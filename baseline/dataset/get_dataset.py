"""Build AV-SpeakerBench question index for evaluation."""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset

from .paths import DATASET_ROOT

DEFAULT_HUB_DATASET_ID = "plnguyen2908/Holistic_AVQA_bench"


def _rows_from_doc(doc) -> list[dict]:
    """Normalize Hub Dataset / list / iterable to a list of row dicts."""
    if hasattr(doc, "__len__") and hasattr(doc, "__getitem__") and not isinstance(doc, list):
        return [dict(doc[i]) for i in range(len(doc))]
    out: list[dict] = []
    for row in doc:
        out.append(dict(row))
    return out


def _filter_rows_by_clip_size(
    rows: list[dict],
    data_path: str,
    max_bytes: int,
    *,
    media_check: str = "all",
) -> tuple[list[dict], int]:
    """
    Drop rows whose local clip(s) exceed ``max_bytes`` (or are missing).

    ``media_check``:
      - ``all``: audiovisual + visual_only + audio_only each present and <= max_bytes
        (safe for OpenAI ``data:`` URIs with split + fallback).
      - ``combined`` / ``visual`` / ``audio``: only that path column.
    """
    root = Path(data_path)
    kept: list[dict] = []
    dropped = 0
    for r in rows:
        rels: list[str] = []
        if media_check == "all":
            rels = [r["audio_visual_path"], r["visual_path"], r["audio_path"]]
        elif media_check == "combined":
            rels = [r["audio_visual_path"]]
        elif media_check == "visual":
            rels = [r["visual_path"]]
        elif media_check == "audio":
            rels = [r["audio_path"]]
        else:
            raise ValueError(f"unknown media_check: {media_check}")

        ok = True
        for rel in rels:
            p = (root / rel).resolve()
            if not p.is_file():
                ok = False
                break
            if p.stat().st_size > max_bytes:
                ok = False
                break
        if ok:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def _stratified_sample_by_key(rows: list[dict], fraction: float, seed: int, key: str) -> list[dict]:
    """Per bucket ``round(n * fraction)`` items; each non-empty bucket keeps at least one."""
    import random

    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1]")
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r[key], []).append(r)
    out: list[dict] = []
    for k in sorted(buckets.keys()):
        items = buckets[k][:]
        rng.shuffle(items)
        n = len(items)
        n_take = int(round(n * fraction))
        n_take = max(1, n_take)
        n_take = min(n, n_take)
        out.extend(items[:n_take])
    rng.shuffle(out)
    return out


class AVQA_Dataset(Dataset):
    def __init__(
        self,
        doc,
        category=None,
        sub_category=None,
        task_id=None,
        sample_fraction: float | None = None,
        sample_seed: int = 0,
        stratify_key: str = "category",
    ):
        self.questions = []

        for question in doc:
            category_check, sub_category_check, task_id_check = True, True, True
            if category is not None and category != question["category"]:
                category_check = False
            if sub_category is not None and sub_category != question["sub_category"]:
                sub_category_check = False
            if task_id is not None and task_id != question["task_id"]:
                task_id_check = False
            if category_check and sub_category_check and task_id_check:
                self.questions.append(question)

        if sample_fraction is not None:
            sf = float(sample_fraction)
            if not (0 < sf <= 1):
                raise ValueError("sample_fraction must be in (0, 1] or omitted")
            self.questions = _stratified_sample_by_key(self.questions, sf, sample_seed, stratify_key)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]


def _load_questions_from_hub():
    return load_dataset(DEFAULT_HUB_DATASET_ID, split="test")


def _pick_metadata_parquet(data_path: str) -> Path | None:
    explicit = os.environ.get("HOLISTIC_METADATA_PARQUET")
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None

    root = Path(data_path)
    if not root.is_dir():
        return None

    matches = list(root.rglob("*.parquet"))
    if not matches:
        return None

    for needle in ("test", "benchmark", "question", "holistic"):
        for p in matches:
            if needle in p.name.lower():
                return p

    for p in sorted(matches, key=lambda x: x.stat().st_size):
        if p.stat().st_size < 80 * 1024 * 1024:
            return p
    return None


def _pick_metadata_csv(data_path: str) -> Path | None:
    """Prefer ``test.csv`` next to clips (see Holistic_AVQA_bench/README.md)."""
    explicit = os.environ.get("HOLISTIC_METADATA_CSV")
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None

    root = Path(data_path)
    if not root.is_dir():
        return None

    for name in ("test.csv", "benchmark.csv", "train.csv"):
        p = root / name
        if p.is_file():
            return p
    return None


def _load_questions_from_csv(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _try_load_local(data_path: str | None) -> object | None:
    """Load questions from parquet/csv under data_path, or None if no table found."""
    if not data_path:
        return None
    root = Path(data_path)
    if not root.is_dir():
        return None
    pq = _pick_metadata_parquet(data_path)
    if pq is not None:
        return load_dataset("parquet", data_files=str(pq), split="train")
    csv_p = _pick_metadata_csv(data_path)
    if csv_p is not None:
        return _load_questions_from_csv(csv_p)
    return None


def get_dataset(
    category=None,
    sub_category=None,
    task_id=None,
    data_path: str | None = None,
    use_local_metadata: bool = False,
    force_hub_metadata: bool = False,
    sample_fraction: float | None = None,
    sample_seed: int = 0,
    stratify_key: str = "category",
    max_clip_bytes: int | None = None,
    clip_size_media_check: str = "all",
):
    """
    By default, if ``data_path`` contains ``test.csv`` (or a small *.parquet), questions
    load from disk and the Hugging Face Hub is not contacted. Pass ``force_hub_metadata=True``
    (CLI: ``--hub_metadata``) to always use the Hub. Pass ``use_local_metadata=True``
    (CLI: ``--use_local_metadata``) or set env ``HOLISTIC_USE_LOCAL_METADATA=1`` to *require*
    a local table and fail if it is missing.

    If ``max_clip_bytes`` is set, rows are **dropped first** (missing or oversize local files),
    then ``sample_fraction`` stratified sampling runs on the remaining pool (same per-bucket rule).
    Use ~14MiB (``14 * 1024 * 1024``) to stay under DashScope OpenAI compatible **~20MiB per data-uri**
    after base64 expansion.

    If ``data_path`` is omitted, ``Holistic_AVQA_bench`` next to the project root is used.
    """
    require_local = use_local_metadata or os.environ.get("HOLISTIC_USE_LOCAL_METADATA", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if data_path is None:
        data_path = str(DATASET_ROOT)

    doc = None
    if not force_hub_metadata:
        doc = _try_load_local(data_path)

    if doc is None:
        if require_local:
            raise FileNotFoundError(
                f"No local question table under {data_path}. "
                "Expected test.csv or *.parquet (see Holistic_AVQA_bench/README.md), "
                "or set HOLISTIC_METADATA_CSV / HOLISTIC_METADATA_PARQUET."
            )
        doc = _load_questions_from_hub()

    oversized_dropped = 0
    if max_clip_bytes is not None and int(max_clip_bytes) > 0:
        rows = _rows_from_doc(doc)
        rows, oversized_dropped = _filter_rows_by_clip_size(
            rows,
            str(data_path),
            int(max_clip_bytes),
            media_check=clip_size_media_check,
        )
        doc = rows

    dataset = AVQA_Dataset(
        doc,
        category,
        sub_category,
        task_id,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        stratify_key=stratify_key,
    )
    dataset.oversized_clips_dropped_prefilter = oversized_dropped
    dataset.max_clip_bytes_prefilter = max_clip_bytes
    return dataset
