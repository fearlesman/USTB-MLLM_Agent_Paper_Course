"""Visualize AV-SpeakerBench metrics from result/*.json files.

Each result file is a flat JSON object whose keys look like:
  "level 1: <category>", "level 2: <sub_category>", "level 3: <task_id>"
with values {"matched", "total", "accuracy"} (accuracy in percent).

Usage:
  python scripts/plot_benchmark_results.py --results result/modelA.json --out_dir result/plots --levels 1,2,3
  python scripts/plot_benchmark_results.py --results a.json b.json --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

_LEVEL_RE = re.compile(r"^level ([123]): (.*)$", re.DOTALL)


def _parse_level_key(key: str) -> tuple[int, str] | None:
    m = _LEVEL_RE.match(key.strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def load_benchmark_json(path: Path) -> dict[int, list[tuple[str, float, int, int]]]:
    """Return rows per level: (name, accuracy_pct, matched, total), sorted by name."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")

    buckets: dict[int, list[tuple[str, float, int, int]]] = {1: [], 2: [], 3: []}
    for k, v in raw.items():
        parsed = _parse_level_key(str(k))
        if parsed is None:
            continue
        level, name = parsed
        if not isinstance(v, dict):
            continue
        acc = v.get("accuracy")
        matched = v.get("matched", 0)
        total = v.get("total", 0)
        if acc is None and total:
            acc = round(100.0 * matched / total, 2)
        elif acc is None:
            acc = 0.0
        buckets[level].append((name, float(acc), int(matched), int(total)))

    for lvl in buckets:
        buckets[lvl].sort(key=lambda x: x[0])
    return buckets


def _model_label(path: Path) -> str:
    return path.stem


def plot_level_horizontal(
    rows: list[tuple[str, float, int, int]],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    labels = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    h = max(4.0, 0.28 * len(labels))
    fig, ax = plt.subplots(figsize=(8, h))
    y = range(len(labels))
    ax.barh(list(y), accs, color="steelblue")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_level_grouped(
    series_by_model: dict[str, dict[int, dict[str, float]]],
    level: int,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = sorted(
        {name for model_rows in series_by_model.values() for name in model_rows.get(level, {})}
    )
    if not names:
        return

    n_models = len(series_by_model)
    x = np.arange(len(names), dtype=float)
    width = min(0.8 / n_models, 0.25)
    fig_h = max(5.0, 0.22 * len(names))
    fig, ax = plt.subplots(figsize=(max(9, 0.35 * len(names)), fig_h))

    for i, (model_name, levels_dict) in enumerate(series_by_model.items()):
        label_map = levels_dict.get(level, {})
        heights = [label_map.get(nm, float("nan")) for nm in names]
        offset = (i - (n_models - 1) / 2) * width
        ax.barh(x + offset, heights, width, label=model_name)

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(0, 100)
    handles, leg_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, leg_labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    path: Path,
    loaded: dict[Path, dict[int, list[tuple[str, float, int, int]]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "name", "accuracy", "matched", "total"])
        for p, buckets in loaded.items():
            label = _model_label(p)
            for lvl in (1, 2, 3):
                for name, acc, matched, total in buckets.get(lvl, []):
                    w.writerow([label, lvl, name, acc, matched, total])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    parser.add_argument(
        "--results",
        nargs="+",
        type=Path,
        required=True,
        help="One or more result JSON paths (e.g. result/Qwen3-Omni-3B.json)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("result/plots"),
        help="Directory for PNG outputs (default: result/plots)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="1,2,3",
        help="Comma-separated levels to plot: 1, 2, 3 (default: 1,2,3)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write results_summary.csv under --out_dir",
    )
    args = parser.parse_args()

    try:
        levels = {int(x.strip()) for x in args.levels.split(",") if x.strip()}
    except ValueError as e:
        raise SystemExit(f"Invalid --levels: {args.levels}") from e
    if not levels.issubset({1, 2, 3}):
        raise SystemExit("--levels must be subset of 1,2,3")

    loaded: dict[Path, dict[int, list[tuple[str, float, int, int]]]] = {}
    for p in args.results:
        if not p.is_file():
            raise SystemExit(f"Missing result file: {p}")
        loaded[p] = load_benchmark_json(p)

    if args.csv:
        write_summary_csv(args.out_dir / "results_summary.csv", loaded)

    titles = {
        1: "Level 1 — category",
        2: "Level 2 — sub_category",
        3: "Level 3 — task_id",
    }

    if len(loaded) == 1:
        path, buckets = next(iter(loaded.items()))
        stem = _model_label(path)
        for lvl in sorted(levels):
            rows = buckets.get(lvl, [])
            out = args.out_dir / f"level{lvl}_{stem}.png"
            if rows:
                plot_level_horizontal(rows, f"{stem}: {titles[lvl]}", out)
                print(f"Wrote {out}")
    else:
        series_by_model: dict[str, dict[int, dict[str, float]]] = {}
        for path, buckets in loaded.items():
            m = _model_label(path)
            series_by_model[m] = {
                lvl: {name: acc for name, acc, _, _ in buckets.get(lvl, [])}
                for lvl in (1, 2, 3)
            }
        for lvl in sorted(levels):
            out = args.out_dir / f"level{lvl}_comparison.png"
            plot_level_grouped(series_by_model, lvl, titles[lvl], out)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
