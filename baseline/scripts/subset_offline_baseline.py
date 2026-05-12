"""Stratified subset evaluation without a model: constant-letter baseline (sanity metrics)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dataset import DATASET_ROOT, get_dataset


def build_metrics(ds, pred_letter: str) -> dict:
    pred_letter = pred_letter.strip().upper()[:1]
    raw: dict[str, dict] = {}
    for i in range(len(ds)):
        q = ds[i]
        gt = (q.get("answer") or "").strip().upper()[:1]
        matched = gt == pred_letter
        for prefix, key in (
            ("level 1", q["category"]),
            ("level 2", q["sub_category"]),
            ("level 3", q["task_id"]),
        ):
            k = f"{prefix}: {key}"
            d = raw.setdefault(k, {"matched": 0, "total": 0})
            d["total"] += 1
            if matched:
                d["matched"] += 1
    out: dict = {}
    for k, v in raw.items():
        t = v["total"]
        out[k] = {
            "matched": v["matched"],
            "total": t,
            "accuracy": round(100.0 * v["matched"] / t, 2) if t else 0.0,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample_fraction", type=float, default=0.3)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--data_path", type=str, default=str(DATASET_ROOT))
    p.add_argument("--pred", type=str, default="A", help="Constant prediction letter")
    p.add_argument("--out", type=Path, default=Path("result/offline_baseline_A_sub30.json"))
    args = p.parse_args()

    ds = get_dataset(data_path=args.data_path, sample_fraction=args.sample_fraction, sample_seed=args.sample_seed)
    cats = Counter(ds[i]["category"] for i in range(len(ds)))
    metrics = build_metrics(ds, args.pred)
    metrics["_subset_meta"] = {
        "mode": "constant_letter_baseline",
        "pred": args.pred.upper(),
        "sample_fraction": args.sample_fraction,
        "sample_seed": args.sample_seed,
        "num_questions": len(ds),
        "per_stratum": dict(sorted(cats.items())),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # overall from level-1 buckets
    tot_m = tot_t = 0
    for k, v in metrics.items():
        if isinstance(k, str) and k.startswith("level 1:"):
            tot_m += v["matched"]
            tot_t += v["total"]
    overall = round(100.0 * tot_m / tot_t, 2) if tot_t else 0.0
    print(f"n={len(ds)} per category: {dict(sorted(cats.items()))}")
    print(f"constant-{args.pred.upper()} overall acc (approx, MC 4-way): {overall}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
