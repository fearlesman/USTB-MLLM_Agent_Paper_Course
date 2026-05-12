"""Verify local dataset: fixed DATASET_ROOT, metadata load, media spot-check, category counts."""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dataset import DATASET_ROOT, get_dataset


def main() -> int:
    root = DATASET_ROOT
    lines = []
    lines.append(f"DATASET_ROOT = {root}")

    if not root.is_dir():
        lines.append("FAIL: directory missing")
        print("\n".join(lines))
        return 2

    csv_path = root / "test.csv"
    if not csv_path.is_file():
        lines.append(f"FAIL: missing {csv_path}")
        print("\n".join(lines))
        return 2
    lines.append(f"test.csv OK ({csv_path.stat().st_size} bytes)")

    ds = get_dataset(data_path=str(root))
    n = len(ds)
    lines.append(f"get_dataset count: {n}")

    rng = random.Random(0)
    k = min(150, n)
    idxs = rng.sample(range(n), k)
    miss_av = miss_a = miss_v = 0
    for i in idxs:
        q = ds[i]
        if not (root / q["audio_visual_path"]).is_file():
            miss_av += 1
        if not (root / q["audio_path"]).is_file():
            miss_a += 1
        if not (root / q["visual_path"]).is_file():
            miss_v += 1
    lines.append(f"spot-check random {k} (seed=0): missing AV={miss_av}, audio={miss_a}, visual={miss_v}")

    cat_counts = Counter(ds[i]["category"] for i in range(n))
    lines.append("category counts:")
    for name in sorted(cat_counts):
        lines.append(f"  {name}: {cat_counts[name]}")

    ok = miss_av == 0 and miss_a == 0 and miss_v == 0
    lines.append(f"RESULT: {'PASS' if ok else 'FAIL (some clips missing)'}")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
