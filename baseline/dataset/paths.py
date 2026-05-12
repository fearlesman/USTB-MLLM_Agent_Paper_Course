"""Single place for repo / dataset root paths."""

from __future__ import annotations

import os
from pathlib import Path

# Root of the evaluation harness (this ``baseline/`` tree).
EVAL_ROOT = Path(__file__).resolve().parent.parent
# Top-level repository root (parent of ``baseline/``; default location of ``Holistic_AVQA_bench/``).
PROJECT_ROOT = EVAL_ROOT.parent

# Back-compat: historically ``REPO_ROOT`` meant the git project root (where the dataset folder lives).
REPO_ROOT = PROJECT_ROOT

# Override with env for nonstandard layouts, e.g. set AV_SPEAKERBENCH_DATA_ROOT
_root = os.environ.get("AV_SPEAKERBENCH_DATA_ROOT")
DATASET_ROOT = Path(_root).resolve() if _root else (PROJECT_ROOT / "Holistic_AVQA_bench").resolve()
