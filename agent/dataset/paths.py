"""Single place for repo / dataset root paths (agent harness)."""

from __future__ import annotations

import os
from pathlib import Path

# Root of the agent harness (``agent/`` tree).
EVAL_ROOT = Path(__file__).resolve().parent.parent
# Top-level repository root (parent of ``agent/``; default location of ``Holistic_AVQA_bench/``).
PROJECT_ROOT = EVAL_ROOT.parent

REPO_ROOT = PROJECT_ROOT

_root = os.environ.get("AV_SPEAKERBENCH_DATA_ROOT")
DATASET_ROOT = Path(_root).resolve() if _root else (PROJECT_ROOT / "Holistic_AVQA_bench").resolve()
