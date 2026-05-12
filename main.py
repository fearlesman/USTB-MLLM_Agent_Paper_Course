"""
Compatibility entry at the repository root.

The frozen **baseline** evaluation harness lives under ``baseline/``. Data stay at the default
``Holistic_AVQA_bench/`` next to this folder (or ``AV_SPEAKERBENCH_DATA_ROOT``).

Run either:

- ``python main.py ...`` from the repo root (this script changes the working directory to ``baseline/``), or
- ``cd baseline && python main.py ...``
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_BASELINE = Path(__file__).resolve().parent / "baseline"
_MAIN = _BASELINE / "main.py"

if not _MAIN.is_file():
    raise SystemExit(f"Expected baseline harness at {_MAIN} (missing).")

os.chdir(_BASELINE)
if str(_BASELINE) not in sys.path:
    sys.path.insert(0, str(_BASELINE))
runpy.run_path(str(_MAIN), run_name="__main__")
