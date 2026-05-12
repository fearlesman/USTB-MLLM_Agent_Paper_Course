"""
Entry point for the **agent** evaluation tree (skills / trace / prefixed artifacts).

The **frozen baseline** shim is ``main.py`` (``baseline/``). Data stay at ``Holistic_AVQA_bench/``
or ``AV_SPEAKERBENCH_DATA_ROOT``.

Examples:

    python main_agent.py --model_name Qwen3-Omni-3B --use_local_metadata

Or:

    cd agent && python main.py ...

``AV_SPEAKERBENCH_EVAL_TRACK`` defaults to ``agent`` inside ``agent/main.py``; override if needed.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_AGENT = Path(__file__).resolve().parent / "agent"
_MAIN = _AGENT / "main.py"

if not _MAIN.is_file():
    raise SystemExit(f"Expected agent harness at {_MAIN} (missing).")

os.chdir(_AGENT)
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))
runpy.run_path(str(_MAIN), run_name="__main__")
