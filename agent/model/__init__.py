"""Load the frozen ``baseline/model`` implementation under a private name and re-export symbols used by ``agent/main.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# ``…/agent/model/__init__.py`` → parents[0]=model dir, [1]=agent, [2]=repo root.
_BASELINE_MODEL_DIR = Path(__file__).resolve().parents[2] / "baseline" / "model"
_IMPL = "_avsb_baseline_model_impl"


def _load_baseline_model():
    spec = importlib.util.spec_from_file_location(
        _IMPL,
        _BASELINE_MODEL_DIR / "__init__.py",
        submodule_search_locations=[str(_BASELINE_MODEL_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline model from {_BASELINE_MODEL_DIR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_IMPL] = mod
    spec.loader.exec_module(mod)
    return mod


_bm = _load_baseline_model()
inference = _bm.inference
flush_experiment_artifacts = _bm.flush_experiment_artifacts
