"""
Whole-clip **RMS / peak / crest** on decoded mono WAV (numpy).

Cheap numeric cue for **Speech Intensity** cohorts (complements discrete prosody bins).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio_vad import VadRunOutcome, load_waveform_mono


@dataclass
class RmsMeterOutcome:
    clip_rms_dbfs: float | None
    peak_dbfs: float | None
    crest_factor_db: float | None
    sample_rate: int
    analyzed_samples: int
    backend: str
    errors: list[dict[str, Any]]


def _dbfs_amp(rms: float) -> float | None:
    """dBFS for amplitudes in ~[-1,1], full scale = 1."""
    eps = 1e-15
    r = float(rms)
    if r <= 0:
        return None
    return float(20.0 * np.log10(max(r, eps)))


def _rms_env_max_samples() -> int:
    raw = os.getenv("AV_SPEAKERBENCH_RMS_METER_MAX_SAMPLES", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def rms_peak_meter(
    wav_path: str | Path,
    vad: VadRunOutcome | None = None,
    *,
    union_speech_only: bool = False,
    max_samples: int = 0,
) -> RmsMeterOutcome:
    """
    If ``union_speech_only`` and ``vad`` has segments, measure only union of speech intervals
    (good for skipping long silence/leaders).

    ``max_samples`` — cap decoded length (uniform head crop) when ``>0`` (or set
    ``AV_SPEAKERBENCH_RMS_METER_MAX_SAMPLES`` when ``max_samples`` omitted).
    """
    errs: list[dict[str, Any]] = []
    path = Path(wav_path).resolve()
    if not path.is_file():
        return RmsMeterOutcome(None, None, None, 0, 0, "none", [{"kind": "file_missing", "detail": str(path)}])

    try:
        x, sr = load_waveform_mono(path)
    except Exception as e:  # noqa: BLE001
        return RmsMeterOutcome(None, None, None, 0, 0, "numpy_wave", [{"kind": "load_failed", "detail": str(e)}])

    cap = int(max_samples) if max_samples and max_samples > 0 else _rms_env_max_samples()
    if cap and len(x) > cap:
        x = x[:cap]
        errs.append({"kind": "truncated_head", "detail": f"max_samples={cap}"})

    if union_speech_only and vad is not None and vad.segments:
        mask = np.zeros_like(x, dtype=bool)
        for vs in vad.segments:
            i0 = int(max(0, min(len(x), vs.t0 * sr)))
            i1 = int(max(0, min(len(x), vs.t1 * sr)))
            if i1 > i0:
                mask[i0:i1] = True
        xs = np.asarray(x[mask], dtype=np.float64)
        if xs.size == 0:
            xs = np.asarray(x, dtype=np.float64)
            errs.append({"kind": "vad_union_empty", "detail": "fell_back_full_clip"})
    else:
        xs = np.asarray(x, dtype=np.float64)

    rms = float(np.sqrt(np.mean(np.square(xs)))) if xs.size else 0.0
    peak = float(np.max(np.abs(xs))) if xs.size else 0.0
    crest = None
    if rms > 1e-12 and peak > 0:
        crest = float(20.0 * np.log10(peak / rms))

    return RmsMeterOutcome(
        _dbfs_amp(rms),
        _dbfs_amp(peak),
        crest,
        int(sr),
        int(xs.size),
        "numpy_rms_meter_v1",
        errs,
    )