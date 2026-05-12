"""
Discrete **energy / rhythm** summaries over anchor windows — numpy-only heuristic.

``Targets``: level 3: Speech Pitch, Speech Rate, Speech Intensity (+ duration variants);
mechanism: perception granularity (baseline acc placeholders per MM_AGENT_DESIGN).

Outputs coarse tertile labels per VAD slice (not phonetic-quality pitch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio_vad import VadRunOutcome, VadSegment, load_waveform_mono


@dataclass
class ProsodyOutcome:
    lines: list[str]
    backend: str
    errors: list[dict[str, Any]]


def _zcr(seg: np.ndarray) -> float:
    x = np.asarray(seg, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    s = np.sign(x)
    s[s == 0] = 1
    crossings = int(np.sum(np.abs(np.diff(s)) > 1e-15))
    return crossings / float(max(1, x.size - 1))


def _rms(seg: np.ndarray) -> float:
    x = np.asarray(seg, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def discrete_prosody_over_vad(
    wav_path: str | Path,
    vad: VadRunOutcome | None,
    *,
    max_segments: int = 12,
) -> ProsodyOutcome:
    errs: list[dict[str, Any]] = []
    path = Path(wav_path).resolve()
    if not path.is_file():
        return ProsodyOutcome([], "none", [{"kind": "file_missing", "detail": str(path)}])

    try:
        x_raw, sr = load_waveform_mono(path)
    except Exception as e:  # noqa: BLE001
        return ProsodyOutcome([], "energy_heuristic_v1", [{"kind": "load_failed", "detail": str(e)}])

    if vad is not None:
        vad_dur = float(vad.duration_s)
    else:
        vad_dur = float(len(x_raw)) / float(max(sr, 1))

    if vad and vad.segments:
        segs = vad.segments[:max_segments]
    else:
        segs = [
            VadSegment(
                t0=0.0,
                t1=min(float(len(x_raw)) / float(max(sr, 1)), vad_dur or float(len(x_raw)) / float(max(sr, 1))),
                conf=None,
            )
        ]

    feats: list[tuple[float, float, float]] = []
    for vs in segs:
        i0 = int(max(0, vs.t0 * sr))
        i1 = int(min(len(x_raw), max(i0 + 1, int(vs.t1 * sr))))
        sl = np.asarray(x_raw[i0:i1])
        feats.append((_rms(sl), _zcr(sl), vs.t1 - vs.t0))

    rmss = sorted(f[0] for f in feats) or [0.0]
    zcrs = sorted(f[1] for f in feats) or [0.0]

    def tertile_label(val: float, arr: list[float]) -> str:
        if len(arr) < 2:
            return "single_bin"
        p33 = arr[int(0.33 * (len(arr) - 1))]
        p67 = arr[int(0.67 * (len(arr) - 1))]
        if val < p33:
            return "low"
        if val < p67:
            return "mid"
        return "high"

    lines_body: list[str] = []
    for vs, fk in zip(segs, feats, strict=False):
        rms_b = tertile_label(fk[0], rmss)
        zcr_b = tertile_label(fk[1], zcrs)
        dur_s = fk[2]
        lines_body.append(
            f"[{vs.t0:.2f},{vs.t1:.2f}] intensity_bin={rms_b} "
            f"zcr_density_bin={zcr_b} slice_dur_s={dur_s:.2f}"
        )

    clip_dur = float(len(x_raw)) / float(max(sr, 1))
    hdr = ["[prosody_discrete_energy_v1]", f"clip_duration_s={clip_dur:.2f}"]
    return ProsodyOutcome(hdr + lines_body, "energy_heuristic_v1", errs)
