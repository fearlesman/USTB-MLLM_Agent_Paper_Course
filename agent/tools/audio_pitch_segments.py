"""Per-segment median F0 (Hz) for short clips — librosa ``pyin`` when available."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio_vad import load_waveform_mono


@dataclass(frozen=True)
class F0SegmentStat:
    label: str
    t0: float
    t1: float
    median_hz: float | None
    voiced_ratio: float


@dataclass
class PitchStatsOutcome:
    segments: list[F0SegmentStat]
    backend: str
    errors: list[dict[str, Any]]


def pitch_median_over_segments(
    wav_path: str | Path,
    segments: list[tuple[float, float, str]],
    *,
    fmin_hz: float = 65.0,
    fmax_hz: float = 450.0,
) -> PitchStatsOutcome:
    errs: list[dict[str, Any]] = []
    path = Path(wav_path).resolve()
    if not path.is_file():
        return PitchStatsOutcome([], "none", [{"kind": "file_missing", "detail": str(path)}])
    try:
        x, sr = load_waveform_mono(path)
    except Exception as e:  # noqa: BLE001
        return PitchStatsOutcome([], "none", [{"kind": "load_failed", "detail": str(e)}])

    try:
        import librosa
    except ImportError as ie:
        return PitchStatsOutcome(
            [],
            "stub",
            [{"kind": "librosa_missing", "detail": str(ie)}],
        )

    x = np.asarray(x, dtype=np.float64)
    out: list[F0SegmentStat] = []
    backend = "librosa_pyin_v1"
    for t0, t1, lab in segments:
        if t1 <= t0:
            continue
        i0 = int(max(0, t0 * sr))
        i1 = int(min(len(x), max(i0 + 2, round(t1 * sr))))
        sl = x[i0:i1]
        if sl.size < int(0.05 * sr):
            out.append(F0SegmentStat(lab, t0, t1, None, 0.0))
            continue
        try:
            f0_hz, voiced_flag, _ = librosa.pyin(  # type: ignore[no-untyped-call]
                sl,
                fmin=float(fmin_hz),
                fmax=float(fmax_hz),
                sr=sr,
                frame_length=2048,
                hop_length=256,
            )
        except Exception as e:  # noqa: BLE001
            errs.append({"kind": "pyin_failed", "detail": str(e), "label": lab})
            out.append(F0SegmentStat(lab, t0, t1, None, 0.0))
            continue
        f0_hz = np.asarray(f0_hz, dtype=np.float64)
        voiced = np.isfinite(f0_hz) & (f0_hz > 1.0)
        voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0
        med = None
        if np.any(voiced):
            med = float(np.median(f0_hz[voiced]))
        out.append(F0SegmentStat(lab, t0, t1, med, voiced_ratio))

    return PitchStatsOutcome(out, backend, errs)


def format_f0_sheet(ot: PitchStatsOutcome) -> str:
    lines = ["[f0_median_hz_by_segment]", f"backend={ot.backend}"]
    for s in ot.segments[:16]:
        hz = "n/a" if s.median_hz is None else f"{s.median_hz:.1f}"
        lines.append(
            f"{s.label} median_hz={hz} voiced_ratio={s.voiced_ratio:.2f} span_s=[{s.t0:.2f},{s.t1:.2f}]"
        )
    if len(ot.segments) > 16:
        lines.append(f"[truncated n_segments={len(ot.segments)}]")
    return "\n".join(lines) + "\n"
