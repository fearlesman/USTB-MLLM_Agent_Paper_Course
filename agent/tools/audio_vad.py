"""
Speech-activity segmentation (Tool T1): anchor windows for ASR / moment / overlap Skills.

Default backend is **energy + adaptive noise floor** on decoded PCM (via ``torchaudio``).
Swap later with WebRTC VAD / Silero by implementing a second backend and selecting via env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VadSegment:
    t0: float
    t1: float
    conf: float | None = None


@dataclass
class VadRunOutcome:
    segments: list[VadSegment]
    duration_s: float
    backend: str
    sample_rate: int
    errors: list[dict[str, Any]]


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _energy_vad_mono(
    x: np.ndarray,
    sr: int,
    *,
    frame_ms: float,
    hop_ms: float,
    energy_margin_db: float,
    min_segment_s: float,
    max_merge_gap_s: float,
    noise_percentile: float,
) -> list[VadSegment]:
    """Simple frame-energy VAD with hysteresis via merge/split."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0 or sr <= 0:
        return []

    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if frame > n:
        frame = n
    if hop > frame:
        hop = frame

    energies: list[float] = []
    for start in range(0, n - frame + 1, hop):
        chunk = x[start : start + frame]
        e = float(np.mean(chunk * chunk) + 1e-12)
        energies.append(10.0 * np.log10(e))
    if not energies:
        return []

    e_arr = np.array(energies, dtype=np.float64)
    noise_floor = float(np.percentile(e_arr, noise_percentile))
    thresh = noise_floor + energy_margin_db
    speech = e_arr > thresh

    times_t: list[float] = []
    for i, sp in enumerate(speech):
        t_center = (i * hop + 0.5 * frame) / sr
        times_t.append(t_center)

    raw_segs: list[tuple[float, float]] = []
    i = 0
    while i < len(speech):
        if not speech[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(speech) and speech[j + 1]:
            j += 1
        t0 = (i * hop) / sr
        t1 = (j * hop + frame) / sr
        raw_segs.append((t0, min(t1, n / sr)))
        i = j + 1

    if not raw_segs:
        return []

    merged: list[list[float]] = [[raw_segs[0][0], raw_segs[0][1]]]
    for t0, t1 in raw_segs[1:]:
        if t0 - merged[-1][1] <= max_merge_gap_s:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])

    out: list[VadSegment] = []
    dur = n / sr
    for a, b in merged:
        if b - a < min_segment_s:
            continue
        a = max(0.0, a)
        b = min(dur, b)
        if b <= a:
            continue
        out.append(VadSegment(t0=a, t1=b, conf=None))
    return out


def _load_waveform_mono(path: str) -> tuple[np.ndarray, int]:
    tor_err: Exception | None = None
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        if wav.dim() != 2:
            raise ValueError("expected 2D waveform from torchaudio.load")
        x = wav.mean(dim=0).detach().cpu().numpy().astype(np.float64)
        return x, int(sr)
    except Exception as e:
        tor_err = e

    import wave

    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
        if sw != 2:
            raise ValueError(f"wave fallback requires 16-bit PCM, got sampwidth={sw}") from tor_err
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        x = x / 32768.0
        return x, int(sr)
    except Exception as wave_exc:
        if tor_err is not None:
            raise tor_err from wave_exc
        raise wave_exc


def load_waveform_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load mono waveform as float64 samples (approximately normalized) using torchaudio or 16‑bit WAV."""
    return _load_waveform_mono(str(Path(path).resolve()))


@lru_cache(maxsize=256)
def _vad_cached(
    resolved_path: str,
    mtime: float,
    frame_ms: float,
    hop_ms: float,
    margin_db: float,
    min_seg: float,
    merge_gap: float,
    noise_pct: float,
    max_dur: float,
) -> tuple[tuple[tuple[float, float, float | None], ...], float, int, str, tuple[tuple[str, str], ...]]:
    """
    Returns:
      (segments as tuples (t0,t1,conf), duration_s, sample_rate, backend, errors as (kind,detail))
    """
    errors: list[tuple[str, str]] = []
    backend = "energy_vad"
    try:
        x, sr = _load_waveform_mono(resolved_path)
    except Exception as e:  # noqa: BLE001 — surface as outcome errors
        errors.append(("load_failed", str(e)))
        return (), 0.0, 0, backend, tuple(errors)

    dur = float(len(x) / max(sr, 1))
    if max_dur > 0 and dur > max_dur:
        n_keep = int(max_dur * sr)
        x = x[:n_keep]
        dur = float(len(x) / max(sr, 1))
        errors.append(("truncated", f"analyzed_first_s={max_dur}"))

    segs = _energy_vad_mono(
        x,
        sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        energy_margin_db=margin_db,
        min_segment_s=min_seg,
        max_merge_gap_s=merge_gap,
        noise_percentile=noise_pct,
    )
    tup = tuple((s.t0, s.t1, s.conf) for s in segs)
    return tup, dur, sr, backend, tuple(errors)


def vad_segments_from_wav_path(path: str | Path) -> VadRunOutcome:
    """
    Decode ``path`` and return speech segments in seconds.

    Tunables (optional env):
      ``AV_SPEAKERBENCH_VAD_FRAME_MS``, ``AV_SPEAKERBENCH_VAD_HOP_MS``,
      ``AV_SPEAKERBENCH_VAD_MARGIN_DB``, ``AV_SPEAKERBENCH_VAD_MIN_SEGMENT_S``,
      ``AV_SPEAKERBENCH_VAD_MERGE_GAP_S``, ``AV_SPEAKERBENCH_VAD_NOISE_PERCENTILE``,
      ``AV_SPEAKERBENCH_VAD_MAX_DURATION_S`` (0 = no cap).
    """
    p = Path(path).resolve()
    if not p.is_file():
        return VadRunOutcome(
            segments=[],
            duration_s=0.0,
            backend="none",
            sample_rate=0,
            errors=[{"kind": "file_missing", "detail": str(p)}],
        )

    frame_ms = _parse_float("AV_SPEAKERBENCH_VAD_FRAME_MS", 25.0)
    hop_ms = _parse_float("AV_SPEAKERBENCH_VAD_HOP_MS", 10.0)
    margin_db = _parse_float("AV_SPEAKERBENCH_VAD_MARGIN_DB", 3.0)
    min_seg = _parse_float("AV_SPEAKERBENCH_VAD_MIN_SEGMENT_S", 0.08)
    merge_gap = _parse_float("AV_SPEAKERBENCH_VAD_MERGE_GAP_S", 0.25)
    noise_pct = _parse_float("AV_SPEAKERBENCH_VAD_NOISE_PERCENTILE", 15.0)
    noise_pct = min(50.0, max(1.0, noise_pct))
    max_dur = _parse_float("AV_SPEAKERBENCH_VAD_MAX_DURATION_S", 0.0)

    mtime = p.stat().st_mtime
    tup, dur, sr, backend, err_tuples = _vad_cached(
        str(p),
        mtime,
        frame_ms,
        hop_ms,
        margin_db,
        min_seg,
        merge_gap,
        noise_pct,
        max_dur,
    )
    segs = [VadSegment(t0=a, t1=b, conf=c) for a, b, c in tup]
    errors = [{"kind": k, "detail": d} for k, d in err_tuples]
    return VadRunOutcome(segments=segs, duration_s=dur, backend=backend, sample_rate=sr, errors=errors)


def format_segments_for_prompt(
    outcome: VadRunOutcome,
    *,
    max_segments: int | None = None,
    precision: int = 2,
) -> str:
    """Compact, token-bounded line for Structured_skill_evidence."""
    lim = max_segments if max_segments is not None else _parse_int("AV_SPEAKERBENCH_VAD_MAX_SEGMENTS_IN_PROMPT", 24)
    segs = outcome.segments[: max(0, lim)]
    inner = ",".join(f"[{s.t0:.{precision}f},{s.t1:.{precision}f}]" for s in segs)
    return (
        f"backend={outcome.backend} sr={outcome.sample_rate} "
        f"duration_s={outcome.duration_s:.{precision}f} "
        f"n_segments={len(outcome.segments)} "
        f"segments_s={inner}"
    )
