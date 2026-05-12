"""
Tool T2 — **anchor-window ASR**: transcript snippets aligned to ``VAD`` speech spans (or a single full chunk).

Backends (``AV_SPEAKERBENCH_ASR_BACKEND``):

- ``auto`` *(default)* — ``faster_whisper`` if importable, else ``openai_whisper``, else ``stub``.
- ``faster_whisper`` / ``fw`` — `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_ .
- ``whisper`` / ``openai_whisper`` / ``ow`` — `openai-whisper <https://github.com/openai/whisper>`_ .
- ``stub`` *(default)* — no model; ``AV_SPEAKERBENCH_STUB_ASR_TEXT`` is injected **only if**
  ``AV_SPEAKERBENCH_ALLOW_SYNTHETIC_ASR=1/true/yes`` (default off — avoids bogus “gain” analyses).

Local ``faster-whisper`` CTranslate2 directory (offline / mirrors): set
``AV_SPEAKERBENCH_FASTER_WHISPER_MODEL_DIR`` to an on-disk folder (e.g. from
``modelscope snapshot-download Systran/faster-whisper-small`` on `ModelScope <https://www.modelscope.cn/models/Systran/faster-whisper-small>`_).
If unset, ``AV_SPEAKERBENCH_WHISPER_MODEL`` is used as Hugging Face model size (``base``, ``small``, …)
or as a directory path when it points to an existing folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .audio_vad import VadRunOutcome, VadSegment, load_waveform_mono


@dataclass
class AsrSegment:
    t0: float
    t1: float
    text: str
    conf: float | None = None


@dataclass
class AsrRunOutcome:
    segments: list[AsrSegment]
    backend: str
    language_hint: str | None
    errors: list[dict[str, Any]]


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def allow_stub_asr_text_prompt() -> bool:
    return os.getenv("AV_SPEAKERBENCH_ALLOW_SYNTHETIC_ASR", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )



def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def mono_to_16k_f32(x: np.ndarray, sr: int) -> np.ndarray:
    """Float32 mono at 16 kHz — expected by Whisper backends."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if sr == 16000:
        return x
    try:
        import torch
        import torchaudio.functional as F

        t = torch.from_numpy(x).unsqueeze(0)
        y = F.resample(t, sr, 16000)
        return y.squeeze(0).numpy().astype(np.float32)
    except ImportError:
        ratio = int(round(sr / 16000))
        if ratio <= 1 or abs(sr / 16000 - ratio) > 0.05:
            raise RuntimeError(
                f"samplerate {sr}≠16kHz and torchaudio missing — install torchaudio or use 16k clips"
            )
        x_down = x[::ratio]
        return np.asarray(x_down, dtype=np.float32)


def _select_backend(explicit: str) -> str:
    e = explicit.strip().lower()
    aliases = {
        "fw": "faster_whisper",
        "openai_whisper": "whisper",
        "ow": "whisper",
    }
    if e in aliases:
        e = aliases[e]

    if e in ("stub", "none", "off"):
        return "stub"
    if e == "auto" or not e:
        try:
            import faster_whisper  # noqa: F401

            return "faster_whisper"
        except ImportError:
            pass
        try:
            import whisper  # noqa: F401

            return "whisper"
        except ImportError:
            return "stub"
    if e == "faster_whisper":
        return "faster_whisper"
    if e == "whisper":
        return "whisper"
    return "stub"


def _windows_for_asr(
    vad_out: VadRunOutcome | None,
    x: np.ndarray,
    sr: int,
    *,
    window_budget_mult: float = 1.0,
) -> list[tuple[float, float]]:
    base = max(1, _parse_int("AV_SPEAKERBENCH_ASR_MAX_WINDOWS", 12))
    cap = float(os.getenv("AV_SPEAKERBENCH_ASR_WINDOW_MULT_CAP", "3") or 3)
    try:
        m = float(window_budget_mult)
    except TypeError:
        m = 1.0
    m = max(0.25, min(cap, m))
    max_windows = max(1, int(round(base * m)))
    max_chunk_s = max(5.0, _parse_float("AV_SPEAKERBENCH_ASR_MAX_CHUNK_S", 120.0))
    dur = len(x) / max(sr, 1)

    vad_segs = list(vad_out.segments) if vad_out and vad_out.segments else []
    if vad_segs:
        out = []
        for s in vad_segs[:max_windows]:
            if s.t1 - s.t0 < 0.05:
                continue
            t0 = max(0.0, float(s.t0))
            t1 = min(dur, float(s.t1))
            if t1 > t0:
                out.append((t0, t1))
        if out:
            return out

    t1 = min(dur, max_chunk_s)
    if t1 <= 0:
        return []
    return [(0.0, t1)]


@lru_cache(maxsize=8)
def _faster_whisper_model(model_ref: str, device: str, compute_type: str):  # type: ignore[no-untyped-def]
    from faster_whisper import WhisperModel

    return WhisperModel(model_ref, device=device, compute_type=compute_type)


@lru_cache(maxsize=4)
def _openai_whisper_model(model_name: str, device_str: str):  # type: ignore[no-untyped-def]
    import whisper

    return whisper.load_model(model_name, device=device_str)


def _resolved_faster_whisper_model_ref() -> str:
    """HF size name (e.g. ``small``) or local CTranslate2 model directory."""
    explicit = os.getenv("AV_SPEAKERBENCH_FASTER_WHISPER_MODEL_DIR", "").strip()
    if explicit:
        ep = Path(explicit)
        if ep.is_dir():
            return str(ep.resolve())
    wm = os.getenv("AV_SPEAKERBENCH_WHISPER_MODEL", "base").strip() or "base"
    wp = Path(wm)
    if wp.is_dir():
        return str(wp.resolve())
    return wm


def _run_faster_whisper(
    wav_16k: np.ndarray,
    win_t0: float,
    language: str | None,
) -> list[tuple[float, float, str, float | None]]:
    model_ref = _resolved_faster_whisper_model_ref()
    device = os.getenv("AV_SPEAKERBENCH_ASR_DEVICE", "cpu").strip() or "cpu"
    ctype = os.getenv("AV_SPEAKERBENCH_ASR_COMPUTE_TYPE", "default").strip() or "default"
    if ctype == "default":
        ctype = "int8" if device == "cpu" else "float16"
    model = _faster_whisper_model(model_ref, device, ctype)
    lang = language if language else None
    segs_it, info = model.transcribe(
        wav_16k.astype(np.float32),
        beam_size=5,
        language=lang,
        vad_filter=False,
    )
    out: list[tuple[float, float, str, float | None]] = []
    avg_logprob = getattr(info, "language_probability", None)
    try:
        for seg in segs_it:
            t0 = win_t0 + float(seg.start)
            t1 = win_t0 + float(seg.end)
            txt = (seg.text or "").strip()
            if txt:
                out.append((t0, t1, txt, getattr(seg, "avg_logprob", avg_logprob)))
    except StopIteration:
        pass
    return out


def _simple_decode_openai_whisper(wav_16k: np.ndarray, win_t0: float, language: str | None) -> list[tuple[float, float, str, float | None]]:
    """
    Uses ``model.transcribe(float32 ndarray)``.
    Processes **<=25 s** slices (Whisper truncates internally to ~30 s per forward).
    """
    import whisper as ow

    name = os.getenv("AV_SPEAKERBENCH_WHISPER_MODEL", "base").strip() or "base"
    dk = os.getenv("AV_SPEAKERBENCH_ASR_DEVICE", "").strip().lower()
    if dk == "cpu":
        dev = "cpu"
    elif dk in ("cuda", "gpu"):
        dev = "cuda"
    else:
        import torch as th

        dev = "cuda" if th.cuda.is_available() else "cpu"

    max_chunk_samples = max(3200, int(_parse_float("AV_SPEAKERBENCH_OW_MAX_SLICE_S", 25.0) * 16000))
    wav_16k = np.asarray(wav_16k, dtype=np.float32)
    model = _openai_whisper_model(name, dev)
    kw: dict[str, Any] = {"fp16": dev == "cuda"}
    if language:
        kw["language"] = language

    out: list[tuple[float, float, str, float | None]] = []
    sr = 16000
    for off in range(0, wav_16k.size, max_chunk_samples):
        piece = wav_16k[off : off + max_chunk_samples].copy()
        t_off_s = off / sr
        ow_res = model.transcribe(piece, **kw)  # type: ignore[arg-type]
        for seg in ow_res.get("segments", []) or []:
            txt = str(seg.get("text", "") or "").strip()
            if not txt:
                continue
            a = win_t0 + t_off_s + float(seg.get("start", 0))
            b = win_t0 + t_off_s + float(seg.get("end", 0))
            out.append((a, b, txt, None))
        full_txt = (ow_res.get("text") or "").strip()
        if full_txt and not ow_res.get("segments"):
            dur = len(piece) / sr
            out.append((win_t0 + t_off_s, win_t0 + t_off_s + dur, full_txt, None))

    return out


def _run_whisper_segments(
    backend: str, wav_slice_16k: np.ndarray, win_t0: float, language: str | None
) -> list[tuple[float, float, str, float | None]]:
    if backend == "faster_whisper":
        return _run_faster_whisper(wav_slice_16k, win_t0, language)
    if backend == "whisper":
        return _simple_decode_openai_whisper(wav_slice_16k, win_t0, language)
    return []


def _stub_env_segments(windows: list[tuple[float, float]]) -> list[AsrSegment]:
    if not allow_stub_asr_text_prompt():
        return []
    stub_txt = os.getenv("AV_SPEAKERBENCH_STUB_ASR_TEXT", "").strip()
    if not stub_txt or not windows:
        return []
    t0 = min(w[0] for w in windows)
    t1 = max(w[1] for w in windows)
    return [
        AsrSegment(
            t0=t0,
            t1=t1,
            text=stub_txt,
            conf=None,
        )
    ]


def asr_anchor_windows(
    wav_path: str | Path,
    vad_out: VadRunOutcome | None,
    *,
    backend_override: str | None = None,
    window_budget_mult: float = 1.0,
) -> AsrRunOutcome:
    """
    Load ``wav_path`` and transcribe anchor windows derived from ``vad_out`` (or fallback head chunk).

    Env:
      ``AV_SPEAKERBENCH_ASR_BACKEND``, ``AV_SPEAKERBENCH_WHISPER_MODEL`` (or dir path),
      ``AV_SPEAKERBENCH_FASTER_WHISPER_MODEL_DIR`` (local CTranslate2 tree, e.g. ModelScope download),
      ``AV_SPEAKERBENCH_ASR_DEVICE``, ``AV_SPEAKERBENCH_ASR_COMPUTE_TYPE`` (faster-whisper),
      ``AV_SPEAKERBENCH_ASR_LANGUAGE`` (ISO-639-1 code or empty for auto),
      ``AV_SPEAKERBENCH_ASR_MAX_WINDOWS``, ``AV_SPEAKERBENCH_ASR_MAX_CHUNK_S``,
      ``AV_SPEAKERBENCH_STUB_ASR_TEXT`` (only if ``AV_SPEAKERBENCH_ALLOW_SYNTHETIC_ASR`` is truthy),
      ``window_budget_mult`` scales ``AV_SPEAKERBENCH_ASR_MAX_WINDOWS`` (clamped by ``..._CAP``).
    """
    err_list: list[dict[str, Any]] = []
    lang = os.getenv("AV_SPEAKERBENCH_ASR_LANGUAGE", "").strip() or None
    raw_be = backend_override if backend_override is not None else os.getenv("AV_SPEAKERBENCH_ASR_BACKEND", "auto")
    backend = _select_backend(raw_be.strip().lower())

    path = Path(wav_path).resolve()
    if not path.is_file():
        return AsrRunOutcome([], backend, lang, [{"kind": "file_missing", "detail": str(path)}])

    if backend == "faster_whisper":
        try:
            import faster_whisper  # noqa: F401
        except ImportError as ie:
            err_list.append({"kind": "asr_import_missing", "detail": "faster_whisper", "import_error": str(ie)})
            backend = "stub"
    elif backend == "whisper":
        try:
            import whisper  # noqa: F401
        except ImportError as ie:
            err_list.append({"kind": "asr_import_missing", "detail": "whisper", "import_error": str(ie)})
            backend = "stub"

    try:
        x_raw, sr = load_waveform_mono(path)
    except Exception as e:  # noqa: BLE001
        return AsrRunOutcome([], backend, lang, [{"kind": "load_failed", "detail": str(e)}])

    windows = _windows_for_asr(vad_out, x_raw, sr, window_budget_mult=window_budget_mult)
    if vad_out:
        err_list.extend(vad_out.errors)

    if not windows:
        return AsrRunOutcome([], backend, lang, err_list)

    aggregated: list[AsrSegment] = []

    if backend == "stub":
        env_segs = _stub_env_segments(windows)
        if env_segs:
            aggregated.extend(env_segs)
        return AsrRunOutcome(aggregated, backend, lang, err_list)

    for t0_w, t1_w in windows:
        i0 = int(max(0, t0_w * sr))
        i1 = int(min(len(x_raw), max(i0 + 1, round(t1_w * sr))))
        chunk = np.asarray(x_raw[i0:i1], dtype=np.float64)

        wav_16k = mono_to_16k_f32(chunk, sr)
        if wav_16k.size < 800:
            continue
        try:
            rows = _run_whisper_segments(backend, wav_16k, win_t0=t0_w, language=lang)
            for tu0, tu1, text, lg in rows:
                if text:
                    aggregated.append(AsrSegment(t0=float(tu0), t1=float(tu1), text=text.strip(), conf=lg))
        except ImportError:
            stub_segs = _stub_env_segments([(t0_w, t1_w)])
            extra = [{"kind": "asr_import_missing", "detail": backend}]
            if not stub_segs and not allow_stub_asr_text_prompt():
                extra.append(
                    {
                        "kind": "synthetic_asr_disabled",
                        "detail": "set AV_SPEAKERBENCH_ALLOW_SYNTHETIC_ASR=1 to inject stub text",
                    }
                )
            return AsrRunOutcome(stub_segs, "stub", lang, [*err_list, *extra])
        except Exception as exc:  # noqa: BLE001
            err_list.append({"kind": "asr_window_failed", "detail": str(exc), "window": [t0_w, t1_w]})

    if not aggregated and backend != "stub":
        env_segs = _stub_env_segments(windows)
        if env_segs:
            aggregated = env_segs
            backend = "stub_env"
        else:
            err_list.append(
                {"kind": "empty_transcript", "detail": f"backend={backend} windows={len(windows)}"}
            )
            if not allow_stub_asr_text_prompt():
                err_list.append(
                    {
                        "kind": "synthetic_asr_disabled",
                        "detail": "no stub filler without AV_SPEAKERBENCH_ALLOW_SYNTHETIC_ASR",
                    }
                )

    return AsrRunOutcome(aggregated, backend, lang, err_list)


def format_asr_for_prompt(
    outcome: AsrRunOutcome,
    *,
    max_lines: int | None = None,
    max_line_chars: int = 220,
) -> str:
    lim = max_lines if max_lines is not None else _parse_int("AV_SPEAKERBENCH_ASR_MAX_LINES_PROMPT", 40)
    parts = []
    lang = outcome.language_hint or "(auto)"
    parts.append(f"asr_backend={outcome.backend} language_hint={lang}")
    shown = outcome.segments[: max(1, lim)]
    for seg in shown:
        line = f"[{seg.t0:.2f},{seg.t1:.2f}] {_truncate(seg.text, max_line_chars)}"
        parts.append(line)
    if len(outcome.segments) > lim:
        parts.append(f"[truncated_remaining={len(outcome.segments)-lim}]")
    return "\n".join(parts)
