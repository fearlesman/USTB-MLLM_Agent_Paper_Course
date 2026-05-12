"""
Tool **A1/A2** — full-clip **word-timestamp ASR** (faster-whisper) + **quote start** search.

Short benchmark clips: single ``transcribe`` with ``word_timestamps=True``.

Env: inherits ``AV_SPEAKERBENCH_ASR_*`` / ``FASTER_WHISPER_MODEL_DIR`` from ``audio_asr``;
add ``AV_SPEAKERBENCH_WORD_ASR_MAX_ITEMS`` (cap words in prompts, default 96).

**Decoupling from anchor ASR:** ``AV_SPEAKERBENCH_WORD_ASR_BACKEND`` (default ``faster_whisper``) controls
word timestamps only. The default runs **faster-whisper** whenever this tool runs, independent of
``AV_SPEAKERBENCH_ASR_BACKEND`` (e.g. openai ``whisper`` for anchor windows). Set to ``follow_global``
**Unifying with anchor / lexical Skills (see ``skills/impl``):** when
``AV_SPEAKERBENCH_ANCHOR_LEXICAL_FROM_WORD_LANE=1`` (default), ``anchor_window_asr`` and
``lexical_asr_bridge`` slice **the same** full-pass word stream into VAD windows instead of calling
``asr_anchor_windows`` again.

**Stub quote matching:** multi-token ``AV_SPEAKERBENCH_STUB_ASR_TEXT`` is split into spaced
``WordPiece`` rows so ``quote_start_time_seconds`` can align quoted phrases.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .audio_asr import (
    AsrRunOutcome,
    AsrSegment,
    allow_stub_asr_text_prompt,
    mono_to_16k_f32,
    _faster_whisper_model,
    _parse_float,
    _parse_int,
    _resolved_faster_whisper_model_ref,
)
from .audio_vad import VadRunOutcome, load_waveform_mono


@dataclass(frozen=True)
class WordPiece:
    t0: float
    t1: float
    text: str


@dataclass
class WordAsrOutcome:
    words: list[WordPiece]
    backend: str
    clip_duration_s: float
    language_hint: str | None
    errors: list[dict[str, Any]]


def norm_text_token(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def words_joined_normalized(words: Sequence[WordPiece]) -> str:
    return norm_text_token(" ".join(w.text for w in words))


def stub_text_to_word_pieces(text: str, clip_dur_s: float) -> list[WordPiece]:
    """Split synthetic stub transcript into per-token WordPieces so ``quote_start_time_seconds`` can match phrases."""
    s = (text or "").strip()
    if not s:
        return []
    toks = [t for t in re.split(r"\s+", s) if t]
    dur = max(0.01, float(clip_dur_s))
    if len(toks) <= 1:
        return [WordPiece(0.0, dur, s)]
    n = len(toks)
    span = dur / n
    out: list[WordPiece] = []
    for i, t in enumerate(toks):
        t0 = i * span
        t1 = dur if i == n - 1 else (i + 1) * span
        out.append(WordPiece(t0=t0, t1=t1, text=t))
    return out


def segments_from_word_lane_for_vad(
    wo: WordAsrOutcome,
    vad: VadRunOutcome | None,
) -> AsrRunOutcome:
    """
    Build VAD-window ASR segments by slicing **cached word timestamps** (one full-pass ASR),
    avoiding a second Whisper run in ``asr_anchor_windows``.
    """
    errs: list[dict[str, Any]] = []
    if not wo.words:
        return AsrRunOutcome([], wo.backend, wo.language_hint, errs)
    be = f"word_lane_v1+{wo.backend}"
    if vad is not None and vad.segments:
        segs: list[AsrSegment] = []
        for vs in sorted(vad.segments, key=lambda s: float(s.t0)):
            t0, t1 = float(vs.t0), float(vs.t1)
            parts: list[str] = []
            for w in wo.words:
                mid = 0.5 * (float(w.t0) + float(w.t1))
                if mid >= t0 and mid <= t1 and w.text.strip():
                    parts.append(w.text.strip())
            txt = " ".join(parts).strip()
            segs.append(AsrSegment(t0=t0, t1=t1, text=txt or "(no_words_in_vad_window)"))
        return AsrRunOutcome(segs, be, wo.language_hint, errs)
    full = " ".join(w.text.strip() for w in wo.words if w.text.strip())
    return AsrRunOutcome(
        [AsrSegment(0.0, max(0.01, wo.clip_duration_s), full)],
        be,
        wo.language_hint,
        errs,
    )


def quote_start_time_seconds(
    words: Sequence[WordPiece],
    phrase: str,
    *,
    norm: Callable[[str], str] | None = None,
) -> float | None:
    """First start time where ``phrase`` matches as contiguous substring in normalized word stream."""
    nf = norm or norm_text_token
    target = nf(phrase)
    if not target or not words:
        return None
    tok = [nf(w.text) for w in words]
    n = len(tok)
    need = target.split()
    if not need:
        return float(words[0].t0)
    m = len(need)
    for i in range(0, max(1, n - m + 1)):
        if i + m > n:
            break
        if tok[i : i + m] == need:
            return float(words[i].t0)
    big = nf(" ".join(tok))
    if target in big:
        fused: list[tuple[float, float, str]] = []
        buf_t0 = words[0].t0
        buf_t1 = words[0].t1
        buf_parts: list[str] = []
        for w in words:
            buf_parts.append(w.text.strip())
            buf_t1 = w.t1
            if target in nf(" ".join(buf_parts)):
                return float(buf_t0)
            if len(" ".join(buf_parts)) > len(target) * 3:
                buf_parts = buf_parts[-m:]
                buf_t0 = w.t0
    return None


def transcribe_words(
    wav_path: str | Path,
    *,
    language: str | None = None,
    max_words: int | None = None,
) -> WordAsrOutcome:
    """Run faster-whisper on **entire** decoded clip; flatten segment words."""
    path = Path(wav_path).resolve()
    errs: list[dict[str, Any]] = []
    if not path.is_file():
        return WordAsrOutcome([], "none", 0.0, language, [{"kind": "file_missing", "detail": str(path)}])

    lim = max_words if max_words is not None else _parse_int("AV_SPEAKERBENCH_WORD_ASR_MAX_ITEMS", 96)

    try:
        x, sr = load_waveform_mono(path)
    except Exception as e:  # noqa: BLE001
        return WordAsrOutcome([], "none", 0.0, language, [{"kind": "load_failed", "detail": str(e)}])

    wav_16k = mono_to_16k_f32(x, sr)
    dur = float(len(wav_16k) / 16000.0)

    raw_be = os.getenv("AV_SPEAKERBENCH_ASR_BACKEND", "auto").strip().lower()
    word_mode = os.getenv("AV_SPEAKERBENCH_WORD_ASR_BACKEND", "faster_whisper").strip().lower()
    follow_global = word_mode in ("follow_global", "follow")
    # Default: word timestamps via faster-whisper regardless of anchor ASR (avoids stub word lane when ASR_BACKEND=whisper).
    use_fw = (
        raw_be in ("", "auto", "faster_whisper", "fw") if follow_global else True
    )

    if use_fw:
        try:
            import faster_whisper  # noqa: F401
        except ImportError as ie:
            errs.append({"kind": "asr_import_missing", "detail": "faster_whisper", "import_error": str(ie)})
            return WordAsrOutcome([], "stub", dur, language, errs)

    if not use_fw:
        errs.append(
            {
                "kind": "word_asr_backend_unsupported",
                "detail": "follow_global is set but AV_SPEAKERBENCH_ASR_BACKEND is not faster-whisper; "
                "unset AV_SPEAKERBENCH_WORD_ASR_BACKEND or set ASR_BACKEND=faster_whisper for word timestamps",
            }
        )
        if allow_stub_asr_text_prompt():
            st = os.getenv("AV_SPEAKERBENCH_STUB_ASR_TEXT", "").strip()
            if st:
                pieces = stub_text_to_word_pieces(st, dur)
                return WordAsrOutcome(pieces, "stub_env", dur, language, errs)
        return WordAsrOutcome([], raw_be or "stub", dur, language, errs)

    model_ref = _resolved_faster_whisper_model_ref()
    device = os.getenv("AV_SPEAKERBENCH_ASR_DEVICE", "cpu").strip() or "cpu"
    ctype = os.getenv("AV_SPEAKERBENCH_ASR_COMPUTE_TYPE", "default").strip() or "default"
    if ctype == "default":
        ctype = "int8" if device == "cpu" else "float16"
    beam = _parse_int("AV_SPEAKERBENCH_WORD_ASR_BEAM_SIZE", 5)

    try:
        model = _faster_whisper_model(model_ref, device, ctype)
        lang = language if language else None
        segs_it, info = model.transcribe(
            wav_16k.astype(np.float32),
            beam_size=max(1, beam),
            language=lang,
            vad_filter=False,
            word_timestamps=True,
        )
        lang_out = getattr(info, "language", None) or language
        flat: list[WordPiece] = []
        for seg in segs_it:
            ws = getattr(seg, "words", None) or []
            for w in ws:
                try:
                    t0 = float(getattr(w, "start", 0.0))
                    t1 = float(getattr(w, "end", t0))
                    tx = str(getattr(w, "word", "") or "").strip()
                except (TypeError, ValueError):
                    continue
                if not tx:
                    continue
                flat.append(WordPiece(t0=t0, t1=t1, text=tx))
                if len(flat) >= max(512, lim * 4):
                    break
            if len(flat) >= max(512, lim * 4):
                break
        flat = flat[:lim]
        if flat:
            return WordAsrOutcome(flat, "faster_whisper", dur, lang_out if isinstance(lang_out, str) else language, errs)
        errs.append({"kind": "empty_word_stream", "detail": "faster_whisper word_timestamps"})
    except Exception as e:  # noqa: BLE001
        errs.append({"kind": "word_asr_failed", "detail": str(e)})

    if allow_stub_asr_text_prompt():
        st = os.getenv("AV_SPEAKERBENCH_STUB_ASR_TEXT", "").strip()
        if st:
            pieces = stub_text_to_word_pieces(st, dur)
            return WordAsrOutcome(pieces, "stub_env", dur, language, errs)

    return WordAsrOutcome([], "stub", dur, language, errs)


def format_words_for_prompt(wo: WordAsrOutcome, *, max_lines: int | None = None) -> str:
    cap = max_lines if max_lines is not None else _parse_int("AV_SPEAKERBENCH_WORD_ASR_MAX_LINES", 48)
    parts = [
        "[word_lane_asr]",
        f"backend={wo.backend} duration_s={wo.clip_duration_s:.2f} n_words_shown={min(len(wo.words), cap)}",
    ]
    for w in wo.words[:cap]:
        parts.append(f"{w.t0:.2f}-{w.t1:.2f} {w.text}")
    if len(wo.words) > cap:
        parts.append(f"[truncated n_total>={len(wo.words)}]")
    return "\n".join(parts) + "\n"
