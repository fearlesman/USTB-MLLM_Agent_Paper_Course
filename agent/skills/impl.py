from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from tools.audio_asr import (
    AsrRunOutcome,
    allow_stub_asr_text_prompt,
    asr_anchor_windows,
    format_asr_for_prompt,
)
from tools.audio_asr_words import (
    WordAsrOutcome,
    format_words_for_prompt,
    quote_start_time_seconds,
    segments_from_word_lane_for_vad,
    transcribe_words,
)
from tools.audio_diar import (
    DiarRunOutcome,
    diarize_wav_path,
    diar_with_vad_fallback,
    format_diar_for_prompt,
    vad_proxy_prompt_footer,
)
from tools.audio_vad import VadRunOutcome, format_segments_for_prompt, vad_segments_from_wav_path
from tools.benchmark_timecode import dataset_span_seconds
from tools.speech_turn_sheet import build_turn_sheet, format_turn_sheet
from tools.video_people_snap import format_people_snap, snap_and_count_people

from . import triggers
from .types import SkillContext, SkillOutcome

SkillFn = Callable[[SkillContext, bool], SkillOutcome]


def _inject_requested() -> bool:
    return os.getenv("AV_SPEAKERBENCH_SKILL_INJECT", "").strip().lower() in ("1", "true", "yes")


def _wav_path(ctx: SkillContext) -> tuple[str | None, bool]:
    ap = Path(ctx.audio_path)
    if ap.suffix.lower() == ".wav" and ap.is_file():
        return str(ap.resolve()), True
    cp = Path(ctx.combined_path)
    return None, cp.is_file()


def _vad_run(ctx: SkillContext) -> VadRunOutcome | None:
    wav, _ = _wav_path(ctx)
    if not wav:
        return None
    return vad_segments_from_wav_path(wav)


def _norm_txt(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _tag_lines(
    *,
    purpose: str,
    confidence: str,
    disclaimer: str | None = None,
) -> list[str]:
    lines = [f"purpose={purpose}", f"confidence={confidence}"]
    if disclaimer:
        lines.append(f"disclaimer={disclaimer}")
    return lines


def _stem_quote_phrases(question: dict[str, Any]) -> list[str]:
    stem = str(question.get("question", ""))
    found: list[str] = []
    for m in triggers.STEM_QUOTES_PAT.finditer(stem):
        frag = next(g for g in m.groups() if g)
        t = frag.strip()
        if t and t not in found:
            found.append(t)
    return found


def _file_size_bytes(path: str | None) -> int | None:
    if not path:
        return None
    p = Path(path)
    try:
        return int(p.stat().st_size) if p.is_file() else None
    except OSError:
        return None


_pipeline_audio_key: tuple[str, int, int] | None = None
_cached_word_asr: WordAsrOutcome | None = None
_cached_diar: DiarRunOutcome | None = None


def _pipeline_audio_cache_reset() -> None:
    global _pipeline_audio_key, _cached_word_asr, _cached_diar
    _pipeline_audio_key = None
    _cached_word_asr = None
    _cached_diar = None


def _wav_cache_key(ap: str) -> tuple[str, int]:
    try:
        p = Path(ap)
        return (str(p.resolve()), p.stat().st_mtime_ns)
    except OSError:
        return (ap, 0)


def _ensure_audio_cache(ap: str, question: dict[str, Any] | None) -> None:
    global _pipeline_audio_key, _cached_word_asr, _cached_diar
    mx = triggers.effective_word_asr_max_items(question or {})
    k = (*_wav_cache_key(ap), mx)
    if _pipeline_audio_key != k:
        _cached_word_asr = None
        _cached_diar = None
        _pipeline_audio_key = k


def _get_cached_word_asr(ap: str, question: dict[str, Any]) -> WordAsrOutcome:
    global _cached_word_asr
    _ensure_audio_cache(ap, question)
    if _cached_word_asr is None:
        lang = os.getenv("AV_SPEAKERBENCH_ASR_LANGUAGE", "").strip() or None
        mx = triggers.effective_word_asr_max_items(question)
        _cached_word_asr = transcribe_words(ap, language=lang, max_words=mx)
    return _cached_word_asr


def _get_cached_diar(ap: str, question: dict[str, Any]) -> DiarRunOutcome:
    global _cached_diar
    _ensure_audio_cache(ap, question)
    if _cached_diar is None:
        _cached_diar = diarize_wav_path(ap)
    return _cached_diar


def _anchor_lexical_from_word_lane_enabled() -> bool:
    """One full-pass word ASR → VAD slices; avoids a second Whisper pass (see ``audio_asr_words``)."""
    return os.getenv("AV_SPEAKERBENCH_ANCHOR_LEXICAL_FROM_WORD_LANE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolve_asr_for_anchor_lexical(
    ap: str,
    vad_out: VadRunOutcome | None,
    q: dict[str, Any],
) -> tuple[AsrRunOutcome | None, str, list[dict[str, Any]]]:
    extra: list[dict[str, Any]] = []
    if not ap:
        return None, "none", extra
    if _anchor_lexical_from_word_lane_enabled():
        w = _get_cached_word_asr(ap, q)
        extra.extend(w.errors)
        if w.words:
            ao = segments_from_word_lane_for_vad(w, vad_out)
            extra.extend(ao.errors)
            return ao, "word_lane_v1", extra
    wmult = triggers.asr_window_budget_multiplier(q)
    ao = asr_anchor_windows(
        ap,
        vad_out if vad_out is not None else None,
        window_budget_mult=wmult,
    )
    extra.extend(ao.errors)
    return ao, "vad_windows", extra


def skill_clip_span_meta(ctx: SkillContext, inject: bool) -> SkillOutcome:
    """Inject dataset ``start_time`` / ``end_time`` (benchmark clip window in authoring)."""
    oid = SkillOutcome("clip_span_meta", invoke_tag="ran", bottleneck_tags=["alignment_pending"])
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.bottleneck_tags = []
        return oid
    if not triggers.should_emit_clip_span_meta(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    st = q.get("start_time", "")
    et = q.get("end_time", "")
    st_s, et_s, span = dataset_span_seconds(q)
    extra = ""
    if span is not None and st_s is not None and et_s is not None:
        extra = f" start_s={st_s:.2f} end_s={et_s:.2f} span_s={span:.2f}"
    oid.injected_text = "\n".join(
        [
            "[clip_dataset_span]",
            *_tag_lines(
                purpose="benchmark_authored_clip_time_window",
                confidence="high",
            ),
            f"start_time={st!s} end_time={et!s}{extra}",
        ]
    ) + "\n"
    oid.invoke_tag = "injected"
    oid.bottleneck_tags = []
    return oid


def skill_lexical_asr_bridge(ctx: SkillContext, inject: bool) -> SkillOutcome:
    """Match stem quotes against ASR spans (substring; normalized)."""
    oid = SkillOutcome(
        "lexical_asr_bridge",
        invoke_tag="stub",
        bottleneck_tags=["alignment_pending"],
    )
    q = dict(ctx.question)
    phrases = _stem_quote_phrases(q)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.bottleneck_tags = []
        return oid
    if not triggers.should_emit_lexical_asr_bridge(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    if not phrases:
        oid.invoke_tag = "no_quotes_in_stem"
        oid.bottleneck_tags = []
        return oid

    ap, _ok = _wav_path(ctx)
    if not ap:
        oid.invoke_tag = "stub"
        oid.injected_text = "[lexical_asr_bridge] audio_wav_missing quotes_only_in_stem\n"
        return oid

    vad_out = _vad_run(ctx)
    if vad_out is not None:
        oid.errors.extend(vad_out.errors)

    ao, asr_src, xerrs = _resolve_asr_for_anchor_lexical(
        ap,
        vad_out if vad_out else None,
        q,
    )
    oid.errors.extend(xerrs)
    lines = [
        "[lexical_asr_bridge]",
        *_tag_lines(
            purpose="check_whether_quoted_phrase_matches_local_asr_span",
            confidence="high" if ao is not None and ao.backend in ("faster_whisper", "whisper") else "medium",
            disclaimer="quote_hit_means_substring_alignment_not_full_semantic_equivalence",
        ),
        f"asr_backend={ao.backend if ao else 'n/a'} asr_source={asr_src}",
    ]
    pn = [_norm_txt(p) for p in phrases[:16]]
    hits = 0
    segs = ao.segments if ao is not None else []
    for raw_p, np in zip(phrases[:16], pn, strict=False):
        if not np:
            continue
        matched = False
        for seg in segs:
            nt = _norm_txt(seg.text)
            if np in nt or nt in np:
                lines.append(
                    f"hit phrase=`{raw_p[:80]}` span_s=[{seg.t0:.2f},{seg.t1:.2f}] preview={seg.text[:120]!s}"
                )
                hits += 1
                matched = True
                break
        if not matched:
            lines.append(f"miss phrase=`{raw_p[:80]}` (no ASR substring match)")

    oid.injected_text = "\n".join(lines) + "\n"
    if hits:
        oid.invoke_tag = "hits_injected"
        oid.bottleneck_tags = []
        be = ao.backend if ao is not None else ""
        if be in ("stub", "stub_env") or (be.startswith("word_lane") and "stub_env" in be):
            oid.bottleneck_tags.append("evidence_synthetic")
    else:
        oid.invoke_tag = "no_hits"
    oid.bottleneck_tags = sorted(set(oid.bottleneck_tags))
    return oid


def skill_anchor_window_asr(ctx: SkillContext, inject: bool) -> SkillOutcome:
    """VAD-window ASR via ``tools/audio_asr``; gated by weak-bucket heuristics (see ``triggers``)."""
    ap, clip_ok = _wav_path(ctx)
    oid = SkillOutcome(
        "anchor_window_asr",
        bottleneck_tags=["perception_pending"],
        invoke_tag="stub",
    )
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    if not triggers.should_emit_anchor_window_asr(dict(ctx.question)):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid

    vad_out = _vad_run(ctx)
    vad_line = ""
    if vad_out is not None:
        vad_line = "vad: " + format_segments_for_prompt(vad_out) + "\n"
        oid.errors.extend(vad_out.errors)

    asr_blk = ""
    ao: AsrRunOutcome | None = None
    stub_preview_line = ""
    asr_src = "none"
    if ap:
        ao, asr_src, xerrs = _resolve_asr_for_anchor_lexical(
            ap,
            vad_out if vad_out is not None else None,
            dict(ctx.question),
        )
        oid.errors.extend(xerrs)
        if ao is not None and ao.segments and any(s.text.strip() for s in ao.segments):
            asr_src_note = f"asr_source={asr_src}\n" if asr_src != "none" else ""
            asr_blk = asr_src_note + "asr:\n" + format_asr_for_prompt(ao) + "\n"
        else:
            sp = os.getenv("AV_SPEAKERBENCH_STUB_ASR_TEXT", "").strip()
            if sp and allow_stub_asr_text_prompt():
                stub_preview_line = (
                    "env_stub_preview="
                    + (sp[:180] + "…" if len(sp) > 180 else sp)
                    + "\n"
                )

    tx = bool(ao and ao.segments and any(s.text.strip() for s in ao.segments))
    be = ao.backend if ao is not None else ""
    oid.injected_text = (
        "[anchor_window_asr]\n"
        + "\n".join(
            _tag_lines(
                purpose="localized_transcript_evidence_near_speech_windows",
                confidence="high" if tx and ao is not None and ao.backend in ("faster_whisper", "whisper") else "medium",
                disclaimer=(
                    "word_lane_or_stub_backends_may_be_proxy_alignment"
                    if (ao is not None and ao.backend.startswith("word_lane")) or stub_preview_line
                    else None
                ),
            )
        )
        + "\n"
        f"audio_wav_present={bool(ap)} audio_path={(ap or 'n/a')} clip_fallback_exists={clip_ok}\n"
        f"{vad_line}"
        f"{asr_blk}"
        f"{stub_preview_line}"
    )

    if tx and be.startswith("word_lane"):
        oid.bottleneck_tags = [b for b in oid.bottleneck_tags if b != "perception_pending"]
        if "stub_env" in be:
            oid.bottleneck_tags.append("evidence_synthetic")
    elif tx and be in ("stub", "stub_env"):
        oid.bottleneck_tags.append("evidence_synthetic")
    if stub_preview_line:
        oid.bottleneck_tags.append("evidence_synthetic")

    if tx and ao is not None:
        if be.startswith("word_lane"):
            oid.invoke_tag = "asr_injected" if "faster_whisper" in be else "asr_stub_text"
        elif be in ("whisper", "faster_whisper"):
            oid.invoke_tag = "asr_injected"
            oid.bottleneck_tags = [b for b in oid.bottleneck_tags if b not in ("perception_pending", "evidence_synthetic")]
        else:
            oid.invoke_tag = "asr_stub_text"
    elif stub_preview_line:
        oid.invoke_tag = "asr_stub_text"
    elif vad_out and vad_out.segments:
        oid.invoke_tag = "vad_injected"
    else:
        oid.invoke_tag = "stub"
    oid.bottleneck_tags = sorted(set(oid.bottleneck_tags))
    return oid


def _vad_pairwise_overlap_s(segments: list) -> float:
    """Sum of temporal overlaps between distinct VAD intervals (coarse overlap pressure)."""
    segs = sorted(((float(s.t0), float(s.t1)) for s in segments), key=lambda x: x[0])
    tot = 0.0
    for i, (a0, a1) in enumerate(segs):
        for b0, b1 in segs[i + 1 :]:
            lo = max(a0, b0)
            hi = min(a1, b1)
            tot += max(0.0, hi - lo)
    return tot


def skill_diar_binding(ctx: SkillContext, inject: bool) -> SkillOutcome:
    """Diarization tool (``pyannote`` optional). ``triggers.TARGETS_DOC['diar_binding']``."""
    oid = SkillOutcome("diar_binding", bottleneck_tags=["perception_pending"], invoke_tag="stub")
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    if not triggers.should_emit_diar_binding(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid

    ap, _ = _wav_path(ctx)
    if not ap:
        oid.invoke_tag = "stub"
        oid.injected_text = "[diarization] audio_wav_missing speaker_spans=[]\n"
        return oid

    d_out = _get_cached_diar(ap, dict(ctx.question))
    oid.errors.extend(d_out.errors)
    if d_out.backend not in ("pyannote", "pyannote_api") or not d_out.segments:
        oid.invoke_tag = "stub_backend"
        oid.injected_text = "\n".join(
            [
                "[diarization]",
                *_tag_lines(
                    purpose="speaker_time_spans_for_binding",
                    confidence="low",
                    disclaimer="low_confidence_diarization_suppressed_from_identity_binding",
                ),
                "backend=stub_or_low_confidence speaker_spans=[]",
            ]
        ) + "\n"
        return oid
    blob = format_diar_for_prompt(d_out)
    extra = ""
    if d_out.segments:
        uniq = len({s.label for s in d_out.segments})
        extra = f"unique_speaker_label_count={uniq}\n"
    oid.injected_text = (
        "[diarization]\n"
        + "\n".join(
            _tag_lines(
                purpose="speaker_time_spans_for_binding",
                confidence="high",
                disclaimer="speaker_labels_are_cluster_ids_not_true_person_names",
            )
        )
        + "\n"
        + blob
        + "\n"
        + extra
    )
    if d_out.backend in ("pyannote", "pyannote_api") and d_out.segments:
        oid.invoke_tag = "diar_injected"
        oid.bottleneck_tags = []
    else:
        oid.invoke_tag = "stub"
    return oid


def skill_asr_word_lane(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("asr_word_lane", invoke_tag="stub", bottleneck_tags=["perception_pending"])
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.bottleneck_tags = []
        return oid
    if not triggers.should_emit_asr_word_lane(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    ap, _ = _wav_path(ctx)
    if not ap:
        oid.invoke_tag = "stub"
        oid.bottleneck_tags = []
        oid.injected_text = "[word_lane_asr] audio_wav_missing\n"
        return oid
    w = _get_cached_word_asr(ap, dict(ctx.question))
    oid.errors.extend(w.errors)
    lead = "\n".join(
        [
            "[word_lane_tags]",
            *_tag_lines(
                purpose="word_timestamp_alignment_for_quotes_and_counting",
                confidence="high" if w.backend == "faster_whisper" else ("medium" if w.words else "low"),
                disclaimer="synthetic_or_proxy_word_timestamps_are_lower_confidence" if w.backend == "stub_env" else None,
            ),
        ]
    ) + "\n"
    oid.injected_text = lead + format_words_for_prompt(w)
    if w.words and w.backend == "faster_whisper":
        oid.invoke_tag = "word_asr_injected"
        oid.bottleneck_tags = []
    elif w.words:
        oid.invoke_tag = w.backend
        oid.bottleneck_tags = []
    else:
        oid.invoke_tag = "stub"
    if w.backend == "stub_env":
        oid.bottleneck_tags.append("evidence_synthetic")
    oid.bottleneck_tags = sorted(set(oid.bottleneck_tags))
    return oid


def skill_anchor_quote_time(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("anchor_quote_time", invoke_tag="stub", bottleneck_tags=["alignment_pending"])
    q = dict(ctx.question)
    phrases = _stem_quote_phrases(q)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.bottleneck_tags = []
        return oid
    if not triggers.should_emit_anchor_quote_time(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    if not phrases:
        oid.invoke_tag = "no_quotes_in_stem"
        oid.bottleneck_tags = []
        return oid
    ap, _ = _wav_path(ctx)
    if not ap:
        oid.invoke_tag = "stub"
        oid.bottleneck_tags = []
        oid.injected_text = "[anchor_quote_s] audio_wav_missing\n"
        return oid
    w = _get_cached_word_asr(ap, dict(ctx.question))
    oid.errors.extend(w.errors)
    ph = phrases[0]
    ts = quote_start_time_seconds(w.words, ph)
    if ts is None:
        oid.invoke_tag = "quote_not_found_in_asr"
        oid.injected_text = "\n".join(
            [
                "[anchor_quote_s]",
                *_tag_lines(
                    purpose="map_quoted_phrase_to_first_time_anchor",
                    confidence="low",
                    disclaimer="quote_not_found_in_current_word_asr",
                ),
                f"phrase={ph!s} t_first_s=n/a",
            ]
        ) + "\n"
        return oid
    oid.injected_text = "\n".join(
        [
            "[anchor_quote_s]",
            *_tag_lines(
                purpose="map_quoted_phrase_to_first_time_anchor",
                confidence="high" if w.backend == "faster_whisper" else "medium",
                disclaimer="timestamp_depends_on_word_asr_alignment" if w.backend != "faster_whisper" else None,
            ),
            f"phrase={ph!s} t_first_s={ts:.2f}",
        ]
    ) + "\n"
    oid.invoke_tag = "aligned" if w.backend == "faster_whisper" else w.backend
    oid.bottleneck_tags = []
    if w.backend == "stub_env":
        oid.bottleneck_tags = ["evidence_synthetic"]
    return oid


def skill_turn_order_sheet(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("turn_order_sheet", invoke_tag="stub", bottleneck_tags=["perception_pending"])
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.bottleneck_tags = []
        return oid
    if not triggers.should_emit_turn_order_sheet(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    ap, _ = _wav_path(ctx)
    if not ap:
        oid.invoke_tag = "stub"
        oid.injected_text = "[turn_order_sheet] audio_wav_missing\n"
        return oid
    vad_out = _vad_run(ctx)
    raw_d = _get_cached_diar(ap, dict(ctx.question))
    oid.errors.extend(raw_d.errors)
    if raw_d.backend not in ("pyannote", "pyannote_api") or not raw_d.segments:
        oid.invoke_tag = "stub_backend"
        oid.injected_text = "\n".join(
            [
                "[turn_order_sheet]",
                *_tag_lines(
                    purpose="speaker_turn_order_from_diar_plus_word_spans",
                    confidence="low",
                    disclaimer="requires_high_confidence_diarization",
                ),
                "diar_backend_low_confidence",
            ]
        ) + "\n"
        return oid
    d_out = diar_with_vad_fallback(raw_d, vad_out)
    words = _get_cached_word_asr(ap, dict(ctx.question))
    oid.errors.extend(words.errors)
    sheet = build_turn_sheet(d_out, words)
    oid.injected_text = format_turn_sheet(sheet)
    oid.invoke_tag = "injected" if sheet.turns else "stub_empty"
    if sheet.turns:
        oid.bottleneck_tags = []
    return oid


def skill_viz_people_anchor(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("viz_people_anchor", invoke_tag="stub", bottleneck_tags=["perception_pending", "stub_backend"])
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.bottleneck_tags = []
        return oid
    if not triggers.should_emit_viz_people_anchor(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    vp = Path(ctx.video_path)
    if not vp.is_file():
        vp = Path(ctx.combined_path)
    if not vp.is_file():
        oid.invoke_tag = "stub"
        oid.injected_text = "[viz_people_snap] video_path_missing\n"
        return oid
    ap, _ = _wav_path(ctx)
    t_anchor: float | None = None
    heuristic_note = ""
    quote_resolved = False
    if ap:
        phrases = _stem_quote_phrases(q)
        w = _get_cached_word_asr(ap, dict(ctx.question))
        oid.errors.extend(w.errors)
        if phrases:
            tq = quote_start_time_seconds(w.words, phrases[0])
            if tq is not None:
                t_anchor = tq
                quote_resolved = True
    if t_anchor is None:
        vo = _vad_run(ctx) if ap else None
        if vo is not None:
            oid.errors.extend(vo.errors)
            t_anchor = 0.5 * float(vo.duration_s)
        else:
            _st_s, _et_s, span = dataset_span_seconds(q)
            if span is not None and span > 0:
                t_anchor = max(0.0, 0.5 * float(span))
            else:
                t_anchor = 0.5
                oid.errors.append({"kind": "viz_anchor_fallback", "detail": "no_duration_using_t=0.5s"})
            heuristic_note = "anchor_heuristic=video_mid_or_dataset_span_no_audio_alignment\n"
    delta = triggers.effective_viz_anchor_delta_s(q, quote_time_resolved=quote_resolved)
    times = sorted({max(0.0, float(t_anchor) + d) for d in (-delta, 0.0, delta)})
    snap = snap_and_count_people(vp, times)
    oid.errors.extend(snap.errors)
    head = "\n".join(
        [
            "[viz_people_anchor_tags]",
            *_tag_lines(
                purpose="visible_people_count_near_anchor_time",
                confidence="high" if any(c is not None for c in snap.person_counts) else "medium",
                disclaimer=(
                    "anchor_time_uses_visual_or_audio_heuristic_without_exact_quote_alignment"
                    if heuristic_note
                    else "tracked_ids_are_detector_track_ids_not_identity_names"
                ),
            ),
        ]
    ) + "\n"
    oid.injected_text = head + heuristic_note + format_people_snap(snap)
    oid.invoke_tag = "frames_injected" if snap.n_frames_extracted_ok > 0 else "stub"
    if any(c is not None for c in snap.person_counts):
        oid.bottleneck_tags = ["perception_pending"]
    return oid


# Stable display order matching the recommendation table (EvidencePack merges at end logically).
_SKILL_REGISTRY: list[tuple[str, SkillFn]] = [
    ("clip_span_meta", skill_clip_span_meta),
    ("asr_word_lane", skill_asr_word_lane),
    ("anchor_quote_time", skill_anchor_quote_time),
    ("anchor_window_asr", skill_anchor_window_asr),
    ("lexical_asr_bridge", skill_lexical_asr_bridge),
    ("diar_binding", skill_diar_binding),
    ("turn_order_sheet", skill_turn_order_sheet),
    ("viz_people_anchor", skill_viz_people_anchor),
]


def _evidence_placement_mode() -> str:
    raw = os.getenv("AV_SPEAKERBENCH_EVIDENCE_PLACEMENT", "tail").strip().lower()
    return raw if raw in ("both", "tail") else "tail"


def _should_keep_injected_block(skill_id: str, out: SkillOutcome) -> bool:
    """
    Default prompt policy: keep high-yield evidence, drop weak / low-confidence blocks.

    Trace still records every Skill status via ``skills_invoked`` even when the actual
    prompt injection is suppressed here.
    """
    tag = out.invoke_tag
    if tag.startswith("skipped_"):
        return False
    if tag in ("stub", "stub_empty", "stub_backend", "no_quotes_in_stem", "quote_not_found_in_asr", "no_hits"):
        return False
    high_value = {
        "clip_span_meta",
        "asr_word_lane",
        "anchor_quote_time",
        "anchor_window_asr",
        "lexical_asr_bridge",
        "diar_binding",
        "turn_order_sheet",
        "viz_people_anchor",
    }
    if skill_id in high_value:
        return True
    return bool(out.injected_text and tag not in ("stub", "stub_empty", "stub_backend"))


def _grounding_intro_text() -> str:
    c = os.getenv("AV_SPEAKERBENCH_EVIDENCE_PREFIX_TEXT", "").strip()
    if c:
        return c
    return (
        "[external_evidence_grounding] If numbered evidence below conflicts with raw "
        "audio/video intuition, rely on timestamped excerpts when choosing A/B/C/D."
    )


def run_skill_pipeline(ctx: SkillContext) -> tuple[str, list[str], list[str], list[dict]]:
    """
    Execute registered Skills.

    Prompt changes only when ``AV_SPEAKERBENCH_SKILL_INJECT=1`` — default leaves MC prompt identical.

    ``AV_SPEAKERBENCH_EVIDENCE_PLACEMENT=both`` prefixes a short grounding line before MC stem;
    structured evidence stays after the stem (suffix). ``tail`` (default): evidence-only suffix.

    Returns:
      (final_prompt, skills_invoked_tags, bottleneck_tags_unique, flattened_errors).
    """
    _pipeline_audio_cache_reset()
    inject = _inject_requested()
    allow = os.getenv("AV_SPEAKERBENCH_SKILLS_ALLOWLIST", "").strip().lower()
    allow_tokens = {x.strip().lower() for x in allow.split(",") if x.strip()}

    parts: list[str] = []
    tags: list[str] = []
    b_accum: list[str] = []
    errs: list[dict[str, Any]] = []

    for sid, fn in _SKILL_REGISTRY:
        if allow and allow != "all" and sid.lower() not in allow_tokens:
            continue
        out = fn(ctx, inject)
        tags.append(f"{out.skill_id}:{out.invoke_tag}")
        b_accum.extend(out.bottleneck_tags)
        errs.extend(out.errors)
        if out.injected_text and _should_keep_injected_block(sid, out):
            parts.append(out.injected_text.rstrip())

    evidence = ""
    if parts and inject:
        evidence = "### Structured_skill_evidence\n" + "\n".join(parts) + "\n"

    base_mc = ctx.question_prompt
    if inject and _evidence_placement_mode() == "both" and evidence.strip():
        base_mc = _grounding_intro_text().rstrip() + "\n\n" + ctx.question_prompt

    final = base_mc
    if evidence.strip():
        final = (base_mc.rstrip() + "\n\n" + evidence.rstrip()).rstrip()

    return final, tags, sorted(set(b_accum)), errs
