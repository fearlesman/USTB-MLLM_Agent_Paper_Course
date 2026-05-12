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
from tools.audio_pitch_segments import format_f0_sheet, pitch_median_over_segments
from tools.audio_prosody import discrete_prosody_over_vad
from tools.audio_rate_proxy import format_rate_sheet, words_per_second_sheet
from tools.audio_rms_meter import rms_peak_meter
from tools.audio_speak_duration import duration_per_diar_label, format_speak_duration
from tools.audio_vad import VadRunOutcome, format_segments_for_prompt, vad_segments_from_wav_path
from tools.benchmark_timecode import dataset_span_seconds
from tools.media_probe import format_probe_for_prompt, probe_media_file
from tools.speech_turn_sheet import build_turn_sheet, format_turn_sheet
from tools.video_people_snap import format_people_snap, snap_and_count_people

from . import triggers
from .types import SkillContext, SkillOutcome

SkillFn = Callable[[SkillContext, bool], SkillOutcome]


def _inject_requested() -> bool:
    return os.getenv("AV_SPEAKERBENCH_SKILL_INJECT", "").strip().lower() in ("1", "true", "yes")


def skill_meta_banner(ctx: SkillContext, inject: bool) -> SkillOutcome:
    """Minimal task taxonomy line for Omni (low token).

    Targets: always-on meta; cite weak buckets separately in EXPERIMENT manifests.
    """
    tid = ctx.question.get("task_id", "")
    cat = ctx.question.get("category", "")
    sub = ctx.question.get("sub_category", "")
    oid = SkillOutcome("meta_banner", invoke_tag="ran")
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    oid.injected_text = f"[task_meta] task_id={tid}; category={cat}; sub_category={sub}\n"
    return oid


def skill_quoted_phrases_from_stem(ctx: SkillContext, inject: bool) -> SkillOutcome:
    """
    Extract quoted phrases from the **stem text** only (not audio-derived ASR).

    Targets: Speech Recognition / Counting lexical alignment (+ generic anchor stems).
    """
    stem = str(ctx.question.get("question", ""))
    pat = (
        r"'([^'\n]{2,160})'"
        r'|"([^\"\n]{2,160})"'
        r"|`([^`\n]{2,160})`"
    )
    found: list[str] = []
    for m in re.finditer(pat, stem):
        frag = next(g for g in m.groups() if g)
        t = frag.strip()
        if t and t not in found:
            found.append(t)
    oid = SkillOutcome("anchor_phrase_hints", invoke_tag="ran")
    if not found:
        oid.invoke_tag = "no_quotes_in_stem"
        return oid
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.errors.append({"kind": "skipped", "detail": f"n_candidates={len(found)}"})
        return oid
    lines = "\n".join(f"- `{q}`" for q in found[:12])
    oid.injected_text = f"[quoted_phrases_in_stem]\n{lines}\n"
    oid.invoke_tag = "injected"
    return oid


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
    oid.injected_text = f"[clip_dataset_span] start_time={st!s} end_time={et!s}{extra}\n"
    oid.invoke_tag = "injected"
    oid.bottleneck_tags = []
    return oid


def skill_media_clip_facts(ctx: SkillContext, inject: bool) -> SkillOutcome:
    """WAV duration / sample rate hints + A/V file presence (cheap file stats)."""
    oid = SkillOutcome("media_clip_facts", invoke_tag="ran", bottleneck_tags=[])
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    if not triggers.should_emit_media_clip_facts(q):
        oid.invoke_tag = "skipped_task_trigger"
        return oid

    ap, clip_ok = _wav_path(ctx)
    lines = ["[media_clip_facts]"]
    vad_out: VadRunOutcome | None = None
    if ap:
        vad_out = _vad_run(ctx)
        if vad_out is not None:
            oid.errors.extend(vad_out.errors)
            lines.append(
                f"audio_wav_duration_s={vad_out.duration_s:.3f} vad_backend={vad_out.backend} "
                f"sr_hint={vad_out.sample_rate}"
            )
        else:
            try:
                import wave

                with wave.open(ap, "rb") as wf:
                    n = wf.getnframes()
                    sr = wf.getframerate() or 1
                    lines.append(f"audio_wav_duration_s={n / float(sr):.3f} sr={sr} (wave)")
            except Exception:
                lines.append(f"audio_wav_path={ap} (duration_parse_failed)")
        b = _file_size_bytes(ap)
        if b is not None:
            lines.append(f"audio_wav_bytes={b}")
    else:
        lines.append("audio_wav_present=false")

    cv = _file_size_bytes(ctx.combined_path if Path(ctx.combined_path).is_file() else None)
    vv = _file_size_bytes(ctx.video_path if Path(ctx.video_path).is_file() else None)
    lines.append(
        f"combined_clip_exists={clip_ok} combined_bytes={cv if cv is not None else 'n/a'} "
        f"visual_only_bytes={vv if vv is not None else 'n/a'}"
    )

    if triggers.should_emit_audio_rms_meter(q) and ap:
        rm = rms_peak_meter(ap, vad_out, union_speech_only=True)
        oid.errors.extend(rm.errors)
        lines.append(
            "[audio_rms_meter_v1]"
            f" backend={rm.backend} sr={rm.sample_rate} analyzed_samples={rm.analyzed_samples}"
            f" union_speech_only=true"
            f" clip_rms_dbfs={(f'{rm.clip_rms_dbfs:.2f}' if rm.clip_rms_dbfs is not None else 'n/a')}"
            f" peak_dbfs={(f'{rm.peak_dbfs:.2f}' if rm.peak_dbfs is not None else 'n/a')}"
            f" crest_db={(f'{rm.crest_factor_db:.2f}' if rm.crest_factor_db is not None else 'n/a')}"
        )

    if triggers.should_emit_media_container_probe(q) and not triggers.should_emit_visual_clip_meta(q):
        cp = Path(ctx.combined_path)
        if cp.is_file():
            pr = probe_media_file(cp)
            oid.errors.extend(pr.errors)
            lines.append("[container_probe] " + format_probe_for_prompt(pr, label="combined"))

    oid.injected_text = "\n".join(lines) + "\n"
    oid.invoke_tag = "injected"
    return oid


def skill_speaker_turn_proxy(ctx: SkillContext, inject: bool) -> SkillOutcome:
    """VAD segment count as coarse ``speech_burst`` index (not literal diarization)."""
    oid = SkillOutcome("speaker_turn_proxy", invoke_tag="stub", bottleneck_tags=[])
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    if not triggers.should_emit_speaker_turn_proxy(q):
        oid.invoke_tag = "skipped_task_trigger"
        return oid

    vad_out = _vad_run(ctx)
    if vad_out is not None:
        oid.errors.extend(vad_out.errors)
    n = len(vad_out.segments) if vad_out else 0
    dur = vad_out.duration_s if vad_out else 0.0
    oid.injected_text = (
        "[speech_burst_proxy_vad]\n"
        f"vad_speech_burst_count={n} clip_audio_duration_s={dur:.3f}\n"
        "note=bursts_are_energy_segments_not_speaker_ids\n"
    )
    oid.invoke_tag = "vad_injected" if n else "stub"
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
    lines = ["[lexical_asr_bridge]", f"asr_backend={ao.backend if ao else 'n/a'} asr_source={asr_src}"]
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


def skill_visual_clip_meta(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("visual_clip_meta", invoke_tag="stub")
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    if not triggers.should_emit_visual_clip_meta(q):
        oid.invoke_tag = "skipped_task_trigger"
        return oid

    vp = ctx.video_path
    cp = ctx.combined_path
    lines = [
        "[visual_clip_meta]",
        f"visual_path_exists={Path(vp).is_file()} visual_bytes={_file_size_bytes(vp)}",
        f"audiovisual_path_exists={Path(cp).is_file()} combined_bytes={_file_size_bytes(cp)}",
    ]
    seen: set[Path] = set()
    for label, p_raw in (("visual_only", vp), ("audiovisual", cp)):
        pth = Path(p_raw).resolve()
        if not pth.is_file() or pth in seen:
            continue
        seen.add(pth)
        pr = probe_media_file(pth)
        oid.errors.extend(pr.errors)
        lines.append("[container_probe] " + format_probe_for_prompt(pr, label=label))
    oid.injected_text = "\n".join(lines) + "\n"
    oid.invoke_tag = "injected"
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

    oid.injected_text = (
        "[anchor_window_asr]\n"
        f"audio_wav_present={bool(ap)} audio_path={(ap or 'n/a')} clip_fallback_exists={clip_ok}\n"
        f"{vad_line}"
        f"{asr_blk}"
        f"{stub_preview_line}"
    )

    tx = bool(ao and ao.segments and any(s.text.strip() for s in ao.segments))
    be = ao.backend if ao is not None else ""
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
    blob = format_diar_for_prompt(d_out)
    extra = ""
    if d_out.segments:
        uniq = len({s.label for s in d_out.segments})
        extra = f"unique_speaker_label_count={uniq}\n"
    oid.injected_text = "[diarization]\n" + blob + "\n" + extra
    if d_out.backend in ("pyannote", "pyannote_api") and d_out.segments:
        oid.invoke_tag = "diar_injected"
        oid.bottleneck_tags = []
    else:
        oid.invoke_tag = "stub"
    return oid


def skill_overlap_split(ctx: SkillContext, inject: bool) -> SkillOutcome:
    """VAD density + coarse overlap proxy. ``triggers.TARGETS_DOC['overlap_split']``."""
    oid = SkillOutcome("overlap_split", bottleneck_tags=["perception_pending"], invoke_tag="stub")
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    if not triggers.should_emit_overlap_skill(dict(ctx.question)):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid

    vad_out = _vad_run(ctx)
    if vad_out is not None:
        oid.errors.extend(vad_out.errors)
    vad_blob = ""
    ov_s = 0.0
    nseg = 0
    if vad_out is not None:
        vad_blob = format_segments_for_prompt(vad_out) + "\n"
        segs = vad_out.segments
        nseg = len(segs)
        if len(segs) >= 2:
            ov_s = _vad_pairwise_overlap_s(segs)
    cov = 0.0
    if vad_out is not None and vad_out.duration_s > 1e-6 and vad_out.segments:
        cov = (
            sum(max(0.0, float(s.t1) - float(s.t0)) for s in vad_out.segments)
            / float(vad_out.duration_s)
        )
    oid.injected_text = (
        "[overlap_candidates]\n"
        f"{vad_blob}"
        f"vad_segment_count={nseg} speech_coverage_ratio={cov:.3f} pairwise_overlap_s_sum={ov_s:.3f}\n"
        "separation_stub per_stream_asr=[]\n"
    )
    if vad_out and vad_out.segments:
        oid.invoke_tag = "vad_injected"
    return oid


def skill_prosody_discrete(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("prosody_discrete", bottleneck_tags=["perception_pending"], invoke_tag="stub")
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    if not triggers.should_emit_prosody_discrete(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid

    ap, _clip = _wav_path(ctx)
    if not ap:
        oid.invoke_tag = "stub"
        oid.injected_text = "[prosody_discrete_energy_v1] audio_wav_missing\n"
        return oid

    vad_out = _vad_run(ctx)
    if vad_out is not None:
        oid.errors.extend(vad_out.errors)
    po = discrete_prosody_over_vad(ap, vad_out)
    oid.errors.extend(po.errors)
    oid.injected_text = "\n".join(po.lines) + "\n"
    oid.invoke_tag = "prosody_injected"
    oid.bottleneck_tags = []
    return oid


def skill_moment_refine(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("moment_refine", bottleneck_tags=["alignment_pending"], invoke_tag="stub")
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    if not triggers.should_emit_moment_refine(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    vad_out = _vad_run(ctx)
    cand = "[[0, clip_end_placeholder]]"
    rationale = "fallback_no_vad_or_no_wav"
    dur_note = ""
    if vad_out is not None:
        oid.errors.extend(vad_out.errors)
        dur_note = f"clip_duration_s={vad_out.duration_s:.2f}"
        if vad_out.segments:
            by_len = sorted(vad_out.segments, key=lambda s: s.t1 - s.t0, reverse=True)
            top = by_len[:3]
            cand = "[" + ",".join(f"[{s.t0:.2f},{s.t1:.2f}]" for s in top) + "]"
            rationale = "energy_vad_top3_by_duration"

    oid.injected_text = (
        "[moment_refine]\n"
        f"candidate_windows={cand}\n"
        f"rationale={rationale}\n"
        f"{dur_note}\n"
    )
    oid.invoke_tag = "vad_injected" if (vad_out and vad_out.segments) else "stub"
    return oid


def skill_visual_anchor_ground(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome(
        "visual_anchor_ground",
        bottleneck_tags=["perception_pending", "stub_backend"],
        invoke_tag="stub",
    )
    cat = str(ctx.question.get("category", "")).lower()
    if "visual" not in cat:
        oid.invoke_tag = "skipped_category_mismatch"
        return oid
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        return oid
    oid.injected_text = (
        "[visual_grounding_stub] detections=[] ocr_regions=[] "
        "(Tool T7 backend pending — see MM_AGENT_DESIGN § Implemented tools)\n"
    )
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
    oid.injected_text = format_words_for_prompt(w)
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
        oid.injected_text = f"[anchor_quote_s] phrase={ph!s} t_first_s=n/a\n"
        return oid
    oid.injected_text = f"[anchor_quote_s] phrase={ph!s} t_first_s={ts:.2f}\n"
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
    d_out = diar_with_vad_fallback(raw_d, vad_out)
    words = _get_cached_word_asr(ap, dict(ctx.question))
    oid.errors.extend(words.errors)
    sheet = build_turn_sheet(d_out, words)
    oid.injected_text = format_turn_sheet(sheet)
    oid.invoke_tag = "injected" if sheet.turns else "stub_empty"
    if sheet.turns:
        oid.bottleneck_tags = []
    return oid


def skill_speak_duration_sheet(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("speak_duration_sheet", invoke_tag="stub", bottleneck_tags=["perception_pending"])
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.bottleneck_tags = []
        return oid
    if not triggers.should_emit_speak_duration_sheet(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    ap, _ = _wav_path(ctx)
    if not ap:
        oid.invoke_tag = "stub"
        oid.injected_text = "[speak_duration_by_speaker] audio_wav_missing\n"
        return oid
    vad_out = _vad_run(ctx)
    raw_d = _get_cached_diar(ap, dict(ctx.question))
    oid.errors.extend(raw_d.errors)
    d_out = diar_with_vad_fallback(raw_d, vad_out)
    if not d_out.segments:
        oid.invoke_tag = "stub_empty"
        oid.injected_text = "[speak_duration_by_speaker] no_segments\n"
        return oid
    dur = duration_per_diar_label(d_out)
    oid.errors.extend(dur.errors)
    oid.injected_text = format_speak_duration(dur)
    oid.invoke_tag = "injected"
    oid.bottleneck_tags = []
    return oid


def skill_f0_rank_shortlist(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("f0_rank_shortlist", invoke_tag="stub", bottleneck_tags=["perception_pending"])
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.bottleneck_tags = []
        return oid
    if not triggers.should_emit_f0_rank_sheet(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    ap, _ = _wav_path(ctx)
    if not ap:
        oid.invoke_tag = "stub"
        oid.injected_text = "[f0_median_hz_by_segment] audio_wav_missing\n"
        return oid
    vad_out = _vad_run(ctx)
    raw_d = _get_cached_diar(ap, dict(ctx.question))
    oid.errors.extend(raw_d.errors)
    d_use = diar_with_vad_fallback(raw_d, vad_out)
    max_seg = triggers.effective_f0_max_diar_segments(q)
    ranked = sorted(
        d_use.segments,
        key=lambda s: float(s.t1) - float(s.t0),
        reverse=True,
    )[: max(1, max_seg)]
    segs = [(float(s.t0), float(s.t1), str(s.label)) for s in ranked]
    if not segs:
        oid.invoke_tag = "stub_empty"
        oid.injected_text = "[f0_median_hz_by_segment] no_segments\n"
        return oid
    po = pitch_median_over_segments(ap, segs)
    oid.errors.extend(po.errors)
    oid.injected_text = format_f0_sheet(po) + vad_proxy_prompt_footer(d_use.backend)
    oid.invoke_tag = "injected" if po.segments else "stub"
    if any(s.median_hz for s in po.segments):
        oid.bottleneck_tags = []
    return oid


def skill_rate_words_per_sec(ctx: SkillContext, inject: bool) -> SkillOutcome:
    oid = SkillOutcome("rate_words_per_sec", invoke_tag="stub", bottleneck_tags=["perception_pending"])
    q = dict(ctx.question)
    if not inject:
        oid.invoke_tag = "skipped_inject_disabled"
        oid.bottleneck_tags = []
        return oid
    if not triggers.should_emit_rate_wps_sheet(q):
        oid.invoke_tag = "skipped_task_trigger"
        oid.bottleneck_tags = []
        return oid
    ap, _ = _wav_path(ctx)
    if not ap:
        oid.invoke_tag = "stub"
        oid.injected_text = "[speech_rate_words_per_second] audio_wav_missing\n"
        return oid
    vad_out = _vad_run(ctx)
    raw_d = _get_cached_diar(ap, dict(ctx.question))
    oid.errors.extend(raw_d.errors)
    d_out = diar_with_vad_fallback(raw_d, vad_out)
    words = _get_cached_word_asr(ap, dict(ctx.question))
    oid.errors.extend(words.errors)
    if not d_out.segments:
        oid.invoke_tag = "stub_empty"
        oid.injected_text = "[speech_rate_words_per_second] no_segments\n"
        return oid
    rs = words_per_second_sheet(d_out, words)
    oid.injected_text = format_rate_sheet(rs)
    oid.invoke_tag = "injected"
    oid.bottleneck_tags = []
    return oid


def _viz_anchor_time_no_wav(video_path: Path, question: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """Midpoint of container duration (ffprobe) or half dataset span; last resort 0.5 s."""
    errs: list[dict[str, Any]] = []
    pr = probe_media_file(video_path)
    errs.extend(pr.errors)
    if pr.duration_s is not None and pr.duration_s > 0.05:
        return max(0.0, 0.5 * float(pr.duration_s)), errs
    st_s, et_s, span = dataset_span_seconds(question)
    if span is not None and span > 0:
        return max(0.0, 0.5 * float(span)), errs
    errs.append({"kind": "viz_anchor_fallback", "detail": "no_duration_using_t=0.5s"})
    return 0.5, errs


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
            t_anchor, xerrs = _viz_anchor_time_no_wav(vp, q)
            oid.errors.extend(xerrs)
            heuristic_note = "anchor_heuristic=video_mid_or_dataset_span_no_audio_alignment\n"
    delta = triggers.effective_viz_anchor_delta_s(q, quote_time_resolved=quote_resolved)
    times = sorted({max(0.0, float(t_anchor) + d) for d in (-delta, 0.0, delta)})
    snap = snap_and_count_people(vp, times)
    oid.errors.extend(snap.errors)
    oid.injected_text = heuristic_note + format_people_snap(snap)
    oid.invoke_tag = "frames_injected" if snap.n_frames_extracted_ok > 0 else "stub"
    if any(c is not None for c in snap.person_counts):
        oid.bottleneck_tags = ["perception_pending"]
    return oid


# Stable display order matching the recommendation table (EvidencePack merges at end logically).
_SKILL_REGISTRY: list[tuple[str, SkillFn]] = [
    ("meta_banner", skill_meta_banner),
    ("clip_span_meta", skill_clip_span_meta),
    ("anchor_phrase_hints", skill_quoted_phrases_from_stem),
    ("media_clip_facts", skill_media_clip_facts),
    ("speaker_turn_proxy", skill_speaker_turn_proxy),
    ("asr_word_lane", skill_asr_word_lane),
    ("anchor_quote_time", skill_anchor_quote_time),
    ("anchor_window_asr", skill_anchor_window_asr),
    ("lexical_asr_bridge", skill_lexical_asr_bridge),
    ("diar_binding", skill_diar_binding),
    ("turn_order_sheet", skill_turn_order_sheet),
    ("speak_duration_sheet", skill_speak_duration_sheet),
    ("f0_rank_shortlist", skill_f0_rank_shortlist),
    ("rate_words_per_sec", skill_rate_words_per_sec),
    ("overlap_split", skill_overlap_split),
    ("prosody_discrete", skill_prosody_discrete),
    ("moment_refine", skill_moment_refine),
    ("viz_people_anchor", skill_viz_people_anchor),
    ("visual_clip_meta", skill_visual_clip_meta),
    ("visual_anchor_ground", skill_visual_anchor_ground),
]


def _evidence_placement_mode() -> str:
    raw = os.getenv("AV_SPEAKERBENCH_EVIDENCE_PLACEMENT", "tail").strip().lower()
    return raw if raw in ("both", "tail") else "tail"


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
        if out.injected_text:
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
