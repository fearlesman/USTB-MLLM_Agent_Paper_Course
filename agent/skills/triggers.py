"""Task-conditional Skill triggers (weak-bucket-aligned). See ``SKILL_INJECT_ABLATION.md``."""

from __future__ import annotations

import os
import re
from typing import Any

# Level-3-ish task cohorts derived from Evidence-from-baseline mapping (placeholder acc — replace via JSON ranking).
_TARGETS_ASR_NOTE = (
    "Targets: level 3: Speech Recognition, Speech Counting, Speech Duration (+ stem quotes); "
    "mechanism: perception|alignment"
)
_TARGETS_DIAR_NOTE = (
    "Targets: level 3: Speaker Recognition, Speaker Counting (+ speaker* subtasks); mechanism: perception|binding"
)
_TARGETS_OVERLAP_NOTE = (
    "Targets: level 3: Speech Counting, Speaker Counting, Speaker Recognition (+ overlap-heavy clips); "
    "mechanism: perception"
)

STEM_QUOTES_PAT = re.compile(
    r"'([^'\n]{2,160})'|" r'"([^\"\n]{2,160})"' r"|`([^`\n]{2,160})`"
)

TASK_IDS_ASR_FOCUS: frozenset[str] = frozenset(
    {
        "Speech Recognition",
        "Speech Counting",
        "Speech Duration",
        "Speech Pitch",
        "Speech Rate",
        "Speech Intensity",
    }
)

TASK_IDS_OVERLAP_FOCUS: frozenset[str] = frozenset(
    {
        "Speech Counting",
        "Speaker Counting",
        "Speaker Recognition",
    }
)

# Extra cohorts: word timestamps help alignment / multi-speaker comparison beyond TASK_IDS_ASR_FOCUS.
TASK_IDS_WORD_LANE_EXTRA: frozenset[str] = frozenset(
    {
        "Speech Duration",
        "Speech Rate",
        "Speaker Recognition",
        "Speaker Detection",
    }
)

# Diarization valuable when comparing speakers or aggregate per (pseudo)speaker.
TASK_IDS_DIAR_EXTENDED: frozenset[str] = frozenset(
    {
        "Speech Duration",
        "Speech Rate",
        "Speaker Detection",
        "Speech Counting",
    }
)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def effective_word_asr_max_items(question: dict[str, Any]) -> int:
    """Larger word caps for counting / recognition / quoted stems (same clip, one ASR pass)."""
    base = _int_env("AV_SPEAKERBENCH_WORD_ASR_MAX_ITEMS", 96)
    cap = _int_env("AV_SPEAKERBENCH_WORD_ASR_MAX_ITEMS_CAP", 256)
    if _trigger_mode_all():
        return min(cap, max(base, _int_env("AV_SPEAKERBENCH_WORD_ASR_MAX_ITEMS_AUDIT", 192)))
    tid = str(question.get("task_id", ""))
    mul = 1
    if tid in ("Speech Counting", "Speech Recognition", "Speaker Recognition"):
        mul = 2
    elif tid in ("Speech Duration", "Speech Rate", "Speaker Counting"):
        mul = 2
    if stem_has_quotelike_phrases(question) and tid in (
        "Speech Counting",
        "Speaker Recognition",
        "Speech Recognition",
    ):
        mul = max(mul, 2)
    return min(cap, base * mul)


def effective_f0_max_diar_segments(question: dict[str, Any]) -> int:
    """More diar/VAD slices for pitch ranking when comparing multiple speakers."""
    base = _int_env("AV_SPEAKERBENCH_F0_MAX_DIAR_SEGMENTS", 48)
    if str(question.get("task_id", "")) == "Speech Pitch":
        bump = _int_env("AV_SPEAKERBENCH_F0_MAX_SEGMENTS_PITCH_BUMP", 16)
        return min(96, base + bump)
    return base


def effective_viz_anchor_delta_s(question: dict[str, Any], *, quote_time_resolved: bool) -> float:
    """Tighter spacing when ASR anchored the quote (visual counting at utterance onset)."""
    raw = os.getenv("AV_SPEAKERBENCH_VIZ_ANCHOR_DELTA_S", "0.25").strip() or "0.25"
    try:
        base = float(raw)
    except ValueError:
        base = 0.25
    if not quote_time_resolved:
        return base
    tight = os.getenv("AV_SPEAKERBENCH_VIZ_ANCHOR_DELTA_QUOTE_S", "0.12").strip() or "0.12"
    try:
        tq = float(tight)
    except ValueError:
        tq = 0.12
    return min(base, tq)


def should_emit_diar_binding(question: dict[str, Any]) -> bool:
    """Per-speaker spans: recognition, counting, detection, multi-speaker duration/rate, etc."""
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", ""))
    tlo = tid.lower()
    sub = str(question.get("sub_category", "")).lower()
    if "speaker" in tlo or "speaker" in sub:
        return True
    if tid in TASK_IDS_DIAR_EXTENDED | TASK_IDS_OVERLAP_FOCUS:
        return True
    return False


def should_emit_prosody_discrete(question: dict[str, Any]) -> bool:
    """Coarse energy / pitch / rate proxies; optional skip for Speech Pitch when F0 sheet runs."""
    if _trigger_mode_all():
        return True
    if os.getenv("AV_SPEAKERBENCH_PROSODY_SKIP_FOR_PITCH", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        if str(question.get("task_id", "")) == "Speech Pitch":
            return False
    tid = str(question.get("task_id", "")).lower()
    return any(k in tid for k in ("pitch", "rate", "intensity", "duration"))


def should_emit_moment_refine(question: dict[str, Any]) -> bool:
    """Clip-scale time localization: duration questions, activity, temporal language."""
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", "")).lower()
    sub = str(question.get("sub_category", "")).lower()
    needles = (
        "duration",
        "activity",
        "after",
        "before",
        "when",
        "time",
        "speech",
        "until",
        "start",
        "finish",
    )
    if any(k in tid for k in needles):
        return True
    if any(x in sub for x in ("activity", "duration")):
        return True
    return False


def _trigger_mode_all() -> bool:
    """``AV_SPEAKERBENCH_SKILLS_TRIGGER_MODE=all`` runs every Skill (minus per-skill structural skips)."""
    return os.getenv("AV_SPEAKERBENCH_SKILLS_TRIGGER_MODE", "auto").strip().lower() in (
        "all",
        "*",
        "force",
    )


def stem_has_quotelike_phrases(question: dict[str, Any]) -> bool:
    stem = str(question.get("question", ""))
    return bool(STEM_QUOTES_PAT.search(stem))


def asr_window_budget_multiplier(question: dict[str, Any]) -> float:
    """Larger Whisper/VAD budgets for lexical / counting tasks (+ quotes)."""
    if _trigger_mode_all():
        return float(os.getenv("AV_SPEAKERBENCH_ASR_WINDOW_MULT_CAP", "2.5") or 2.5)
    tid = str(question.get("task_id", ""))
    m = float(os.getenv("AV_SPEAKERBENCH_ASR_WINDOW_MULT_DEFAULT", "1") or 1)
    focus = os.getenv("AV_SPEAKERBENCH_ASR_PRIORITY_TASK_IDS", "")
    ids: set[str] = TASK_IDS_ASR_FOCUS.copy()
    if focus.strip():
        ids = {x.strip() for x in focus.split(",") if x.strip()}
    if tid in ids or stem_has_quotelike_phrases(question):
        return float(os.getenv("AV_SPEAKERBENCH_ASR_WINDOW_MULT_FOCUS", "2") or 2)
    return m


def should_emit_anchor_window_asr(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", ""))
    if tid in TASK_IDS_ASR_FOCUS:
        return True
    if stem_has_quotelike_phrases(question):
        return True
    return False


def should_emit_overlap_skill(question: dict[str, Any]) -> bool:
    """Narrow cohorts where overlap / turn-taking hypotheses help."""
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", ""))
    cat = str(question.get("category", "")).lower()
    return tid in TASK_IDS_OVERLAP_FOCUS or "speaker" in cat


def should_emit_clip_span_meta(question: dict[str, Any]) -> bool:
    """Official clip window from ``test.csv`` (`start_time` / `end_time`)."""
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", "")).lower()
    sub = str(question.get("sub_category", "")).lower()
    needles = (
        "activity",
        "duration",
        "counting",
        "when",
        "time",
        "recognition",
        "pitch",
        "rate",
        "detect",
    )
    if any(x in tid for x in needles) or any(x in sub for x in needles):
        return True
    return str(question.get("category", "")).lower() == "speaker-centric"


def should_emit_media_clip_facts(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", "")).lower()
    cat = str(question.get("category", "")).lower()
    asrish = {x.lower() for x in TASK_IDS_ASR_FOCUS}
    overish = {x.lower() for x in TASK_IDS_OVERLAP_FOCUS}
    if tid in asrish | overish:
        return True
    if cat in ("audio-centric", "visual-centric", "speaker-centric"):
        return True
    return stem_has_quotelike_phrases(question)


def should_emit_lexical_asr_bridge(question: dict[str, Any]) -> bool:
    if not stem_has_quotelike_phrases(question):
        return False
    if _trigger_mode_all():
        return True
    return should_emit_anchor_window_asr(question)


def should_emit_speaker_turn_proxy(question: dict[str, Any]) -> bool:
    """VAD bursts ≈ coarse speech activity index (not literal speaker count)."""
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", ""))
    return tid in ("Speaker Counting", "Speech Counting")


def should_emit_visual_clip_meta(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    cat = str(question.get("category", "")).lower()
    tid = str(question.get("task_id", "")).lower()
    if "visual" in cat:
        return True
    return "visual" in tid or "attribute" in tid


def should_emit_media_container_probe(question: dict[str, Any]) -> bool:
    """ffprobe on ``.mp4`` paths — narrow default to limit eval wall time."""
    if _trigger_mode_all():
        return True
    if should_emit_visual_clip_meta(question):
        return True
    tid = str(question.get("task_id", "")).lower()
    if "duration" in tid:
        return True
    cat = str(question.get("category", "")).lower()
    return cat == "speaker-centric"


def should_emit_audio_rms_meter(question: dict[str, Any]) -> bool:
    """Extra RMS/peak line (second decode); keep to intensity-focused rows."""
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", "")).lower()
    return "intensity" in tid


def should_emit_asr_word_lane(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", ""))
    if tid in TASK_IDS_ASR_FOCUS | TASK_IDS_WORD_LANE_EXTRA:
        return True
    return stem_has_quotelike_phrases(question)


def should_emit_anchor_quote_time(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", ""))
    if tid == "Visual Counting":
        return True
    if tid == "Speaker Recognition" and stem_has_quotelike_phrases(question):
        return True
    if "Counting" in tid and stem_has_quotelike_phrases(question):
        return True
    return False


def should_emit_turn_order_sheet(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    tid = str(question.get("task_id", ""))
    stem = str(question.get("question", "")).lower()
    order_kw = (
        "before",
        "after",
        "until",
        "start",
        "finish",
        "immediately",
        "right ",
        "first",
        "second",
        "who speaks",
        "speaks before",
        "speaks after",
        "order",
    )
    if tid == "Speaker Recognition":
        return True
    if tid in ("Speaker Counting", "Speech Counting") and (
        any(k in stem for k in order_kw) or stem_has_quotelike_phrases(question)
    ):
        return True
    return False


def should_emit_speak_duration_sheet(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    return str(question.get("task_id", "")) == "Speech Duration"


def should_emit_f0_rank_sheet(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    return "Pitch" in str(question.get("task_id", ""))


def should_emit_rate_wps_sheet(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    return "Rate" in str(question.get("task_id", ""))


def should_emit_viz_people_anchor(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    return str(question.get("task_id", "")) == "Visual Counting"


TARGETS_DOC = {
    "anchor_window_asr": _TARGETS_ASR_NOTE,
    "clip_span_meta": (
        "Targets: alignment-heavy tasks (timing, counting, recognition cohorts); "
        "mechanism: alignment (dataset clip span)"
    ),
    "media_clip_facts": (
        "Targets: A/V/Speaker-centric + ASR/overlap cohorts; mechanism: perception (clip stats)"
    ),
    "speaker_turn_proxy": (
        "Targets: Speaker Counting, Speech Counting; mechanism: perception (VAD bursts, not diar IDs)"
    ),
    "lexical_asr_bridge": (
        "Targets: Speech* + stem quotes; mechanism: alignment (quotes vs ASR windows)"
    ),
    "visual_clip_meta": (
        "Targets: Visual-centric / visual-attribute tasks; mechanism: perception (file presence)"
    ),
    "diar_binding": (
        "Targets: speaker-named tasks, Speaker Detection, Speech Duration/Rate (multi-speaker), "
        "Speech/Speaker Counting; mechanism: diar (binding)"
    ),
    "overlap_split": _TARGETS_OVERLAP_NOTE,
    "asr_word_lane": (
        "Targets: ASR-focus tasks + Speech Duration/Rate, Speaker Recognition/Detection, stem quotes; "
        "mechanism: word timestamps (budget via effective_word_asr_max_items)"
    ),
    "anchor_quote_time": (
        "Targets: Visual Counting; Speaker Recognition w/ quotes; quote-bounded counting; "
        "mechanism: ASR alignment"
    ),
    "turn_order_sheet": (
        "Targets: Speaker Recognition; temporal/order counting (before/after/immediately/…); "
        "mechanism: diar + word spans"
    ),
    "speak_duration_sheet": "Targets: Speech Duration; mechanism: perception",
    "f0_rank_shortlist": (
        "Targets: Speech Pitch; mechanism: pyin per segment (extra segments on Pitch via env bump)"
    ),
    "rate_words_per_sec": "Targets: Speech Rate; mechanism: perception",
    "viz_people_anchor": "Targets: Visual Counting at anchor time; mechanism: perception (tighter delta when quote aligned)",
    "prosody_discrete": (
        "Targets: Intensity, Rate, Duration coarse energy; skipped for Speech Pitch if PROSODY_SKIP_FOR_PITCH=1 (F0 covers)"
    ),
    "moment_refine": (
        "Targets: temporal questions (duration/activity/before/after/when); mechanism: VAD top windows"
    ),
}
