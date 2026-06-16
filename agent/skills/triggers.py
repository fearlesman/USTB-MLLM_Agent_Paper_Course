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

# Extra cohorts: word timestamps help alignment / multi-speaker comparison beyond TASK_IDS_ASR_FOCUS.
TASK_IDS_WORD_LANE_EXTRA: frozenset[str] = frozenset(
    {
        "Speech Duration",
        "Speech Rate",
        "Speaker Recognition",
        "Speaker Detection",
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


def _experimental_speaker_order_enabled() -> bool:
    """
    Temporal speaker-order tables outside core Speaker Recognition are noisier
    than the ASR-centric evidence pack, so keep them opt-in.
    """
    raw = os.getenv("AV_SPEAKERBENCH_ENABLE_EXPERIMENTAL_SPEAKER_ORDER", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def question_category(question: dict[str, Any]) -> str:
    return str(question.get("category", "")).strip().lower()


def is_audio_centric(question: dict[str, Any]) -> bool:
    return question_category(question) == "audio-centric"


def is_visual_centric(question: dict[str, Any]) -> bool:
    return question_category(question) == "visual-centric"


def is_speaker_centric(question: dict[str, Any]) -> bool:
    return question_category(question) == "speaker-centric"


def is_composite_av_task(question: dict[str, Any]) -> bool:
    """
    Speaker-centric rows are treated as composite A/V tasks in the current agent track:
    they may need audio evidence plus visual identity disambiguation and temporal binding.
    """
    return is_speaker_centric(question)


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
    if is_visual_centric(question):
        return False
    tid = str(question.get("task_id", ""))
    if tid == "Speaker Recognition":
        return True
    if _experimental_speaker_order_enabled() and tid in ("Speaker Counting", "Speaker Detection"):
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
    if is_visual_centric(question) and not stem_has_quotelike_phrases(question):
        return False
    tid = str(question.get("task_id", ""))
    if tid in TASK_IDS_ASR_FOCUS:
        return True
    if stem_has_quotelike_phrases(question):
        return True
    return False


def should_emit_clip_span_meta(question: dict[str, Any]) -> bool:
    """Official clip window from ``test.csv`` (`start_time` / `end_time`)."""
    if _trigger_mode_all():
        return True
    if is_audio_centric(question) or is_visual_centric(question) or is_composite_av_task(question):
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


def should_emit_lexical_asr_bridge(question: dict[str, Any]) -> bool:
    if not stem_has_quotelike_phrases(question):
        return False
    if _trigger_mode_all():
        return True
    if is_visual_centric(question) or is_audio_centric(question) or is_composite_av_task(question):
        return True
    return should_emit_anchor_window_asr(question)


def should_emit_asr_word_lane(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    if is_visual_centric(question) and not stem_has_quotelike_phrases(question):
        return False
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
    if is_visual_centric(question):
        return False
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
    if _experimental_speaker_order_enabled() and tid in ("Speaker Counting", "Speech Counting") and (
        any(k in stem for k in order_kw) or stem_has_quotelike_phrases(question)
    ):
        return True
    return False


def should_emit_viz_people_anchor(question: dict[str, Any]) -> bool:
    if _trigger_mode_all():
        return True
    if is_audio_centric(question):
        return False
    return str(question.get("task_id", "")) == "Visual Counting"


TARGETS_DOC = {
    "anchor_window_asr": _TARGETS_ASR_NOTE,
    "clip_span_meta": (
        "Targets: alignment-heavy tasks (timing, counting, recognition cohorts); "
        "mechanism: alignment (dataset clip span)"
    ),
    "lexical_asr_bridge": (
        "Targets: Speech* + stem quotes; mechanism: alignment (quotes vs ASR windows)"
    ),
    "diar_binding": (
        "Targets: speaker-named tasks; mechanism: diar (binding)"
    ),
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
    "viz_people_anchor": "Targets: Visual Counting at anchor time; mechanism: perception (tighter delta when quote aligned)",
}
