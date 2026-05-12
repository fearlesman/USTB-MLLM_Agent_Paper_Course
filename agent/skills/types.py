from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillContext:
    """Per-question context passed through the Skill pipeline."""

    question_prompt: str
    question: dict[str, Any]
    args: Any
    video_path: str
    audio_path: str
    combined_path: str


@dataclass
class SkillOutcome:
    """Return value from a single Skill."""

    skill_id: str
    injected_text: str = ""
    """Non-empty snippets aggregated into ``Structured_skill_evidence`` when injection is on."""

    invoke_tag: str = ""
    """Short status for traces, e.g. ``stub`` or ``skipped_inject_disabled``."""

    bottleneck_tags: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
