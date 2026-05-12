from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentPrepResult:
    """Output of ``augment_prompt_for_inference`` consumed by ``model.inference``."""

    final_prompt: str
    skills_invoked: list[str] = field(default_factory=list)
    """Skill names that ran during prep (including no-op placeholders)."""

    bottleneck_tags: list[str] = field(default_factory=list)
    """Suggested taxonomy hooks: perception, alignment, reasoning, tool_boundary — filled by Skills later."""

    errors: list[dict] = field(default_factory=list)
    """Structured issues, e.g. ``{"kind": "tool_failure", "detail": "..."}``."""
