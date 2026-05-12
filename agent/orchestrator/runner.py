"""Single-step inference prep: Skills pipeline; optional prompt injection via env."""

from __future__ import annotations

import os
from typing import Any

from .context import AgentPrepResult
from skills import SkillContext, run_skill_pipeline


def _skill_gate_is_off() -> bool:
    """``AV_SPEAKERBENCH_SKILLS=off`` forces LM-only prompt (Skills registry ignored)."""
    return os.getenv("AV_SPEAKERBENCH_SKILLS", "").strip().lower() in ("0", "off", "false", "no")


def augment_prompt_for_inference(
    *,
    question_prompt: str,
    question: dict[str, Any],
    args: Any,
    video_path: str,
    audio_path: str,
    combined_path: str,
) -> AgentPrepResult:
    """
    Run the ordered Skill pipeline (see ``agent/skills/impl.py``).

    * By default **does not change** the MC prompt unless ``AV_SPEAKERBENCH_SKILL_INJECT=1``.
    * Trace still receives ``skills_invoked`` status tags when the agent track is active.
    """
    if _skill_gate_is_off():
        return AgentPrepResult(question_prompt)

    ctx = SkillContext(
        question_prompt=question_prompt,
        question=question,
        args=args,
        video_path=video_path,
        audio_path=audio_path,
        combined_path=combined_path,
    )
    final_prompt, tags, btags, errs = run_skill_pipeline(ctx)
    return AgentPrepResult(
        final_prompt,
        skills_invoked=tags,
        bottleneck_tags=btags,
        errors=errs,
    )
