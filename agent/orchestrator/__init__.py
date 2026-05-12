"""Agent orchestration layer: ``augment_prompt_for_inference`` delegates to ``skills`` pipeline."""

from .context import AgentPrepResult
from .runner import augment_prompt_for_inference

__all__ = ["AgentPrepResult", "augment_prompt_for_inference"]
