"""Aggregate speech duration per diarization label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audio_diar import DiarRunOutcome, vad_proxy_prompt_footer


@dataclass
class SpeakDurationOutcome:
    seconds_by_speaker: dict[str, float]
    total_speech_s: float
    backend: str
    errors: list[dict[str, Any]]


def duration_per_diar_label(diar: DiarRunOutcome) -> SpeakDurationOutcome:
    errs: list[dict[str, Any]] = list(diar.errors)
    acc: dict[str, float] = {}
    for s in diar.segments:
        lab = str(s.label)
        dt = max(0.0, float(s.t1) - float(s.t0))
        acc[lab] = acc.get(lab, 0.0) + dt
    tot = sum(acc.values())
    return SpeakDurationOutcome(acc, tot, diar.backend, errs)


def format_speak_duration(ot: SpeakDurationOutcome) -> str:
    ranked = sorted(ot.seconds_by_speaker.items(), key=lambda x: x[1], reverse=True)
    parts = [
        "[speak_duration_by_speaker]",
        f"backend={ot.backend} total_diar_speech_s={ot.total_speech_s:.2f}",
    ]
    for lab, sec in ranked[:12]:
        parts.append(f"{lab}={sec:.2f}s")
    if not ranked:
        parts.append("no_diar_spans")
    foot = vad_proxy_prompt_footer(ot.backend)
    if foot:
        parts.append(foot.rstrip())
    return "\n".join(parts) + "\n"
