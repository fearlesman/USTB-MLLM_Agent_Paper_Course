"""Turn-ordered transcript rows from diarization + word ASR (short clips)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audio_asr_words import WordAsrOutcome, WordPiece
from .audio_diar import DiarRunOutcome, vad_proxy_prompt_footer


@dataclass(frozen=True)
class TurnRow:
    t0: float
    t1: float
    speaker_label: str
    text_preview: str


@dataclass
class TurnSheetOutcome:
    turns: list[TurnRow]
    backend: str
    errors: list[dict[str, Any]]


def _mid(w: WordPiece) -> float:
    return 0.5 * (w.t0 + w.t1)


def build_turn_sheet(
    diar: DiarRunOutcome,
    words: WordAsrOutcome,
    *,
    preview_chars: int = 100,
) -> TurnSheetOutcome:
    errs: list[dict[str, Any]] = []
    if not diar.segments:
        return TurnSheetOutcome([], diar.backend, errs)

    rows: list[TurnRow] = []
    for sp in sorted(diar.segments, key=lambda s: (s.t0, s.label)):
        picked: list[str] = []
        for w in words.words:
            if _mid(w) >= float(sp.t0) and _mid(w) <= float(sp.t1):
                picked.append(w.text)
        prev = " ".join(picked).strip()
        if len(prev) > preview_chars:
            prev = prev[: preview_chars - 1].rstrip() + "…"
        rows.append(
            TurnRow(
                t0=float(sp.t0),
                t1=float(sp.t1),
                speaker_label=str(sp.label),
                text_preview=prev or "(no_asr_words_in_span)",
            )
        )
    return TurnSheetOutcome(rows, diar.backend, errs)


def format_turn_sheet(ot: TurnSheetOutcome) -> str:
    lines = ["[turn_order_sheet]", f"diar_backend={ot.backend} n_turns={len(ot.turns)}"]
    for r in ot.turns[:32]:
        lines.append(f"{r.t0:.2f}-{r.t1:.2f} spk={r.speaker_label} | {r.text_preview}")
    if len(ot.turns) > 32:
        lines.append(f"[truncated turns>{32}]")
    foot = vad_proxy_prompt_footer(ot.backend)
    if foot:
        lines.append(foot.rstrip())
    return "\n".join(lines) + "\n"
