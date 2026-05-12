"""Words-per-second proxy per speaker span (word midpoint in diar interval)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audio_asr_words import WordAsrOutcome, WordPiece
from .audio_diar import DiarRunOutcome, vad_proxy_prompt_footer


@dataclass(frozen=True)
class RateRow:
    speaker_label: str
    t0: float
    t1: float
    n_words: int
    words_per_second: float


@dataclass
class RateSheetOutcome:
    rows: list[RateRow]
    backend: str
    errors: list[dict[str, Any]]


def _mid(w: WordPiece) -> float:
    return 0.5 * (w.t0 + w.t1)


def words_per_second_sheet(diar: DiarRunOutcome, words: WordAsrOutcome) -> RateSheetOutcome:
    errs: list[dict[str, Any]] = []
    rows: list[RateRow] = []
    if not diar.segments:
        return RateSheetOutcome([], words.backend, errs)

    for sp in diar.segments:
        dur = max(1e-3, float(sp.t1) - float(sp.t0))
        n = 0
        for w in words.words:
            if _mid(w) >= float(sp.t0) and _mid(w) <= float(sp.t1):
                if w.text.strip():
                    n += 1
        rows.append(
            RateRow(
                speaker_label=str(sp.label),
                t0=float(sp.t0),
                t1=float(sp.t1),
                n_words=n,
                words_per_second=n / dur,
            )
        )
    return RateSheetOutcome(rows, f"{words.backend}+{diar.backend}", errs)


def format_rate_sheet(ot: RateSheetOutcome) -> str:
    ranked = sorted(ot.rows, key=lambda r: r.words_per_second, reverse=True)
    lines = ["[speech_rate_words_per_second]", f"backend={ot.backend}"]
    for r in ranked[:16]:
        lines.append(
            f"{r.speaker_label} wps={r.words_per_second:.2f} n_words={r.n_words} "
            f"span=[{r.t0:.2f},{r.t1:.2f}]"
        )
    foot = vad_proxy_prompt_footer(ot.backend)
    if foot:
        lines.append(foot.rstrip())
    return "\n".join(lines) + "\n"
