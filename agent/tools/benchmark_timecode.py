"""
Parse ``start_time`` / ``end_time`` fields from ``test.csv`` (typically ``MM:SS``).

Used for alignment Skills and cheap duration checks vs container / WAV length.
"""

from __future__ import annotations

from typing import Any


def parse_benchmark_clock(token: str) -> float | None:
    """
    Return seconds from a dataset time token.

    Accepts ``MM:SS``, ``H:MM:SS``, or a plain float string (fallback).
    """
    t = (token or "").strip()
    if not t:
        return None
    parts = t.split(":")
    try:
        if len(parts) == 2:
            m, sec = parts
            return float(int(m) * 60 + float(sec))
        if len(parts) == 3:
            h, m, sec = parts
            return float(int(h) * 3600 + int(m) * 60 + float(sec))
        return float(t)
    except ValueError:
        return None


def dataset_span_seconds(question: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """
    Returns ``(start_s, end_s, span_s)`` where ``span_s = end - start`` if both ends parse.
    """
    st = parse_benchmark_clock(str(question.get("start_time", "")))
    et = parse_benchmark_clock(str(question.get("end_time", "")))
    if st is None or et is None:
        return st, et, None
    return st, et, max(0.0, et - st)
