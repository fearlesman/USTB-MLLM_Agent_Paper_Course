"""Append-only JSONL for per-question agent traces."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def append_trace_record(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(record)
    payload.setdefault("ts_utc", datetime.now(timezone.utc).isoformat())
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def prompt_fingerprint(prompt: str) -> str | None:
    """Short stable hash placeholder (full SHA optional via ``hashlib`` if needed later)."""
    if not prompt:
        return None
    return hex(hash(prompt) & ((1 << 64) - 1))[2:]

