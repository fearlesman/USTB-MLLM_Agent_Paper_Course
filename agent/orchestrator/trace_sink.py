"""Append-only JSONL for per-question agent traces."""

from __future__ import annotations

import json
import os
import hashlib
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
    """Short stable SHA-256 fingerprint for comparing model-visible prompts."""
    if not prompt:
        return None
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
