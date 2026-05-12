"""
Container / stream probe via **ffprobe** (optional).

``AV_SPEAKERBENCH_MEDIA_PROBE``: ``auto`` (default: run if ``ffprobe`` resolves), ``on``, ``off``.
``AV_SPEAKERBENCH_FFPROBE_BIN``: executable name or path (default ``ffprobe``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MediaProbeOutcome:
    path: str
    duration_s: float | None
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_codec: str | None
    has_video_stream: bool
    backend: str
    errors: list[dict[str, Any]]


def _parse_ratio(s: str | None) -> float | None:
    if not s or s in ("0/0", "N/A"):
        return None
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            na, nb = float(a), float(b)
            return na / nb if nb else None
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _probe_mode() -> str:
    return os.getenv("AV_SPEAKERBENCH_MEDIA_PROBE", "auto").strip().lower()


def _ffprobe_bin() -> str:
    return os.getenv("AV_SPEAKERBENCH_FFPROBE_BIN", "ffprobe").strip() or "ffprobe"


def _resolve_ffprobe() -> str | None:
    exe = _ffprobe_bin()
    if os.sep in exe or (len(exe) > 2 and exe[1] == ":"):
        p = Path(exe)
        return str(p) if p.is_file() else None
    return shutil.which(exe)


def probe_media_file(path: str | Path, *, timeout_s: float | None = None) -> MediaProbeOutcome:
    """
    Return stream/format summary using ffprobe JSON, or stub when disabled / missing.
    """
    p = Path(path).resolve()
    errs: list[dict[str, Any]] = []
    mode = _probe_mode()
    if mode in ("off", "0", "false", "no", "none"):
        return MediaProbeOutcome(
            str(p),
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            "disabled",
            [],
        )

    if not p.is_file():
        errs.append({"kind": "file_missing", "detail": str(p)})
        return MediaProbeOutcome(str(p), None, None, None, None, None, None, False, "none", errs)

    resolved = _resolve_ffprobe()
    if mode == "auto" and not resolved:
        return MediaProbeOutcome(str(p), None, None, None, None, None, None, False, "ffprobe_unavailable", [])

    if not resolved:
        errs.append({"kind": "ffprobe_not_found", "detail": _ffprobe_bin()})
        return MediaProbeOutcome(str(p), None, None, None, None, None, None, False, "none", errs)

    to = timeout_s if timeout_s is not None else float(os.getenv("AV_SPEAKERBENCH_FFPROBE_TIMEOUT_S", "45") or 45)

    cmd = [
        resolved,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(p),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1.0, to),
            check=False,
        )
    except subprocess.TimeoutExpired:
        errs.append({"kind": "ffprobe_timeout", "detail": f"timeout_s={to}"})
        return MediaProbeOutcome(str(p), None, None, None, None, None, None, False, "error", errs)
    except OSError as e:
        errs.append({"kind": "ffprobe_spawn_failed", "detail": str(e)})
        return MediaProbeOutcome(str(p), None, None, None, None, None, None, False, "error", errs)

    if proc.returncode != 0:
        err_tail = (proc.stderr or "").strip()[:400]
        errs.append({"kind": "ffprobe_failed", "detail": err_tail or f"exit={proc.returncode}"})
        return MediaProbeOutcome(str(p), None, None, None, None, None, None, False, "error", errs)

    try:
        blob = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        errs.append({"kind": "ffprobe_json", "detail": str(e)})
        return MediaProbeOutcome(str(p), None, None, None, None, None, None, False, "error", errs)

    fmt = blob.get("format") or {}
    dur_raw = fmt.get("duration")
    duration_s: float | None = None
    try:
        if dur_raw is not None:
            duration_s = float(dur_raw)
    except (TypeError, ValueError):
        duration_s = None

    width = height = None
    fps = None
    vcodec = acodec = None
    has_video = False
    for st in blob.get("streams") or []:
        ctype = (st.get("codec_type") or "").lower()
        if ctype == "video" and not has_video:
            has_video = True
            vcodec = st.get("codec_name")
            try:
                width = int(st.get("width"))
            except (TypeError, ValueError):
                width = None
            try:
                height = int(st.get("height"))
            except (TypeError, ValueError):
                height = None
            fps = _parse_ratio(st.get("r_frame_rate")) or _parse_ratio(st.get("avg_frame_rate"))
        elif ctype == "audio" and acodec is None:
            acodec = st.get("codec_name")

    return MediaProbeOutcome(
        str(p),
        duration_s,
        width,
        height,
        fps,
        vcodec,
        acodec,
        has_video,
        "ffprobe",
        errs,
    )


def format_probe_for_prompt(otc: MediaProbeOutcome, *, label: str) -> str:
    """Single compact line for Structured_skill_evidence."""
    parts = [f"path_label={label}", f"backend={otc.backend}"]
    if otc.duration_s is not None:
        parts.append(f"duration_s={otc.duration_s:.3f}")
    if otc.has_video_stream and otc.width and otc.height:
        parts.append(f"frame={otc.width}x{otc.height}")
    if otc.fps is not None:
        parts.append(f"fps={otc.fps:.3f}")
    if otc.video_codec:
        parts.append(f"vcodec={otc.video_codec}")
    if otc.audio_codec:
        parts.append(f"acodec={otc.audio_codec}")
    return " ".join(parts)
