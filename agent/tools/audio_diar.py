"""
Tool — **speaker diarization**: optional **pyannoteAI cloud API**, local **HF Pipeline**, or ``stub``.

``Targets``: level 3: Speaker Recognition, Speaker Counting; mechanism: perception|binding
(replace baseline acc/n from ``result/*.json`` per MM_AGENT_DESIGN).

Env:

- ``AV_SPEAKERBENCH_DIAR_BACKEND`` — ``auto`` (default), ``stub``, ``pyannote``, ``pyannote_api`` (cloud only).
- **Cloud** (see `pyannoteAI quickstart <https://docs.pyannote.ai/quickstart>`_):

  - ``PYANNOTE_API_KEY`` — Bearer token; uploads WAV via ``/v1/media/input`` then ``/v1/diarize``.
  - ``PYANNOTE_API_BASE`` — default ``https://api.pyannote.ai``.
  - ``AV_SPEAKERBENCH_PYANNOTE_API_POLL_S`` / ``AV_SPEAKERBENCH_PYANNOTE_API_MAX_WAIT_S`` — polling.

- **Local HF** (when ``DIAR_BACKEND=pyannote`` and **no** ``PYANNOTE_API_KEY``):

  - ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` — auth for ``Pipeline.from_pretrained``.
  - ``AV_SPEAKERBENCH_PYANNOTE_MODEL`` — default ``pyannote/speaker-diarization-community-1``.

If ``DIAR_BACKEND=pyannote`` and ``PYANNOTE_API_KEY`` is set, the **cloud API** is used first (no local ``pyannote.audio`` weights).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .audio_vad import VadRunOutcome


@dataclass(frozen=True)
class DiarizedSpan:
    t0: float
    t1: float
    label: str
    conf: float | None


@dataclass
class DiarRunOutcome:
    segments: list[DiarizedSpan]
    backend: str
    duration_s: float
    errors: list[dict[str, Any]]


def _select_diar_backend() -> str:
    raw = os.getenv("AV_SPEAKERBENCH_DIAR_BACKEND", "auto").strip().lower()
    aliases = {
        "default": "auto",
        "cloud": "pyannote_api",
    }
    raw = aliases.get(raw, raw)
    if raw in ("pyannote", "pyannote_api", "stub"):
        return raw
    if os.getenv("PYANNOTE_API_KEY", "").strip():
        return "pyannote_api"
    try:
        import pyannote.audio  # noqa: F401

        return "pyannote"
    except ImportError:
        return "stub"


def _http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    body: bytes | None = None,
    timeout: float = 120.0,
) -> tuple[int, bytes]:
    req = Request(url, data=body, method=method, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return int(resp.status), resp.read()
    except HTTPError as e:
        return int(e.code), e.read()
    except URLError as e:
        return -1, str(e).encode("utf-8", errors="replace")


def _diarize_pyannote_api(path: Path, duration_s: float, errs: list[dict[str, Any]]) -> DiarRunOutcome:
    """Upload local WAV to pyannoteAI temporary media, run diarization, poll job (OpenAPI shape)."""
    api_key = os.getenv("PYANNOTE_API_KEY", "").strip()
    base = os.getenv("PYANNOTE_API_BASE", "https://api.pyannote.ai").rstrip("/")
    poll_s = float(os.getenv("AV_SPEAKERBENCH_PYANNOTE_API_POLL_S", "8") or 8)
    max_wait = float(os.getenv("AV_SPEAKERBENCH_PYANNOTE_API_MAX_WAIT_S", "600") or 600)
    if not api_key:
        errs.append({"kind": "pyannote_api_key_missing", "detail": "set PYANNOTE_API_KEY (see docs.pyannote.ai)"})
        return DiarRunOutcome([], "stub", duration_s, errs)

    auth_h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    object_key = f"avspeakerbench-{uuid.uuid4().hex}"
    media_ref = f"media://{object_key}"

    cre_body = json.dumps({"url": media_ref}).encode("utf-8")
    code, raw = _http_request("POST", f"{base}/v1/media/input", headers=auth_h, body=cre_body, timeout=60.0)
    if code != 200:
        errs.append(
            {
                "kind": "pyannote_media_input_failed",
                "detail": raw.decode("utf-8", errors="replace")[:800],
                "http_status": code,
            }
        )
        return DiarRunOutcome([], "stub", duration_s, errs)
    try:
        presigned = json.loads(raw.decode("utf-8")).get("url")
    except (json.JSONDecodeError, TypeError) as e:
        errs.append({"kind": "pyannote_media_input_parse", "detail": str(e)})
        return DiarRunOutcome([], "stub", duration_s, errs)
    if not presigned or not isinstance(presigned, str):
        errs.append({"kind": "pyannote_media_input_no_url", "detail": raw.decode("utf-8", errors="replace")[:400]})
        return DiarRunOutcome([], "stub", duration_s, errs)

    try:
        data_bin = path.read_bytes()
    except OSError as e:
        errs.append({"kind": "wav_read_failed", "detail": str(e)})
        return DiarRunOutcome([], "stub", duration_s, errs)

    put_code, put_raw = _http_request(
        "PUT",
        presigned,
        headers={"Content-Type": "application/octet-stream"},
        body=data_bin,
        timeout=max(120.0, min(600.0, len(data_bin) / 1e6 * 30)),
    )
    if put_code not in (200, 201, 204):
        errs.append(
            {
                "kind": "pyannote_upload_failed",
                "detail": put_raw.decode("utf-8", errors="replace")[:800],
                "http_status": put_code,
            }
        )
        return DiarRunOutcome([], "stub", duration_s, errs)

    di_body = json.dumps({"url": media_ref}).encode("utf-8")
    code2, raw2 = _http_request("POST", f"{base}/v1/diarize", headers=auth_h, body=di_body, timeout=60.0)
    if code2 != 200:
        errs.append(
            {
                "kind": "pyannote_diarize_submit_failed",
                "detail": raw2.decode("utf-8", errors="replace")[:800],
                "http_status": code2,
            }
        )
        return DiarRunOutcome([], "stub", duration_s, errs)
    try:
        dj = json.loads(raw2.decode("utf-8"))
    except json.JSONDecodeError as e:
        errs.append({"kind": "pyannote_diarize_submit_parse", "detail": str(e)})
        return DiarRunOutcome([], "stub", duration_s, errs)
    job_id = dj.get("jobId")
    if not job_id:
        errs.append({"kind": "pyannote_no_job_id", "detail": str(dj)[:400]})
        return DiarRunOutcome([], "stub", duration_s, errs)

    deadline = time.monotonic() + max(30.0, max_wait)
    poll_headers = {"Authorization": f"Bearer {api_key}"}
    last_status = ""
    while time.monotonic() < deadline:
        jc, jr = _http_request("GET", f"{base}/v1/jobs/{job_id}", headers=poll_headers, timeout=60.0)
        if jc != 200:
            errs.append(
                {
                    "kind": "pyannote_job_poll_failed",
                    "detail": jr.decode("utf-8", errors="replace")[:600],
                    "http_status": jc,
                }
            )
            return DiarRunOutcome([], "stub", duration_s, errs)
        try:
            job = json.loads(jr.decode("utf-8"))
        except json.JSONDecodeError:
            time.sleep(max(1.0, poll_s))
            continue
        last_status = str(job.get("status", ""))
        if last_status == "succeeded":
            output = job.get("output")
            segments_data: list[dict[str, Any]] = []
            if isinstance(output, dict):
                segments_data = list(output.get("diarization") or [])
            spans: list[DiarizedSpan] = []
            for seg in segments_data:
                try:
                    spk = str(seg.get("speaker", "")).strip() or "?"
                    t0 = float(seg.get("start", 0.0))
                    t1 = float(seg.get("end", 0.0))
                except (TypeError, ValueError):
                    continue
                if t1 <= t0:
                    continue
                cf = seg.get("confidence")
                cnf = None
                if isinstance(cf, dict) and cf:
                    try:
                        cnf = max(float(v) for v in cf.values() if isinstance(v, (int, float)))
                    except (TypeError, ValueError):
                        cnf = None
                spans.append(DiarizedSpan(t0=t0, t1=t1, label=spk, conf=cnf))
            spans.sort(key=lambda s: (s.t0, s.label))
            if spans:
                return DiarRunOutcome(spans, "pyannote_api", duration_s, errs)
            errs.append({"kind": "pyannote_api_empty_segments", "detail": str(output)[:400]})
            return DiarRunOutcome([], "stub", duration_s, errs)
        if last_status in ("failed", "canceled"):
            errs.append({"kind": "pyannote_job_terminal", "detail": last_status})
            return DiarRunOutcome([], "stub", duration_s, errs)
        time.sleep(max(1.0, poll_s))

    errs.append({"kind": "pyannote_job_timeout", "detail": f"last_status={last_status} job_id={job_id}"})
    return DiarRunOutcome([], "stub", duration_s, errs)


def format_diar_for_prompt(outcome: DiarRunOutcome, *, max_lines: int = 24) -> str:
    if not outcome.segments:
        return (
            f"backend={outcome.backend} duration_s={outcome.duration_s:.2f} "
            "speaker_spans=[]"
        )
    lines = [f"backend={outcome.backend} duration_s={outcome.duration_s:.2f}"]
    for s in outcome.segments[:max_lines]:
        lc = "" if s.conf is None else f" conf={s.conf:.3f}"
        lines.append(f"[{s.t0:.2f},{s.t1:.2f}] speaker={s.label}{lc}")
    if len(outcome.segments) > max_lines:
        lines.append(f"[truncated remaining={len(outcome.segments) - max_lines}]")
    return "\n".join(lines)


def diar_with_vad_fallback(diar: DiarRunOutcome, vad: VadRunOutcome | None) -> DiarRunOutcome:
    """Use VAD segments as pseudo speaker labels when diar is empty (short clips)."""
    if diar.segments:
        return diar
    if vad is not None and vad.segments:
        fake = [
            DiarizedSpan(t0=float(s.t0), t1=float(s.t1), label=f"vad_{i}", conf=s.conf)
            for i, s in enumerate(vad.segments[:24])
        ]
        return DiarRunOutcome(
            fake,
            f"vad_as_diar_proxy({vad.backend})",
            vad.duration_s,
            [*list(diar.errors), *list(vad.errors)],
        )
    return diar


def vad_proxy_prompt_footer(diar_backend: str) -> str:
    """Append to sheet prompts when ``diar_with_vad_fallback`` produced pseudo speaker labels."""
    if "vad_as_diar_proxy" not in diar_backend:
        return ""
    return "disclaimer=speaker_labels_are_vad_energy_segments_not_true_speaker_identity\n"


def diarize_wav_path(wav_path: str | Path) -> DiarRunOutcome:
    path = Path(wav_path).resolve()
    errs: list[dict[str, Any]] = []
    if not path.is_file():
        return DiarRunOutcome([], "none", 0.0, [{"kind": "file_missing", "detail": str(path)}])

    backend = _select_diar_backend()

    duration_s = 0.0
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            duration_s = wf.getnframes() / float(wf.getframerate() or 1)
    except Exception:
        duration_s = 0.0

    if backend == "pyannote_api":
        return _diarize_pyannote_api(path, duration_s, errs)

    if backend != "pyannote":
        return DiarRunOutcome([], backend, duration_s, errs)

    api_key_cloud = os.getenv("PYANNOTE_API_KEY", "").strip()
    if api_key_cloud:
        return _diarize_pyannote_api(path, duration_s, errs)

    token_raw = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
    )
    if not token_raw:
        errs.append({"kind": "hf_token_missing", "detail": "set HF_TOKEN for pyannote backend"})
        return DiarRunOutcome([], "stub", duration_s, errs)
    token: str | bool = token_raw

    model_id = (
        os.getenv("AV_SPEAKERBENCH_PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1").strip()
        or "pyannote/speaker-diarization-community-1"
    )
    try:
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]

        try:
            pipeline = Pipeline.from_pretrained(model_id, token=token)  # type: ignore[arg-type]
        except TypeError:
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)  # type: ignore[arg-type]

        diar_ann = pipeline(str(path))

        spans: list[DiarizedSpan] = []

        if hasattr(diar_ann, "itertracks"):
            try:
                for turn, _, label in diar_ann.itertracks(yield_label=True):  # type: ignore[attr-defined]
                    spans.append(
                        DiarizedSpan(
                            t0=float(getattr(turn, "start", 0.0)),
                            t1=float(getattr(turn, "end", 0.0)),
                            label=str(label),
                            conf=None,
                        )
                    )
            except Exception:
                pass

        spans.sort(key=lambda s: (s.t0, s.label))
        if spans:
            return DiarRunOutcome(spans, "pyannote", duration_s, errs)

        errs.append({"kind": "diar_parse_unsupported_layout", "detail": repr(type(diar_ann))})
        return DiarRunOutcome([], "stub", duration_s, errs)

    except ImportError:
        errs.append({"kind": "diar_import_missing", "detail": "pip install pyannote.audio"})
        return DiarRunOutcome([], "stub", duration_s, errs)
    except Exception as e:  # noqa: BLE001
        errs.append({"kind": "diar_inference_failed", "detail": str(e)})
        return DiarRunOutcome([], "stub", duration_s, errs)
