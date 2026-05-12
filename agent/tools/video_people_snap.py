"""Decode 1–3 video frames near anchor time; optional Ultralytics ``person`` count."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from subprocess import run as subprocess_run
from typing import Any, Sequence


@dataclass(frozen=True)
class SnapFrameOutcome:
    time_s: float
    jpeg_path: str | None

    @property
    def ok(self) -> bool:
        return bool(self.jpeg_path and Path(self.jpeg_path).is_file())


@dataclass
class PeopleSnapOutcome:
    frames: list[SnapFrameOutcome]
    person_counts: list[int | None]
    backend: str
    notes: str
    errors: list[dict[str, Any]]
    n_frames_extracted_ok: int = 0


def _ffmpeg_bin() -> str:
    return os.getenv("AV_SPEAKERBENCH_FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"


def extract_frame_jpeg(video_path: str | Path, t_s: float, out_path: Path) -> tuple[bool, str]:
    vid = Path(video_path)
    if not vid.is_file():
        return False, "video_missing"
    t_s = max(0.0, float(t_s))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t_s:.3f}",
        "-i",
        str(vid),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        "-y",
        str(out_path),
    ]
    try:
        p: CompletedProcess[str] = subprocess_run(cmd, capture_output=True, text=True, timeout=90)
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    if p.returncode != 0 or not out_path.is_file():
        return False, (p.stderr or "")[:400]
    return True, ""


def _try_yolo_person_count(jpeg_paths: Sequence[str]) -> tuple[list[int | None], str]:
    counts: list[int | None] = []
    backend = os.getenv("AV_SPEAKERBENCH_YOLO_WEIGHTS", "yolov8n.pt").strip() or "yolov8n.pt"
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        return [None] * len(jpeg_paths), "ultralytics_not_installed"

    try:
        model = YOLO(backend)
    except Exception:
        model = YOLO("yolov8n.pt")

    coco_person = 0
    for jp in jpeg_paths:
        if not jp or not Path(jp).is_file():
            counts.append(None)
            continue
        try:
            r = model.predict(jp, verbose=False, classes=[coco_person], conf=0.25)
            n = len(r[0].boxes) if r and len(r[0].boxes) else 0  # noqa: SLF001
            counts.append(int(n))
        except Exception:  # noqa: BLE001
            counts.append(None)
    return counts, "yolov8_person_class0"


def snap_and_count_people(
    video_path: str | Path,
    times_s: Sequence[float],
    *,
    enable_yolo: bool | None = None,
) -> PeopleSnapOutcome:
    errs: list[dict[str, Any]] = []
    vp = Path(video_path)
    if not vp.is_file():
        return PeopleSnapOutcome([], [], "none", "no_video", [{"kind": "file_missing", "detail": str(vp)}], 0)

    if enable_yolo is None:
        enable_yolo = os.getenv("AV_SPEAKERBENCH_VIS_COUNT_YOLO", "").strip().lower() in ("1", "true", "yes")

    tmpdir = tempfile.mkdtemp(prefix="avsb_snap_")
    frames_out: list[SnapFrameOutcome] = []
    paths: list[str] = []
    counts: list[int | None] = []
    snap_backend = "ffmpeg_frame_extract"
    snap_notes = "(set AV_SPEAKERBENCH_VIS_COUNT_YOLO=1 for COCO person-class counts)"

    try:
        for i, t in enumerate(times_s[:5]):
            outp = Path(tmpdir) / f"f{i}.jpg"
            ok, detail = extract_frame_jpeg(vp, float(t), outp)
            pth = str(outp) if ok else None
            if not ok:
                errs.append({"kind": "ffmpeg_frame_failed", "detail": detail, "t_s": float(t)})
            frames_out.append(SnapFrameOutcome(time_s=float(t), jpeg_path=pth))
            if pth:
                paths.append(pth)

        counts = [None] * len(frames_out)
        if enable_yolo and paths:
            rawc, detector_note = _try_yolo_person_count(paths)
            j = 0
            for i, fr in enumerate(frames_out):
                if fr.jpeg_path:
                    counts[i] = rawc[j] if j < len(rawc) else None
                    j += 1
            snap_backend = "yolov8_person_detector"
            snap_notes = detector_note
    finally:
        n_ok = sum(1 for f in frames_out if f.jpeg_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

    cleaned = [SnapFrameOutcome(f.time_s, None) for f in frames_out]
    return PeopleSnapOutcome(cleaned, counts, snap_backend, snap_notes, errs, n_ok)


def format_people_snap(ot: PeopleSnapOutcome) -> str:
    xs = ",".join(f"{f.time_s:.2f}" for f in ot.frames)
    cs = ",".join("n/a" if c is None else str(c) for c in ot.person_counts[: len(ot.frames)])
    return (
        f"[viz_people_snap] backend={ot.backend} n_frames_ok={ot.n_frames_extracted_ok} anchor_times_s=[{xs}] "
        f"person_counts=[{cs}] note={ot.notes}\n"
        "disclaimer=headcount_from_detector_not_unique_face_id_matching\n"
    )
