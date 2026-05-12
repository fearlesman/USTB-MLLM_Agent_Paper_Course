"""Optional perception backends invoked from Skills (see ``audio_*`` modules)."""

from .audio_asr import (
    AsrRunOutcome,
    AsrSegment,
    allow_stub_asr_text_prompt,
    asr_anchor_windows,
    format_asr_for_prompt,
)
from .audio_diar import DiarRunOutcome, DiarizedSpan, diarize_wav_path, format_diar_for_prompt
from .audio_prosody import ProsodyOutcome, discrete_prosody_over_vad
from .audio_rms_meter import RmsMeterOutcome, rms_peak_meter
from .audio_vad import VadRunOutcome, vad_segments_from_wav_path
from .benchmark_timecode import dataset_span_seconds, parse_benchmark_clock
from .media_probe import MediaProbeOutcome, format_probe_for_prompt, probe_media_file

__all__ = [
    "AsrRunOutcome",
    "AsrSegment",
    "DiarRunOutcome",
    "DiarizedSpan",
    "MediaProbeOutcome",
    "ProsodyOutcome",
    "RmsMeterOutcome",
    "VadRunOutcome",
    "allow_stub_asr_text_prompt",
    "asr_anchor_windows",
    "dataset_span_seconds",
    "diarize_wav_path",
    "discrete_prosody_over_vad",
    "format_asr_for_prompt",
    "format_diar_for_prompt",
    "format_probe_for_prompt",
    "parse_benchmark_clock",
    "probe_media_file",
    "rms_peak_meter",
    "vad_segments_from_wav_path",
]
