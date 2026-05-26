#!/usr/bin/env python3
"""Low-cost checks that the agent facade routes prompts through the orchestrator."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import types
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


def _install_optional_backend_stubs() -> None:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.types = types.SimpleNamespace()
    google.genai = genai
    sys.modules.setdefault("google", google)
    sys.modules.setdefault("google.genai", genai)

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)

    dashscope = types.ModuleType("dashscope")
    dashscope.base_http_api_url = ""
    dashscope.MultiModalConversation = types.SimpleNamespace(call=lambda *a, **k: None)
    sys.modules.setdefault("dashscope", dashscope)

    moviepy = types.ModuleType("moviepy")
    moviepy.VideoFileClip = object
    moviepy.AudioFileClip = object
    moviepy.CompositeAudioClip = object
    sys.modules.setdefault("moviepy", moviepy)


def _args(tmpdir: Path) -> Namespace:
    return Namespace(
        model_name="unit-smoke",
        task_id=None,
        audio=False,
        visual=False,
        temp_dir="temp",
        data_path=str(tmpdir),
    )


def _dataset() -> list[dict[str, str]]:
    return [
        {
            "question_id": "q1",
            "video_id": "v1",
            "visual_path": "v.mp4",
            "audio_path": "a.wav",
            "audio_visual_path": "av.mp4",
            "choices": "['A. yes', 'B. no', 'C. maybe', 'D. unknown']",
            "question": 'Who says "hello" first?',
            "answer": "A",
            "category": "audio-centric",
            "sub_category": "speech",
            "task_id": "Speech Recognition",
            "start_time": "00:00",
            "end_time": "00:05",
        }
    ]


def _run_once(*, inject: bool, tmpdir: Path):
    for key in (
        "AV_SPEAKERBENCH_SKILLS",
        "AV_SPEAKERBENCH_SKILL_INJECT",
        "AV_SPEAKERBENCH_SKILLS_ALLOWLIST",
        "AV_SPEAKERBENCH_ALLOW_SYNTHETIC_ASR",
        "AV_SPEAKERBENCH_STUB_ASR_TEXT",
    ):
        os.environ.pop(key, None)
    os.environ["AV_SPEAKERBENCH_EVAL_TRACK"] = "agent"
    os.environ["AV_SPEAKERBENCH_SKILLS_ALLOWLIST"] = "meta_banner"
    if inject:
        os.environ["AV_SPEAKERBENCH_SKILL_INJECT"] = "1"

    model = importlib.import_module("model")
    model._call_backend = lambda **kwargs: "A"
    result, records = model.inference(_args(tmpdir), _dataset())
    record = records["q1"]
    assert result["level 3: Speech Recognition"]["matched"] == 1
    return record


def _assert_active_injection(tmpdir: Path) -> None:
    record = _run_once(inject=True, tmpdir=tmpdir)
    prompt = record["agent_model_prompt"]
    assert "Structured_skill_evidence" in prompt, prompt
    assert record["agent_prompt_changed"] is True
    assert record["agent_evidence_injected"] is True
    assert record["agent_orchestrator_ran"] is True
    assert record["skills_invoked"], record


def _assert_parity_without_injection(tmpdir: Path) -> None:
    record = _run_once(inject=False, tmpdir=tmpdir)
    assert record["agent_model_prompt"] == record["question_prompt"]
    assert record["agent_prompt_changed"] is False
    assert record["agent_evidence_injected"] is False
    assert record["agent_orchestrator_ran"] is True
    assert record["skills_invoked"], record


def _assert_stable_fingerprint() -> None:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(AGENT_DIR)!r}); "
        "from orchestrator.trace_sink import prompt_fingerprint; "
        "print(prompt_fingerprint('same prompt'))"
    )
    a = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    b = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert a == b and a, (a, b)


def main() -> None:
    _install_optional_backend_stubs()
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        os.chdir(AGENT_DIR)
        _assert_active_injection(tmpdir)
        _assert_parity_without_injection(tmpdir)
        _assert_stable_fingerprint()
    print("agent smoke checks passed")


if __name__ == "__main__":
    main()
