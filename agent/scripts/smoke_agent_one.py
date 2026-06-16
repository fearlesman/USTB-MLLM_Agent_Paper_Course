from __future__ import annotations

import ast
import os
import sys
from argparse import Namespace
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from dataset.get_dataset import get_dataset
from orchestrator.runner import augment_prompt_for_inference


def _build_prompt(question: dict) -> str:
    raw_choices = question.get("choices", [])
    if isinstance(raw_choices, str):
        try:
            choices = ast.literal_eval(raw_choices)
        except Exception:
            choices = [raw_choices]
    else:
        choices = list(raw_choices)
    choices_str = "\n".join(str(x) for x in choices)
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        f"{question['question']}\n{choices_str}\nThe best answer is:"
    )


def main() -> None:
    os.environ.setdefault("AV_SPEAKERBENCH_SKILL_INJECT", "1")
    os.environ.setdefault("AV_SPEAKERBENCH_SKILLS", "on")
    os.environ.setdefault("AV_SPEAKERBENCH_EVAL_TRACK", "agent")

    ds = get_dataset(use_local_metadata=True, sample_fraction=0.01, sample_seed=0, stratify_key="category")
    if len(ds) == 0:
        raise SystemExit("Dataset is empty after sampling.")
    q = ds[0]
    data_root = Path(os.environ.get("AV_SPEAKERBENCH_DATA_ROOT", "")) if os.environ.get("AV_SPEAKERBENCH_DATA_ROOT") else None
    if data_root is None or not data_root.exists():
        from dataset.paths import DATASET_ROOT

        data_root = DATASET_ROOT

    video_path = str((data_root / q["visual_path"]).resolve())
    audio_path = str((data_root / q["audio_path"]).resolve())
    combined_path = str((data_root / q["audio_visual_path"]).resolve())

    prompt = _build_prompt(q)
    prep = augment_prompt_for_inference(
        question_prompt=prompt,
        question=q,
        args=Namespace(model_name="smoke", audio=False, visual=False, temp_dir="temp"),
        video_path=video_path,
        audio_path=audio_path,
        combined_path=combined_path,
    )

    print("question_id:", q.get("id", q.get("question_id", "n/a")))
    print("skills_invoked:", ",".join(prep.skills_invoked))
    print("bottleneck_tags:", ",".join(prep.bottleneck_tags))
    print("errors:", len(prep.errors))
    print("--- prompt preview ---")
    print(prep.final_prompt[:3000])


if __name__ == "__main__":
    main()
