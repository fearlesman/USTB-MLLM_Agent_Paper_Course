"""
Rebuild ``record/*_experiment.json`` from existing ``result/*.json`` and ``record/*_record*.json``.

Use the **same CLI flags as the original eval** so paths resolve like ``main.py`` (especially Qwen3 / Gemini naming).

Example (your previous run):

  python scripts/backfill_experiment_manifest.py \\
    --model_name Qwen3-Omni-3B \\
    --sample_fraction 0.1 --sample_seed 42 \\
    --use_local_metadata --resume

Run from the AV-SpeakerBench repo root so ``result/`` and ``record/`` resolve correctly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    root = _repo_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    from dataset.paths import DATASET_ROOT
    from model import experiment_record_path, experiment_result_path, write_run_manifest_only

    parser = argparse.ArgumentParser(description="Backfill *_experiment.json run manifest.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cfg_path", default="eval_configs/video_llama_eval_withaudio.yaml")
    parser.add_argument("--options", nargs="+", default=None)
    parser.add_argument("--gpu-id", type=int, default=0, dest="gpu_id")
    parser.add_argument("--model_type", type=str, default="vicuna")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--task_id", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--sub_category", type=str, default=None)
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--data_path", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--dashscope_model", type=str, default="qwen3.5-omni-plus-2026-03-15")
    parser.add_argument("--local_qwen_weights", action="store_true")
    parser.add_argument("--use_local_metadata", action="store_true")
    parser.add_argument("--hub_metadata", action="store_true")
    parser.add_argument("--sample_fraction", type=float, default=None)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--stratify_key", type=str, default="category")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_clip_bytes", type=int, default=None)
    parser.add_argument("--openai_safe_clip_filter", action="store_true")
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Override path to aggregate metrics JSON (default: infer from model_name).",
    )
    parser.add_argument(
        "--records-json",
        default=None,
        help="Override path to per-question records JSON (default: infer from model_name).",
    )
    parser.add_argument(
        "--num_dataset_items",
        type=int,
        default=None,
        help="Stored in manifest; defaults to len(records) when omitted.",
    )
    parser.add_argument(
        "--inference_wall_seconds",
        type=float,
        default=None,
        help="Optional wall time (seconds); omit if unknown.",
    )

    args = parser.parse_args()
    if getattr(args, "local_qwen_weights", False):
        args.dashscope_model = None
    elif getattr(args, "dashscope_model", "") == "":
        args.dashscope_model = None

    res_path = args.metrics_json or experiment_result_path(args)
    rec_path = args.records_json or experiment_record_path(args)

    if not os.path.isfile(res_path):
        print(f"Missing metrics file: {res_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(rec_path):
        print(f"Missing records file: {rec_path}", file=sys.stderr)
        return 1

    with open(res_path, encoding="utf-8") as f:
        result_obj = json.load(f)
    with open(rec_path, encoding="utf-8") as f:
        records_obj = json.load(f)

    nd = args.num_dataset_items if args.num_dataset_items is not None else len(records_obj)
    out = write_run_manifest_only(
        args,
        result_obj,
        records_obj,
        elapsed_sec=args.inference_wall_seconds,
        num_dataset_items=nd,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
