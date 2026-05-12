import os

# huggingface_hub uses these when talking to the API (file tree / dataset card). Defaults are ~10s and often fail on slow or filtered links while blob downloads still work.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

from dataset import *
from model import inference, flush_experiment_artifacts
from torch.utils.data import DataLoader
import argparse
import time
import sys
# proj_root = os.path.abspath(os.path.join(__file__, "..", "model", "open", "LAVIS_XInstructBLIP"))
# if proj_root not in sys.path:
#     sys.path.insert(0, proj_root)
# print("\n".join(sys.path))

from dataset.paths import DATASET_ROOT as _DATASET_ROOT

_DEFAULT_DATA_PATH = str(_DATASET_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, required = True, help = "model name to be tested")
    parser.add_argument("--cfg_path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument("--num_frames", type=int, help="the number of frames for phi4")
    parser.add_argument("--task_id", type=str, default = None)
    parser.add_argument("--category", type=str, default = None)
    parser.add_argument("--sub_category", type=str, default = None)
    parser.add_argument("--audio", dest="audio", action='store_true')
    parser.add_argument("--visual", dest="visual", action='store_true')
    parser.add_argument("--temp_dir", type=str, default = "temp")
    parser.add_argument(
        "--data_path",
        type=str,
        default=_DEFAULT_DATA_PATH,
        help="Dataset root (clips + test.csv). Default: Holistic_AVQA_bench next to the repo root (parent of baseline/).",
    )
    parser.add_argument(
        "--dashscope_model",
        type=str,
        default="qwen3.5-omni-plus-2026-03-15",
        help="DashScope Qwen3-Omni API model id (default: qwen3.5-omni-plus-2026-03-15). "
        "Ignored for Qwen3 when --local_qwen_weights is set. Requires API access for API mode.",
    )
    parser.add_argument(
        "--local_qwen_weights",
        action="store_true",
        help="For Qwen3, load weights from disk instead of DashScope (--dashscope_model ignored).",
    )
    parser.add_argument(
        "--use_local_metadata",
        action="store_true",
        help="Require local test.csv/parquet under --data_path; error if missing (no Hub fallback).",
    )
    parser.add_argument(
        "--hub_metadata",
        action="store_true",
        help="Load the question table from Hugging Face Hub only (ignore local test.csv/parquet).",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=None,
        help="If set in (0,1], stratified subsample per --stratify_key (default category), ~fraction per bucket, at least 1 per bucket.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="RNG seed for stratified subsample (default 0).",
    )
    parser.add_argument(
        "--stratify_key",
        type=str,
        default="category",
        help="Metadata column for stratified sampling (default: category).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Load existing record/*/ checkpoint (same naming as outputs) before eval and flush after each "
            "question — safe to rerun if the process stops mid-stream. Match --sample_seed/--task_id/ablation flags "
            "to the original run."
        ),
    )
    parser.add_argument(
        "--max_clip_bytes",
        type=int,
        default=None,
        help=(
            "Before stratified sampling, drop rows whose local clip file(s) exceed this size (bytes), or are "
            "missing. For default A+V + OpenAI data: URLs, ~14680064 (14MiB) avoids DashScope per-item ~20MiB "
            "base64 limits. Override with env AV_SPEAKERBENCH_MAX_CLIP_BYTES."
        ),
    )
    parser.add_argument(
        "--openai_safe_clip_filter",
        action="store_true",
        help=(
            "If --max_clip_bytes is not set, apply a 14MiB per-file cap (audiovisual + visual + audio paths) "
            "so OpenAI-compatible data: URIs stay under the gateway limit."
        ),
    )

    args = parser.parse_args()
    if args.local_qwen_weights:
        args.dashscope_model = None
    elif getattr(args, "dashscope_model", None) == "":
        args.dashscope_model = None

    def _clip_size_media_check(a) -> str:
        if a.audio and not a.visual:
            return "audio"
        if a.visual and not a.audio:
            return "visual"
        return "all"

    max_clip = args.max_clip_bytes
    if max_clip is None:
        env_m = os.environ.get("AV_SPEAKERBENCH_MAX_CLIP_BYTES", "").strip()
        if env_m:
            max_clip = int(env_m)
    if getattr(args, "openai_safe_clip_filter", False) and max_clip is None:
        max_clip = 14 * 1024 * 1024

    test_loader = get_dataset(
        category=args.category,
        sub_category=args.sub_category,
        task_id=args.task_id,
        data_path=args.data_path,
        use_local_metadata=args.use_local_metadata,
        force_hub_metadata=args.hub_metadata,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
        stratify_key=args.stratify_key,
        max_clip_bytes=max_clip,
        clip_size_media_check=_clip_size_media_check(args),
    )
    if max_clip is not None:
        print(
            f"Clip prefilter: max {max_clip} bytes (paths={_clip_size_media_check(args)}); "
            f"dropped {getattr(test_loader, 'oversized_clips_dropped_prefilter', 0)} rows before stratified sampling."
        )

    os.makedirs("result", exist_ok = True)
    os.makedirs("record", exist_ok = True)


    t_start = time.perf_counter()
    result, records = inference(args, test_loader)
    elapsed = time.perf_counter() - t_start

    if args.sample_fraction is not None and 0 < args.sample_fraction <= 1:
        from collections import Counter

        _cats = Counter(test_loader[i]["category"] for i in range(len(test_loader)))
        result["_subset_meta"] = {
            "sample_fraction": args.sample_fraction,
            "sample_seed": args.sample_seed,
            "stratify_key": args.stratify_key,
            "num_questions": len(test_loader),
            "per_stratum": dict(sorted(_cats.items())),
            "max_clip_bytes_prefilter": getattr(test_loader, "max_clip_bytes_prefilter", None),
            "oversized_clips_dropped_prefilter": getattr(
                test_loader, "oversized_clips_dropped_prefilter", 0
            ),
        }

    res_path, rec_path, run_path = flush_experiment_artifacts(
        args,
        result,
        records,
        elapsed_sec=elapsed,
        num_dataset_items=len(test_loader),
    )
    print("=" * 5 + f"evaluate {args.model_name}" + "=" * 5)
    print(f"Wrote metrics JSON: {res_path}")
    print(f"Wrote per-question records: {rec_path}")
    print(f"Wrote experiment run manifest: {run_path}")


if __name__ == "__main__":
    os.environ.setdefault(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )
    main()