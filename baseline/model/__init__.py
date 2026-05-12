from .closed import *
from .closed.qwen3omni_api.inference import ClipPayloadTooLargeError
from .open_model import *
import os, tqdm, textwrap, ast, re
# moviepy < 2.0
# from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
# moviepy > 2.0
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
import shutil
import time
import random
random.seed(0)
import json
from datetime import datetime, timezone

result = {}
records = {}


def _is_qwen3_model(name: str) -> bool:
    """Match DashScope ids like ``qwen3.5-omni-plus-2026-03-15`` (case-insensitive)."""
    return "qwen3" in (name or "").lower()


def _task_id_segment(args) -> str:
    """Filename segment matching historical ``f\"...{args.task_id}...\"`` (``None`` → ``\"None\"``)."""
    return "None" if args.task_id is None else str(args.task_id)


def _record_json_path(args) -> str:
    return os.path.join(
        "record",
        f"{args.model_name}_record_{_task_id_segment(args)}_audio_{args.audio}_visual_{args.visual}.json",
    )


# Models that historically wrote ``record/<name>_record_<task>.json`` (no audio/visual suffix).
_LEGACY_RECORD_MODELS = frozenset(
    {
        "video_llama_13b",
        "video_llama_7b",
        "video_llama2_7b",
        "pandagpt_7b",
        "pandagpt_13b",
        "phi4",
        "onellm",
        "uio2-large",
        "uio2-xl",
        "uio2-xxl",
        "NExTGPT",
        "AnyGPT",
        "vita1",
        "vita1_5",
        "xblip_7b",
        "xblip_13b",
        "Qwen2.5-Omni-3B",
        "Qwen2.5-Omni-7B",
        "ola",
    }
)


def experiment_result_path(args) -> str:
    """Path for aggregate metrics JSON under ``result/`` (stable, filesystem-safe)."""
    if args.model_name == "stream-omni":
        return os.path.join("result", "StreamOmni.json")
    safe = args.model_name.replace("/", "_").replace("\\", "_")
    return os.path.join("result", f"{safe}.json")


def experiment_record_path(args) -> str:
    """Per-question record path; matches checkpoint resume paths for Gemini / Qwen3 API."""
    if "gemini" in args.model_name or _is_qwen3_model(args.model_name):
        return _record_json_path(args)
    task = _task_id_segment(args)
    if args.model_name == "stream-omni":
        return os.path.join("record", f"StreamOmni_record_{task}.json")
    base = args.model_name.replace("/", "_").replace("\\", "_")
    if args.model_name in _LEGACY_RECORD_MODELS:
        return os.path.join("record", f"{base}_record_{task}.json")
    return os.path.join("record", f"{base}_record_{task}_audio_{args.audio}_visual_{args.visual}.json")


def experiment_run_manifest_path(args) -> str:
    """Companion JSON next to ``experiment_record_path`` describing this benchmark run."""
    rec = experiment_record_path(args)
    stem, ext = os.path.splitext(rec)
    if not ext:
        ext = ".json"
    return f"{stem}_experiment{ext}"


def _jsonable_arg_value(val):
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return [_jsonable_arg_value(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _jsonable_arg_value(v) for k, v in val.items()}
    return repr(val)


def _args_snapshot(args) -> dict:
    return {k: _jsonable_arg_value(v) for k, v in vars(args).items()}


def _summarize_records(records: dict) -> dict:
    n = len(records)
    if not n:
        return {"num_questions": 0, "matched": 0, "accuracy_percent": 0.0, "empty_response_count": 0}
    matched = sum(1 for r in records.values() if r.get("matched"))
    empty = sum(1 for r in records.values() if not str(r.get("llm response", "")).strip())
    return {
        "num_questions": n,
        "matched": matched,
        "accuracy_percent": round(100.0 * matched / n, 4),
        "empty_response_count": empty,
    }


def _aggregate_metrics_preview(result: dict) -> dict:
    """Compact view of ``result`` (per-bucket matched/total/accuracy) without duplicating huge blobs."""
    preview: dict = {}
    for k, v in result.items():
        if k == "_subset_meta" and isinstance(v, dict):
            preview[k] = v
            continue
        if isinstance(v, dict) and "matched" in v and "total" in v:
            preview[k] = {
                "matched": v["matched"],
                "total": v["total"],
                "accuracy": v.get("accuracy"),
            }
    return preview


def _experiment_manifest_dict(
    args,
    result,
    records,
    *,
    res_path: str,
    rec_path: str,
    run_path: str,
    elapsed_sec: float | None = None,
    num_dataset_items: int | None = None,
) -> dict:
    return {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "metrics_json": os.path.abspath(res_path),
        "records_json": os.path.abspath(rec_path),
        "run_manifest_json": os.path.abspath(run_path),
        "num_dataset_items": num_dataset_items,
        "inference_wall_seconds": None if elapsed_sec is None else round(float(elapsed_sec), 6),
        "args": _args_snapshot(args),
        "record_summary": _summarize_records(records),
        "aggregate_metrics": _aggregate_metrics_preview(result),
    }


def write_run_manifest_only(
    args,
    result,
    records,
    *,
    elapsed_sec: float | None = None,
    num_dataset_items: int | None = None,
) -> str:
    """Write only ``*_experiment.json`` (reuse existing metrics + records on disk paths implied by ``args``)."""
    run_path = experiment_run_manifest_path(args)
    res_path = experiment_result_path(args)
    rec_path = experiment_record_path(args)
    manifest = _experiment_manifest_dict(
        args,
        result,
        records,
        res_path=res_path,
        rec_path=rec_path,
        run_path=run_path,
        elapsed_sec=elapsed_sec,
        num_dataset_items=num_dataset_items,
    )
    _atomic_write_json(run_path, manifest)
    return run_path


def flush_experiment_artifacts(
    args,
    result,
    records,
    *,
    elapsed_sec: float | None = None,
    num_dataset_items: int | None = None,
) -> tuple[str, str, str]:
    """Atomically write metrics JSON, per-question records, and a run manifest for every completed eval."""
    res_path = experiment_result_path(args)
    rec_path = experiment_record_path(args)
    run_path = experiment_run_manifest_path(args)
    _atomic_write_json(res_path, result)
    _atomic_write_json(rec_path, records)
    manifest = _experiment_manifest_dict(
        args,
        result,
        records,
        res_path=res_path,
        rec_path=rec_path,
        run_path=run_path,
        elapsed_sec=elapsed_sec,
        num_dataset_items=num_dataset_items,
    )
    _atomic_write_json(run_path, manifest)
    return res_path, rec_path, run_path


def _resume_requested(args) -> bool:
    if getattr(args, "resume", False):
        return True
    return os.getenv("AV_SPEAKERBENCH_RESUME", "").strip().lower() in ("1", "true", "yes")


def _checkpoint_records_supported(args) -> bool:
    return "gemini" in args.model_name or _is_qwen3_model(args.model_name)


def _atomic_write_json(path: str, obj) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


answer_prefixes = [
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is"
    "The correct option is",
    "Best answer:"
    "Best option:",
    "Answer:",
    "Option:",
    "The correct answer",
    "The correct option",
    "Based",
    "Correct answer",
    "\u261e",
    "<|im_end|>"
]

def extract_characters_regex(s):
    if s is None:
        c = ['A', 'B', 'C', 'D'][random.randint(0,3)]
        return c
    
    s = s.strip()
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    pattern = r"[.,:!'\";/\?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]"
    
    s = re.sub(pattern, " ", s)

    parsed = s.split()
    # print(parsed)
    matches = None
    for char in parsed:
        if char == "A" or char == "B" or char == "C" or char == "D" or char == "E":
            matches = char
            break

    if matches is None:
        return ""
    return matches[0]

def process_answer(gt, pred, question):
    if gt == pred:
        result[f'level 3: {question["task_id"]}']["matched"] += 1
        result[f'level 2: {question["sub_category"]}']["matched"] += 1
        result[f'level 1: {question["category"]}']["matched"] += 1
    
    result[f'level 3: {question["task_id"]}']["accuracy"] = round(result[f'level 3: {question["task_id"]}']["matched"] / result[f'level 3: {question["task_id"]}']["total"] * 100, 2)
    result[f'level 2: {question["sub_category"]}']["accuracy"] = round(result[f'level 2: {question["sub_category"]}']["matched"] / result[f'level 2: {question["sub_category"]}']["total"] * 100, 2)
    result[f'level 1: {question["category"]}']["accuracy"] = round(result[f'level 1: {question["category"]}']["matched"] / result[f'level 1: {question["category"]}']["total"] * 100, 2)

    return (gt == pred)

def inference(args, dataset):
    global result, records

    result.clear()
    records.clear()

    temporary_dir = os.path.join(os.getcwd(), args.temp_dir)
    try:
        shutil.rmtree(temporary_dir)
    except Exception as e:
        pass

    os.makedirs(temporary_dir, exist_ok = True)

    if args.model_name == "video_llama_13b" or args.model_name == "video_llama_7b":
        chat = model_init(args)
    elif args.model_name == "video_llama2_7b":
        model, processor, tokenizer = video_llama2_model_init()
    elif "panda" in args.model_name:
        model = pandagpt_model_init(args)
    elif "phi" in args.model_name:
        model, processor, generation_config = phi4_model_init(args)
    elif "onellm" in args.model_name:
        model = onellm_model_init(args)
    elif "uio" in args.model_name:
        model, processor = uio2_model_init(args)
    elif "NExTGPT" in args.model_name:
        model = NExTGPT_model_init(args)
    elif "AnyGPT" in args.model_name:
        model = anygpt_model_init(args)
    elif args.model_name == "vita1":
        model, audio_processor, image_processor, tokenizer = vita1_model_init(args)
    elif args.model_name == "vita1_5":
        model, audio_processor, image_processor, tokenizer = vita1_5_model_init(args)
    elif "xblip" in args.model_name:
        model = xblip_model_init(args)
    elif "Qwen2.5" in args.model_name:
        model, processor = qwen2_5Omni_model_init(args)
    elif "stream-omni" in args.model_name:
        model, tokenizer, image_processor, cosyvoice = streamomni_model_init(args)
    elif "ola" in args.model_name:
        model, tokenizer, image_processor = ola_model_init(args)
    elif _is_qwen3_model(args.model_name):
        if getattr(args, "dashscope_model", None):
            model, processor = None, args.dashscope_model
        else:
            from .open_model.Qwen3Omni.inference import qwen3omni_model_init as _qwen3_local_init

            model, processor = _qwen3_local_init(args)

    cnt = 0
    wrong_cnt = 0

    record_ckpt_path = experiment_record_path(args)
    if _checkpoint_records_supported(args) and _resume_requested(args) and os.path.isfile(record_ckpt_path):
        with open(record_ckpt_path, encoding="utf-8") as f:
            records.update(json.load(f))

    _infer_limit = os.environ.get("AV_SPEAKERBENCH_INFERENCE_LIMIT")
    _infer_limit = int(_infer_limit) if _infer_limit else None

    for idx, question in tqdm.tqdm(enumerate(dataset), total = len(dataset)):
        if _infer_limit is not None and idx >= _infer_limit:
            break
        new_video_path = os.path.join(args.data_path, question["visual_path"])
        new_audio_path = os.path.join(args.data_path, question["audio_path"])
        new_combined_path = os.path.join(args.data_path, question["audio_visual_path"])

        choices = ast.literal_eval(question["choices"])
        choices_str = "\n".join(choices)

        question_prompt = f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n{question['question']}\n{choices_str}\nThe best answer is:"

        # break
        result[f'level 1: {question["category"]}'] = result.get(f'level 1: {question["category"]}', {})
        result[f'level 1: {question["category"]}']["matched"]= result[f'level 1: {question["category"]}'].get("matched", 0)
        result[f'level 1: {question["category"]}']["total"] = result[f'level 1: {question["category"]}'].get("total", 0) + 1

        result[f'level 2: {question["sub_category"]}'] = result.get(f'level 2: {question["sub_category"]}', {})
        result[f'level 2: {question["sub_category"]}']["matched"]= result[f'level 2: {question["sub_category"]}'].get("matched", 0)
        result[f'level 2: {question["sub_category"]}']["total"]= result[f'level 2: {question["sub_category"]}'].get("total", 0) + 1

        result[f'level 3: {question["task_id"]}'] = result.get(f'level 3: {question["task_id"]}', {})
        result[f'level 3: {question["task_id"]}']["matched"]= result[f'level 3: {question["task_id"]}'].get("matched", 0)
        result[f'level 3: {question["task_id"]}']["total"]= result[f'level 3: {question["task_id"]}'].get("total", 0) + 1
        
        # add more if needed for ablation
        id = question["question_id"]

        ans = ""
        if "gemini" in args.model_name:
            if id in records and records[id]["question_prompt"] == question_prompt:
                print(f"{id} exists")
                ans = records[id]["llm response"]
            elif args.audio:
                # print(new_audio_path)
                ans = gemini_process(new_audio_path, question_prompt, args.model_name, idx)
            elif args.visual:
                # print(new_video_path)
                ans = gemini_process(new_video_path, question_prompt, args.model_name, idx)
            else:
                ans = gemini_process(new_combined_path, question_prompt, args.model_name, idx)
        elif args.model_name == "video_llama_13b" or args.model_name == "video_llama_7b":
            ans = video_llama_process(chat, new_combined_path, question_prompt)
        elif args.model_name == "video_llama2_7b":
            ans = video_llama2_process(new_combined_path, question_prompt, model, processor, tokenizer)
            # print(ans)
        elif args.model_name == "pandagpt_7b" or args.model_name == "pandagpt_13b":
            ans = pandagpt_process(model, question_prompt, new_audio_path, new_video_path, 512, [])
        elif args.model_name == "phi4":
            ans = phi4_process(new_audio_path, new_video_path, args.num_frames, question_prompt, model = model, processor = processor, generation_config = generation_config)
        elif args.model_name == "onellm":
            ans = onellm_process(model, new_audio_path, new_video_path, question_prompt)
        elif args.model_name ==  "uio2-large" or args.model_name == "uio2-xl" or args.model_name == "uio2-xxl":
            ans = uio2_process(model, processor, new_combined_path, question_prompt)
        elif args.model_name ==  "NExTGPT":
            ans = NExTGPT_process(model, new_video_path, new_audio_path, question_prompt)
            ans = ans[0]
        elif args.model_name == "AnyGPT":
            ans = anygpt_process(model, new_video_path, new_audio_path, question_prompt)
        elif args.model_name == "vita1":
            ans = vita1_process(model, audio_processor, image_processor, tokenizer, new_video_path, new_audio_path, question_prompt)
        elif args.model_name == "vita1_5":
            ans = vita1_5_process(model, audio_processor, image_processor, tokenizer, new_video_path, new_audio_path, question_prompt)
        elif "xblip" in args.model_name:
            ans = xblip_process(model, new_audio_path, new_video_path, question_prompt)
        elif "Qwen2.5" in args.model_name:
            ans = qwen2_5Omni_process(model, processor, new_combined_path, question_prompt)
        elif "stream-omni" in args.model_name:
            ans = streamomni_process(model, tokenizer, image_processor, cosyvoice, new_audio_path, new_video_path, question_prompt)
        elif "ola" in args.model_name:
            ans = ola_process(model, tokenizer, image_processor, new_video_path, new_audio_path, question_prompt)
        elif _is_qwen3_model(args.model_name):
            if id in records and records[id]["question_prompt"] == question_prompt:
                print(f"{id} exists")
                ans = records[id]["llm response"]
            else:
                media_type = "video" if args.visual else ("audio" if args.audio else "both")
                media_path = new_video_path if args.visual else (new_audio_path if args.audio else new_combined_path)
                if model is None:
                    try:
                        ans = qwen3omni_api_process(
                            processor,
                            media_type,
                            media_path,
                            question_prompt,
                            split_visual_path=new_video_path,
                            split_audio_path=new_audio_path,
                        )
                    except ClipPayloadTooLargeError as cle:
                        print(f"[SKIP] clip exceeds DashScope data-uri/embed limit (~20MiB): {cle}")
                        ans = ""
                    except FileNotFoundError as fnf:
                        print(f"[SKIP] missing dataset media for question_id={id}: {fnf}")
                        ans = ""
                    except RuntimeError as err:
                        c = getattr(err, "__cause__", None)
                        if isinstance(c, FileNotFoundError):
                            print(f"[SKIP] missing dataset media for question_id={id}: {c}")
                            ans = ""
                        else:
                            raise
                else:
                    from .open_model.Qwen3Omni.inference import qwen3omni_process as _qwen3_local_process

                    try:
                        ans = _qwen3_local_process(model, processor, media_type, media_path, question_prompt)
                    except FileNotFoundError as fnf:
                        print(f"[SKIP] missing dataset media for question_id={id}: {fnf}")
                        ans = ""
            
        
        records[id] = {
            "video_id": question["video_id"],
            "task_id": question["task_id"],
            "question_prompt": question_prompt,
            "answer": question["answer"],
            "llm response": ans
        }

        print(ans)
        print(question["answer"])
        
        ans = extract_characters_regex(ans)
        print(ans)

        if ans != question["answer"]:
            wrong_cnt += 1
        cnt += 1

        print(f"current error rate: {wrong_cnt / cnt}")

        records[id]["parsed llm answer"] = ans

        # process_answer(choices[ord(question["answer"]) - ord("A")], ans, question)
        matched = process_answer(question["answer"], ans, question)

        records[id]["matched"] = matched

        if _checkpoint_records_supported(args):
            _atomic_write_json(record_ckpt_path, records)

    print(result)
    return result, records