"""Agent-track model facade.

The frozen baseline module still owns backend initialization, answer parsing, metric
updates, and artifact flushing. This facade replaces only the inference loop so
agent runs actually pass through ``orchestrator.augment_prompt_for_inference``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

_AGENT_DIR = Path(__file__).resolve().parents[1]
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

try:
    import tqdm
except ModuleNotFoundError:
    tqdm = types.SimpleNamespace(tqdm=lambda iterable, total=None: iterable)
    sys.modules.setdefault("tqdm", tqdm)

from orchestrator import AgentPrepResult, augment_prompt_for_inference
from orchestrator.trace_sink import append_trace_record, prompt_fingerprint

# ``.../agent/model/__init__.py`` -> parents[0]=model dir, [1]=agent, [2]=repo root.
_BASELINE_MODEL_DIR = Path(__file__).resolve().parents[2] / "baseline" / "model"
_IMPL = "_avsb_baseline_model_impl"


def _load_baseline_model():
    spec = importlib.util.spec_from_file_location(
        _IMPL,
        _BASELINE_MODEL_DIR / "__init__.py",
        submodule_search_locations=[str(_BASELINE_MODEL_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline model from {_BASELINE_MODEL_DIR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_IMPL] = mod
    spec.loader.exec_module(mod)
    return mod


_bm = _load_baseline_model()


def _eval_track() -> str:
    return os.getenv("AV_SPEAKERBENCH_EVAL_TRACK", "agent").strip() or "agent"


def _track_path(path: str) -> str:
    if _eval_track() in ("", "baseline", "none", "off"):
        return path
    dirname, basename = os.path.split(path)
    prefix = f"{_eval_track()}_"
    if basename.startswith(prefix):
        return path
    return os.path.join(dirname, prefix + basename)


def experiment_result_path(args: Any) -> str:
    return _track_path(_bm.experiment_result_path(args))


def experiment_record_path(args: Any) -> str:
    return _track_path(_bm.experiment_record_path(args))


def experiment_run_manifest_path(args: Any) -> str:
    rec = experiment_record_path(args)
    stem, ext = os.path.splitext(rec)
    return f"{stem}_experiment{ext or '.json'}"


def flush_experiment_artifacts(
    args: Any,
    result: dict,
    records: dict,
    *,
    elapsed_sec: float | None = None,
    num_dataset_items: int | None = None,
) -> tuple[str, str, str]:
    res_path = experiment_result_path(args)
    rec_path = experiment_record_path(args)
    run_path = experiment_run_manifest_path(args)
    _bm._atomic_write_json(res_path, result)
    _bm._atomic_write_json(rec_path, records)
    manifest = _bm._experiment_manifest_dict(
        args,
        result,
        records,
        res_path=res_path,
        rec_path=rec_path,
        run_path=run_path,
        elapsed_sec=elapsed_sec,
        num_dataset_items=num_dataset_items,
    )
    _bm._atomic_write_json(run_path, manifest)
    return res_path, rec_path, run_path


def _skills_disabled() -> bool:
    return os.getenv("AV_SPEAKERBENCH_SKILLS", "").strip().lower() in ("0", "off", "false", "no")


def _skill_inject_enabled() -> bool:
    return os.getenv("AV_SPEAKERBENCH_SKILL_INJECT", "").strip().lower() in ("1", "true", "yes")


def _trace_path(args: Any) -> str:
    track = os.getenv("AV_SPEAKERBENCH_EVAL_TRACK", "agent").strip() or "agent"
    safe_model = str(args.model_name).replace("/", "_").replace("\\", "_")
    task = "None" if getattr(args, "task_id", None) is None else str(args.task_id)
    return os.path.join(
        "record",
        f"{track}_trace_{safe_model}_{task}_audio_{args.audio}_visual_{args.visual}.jsonl",
    )


def _baseline_prompt(question: dict[str, Any]) -> str:
    choices = ast.literal_eval(question["choices"])
    choices_str = "\n".join(choices)
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        f"{question['question']}\n"
        f"{choices_str}\n"
        "The best answer is:"
    )


def _init_metric_bucket(question: dict[str, Any]) -> None:
    result = _bm.result
    result[f'level 1: {question["category"]}'] = result.get(f'level 1: {question["category"]}', {})
    result[f'level 1: {question["category"]}']["matched"] = result[
        f'level 1: {question["category"]}'
    ].get("matched", 0)
    result[f'level 1: {question["category"]}']["total"] = (
        result[f'level 1: {question["category"]}'].get("total", 0) + 1
    )

    result[f'level 2: {question["sub_category"]}'] = result.get(
        f'level 2: {question["sub_category"]}', {}
    )
    result[f'level 2: {question["sub_category"]}']["matched"] = result[
        f'level 2: {question["sub_category"]}'
    ].get("matched", 0)
    result[f'level 2: {question["sub_category"]}']["total"] = (
        result[f'level 2: {question["sub_category"]}'].get("total", 0) + 1
    )

    result[f'level 3: {question["task_id"]}'] = result.get(f'level 3: {question["task_id"]}', {})
    result[f'level 3: {question["task_id"]}']["matched"] = result[
        f'level 3: {question["task_id"]}'
    ].get("matched", 0)
    result[f'level 3: {question["task_id"]}']["total"] = (
        result[f'level 3: {question["task_id"]}'].get("total", 0) + 1
    )


def _cached_record_matches(record: dict[str, Any], prompt: str, prompt_fp: str | None) -> bool:
    if record.get("agent_prompt_fp") is not None:
        return record.get("agent_prompt_fp") == prompt_fp
    return record.get("question_prompt") == prompt


def _agent_prep(
    *,
    question_prompt: str,
    question: dict[str, Any],
    args: Any,
    video_path: str,
    audio_path: str,
    combined_path: str,
) -> tuple[AgentPrepResult, bool]:
    if _skills_disabled():
        return AgentPrepResult(question_prompt), False
    return (
        augment_prompt_for_inference(
            question_prompt=question_prompt,
            question=question,
            args=args,
            video_path=video_path,
            audio_path=audio_path,
            combined_path=combined_path,
        ),
        True,
    )


def _call_backend(
    *,
    args: Any,
    state: dict[str, Any],
    idx: int,
    question_id: str,
    video_path: str,
    audio_path: str,
    combined_path: str,
    prompt: str,
) -> str:
    if "gemini" in args.model_name:
        if args.audio:
            return _bm.gemini_process(audio_path, prompt, args.model_name, idx)
        if args.visual:
            return _bm.gemini_process(video_path, prompt, args.model_name, idx)
        return _bm.gemini_process(combined_path, prompt, args.model_name, idx)
    if args.model_name == "video_llama_13b" or args.model_name == "video_llama_7b":
        return _bm.video_llama_process(state["chat"], combined_path, prompt)
    if args.model_name == "video_llama2_7b":
        return _bm.video_llama2_process(
            combined_path,
            prompt,
            state["model"],
            state["processor"],
            state["tokenizer"],
        )
    if args.model_name == "pandagpt_7b" or args.model_name == "pandagpt_13b":
        return _bm.pandagpt_process(state["model"], prompt, audio_path, video_path, 512, [])
    if args.model_name == "phi4":
        return _bm.phi4_process(
            audio_path,
            video_path,
            args.num_frames,
            prompt,
            model=state["model"],
            processor=state["processor"],
            generation_config=state["generation_config"],
        )
    if args.model_name == "onellm":
        return _bm.onellm_process(state["model"], audio_path, video_path, prompt)
    if args.model_name == "uio2-large" or args.model_name == "uio2-xl" or args.model_name == "uio2-xxl":
        return _bm.uio2_process(state["model"], state["processor"], combined_path, prompt)
    if args.model_name == "NExTGPT":
        return _bm.NExTGPT_process(state["model"], video_path, audio_path, prompt)[0]
    if args.model_name == "AnyGPT":
        return _bm.anygpt_process(state["model"], video_path, audio_path, prompt)
    if args.model_name == "vita1":
        return _bm.vita1_process(
            state["model"],
            state["audio_processor"],
            state["image_processor"],
            state["tokenizer"],
            video_path,
            audio_path,
            prompt,
        )
    if args.model_name == "vita1_5":
        return _bm.vita1_5_process(
            state["model"],
            state["audio_processor"],
            state["image_processor"],
            state["tokenizer"],
            video_path,
            audio_path,
            prompt,
        )
    if "xblip" in args.model_name:
        return _bm.xblip_process(state["model"], audio_path, video_path, prompt)
    if "Qwen2.5" in args.model_name:
        return _bm.qwen2_5Omni_process(state["model"], state["processor"], combined_path, prompt)
    if "stream-omni" in args.model_name:
        return _bm.streamomni_process(
            state["model"],
            state["tokenizer"],
            state["image_processor"],
            state["cosyvoice"],
            audio_path,
            video_path,
            prompt,
        )
    if "ola" in args.model_name:
        return _bm.ola_process(
            state["model"],
            state["tokenizer"],
            state["image_processor"],
            video_path,
            audio_path,
            prompt,
        )
    if _bm._is_qwen3_model(args.model_name):
        media_type = "video" if args.visual else ("audio" if args.audio else "both")
        media_path = video_path if args.visual else (audio_path if args.audio else combined_path)
        if state["model"] is None:
            try:
                return _bm.qwen3omni_api_process(
                    state["processor"],
                    media_type,
                    media_path,
                    prompt,
                    split_visual_path=video_path,
                    split_audio_path=audio_path,
                )
            except _bm.ClipPayloadTooLargeError as cle:
                print(f"[SKIP] clip exceeds DashScope data-uri/embed limit (~20MiB): {cle}")
                return ""
            except FileNotFoundError as fnf:
                print(f"[SKIP] missing dataset media for question_id={question_id}: {fnf}")
                return ""
            except RuntimeError as err:
                cause = getattr(err, "__cause__", None)
                if isinstance(cause, FileNotFoundError):
                    print(f"[SKIP] missing dataset media for question_id={question_id}: {cause}")
                    return ""
                raise
        try:
            return state["qwen3_local_process"](
                state["model"],
                state["processor"],
                media_type,
                media_path,
                prompt,
            )
        except FileNotFoundError as fnf:
            print(f"[SKIP] missing dataset media for question_id={question_id}: {fnf}")
            return ""
    return ""


def _init_backend_state(args: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if args.model_name == "video_llama_13b" or args.model_name == "video_llama_7b":
        state["chat"] = _bm.model_init(args)
    elif args.model_name == "video_llama2_7b":
        state["model"], state["processor"], state["tokenizer"] = _bm.video_llama2_model_init()
    elif "panda" in args.model_name:
        state["model"] = _bm.pandagpt_model_init(args)
    elif "phi" in args.model_name:
        state["model"], state["processor"], state["generation_config"] = _bm.phi4_model_init(args)
    elif "onellm" in args.model_name:
        state["model"] = _bm.onellm_model_init(args)
    elif "uio" in args.model_name:
        state["model"], state["processor"] = _bm.uio2_model_init(args)
    elif "NExTGPT" in args.model_name:
        state["model"] = _bm.NExTGPT_model_init(args)
    elif "AnyGPT" in args.model_name:
        state["model"] = _bm.anygpt_model_init(args)
    elif args.model_name == "vita1":
        (
            state["model"],
            state["audio_processor"],
            state["image_processor"],
            state["tokenizer"],
        ) = _bm.vita1_model_init(args)
    elif args.model_name == "vita1_5":
        (
            state["model"],
            state["audio_processor"],
            state["image_processor"],
            state["tokenizer"],
        ) = _bm.vita1_5_model_init(args)
    elif "xblip" in args.model_name:
        state["model"] = _bm.xblip_model_init(args)
    elif "Qwen2.5" in args.model_name:
        state["model"], state["processor"] = _bm.qwen2_5Omni_model_init(args)
    elif "stream-omni" in args.model_name:
        (
            state["model"],
            state["tokenizer"],
            state["image_processor"],
            state["cosyvoice"],
        ) = _bm.streamomni_model_init(args)
    elif "ola" in args.model_name:
        state["model"], state["tokenizer"], state["image_processor"] = _bm.ola_model_init(args)
    elif _bm._is_qwen3_model(args.model_name):
        if getattr(args, "dashscope_model", None):
            state["model"], state["processor"] = None, args.dashscope_model
        else:
            from .open_model.Qwen3Omni.inference import qwen3omni_model_init, qwen3omni_process

            state["model"], state["processor"] = qwen3omni_model_init(args)
            state["qwen3_local_process"] = qwen3omni_process
    return state


def inference(args: Any, dataset: Any):
    """Agent-track inference with orchestrator prompt preparation and trace records."""
    _bm.result.clear()
    _bm.records.clear()

    temporary_dir = os.path.join(os.getcwd(), args.temp_dir)
    try:
        shutil.rmtree(temporary_dir)
    except Exception:
        pass
    os.makedirs(temporary_dir, exist_ok=True)

    state = _init_backend_state(args)
    cnt = 0
    wrong_cnt = 0

    record_ckpt_path = experiment_record_path(args)
    if _bm._checkpoint_records_supported(args) and _bm._resume_requested(args) and os.path.isfile(record_ckpt_path):
        with open(record_ckpt_path, encoding="utf-8") as f:
            _bm.records.update(json.load(f))

    infer_limit_raw = os.environ.get("AV_SPEAKERBENCH_INFERENCE_LIMIT")
    infer_limit = int(infer_limit_raw) if infer_limit_raw else None
    trace_path = _trace_path(args)

    for idx, question in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        if infer_limit is not None and idx >= infer_limit:
            break

        new_video_path = os.path.join(args.data_path, question["visual_path"])
        new_audio_path = os.path.join(args.data_path, question["audio_path"])
        new_combined_path = os.path.join(args.data_path, question["audio_visual_path"])
        question_prompt = _baseline_prompt(question)
        _init_metric_bucket(question)

        prep, orchestrator_ran = _agent_prep(
            question_prompt=question_prompt,
            question=question,
            args=args,
            video_path=new_video_path,
            audio_path=new_audio_path,
            combined_path=new_combined_path,
        )
        model_prompt = prep.final_prompt
        original_fp = prompt_fingerprint(question_prompt)
        agent_fp = prompt_fingerprint(model_prompt)
        prompt_changed = model_prompt != question_prompt
        evidence_injected = "Structured_skill_evidence" in model_prompt and prompt_changed
        question_id = question["question_id"]

        ans = ""
        reused_cache = False
        rec = _bm.records.get(question_id)
        if (
            rec
            and _bm._checkpoint_records_supported(args)
            and _cached_record_matches(rec, model_prompt, agent_fp)
        ):
            print(f"{question_id} exists")
            ans = rec["llm response"]
            reused_cache = True
        else:
            t0 = time.perf_counter()
            ans = _call_backend(
                args=args,
                state=state,
                idx=idx,
                question_id=question_id,
                video_path=new_video_path,
                audio_path=new_audio_path,
                combined_path=new_combined_path,
                prompt=model_prompt,
            )
            infer_wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if reused_cache:
            infer_wall_ms = 0.0

        _bm.records[question_id] = {
            "video_id": question["video_id"],
            "task_id": question["task_id"],
            "question_prompt": question_prompt,
            "agent_model_prompt": model_prompt,
            "agent_original_prompt_fp": original_fp,
            "agent_prompt_fp": agent_fp,
            "agent_prompt_changed": prompt_changed,
            "agent_evidence_injected": evidence_injected,
            "agent_orchestrator_ran": orchestrator_ran,
            "agent_skills_disabled": _skills_disabled(),
            "agent_skill_inject_requested": _skill_inject_enabled(),
            "skills_invoked": prep.skills_invoked,
            "bottleneck_tags": prep.bottleneck_tags,
            "orchestrator_errors": prep.errors,
            "answer": question["answer"],
            "llm response": ans,
        }

        print(ans)
        print(question["answer"])

        parsed = _bm.extract_characters_regex(ans)
        print(parsed)

        if parsed != question["answer"]:
            wrong_cnt += 1
        cnt += 1
        print(f"current error rate: {wrong_cnt / cnt}")

        _bm.records[question_id]["parsed llm answer"] = parsed
        matched = _bm.process_answer(question["answer"], parsed, question)
        _bm.records[question_id]["matched"] = matched

        append_trace_record(
            trace_path,
            {
                "question_id": question_id,
                "video_id": question.get("video_id"),
                "task_id": question.get("task_id"),
                "category": question.get("category"),
                "sub_category": question.get("sub_category"),
                "orchestrator_ran": orchestrator_ran,
                "skills_disabled": _skills_disabled(),
                "skill_inject_requested": _skill_inject_enabled(),
                "agent_evidence_injected": evidence_injected,
                "original_prompt_fp": original_fp,
                "agent_prompt_fp": agent_fp,
                "agent_prompt_changed": prompt_changed,
                "skills_invoked": prep.skills_invoked,
                "bottleneck_tags": prep.bottleneck_tags,
                "errors": prep.errors,
                "infer_wall_ms": infer_wall_ms,
                "reused_cache": reused_cache,
                "llm_response_empty": not str(ans or "").strip(),
                "parsed_answer": parsed,
                "matched": matched,
            },
        )

        if _bm._checkpoint_records_supported(args):
            _bm._atomic_write_json(record_ckpt_path, _bm.records)

    print(_bm.result)
    return _bm.result, _bm.records
