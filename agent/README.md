# Multimodal Agent track (`agent/`)

This tree is a **working copy** of the AV-SpeakerBench harness plus Agent-oriented extensions: **artifact isolation**, **per-question JSONL traces**, and a **stub orchestrator** (`orchestrator/`) for Skills/Tools.

**Dataset** stays at the repo root **`Holistic_AVQA_bench/`** (or **`AV_SPEAKERBENCH_DATA_ROOT`**). Paths resolve via [`dataset/paths.py`](dataset/paths.py).

Smoke (1 question, Skill inject + Tools, loads repo-root ``.env``):

```bash
python agent/scripts/smoke_agent_one.py
```

## Environment

Preferred flow: start from the baseline env, then add only the agent-side extras.

```bash
mamba env create -f baseline/environment-mamba.yml
mamba activate av-speakerbench
pip install -r agent/requirements-agent.txt
```

If you prefer a separate env for the agent track:

```bash
mamba env create -f agent/environment-mamba.yml
mamba activate av-speakerbench-agent
```

What the agent layer adds on top of baseline:

- `silero-vad`, `faster-whisper`, `whisperx`, `pyannote.audio`
- `ultralytics`, `opencv-python`
- `pandas`, `numexpr`, `bottleneck` to suppress current warning noise in analysis utilities

Notes:

- These backends are still **optional at runtime**; the tool layer falls back to lighter paths or stub outputs when they are missing.
- `whisperx` and `pyannote.audio` are the heaviest installs. Use the incremental `requirements-agent.txt` path if you do not want a second conda env.

## How to run

From repository root:

```bash
python main_agent.py --model_name Qwen3-Omni-3B --use_local_metadata
```

From this directory:

```bash
python main.py --model_name Qwen3-Omni-3B --use_local_metadata
```

`agent/main.py` sets **`AV_SPEAKERBENCH_EVAL_TRACK=agent` by default** (filename prefix **`agent_`** on `result/` / `record/`). Clear it only if you intentionally want legacy names:

```bash
set AV_SPEAKERBENCH_EVAL_TRACK=
python main.py ...
```

## Outputs

| Output | Purpose |
|--------|---------|
| `result/agent_<model>.json` (typical) | Aggregate metrics JSON |
| `record/agent_<model>_record_*.json` | Per-question records + resume checkpoints (Gemini / Qwen3 API) |
| `record/agent_*_experiment.json` | Run manifest (`args`, paths, timings) |
| `record/agent_trace_*.jsonl` | One JSON object per question (skills, bottleneck tags, errors, `infer_wall_ms`) |

### Trace JSONL schema (one line per question)

- `eval_track`, `question_id`, `video_id`, `task_id`, `category`, `sub_category`
- `skills_invoked` — ordered status tags per registered Skill (e.g. `clip_span_meta:injected`, `asr_word_lane:word_asr_injected`) unless `AV_SPEAKERBENCH_SKILLS=off`
- `bottleneck_tags` — design-time hooks (`perception_pending`, `alignment_pending`, **`evidence_synthetic`** for stub ASR, `stub_backend`, etc.)
- `errors` — list of `{kind, detail}` dicts
- `infer_wall_ms`, `prompt_fp`, `matched`, `parsed_answer`, `llm_response_empty`

### Skills gate / env (operational)

- **`AV_SPEAKERBENCH_SKILLS=off`** — orchestrator returns LM-only prep (pipeline skipped).
- **`AV_SPEAKERBENCH_SKILL_INJECT=1`** (or `true`/`yes`) — append Skill blocks to the MC prompt; **omit or unset for LM-only parity** on the MC stem (skills may still run for trace unless `SKILLS=off`).
- **`AV_SPEAKERBENCH_SKILLS_ALLOWLIST`** — comma-separated skill ids, or `all`/empty for full registry **subject to triggers** (see `skills/triggers.py` and the **Task → Skill → Tool matrix** below). Registered ids: `clip_span_meta`, `asr_word_lane`, `anchor_quote_time`, `anchor_window_asr`, `lexical_asr_bridge`, `diar_binding`, `turn_order_sheet`, `viz_people_anchor`.
- **`AV_SPEAKERBENCH_SKILLS_TRIGGER_MODE`** — `auto` (default) or **`all`** (force every Skill payload for audits).
- **`AV_SPEAKERBENCH_ALLOW_SYNTHETIC_ASR`** — default **off**; when off, **`AV_SPEAKERBENCH_STUB_ASR_TEXT`** is never injected and trace may tag missing synthetic rows. Smoke script defaults allow=1 for stub demos only.
- **Prompt fusion:** **`AV_SPEAKERBENCH_EVIDENCE_PLACEMENT=tail`** (default) vs **`both`** — `both` prepends `AV_SPEAKERBENCH_EVIDENCE_PREFIX_TEXT` or the default grounding sentence before the MC stem (structured evidence remains after the stem).
- **Conservative defaults:** the active prompt path is intentionally small and keeps only clip span, localized ASR, quote alignment, core diarization / turn order, and visual people snapshots.
- **Diarization:** **`AV_SPEAKERBENCH_DIAR_BACKEND=auto`** (default) vs **`pyannote`** (local `pyannote.audio` + `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`) vs **`pyannote_api`** (cloud only) vs `stub`. Local default model is **`pyannote/speaker-diarization-community-1`** unless overridden by `AV_SPEAKERBENCH_PYANNOTE_MODEL`. If `DIAR_BACKEND=pyannote` and **`PYANNOTE_API_KEY`** is set, the [pyannoteAI API](https://docs.pyannote.ai/quickstart) is used ([upload + diarize](https://docs.pyannote.ai/tutorials/how-to-upload-files)); optional `PYANNOTE_API_BASE`, `AV_SPEAKERBENCH_PYANNOTE_API_POLL_S`, `AV_SPEAKERBENCH_PYANNOTE_API_MAX_WAIT_S`.

## Tool / Skill categories

Default routing now follows benchmark `category`:

- **Audio-centric**: keep audio-related Tools / Skills only.
- **Visual-centric**: keep visual-related Tools / Skills only, plus quote-alignment only when the question stem explicitly needs audio anchoring.
- **Speaker-centric / composite A-V**: keep both audio and visual Tool / Skill families because these tasks often need `who + when + where` jointly.

### Audio-centric

**Tools**

- `audio_vad` — speech/non-speech segmentation; provides anchor windows for localized ASR.
- `audio_asr` — anchor-window ASR over VAD spans; extracts what was said in the relevant interval.
- `audio_asr_words` — word-timestamp ASR; supports quote lookup and token-level alignment.
- `benchmark_timecode` — parses benchmark `start_time` / `end_time` into seconds.

**Skills**

- `clip_span_meta` — injects the official benchmark clip span.
- `asr_word_lane` — injects word-timestamp transcript snippets for recognition/counting/alignment.
- `anchor_window_asr` — injects VAD-bounded transcript excerpts near likely answer regions.
- `lexical_asr_bridge` — checks whether quoted phrases in the question appear in ASR evidence.
- `anchor_quote_time` — resolves a quoted phrase to its first timestamp when needed.

### Visual-centric

**Tools**

- `video_people_snap` — frame extraction plus optional person detection/tracking around anchor time.
- `benchmark_timecode` — clip span parsing for event windows.

**Skills**

- `clip_span_meta` — injects the benchmark clip span for visual event scoping.
- `viz_people_anchor` — samples frames around the resolved anchor time and summarizes visible people counts.
- `anchor_quote_time` — only retained when a visual task is explicitly anchored by a spoken quote.

### Speaker-centric / composite A-V

**Tools**

- `audio_vad` — coarse speech windows for transcript anchoring.
- `audio_asr` — anchor-window transcript extraction.
- `audio_asr_words` — quote timing and token-level evidence.
- `audio_diar` — speaker diarization; only high-confidence `pyannote` outputs are injected by default.
- `speech_turn_sheet` — converts diarization + word timestamps into structured turn order.
- `video_people_snap` — visual people snapshots for speaker/identity disambiguation when relevant.
- `benchmark_timecode` — clip span parsing for temporal anchoring.

**Skills**

- `clip_span_meta` — injects official clip span.
- `asr_word_lane` — injects word-timestamp transcript evidence.
- `anchor_window_asr` — injects VAD-bounded transcript windows.
- `lexical_asr_bridge` — bridges quoted phrases to ASR spans.
- `diar_binding` — injects speaker spans only when diarization backend is high-confidence.
- `turn_order_sheet` — injects structured speaker-turn order for core speaker recognition.
- `viz_people_anchor` — retained for visual counting or visible-people cues inside composite tasks.

### VAD (`tools/audio_vad.py`)

When inject is on and a per-clip `.wav` exists, VAD backs [`anchor_window_asr`](skills/impl.py) and [`lexical_asr_bridge`](skills/impl.py). **`AV_SPEAKERBENCH_VAD_BACKEND=auto`** prefers Silero VAD when installed and falls back to energy VAD; set `energy` or `silero` explicitly if needed. Tunables include `AV_SPEAKERBENCH_VAD_FRAME_MS`, `AV_SPEAKERBENCH_VAD_HOP_MS`, `AV_SPEAKERBENCH_VAD_MARGIN_DB`, `AV_SPEAKERBENCH_VAD_MIN_SEGMENT_S`, `AV_SPEAKERBENCH_VAD_MERGE_GAP_S`, `AV_SPEAKERBENCH_VAD_NOISE_PERCENTILE`, `AV_SPEAKERBENCH_VAD_MAX_DURATION_S`, `AV_SPEAKERBENCH_VAD_MAX_SEGMENTS_IN_PROMPT`, `AV_SPEAKERBENCH_VAD_SILERO_THRESHOLD`, `AV_SPEAKERBENCH_VAD_SILERO_SPEECH_PAD_MS`.

### ASR (`tools/audio_asr.py`)

Anchor-window transcription. Set `AV_SPEAKERBENCH_ASR_BACKEND` = `auto` (default), `faster_whisper`, `whisper`, or `stub`. Word-timestamp ASR uses `AV_SPEAKERBENCH_WORD_ASR_BACKEND`; `auto` prefers **WhisperX** when installed and falls back to `faster_whisper`. Also: `AV_SPEAKERBENCH_STUB_ASR_TEXT` (**requires** `ALLOW_SYNTHETIC_ASR` for stub filler), `AV_SPEAKERBENCH_WHISPER_MODEL` (size name **or** existing local directory), **`AV_SPEAKERBENCH_FASTER_WHISPER_MODEL_DIR`** (recommended: CTranslate2 folder from [ModelScope `Systran/faster-whisper-small`](https://www.modelscope.cn/models/Systran/faster-whisper-small) via `modelscope snapshot-download …`), `AV_SPEAKERBENCH_ASR_DEVICE`, `AV_SPEAKERBENCH_ASR_COMPUTE_TYPE`, `AV_SPEAKERBENCH_ASR_LANGUAGE`, `AV_SPEAKERBENCH_ASR_MAX_WINDOWS`, `AV_SPEAKERBENCH_ASR_MAX_CHUNK_S`, `AV_SPEAKERBENCH_ASR_WINDOW_MULT_*` / `AV_SPEAKERBENCH_ASR_PRIORITY_TASK_IDS`, `AV_SPEAKERBENCH_ASR_MAX_LINES_PROMPT`, `AV_SPEAKERBENCH_OW_MAX_SLICE_S`, `AV_SPEAKERBENCH_WORD_ASR_BATCH_SIZE`.

### Clip timecodes (`tools/benchmark_timecode.py`)

Parses Holistic CSV `MM:SS` (and `H:MM:SS`) fields into seconds; used by **`clip_span_meta`** for `span_s` cues.

### Task → Skill → Tool matrix (for operators & agent design)

Holistic metadata uses **`task_id`** (level-3, same column as in `test.csv`) and **`category`** / **`sub_category`**. Unless **`AV_SPEAKERBENCH_SKILLS_TRIGGER_MODE=all`**, only Skills whose **`should_emit_*`** passes in [`skills/triggers.py`](skills/triggers.py) run and may inject evidence (when **`AV_SPEAKERBENCH_SKILL_INJECT=1`**).

**Legend**

- **●** — intended for this task under default **auto** triggers (may still `skip_*` if inject off, missing media, or heuristics miss).
- **○** — conditional (e.g. quoted phrase in stem, keywords like before/after, or sub_category).
- **—** — not targeted; may appear only under **`TRIGGER_MODE=all`** or allowlist experiments.

#### By `task_id`

| `task_id` | Intent | Skills (registry id) | Underlying `tools/` (plus shared) |
|-----------|--------|-------------------------|-------------------------------------|
| **Speech Recognition** | Verbatim **what** was said | ● `asr_word_lane`, `anchor_window_asr`; ○ `lexical_asr_bridge` (quotes) | `audio_asr_words`, `audio_asr`, `audio_vad` |
| **Speech Counting** | Count mentions / bounded intervals | ● `asr_word_lane`, `anchor_window_asr`; ○ `lexical_asr_bridge`, `anchor_quote_time` | `audio_asr_words`, `audio_asr`, `audio_vad` |
| **Speaker Recognition** | **Who** speaks before/after / order | ● `diar_binding`, `turn_order_sheet`, `asr_word_lane`; ○ `anchor_quote_time`, `lexical_asr_bridge` | `audio_diar`, `speech_turn_sheet`, `audio_asr_words`, `audio_asr`, `audio_vad` |
| **Visual Counting** | People **visible** at an event/time | ● `viz_people_anchor`, `anchor_quote_time` | `video_people_snap`, `audio_asr_words`, `benchmark_timecode` |
| **Other rows** | Weak-support / fallback | ● `clip_span_meta`; ○ `anchor_window_asr` | `benchmark_timecode`, `audio_asr` |

**Always-on / broad Skill**: **`clip_span_meta`** when timing/counting/activity-style heuristics hit.

**Shared stack** almost everywhere a `.wav` exists: **`audio_vad`** feeds anchor ASR windows and quote-localized evidence; **`audio_diar`** is injected only when backend confidence is high enough.

#### By `category` (coarse)

| `category` | Emphasis |
|------------|----------|
| **Audio-centric** | ASR family (`audio_asr`, `audio_asr_words`) plus VAD and clip span. |
| **Speaker-centric** | Diarization (`audio_diar`), turn order, word lane, and quote alignment. |
| **Visual-centric** | `video_people_snap` plus quote→time alignment when the visual question is audio-anchored. |

Implementation details and env knobs for per-task budgets (word cap, viz Δt) live in **`effective_*`** helpers inside [`skills/triggers.py`](skills/triggers.py).

## Design docs

- [`docs/MM_AGENT_DESIGN.md`](docs/MM_AGENT_DESIGN.md) — tasks, bottlenecks, Skill/Tool principles, resource table.
- [`docs/AV_TOOL_SKILL_SURVEY.md`](docs/AV_TOOL_SKILL_SURVEY.md) — external AV tools / candidate Skills survey, prioritized for this repo's `agent/tools` and `agent/skills`.
- [`docs/SKILL_INJECT_ABLATION.md`](docs/SKILL_INJECT_ABLATION.md) — **protocol**: inject off vs on, same split + bucket diff.
- [`docs/HARD_SUBSETS.md`](docs/HARD_SUBSETS.md) — reproducible subset recipes using existing CLI filters.
- **Tooling scripts:** [`scripts/compare_result_buckets.py`](scripts/compare_result_buckets.py) (baseline vs treatment JSON), [`scripts/rank_buckets_from_result.py`](scripts/rank_buckets_from_result.py).

## Code layout

| Path | Role |
|------|------|
| [`main.py`](main.py) | CLI entry; sets default eval track |
| [`model/`](model/) | Inference + prefixed `flush_experiment_artifacts` + trace append |
| [`orchestrator/`](orchestrator/) | `augment_prompt_for_inference` → [`skills/`](skills/) pipeline, `trace_sink` |
| [`skills/`](skills/) | `impl.py` registry, `triggers.py` task cohorts |
| [`tools/`](tools/) | Optional perception backends (`audio_vad`, …) used by Skills |
| [`dataset/`](dataset/) | Loaders (shared semantics with `baseline/`) |

## Maintenance strategy (vs `baseline/`)

**Chosen approach for this repo:** **fork-style parallel trees**

- **`baseline/`** — frozen reference harness; match paper / leaderboard reproductions here.
- **`agent/`** — primary development for multimodal Agent, Skills, and traces.
- **Long-term options** (if drift hurts):
  1. Extract a small **shared** Python package (loaders + metric aggregation only), or
  2. Use **git branches** (`baseline-frozen` vs `agent-dev`) with tags for submission snapshots.

We do **not** auto-sync `agent/` ← `baseline/`; port fixes manually or via shared package when you adopt option (1).

## See also

- Repo root [`README.md`](../README.md) for the original benchmark narrative.
- Baseline [`../baseline/README.md`](../baseline/README.md).
