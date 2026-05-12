# Multimodal Agent track (`agent/`)

This tree is a **working copy** of the AV-SpeakerBench harness plus Agent-oriented extensions: **artifact isolation**, **per-question JSONL traces**, and a **stub orchestrator** (`orchestrator/`) for Skills/Tools.

**Dataset** stays at the repo root **`Holistic_AVQA_bench/`** (or **`AV_SPEAKERBENCH_DATA_ROOT`**). Paths resolve via [`dataset/paths.py`](dataset/paths.py).

Smoke (1 question, Skill inject + Tools, loads repo-root ``.env``):

```bash
python agent/scripts/smoke_agent_one.py
```

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
- `skills_invoked` — ordered status tags per registered Skill (e.g. `meta_banner:skipped_inject_disabled`, `anchor_phrase_hints:injected`) unless `AV_SPEAKERBENCH_SKILLS=off`
- `bottleneck_tags` — design-time hooks (`perception_pending`, `alignment_pending`, **`evidence_synthetic`** for stub ASR, `stub_backend`, etc.)
- `errors` — list of `{kind, detail}` dicts
- `infer_wall_ms`, `prompt_fp`, `matched`, `parsed_answer`, `llm_response_empty`

### Skills gate / env (operational)

- **`AV_SPEAKERBENCH_SKILLS=off`** — orchestrator returns LM-only prep (pipeline skipped).
- **`AV_SPEAKERBENCH_SKILL_INJECT=1`** (or `true`/`yes`) — append Skill blocks to the MC prompt; **omit or unset for LM-only parity** on the MC stem (skills may still run for trace unless `SKILLS=off`).
- **`AV_SPEAKERBENCH_SKILLS_ALLOWLIST`** — comma-separated skill ids, or `all`/empty for full registry **subject to triggers** (see `skills/triggers.py` and the **Task → Skill → Tool matrix** below). Registered ids: `meta_banner`, `clip_span_meta`, `anchor_phrase_hints`, `media_clip_facts`, `speaker_turn_proxy`, `asr_word_lane`, `anchor_quote_time`, `anchor_window_asr`, `lexical_asr_bridge`, `diar_binding`, `turn_order_sheet`, `speak_duration_sheet`, `f0_rank_shortlist`, `rate_words_per_sec`, `overlap_split`, `prosody_discrete`, `moment_refine`, `viz_people_anchor`, `visual_clip_meta`, `visual_anchor_ground`.
- **`AV_SPEAKERBENCH_SKILLS_TRIGGER_MODE`** — `auto` (default) or **`all`** (force every Skill payload for audits).
- **`AV_SPEAKERBENCH_ALLOW_SYNTHETIC_ASR`** — default **off**; when off, **`AV_SPEAKERBENCH_STUB_ASR_TEXT`** is never injected and trace may tag missing synthetic rows. Smoke script defaults allow=1 for stub demos only.
- **Prompt fusion:** **`AV_SPEAKERBENCH_EVIDENCE_PLACEMENT=tail`** (default) vs **`both`** — `both` prepends `AV_SPEAKERBENCH_EVIDENCE_PREFIX_TEXT` or the default grounding sentence before the MC stem (structured evidence remains after the stem).
- **Diarization:** **`AV_SPEAKERBENCH_DIAR_BACKEND=stub`** (default) vs **`pyannote`** (local `pyannote.audio` + `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`) vs **`pyannote_api`** (cloud only). If `DIAR_BACKEND=pyannote` and **`PYANNOTE_API_KEY`** is set, the [pyannoteAI API](https://docs.pyannote.ai/quickstart) is used ([upload + diarize](https://docs.pyannote.ai/tutorials/how-to-upload-files)); optional `PYANNOTE_API_BASE`, `AV_SPEAKERBENCH_PYANNOTE_API_POLL_S`, `AV_SPEAKERBENCH_PYANNOTE_API_MAX_WAIT_S`.

### VAD (`tools/audio_vad.py`)

When inject is on and a per-clip `.wav` exists, energy-VAD backs [`anchor_window_asr`](skills/impl.py), [`lexical_asr_bridge`](skills/impl.py), [`media_clip_facts`](skills/impl.py) (duration fallback), [`speaker_turn_proxy`](skills/impl.py), [`moment_refine`](skills/impl.py), [`overlap_split`](skills/impl.py), and [`prosody_discrete`](skills/impl.py). Tunables include `AV_SPEAKERBENCH_VAD_FRAME_MS`, `AV_SPEAKERBENCH_VAD_HOP_MS`, `AV_SPEAKERBENCH_VAD_MARGIN_DB`, `AV_SPEAKERBENCH_VAD_MIN_SEGMENT_S`, `AV_SPEAKERBENCH_VAD_MERGE_GAP_S`, `AV_SPEAKERBENCH_VAD_NOISE_PERCENTILE`, `AV_SPEAKERBENCH_VAD_MAX_DURATION_S`, `AV_SPEAKERBENCH_VAD_MAX_SEGMENTS_IN_PROMPT`.

### ASR (`tools/audio_asr.py`)

Anchor-window transcription. Set `AV_SPEAKERBENCH_ASR_BACKEND` = `auto` (default), `faster_whisper`, `whisper`, or `stub`. Also: `AV_SPEAKERBENCH_STUB_ASR_TEXT` (**requires** `ALLOW_SYNTHETIC_ASR` for stub filler), `AV_SPEAKERBENCH_WHISPER_MODEL` (size name **or** existing local directory), **`AV_SPEAKERBENCH_FASTER_WHISPER_MODEL_DIR`** (recommended: CTranslate2 folder from [ModelScope `Systran/faster-whisper-small`](https://www.modelscope.cn/models/Systran/faster-whisper-small) via `modelscope snapshot-download …`), `AV_SPEAKERBENCH_ASR_DEVICE`, `AV_SPEAKERBENCH_ASR_COMPUTE_TYPE`, `AV_SPEAKERBENCH_ASR_LANGUAGE`, `AV_SPEAKERBENCH_ASR_MAX_WINDOWS`, `AV_SPEAKERBENCH_ASR_MAX_CHUNK_S`, `AV_SPEAKERBENCH_ASR_WINDOW_MULT_*` / `AV_SPEAKERBENCH_ASR_PRIORITY_TASK_IDS`, `AV_SPEAKERBENCH_ASR_MAX_LINES_PROMPT`, `AV_SPEAKERBENCH_OW_MAX_SLICE_S`.

### Clip timecodes (`tools/benchmark_timecode.py`)

Parses Holistic CSV `MM:SS` (and `H:MM:SS`) fields into seconds; used by **`clip_span_meta`** for `span_s` cues.

### Container probe (`tools/media_probe.py`)

Optional **ffprobe** JSON on local clip MP4s. Env: **`AV_SPEAKERBENCH_MEDIA_PROBE`** = `auto` (run if `ffprobe` is on `PATH`), `on` (**always** attempt), or `off` / `none`. **`AV_SPEAKERBENCH_FFPROBE_BIN`** (default `ffprobe`), **`AV_SPEAKERBENCH_FFPROBE_TIMEOUT_S`** (default `45`). Wired into **`visual_clip_meta`** (both visual + audiovisual paths) and conditionally **`media_clip_facts`** (combined path only when the visual cohort would not duplicate it — see `skills/triggers.py` `should_emit_media_container_probe`).

### RMS meter (`tools/audio_rms_meter.py`)

Whole-clip **RMS / peak / crest-factor (dB)** on decoded mono WAV with optional **VAD speech-union** mask (Speech Intensity cohort via `skills/triggers.py`). Optional cap: **`AV_SPEAKERBENCH_RMS_METER_MAX_SAMPLES`** (head truncate; default uncapped for short benchmark clips).

### Task → Skill → Tool matrix (for operators & agent design)

Holistic metadata uses **`task_id`** (level-3, same column as in `test.csv`) and **`category`** / **`sub_category`**. Unless **`AV_SPEAKERBENCH_SKILLS_TRIGGER_MODE=all`**, only Skills whose **`should_emit_*`** passes in [`skills/triggers.py`](skills/triggers.py) run and may inject evidence (when **`AV_SPEAKERBENCH_SKILL_INJECT=1`**).

**Legend**

- **●** — intended for this task under default **auto** triggers (may still `skip_*` if inject off, missing media, or heuristics miss).
- **○** — conditional (e.g. quoted phrase in stem, keywords like before/after, or sub_category).
- **—** — not targeted; may appear only under **`TRIGGER_MODE=all`** or allowlist experiments.

#### By `task_id`

| `task_id` | Intent | Skills (registry id) | Underlying `tools/` (plus shared) |
|-----------|--------|-------------------------|-------------------------------------|
| **Speech Recognition** | Verbatim **what** was said | ● `asr_word_lane`, `anchor_window_asr`; ○ `lexical_asr_bridge` (quotes), `anchor_phrase_hints` | `audio_asr_words` (word lane), `audio_asr` (anchor / word-lane merge), `audio_vad` |
| **Speech Counting** | Count mentions / bounded intervals | ● `asr_word_lane`, `anchor_window_asr`, `overlap_split`, `speaker_turn_proxy`; ○ `lexical_asr_bridge`, `turn_order_sheet` (order language), `anchor_quote_time` (quote+counting) | `audio_asr_words`, `audio_asr`, `audio_vad`; `speech_turn_sheet` when order triggers |
| **Speech Duration** | **Who talks longest/least** (time) | ● `speak_duration_sheet`, `anchor_window_asr`, `diar_binding`, `moment_refine`; ○ `clip_span_meta`, `asr_word_lane` | `audio_speak_duration`, `audio_diar` (+ `diar_with_vad_fallback`), `audio_asr`, `audio_asr_words`, `audio_vad`, `benchmark_timecode` |
| **Speech Pitch** | Compare **F0** across speakers | ● `f0_rank_shortlist`, `anchor_window_asr`; ○ `prosody_discrete` (**off** by default for Pitch if `PROSODY_SKIP_FOR_PITCH=1`) | `audio_pitch_segments` (librosa pyin), `audio_asr`, `audio_diar`/`vad` for segments; `audio_prosody` optional |
| **Speech Rate** | Words/sec style comparison | ● `rate_words_per_sec`, `anchor_window_asr`, `diar_binding`, `asr_word_lane` | `audio_rate_proxy`, `audio_asr_words`, `audio_diar`, `audio_asr`, `audio_vad` |
| **Speech Intensity** | Loudness / dynamics | ● `prosody_discrete`, `anchor_window_asr`, `media_clip_facts` (RMS line); ○ `asr_word_lane` | `audio_prosody`, `audio_rms_meter`, `audio_asr`, `audio_vad`, `audio_asr_words` |
| **Speaker Detection** | Yes/no or **whether** someone speaks | ● `diar_binding`, `overlap_split`, `asr_word_lane` | `audio_diar`, `audio_vad`, `audio_asr_words` |
| **Speaker Recognition** | **Who** speaks before/after / order | ● `diar_binding`, `turn_order_sheet`, `overlap_split`, `asr_word_lane`; ○ `anchor_quote_time` (stem quotes), `lexical_asr_bridge` | `audio_diar`, `speech_turn_sheet`, `audio_asr_words`, `audio_asr`, `audio_vad` |
| **Speaker Counting** | **How many** distinct speakers (± visible people) | ● `speaker_turn_proxy`, `overlap_split`, `diar_binding`, `asr_word_lane`; ○ `turn_order_sheet` | `audio_vad`, `audio_diar`, `audio_asr_words` |
| **Visual Counting** | People **visible** at an event/time | ● `viz_people_anchor`, `anchor_quote_time`; ○ `visual_clip_meta` | `video_people_snap` (ffmpeg ± YOLO), `audio_asr_words` (quote→time), `media_probe` |
| **Attribute Recognition** | Visual attributes | ● `visual_clip_meta`; ○ `anchor_window_asr` if stem needs audio | `media_probe`, `audio_asr` (if triggered as ASR-focus) |
| **Activity Recognition** | What happens / when | ● `moment_refine`, `clip_span_meta`, `visual_clip_meta`; ○ `anchor_window_asr` | `audio_vad`, `benchmark_timecode`, `media_probe`, `audio_asr` |

**Always-on / broad Skills** (many rows above also get these when triggers match): `meta_banner`; **`media_clip_facts`** on most audio/speaker/visual rows; **`clip_span_meta`** when timing/counting/activity-style heuristics hit; **`speaker_turn_proxy`** only **Speaker Counting** & **Speech Counting**. **`visual_anchor_ground`** is a **stub** (no real detector yet).

**Shared stack** almost everywhere a `.wav` exists: **`audio_vad`** (energy VAD) feeds anchor ASR windows, overlap, prosody, moment hints, and **VAD-as-diar** fallback when diar is empty (`audio_diar.diar_with_vad_fallback`).

#### By `category` (coarse)

| `category` | Emphasis |
|------------|----------|
| **Audio-centric** | ASR family (`audio_asr`, `audio_asr_words`), pitch/intensity/rate/duration tools, VAD. |
| **Speaker-centric** | Diarization (`audio_diar`), turn / overlap / duration / rate sheets, word lane. |
| **Visual-centric** | `media_probe`, `video_people_snap`, `visual_clip_meta`; quote→time uses `audio_asr_words`. |

Implementation details and env knobs for per-task budgets (word cap, F0 segment count, viz Δt) live in **`effective_*`** helpers inside [`skills/triggers.py`](skills/triggers.py).

## Design docs

- [`docs/MM_AGENT_DESIGN.md`](docs/MM_AGENT_DESIGN.md) — tasks, bottlenecks, Skill/Tool principles, resource table.
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
