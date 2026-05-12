# Reproducible subsets (no dataset move)

Hard / focused evaluation uses **the same** `Holistic_AVQA_bench/test.csv` (or `--data_path`). Below are **recipe-style** filters you can combine with **`baseline/main.py`** (frozen leaderboard-style harness) or **`agent/main.py`** (prefixed artifacts + traces).

## Baseline artifacts for Skill design (freeze your command)

Skill / Tool prioritisation must cite metrics from a **saved** aggregate JSON (see [`MM_AGENT_DESIGN.md — Evidence-from-baseline`](MM_AGENT_DESIGN.md#evidence-from-baseline-skill--tool-prioritisation)).

| Track | Working directory | Typical outputs (relative to cwd) | Notes |
|-------|-------------------|-----------------------------------|--------|
| **Baseline** | `baseline/` | `result/<model>.json`, `record/<model>_record_*` | No `agent_` filename prefix; use for paper-aligned reproduction. |
| **Agent** | `agent/` or `python main_agent.py` from repo root | `result/agent_<model>.json`, `record/agent_*`, `record/agent_trace_*.jsonl` | Default `AV_SPEAKERBENCH_EVAL_TRACK=agent`. |

### Full-benchmark baseline (example)

Run from **`baseline/`** after `cd`:

```bash
cd baseline
python main.py --model_name Qwen3-Omni-3B --use_local_metadata --resume
```

Produces (when finished) at least **`result/Qwen3-Omni-3B.json`** and **`record/Qwen3-Omni-3B_record_None_audio_False_visual_False.json`** (`task_id`/`category` filters change the middle segments of filenames).

### Stratified subset baseline (example)

Must record **`sample_fraction`**, **`sample_seed`**, and **`stratify_key`**; the aggregate JSON includes **`_subset_meta`** after the run.

```bash
cd baseline
python main.py --model_name Qwen3-Omni-3B --sample_fraction 0.1 --sample_seed 42 --use_local_metadata --resume
```

**Rule:** Rank buckets **only within this subset** when using these numbers for Skill targeting — do not compare headline accuracy to full 3,212-run tables without relabeling.

### Rank buckets from saved JSON

```bash
python scripts/rank_buckets_from_result.py result/Qwen3-Omni-3B.json --bottom-ratio 0.35
```

(Works from **`agent/`** cwd; pass a path to **`../baseline/result/...`** if needed.)

## Columns available

Typical rows expose at least: `question_id`, `video_id`, `question`, `choices`, `answer`, `task_id`, `category`, `sub_category`, and clip paths. Confirm with your local schema:

```bash
python -c "import pandas as pd; print(pd.read_csv(\"Holistic_AVQA_bench/test.csv\", nrows=0).columns)"
```

## Built-in filters (no code change)

| Goal | How |
|------|-----|
| Single task family | `--task_id "<TaskName>"` (matches `task_id` column) |
| One L1 bucket | `--category "<Category>"` |
| One L2 bucket | `--sub_category "<SubCategory>"` |
| Stratified sample | `--sample_fraction 0.1 --sample_seed 42 --stratify_key category` |

## Suggested “hard” slices (align with research themes)

Use **task / category / sub_category** from your table; names follow the paper-style hierarchy (Audio-centric, Speaker-centric, Visual-centric and their sub-tasks).

| Research theme | Where to start (examples) | Notes |
|----------------|---------------------------|--------|
| Multi-speaker / overlap stress | `category` **Speaker-centric**; sub-tasks like **Speaker Counting**, **Speaker Recognition**, **Speech Counting** | Overlap is a *data property*; if `test.csv` has no speaker-count column, filter by `video_id` after manual tagging or use clip-level heuristics offline. |
| Local visual grounding | `category` **Visual-centric**; **Attribute Recognition**, **Visual Counting** | Many items are anchor–target by design. |
| Temporal / speech quantity | **Speech Duration**, **Speech Rate**, **Speech Intensity**, **Speech Pitch** | Short clips still stress *relative* timing within clip. |
| ASR / lexicon | **Speech Recognition** | Good proxy for “what was said” under noise. |

## Repro examples

```bash
cd agent
# Speaker-centric only (adjust string to match your CSV exactly)
python main.py --model_name Qwen3-Omni-3B --category "Speaker-centric" --use_local_metadata

# Single task_id (see unique values in test.csv)
python main.py --model_name Qwen3-Omni-3B --task_id "Speaker Counting" --use_local_metadata

# 10% stratified by category (~322 questions full run scale; scaled by fraction)
python main.py --model_name Qwen3-Omni-3B --sample_fraction 0.1 --sample_seed 42 --use_local_metadata
```

## Advanced: CSV-derived masks (optional follow-up)

For rules like “videos with ≥5 speakers” or “overlap segments”, add a **small offline script** that outputs a list of `question_id` / `video_id`, then intersect with `test.csv`. Keep the mask file under `agent/docs/` or `record/` with a checksum for paper reproducibility.

## Long-video extension (protocol only)

Current clips are benchmark-short by design. If you introduce longer sources later:

1. Freeze a **protocol id** (`long_video_protocol_v1`) in the mask file.  
2. Do **not** mix long-video aggregates with vanilla AV-SpeakerBench tables in the same row of a leaderboard without labeling the protocol.
