# Baseline evaluation harness (AV-SpeakerBench)

This directory contains the **original-style** benchmark code: `main.py`, `dataset/`, `model/`, and helper scripts. It is kept separate from the **`../agent/`** research track (multimodal Agent, skills, tools).

## Dataset location (unchanged)

By default, clips and `test.csv` are read from:

- **`../Holistic_AVQA_bench/`** (sibling of this `baseline/` folder at the repository root)

Override with:

- **`AV_SPEAKERBENCH_DATA_ROOT`** — absolute path to the dataset root, or
- **`--data_path`** on the command line.

## Run

From this directory:

```bash
python main.py --model_name Qwen3-Omni-3B --use_local_metadata
```

Or from the **repository root**, the shim `../main.py` switches into `baseline/` for you:

```bash
python main.py --model_name Qwen3-Omni-3B --use_local_metadata
```

Outputs (`result/`, `record/`) are created relative to the **current working directory** after the shim runs — i.e. under `baseline/` when launching from root via `../main.py`.

## Scripts

- `scripts/backfill_experiment_manifest.py` — rebuild `*_experiment.json` from existing metrics + records.

## Environment

See root **`../README.md`** and `environment.yml` in this folder.
