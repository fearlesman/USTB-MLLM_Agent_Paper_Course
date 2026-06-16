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

Use the **baseline layer** only:

```bash
mamba env create -f environment-mamba.yml
mamba activate av-speakerbench
```

`environment.yml` and `environment-mamba.yml` now describe the same portable baseline env. If you already have a Python env, install the same deps with:

```bash
pip install -r requirements-baseline.txt
```

This baseline layer does **not** include the heavier agent-only backends such as WhisperX, pyannote.audio, Ultralytics, or Silero VAD. Those live under [`../agent/requirements-agent.txt`](../agent/requirements-agent.txt) and [`../agent/environment-mamba.yml`](../agent/environment-mamba.yml).

For local open-model inference, install model-specific extras separately. The current Qwen3-Omni code path expects `transformers==4.42.2`, `peft==0.13.2`, and related runtime helpers such as `accelerate`, `sentencepiece`, plus the `qwen_omni_utils` module imported by [`model/open_model/Qwen3Omni/inference.py`](model/open_model/Qwen3Omni/inference.py).
