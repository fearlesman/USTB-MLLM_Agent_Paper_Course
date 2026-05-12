# Skill / Tool inject — ablation protocol

This document fixes a reproducible **baseline vs treatment** comparison when evaluating whether `Structured_skill_evidence` changes MC accuracy.

## Defaults that matter

- **LM-only prompt (pairable with baseline semantics)**  
  - Either run with **`AV_SPEAKERBENCH_EVAL_TRACK=`** (empty; no augment hook in the current harness),  
  - **or** keep the agent track but set **`AV_SPEAKERBENCH_SKILL_INJECT`** unset / not `1` — the MC stem is unchanged (Skills still emit trace tags unless `AV_SPEAKERBENCH_SKILLS=off`).

- **Evidence on**  
  - Set **`AV_SPEAKERBENCH_SKILL_INJECT=1`**.  
  - For conclusions about perception tools, prefer **`AV_SPEAKERBENCH_ALLOW_SYNTHETIC_ASR=0`** (default) and **`AV_SPEAKERBENCH_ASR_BACKEND=faster_whisper`** or **`whisper`** (after installing deps). Runs that use **`ASR_BACKEND=stub`** plus `AV_SPEAKERBENCH_STUB_ASR_TEXT` are **pipeline smoke only** — trace will include **`evidence_synthetic`**.

## Controlled comparison

Use the **same** CLI except inject-related env vars:

| Knob | Baseline-like run | Treatment run |
|------|-------------------|----------------|
| `AV_SPEAKERBENCH_SKILL_INJECT` | unset / `0` | `1` |
| `--sample_fraction` / `--sample_seed` / `--task_id` | identical | identical |
| `--model_name` / `--data_path` | identical | identical |

Write two aggregate JSON paths, e.g. `result/agent_Model_inject_off.json` vs `result/agent_Model_inject_on.json`.

## Bucket-level analysis (required)

Overall accuracy hides weak-bucket shifts. Use:

```bash
cd agent
python scripts/rank_buckets_from_result.py path/to/off.json --bottom-k 12
python scripts/rank_buckets_from_result.py path/to/on.json --bottom-k 12
python scripts/compare_result_buckets.py ^
  --baseline path/to/off.json ^
  --treatment path/to/on.json ^
  --levels both ^
  --min-total 10
```

Interpret **delta_acc** per `level 2:` / `level 3:` key; correlate with `record/agent_trace_*.jsonl` (under this `agent/` tree) joined on `question_id` for error mining.

## Trace tags

Filter JSONL rows by **`bottleneck_tags`** containing **`evidence_synthetic`** so synthetic‑stub runs are excluded from “real Tool gain” spreadsheets.
