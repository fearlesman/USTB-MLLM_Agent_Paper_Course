## Context

The repository separates `baseline/` and `agent/`, but `agent/model/__init__.py` currently loads `baseline/model/__init__.py` and re-exports `inference` unchanged. The baseline inference loop constructs `question_prompt` and sends it directly to model backends; it does not import `agent/orchestrator/runner.py`, does not call `augment_prompt_for_inference`, and does not append agent traces. As a result, `main_agent.py` can produce `agent_`-prefixed artifacts while still using baseline-only prompts.

The Skills pipeline already contains useful pieces: task triggers, ASR/VAD/diarization helpers, evidence formatting, environment gates such as `AV_SPEAKERBENCH_SKILL_INJECT`, and status tags. The missing piece is a reliable integration point in the evaluation loop plus observability proving that integration happened.

## Goals / Non-Goals

**Goals:**

- Ensure agent-track inference always passes through the orchestrator unless explicitly disabled.
- Preserve baseline metric structure, answer parsing, model backend selection, checkpoint/resume behavior, and output file naming.
- Keep LM-only parity runs possible for ablation, but make them visibly different from evidence-injected agent runs in records and traces.
- Make trace fingerprints stable across Python processes and capture enough prompt/evidence metadata for debugging.
- Add tests or smoke scripts that detect a disconnected orchestrator.

**Non-Goals:**

- Rewriting model backend implementations.
- Making every planned Skill production-grade in this change.
- Changing the dataset schema or benchmark metric definitions.
- Making evidence injection the default for every run without an explicit configuration decision.

## Decisions

1. Implement an agent-owned inference loop instead of mutating the frozen baseline loop in place.

   Rationale: the baseline tree is intended to remain paper-compatible. The agent loop can copy the small amount of orchestration-sensitive control flow from baseline and call shared backend helpers, while keeping `baseline/main.py` behavior unchanged.

   Alternative considered: patch `baseline/model/__init__.py` to conditionally call the orchestrator based on `AV_SPEAKERBENCH_EVAL_TRACK`. This is lower code duplication but makes baseline behavior depend on agent imports and environment state.

2. Treat the original MC prompt and prepared agent prompt as separate artifacts.

   Rationale: resume checks and ablation analysis need to know whether a cached answer corresponds to the same model-visible prompt. Store the original prompt for compatibility and store `agent_prompt_fp`, `agent_prompt_changed`, `agent_evidence_injected`, and the final prompt used for cache comparison when practical.

   Alternative considered: overwrite `question_prompt` only. That is simple, but it hides whether differences came from original question text or agent evidence.

3. Keep `AV_SPEAKERBENCH_SKILLS=off` as the explicit orchestrator bypass and `AV_SPEAKERBENCH_SKILL_INJECT=1` as the evidence injection gate.

   Rationale: this preserves existing ablation semantics. The main behavioral change is that the orchestrator is invoked by the agent loop and emits observability even when injection is off.

   Alternative considered: make Skill evidence injection on by default for all agent runs. That would make the agent visibly active immediately, but it risks breaking existing parity scripts and could contaminate baseline comparisons.

4. Use deterministic hashing for prompt fingerprints.

   Rationale: Python's built-in `hash()` is process-randomized and unsuitable for comparing records across runs. A short SHA-256 prefix is stable and cheap.

   Alternative considered: store full prompts only. Full prompts are useful for records, but stable fingerprints are better for compact trace joins and privacy-conscious summaries.

5. Add a minimal smoke backend path for orchestration tests.

   Rationale: verifying that the model-visible prompt changes should not require paid API calls or GPU inference. A smoke script can monkeypatch or use a deterministic model stub to assert prompt routing, Skill tags, and trace fields.

   Alternative considered: rely on full benchmark runs. That is too slow and too expensive for regression testing this integration boundary.

## Risks / Trade-offs

- Agent loop diverges from baseline loop over time -> Keep the diff narrow, document copied sections, and add a comparison checklist for future baseline ports.
- Evidence injection can degrade some model answers through prompt noise -> Keep injection environment-gated, support Skill allowlists, and require per-bucket ablations.
- Optional tools may be missing or slow -> Preserve current backend fallbacks and trace `errors`/`bottleneck_tags` instead of failing whole evaluations by default.
- Resume behavior can reuse answers from a different prompt if cache keys are wrong -> Compare against the model-visible prompt fingerprint and record injection state.
- Trace files can grow quickly -> Store compact hashes and status fields by default; keep full prompts in per-question records only where already expected.

## Migration Plan

1. Add agent-owned inference wiring and deterministic trace hashing.
2. Run smoke tests with Skills off, Skills on without injection, and Skills on with injection.
3. Run a small local subset with `AV_SPEAKERBENCH_INFERENCE_LIMIT=1` to verify model backend routing and artifact schema.
4. Compare baseline and agent LM-only parity outputs on a tiny deterministic or cached subset.
5. Update `agent/README.md` with the exact environment combinations for parity, active-agent, and synthetic-evidence smoke runs.
