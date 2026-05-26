## 1. Baseline Diagnosis

- [x] 1.1 Add or update a lightweight test that demonstrates `agent/model/__init__.py` currently re-exports baseline inference without calling `augment_prompt_for_inference`.
- [x] 1.2 Identify the minimum backend helper functions and record/flush utilities the agent inference loop must reuse from `baseline/model/__init__.py`.

## 2. Agent Inference Wiring

- [x] 2.1 Implement an agent-owned `inference(args, dataset)` path that constructs the baseline MC prompt, resolves media paths, and calls `augment_prompt_for_inference` before backend dispatch.
- [x] 2.2 Pass `AgentPrepResult.final_prompt` to every model backend branch in the agent inference path.
- [x] 2.3 Preserve baseline answer parsing, metric bucket updates, result accuracy calculations, checkpoint writes, and model initialization semantics.
- [x] 2.4 Preserve `AV_SPEAKERBENCH_SKILLS=off` as an explicit Skills-disabled parity path.
- [x] 2.5 Update resume/cache checks so cached answers are reused only when the model-visible prompt fingerprint and injection state match.

## 3. Evidence And Trace Observability

- [x] 3.1 Replace `prompt_fingerprint()` with a deterministic SHA-256 based fingerprint.
- [x] 3.2 Add per-question record fields for original prompt fingerprint, agent prompt fingerprint, `agent_prompt_changed`, `agent_evidence_injected`, `skills_invoked`, `bottleneck_tags`, and orchestrator errors.
- [x] 3.3 Append agent trace records from the agent inference path with prompt fingerprints, Skill status tags, bottleneck tags, parsed answer, match status, and inference latency.
- [x] 3.4 Ensure synthetic/stub evidence and missing backend/media failures are visible through `bottleneck_tags` or structured errors.

## 4. Verification

- [x] 4.1 Add a smoke check that fails if `AV_SPEAKERBENCH_SKILL_INJECT=1` does not produce a model-visible prompt containing `Structured_skill_evidence` when a Skill injects evidence.
- [x] 4.2 Add a parity smoke check showing Skills-enabled but injection-disabled runs keep the model-visible prompt equal to the baseline prompt while still reporting Skill tags.
- [x] 4.3 Add a deterministic fingerprint test that verifies identical prompts hash the same across separate Python processes.
- [x] 4.4 Run the existing agent smoke script and the new focused checks locally.

## 5. Documentation

- [x] 5.1 Update `agent/README.md` to explain the three supported modes: Skills off, Skills traced without injection, and evidence-injected active agent.
- [x] 5.2 Document the record and trace fields used to prove whether the model saw agent evidence.
- [x] 5.3 Document a recommended small-subset ablation command comparing baseline, agent LM-only parity, and evidence-injected agent runs.
