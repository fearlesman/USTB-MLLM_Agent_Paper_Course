## Why

The agent track currently looks active from the folder layout and trace-oriented code, but the main evaluation path re-exports the baseline inference function and never calls `augment_prompt_for_inference`. Even when Skills are run manually, the default configuration leaves the multiple-choice prompt unchanged, so agent tools can appear in traces without affecting model decisions.

## What Changes

- Route `agent/main.py` evaluations through an agent-owned inference wrapper that invokes the orchestrator before each model call.
- Preserve baseline-compatible metrics and resume behavior while recording both the original prompt and the agent-prepared prompt metadata.
- Make Skill evidence activation explicit and auditable: support LM-only parity, evidence-injected agent runs, allowlisted Skill runs, and hard failure tags when expected evidence is missing.
- Replace non-deterministic trace prompt fingerprints with stable hashes and add enough trace fields to prove whether the model saw agent evidence.
- Add smoke and focused regression checks that fail if the agent path silently falls back to baseline-only inference.

## Capabilities

### New Capabilities
- `agent-orchestrated-inference`: Agent-track evaluation must execute the orchestrator and pass its prepared prompt to the selected model backend when enabled.
- `agent-evidence-observability`: Agent runs must expose stable, per-question observability showing which Skills ran, whether evidence was injected, what prompt was used, and whether any evidence was synthetic or missing.

### Modified Capabilities

None.

## Impact

- Affected code: `agent/model/__init__.py`, `agent/orchestrator/*`, `agent/skills/*`, `agent/scripts/*`, and possibly shared logic currently embedded in `baseline/model/__init__.py`.
- Affected behavior: `python main_agent.py ...` and `cd agent && python main.py ...` should no longer be indistinguishable from baseline inference when Skills are enabled.
- Compatibility: existing result breakdown keys remain unchanged; record schema gains agent-specific fields.
- Dependencies: no new mandatory external services; optional ASR/diarization backends remain environment-gated.
