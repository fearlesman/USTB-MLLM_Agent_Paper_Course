## ADDED Requirements

### Requirement: Agent records expose orchestration state
Each agent per-question record SHALL include fields that identify whether the orchestrator ran, whether the model-visible prompt changed, which Skills were invoked, and which bottleneck or error tags were emitted.

#### Scenario: Skills run without injection
- **WHEN** an agent question is processed with Skills enabled and injection disabled
- **THEN** the record includes Skill invocation tags
- **AND** the record marks `agent_prompt_changed` as false
- **AND** the record marks evidence injection as false

#### Scenario: Skills inject evidence
- **WHEN** an agent question is processed with evidence injection enabled and at least one Skill injects text
- **THEN** the record marks evidence injection as true
- **AND** the record includes the prepared prompt fingerprint and Skill invocation tags

### Requirement: Trace prompt fingerprints are stable
Agent trace prompt fingerprints SHALL be deterministic across Python processes for identical prompt text.

#### Scenario: Same prompt across processes
- **WHEN** the same prompt text is fingerprinted in two separate Python processes
- **THEN** the returned fingerprint is identical

#### Scenario: Different prompt after injection
- **WHEN** Skill evidence changes the model-visible prompt
- **THEN** the trace fingerprint for the prepared prompt differs from the original baseline prompt fingerprint

### Requirement: Trace distinguishes real evidence from placeholders
Agent traces SHALL expose whether evidence came from real tool output, skipped Skills, missing backends, or synthetic/stub evidence.

#### Scenario: Stub ASR evidence is used
- **WHEN** ASR evidence is produced from a stub or synthetic environment value
- **THEN** the trace or record includes an `evidence_synthetic` bottleneck tag

#### Scenario: Expected evidence is missing
- **WHEN** a triggered Skill cannot produce evidence because required media or backend support is missing
- **THEN** the trace or record includes a structured error or bottleneck tag that makes the missing evidence visible

### Requirement: Smoke checks prove active agent behavior
The project SHALL provide a low-cost verification path that fails when the agent entrypoint does not invoke the orchestrator or does not pass the prepared prompt to inference under injection.

#### Scenario: Orchestrator disconnected
- **WHEN** the smoke check runs with evidence injection enabled
- **AND** the model-visible prompt does not include orchestrator-produced evidence
- **THEN** the smoke check fails

#### Scenario: LM-only parity smoke
- **WHEN** the smoke check runs with injection disabled
- **THEN** it verifies that Skill tags can be recorded while the model-visible prompt remains unchanged
