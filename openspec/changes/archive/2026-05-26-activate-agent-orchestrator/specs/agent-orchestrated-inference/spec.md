## ADDED Requirements

### Requirement: Agent evaluation invokes orchestrator
The agent evaluation path SHALL invoke `augment_prompt_for_inference` for each evaluated question unless Skills are explicitly disabled through the documented environment gate.

#### Scenario: Default agent run reaches orchestrator
- **WHEN** `python main_agent.py` or `cd agent && python main.py` evaluates a question with `AV_SPEAKERBENCH_SKILLS` unset
- **THEN** the orchestrator is called with the question metadata and resolved audio, visual, and audiovisual paths

#### Scenario: Skills disabled bypasses orchestrator effects
- **WHEN** an agent evaluation runs with `AV_SPEAKERBENCH_SKILLS=off`
- **THEN** the model-visible prompt remains the baseline multiple-choice prompt and the record marks Skills as disabled

### Requirement: Agent prompt is model-visible when evidence injection is enabled
The agent evaluation path SHALL pass `AgentPrepResult.final_prompt` to the selected model backend whenever Skill evidence injection is enabled and the orchestrator returns an augmented prompt.

#### Scenario: Evidence-injected prompt is used for inference
- **WHEN** `AV_SPEAKERBENCH_SKILL_INJECT=1` and at least one Skill returns `injected_text`
- **THEN** the prompt sent to the model contains `Structured_skill_evidence`
- **AND** the original baseline prompt is not the prompt used for that model call

#### Scenario: Injection disabled preserves LM-only parity
- **WHEN** `AV_SPEAKERBENCH_SKILL_INJECT` is unset and Skills are not disabled
- **THEN** Skills may run for tracing
- **AND** the prompt sent to the model is byte-for-byte equal to the baseline multiple-choice prompt

### Requirement: Baseline compatibility is preserved
The agent evaluation path SHALL preserve the benchmark result keys, answer parsing behavior, model backend selection, and final aggregate metric semantics used by baseline evaluation.

#### Scenario: Agent run writes compatible metrics
- **WHEN** an agent evaluation completes
- **THEN** result JSON still contains `level 1`, `level 2`, and `level 3` bucket metrics with `matched`, `total`, and `accuracy` semantics compatible with baseline output

#### Scenario: Existing model backends remain selectable
- **WHEN** a supported `--model_name` is used from the agent entrypoint
- **THEN** the same backend branch that baseline would use for that model is selected after prompt preparation

### Requirement: Resume uses model-visible prompt identity
The agent evaluation path SHALL avoid reusing a cached answer when the model-visible prompt differs from the prompt used to create that cached answer.

#### Scenario: Injection state changes between runs
- **WHEN** a question has an existing record from a run with injection disabled
- **AND** the same question is evaluated with `AV_SPEAKERBENCH_SKILL_INJECT=1`
- **THEN** the cached answer is not reused unless the stored model-visible prompt fingerprint matches the newly prepared prompt fingerprint
