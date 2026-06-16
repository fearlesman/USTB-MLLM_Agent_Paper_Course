# AV Multimodal Tool and Skill Survey

Survey date: 2026-06-15

This note collects **external AV perception tools** and translates them into **project-facing Skill / Tool candidates** for the `agent/` track. The goal is not to list every popular repo, but to identify components that fit the current `agent/tools` + `agent/skills` architecture and can be evaluated with the existing AV-SpeakerBench trace pipeline.

## Selection rules

- Prefer **primary sources**: official repositories, official docs, model cards, or author-maintained pages.
- Prefer tools that already have a **Python inference path** and can run on local files.
- Prefer components that map to current bottlenecks in `agent/docs/MM_AGENT_DESIGN.md`: perception, alignment, reasoning, and tool-boundary errors.
- Treat every addition as **optional evidence**, not a replacement for the multimodal LM backbone.

## Candidate tools by capability

| Capability | Candidate | Why it matters here | Likely integration point | Phase |
|------------|-----------|---------------------|--------------------------|-------|
| Voice activity detection | [Silero VAD](https://github.com/snakers4/silero-vad) | Stronger speech/non-speech gating than the current energy-only fallback, especially on noisy clips and short pauses. | Add `silero` backend to `agent/tools/audio_vad.py`; keep current energy VAD as zero-dependency fallback. | P0 |
| Fast ASR | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Already aligned with current code style; good local default for anchor-window ASR. | Keep as the main local backend in `agent/tools/audio_asr.py`. | Existing / P0 hardening |
| Word-level ASR + alignment | [WhisperX](https://github.com/m-bain/whisperx) | Adds word timestamps and diarization-oriented alignment, which is directly useful for quote anchoring and turn order. | Optional backend in `agent/tools/audio_asr_words.py`. | P0 |
| Speaker diarization | [pyannote.audio](https://github.com/pyannote/pyannote-audio), [Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1), [pyannoteAI](https://docs.pyannote.ai/quickstart) | Best fit for `who spoke when`; aligns with current `audio_diar.py` abstraction and existing env flags. | Upgrade `agent/tools/audio_diar.py` from `speaker-diarization-3.1` default toward `community-1`; keep cloud path as optional. | P0 |
| Alternative diarization stack | [NVIDIA NeMo diarization examples](https://github.com/NVIDIA/NeMo/tree/main/examples/speaker_tasks/diarization) | Useful when diarization plus speaker-count control or NeMo-based ASR pipelines are needed. Heavier than pyannote. | New optional backend in `agent/tools/audio_diar.py` or a separate `audio_diar_nemo.py`. | P1 |
| Speaker identification / voiceprints | [SpeechBrain](https://github.com/speechbrain/speechbrain) | Complements diarization by binding anonymous clusters to known or repeated speakers across clips. | New tool for embeddings / voiceprint matching; feeds a `known_speaker_voiceprint` skill. | P1 |
| Active speaker detection | [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) | Directly addresses the benchmark's hardest cross-modal question: which visible face is actually speaking. | New `agent/tools/video_active_speaker.py`; feeds speaker-binding skills. | P2 |
| Person detection + tracking | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [Track mode docs](https://docs.ultralytics.com/modes/track) | More reliable people snapshots and trajectories than isolated frame counting; useful for visual counting and anchor persistence. | Extend `agent/tools/video_people_snap.py` with tracked IDs over sampled frames. | P0 |
| Face / pose / holistic landmarks | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker), [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker), [Holistic Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker) | Good fit for visual anchors like "wiggles fingers", "raises hand", "turns head", or coarse activity cues. | New `video_landmarks.py` tool; feeds gesture- or pose-based anchor skills. | P1 |
| OCR / scene text | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Helps when the visual answer cue or anchor depends on signs, subtitles, captions, or overlaid labels. | New `video_ocr.py` tool; feeds `ocr_scene_cue`. | P1 |
| Open-vocabulary object grounding | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SAM](https://github.com/facebookresearch/segment-anything) | Best fit for replacing the current `visual_anchor_ground` stub with phrase-conditioned regions or masks. | New `video_open_vocab_ground.py`; feeds `visual_anchor_ground`. | P2 |
| Multi-object tracking core | [ByteTrack](https://github.com/ifzhang/ByteTrack) | Helps maintain person identity over time when a detector alone fragments tracks. | Either direct integration or rely on Ultralytics track mode if that is sufficient. | P1 |
| Speech separation | [Asteroid](https://github.com/asteroid-team/asteroid) | Useful for overlap-heavy speaker tasks where diarization or ASR collapses under simultaneous speech. | New `audio_separate.py`; use only as a retry path on high-overlap traces. | P2 |

## Recommended Skill additions for this repo

The current codebase already separates **backend tools** from **prompt-injected skills**. The following additions fit that contract.

| Skill id | Backing tool(s) | Output contract | Best target tasks | Notes |
|----------|------------------|-----------------|-------------------|-------|
| `word_aligned_quote_anchor` | WhisperX or `faster-whisper` + alignment | `quote -> [t0, t1]`, confidence, local transcript slice | Speech Recognition, Speech Counting, Visual Counting | Stronger than coarse VAD windows when the question includes a quoted phrase. |
| `active_speaker_bind` | TalkNet-ASD + face tracks + diar/ASR | `time span -> visible speaker candidate ids` | Speaker Recognition, Speaker Detection, Speaker Counting | High value, but highest engineering cost. |
| `person_track_sheet` | Ultralytics track mode or ByteTrack | time-indexed person IDs and visibility ranges | Visual Counting, Activity Recognition | Cleaner evidence than one-frame people snapshots. |
| `gesture_anchor_ground` | MediaPipe pose / holistic landmarks | event markers such as hand raise, head turn, gesture spans | Activity Recognition, Speech Counting with visual anchors | Best for "before/after X does Y" stems. |
| `ocr_scene_cue` | PaddleOCR | extracted text with frame timestamps | Attribute Recognition, Activity Recognition | Only trigger when stems mention signs, captions, labels, or on-screen text. |
| `known_speaker_voiceprint` | SpeechBrain embeddings | diar cluster -> known speaker label scores | Speaker Recognition | Useful only if repeated characters or curated references exist. |
| `open_vocab_visual_anchor` | GroundingDINO + SAM | phrase-conditioned boxes / masks over sampled frames | Visual Counting, Attribute Recognition | Natural replacement for the current `visual_anchor_ground` stub. |
| `overlap_separation_retry` | Asteroid + ASR/diar rerun | alternate transcript / speaker spans for overlap windows | Speech Counting, Speaker Recognition | Keep behind a narrow trigger to avoid latency blowups. |

## Priority roadmap

### P0: direct value with modest code churn

1. **Add `silero` as a VAD backend** in `agent/tools/audio_vad.py`.
2. **Add `whisperx` as an optional word-timestamp backend** in `agent/tools/audio_asr_words.py`.
3. **Refresh pyannote defaults** in `agent/tools/audio_diar.py` so the local path targets `community-1` rather than older defaults.
4. **Upgrade `video_people_snap.py` from count-only toward short-horizon tracking**, ideally through Ultralytics track mode first.

Why P0 first:

- These tools map directly to existing stubs and current env-controlled backends.
- They improve the benchmark's core weak spots without forcing a redesign of the orchestrator.
- They can be evaluated with the existing `agent_trace_*.jsonl` outputs and prompt-injection ablation flow.

### P1: structured visual grounding and speaker identity

1. Add `video_landmarks.py` based on MediaPipe pose / face / holistic tasks.
2. Add `video_ocr.py` based on PaddleOCR.
3. Add a speaker-embedding tool based on SpeechBrain.
4. Optionally add a NeMo diarization backend if pyannote coverage is insufficient on your clips.

Why P1 second:

- These additions unlock better visual anchors and repeated-speaker reasoning.
- They benefit from trace-driven trigger design, not blanket invocation.

### P2: research-heavy components

1. Add `video_active_speaker.py` around TalkNet-ASD.
2. Replace `visual_anchor_ground` stub with GroundingDINO + SAM.
3. Add `audio_separate.py` as a retry-only path for overlap-heavy cases.

Why P2 last:

- These are high upside, but they also add the largest dependency, runtime, and debugging burden.
- They should be justified by weak buckets in your local `result/*.json`, not by generic tool popularity.

## Suggested code-level landing points

| File / area | Suggested upgrade |
|-------------|-------------------|
| `agent/tools/audio_vad.py` | Introduce backend registry: `energy` / `silero`. |
| `agent/tools/audio_asr_words.py` | Add `whisperx` backend for word timestamps and alignment. |
| `agent/tools/audio_diar.py` | Keep `stub` + `pyannote_api`; add clearer local pyannote model selection and optional NeMo backend. |
| `agent/tools/video_people_snap.py` | Evolve from frame snapshots into sampled tracking summaries. |
| `agent/skills/impl.py` | Add new skill wrappers with narrow triggers and short evidence payloads. |
| `agent/skills/triggers.py` | Gate expensive tools by `task_id`, quote presence, overlap suspicion, or visual-anchor language. |
| `agent/docs/MM_AGENT_DESIGN.md` | Use local result buckets to justify which P1/P2 tools are worth implementing next. |

## Integration constraints

- Do not make heavy tools default-on. Preserve a **cheap local baseline** and add stronger backends behind env flags.
- Every new skill should emit **compact structured evidence**, not long prose.
- Every heavy backend should add trace tags for:
  - backend used
  - latency
  - empty-output vs hard error
  - fallback path taken
- New tools should be evaluated with **same-split ablations** against the current LM-only or lightweight-tool baseline.

## Source list

### Speech / diarization

- Silero VAD: <https://github.com/snakers4/silero-vad>
- faster-whisper: <https://github.com/SYSTRAN/faster-whisper>
- WhisperX: <https://github.com/m-bain/whisperx>
- pyannote.audio: <https://github.com/pyannote/pyannote-audio>
- pyannote `speaker-diarization-community-1`: <https://huggingface.co/pyannote/speaker-diarization-community-1>
- pyannoteAI quickstart: <https://docs.pyannote.ai/quickstart>
- NVIDIA NeMo diarization examples: <https://github.com/NVIDIA/NeMo/tree/main/examples/speaker_tasks/diarization>
- SpeechBrain: <https://github.com/speechbrain/speechbrain>
- Asteroid: <https://github.com/asteroid-team/asteroid>

### Vision / audiovisual grounding

- TalkNet-ASD: <https://github.com/TaoRuijie/TalkNet-ASD>
- Ultralytics YOLO: <https://github.com/ultralytics/ultralytics>
- Ultralytics track mode: <https://docs.ultralytics.com/modes/track>
- MediaPipe Face Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker>
- MediaPipe Pose Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker>
- MediaPipe Holistic Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker>
- PaddleOCR: <https://github.com/PaddlePaddle/PaddleOCR>
- GroundingDINO: <https://github.com/IDEA-Research/GroundingDINO>
- Segment Anything (SAM): <https://github.com/facebookresearch/segment-anything>
- ByteTrack: <https://github.com/ifzhang/ByteTrack>

## Bottom line

If the goal is to improve AV-SpeakerBench agent performance without destabilizing the codebase, the most defensible next moves are:

1. stronger VAD,
2. word-level aligned ASR,
3. better diarization defaults, and
4. tracked visual people summaries.

Everything else should be justified by local weak-bucket evidence before it becomes an engineering project.
