# Project Timeline: Live Audio-to-Lyrics Alignment Engine

## 1. Project Objective

Build a production-ready MVP that can:

* capture live church audio;
* extract audio features using ONNX Runtime;
* align live performance audio with a known reference track;
* estimate the current song position in real time;
* map the detected position to lyric slides;
* trigger presentation software through OSC;
* freeze slide changes when confidence is low;
* allow manual operator override.

The first release will support one selected song at a time. Automatic song recognition will be treated as a later phase.

---

# 2. Recommended Architecture

Use a modular monolith with hexagonal architecture.

```text
Live Audio
    │
    ▼
Audio Input Adapter
    │
    ▼
Bounded Audio Queue
    │
    ▼
Feature Extraction
    │
    ▼
Online Alignment Engine
    │
    ▼
Position Stabilizer
    │
    ▼
Slide Decision Engine
    │
    ▼
Presentation Adapter
    │
    ├── OSC
    ├── FreeShow
    └── ProPresenter
```

Reference-track creation should remain a separate offline workflow:

```text
Reference Audio
    │
    ▼
Offline Feature Extraction
    │
    ▼
Reference Profile
    │
    ├── reference_features.npy
    ├── profile.json
    └── metadata.json
```

## Main architectural modules

### Domain

Contains pure business and alignment logic:

* audio and feature models;
* DTW calculation;
* confidence calculation;
* position tracking;
* slide resolution;
* slide stability rules.

### Application

Coordinates the runtime:

* processing workers;
* bounded queues;
* lifecycle management;
* metrics;
* error recovery.

### Adapters

Connects external systems:

* microphone input;
* WAV input;
* ONNX Runtime;
* filesystem profiles;
* OSC;
* console output;
* simulation mode.

The domain layer must not depend on `sounddevice`, `onnxruntime`, or `python-osc`.

---

# 3. Project Scope

## In scope for MVP

* one selected song at a time;
* one reference performance per song;
* 16 kHz mono audio input;
* dedicated vocal-heavy mixer feed;
* ONNX-based feature extraction;
* online subsequence DTW;
* confidence-based tracking;
* timestamp-to-slide mapping;
* OSC slide triggering;
* console monitoring;
* simulation mode;
* manual override integration point;
* recorded replay testing;
* basic operational metrics.

## Out of scope for MVP

* automatic song recognition;
* support for an unlimited song library;
* cloud processing;
* mobile application;
* multi-church tenancy;
* automatic lyrics licensing;
* AI-generated lyrics;
* advanced source separation;
* adaptive machine learning from user feedback;
* full ProPresenter or FreeShow plugin development;
* multi-language lyric matching.

These can be added after the alignment MVP is proven.

---

# 4. Project Duration

## Total duration

```text
12 weeks development
+ 2 weeks pilot and stabilization
= 14 weeks total
```

## Sprint model

* Sprint length: 2 weeks
* Development sprints: 6
* Pilot and hardening: 2 weeks
* Demo at the end of every sprint
* Retrospective after every sprint
* Architecture review after Sprints 2 and 4

---

# 5. Delivery Timeline

```text
Week 1–2     Sprint 1: Foundation and Audio Ingestion
Week 3–4     Sprint 2: ONNX Feature Extraction
Week 5–6     Sprint 3: Reference Profile Builder
Week 7–8     Sprint 4: Offline Alignment
Week 9–10    Sprint 5: Online Tracking and Stability
Week 11–12   Sprint 6: OSC Integration and End-to-End MVP
Week 13      Church Pilot
Week 14      Stabilization and Release
```

---

# 6. Sprint Plan

## Sprint 1 — Foundation and Audio Ingestion

### Goal

Create the project foundation and capture live audio safely.

### Scope

* create repository structure;
* configure Python project;
* add linting, formatting, type checking, and testing;
* define domain models and interfaces;
* implement application lifecycle;
* implement structured logging;
* implement graceful shutdown;
* implement sound-device audio input;
* implement bounded audio queue;
* implement simulated audio source;
* implement silence and clipping detection;
* implement audio metrics.

### Deliverables

* runnable command-line application;
* working live microphone capture;
* simulation mode;
* audio diagnostics;
* initial CI pipeline;
* architecture decision records.

### Demo

The application captures audio and prints:

```text
device=Church Mixer
rms=0.21
peak=0.78
queue=1/16
chunks_received=500
chunks_dropped=0
```

### Acceptance criteria

* audio runs continuously for 30 minutes;
* callback does not block;
* memory usage remains stable;
* dropped chunks are counted;
* application shuts down cleanly;
* automated tests pass.

---

## Sprint 2 — ONNX Feature Extraction

### Goal

Convert audio chunks into normalized feature frames.

### Scope

* implement ONNX Runtime adapter;
* inspect model metadata;
* dynamically detect input and output names;
* support common input shapes;
* normalize waveform input;
* normalize output features;
* process multiple feature frames per chunk;
* add provider fallback;
* add deterministic simulated extractor;
* add inference latency metrics;
* handle invalid outputs safely.

### Deliverables

* ONNX model inspection tool;
* feature extraction pipeline;
* simulation fallback;
* inference error handling;
* feature-level metrics.

### Demo

```text
chunk=42
input_shape=(1, 4096)
output_shape=(1, 12, 64)
frames=12
latency_ms=24.7
```

### Acceptance criteria

* model loads successfully when available;
* application still runs without the model;
* malformed output does not crash the runtime;
* feature output contains finite values;
* inference runs outside the audio callback;
* average processing stays below real-time limits.

---

## Sprint 3 — Reference Profile Builder

### Goal

Create reusable reference profiles from studio or church recordings.

### Scope

* build offline reference CLI;
* read WAV files;
* convert audio to mono;
* resample to 16 kHz;
* run feature extraction;
* save reference features;
* define slide intervals;
* validate slide timestamps;
* save metadata;
* load and validate profiles;
* add profile versioning.

### Deliverables

```text
profiles/amazing-grace/
├── profile.json
├── reference_features.npy
└── metadata.json
```

### Demo

```bash
python tools/build_reference.py \
  --audio amazing-grace.wav \
  --slides amazing-grace-slides.json \
  --output profiles/amazing-grace
```

### Acceptance criteria

* reference profiles are reproducible;
* feature dimensions match live extraction;
* invalid profiles fail with clear errors;
* profiles load without rerunning inference;
* slide timestamps map correctly to reference frames.

---

## Sprint 4 — Offline Alignment

### Goal

Prove the alignment method against recorded performances.

### Scope

* implement cosine distance;
* implement normalized Euclidean distance;
* implement baseline DTW;
* implement subsequence DTW;
* calculate normalized path cost;
* calculate confidence;
* compare live recording with reference profile;
* output timestamp estimates;
* create evaluation reports;
* create replay-test fixtures.

### Deliverables

* offline alignment tool;
* baseline alignment results;
* timestamp error report;
* repeatable replay tests;
* initial threshold recommendations.

### Demo

```text
Live 00:08.20 → Reference 00:07.94
Confidence: 0.89
Timestamp error: 0.26 seconds
```

### Acceptance criteria

* clean recordings align correctly;
* slower and faster performances remain trackable;
* silence produces low confidence;
* repeated runs produce consistent results;
* timestamp error is measurable;
* performance is documented.

### Decision gate

Do not move to live slide control unless offline alignment achieves acceptable accuracy on recorded samples.

Recommended initial target:

```text
Median timestamp error: less than 1 second
95th percentile error: less than 2.5 seconds
False large jumps: less than 2 per song
```

---

## Sprint 5 — Online Tracking and Slide Stability

### Goal

Convert offline alignment into a real-time tracker.

### Scope

* implement incremental online subsequence DTW;
* use bounded search windows;
* preserve previous and current DTW rows;
* support limited backward recovery;
* allow larger forward movement;
* add candidate ambiguity detection;
* add tracking state machine;
* implement lost-position recovery;
* smooth timestamp output;
* implement confidence hysteresis;
* implement consecutive-match validation;
* add slide-boundary stability;
* add duplicate suppression;
* add trigger cooldown;
* add 200 ms look-ahead.

### Tracking states

```text
UNINITIALIZED
    ↓
SEARCHING
    ↓
TRACKING
    ↓
UNCERTAIN
    ├── recover → TRACKING
    └── timeout → SEARCHING
```

### Deliverables

* online DTW tracker;
* position stabilizer;
* slide decision engine;
* console presentation adapter;
* performance benchmarks.

### Demo

```text
TRACKING
reference_time=12.42
confidence=0.88
candidate_slide=3
active_slide=3
latency_ms=84
```

### Acceptance criteria

* memory usage remains bounded;
* online update runs faster than feature arrival;
* low confidence freezes the current slide;
* one bad frame cannot change slides;
* duplicate commands are suppressed;
* tracker can recover after temporary loss.

---

## Sprint 6 — OSC Integration and End-to-End MVP

### Goal

Connect the full system to presentation software.

### Scope

* implement presentation gateway interface;
* implement OSC sender;
* create separate command queue;
* serialize JSON payload;
* handle UDP failures;
* add sender retry policy;
* add console fallback;
* add runtime health summary;
* add configuration CLI;
* add startup validation;
* integrate full processing pipeline;
* add manual override hook;
* add end-to-end tests.

### OSC payload

```json
{
  "slide": 2,
  "section": "Verse 1",
  "lyrics": "Amazing grace, how sweet the sound...",
  "reference_timestamp": 5.42,
  "confidence": 0.91
}
```

### Deliverables

* complete live runtime;
* OSC slide triggering;
* simulation demonstration;
* operator configuration;
* health monitoring;
* MVP deployment package.

### Demo

```text
AUDIO: OK
MODEL: OK
ALIGNMENT: TRACKING
POSITION: 00:12.42
CONFIDENCE: 0.88
SLIDE: 3
QUEUE: 1/16
LATENCY: 96 ms
```

### Acceptance criteria

* OSC messages reach the presentation client;
* network errors do not block alignment;
* application survives temporary presentation-client failure;
* low-confidence audio does not trigger slides;
* manual override remains possible;
* complete song can run from start to finish.

---

# 7. Pilot Phase

## Week 13 — Church Pilot

### Goal

Test the system in realistic worship conditions.

### Test scenarios

* lead vocal only;
* lead and backing vocals;
* full band;
* congregation singing;
* instrumental section;
* pastor speaking;
* verse skipped;
* chorus repeated;
* song restarted;
* tempo increased;
* tempo decreased;
* audio clipping;
* temporary audio loss;
* presentation client restart.

### Pilot data to collect

* audio recordings;
* estimated reference positions;
* confidence scores;
* slide transitions;
* manual corrections;
* false transitions;
* missed transitions;
* recovery time;
* end-to-end latency;
* CPU and memory usage.

### Pilot success criteria

* no application crashes;
* no uncontrolled slide oscillation;
* acceptable operator workload;
* manual override remains responsive;
* alignment recovers after temporary loss;
* most slide transitions occur within an acceptable window.

Recommended target:

```text
Correct slide shown within ±1.5 seconds: at least 90%
False slide transitions: fewer than 2 per song
Recovery after lost tracking: under 5 seconds
End-to-end latency: under 500 ms
```

---

# 8. Stabilization Phase

## Week 14 — Hardening and Release

### Scope

* analyze pilot recordings;
* tune thresholds;
* fix critical bugs;
* optimize slow paths;
* improve recovery logic;
* improve logs;
* improve configuration;
* document deployment;
* document operator workflow;
* document troubleshooting;
* prepare release candidate;
* conduct final acceptance test.

### Deliverables

* release candidate;
* deployment guide;
* operator guide;
* troubleshooting guide;
* architecture documentation;
* known limitations;
* pilot evaluation report;
* backlog for Phase 2.

---

# 9. Team Structure

## Minimum team

### Project Manager

Responsibilities:

* timeline;
* sprint planning;
* scope control;
* risk tracking;
* stakeholder communication;
* pilot coordination;
* release readiness.

### Senior Python/DSP Engineer

Responsibilities:

* audio pipeline;
* ONNX Runtime;
* DTW;
* feature processing;
* performance optimization;
* concurrency.

### Machine Learning Engineer

Responsibilities:

* model evaluation;
* feature quality;
* ONNX export validation;
* confidence calibration;
* replay dataset analysis.

### QA Engineer

Responsibilities:

* test planning;
* replay scenarios;
* regression testing;
* performance testing;
* pilot validation.

### Worship or Presentation Operator

Responsibilities:

* slide mapping;
* presentation workflow;
* real-world feedback;
* manual override requirements;
* pilot acceptance.

## Lean team option

For a small MVP team:

```text
1 Senior Python/ML Engineer
1 Part-time QA Engineer
1 Product Owner or Project Manager
1 Church Presentation Operator
```

---

# 10. Project Roles and Ownership

| Area               | Owner                           |
| ------------------ | ------------------------------- |
| Architecture       | Senior Engineer                 |
| Sprint planning    | Project Manager                 |
| Audio ingestion    | Python Engineer                 |
| ONNX integration   | ML/Python Engineer              |
| DTW alignment      | DSP Engineer                    |
| Slide logic        | Python Engineer                 |
| OSC integration    | Python Engineer                 |
| Test automation    | QA Engineer                     |
| Pilot coordination | Project Manager                 |
| Acceptance         | Product Owner / Church Operator |

---

# 11. Core Backlog

## Epic 1 — Audio Platform

* capture microphone audio;
* support mixer device selection;
* implement queue backpressure;
* detect silence and clipping;
* support recorded WAV input;
* support simulation mode.

## Epic 2 — Feature Extraction

* load ONNX model;
* inspect input shape;
* normalize audio;
* extract feature frames;
* validate outputs;
* measure latency;
* provide simulated features.

## Epic 3 — Song Profiles

* create profile format;
* extract reference features;
* map timestamps to slides;
* validate metadata;
* load profiles;
* version profiles.

## Epic 4 — Alignment

* implement distance functions;
* implement offline DTW;
* implement subsequence DTW;
* implement online DTW;
* calculate confidence;
* detect ambiguity;
* implement recovery.

## Epic 5 — Slide Control

* resolve timestamps;
* stabilize candidate positions;
* add look-ahead;
* add cooldown;
* add hysteresis;
* suppress duplicates;
* freeze on low confidence.

## Epic 6 — Presentation Integration

* create OSC adapter;
* serialize commands;
* isolate network sending;
* handle failures;
* support console fallback;
* add manual override integration.

## Epic 7 — Operations

* structured logging;
* metrics;
* health status;
* configuration;
* graceful shutdown;
* deployment scripts;
* troubleshooting documentation.

---

# 12. Non-Functional Requirements

## Performance

* audio callback must never block;
* processing must run faster than real time;
* memory usage must remain bounded;
* average DTW update should be below 5 ms;
* total slide-trigger latency should remain below 500 ms.

## Reliability

* one invalid chunk must not crash the system;
* OSC failure must not stop alignment;
* microphone loss must be reported clearly;
* low confidence must freeze slide control;
* application must shut down cleanly.

## Maintainability

* domain logic must remain independent of external libraries;
* interfaces must be explicit;
* test coverage must focus on alignment and slide safety;
* configuration must not be hardcoded across modules.

## Observability

Track:

* chunks received;
* chunks dropped;
* feature frames processed;
* invalid model outputs;
* low-confidence matches;
* accepted matches;
* slide triggers;
* OSC failures;
* average latency;
* maximum latency;
* queue high-water mark.

---

# 13. Risks

## Risk 1 — Poor audio quality

Impact: High

Mitigation:

* use a dedicated vocal-heavy mixer feed;
* reduce drums and instruments;
* monitor clipping and silence;
* record pilot sessions.

## Risk 2 — Speech model performs poorly on singing

Impact: High

Mitigation:

* evaluate several ONNX feature models;
* add melody and chroma features later;
* use replay recordings;
* avoid depending only on phoneme output.

## Risk 3 — Repeated choruses cause wrong position

Impact: High

Mitigation:

* use song-section constraints;
* limit backward movement;
* add section state machine;
* require multiple stable matches.

## Risk 4 — Different live arrangement from reference

Impact: High

Mitigation:

* allow controlled forward jumps;
* implement recovery mode;
* add multiple reference versions later;
* provide manual override.

## Risk 5 — Real-time performance issues

Impact: Medium

Mitigation:

* bounded queues;
* separate workers;
* NumPy optimization;
* avoid full-song DTW on every frame;
* benchmark from Sprint 4 onward.

## Risk 6 — False slide transitions

Impact: High

Mitigation:

* confidence threshold;
* hysteresis;
* cooldown;
* duplicate suppression;
* consecutive-match rule;
* freeze on uncertainty.

---

# 14. Success Metrics

## Technical metrics

* alignment median error below 1 second;
* alignment 95th percentile below 2.5 seconds;
* end-to-end latency below 500 ms;
* no memory growth during a full service;
* audio chunk drop rate below 1%;
* recovery time below 5 seconds.

## Product metrics

* at least 90% of lyric slides triggered correctly;
* fewer than 2 false transitions per song;
* operator manually corrects fewer than 3 times per song;
* application survives a full worship service;
* operator considers the system useful and predictable.

---

# 15. Release Milestones

## Milestone 1 — Audio Foundation

End of Sprint 2

Includes:

* live audio capture;
* simulation;
* ONNX feature extraction;
* metrics.

## Milestone 2 — Alignment Proof

End of Sprint 4

Includes:

* reference profile builder;
* offline alignment;
* evaluation report.

## Milestone 3 — Real-Time Tracker

End of Sprint 5

Includes:

* online sDTW;
* confidence;
* recovery;
* slide stability.

## Milestone 4 — MVP Release Candidate

End of Sprint 6

Includes:

* OSC integration;
* full runtime;
* health monitoring;
* end-to-end demonstration.

## Milestone 5 — Pilot Release

End of Week 14

Includes:

* pilot results;
* tuned configuration;
* release documentation;
* prioritized Phase 2 backlog.

---

# 16. Phase 2 Roadmap

After the MVP is proven, prioritize:

* automatic song recognition;
* multiple reference performances per song;
* chroma and pitch features;
* vocal source separation;
* repeated-section state graph;
* lyric-line probability tracking;
* adaptive confidence thresholds;
* operator feedback learning;
* automatic profile selection;
* browser-based monitoring dashboard;
* FreeShow and ProPresenter-specific integrations;
* CCLI or licensed lyrics integration.

---

# 17. First Sprint Starting Backlog

## Sprint 1 priority order

1. Create repository structure.
2. Configure `pyproject.toml`.
3. Add Ruff, MyPy, Pytest, and coverage.
4. Define domain models.
5. Define application ports.
6. Implement configuration.
7. Implement logging.
8. Implement lifecycle and signal handling.
9. Implement simulated audio source.
10. Implement sound-device source.
11. Implement bounded audio queue.
12. Implement audio metrics.
13. Add unit and integration tests.
14. Run a 30-minute stability test.
15. Demo live audio diagnostics.

## Sprint 1 output

At the end of Sprint 1, the team should have a stable foundation that captures audio safely and can be extended without architectural rework.
