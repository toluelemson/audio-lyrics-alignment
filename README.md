# Audio-to-Lyrics Alignment Engine

A modular Python MVP for aligning live worship audio with a known reference performance and triggering lyric slides through OSC.

## Architecture

The project uses a modular monolith with hexagonal boundaries:

- `domain`: pure models and alignment rules
- `application`: orchestration, lifecycle, and metrics
- `ports`: interfaces for external capabilities
- `adapters`: microphone, simulation, ONNX, profile storage, and presentation integrations

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap and sprint plan.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
mypy src
```

## Running

Simulated input:

```bash
.venv/bin/python -m lyrics_aligner.main
```

Simulated input with explicit simulated feature extraction:

```bash
.venv/bin/python -m lyrics_aligner.main --audio-source simulated --feature-extractor simulated
```

ONNX feature extraction:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor onnx \
  --feature-model-path /absolute/path/to/model.onnx
```

With a prepared reference profile directory:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor onnx \
  --feature-model-path /absolute/path/to/model.onnx \
  --reference-profile-path /absolute/path/to/reference-profile
```

With baseline matching against the loaded reference profile:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor simulated \
  --reference-profile-path /absolute/path/to/reference-profile \
  --match-confidence-threshold 0.6
```

With simple match stabilization:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor simulated \
  --reference-profile-path /absolute/path/to/reference-profile \
  --match-confidence-threshold 0.6 \
  --match-max-forward-jump-frames 4 \
  --match-large-jump-threshold-frames 2 \
  --match-confirmation-count 2
```

Relevant runtime selectors:

- `--audio-source simulated|microphone`
- `--feature-extractor simulated|onnx`
- `--feature-model-path /absolute/path/to/model.onnx` when using `onnx`
- `--reference-profile-path /absolute/path/to/reference-profile`
- `--match-confidence-threshold 0.0-1.0`
- `--match-max-forward-jump-frames`
- `--match-large-jump-threshold-frames`
- `--match-confirmation-count`

Microphone input:

```bash
.venv/bin/python -m lyrics_aligner.main --audio-source microphone --input-device "Built-in Microphone"
```

Environment variables are also supported:

```bash
export LYRICS_ALIGNER_AUDIO_SOURCE=microphone
export LYRICS_ALIGNER_INPUT_DEVICE="Built-in Microphone"
.venv/bin/python -m lyrics_aligner.main
```

Feature extraction can also be configured with environment variables:

```bash
export LYRICS_ALIGNER_FEATURE_EXTRACTOR=onnx
export LYRICS_ALIGNER_FEATURE_MODEL_PATH=/absolute/path/to/model.onnx
export LYRICS_ALIGNER_REFERENCE_PROFILE_PATH=/absolute/path/to/reference-profile
export LYRICS_ALIGNER_MATCH_CONFIDENCE_THRESHOLD=0.6
export LYRICS_ALIGNER_MATCH_MAX_FORWARD_JUMP_FRAMES=4
export LYRICS_ALIGNER_MATCH_LARGE_JUMP_THRESHOLD_FRAMES=2
export LYRICS_ALIGNER_MATCH_CONFIRMATION_COUNT=2
.venv/bin/python -m lyrics_aligner.main --audio-source simulated
```

Prepared reference profile directory contents:

- `profile.json`: profile name, frame duration, and optional timestamps
- `reference_features.npy`: 2D NumPy array of reference feature vectors
- `metadata.json`: optional extra labels for future matching and presentation steps
- optional `slides` list inside `profile.json`: slide cues with `slide_number`, `section`, `lyrics`, and `reference_timestamp`

## Sprint 3 Reference Builder

Build a reusable reference profile from a WAV file and slide cue JSON:

```bash
python tools/build_reference.py \
  --audio amazing-grace.wav \
  --slides amazing-grace-slides.json \
  --output profiles/amazing-grace
```

Optional selectors:

- `--profile-name amazing-grace`
- `--feature-extractor simulated|onnx`
- `--feature-model-path /absolute/path/to/model.onnx` when using `onnx`
- `--block-size 4096`

The builder:

- reads WAV input
- converts stereo to mono when needed
- resamples to `16 kHz`
- runs offline feature extraction
- validates slide timestamps
- writes `profile.json`, `reference_features.npy`, and `metadata.json`

## Sprint 4 Offline Alignment

Compare a recorded performance against a saved reference profile:

```bash
python tools/offline_align.py \
  --live-audio amazing-grace-live.wav \
  --reference-profile profiles/amazing-grace
```

Optional selectors:

- `--feature-extractor simulated|onnx`
- `--feature-model-path /absolute/path/to/model.onnx` when using `onnx`
- `--method baseline|subsequence`
- `--metric cosine|euclidean`
- `--expected-alignments expected-timestamps.json`

The offline alignment tool:

- extracts feature frames from a recorded WAV file
- compares them against the saved reference profile
- supports baseline DTW and subsequence DTW
- calculates normalized path cost and confidence
- prints timestamp estimates
- optionally reports alignment error against expected timestamp fixtures

## Sprint 5 Online Tracking And Slide Stability

The live runtime now adds a tracking state machine on top of frame-level
matches:

- `SEARCHING` waits for consecutive plausible matches before trusting position
- `TRACKING` accepts stable forward progress and keeps reporting position
- `UNCERTAIN` freezes slide output when confidence drops or jumps look wrong
- recovery requires consecutive good matches before tracking resumes

The slide resolver now adds:

- `200 ms` cue look-ahead
- consecutive-match validation at slide boundaries
- duplicate suppression
- trigger cooldown between slide commands

## Testing The Current Sprint

Quality checks:

```bash
.venv/bin/pytest
.venv/bin/ruff check .
/private/tmp/audio-lyrics-py311-test/bin/mypy src
```

Regenerate and verify the walkthrough PDF:

```bash
python3 scripts/generate_runtime_walkthrough_pdf.py
git diff --exit-code -- docs/runtime-flow-walkthrough.pdf
```

Manual simulated runtime test:

```bash
.venv/bin/python -m lyrics_aligner.main --audio-source simulated --simulation-duration-seconds 5.0
```

Expected behavior:

- `chunks_received` increases steadily
- `chunks_dropped=0`
- `queue_high_water_mark` stays low
- `rms` stays around `0.35` for the default sine wave
- `peak` stays around `0.50`
- `silent_chunks=0`
- `clipped_chunks=0`
- `feature_frames_processed` increases with processed chunks

Manual microphone runtime test:

```bash
.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
.venv/bin/python -m lyrics_aligner.main --audio-source microphone --input-device "Built-in Microphone"
```

Expected behavior:

- `chunks_received` increases continuously
- `chunks_dropped=0`
- `silent_chunks` rises in a quiet room
- `rms` and `peak` rise when speaking
- `clipped_chunks` stays low unless the input is too hot

Manual ONNX extractor wiring test:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor onnx \
  --feature-model-path /absolute/path/to/model.onnx
```

Expected behavior:

- the process starts without configuration errors
- `feature_frames_processed` rises when the model returns valid frames
- `invalid_inference_outputs` stays at `0` for a healthy model output

Manual reference profile loading test:

```bash
mkdir -p /tmp/reference-profile
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

base = Path("/tmp/reference-profile")
np.save(base / "reference_features.npy", np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
(base / "profile.json").write_text(
    json.dumps(
        {
            "name": "demo-song",
            "frame_duration_seconds": 0.25,
            "timestamps_seconds": [0.0, 0.25],
        }
    ),
    encoding="utf-8",
)
(base / "metadata.json").write_text(json.dumps({"song": "Demo Song"}), encoding="utf-8")
PY
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor simulated \
  --reference-profile-path /tmp/reference-profile \
  --simulation-duration-seconds 1.0
```

Expected behavior:

- startup logs `Loaded reference profile`
- the log reports the profile name and frame count
- runtime then continues with normal audio ingestion and feature extraction

Manual baseline matching test:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor simulated \
  --reference-profile-path /tmp/reference-profile \
  --match-confidence-threshold 0.2 \
  --simulation-duration-seconds 1.0
```

Expected behavior:

- diagnostics include `accepted_matches` and `low_confidence_matches`
- `last_reference_timestamp` changes from `none` once matching begins
- the final summary includes match counts

Expected tracking behavior:

- diagnostics include `tracking_state`
- startup begins in `SEARCHING`
- after repeated good matches the runtime enters `TRACKING`
- a weak or implausible match moves the runtime to `UNCERTAIN`
- repeated recovery matches return the runtime to `TRACKING`
- low confidence freezes slide output instead of advancing slides

Manual slide-trigger test:

```bash
mkdir -p /tmp/reference-profile
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

base = Path("/tmp/reference-profile")
np.save(base / "reference_features.npy", np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
(base / "profile.json").write_text(
    json.dumps(
        {
            "name": "demo-song",
            "frame_duration_seconds": 0.25,
            "timestamps_seconds": [0.0, 0.25],
            "slides": [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                    "reference_timestamp": 0.0,
                }
            ],
        }
    ),
    encoding="utf-8",
)
PY
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor simulated \
  --reference-profile-path /tmp/reference-profile \
  --simulation-duration-seconds 1.0
```

Expected behavior:

- logs include `Triggered slide`
- diagnostics include `slide_triggers_sent`
- final summary reports slide trigger counts
- slide output does not duplicate on one noisy frame
- cue firing can happen slightly before the exact cue because of the `200 ms` look-ahead
- rapid back-to-back cue emissions are limited by cooldown

Manual Sprint 3 builder test:

```bash
python tools/build_reference.py \
  --audio amazing-grace.wav \
  --slides amazing-grace-slides.json \
  --output profiles/amazing-grace
```

Expected behavior:

- the command prints a JSON summary
- `profiles/amazing-grace/profile.json` is created
- `profiles/amazing-grace/reference_features.npy` is created
- `profiles/amazing-grace/metadata.json` is created
- rerunning the command with the same inputs produces the same feature shape

Manual Sprint 4 offline alignment test:

```bash
python tools/offline_align.py \
  --live-audio amazing-grace-live.wav \
  --reference-profile profiles/amazing-grace \
  --method subsequence \
  --metric cosine
```

Expected behavior:

- the command prints a `Live mm:ss.xx -> Reference mm:ss.xx` estimate
- it prints a confidence value
- it prints a JSON summary with normalized path cost
- repeated runs on the same inputs produce the same result

## Documentation

Documentation is expected to move with the code.

- update `README.md` when setup, run, or test commands change
- update `docs/adr/` when architecture decisions change
- update `docs/runtime-flow-walkthrough.html` when runtime flow changes
- regenerate `docs/runtime-flow-walkthrough.pdf` after changing the walkthrough HTML

```bash
python3 scripts/generate_runtime_walkthrough_pdf.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repo's documentation and PR expectations.

## Current scope

Sprint 1 established the architecture, configuration, runtime ingestion, diagnostics, CI, and test foundation.

The current code now spans the planned work through Sprint 5:

- deterministic feature extraction from `AudioChunk` to `FeatureFrame`
- ONNX-backed feature extraction behind the same `FeatureExtractor` port
- filesystem-backed loading of prepared reference profiles
- offline reference profile builder CLI for Sprint 3
- offline alignment tool with cosine distance, normalized Euclidean distance, baseline DTW, and subsequence DTW
- nearest-neighbor matching from live feature frames to reference frames
- online tracking state transitions across `SEARCHING`, `TRACKING`, and `UNCERTAIN`
- recovery hysteresis after low-confidence or implausible jumps
- slide resolution from stable matches to `SlideCommand`
- slide-boundary confirmation, duplicate suppression, cue look-ahead, and cooldown
- logging presentation gateway for local slide-trigger visibility
- runtime accounting for `feature_frames_processed`
- invalid feature frame detection via `invalid_inference_outputs`
- match accounting via `accepted_matches` and `low_confidence_matches`
- slide trigger accounting via `slide_triggers_sent` and `osc_send_failures`

Real OSC delivery remains Sprint 6 follow-up work.
