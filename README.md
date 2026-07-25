# Audio-to-Lyrics Alignment Engine

A Python MVP that listens to one live song, estimates the current song position,
and triggers the matching lyric slide.

## Architecture

The project uses a modular monolith with hexagonal boundaries:

- `domain`: pure models and alignment rules
- `application`: orchestration, lifecycle, and metrics
- `ports`: interfaces for external capabilities
- `adapters`: microphone, simulation, ONNX, profile storage, and presentation integrations

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap and sprint plan.
For the next production direction, see
[docs/adr/003-layered-show-control-architecture.md](docs/adr/003-layered-show-control-architecture.md).

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

The runtime is intentionally streamlined to one core path:

```text
WAV or microphone input
-> ONNX feature extraction
-> reference matching with tracking
-> timeline slide resolution
-> logging or OSC output
```

Required inputs:

- `--reference-profile-path /absolute/path/to/reference-profile`
- `--feature-model-path /absolute/path/to/model.onnx`

Supported audio inputs:

- `--audio-source wav --audio-file-path /absolute/path/to/file.wav`
- `--audio-source microphone --input-device "Built-in Microphone"`

Supported outputs:

- `--presentation-mode logging`
- `--presentation-mode osc`
- `--presentation-mode none`

Microphone input:

```bash
.venv/bin/python -m lyrics_aligner.main --audio-source microphone --input-device "Built-in Microphone"
```

WAV replay input:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source wav \
  --audio-file-path demo_assets/onnx_demo_song.wav \
  --feature-model-path models/wav2vec2-base.onnx \
  --reference-profile-path profiles/demo-wav2vec2 \
  --presentation-mode logging
```

Environment variables are also supported:

```bash
export LYRICS_ALIGNER_AUDIO_SOURCE=microphone
export LYRICS_ALIGNER_INPUT_DEVICE="Built-in Microphone"
.venv/bin/python -m lyrics_aligner.main
```

WAV replay can also be configured with environment variables:

```bash
export LYRICS_ALIGNER_AUDIO_SOURCE=wav
export LYRICS_ALIGNER_AUDIO_FILE_PATH=/absolute/path/to/file.wav
.venv/bin/python -m lyrics_aligner.main
```

The same core runtime can be configured with environment variables:

```bash
export LYRICS_ALIGNER_FEATURE_MODEL_PATH=/absolute/path/to/model.onnx
export LYRICS_ALIGNER_REFERENCE_PROFILE_PATH=/absolute/path/to/reference-profile
export LYRICS_ALIGNER_MATCH_CONFIDENCE_THRESHOLD=0.6
export LYRICS_ALIGNER_MATCH_CONFIRMATION_COUNT=2
.venv/bin/python -m lyrics_aligner.main --audio-source wav --audio-file-path /absolute/path/to/file.wav
```

Prepared reference profile directory contents:

- `profile.json`: profile name, frame duration, and optional timestamps
- `reference_features.npy`: 2D NumPy array of reference feature vectors
- `metadata.json`: optional extra labels for future matching and presentation steps
- optional `slides` list inside `profile.json`: slide cues with `slide_number`, `section`, `lyrics`, and `reference_timestamp`

## Sprint 3 Reference Builder

Build a reusable reference profile from a WAV file and a slide cue JSON file
captured from user slide-change clicks:

1. Generate or prepare the slide order file with `slide_number`, `section`, and
   `lyrics`.
2. Play the reference song once and capture user click timings.
3. Build the reference profile from the audio plus captured click JSON.

Generate a slide-order scaffold from plain lyrics:

```bash
.venv/bin/python tools/generate_slide_cues.py \
  --lyrics amazing-grace.txt \
  --output amazing-grace-slides.json
```

Capture click timings in the terminal while the song plays:

```bash
.venv/bin/python tools/capture_slide_clicks.py \
  --slides amazing-grace-slides.json \
  --output amazing-grace-clicks.json
```

Then build the profile:

```bash
.venv/bin/python tools/build_reference.py \
  --audio amazing-grace.wav \
  --slides amazing-grace-clicks.json \
  --output profiles/amazing-grace
```

Optional selectors:

- `--profile-name amazing-grace`
- `--feature-extractor simulated|onnx`
- `--feature-model-path /absolute/path/to/model.onnx` when using `onnx`
- `--block-size 4096`
- `--audio-role mixed|vocals|instrumental|other`
- `--companion-audio /absolute/path/to/original-mix.wav`

The builder:

- reads WAV input
- converts stereo to mono when needed
- resamples to `16 kHz`
- runs offline feature extraction
- uses recorded click times as the slide timings
- writes `profile.json`, `reference_features.npy`, and `metadata.json`

The slide cue input should contain slide content plus `click_timestamp` for
each slide. Older `reference_timestamp` files are still accepted for backward
compatibility, but new reference prep should come from recorded user clicks.

For live tracking, the most reliable setup is a vocals-only reference profile.
If you can export or obtain a vocal stem, build the profile from that stem and
keep the full mix only as companion metadata:

```bash
.venv/bin/python tools/build_vocal_reference.py \
  --vocals demo_assets/amazing-grace-vocals.wav \
  --mix demo_assets/amazing-grace.wav \
  --slides demo_assets/amazing-grace-clicks.json \
  --output profiles/amazing-grace-vocals \
  --feature-extractor onnx \
  --feature-model-path models/wav2vec2-base.onnx
```

That helper stores the profile as `reference_audio_role=vocals`, which makes it
clear that live matching should be judged against sung content rather than the
full arrangement.

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

## Sprint 6 OSC Integration And End-To-End MVP

The runtime now supports operator-selectable presentation output:

- `logging` for local visibility
- `terminal` for a live in-terminal verse view with the current section highlighted
- `osc` for real UDP OSC delivery
- `both` for console plus OSC
- `none` or `--manual-override` to keep tracking active without sending slides

The presentation path now adds:

- a non-blocking command queue separate from audio ingestion
- JSON payload serialization for outgoing OSC messages
- retry with backoff on temporary UDP send failures
- startup validation for OSC configuration
- runtime health summaries including audio, alignment, and presentation status

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

Manual live BlackHole test with terminal verse view:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source microphone \
  --input-device "BlackHole 2ch" \
  --feature-extractor onnx \
  --feature-model-path models/wav2vec2-base.onnx \
  --reference-profile-path profiles/amazing-grace-vocals \
  --presentation-mode terminal
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

Manual OSC delivery test:

```bash
.venv/bin/python - <<'PY'
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 7000))
print("listening on udp/7000")
data, address = sock.recvfrom(4096)
print("from", address)
print(data)
PY
```

In a second shell:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor simulated \
  --reference-profile-path /tmp/reference-profile \
  --presentation-mode osc \
  --osc-host 127.0.0.1 \
  --osc-port 7000 \
  --osc-path /presentation/trigger-slide \
  --simulation-duration-seconds 5.0
```

Expected behavior:

- the UDP listener receives OSC packets
- runtime continues tracking even if the receiver is restarted
- `osc_send_failures` stays at `0` when the receiver is available
- the final logs include the health summary line with `PRESENTATION: OSC`

Manual override test:

```bash
.venv/bin/python -m lyrics_aligner.main \
  --audio-source simulated \
  --feature-extractor simulated \
  --reference-profile-path /tmp/reference-profile \
  --manual-override \
  --simulation-duration-seconds 5.0
```

Expected behavior:

- matching and tracking still run normally
- slide commands are suppressed until the operator switches back to `auto`
- the final health summary shows `PRESENTATION: MANUAL_OVERRIDE` while override remains active
- type `manual`, `auto`, `toggle`, `status`, or `help` in the terminal to control handoff at runtime

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

ONNX-backed demo profile test:

```bash
.venv/bin/python tools/generate_demo_reference_assets.py --output-dir demo_assets
.venv/bin/python tools/build_reference.py \
  --audio demo_assets/onnx_demo_song.wav \
  --slides demo_assets/onnx_demo_slides.json \
  --output profiles/demo-wav2vec2 \
  --feature-extractor onnx \
  --feature-model-path models/wav2vec2-base.onnx
```

Expected behavior:

- the generator creates `demo_assets/onnx_demo_song.wav`
- the generator creates `demo_assets/onnx_demo_slides.json`
- the builder writes `profiles/demo-wav2vec2/profile.json`
- `reference_features.npy` has a real ONNX-derived feature matrix
- the printed summary reports a non-zero `frames` count and feature size `768`

Plain lyrics to cue-template test:

```bash
python tools/generate_slide_cues.py \
  --lyrics amazing-grace.txt \
  --output amazing-grace-slides.json \
  --timestamp-step 20
```

Expected behavior:

- the command reads section headings like `**Verse 1**`
- the output JSON contains slide entries with placeholder ascending timestamps
- users only need to adjust timestamps instead of hand-writing the full JSON structure

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

The current code now spans the planned work through Sprint 6:

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
- logging, OSC, or combined presentation gateways
- non-blocking presentation command dispatch queue
- OSC JSON payload delivery with retry/backoff
- operator manual override and presentation mode selection
- runtime health summary output
- runtime accounting for `feature_frames_processed`
- invalid feature frame detection via `invalid_inference_outputs`
- match accounting via `accepted_matches` and `low_confidence_matches`
- slide trigger accounting via `slide_triggers_sent` and `osc_send_failures`
