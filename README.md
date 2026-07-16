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

Relevant runtime selectors:

- `--audio-source simulated|microphone`
- `--feature-extractor simulated|onnx`
- `--feature-model-path /absolute/path/to/model.onnx` when using `onnx`
- `--reference-profile-path /absolute/path/to/reference-profile`

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
.venv/bin/python -m lyrics_aligner.main --audio-source simulated
```

Prepared reference profile directory contents:

- `profile.json`: profile name, frame duration, and optional timestamps
- `reference_features.npy`: 2D NumPy array of reference feature vectors
- `metadata.json`: optional extra labels for future matching and presentation steps

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

The current code also includes the first Sprint 2 vertical slice:

- deterministic feature extraction from `AudioChunk` to `FeatureFrame`
- ONNX-backed feature extraction behind the same `FeatureExtractor` port
- filesystem-backed loading of prepared reference profiles
- runtime accounting for `feature_frames_processed`
- invalid feature frame detection via `invalid_inference_outputs`

Alignment, live-to-reference comparison, and slide decisions are still follow-up work.
