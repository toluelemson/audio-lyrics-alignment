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

## Current scope

Sprint 1 establishes the architecture, configuration, domain contracts, CI, and test foundation. Live audio and ONNX adapters will be implemented in follow-up pull requests.
