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

## Current scope

Sprint 1 establishes the architecture, configuration, domain contracts, CI, and test foundation. Live audio and ONNX adapters will be implemented in follow-up pull requests.
