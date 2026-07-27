# Contributing

## Definition Of Done

A change is not done when only the code works. It is done when:

- behavior is implemented
- tests are updated where applicable
- documentation is updated where applicable
- quality checks pass

## Documentation Rules

Update documentation in the same PR as the code change.

Use this guide:

- `README.md`
  - update when run commands, setup steps, or testing workflow changes
- `docs/adr/`
  - update or add an ADR when an architectural decision changes
- `docs/runtime-flow-walkthrough.html`
  - update when ingestion flow, adapter behavior, runtime queueing, metrics, or threading changes
- `docs/runtime-flow-walkthrough.pdf`
  - regenerate after updating the walkthrough HTML
- code comments and docstrings
  - update when local implementation intent changes

## Regenerating The Walkthrough PDF

The walkthrough PDF is generated from the HTML source:

```bash
python3 scripts/generate_runtime_walkthrough_pdf.py
```

If the HTML changes, the PDF should change in the same PR.

## Quality Checks

Use the existing local environments:

```bash
.venv/bin/pytest
.venv/bin/ruff check .
/private/tmp/audio-lyrics-py311-test/bin/mypy src
```

## PR Expectations

Every PR should answer:

- what changed?
- how was it tested?
- what documentation changed?
- if docs did not change, why not?
