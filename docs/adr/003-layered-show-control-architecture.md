# ADR-003: Use layered show control instead of a single inference path

## Status

Accepted

## Decision

Drive lyric slides through a layered control model:

1. planned cue timeline as the primary path
2. operator override as the safety path
3. live validation and recovery as the adaptive path

The runtime must not depend on a single inference strategy such as reference-audio
matching alone. Automatic signals should inform slide decisions, but deterministic
cue flow and operator control remain first-class parts of the architecture.

## Rationale

Pure reference matching proved too sensitive to routing quality, accompaniment
content, silence, and live signal instability. Pure ASR would reduce some of those
 problems, but it would still be probabilistic and brittle under singing,
 overlap, and noisy room conditions.

Cue-based deterministic control is the most reliable core mechanism, but it fails
 when the worship leader changes order, repeats a section, pauses, or moves into
 spontaneous worship. A layered model addresses those limits by combining:

- a repeatable planned arrangement
- explicit operator intervention
- live evidence for validation or recovery

This keeps the happy path deterministic while still allowing the system to adapt
when the room diverges from the plan.

## Architecture

### 1. Planned cue timeline

The system stores a song arrangement as ordered sections rather than only as raw
slide timestamps.

Recommended song structure model:

- song id
- arrangement id
- ordered sections such as `verse_1`, `chorus`, `bridge`, `tag`
- per-section slide sequence
- optional estimated durations or cue markers
- allowed transitions such as `chorus -> chorus`, `bridge -> tag`

This becomes the primary source of slide progression.

### 2. Operator override

The operator can:

- hold auto-advance
- resume auto-advance
- jump to the next section
- jump to the previous section
- jump directly to a named section

This path must remain available even when automatic inference is uncertain.

### 3. Live validation and recovery

Live signals do not directly control the slide deck by default. They validate or
repair the planned cue flow.

Supported live signals can include:

- reference-audio matching
- live ASR plus lyric matching
- future MIDI, click, or marker input

The runtime should use them to:

- confirm that the current section still fits the live audio
- detect when the planned arrangement is no longer plausible
- suggest a likely recovery section
- pause auto-advance when confidence collapses

## Repo mapping

This design fits the current codebase with incremental changes.

Current pieces that already exist:

- `src/lyrics_aligner/application/runtime.py`
  - owns the real-time loop and status reporting
- `src/lyrics_aligner/adapters/slides/timeline.py`
  - already resolves slide cues from a timeline
- `src/lyrics_aligner/adapters/control/terminal_manual_override.py`
  - already exposes manual/auto control
- `src/lyrics_aligner/adapters/presentation/*`
  - already separates presentation output from decision logic

Recommended new components:

- `src/lyrics_aligner/domain/show_plan.py`
  - arrangement sections, transitions, and cue-plan models
- `src/lyrics_aligner/ports/plan_validator.py`
  - protocol for live validation signals
- `src/lyrics_aligner/application/show_controller.py`
  - central decision-maker that combines plan state, operator state, and
    validation state
- `src/lyrics_aligner/adapters/validation/reference_audio.py`
  - wraps the current matcher as one validator input
- `src/lyrics_aligner/adapters/validation/asr_lyrics.py`
  - future transcript-to-lyrics validator

## Runtime flow

Planned runtime behavior:

1. load the song arrangement and slide plan
2. initialize the show controller at section 1
3. accept operator commands at all times
4. advance slides from the plan when the current section remains valid
5. if validation confidence drops, enter recovery mode
6. in recovery mode, hold the current slide, suggest a likely section, or wait
   for operator confirmation

## Consequences

Benefits:

- deterministic happy path
- lower dependence on perfect audio routing
- explicit handling of repeats, skips, and spontaneous sections
- easier production debugging because plan state is inspectable

Costs:

- requires arrangement authoring
- adds a show-controller layer beyond raw matching
- requires clear operator UX

## Implementation order

1. represent songs as section-aware plans
2. insert a show-controller layer between matching and slide emission
3. route existing manual override through the show controller
4. treat the current audio matcher as a validator, not the sole authority
5. add transcript-based validation later if needed
