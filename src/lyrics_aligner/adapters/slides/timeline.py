"""Resolve slide commands from stable reference-timeline matches."""

from __future__ import annotations

from lyrics_aligner.domain.models import MatchResult, ReferenceProfile, SlideCommand


class TimelineSlideResolver:
    """Emit each slide once when a stable match reaches its cue timestamp."""

    def __init__(self, profile: ReferenceProfile) -> None:
        self._slide_cues = tuple(
            sorted(profile.slide_cues, key=lambda cue: cue.reference_timestamp)
        )
        self._next_index = 0

    def resolve(self, match: MatchResult) -> SlideCommand | None:
        if not match.valid or self._next_index >= len(self._slide_cues):
            return None

        cue = self._slide_cues[self._next_index]
        if match.reference_timestamp < cue.reference_timestamp:
            return None

        self._next_index += 1
        return SlideCommand(
            slide_number=cue.slide_number,
            section=cue.section,
            lyrics=cue.lyrics,
            reference_timestamp=cue.reference_timestamp,
            confidence=match.confidence,
        )
