"""Resolve slide commands from stable reference-timeline matches."""

from __future__ import annotations

from dataclasses import dataclass

from lyrics_aligner.domain.models import MatchResult, ReferenceProfile, SlideCommand


@dataclass(frozen=True, slots=True)
class TimelineSlideResolverConfig:
    lookahead_seconds: float = 0.2
    cooldown_seconds: float = 0.5
    consecutive_match_count: int = 2

    def __post_init__(self) -> None:
        if self.lookahead_seconds < 0:
            raise ValueError("lookahead_seconds must be non-negative")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if self.consecutive_match_count <= 0:
            raise ValueError("consecutive_match_count must be greater than zero")


class TimelineSlideResolver:
    """Emit each slide once when a stable match reaches its cue timestamp."""

    def __init__(
        self,
        profile: ReferenceProfile,
        config: TimelineSlideResolverConfig | None = None,
    ) -> None:
        self._slide_cues = tuple(
            sorted(profile.slide_cues, key=lambda cue: cue.reference_timestamp)
        )
        self._config = config or TimelineSlideResolverConfig()
        self._next_index = 0
        self._candidate_count = 0
        self._last_candidate_slide_number: int | None = None
        self._last_emitted_slide_number: int | None = None
        self._last_emitted_reference_timestamp: float | None = None

    def resolve(self, match: MatchResult) -> SlideCommand | None:
        if not match.valid or self._next_index >= len(self._slide_cues):
            self._reset_candidate()
            return None

        cue = self._slide_cues[self._next_index]
        if match.reference_timestamp + self._config.lookahead_seconds < cue.reference_timestamp:
            self._reset_candidate()
            return None

        if (
            self._last_emitted_reference_timestamp is not None
            and match.reference_timestamp - self._last_emitted_reference_timestamp
            < self._config.cooldown_seconds
        ):
            self._reset_candidate()
            return None

        if cue.slide_number == self._last_candidate_slide_number:
            self._candidate_count += 1
        else:
            self._last_candidate_slide_number = cue.slide_number
            self._candidate_count = 1

        if self._candidate_count < self._config.consecutive_match_count:
            return None

        if cue.slide_number == self._last_emitted_slide_number:
            self._reset_candidate()
            return None

        self._next_index += 1
        self._last_emitted_slide_number = cue.slide_number
        self._last_emitted_reference_timestamp = cue.reference_timestamp
        self._reset_candidate()
        return SlideCommand(
            slide_number=cue.slide_number,
            section=cue.section,
            lyrics=cue.lyrics,
            reference_timestamp=cue.reference_timestamp,
            confidence=match.confidence,
        )

    def _reset_candidate(self) -> None:
        self._candidate_count = 0
        self._last_candidate_slide_number = None
