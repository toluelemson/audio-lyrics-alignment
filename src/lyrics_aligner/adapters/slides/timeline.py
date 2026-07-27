"""Resolve slide commands from stable reference-timeline matches."""

from __future__ import annotations

from dataclasses import dataclass

from lyrics_aligner.domain.models import MatchResult, ReferenceProfile, SlideCommand


@dataclass(frozen=True, slots=True)
class TimelineSlideResolverConfig:
    lookahead_seconds: float = 0.2
    cooldown_seconds: float = 0.5
    consecutive_match_count: int = 2
    max_emit_lag_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.lookahead_seconds < 0:
            raise ValueError("lookahead_seconds must be non-negative")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if self.consecutive_match_count <= 0:
            raise ValueError("consecutive_match_count must be greater than zero")
        if self.max_emit_lag_seconds < 0:
            raise ValueError("max_emit_lag_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class SlideGroup:
    slide_number: int
    section: str
    lyrics: str
    reference_timestamp: float


class TimelineSlideResolver:
    """Emit each slide once when a stable match reaches its cue timestamp."""

    def __init__(
        self,
        profile: ReferenceProfile,
        config: TimelineSlideResolverConfig | None = None,
    ) -> None:
        self._slide_groups = _build_slide_groups(profile.slide_cues)
        self._config = config or TimelineSlideResolverConfig()
        self._next_index = 0
        self._candidate_count = 0
        self._last_candidate_slide_number: int | None = None
        self._last_emitted_slide_number: int | None = None
        self._last_emitted_reference_timestamp: float | None = None

    def resolve(self, match: MatchResult) -> SlideCommand | None:
        if not match.valid or self._next_index >= len(self._slide_groups):
            self._reset_candidate()
            return None

        self._skip_stale_cues(match.reference_timestamp)
        if self._next_index >= len(self._slide_groups):
            self._reset_candidate()
            return None

        cue = self._slide_groups[self._next_index]
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

    def seek_to_slide(self, slide_number: int) -> None:
        target_index = None
        for index, cue in enumerate(self._slide_groups):
            if cue.slide_number == slide_number:
                target_index = index
                break
        if target_index is None:
            raise ValueError(f"Unknown slide number: {slide_number}")

        cue = self._slide_groups[target_index]
        self._next_index = target_index + 1
        self._last_emitted_slide_number = cue.slide_number
        self._last_emitted_reference_timestamp = cue.reference_timestamp
        self._reset_candidate()

    def slide_command_for_slide(
        self,
        slide_number: int,
        *,
        confidence: float = 1.0,
    ) -> SlideCommand:
        for cue in self._slide_groups:
            if cue.slide_number == slide_number:
                return SlideCommand(
                    slide_number=cue.slide_number,
                    section=cue.section,
                    lyrics=cue.lyrics,
                    reference_timestamp=cue.reference_timestamp,
                    confidence=confidence,
                )
        raise ValueError(f"Unknown slide number: {slide_number}")

    def slide_commands(self) -> tuple[SlideCommand, ...]:
        return tuple(
            SlideCommand(
                slide_number=cue.slide_number,
                section=cue.section,
                lyrics=cue.lyrics,
                reference_timestamp=cue.reference_timestamp,
                confidence=1.0,
            )
            for cue in self._slide_groups
        )

    def active_slide_command_for_timestamp(
        self,
        reference_timestamp: float,
        *,
        confidence: float = 1.0,
    ) -> SlideCommand | None:
        if not self._slide_groups:
            return None
        target = reference_timestamp + self._config.lookahead_seconds
        active_cue = self._slide_groups[0]
        for cue in self._slide_groups:
            if cue.reference_timestamp > target:
                break
            active_cue = cue
        return SlideCommand(
            slide_number=active_cue.slide_number,
            section=active_cue.section,
            lyrics=active_cue.lyrics,
            reference_timestamp=active_cue.reference_timestamp,
            confidence=confidence,
        )

    def _skip_stale_cues(self, reference_timestamp: float) -> None:
        while self._next_index < len(self._slide_groups):
            cue = self._slide_groups[self._next_index]
            if reference_timestamp - cue.reference_timestamp <= self._config.max_emit_lag_seconds:
                return
            self._next_index += 1
            self._last_emitted_slide_number = cue.slide_number
            self._last_emitted_reference_timestamp = cue.reference_timestamp
        self._reset_candidate()

    def _reset_candidate(self) -> None:
        self._candidate_count = 0
        self._last_candidate_slide_number = None


def _build_slide_groups(slide_cues: tuple) -> tuple[SlideGroup, ...]:
    sorted_cues = tuple(sorted(slide_cues, key=lambda cue: cue.reference_timestamp))
    if not sorted_cues:
        return ()

    groups: list[SlideGroup] = []
    current_slide_number = sorted_cues[0].slide_number
    current_section = sorted_cues[0].section
    current_timestamp = sorted_cues[0].reference_timestamp
    current_lines: list[str] = []

    def flush_group() -> None:
        if not current_lines:
            return
        groups.append(
            SlideGroup(
                slide_number=current_slide_number,
                section=current_section,
                lyrics="\n".join(current_lines),
                reference_timestamp=current_timestamp,
            )
        )

    for cue in sorted_cues:
        if cue.slide_number != current_slide_number:
            flush_group()
            current_slide_number = cue.slide_number
            current_section = cue.section
            current_timestamp = cue.reference_timestamp
            current_lines = []
        current_lines.append(cue.lyrics)

    flush_group()
    return tuple(groups)
