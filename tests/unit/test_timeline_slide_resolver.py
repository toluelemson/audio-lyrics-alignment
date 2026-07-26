import pytest

from lyrics_aligner.adapters.slides import (
    TimelineSlideResolver,
    TimelineSlideResolverConfig,
)
from lyrics_aligner.domain.models import MatchResult, ReferenceProfile, SlideCue


def test_resolver_emits_slide_once_when_timestamp_is_reached() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(),
        metadata={},
        slide_cues=(
            SlideCue(1, "Verse 1", "Amazing grace", 0.5),
            SlideCue(2, "Chorus", "How sweet the sound", 1.0),
        ),
    )
    resolver = TimelineSlideResolver(
        profile,
        TimelineSlideResolverConfig(
            lookahead_seconds=0.0,
            cooldown_seconds=0.0,
            consecutive_match_count=1,
        ),
    )

    first = resolver.resolve(MatchResult(1, 0.25, 0.1, 0.1, 0.9, True))
    second = resolver.resolve(MatchResult(2, 0.5, 0.1, 0.1, 0.9, True))
    third = resolver.resolve(MatchResult(3, 1.2, 0.1, 0.1, 0.8, True))

    assert first is None
    assert second is not None
    assert second.slide_number == 1
    assert third is not None
    assert third.slide_number == 2


def test_resolver_ignores_invalid_matches() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(),
        metadata={},
        slide_cues=(SlideCue(1, "Verse 1", "Amazing grace", 0.5),),
    )
    resolver = TimelineSlideResolver(
        profile,
        TimelineSlideResolverConfig(
            lookahead_seconds=0.0,
            cooldown_seconds=0.0,
            consecutive_match_count=1,
        ),
    )

    result = resolver.resolve(MatchResult(2, 1.0, 0.1, 0.1, 0.2, False))

    assert result is None


def test_resolver_requires_consecutive_matches_and_honors_lookahead() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(),
        metadata={},
        slide_cues=(SlideCue(1, "Verse 1", "Amazing grace", 1.0),),
    )
    resolver = TimelineSlideResolver(
        profile,
        TimelineSlideResolverConfig(
            lookahead_seconds=0.2,
            cooldown_seconds=0.0,
            consecutive_match_count=2,
        ),
    )

    first = resolver.resolve(MatchResult(2, 0.79, 0.1, 0.1, 0.9, True))
    second = resolver.resolve(MatchResult(3, 0.81, 0.1, 0.1, 0.9, True))
    third = resolver.resolve(MatchResult(4, 0.82, 0.1, 0.1, 0.9, True))

    assert first is None
    assert second is None
    assert third is not None
    assert third.slide_number == 1


def test_resolver_suppresses_new_slide_during_cooldown() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(),
        metadata={},
        slide_cues=(
            SlideCue(1, "Verse 1", "Amazing grace", 1.0),
            SlideCue(2, "Chorus", "How sweet the sound", 1.2),
        ),
    )
    resolver = TimelineSlideResolver(
        profile,
        TimelineSlideResolverConfig(
            lookahead_seconds=0.0,
            cooldown_seconds=0.5,
            consecutive_match_count=1,
        ),
    )

    first = resolver.resolve(MatchResult(4, 1.0, 0.1, 0.1, 0.9, True))
    blocked = resolver.resolve(MatchResult(5, 1.21, 0.1, 0.1, 0.9, True))
    released = resolver.resolve(MatchResult(6, 1.6, 0.1, 0.1, 0.9, True))

    assert first is not None
    assert blocked is None
    assert released is not None
    assert released.slide_number == 2


def test_resolver_skips_stale_cues_after_late_relock() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(),
        metadata={},
        slide_cues=(
            SlideCue(1, "Verse 1", "Amazing grace", 1.0),
            SlideCue(2, "Verse 2", "How sweet the sound", 5.0),
            SlideCue(3, "Verse 3", "That saved a wretch", 9.0),
        ),
    )
    resolver = TimelineSlideResolver(
        profile,
        TimelineSlideResolverConfig(
            lookahead_seconds=0.0,
            cooldown_seconds=0.0,
            consecutive_match_count=1,
            max_emit_lag_seconds=1.0,
        ),
    )

    result = resolver.resolve(MatchResult(10, 9.2, 0.1, 0.1, 0.9, True))

    assert result is not None
    assert result.slide_number == 3


def test_resolver_seek_to_slide_advances_future_emissions() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(),
        metadata={},
        slide_cues=(
            SlideCue(1, "Verse 1", "Amazing grace", 1.0),
            SlideCue(2, "Verse 2", "How sweet the sound", 5.0),
            SlideCue(3, "Verse 3", "That saved a wretch", 9.0),
        ),
    )
    resolver = TimelineSlideResolver(
        profile,
        TimelineSlideResolverConfig(
            lookahead_seconds=0.0,
            cooldown_seconds=0.0,
            consecutive_match_count=1,
        ),
    )

    resolver.seek_to_slide(2)
    result = resolver.resolve(MatchResult(10, 9.2, 0.1, 0.1, 0.9, True))

    assert result is not None
    assert result.slide_number == 3


def test_resolver_groups_consecutive_line_cues_into_single_slide_output() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(),
        metadata={},
        slide_cues=(
            SlideCue(1, "Verse 1", "Amazing grace", 0.5),
            SlideCue(1, "Verse 1", "How sweet the sound", 0.8),
            SlideCue(2, "Verse 2", "That saved a wretch", 1.2),
        ),
    )
    resolver = TimelineSlideResolver(
        profile,
        TimelineSlideResolverConfig(
            lookahead_seconds=0.0,
            cooldown_seconds=0.0,
            consecutive_match_count=1,
        ),
    )

    first = resolver.resolve(MatchResult(1, 0.5, 0.1, 0.1, 0.9, True))
    second = resolver.resolve(MatchResult(2, 0.81, 0.1, 0.1, 0.9, True))
    third = resolver.resolve(MatchResult(3, 1.21, 0.1, 0.1, 0.9, True))

    assert first is not None
    assert first.slide_number == 1
    assert first.reference_timestamp == pytest.approx(0.5)
    assert first.lyrics == "Amazing grace\nHow sweet the sound"
    assert second is None
    assert third is not None
    assert third.slide_number == 2
