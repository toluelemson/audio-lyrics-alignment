from lyrics_aligner.adapters.slides import TimelineSlideResolver
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
    resolver = TimelineSlideResolver(profile)

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
    resolver = TimelineSlideResolver(profile)

    result = resolver.resolve(MatchResult(2, 1.0, 0.1, 0.1, 0.2, False))

    assert result is None
