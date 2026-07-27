import numpy as np

from lyrics_aligner.adapters.matching import (
    RollingWindowFeatureMatcher,
    RollingWindowFeatureMatcherConfig,
)
from lyrics_aligner.domain.models import CorrectionAnchor, FeatureFrame, ReferenceProfile


def _reference_frame(values: list[float], observed_at: float) -> FeatureFrame:
    return FeatureFrame(
        values=np.array(values, dtype=np.float32),
        observed_at=observed_at,
        frame_duration_seconds=0.25,
    )


def _live_frame(values: list[float]) -> FeatureFrame:
    return FeatureFrame(
        values=np.array(values, dtype=np.float32),
        observed_at=10.0,
        frame_duration_seconds=0.25,
    )


def test_match_requires_minimum_sequence_before_accepting() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            _reference_frame([0.1, 0.1], 0.0),
            _reference_frame([0.2, 0.2], 0.25),
            _reference_frame([0.3, 0.3], 0.5),
        ),
        metadata={},
    )
    matcher = RollingWindowFeatureMatcher(
        profile,
        RollingWindowFeatureMatcherConfig(
            confidence_threshold=0.5,
            min_sequence_frames=2,
            max_sequence_frames=4,
        ),
    )

    first = matcher.match(_live_frame([0.1, 0.1]))
    second = matcher.match(_live_frame([0.2, 0.2]))

    assert first.reference_frame == -1
    assert first.valid is False
    assert second.reference_frame == 1
    assert second.reference_timestamp == 0.25
    assert second.valid is True


def test_match_disambiguates_repeated_single_frame_patterns_with_sequence_context() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            _reference_frame([0.10, 0.10], 0.0),
            _reference_frame([0.20, 0.20], 0.25),
            _reference_frame([0.30, 0.30], 0.5),
            _reference_frame([0.10, 0.10], 10.0),
            _reference_frame([0.20, 0.20], 10.25),
            _reference_frame([0.90, 0.90], 10.5),
        ),
        metadata={},
    )
    matcher = RollingWindowFeatureMatcher(
        profile,
        RollingWindowFeatureMatcherConfig(
            confidence_threshold=0.5,
            min_sequence_frames=3,
            max_sequence_frames=3,
        ),
    )

    matcher.match(_live_frame([0.10, 0.10]))
    matcher.match(_live_frame([0.20, 0.20]))
    result = matcher.match(_live_frame([0.90, 0.90]))

    assert result.reference_frame == 5
    assert result.reference_timestamp == 10.5
    assert result.valid is True
    assert matcher.last_decision is not None
    assert matcher.last_decision.sequence_length_frames == 3


def test_match_prefers_expected_window_when_configured() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            _reference_frame([0.10, 0.10], 0.0),
            _reference_frame([0.20, 0.20], 0.25),
            _reference_frame([0.30, 0.30], 0.5),
            _reference_frame([0.10, 0.10], 100.0),
            _reference_frame([0.20, 0.20], 100.25),
            _reference_frame([0.30, 0.30], 100.5),
        ),
        metadata={},
    )
    matcher = RollingWindowFeatureMatcher(
        profile,
        RollingWindowFeatureMatcherConfig(
            confidence_threshold=0.5,
            min_sequence_frames=3,
            max_sequence_frames=3,
        ),
    )
    matcher.set_expected_window(100.5, 2.0)

    matcher.match(_live_frame([0.10, 0.10]))
    matcher.match(_live_frame([0.20, 0.20]))
    result = matcher.match(_live_frame([0.30, 0.30]))

    assert result.reference_frame == 5
    assert result.reference_timestamp == 100.5
    assert result.valid is True


def test_invalid_frame_clears_sequence_state() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            _reference_frame([0.10, 0.10], 0.0),
            _reference_frame([0.20, 0.20], 0.25),
            _reference_frame([0.30, 0.30], 0.5),
        ),
        metadata={},
    )
    matcher = RollingWindowFeatureMatcher(
        profile,
        RollingWindowFeatureMatcherConfig(
            confidence_threshold=0.5,
            min_sequence_frames=2,
            max_sequence_frames=3,
        ),
    )

    matcher.match(_live_frame([0.10, 0.10]))
    invalid = matcher.match(
        FeatureFrame(
            values=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            observed_at=10.0,
            frame_duration_seconds=0.25,
        )
    )
    restarted = matcher.match(_live_frame([0.20, 0.20]))

    assert invalid.reference_frame == -1
    assert invalid.valid is False
    assert restarted.reference_frame == -1
    assert restarted.valid is False


def test_match_biases_toward_persisted_correction_anchor() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            _reference_frame([0.10, 0.10], 0.0),
            _reference_frame([0.20, 0.20], 0.25),
            _reference_frame([0.30, 0.30], 0.5),
            _reference_frame([0.10, 0.10], 10.0),
            _reference_frame([0.20, 0.20], 10.25),
            _reference_frame([0.305, 0.305], 10.5),
        ),
        metadata={},
    )
    matcher = RollingWindowFeatureMatcher(
        profile,
        RollingWindowFeatureMatcherConfig(
            confidence_threshold=0.5,
            min_sequence_frames=3,
            max_sequence_frames=3,
            anchor_activation_tolerance_seconds=0.75,
            anchor_target_tolerance_seconds=0.75,
            anchor_bias=0.02,
        ),
        correction_anchors=(
            CorrectionAnchor(
                profile_name="song-a",
                source_reference_timestamp=0.5,
                target_reference_timestamp=10.5,
                slide_number=2,
                section="Verse 2",
                lyrics="Target slide",
                correction_count=3,
                session_count=2,
                support_score=3.0,
                last_seen_at="2026-07-26T00:00:00+00:00",
            ),
        ),
    )

    matcher.match(_live_frame([0.10, 0.10]))
    matcher.match(_live_frame([0.20, 0.20]))
    result = matcher.match(_live_frame([0.302, 0.302]))

    assert result.reference_frame == 5
    assert result.reference_timestamp == 10.5
    assert result.valid is True
