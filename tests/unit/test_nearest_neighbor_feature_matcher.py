import numpy as np

from lyrics_aligner.adapters.matching import (
    NearestNeighborFeatureMatcher,
    NearestNeighborFeatureMatcherConfig,
)
from lyrics_aligner.domain.models import FeatureFrame, ReferenceProfile


def test_match_returns_best_reference_frame() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            FeatureFrame(
                values=np.array([0.1, 0.2], dtype=np.float32),
                observed_at=0.0,
                frame_duration_seconds=0.25,
            ),
            FeatureFrame(
                values=np.array([0.8, 0.9], dtype=np.float32),
                observed_at=0.25,
                frame_duration_seconds=0.25,
            ),
        ),
        metadata={},
    )
    matcher = NearestNeighborFeatureMatcher(
        profile,
        NearestNeighborFeatureMatcherConfig(confidence_threshold=0.5),
    )

    result = matcher.match(
        FeatureFrame(
            values=np.array([0.12, 0.18], dtype=np.float32),
            observed_at=1.0,
            frame_duration_seconds=0.25,
        )
    )

    assert result.reference_frame == 0
    assert result.reference_timestamp == 0.0
    assert result.valid is True
    assert result.confidence > 0.9


def test_match_returns_invalid_for_mismatched_feature_size() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            FeatureFrame(
                values=np.array([0.1, 0.2], dtype=np.float32),
                observed_at=0.0,
                frame_duration_seconds=0.25,
            ),
        ),
        metadata={},
    )
    matcher = NearestNeighborFeatureMatcher(profile)

    result = matcher.match(
        FeatureFrame(
            values=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            observed_at=1.0,
            frame_duration_seconds=0.25,
        )
    )

    assert result.reference_frame == -1
    assert result.valid is False
    assert result.confidence == 0.0


def test_match_rejects_ambiguous_distant_second_best_region() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            FeatureFrame(
                values=np.array([0.10, 0.10], dtype=np.float32),
                observed_at=0.0,
                frame_duration_seconds=0.25,
            ),
            FeatureFrame(
                values=np.array([0.11, 0.10], dtype=np.float32),
                observed_at=30.0,
                frame_duration_seconds=0.25,
            ),
            FeatureFrame(
                values=np.array([0.90, 0.90], dtype=np.float32),
                observed_at=30.25,
                frame_duration_seconds=0.25,
            ),
        ),
        metadata={},
    )
    matcher = NearestNeighborFeatureMatcher(
        profile,
        NearestNeighborFeatureMatcherConfig(
            confidence_threshold=0.5,
            ambiguity_distance_margin=0.02,
            ambiguity_min_separation_seconds=15.0,
        ),
    )

    result = matcher.match(
        FeatureFrame(
            values=np.array([0.105, 0.10], dtype=np.float32),
            observed_at=1.0,
            frame_duration_seconds=0.25,
        )
    )

    assert result.reference_frame == 0
    assert result.confidence > 0.9
    assert result.valid is False
    assert matcher.last_decision is not None
    assert matcher.last_decision.reason == "ambiguous_distant_second_best"
    assert matcher.last_decision.second_time_gap_seconds == 30.0


def test_match_prefers_expected_window_when_configured() -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            FeatureFrame(
                values=np.array([0.10, 0.10], dtype=np.float32),
                observed_at=0.0,
                frame_duration_seconds=0.25,
            ),
            FeatureFrame(
                values=np.array([0.90, 0.90], dtype=np.float32),
                observed_at=50.0,
                frame_duration_seconds=0.25,
            ),
            FeatureFrame(
                values=np.array([0.11, 0.10], dtype=np.float32),
                observed_at=100.0,
                frame_duration_seconds=0.25,
            ),
        ),
        metadata={},
    )
    matcher = NearestNeighborFeatureMatcher(
        profile,
        NearestNeighborFeatureMatcherConfig(confidence_threshold=0.5),
    )
    matcher.set_expected_window(95.0, 10.0)

    result = matcher.match(
        FeatureFrame(
            values=np.array([0.105, 0.10], dtype=np.float32),
            observed_at=1.0,
            frame_duration_seconds=0.25,
        )
    )

    assert result.reference_frame == 2
    assert result.reference_timestamp == 100.0
    assert matcher.last_decision is not None
    assert matcher.last_decision.best_timestamp == 100.0
