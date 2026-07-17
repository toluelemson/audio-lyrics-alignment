import numpy as np

from lyrics_aligner.adapters.matching import (
    TrackingFeatureMatcher,
    TrackingFeatureMatcherConfig,
)
from lyrics_aligner.domain.models import FeatureFrame, MatchResult
from lyrics_aligner.ports.feature_matcher import FeatureMatcher


class SequenceMatcher:
    def __init__(self, results: list[MatchResult]) -> None:
        self._results = results
        self._index = 0

    def match(self, frame: FeatureFrame) -> MatchResult:
        del frame
        result = self._results[self._index]
        self._index += 1
        return result


def _frame() -> FeatureFrame:
    return FeatureFrame(
        values=np.array([0.1, 0.2], dtype=np.float32),
        observed_at=0.0,
        frame_duration_seconds=0.25,
    )


def _result(reference_frame: int, confidence: float = 0.9, *, valid: bool = True) -> MatchResult:
    return MatchResult(
        reference_frame=reference_frame,
        reference_timestamp=reference_frame * 0.25,
        raw_distance=0.1,
        normalized_distance=0.1,
        confidence=confidence,
        valid=valid,
    )


def test_tracker_requires_consecutive_search_matches_before_tracking() -> None:
    matcher: FeatureMatcher = SequenceMatcher([_result(1), _result(2)])
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
        ),
    )

    first = tracker.match(_frame())
    second = tracker.match(_frame())

    assert first.valid is False
    assert tracker.state_name == "TRACKING"
    assert second.valid is True
    assert second.reference_frame == 2


def test_tracker_enters_uncertain_and_recovers_after_stable_matches() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            _result(1),
            _result(2),
            _result(3, confidence=0.5),
            _result(3, confidence=0.8),
            _result(4, confidence=0.8),
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            tracking_confidence_threshold=0.7,
            recovery_confidence_threshold=0.6,
        ),
    )

    assert tracker.match(_frame()).valid is False
    assert tracker.match(_frame()).valid is True

    uncertain = tracker.match(_frame())
    recovering = tracker.match(_frame())
    recovered = tracker.match(_frame())

    assert uncertain.valid is False
    assert tracker.state_name == "TRACKING"
    assert recovering.valid is False
    assert recovered.valid is True
    assert recovered.reference_frame == 4


def test_tracker_falls_back_to_searching_after_too_many_losses() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            _result(1),
            _result(2),
            _result(2, valid=False, confidence=0.0),
            _result(2, valid=False, confidence=0.0),
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            lost_match_patience=2,
        ),
    )

    assert tracker.match(_frame()).valid is False
    assert tracker.match(_frame()).valid is True
    assert tracker.match(_frame()).valid is False
    lost = tracker.match(_frame())

    assert lost.valid is False
    assert tracker.state_name == "SEARCHING"
