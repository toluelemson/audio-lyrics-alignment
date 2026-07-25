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


def _frame(observed_at: float = 0.0) -> FeatureFrame:
    return FeatureFrame(
        values=np.array([0.1, 0.2], dtype=np.float32),
        observed_at=observed_at,
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
            search_min_duration_seconds=0.0,
            recovery_min_duration_seconds=0.0,
        ),
    )

    first = tracker.match(_frame(0.0))
    second = tracker.match(_frame(0.1))

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
            search_min_duration_seconds=0.0,
            recovery_min_duration_seconds=0.0,
            tracking_confidence_threshold=0.7,
            recovery_confidence_threshold=0.6,
            max_forward_jump_seconds=0.6,
        ),
    )

    assert tracker.match(_frame(0.0)).valid is False
    assert tracker.match(_frame(0.1)).valid is True

    uncertain = tracker.match(_frame(0.2))
    assert uncertain.valid is False
    assert tracker.last_decision is not None
    assert tracker.last_decision.state == "UNCERTAIN"

    recovering = tracker.match(_frame(0.3))
    recovered = tracker.match(_frame(0.4))

    assert recovering.valid is False
    assert recovered.valid is True
    assert recovered.reference_frame == 4
    assert tracker.state_name == "TRACKING"


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
            search_min_duration_seconds=0.0,
            recovery_min_duration_seconds=0.0,
            lost_match_patience=2,
        ),
    )

    assert tracker.match(_frame(0.0)).valid is False
    assert tracker.match(_frame(0.1)).valid is True
    assert tracker.match(_frame(0.2)).valid is False
    lost = tracker.match(_frame(0.3))

    assert lost.valid is False
    assert tracker.state_name == "SEARCHING"


def test_tracker_exposes_rejection_reason_for_large_jump() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            _result(1),
            _result(2),
            _result(20),
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            search_min_duration_seconds=0.0,
            recovery_min_duration_seconds=0.0,
            max_forward_jump_seconds=0.5,
        ),
    )

    assert tracker.match(_frame(0.0)).valid is False
    assert tracker.match(_frame(0.1)).valid is True
    rejected = tracker.match(_frame(0.2))

    assert rejected.valid is False
    assert tracker.last_decision is not None
    assert tracker.last_decision.reason == "tracking_rejected_large_jump"
    assert tracker.last_decision.state == "UNCERTAIN"


def test_tracker_tolerates_brief_bad_frame_while_tracking() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            _result(1),
            _result(2),
            _result(20),
            _result(3),
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            search_min_duration_seconds=0.0,
            recovery_min_duration_seconds=0.0,
            tracking_miss_patience=1,
            max_forward_jump_seconds=0.5,
        ),
    )

    assert tracker.match(_frame(0.0)).valid is False
    assert tracker.match(_frame(0.1)).valid is True
    held = tracker.match(_frame(0.2))

    assert held.valid is False
    assert tracker.last_decision is not None
    assert tracker.last_decision.reason == "tracking_waiting_for_reacquire"
    assert tracker.state_name == "TRACKING"
    recovered = tracker.match(_frame(0.3))
    assert recovered.valid is True
    assert tracker.state_name == "TRACKING"


def test_tracker_requires_min_search_duration_before_locking() -> None:
    matcher: FeatureMatcher = SequenceMatcher([_result(1), _result(2), _result(3)])
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            search_min_duration_seconds=0.25,
            recovery_min_duration_seconds=0.0,
        ),
    )

    assert tracker.match(_frame(0.00)).valid is False
    assert tracker.match(_frame(0.10)).valid is False
    locked = tracker.match(_frame(0.30))

    assert locked.valid is True
    assert tracker.state_name == "TRACKING"


def test_tracker_search_accepts_progress_scaled_to_observed_time() -> None:
    matcher: FeatureMatcher = SequenceMatcher([_result(0), _result(2), _result(4)])
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            search_min_duration_seconds=0.25,
            recovery_min_duration_seconds=0.0,
            max_forward_jump_seconds=0.35,
        ),
    )

    first = tracker.match(_frame(0.00))
    second = tracker.match(_frame(0.51))

    assert first.valid is False
    assert second.valid is True
    assert second.reference_frame == 2
    assert tracker.state_name == "TRACKING"


def test_tracker_rejects_search_candidate_outside_expected_position_window() -> None:
    matcher: FeatureMatcher = SequenceMatcher([_result(0), _result(1), _result(40)])
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=3,
            recovery_min_consecutive_matches=2,
            expected_position_tolerance_seconds=2.0,
            search_min_duration_seconds=0.0,
            recovery_min_duration_seconds=0.0,
        ),
    )

    first = tracker.match(_frame(0.0))
    second = tracker.match(_frame(0.1))
    third = tracker.match(_frame(0.2))

    assert first.valid is False
    assert second.valid is False
    assert third.valid is False
    assert tracker.state_name == "SEARCHING"
    assert tracker.last_decision is not None
    assert tracker.last_decision.reason == "search_waiting_for_reacquire"


def test_tracker_force_anchor_guides_recovery_to_timestamp() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            MatchResult(
                reference_frame=40,
                reference_timestamp=10.0,
                raw_distance=0.1,
                normalized_distance=0.1,
                confidence=0.9,
                valid=True,
            )
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            recovery_min_consecutive_matches=1,
            recovery_min_duration_seconds=0.0,
            recovery_confidence_threshold=0.6,
            max_forward_jump_seconds=0.5,
        ),
    )

    tracker.force_anchor(10.0, observed_at=30.0)
    anchored = tracker.match(_frame(30.0))

    assert anchored.valid is True
    assert anchored.reference_timestamp == 10.0
    assert tracker.state_name == "TRACKING"


def test_tracker_prefers_anchored_timeline_over_distant_live_match() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            MatchResult(
                reference_frame=900,
                reference_timestamp=225.0,
                raw_distance=0.1,
                normalized_distance=0.1,
                confidence=0.91,
                valid=True,
            )
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            anchored_timeline_tolerance_seconds=1.5,
            max_forward_jump_seconds=0.5,
        ),
    )

    tracker.force_anchor(10.0, observed_at=30.0)
    anchored = tracker.match(_frame(30.5))

    assert anchored.valid is True
    assert anchored.reference_timestamp == 10.5
    assert anchored.reference_frame == 42
    assert tracker.last_decision is not None
    assert tracker.last_decision.reason == "tracking_guided_by_anchor"
    assert tracker.state_name == "TRACKING"


def test_tracker_accepts_live_match_when_it_stays_close_to_anchor_timeline() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            MatchResult(
                reference_frame=42,
                reference_timestamp=10.4,
                raw_distance=0.1,
                normalized_distance=0.1,
                confidence=0.91,
                valid=True,
            )
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            anchored_timeline_tolerance_seconds=1.5,
            max_forward_jump_seconds=0.5,
        ),
    )

    tracker.force_anchor(10.0, observed_at=30.0)
    accepted = tracker.match(_frame(30.5))

    assert accepted.valid is True
    assert accepted.reference_timestamp == 10.4
    assert tracker.last_decision is not None
    assert tracker.last_decision.reason == "tracking_accepted"


def test_tracker_search_seeds_expected_origin_from_first_strong_candidate() -> None:
    matcher: FeatureMatcher = SequenceMatcher([_result(80), _result(81), _result(82)])
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            expected_position_tolerance_seconds=2.0,
            search_min_duration_seconds=0.0,
            recovery_min_duration_seconds=0.0,
        ),
    )

    first = tracker.match(_frame(20.0))
    second = tracker.match(_frame(20.25))

    assert first.valid is False
    assert second.valid is True
    assert second.reference_frame == 81
    assert tracker.state_name == "TRACKING"


def test_tracker_search_survives_brief_miss_and_then_locks() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            _result(0),
            _result(500, valid=False, confidence=0.0),
            _result(1),
            _result(1),
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            search_miss_patience=1,
            expected_position_tolerance_seconds=10.0,
            search_min_duration_seconds=0.25,
            recovery_min_duration_seconds=0.0,
        ),
    )

    assert tracker.match(_frame(0.00)).valid is False
    missed = tracker.match(_frame(0.10))
    assert missed.valid is False
    assert tracker.last_decision is not None
    assert tracker.last_decision.reason == "search_waiting_for_reacquire"
    assert tracker.match(_frame(0.20)).valid is False
    locked = tracker.match(_frame(0.30))

    assert locked.valid is True
    assert locked.reference_frame == 1
    assert tracker.state_name == "TRACKING"


def test_tracker_updates_origin_after_lock_and_rejects_large_expected_time_drift() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            _result(1),
            _result(2),
            _result(3),
            _result(40),
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            expected_position_tolerance_seconds=1.0,
            search_min_duration_seconds=0.0,
            recovery_min_duration_seconds=0.0,
            max_forward_jump_seconds=10.0,
        ),
    )

    assert tracker.match(_frame(0.25)).valid is False
    locked = tracker.match(_frame(0.50))
    stable = tracker.match(_frame(0.75))
    drifted = tracker.match(_frame(1.00))

    assert locked.valid is True
    assert stable.valid is True
    assert drifted.valid is False
    assert tracker.last_decision is not None
    assert tracker.last_decision.reason == "tracking_rejected_outside_expected_window"


def test_tracker_enters_uncertain_after_exceeding_tracking_miss_patience() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [
            _result(1),
            _result(2),
            _result(20),
            _result(20),
        ]
    )
    tracker = TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=2,
            recovery_min_consecutive_matches=2,
            search_min_duration_seconds=0.0,
            recovery_min_duration_seconds=0.0,
            tracking_miss_patience=1,
            max_forward_jump_seconds=0.5,
        ),
    )

    assert tracker.match(_frame(0.0)).valid is False
    assert tracker.match(_frame(0.1)).valid is True
    assert tracker.match(_frame(0.2)).valid is False
    rejected = tracker.match(_frame(0.3))

    assert rejected.valid is False
    assert tracker.last_decision is not None
    assert tracker.last_decision.reason == "tracking_rejected_large_jump"
    assert tracker.state_name == "UNCERTAIN"
