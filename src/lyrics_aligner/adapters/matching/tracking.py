"""Stateful online tracking around frame-level match candidates."""

from __future__ import annotations

from dataclasses import dataclass

from lyrics_aligner.domain.models import FeatureFrame, MatchResult
from lyrics_aligner.ports.feature_matcher import FeatureMatcher


@dataclass(frozen=True, slots=True)
class TrackingFeatureMatcherConfig:
    search_min_consecutive_matches: int = 2
    recovery_min_consecutive_matches: int = 2
    tracking_confidence_threshold: float = 0.7
    recovery_confidence_threshold: float = 0.6
    max_forward_jump_frames: int = 6
    max_backward_recovery_frames: int = 2
    lost_match_patience: int = 3

    def __post_init__(self) -> None:
        if self.search_min_consecutive_matches <= 0:
            raise ValueError("search_min_consecutive_matches must be greater than zero")
        if self.recovery_min_consecutive_matches <= 0:
            raise ValueError("recovery_min_consecutive_matches must be greater than zero")
        if not 0.0 <= self.tracking_confidence_threshold <= 1.0:
            raise ValueError("tracking_confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.recovery_confidence_threshold <= 1.0:
            raise ValueError("recovery_confidence_threshold must be between 0.0 and 1.0")
        if self.max_forward_jump_frames < 0:
            raise ValueError("max_forward_jump_frames must be non-negative")
        if self.max_backward_recovery_frames < 0:
            raise ValueError("max_backward_recovery_frames must be non-negative")
        if self.lost_match_patience <= 0:
            raise ValueError("lost_match_patience must be greater than zero")


class TrackingFeatureMatcher:
    """Track a stable position with SEARCHING/TRACKING/UNCERTAIN states."""

    def __init__(
        self,
        matcher: FeatureMatcher,
        config: TrackingFeatureMatcherConfig | None = None,
    ) -> None:
        self._matcher = matcher
        self._config = config or TrackingFeatureMatcherConfig()
        self._state = "UNINITIALIZED"
        self._last_accepted_frame: int | None = None
        self._search_anchor_frame: int | None = None
        self._search_streak = 0
        self._recovery_anchor_frame: int | None = None
        self._recovery_streak = 0
        self._lost_count = 0

    @property
    def state_name(self) -> str:
        return self._state

    def match(self, frame: FeatureFrame) -> MatchResult:
        candidate = self._matcher.match(frame)
        if self._state in ("UNINITIALIZED", "SEARCHING"):
            return self._search(candidate)
        if self._state == "TRACKING":
            return self._track(candidate)
        return self._recover(candidate)

    def _search(self, candidate: MatchResult) -> MatchResult:
        self._state = "SEARCHING"
        if not candidate.valid or candidate.confidence < self._config.recovery_confidence_threshold:
            self._reset_search()
            return self._invalidate(candidate)

        if self._is_consistent_with_anchor(candidate.reference_frame, self._search_anchor_frame):
            self._search_streak += 1
        else:
            self._search_streak = 1
        self._search_anchor_frame = candidate.reference_frame

        if self._search_streak < self._config.search_min_consecutive_matches:
            return self._invalidate(candidate)

        self._last_accepted_frame = candidate.reference_frame
        self._reset_search()
        self._reset_recovery()
        self._lost_count = 0
        self._state = "TRACKING"
        return candidate

    def _track(self, candidate: MatchResult) -> MatchResult:
        if not self._is_tracking_candidate_acceptable(candidate):
            self._enter_uncertain(candidate)
            return self._invalidate(candidate)

        self._last_accepted_frame = candidate.reference_frame
        self._lost_count = 0
        self._reset_recovery()
        return candidate

    def _recover(self, candidate: MatchResult) -> MatchResult:
        if not self._is_recovery_candidate_acceptable(candidate):
            self._lost_count += 1
            self._reset_recovery()
            if self._lost_count >= self._config.lost_match_patience:
                self._state = "SEARCHING"
                self._lost_count = 0
                self._reset_search()
            return self._invalidate(candidate)

        self._lost_count = 0
        if self._is_consistent_with_anchor(candidate.reference_frame, self._recovery_anchor_frame):
            self._recovery_streak += 1
        else:
            self._recovery_streak = 1
        self._recovery_anchor_frame = candidate.reference_frame

        if self._recovery_streak < self._config.recovery_min_consecutive_matches:
            return self._invalidate(candidate)

        self._last_accepted_frame = candidate.reference_frame
        self._reset_recovery()
        self._reset_search()
        self._state = "TRACKING"
        return candidate

    def _enter_uncertain(self, candidate: MatchResult) -> None:
        self._state = "UNCERTAIN"
        self._lost_count = 1 if not candidate.valid else 0
        self._reset_recovery()
        self._reset_search()

    def _is_tracking_candidate_acceptable(self, candidate: MatchResult) -> bool:
        if not candidate.valid:
            return False
        if candidate.confidence < self._config.tracking_confidence_threshold:
            return False
        return self._is_relative_jump_acceptable(candidate.reference_frame)

    def _is_recovery_candidate_acceptable(self, candidate: MatchResult) -> bool:
        if not candidate.valid:
            return False
        if candidate.confidence < self._config.recovery_confidence_threshold:
            return False
        return self._is_relative_jump_acceptable(candidate.reference_frame)

    def _is_relative_jump_acceptable(self, candidate_frame: int) -> bool:
        if self._last_accepted_frame is None:
            return True
        delta = candidate_frame - self._last_accepted_frame
        return (
            delta >= -self._config.max_backward_recovery_frames
            and delta <= self._config.max_forward_jump_frames
        )

    def _is_consistent_with_anchor(
        self,
        candidate_frame: int,
        anchor_frame: int | None,
    ) -> bool:
        if anchor_frame is None:
            return False
        delta = candidate_frame - anchor_frame
        return (
            delta >= -self._config.max_backward_recovery_frames
            and delta <= self._config.max_forward_jump_frames
        )

    def _reset_search(self) -> None:
        self._search_anchor_frame = None
        self._search_streak = 0

    def _reset_recovery(self) -> None:
        self._recovery_anchor_frame = None
        self._recovery_streak = 0

    @staticmethod
    def _invalidate(candidate: MatchResult) -> MatchResult:
        return MatchResult(
            reference_frame=candidate.reference_frame,
            reference_timestamp=candidate.reference_timestamp,
            raw_distance=candidate.raw_distance,
            normalized_distance=candidate.normalized_distance,
            confidence=candidate.confidence,
            valid=False,
        )
