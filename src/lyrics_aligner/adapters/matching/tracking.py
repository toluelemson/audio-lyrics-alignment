"""Stateful online tracking around frame-level match candidates."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from lyrics_aligner.domain.models import FeatureFrame, MatchResult
from lyrics_aligner.ports.feature_matcher import FeatureMatcher


@dataclass(frozen=True, slots=True)
class TrackingFeatureMatcherConfig:
    search_min_consecutive_matches: int = 2
    recovery_min_consecutive_matches: int = 2
    search_miss_patience: int = 2
    tracking_miss_patience: int = 0
    search_min_duration_seconds: float = 0.25
    recovery_min_duration_seconds: float = 0.25
    tracking_confidence_threshold: float = 0.7
    recovery_confidence_threshold: float = 0.6
    tracking_match_window_seconds: float = 12.0
    recovery_match_window_seconds: float = 18.0
    expected_position_tolerance_seconds: float = 12.0
    anchored_timeline_tolerance_seconds: float = 1.5
    max_forward_jump_seconds: float = 0.35
    max_backward_jump_seconds: float = 0.2
    ambiguous_second_distance_gap_threshold: float = 0.01
    ambiguous_second_time_gap_seconds: float = 0.5
    lost_match_patience: int = 3
    anchored_timeline_confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.search_min_consecutive_matches <= 0:
            raise ValueError("search_min_consecutive_matches must be greater than zero")
        if self.recovery_min_consecutive_matches <= 0:
            raise ValueError("recovery_min_consecutive_matches must be greater than zero")
        if self.search_miss_patience < 0:
            raise ValueError("search_miss_patience must be non-negative")
        if self.tracking_miss_patience < 0:
            raise ValueError("tracking_miss_patience must be non-negative")
        if self.search_min_duration_seconds < 0.0:
            raise ValueError("search_min_duration_seconds must be non-negative")
        if self.recovery_min_duration_seconds < 0.0:
            raise ValueError("recovery_min_duration_seconds must be non-negative")
        if not 0.0 <= self.tracking_confidence_threshold <= 1.0:
            raise ValueError("tracking_confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.recovery_confidence_threshold <= 1.0:
            raise ValueError("recovery_confidence_threshold must be between 0.0 and 1.0")
        if self.tracking_match_window_seconds < 0.0:
            raise ValueError("tracking_match_window_seconds must be non-negative")
        if self.recovery_match_window_seconds < 0.0:
            raise ValueError("recovery_match_window_seconds must be non-negative")
        if self.expected_position_tolerance_seconds < 0.0:
            raise ValueError("expected_position_tolerance_seconds must be non-negative")
        if self.anchored_timeline_tolerance_seconds < 0.0:
            raise ValueError(
                "anchored_timeline_tolerance_seconds must be non-negative"
            )
        if self.max_forward_jump_seconds < 0.0:
            raise ValueError("max_forward_jump_seconds must be non-negative")
        if self.max_backward_jump_seconds < 0.0:
            raise ValueError("max_backward_jump_seconds must be non-negative")
        if self.ambiguous_second_distance_gap_threshold < 0.0:
            raise ValueError(
                "ambiguous_second_distance_gap_threshold must be non-negative"
            )
        if self.ambiguous_second_time_gap_seconds < 0.0:
            raise ValueError(
                "ambiguous_second_time_gap_seconds must be non-negative"
            )
        if self.lost_match_patience <= 0:
            raise ValueError("lost_match_patience must be greater than zero")
        if not 0.0 <= self.anchored_timeline_confidence <= 1.0:
            raise ValueError(
                "anchored_timeline_confidence must be between 0.0 and 1.0"
            )


@dataclass(frozen=True, slots=True)
class TrackingDecision:
    previous_state: str
    state: str
    candidate_frame: int
    candidate_timestamp: float
    candidate_confidence: float
    candidate_valid: bool
    accepted: bool
    reason: str
    delta_from_last_accepted: int | None
    matcher_reason: str | None
    matcher_second_time_gap_seconds: float | None
    matcher_second_distance_gap: float | None


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
        self._last_accepted_timestamp: float | None = None
        self._timeline_origin_observed_at: float | None = None
        self._search_anchor_timestamp: float | None = None
        self._search_anchor_observed_at: float | None = None
        self._search_start_observed_at: float | None = None
        self._search_streak = 0
        self._search_miss_count = 0
        self._tracking_miss_count = 0
        self._recovery_anchor_timestamp: float | None = None
        self._recovery_anchor_observed_at: float | None = None
        self._recovery_start_observed_at: float | None = None
        self._recovery_streak = 0
        self._global_reacquire_anchor_timestamp: float | None = None
        self._global_reacquire_anchor_observed_at: float | None = None
        self._global_reacquire_start_observed_at: float | None = None
        self._global_reacquire_streak = 0
        self._lost_count = 0
        self._anchored_timeline_active = False
        self._manual_anchor_active = False
        self._last_decision: TrackingDecision | None = None

    @property
    def state_name(self) -> str:
        return self._state

    @property
    def last_decision(self) -> TrackingDecision | None:
        return self._last_decision

    def reset(self) -> None:
        self._state = "UNINITIALIZED"
        self._last_accepted_frame = None
        self._last_accepted_timestamp = None
        self._timeline_origin_observed_at = None
        self._reset_search()
        self._reset_recovery()
        self._reset_global_reacquire()
        self._lost_count = 0
        self._tracking_miss_count = 0
        self._anchored_timeline_active = False
        self._manual_anchor_active = False
        self._last_decision = None

    def force_anchor(
        self,
        reference_timestamp: float,
        observed_at: float | None = None,
    ) -> None:
        if reference_timestamp < 0.0:
            raise ValueError("reference_timestamp must be non-negative")
        current_observed_at = monotonic() if observed_at is None else observed_at
        self._state = "TRACKING"
        self._last_accepted_timestamp = reference_timestamp
        self._last_accepted_frame = None
        self._timeline_origin_observed_at = current_observed_at - reference_timestamp
        self._tracking_miss_count = 0
        self._lost_count = 0
        self._anchored_timeline_active = True
        self._manual_anchor_active = True
        self._reset_search()
        self._reset_recovery()
        self._reset_global_reacquire()
        self._recovery_anchor_timestamp = reference_timestamp
        self._recovery_anchor_observed_at = current_observed_at
        self._recovery_start_observed_at = current_observed_at

    def match(self, frame: FeatureFrame) -> MatchResult:
        self._prepare_matcher_window(frame.observed_at)
        candidate = self._matcher.match(frame)
        previous_state = self._state
        if self._state in ("UNINITIALIZED", "SEARCHING"):
            return self._search(frame, candidate, previous_state)
        if self._state == "TRACKING":
            return self._track(frame, candidate, previous_state)
        return self._recover(frame, candidate, previous_state)

    def _search(
        self,
        frame: FeatureFrame,
        candidate: MatchResult,
        previous_state: str,
    ) -> MatchResult:
        self._state = "SEARCHING"
        self._ensure_timeline_origin(frame.observed_at)
        if not candidate.valid or candidate.confidence < self._config.recovery_confidence_threshold:
            self._search_miss_count += 1
            if self._search_miss_count > self._config.search_miss_patience:
                self._reset_search()
            result = self._invalidate(candidate)
            self._record_decision(
                previous_state,
                result,
                accepted=False,
                reason=(
                    "search_rejected_low_confidence"
                    if self._search_streak == 0
                    else "search_waiting_for_reacquire"
                ),
            )
            return result
        if self._should_seed_search_origin():
            self._update_timeline_origin(frame.observed_at, candidate.reference_timestamp)
        if not self._is_expected_position_acceptable(
            frame.observed_at,
            candidate.reference_timestamp,
        ):
            self._search_miss_count += 1
            if self._search_miss_count > self._config.search_miss_patience:
                self._reset_search()
            result = self._invalidate(candidate)
            self._record_decision(
                previous_state,
                result,
                accepted=False,
                reason=(
                    "search_rejected_outside_expected_window"
                    if self._search_streak == 0
                    else "search_waiting_for_reacquire"
                ),
            )
            return result

        if self._is_consistent_with_anchor(
            candidate.reference_timestamp,
            self._search_anchor_timestamp,
            frame.observed_at,
            self._search_anchor_observed_at,
        ):
            self._search_streak += 1
        else:
            self._search_streak = 1
            self._search_start_observed_at = frame.observed_at
        if self._search_start_observed_at is None:
            self._search_start_observed_at = frame.observed_at
        self._search_anchor_timestamp = candidate.reference_timestamp
        self._search_anchor_observed_at = frame.observed_at
        self._search_miss_count = 0

        search_duration = frame.observed_at - self._search_start_observed_at
        if (
            self._search_streak < self._config.search_min_consecutive_matches
            or search_duration < self._config.search_min_duration_seconds
        ):
            result = self._invalidate(candidate)
            self._record_decision(
                previous_state,
                result,
                accepted=False,
                reason=(
                    "search_waiting_for_duration"
                    if search_duration < self._config.search_min_duration_seconds
                    else "search_waiting_for_confirmation"
                ),
            )
            return result

        previous_accepted_frame = self._last_accepted_frame
        self._last_accepted_frame = candidate.reference_frame
        self._last_accepted_timestamp = candidate.reference_timestamp
        self._update_timeline_origin(frame.observed_at, candidate.reference_timestamp)
        self._reset_search()
        self._reset_recovery()
        self._reset_global_reacquire()
        self._lost_count = 0
        self._state = "TRACKING"
        self._record_decision(
            previous_state,
            candidate,
            accepted=True,
            reason="search_confirmed_tracking_lock",
            previous_accepted_frame=previous_accepted_frame,
        )
        return candidate

    def _track(
        self,
        frame: FeatureFrame,
        candidate: MatchResult,
        previous_state: str,
    ) -> MatchResult:
        if not self._is_tracking_candidate_acceptable(frame.observed_at, candidate):
            anchored_result = self._anchored_timeline_result(frame, candidate)
            if anchored_result is not None:
                previous_accepted_frame = self._last_accepted_frame
                self._last_accepted_frame = anchored_result.reference_frame
                self._last_accepted_timestamp = anchored_result.reference_timestamp
                self._tracking_miss_count = 0
                self._lost_count = 0
                self._reset_recovery()
                self._record_decision(
                    previous_state,
                    anchored_result,
                    accepted=True,
                    reason="tracking_guided_by_anchor",
                    previous_accepted_frame=previous_accepted_frame,
                )
                return anchored_result
            self._tracking_miss_count += 1
            if self._tracking_miss_count <= self._config.tracking_miss_patience:
                result = self._invalidate(candidate)
                self._record_decision(
                    previous_state,
                    result,
                    accepted=False,
                    reason="tracking_waiting_for_reacquire",
                )
                return result
            self._enter_uncertain(candidate)
            result = self._invalidate(candidate)
            self._record_decision(
                previous_state,
                result,
                accepted=False,
                reason=self._tracking_rejection_reason(frame.observed_at, candidate),
            )
            return result

        previous_accepted_frame = self._last_accepted_frame
        self._last_accepted_frame = candidate.reference_frame
        self._last_accepted_timestamp = candidate.reference_timestamp
        self._update_timeline_origin(frame.observed_at, candidate.reference_timestamp)
        self._lost_count = 0
        self._tracking_miss_count = 0
        self._reset_recovery()
        self._reset_global_reacquire()
        self._record_decision(
            previous_state,
            candidate,
            accepted=True,
            reason="tracking_accepted",
            previous_accepted_frame=previous_accepted_frame,
        )
        return candidate

    def _recover(
        self,
        frame: FeatureFrame,
        candidate: MatchResult,
        previous_state: str,
    ) -> MatchResult:
        self._ensure_timeline_origin(frame.observed_at)
        global_reacquired = self._try_global_reacquire(frame, candidate, previous_state)
        if global_reacquired is not None:
            return global_reacquired
        if not self._is_recovery_candidate_acceptable(frame.observed_at, candidate):
            anchored_result = self._anchored_timeline_result(frame, candidate)
            if anchored_result is not None:
                previous_accepted_frame = self._last_accepted_frame
                self._last_accepted_frame = anchored_result.reference_frame
                self._last_accepted_timestamp = anchored_result.reference_timestamp
                self._tracking_miss_count = 0
                self._lost_count = 0
                self._reset_recovery()
                self._reset_search()
                self._reset_global_reacquire()
                self._state = "TRACKING"
                self._record_decision(
                    previous_state,
                    anchored_result,
                    accepted=True,
                    reason="recovery_guided_by_anchor",
                    previous_accepted_frame=previous_accepted_frame,
                )
                return anchored_result
            self._lost_count += 1
            self._reset_recovery()
            if self._lost_count >= self._config.lost_match_patience:
                self._state = "UNCERTAIN" if self._manual_anchor_active else "SEARCHING"
                self._lost_count = 0
                self._reset_search()
                self._reset_global_reacquire()
                result = self._invalidate(candidate)
                self._record_decision(
                    previous_state,
                    result,
                    accepted=False,
                    reason=(
                        "recovery_failed_waiting_for_manual_reanchor"
                        if self._manual_anchor_active
                        else "recovery_failed_back_to_search"
                    ),
                )
                return result
            result = self._invalidate(candidate)
            self._record_decision(
                previous_state,
                result,
                accepted=False,
                reason=self._recovery_rejection_reason(frame.observed_at, candidate),
            )
            return result

        self._lost_count = 0
        if self._is_consistent_with_anchor(
            candidate.reference_timestamp,
            self._recovery_anchor_timestamp,
            frame.observed_at,
            self._recovery_anchor_observed_at,
        ):
            self._recovery_streak += 1
        else:
            self._recovery_streak = 1
            self._recovery_start_observed_at = frame.observed_at
        if self._recovery_start_observed_at is None:
            self._recovery_start_observed_at = frame.observed_at
        self._recovery_anchor_timestamp = candidate.reference_timestamp
        self._recovery_anchor_observed_at = frame.observed_at

        recovery_duration = frame.observed_at - self._recovery_start_observed_at
        if (
            self._recovery_streak < self._config.recovery_min_consecutive_matches
            or recovery_duration < self._config.recovery_min_duration_seconds
        ):
            result = self._invalidate(candidate)
            self._record_decision(
                previous_state,
                result,
                accepted=False,
                reason=(
                    "recovery_waiting_for_duration"
                    if recovery_duration < self._config.recovery_min_duration_seconds
                    else "recovery_waiting_for_confirmation"
                ),
            )
            return result

        previous_accepted_frame = self._last_accepted_frame
        self._last_accepted_frame = candidate.reference_frame
        self._last_accepted_timestamp = candidate.reference_timestamp
        self._update_timeline_origin(frame.observed_at, candidate.reference_timestamp)
        self._reset_recovery()
        self._reset_search()
        self._reset_global_reacquire()
        self._state = "TRACKING"
        self._record_decision(
            previous_state,
            candidate,
            accepted=True,
            reason="recovery_confirmed_tracking_lock",
            previous_accepted_frame=previous_accepted_frame,
        )
        return candidate

    def _enter_uncertain(self, candidate: MatchResult) -> None:
        self._state = "UNCERTAIN"
        self._lost_count = 1 if not candidate.valid else 0
        self._tracking_miss_count = 0
        self._reset_recovery()
        self._reset_search()
        self._reset_global_reacquire()

    def _is_tracking_candidate_acceptable(
        self,
        observed_at: float,
        candidate: MatchResult,
    ) -> bool:
        if not candidate.valid:
            return False
        if candidate.confidence < self._config.tracking_confidence_threshold:
            return False
        if self._is_matcher_ambiguity_unacceptable():
            return False
        if not self._is_expected_position_acceptable(
            observed_at,
            candidate.reference_timestamp,
        ):
            return False
        return self._is_relative_jump_acceptable(candidate.reference_timestamp)

    def _is_recovery_candidate_acceptable(
        self,
        observed_at: float,
        candidate: MatchResult,
    ) -> bool:
        if not candidate.valid:
            return False
        if candidate.confidence < self._config.recovery_confidence_threshold:
            return False
        if self._is_matcher_ambiguity_unacceptable():
            return False
        if not self._is_expected_position_acceptable(
            observed_at,
            candidate.reference_timestamp,
        ):
            return False
        return self._is_relative_jump_acceptable(candidate.reference_timestamp)

    def _tracking_rejection_reason(self, observed_at: float, candidate: MatchResult) -> str:
        if not candidate.valid:
            return "tracking_rejected_invalid_candidate"
        if candidate.confidence < self._config.tracking_confidence_threshold:
            return "tracking_rejected_low_confidence"
        if self._is_matcher_ambiguity_unacceptable():
            return "tracking_rejected_ambiguous_candidate"
        if not self._is_expected_position_acceptable(
            observed_at,
            candidate.reference_timestamp,
        ):
            return "tracking_rejected_outside_expected_window"
        return "tracking_rejected_large_jump"

    def _recovery_rejection_reason(self, observed_at: float, candidate: MatchResult) -> str:
        if not candidate.valid:
            return "recovery_rejected_invalid_candidate"
        if candidate.confidence < self._config.recovery_confidence_threshold:
            return "recovery_rejected_low_confidence"
        if self._is_matcher_ambiguity_unacceptable():
            return "recovery_rejected_ambiguous_candidate"
        if not self._is_expected_position_acceptable(
            observed_at,
            candidate.reference_timestamp,
        ):
            return "recovery_rejected_outside_expected_window"
        return "recovery_rejected_large_jump"

    def _is_matcher_ambiguity_unacceptable(self) -> bool:
        matcher_decision = getattr(self._matcher, "last_decision", None)
        if matcher_decision is None:
            return False
        second_gap = getattr(matcher_decision, "second_distance_gap", None)
        second_time_gap = getattr(matcher_decision, "second_time_gap_seconds", None)
        if second_gap is None or second_time_gap is None:
            return False
        return (
            second_gap <= self._config.ambiguous_second_distance_gap_threshold
            and second_time_gap >= self._config.ambiguous_second_time_gap_seconds
        )

    def _is_relative_jump_acceptable(self, candidate_timestamp: float) -> bool:
        if self._last_accepted_timestamp is None:
            return True
        delta = candidate_timestamp - self._last_accepted_timestamp
        return (
            delta >= -self._config.max_backward_jump_seconds
            and delta <= self._config.max_forward_jump_seconds
        )

    def _is_consistent_with_anchor(
        self,
        candidate_timestamp: float,
        anchor_timestamp: float | None,
        observed_at: float,
        anchor_observed_at: float | None,
    ) -> bool:
        if anchor_timestamp is None or anchor_observed_at is None:
            return False
        expected_delta = max(0.0, observed_at - anchor_observed_at)
        delta = candidate_timestamp - anchor_timestamp
        return (
            delta >= -self._config.max_backward_jump_seconds
            and delta <= expected_delta + self._config.max_forward_jump_seconds
        )

    def _should_seed_search_origin(self) -> bool:
        return (
            self._last_accepted_timestamp is None
            and self._search_anchor_timestamp is None
        )

    def _ensure_timeline_origin(self, observed_at: float) -> None:
        if self._timeline_origin_observed_at is None:
            self._timeline_origin_observed_at = observed_at

    def _update_timeline_origin(
        self,
        observed_at: float,
        reference_timestamp: float,
    ) -> None:
        self._timeline_origin_observed_at = observed_at - reference_timestamp

    def _is_expected_position_acceptable(
        self,
        observed_at: float,
        candidate_timestamp: float,
    ) -> bool:
        self._ensure_timeline_origin(observed_at)
        if self._timeline_origin_observed_at is None:
            return True
        expected_timestamp = observed_at - self._timeline_origin_observed_at
        tolerance_seconds = (
            self._config.anchored_timeline_tolerance_seconds
            if self._anchored_timeline_active
            else self._config.expected_position_tolerance_seconds
        )
        return abs(candidate_timestamp - expected_timestamp) <= tolerance_seconds

    def _anchored_timeline_result(
        self,
        frame: FeatureFrame,
        candidate: MatchResult,
    ) -> MatchResult | None:
        if not self._anchored_timeline_active or not self._manual_anchor_active:
            return None
        if candidate.confidence < self._config.recovery_confidence_threshold:
            return None
        if not self._matcher_reason_allows_anchor_guidance():
            return None
        if self._is_matcher_ambiguity_unacceptable():
            return None
        self._ensure_timeline_origin(frame.observed_at)
        if self._timeline_origin_observed_at is None:
            return None
        expected_timestamp = max(
            0.0,
            frame.observed_at - self._timeline_origin_observed_at,
        )
        reference_frame = (
            self._last_accepted_frame
            if self._last_accepted_frame is not None
            else (candidate.reference_frame if candidate.reference_frame >= 0 else 0)
        )
        return MatchResult(
            reference_frame=reference_frame,
            reference_timestamp=expected_timestamp,
            raw_distance=candidate.raw_distance,
            normalized_distance=candidate.normalized_distance,
            confidence=candidate.confidence,
            valid=True,
        )

    def _matcher_reason_allows_anchor_guidance(self) -> bool:
        matcher_decision = getattr(self._matcher, "last_decision", None)
        if matcher_decision is None:
            return True
        reason = getattr(matcher_decision, "reason", None)
        if reason is None:
            return True
        return reason == "accepted"

    def _reset_search(self) -> None:
        self._search_anchor_timestamp = None
        self._search_anchor_observed_at = None
        self._search_start_observed_at = None
        self._search_streak = 0
        self._search_miss_count = 0

    def _reset_recovery(self) -> None:
        self._recovery_anchor_timestamp = None
        self._recovery_anchor_observed_at = None
        self._recovery_start_observed_at = None
        self._recovery_streak = 0

    def _reset_global_reacquire(self) -> None:
        self._global_reacquire_anchor_timestamp = None
        self._global_reacquire_anchor_observed_at = None
        self._global_reacquire_start_observed_at = None
        self._global_reacquire_streak = 0

    def _try_global_reacquire(
        self,
        frame: FeatureFrame,
        candidate: MatchResult,
        previous_state: str,
    ) -> MatchResult | None:
        if self._manual_anchor_active:
            self._reset_global_reacquire()
            return None
        if not candidate.valid or candidate.confidence < self._config.recovery_confidence_threshold:
            self._reset_global_reacquire()
            return None
        if self._is_recovery_candidate_acceptable(frame.observed_at, candidate):
            self._reset_global_reacquire()
            return None

        if self._is_consistent_with_anchor(
            candidate.reference_timestamp,
            self._global_reacquire_anchor_timestamp,
            frame.observed_at,
            self._global_reacquire_anchor_observed_at,
        ):
            self._global_reacquire_streak += 1
        else:
            self._global_reacquire_streak = 1
            self._global_reacquire_start_observed_at = frame.observed_at
        if self._global_reacquire_start_observed_at is None:
            self._global_reacquire_start_observed_at = frame.observed_at
        self._global_reacquire_anchor_timestamp = candidate.reference_timestamp
        self._global_reacquire_anchor_observed_at = frame.observed_at

        reacquire_duration = frame.observed_at - self._global_reacquire_start_observed_at
        if (
            self._global_reacquire_streak < self._config.search_min_consecutive_matches
            or reacquire_duration < self._config.search_min_duration_seconds
        ):
            return None

        previous_accepted_frame = self._last_accepted_frame
        self._last_accepted_frame = candidate.reference_frame
        self._last_accepted_timestamp = candidate.reference_timestamp
        self._update_timeline_origin(frame.observed_at, candidate.reference_timestamp)
        self._tracking_miss_count = 0
        self._lost_count = 0
        self._reset_recovery()
        self._reset_search()
        self._reset_global_reacquire()
        self._state = "TRACKING"
        self._record_decision(
            previous_state,
            candidate,
            accepted=True,
            reason="recovery_reacquired_global_lock",
            previous_accepted_frame=previous_accepted_frame,
        )
        return candidate

    def _record_decision(
        self,
        previous_state: str,
        candidate: MatchResult,
        *,
        accepted: bool,
        reason: str,
        previous_accepted_frame: int | None = None,
    ) -> None:
        delta = None
        anchor = (
            previous_accepted_frame
            if previous_accepted_frame is not None
            else self._last_accepted_frame
        )
        if anchor is not None:
            delta = candidate.reference_frame - anchor
        matcher_decision = getattr(self._matcher, "last_decision", None)
        self._last_decision = TrackingDecision(
            previous_state=previous_state,
            state=self._state,
            candidate_frame=candidate.reference_frame,
            candidate_timestamp=candidate.reference_timestamp,
            candidate_confidence=candidate.confidence,
            candidate_valid=candidate.valid,
            accepted=accepted,
            reason=reason,
            delta_from_last_accepted=delta,
            matcher_reason=(
                getattr(matcher_decision, "reason", None) if matcher_decision is not None else None
            ),
            matcher_second_time_gap_seconds=(
                getattr(matcher_decision, "second_time_gap_seconds", None)
                if matcher_decision is not None
                else None
            ),
            matcher_second_distance_gap=(
                getattr(matcher_decision, "second_distance_gap", None)
                if matcher_decision is not None
                else None
            ),
        )

    def _prepare_matcher_window(self, observed_at: float) -> None:
        clear_window = getattr(self._matcher, "clear_expected_window", None)
        set_window = getattr(self._matcher, "set_expected_window", None)
        if not callable(set_window):
            return
        if (
            self._state in ("UNINITIALIZED", "SEARCHING")
            or self._timeline_origin_observed_at is None
        ):
            if callable(clear_window):
                clear_window()
            return
        if self._state == "UNCERTAIN":
            if callable(clear_window):
                clear_window()
            return
        expected_timestamp = observed_at - self._timeline_origin_observed_at
        window_seconds = (
            self._config.tracking_match_window_seconds
            if self._state == "TRACKING"
            else self._config.recovery_match_window_seconds
        )
        if self._anchored_timeline_active:
            window_seconds = min(
                window_seconds,
                max(
                    self._config.anchored_timeline_tolerance_seconds,
                    self._config.max_forward_jump_seconds,
                    self._config.max_backward_jump_seconds,
                ),
            )
        set_window(expected_timestamp, window_seconds)

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
