"""Stateful post-processing for stabilizing feature matches."""

from __future__ import annotations

from dataclasses import dataclass

from lyrics_aligner.domain.models import FeatureFrame, MatchResult
from lyrics_aligner.ports.feature_matcher import FeatureMatcher


@dataclass(frozen=True, slots=True)
class StabilizedFeatureMatcherConfig:
    max_forward_jump_frames: int = 4
    large_jump_threshold_frames: int = 2
    confirmation_count: int = 2

    def __post_init__(self) -> None:
        if self.max_forward_jump_frames < 0:
            raise ValueError("max_forward_jump_frames must be non-negative")
        if self.large_jump_threshold_frames < 0:
            raise ValueError("large_jump_threshold_frames must be non-negative")
        if self.confirmation_count <= 0:
            raise ValueError("confirmation_count must be greater than zero")


class StabilizedFeatureMatcher:
    """Wrap a matcher with simple monotonicity and confirmation rules."""

    def __init__(
        self,
        matcher: FeatureMatcher,
        config: StabilizedFeatureMatcherConfig | None = None,
    ) -> None:
        self._matcher = matcher
        self._config = config or StabilizedFeatureMatcherConfig()
        self._last_accepted_frame: int | None = None
        self._pending_frame: int | None = None
        self._pending_count = 0

    def match(self, frame: FeatureFrame) -> MatchResult:
        candidate = self._matcher.match(frame)
        if not candidate.valid:
            self._clear_pending()
            return candidate

        if self._last_accepted_frame is None:
            self._last_accepted_frame = candidate.reference_frame
            return candidate

        delta = candidate.reference_frame - self._last_accepted_frame
        if delta < 0 or delta > self._config.max_forward_jump_frames:
            self._clear_pending()
            return self._invalidate(candidate)

        if delta <= self._config.large_jump_threshold_frames:
            self._last_accepted_frame = candidate.reference_frame
            self._clear_pending()
            return candidate

        if candidate.reference_frame == self._pending_frame:
            self._pending_count += 1
        else:
            self._pending_frame = candidate.reference_frame
            self._pending_count = 1

        if self._pending_count >= self._config.confirmation_count:
            self._last_accepted_frame = candidate.reference_frame
            self._clear_pending()
            return candidate

        return self._invalidate(candidate)

    def _clear_pending(self) -> None:
        self._pending_frame = None
        self._pending_count = 0

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
