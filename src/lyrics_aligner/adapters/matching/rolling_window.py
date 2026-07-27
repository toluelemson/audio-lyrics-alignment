"""Rolling-window feature matching for short-sequence alignment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from lyrics_aligner.domain.models import (
    CorrectionAnchor,
    FeatureFrame,
    MatchResult,
    ReferenceProfile,
)


@dataclass(frozen=True, slots=True)
class RollingWindowFeatureMatcherConfig:
    confidence_threshold: float = 0.6
    ambiguity_distance_margin: float = 0.03
    ambiguity_min_separation_seconds: float = 15.0
    min_sequence_frames: int = 3
    max_sequence_frames: int = 10
    anchor_activation_tolerance_seconds: float = 1.5
    anchor_target_tolerance_seconds: float = 1.5
    anchor_bias: float = 0.08

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.ambiguity_distance_margin < 0.0:
            raise ValueError("ambiguity_distance_margin must be non-negative")
        if self.ambiguity_min_separation_seconds < 0.0:
            raise ValueError("ambiguity_min_separation_seconds must be non-negative")
        if self.min_sequence_frames <= 0:
            raise ValueError("min_sequence_frames must be greater than zero")
        if self.max_sequence_frames < self.min_sequence_frames:
            raise ValueError("max_sequence_frames must be at least min_sequence_frames")
        if self.anchor_activation_tolerance_seconds < 0.0:
            raise ValueError("anchor_activation_tolerance_seconds must be non-negative")
        if self.anchor_target_tolerance_seconds < 0.0:
            raise ValueError("anchor_target_tolerance_seconds must be non-negative")
        if self.anchor_bias < 0.0:
            raise ValueError("anchor_bias must be non-negative")


@dataclass(frozen=True, slots=True)
class RollingWindowDecision:
    best_index: int
    best_timestamp: float
    best_distance: float
    best_confidence: float
    second_index: int | None
    second_timestamp: float | None
    second_distance: float | None
    second_confidence: float | None
    second_distance_gap: float | None
    second_time_gap_seconds: float | None
    ambiguous: bool
    reason: str
    sequence_length_frames: int


class RollingWindowFeatureMatcher:
    """Match recent live feature sequences against contiguous reference windows."""

    def __init__(
        self,
        profile: ReferenceProfile,
        config: RollingWindowFeatureMatcherConfig | None = None,
        correction_anchors: tuple[CorrectionAnchor, ...] = (),
    ) -> None:
        if not profile.frames:
            raise ValueError("reference profile must contain at least one frame")

        self._profile = profile
        self._config = config or RollingWindowFeatureMatcherConfig()
        self._reference_matrix = np.stack(
            [np.asarray(frame.values, dtype=np.float32) for frame in profile.frames],
            axis=0,
        )
        self._feature_size = int(self._reference_matrix.shape[1])
        self._reference_timestamps = np.array(
            [float(frame.observed_at) for frame in profile.frames],
            dtype=np.float32,
        )
        self._live_window: deque[np.ndarray] = deque(maxlen=self._config.max_sequence_frames)
        self._last_decision: RollingWindowDecision | None = None
        self._preferred_timestamp: float | None = None
        self._preferred_window_seconds: float | None = None
        self._correction_anchors = correction_anchors

    @property
    def last_decision(self) -> RollingWindowDecision | None:
        return self._last_decision

    def reset(self) -> None:
        self._live_window.clear()
        self._last_decision = None

    def set_expected_window(
        self,
        preferred_timestamp: float,
        window_seconds: float,
    ) -> None:
        self._preferred_timestamp = preferred_timestamp
        self._preferred_window_seconds = max(0.0, window_seconds)

    def clear_expected_window(self) -> None:
        self._preferred_timestamp = None
        self._preferred_window_seconds = None

    def set_correction_anchors(self, anchors: tuple[CorrectionAnchor, ...]) -> None:
        self._correction_anchors = anchors

    def match(self, frame: FeatureFrame) -> MatchResult:
        values = np.asarray(frame.values, dtype=np.float32)
        if values.ndim != 1 or values.size != self._feature_size or not np.isfinite(values).all():
            self.reset()
            return MatchResult(
                reference_frame=-1,
                reference_timestamp=0.0,
                raw_distance=float("inf"),
                normalized_distance=float("inf"),
                confidence=0.0,
                valid=False,
            )

        self._live_window.append(np.array(values, dtype=np.float32, copy=True))
        sequence_length = len(self._live_window)
        if sequence_length < self._config.min_sequence_frames:
            self._last_decision = None
            return MatchResult(
                reference_frame=-1,
                reference_timestamp=0.0,
                raw_distance=float("inf"),
                normalized_distance=float("inf"),
                confidence=0.0,
                valid=False,
            )

        live_matrix = np.stack(tuple(self._live_window), axis=0)
        candidate_indices = self._candidate_indices(sequence_length)
        distances = np.array(
            [
                self._window_distance(live_matrix, end_index)
                for end_index in candidate_indices
            ],
            dtype=np.float32,
        )
        distances = self._apply_correction_anchor_bias(candidate_indices, distances)
        best_offset = int(np.argmin(distances))
        best_index = int(candidate_indices[best_offset])
        normalized_distance = float(distances[best_offset])
        raw_distance = float(normalized_distance * np.sqrt(float(self._feature_size)))
        confidence = float(max(0.0, 1.0 - normalized_distance))
        (
            second_index,
            second_distance,
            second_distance_gap,
            second_time_gap_seconds,
        ) = self._second_best(best_offset, candidate_indices, distances)
        second_confidence = None
        if second_distance is not None:
            second_confidence = float(max(0.0, 1.0 - second_distance))
        valid = bool(confidence >= self._config.confidence_threshold)
        ambiguous = False
        reason = "accepted"
        if not valid:
            reason = "below_confidence_threshold"
        elif self._is_ambiguous(second_distance_gap, second_time_gap_seconds):
            valid = False
            ambiguous = True
            reason = "ambiguous_distant_second_best"

        best_frame = self._profile.frames[best_index]
        self._last_decision = RollingWindowDecision(
            best_index=best_index,
            best_timestamp=float(best_frame.observed_at),
            best_distance=raw_distance,
            best_confidence=confidence,
            second_index=second_index,
            second_timestamp=(
                float(self._reference_timestamps[second_index])
                if second_index is not None
                else None
            ),
            second_distance=second_distance,
            second_confidence=second_confidence,
            second_distance_gap=second_distance_gap,
            second_time_gap_seconds=second_time_gap_seconds,
            ambiguous=ambiguous,
            reason=reason,
            sequence_length_frames=sequence_length,
        )
        return MatchResult(
            reference_frame=best_index,
            reference_timestamp=best_frame.observed_at,
            raw_distance=raw_distance,
            normalized_distance=normalized_distance,
            confidence=confidence,
            valid=valid,
        )

    def _candidate_indices(self, sequence_length: int) -> np.ndarray:
        minimum_end_index = sequence_length - 1
        if self._preferred_timestamp is None or self._preferred_window_seconds is None:
            return np.arange(minimum_end_index, self._reference_timestamps.size, dtype=np.int32)
        lower = self._preferred_timestamp - self._preferred_window_seconds
        upper = self._preferred_timestamp + self._preferred_window_seconds
        mask = (self._reference_timestamps >= lower) & (self._reference_timestamps <= upper)
        mask[:minimum_end_index] = False
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return np.arange(minimum_end_index, self._reference_timestamps.size, dtype=np.int32)
        return indices.astype(np.int32, copy=False)

    def _window_distance(self, live_matrix: np.ndarray, end_index: int) -> float:
        start_index = end_index - live_matrix.shape[0] + 1
        reference_window = self._reference_matrix[start_index : end_index + 1]
        frame_distances = np.linalg.norm(reference_window - live_matrix, axis=1)
        return float(np.mean(frame_distances / np.sqrt(float(self._feature_size))))

    def _apply_correction_anchor_bias(
        self,
        candidate_indices: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        if not self._correction_anchors or candidate_indices.size == 0:
            return distances
        adjusted = np.array(distances, copy=True)
        base_best_offset = int(np.argmin(distances))
        base_best_timestamp = float(self._reference_timestamps[int(candidate_indices[base_best_offset])])
        for anchor in self._correction_anchors:
            if (
                abs(base_best_timestamp - anchor.source_reference_timestamp)
                > self._config.anchor_activation_tolerance_seconds
            ):
                continue
            target_mask = np.abs(
                self._reference_timestamps[candidate_indices] - anchor.target_reference_timestamp
            ) <= self._config.anchor_target_tolerance_seconds
            if not np.any(target_mask):
                continue
            support_units = anchor.support_score if anchor.support_score > 0.0 else float(anchor.correction_count)
            bias = min(support_units, 5.0) * self._config.anchor_bias
            adjusted[target_mask] = np.maximum(0.0, adjusted[target_mask] - bias)
        return adjusted

    def _second_best(
        self,
        best_offset: int,
        candidate_indices: np.ndarray,
        distances: np.ndarray,
    ) -> tuple[int | None, float | None, float | None, float | None]:
        if distances.size < 2:
            return None, None, None, None
        best_distance = float(distances[best_offset])
        best_index = int(candidate_indices[best_offset])
        for offset in np.argsort(distances):
            candidate_offset = int(offset)
            if candidate_offset == best_offset:
                continue
            candidate_index = int(candidate_indices[candidate_offset])
            candidate_distance = float(distances[candidate_offset])
            timestamp_gap = abs(
                float(self._reference_timestamps[candidate_index])
                - float(self._reference_timestamps[best_index])
            )
            return (
                candidate_index,
                candidate_distance,
                candidate_distance - best_distance,
                timestamp_gap,
            )
        return None, None, None, None

    def _is_ambiguous(
        self,
        second_distance_gap: float | None,
        second_time_gap_seconds: float | None,
    ) -> bool:
        if second_distance_gap is None or second_time_gap_seconds is None:
            return False
        return (
            second_distance_gap <= self._config.ambiguity_distance_margin
            and second_time_gap_seconds >= self._config.ambiguity_min_separation_seconds
        )
