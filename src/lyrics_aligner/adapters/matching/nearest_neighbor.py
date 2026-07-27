"""Baseline nearest-neighbor feature matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lyrics_aligner.domain.models import FeatureFrame, MatchResult, ReferenceProfile


@dataclass(frozen=True, slots=True)
class NearestNeighborFeatureMatcherConfig:
    confidence_threshold: float = 0.6
    ambiguity_distance_margin: float = 0.03
    ambiguity_min_separation_seconds: float = 15.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.ambiguity_distance_margin < 0.0:
            raise ValueError("ambiguity_distance_margin must be non-negative")
        if self.ambiguity_min_separation_seconds < 0.0:
            raise ValueError("ambiguity_min_separation_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class NearestNeighborDecision:
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


class NearestNeighborFeatureMatcher:
    """Match live frames to the closest reference frame in feature space."""

    def __init__(
        self,
        profile: ReferenceProfile,
        config: NearestNeighborFeatureMatcherConfig | None = None,
    ) -> None:
        if not profile.frames:
            raise ValueError("reference profile must contain at least one frame")

        self._profile = profile
        self._config = config or NearestNeighborFeatureMatcherConfig()
        self._reference_matrix = np.stack(
            [np.asarray(frame.values, dtype=np.float32) for frame in profile.frames],
            axis=0,
        )
        self._feature_size = int(self._reference_matrix.shape[1])
        self._reference_timestamps = np.array(
            [float(frame.observed_at) for frame in profile.frames],
            dtype=np.float32,
        )
        self._last_decision: NearestNeighborDecision | None = None
        self._preferred_timestamp: float | None = None
        self._preferred_window_seconds: float | None = None

    @property
    def last_decision(self) -> NearestNeighborDecision | None:
        return self._last_decision

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

    def match(self, frame: FeatureFrame) -> MatchResult:
        values = np.asarray(frame.values, dtype=np.float32)
        if values.ndim != 1 or values.size != self._feature_size or not np.isfinite(values).all():
            self._last_decision = None
            return MatchResult(
                reference_frame=-1,
                reference_timestamp=0.0,
                raw_distance=float("inf"),
                normalized_distance=float("inf"),
                confidence=0.0,
                valid=False,
            )

        candidate_indices = self._candidate_indices()
        reference_matrix = self._reference_matrix[candidate_indices]
        deltas = reference_matrix - values
        distances = np.linalg.norm(deltas, axis=1)
        best_offset = int(np.argmin(distances))
        best_index = int(candidate_indices[best_offset])
        raw_distance = float(distances[best_offset])
        normalized_distance = float(raw_distance / np.sqrt(float(self._feature_size)))
        confidence = float(max(0.0, 1.0 - normalized_distance))
        best_frame = self._profile.frames[best_index]
        (
            second_index,
            second_distance,
            second_distance_gap,
            second_time_gap_seconds,
        ) = self._second_best(best_offset, candidate_indices, distances)
        second_confidence = None
        if second_distance is not None:
            second_confidence = float(
                max(0.0, 1.0 - (second_distance / np.sqrt(float(self._feature_size))))
            )
        valid = bool(confidence >= self._config.confidence_threshold)
        ambiguous = False
        reason = "accepted"
        if not valid:
            reason = "below_confidence_threshold"
        elif self._is_ambiguous(second_distance_gap, second_time_gap_seconds):
            valid = False
            ambiguous = True
            reason = "ambiguous_distant_second_best"
        self._last_decision = NearestNeighborDecision(
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
        )
        return MatchResult(
            reference_frame=best_index,
            reference_timestamp=best_frame.observed_at,
            raw_distance=raw_distance,
            normalized_distance=normalized_distance,
            confidence=confidence,
            valid=valid,
        )

    def _second_best(
        self,
        best_offset: int,
        candidate_indices: np.ndarray,
        distances: np.ndarray,
    ) -> tuple[int | None, float | None, float | None, float | None]:
        if distances.size < 2:
            return None, None, None, None
        best_distance = float(distances[best_offset])
        sorted_offsets = np.argsort(distances)
        best_index = int(candidate_indices[best_offset])
        for offset in sorted_offsets:
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

    def _candidate_indices(self) -> np.ndarray:
        if self._preferred_timestamp is None or self._preferred_window_seconds is None:
            return np.arange(self._reference_timestamps.size, dtype=np.int32)
        lower = self._preferred_timestamp - self._preferred_window_seconds
        upper = self._preferred_timestamp + self._preferred_window_seconds
        mask = (self._reference_timestamps >= lower) & (self._reference_timestamps <= upper)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return np.arange(self._reference_timestamps.size, dtype=np.int32)
        return indices.astype(np.int32, copy=False)

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
