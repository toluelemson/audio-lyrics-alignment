"""Baseline nearest-neighbor feature matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lyrics_aligner.domain.models import FeatureFrame, MatchResult, ReferenceProfile


@dataclass(frozen=True, slots=True)
class NearestNeighborFeatureMatcherConfig:
    confidence_threshold: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")


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

    def match(self, frame: FeatureFrame) -> MatchResult:
        values = np.asarray(frame.values, dtype=np.float32)
        if values.ndim != 1 or values.size != self._feature_size or not np.isfinite(values).all():
            return MatchResult(
                reference_frame=-1,
                reference_timestamp=0.0,
                raw_distance=float("inf"),
                normalized_distance=float("inf"),
                confidence=0.0,
                valid=False,
            )

        deltas = self._reference_matrix - values
        distances = np.linalg.norm(deltas, axis=1)
        best_index = int(np.argmin(distances))
        raw_distance = float(distances[best_index])
        normalized_distance = float(raw_distance / np.sqrt(float(self._feature_size)))
        confidence = float(max(0.0, 1.0 - normalized_distance))
        best_frame = self._profile.frames[best_index]
        return MatchResult(
            reference_frame=best_index,
            reference_timestamp=best_frame.observed_at,
            raw_distance=raw_distance,
            normalized_distance=normalized_distance,
            confidence=confidence,
            valid=bool(confidence >= self._config.confidence_threshold),
        )
