"""Coarse signature matching for approximate song-position lock."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lyrics_aligner.domain.models import FeatureFrame, MatchResult, ReferenceProfile


@dataclass(frozen=True, slots=True)
class CoarseFingerprintMatcherConfig:
    confidence_threshold: float = 0.82
    ambiguity_margin: float = 0.03
    window_frames: int = 24
    seed_cooldown_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.ambiguity_margin < 0.0:
            raise ValueError("ambiguity_margin must be non-negative")
        if self.window_frames <= 0:
            raise ValueError("window_frames must be greater than zero")
        if self.seed_cooldown_seconds < 0.0:
            raise ValueError("seed_cooldown_seconds must be non-negative")


class CoarseFingerprintMatcher:
    """Match a rolling live window against precomputed coarse signatures."""

    def __init__(
        self,
        profile: ReferenceProfile,
        config: CoarseFingerprintMatcherConfig | None = None,
    ) -> None:
        self._profile = profile
        self._config = config or CoarseFingerprintMatcherConfig()
        profile_path = profile.metadata.get("profile_path")
        if not profile_path:
            raise ValueError("profile metadata must contain profile_path for coarse matching")
        base_path = Path(profile_path).expanduser().resolve()
        signatures_path = base_path / "coarse_signatures.npy"
        timestamps_path = base_path / "coarse_timestamps.npy"
        frame_indexes_path = base_path / "coarse_frame_indexes.npy"
        if not signatures_path.exists() or not timestamps_path.exists() or not frame_indexes_path.exists():
            raise FileNotFoundError("coarse profile artifacts are missing")
        self._signatures = np.asarray(np.load(signatures_path), dtype=np.float32)
        self._timestamps = np.asarray(np.load(timestamps_path), dtype=np.float32)
        self._frame_indexes = np.asarray(np.load(frame_indexes_path), dtype=np.int32)
        if self._signatures.ndim != 2 or self._signatures.shape[0] == 0:
            raise ValueError("coarse signatures must be a non-empty 2D matrix")
        self._window: deque[np.ndarray] = deque(maxlen=self._config.window_frames)

    def reset(self) -> None:
        self._window.clear()

    def match(self, frame: FeatureFrame) -> MatchResult:
        values = np.asarray(frame.values, dtype=np.float32)
        if values.ndim != 1 or values.size != self._signatures.shape[1] or not np.isfinite(values).all():
            return MatchResult(0, 0.0, float("inf"), float("inf"), 0.0, False)
        self._window.append(_normalize(values))
        if len(self._window) < self._config.window_frames:
            return MatchResult(0, 0.0, float("inf"), float("inf"), 0.0, False)

        query = _normalize(np.mean(np.stack(tuple(self._window), axis=0), axis=0, dtype=np.float32))
        similarities = self._signatures @ query
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        second_similarity = float(np.partition(similarities, -2)[-2]) if similarities.size > 1 else -1.0
        confidence = float(max(0.0, min(1.0, (best_similarity + 1.0) / 2.0)))
        valid = bool(
            confidence >= self._config.confidence_threshold
            and best_similarity - second_similarity >= self._config.ambiguity_margin
        )
        distance = float(1.0 - best_similarity)
        return MatchResult(
            reference_frame=int(self._frame_indexes[best_index]),
            reference_timestamp=float(self._timestamps[best_index]),
            raw_distance=distance,
            normalized_distance=distance,
            confidence=confidence,
            valid=valid,
        )


class CoarseAnchorFeatureMatcher:
    """Seed a fine matcher from coarse signatures when global search is unstable."""

    def __init__(
        self,
        matcher,
        coarse_matcher: CoarseFingerprintMatcher,
        config: CoarseFingerprintMatcherConfig | None = None,
    ) -> None:
        self._matcher = matcher
        self._coarse_matcher = coarse_matcher
        self._config = config or CoarseFingerprintMatcherConfig()
        self._last_seeded_at: float | None = None

    @property
    def state_name(self) -> str | None:
        return getattr(self._matcher, "state_name", None)

    @property
    def last_decision(self):
        return getattr(self._matcher, "last_decision", None)

    def reset(self) -> None:
        reset = getattr(self._matcher, "reset", None)
        if callable(reset):
            reset()
        self._coarse_matcher.reset()
        self._last_seeded_at = None

    def force_anchor(self, reference_timestamp: float, observed_at: float | None = None) -> None:
        force_anchor = getattr(self._matcher, "force_anchor", None)
        if callable(force_anchor):
            force_anchor(reference_timestamp, observed_at=observed_at)
        self._last_seeded_at = observed_at

    def match(self, frame: FeatureFrame) -> MatchResult:
        result = self._matcher.match(frame)
        state_name = getattr(self._matcher, "state_name", None)
        if state_name not in {"UNINITIALIZED", "SEARCHING"}:
            return result
        if result.valid:
            return result
        if (
            self._last_seeded_at is not None
            and frame.observed_at - self._last_seeded_at < self._config.seed_cooldown_seconds
        ):
            return result

        coarse = self._coarse_matcher.match(frame)
        if not coarse.valid:
            return result
        force_anchor = getattr(self._matcher, "force_anchor", None)
        if not callable(force_anchor):
            return result
        force_anchor(coarse.reference_timestamp, observed_at=frame.observed_at)
        self._last_seeded_at = frame.observed_at
        return self._matcher.match(frame)


def _normalize(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.zeros_like(vector)
    return vector / norm
