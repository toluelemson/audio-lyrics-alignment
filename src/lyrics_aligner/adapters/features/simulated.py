"""Deterministic feature extraction for runtime and test wiring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lyrics_aligner.domain.models import AudioChunk, FeatureFrame


@dataclass(frozen=True, slots=True)
class SimulatedFeatureExtractorConfig:
    sample_rate: int = 16_000
    feature_size: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if isinstance(self.feature_size, bool) or not isinstance(self.feature_size, int):
            raise TypeError("feature_size must be an integer")
        if self.feature_size != 4:
            raise ValueError("feature_size must be exactly four for the simulated extractor")


class SimulatedFeatureExtractor:
    """Convert audio chunks into deterministic low-dimensional feature frames."""

    def __init__(self, config: SimulatedFeatureExtractorConfig | None = None) -> None:
        self._config = config or SimulatedFeatureExtractorConfig()

    def extract(self, chunk: AudioChunk) -> list[FeatureFrame]:
        if chunk.samples.size == 0:
            return []

        samples = np.asarray(chunk.samples, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32), dtype=np.float32)))
        peak = float(np.max(np.abs(samples)))
        mean_abs = float(np.mean(np.abs(samples), dtype=np.float32))
        zero_crossings = float(np.mean(samples[:-1] * samples[1:] < 0)) if samples.size > 1 else 0.0

        values = np.array(
            [rms, peak, mean_abs, zero_crossings],
            dtype=np.float32,
        )
        return [
            FeatureFrame(
                values=values,
                observed_at=chunk.captured_at,
                frame_duration_seconds=samples.size / self._config.sample_rate,
            )
        ]
