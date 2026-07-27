"""Lightweight pitch helpers for contour-aware feature extraction."""

from __future__ import annotations

import math

import numpy as np


def estimate_normalized_pitch(
    samples: np.ndarray,
    *,
    sample_rate: int,
    min_pitch_hz: float = 80.0,
    max_pitch_hz: float = 1_000.0,
    min_correlation: float = 0.3,
    silence_rms_threshold: float = 0.01,
) -> float:
    """Estimate a coarse normalized pitch from a short audio segment.

    Returns `0.0` when the segment is silent or not confidently voiced.
    For voiced segments, returns a clipped log-pitch value in the range [-1, 1].
    """
    values = np.asarray(samples, dtype=np.float32)
    if values.size < 4:
        return 0.0

    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float32), dtype=np.float32)))
    if rms <= silence_rms_threshold:
        return 0.0

    centered = values - float(np.mean(values, dtype=np.float32))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-8:
        return 0.0

    min_lag = max(1, int(sample_rate / max_pitch_hz))
    max_lag = min(values.size - 1, int(sample_rate / min_pitch_hz))
    if max_lag <= min_lag:
        return 0.0

    autocorrelation = np.correlate(centered, centered, mode="full")[values.size - 1 :]
    if autocorrelation.size <= max_lag:
        return 0.0

    normalized = autocorrelation / max(autocorrelation[0], 1e-8)
    lag_slice = normalized[min_lag : max_lag + 1]
    best_offset = int(np.argmax(lag_slice))
    best_lag = min_lag + best_offset
    best_correlation = float(lag_slice[best_offset])
    if best_correlation < min_correlation:
        return 0.0

    pitch_hz = sample_rate / best_lag
    if not np.isfinite(pitch_hz) or pitch_hz <= 0.0:
        return 0.0

    # Compress pitch into a bounded value so it acts as a backup cue, not a dominant signal.
    normalized_pitch = math.log2(pitch_hz / 440.0) / 2.0
    return float(np.clip(normalized_pitch, -1.0, 1.0))
