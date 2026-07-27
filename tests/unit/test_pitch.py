import numpy as np

from lyrics_aligner.adapters.features.pitch import estimate_normalized_pitch


def test_estimate_normalized_pitch_returns_zero_for_silence() -> None:
    value = estimate_normalized_pitch(
        np.zeros(128, dtype=np.float32),
        sample_rate=1_000,
    )

    assert value == 0.0


def test_estimate_normalized_pitch_detects_voiced_sine_wave() -> None:
    sample_rate = 8_000
    frequency_hz = 440.0
    time = np.arange(0, 0.05, 1 / sample_rate, dtype=np.float32)
    samples = np.sin(2.0 * np.pi * frequency_hz * time).astype(np.float32)

    value = estimate_normalized_pitch(samples, sample_rate=sample_rate)

    assert abs(value) < 0.1
