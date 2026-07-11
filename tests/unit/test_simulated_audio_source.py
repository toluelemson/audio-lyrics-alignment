import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from lyrics_aligner.adapters.audio.simulated import (
    SimulatedAudioConfig,
    SimulatedAudioSource,
)
from lyrics_aligner.domain.models import AudioChunk
from lyrics_aligner.ports.audio_source import AudioSource


def test_produces_mono_float32_chunks() -> None:
    source = SimulatedAudioSource(
        SimulatedAudioConfig(sample_rate=16_000, block_size=160, duration=0.02)
    )

    chunks = list(source.chunks())

    assert [chunk.samples.shape for chunk in chunks] == [(160,), (160,)]
    assert all(chunk.samples.dtype == np.float32 for chunk in chunks)
    assert np.max(np.abs(chunks[0].samples)) <= 0.5


def test_sequence_numbers_and_timestamps_follow_sample_positions() -> None:
    source = SimulatedAudioSource(
        SimulatedAudioConfig(sample_rate=1_000, block_size=10, frequency=100, duration=0.03)
    )

    chunks = list(source.chunks())

    assert [chunk.sequence_number for chunk in chunks] == [0, 1, 2]
    assert chunks[1].captured_at - chunks[0].captured_at == pytest.approx(0.01)
    assert chunks[2].captured_at - chunks[1].captured_at == pytest.approx(0.01)


def test_sine_wave_is_deterministic() -> None:
    config = SimulatedAudioConfig(
        sample_rate=1_000, block_size=10, frequency=100, amplitude=0.25, duration=0.01
    )

    samples = next(SimulatedAudioSource(config).chunks()).samples
    expected = (0.25 * np.sin(2 * np.pi * 100 * np.arange(10) / 1_000)).astype(np.float32)

    np.testing.assert_array_equal(samples, expected)


def test_conforms_to_audio_source_port() -> None:
    source: AudioSource = SimulatedAudioSource(
        SimulatedAudioConfig(sample_rate=1_000, block_size=10, frequency=100, duration=0.01)
    )
    chunks: Iterator[AudioChunk] = source.chunks()

    assert next(chunks).sequence_number == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("sample_rate", 0, ValueError),
        ("sample_rate", 16_000.0, TypeError),
        ("block_size", 0, ValueError),
        ("block_size", True, TypeError),
        ("frequency", 0, ValueError),
        ("frequency", 8_000, ValueError),
        ("frequency", float("nan"), ValueError),
        ("amplitude", -0.1, ValueError),
        ("amplitude", 1.1, ValueError),
        ("duration", 0, ValueError),
        ("duration", float("inf"), ValueError),
    ],
)
def test_rejects_invalid_configuration(field: str, value: object, error: type[Exception]) -> None:
    values: dict[str, object] = {
        "sample_rate": 16_000,
        "block_size": 160,
        "frequency": 440.0,
        "amplitude": 0.5,
        "duration": 1.0,
    }
    values[field] = value

    with pytest.raises(error):
        SimulatedAudioConfig(**values)  # type: ignore[arg-type]


def test_stop_interrupts_waiting_iterator_promptly() -> None:
    source = SimulatedAudioSource(
        SimulatedAudioConfig(sample_rate=16_000, block_size=16_000, duration=10)
    )
    iterator = source.chunks()
    assert next(iterator).sequence_number == 0

    worker = threading.Thread(target=lambda: next(iterator, None))
    worker.start()
    time.sleep(0.02)
    source.stop()
    worker.join(timeout=0.2)

    assert not worker.is_alive()
