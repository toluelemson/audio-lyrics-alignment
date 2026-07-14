import logging

import pytest

from lyrics_aligner.adapters.audio.simulated import SimulatedAudioConfig, SimulatedAudioSource
from lyrics_aligner.application.runtime import AudioIngestionRuntime, BoundedAudioQueue


def test_bounded_audio_queue_drops_oldest_when_full() -> None:
    source = SimulatedAudioSource(
        SimulatedAudioConfig(sample_rate=1_000, block_size=4, duration=0.012, frequency=100)
    )
    queue = BoundedAudioQueue(capacity=2)

    chunks = list(source.chunks())

    assert queue.put_drop_oldest(chunks[0]) is False
    assert queue.put_drop_oldest(chunks[1]) is False
    assert queue.put_drop_oldest(chunks[2]) is True

    assert queue.get().sequence_number == 1
    assert queue.get().sequence_number == 2
    assert queue.get(timeout=0.001) is None


def test_runtime_consumes_simulated_audio_and_returns_metrics(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.03,
                frequency=100,
                amplitude=0.25,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime"),
        diagnostics_interval_seconds=0.01,
    )

    report = runtime.run()

    assert report.metrics.chunks_received == 3
    assert report.metrics.chunks_dropped == 0
    assert report.metrics.queue_high_water_mark >= 1
    assert report.queue_size == 0
    assert report.peak == pytest.approx(0.25, rel=0.05)
    assert report.rms > 0
    assert "device=SimulatedAudioSource" in caplog.text


def test_bounded_audio_queue_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError):
        BoundedAudioQueue(capacity=0)
