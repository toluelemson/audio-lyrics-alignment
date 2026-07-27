import threading

import numpy as np

from lyrics_aligner.adapters.audio.microphone import (
    MicrophoneAudioConfig,
    MicrophoneAudioSource,
)
from lyrics_aligner.domain.models import AudioChunk


class FakeInputStream:
    def __init__(self, **kwargs: object) -> None:
        self._callback = kwargs["callback"]
        self._entered = False

    def __enter__(self) -> "FakeInputStream":
        self._entered = True
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def emit(self, samples: np.ndarray) -> None:
        if not self._entered:
            raise RuntimeError("stream must be entered before emitting samples")
        self._callback(samples, samples.shape[0], None, None)


def test_emits_float32_mono_chunks_from_stream_callback() -> None:
    created_streams: list[FakeInputStream] = []
    stream_ready = threading.Event()

    def stream_factory(**kwargs: object) -> FakeInputStream:
        stream = FakeInputStream(**kwargs)
        created_streams.append(stream)
        stream_ready.set()
        return stream

    source = MicrophoneAudioSource(
        MicrophoneAudioConfig(sample_rate=1_000, channels=1, block_size=4),
        stream_factory=stream_factory,
    )

    iterator = source.chunks()
    results: list[AudioChunk] = []
    worker = threading.Thread(target=lambda: results.append(next(iterator)))
    worker.start()

    assert stream_ready.wait(timeout=0.2)
    created_streams[0].emit(np.array([[0.1], [-0.2], [0.3], [-0.4]], dtype=np.float32))
    worker.join(timeout=0.2)
    source.stop()

    assert results
    chunk = results[0]
    assert chunk.samples.dtype == np.float32
    np.testing.assert_array_equal(
        chunk.samples,
        np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
    )
    assert chunk.sequence_number == 0


def test_stop_unblocks_waiting_iterator() -> None:
    created_streams: list[FakeInputStream] = []
    stream_ready = threading.Event()

    def stream_factory(**kwargs: object) -> FakeInputStream:
        stream = FakeInputStream(**kwargs)
        created_streams.append(stream)
        stream_ready.set()
        return stream

    source = MicrophoneAudioSource(
        MicrophoneAudioConfig(
            sample_rate=1_000,
            channels=1,
            block_size=4,
            queue_timeout_seconds=0.01,
        ),
        stream_factory=stream_factory,
    )

    iterator = source.chunks()
    worker = threading.Thread(target=lambda: next(iterator, None))
    worker.start()

    assert stream_ready.wait(timeout=0.2)
    source.stop()
    worker.join(timeout=0.2)

    assert not worker.is_alive()
