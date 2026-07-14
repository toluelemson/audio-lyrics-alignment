"""Live microphone audio input through sounddevice."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event
from time import monotonic
from typing import Protocol, cast

import numpy as np

from lyrics_aligner.domain.models import AudioChunk


class AudioStream(Protocol):
    def __enter__(self) -> AudioStream: ...

    def __exit__(
        self,
        exc_type: object | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None: ...


StreamFactory = Callable[..., AudioStream]


@dataclass(frozen=True, slots=True)
class MicrophoneAudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    block_size: int = 4_096
    device: str | int | None = None
    dtype: str = "float32"
    queue_timeout_seconds: float = 0.25


class MicrophoneAudioSource:
    """Capture mono float32 chunks from a live audio input device."""

    def __init__(
        self,
        config: MicrophoneAudioConfig | None = None,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self._config = config or MicrophoneAudioConfig()
        self._stream_factory = stream_factory
        self._chunks: Queue[AudioChunk | None] = Queue()
        self._stop_requested = Event()
        self._sequence_number = 0

    @property
    def device_name(self) -> str:
        if self._config.device is None:
            return "Microphone"
        return str(self._config.device)

    def chunks(self) -> Iterator[AudioChunk]:
        self._stop_requested.clear()
        stream_factory = self._stream_factory or self._default_stream_factory()
        with stream_factory(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            blocksize=self._config.block_size,
            dtype=self._config.dtype,
            device=self._config.device,
            callback=self._on_audio,
        ):
            while True:
                try:
                    item = self._chunks.get(timeout=self._config.queue_timeout_seconds)
                except Empty:
                    if self._stop_requested.is_set():
                        return
                    continue

                if item is None:
                    return
                yield item

    def stop(self) -> None:
        self._stop_requested.set()
        self._chunks.put_nowait(None)

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del frames, time_info, status
        if self._stop_requested.is_set():
            return

        samples = np.asarray(indata, dtype=np.float32)
        mono = samples[:, 0] if samples.ndim == 2 else samples

        self._chunks.put_nowait(
            AudioChunk(
                samples=np.array(mono, dtype=np.float32, copy=True),
                captured_at=monotonic(),
                sequence_number=self._sequence_number,
            )
        )
        self._sequence_number += 1

    @staticmethod
    def _default_stream_factory() -> StreamFactory:
        import sounddevice as sd  # type: ignore[import-untyped]

        return cast(StreamFactory, sd.InputStream)
