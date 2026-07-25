"""Replay mono audio chunks from a WAV file in real time."""

from __future__ import annotations

import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from time import monotonic

import numpy as np

from lyrics_aligner.domain.models import AudioChunk


@dataclass(frozen=True, slots=True)
class WavFileAudioConfig:
    path: str
    sample_rate: int = 16_000
    block_size: int = 4_096

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path must not be empty")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise TypeError("block_size must be an integer")
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero")


class WavFileAudioSource:
    """Replay PCM WAV samples as mono float32 chunks in wall-clock time."""

    def __init__(self, config: WavFileAudioConfig) -> None:
        self._config = config
        self._stopped = Event()

    @property
    def device_name(self) -> str:
        return Path(self._config.path).name

    def chunks(self) -> Iterator[AudioChunk]:
        self._stopped.clear()
        samples = _load_wav_mono(self._config.path)
        started_at = monotonic()

        for sequence_number, start in enumerate(range(0, samples.size, self._config.block_size)):
            captured_at = started_at + start / self._config.sample_rate
            delay = captured_at - monotonic()
            if self._stopped.wait(max(0.0, delay)):
                return

            stop = min(start + self._config.block_size, samples.size)
            yield AudioChunk(
                samples=np.array(samples[start:stop], dtype=np.float32, copy=True),
                captured_at=captured_at,
                sequence_number=sequence_number,
            )

    def stop(self) -> None:
        self._stopped.set()


def _load_wav_mono(path: str) -> np.ndarray:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {resolved_path}")

    with wave.open(str(resolved_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        pcm = handle.readframes(frame_count)

    if sample_width not in (1, 2, 4):
        raise ValueError("Only 8-bit, 16-bit, or 32-bit PCM WAV files are supported")

    if sample_width == 1:
        data = np.frombuffer(pcm, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768.0
    else:
        data = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2_147_483_648.0

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1, dtype=np.float32)

    mono = np.asarray(data, dtype=np.float32)
    if sample_rate == 16_000:
        return mono
    return _resample_audio(mono, source_sample_rate=sample_rate, target_sample_rate=16_000)


def _resample_audio(
    samples: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return np.asarray(samples, dtype=np.float32, copy=True)
    if samples.size == 0:
        return np.array([], dtype=np.float32)

    duration_seconds = samples.size / source_sample_rate
    target_length = max(1, int(round(duration_seconds * target_sample_rate)))
    source_positions = np.linspace(0.0, duration_seconds, num=samples.size, endpoint=False)
    target_positions = np.linspace(0.0, duration_seconds, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, samples)
    return np.asarray(resampled, dtype=np.float32)
