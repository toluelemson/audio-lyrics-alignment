"""Deterministic, real-time simulated audio input."""

from collections.abc import Iterator
from dataclasses import dataclass
from threading import Event
from time import monotonic

import numpy as np

from lyrics_aligner.domain.models import AudioChunk


@dataclass(frozen=True, slots=True)
class SimulatedAudioConfig:
    """Configuration for a finite sine-wave audio stream."""

    sample_rate: int = 16_000
    block_size: int = 4_096
    frequency: float = 440.0
    amplitude: float = 0.5
    duration: float = 10.0

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise TypeError("block_size must be an integer")
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        self._validate_finite_number("frequency", self.frequency)
        self._validate_finite_number("amplitude", self.amplitude)
        self._validate_finite_number("duration", self.duration)
        if self.frequency <= 0 or self.frequency >= self.sample_rate / 2:
            raise ValueError("frequency must be greater than zero and below the Nyquist frequency")
        if self.amplitude < 0 or self.amplitude > 1:
            raise ValueError("amplitude must be between zero and one")
        if self.duration <= 0:
            raise ValueError("duration must be greater than zero")

    @staticmethod
    def _validate_finite_number(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")


class SimulatedAudioSource:
    """Produce deterministic mono float32 sine-wave chunks in real time."""

    def __init__(self, config: SimulatedAudioConfig | None = None) -> None:
        self._config = config or SimulatedAudioConfig()
        self._stopped = Event()

    def chunks(self) -> Iterator[AudioChunk]:
        """Yield audio blocks at their simulated capture times until stopped."""
        started_at = monotonic()
        total_samples = round(self._config.duration * self._config.sample_rate)

        for sequence_number, start in enumerate(
            range(0, total_samples, self._config.block_size)
        ):
            captured_at = started_at + start / self._config.sample_rate
            delay = captured_at - monotonic()
            if self._stopped.wait(max(0.0, delay)):
                return

            stop = min(start + self._config.block_size, total_samples)
            sample_numbers = np.arange(start, stop, dtype=np.float64)
            phase = 2.0 * np.pi * self._config.frequency * sample_numbers
            samples = (self._config.amplitude * np.sin(phase / self._config.sample_rate)).astype(
                np.float32
            )
            yield AudioChunk(
                samples=samples,
                captured_at=captured_at,
                sequence_number=sequence_number,
            )

    def stop(self) -> None:
        """Stop iteration and wake any real-time wait immediately."""
        self._stopped.set()
