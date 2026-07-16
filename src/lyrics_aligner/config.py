from __future__ import annotations

import os
from dataclasses import dataclass


def _read_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _read_optional_device(name: str) -> str | int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value) if value.isdigit() else value


@dataclass(frozen=True, slots=True)
class AppConfig:
    audio_source: str = "simulated"
    sample_rate: int = 16_000
    channels: int = 1
    block_size: int = 4_096
    audio_queue_capacity: int = 16
    input_device: str | int | None = None
    simulation_duration_seconds: float = 2.0
    diagnostics_interval_seconds: float = 0.5
    silence_threshold_rms: float = 0.01
    clipping_threshold_peak: float = 0.99
    osc_host: str = "127.0.0.1"
    osc_port: int = 7_000
    osc_path: str = "/presentation/trigger-slide"

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            audio_source=_read_str("LYRICS_ALIGNER_AUDIO_SOURCE", "simulated"),
            sample_rate=_read_int("LYRICS_ALIGNER_SAMPLE_RATE", 16_000),
            channels=_read_int("LYRICS_ALIGNER_CHANNELS", 1),
            block_size=_read_int("LYRICS_ALIGNER_BLOCK_SIZE", 4_096),
            audio_queue_capacity=_read_int("LYRICS_ALIGNER_AUDIO_QUEUE_CAPACITY", 16),
            input_device=_read_optional_device("LYRICS_ALIGNER_INPUT_DEVICE"),
            simulation_duration_seconds=_read_float(
                "LYRICS_ALIGNER_SIMULATION_DURATION_SECONDS",
                2.0,
            ),
            diagnostics_interval_seconds=_read_float(
                "LYRICS_ALIGNER_DIAGNOSTICS_INTERVAL_SECONDS",
                0.5,
            ),
            silence_threshold_rms=_read_float(
                "LYRICS_ALIGNER_SILENCE_THRESHOLD_RMS",
                0.01,
            ),
            clipping_threshold_peak=_read_float(
                "LYRICS_ALIGNER_CLIPPING_THRESHOLD_PEAK",
                0.99,
            ),
            osc_host=_read_str("LYRICS_ALIGNER_OSC_HOST", "127.0.0.1"),
            osc_port=_read_int("LYRICS_ALIGNER_OSC_PORT", 7_000),
            osc_path=_read_str(
                "LYRICS_ALIGNER_OSC_PATH",
                "/presentation/trigger-slide",
            ),
        )
