from dataclasses import dataclass


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
