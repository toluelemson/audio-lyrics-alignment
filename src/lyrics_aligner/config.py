from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    sample_rate: int = 16_000
    channels: int = 1
    block_size: int = 4_096
    audio_queue_capacity: int = 16
    osc_host: str = "127.0.0.1"
    osc_port: int = 7_000
    osc_path: str = "/presentation/trigger-slide"
