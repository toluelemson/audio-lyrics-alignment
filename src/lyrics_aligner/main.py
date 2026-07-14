import logging

from lyrics_aligner.adapters.audio import (
    MicrophoneAudioConfig,
    MicrophoneAudioSource,
    SimulatedAudioConfig,
    SimulatedAudioSource,
)
from lyrics_aligner.application import AudioIngestionRuntime
from lyrics_aligner.config import AppConfig
from lyrics_aligner.ports.audio_source import AudioSource


def _build_audio_source(config: AppConfig) -> AudioSource:
    if config.audio_source == "microphone":
        return MicrophoneAudioSource(
            MicrophoneAudioConfig(
                sample_rate=config.sample_rate,
                channels=config.channels,
                block_size=config.block_size,
                device=config.input_device,
            )
        )
    if config.audio_source == "simulated":
        return SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=config.sample_rate,
                block_size=config.block_size,
                duration=config.simulation_duration_seconds,
            )
        )
    raise ValueError(f"Unsupported audio_source: {config.audio_source}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = AppConfig()
    logger = logging.getLogger(__name__)
    source = _build_audio_source(config)
    runtime = AudioIngestionRuntime(
        source=source,
        queue_capacity=config.audio_queue_capacity,
        logger=logger,
        diagnostics_interval_seconds=config.diagnostics_interval_seconds,
        device_name=getattr(source, "device_name", source.__class__.__name__),
        silence_threshold_rms=config.silence_threshold_rms,
        clipping_threshold_peak=config.clipping_threshold_peak,
    )

    report = runtime.run()
    logger.info(
        "Audio ingestion finished source=%s sample_rate=%s block_size=%s chunks_received=%s chunks_dropped=%s silent_chunks=%s clipped_chunks=%s queue_high_water_mark=%s",
        config.audio_source,
        config.sample_rate,
        config.block_size,
        report.metrics.chunks_received,
        report.metrics.chunks_dropped,
        report.metrics.silent_chunks,
        report.metrics.clipped_chunks,
        report.metrics.queue_high_water_mark,
    )


if __name__ == "__main__":
    main()
