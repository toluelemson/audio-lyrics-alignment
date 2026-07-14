import logging

from lyrics_aligner.adapters.audio.simulated import SimulatedAudioConfig, SimulatedAudioSource
from lyrics_aligner.application import AudioIngestionRuntime
from lyrics_aligner.config import AppConfig


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = AppConfig()
    logger = logging.getLogger(__name__)
    source = SimulatedAudioSource(
        SimulatedAudioConfig(
            sample_rate=config.sample_rate,
            block_size=config.block_size,
            duration=config.simulation_duration_seconds,
        )
    )
    runtime = AudioIngestionRuntime(
        source=source,
        queue_capacity=config.audio_queue_capacity,
        logger=logger,
        diagnostics_interval_seconds=config.diagnostics_interval_seconds,
    )

    report = runtime.run()
    logger.info(
        "Simulation finished sample_rate=%s block_size=%s chunks_received=%s chunks_dropped=%s queue_high_water_mark=%s",
        config.sample_rate,
        config.block_size,
        report.metrics.chunks_received,
        report.metrics.chunks_dropped,
        report.metrics.queue_high_water_mark,
    )


if __name__ == "__main__":
    main()
