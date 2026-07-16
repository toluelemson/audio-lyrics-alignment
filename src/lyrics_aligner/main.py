import argparse
import logging

from lyrics_aligner.adapters.audio import (
    MicrophoneAudioConfig,
    MicrophoneAudioSource,
    SimulatedAudioConfig,
    SimulatedAudioSource,
)
from lyrics_aligner.adapters.features import (
    OnnxFeatureExtractor,
    OnnxFeatureExtractorConfig,
    SimulatedFeatureExtractor,
    SimulatedFeatureExtractorConfig,
)
from lyrics_aligner.adapters.matching import (
    NearestNeighborFeatureMatcher,
    NearestNeighborFeatureMatcherConfig,
    StabilizedFeatureMatcher,
    StabilizedFeatureMatcherConfig,
)
from lyrics_aligner.adapters.presentation import LoggingPresentationGateway
from lyrics_aligner.adapters.reference_profiles import (
    FilesystemReferenceProfileRepository,
)
from lyrics_aligner.adapters.slides import TimelineSlideResolver
from lyrics_aligner.application import AudioIngestionRuntime
from lyrics_aligner.config import AppConfig
from lyrics_aligner.domain.models import ReferenceProfile
from lyrics_aligner.ports.audio_source import AudioSource
from lyrics_aligner.ports.feature_extractor import FeatureExtractor
from lyrics_aligner.ports.feature_matcher import FeatureMatcher
from lyrics_aligner.ports.presentation_gateway import PresentationGateway
from lyrics_aligner.ports.reference_profile_repository import (
    ReferenceProfileRepository,
)
from lyrics_aligner.ports.slide_resolver import SlideResolver


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


def _build_feature_extractor(config: AppConfig) -> FeatureExtractor:
    if config.feature_extractor == "simulated":
        return SimulatedFeatureExtractor(
            SimulatedFeatureExtractorConfig(sample_rate=config.sample_rate)
        )
    if config.feature_extractor == "onnx":
        if config.feature_model_path is None:
            raise ValueError("feature_model_path is required when feature_extractor=onnx")
        return OnnxFeatureExtractor(
            OnnxFeatureExtractorConfig(
                model_path=config.feature_model_path,
                sample_rate=config.sample_rate,
            )
        )
    raise ValueError(f"Unsupported feature_extractor: {config.feature_extractor}")


def _build_reference_profile_repository() -> ReferenceProfileRepository:
    return FilesystemReferenceProfileRepository()


def _build_feature_matcher(
    config: AppConfig,
    profile: ReferenceProfile | None,
) -> FeatureMatcher | None:
    if profile is None:
        return None
    matcher = NearestNeighborFeatureMatcher(
        profile,
        NearestNeighborFeatureMatcherConfig(
            confidence_threshold=config.match_confidence_threshold,
        ),
    )
    return StabilizedFeatureMatcher(
        matcher,
        StabilizedFeatureMatcherConfig(
            max_forward_jump_frames=config.match_max_forward_jump_frames,
            large_jump_threshold_frames=config.match_large_jump_threshold_frames,
            confirmation_count=config.match_confirmation_count,
        ),
    )


def _build_slide_resolver(profile: ReferenceProfile | None) -> SlideResolver | None:
    if profile is None or not profile.slide_cues:
        return None
    return TimelineSlideResolver(profile)


def _build_presentation_gateway(
    logger: logging.Logger,
    profile: ReferenceProfile | None,
) -> PresentationGateway | None:
    if profile is None or not profile.slide_cues:
        return None
    return LoggingPresentationGateway(logger)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the audio-to-lyrics ingestion runtime.",
    )
    parser.add_argument(
        "--audio-source",
        choices=("simulated", "microphone"),
        help="Choose the runtime audio input source.",
    )
    parser.add_argument(
        "--input-device",
        help="Audio input device name or numeric device index.",
    )
    parser.add_argument(
        "--feature-extractor",
        choices=("simulated", "onnx"),
        help="Choose the runtime feature extractor implementation.",
    )
    parser.add_argument(
        "--feature-model-path",
        help="Path to the ONNX feature extraction model when using --feature-extractor onnx.",
    )
    parser.add_argument(
        "--simulation-duration-seconds",
        type=float,
        help="Simulated audio duration when using the simulated source.",
    )
    parser.add_argument(
        "--reference-profile-path",
        help="Path to a prepared reference profile directory.",
    )
    parser.add_argument(
        "--match-confidence-threshold",
        type=float,
        help="Confidence threshold used to accept feature matches.",
    )
    parser.add_argument(
        "--match-max-forward-jump-frames",
        type=int,
        help="Largest forward frame jump accepted without rejecting the match.",
    )
    parser.add_argument(
        "--match-large-jump-threshold-frames",
        type=int,
        help="Forward jump size that begins requiring repeated confirmation.",
    )
    parser.add_argument(
        "--match-confirmation-count",
        type=int,
        help="Number of repeated large-jump matches required before acceptance.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_env()
    input_device: str | int | None = config.input_device
    if args.input_device is not None:
        input_device = int(args.input_device) if args.input_device.isdigit() else args.input_device
    feature_model_path = args.feature_model_path or config.feature_model_path
    reference_profile_path = args.reference_profile_path or config.reference_profile_path
    match_confidence_threshold = (
        args.match_confidence_threshold
        if args.match_confidence_threshold is not None
        else config.match_confidence_threshold
    )
    match_max_forward_jump_frames = (
        args.match_max_forward_jump_frames
        if args.match_max_forward_jump_frames is not None
        else config.match_max_forward_jump_frames
    )
    match_large_jump_threshold_frames = (
        args.match_large_jump_threshold_frames
        if args.match_large_jump_threshold_frames is not None
        else config.match_large_jump_threshold_frames
    )
    match_confirmation_count = (
        args.match_confirmation_count
        if args.match_confirmation_count is not None
        else config.match_confirmation_count
    )

    return AppConfig(
        audio_source=args.audio_source or config.audio_source,
        feature_extractor=args.feature_extractor or config.feature_extractor,
        reference_profile_path=reference_profile_path,
        match_confidence_threshold=match_confidence_threshold,
        match_max_forward_jump_frames=match_max_forward_jump_frames,
        match_large_jump_threshold_frames=match_large_jump_threshold_frames,
        match_confirmation_count=match_confirmation_count,
        sample_rate=config.sample_rate,
        channels=config.channels,
        block_size=config.block_size,
        audio_queue_capacity=config.audio_queue_capacity,
        input_device=input_device,
        feature_model_path=feature_model_path,
        simulation_duration_seconds=(
            args.simulation_duration_seconds
            if args.simulation_duration_seconds is not None
            else config.simulation_duration_seconds
        ),
        diagnostics_interval_seconds=config.diagnostics_interval_seconds,
        silence_threshold_rms=config.silence_threshold_rms,
        clipping_threshold_peak=config.clipping_threshold_peak,
        osc_host=config.osc_host,
        osc_port=config.osc_port,
        osc_path=config.osc_path,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = _config_from_args(_parse_args())
    logger = logging.getLogger(__name__)
    profile: ReferenceProfile | None = None
    if config.reference_profile_path is not None:
        repository = _build_reference_profile_repository()
        profile = repository.load(config.reference_profile_path)
        logger.info(
            "Loaded reference profile name=%s frames=%s path=%s",
            profile.name,
            len(profile.frames),
            config.reference_profile_path,
        )
    source = _build_audio_source(config)
    feature_extractor = _build_feature_extractor(config)
    feature_matcher = _build_feature_matcher(config, profile)
    slide_resolver = _build_slide_resolver(profile)
    presentation_gateway = _build_presentation_gateway(logger, profile)
    runtime = AudioIngestionRuntime(
        source=source,
        queue_capacity=config.audio_queue_capacity,
        logger=logger,
        feature_extractor=feature_extractor,
        feature_matcher=feature_matcher,
        slide_resolver=slide_resolver,
        presentation_gateway=presentation_gateway,
        diagnostics_interval_seconds=config.diagnostics_interval_seconds,
        device_name=getattr(source, "device_name", source.__class__.__name__),
        silence_threshold_rms=config.silence_threshold_rms,
        clipping_threshold_peak=config.clipping_threshold_peak,
    )

    report = runtime.run()
    logger.info(
        "Audio ingestion finished source=%s feature_extractor=%s sample_rate=%s block_size=%s "
        "chunks_received=%s chunks_dropped=%s silent_chunks=%s "
        "clipped_chunks=%s feature_frames_processed=%s accepted_matches=%s "
        "low_confidence_matches=%s slide_triggers_sent=%s osc_send_failures=%s "
        "queue_high_water_mark=%s",
        config.audio_source,
        config.feature_extractor,
        config.sample_rate,
        config.block_size,
        report.metrics.chunks_received,
        report.metrics.chunks_dropped,
        report.metrics.silent_chunks,
        report.metrics.clipped_chunks,
        report.metrics.feature_frames_processed,
        report.metrics.accepted_matches,
        report.metrics.low_confidence_matches,
        report.metrics.slide_triggers_sent,
        report.metrics.osc_send_failures,
        report.metrics.queue_high_water_mark,
    )


if __name__ == "__main__":
    main()
