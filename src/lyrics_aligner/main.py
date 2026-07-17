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
    TrackingFeatureMatcher,
    TrackingFeatureMatcherConfig,
)
from lyrics_aligner.adapters.presentation import (
    CompositePresentationGateway,
    LoggingPresentationGateway,
    OscPresentationGateway,
    OscPresentationGatewayConfig,
)
from lyrics_aligner.adapters.reference_profiles import (
    FilesystemReferenceProfileRepository,
)
from lyrics_aligner.adapters.slides import (
    TimelineSlideResolver,
    TimelineSlideResolverConfig,
)
from lyrics_aligner.application import AudioIngestionRuntime
from lyrics_aligner.application.runtime import RuntimeReport
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
    return TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=config.match_confirmation_count,
            recovery_min_consecutive_matches=config.match_confirmation_count,
            tracking_confidence_threshold=config.match_confidence_threshold,
            recovery_confidence_threshold=config.tracking_recovery_confidence_threshold,
            max_forward_jump_frames=config.match_max_forward_jump_frames,
            max_backward_recovery_frames=config.match_large_jump_threshold_frames,
            lost_match_patience=config.tracking_lost_match_patience,
        ),
    )


def _build_slide_resolver(
    config: AppConfig,
    profile: ReferenceProfile | None,
) -> SlideResolver | None:
    if profile is None or not profile.slide_cues:
        return None
    return TimelineSlideResolver(
        profile,
        TimelineSlideResolverConfig(
            lookahead_seconds=config.slide_lookahead_seconds,
            cooldown_seconds=config.slide_trigger_cooldown_seconds,
            consecutive_match_count=config.slide_consecutive_match_count,
        ),
    )


def _build_presentation_gateway(
    config: AppConfig,
    logger: logging.Logger,
    profile: ReferenceProfile | None,
) -> PresentationGateway | None:
    if profile is None or not profile.slide_cues:
        return None
    if config.manual_override or config.presentation_mode == "none":
        return None

    gateways: list[PresentationGateway] = []
    if config.presentation_mode in {"logging", "both"}:
        gateways.append(LoggingPresentationGateway(logger))
    if config.presentation_mode in {"osc", "both"}:
        gateways.append(
            OscPresentationGateway(
                OscPresentationGatewayConfig(
                    host=config.osc_host,
                    port=config.osc_port,
                    path=config.osc_path,
                    retry_count=config.osc_retry_count,
                    retry_backoff_seconds=config.osc_retry_backoff_seconds,
                )
            )
        )
    if not gateways:
        return None
    if len(gateways) == 1:
        return gateways[0]
    return CompositePresentationGateway(*gateways)


def _validate_config(config: AppConfig) -> None:
    valid_presentation_modes = {"none", "logging", "osc", "both"}
    if config.presentation_mode not in valid_presentation_modes:
        raise ValueError(
            f"presentation_mode must be one of {sorted(valid_presentation_modes)!r}"
        )
    if config.presentation_command_queue_capacity <= 0:
        raise ValueError("presentation_command_queue_capacity must be greater than zero")
    if config.presentation_mode in {"osc", "both"}:
        OscPresentationGatewayConfig(
            host=config.osc_host,
            port=config.osc_port,
            path=config.osc_path,
            retry_count=config.osc_retry_count,
            retry_backoff_seconds=config.osc_retry_backoff_seconds,
        )
    if config.manual_override and config.presentation_mode == "osc":
        raise ValueError("manual_override cannot be combined with presentation_mode='osc'")


def _log_runtime_health(
    logger: logging.Logger,
    config: AppConfig,
    report: RuntimeReport,
) -> None:
    logger.info(
        "AUDIO: %s | MODEL: %s | ALIGNMENT: %s | POSITION: %s | CONFIDENCE: %.2f | "
        "SLIDE: %s | AUDIO_QUEUE: %s/%s | COMMAND_QUEUE: %s/%s | PRESENTATION: %s",
        "OK" if report.metrics.chunks_received > 0 else "NO DATA",
        config.feature_extractor.upper(),
        report.tracking_state or "IDLE",
        (
            f"{report.last_match.reference_timestamp:05.2f}s"
            if report.last_match is not None
            else "none"
        ),
        report.last_match.confidence if report.last_match is not None else 0.0,
        (
            report.last_slide_command.slide_number
            if report.last_slide_command is not None
            else "none"
        ),
        report.queue_size,
        report.queue_capacity,
        report.command_queue_size,
        report.command_queue_capacity,
        "MANUAL_OVERRIDE" if config.manual_override else config.presentation_mode.upper(),
    )


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
        "--presentation-mode",
        choices=("none", "logging", "osc", "both"),
        help="Select whether to log, emit OSC, do both, or suppress presentation output.",
    )
    parser.add_argument(
        "--manual-override",
        action="store_true",
        help="Disable presentation sending while keeping alignment and tracking active.",
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
    parser.add_argument(
        "--tracking-recovery-confidence-threshold",
        type=float,
        help="Confidence threshold used while recovering from an uncertain position.",
    )
    parser.add_argument(
        "--tracking-lost-match-patience",
        type=int,
        help="Number of bad frames tolerated before falling back to searching mode.",
    )
    parser.add_argument(
        "--slide-lookahead-seconds",
        type=float,
        help="Allow slide cues to trigger slightly before the exact cue timestamp.",
    )
    parser.add_argument(
        "--slide-trigger-cooldown-seconds",
        type=float,
        help="Minimum reference-time gap between emitted slide commands.",
    )
    parser.add_argument(
        "--slide-consecutive-match-count",
        type=int,
        help="Number of stable cue-reaching matches required before emitting a slide.",
    )
    parser.add_argument(
        "--presentation-command-queue-capacity",
        type=int,
        help="Capacity of the non-blocking presentation command queue.",
    )
    parser.add_argument(
        "--osc-host",
        help="OSC destination host.",
    )
    parser.add_argument(
        "--osc-port",
        type=int,
        help="OSC destination UDP port.",
    )
    parser.add_argument(
        "--osc-path",
        help="OSC path for emitted slide payloads.",
    )
    parser.add_argument(
        "--osc-retry-count",
        type=int,
        help="Number of retries after an OSC send failure.",
    )
    parser.add_argument(
        "--osc-retry-backoff-seconds",
        type=float,
        help="Backoff between OSC retries in seconds.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_env()
    input_device: str | int | None = config.input_device
    if args.input_device is not None:
        input_device = int(args.input_device) if args.input_device.isdigit() else args.input_device
    feature_model_path = args.feature_model_path or config.feature_model_path
    reference_profile_path = args.reference_profile_path or config.reference_profile_path
    presentation_mode = args.presentation_mode or config.presentation_mode
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
    tracking_recovery_confidence_threshold = (
        args.tracking_recovery_confidence_threshold
        if args.tracking_recovery_confidence_threshold is not None
        else config.tracking_recovery_confidence_threshold
    )
    tracking_lost_match_patience = (
        args.tracking_lost_match_patience
        if args.tracking_lost_match_patience is not None
        else config.tracking_lost_match_patience
    )
    slide_lookahead_seconds = (
        args.slide_lookahead_seconds
        if args.slide_lookahead_seconds is not None
        else config.slide_lookahead_seconds
    )
    slide_trigger_cooldown_seconds = (
        args.slide_trigger_cooldown_seconds
        if args.slide_trigger_cooldown_seconds is not None
        else config.slide_trigger_cooldown_seconds
    )
    slide_consecutive_match_count = (
        args.slide_consecutive_match_count
        if args.slide_consecutive_match_count is not None
        else config.slide_consecutive_match_count
    )
    presentation_command_queue_capacity = (
        args.presentation_command_queue_capacity
        if args.presentation_command_queue_capacity is not None
        else config.presentation_command_queue_capacity
    )
    osc_host = args.osc_host or config.osc_host
    osc_port = args.osc_port if args.osc_port is not None else config.osc_port
    osc_path = args.osc_path or config.osc_path
    osc_retry_count = (
        args.osc_retry_count if args.osc_retry_count is not None else config.osc_retry_count
    )
    osc_retry_backoff_seconds = (
        args.osc_retry_backoff_seconds
        if args.osc_retry_backoff_seconds is not None
        else config.osc_retry_backoff_seconds
    )

    return AppConfig(
        audio_source=args.audio_source or config.audio_source,
        feature_extractor=args.feature_extractor or config.feature_extractor,
        reference_profile_path=reference_profile_path,
        presentation_mode=presentation_mode,
        manual_override=args.manual_override or config.manual_override,
        match_confidence_threshold=match_confidence_threshold,
        match_max_forward_jump_frames=match_max_forward_jump_frames,
        match_large_jump_threshold_frames=match_large_jump_threshold_frames,
        match_confirmation_count=match_confirmation_count,
        tracking_recovery_confidence_threshold=tracking_recovery_confidence_threshold,
        tracking_lost_match_patience=tracking_lost_match_patience,
        slide_lookahead_seconds=slide_lookahead_seconds,
        slide_trigger_cooldown_seconds=slide_trigger_cooldown_seconds,
        slide_consecutive_match_count=slide_consecutive_match_count,
        presentation_command_queue_capacity=presentation_command_queue_capacity,
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
        osc_host=osc_host,
        osc_port=osc_port,
        osc_path=osc_path,
        osc_retry_count=osc_retry_count,
        osc_retry_backoff_seconds=osc_retry_backoff_seconds,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = _config_from_args(_parse_args())
    _validate_config(config)
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
    slide_resolver = _build_slide_resolver(config, profile)
    presentation_gateway = _build_presentation_gateway(config, logger, profile)
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
        command_queue_capacity=config.presentation_command_queue_capacity,
    )

    report = runtime.run()
    _log_runtime_health(logger, config, report)
    logger.info(
        "Audio ingestion finished source=%s feature_extractor=%s presentation_mode=%s "
        "sample_rate=%s block_size=%s "
        "chunks_received=%s chunks_dropped=%s silent_chunks=%s "
        "clipped_chunks=%s feature_frames_processed=%s accepted_matches=%s "
        "low_confidence_matches=%s slide_triggers_sent=%s osc_send_failures=%s "
        "queue_high_water_mark=%s command_queue_capacity=%s",
        config.audio_source,
        config.feature_extractor,
        "manual_override" if config.manual_override else config.presentation_mode,
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
        report.command_queue_capacity,
    )


if __name__ == "__main__":
    main()
