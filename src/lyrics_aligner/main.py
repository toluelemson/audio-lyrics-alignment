from __future__ import annotations

import argparse
import logging

from lyrics_aligner.adapters.audio import (
    MicrophoneAudioConfig,
    MicrophoneAudioSource,
    WavFileAudioConfig,
    WavFileAudioSource,
)
from lyrics_aligner.adapters.features import (
    OnnxFeatureExtractor,
    OnnxFeatureExtractorConfig,
)
from lyrics_aligner.adapters.matching import (
    NearestNeighborFeatureMatcher,
    NearestNeighborFeatureMatcherConfig,
    TrackingFeatureMatcher,
    TrackingFeatureMatcherConfig,
)
from lyrics_aligner.adapters.presentation import (
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
from lyrics_aligner.ports.feature_matcher import FeatureMatcher
from lyrics_aligner.ports.presentation_gateway import PresentationGateway
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
    if config.audio_source == "wav":
        if config.audio_file_path is None:
            raise ValueError("audio_file_path is required when audio_source=wav")
        return WavFileAudioSource(
            WavFileAudioConfig(
                path=config.audio_file_path,
                sample_rate=config.sample_rate,
                block_size=config.block_size,
            )
        )
    raise ValueError("audio_source must be 'wav' or 'microphone'")


def _build_feature_matcher(
    config: AppConfig,
    profile: ReferenceProfile,
) -> FeatureMatcher:
    matcher = NearestNeighborFeatureMatcher(
        profile,
        NearestNeighborFeatureMatcherConfig(
            confidence_threshold=config.match_confidence_threshold,
            ambiguity_distance_margin=config.match_ambiguity_distance_margin,
            ambiguity_min_separation_seconds=config.match_ambiguity_min_separation_seconds,
        ),
    )
    return TrackingFeatureMatcher(
        matcher,
        TrackingFeatureMatcherConfig(
            search_min_consecutive_matches=config.match_confirmation_count,
            recovery_min_consecutive_matches=config.match_confirmation_count,
            search_miss_patience=config.tracking_search_miss_patience,
            tracking_miss_patience=config.tracking_miss_patience,
            search_min_duration_seconds=config.tracking_search_min_duration_seconds,
            recovery_min_duration_seconds=config.tracking_recovery_min_duration_seconds,
            tracking_confidence_threshold=config.match_confidence_threshold,
            recovery_confidence_threshold=config.tracking_recovery_confidence_threshold,
            tracking_match_window_seconds=config.tracking_match_window_seconds,
            recovery_match_window_seconds=config.recovery_match_window_seconds,
            expected_position_tolerance_seconds=(
                config.tracking_expected_position_tolerance_seconds
            ),
            anchored_timeline_tolerance_seconds=(
                config.tracking_anchored_timeline_tolerance_seconds
            ),
            max_forward_jump_seconds=config.tracking_max_forward_jump_seconds,
            max_backward_jump_seconds=config.tracking_max_backward_jump_seconds,
            lost_match_patience=config.tracking_lost_match_patience,
        ),
    )


def _build_slide_resolver(
    config: AppConfig,
    profile: ReferenceProfile,
) -> SlideResolver | None:
    if not profile.slide_cues:
        return None
    return TimelineSlideResolver(
        profile,
        TimelineSlideResolverConfig(
            lookahead_seconds=config.slide_lookahead_seconds,
            cooldown_seconds=config.slide_trigger_cooldown_seconds,
            consecutive_match_count=config.slide_consecutive_match_count,
            max_emit_lag_seconds=config.slide_max_emit_lag_seconds,
        ),
    )


def _build_presentation_gateway(
    config: AppConfig,
    logger: logging.Logger,
    profile: ReferenceProfile,
) -> PresentationGateway | None:
    if not profile.slide_cues or config.presentation_mode == "none":
        return None
    if config.presentation_mode == "logging":
        return LoggingPresentationGateway(logger)
    if config.presentation_mode == "osc":
        return OscPresentationGateway(
            OscPresentationGatewayConfig(
                host=config.osc_host,
                port=config.osc_port,
                path=config.osc_path,
                retry_count=config.osc_retry_count,
                retry_backoff_seconds=config.osc_retry_backoff_seconds,
            )
        )
    raise ValueError("presentation_mode must be 'logging', 'osc', or 'none'")


def _validate_config(config: AppConfig) -> None:
    if config.reference_profile_path is None:
        raise ValueError("reference_profile_path is required")
    if config.feature_model_path is None:
        raise ValueError("feature_model_path is required")
    if config.audio_source == "wav" and config.audio_file_path is None:
        raise ValueError("audio_file_path is required when audio_source=wav")
    if config.presentation_command_queue_capacity <= 0:
        raise ValueError("presentation_command_queue_capacity must be greater than zero")
    if config.presentation_mode == "osc":
        OscPresentationGatewayConfig(
            host=config.osc_host,
            port=config.osc_port,
            path=config.osc_path,
            retry_count=config.osc_retry_count,
            retry_backoff_seconds=config.osc_retry_backoff_seconds,
        )


def _log_runtime_health(
    logger: logging.Logger,
    config: AppConfig,
    report: RuntimeReport,
) -> None:
    logger.info(
        "AUDIO: %s | MODEL: ONNX | ALIGNMENT: %s | POSITION: %s | CONFIDENCE: %.2f | "
        "SLIDE: %s | AUDIO_QUEUE: %s/%s | COMMAND_QUEUE: %s/%s | PRESENTATION: %s",
        "OK" if report.metrics.chunks_received > 0 else "NO DATA",
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
        config.presentation_mode.upper(),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the core audio-to-lyrics alignment runtime.",
    )
    parser.add_argument(
        "--audio-source",
        choices=("wav", "microphone"),
        help="Choose the runtime audio input source.",
    )
    parser.add_argument(
        "--audio-file-path",
        help="Path to a WAV file when using --audio-source wav.",
    )
    parser.add_argument(
        "--input-device",
        help="Audio input device name or numeric device index.",
    )
    parser.add_argument(
        "--feature-model-path",
        help="Path to the ONNX feature extraction model.",
    )
    parser.add_argument(
        "--reference-profile-path",
        help="Path to a prepared reference profile directory.",
    )
    parser.add_argument(
        "--presentation-mode",
        choices=("logging", "osc", "none"),
        help="Choose the presentation output.",
    )
    parser.add_argument(
        "--match-debug-logging",
        action="store_true",
        help="Log per-frame tracker decisions for matching diagnostics.",
    )
    parser.add_argument(
        "--match-confidence-threshold",
        type=float,
        help="Confidence threshold used to accept feature matches.",
    )
    parser.add_argument(
        "--match-ambiguity-distance-margin",
        type=float,
        help="Reject matches when a distant second-best region scores too similarly.",
    )
    parser.add_argument(
        "--match-ambiguity-min-separation-seconds",
        type=float,
        help="Minimum reference-time gap used when testing ambiguous alternatives.",
    )
    parser.add_argument(
        "--match-confirmation-count",
        type=int,
        help="Consecutive stable matches required before tracking locks.",
    )
    parser.add_argument(
        "--tracking-recovery-confidence-threshold",
        type=float,
        help="Confidence threshold used while recovering position.",
    )
    parser.add_argument(
        "--tracking-lost-match-patience",
        type=int,
        help="Bad frames tolerated before returning to search mode.",
    )
    parser.add_argument(
        "--tracking-search-miss-patience",
        type=int,
        help="Interrupted search frames tolerated before resetting confirmation.",
    )
    parser.add_argument(
        "--tracking-miss-patience",
        type=int,
        help="Bad tracking frames tolerated before dropping into recovery.",
    )
    parser.add_argument(
        "--tracking-match-window-seconds",
        type=float,
        help="Reference-time window searched while tracking a stable position.",
    )
    parser.add_argument(
        "--recovery-match-window-seconds",
        type=float,
        help="Reference-time window searched while recovering from an uncertain position.",
    )
    parser.add_argument(
        "--tracking-max-forward-jump-seconds",
        type=float,
        help="Maximum forward reference-time jump accepted while tracking.",
    )
    parser.add_argument(
        "--tracking-max-backward-jump-seconds",
        type=float,
        help="Maximum backward reference-time jump accepted while tracking.",
    )
    parser.add_argument(
        "--tracking-expected-position-tolerance-seconds",
        type=float,
        help="Maximum difference between elapsed stream time and candidate reference time.",
    )
    parser.add_argument(
        "--tracking-search-min-duration-seconds",
        type=float,
        help="Minimum live duration of stable matches required before locking from search.",
    )
    parser.add_argument(
        "--tracking-recovery-min-duration-seconds",
        type=float,
        help="Minimum live duration of stable matches required before relocking from recovery.",
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
        help="Stable cue-reaching matches required before emitting a slide.",
    )
    parser.add_argument(
        "--slide-max-emit-lag-seconds",
        type=float,
        help="Skip stale cues instead of emitting all missed slides after a late relock.",
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
    input_device = config.input_device
    if args.input_device is not None:
        input_device = int(args.input_device) if args.input_device.isdigit() else args.input_device
    return AppConfig(
        audio_source=args.audio_source or config.audio_source,
        audio_file_path=args.audio_file_path or config.audio_file_path,
        reference_profile_path=(
            args.reference_profile_path or config.reference_profile_path
        ),
        feature_model_path=args.feature_model_path or config.feature_model_path,
        presentation_mode=args.presentation_mode or config.presentation_mode,
        match_confidence_threshold=(
            args.match_confidence_threshold
            if args.match_confidence_threshold is not None
            else config.match_confidence_threshold
        ),
        match_ambiguity_distance_margin=(
            args.match_ambiguity_distance_margin
            if args.match_ambiguity_distance_margin is not None
            else config.match_ambiguity_distance_margin
        ),
        match_ambiguity_min_separation_seconds=(
            args.match_ambiguity_min_separation_seconds
            if args.match_ambiguity_min_separation_seconds is not None
            else config.match_ambiguity_min_separation_seconds
        ),
        match_confirmation_count=(
            args.match_confirmation_count
            if args.match_confirmation_count is not None
            else config.match_confirmation_count
        ),
        tracking_search_miss_patience=(
            args.tracking_search_miss_patience
            if args.tracking_search_miss_patience is not None
            else config.tracking_search_miss_patience
        ),
        tracking_miss_patience=(
            args.tracking_miss_patience
            if args.tracking_miss_patience is not None
            else config.tracking_miss_patience
        ),
        tracking_match_window_seconds=(
            args.tracking_match_window_seconds
            if args.tracking_match_window_seconds is not None
            else config.tracking_match_window_seconds
        ),
        recovery_match_window_seconds=(
            args.recovery_match_window_seconds
            if args.recovery_match_window_seconds is not None
            else config.recovery_match_window_seconds
        ),
        tracking_max_forward_jump_seconds=(
            args.tracking_max_forward_jump_seconds
            if args.tracking_max_forward_jump_seconds is not None
            else config.tracking_max_forward_jump_seconds
        ),
        tracking_max_backward_jump_seconds=(
            args.tracking_max_backward_jump_seconds
            if args.tracking_max_backward_jump_seconds is not None
            else config.tracking_max_backward_jump_seconds
        ),
        tracking_expected_position_tolerance_seconds=(
            args.tracking_expected_position_tolerance_seconds
            if args.tracking_expected_position_tolerance_seconds is not None
            else config.tracking_expected_position_tolerance_seconds
        ),
        tracking_anchored_timeline_tolerance_seconds=(
            config.tracking_anchored_timeline_tolerance_seconds
        ),
        tracking_search_min_duration_seconds=(
            args.tracking_search_min_duration_seconds
            if args.tracking_search_min_duration_seconds is not None
            else config.tracking_search_min_duration_seconds
        ),
        tracking_recovery_min_duration_seconds=(
            args.tracking_recovery_min_duration_seconds
            if args.tracking_recovery_min_duration_seconds is not None
            else config.tracking_recovery_min_duration_seconds
        ),
        tracking_recovery_confidence_threshold=(
            args.tracking_recovery_confidence_threshold
            if args.tracking_recovery_confidence_threshold is not None
            else config.tracking_recovery_confidence_threshold
        ),
        tracking_lost_match_patience=(
            args.tracking_lost_match_patience
            if args.tracking_lost_match_patience is not None
            else config.tracking_lost_match_patience
        ),
        slide_lookahead_seconds=(
            args.slide_lookahead_seconds
            if args.slide_lookahead_seconds is not None
            else config.slide_lookahead_seconds
        ),
        slide_trigger_cooldown_seconds=(
            args.slide_trigger_cooldown_seconds
            if args.slide_trigger_cooldown_seconds is not None
            else config.slide_trigger_cooldown_seconds
        ),
        slide_consecutive_match_count=(
            args.slide_consecutive_match_count
            if args.slide_consecutive_match_count is not None
            else config.slide_consecutive_match_count
        ),
        slide_max_emit_lag_seconds=(
            args.slide_max_emit_lag_seconds
            if args.slide_max_emit_lag_seconds is not None
            else config.slide_max_emit_lag_seconds
        ),
        sample_rate=config.sample_rate,
        channels=config.channels,
        block_size=config.block_size,
        audio_queue_capacity=config.audio_queue_capacity,
        presentation_command_queue_capacity=config.presentation_command_queue_capacity,
        input_device=input_device,
        diagnostics_interval_seconds=config.diagnostics_interval_seconds,
        match_debug_logging=args.match_debug_logging or config.match_debug_logging,
        silence_threshold_rms=config.silence_threshold_rms,
        silence_reset_chunk_count=config.silence_reset_chunk_count,
        clipping_threshold_peak=config.clipping_threshold_peak,
        osc_host=args.osc_host or config.osc_host,
        osc_port=args.osc_port if args.osc_port is not None else config.osc_port,
        osc_path=args.osc_path or config.osc_path,
        osc_retry_count=(
            args.osc_retry_count
            if args.osc_retry_count is not None
            else config.osc_retry_count
        ),
        osc_retry_backoff_seconds=(
            args.osc_retry_backoff_seconds
            if args.osc_retry_backoff_seconds is not None
            else config.osc_retry_backoff_seconds
        ),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = _config_from_args(_parse_args())
    _validate_config(config)
    logger = logging.getLogger(__name__)

    profile = FilesystemReferenceProfileRepository().load(config.reference_profile_path)
    logger.info(
        "Loaded reference profile name=%s frames=%s path=%s",
        profile.name,
        len(profile.frames),
        config.reference_profile_path,
    )

    source = _build_audio_source(config)
    feature_extractor = OnnxFeatureExtractor(
        OnnxFeatureExtractorConfig(
            model_path=config.feature_model_path,
            sample_rate=config.sample_rate,
        )
    )
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
        silence_reset_chunk_count=config.silence_reset_chunk_count,
        clipping_threshold_peak=config.clipping_threshold_peak,
        command_queue_capacity=config.presentation_command_queue_capacity,
        emit_match_debug_logs=config.match_debug_logging,
    )
    report = runtime.run()
    _log_runtime_health(logger, config, report)
    logger.info(
        "Audio ingestion finished source=%s presentation_mode=%s sample_rate=%s "
        "block_size=%s chunks_received=%s chunks_dropped=%s silent_chunks=%s "
        "clipped_chunks=%s feature_frames_processed=%s accepted_matches=%s "
        "low_confidence_matches=%s slide_triggers_sent=%s osc_send_failures=%s "
        "queue_high_water_mark=%s command_queue_capacity=%s",
        config.audio_source,
        config.presentation_mode,
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
