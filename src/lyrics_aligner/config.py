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


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value in {"1", "true", "TRUE", "yes", "YES"}


def _read_optional_device(name: str) -> str | int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value) if value.isdigit() else value


@dataclass(frozen=True, slots=True)
class AppConfig:
    audio_source: str = "wav"
    audio_file_path: str | None = None
    reference_profile_path: str | None = None
    feature_model_path: str | None = None
    live_tracking_mode: str = "live_audio_inference"
    presentation_mode: str = "logging"
    match_confidence_threshold: float = 0.6
    match_ambiguity_distance_margin: float = 0.03
    match_ambiguity_min_separation_seconds: float = 15.0
    match_confirmation_count: int = 2
    tracking_search_miss_patience: int = 2
    tracking_miss_patience: int = 3
    tracking_match_window_seconds: float = 14.0
    recovery_match_window_seconds: float = 20.0
    tracking_max_forward_jump_seconds: float = 0.35
    tracking_max_backward_jump_seconds: float = 0.2
    tracking_expected_position_tolerance_seconds: float = 12.0
    tracking_anchored_timeline_tolerance_seconds: float = 1.5
    tracking_search_min_duration_seconds: float = 0.25
    tracking_recovery_min_duration_seconds: float = 0.25
    tracking_recovery_confidence_threshold: float = 0.6
    tracking_lost_match_patience: int = 3
    slide_lookahead_seconds: float = 0.2
    slide_trigger_cooldown_seconds: float = 0.5
    slide_consecutive_match_count: int = 2
    slide_max_emit_lag_seconds: float = 2.0
    sample_rate: int = 16_000
    channels: int = 1
    block_size: int = 4_096
    audio_queue_capacity: int = 16
    presentation_command_queue_capacity: int = 8
    input_device: str | int | None = None
    diagnostics_interval_seconds: float = 0.5
    match_debug_logging: bool = False
    silence_threshold_rms: float = 0.01
    silence_reset_chunk_count: int = 3
    clipping_threshold_peak: float = 0.99
    osc_host: str = "127.0.0.1"
    osc_port: int = 7_000
    osc_path: str = "/presentation/trigger-slide"
    osc_retry_count: int = 2
    osc_retry_backoff_seconds: float = 0.05

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            audio_source=_read_str("LYRICS_ALIGNER_AUDIO_SOURCE", "wav"),
            audio_file_path=os.getenv("LYRICS_ALIGNER_AUDIO_FILE_PATH"),
            reference_profile_path=os.getenv("LYRICS_ALIGNER_REFERENCE_PROFILE_PATH"),
            feature_model_path=os.getenv("LYRICS_ALIGNER_FEATURE_MODEL_PATH"),
            live_tracking_mode=_read_str(
                "LYRICS_ALIGNER_LIVE_TRACKING_MODE",
                "live_audio_inference",
            ),
            presentation_mode=_read_str("LYRICS_ALIGNER_PRESENTATION_MODE", "logging"),
            match_confidence_threshold=_read_float(
                "LYRICS_ALIGNER_MATCH_CONFIDENCE_THRESHOLD",
                0.6,
            ),
            match_ambiguity_distance_margin=_read_float(
                "LYRICS_ALIGNER_MATCH_AMBIGUITY_DISTANCE_MARGIN",
                0.03,
            ),
            match_ambiguity_min_separation_seconds=_read_float(
                "LYRICS_ALIGNER_MATCH_AMBIGUITY_MIN_SEPARATION_SECONDS",
                15.0,
            ),
            match_confirmation_count=_read_int(
                "LYRICS_ALIGNER_MATCH_CONFIRMATION_COUNT",
                2,
            ),
            tracking_search_miss_patience=_read_int(
                "LYRICS_ALIGNER_TRACKING_SEARCH_MISS_PATIENCE",
                2,
            ),
            tracking_miss_patience=_read_int(
                "LYRICS_ALIGNER_TRACKING_MISS_PATIENCE",
                3,
            ),
            tracking_match_window_seconds=_read_float(
                "LYRICS_ALIGNER_TRACKING_MATCH_WINDOW_SECONDS",
                14.0,
            ),
            recovery_match_window_seconds=_read_float(
                "LYRICS_ALIGNER_RECOVERY_MATCH_WINDOW_SECONDS",
                20.0,
            ),
            tracking_max_forward_jump_seconds=_read_float(
                "LYRICS_ALIGNER_TRACKING_MAX_FORWARD_JUMP_SECONDS",
                0.35,
            ),
            tracking_max_backward_jump_seconds=_read_float(
                "LYRICS_ALIGNER_TRACKING_MAX_BACKWARD_JUMP_SECONDS",
                0.2,
            ),
            tracking_expected_position_tolerance_seconds=_read_float(
                "LYRICS_ALIGNER_TRACKING_EXPECTED_POSITION_TOLERANCE_SECONDS",
                12.0,
            ),
            tracking_anchored_timeline_tolerance_seconds=_read_float(
                "LYRICS_ALIGNER_TRACKING_ANCHORED_TIMELINE_TOLERANCE_SECONDS",
                1.5,
            ),
            tracking_search_min_duration_seconds=_read_float(
                "LYRICS_ALIGNER_TRACKING_SEARCH_MIN_DURATION_SECONDS",
                0.25,
            ),
            tracking_recovery_min_duration_seconds=_read_float(
                "LYRICS_ALIGNER_TRACKING_RECOVERY_MIN_DURATION_SECONDS",
                0.25,
            ),
            tracking_recovery_confidence_threshold=_read_float(
                "LYRICS_ALIGNER_TRACKING_RECOVERY_CONFIDENCE_THRESHOLD",
                0.6,
            ),
            tracking_lost_match_patience=_read_int(
                "LYRICS_ALIGNER_TRACKING_LOST_MATCH_PATIENCE",
                3,
            ),
            slide_lookahead_seconds=_read_float(
                "LYRICS_ALIGNER_SLIDE_LOOKAHEAD_SECONDS",
                0.2,
            ),
            slide_trigger_cooldown_seconds=_read_float(
                "LYRICS_ALIGNER_SLIDE_TRIGGER_COOLDOWN_SECONDS",
                0.5,
            ),
            slide_consecutive_match_count=_read_int(
                "LYRICS_ALIGNER_SLIDE_CONSECUTIVE_MATCH_COUNT",
                2,
            ),
            slide_max_emit_lag_seconds=_read_float(
                "LYRICS_ALIGNER_SLIDE_MAX_EMIT_LAG_SECONDS",
                2.0,
            ),
            sample_rate=_read_int("LYRICS_ALIGNER_SAMPLE_RATE", 16_000),
            channels=_read_int("LYRICS_ALIGNER_CHANNELS", 1),
            block_size=_read_int("LYRICS_ALIGNER_BLOCK_SIZE", 4_096),
            audio_queue_capacity=_read_int("LYRICS_ALIGNER_AUDIO_QUEUE_CAPACITY", 16),
            presentation_command_queue_capacity=_read_int(
                "LYRICS_ALIGNER_PRESENTATION_COMMAND_QUEUE_CAPACITY",
                8,
            ),
            input_device=_read_optional_device("LYRICS_ALIGNER_INPUT_DEVICE"),
            diagnostics_interval_seconds=_read_float(
                "LYRICS_ALIGNER_DIAGNOSTICS_INTERVAL_SECONDS",
                0.5,
            ),
            match_debug_logging=_read_bool("LYRICS_ALIGNER_MATCH_DEBUG_LOGGING", False),
            silence_threshold_rms=_read_float(
                "LYRICS_ALIGNER_SILENCE_THRESHOLD_RMS",
                0.01,
            ),
            silence_reset_chunk_count=_read_int(
                "LYRICS_ALIGNER_SILENCE_RESET_CHUNK_COUNT",
                3,
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
            osc_retry_count=_read_int("LYRICS_ALIGNER_OSC_RETRY_COUNT", 2),
            osc_retry_backoff_seconds=_read_float(
                "LYRICS_ALIGNER_OSC_RETRY_BACKOFF_SECONDS",
                0.05,
            ),
        )
