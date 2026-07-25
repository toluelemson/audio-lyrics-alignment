import pytest

from lyrics_aligner.config import AppConfig


def test_default_configuration_is_streamlined_to_core_runtime() -> None:
    config = AppConfig()

    assert config.audio_source == "wav"
    assert config.audio_file_path is None
    assert config.reference_profile_path is None
    assert config.feature_model_path is None
    assert config.presentation_mode == "logging"
    assert config.match_confidence_threshold == 0.6
    assert config.match_confirmation_count == 2
    assert config.slide_lookahead_seconds == 0.2
    assert config.slide_trigger_cooldown_seconds == 0.5
    assert config.sample_rate == 16_000
    assert config.channels == 1
    assert config.block_size == 4_096
    assert config.input_device is None
    assert config.match_debug_logging is False
    assert config.silence_threshold_rms == 0.01
    assert config.silence_reset_chunk_count == 3
    assert config.clipping_threshold_peak == 0.99
    assert config.osc_port == 7_000
    assert config.osc_retry_count == 2
    assert config.osc_retry_backoff_seconds == 0.05


def test_from_env_overrides_core_runtime_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LYRICS_ALIGNER_AUDIO_SOURCE", "microphone")
    monkeypatch.setenv("LYRICS_ALIGNER_AUDIO_FILE_PATH", "demo.wav")
    monkeypatch.setenv("LYRICS_ALIGNER_REFERENCE_PROFILE_PATH", "profiles/song-a")
    monkeypatch.setenv("LYRICS_ALIGNER_FEATURE_MODEL_PATH", "models/extractor.onnx")
    monkeypatch.setenv("LYRICS_ALIGNER_PRESENTATION_MODE", "osc")
    monkeypatch.setenv("LYRICS_ALIGNER_MATCH_CONFIDENCE_THRESHOLD", "0.75")
    monkeypatch.setenv("LYRICS_ALIGNER_MATCH_AMBIGUITY_DISTANCE_MARGIN", "0.015")
    monkeypatch.setenv("LYRICS_ALIGNER_MATCH_AMBIGUITY_MIN_SEPARATION_SECONDS", "12.0")
    monkeypatch.setenv("LYRICS_ALIGNER_MATCH_CONFIRMATION_COUNT", "4")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_SEARCH_MISS_PATIENCE", "1")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_MISS_PATIENCE", "2")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_MATCH_WINDOW_SECONDS", "9.0")
    monkeypatch.setenv("LYRICS_ALIGNER_RECOVERY_MATCH_WINDOW_SECONDS", "14.0")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_MAX_FORWARD_JUMP_SECONDS", "0.6")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_MAX_BACKWARD_JUMP_SECONDS", "0.3")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_EXPECTED_POSITION_TOLERANCE_SECONDS", "6.0")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_ANCHORED_TIMELINE_TOLERANCE_SECONDS", "1.2")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_SEARCH_MIN_DURATION_SECONDS", "0.4")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_RECOVERY_MIN_DURATION_SECONDS", "0.5")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_RECOVERY_CONFIDENCE_THRESHOLD", "0.65")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_LOST_MATCH_PATIENCE", "5")
    monkeypatch.setenv("LYRICS_ALIGNER_SLIDE_LOOKAHEAD_SECONDS", "0.15")
    monkeypatch.setenv("LYRICS_ALIGNER_SLIDE_TRIGGER_COOLDOWN_SECONDS", "0.7")
    monkeypatch.setenv("LYRICS_ALIGNER_SLIDE_CONSECUTIVE_MATCH_COUNT", "3")
    monkeypatch.setenv("LYRICS_ALIGNER_SLIDE_MAX_EMIT_LAG_SECONDS", "1.5")
    monkeypatch.setenv("LYRICS_ALIGNER_INPUT_DEVICE", "2")
    monkeypatch.setenv("LYRICS_ALIGNER_MATCH_DEBUG_LOGGING", "true")
    monkeypatch.setenv("LYRICS_ALIGNER_SILENCE_THRESHOLD_RMS", "0.02")
    monkeypatch.setenv("LYRICS_ALIGNER_SILENCE_RESET_CHUNK_COUNT", "5")
    monkeypatch.setenv("LYRICS_ALIGNER_OSC_RETRY_COUNT", "4")
    monkeypatch.setenv("LYRICS_ALIGNER_OSC_RETRY_BACKOFF_SECONDS", "0.2")

    config = AppConfig.from_env()

    assert config.audio_source == "microphone"
    assert config.audio_file_path == "demo.wav"
    assert config.reference_profile_path == "profiles/song-a"
    assert config.feature_model_path == "models/extractor.onnx"
    assert config.presentation_mode == "osc"
    assert config.match_confidence_threshold == 0.75
    assert config.match_ambiguity_distance_margin == 0.015
    assert config.match_ambiguity_min_separation_seconds == 12.0
    assert config.match_confirmation_count == 4
    assert config.tracking_search_miss_patience == 1
    assert config.tracking_miss_patience == 2
    assert config.tracking_match_window_seconds == 9.0
    assert config.recovery_match_window_seconds == 14.0
    assert config.tracking_max_forward_jump_seconds == 0.6
    assert config.tracking_max_backward_jump_seconds == 0.3
    assert config.tracking_expected_position_tolerance_seconds == 6.0
    assert config.tracking_anchored_timeline_tolerance_seconds == 1.2
    assert config.tracking_search_min_duration_seconds == 0.4
    assert config.tracking_recovery_min_duration_seconds == 0.5
    assert config.tracking_recovery_confidence_threshold == 0.65
    assert config.tracking_lost_match_patience == 5
    assert config.slide_lookahead_seconds == 0.15
    assert config.slide_trigger_cooldown_seconds == 0.7
    assert config.slide_consecutive_match_count == 3
    assert config.slide_max_emit_lag_seconds == 1.5
    assert config.input_device == 2
    assert config.match_debug_logging is True
    assert config.silence_threshold_rms == 0.02
    assert config.silence_reset_chunk_count == 5
    assert config.osc_retry_count == 4
    assert config.osc_retry_backoff_seconds == 0.2
