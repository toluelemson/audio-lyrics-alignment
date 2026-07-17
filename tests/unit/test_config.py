import pytest

from lyrics_aligner.config import AppConfig


def test_default_audio_configuration() -> None:
    config = AppConfig()

    assert config.audio_source == "simulated"
    assert config.feature_extractor == "simulated"
    assert config.reference_profile_path is None
    assert config.presentation_mode == "logging"
    assert config.manual_override is False
    assert config.match_confidence_threshold == 0.6
    assert config.match_max_forward_jump_frames == 4
    assert config.match_large_jump_threshold_frames == 2
    assert config.match_confirmation_count == 2
    assert config.tracking_recovery_confidence_threshold == 0.6
    assert config.tracking_lost_match_patience == 3
    assert config.slide_lookahead_seconds == 0.2
    assert config.slide_trigger_cooldown_seconds == 0.5
    assert config.slide_consecutive_match_count == 2
    assert config.presentation_command_queue_capacity == 8
    assert config.sample_rate == 16_000
    assert config.channels == 1
    assert config.block_size == 4_096
    assert config.silence_threshold_rms == 0.01
    assert config.clipping_threshold_peak == 0.99
    assert config.osc_port == 7_000
    assert config.osc_retry_count == 2
    assert config.osc_retry_backoff_seconds == 0.05


def test_from_env_overrides_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LYRICS_ALIGNER_AUDIO_SOURCE", "microphone")
    monkeypatch.setenv("LYRICS_ALIGNER_FEATURE_EXTRACTOR", "onnx")
    monkeypatch.setenv("LYRICS_ALIGNER_FEATURE_MODEL_PATH", "models/extractor.onnx")
    monkeypatch.setenv("LYRICS_ALIGNER_REFERENCE_PROFILE_PATH", "profiles/song-a")
    monkeypatch.setenv("LYRICS_ALIGNER_PRESENTATION_MODE", "both")
    monkeypatch.setenv("LYRICS_ALIGNER_MANUAL_OVERRIDE", "true")
    monkeypatch.setenv("LYRICS_ALIGNER_MATCH_CONFIDENCE_THRESHOLD", "0.75")
    monkeypatch.setenv("LYRICS_ALIGNER_MATCH_MAX_FORWARD_JUMP_FRAMES", "6")
    monkeypatch.setenv("LYRICS_ALIGNER_MATCH_LARGE_JUMP_THRESHOLD_FRAMES", "3")
    monkeypatch.setenv("LYRICS_ALIGNER_MATCH_CONFIRMATION_COUNT", "4")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_RECOVERY_CONFIDENCE_THRESHOLD", "0.65")
    monkeypatch.setenv("LYRICS_ALIGNER_TRACKING_LOST_MATCH_PATIENCE", "5")
    monkeypatch.setenv("LYRICS_ALIGNER_SLIDE_LOOKAHEAD_SECONDS", "0.15")
    monkeypatch.setenv("LYRICS_ALIGNER_SLIDE_TRIGGER_COOLDOWN_SECONDS", "0.7")
    monkeypatch.setenv("LYRICS_ALIGNER_SLIDE_CONSECUTIVE_MATCH_COUNT", "3")
    monkeypatch.setenv("LYRICS_ALIGNER_PRESENTATION_COMMAND_QUEUE_CAPACITY", "12")
    monkeypatch.setenv("LYRICS_ALIGNER_INPUT_DEVICE", "2")
    monkeypatch.setenv("LYRICS_ALIGNER_SIMULATION_DURATION_SECONDS", "5.5")
    monkeypatch.setenv("LYRICS_ALIGNER_SILENCE_THRESHOLD_RMS", "0.02")
    monkeypatch.setenv("LYRICS_ALIGNER_OSC_RETRY_COUNT", "4")
    monkeypatch.setenv("LYRICS_ALIGNER_OSC_RETRY_BACKOFF_SECONDS", "0.2")

    config = AppConfig.from_env()

    assert config.audio_source == "microphone"
    assert config.feature_extractor == "onnx"
    assert config.feature_model_path == "models/extractor.onnx"
    assert config.reference_profile_path == "profiles/song-a"
    assert config.presentation_mode == "both"
    assert config.manual_override is True
    assert config.match_confidence_threshold == 0.75
    assert config.match_max_forward_jump_frames == 6
    assert config.match_large_jump_threshold_frames == 3
    assert config.match_confirmation_count == 4
    assert config.tracking_recovery_confidence_threshold == 0.65
    assert config.tracking_lost_match_patience == 5
    assert config.slide_lookahead_seconds == 0.15
    assert config.slide_trigger_cooldown_seconds == 0.7
    assert config.slide_consecutive_match_count == 3
    assert config.presentation_command_queue_capacity == 12
    assert config.input_device == 2
    assert config.simulation_duration_seconds == 5.5
    assert config.silence_threshold_rms == 0.02
    assert config.osc_retry_count == 4
    assert config.osc_retry_backoff_seconds == 0.2
