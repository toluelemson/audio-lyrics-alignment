import pytest

from lyrics_aligner.config import AppConfig


def test_default_audio_configuration() -> None:
    config = AppConfig()

    assert config.audio_source == "simulated"
    assert config.feature_extractor == "simulated"
    assert config.reference_profile_path is None
    assert config.sample_rate == 16_000
    assert config.channels == 1
    assert config.block_size == 4_096
    assert config.silence_threshold_rms == 0.01
    assert config.clipping_threshold_peak == 0.99
    assert config.osc_port == 7_000


def test_from_env_overrides_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LYRICS_ALIGNER_AUDIO_SOURCE", "microphone")
    monkeypatch.setenv("LYRICS_ALIGNER_FEATURE_EXTRACTOR", "onnx")
    monkeypatch.setenv("LYRICS_ALIGNER_FEATURE_MODEL_PATH", "models/extractor.onnx")
    monkeypatch.setenv("LYRICS_ALIGNER_REFERENCE_PROFILE_PATH", "profiles/song-a")
    monkeypatch.setenv("LYRICS_ALIGNER_INPUT_DEVICE", "2")
    monkeypatch.setenv("LYRICS_ALIGNER_SIMULATION_DURATION_SECONDS", "5.5")
    monkeypatch.setenv("LYRICS_ALIGNER_SILENCE_THRESHOLD_RMS", "0.02")

    config = AppConfig.from_env()

    assert config.audio_source == "microphone"
    assert config.feature_extractor == "onnx"
    assert config.feature_model_path == "models/extractor.onnx"
    assert config.reference_profile_path == "profiles/song-a"
    assert config.input_device == 2
    assert config.simulation_duration_seconds == 5.5
    assert config.silence_threshold_rms == 0.02
