from lyrics_aligner.config import AppConfig


def test_default_audio_configuration() -> None:
    config = AppConfig()

    assert config.audio_source == "simulated"
    assert config.sample_rate == 16_000
    assert config.channels == 1
    assert config.block_size == 4_096
    assert config.silence_threshold_rms == 0.01
    assert config.clipping_threshold_peak == 0.99
    assert config.osc_port == 7_000
