from lyrics_aligner.config import AppConfig


def test_default_audio_configuration() -> None:
    config = AppConfig()

    assert config.sample_rate == 16_000
    assert config.channels == 1
    assert config.block_size == 4_096
    assert config.osc_port == 7_000
