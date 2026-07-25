import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from lyrics_aligner.adapters.audio.wav_file import WavFileAudioConfig, WavFileAudioSource
from lyrics_aligner.ports.audio_source import AudioSource


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32_767.0).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm[:, np.newaxis], channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_emits_float32_chunks_from_wav_file(tmp_path: Path) -> None:
    path = tmp_path / "demo.wav"
    _write_wav(path, np.linspace(-0.5, 0.5, num=20, dtype=np.float32), sample_rate=16_000)

    source = WavFileAudioSource(
        WavFileAudioConfig(path=str(path), sample_rate=16_000, block_size=8)
    )
    chunks = list(source.chunks())

    assert [chunk.samples.shape for chunk in chunks] == [(8,), (8,), (4,)]
    assert all(chunk.samples.dtype == np.float32 for chunk in chunks)
    assert [chunk.sequence_number for chunk in chunks] == [0, 1, 2]


def test_resamples_non_16khz_wav_to_target_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "resample.wav"
    _write_wav(path, np.sin(2 * np.pi * 110 * np.arange(8_000) / 8_000).astype(np.float32), 8_000)

    source = WavFileAudioSource(
        WavFileAudioConfig(path=str(path), sample_rate=16_000, block_size=4_096)
    )
    chunks = source.chunks()

    assert sum(chunk.samples.shape[0] for chunk in chunks) == 16_000


def test_conforms_to_audio_source_port(tmp_path: Path) -> None:
    path = tmp_path / "port.wav"
    _write_wav(path, np.ones(16, dtype=np.float32) * 0.25, 16_000)

    source: AudioSource = WavFileAudioSource(
        WavFileAudioConfig(path=str(path), sample_rate=16_000, block_size=8)
    )

    assert next(source.chunks()).sequence_number == 0


def test_stop_interrupts_real_time_replay(tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    _write_wav(path, np.ones(160_000, dtype=np.float32) * 0.25, 16_000)
    source = WavFileAudioSource(
        WavFileAudioConfig(path=str(path), sample_rate=16_000, block_size=16_000)
    )
    iterator = iter(source.chunks())
    assert next(iterator).sequence_number == 0

    results: list[object] = []
    worker = threading.Thread(target=lambda: results.append(next(iterator, None)))
    worker.start()
    time.sleep(0.02)
    source.stop()
    worker.join(timeout=0.2)

    assert not worker.is_alive()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("path", "", ValueError),
        ("sample_rate", 0, ValueError),
        ("sample_rate", 16_000.0, TypeError),
        ("block_size", 0, ValueError),
        ("block_size", True, TypeError),
    ],
)
def test_rejects_invalid_configuration(field: str, value: object, error: type[Exception]) -> None:
    values: dict[str, object] = {
        "path": "demo.wav",
        "sample_rate": 16_000,
        "block_size": 4_096,
    }
    values[field] = value

    with pytest.raises(error):
        WavFileAudioConfig(**values)  # type: ignore[arg-type]
