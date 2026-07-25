import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from lyrics_aligner.adapters.features.simulated import (
    SimulatedFeatureExtractor,
    SimulatedFeatureExtractorConfig,
)
from lyrics_aligner.application.reference_builder import (
    PROFILE_VERSION,
    ReferenceBuilderConfig,
    ReferenceProfileBuilder,
    extract_reference_frames,
    load_slide_intervals,
    load_wav_mono,
    resample_audio,
)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    int_samples = np.clip(samples, -1.0, 1.0)
    pcm = (int_samples * 32_767.0).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm[:, np.newaxis], channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_load_wav_mono_converts_stereo_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    samples = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    _write_wav(path, samples, sample_rate=8_000, channels=2)

    mono, sample_rate = load_wav_mono(str(path))

    assert sample_rate == 8_000
    assert mono.dtype == np.float32
    assert mono.shape == (4,)


def test_resample_audio_changes_length_for_target_rate() -> None:
    samples = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)

    resampled = resample_audio(samples, source_sample_rate=8_000, target_sample_rate=16_000)

    assert resampled.dtype == np.float32
    assert resampled.shape[0] == 8


def test_extract_reference_frames_uses_feature_extractor() -> None:
    extractor = SimulatedFeatureExtractor(
        SimulatedFeatureExtractorConfig(sample_rate=16_000)
    )
    samples = np.ones(8_192, dtype=np.float32) * 0.5

    frames = extract_reference_frames(
        samples=samples,
        sample_rate=16_000,
        block_size=4_096,
        feature_extractor=extractor,
    )

    assert len(frames) == 2
    assert frames[0].observed_at == 0.0
    assert frames[1].observed_at == 0.256


def test_load_slide_intervals_maps_timestamps_to_frame_indices(tmp_path: Path) -> None:
    slides_path = tmp_path / "slides.json"
    slides_path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                    "click_timestamp": 0.0,
                },
                {
                    "slide_number": 2,
                    "section": "Chorus",
                    "lyrics": "How sweet the sound",
                    "click_timestamp": 0.5,
                },
            ]
        ),
        encoding="utf-8",
    )

    intervals = load_slide_intervals(
        slides_path=str(slides_path),
        total_duration_seconds=1.0,
        timestamps_seconds=(0.0, 0.25, 0.5, 0.75),
    )

    assert len(intervals) == 2
    assert intervals[0].start_frame == 0
    assert intervals[0].end_frame == 2
    assert intervals[1].start_frame == 2
    assert intervals[1].end_frame == 3


def test_load_slide_intervals_accepts_legacy_reference_timestamp(tmp_path: Path) -> None:
    slides_path = tmp_path / "slides.json"
    slides_path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                    "reference_timestamp": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    intervals = load_slide_intervals(
        slides_path=str(slides_path),
        total_duration_seconds=1.0,
        timestamps_seconds=(0.0, 0.25, 0.5, 0.75),
    )

    assert len(intervals) == 1
    assert intervals[0].reference_timestamp == pytest.approx(0.0)


def test_builder_creates_reproducible_profile_files(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    slides_path = tmp_path / "slides.json"
    output_path = tmp_path / "profile"
    sample_count = 16_000
    time = np.arange(sample_count, dtype=np.float32) / 16_000.0
    waveform = 0.5 * np.sin(2 * np.pi * 220.0 * time, dtype=np.float32)
    _write_wav(audio_path, waveform, sample_rate=16_000)
    slides_path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                    "click_timestamp": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    builder = ReferenceProfileBuilder(
        feature_extractor=SimulatedFeatureExtractor(
            SimulatedFeatureExtractorConfig(sample_rate=16_000)
        ),
        config=ReferenceBuilderConfig(sample_rate=16_000, block_size=4_096),
    )

    profile = builder.build(str(audio_path), str(slides_path), "amazing-grace")
    builder.save(profile, str(output_path))

    saved_profile = json.loads((output_path / "profile.json").read_text(encoding="utf-8"))
    saved_metadata = json.loads((output_path / "metadata.json").read_text(encoding="utf-8"))
    saved_features = np.load(output_path / "reference_features.npy")

    assert saved_profile["profile_version"] == PROFILE_VERSION
    assert saved_profile["name"] == "amazing-grace"
    assert saved_profile["slides"][0]["slide_number"] == 1
    assert saved_metadata["profile_version"] == PROFILE_VERSION
    assert saved_metadata["reference_audio_role"] == "mixed"
    assert saved_features.dtype == np.float32
    assert saved_features.shape[1] == 4


def test_builder_records_vocal_reference_metadata(tmp_path: Path) -> None:
    audio_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    slides_path = tmp_path / "slides.json"
    sample_count = 16_000
    time = np.arange(sample_count, dtype=np.float32) / 16_000.0
    waveform = 0.5 * np.sin(2 * np.pi * 220.0 * time, dtype=np.float32)
    _write_wav(audio_path, waveform, sample_rate=16_000)
    _write_wav(mix_path, waveform, sample_rate=16_000)
    slides_path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                    "click_timestamp": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    builder = ReferenceProfileBuilder(
        feature_extractor=SimulatedFeatureExtractor(
            SimulatedFeatureExtractorConfig(sample_rate=16_000)
        ),
        config=ReferenceBuilderConfig(sample_rate=16_000, block_size=4_096),
    )

    profile = builder.build(
        str(audio_path),
        str(slides_path),
        "amazing-grace-vocals",
        audio_role="vocals",
        companion_audio_path=str(mix_path),
    )

    assert profile.metadata["reference_audio_role"] == "vocals"
    assert profile.metadata["companion_audio_path"] == str(mix_path.resolve())


def test_builder_rejects_unknown_audio_role(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    slides_path = tmp_path / "slides.json"
    waveform = np.zeros(16_000, dtype=np.float32)
    _write_wav(audio_path, waveform, sample_rate=16_000)
    slides_path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                    "click_timestamp": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    builder = ReferenceProfileBuilder(
        feature_extractor=SimulatedFeatureExtractor(
            SimulatedFeatureExtractorConfig(sample_rate=16_000)
        ),
        config=ReferenceBuilderConfig(sample_rate=16_000, block_size=4_096),
    )

    with pytest.raises(ValueError, match="audio_role"):
        builder.build(
            str(audio_path),
            str(slides_path),
            "bad-role",
            audio_role="choir-only",
        )


def test_cli_builds_reference_profile_from_repo_root(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    slides_path = tmp_path / "slides.json"
    output_path = tmp_path / "profile"
    sample_count = 8_000
    time = np.arange(sample_count, dtype=np.float32) / 8_000.0
    waveform = 0.5 * np.sin(2 * np.pi * 110.0 * time).astype(np.float32)
    _write_wav(audio_path, waveform, sample_rate=8_000)
    slides_path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                    "click_timestamp": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_reference.py",
            "--audio",
            str(audio_path),
            "--slides",
            str(slides_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    summary = json.loads(result.stdout)
    assert summary["profile_name"] == "profile"
    assert summary["profile_version"] == PROFILE_VERSION
    assert summary["reference_audio_role"] == "mixed"
    assert (output_path / "profile.json").exists()


def test_vocal_cli_builds_vocal_reference_profile(tmp_path: Path) -> None:
    vocals_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    slides_path = tmp_path / "slides.json"
    output_path = tmp_path / "profile"
    sample_count = 8_000
    time = np.arange(sample_count, dtype=np.float32) / 8_000.0
    waveform = 0.5 * np.sin(2 * np.pi * 110.0 * time).astype(np.float32)
    _write_wav(vocals_path, waveform, sample_rate=8_000)
    _write_wav(mix_path, waveform, sample_rate=8_000)
    slides_path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                    "click_timestamp": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_vocal_reference.py",
            "--vocals",
            str(vocals_path),
            "--mix",
            str(mix_path),
            "--slides",
            str(slides_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    summary = json.loads(result.stdout)
    metadata = json.loads((output_path / "metadata.json").read_text(encoding="utf-8"))
    assert summary["reference_audio_role"] == "vocals"
    assert metadata["reference_audio_role"] == "vocals"
    assert metadata["companion_audio_path"] == str(mix_path.resolve())
