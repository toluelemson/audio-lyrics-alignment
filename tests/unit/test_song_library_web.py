import wave
from pathlib import Path

from tools.song_library_web import (
    _profile_has_coarse_artifacts,
    _suggest_profile_clip_range,
)


def _write_silent_wav(path: Path, *, sample_rate: int, duration_seconds: float) -> None:
    frame_count = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


def test_suggest_profile_clip_range_uses_saved_timings_window(tmp_path) -> None:
    audio_path = tmp_path / "source.wav"
    timings_path = tmp_path / "timings.json"
    _write_silent_wav(audio_path, sample_rate=16_000, duration_seconds=858.282)
    timings_path.write_text(
        """
[
  {"slide_number": 1, "section": "Verse 1", "lyrics": "Line 1", "click_timestamp": 10.571},
  {"slide_number": 2, "section": "Verse 1", "lyrics": "Line 2", "click_timestamp": 212.317}
]
""".strip(),
        encoding="utf-8",
    )

    clip_range = _suggest_profile_clip_range(audio_path, timings_path)

    assert clip_range == (10.571, 214.317)


def test_suggest_profile_clip_range_skips_when_timings_cover_full_audio(tmp_path) -> None:
    audio_path = tmp_path / "source.wav"
    timings_path = tmp_path / "timings.json"
    _write_silent_wav(audio_path, sample_rate=16_000, duration_seconds=30.0)
    timings_path.write_text(
        """
[
  {"slide_number": 1, "section": "Verse 1", "lyrics": "Line 1", "click_timestamp": 0.0},
  {"slide_number": 2, "section": "Verse 1", "lyrics": "Line 2", "click_timestamp": 29.0}
]
""".strip(),
        encoding="utf-8",
    )

    clip_range = _suggest_profile_clip_range(audio_path, timings_path)

    assert clip_range is None


def test_profile_has_coarse_artifacts_requires_new_coarse_files(tmp_path) -> None:
    profile_path = tmp_path / "profile"
    profile_path.mkdir()
    for name in ("profile.json", "reference_features.npy"):
        (profile_path / name).write_bytes(b"stub")

    assert _profile_has_coarse_artifacts(profile_path) is False

    for name in (
        "coarse_signatures.npy",
        "coarse_timestamps.npy",
        "coarse_frame_indexes.npy",
    ):
        (profile_path / name).write_bytes(b"stub")

    assert _profile_has_coarse_artifacts(profile_path) is True
