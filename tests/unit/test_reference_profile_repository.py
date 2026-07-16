import json
from pathlib import Path

import numpy as np
import pytest

from lyrics_aligner.adapters.reference_profiles import (
    FilesystemReferenceProfileRepository,
)


def _write_profile_fixture(
    path: Path,
    *,
    features: np.ndarray | None = None,
    profile: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    path.mkdir()
    np.save(
        path / "reference_features.npy",
        features if features is not None else np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )
    (path / "profile.json").write_text(
        json.dumps(
            profile
            if profile is not None
            else {
                "name": "song-a",
                "frame_duration_seconds": 0.25,
                "timestamps_seconds": [0.0, 0.25],
            }
        ),
        encoding="utf-8",
    )
    if metadata is not None:
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_loads_reference_profile_from_directory(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile"
    _write_profile_fixture(
        profile_path,
        metadata={"song": "Example Song", "section": "Verse 1"},
    )

    repository = FilesystemReferenceProfileRepository()
    profile = repository.load(str(profile_path))

    assert profile.name == "song-a"
    assert len(profile.frames) == 2
    assert profile.frames[0].observed_at == pytest.approx(0.0)
    assert profile.frames[1].frame_duration_seconds == pytest.approx(0.25)
    assert profile.metadata == {"song": "Example Song", "section": "Verse 1"}


def test_load_raises_when_required_file_is_missing(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile"
    profile_path.mkdir()
    (profile_path / "profile.json").write_text(
        json.dumps({"name": "song-a", "frame_duration_seconds": 0.25}),
        encoding="utf-8",
    )

    repository = FilesystemReferenceProfileRepository()

    with pytest.raises(FileNotFoundError, match="reference_features.npy"):
        repository.load(str(profile_path))


def test_load_raises_when_feature_matrix_is_invalid(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile"
    _write_profile_fixture(
        profile_path,
        features=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )

    repository = FilesystemReferenceProfileRepository()

    with pytest.raises(ValueError, match="2D feature matrix"):
        repository.load(str(profile_path))


def test_load_raises_when_timestamp_count_does_not_match_features(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile"
    _write_profile_fixture(
        profile_path,
        profile={
            "name": "song-a",
            "frame_duration_seconds": 0.25,
            "timestamps_seconds": [0.0],
        },
    )

    repository = FilesystemReferenceProfileRepository()

    with pytest.raises(ValueError, match="matching the feature row count"):
        repository.load(str(profile_path))
