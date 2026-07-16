"""Filesystem-backed reference profile loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lyrics_aligner.domain.models import FeatureFrame, ReferenceProfile


class FilesystemReferenceProfileRepository:
    """Load a reference profile from a directory of prepared artifacts."""

    def load(self, path: str) -> ReferenceProfile:
        base_path = Path(path).expanduser().resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"Reference profile path does not exist: {base_path}")
        if not base_path.is_dir():
            raise ValueError(f"Reference profile path must be a directory: {base_path}")

        profile_data = self._read_json(base_path / "profile.json")
        features = self._read_features(base_path / "reference_features.npy")
        metadata = self._read_metadata(base_path / "metadata.json")

        name = profile_data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("profile.json must contain a non-empty string 'name'")

        frame_duration = profile_data.get("frame_duration_seconds")
        if not isinstance(frame_duration, (int, float)) or frame_duration <= 0:
            raise ValueError(
                "profile.json must contain a positive 'frame_duration_seconds' value"
            )

        timestamps = profile_data.get("timestamps_seconds")
        if timestamps is None:
            timestamp_values = [
                float(index) * float(frame_duration) for index in range(features.shape[0])
            ]
        else:
            if not isinstance(timestamps, list) or len(timestamps) != features.shape[0]:
                raise ValueError(
                    "profile.json 'timestamps_seconds' must be a list matching the "
                    "feature row count"
                )
            timestamp_values = [float(value) for value in timestamps]

        frames = tuple(
            FeatureFrame(
                values=np.array(row, dtype=np.float32, copy=True),
                observed_at=observed_at,
                frame_duration_seconds=float(frame_duration),
            )
            for observed_at, row in zip(timestamp_values, features, strict=True)
        )
        return ReferenceProfile(name=name.strip(), frames=frames, metadata=metadata)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(f"Missing reference profile file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
        if not isinstance(content, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return content

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        content = FilesystemReferenceProfileRepository._read_json(path)
        metadata: dict[str, str] = {}
        for key, value in content.items():
            metadata[str(key)] = str(value)
        return metadata

    @staticmethod
    def _read_features(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing reference profile file: {path}")
        features = np.load(path)
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("reference_features.npy must contain a 2D feature matrix")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("reference_features.npy must not be empty")
        if not np.isfinite(matrix).all():
            raise ValueError("reference_features.npy must contain only finite values")
        return matrix
