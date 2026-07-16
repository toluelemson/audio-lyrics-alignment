"""Filesystem-backed reference profile loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lyrics_aligner.application.reference_builder import PROFILE_VERSION
from lyrics_aligner.domain.models import FeatureFrame, ReferenceProfile, SlideCue


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
        profile_version = profile_data.get("profile_version")
        if profile_version != PROFILE_VERSION:
            raise ValueError(
                f"profile.json must contain supported profile_version={PROFILE_VERSION!r}"
            )

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
        slide_cues = self._read_slide_cues(profile_data.get("slides"))
        return ReferenceProfile(
            name=name.strip(),
            frames=frames,
            metadata=metadata,
            slide_cues=slide_cues,
        )

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

    @staticmethod
    def _read_slide_cues(raw_cues: object) -> tuple[SlideCue, ...]:
        if raw_cues is None:
            return ()
        if not isinstance(raw_cues, list):
            raise ValueError("profile.json 'slides' must be a list when provided")

        cues: list[SlideCue] = []
        for entry in raw_cues:
            if not isinstance(entry, dict):
                raise ValueError("profile.json 'slides' entries must be objects")
            slide_number = entry.get("slide_number")
            section = entry.get("section")
            lyrics = entry.get("lyrics")
            reference_timestamp = entry.get("reference_timestamp")
            if (
                isinstance(slide_number, bool)
                or not isinstance(slide_number, int)
                or slide_number <= 0
            ):
                raise ValueError("slide_number must be a positive integer")
            if not isinstance(section, str) or not section.strip():
                raise ValueError("slide section must be a non-empty string")
            if not isinstance(lyrics, str) or not lyrics.strip():
                raise ValueError("slide lyrics must be a non-empty string")
            if not isinstance(reference_timestamp, (int, float)) or reference_timestamp < 0:
                raise ValueError("slide reference_timestamp must be a non-negative number")
            cues.append(
                SlideCue(
                    slide_number=slide_number,
                    section=section.strip(),
                    lyrics=lyrics,
                    reference_timestamp=float(reference_timestamp),
                )
            )
        return tuple(sorted(cues, key=lambda cue: cue.reference_timestamp))
