import logging

import numpy as np

from lyrics_aligner.domain.models import FeatureFrame, ReferenceProfile
from lyrics_aligner.main import _refresh_profile_slide_cues_from_saved_timings


def test_refresh_profile_slide_cues_from_saved_timings_uses_latest_saved_lines(tmp_path) -> None:
    slides_path = tmp_path / "timings.json"
    slides_path.write_text(
        """
[
  {
    "slide_number": 1,
    "section": "Verse 1",
    "lyrics": "Amazing Grace",
    "line_number": 1,
    "line_count": 1,
    "click_timestamp": 4.2
  },
  {
    "slide_number": 2,
    "section": "Verse 1",
    "lyrics": "How sweet the sound,",
    "line_number": 1,
    "line_count": 1,
    "click_timestamp": 8.5
  }
]
""".strip(),
        encoding="utf-8",
    )
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            FeatureFrame(
                values=np.array([0.1, 0.2], dtype=np.float32),
                observed_at=0.0,
                frame_duration_seconds=0.01,
            ),
        ),
        metadata={"slides_path": str(slides_path)},
        slide_cues=(),
    )

    refreshed = _refresh_profile_slide_cues_from_saved_timings(profile)

    assert [cue.slide_number for cue in refreshed.slide_cues] == [1, 2]
    assert [cue.lyrics for cue in refreshed.slide_cues] == [
        "Amazing Grace",
        "How sweet the sound,",
    ]
    assert [cue.reference_timestamp for cue in refreshed.slide_cues] == [4.2, 8.5]


def test_refresh_profile_slide_cues_from_saved_timings_keeps_original_when_file_missing(tmp_path) -> None:
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            FeatureFrame(
                values=np.array([0.1, 0.2], dtype=np.float32),
                observed_at=0.0,
                frame_duration_seconds=0.01,
            ),
        ),
        metadata={"slides_path": str(tmp_path / "missing.json")},
        slide_cues=(),
    )

    refreshed = _refresh_profile_slide_cues_from_saved_timings(profile)

    assert refreshed == profile
