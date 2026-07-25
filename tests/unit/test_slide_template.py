import json
import subprocess
import sys
from pathlib import Path

import pytest

from lyrics_aligner.application.slide_template import (
    SlideTemplateConfig,
    build_slide_template,
)

AMAZING_GRACE_LYRICS = """
John Newton, 1779
*[Key: Eb]*

**Verse 1**
Amazing Grace
How sweet the sound,
That saved a wretch like me!
I once was lost,
But now am found;
Was blind, but now I see.

**Verse 2**
'Twas grace that taught
My heart to fear,
And grace my fears relieved;
How precious
Did that grace appear
The hour I first believed!
"""


def test_build_slide_template_extracts_sections_and_placeholder_timestamps() -> None:
    cues = build_slide_template(
        AMAZING_GRACE_LYRICS,
        SlideTemplateConfig(initial_timestamp=5.0, timestamp_step=18.0),
    )

    assert len(cues) == 2
    assert cues[0].slide_number == 1
    assert cues[0].section == "Verse 1"
    assert cues[0].reference_timestamp == pytest.approx(5.0)
    assert cues[0].lyrics.startswith("Amazing Grace")
    assert cues[1].section == "Verse 2"
    assert cues[1].reference_timestamp == pytest.approx(23.0)


def test_build_slide_template_rejects_missing_sections() -> None:
    with pytest.raises(ValueError, match="no slide sections"):
        build_slide_template("Amazing Grace\nHow sweet the sound")


def test_cli_generates_slide_cue_json(tmp_path: Path) -> None:
    lyrics_path = tmp_path / "lyrics.txt"
    output_path = tmp_path / "slides.json"
    lyrics_path.write_text(AMAZING_GRACE_LYRICS, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/generate_slide_cues.py",
            "--lyrics",
            str(lyrics_path),
            "--output",
            str(output_path),
            "--timestamp-step",
            "15",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    summary = json.loads(result.stdout)
    generated = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["slides"] == 2
    assert generated[0]["section"] == "Verse 1"
    assert generated[1]["click_timestamp"] == pytest.approx(15.0)
