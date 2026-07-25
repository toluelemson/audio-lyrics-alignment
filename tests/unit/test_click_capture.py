import json
from pathlib import Path

import pytest

from lyrics_aligner.application.click_capture import (
    ClickCaptureSession,
    build_captured_slide_cues,
    load_slide_script,
    write_captured_slide_cues,
)


def test_load_slide_script_reads_slide_order_without_timing(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                },
                {
                    "slide_number": 2,
                    "section": "Verse 2",
                    "lyrics": "How sweet the sound",
                },
            ]
        ),
        encoding="utf-8",
    )

    cues = load_slide_script(str(path))

    assert [cue.slide_number for cue in cues] == [1, 2]
    assert cues[0].section == "Verse 1"
    assert cues[1].lyrics == "How sweet the sound"
    assert cues[0].line_number == 1
    assert cues[0].line_count == 1


def test_load_slide_script_expands_multiline_lyrics_into_line_cues(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace\nHow sweet the sound",
                }
            ]
        ),
        encoding="utf-8",
    )

    cues = load_slide_script(str(path))

    assert len(cues) == 2
    assert cues[0].lyrics == "Amazing grace"
    assert cues[0].line_number == 1
    assert cues[0].line_count == 2
    assert cues[1].lyrics == "How sweet the sound"
    assert cues[1].line_number == 2
    assert cues[1].line_count == 2


def test_build_captured_slide_cues_applies_click_timestamps_in_order(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                },
                {
                    "slide_number": 2,
                    "section": "Verse 2",
                    "lyrics": "How sweet the sound",
                },
            ]
        ),
        encoding="utf-8",
    )

    cues = load_slide_script(str(path))
    captured = build_captured_slide_cues(cues, (0.0, 12.5))

    assert captured[0].click_timestamp == pytest.approx(0.0)
    assert captured[1].click_timestamp == pytest.approx(12.5)


def test_click_capture_session_tracks_progress_and_completion(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                },
                {
                    "slide_number": 2,
                    "section": "Verse 2",
                    "lyrics": "How sweet the sound",
                },
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    session.started_at = 100.0
    first = session.mark_current(now=100.25)
    second = session.mark_current(now=101.0)

    assert first.click_timestamp == pytest.approx(0.25)
    assert second.click_timestamp == pytest.approx(1.0)
    assert session.current_index() == 2
    assert session.is_complete() is True


def test_click_capture_session_accepts_explicit_elapsed_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                },
                {
                    "slide_number": 2,
                    "section": "Verse 2",
                    "lyrics": "How sweet the sound",
                },
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    first = session.mark_current(elapsed=3.25)
    second = session.mark_current(elapsed=7.5)

    assert first.click_timestamp == pytest.approx(3.25)
    assert second.click_timestamp == pytest.approx(7.5)
    assert session.current_index() == 2


def test_click_capture_session_can_undo_last_click(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                },
                {
                    "slide_number": 2,
                    "section": "Verse 2",
                    "lyrics": "How sweet the sound",
                },
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    session.started_at = 50.0
    session.mark_current(now=50.5)
    session.mark_current(now=51.0)
    removed = session.undo_last()

    assert removed.slide_number == 2
    assert removed.click_timestamp == pytest.approx(1.0)
    assert session.current_index() == 1
    assert session.is_complete() is False


def test_click_capture_session_can_finish_and_lock_capture(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                }
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    session.started_at = 10.0
    session.mark_current(now=10.25)
    captured = session.finish()

    assert session.finished is True
    assert captured[0].click_timestamp == pytest.approx(0.25)
    with pytest.raises(ValueError, match="already finished"):
        session.mark_current(now=10.5)


def test_click_capture_session_rejects_incomplete_finish(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                },
                {
                    "slide_number": 2,
                    "section": "Verse 2",
                    "lyrics": "How sweet the sound",
                },
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    session.started_at = 10.0
    session.mark_current(now=10.25)

    with pytest.raises(ValueError, match="all slide clicks must be captured"):
        session.finish()


def test_build_captured_slide_cues_rejects_descending_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                },
                {
                    "slide_number": 2,
                    "section": "Verse 2",
                    "lyrics": "How sweet the sound",
                },
            ]
        ),
        encoding="utf-8",
    )

    cues = load_slide_script(str(path))

    with pytest.raises(ValueError, match="sorted ascending"):
        build_captured_slide_cues(cues, (5.0, 4.0))


def test_write_captured_slide_cues_writes_click_timestamp_json(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                }
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    captured = build_captured_slide_cues(cues, (0.75,))
    output_path = tmp_path / "captured.json"

    write_captured_slide_cues(str(output_path), captured)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved[0]["click_timestamp"] == pytest.approx(0.75)
    assert saved[0]["section"] == "Verse 1"
    assert saved[0]["line_number"] == 1
    assert saved[0]["line_count"] == 1
