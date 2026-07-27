import json
from pathlib import Path

import pytest

from lyrics_aligner.application.click_capture import (
    ClickCaptureSession,
    build_captured_slide_cues,
    detect_onset_candidates,
    detect_vocal_candidates,
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


def test_click_capture_session_reports_first_line_before_start(tmp_path: Path) -> None:
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

    assert session.current_index() == 0
    assert session.is_complete() is False


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
    assert session.cue_states()[0]["locked"] is True


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


def test_click_capture_session_allows_reanchoring_after_finish(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {"slide_number": 1, "section": "Verse 1", "lyrics": "First"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Second"},
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    session.mark_current(elapsed=1.0)
    session.mark_current(elapsed=2.0)
    session.finish()

    corrected = session.set_anchor(1, elapsed=1.8)

    assert corrected.click_timestamp == pytest.approx(1.8)
    assert session.finished is False
    assert session.cue_states()[1]["locked"] is True


def test_click_capture_session_can_stop_before_last_line(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {"slide_number": 1, "section": "Verse 1", "lyrics": "First"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Second"},
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    session.mark_current(elapsed=1.0)

    captured = session.stop()

    assert session.finished is True
    assert len(captured) == 1
    assert captured[0].click_timestamp == pytest.approx(1.0)


def test_click_capture_session_can_restart_from_selected_line_and_keep_earlier_timings(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {"slide_number": 1, "section": "Verse 1", "lyrics": "First"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Second"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Third"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Fourth"},
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    session.mark_current(elapsed=1.0)
    session.mark_current(elapsed=2.0)
    session.mark_current(elapsed=3.0)
    session.mark_current(elapsed=4.0)
    session.stop()

    restarted = session.start_from(2, elapsed=3.5)

    assert restarted.click_timestamp == pytest.approx(3.5)
    assert session.finished is False
    assert session.cue_states()[0]["click_timestamp"] == pytest.approx(1.0)
    assert session.cue_states()[1]["click_timestamp"] == pytest.approx(2.0)
    assert session.cue_states()[2]["click_timestamp"] == pytest.approx(3.5)
    assert session.cue_states()[3]["click_timestamp"] is None
    assert session.current_index() == 3


def test_click_capture_session_can_draft_between_locked_anchors(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Short"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "A much longer middle line"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "End"},
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    session.set_anchor(0, elapsed=10.0)
    session.set_anchor(2, elapsed=22.0)
    drafted = session.draft_gaps()

    assert len(drafted) == 3
    assert drafted[0].click_timestamp == pytest.approx(10.0)
    assert drafted[2].click_timestamp == pytest.approx(22.0)
    assert drafted[1].click_timestamp > 10.0
    assert drafted[1].click_timestamp < 22.0
    assert session.cue_states()[1]["drafted"] is True
    assert session.cue_states()[1]["locked"] is False


def test_click_capture_session_nudging_promotes_line_to_locked_anchor(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {"slide_number": 1, "section": "Verse 1", "lyrics": "First"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Middle line"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Last"},
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(cues)

    session.start()
    session.set_anchor(0, elapsed=4.0)
    session.set_anchor(2, elapsed=10.0)
    session.draft_gaps()
    nudged = session.nudge_cue(1, 0.25)

    assert nudged.click_timestamp == pytest.approx(session.cue_timestamps[1])
    assert session.cue_states()[1]["locked"] is True
    assert session.cue_states()[1]["drafted"] is False
    assert session.anchor_history[-1] == 1


def test_click_capture_session_snaps_drafted_line_to_nearby_onset(tmp_path: Path) -> None:
    path = tmp_path / "slides.json"
    path.write_text(
        json.dumps(
            [
                {"slide_number": 1, "section": "Verse 1", "lyrics": "First"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Middle"},
                {"slide_number": 1, "section": "Verse 1", "lyrics": "Last"},
            ]
        ),
        encoding="utf-8",
    )
    cues = load_slide_script(str(path))
    session = ClickCaptureSession(
        cues,
        onset_candidates=(1.82,),
        onset_snap_tolerance_seconds=0.4,
    )

    session.start()
    session.set_anchor(0, elapsed=1.0)
    session.set_anchor(2, elapsed=3.0)
    drafted = session.draft_gaps()

    assert drafted[1].click_timestamp == pytest.approx(1.82)
    assert session.cue_states()[1]["drafted"] is True


def test_detect_onset_candidates_finds_simple_energy_rises(tmp_path: Path) -> None:
    import wave

    import numpy as np

    audio_path = tmp_path / "reference.wav"
    sample_rate = 16_000
    samples = np.zeros(sample_rate * 2, dtype=np.float32)
    samples[4_000:4_400] = 0.8
    samples[12_000:12_400] = 0.8
    pcm = np.asarray(np.clip(samples, -1.0, 1.0) * 32767, dtype="<i2")
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    candidates = detect_onset_candidates(str(audio_path))

    assert any(abs(candidate - 0.25) < 0.08 for candidate in candidates)
    assert any(abs(candidate - 0.75) < 0.08 for candidate in candidates)


def test_detect_vocal_candidates_prefers_voiced_segment_after_non_voiced_onset(tmp_path: Path) -> None:
    import wave

    import numpy as np

    audio_path = tmp_path / "reference.wav"
    sample_rate = 16_000
    duration_seconds = 2.0
    sample_count = int(sample_rate * duration_seconds)
    samples = np.zeros(sample_count, dtype=np.float32)

    burst_start = int(0.20 * sample_rate)
    burst_end = burst_start + 320
    noise = np.random.default_rng(7).uniform(-0.9, 0.9, burst_end - burst_start).astype(np.float32)
    samples[burst_start:burst_end] = noise

    vocal_start = int(0.90 * sample_rate)
    vocal_end = vocal_start + int(0.35 * sample_rate)
    positions = np.arange(vocal_end - vocal_start, dtype=np.float32)
    samples[vocal_start:vocal_end] = 0.35 * np.sin(2.0 * np.pi * 220.0 * positions / sample_rate)

    pcm = np.asarray(np.clip(samples, -1.0, 1.0) * 32767, dtype="<i2")
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    candidates = detect_vocal_candidates(str(audio_path))

    assert candidates
    assert abs(candidates[0] - 0.90) < 0.12


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
