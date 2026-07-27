import pytest

from lyrics_aligner.application.operator_corrections import (
    CorrectionAnchorStoreConfig,
    ProfileCorrectionStore,
)


def test_store_writes_corrections_and_builds_anchor_after_repeated_matches(tmp_path) -> None:
    store = ProfileCorrectionStore(
        str(tmp_path / "profiles" / "song-a"),
        CorrectionAnchorStoreConfig(
            min_corrections_for_anchor=2,
            min_support_score_for_anchor=1.5,
            source_group_tolerance_seconds=1.0,
            target_group_tolerance_seconds=1.0,
        ),
    )

    store.record(
        store.build_record(
            profile_name="song-a",
            detected_reference_timestamp=12.0,
            chosen_reference_timestamp=18.0,
            chosen_slide_number=4,
            chosen_section="Verse 2",
            chosen_lyrics="Line 4",
        )
    )
    no_anchor = store.load_anchors()
    store.record(
        store.build_record(
            profile_name="song-a",
            detected_reference_timestamp=12.4,
            chosen_reference_timestamp=18.2,
            chosen_slide_number=4,
            chosen_section="Verse 2",
            chosen_lyrics="Line 4",
        )
    )
    anchors = store.load_anchors()

    assert no_anchor == ()
    assert len(anchors) == 1
    assert anchors[0].profile_name == "song-a"
    assert anchors[0].slide_number == 4
    assert anchors[0].correction_count == 2
    assert anchors[0].session_count == 2
    assert anchors[0].support_score > 1.99
    assert anchors[0].last_seen_at != ""
    assert anchors[0].source_reference_timestamp == pytest.approx(12.2)
    assert anchors[0].target_reference_timestamp == pytest.approx(18.1)


def test_store_ignores_corrections_without_detected_timestamp_for_anchor_building(tmp_path) -> None:
    store = ProfileCorrectionStore(str(tmp_path / "profiles" / "song-b"))

    store.record(
        store.build_record(
            profile_name="song-b",
            detected_reference_timestamp=None,
            chosen_reference_timestamp=8.0,
            chosen_slide_number=2,
            chosen_section="Chorus",
            chosen_lyrics="Line 2",
        )
    )

    assert store.load_anchors() == ()


def test_store_weights_low_confidence_and_no_vocal_corrections_more_strongly(tmp_path) -> None:
    store = ProfileCorrectionStore(
        str(tmp_path / "profiles" / "song-c"),
        CorrectionAnchorStoreConfig(
            min_corrections_for_anchor=2,
            min_support_score_for_anchor=2.2,
            source_group_tolerance_seconds=1.0,
            target_group_tolerance_seconds=1.0,
        ),
    )

    store.record(
        store.build_record(
            profile_name="song-c",
            detected_reference_timestamp=25.0,
            detected_confidence=0.2,
            chosen_reference_timestamp=30.0,
            chosen_slide_number=5,
            chosen_section="Bridge",
            chosen_lyrics="Line 5",
            no_vocal_detected=True,
            session_id="session-a",
        )
    )
    store.record(
        store.build_record(
            profile_name="song-c",
            detected_reference_timestamp=25.4,
            detected_confidence=0.3,
            chosen_reference_timestamp=30.3,
            chosen_slide_number=5,
            chosen_section="Bridge",
            chosen_lyrics="Line 5",
            no_vocal_detected=False,
            session_id="session-b",
        )
    )

    anchors = store.load_anchors()

    assert len(anchors) == 1
    assert anchors[0].support_score > 2.2
