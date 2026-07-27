import json
import wave
from pathlib import Path

from lyrics_aligner.application.song_library import SongLibrary


def _write_test_wav(path: Path, *, sample_rate: int = 1000, frame_count: int = 1000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


def test_song_library_creates_song_assets_and_index(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))

    record = library.create_song(
        title="Amazing Grace",
        lyrics_text="Verse 1\nAmazing grace\nHow sweet the sound",
        audio_filename="amazing-grace.wav",
        audio_bytes=b"RIFFdemo",
    )

    assert record.song_id == "amazing-grace"
    assert library.audio_path(record.song_id).read_bytes() == b"RIFFdemo"
    assert library.reference_audio_path(record.song_id).read_bytes() == b"RIFFdemo"
    assert library.lyrics_path(record.song_id).read_text(encoding="utf-8").startswith("Verse 1")
    slides = json.loads(library.slides_path(record.song_id).read_text(encoding="utf-8"))
    assert slides[0]["section"] == "Verse 1"
    assert slides[0]["lyrics"] == "Amazing grace\nHow sweet the sound"
    assert library.list_songs()[0].title == "Amazing Grace"


def test_song_library_can_create_title_only_song(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))

    record = library.create_song(title="Untitled Capture")

    assert record.song_id == "untitled-capture"
    assert library.audio_path(record.song_id) is None
    assert library.reference_audio_path(record.song_id) is None
    assert library.lyrics_text(record.song_id) == ""
    slides = json.loads(library.slides_path(record.song_id).read_text(encoding="utf-8"))
    assert slides == []


def test_song_library_adds_default_section_when_lyrics_have_no_headings(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))

    record = library.create_song(
        title="Simple Song",
        lyrics_text="Amazing grace\nHow sweet the sound",
        audio_filename="simple.wav",
        audio_bytes=b"RIFFdemo",
    )

    lyrics = library.lyrics_path(record.song_id).read_text(encoding="utf-8")

    assert lyrics.startswith("Verse 1\n")


def test_song_library_converts_non_wav_uploads_for_reference_build(tmp_path: Path) -> None:
    conversions: list[tuple[str, str]] = []

    def converter(source: Path, target: Path) -> None:
        conversions.append((source.name, target.name))
        target.write_bytes(b"RIFFconverted")

    library = SongLibrary(str(tmp_path / "library"), audio_converter=converter)

    record = library.create_song(
        title="Simple Song",
        lyrics_text="Verse 1\nAmazing grace",
        audio_filename="simple.mp3",
        audio_bytes=b"ID3demo",
    )

    assert conversions == [("simple.mp3", "reference.wav")]
    assert library.audio_path(record.song_id).read_bytes() == b"ID3demo"
    assert library.reference_audio_path(record.song_id).read_bytes() == b"RIFFconverted"


def test_song_library_can_create_song_without_uploaded_audio(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))

    record = library.create_song(
        title="Lyrics Only Song",
        lyrics_text="Verse 1\nLine one\nLine two",
    )

    assert record.audio_filename == ""
    assert record.reference_audio_filename == ""
    assert record.clip_start_seconds == 0.0
    assert record.clip_end_seconds is None
    assert library.audio_path(record.song_id) is None
    assert library.reference_audio_path(record.song_id) is None
    assert library.lyrics_path(record.song_id).read_text(encoding="utf-8").startswith("Verse 1")


def test_song_library_can_attach_audio_later_to_existing_song(tmp_path: Path) -> None:
    conversions: list[tuple[str, str]] = []

    def converter(source: Path, target: Path) -> None:
        conversions.append((source.name, target.name))
        target.write_bytes(b"RIFFattached")

    library = SongLibrary(str(tmp_path / "library"), audio_converter=converter)
    record = library.create_song(
        title="Lyrics Only Song",
        lyrics_text="Verse 1\nLine one\nLine two",
    )

    updated = library.attach_audio(
        record.song_id,
        audio_filename="later.m4a",
        audio_bytes=b"m4adata",
    )

    assert conversions == [("later.m4a", "reference.wav")]
    assert updated.audio_filename == "later.m4a"
    assert updated.reference_audio_filename == "reference.wav"
    assert library.audio_path(record.song_id).read_bytes() == b"m4adata"
    assert library.reference_audio_path(record.song_id).read_bytes() == b"RIFFattached"


def test_song_library_can_delete_audio_and_clear_profile_state(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    source_audio = tmp_path / "source.wav"
    _write_test_wav(source_audio, sample_rate=1000, frame_count=2000)
    record = library.create_song(
        title="Audio Song",
        lyrics_text="Verse 1\nLine one\nLine two",
        audio_filename="audio-song.wav",
        audio_bytes=source_audio.read_bytes(),
    )
    library.update_timings(record.song_id)
    library.update_profile_directory(record.song_id)
    profile_path = library.profile_path(record.song_id)
    profile_path.mkdir(parents=True, exist_ok=True)

    updated = library.delete_audio(record.song_id)

    assert updated.audio_filename == ""
    assert updated.reference_audio_filename == ""
    assert updated.profile_directory == ""
    assert updated.clip_reference_audio_filename == ""
    assert library.audio_path(record.song_id) is None
    assert library.reference_audio_path(record.song_id) is None
    assert not profile_path.exists()


def test_song_library_updates_song_status_when_timings_are_saved(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    record = library.create_song(
        title="Amazing Grace",
        lyrics_text="Verse 1\nAmazing grace",
        audio_filename="amazing-grace.wav",
        audio_bytes=b"RIFFdemo",
    )

    updated = library.update_timings(record.song_id)

    assert updated.timings_filename == "timings.json"
    assert library.get_song(record.song_id).status == "Timed"


def test_song_library_updates_song_status_when_profile_is_saved(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    record = library.create_song(
        title="Amazing Grace",
        lyrics_text="Verse 1\nAmazing grace",
        audio_filename="amazing-grace.wav",
        audio_bytes=b"RIFFdemo",
    )

    updated = library.update_profile_directory(record.song_id)

    assert updated.profile_directory == "profile"
    assert library.get_song(record.song_id).status == "Ready"


def test_song_library_can_save_clip_and_sync_timing_state(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    source_audio = tmp_path / "source.wav"
    _write_test_wav(source_audio, sample_rate=1000, frame_count=2000)
    record = library.create_song(
        title="Amazing Grace",
        lyrics_text="Verse 1\nAmazing grace\nHow sweet the sound\nThat saved a wretch like me",
        audio_filename="amazing-grace.wav",
        audio_bytes=source_audio.read_bytes(),
    )
    library.update_timings(record.song_id)
    library.update_profile_directory(record.song_id)
    library.timings_path(record.song_id).write_text(
        json.dumps(
            [
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "Amazing grace",
                    "line_number": 1,
                    "line_count": 3,
                    "click_timestamp": 0.1,
                },
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "How sweet the sound",
                    "line_number": 2,
                    "line_count": 3,
                    "click_timestamp": 0.8,
                },
                {
                    "slide_number": 1,
                    "section": "Verse 1",
                    "lyrics": "That saved a wretch like me",
                    "line_number": 3,
                    "line_count": 3,
                    "click_timestamp": 1.6,
                },
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    profile_path = library.profile_path(record.song_id)
    profile_path.mkdir(parents=True, exist_ok=True)

    updated = library.update_clip(
        record.song_id,
        clip_start_seconds=0.5,
        clip_end_seconds=1.5,
    )

    assert updated.clip_start_seconds == 0.5
    assert updated.clip_end_seconds == 1.5
    assert updated.timings_filename == "timings.json"
    assert updated.profile_directory == ""
    assert not profile_path.exists()
    synced_timings = json.loads(library.timings_path(record.song_id).read_text(encoding="utf-8"))
    assert [entry["lyrics"] for entry in synced_timings] == ["How sweet the sound"]
    assert synced_timings[0]["click_timestamp"] == 0.3
    clip_path = library.reference_audio_path(record.song_id)
    assert clip_path is not None
    assert clip_path.name == "reference_clip.wav"


def test_song_library_clip_export_uses_trimmed_reference_audio(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    source_audio = tmp_path / "source.wav"
    _write_test_wav(source_audio, sample_rate=1000, frame_count=5000)
    record = library.create_song(
        title="Clip Test",
        lyrics_text="Verse 1\nLine one",
        audio_filename="clip-test.wav",
        audio_bytes=source_audio.read_bytes(),
    )

    updated = library.update_clip(
        record.song_id,
        clip_start_seconds=1.0,
        clip_end_seconds=3.5,
    )

    assert updated.clip_reference_audio_filename == "reference_clip.wav"
    clip_path = library.reference_audio_path(record.song_id)
    assert clip_path is not None
    with wave.open(str(clip_path), "rb") as handle:
        assert handle.getframerate() == 1000
        assert handle.getnframes() == 2500


def test_song_library_can_reset_saved_timings_and_profile(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    record = library.create_song(
        title="Reset Me",
        lyrics_text="Verse 1\nLine one",
        audio_filename="reset.wav",
        audio_bytes=b"RIFFdemo",
    )
    library.update_timings(record.song_id)
    library.update_profile_directory(record.song_id)
    library.timings_path(record.song_id).write_text("[]\n", encoding="utf-8")
    profile_path = library.profile_path(record.song_id)
    profile_path.mkdir(parents=True, exist_ok=True)

    updated = library.reset_timings(record.song_id)

    assert updated.timings_filename == ""
    assert updated.profile_directory == ""
    assert not library.timings_path(record.song_id).exists()
    assert not profile_path.exists()


def test_song_library_edit_lyrics_regenerates_slides_and_clears_timing_state(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    record = library.create_song(
        title="Amazing Grace",
        lyrics_text="Verse 1\nAmazing grace",
        audio_filename="amazing-grace.wav",
        audio_bytes=b"RIFFdemo",
    )
    library.update_timings(record.song_id)
    library.update_profile_directory(record.song_id)

    updated = library.update_lyrics(
        record.song_id,
        "Verse 1\nNew line one\nNew line two",
    )

    assert updated.status == "Not timed"
    assert library.get_song(record.song_id).timings_filename == ""
    assert library.get_song(record.song_id).profile_directory == ""
    slides = json.loads(library.slides_path(record.song_id).read_text(encoding="utf-8"))
    assert slides[0]["lyrics"] == "New line one\nNew line two"


def test_song_library_can_replace_slides_and_clear_profile_state(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    record = library.create_song(
        title="Smooth Criminal",
        lyrics_text="Verse 1\nLine one\nLine two\nLine three",
        audio_filename="smooth.wav",
        audio_bytes=b"RIFFdemo",
    )
    library.update_profile_directory(record.song_id)
    profile_path = library.profile_path(record.song_id)
    profile_path.mkdir(parents=True, exist_ok=True)

    updated = library.replace_slides(
        record.song_id,
        [
            {"slide_number": 1, "section": "Verse 1", "lyrics": "Line one\nLine two"},
            {"slide_number": 2, "section": "Verse 1", "lyrics": "Line three"},
        ],
    )

    slides = json.loads(library.slides_path(record.song_id).read_text(encoding="utf-8"))
    assert updated.profile_directory == ""
    assert slides[1]["slide_number"] == 2
    assert slides[1]["lyrics"] == "Line three"
    assert not profile_path.exists()


def test_song_library_can_replace_slides_with_one_line_per_slide(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    record = library.create_song(
        title="Line By Line",
        lyrics_text="Verse 1\nLine one\nLine two\nLine three",
        audio_filename="line.wav",
        audio_bytes=b"RIFFdemo",
    )

    library.replace_slides(
        record.song_id,
        [
            {"slide_number": 1, "section": "Verse 1", "lyrics": "Line one"},
            {"slide_number": 2, "section": "Verse 1", "lyrics": "Line two"},
            {"slide_number": 3, "section": "Verse 1", "lyrics": "Line three"},
        ],
    )

    slides = json.loads(library.slides_path(record.song_id).read_text(encoding="utf-8"))
    assert [entry["slide_number"] for entry in slides] == [1, 2, 3]
    assert [entry["lyrics"] for entry in slides] == ["Line one", "Line two", "Line three"]


def test_song_library_can_delete_song_and_remove_assets(tmp_path: Path) -> None:
    library = SongLibrary(str(tmp_path / "library"))
    record = library.create_song(
        title="Delete Me",
        lyrics_text="Verse 1\nLine one",
        audio_filename="delete-me.wav",
        audio_bytes=b"RIFFdemo",
    )

    library.delete_song(record.song_id)

    assert library.list_songs() == ()
    assert not library.song_path(record.song_id).exists()
