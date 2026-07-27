from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from lyrics_aligner.application.slide_template import SlideTemplateConfig, build_slide_template


class AudioConverter(Protocol):
    def __call__(self, source_path: Path, target_wav_path: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class SongRecord:
    song_id: str
    title: str
    audio_filename: str
    reference_audio_filename: str
    lyrics_filename: str
    slides_filename: str
    timings_filename: str
    profile_directory: str
    clip_start_seconds: float
    clip_end_seconds: float | None
    clip_reference_audio_filename: str
    created_at: str
    updated_at: str

    @property
    def status(self) -> str:
        if self.profile_directory:
            return "Ready"
        if self.timings_filename:
            return "Timed"
        return "Not timed"


class SongLibrary:
    def __init__(
        self,
        root_path: str,
        *,
        audio_converter: AudioConverter | None = None,
    ) -> None:
        self._root_path = Path(root_path).expanduser().resolve()
        self._songs_path = self._root_path / "songs"
        self._index_path = self._root_path / "index.json"
        self._audio_converter = audio_converter or convert_audio_to_wav

    def list_songs(self) -> tuple[SongRecord, ...]:
        payload = self._read_index()
        records = [
            SongRecord(
                song_id=str(entry["song_id"]),
                title=str(entry["title"]),
                audio_filename=str(entry["audio_filename"]),
                reference_audio_filename=str(
                    entry.get("reference_audio_filename", entry["audio_filename"])
                ),
                lyrics_filename=str(entry["lyrics_filename"]),
                slides_filename=str(entry["slides_filename"]),
                timings_filename=str(entry.get("timings_filename", "")),
                profile_directory=str(entry.get("profile_directory", "")),
                clip_start_seconds=float(entry.get("clip_start_seconds", 0.0)),
                clip_end_seconds=(
                    None
                    if entry.get("clip_end_seconds") in (None, "")
                    else float(entry["clip_end_seconds"])
                ),
                clip_reference_audio_filename=str(entry.get("clip_reference_audio_filename", "")),
                created_at=str(entry["created_at"]),
                updated_at=str(entry["updated_at"]),
            )
            for entry in payload
        ]
        return tuple(records)

    def create_song(
        self,
        *,
        title: str,
        lyrics_text: str,
        audio_filename: str = "",
        audio_bytes: bytes = b"",
    ) -> SongRecord:
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("title must not be empty")
        if not lyrics_text.strip():
            raise ValueError("lyrics text must not be empty")
        if bool(audio_filename.strip()) != bool(audio_bytes):
            raise ValueError("audio filename and audio bytes must be provided together")

        song_id = self._next_song_id(normalized_title)
        song_path = self.song_path(song_id)
        song_path.mkdir(parents=True, exist_ok=False)

        safe_audio_name = _sanitize_filename(audio_filename) if audio_filename.strip() else ""
        lyrics_filename = "lyrics.txt"
        slides_filename = "slides.json"
        timings_filename = ""
        profile_directory = ""
        clip_start_seconds = 0.0
        clip_end_seconds = None
        clip_reference_audio_filename = ""
        reference_audio_filename = safe_audio_name

        if safe_audio_name:
            audio_path = song_path / safe_audio_name
            audio_path.write_bytes(audio_bytes)
            if audio_path.suffix.lower() != ".wav":
                reference_audio_filename = "reference.wav"
                self._audio_converter(audio_path, song_path / reference_audio_filename)
        else:
            reference_audio_filename = ""

        normalized_lyrics = _normalize_lyrics_text(lyrics_text)
        (song_path / lyrics_filename).write_text(normalized_lyrics + "\n", encoding="utf-8")
        slides = build_slide_template(
            normalized_lyrics,
            SlideTemplateConfig(initial_timestamp=0.0, timestamp_step=20.0),
        )
        (song_path / slides_filename).write_text(
            json.dumps(
                [
                    {
                        "slide_number": cue.slide_number,
                        "section": cue.section,
                        "lyrics": cue.lyrics,
                    }
                    for cue in slides
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        now = _utc_now()
        record = SongRecord(
            song_id=song_id,
            title=normalized_title,
            audio_filename=safe_audio_name,
            reference_audio_filename=reference_audio_filename,
            lyrics_filename=lyrics_filename,
            slides_filename=slides_filename,
            timings_filename=timings_filename,
            profile_directory=profile_directory,
            clip_start_seconds=clip_start_seconds,
            clip_end_seconds=clip_end_seconds,
            clip_reference_audio_filename=clip_reference_audio_filename,
            created_at=now,
            updated_at=now,
        )
        payload = self._read_index()
        payload.append(asdict(record))
        self._write_index(payload)
        return record

    def update_timings(self, song_id: str, timings_filename: str = "timings.json") -> SongRecord:
        payload = self._read_index()
        for index, entry in enumerate(payload):
            if str(entry["song_id"]) != song_id:
                continue
            entry["timings_filename"] = timings_filename
            entry["updated_at"] = _utc_now()
            payload[index] = entry
            self._write_index(payload)
            return self._record_from_entry(entry)
        raise ValueError(f"unknown song id: {song_id}")

    def update_profile_directory(
        self,
        song_id: str,
        profile_directory: str = "profile",
    ) -> SongRecord:
        payload = self._read_index()
        for index, entry in enumerate(payload):
            if str(entry["song_id"]) != song_id:
                continue
            entry["profile_directory"] = profile_directory
            entry["updated_at"] = _utc_now()
            payload[index] = entry
            self._write_index(payload)
            return self._record_from_entry(entry)
        raise ValueError(f"unknown song id: {song_id}")

    def update_clip(
        self,
        song_id: str,
        *,
        clip_start_seconds: float,
        clip_end_seconds: float | None,
    ) -> SongRecord:
        if clip_start_seconds < 0:
            raise ValueError("clip start must be non-negative")
        if clip_end_seconds is not None and clip_end_seconds <= clip_start_seconds:
            raise ValueError("clip end must be greater than clip start")

        payload = self._read_index()
        for index, entry in enumerate(payload):
            if str(entry["song_id"]) != song_id:
                continue
            song_path = self.song_path(song_id)
            previous_clip_start_seconds = float(entry.get("clip_start_seconds", 0.0))
            timings_filename = str(entry.get("timings_filename", ""))
            synced_timings_available = False
            if timings_filename:
                timings_path = song_path / timings_filename
                if timings_path.exists():
                    synced_timings_available = self._sync_timings_to_clip(
                        timings_path,
                        previous_clip_start_seconds=previous_clip_start_seconds,
                        clip_start_seconds=clip_start_seconds,
                        clip_end_seconds=clip_end_seconds,
                    )
            profile_directory = str(entry.get("profile_directory", ""))
            if profile_directory:
                profile_path = song_path / profile_directory
                if profile_path.exists():
                    shutil.rmtree(profile_path)
            source_reference_filename = str(entry.get("reference_audio_filename", ""))
            clip_reference_audio_filename = ""
            if source_reference_filename:
                source_reference_path = song_path / source_reference_filename
                if source_reference_path.exists() and (
                    clip_start_seconds > 0 or clip_end_seconds is not None
                ):
                    clip_reference_audio_filename = "reference_clip.wav"
                    _export_wav_clip(
                        source_reference_path,
                        song_path / clip_reference_audio_filename,
                        clip_start_seconds=clip_start_seconds,
                        clip_end_seconds=clip_end_seconds,
                    )
            previous_clip_reference = str(entry.get("clip_reference_audio_filename", ""))
            if previous_clip_reference and previous_clip_reference != clip_reference_audio_filename:
                previous_clip_path = song_path / previous_clip_reference
                if previous_clip_path.exists():
                    previous_clip_path.unlink()
            entry["clip_start_seconds"] = round(clip_start_seconds, 3)
            entry["clip_end_seconds"] = (
                None if clip_end_seconds is None else round(clip_end_seconds, 3)
            )
            entry["clip_reference_audio_filename"] = clip_reference_audio_filename
            entry["timings_filename"] = timings_filename if synced_timings_available else ""
            entry["profile_directory"] = ""
            entry["updated_at"] = _utc_now()
            payload[index] = entry
            self._write_index(payload)
            return self._record_from_entry(entry)
        raise ValueError(f"unknown song id: {song_id}")

    def reset_timings(self, song_id: str) -> SongRecord:
        payload = self._read_index()
        for index, entry in enumerate(payload):
            if str(entry["song_id"]) != song_id:
                continue
            song_path = self.song_path(song_id)
            timings_filename = str(entry.get("timings_filename", ""))
            if timings_filename:
                timings_path = song_path / timings_filename
                if timings_path.exists():
                    timings_path.unlink()
            profile_directory = str(entry.get("profile_directory", ""))
            if profile_directory:
                profile_path = song_path / profile_directory
                if profile_path.exists():
                    shutil.rmtree(profile_path)
            entry["timings_filename"] = ""
            entry["profile_directory"] = ""
            entry["updated_at"] = _utc_now()
            payload[index] = entry
            self._write_index(payload)
            return self._record_from_entry(entry)
        raise ValueError(f"unknown song id: {song_id}")

    def attach_audio(
        self,
        song_id: str,
        *,
        audio_filename: str,
        audio_bytes: bytes,
    ) -> SongRecord:
        if not audio_filename.strip():
            raise ValueError("audio filename must not be empty")
        if not audio_bytes:
            raise ValueError("audio file must not be empty")

        payload = self._read_index()
        for index, entry in enumerate(payload):
            if str(entry["song_id"]) != song_id:
                continue

            song_path = self.song_path(song_id)
            previous_audio_filename = str(entry.get("audio_filename", ""))
            previous_reference_filename = str(
                entry.get("reference_audio_filename", previous_audio_filename)
            )
            for filename in {previous_audio_filename, previous_reference_filename}:
                if not filename:
                    continue
                existing_path = song_path / filename
                if existing_path.exists():
                    existing_path.unlink()

            safe_audio_name = _sanitize_filename(audio_filename)
            reference_audio_filename = safe_audio_name
            audio_path = song_path / safe_audio_name
            audio_path.write_bytes(audio_bytes)
            if audio_path.suffix.lower() != ".wav":
                reference_audio_filename = "reference.wav"
                self._audio_converter(audio_path, song_path / reference_audio_filename)

            entry["audio_filename"] = safe_audio_name
            entry["reference_audio_filename"] = reference_audio_filename
            entry["profile_directory"] = ""
            entry["clip_start_seconds"] = 0.0
            entry["clip_end_seconds"] = None
            previous_clip_reference = str(entry.get("clip_reference_audio_filename", ""))
            if previous_clip_reference:
                previous_clip_path = song_path / previous_clip_reference
                if previous_clip_path.exists():
                    previous_clip_path.unlink()
            entry["clip_reference_audio_filename"] = ""
            entry["updated_at"] = _utc_now()
            payload[index] = entry
            self._write_index(payload)
            return self._record_from_entry(entry)
        raise ValueError(f"unknown song id: {song_id}")

    def delete_audio(self, song_id: str) -> SongRecord:
        payload = self._read_index()
        for index, entry in enumerate(payload):
            if str(entry["song_id"]) != song_id:
                continue

            song_path = self.song_path(song_id)
            filenames_to_remove = {
                str(entry.get("audio_filename", "")),
                str(entry.get("reference_audio_filename", "")),
                str(entry.get("clip_reference_audio_filename", "")),
            }
            for filename in filenames_to_remove:
                if not filename:
                    continue
                existing_path = song_path / filename
                if existing_path.exists():
                    existing_path.unlink()

            profile_directory = str(entry.get("profile_directory", ""))
            if profile_directory:
                profile_path = song_path / profile_directory
                if profile_path.exists():
                    shutil.rmtree(profile_path)

            entry["audio_filename"] = ""
            entry["reference_audio_filename"] = ""
            entry["clip_reference_audio_filename"] = ""
            entry["clip_start_seconds"] = 0.0
            entry["clip_end_seconds"] = None
            entry["profile_directory"] = ""
            entry["updated_at"] = _utc_now()
            payload[index] = entry
            self._write_index(payload)
            return self._record_from_entry(entry)
        raise ValueError(f"unknown song id: {song_id}")

    def update_lyrics(
        self,
        song_id: str,
        lyrics_text: str,
    ) -> SongRecord:
        normalized_lyrics = _normalize_lyrics_text(lyrics_text)
        payload = self._read_index()
        for index, entry in enumerate(payload):
            if str(entry["song_id"]) != song_id:
                continue
            record = self._record_from_entry(entry)
            song_path = self.song_path(song_id)
            (song_path / record.lyrics_filename).write_text(
                normalized_lyrics + "\n",
                encoding="utf-8",
            )
            slides = build_slide_template(
                normalized_lyrics,
                SlideTemplateConfig(initial_timestamp=0.0, timestamp_step=20.0),
            )
            (song_path / record.slides_filename).write_text(
                json.dumps(
                    [
                        {
                            "slide_number": cue.slide_number,
                            "section": cue.section,
                            "lyrics": cue.lyrics,
                        }
                        for cue in slides
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            entry["timings_filename"] = ""
            entry["profile_directory"] = ""
            timings_path = song_path / record.timings_filename if record.timings_filename else None
            if timings_path is not None and timings_path.exists():
                timings_path.unlink()
            profile_path = song_path / record.profile_directory if record.profile_directory else None
            if profile_path is not None and profile_path.exists():
                shutil.rmtree(profile_path)
            clip_reference_path = (
                song_path / str(entry.get("clip_reference_audio_filename", ""))
                if str(entry.get("clip_reference_audio_filename", ""))
                else None
            )
            if clip_reference_path is not None and clip_reference_path.exists():
                clip_reference_path.unlink()
            entry["clip_reference_audio_filename"] = ""
            entry["clip_start_seconds"] = 0.0
            entry["clip_end_seconds"] = None
            entry["updated_at"] = _utc_now()
            payload[index] = entry
            self._write_index(payload)
            return self._record_from_entry(entry)
        raise ValueError(f"unknown song id: {song_id}")

    def delete_song(self, song_id: str) -> None:
        payload = self._read_index()
        remaining = [entry for entry in payload if str(entry["song_id"]) != song_id]
        if len(remaining) == len(payload):
            raise ValueError(f"unknown song id: {song_id}")
        song_path = self.song_path(song_id)
        if song_path.exists():
            shutil.rmtree(song_path)
        self._write_index(remaining)

    def get_song(self, song_id: str) -> SongRecord:
        for record in self.list_songs():
            if record.song_id == song_id:
                return record
        raise ValueError(f"unknown song id: {song_id}")

    def song_path(self, song_id: str) -> Path:
        return self._songs_path / song_id

    def audio_path(self, song_id: str) -> Path | None:
        record = self.get_song(song_id)
        if not record.audio_filename:
            return None
        return self.song_path(song_id) / record.audio_filename

    def lyrics_path(self, song_id: str) -> Path:
        record = self.get_song(song_id)
        return self.song_path(song_id) / record.lyrics_filename

    def reference_audio_path(self, song_id: str) -> Path | None:
        record = self.get_song(song_id)
        if record.clip_reference_audio_filename:
            return self.song_path(song_id) / record.clip_reference_audio_filename
        if not record.reference_audio_filename:
            return None
        return self.song_path(song_id) / record.reference_audio_filename

    def source_reference_audio_path(self, song_id: str) -> Path | None:
        record = self.get_song(song_id)
        if not record.reference_audio_filename:
            return None
        return self.song_path(song_id) / record.reference_audio_filename

    def slides_path(self, song_id: str) -> Path:
        record = self.get_song(song_id)
        return self.song_path(song_id) / record.slides_filename

    def timings_path(self, song_id: str, fallback_filename: str = "timings.json") -> Path:
        record = self.get_song(song_id)
        filename = record.timings_filename or fallback_filename
        return self.song_path(song_id) / filename

    def profile_path(self, song_id: str, fallback_directory: str = "profile") -> Path:
        record = self.get_song(song_id)
        directory = record.profile_directory or fallback_directory
        return self.song_path(song_id) / directory

    def _read_index(self) -> list[dict[str, object]]:
        if not self._index_path.exists():
            return []
        content = json.loads(self._index_path.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            raise ValueError("song library index must contain a JSON list")
        return [entry for entry in content if isinstance(entry, dict)]

    def _write_index(self, payload: list[dict[str, object]]) -> None:
        self._root_path.mkdir(parents=True, exist_ok=True)
        self._songs_path.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _sync_timings_to_clip(
        self,
        timings_path: Path,
        *,
        previous_clip_start_seconds: float,
        clip_start_seconds: float,
        clip_end_seconds: float | None,
    ) -> bool:
        content = json.loads(timings_path.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            timings_path.unlink()
            return False
        synced_entries: list[dict[str, object]] = []
        for entry in content:
            if not isinstance(entry, dict):
                continue
            timestamp = entry.get("click_timestamp")
            if not isinstance(timestamp, (int, float)):
                continue
            absolute_seconds = previous_clip_start_seconds + float(timestamp)
            if absolute_seconds < clip_start_seconds:
                continue
            if clip_end_seconds is not None and absolute_seconds > clip_end_seconds:
                continue
            synced_entry = dict(entry)
            synced_entry["click_timestamp"] = round(absolute_seconds - clip_start_seconds, 3)
            synced_entries.append(synced_entry)
        synced_entries.sort(key=lambda item: float(item.get("click_timestamp", 0.0)))
        if not synced_entries:
            timings_path.unlink()
            return False
        timings_path.write_text(json.dumps(synced_entries, indent=2) + "\n", encoding="utf-8")
        return True

    def _next_song_id(self, title: str) -> str:
        slug = _slugify(title)
        existing_ids = {record.song_id for record in self.list_songs()}
        if slug not in existing_ids:
            return slug
        suffix = 2
        while f"{slug}-{suffix}" in existing_ids:
            suffix += 1
        return f"{slug}-{suffix}"

    @staticmethod
    def _record_from_entry(entry: dict[str, object]) -> SongRecord:
        return SongRecord(
            song_id=str(entry["song_id"]),
            title=str(entry["title"]),
            audio_filename=str(entry["audio_filename"]),
            reference_audio_filename=str(
                entry.get("reference_audio_filename", entry["audio_filename"])
            ),
            lyrics_filename=str(entry["lyrics_filename"]),
            slides_filename=str(entry["slides_filename"]),
            timings_filename=str(entry.get("timings_filename", "")),
            profile_directory=str(entry.get("profile_directory", "")),
            clip_start_seconds=float(entry.get("clip_start_seconds", 0.0)),
            clip_end_seconds=(
                None
                if entry.get("clip_end_seconds") in (None, "")
                else float(entry["clip_end_seconds"])
            ),
            clip_reference_audio_filename=str(entry.get("clip_reference_audio_filename", "")),
            created_at=str(entry["created_at"]),
            updated_at=str(entry["updated_at"]),
        )


def _export_wav_clip(
    source_path: Path,
    target_path: Path,
    *,
    clip_start_seconds: float,
    clip_end_seconds: float | None,
) -> None:
    import wave

    with wave.open(str(source_path), "rb") as source:
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        start_frame = max(0, min(total_frames, round(clip_start_seconds * frame_rate)))
        end_frame = total_frames
        if clip_end_seconds is not None:
            end_frame = max(start_frame, min(total_frames, round(clip_end_seconds * frame_rate)))
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)

        with wave.open(str(target_path), "wb") as target:
            target.setnchannels(source.getnchannels())
            target.setsampwidth(source.getsampwidth())
            target.setframerate(frame_rate)
            target.writeframes(frames)


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return normalized or "song"


def _sanitize_filename(filename: str) -> str:
    base = Path(filename).name.strip()
    if not base:
        return "audio.bin"
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def _normalize_lyrics_text(lyrics_text: str) -> str:
    stripped = lyrics_text.strip()
    if not stripped:
        raise ValueError("lyrics text must not be empty")
    first_meaningful_line = next(
        (line.strip() for line in stripped.splitlines() if line.strip()),
        "",
    )
    if not first_meaningful_line.lower().startswith(
        ("verse", "chorus", "bridge", "intro", "outro", "tag", "refrain", "pre-chorus")
    ):
        return f"Verse 1\n{stripped}"
    return stripped


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def convert_audio_to_wav(source_path: Path, target_wav_path: Path) -> None:
    source = source_path.expanduser().resolve()
    target = target_wav_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".wav":
        shutil.copyfile(source, target)
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(target),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and target.exists():
            return

    afconvert = shutil.which("afconvert")
    if afconvert is not None:
        result = subprocess.run(
            [
                afconvert,
                "-f",
                "WAVE",
                "-d",
                "LEI16@16000",
                str(source),
                str(target),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and target.exists():
            return

    raise ValueError(
        "could not convert uploaded audio to WAV; install ffmpeg or afconvert and retry"
    )
