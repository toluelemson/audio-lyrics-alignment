from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lyrics_aligner.domain.models import AudioChunk, FeatureFrame
from lyrics_aligner.ports.feature_extractor import FeatureExtractor

PROFILE_VERSION = "1"
TARGET_SAMPLE_RATE = 16_000
SUPPORTED_REFERENCE_AUDIO_ROLES = ("mixed", "vocals", "instrumental", "other")


@dataclass(frozen=True, slots=True)
class SlideInterval:
    slide_number: int
    section: str
    lyrics: str
    reference_timestamp: float
    end_timestamp: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True, slots=True)
class BuiltReferenceProfile:
    name: str
    frame_duration_seconds: float
    timestamps_seconds: tuple[float, ...]
    feature_matrix: np.ndarray
    slides: tuple[SlideInterval, ...]
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class ReferenceBuilderConfig:
    sample_rate: int = TARGET_SAMPLE_RATE
    block_size: int = 4_096

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise TypeError("block_size must be an integer")
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero")


class ReferenceProfileBuilder:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        config: ReferenceBuilderConfig | None = None,
    ) -> None:
        self._feature_extractor = feature_extractor
        self._config = config or ReferenceBuilderConfig()

    def build(
        self,
        audio_path: str,
        slides_path: str,
        profile_name: str,
        *,
        audio_role: str = "mixed",
        companion_audio_path: str | None = None,
    ) -> BuiltReferenceProfile:
        if audio_role not in SUPPORTED_REFERENCE_AUDIO_ROLES:
            raise ValueError(
                "audio_role must be one of "
                f"{SUPPORTED_REFERENCE_AUDIO_ROLES!r}"
            )
        samples, source_sample_rate = load_wav_mono(audio_path)
        if source_sample_rate != self._config.sample_rate:
            samples = resample_audio(samples, source_sample_rate, self._config.sample_rate)

        frames = extract_reference_frames(
            samples=samples,
            sample_rate=self._config.sample_rate,
            block_size=self._config.block_size,
            feature_extractor=self._feature_extractor,
        )
        if not frames:
            raise ValueError("feature extraction produced no frames for the reference audio")

        feature_matrix = np.stack(
            [np.asarray(frame.values, dtype=np.float32) for frame in frames],
            axis=0,
        )
        timestamps_seconds = tuple(float(frame.observed_at) for frame in frames)
        frame_duration_seconds = float(frames[0].frame_duration_seconds)
        total_duration_seconds = len(samples) / self._config.sample_rate
        slides = load_slide_intervals(
            slides_path=slides_path,
            total_duration_seconds=total_duration_seconds,
            timestamps_seconds=timestamps_seconds,
        )

        metadata = {
            "profile_version": PROFILE_VERSION,
            "feature_extractor": self._feature_extractor.__class__.__name__,
            "sample_rate": str(self._config.sample_rate),
            "block_size": str(self._config.block_size),
            "source_sample_rate": str(source_sample_rate),
            "reference_audio_role": audio_role,
            "audio_path": str(Path(audio_path).resolve()),
            "slides_path": str(Path(slides_path).resolve()),
        }
        if companion_audio_path is not None:
            metadata["companion_audio_path"] = str(
                Path(companion_audio_path).expanduser().resolve()
            )
        return BuiltReferenceProfile(
            name=profile_name,
            frame_duration_seconds=frame_duration_seconds,
            timestamps_seconds=timestamps_seconds,
            feature_matrix=feature_matrix,
            slides=slides,
            metadata=metadata,
        )

    def save(self, profile: BuiltReferenceProfile, output_path: str) -> None:
        target = Path(output_path).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)

        np.save(target / "reference_features.npy", profile.feature_matrix)
        (target / "metadata.json").write_text(
            json.dumps(profile.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        slides = [
            {
                "slide_number": cue.slide_number,
                "section": cue.section,
                "lyrics": cue.lyrics,
                "reference_timestamp": cue.reference_timestamp,
                "end_timestamp": cue.end_timestamp,
                "start_frame": cue.start_frame,
                "end_frame": cue.end_frame,
            }
            for cue in profile.slides
        ]
        (target / "profile.json").write_text(
            json.dumps(
                {
                    "profile_version": PROFILE_VERSION,
                    "name": profile.name,
                    "frame_duration_seconds": profile.frame_duration_seconds,
                    "timestamps_seconds": list(profile.timestamps_seconds),
                    "slides": slides,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def load_wav_mono(audio_path: str) -> tuple[np.ndarray, int]:
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        pcm = handle.readframes(frame_count)

    if sample_width not in (1, 2, 4):
        raise ValueError("Only 8-bit, 16-bit, or 32-bit PCM WAV files are supported")

    if sample_width == 1:
        data = np.frombuffer(pcm, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768.0
    else:
        data = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2_147_483_648.0

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1, dtype=np.float32)

    return np.asarray(data, dtype=np.float32), sample_rate


def resample_audio(
    samples: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return np.asarray(samples, dtype=np.float32, copy=True)
    if samples.size == 0:
        return np.array([], dtype=np.float32)

    duration_seconds = samples.size / source_sample_rate
    target_length = max(1, int(round(duration_seconds * target_sample_rate)))
    source_positions = np.linspace(0.0, duration_seconds, num=samples.size, endpoint=False)
    target_positions = np.linspace(0.0, duration_seconds, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, samples)
    return np.asarray(resampled, dtype=np.float32)


def extract_reference_frames(
    samples: np.ndarray,
    sample_rate: int,
    block_size: int,
    feature_extractor: FeatureExtractor,
) -> list[FeatureFrame]:
    frames: list[FeatureFrame] = []
    for sequence_number, start in enumerate(range(0, samples.size, block_size)):
        end = min(start + block_size, samples.size)
        chunk = AudioChunk(
            samples=np.array(samples[start:end], dtype=np.float32, copy=True),
            captured_at=start / sample_rate,
            sequence_number=sequence_number,
        )
        frames.extend(feature_extractor.extract(chunk))
    return frames


def load_slide_intervals(
    slides_path: str,
    total_duration_seconds: float,
    timestamps_seconds: tuple[float, ...],
) -> tuple[SlideInterval, ...]:
    path = Path(slides_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Slides file does not exist: {path}")

    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError("Slides file must contain a JSON list")
    if not content:
        raise ValueError("Slides file must contain at least one slide entry")

    validated: list[tuple[int, str, str, float]] = []
    last_timestamp = -1.0
    for entry in content:
        if not isinstance(entry, dict):
            raise ValueError("Each slide entry must be a JSON object")
        slide_number = entry.get("slide_number")
        section = entry.get("section")
        lyrics = entry.get("lyrics")
        click_timestamp = entry.get("click_timestamp")
        reference_timestamp = entry.get("reference_timestamp")
        if isinstance(slide_number, bool) or not isinstance(slide_number, int) or slide_number <= 0:
            raise ValueError("slide_number must be a positive integer")
        if not isinstance(section, str) or not section.strip():
            raise ValueError("section must be a non-empty string")
        if not isinstance(lyrics, str) or not lyrics.strip():
            raise ValueError("lyrics must be a non-empty string")
        if isinstance(click_timestamp, (int, float)) and click_timestamp >= 0:
            timestamp = float(click_timestamp)
        elif isinstance(reference_timestamp, (int, float)) and reference_timestamp >= 0:
            timestamp = float(reference_timestamp)
        else:
            raise ValueError(
                "slide timing must include a non-negative click_timestamp "
                "or reference_timestamp"
            )
        if timestamp > total_duration_seconds:
            raise ValueError("slide timing must not exceed the audio duration")
        if timestamp < last_timestamp:
            raise ValueError("slide timing values must be sorted ascending")
        validated.append((slide_number, section.strip(), lyrics, timestamp))
        last_timestamp = timestamp

    intervals: list[SlideInterval] = []
    for index, (slide_number, section, lyrics, start_timestamp) in enumerate(validated):
        end_timestamp = (
            validated[index + 1][3] if index + 1 < len(validated) else total_duration_seconds
        )
        start_frame = _frame_index_for_timestamp(start_timestamp, timestamps_seconds)
        end_frame = _frame_index_for_timestamp(end_timestamp, timestamps_seconds)
        intervals.append(
            SlideInterval(
                slide_number=slide_number,
                section=section,
                lyrics=lyrics,
                reference_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    return tuple(intervals)


def _frame_index_for_timestamp(timestamp: float, timestamps_seconds: tuple[float, ...]) -> int:
    if not timestamps_seconds:
        return 0
    for index, observed_at in enumerate(timestamps_seconds):
        if observed_at >= timestamp:
            return index
    return len(timestamps_seconds) - 1
