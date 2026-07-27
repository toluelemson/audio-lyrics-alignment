from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, time

import numpy as np

from lyrics_aligner.adapters.features.pitch import estimate_normalized_pitch
from lyrics_aligner.application.reference_builder import load_wav_mono, resample_audio

ONSET_ANALYSIS_SAMPLE_RATE = 16_000
ONSET_FRAME_SIZE = 1_024
ONSET_HOP_SIZE = 256
DEFAULT_ONSET_SNAP_TOLERANCE_SECONDS = 0.35
VOCAL_ANALYSIS_SAMPLE_RATE = 16_000
VOCAL_FRAME_SIZE = 2_048
VOCAL_HOP_SIZE = 256


@dataclass(frozen=True, slots=True)
class SlideScriptCue:
    slide_number: int
    section: str
    lyrics: str
    line_number: int
    line_count: int


@dataclass(frozen=True, slots=True)
class CapturedSlideCue:
    slide_number: int
    section: str
    lyrics: str
    line_number: int
    line_count: int
    click_timestamp: float


@dataclass(slots=True)
class ClickCaptureSession:
    script_cues: tuple[SlideScriptCue, ...]
    onset_candidates: tuple[float, ...] = ()
    onset_snap_tolerance_seconds: float = DEFAULT_ONSET_SNAP_TOLERANCE_SECONDS
    started_at: float | None = None
    started_at_wall_seconds: float | None = None
    cue_timestamps: list[float | None] = field(default_factory=list)
    locked_indices: set[int] = field(default_factory=set)
    anchor_history: list[int] = field(default_factory=list)
    finished: bool = False

    def start(self) -> None:
        self.started_at = monotonic()
        self.started_at_wall_seconds = time()
        self.cue_timestamps = [None] * len(self.script_cues)
        self.locked_indices.clear()
        self.anchor_history.clear()
        self.finished = False

    def start_from(self, cue_index: int, *, elapsed: float) -> CapturedSlideCue:
        if cue_index < 0 or cue_index >= len(self.script_cues):
            raise ValueError("cue index is out of range")
        preserved_timestamps = list(self._ensure_timestamp_buffer())
        preserved_locked = sorted(
            index for index in self.locked_indices if index < cue_index and preserved_timestamps[index] is not None
        )

        self.started_at = monotonic()
        self.started_at_wall_seconds = time()
        self.cue_timestamps = [None] * len(self.script_cues)
        self.locked_indices.clear()
        self.anchor_history.clear()
        self.finished = False

        for index in preserved_locked:
            timestamp = preserved_timestamps[index]
            if timestamp is None:
                continue
            self.cue_timestamps[index] = timestamp
            self.locked_indices.add(index)
            self.anchor_history.append(index)

        captured = self.set_anchor(cue_index, elapsed=elapsed)
        self.anchor_history = [*preserved_locked, cue_index]
        return captured

    def reset(self) -> None:
        self.started_at = None
        self.started_at_wall_seconds = None
        self.cue_timestamps = []
        self.locked_indices.clear()
        self.anchor_history.clear()
        self.finished = False

    def mark_current(
        self,
        *,
        now: float | None = None,
        elapsed: float | None = None,
    ) -> CapturedSlideCue:
        if self.started_at is None:
            raise ValueError("capture session has not started")
        if self.finished:
            raise ValueError("capture session is already finished")
        index = self.current_index()
        if index >= len(self.script_cues):
            raise ValueError("all slide clicks have already been captured")
        click_timestamp = self._capture_timestamp(now=now, elapsed=elapsed)
        return self._set_timestamp(index, click_timestamp, locked=True)

    def set_anchor(
        self,
        cue_index: int,
        *,
        now: float | None = None,
        elapsed: float | None = None,
    ) -> CapturedSlideCue:
        if self.started_at is None:
            raise ValueError("capture session has not started")
        click_timestamp = self._capture_timestamp(now=now, elapsed=elapsed)
        return self._set_timestamp(cue_index, click_timestamp, locked=True)

    def draft_gaps(self) -> tuple[CapturedSlideCue, ...]:
        if self.started_at is None:
            raise ValueError("capture session has not started")
        if len(self.locked_indices) < 2:
            raise ValueError("at least two user anchors are required before drafting gaps")
        self._rebuild_drafts()
        return self.captured_cues()

    def nudge_cue(self, cue_index: int, delta_seconds: float) -> CapturedSlideCue:
        if self.started_at is None:
            raise ValueError("capture session has not started")
        if cue_index < 0 or cue_index >= len(self.script_cues):
            raise ValueError("cue index is out of range")
        current_timestamp = self.cue_timestamps[cue_index]
        if current_timestamp is None:
            raise ValueError("cue does not have a timestamp to nudge")
        new_timestamp = round(current_timestamp + delta_seconds, 3)
        return self._set_timestamp(cue_index, new_timestamp, locked=True)

    def current_index(self) -> int:
        if len(self.cue_timestamps) != len(self.script_cues):
            return 0
        for index, timestamp in enumerate(self.cue_timestamps):
            if timestamp is None:
                return index
        return len(self.script_cues)

    def is_complete(self) -> bool:
        return (
            bool(self.script_cues)
            and len(self.cue_timestamps) == len(self.script_cues)
            and all(
            timestamp is not None for timestamp in self.cue_timestamps
        )
        )

    def undo_last(self) -> CapturedSlideCue:
        if not self.anchor_history:
            raise ValueError("no captured slide clicks are available to undo")
        self.finished = False
        cue_index = self.anchor_history.pop()
        timestamp = self.cue_timestamps[cue_index]
        self.locked_indices.discard(cue_index)
        self.cue_timestamps[cue_index] = None
        self._rebuild_drafts()
        if timestamp is None:
            raise ValueError("last anchor did not have a timestamp")
        return self._captured_cue(cue_index, timestamp)

    def finish(self) -> tuple[CapturedSlideCue, ...]:
        if not any(timestamp is not None for timestamp in self.cue_timestamps):
            raise ValueError("at least one slide click must be captured before finishing")
        if not self.is_complete():
            raise ValueError("all slide clicks must be captured before finishing")
        self.finished = True
        return self.captured_cues()

    def stop(self) -> tuple[CapturedSlideCue, ...]:
        if not any(timestamp is not None for timestamp in self.cue_timestamps):
            raise ValueError("at least one slide click must be captured before stopping")
        self.finished = True
        return self.captured_cues()

    def captured_cues(self) -> tuple[CapturedSlideCue, ...]:
        captured: list[CapturedSlideCue] = []
        for index, timestamp in enumerate(self.cue_timestamps):
            if timestamp is None:
                continue
            captured.append(self._captured_cue(index, timestamp))
        return tuple(captured)

    def cue_states(self) -> tuple[dict[str, object], ...]:
        states: list[dict[str, object]] = []
        for index, cue in enumerate(self.script_cues):
            timestamp = self.cue_timestamps[index] if index < len(self.cue_timestamps) else None
            states.append(
                {
                    "slide_number": cue.slide_number,
                    "section": cue.section,
                    "lyrics": cue.lyrics,
                    "line_number": cue.line_number,
                    "line_count": cue.line_count,
                    "click_timestamp": timestamp,
                    "locked": index in self.locked_indices,
                    "drafted": timestamp is not None and index not in self.locked_indices,
                }
            )
        return tuple(states)

    def _capture_timestamp(
        self,
        *,
        now: float | None = None,
        elapsed: float | None = None,
    ) -> float:
        if elapsed is None:
            if self.started_at is None:
                raise ValueError("capture session has not started")
            current_time = monotonic() if now is None else now
            click_timestamp = round(current_time - self.started_at, 3)
        else:
            click_timestamp = round(elapsed, 3)
        if click_timestamp < 0.0:
            raise ValueError("captured click time must be non-negative")
        return click_timestamp

    def _set_timestamp(
        self,
        cue_index: int,
        click_timestamp: float,
        *,
        locked: bool,
    ) -> CapturedSlideCue:
        if cue_index < 0 or cue_index >= len(self.script_cues):
            raise ValueError("cue index is out of range")
        self._ensure_timestamp_order(cue_index, click_timestamp)
        self.cue_timestamps = self._ensure_timestamp_buffer()
        self.cue_timestamps[cue_index] = click_timestamp
        if locked:
            self.locked_indices.add(cue_index)
            self.anchor_history.append(cue_index)
        self.finished = False
        self._rebuild_drafts()
        locked_timestamp = self.cue_timestamps[cue_index]
        if locked_timestamp is None:
            raise ValueError("cue timestamp could not be stored")
        return self._captured_cue(cue_index, locked_timestamp)

    def _ensure_timestamp_buffer(self) -> list[float | None]:
        if len(self.cue_timestamps) != len(self.script_cues):
            return [None] * len(self.script_cues)
        return self.cue_timestamps

    def _ensure_timestamp_order(self, cue_index: int, click_timestamp: float) -> None:
        previous_timestamp = next(
            (
                timestamp
                for timestamp in reversed(self.cue_timestamps[:cue_index])
                if timestamp is not None
            ),
            None,
        )
        next_timestamp = next(
            (
                timestamp
                for timestamp in self.cue_timestamps[cue_index + 1 :]
                if timestamp is not None
            ),
            None,
        )
        if previous_timestamp is not None and click_timestamp < previous_timestamp:
            raise ValueError("captured click time must not move backward")
        if next_timestamp is not None and click_timestamp > next_timestamp:
            raise ValueError("captured click time must not pass the next locked line")

    def _rebuild_drafts(self) -> None:
        timestamps = self._ensure_timestamp_buffer()
        for index in range(len(timestamps)):
            if index not in self.locked_indices:
                timestamps[index] = None
        anchor_indices = sorted(self.locked_indices)
        for start_index, end_index in zip(anchor_indices, anchor_indices[1:], strict=False):
            if end_index - start_index <= 1:
                continue
            start_time = timestamps[start_index]
            end_time = timestamps[end_index]
            if start_time is None or end_time is None:
                continue
            interval_weights = [
                self._cue_weight(self.script_cues[index].lyrics)
                for index in range(start_index + 1, end_index + 1)
            ]
            total_weight = sum(interval_weights)
            if total_weight <= 0:
                continue
            elapsed = 0.0
            for offset, cue_index in enumerate(range(start_index + 1, end_index), start=0):
                elapsed += interval_weights[offset]
                ratio = elapsed / total_weight
                interpolated_timestamp = start_time + ((end_time - start_time) * ratio)
                lower_bound = start_time if cue_index == start_index + 1 else timestamps[cue_index - 1]
                upper_bound = end_time
                if cue_index + 1 < len(timestamps) and cue_index + 1 in self.locked_indices:
                    upper_bound = timestamps[cue_index + 1] or end_time
                timestamps[cue_index] = round(
                    self._snap_to_onset(
                        interpolated_timestamp,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    ),
                    3,
                )

    @staticmethod
    def _cue_weight(lyrics: str) -> float:
        words = [word for word in lyrics.split() if word.strip()]
        if words:
            return float(sum(max(1, len(word)) for word in words))
        return max(1.0, float(len(lyrics.strip())))

    def _captured_cue(self, cue_index: int, timestamp: float) -> CapturedSlideCue:
        cue = self.script_cues[cue_index]
        return CapturedSlideCue(
            slide_number=cue.slide_number,
            section=cue.section,
            lyrics=cue.lyrics,
            line_number=cue.line_number,
            line_count=cue.line_count,
            click_timestamp=timestamp,
        )

    def _snap_to_onset(
        self,
        timestamp: float,
        *,
        lower_bound: float | None,
        upper_bound: float | None,
    ) -> float:
        if not self.onset_candidates:
            return timestamp

        best_candidate = timestamp
        best_distance = self.onset_snap_tolerance_seconds + 1.0
        minimum_gap = 0.001

        for candidate in self.onset_candidates:
            if lower_bound is not None and candidate <= lower_bound + minimum_gap:
                continue
            if upper_bound is not None and candidate >= upper_bound - minimum_gap:
                continue
            distance = abs(candidate - timestamp)
            if distance > self.onset_snap_tolerance_seconds:
                continue
            if distance < best_distance:
                best_candidate = candidate
                best_distance = distance
        return best_candidate


def detect_onset_candidates(
    audio_path: str,
    *,
    target_sample_rate: int = ONSET_ANALYSIS_SAMPLE_RATE,
    frame_size: int = ONSET_FRAME_SIZE,
    hop_size: int = ONSET_HOP_SIZE,
) -> tuple[float, ...]:
    samples, source_sample_rate = load_wav_mono(audio_path)
    if source_sample_rate != target_sample_rate:
        samples = resample_audio(samples, source_sample_rate, target_sample_rate)
    if samples.size < frame_size or hop_size <= 0:
        return ()

    frames: list[np.ndarray] = []
    for start in range(0, samples.size - frame_size + 1, hop_size):
        frames.append(samples[start : start + frame_size])
    if not frames:
        return ()

    energy = np.asarray(
        [float(np.sqrt(np.mean(np.square(frame), dtype=np.float64))) for frame in frames],
        dtype=np.float32,
    )
    spectral_flux = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    if spectral_flux.size == 0:
        return ()

    threshold = max(
        float(np.mean(spectral_flux) + np.std(spectral_flux)),
        0.01,
    )
    candidates: list[float] = []
    last_timestamp = -1.0
    min_spacing_seconds = 0.12
    for index in range(1, spectral_flux.size - 1):
        value = float(spectral_flux[index])
        if value < threshold:
            continue
        if value < float(spectral_flux[index - 1]) or value < float(spectral_flux[index + 1]):
            continue
        timestamp = round((index * hop_size) / target_sample_rate, 3)
        if timestamp - last_timestamp < min_spacing_seconds:
            if candidates and value > spectral_flux[index - 1]:
                candidates[-1] = timestamp
                last_timestamp = timestamp
            continue
        candidates.append(timestamp)
        last_timestamp = timestamp
    return tuple(candidates)


def detect_vocal_candidates(
    audio_path: str,
    *,
    target_sample_rate: int = VOCAL_ANALYSIS_SAMPLE_RATE,
    frame_size: int = VOCAL_FRAME_SIZE,
    hop_size: int = VOCAL_HOP_SIZE,
    silence_rms_threshold: float = 0.02,
    min_voiced_run_seconds: float = 0.18,
) -> tuple[float, ...]:
    samples, source_sample_rate = load_wav_mono(audio_path)
    if source_sample_rate != target_sample_rate:
        samples = resample_audio(samples, source_sample_rate, target_sample_rate)
    if samples.size < frame_size or hop_size <= 0:
        return ()

    rms_values: list[float] = []
    voiced_flags: list[bool] = []
    timestamps: list[float] = []
    for start in range(0, samples.size - frame_size + 1, hop_size):
        frame = samples[start : start + frame_size]
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        pitch = estimate_normalized_pitch(
            frame,
            sample_rate=target_sample_rate,
            silence_rms_threshold=silence_rms_threshold,
        )
        rms_values.append(rms)
        voiced_flags.append(bool(pitch != 0.0))
        timestamps.append(round(start / target_sample_rate, 3))
    if not voiced_flags:
        return ()

    rms_array = np.asarray(rms_values, dtype=np.float32)
    energy_floor = max(float(np.mean(rms_array) + 0.5 * np.std(rms_array)), silence_rms_threshold)
    min_run_frames = max(1, int(round((min_voiced_run_seconds * target_sample_rate) / hop_size)))

    candidates: list[float] = []
    index = 0
    while index < len(voiced_flags):
        if not voiced_flags[index] or rms_values[index] < energy_floor:
            index += 1
            continue
        run_start = index
        run_end = index + 1
        while run_end < len(voiced_flags) and voiced_flags[run_end]:
            run_end += 1
        voiced_run = sum(
            1 for frame_index in range(run_start, run_end) if rms_values[frame_index] >= energy_floor
        )
        if voiced_run >= min_run_frames:
            candidates.append(timestamps[run_start])
        index = run_end
    return tuple(candidates)


def load_slide_script(path: str) -> tuple[SlideScriptCue, ...]:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Slide script file does not exist: {resolved_path}")

    content = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError("Slide script file must contain a JSON list")
    if not content:
        raise ValueError("Slide script file must contain at least one slide entry")

    cues: list[SlideScriptCue] = []
    for entry in content:
        if not isinstance(entry, dict):
            raise ValueError("Each slide script entry must be a JSON object")
        slide_number = entry.get("slide_number")
        section = entry.get("section")
        lyrics = entry.get("lyrics")
        if isinstance(slide_number, bool) or not isinstance(slide_number, int) or slide_number <= 0:
            raise ValueError("slide_number must be a positive integer")
        if not isinstance(section, str) or not section.strip():
            raise ValueError("section must be a non-empty string")
        if not isinstance(lyrics, str) or not lyrics.strip():
            raise ValueError("lyrics must be a non-empty string")
        lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
        if not lines:
            raise ValueError("lyrics must contain at least one non-empty line")
        for line_index, line in enumerate(lines, start=1):
            cues.append(
                SlideScriptCue(
                    slide_number=slide_number,
                    section=section.strip(),
                    lyrics=line,
                    line_number=line_index,
                    line_count=len(lines),
                )
            )
    return tuple(cues)


def build_captured_slide_cues(
    script_cues: tuple[SlideScriptCue, ...],
    click_timestamps: tuple[float, ...],
) -> tuple[CapturedSlideCue, ...]:
    if len(script_cues) != len(click_timestamps):
        raise ValueError("slide cue count must match click timestamp count")

    captured: list[CapturedSlideCue] = []
    last_timestamp = -1.0
    for cue, timestamp in zip(script_cues, click_timestamps, strict=True):
        if timestamp < 0.0:
            raise ValueError("click timestamps must be non-negative")
        if timestamp < last_timestamp:
            raise ValueError("click timestamps must be sorted ascending")
        captured.append(
            CapturedSlideCue(
                slide_number=cue.slide_number,
                section=cue.section,
                lyrics=cue.lyrics,
                line_number=cue.line_number,
                line_count=cue.line_count,
                click_timestamp=timestamp,
            )
        )
        last_timestamp = timestamp
    return tuple(captured)


def write_captured_slide_cues(
    output_path: str,
    captured_cues: tuple[CapturedSlideCue, ...],
) -> None:
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(
            [
                {
                    "slide_number": cue.slide_number,
                    "section": cue.section,
                    "lyrics": cue.lyrics,
                    "line_number": cue.line_number,
                    "line_count": cue.line_count,
                    "click_timestamp": cue.click_timestamp,
                }
                for cue in captured_cues
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
