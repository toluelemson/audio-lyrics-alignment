from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, time


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
    started_at: float | None = None
    started_at_wall_seconds: float | None = None
    click_timestamps: list[float] = field(default_factory=list)
    finished: bool = False

    def start(self) -> None:
        self.started_at = monotonic()
        self.started_at_wall_seconds = time()
        self.click_timestamps.clear()
        self.finished = False

    def reset(self) -> None:
        self.started_at = None
        self.started_at_wall_seconds = None
        self.click_timestamps.clear()
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
        if len(self.click_timestamps) >= len(self.script_cues):
            raise ValueError("all slide clicks have already been captured")

        if elapsed is None:
            current_time = monotonic() if now is None else now
            click_timestamp = round(current_time - self.started_at, 3)
        else:
            click_timestamp = round(elapsed, 3)

        if click_timestamp < 0.0:
            raise ValueError("captured click time must be non-negative")
        if self.click_timestamps and click_timestamp < self.click_timestamps[-1]:
            raise ValueError("captured click time must not move backward")

        self.click_timestamps.append(click_timestamp)
        cue = self.script_cues[len(self.click_timestamps) - 1]
        return CapturedSlideCue(
            slide_number=cue.slide_number,
            section=cue.section,
            lyrics=cue.lyrics,
            line_number=cue.line_number,
            line_count=cue.line_count,
            click_timestamp=click_timestamp,
        )

    def current_index(self) -> int:
        return len(self.click_timestamps)

    def is_complete(self) -> bool:
        return len(self.click_timestamps) == len(self.script_cues)

    def undo_last(self) -> CapturedSlideCue:
        if not self.click_timestamps:
            raise ValueError("no captured slide clicks are available to undo")
        self.finished = False
        timestamp = self.click_timestamps.pop()
        cue = self.script_cues[len(self.click_timestamps)]
        return CapturedSlideCue(
            slide_number=cue.slide_number,
            section=cue.section,
            lyrics=cue.lyrics,
            line_number=cue.line_number,
            line_count=cue.line_count,
            click_timestamp=timestamp,
        )

    def finish(self) -> tuple[CapturedSlideCue, ...]:
        if not self.click_timestamps:
            raise ValueError("at least one slide click must be captured before finishing")
        if not self.is_complete():
            raise ValueError("all slide clicks must be captured before finishing")
        self.finished = True
        return self.captured_cues()

    def captured_cues(self) -> tuple[CapturedSlideCue, ...]:
        captured: list[CapturedSlideCue] = []
        for cue, timestamp in zip(
            self.script_cues,
            self.click_timestamps,
            strict=False,
        ):
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
        return tuple(captured)


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
