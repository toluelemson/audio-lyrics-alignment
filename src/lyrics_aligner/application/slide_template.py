from __future__ import annotations

import re
from dataclasses import dataclass

SECTION_PATTERN = re.compile(
    r"^(?:\*\*)?\s*"
    r"(?P<section>(?:verse|chorus|bridge|intro|outro|tag|refrain|pre-chorus)"
    r"(?:\s+\d+)?)"
    r"\s*(?:\*\*)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SlideTemplateConfig:
    initial_timestamp: float = 0.0
    timestamp_step: float = 20.0

    def __post_init__(self) -> None:
        if self.initial_timestamp < 0:
            raise ValueError("initial_timestamp must be non-negative")
        if self.timestamp_step <= 0:
            raise ValueError("timestamp_step must be greater than zero")


@dataclass(frozen=True, slots=True)
class SlideTemplateCue:
    slide_number: int
    section: str
    lyrics: str
    reference_timestamp: float


def build_slide_template(
    lyrics_text: str,
    config: SlideTemplateConfig | None = None,
) -> list[SlideTemplateCue]:
    if not lyrics_text.strip():
        raise ValueError("lyrics text must not be empty")

    settings = config or SlideTemplateConfig()
    cues: list[SlideTemplateCue] = []
    current_section: str | None = None
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_section, current_lines
        if current_section is None:
            current_lines = []
            return
        lyrics = "\n".join(current_lines).strip()
        if not lyrics:
            raise ValueError(f"section {current_section!r} does not contain any lyric lines")
        cues.append(
            SlideTemplateCue(
                slide_number=len(cues) + 1,
                section=current_section,
                lyrics=lyrics,
                reference_timestamp=(
                    settings.initial_timestamp + (len(cues) * settings.timestamp_step)
                ),
            )
        )
        current_section = None
        current_lines = []

    for raw_line in lyrics_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section = _parse_section_heading(line)
        if section is not None:
            flush_section()
            current_section = section
            continue
        if _is_metadata_line(line):
            continue

        if current_section is None:
            continue
        current_lines.append(line)

    flush_section()
    if not cues:
        raise ValueError("no slide sections were found in the lyrics text")
    return cues


def _parse_section_heading(line: str) -> str | None:
    match = SECTION_PATTERN.match(line)
    if match is None:
        return None
    return _normalize_section_name(match.group("section"))


def _normalize_section_name(section: str) -> str:
    normalized = " ".join(part.capitalize() for part in section.split())
    return normalized.replace("Pre-chorus", "Pre-Chorus")


def _is_metadata_line(line: str) -> bool:
    return line.startswith("*") and line.endswith("*")
