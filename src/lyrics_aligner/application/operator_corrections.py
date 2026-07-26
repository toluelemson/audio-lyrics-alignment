from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lyrics_aligner.domain.models import CorrectionAnchor, OperatorCorrectionRecord


@dataclass(frozen=True, slots=True)
class CorrectionAnchorStoreConfig:
    min_corrections_for_anchor: int = 2
    source_group_tolerance_seconds: float = 2.0
    target_group_tolerance_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.min_corrections_for_anchor <= 0:
            raise ValueError("min_corrections_for_anchor must be greater than zero")
        if self.source_group_tolerance_seconds < 0.0:
            raise ValueError("source_group_tolerance_seconds must be non-negative")
        if self.target_group_tolerance_seconds < 0.0:
            raise ValueError("target_group_tolerance_seconds must be non-negative")


class ProfileCorrectionStore:
    """Persist operator corrections and derive stable anchors from repeated jumps."""

    def __init__(
        self,
        profile_path: str,
        config: CorrectionAnchorStoreConfig | None = None,
    ) -> None:
        self._profile_path = Path(profile_path).expanduser().resolve()
        self._config = config or CorrectionAnchorStoreConfig()
        self._corrections_path = self._profile_path / "operator_corrections.jsonl"
        self._anchors_path = self._profile_path / "operator_correction_anchors.json"

    def record(self, record: OperatorCorrectionRecord) -> None:
        self._profile_path.mkdir(parents=True, exist_ok=True)
        with self._corrections_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True))
            handle.write("\n")
        self._write_anchor_file(self.load_anchors())

    def load_anchors(self) -> tuple[CorrectionAnchor, ...]:
        groups: list[_AnchorGroup] = []
        for record in self._read_records():
            if record.detected_reference_timestamp is None:
                continue
            match = next(
                (
                    group
                    for group in groups
                    if group.matches(record, self._config)
                ),
                None,
            )
            if match is None:
                groups.append(_AnchorGroup.from_record(record))
                continue
            match.add(record)

        anchors = tuple(
            group.to_anchor()
            for group in groups
            if group.count >= self._config.min_corrections_for_anchor
        )
        return tuple(
            sorted(
                anchors,
                key=lambda anchor: (anchor.source_reference_timestamp, anchor.target_reference_timestamp),
            )
        )

    @staticmethod
    def build_record(
        *,
        profile_name: str,
        detected_reference_timestamp: float | None,
        chosen_reference_timestamp: float,
        chosen_slide_number: int,
        chosen_section: str,
        chosen_lyrics: str,
    ) -> OperatorCorrectionRecord:
        return OperatorCorrectionRecord(
            profile_name=profile_name,
            detected_reference_timestamp=detected_reference_timestamp,
            chosen_reference_timestamp=chosen_reference_timestamp,
            chosen_slide_number=chosen_slide_number,
            chosen_section=chosen_section,
            chosen_lyrics=chosen_lyrics,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _read_records(self) -> tuple[OperatorCorrectionRecord, ...]:
        if not self._corrections_path.exists():
            return ()

        records: list[OperatorCorrectionRecord] = []
        for raw_line in self._corrections_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                OperatorCorrectionRecord(
                    profile_name=str(payload["profile_name"]),
                    detected_reference_timestamp=(
                        None
                        if payload.get("detected_reference_timestamp") is None
                        else float(payload["detected_reference_timestamp"])
                    ),
                    chosen_reference_timestamp=float(payload["chosen_reference_timestamp"]),
                    chosen_slide_number=int(payload["chosen_slide_number"]),
                    chosen_section=str(payload["chosen_section"]),
                    chosen_lyrics=str(payload["chosen_lyrics"]),
                    created_at=str(payload["created_at"]),
                )
            )
        return tuple(records)

    def _write_anchor_file(self, anchors: tuple[CorrectionAnchor, ...]) -> None:
        payload = {
            "anchors": [asdict(anchor) for anchor in anchors],
            "min_corrections_for_anchor": self._config.min_corrections_for_anchor,
        }
        self._anchors_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


@dataclass(slots=True)
class _AnchorGroup:
    profile_name: str
    source_reference_timestamp: float
    target_reference_timestamp: float
    slide_number: int
    section: str
    lyrics: str
    count: int = 1

    @classmethod
    def from_record(cls, record: OperatorCorrectionRecord) -> _AnchorGroup:
        if record.detected_reference_timestamp is None:
            raise ValueError("record must include detected_reference_timestamp")
        return cls(
            profile_name=record.profile_name,
            source_reference_timestamp=record.detected_reference_timestamp,
            target_reference_timestamp=record.chosen_reference_timestamp,
            slide_number=record.chosen_slide_number,
            section=record.chosen_section,
            lyrics=record.chosen_lyrics,
        )

    def matches(
        self,
        record: OperatorCorrectionRecord,
        config: CorrectionAnchorStoreConfig,
    ) -> bool:
        if record.detected_reference_timestamp is None:
            return False
        if record.profile_name != self.profile_name:
            return False
        if record.chosen_slide_number != self.slide_number:
            return False
        return (
            abs(record.detected_reference_timestamp - self.source_reference_timestamp)
            <= config.source_group_tolerance_seconds
            and abs(record.chosen_reference_timestamp - self.target_reference_timestamp)
            <= config.target_group_tolerance_seconds
        )

    def add(self, record: OperatorCorrectionRecord) -> None:
        if record.detected_reference_timestamp is None:
            return
        next_count = self.count + 1
        self.source_reference_timestamp = (
            (self.source_reference_timestamp * self.count) + record.detected_reference_timestamp
        ) / next_count
        self.target_reference_timestamp = (
            (self.target_reference_timestamp * self.count) + record.chosen_reference_timestamp
        ) / next_count
        self.count = next_count
        self.section = record.chosen_section
        self.lyrics = record.chosen_lyrics

    def to_anchor(self) -> CorrectionAnchor:
        return CorrectionAnchor(
            profile_name=self.profile_name,
            source_reference_timestamp=self.source_reference_timestamp,
            target_reference_timestamp=self.target_reference_timestamp,
            slide_number=self.slide_number,
            section=self.section,
            lyrics=self.lyrics,
            correction_count=self.count,
        )
