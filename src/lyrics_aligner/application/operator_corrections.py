from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lyrics_aligner.domain.models import CorrectionAnchor, OperatorCorrectionRecord


@dataclass(frozen=True, slots=True)
class CorrectionAnchorStoreConfig:
    min_corrections_for_anchor: int = 2
    min_support_score_for_anchor: float = 2.0
    source_group_tolerance_seconds: float = 2.0
    target_group_tolerance_seconds: float = 2.0
    recency_half_life_days: float = 45.0
    low_confidence_weight_gain: float = 0.75
    no_vocal_weight_bonus: float = 0.25

    def __post_init__(self) -> None:
        if self.min_corrections_for_anchor <= 0:
            raise ValueError("min_corrections_for_anchor must be greater than zero")
        if self.min_support_score_for_anchor <= 0.0:
            raise ValueError("min_support_score_for_anchor must be greater than zero")
        if self.source_group_tolerance_seconds < 0.0:
            raise ValueError("source_group_tolerance_seconds must be non-negative")
        if self.target_group_tolerance_seconds < 0.0:
            raise ValueError("target_group_tolerance_seconds must be non-negative")
        if self.recency_half_life_days <= 0.0:
            raise ValueError("recency_half_life_days must be greater than zero")
        if self.low_confidence_weight_gain < 0.0:
            raise ValueError("low_confidence_weight_gain must be non-negative")
        if self.no_vocal_weight_bonus < 0.0:
            raise ValueError("no_vocal_weight_bonus must be non-negative")


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
        records = self._read_records()
        now = datetime.now(UTC)
        for record in records:
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
                groups.append(_AnchorGroup.from_record(record, self._record_weight(record, now)))
                continue
            match.add(record, self._record_weight(record, now))

        anchors = tuple(
            group.to_anchor()
            for group in groups
            if (
                group.count >= self._config.min_corrections_for_anchor
                and group.support_score >= self._config.min_support_score_for_anchor
            )
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
        detected_confidence: float | None = None,
        chosen_reference_timestamp: float,
        chosen_slide_number: int,
        chosen_section: str,
        chosen_lyrics: str,
        no_vocal_detected: bool = False,
        session_id: str = "",
    ) -> OperatorCorrectionRecord:
        return OperatorCorrectionRecord(
            profile_name=profile_name,
            detected_reference_timestamp=detected_reference_timestamp,
            detected_confidence=detected_confidence,
            chosen_reference_timestamp=chosen_reference_timestamp,
            chosen_slide_number=chosen_slide_number,
            chosen_section=chosen_section,
            chosen_lyrics=chosen_lyrics,
            created_at=datetime.now(UTC).isoformat(),
            no_vocal_detected=no_vocal_detected,
            session_id=session_id,
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
                    detected_confidence=(
                        None
                        if payload.get("detected_confidence") is None
                        else float(payload["detected_confidence"])
                    ),
                    chosen_reference_timestamp=float(payload["chosen_reference_timestamp"]),
                    chosen_slide_number=int(payload["chosen_slide_number"]),
                    chosen_section=str(payload["chosen_section"]),
                    chosen_lyrics=str(payload["chosen_lyrics"]),
                    created_at=str(payload["created_at"]),
                    no_vocal_detected=bool(payload.get("no_vocal_detected", False)),
                    session_id=str(payload.get("session_id", "")),
                )
            )
        return tuple(records)

    def _write_anchor_file(self, anchors: tuple[CorrectionAnchor, ...]) -> None:
        payload = {
            "anchors": [asdict(anchor) for anchor in anchors],
            "min_corrections_for_anchor": self._config.min_corrections_for_anchor,
            "min_support_score_for_anchor": self._config.min_support_score_for_anchor,
        }
        self._anchors_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _record_weight(
        self,
        record: OperatorCorrectionRecord,
        now: datetime,
    ) -> float:
        weight = 1.0
        if record.detected_confidence is not None:
            confidence = min(max(record.detected_confidence, 0.0), 1.0)
            weight += (1.0 - confidence) * self._config.low_confidence_weight_gain
        if record.no_vocal_detected:
            weight += self._config.no_vocal_weight_bonus
        try:
            created_at = datetime.fromisoformat(record.created_at)
        except ValueError:
            return weight
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        age_days = max(0.0, (now - created_at.astimezone(UTC)).total_seconds() / 86_400.0)
        return weight * (0.5 ** (age_days / self._config.recency_half_life_days))


@dataclass(slots=True)
class _AnchorGroup:
    profile_name: str
    source_reference_timestamp: float
    target_reference_timestamp: float
    slide_number: int
    section: str
    lyrics: str
    support_score: float
    last_seen_at: str
    count: int = 1
    detected_confidence_total: float = 0.0
    detected_confidence_count: int = 0
    session_ids: set[str] | None = None

    @classmethod
    def from_record(
        cls,
        record: OperatorCorrectionRecord,
        weight: float,
    ) -> _AnchorGroup:
        if record.detected_reference_timestamp is None:
            raise ValueError("record must include detected_reference_timestamp")
        session_ids = {record.session_id} if record.session_id else {_legacy_session_id(record)}
        return cls(
            profile_name=record.profile_name,
            source_reference_timestamp=record.detected_reference_timestamp,
            target_reference_timestamp=record.chosen_reference_timestamp,
            slide_number=record.chosen_slide_number,
            section=record.chosen_section,
            lyrics=record.chosen_lyrics,
            support_score=weight,
            last_seen_at=record.created_at,
            detected_confidence_total=(
                0.0 if record.detected_confidence is None else record.detected_confidence
            ),
            detected_confidence_count=0 if record.detected_confidence is None else 1,
            session_ids=session_ids,
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

    def add(self, record: OperatorCorrectionRecord, weight: float) -> None:
        if record.detected_reference_timestamp is None:
            return
        next_count = self.count + 1
        current_weight = self.support_score
        combined_weight = current_weight + weight
        self.source_reference_timestamp = (
            (self.source_reference_timestamp * current_weight)
            + (record.detected_reference_timestamp * weight)
        ) / combined_weight
        self.target_reference_timestamp = (
            (self.target_reference_timestamp * current_weight)
            + (record.chosen_reference_timestamp * weight)
        ) / combined_weight
        self.count = next_count
        self.support_score = combined_weight
        self.section = record.chosen_section
        self.lyrics = record.chosen_lyrics
        self.last_seen_at = max(self.last_seen_at, record.created_at)
        if record.detected_confidence is not None:
            self.detected_confidence_total += record.detected_confidence
            self.detected_confidence_count += 1
        if self.session_ids is None:
            self.session_ids = set()
        self.session_ids.add(record.session_id or _legacy_session_id(record))

    def to_anchor(self) -> CorrectionAnchor:
        return CorrectionAnchor(
            profile_name=self.profile_name,
            source_reference_timestamp=self.source_reference_timestamp,
            target_reference_timestamp=self.target_reference_timestamp,
            slide_number=self.slide_number,
            section=self.section,
            lyrics=self.lyrics,
            correction_count=self.count,
            session_count=0 if self.session_ids is None else len(self.session_ids),
            support_score=self.support_score,
            last_seen_at=self.last_seen_at,
        )


def _legacy_session_id(record: OperatorCorrectionRecord) -> str:
    return (
        f"legacy:{record.created_at}:{record.chosen_slide_number}:"
        f"{record.chosen_reference_timestamp:.3f}"
    )
