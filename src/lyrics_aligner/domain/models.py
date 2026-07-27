from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AudioChunk:
    samples: FloatArray
    captured_at: float
    sequence_number: int


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    values: FloatArray
    observed_at: float
    frame_duration_seconds: float


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    name: str
    frames: tuple[FeatureFrame, ...]
    metadata: dict[str, str]
    slide_cues: tuple["SlideCue", ...] = ()


@dataclass(frozen=True, slots=True)
class MatchResult:
    reference_frame: int
    reference_timestamp: float
    raw_distance: float
    normalized_distance: float
    confidence: float
    valid: bool


@dataclass(frozen=True, slots=True)
class SlideCommand:
    slide_number: int
    section: str
    lyrics: str
    reference_timestamp: float
    confidence: float


@dataclass(frozen=True, slots=True)
class SlideCue:
    slide_number: int
    section: str
    lyrics: str
    reference_timestamp: float


@dataclass(frozen=True, slots=True)
class OperatorCorrectionRecord:
    profile_name: str
    detected_reference_timestamp: float | None
    detected_confidence: float | None
    chosen_reference_timestamp: float
    chosen_slide_number: int
    chosen_section: str
    chosen_lyrics: str
    created_at: str
    no_vocal_detected: bool = False
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class CorrectionAnchor:
    profile_name: str
    source_reference_timestamp: float
    target_reference_timestamp: float
    slide_number: int
    section: str
    lyrics: str
    correction_count: int
    session_count: int = 0
    support_score: float = 0.0
    last_seen_at: str = ""
