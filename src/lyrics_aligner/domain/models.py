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
