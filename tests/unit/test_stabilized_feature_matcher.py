import numpy as np

from lyrics_aligner.adapters.matching import (
    StabilizedFeatureMatcher,
    StabilizedFeatureMatcherConfig,
)
from lyrics_aligner.domain.models import FeatureFrame, MatchResult
from lyrics_aligner.ports.feature_matcher import FeatureMatcher


class SequenceMatcher:
    def __init__(self, results: list[MatchResult]) -> None:
        self._results = results
        self._index = 0

    def match(self, frame: FeatureFrame) -> MatchResult:
        del frame
        result = self._results[self._index]
        self._index += 1
        return result


def _frame() -> FeatureFrame:
    return FeatureFrame(
        values=np.array([0.1, 0.2], dtype=np.float32),
        observed_at=0.0,
        frame_duration_seconds=0.25,
    )


def _result(reference_frame: int, confidence: float = 0.9, *, valid: bool = True) -> MatchResult:
    return MatchResult(
        reference_frame=reference_frame,
        reference_timestamp=reference_frame * 0.25,
        raw_distance=0.1,
        normalized_distance=0.1,
        confidence=confidence,
        valid=valid,
    )


def test_stabilizer_blocks_backward_jumps() -> None:
    matcher: FeatureMatcher = SequenceMatcher([_result(2), _result(1)])
    stabilizer = StabilizedFeatureMatcher(matcher)

    first = stabilizer.match(_frame())
    second = stabilizer.match(_frame())

    assert first.valid is True
    assert second.reference_frame == 1
    assert second.valid is False


def test_stabilizer_requires_confirmation_for_large_forward_jump() -> None:
    matcher: FeatureMatcher = SequenceMatcher([_result(1), _result(4), _result(4)])
    stabilizer = StabilizedFeatureMatcher(
        matcher,
        StabilizedFeatureMatcherConfig(
            max_forward_jump_frames=4,
            large_jump_threshold_frames=1,
            confirmation_count=2,
        ),
    )

    first = stabilizer.match(_frame())
    second = stabilizer.match(_frame())
    third = stabilizer.match(_frame())

    assert first.valid is True
    assert second.reference_frame == 4
    assert second.valid is False
    assert third.reference_frame == 4
    assert third.valid is True


def test_stabilizer_rejects_forward_jump_beyond_limit() -> None:
    matcher: FeatureMatcher = SequenceMatcher([_result(1), _result(7)])
    stabilizer = StabilizedFeatureMatcher(
        matcher,
        StabilizedFeatureMatcherConfig(max_forward_jump_frames=3),
    )

    first = stabilizer.match(_frame())
    second = stabilizer.match(_frame())

    assert first.valid is True
    assert second.reference_frame == 7
    assert second.valid is False
