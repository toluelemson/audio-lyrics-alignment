from typing import Protocol

from lyrics_aligner.domain.models import FeatureFrame, MatchResult


class FeatureMatcher(Protocol):
    def match(self, frame: FeatureFrame) -> MatchResult: ...
