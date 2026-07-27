from typing import Protocol

from lyrics_aligner.domain.models import MatchResult, SlideCommand


class SlideResolver(Protocol):
    def resolve(self, match: MatchResult) -> SlideCommand | None: ...
