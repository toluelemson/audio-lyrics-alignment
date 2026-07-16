from typing import Protocol

from lyrics_aligner.domain.models import ReferenceProfile


class ReferenceProfileRepository(Protocol):
    def load(self, path: str) -> ReferenceProfile: ...
