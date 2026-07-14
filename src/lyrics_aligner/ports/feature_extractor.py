from typing import Protocol

from lyrics_aligner.domain.models import AudioChunk, FeatureFrame


class FeatureExtractor(Protocol):
    def extract(self, chunk: AudioChunk) -> list[FeatureFrame]: ...
