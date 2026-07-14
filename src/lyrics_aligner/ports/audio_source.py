from collections.abc import Iterator
from typing import Protocol

from lyrics_aligner.domain.models import AudioChunk


class AudioSource(Protocol):
    def chunks(self) -> Iterator[AudioChunk]: ...
