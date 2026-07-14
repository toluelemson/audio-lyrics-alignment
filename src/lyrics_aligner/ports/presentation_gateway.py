from typing import Protocol

from lyrics_aligner.domain.models import SlideCommand


class PresentationGateway(Protocol):
    def send(self, command: SlideCommand) -> None: ...
