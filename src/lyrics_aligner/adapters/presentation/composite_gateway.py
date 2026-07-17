"""Presentation gateway composition helpers."""

from __future__ import annotations

from lyrics_aligner.domain.models import SlideCommand
from lyrics_aligner.ports.presentation_gateway import PresentationGateway


class CompositePresentationGateway:
    """Send the same command to multiple presentation gateways."""

    def __init__(self, *gateways: PresentationGateway) -> None:
        self._gateways = tuple(gateways)

    def send(self, command: SlideCommand) -> None:
        for gateway in self._gateways:
            gateway.send(command)
