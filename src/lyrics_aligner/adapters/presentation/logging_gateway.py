"""Logging presentation gateway for local development."""

from __future__ import annotations

import logging

from lyrics_aligner.domain.models import SlideCommand


class LoggingPresentationGateway:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def send(self, command: SlideCommand) -> None:
        self._logger.info(
            "Triggered slide slide_number=%s section=%s "
            "reference_timestamp=%.2f confidence=%.2f lyrics=%r",
            command.slide_number,
            command.section,
            command.reference_timestamp,
            command.confidence,
            command.lyrics,
        )
