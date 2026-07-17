"""OSC-backed presentation gateway."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from pythonosc.udp_client import SimpleUDPClient

from lyrics_aligner.domain.models import SlideCommand


@dataclass(frozen=True, slots=True)
class OscPresentationGatewayConfig:
    host: str
    port: int
    path: str
    retry_count: int = 2
    retry_backoff_seconds: float = 0.05

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host must be a non-empty string")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        if not self.path.startswith("/"):
            raise ValueError("path must start with '/'")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")


class OscPresentationGateway:
    """Send slide commands as JSON payloads over OSC/UDP."""

    def __init__(
        self,
        config: OscPresentationGatewayConfig,
        client: SimpleUDPClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or SimpleUDPClient(config.host, config.port)

    def send(self, command: SlideCommand) -> None:
        payload = json.dumps(
            {
                "slide": command.slide_number,
                "section": command.section,
                "lyrics": command.lyrics,
                "reference_timestamp": command.reference_timestamp,
                "confidence": command.confidence,
            },
            sort_keys=True,
        )
        attempts = self._config.retry_count + 1
        for attempt in range(attempts):
            try:
                self._client.send_message(self._config.path, payload)
            except OSError:
                if attempt + 1 >= attempts:
                    raise
                time.sleep(self._config.retry_backoff_seconds)
                continue
            return
