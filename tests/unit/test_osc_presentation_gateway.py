import json

import pytest

from lyrics_aligner.adapters.presentation import (
    OscPresentationGateway,
    OscPresentationGatewayConfig,
)
from lyrics_aligner.domain.models import SlideCommand


class RecordingOscClient:
    def __init__(self, failures_before_success: int = 0) -> None:
        self.failures_before_success = failures_before_success
        self.messages: list[tuple[str, str]] = []

    def send_message(self, path: str, payload: str) -> None:
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise OSError("network down")
        self.messages.append((path, payload))


def _command() -> SlideCommand:
    return SlideCommand(
        slide_number=2,
        section="Verse 1",
        lyrics="Amazing grace",
        reference_timestamp=5.42,
        confidence=0.91,
    )


def test_gateway_serializes_json_payload_and_sends_to_path() -> None:
    client = RecordingOscClient()
    gateway = OscPresentationGateway(
        OscPresentationGatewayConfig(
            host="127.0.0.1",
            port=7000,
            path="/presentation/trigger-slide",
        ),
        client=client,
    )

    gateway.send(_command())

    assert len(client.messages) == 1
    path, payload = client.messages[0]
    assert path == "/presentation/trigger-slide"
    assert json.loads(payload) == {
        "confidence": 0.91,
        "lyrics": "Amazing grace",
        "reference_timestamp": 5.42,
        "section": "Verse 1",
        "slide": 2,
    }


def test_gateway_retries_after_temporary_os_error() -> None:
    client = RecordingOscClient(failures_before_success=1)
    gateway = OscPresentationGateway(
        OscPresentationGatewayConfig(
            host="127.0.0.1",
            port=7000,
            path="/presentation/trigger-slide",
            retry_count=1,
            retry_backoff_seconds=0.0,
        ),
        client=client,
    )

    gateway.send(_command())

    assert len(client.messages) == 1


def test_gateway_raises_after_exhausting_retries() -> None:
    client = RecordingOscClient(failures_before_success=2)
    gateway = OscPresentationGateway(
        OscPresentationGatewayConfig(
            host="127.0.0.1",
            port=7000,
            path="/presentation/trigger-slide",
            retry_count=1,
            retry_backoff_seconds=0.0,
        ),
        client=client,
    )

    with pytest.raises(OSError):
        gateway.send(_command())
