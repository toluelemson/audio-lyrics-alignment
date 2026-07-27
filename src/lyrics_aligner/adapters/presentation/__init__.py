"""Presentation gateway adapters."""

from lyrics_aligner.adapters.presentation.logging_gateway import LoggingPresentationGateway
from lyrics_aligner.adapters.presentation.osc_gateway import (
    OscPresentationGateway,
    OscPresentationGatewayConfig,
)

__all__ = [
    "LoggingPresentationGateway",
    "OscPresentationGateway",
    "OscPresentationGatewayConfig",
]
