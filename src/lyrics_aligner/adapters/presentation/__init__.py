"""Presentation gateway adapters."""

from lyrics_aligner.adapters.presentation.composite_gateway import CompositePresentationGateway
from lyrics_aligner.adapters.presentation.logging_gateway import LoggingPresentationGateway
from lyrics_aligner.adapters.presentation.osc_gateway import (
    OscPresentationGateway,
    OscPresentationGatewayConfig,
)

__all__ = [
    "CompositePresentationGateway",
    "LoggingPresentationGateway",
    "OscPresentationGateway",
    "OscPresentationGatewayConfig",
]
