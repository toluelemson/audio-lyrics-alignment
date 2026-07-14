"""Audio source adapters."""

from lyrics_aligner.adapters.audio.microphone import (
    MicrophoneAudioConfig,
    MicrophoneAudioSource,
)
from lyrics_aligner.adapters.audio.simulated import SimulatedAudioConfig, SimulatedAudioSource

__all__ = [
    "MicrophoneAudioConfig",
    "MicrophoneAudioSource",
    "SimulatedAudioConfig",
    "SimulatedAudioSource",
]
