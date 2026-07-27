"""Audio source adapters."""

from lyrics_aligner.adapters.audio.microphone import (
    MicrophoneAudioConfig,
    MicrophoneAudioSource,
)
from lyrics_aligner.adapters.audio.simulated import SimulatedAudioConfig, SimulatedAudioSource
from lyrics_aligner.adapters.audio.wav_file import WavFileAudioConfig, WavFileAudioSource

__all__ = [
    "MicrophoneAudioConfig",
    "MicrophoneAudioSource",
    "SimulatedAudioConfig",
    "SimulatedAudioSource",
    "WavFileAudioConfig",
    "WavFileAudioSource",
]
