"""Feature extraction adapters."""

from lyrics_aligner.adapters.features.onnx_runtime import (
    OnnxFeatureExtractor,
    OnnxFeatureExtractorConfig,
)
from lyrics_aligner.adapters.features.simulated import (
    SimulatedFeatureExtractor,
    SimulatedFeatureExtractorConfig,
)

__all__ = [
    "OnnxFeatureExtractor",
    "OnnxFeatureExtractorConfig",
    "SimulatedFeatureExtractor",
    "SimulatedFeatureExtractorConfig",
]
