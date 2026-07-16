"""ONNX Runtime-backed feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from lyrics_aligner.domain.models import AudioChunk, FeatureFrame


class InferenceSession(Protocol):
    def get_inputs(self) -> list[Any]: ...

    def get_outputs(self) -> list[Any]: ...

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class OnnxFeatureExtractorConfig:
    model_path: str
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")


class OnnxFeatureExtractor:
    """Run feature extraction with an ONNX Runtime inference session."""

    def __init__(
        self,
        config: OnnxFeatureExtractorConfig,
        session: InferenceSession | None = None,
    ) -> None:
        self._config = config
        self._session = session or self._create_session(config.model_path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]

    def extract(self, chunk: AudioChunk) -> list[FeatureFrame]:
        samples = np.asarray(chunk.samples, dtype=np.float32)
        if samples.size == 0:
            return []

        max_abs = float(np.max(np.abs(samples)))
        normalized = samples if max_abs == 0.0 else samples / max_abs
        model_input = normalized[np.newaxis, :]
        outputs = self._session.run(self._output_names, {self._input_name: model_input})
        if not outputs:
            return []

        features = np.asarray(outputs[0], dtype=np.float32)
        if features.ndim == 1:
            feature_rows = features[np.newaxis, :]
        elif features.ndim == 2:
            feature_rows = features
        else:
            feature_rows = features.reshape(features.shape[0], -1)

        frame_duration = samples.size / self._config.sample_rate
        return [
            FeatureFrame(
                values=np.array(row, dtype=np.float32, copy=True),
                observed_at=chunk.captured_at + index * frame_duration,
                frame_duration_seconds=frame_duration,
            )
            for index, row in enumerate(feature_rows)
        ]

    @staticmethod
    def _create_session(model_path: str) -> InferenceSession:
        import onnxruntime as ort  # type: ignore[import-untyped]

        resolved_path = str(Path(model_path).expanduser().resolve())
        return cast(InferenceSession, ort.InferenceSession(resolved_path))
