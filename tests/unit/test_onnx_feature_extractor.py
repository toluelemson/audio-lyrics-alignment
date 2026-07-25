import numpy as np
import pytest

from lyrics_aligner.adapters.features.onnx_runtime import (
    OnnxFeatureExtractor,
    OnnxFeatureExtractorConfig,
)
from lyrics_aligner.domain.models import AudioChunk
from lyrics_aligner.ports.feature_extractor import FeatureExtractor


class _StubValueInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubSession:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outputs = outputs
        self.seen_inputs: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[_StubValueInfo]:
        return [_StubValueInfo("waveform")]

    def get_outputs(self) -> list[_StubValueInfo]:
        return [_StubValueInfo("features")]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert output_names == ["features"]
        self.seen_inputs.append(input_feed)
        return self._outputs


def test_extract_normalizes_waveform_and_returns_feature_frames() -> None:
    session = _StubSession(
        outputs=[
            np.array(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            )
        ]
    )
    extractor = OnnxFeatureExtractor(
        OnnxFeatureExtractorConfig(model_path="dummy.onnx", sample_rate=1_000),
        session=session,
    )
    chunk = AudioChunk(
        samples=np.array([0.0, 2.0, -2.0, 1.0], dtype=np.float32),
        captured_at=5.0,
        sequence_number=4,
    )

    frames = extractor.extract(chunk)

    assert len(frames) == 2
    np.testing.assert_allclose(
        session.seen_inputs[0]["waveform"],
        np.array([[0.0, 1.0, -1.0, 0.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(frames[0].values, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(frames[1].values, np.array([3.0, 4.0], dtype=np.float32))
    assert frames[0].observed_at == 5.0
    assert frames[1].observed_at == pytest.approx(5.002)
    assert frames[0].frame_duration_seconds == pytest.approx(0.002)


def test_extract_splits_sequence_model_outputs_into_per_step_feature_frames() -> None:
    session = _StubSession(
        outputs=[
            np.array(
                [
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ],
                dtype=np.float32,
            )
        ]
    )
    extractor = OnnxFeatureExtractor(
        OnnxFeatureExtractorConfig(model_path="dummy.onnx", sample_rate=1_000),
        session=session,
    )

    frames = extractor.extract(
        AudioChunk(
            samples=np.ones(6, dtype=np.float32),
            captured_at=2.0,
            sequence_number=0,
        )
    )

    assert len(frames) == 3
    np.testing.assert_array_equal(frames[0].values, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(frames[1].values, np.array([4.0, 5.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(frames[2].values, np.array([7.0, 8.0, 9.0], dtype=np.float32))
    assert frames[0].observed_at == 2.0
    assert frames[1].observed_at == pytest.approx(2.002)
    assert frames[2].observed_at == pytest.approx(2.004)
    assert frames[0].frame_duration_seconds == pytest.approx(0.002)


def test_empty_chunk_produces_no_frames() -> None:
    extractor = OnnxFeatureExtractor(
        OnnxFeatureExtractorConfig(model_path="dummy.onnx"),
        session=_StubSession(outputs=[np.ones((1, 2), dtype=np.float32)]),
    )

    frames = extractor.extract(
        AudioChunk(
            samples=np.array([], dtype=np.float32),
            captured_at=0.0,
            sequence_number=0,
        )
    )

    assert frames == []


def test_conforms_to_feature_extractor_port() -> None:
    extractor: FeatureExtractor = OnnxFeatureExtractor(
        OnnxFeatureExtractorConfig(model_path="dummy.onnx"),
        session=_StubSession(outputs=[np.ones((1, 2), dtype=np.float32)]),
    )

    frames = extractor.extract(
        AudioChunk(
            samples=np.ones(8, dtype=np.float32),
            captured_at=1.0,
            sequence_number=0,
        )
    )

    assert len(frames) == 1


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("model_path", "", ValueError),
        ("sample_rate", 0, ValueError),
        ("sample_rate", 16_000.0, TypeError),
    ],
)
def test_rejects_invalid_configuration(field: str, value: object, error: type[Exception]) -> None:
    values: dict[str, object] = {
        "model_path": "model.onnx",
        "sample_rate": 16_000,
    }
    values[field] = value

    with pytest.raises(error):
        OnnxFeatureExtractorConfig(**values)  # type: ignore[arg-type]
