import numpy as np
import pytest

from lyrics_aligner.adapters.features.simulated import (
    SimulatedFeatureExtractor,
    SimulatedFeatureExtractorConfig,
)
from lyrics_aligner.domain.models import AudioChunk
from lyrics_aligner.ports.feature_extractor import FeatureExtractor


def test_extract_returns_deterministic_feature_frame() -> None:
    extractor = SimulatedFeatureExtractor(SimulatedFeatureExtractorConfig(sample_rate=1_000))
    chunk = AudioChunk(
        samples=np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32),
        captured_at=2.5,
        sequence_number=3,
    )

    frames = extractor.extract(chunk)

    assert len(frames) == 1
    np.testing.assert_allclose(
        frames[0].values,
        np.array([0.70710677, 1.0, 0.5, 0.0], dtype=np.float32),
        rtol=1e-6,
    )
    assert frames[0].observed_at == 2.5
    assert frames[0].frame_duration_seconds == pytest.approx(0.004)


def test_empty_chunk_produces_no_frames() -> None:
    extractor = SimulatedFeatureExtractor()
    chunk = AudioChunk(
        samples=np.array([], dtype=np.float32),
        captured_at=1.0,
        sequence_number=0,
    )

    assert extractor.extract(chunk) == []


def test_conforms_to_feature_extractor_port() -> None:
    extractor: FeatureExtractor = SimulatedFeatureExtractor()

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
        ("sample_rate", 0, ValueError),
        ("sample_rate", 16_000.0, TypeError),
        ("feature_size", 3, ValueError),
        ("feature_size", True, TypeError),
    ],
)
def test_rejects_invalid_configuration(field: str, value: object, error: type[Exception]) -> None:
    values: dict[str, object] = {
        "sample_rate": 16_000,
        "feature_size": 4,
    }
    values[field] = value

    with pytest.raises(error):
        SimulatedFeatureExtractorConfig(**values)  # type: ignore[arg-type]
