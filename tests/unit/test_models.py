import numpy as np

from lyrics_aligner.domain.models import AudioChunk


def test_audio_chunk_preserves_sequence_metadata() -> None:
    chunk = AudioChunk(
        samples=np.zeros(4_096, dtype=np.float32),
        captured_at=1.25,
        sequence_number=7,
    )

    assert chunk.samples.dtype == np.float32
    assert chunk.samples.shape == (4_096,)
    assert chunk.sequence_number == 7
