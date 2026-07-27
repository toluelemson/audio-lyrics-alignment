import numpy as np

from lyrics_aligner.domain.models import AudioChunk, FeatureFrame, ReferenceProfile


def test_audio_chunk_preserves_sequence_metadata() -> None:
    chunk = AudioChunk(
        samples=np.zeros(4_096, dtype=np.float32),
        captured_at=1.25,
        sequence_number=7,
    )

    assert chunk.samples.dtype == np.float32
    assert chunk.samples.shape == (4_096,)
    assert chunk.sequence_number == 7


def test_reference_profile_preserves_frames_and_metadata() -> None:
    frame = FeatureFrame(
        values=np.array([0.1, 0.2], dtype=np.float32),
        observed_at=0.5,
        frame_duration_seconds=0.25,
    )

    profile = ReferenceProfile(
        name="verse-1",
        frames=(frame,),
        metadata={"song": "Example"},
    )

    assert profile.name == "verse-1"
    assert len(profile.frames) == 1
    assert profile.frames[0].values.dtype == np.float32
    assert profile.metadata["song"] == "Example"
