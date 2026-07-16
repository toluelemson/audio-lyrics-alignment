import logging

import numpy as np
import pytest

from lyrics_aligner.adapters.audio.simulated import SimulatedAudioConfig, SimulatedAudioSource
from lyrics_aligner.adapters.matching import (
    NearestNeighborFeatureMatcher,
    NearestNeighborFeatureMatcherConfig,
    StabilizedFeatureMatcher,
    StabilizedFeatureMatcherConfig,
)
from lyrics_aligner.application.runtime import AudioIngestionRuntime, BoundedAudioQueue
from lyrics_aligner.domain.models import (
    FeatureFrame,
    MatchResult,
    ReferenceProfile,
    SlideCommand,
)
from lyrics_aligner.ports.feature_extractor import FeatureExtractor
from lyrics_aligner.ports.feature_matcher import FeatureMatcher
from lyrics_aligner.ports.presentation_gateway import PresentationGateway
from lyrics_aligner.ports.slide_resolver import SlideResolver


def test_bounded_audio_queue_drops_oldest_when_full() -> None:
    source = SimulatedAudioSource(
        SimulatedAudioConfig(sample_rate=1_000, block_size=4, duration=0.012, frequency=100)
    )
    queue = BoundedAudioQueue(capacity=2)

    chunks = list(source.chunks())

    assert queue.put_drop_oldest(chunks[0]) is False
    assert queue.put_drop_oldest(chunks[1]) is False
    assert queue.put_drop_oldest(chunks[2]) is True

    assert queue.get().sequence_number == 1
    assert queue.get().sequence_number == 2
    assert queue.get(timeout=0.001) is None


def test_runtime_consumes_simulated_audio_and_returns_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.03,
                frequency=100,
                amplitude=0.25,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime"),
        diagnostics_interval_seconds=0.01,
    )

    report = runtime.run()

    assert report.metrics.chunks_received == 3
    assert report.metrics.chunks_dropped == 0
    assert report.metrics.silent_chunks == 0
    assert report.metrics.clipped_chunks == 0
    assert report.metrics.feature_frames_processed == 0
    assert report.metrics.queue_high_water_mark >= 1
    assert report.queue_size == 0
    assert report.last_match is None
    assert report.peak == pytest.approx(0.25, rel=0.05)
    assert report.rms > 0
    assert "device=SimulatedAudioSource" in caplog.text


def test_bounded_audio_queue_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError):
        BoundedAudioQueue(capacity=0)


def test_runtime_counts_silent_and_clipped_chunks() -> None:
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.02,
                frequency=250,
                amplitude=1.0,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime-thresholds"),
        diagnostics_interval_seconds=1.0,
        silence_threshold_rms=1.0,
        clipping_threshold_peak=0.99,
    )

    report = runtime.run()

    assert report.metrics.chunks_received == 2
    assert report.metrics.silent_chunks == 2
    assert report.metrics.clipped_chunks == 2


class StubFeatureExtractor:
    def extract(self, chunk: object) -> list[FeatureFrame]:
        del chunk
        return [
            FeatureFrame(
                values=np.array([0.1, 0.2], dtype=np.float32),
                observed_at=1.0,
                frame_duration_seconds=0.01,
            ),
            FeatureFrame(
                values=np.array([np.nan], dtype=np.float32),
                observed_at=1.0,
                frame_duration_seconds=0.01,
            ),
        ]


def test_runtime_counts_processed_and_invalid_feature_frames() -> None:
    extractor: FeatureExtractor = StubFeatureExtractor()
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.02,
                frequency=100,
                amplitude=0.25,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime-features"),
        feature_extractor=extractor,
        diagnostics_interval_seconds=1.0,
    )

    report = runtime.run()

    assert report.metrics.chunks_received == 2
    assert report.metrics.feature_frames_processed == 2
    assert report.metrics.invalid_inference_outputs == 2


def test_runtime_counts_accepted_and_low_confidence_matches() -> None:
    extractor: FeatureExtractor = StubFeatureExtractor()
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            FeatureFrame(
                values=np.array([0.1, 0.2], dtype=np.float32),
                observed_at=0.0,
                frame_duration_seconds=0.01,
            ),
            FeatureFrame(
                values=np.array([1.0, 1.0], dtype=np.float32),
                observed_at=0.25,
                frame_duration_seconds=0.01,
            ),
        ),
        metadata={},
    )
    matcher = NearestNeighborFeatureMatcher(
        profile,
        NearestNeighborFeatureMatcherConfig(confidence_threshold=0.9),
    )
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.02,
                frequency=100,
                amplitude=0.25,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime-matches"),
        feature_extractor=extractor,
        feature_matcher=matcher,
        diagnostics_interval_seconds=1.0,
    )

    report = runtime.run()

    assert report.metrics.feature_frames_processed == 2
    assert report.metrics.accepted_matches == 2
    assert report.metrics.low_confidence_matches == 0
    assert report.last_match is not None
    assert report.last_match.reference_frame == 0


class LowConfidenceFeatureExtractor:
    def extract(self, chunk: object) -> list[FeatureFrame]:
        del chunk
        return [
            FeatureFrame(
                values=np.array([0.7, 0.7], dtype=np.float32),
                observed_at=1.0,
                frame_duration_seconds=0.01,
            )
        ]


def test_runtime_counts_low_confidence_matches() -> None:
    extractor: FeatureExtractor = LowConfidenceFeatureExtractor()
    profile = ReferenceProfile(
        name="song-a",
        frames=(
            FeatureFrame(
                values=np.array([0.1, 0.2], dtype=np.float32),
                observed_at=0.0,
                frame_duration_seconds=0.01,
            ),
        ),
        metadata={},
    )
    matcher = NearestNeighborFeatureMatcher(
        profile,
        NearestNeighborFeatureMatcherConfig(confidence_threshold=0.95),
    )
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.01,
                frequency=100,
                amplitude=0.25,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime-low-confidence"),
        feature_extractor=extractor,
        feature_matcher=matcher,
        diagnostics_interval_seconds=1.0,
    )

    report = runtime.run()

    assert report.metrics.accepted_matches == 0
    assert report.metrics.low_confidence_matches == 1
    assert report.last_match is not None
    assert report.last_match.valid is False


class RepeatingFeatureExtractor:
    def extract(self, chunk: object) -> list[FeatureFrame]:
        del chunk
        return [
            FeatureFrame(
                values=np.array([0.1, 0.2], dtype=np.float32),
                observed_at=1.0,
                frame_duration_seconds=0.01,
            )
        ]


class SequenceMatcher:
    def __init__(self, results: list[MatchResult]) -> None:
        self._results = results
        self._index = 0

    def match(self, frame: FeatureFrame) -> MatchResult:
        del frame
        result = self._results[self._index]
        self._index += 1
        return result


def test_runtime_uses_stabilizer_to_filter_large_forward_jump() -> None:
    extractor: FeatureExtractor = RepeatingFeatureExtractor()
    raw_matcher: FeatureMatcher = SequenceMatcher(
        [
            MatchResult(1, 0.25, 0.1, 0.1, 0.95, True),
            MatchResult(4, 1.0, 0.1, 0.1, 0.95, True),
            MatchResult(4, 1.0, 0.1, 0.1, 0.95, True),
        ]
    )
    matcher = StabilizedFeatureMatcher(
        raw_matcher,
        StabilizedFeatureMatcherConfig(
            max_forward_jump_frames=4,
            large_jump_threshold_frames=1,
            confirmation_count=2,
        ),
    )
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.03,
                frequency=100,
                amplitude=0.25,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime-stabilized-matches"),
        feature_extractor=extractor,
        feature_matcher=matcher,
        diagnostics_interval_seconds=1.0,
    )

    report = runtime.run()

    assert report.metrics.feature_frames_processed == 3
    assert report.metrics.accepted_matches == 2
    assert report.metrics.low_confidence_matches == 1
    assert report.last_match is not None
    assert report.last_match.reference_frame == 4
    assert report.last_match.valid is True


class SingleCommandSlideResolver:
    def __init__(self) -> None:
        self._emitted = False

    def resolve(self, match: MatchResult) -> SlideCommand | None:
        if self._emitted or not match.valid:
            return None
        self._emitted = True
        return SlideCommand(
            slide_number=1,
            section="Verse 1",
            lyrics="Amazing grace",
            reference_timestamp=match.reference_timestamp,
            confidence=match.confidence,
        )


class RecordingPresentationGateway:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.commands: list[SlideCommand] = []

    def send(self, command: SlideCommand) -> None:
        if self.fail:
            raise RuntimeError("boom")
        self.commands.append(command)


def test_runtime_sends_slide_command_when_resolver_returns_one() -> None:
    extractor: FeatureExtractor = RepeatingFeatureExtractor()
    matcher: FeatureMatcher = SequenceMatcher(
        [MatchResult(1, 0.25, 0.1, 0.1, 0.95, True)]
    )
    slide_resolver: SlideResolver = SingleCommandSlideResolver()
    gateway = RecordingPresentationGateway()
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.01,
                frequency=100,
                amplitude=0.25,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime-slide-send"),
        feature_extractor=extractor,
        feature_matcher=matcher,
        slide_resolver=slide_resolver,
        presentation_gateway=gateway,
        diagnostics_interval_seconds=1.0,
    )

    report = runtime.run()

    assert report.metrics.slide_triggers_sent == 1
    assert report.metrics.osc_send_failures == 0
    assert report.last_slide_command is not None
    assert report.last_slide_command.slide_number == 1
    assert len(gateway.commands) == 1


def test_runtime_counts_gateway_failures_for_slide_command() -> None:
    extractor: FeatureExtractor = RepeatingFeatureExtractor()
    matcher: FeatureMatcher = SequenceMatcher(
        [MatchResult(1, 0.25, 0.1, 0.1, 0.95, True)]
    )
    slide_resolver: SlideResolver = SingleCommandSlideResolver()
    gateway: PresentationGateway = RecordingPresentationGateway(fail=True)
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.01,
                frequency=100,
                amplitude=0.25,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime-slide-failure"),
        feature_extractor=extractor,
        feature_matcher=matcher,
        slide_resolver=slide_resolver,
        presentation_gateway=gateway,
        diagnostics_interval_seconds=1.0,
    )

    report = runtime.run()

    assert report.metrics.slide_triggers_sent == 0
    assert report.metrics.osc_send_failures == 1
    assert report.last_slide_command is not None
