import logging
from collections.abc import Iterator

import numpy as np
import pytest

from lyrics_aligner.adapters.audio.simulated import SimulatedAudioConfig, SimulatedAudioSource
from lyrics_aligner.adapters.matching import (
    NearestNeighborFeatureMatcher,
    NearestNeighborFeatureMatcherConfig,
    StabilizedFeatureMatcher,
    StabilizedFeatureMatcherConfig,
)
from lyrics_aligner.application.runtime import (
    AudioIngestionRuntime,
    BoundedAudioQueue,
    ManualOverrideController,
    RuntimeStatusSnapshot,
)
from lyrics_aligner.domain.models import (
    FeatureFrame,
    MatchResult,
    OperatorCorrectionRecord,
    ReferenceProfile,
    SlideCommand,
    AudioChunk,
)
from lyrics_aligner.ports.feature_extractor import FeatureExtractor
from lyrics_aligner.ports.feature_matcher import FeatureMatcher
from lyrics_aligner.ports.presentation_gateway import PresentationGateway
from lyrics_aligner.ports.slide_resolver import SlideResolver


class SequenceAudioSource:
    def __init__(self, chunks: list[AudioChunk]) -> None:
        self._chunks = chunks

    def chunks(self) -> Iterator[AudioChunk]:
        yield from self._chunks


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
    assert report.command_queue_size == 0
    assert report.manual_override_active is False
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


class UnvoicedPitchFeatureExtractor:
    def extract(self, chunk: object) -> list[FeatureFrame]:
        del chunk
        return [
            FeatureFrame(
                values=np.array([0.1, 0.2, 0.0], dtype=np.float32),
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


class ResettableSequenceMatcher:
    def __init__(self, results: list[MatchResult]) -> None:
        self._results = results
        self._index = 0
        self.reset_calls = 0

    def match(self, frame: FeatureFrame) -> MatchResult:
        del frame
        result = self._results[self._index]
        self._index += 1
        return result

    def reset(self) -> None:
        self.reset_calls += 1


class SequenceMatcher:
    def __init__(self, results: list[MatchResult]) -> None:
        self._results = results
        self._index = 0

    def match(self, frame: FeatureFrame) -> MatchResult:
        del frame
        result = self._results[self._index]
        self._index += 1
        return result


class ForceAnchorSequenceMatcher(SequenceMatcher):
    def __init__(self, results: list[MatchResult]) -> None:
        super().__init__(results)
        self.forced_timestamps: list[float] = []

    def force_anchor(self, reference_timestamp: float) -> None:
        self.forced_timestamps.append(reference_timestamp)


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


class ThresholdSlideResolver:
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._emitted = False

    def resolve(self, match: MatchResult) -> SlideCommand | None:
        if self._emitted or not match.valid or match.reference_timestamp < self._threshold:
            return None
        self._emitted = True
        return SlideCommand(
            slide_number=2,
            section="Verse 2",
            lyrics="Held through silence",
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


class MultiCommandSlideResolver:
    def __init__(self) -> None:
        self._slide_number = 0

    def resolve(self, match: MatchResult) -> SlideCommand | None:
        if not match.valid:
            return None
        self._slide_number += 1
        return SlideCommand(
            slide_number=self._slide_number,
            section=f"Section {self._slide_number}",
            lyrics="Amazing grace",
            reference_timestamp=match.reference_timestamp,
            confidence=match.confidence,
        )


class OperatorJumpSlideResolver:
    def __init__(self) -> None:
        self.seek_calls: list[int] = []

    def resolve(self, match: MatchResult) -> SlideCommand | None:
        del match
        return None

    def seek_to_slide(self, slide_number: int) -> None:
        self.seek_calls.append(slide_number)

    def slide_command_for_slide(
        self,
        slide_number: int,
        *,
        confidence: float = 1.0,
    ) -> SlideCommand:
        return SlideCommand(
            slide_number=slide_number,
            section=f"Section {slide_number}",
            lyrics=f"Slide {slide_number}",
            reference_timestamp=float(slide_number) * 10.0,
            confidence=confidence,
        )


class RecordingCorrectionSink:
    def __init__(self) -> None:
        self.records: list[OperatorCorrectionRecord] = []

    def build_record(
        self,
        *,
        profile_name: str,
        detected_reference_timestamp: float | None,
        detected_confidence: float | None = None,
        chosen_reference_timestamp: float,
        chosen_slide_number: int,
        chosen_section: str,
        chosen_lyrics: str,
        no_vocal_detected: bool = False,
        session_id: str = "",
    ) -> OperatorCorrectionRecord:
        return OperatorCorrectionRecord(
            profile_name=profile_name,
            detected_reference_timestamp=detected_reference_timestamp,
            detected_confidence=detected_confidence,
            chosen_reference_timestamp=chosen_reference_timestamp,
            chosen_slide_number=chosen_slide_number,
            chosen_section=chosen_section,
            chosen_lyrics=chosen_lyrics,
            created_at="2026-07-25T00:00:00+00:00",
            no_vocal_detected=no_vocal_detected,
            session_id=session_id,
        )

    def record(self, record: OperatorCorrectionRecord) -> None:
        self.records.append(record)


class RecordingStatusObserver:
    def __init__(self) -> None:
        self.snapshots: list[RuntimeStatusSnapshot] = []

    def update(self, snapshot: RuntimeStatusSnapshot) -> None:
        self.snapshots.append(snapshot)


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
    assert report.command_queue_capacity == 8
    assert report.command_queue_size == 0
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
    assert report.command_queue_size == 0
    assert report.last_slide_command is not None


def test_runtime_skips_matching_and_resets_tracker_during_sustained_silence() -> None:
    extractor: FeatureExtractor = RepeatingFeatureExtractor()
    matcher = ResettableSequenceMatcher([MatchResult(1, 0.25, 0.1, 0.1, 0.95, True)])
    runtime = AudioIngestionRuntime(
        source=SimulatedAudioSource(
            SimulatedAudioConfig(
                sample_rate=1_000,
                block_size=10,
                duration=0.03,
                frequency=100,
                amplitude=0.0,
            )
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime-silence-reset"),
        feature_extractor=extractor,
        feature_matcher=matcher,
        diagnostics_interval_seconds=1.0,
        silence_threshold_rms=0.01,
        silence_reset_chunk_count=2,
    )

    report = runtime.run()

    assert report.metrics.silent_chunks == 3
    assert report.metrics.feature_frames_processed == 0
    assert report.metrics.accepted_matches == 0
    assert matcher.reset_calls >= 1
    assert report.last_match is None


def test_runtime_brief_silence_holds_timeline_and_can_continue_slide_progression() -> None:
    matcher: FeatureMatcher = SequenceMatcher(
        [MatchResult(1, 0.25, 0.1, 0.1, 0.95, True)]
    )
    slide_resolver: SlideResolver = ThresholdSlideResolver(0.5)
    gateway = RecordingPresentationGateway()
    runtime = AudioIngestionRuntime(
        source=SequenceAudioSource(
            [
                AudioChunk(
                    samples=np.ones(250, dtype=np.float32) * 0.25,
                    captured_at=0.0,
                    sequence_number=0,
                ),
                AudioChunk(
                    samples=np.zeros(250, dtype=np.float32),
                    captured_at=0.25,
                    sequence_number=1,
                ),
            ]
        ),
        queue_capacity=4,
        logger=logging.getLogger("test-runtime-silence-hold"),
        feature_extractor=RepeatingFeatureExtractor(),
        feature_matcher=matcher,
        slide_resolver=slide_resolver,
        presentation_gateway=gateway,
        diagnostics_interval_seconds=1.0,
        silence_threshold_rms=0.01,
        silence_reset_chunk_count=3,
        audio_sample_rate_hz=1_000,
    )

    report = runtime.run()

    assert report.metrics.silent_chunks == 1
    assert report.metrics.accepted_matches == 1
    assert report.last_match is not None
    assert report.last_match.reference_timestamp == 0.5
    assert report.metrics.slide_triggers_sent == 1
    assert len(gateway.commands) == 1
    assert gateway.commands[0].slide_number == 2


def test_runtime_detects_non_voiced_chunks_for_pitch_aware_profiles() -> None:
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
        logger=logging.getLogger("test-runtime-no-vocal"),
        feature_extractor=UnvoicedPitchFeatureExtractor(),
        diagnostics_interval_seconds=1.0,
        vocal_presence_detection_enabled=True,
        silence_reset_chunk_count=3,
        audio_sample_rate_hz=1_000,
    )

    report = runtime.run()

    assert report.metrics.non_voiced_chunks == 2
    assert report.metrics.accepted_matches == 0
    assert report.no_vocal_detected is True


def test_runtime_suppresses_slide_command_during_manual_override() -> None:
    extractor: FeatureExtractor = RepeatingFeatureExtractor()
    matcher: FeatureMatcher = SequenceMatcher(
        [MatchResult(1, 0.25, 0.1, 0.1, 0.95, True)]
    )
    slide_resolver: SlideResolver = SingleCommandSlideResolver()
    gateway = RecordingPresentationGateway()
    controller = ManualOverrideController(active=True)
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
        logger=logging.getLogger("test-runtime-manual-override"),
        feature_extractor=extractor,
        feature_matcher=matcher,
        slide_resolver=slide_resolver,
        presentation_gateway=gateway,
        diagnostics_interval_seconds=1.0,
        manual_override_controller=controller,
    )

    report = runtime.run()

    assert report.metrics.slide_triggers_sent == 0
    assert report.metrics.manual_override_suppressed_triggers == 1
    assert report.manual_override_active is True
    assert gateway.commands == []


class ToggleAfterFirstExtractFeatureExtractor:
    def __init__(self, controller: ManualOverrideController) -> None:
        self._controller = controller
        self._calls = 0

    def extract(self, chunk: object) -> list[FeatureFrame]:
        del chunk
        self._calls += 1
        if self._calls == 2:
            self._controller.deactivate()
        return [
            FeatureFrame(
                values=np.array([0.1, 0.2], dtype=np.float32),
                observed_at=1.0,
                frame_duration_seconds=0.01,
            )
        ]


def test_runtime_resumes_slide_delivery_after_manual_override_is_disabled() -> None:
    controller = ManualOverrideController(active=True)
    extractor: FeatureExtractor = ToggleAfterFirstExtractFeatureExtractor(controller)
    matcher: FeatureMatcher = SequenceMatcher(
        [
            MatchResult(1, 0.25, 0.1, 0.1, 0.95, True),
            MatchResult(2, 0.50, 0.1, 0.1, 0.95, True),
        ]
    )
    slide_resolver: SlideResolver = MultiCommandSlideResolver()
    gateway = RecordingPresentationGateway()
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
        logger=logging.getLogger("test-runtime-manual-override-resume"),
        feature_extractor=extractor,
        feature_matcher=matcher,
        slide_resolver=slide_resolver,
        presentation_gateway=gateway,
        diagnostics_interval_seconds=1.0,
        manual_override_controller=controller,
    )

    report = runtime.run()

    assert report.metrics.slide_triggers_sent == 1
    assert report.metrics.manual_override_suppressed_triggers == 1
    assert report.manual_override_active is False
    assert len(gateway.commands) == 1
    assert gateway.commands[0].slide_number == 2


def test_runtime_jump_to_slide_activates_manual_override_and_reanchors() -> None:
    matcher = ForceAnchorSequenceMatcher([])
    slide_resolver = OperatorJumpSlideResolver()
    gateway = RecordingPresentationGateway()
    correction_sink = RecordingCorrectionSink()
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
        logger=logging.getLogger("test-runtime-jump-to-slide"),
        feature_matcher=matcher,
        slide_resolver=slide_resolver,
        presentation_gateway=gateway,
        diagnostics_interval_seconds=1.0,
        operator_correction_sink=correction_sink,
        profile_name="song-a",
    )
    runtime._last_match = MatchResult(1, 12.5, 0.1, 0.1, 0.95, True)  # noqa: SLF001

    command = runtime.jump_to_slide(3)

    assert command.slide_number == 3
    assert runtime._manual_override_controller.is_active is True
    assert slide_resolver.seek_calls == [3]
    assert matcher.forced_timestamps == [30.0]
    assert len(gateway.commands) == 1
    assert gateway.commands[0].slide_number == 3
    assert len(correction_sink.records) == 1
    assert correction_sink.records[0].profile_name == "song-a"
    assert correction_sink.records[0].detected_reference_timestamp == 12.5
    assert correction_sink.records[0].detected_confidence == 0.95
    assert correction_sink.records[0].chosen_reference_timestamp == 30.0
    assert correction_sink.records[0].chosen_slide_number == 3
    assert correction_sink.records[0].session_id != ""


def test_runtime_exposes_operator_slide_targets() -> None:
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
        logger=logging.getLogger("test-runtime-operator-targets"),
        operator_slide_targets=(
            SlideCommand(1, "Verse 1", "Amazing grace", 1.0, 1.0),
            SlideCommand(2, "Verse 2", "How sweet the sound", 5.0, 1.0),
        ),
    )

    assert [target.slide_number for target in runtime.operator_slide_targets()] == [1, 2]


def test_runtime_marks_alignment_hold_for_low_confidence_matches() -> None:
    observer = RecordingStatusObserver()
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
        logger=logging.getLogger("test-runtime-alignment-hold"),
        feature_extractor=LowConfidenceFeatureExtractor(),
        feature_matcher=NearestNeighborFeatureMatcher(
            ReferenceProfile(
                name="song-a",
                frames=(
                    FeatureFrame(
                        values=np.array([0.1, 0.2], dtype=np.float32),
                        observed_at=0.0,
                        frame_duration_seconds=0.01,
                    ),
                ),
                metadata={},
            ),
            NearestNeighborFeatureMatcherConfig(confidence_threshold=0.95),
        ),
        diagnostics_interval_seconds=1.0,
        status_observer=observer,
    )

    report = runtime.run()

    assert report.alignment_hold_active is True
    assert report.alignment_hold_reason == "low_confidence"
    assert observer.snapshots[-1].alignment_hold_active is True
    assert observer.snapshots[-1].alignment_hold_reason == "low_confidence"
