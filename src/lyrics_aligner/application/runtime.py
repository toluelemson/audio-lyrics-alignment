from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from time import monotonic
from typing import Protocol
from uuid import uuid4

import numpy as np

from lyrics_aligner.application.metrics import RuntimeMetrics
from lyrics_aligner.domain.models import (
    AudioChunk,
    FeatureFrame,
    MatchResult,
    OperatorCorrectionRecord,
    SlideCommand,
)
from lyrics_aligner.ports.audio_source import AudioSource
from lyrics_aligner.ports.feature_extractor import FeatureExtractor
from lyrics_aligner.ports.feature_matcher import FeatureMatcher
from lyrics_aligner.ports.presentation_gateway import PresentationGateway
from lyrics_aligner.ports.slide_resolver import SlideResolver


class BoundedAudioQueue:
    """Thread-safe queue that drops the oldest item instead of blocking producers."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        self._capacity = capacity
        self._items: deque[AudioChunk] = deque()
        self._condition = Condition()
        self._closed = False

    @property
    def capacity(self) -> int:
        return self._capacity

    def put_drop_oldest(self, item: AudioChunk) -> bool:
        with self._condition:
            dropped = False
            if len(self._items) == self._capacity:
                self._items.popleft()
                dropped = True
            self._items.append(item)
            self._condition.notify()
            return dropped

    def get(self, timeout: float | None = None) -> AudioChunk | None:
        with self._condition:
            if timeout is None:
                while not self._items and not self._closed:
                    self._condition.wait()
            else:
                deadline = monotonic() + timeout
                while not self._items and not self._closed:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)

            if self._items:
                return self._items.popleft()
            return None

    def get_latest(self, timeout: float | None = None) -> tuple[AudioChunk | None, int]:
        with self._condition:
            if timeout is None:
                while not self._items and not self._closed:
                    self._condition.wait()
            else:
                deadline = monotonic() + timeout
                while not self._items and not self._closed:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return None, 0
                    self._condition.wait(remaining)

            if not self._items:
                return None, 0
            dropped = max(0, len(self._items) - 1)
            latest = self._items.pop()
            self._items.clear()
            return latest, dropped

    def qsize(self) -> int:
        with self._condition:
            return len(self._items)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class BoundedCommandQueue:
    """Thread-safe queue for slide commands with drop-oldest backpressure."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        self._capacity = capacity
        self._items: deque[SlideCommand] = deque()
        self._condition = Condition()
        self._closed = False

    @property
    def capacity(self) -> int:
        return self._capacity

    def put_drop_oldest(self, item: SlideCommand) -> bool:
        with self._condition:
            dropped = False
            if len(self._items) == self._capacity:
                self._items.popleft()
                dropped = True
            self._items.append(item)
            self._condition.notify()
            return dropped

    def get(self, timeout: float | None = None) -> SlideCommand | None:
        with self._condition:
            if timeout is None:
                while not self._items and not self._closed:
                    self._condition.wait()
            else:
                deadline = monotonic() + timeout
                while not self._items and not self._closed:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)

            if self._items:
                return self._items.popleft()
            return None

    def qsize(self) -> int:
        with self._condition:
            return len(self._items)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class ManualOverrideController:
    """Thread-safe controller for operator handoff between automatic and manual mode."""

    def __init__(self, active: bool = False) -> None:
        self._active = active
        self._lock = Lock()

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def activate(self) -> bool:
        with self._lock:
            changed = not self._active
            self._active = True
            return changed

    def deactivate(self) -> bool:
        with self._lock:
            changed = self._active
            self._active = False
            return changed

    def toggle(self) -> bool:
        with self._lock:
            self._active = not self._active
            return self._active


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    device_name: str
    metrics: RuntimeMetrics
    rms: float
    peak: float
    queue_size: int
    queue_capacity: int
    command_queue_size: int
    command_queue_capacity: int
    manual_override_active: bool
    tracking_state: str | None
    last_match: MatchResult | None
    last_slide_command: SlideCommand | None
    matched_slide_command: SlideCommand | None = None
    current_section_key: str | None = None
    current_section_label: str | None = None
    current_section_slide_number: int | None = None
    recovery_active: bool = False
    suggested_section_key: str | None = None
    suggested_section_label: str | None = None
    no_vocal_detected: bool = False
    alignment_hold_active: bool = False
    alignment_hold_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStatusSnapshot:
    device_name: str
    metrics: RuntimeMetrics
    rms: float
    peak: float
    queue_size: int
    queue_capacity: int
    command_queue_size: int
    command_queue_capacity: int
    manual_override_active: bool
    tracking_state: str | None
    last_match: MatchResult | None
    last_slide_command: SlideCommand | None
    matched_slide_command: SlideCommand | None = None
    current_section_key: str | None = None
    current_section_label: str | None = None
    current_section_slide_number: int | None = None
    recovery_active: bool = False
    suggested_section_key: str | None = None
    suggested_section_label: str | None = None
    no_vocal_detected: bool = False
    alignment_hold_active: bool = False
    alignment_hold_reason: str | None = None


class RuntimeStatusObserver(Protocol):
    def update(self, snapshot: RuntimeStatusSnapshot) -> None: ...


class OperatorCorrectionSink(Protocol):
    def record(self, record: OperatorCorrectionRecord) -> None: ...


class AudioIngestionRuntime:
    """Run audio capture and queue consumption with periodic diagnostics."""

    def __init__(
        self,
        source: AudioSource,
        queue_capacity: int,
        logger: logging.Logger,
        feature_extractor: FeatureExtractor | None = None,
        feature_matcher: FeatureMatcher | None = None,
        slide_resolver: SlideResolver | None = None,
        presentation_gateway: PresentationGateway | None = None,
        diagnostics_interval_seconds: float = 0.5,
        device_name: str = "SimulatedAudioSource",
        silence_threshold_rms: float = 0.01,
        clipping_threshold_peak: float = 0.99,
        silence_reset_chunk_count: int = 3,
        command_queue_capacity: int = 8,
        manual_override_controller: ManualOverrideController | None = None,
        status_observer: RuntimeStatusObserver | None = None,
        operator_correction_sink: OperatorCorrectionSink | None = None,
        profile_name: str = "unknown-profile",
        operator_slide_targets: tuple[SlideCommand, ...] = (),
        audio_sample_rate_hz: int = 16_000,
        vocal_presence_detection_enabled: bool = False,
        voiced_pitch_threshold: float = 0.02,
        timeline_hold_max_seconds: float = 4.0,
        queue_catchup_threshold: int = 4,
        clock_resync_tolerance_seconds: float = 3.0,
        fast_forward_accept_tolerance_seconds: float = 6.0,
        fast_forward_confidence_floor: float = 0.8,
        reacquire_consistency_tolerance_seconds: float = 2.0,
        reacquire_min_consecutive_matches: int = 2,
        reacquire_min_duration_seconds: float = 0.0,
        emit_diagnostic_logs: bool = True,
        emit_match_debug_logs: bool = False,
    ) -> None:
        self._source = source
        self._queue = BoundedAudioQueue(queue_capacity)
        self._command_queue = BoundedCommandQueue(command_queue_capacity)
        self._logger = logger
        self._feature_extractor = feature_extractor
        self._feature_matcher = feature_matcher
        self._slide_resolver = slide_resolver
        self._presentation_gateway = presentation_gateway
        self._diagnostics_interval_seconds = diagnostics_interval_seconds
        self._device_name = device_name
        self._silence_threshold_rms = silence_threshold_rms
        self._clipping_threshold_peak = clipping_threshold_peak
        self._silence_reset_chunk_count = silence_reset_chunk_count
        self._manual_override_controller = (
            manual_override_controller or ManualOverrideController()
        )
        self._status_observer = status_observer
        self._operator_correction_sink = operator_correction_sink
        self._profile_name = profile_name
        self._operator_slide_targets = operator_slide_targets
        self._audio_sample_rate_hz = audio_sample_rate_hz
        self._session_id = uuid4().hex
        self._vocal_presence_detection_enabled = vocal_presence_detection_enabled
        self._voiced_pitch_threshold = voiced_pitch_threshold
        self._timeline_hold_max_seconds = timeline_hold_max_seconds
        self._queue_catchup_threshold = queue_catchup_threshold
        self._clock_resync_tolerance_seconds = clock_resync_tolerance_seconds
        self._fast_forward_accept_tolerance_seconds = (
            fast_forward_accept_tolerance_seconds
        )
        self._fast_forward_confidence_floor = fast_forward_confidence_floor
        self._reacquire_consistency_tolerance_seconds = (
            reacquire_consistency_tolerance_seconds
        )
        self._reacquire_min_consecutive_matches = reacquire_min_consecutive_matches
        self._reacquire_min_duration_seconds = reacquire_min_duration_seconds
        self._emit_diagnostic_logs = emit_diagnostic_logs
        self._emit_match_debug_logs = emit_match_debug_logs
        self._metrics = RuntimeMetrics()
        self._metrics_lock = Lock()
        self._stop_requested = Event()
        self._last_rms = 0.0
        self._last_peak = 0.0
        self._consecutive_silent_chunks = 0
        self._consecutive_non_voiced_chunks = 0
        self._last_match: MatchResult | None = None
        self._last_slide_command: SlideCommand | None = None
        self._timeline_mode = "SEARCHING"
        self._timeline_anchor_reference_timestamp: float | None = None
        self._timeline_anchor_observed_at: float | None = None
        self._timeline_anchor_confidence: float = 0.0
        self._timeline_anchor_reference_frame: int | None = None
        self._timeline_anchor_raw_distance: float = 0.0
        self._timeline_anchor_normalized_distance: float = 0.0
        self._timeline_rate_estimate = 1.0
        self._reacquire_candidate_timestamp: float | None = None
        self._reacquire_candidate_observed_at: float | None = None
        self._reacquire_start_observed_at: float | None = None
        self._reacquire_streak = 0
        self._alignment_hold_active = False
        self._alignment_hold_reason: str | None = None
        self._manual_jump_display_active = False
        self._manual_timeline_anchor_active = False
        self._timing_only_hold_elapsed_seconds = 0.0
        self._last_auto_emitted_slide_number: int | None = None
        self._last_auto_emitted_at: float | None = None
        self._last_queue_catchup_drop_count = 0
        self._last_queue_catchup_at: float | None = None
        self._last_producer_drop_at: float | None = None

    def run(self) -> RuntimeReport:
        producer = Thread(target=self._produce, name="audio-producer", daemon=True)
        dispatcher = Thread(target=self._dispatch_commands, name="slide-dispatcher", daemon=True)
        producer.start()
        dispatcher.start()

        next_report_at = monotonic() + self._diagnostics_interval_seconds
        try:
            while producer.is_alive() or self._queue.qsize() > 0:
                if self._queue.qsize() >= self._queue_catchup_threshold:
                    chunk, dropped = self._queue.get_latest(timeout=0.1)
                    if dropped:
                        self._last_queue_catchup_drop_count = dropped
                        self._last_queue_catchup_at = monotonic()
                        with self._metrics_lock:
                            self._metrics.chunks_dropped += dropped
                else:
                    chunk = self._queue.get(timeout=0.1)
                if chunk is not None:
                    self._process(chunk)

                if monotonic() >= next_report_at:
                    self._publish_status()
                    next_report_at = monotonic() + self._diagnostics_interval_seconds
        finally:
            self.stop()
            producer.join()
            dispatcher.join()

        self._publish_status()
        snapshot = self._build_status_snapshot()
        return RuntimeReport(
            device_name=snapshot.device_name,
            metrics=snapshot.metrics,
            rms=snapshot.rms,
            peak=snapshot.peak,
            queue_size=snapshot.queue_size,
            queue_capacity=snapshot.queue_capacity,
            command_queue_size=snapshot.command_queue_size,
            command_queue_capacity=snapshot.command_queue_capacity,
            manual_override_active=snapshot.manual_override_active,
            tracking_state=snapshot.tracking_state,
            last_match=snapshot.last_match,
            last_slide_command=snapshot.last_slide_command,
            matched_slide_command=snapshot.matched_slide_command,
            current_section_key=snapshot.current_section_key,
            current_section_label=snapshot.current_section_label,
            current_section_slide_number=snapshot.current_section_slide_number,
            recovery_active=snapshot.recovery_active,
            suggested_section_key=snapshot.suggested_section_key,
            suggested_section_label=snapshot.suggested_section_label,
            no_vocal_detected=snapshot.no_vocal_detected,
            alignment_hold_active=snapshot.alignment_hold_active,
            alignment_hold_reason=snapshot.alignment_hold_reason,
        )

    def stop(self) -> None:
        self._stop_requested.set()
        self._queue.close()
        self._command_queue.close()
        stop_method = getattr(self._source, "stop", None)
        if callable(stop_method):
            stop_method()

    def emit_operator_slide_command(self, command: SlideCommand) -> None:
        self._log_slide_command_decision(
            source="manual",
            command=command,
            match_result=self._last_match,
        )
        self._last_slide_command = command
        if self._presentation_gateway is None:
            with self._metrics_lock:
                self._metrics.slide_triggers_sent += 1
            self._publish_status()
            return
        try:
            self._presentation_gateway.send(command)
        except Exception:
            self._logger.exception(
                "Failed to send operator slide command slide_number=%s section=%s",
                command.slide_number,
                command.section,
            )
            with self._metrics_lock:
                self._metrics.osc_send_failures += 1
            self._publish_status()
            return
        with self._metrics_lock:
            self._metrics.slide_triggers_sent += 1
        self._publish_status()

    def enable_manual_override(self) -> bool:
        changed = self._manual_override_controller.activate()
        self._publish_status()
        return changed

    def disable_manual_override(self) -> bool:
        changed = self._manual_override_controller.deactivate()
        self._publish_status()
        return changed

    def toggle_manual_override(self) -> bool:
        active = self._manual_override_controller.toggle()
        self._publish_status()
        return active

    def force_match_anchor(self, reference_timestamp: float) -> bool:
        force_anchor = getattr(self._feature_matcher, "force_anchor", None)
        if not callable(force_anchor):
            return False
        force_anchor(reference_timestamp)
        self._publish_status()
        return True

    def jump_to_slide(
        self,
        slide_number: int,
        *,
        activate_manual_override: bool = False,
        realign_tracker: bool = False,
        record_correction: bool = False,
    ) -> SlideCommand:
        build_command = getattr(self._slide_resolver, "slide_command_for_slide", None)
        if not callable(build_command):
            raise ValueError("slide resolver does not support operator slide jumps")

        command = build_command(slide_number, confidence=1.0)
        if activate_manual_override:
            self._manual_override_controller.activate()
        if realign_tracker:
            seek_to_slide = getattr(self._slide_resolver, "seek_to_slide", None)
            if not callable(seek_to_slide):
                raise ValueError("slide resolver does not support operator seeking")
            seek_to_slide(slide_number)
            self.force_match_anchor(command.reference_timestamp)
            self._manual_timeline_anchor_active = True
        self._alignment_hold_active = False
        self._alignment_hold_reason = None
        if record_correction:
            self._record_operator_correction(command)
        self._manual_jump_display_active = True
        self.emit_operator_slide_command(command)
        return command

    def operator_slide_targets(self) -> tuple[SlideCommand, ...]:
        return self._operator_slide_targets

    def _produce(self) -> None:
        try:
            for chunk in self._source.chunks():
                if self._stop_requested.is_set():
                    return
                dropped = self._queue.put_drop_oldest(chunk)
                with self._metrics_lock:
                    self._metrics.chunks_received += 1
                    if dropped:
                        self._metrics.chunks_dropped += 1
                        self._last_producer_drop_at = monotonic()
                    self._metrics.queue_high_water_mark = max(
                        self._metrics.queue_high_water_mark,
                        self._queue.qsize(),
                    )
        finally:
            self._queue.close()

    def _process(self, chunk: AudioChunk) -> None:
        squared = np.square(chunk.samples, dtype=np.float32)
        self._last_rms = float(np.sqrt(np.mean(squared, dtype=np.float32)))
        self._last_peak = float(np.max(np.abs(chunk.samples)))
        is_silent = self._last_rms <= self._silence_threshold_rms
        with self._metrics_lock:
            if is_silent:
                self._metrics.silent_chunks += 1
            if self._last_peak >= self._clipping_threshold_peak:
                self._metrics.clipped_chunks += 1
        if is_silent:
            self._consecutive_silent_chunks += 1
            if self._consecutive_silent_chunks < self._silence_reset_chunk_count:
                self._hold_timeline_during_silence(chunk)
                return
            if self._manual_timeline_anchor_active:
                self._alignment_hold_active = True
                self._alignment_hold_reason = "manual_anchor_timing"
                if not self._hold_timeline_during_silence(chunk):
                    self._reset_matching_due_to_silence()
                return
            if self._consecutive_silent_chunks >= self._silence_reset_chunk_count:
                self._reset_matching_due_to_silence()
            return

        self._consecutive_silent_chunks = 0
        if self._feature_extractor is not None:
            frames = self._feature_extractor.extract(chunk)
            if self._vocal_presence_detection_enabled and not self._frames_include_voiced_pitch(frames):
                self._consecutive_non_voiced_chunks += 1
                with self._metrics_lock:
                    self._metrics.non_voiced_chunks += 1
                self._record_feature_frames(frames, allow_matching=False)
                if self._consecutive_non_voiced_chunks < self._silence_reset_chunk_count:
                    self._hold_timeline_during_silence(chunk)
                elif self._manual_timeline_anchor_active:
                    self._alignment_hold_active = True
                    self._alignment_hold_reason = "manual_anchor_timing"
                    if not self._hold_timeline_during_silence(chunk):
                        self._reset_matching_due_to_silence()
                else:
                    self._reset_matching_due_to_silence()
                return
            self._consecutive_non_voiced_chunks = 0
            self._timing_only_hold_elapsed_seconds = 0.0
            self._record_feature_frames(frames, allow_matching=True)

    def _record_operator_correction(self, command: SlideCommand) -> None:
        if self._operator_correction_sink is None:
            return
        record = OperatorCorrectionRecord(
            profile_name=self._profile_name,
            detected_reference_timestamp=(
                None if self._last_match is None else self._last_match.reference_timestamp
            ),
            detected_confidence=None if self._last_match is None else self._last_match.confidence,
            chosen_reference_timestamp=command.reference_timestamp,
            chosen_slide_number=command.slide_number,
            chosen_section=command.section,
            chosen_lyrics=command.lyrics,
            created_at="",
            no_vocal_detected=self._build_status_snapshot().no_vocal_detected,
            session_id=self._session_id,
        )
        build_record = getattr(self._operator_correction_sink, "build_record", None)
        if callable(build_record):
            record = build_record(
                profile_name=record.profile_name,
                detected_reference_timestamp=record.detected_reference_timestamp,
                detected_confidence=record.detected_confidence,
                chosen_reference_timestamp=record.chosen_reference_timestamp,
                chosen_slide_number=record.chosen_slide_number,
                chosen_section=record.chosen_section,
                chosen_lyrics=record.chosen_lyrics,
                no_vocal_detected=record.no_vocal_detected,
                session_id=record.session_id,
            )
        self._operator_correction_sink.record(record)

    def _publish_status(self) -> None:
        snapshot = self._build_status_snapshot()
        if self._emit_diagnostic_logs:
            self._log_diagnostics(snapshot)
        if self._status_observer is not None:
            self._status_observer.update(snapshot)

    def _log_diagnostics(self, snapshot: RuntimeStatusSnapshot) -> None:
        self._logger.info(
            "device=%s rms=%.2f peak=%.2f queue=%s/%s "
            "command_queue=%s/%s "
            "chunks_received=%s chunks_dropped=%s "
            "silent_chunks=%s clipped_chunks=%s feature_frames_processed=%s "
            "non_voiced_chunks=%s "
            "accepted_matches=%s low_confidence_matches=%s slide_triggers_sent=%s "
            "manual_override_suppressed=%s "
            "osc_send_failures=%s tracking_state=%s last_reference_timestamp=%s "
            "last_slide_number=%s last_confidence=%.2f manual_override=%s no_vocal=%s",
            snapshot.device_name,
            snapshot.rms,
            snapshot.peak,
            snapshot.queue_size,
            snapshot.queue_capacity,
            snapshot.command_queue_size,
            snapshot.command_queue_capacity,
            snapshot.metrics.chunks_received,
            snapshot.metrics.chunks_dropped,
            snapshot.metrics.silent_chunks,
            snapshot.metrics.clipped_chunks,
            snapshot.metrics.feature_frames_processed,
            snapshot.metrics.non_voiced_chunks,
            snapshot.metrics.accepted_matches,
            snapshot.metrics.low_confidence_matches,
            snapshot.metrics.slide_triggers_sent,
            snapshot.metrics.manual_override_suppressed_triggers,
            snapshot.metrics.osc_send_failures,
            snapshot.tracking_state or "none",
            (
                f"{snapshot.last_match.reference_timestamp:.2f}"
                if snapshot.last_match is not None
                else "none"
            ),
            (
                snapshot.last_slide_command.slide_number
                if snapshot.last_slide_command is not None
                else "none"
            ),
            snapshot.last_match.confidence if snapshot.last_match is not None else 0.0,
            "active" if snapshot.manual_override_active else "auto",
            "yes" if snapshot.no_vocal_detected else "no",
        )

    def _record_feature_frames(
        self,
        frames: list[FeatureFrame],
        *,
        allow_matching: bool,
    ) -> None:
        valid_frames = 0
        invalid_frames = 0
        for frame in frames:
            values = np.asarray(frame.values, dtype=np.float32)
            if values.size == 0 or not np.isfinite(values).all():
                invalid_frames += 1
                continue
            valid_frames += 1
            if allow_matching and self._feature_matcher is not None:
                self._record_match(self._feature_matcher.match(frame), frame=frame)

        with self._metrics_lock:
            self._metrics.feature_frames_processed += valid_frames
            self._metrics.invalid_inference_outputs += invalid_frames

    def _record_match(self, result: MatchResult, *, frame: FeatureFrame) -> None:
        with self._metrics_lock:
            if result.valid:
                self._metrics.accepted_matches += 1
            else:
                self._metrics.low_confidence_matches += 1
        applied_result = self._select_runtime_result(result, frame=frame)
        self._log_match_acceptance(
            candidate=result,
            applied_result=applied_result,
            frame=frame,
        )
        self._apply_match_result(applied_result, count_metrics=False)

    def _apply_match_result(
        self,
        result: MatchResult,
        *,
        count_metrics: bool,
    ) -> None:
        self._log_match_debug()
        self._last_match = result
        self._alignment_hold_active = not result.valid
        self._alignment_hold_reason = "low_confidence" if not result.valid else None
        if count_metrics:
            with self._metrics_lock:
                if result.valid:
                    self._metrics.accepted_matches += 1
                else:
                    self._metrics.low_confidence_matches += 1
        if self._restore_from_manual_jump_if_needed(result):
            return
        tracking_state = self._tracking_state()
        if tracking_state in {"SEARCHING", "UNINITIALIZED", "UNCERTAIN"}:
            return
        if self._slide_resolver is not None:
            command = self._slide_resolver.resolve(result)
            if command is not None:
                self._manual_jump_display_active = False
                self._log_slide_command_decision(
                    source="auto",
                    command=command,
                    match_result=result,
                )
                self._emit_slide_command(command)

    def _hold_timeline_during_silence(self, chunk: AudioChunk) -> bool:
        if self._last_match is None or not self._last_match.valid:
            return False
        chunk_duration_seconds = float(chunk.samples.size) if chunk.samples.size else 0.0
        if chunk_duration_seconds <= 0.0:
            return False
        if (
            self._timing_only_hold_elapsed_seconds + chunk_duration_seconds / float(self._audio_sample_rate_hz)
            > self._timeline_hold_max_seconds
        ):
            return False
        self._timing_only_hold_elapsed_seconds += (
            chunk_duration_seconds / float(self._audio_sample_rate_hz)
        )
        held_result = MatchResult(
            reference_frame=self._last_match.reference_frame,
            reference_timestamp=(
                self._last_match.reference_timestamp
                + chunk_duration_seconds / float(self._audio_sample_rate_hz)
            ),
            raw_distance=self._last_match.raw_distance,
            normalized_distance=self._last_match.normalized_distance,
            confidence=self._last_match.confidence,
            valid=True,
        )
        self._apply_match_result(held_result, count_metrics=False)
        return True

    def _restore_from_manual_jump_if_needed(self, result: MatchResult) -> bool:
        if not result.valid or not self._manual_jump_display_active or self._slide_resolver is None:
            return False
        build_command = getattr(
            self._slide_resolver,
            "active_slide_command_for_timestamp",
            None,
        )
        if not callable(build_command):
            return False
        command = build_command(result.reference_timestamp, confidence=result.confidence)
        if command is None:
            return False
        if (
            self._last_slide_command is not None
            and command.slide_number == self._last_slide_command.slide_number
            and abs(command.reference_timestamp - self._last_slide_command.reference_timestamp)
            < 1e-6
        ):
            self._manual_jump_display_active = False
            return False
        self._manual_jump_display_active = False
        self._log_slide_command_decision(
            source="restore",
            command=command,
            match_result=result,
        )
        self._emit_slide_command(command)
        return True

    def _log_slide_command_decision(
        self,
        *,
        source: str,
        command: SlideCommand,
        match_result: MatchResult | None,
    ) -> None:
        resolver_state = self._slide_resolver_debug_state()
        self._logger.info(
            "SLIDE_DECISION source=%s command_slide=%s command_timestamp=%.2f "
            "tracking_state=%s match_timestamp=%s match_confidence=%s match_valid=%s "
            "last_slide=%s manual_display_active=%s resolver=%s",
            source,
            command.slide_number,
            command.reference_timestamp,
            self._tracking_state() or "none",
            (
                f"{match_result.reference_timestamp:.2f}"
                if match_result is not None
                else "none"
            ),
            (
                f"{match_result.confidence:.2f}"
                if match_result is not None
                else "none"
            ),
            (
                match_result.valid
                if match_result is not None
                else "none"
            ),
            (
                self._last_slide_command.slide_number
                if self._last_slide_command is not None
                else "none"
            ),
            self._manual_jump_display_active,
            resolver_state,
        )

    def _slide_resolver_debug_state(self) -> str:
        if self._slide_resolver is None:
            return "none"
        parts: list[str] = [type(self._slide_resolver).__name__]
        for name in (
            "_next_index",
            "_candidate_count",
            "_last_candidate_slide_number",
            "_last_emitted_slide_number",
            "_last_emitted_reference_timestamp",
        ):
            if hasattr(self._slide_resolver, name):
                parts.append(f"{name[1:]}={getattr(self._slide_resolver, name)!r}")
        return " ".join(parts)

    def _frames_include_voiced_pitch(self, frames: list[FeatureFrame]) -> bool:
        for frame in frames:
            values = np.asarray(frame.values, dtype=np.float32)
            if values.size == 0 or not np.isfinite(values).all():
                continue
            if abs(float(values[-1])) > self._voiced_pitch_threshold:
                return True
        return False

    def _log_match_debug(self) -> None:
        if not self._emit_match_debug_logs or self._feature_matcher is None:
            return
        decision = getattr(self._feature_matcher, "last_decision", None)
        if decision is None:
            return
        self._logger.info(
            "MATCH_DEBUG prev_state=%s state=%s accepted=%s reason=%s "
            "frame=%s timestamp=%.2f confidence=%.2f candidate_valid=%s delta=%s "
            "matcher_reason=%s second_gap=%s second_time_gap=%s",
            decision.previous_state,
            decision.state,
            decision.accepted,
            decision.reason,
            decision.candidate_frame,
            decision.candidate_timestamp,
            decision.candidate_confidence,
            decision.candidate_valid,
            (
                decision.delta_from_last_accepted
                if decision.delta_from_last_accepted is not None
                else "none"
            ),
            decision.matcher_reason or "none",
            (
                f"{decision.matcher_second_distance_gap:.4f}"
                if decision.matcher_second_distance_gap is not None
                else "none"
            ),
            (
                f"{decision.matcher_second_time_gap_seconds:.2f}"
                if decision.matcher_second_time_gap_seconds is not None
                else "none"
            ),
        )

    def _log_match_acceptance(
        self,
        *,
        candidate: MatchResult,
        applied_result: MatchResult,
        frame: FeatureFrame,
    ) -> None:
        if self._feature_matcher is None:
            return
        decision = getattr(self._feature_matcher, "last_decision", None)
        if decision is None or not getattr(decision, "accepted", False):
            return
        predicted_result = self._predict_match_from_anchor(frame)
        queue_catchup_age = (
            None
            if self._last_queue_catchup_at is None
            else max(0.0, monotonic() - self._last_queue_catchup_at)
        )
        producer_drop_age = (
            None
            if self._last_producer_drop_at is None
            else max(0.0, monotonic() - self._last_producer_drop_at)
        )
        self._logger.info(
            "MATCH_ACCEPT reason=%s prev_state=%s state=%s frame_observed_at=%.3f "
            "candidate_timestamp=%.2f candidate_confidence=%.2f candidate_valid=%s "
            "applied_timestamp=%.2f applied_confidence=%.2f applied_valid=%s "
            "delta_from_last_accepted=%s timeline_mode=%s "
            "anchor_timestamp=%s anchor_confidence=%.2f predicted_timestamp=%s "
            "anchor_observed_at=%s queue_size=%s catchup_drop_count=%s catchup_age=%s "
            "producer_drop_age=%s matcher_reason=%s matcher_second_gap=%s matcher_second_time_gap=%s",
            decision.reason,
            decision.previous_state,
            decision.state,
            frame.observed_at,
            candidate.reference_timestamp,
            candidate.confidence,
            candidate.valid,
            applied_result.reference_timestamp,
            applied_result.confidence,
            applied_result.valid,
            (
                decision.delta_from_last_accepted
                if decision.delta_from_last_accepted is not None
                else "none"
            ),
            self._timeline_mode,
            (
                f"{self._timeline_anchor_reference_timestamp:.2f}"
                if self._timeline_anchor_reference_timestamp is not None
                else "none"
            ),
            self._timeline_anchor_confidence,
            (
                f"{predicted_result.reference_timestamp:.2f}"
                if predicted_result is not None
                else "none"
            ),
            (
                f"{self._timeline_anchor_observed_at:.3f}"
                if self._timeline_anchor_observed_at is not None
                else "none"
            ),
            self._queue.qsize(),
            self._last_queue_catchup_drop_count,
            (
                f"{queue_catchup_age:.3f}"
                if queue_catchup_age is not None
                else "none"
            ),
            (
                f"{producer_drop_age:.3f}"
                if producer_drop_age is not None
                else "none"
            ),
            decision.matcher_reason or "none",
            (
                f"{decision.matcher_second_distance_gap:.4f}"
                if decision.matcher_second_distance_gap is not None
                else "none"
            ),
            (
                f"{decision.matcher_second_time_gap_seconds:.2f}"
                if decision.matcher_second_time_gap_seconds is not None
                else "none"
            ),
        )

    def _reset_matching_due_to_silence(self) -> None:
        reset_method = getattr(self._feature_matcher, "reset", None)
        if callable(reset_method):
            reset_method()
        self._last_match = None
        self._timeline_mode = "SEARCHING"
        self._timeline_anchor_reference_timestamp = None
        self._timeline_anchor_observed_at = None
        self._timeline_anchor_confidence = 0.0
        self._timeline_anchor_reference_frame = None
        self._timeline_anchor_raw_distance = 0.0
        self._timeline_anchor_normalized_distance = 0.0
        self._timeline_rate_estimate = 1.0
        self._reset_reacquire_candidate()
        self._manual_timeline_anchor_active = False
        self._timing_only_hold_elapsed_seconds = 0.0
        self._alignment_hold_active = True
        self._alignment_hold_reason = "silence"

    def _update_timeline_anchor(
        self,
        result: MatchResult,
        *,
        observed_at: float,
    ) -> None:
        self._update_timeline_rate_estimate(
            observed_at=observed_at,
            reference_timestamp=result.reference_timestamp,
        )
        self._timeline_anchor_reference_timestamp = result.reference_timestamp
        self._timeline_anchor_observed_at = observed_at
        self._timeline_anchor_confidence = result.confidence
        self._timeline_anchor_reference_frame = result.reference_frame
        self._timeline_anchor_raw_distance = result.raw_distance
        self._timeline_anchor_normalized_distance = result.normalized_distance
        self._timing_only_hold_elapsed_seconds = 0.0
        self._timeline_mode = "LOCKED_CLOCK"
        self._reset_reacquire_candidate()

    def _update_timeline_rate_estimate(
        self,
        *,
        observed_at: float,
        reference_timestamp: float,
    ) -> None:
        if (
            self._timeline_anchor_reference_timestamp is None
            or self._timeline_anchor_observed_at is None
        ):
            self._timeline_rate_estimate = 1.0
            return
        observed_delta = observed_at - self._timeline_anchor_observed_at
        if observed_delta < 0.1:
            return
        reference_delta = reference_timestamp - self._timeline_anchor_reference_timestamp
        instantaneous_rate = reference_delta / observed_delta
        if not np.isfinite(instantaneous_rate):
            return
        instantaneous_rate = min(max(instantaneous_rate, 0.9), 1.1)
        self._timeline_rate_estimate = (
            (self._timeline_rate_estimate * 0.8) + (instantaneous_rate * 0.2)
        )

    def _reset_reacquire_candidate(self) -> None:
        self._reacquire_candidate_timestamp = None
        self._reacquire_candidate_observed_at = None
        self._reacquire_start_observed_at = None
        self._reacquire_streak = 0

    def _anchor_age_seconds(self, observed_at: float) -> float | None:
        if self._timeline_anchor_observed_at is None:
            return None
        return max(0.0, observed_at - self._timeline_anchor_observed_at)

    def _select_runtime_result(
        self,
        result: MatchResult,
        *,
        frame: FeatureFrame,
    ) -> MatchResult:
        if result.valid:
            return self._select_runtime_valid_result(result, frame=frame)
        return self._select_runtime_invalid_result(result, frame=frame)

    def _select_runtime_valid_result(
        self,
        result: MatchResult,
        *,
        frame: FeatureFrame,
    ) -> MatchResult:
        if self._timeline_anchor_reference_timestamp is None or self._timeline_mode == "SEARCHING":
            self._update_timeline_anchor(result, observed_at=frame.observed_at)
            return result

        predicted_result = self._predict_match_from_anchor(frame)
        if predicted_result is None:
            self._update_timeline_anchor(result, observed_at=frame.observed_at)
            return result

        if (
            abs(result.reference_timestamp - predicted_result.reference_timestamp)
            <= self._clock_resync_tolerance_seconds
        ):
            self._update_timeline_anchor(result, observed_at=frame.observed_at)
            return result

        forward_delta = result.reference_timestamp - predicted_result.reference_timestamp
        required_confidence = max(
            self._fast_forward_confidence_floor,
            self._timeline_anchor_confidence - 0.1,
        )
        if (
            forward_delta > 0.0
            and forward_delta <= self._fast_forward_accept_tolerance_seconds
            and result.confidence >= required_confidence
        ):
            self._update_timeline_anchor(result, observed_at=frame.observed_at)
            return result

        if self._reacquire_candidate_confirmed(result, observed_at=frame.observed_at):
            self._update_timeline_anchor(result, observed_at=frame.observed_at)
            return result

        anchor_age = self._anchor_age_seconds(frame.observed_at)
        if anchor_age is not None and anchor_age <= self._timeline_hold_max_seconds:
            self._timeline_mode = "SOFT_HOLD"
            return predicted_result

        self._timeline_mode = "REACQUIRE"
        return self._invalidate(result)

    def _select_runtime_invalid_result(
        self,
        result: MatchResult,
        *,
        frame: FeatureFrame,
    ) -> MatchResult:
        predicted_result = self._predict_match_from_anchor(frame)
        anchor_age = self._anchor_age_seconds(frame.observed_at)
        if (
            predicted_result is not None
            and anchor_age is not None
            and anchor_age <= self._timeline_hold_max_seconds
        ):
            self._timeline_mode = "SOFT_HOLD"
            return predicted_result
        self._timeline_mode = "REACQUIRE"
        self._reset_reacquire_candidate()
        return result

    def _reacquire_candidate_confirmed(
        self,
        result: MatchResult,
        *,
        observed_at: float,
    ) -> bool:
        if self._timeline_mode not in {"REACQUIRE", "SOFT_HOLD", "LOCKED_CLOCK"}:
            return False
        if (
            self._reacquire_candidate_timestamp is not None
            and abs(result.reference_timestamp - self._reacquire_candidate_timestamp)
            <= self._reacquire_consistency_tolerance_seconds
        ):
            self._reacquire_streak += 1
        else:
            self._reacquire_streak = 1
            self._reacquire_start_observed_at = observed_at
        if self._reacquire_start_observed_at is None:
            self._reacquire_start_observed_at = observed_at
        self._reacquire_candidate_timestamp = result.reference_timestamp
        self._reacquire_candidate_observed_at = observed_at
        reacquire_duration = observed_at - self._reacquire_start_observed_at
        if self._reacquire_streak < self._reacquire_min_consecutive_matches:
            self._timeline_mode = "REACQUIRE"
            return False
        if reacquire_duration < self._reacquire_min_duration_seconds:
            self._timeline_mode = "REACQUIRE"
            return False
        self._timeline_mode = "REACQUIRE"
        return True

    def _predict_match_from_anchor(self, frame: FeatureFrame) -> MatchResult | None:
        if (
            self._timeline_anchor_reference_timestamp is None
            or self._timeline_anchor_observed_at is None
            or self._timeline_anchor_reference_frame is None
        ):
            return None
        elapsed_seconds = frame.observed_at - self._timeline_anchor_observed_at
        if elapsed_seconds <= 0 or elapsed_seconds > self._timeline_hold_max_seconds:
            return None
        return MatchResult(
            reference_frame=self._timeline_anchor_reference_frame,
            reference_timestamp=(
                self._timeline_anchor_reference_timestamp
                + (elapsed_seconds * self._timeline_rate_estimate)
            ),
            raw_distance=self._timeline_anchor_raw_distance,
            normalized_distance=self._timeline_anchor_normalized_distance,
            confidence=self._timeline_anchor_confidence,
            valid=True,
        )

    def _emit_slide_command(self, command: SlideCommand) -> None:
        now = monotonic()
        if (
            self._last_auto_emitted_slide_number == command.slide_number
            and self._last_auto_emitted_at is not None
            and now - self._last_auto_emitted_at < 2.0
        ):
            return
        self._last_auto_emitted_slide_number = command.slide_number
        self._last_auto_emitted_at = now
        self._last_slide_command = command
        self._publish_status()
        if self._manual_override_controller.is_active:
            self._suppress_slide_command(command)
            return
        if self._presentation_gateway is None:
            with self._metrics_lock:
                self._metrics.slide_triggers_sent += 1
            self._publish_status()
            return
        dropped = self._command_queue.put_drop_oldest(command)
        if dropped:
            self._logger.warning(
                "Presentation command queue full; dropped oldest queued command"
            )

    def _dispatch_commands(self) -> None:
        while not self._stop_requested.is_set() or self._command_queue.qsize() > 0:
            command = self._command_queue.get(timeout=0.1)
            if command is None:
                continue
            if self._manual_override_controller.is_active:
                self._suppress_slide_command(command)
                continue

            try:
                if self._presentation_gateway is not None:
                    self._presentation_gateway.send(command)
            except Exception:
                self._logger.exception(
                    "Failed to send slide command slide_number=%s section=%s",
                    command.slide_number,
                    command.section,
                )
                with self._metrics_lock:
                    self._metrics.osc_send_failures += 1
                continue

            with self._metrics_lock:
                self._metrics.slide_triggers_sent += 1
            self._publish_status()

    def _tracking_state(self) -> str | None:
        if self._feature_matcher is None:
            return None
        state_name = getattr(self._feature_matcher, "state_name", None)
        return state_name if isinstance(state_name, str) else None

    def _suppress_slide_command(self, command: SlideCommand) -> None:
        self._logger.info(
            "Manual override active; suppressed slide command slide_number=%s section=%s",
            command.slide_number,
            command.section,
        )
        with self._metrics_lock:
            self._metrics.manual_override_suppressed_triggers += 1

    def _snapshot_metrics(self) -> RuntimeMetrics:
        with self._metrics_lock:
            return RuntimeMetrics(
                chunks_received=self._metrics.chunks_received,
                chunks_dropped=self._metrics.chunks_dropped,
                silent_chunks=self._metrics.silent_chunks,
                non_voiced_chunks=self._metrics.non_voiced_chunks,
                clipped_chunks=self._metrics.clipped_chunks,
                feature_frames_processed=self._metrics.feature_frames_processed,
                invalid_inference_outputs=self._metrics.invalid_inference_outputs,
                low_confidence_matches=self._metrics.low_confidence_matches,
                accepted_matches=self._metrics.accepted_matches,
                slide_triggers_sent=self._metrics.slide_triggers_sent,
                manual_override_suppressed_triggers=(
                    self._metrics.manual_override_suppressed_triggers
                ),
                osc_send_failures=self._metrics.osc_send_failures,
                queue_high_water_mark=self._metrics.queue_high_water_mark,
            )

    def _matched_slide_command(self) -> SlideCommand | None:
        if self._last_match is None or not self._last_match.valid or self._slide_resolver is None:
            return None
        build_command = getattr(
            self._slide_resolver,
            "active_slide_command_for_timestamp",
            None,
        )
        if not callable(build_command):
            return None
        return build_command(
            self._last_match.reference_timestamp,
            confidence=self._last_match.confidence,
        )

    def _build_status_snapshot(self) -> RuntimeStatusSnapshot:
        controller_status = getattr(self._slide_resolver, "status", None)
        return RuntimeStatusSnapshot(
            device_name=self._device_name,
            metrics=self._snapshot_metrics(),
            rms=self._last_rms,
            peak=self._last_peak,
            queue_size=self._queue.qsize(),
            queue_capacity=self._queue.capacity,
            command_queue_size=self._command_queue.qsize(),
            command_queue_capacity=self._command_queue.capacity,
            manual_override_active=self._manual_override_controller.is_active,
            tracking_state=self._tracking_state(),
            last_match=self._last_match,
            last_slide_command=self._last_slide_command,
            matched_slide_command=self._matched_slide_command(),
            current_section_key=getattr(controller_status, "current_section_key", None),
            current_section_label=getattr(controller_status, "current_section_label", None),
            current_section_slide_number=getattr(
                controller_status, "current_slide_number", None
            ),
            recovery_active=bool(
                getattr(controller_status, "recovery_active", False)
            ),
            suggested_section_key=getattr(
                controller_status, "suggested_section_key", None
            ),
            suggested_section_label=getattr(
                controller_status, "suggested_section_label", None
            ),
            no_vocal_detected=(
                self._vocal_presence_detection_enabled
                and self._consecutive_non_voiced_chunks > 0
            ),
            alignment_hold_active=self._alignment_hold_active,
            alignment_hold_reason=self._alignment_hold_reason,
        )
