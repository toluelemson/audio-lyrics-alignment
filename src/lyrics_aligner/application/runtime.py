from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from time import monotonic

import numpy as np

from lyrics_aligner.application.metrics import RuntimeMetrics
from lyrics_aligner.domain.models import AudioChunk, FeatureFrame, MatchResult, SlideCommand
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

    def qsize(self) -> int:
        with self._condition:
            return len(self._items)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    device_name: str
    metrics: RuntimeMetrics
    rms: float
    peak: float
    queue_size: int
    queue_capacity: int
    last_match: MatchResult | None
    last_slide_command: SlideCommand | None


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
    ) -> None:
        self._source = source
        self._queue = BoundedAudioQueue(queue_capacity)
        self._logger = logger
        self._feature_extractor = feature_extractor
        self._feature_matcher = feature_matcher
        self._slide_resolver = slide_resolver
        self._presentation_gateway = presentation_gateway
        self._diagnostics_interval_seconds = diagnostics_interval_seconds
        self._device_name = device_name
        self._silence_threshold_rms = silence_threshold_rms
        self._clipping_threshold_peak = clipping_threshold_peak
        self._metrics = RuntimeMetrics()
        self._metrics_lock = Lock()
        self._stop_requested = Event()
        self._last_rms = 0.0
        self._last_peak = 0.0
        self._last_match: MatchResult | None = None
        self._last_slide_command: SlideCommand | None = None

    def run(self) -> RuntimeReport:
        producer = Thread(target=self._produce, name="audio-producer", daemon=True)
        producer.start()

        next_report_at = monotonic() + self._diagnostics_interval_seconds
        try:
            while producer.is_alive() or self._queue.qsize() > 0:
                chunk = self._queue.get(timeout=0.1)
                if chunk is not None:
                    self._process(chunk)

                if monotonic() >= next_report_at:
                    self._log_diagnostics()
                    next_report_at = monotonic() + self._diagnostics_interval_seconds
        finally:
            self.stop()
            producer.join()

        self._log_diagnostics()
        return RuntimeReport(
            device_name=self._device_name,
            metrics=self._snapshot_metrics(),
            rms=self._last_rms,
            peak=self._last_peak,
            queue_size=self._queue.qsize(),
            queue_capacity=self._queue.capacity,
            last_match=self._last_match,
            last_slide_command=self._last_slide_command,
        )

    def stop(self) -> None:
        self._stop_requested.set()
        self._queue.close()
        stop_method = getattr(self._source, "stop", None)
        if callable(stop_method):
            stop_method()

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
        with self._metrics_lock:
            if self._last_rms <= self._silence_threshold_rms:
                self._metrics.silent_chunks += 1
            if self._last_peak >= self._clipping_threshold_peak:
                self._metrics.clipped_chunks += 1
        if self._feature_extractor is not None:
            self._record_feature_frames(self._feature_extractor.extract(chunk))

    def _log_diagnostics(self) -> None:
        metrics = self._snapshot_metrics()
        self._logger.info(
            "device=%s rms=%.2f peak=%.2f queue=%s/%s "
            "chunks_received=%s chunks_dropped=%s "
            "silent_chunks=%s clipped_chunks=%s feature_frames_processed=%s "
            "accepted_matches=%s low_confidence_matches=%s slide_triggers_sent=%s "
            "osc_send_failures=%s last_reference_timestamp=%s "
            "last_slide_number=%s last_confidence=%.2f",
            self._device_name,
            self._last_rms,
            self._last_peak,
            self._queue.qsize(),
            self._queue.capacity,
            metrics.chunks_received,
            metrics.chunks_dropped,
            metrics.silent_chunks,
            metrics.clipped_chunks,
            metrics.feature_frames_processed,
            metrics.accepted_matches,
            metrics.low_confidence_matches,
            metrics.slide_triggers_sent,
            metrics.osc_send_failures,
            (
                f"{self._last_match.reference_timestamp:.2f}"
                if self._last_match is not None
                else "none"
            ),
            (
                self._last_slide_command.slide_number
                if self._last_slide_command is not None
                else "none"
            ),
            self._last_match.confidence if self._last_match is not None else 0.0,
        )

    def _record_feature_frames(self, frames: list[FeatureFrame]) -> None:
        valid_frames = 0
        invalid_frames = 0
        for frame in frames:
            values = np.asarray(frame.values, dtype=np.float32)
            if values.size == 0 or not np.isfinite(values).all():
                invalid_frames += 1
                continue
            valid_frames += 1
            if self._feature_matcher is not None:
                self._record_match(self._feature_matcher.match(frame))

        with self._metrics_lock:
            self._metrics.feature_frames_processed += valid_frames
            self._metrics.invalid_inference_outputs += invalid_frames

    def _record_match(self, result: MatchResult) -> None:
        self._last_match = result
        with self._metrics_lock:
            if result.valid:
                self._metrics.accepted_matches += 1
            else:
                self._metrics.low_confidence_matches += 1
        if self._slide_resolver is not None:
            command = self._slide_resolver.resolve(result)
            if command is not None:
                self._emit_slide_command(command)

    def _emit_slide_command(self, command: SlideCommand) -> None:
        self._last_slide_command = command
        if self._presentation_gateway is None:
            with self._metrics_lock:
                self._metrics.slide_triggers_sent += 1
            return

        try:
            self._presentation_gateway.send(command)
        except Exception:
            self._logger.exception(
                "Failed to send slide command slide_number=%s section=%s",
                command.slide_number,
                command.section,
            )
            with self._metrics_lock:
                self._metrics.osc_send_failures += 1
            return

        with self._metrics_lock:
            self._metrics.slide_triggers_sent += 1

    def _snapshot_metrics(self) -> RuntimeMetrics:
        with self._metrics_lock:
            return RuntimeMetrics(
                chunks_received=self._metrics.chunks_received,
                chunks_dropped=self._metrics.chunks_dropped,
                silent_chunks=self._metrics.silent_chunks,
                clipped_chunks=self._metrics.clipped_chunks,
                feature_frames_processed=self._metrics.feature_frames_processed,
                invalid_inference_outputs=self._metrics.invalid_inference_outputs,
                low_confidence_matches=self._metrics.low_confidence_matches,
                accepted_matches=self._metrics.accepted_matches,
                slide_triggers_sent=self._metrics.slide_triggers_sent,
                osc_send_failures=self._metrics.osc_send_failures,
                queue_high_water_mark=self._metrics.queue_high_water_mark,
            )
