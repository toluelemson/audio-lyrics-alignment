from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from time import monotonic

import numpy as np

from lyrics_aligner.application.metrics import RuntimeMetrics
from lyrics_aligner.domain.models import AudioChunk
from lyrics_aligner.ports.audio_source import AudioSource


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


class AudioIngestionRuntime:
    """Run audio capture and queue consumption with periodic diagnostics."""

    def __init__(
        self,
        source: AudioSource,
        queue_capacity: int,
        logger: logging.Logger,
        diagnostics_interval_seconds: float = 0.5,
        device_name: str = "SimulatedAudioSource",
        silence_threshold_rms: float = 0.01,
        clipping_threshold_peak: float = 0.99,
    ) -> None:
        self._source = source
        self._queue = BoundedAudioQueue(queue_capacity)
        self._logger = logger
        self._diagnostics_interval_seconds = diagnostics_interval_seconds
        self._device_name = device_name
        self._silence_threshold_rms = silence_threshold_rms
        self._clipping_threshold_peak = clipping_threshold_peak
        self._metrics = RuntimeMetrics()
        self._metrics_lock = Lock()
        self._stop_requested = Event()
        self._last_rms = 0.0
        self._last_peak = 0.0

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

    def _log_diagnostics(self) -> None:
        metrics = self._snapshot_metrics()
        self._logger.info(
            "device=%s rms=%.2f peak=%.2f queue=%s/%s "
            "chunks_received=%s chunks_dropped=%s "
            "silent_chunks=%s clipped_chunks=%s",
            self._device_name,
            self._last_rms,
            self._last_peak,
            self._queue.qsize(),
            self._queue.capacity,
            metrics.chunks_received,
            metrics.chunks_dropped,
            metrics.silent_chunks,
            metrics.clipped_chunks,
        )

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
