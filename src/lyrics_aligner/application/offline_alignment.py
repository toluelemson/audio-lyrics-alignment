from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lyrics_aligner.application.reference_builder import (
    TARGET_SAMPLE_RATE,
    extract_reference_frames,
    load_wav_mono,
    resample_audio,
)
from lyrics_aligner.domain.models import FeatureFrame, ReferenceProfile
from lyrics_aligner.ports.feature_extractor import FeatureExtractor


@dataclass(frozen=True, slots=True)
class AlignmentTracePoint:
    live_frame_index: int
    live_timestamp: float
    reference_frame_index: int
    reference_timestamp: float
    distance: float


@dataclass(frozen=True, slots=True)
class TimestampExpectation:
    live_timestamp: float
    reference_timestamp: float


@dataclass(frozen=True, slots=True)
class TimestampErrorReport:
    median_error_seconds: float
    p95_error_seconds: float
    max_error_seconds: float
    compared_points: int


@dataclass(frozen=True, slots=True)
class OfflineAlignmentResult:
    method: str
    normalized_path_cost: float
    confidence: float
    trace: tuple[AlignmentTracePoint, ...]
    start_reference_timestamp: float
    end_reference_timestamp: float
    timestamp_error_report: TimestampErrorReport | None = None


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def normalized_euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("vectors must have matching shape")
    if left.size == 0:
        return 0.0
    distance = float(np.linalg.norm(left - right))
    dimension_scale = float(np.sqrt(float(left.size)))
    return distance / dimension_scale


def pairwise_distance_matrix(
    live_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    if live_matrix.ndim != 2 or reference_matrix.ndim != 2:
        raise ValueError("live and reference feature matrices must both be 2D")
    if live_matrix.shape[1] != reference_matrix.shape[1]:
        raise ValueError("live and reference feature dimensions must match")

    distances = np.zeros((live_matrix.shape[0], reference_matrix.shape[0]), dtype=np.float32)
    for live_index, live_row in enumerate(live_matrix):
        for reference_index, reference_row in enumerate(reference_matrix):
            if metric == "cosine":
                distances[live_index, reference_index] = cosine_distance(live_row, reference_row)
            elif metric == "euclidean":
                distances[live_index, reference_index] = normalized_euclidean_distance(
                    live_row,
                    reference_row,
                )
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    return distances


def baseline_dtw(distance_matrix: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    rows, columns = distance_matrix.shape
    costs = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            local_distance = float(distance_matrix[row - 1, column - 1])
            costs[row, column] = local_distance + min(
                costs[row - 1, column],
                costs[row, column - 1],
                costs[row - 1, column - 1],
            )

    return float(costs[rows, columns]), _backtrack_path(costs, rows, columns)


def subsequence_dtw(distance_matrix: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    rows, columns = distance_matrix.shape
    costs = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    costs[0, :] = 0.0

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            local_distance = float(distance_matrix[row - 1, column - 1])
            costs[row, column] = local_distance + min(
                costs[row - 1, column],
                costs[row, column - 1],
                costs[row - 1, column - 1],
            )

    end_column = int(np.argmin(costs[rows, 1:])) + 1
    total_cost = float(costs[rows, end_column])
    return total_cost, _backtrack_path_until_first_row(costs, rows, end_column)


def _backtrack_path(costs: np.ndarray, row: int, column: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    while row > 0 and column > 0:
        path.append((row - 1, column - 1))
        options = (
            (costs[row - 1, column], row - 1, column),
            (costs[row, column - 1], row, column - 1),
            (costs[row - 1, column - 1], row - 1, column - 1),
        )
        _, row, column = min(options, key=lambda item: item[0])
    path.reverse()
    return path


def _backtrack_path_until_first_row(
    costs: np.ndarray,
    row: int,
    column: int,
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    while row > 0 and column > 0:
        path.append((row - 1, column - 1))
        options = (
            (costs[row - 1, column], row - 1, column),
            (costs[row, column - 1], row, column - 1),
            (costs[row - 1, column - 1], row - 1, column - 1),
        )
        _, next_row, next_column = min(options, key=lambda item: item[0])
        row, column = next_row, next_column
        if row == 0:
            break
    path.reverse()
    return path


def normalized_path_cost(total_cost: float, path_length: int) -> float:
    if path_length <= 0:
        raise ValueError("path_length must be greater than zero")
    return total_cost / path_length


def confidence_from_cost(path_cost: float) -> float:
    return 1.0 / (1.0 + max(0.0, path_cost))


def align_live_recording(
    live_frames: tuple[FeatureFrame, ...],
    reference_profile: ReferenceProfile,
    *,
    method: str,
    metric: str,
    expectations: tuple[TimestampExpectation, ...] = (),
) -> OfflineAlignmentResult:
    if not live_frames:
        raise ValueError("live_frames must not be empty")
    if not reference_profile.frames:
        raise ValueError("reference_profile.frames must not be empty")

    live_matrix = np.stack(
        [np.asarray(frame.values, dtype=np.float32) for frame in live_frames],
        axis=0,
    )
    reference_matrix = np.stack(
        [np.asarray(frame.values, dtype=np.float32) for frame in reference_profile.frames],
        axis=0,
    )
    distance_matrix = pairwise_distance_matrix(live_matrix, reference_matrix, metric=metric)
    if method == "baseline":
        total_cost, path = baseline_dtw(distance_matrix)
    elif method == "subsequence":
        total_cost, path = subsequence_dtw(distance_matrix)
    else:
        raise ValueError(f"Unsupported alignment method: {method}")

    path_cost = normalized_path_cost(total_cost, len(path))
    confidence = confidence_from_cost(path_cost)
    trace = tuple(
        AlignmentTracePoint(
            live_frame_index=live_index,
            live_timestamp=float(live_frames[live_index].observed_at),
            reference_frame_index=reference_index,
            reference_timestamp=float(reference_profile.frames[reference_index].observed_at),
            distance=float(distance_matrix[live_index, reference_index]),
        )
        for live_index, reference_index in path
    )
    error_report = (
        build_timestamp_error_report(expectations, trace) if expectations else None
    )
    return OfflineAlignmentResult(
        method=method,
        normalized_path_cost=path_cost,
        confidence=confidence,
        trace=trace,
        start_reference_timestamp=trace[0].reference_timestamp,
        end_reference_timestamp=trace[-1].reference_timestamp,
        timestamp_error_report=error_report,
    )


def extract_live_feature_frames(
    audio_path: str,
    feature_extractor: FeatureExtractor,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    block_size: int = 4_096,
) -> tuple[FeatureFrame, ...]:
    samples, source_sample_rate = load_wav_mono(audio_path)
    if source_sample_rate != sample_rate:
        samples = resample_audio(samples, source_sample_rate, sample_rate)
    frames = extract_reference_frames(
        samples=samples,
        sample_rate=sample_rate,
        block_size=block_size,
        feature_extractor=feature_extractor,
    )
    if not frames:
        raise ValueError("feature extraction produced no live frames")
    return tuple(frames)


def load_timestamp_expectations(path: str) -> tuple[TimestampExpectation, ...]:
    content = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError("Expected timestamp expectations to be a JSON list")
    expectations: list[TimestampExpectation] = []
    for entry in content:
        if not isinstance(entry, dict):
            raise ValueError("Each timestamp expectation must be a JSON object")
        live_timestamp = entry.get("live_timestamp")
        reference_timestamp = entry.get("reference_timestamp")
        if not isinstance(live_timestamp, (int, float)) or not isinstance(
            reference_timestamp,
            (int, float),
        ):
            raise ValueError("Timestamp expectations must contain numeric timestamps")
        expectations.append(
            TimestampExpectation(
                live_timestamp=float(live_timestamp),
                reference_timestamp=float(reference_timestamp),
            )
        )
    return tuple(expectations)


def build_timestamp_error_report(
    expectations: tuple[TimestampExpectation, ...],
    trace: tuple[AlignmentTracePoint, ...],
) -> TimestampErrorReport:
    if not expectations:
        raise ValueError("expectations must not be empty")
    if not trace:
        raise ValueError("trace must not be empty")

    errors = np.array(
        [
            abs(
                _reference_timestamp_for_live_time(
                    expectation.live_timestamp,
                    trace,
                )
                - expectation.reference_timestamp
            )
            for expectation in expectations
        ],
        dtype=np.float32,
    )
    return TimestampErrorReport(
        median_error_seconds=float(np.median(errors)),
        p95_error_seconds=float(np.percentile(errors, 95)),
        max_error_seconds=float(np.max(errors)),
        compared_points=int(errors.size),
    )


def _reference_timestamp_for_live_time(
    live_timestamp: float,
    trace: tuple[AlignmentTracePoint, ...],
) -> float:
    best = min(
        trace,
        key=lambda point: abs(point.live_timestamp - live_timestamp),
    )
    return best.reference_timestamp
