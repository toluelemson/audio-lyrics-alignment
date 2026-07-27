import json
from pathlib import Path

import numpy as np
import pytest

from lyrics_aligner.application.offline_alignment import (
    align_live_recording,
    baseline_dtw,
    build_timestamp_error_report,
    confidence_from_cost,
    cosine_distance,
    load_timestamp_expectations,
    normalized_euclidean_distance,
    normalized_path_cost,
    pairwise_distance_matrix,
    subsequence_dtw,
)
from lyrics_aligner.domain.models import FeatureFrame, ReferenceProfile


def _frame(values: list[float], observed_at: float) -> FeatureFrame:
    return FeatureFrame(
        values=np.array(values, dtype=np.float32),
        observed_at=observed_at,
        frame_duration_seconds=0.25,
    )


def _profile(values: list[list[float]]) -> ReferenceProfile:
    frames = tuple(_frame(value, index * 0.25) for index, value in enumerate(values))
    return ReferenceProfile(name="song-a", frames=frames, metadata={})


def test_cosine_distance_is_zero_for_identical_vectors() -> None:
    value = cosine_distance(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32),
    )

    assert value == pytest.approx(0.0)


def test_normalized_euclidean_distance_is_scaled_by_dimension() -> None:
    value = normalized_euclidean_distance(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    )

    assert value == pytest.approx(1.0)


def test_baseline_dtw_aligns_identical_sequences() -> None:
    distance_matrix = pairwise_distance_matrix(
        np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        metric="euclidean",
    )

    total_cost, path = baseline_dtw(distance_matrix)

    assert total_cost == pytest.approx(0.0)
    assert path == [(0, 0), (1, 1), (2, 2)]


def test_subsequence_dtw_finds_best_reference_window() -> None:
    distance_matrix = pairwise_distance_matrix(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        metric="euclidean",
    )

    total_cost, path = subsequence_dtw(distance_matrix)

    assert total_cost == pytest.approx(0.0)
    assert path == [(0, 1), (1, 2)]


def test_align_live_recording_returns_trace_and_confidence() -> None:
    live_frames = (
        _frame([1.0, 0.0], 0.0),
        _frame([0.0, 1.0], 0.25),
    )
    profile = _profile([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])

    result = align_live_recording(
        live_frames,
        profile,
        method="subsequence",
        metric="cosine",
    )

    assert result.trace[0].reference_frame_index == 1
    assert result.trace[-1].reference_frame_index == 2
    assert result.confidence > 0.9


def test_timestamp_error_report_is_computed_from_expectations() -> None:
    live_frames = (
        _frame([1.0, 0.0], 0.0),
        _frame([0.0, 1.0], 0.25),
    )
    profile = _profile([[1.0, 0.0], [0.0, 1.0]])

    result = align_live_recording(
        live_frames,
        profile,
        method="baseline",
        metric="cosine",
        expectations=load_timestamp_expectations_from_data(
            [
                {"live_timestamp": 0.0, "reference_timestamp": 0.0},
                {"live_timestamp": 0.25, "reference_timestamp": 0.25},
            ]
        ),
    )

    assert result.timestamp_error_report is not None
    assert result.timestamp_error_report.median_error_seconds == pytest.approx(0.0)
    assert result.timestamp_error_report.p95_error_seconds == pytest.approx(0.0)


def test_load_timestamp_expectations_reads_fixture_file(tmp_path: Path) -> None:
    fixture_path = tmp_path / "expected.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"live_timestamp": 0.0, "reference_timestamp": 0.1},
                {"live_timestamp": 1.0, "reference_timestamp": 1.2},
            ]
        ),
        encoding="utf-8",
    )

    expectations = load_timestamp_expectations(str(fixture_path))

    assert len(expectations) == 2
    assert expectations[1].reference_timestamp == pytest.approx(1.2)


def test_build_timestamp_error_report_uses_nearest_trace_point() -> None:
    report = build_timestamp_error_report(
        load_timestamp_expectations_from_data(
            [
                {"live_timestamp": 0.24, "reference_timestamp": 0.25},
            ]
        ),
        (
            align_point(0, 0.0, 0, 0.0, 0.1),
            align_point(1, 0.25, 1, 0.25, 0.1),
        ),
    )

    assert report.median_error_seconds == pytest.approx(0.0)
    assert report.compared_points == 1


def test_confidence_decreases_as_path_cost_increases() -> None:
    assert confidence_from_cost(0.0) == pytest.approx(1.0)
    assert confidence_from_cost(1.0) < confidence_from_cost(0.5)


def test_normalized_path_cost_divides_by_path_length() -> None:
    assert normalized_path_cost(2.0, 4) == pytest.approx(0.5)


def load_timestamp_expectations_from_data(
    data: list[dict[str, float]],
) -> tuple[object, ...]:
    temp_path = Path(__file__).with_name("_tmp_expectations.json")
    temp_path.write_text(json.dumps(data), encoding="utf-8")
    try:
        return tuple(load_timestamp_expectations(str(temp_path)))
    finally:
        temp_path.unlink(missing_ok=True)


def align_point(
    live_frame_index: int,
    live_timestamp: float,
    reference_frame_index: int,
    reference_timestamp: float,
    distance: float,
):
    from lyrics_aligner.application.offline_alignment import AlignmentTracePoint

    return AlignmentTracePoint(
        live_frame_index=live_frame_index,
        live_timestamp=live_timestamp,
        reference_frame_index=reference_frame_index,
        reference_timestamp=reference_timestamp,
        distance=distance,
    )
