from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline alignment against a reference profile.",
    )
    parser.add_argument("--live-audio", required=True, help="Path to the recorded live WAV file.")
    parser.add_argument(
        "--reference-profile",
        required=True,
        help="Path to the prepared reference profile directory.",
    )
    parser.add_argument(
        "--feature-extractor",
        choices=("simulated", "onnx"),
        default="simulated",
        help="Feature extractor used for offline alignment.",
    )
    parser.add_argument(
        "--feature-model-path",
        help="ONNX model path when using --feature-extractor onnx.",
    )
    parser.add_argument(
        "--method",
        choices=("baseline", "subsequence"),
        default="subsequence",
        help="Offline alignment algorithm.",
    )
    parser.add_argument(
        "--metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Frame-to-frame distance metric.",
    )
    parser.add_argument(
        "--expected-alignments",
        help="Optional JSON file with live/reference timestamp expectations for error reporting.",
    )
    return parser.parse_args()


def _build_feature_extractor(args: argparse.Namespace) -> object:
    from lyrics_aligner.adapters.features import (
        OnnxFeatureExtractor,
        OnnxFeatureExtractorConfig,
        SimulatedFeatureExtractor,
        SimulatedFeatureExtractorConfig,
    )

    if args.feature_extractor == "simulated":
        return SimulatedFeatureExtractor(
            SimulatedFeatureExtractorConfig(sample_rate=16_000)
        )
    if args.feature_model_path is None:
        raise ValueError("--feature-model-path is required when --feature-extractor onnx")
    return OnnxFeatureExtractor(
        OnnxFeatureExtractorConfig(
            model_path=args.feature_model_path,
            sample_rate=16_000,
        )
    )


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes:02d}:{remainder:05.2f}"


def main() -> None:
    from lyrics_aligner.adapters.reference_profiles import (
        FilesystemReferenceProfileRepository,
    )
    from lyrics_aligner.application.offline_alignment import (
        align_live_recording,
        extract_live_feature_frames,
        load_timestamp_expectations,
    )

    args = _parse_args()
    profile = FilesystemReferenceProfileRepository().load(args.reference_profile)
    extractor = _build_feature_extractor(args)
    live_frames = extract_live_feature_frames(args.live_audio, extractor)
    expectations = (
        load_timestamp_expectations(args.expected_alignments)
        if args.expected_alignments is not None
        else ()
    )
    result = align_live_recording(
        live_frames,
        profile,
        method=args.method,
        metric=args.metric,
        expectations=expectations,
    )

    summary = {
        "method": result.method,
        "metric": args.metric,
        "live_frames": len(live_frames),
        "reference_frames": len(profile.frames),
        "normalized_path_cost": result.normalized_path_cost,
        "confidence": result.confidence,
        "start_reference_timestamp": result.start_reference_timestamp,
        "end_reference_timestamp": result.end_reference_timestamp,
    }
    if result.timestamp_error_report is not None:
        summary["timestamp_error_report"] = {
            "median_error_seconds": result.timestamp_error_report.median_error_seconds,
            "p95_error_seconds": result.timestamp_error_report.p95_error_seconds,
            "max_error_seconds": result.timestamp_error_report.max_error_seconds,
            "compared_points": result.timestamp_error_report.compared_points,
        }

    last_point = result.trace[-1]
    print(
        f"Live {_format_timestamp(last_point.live_timestamp)} -> "
        f"Reference {_format_timestamp(last_point.reference_timestamp)}"
    )
    print(f"Confidence: {result.confidence:.2f}")
    if result.timestamp_error_report is not None:
        print(
            "Timestamp error: "
            f"{result.timestamp_error_report.median_error_seconds:.2f} seconds (median)"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
