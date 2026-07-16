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
    parser = argparse.ArgumentParser(description="Build an offline reference profile from WAV.")
    parser.add_argument("--audio", required=True, help="Path to the reference WAV file.")
    parser.add_argument("--slides", required=True, help="Path to the slide cue JSON file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the reference profile.",
    )
    parser.add_argument(
        "--profile-name",
        help="Optional profile name. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--feature-extractor",
        choices=("simulated", "onnx"),
        default="simulated",
        help="Feature extractor used for offline profile generation.",
    )
    parser.add_argument(
        "--feature-model-path",
        help="ONNX model path when using --feature-extractor onnx.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4_096,
        help="Chunk size used during offline feature extraction.",
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


def main() -> None:
    from lyrics_aligner.application.reference_builder import (
        ReferenceBuilderConfig,
        ReferenceProfileBuilder,
    )

    args = _parse_args()
    profile_name = args.profile_name or Path(args.output).expanduser().name
    builder = ReferenceProfileBuilder(
        feature_extractor=_build_feature_extractor(args),
        config=ReferenceBuilderConfig(sample_rate=16_000, block_size=args.block_size),
    )
    profile = builder.build(
        audio_path=args.audio,
        slides_path=args.slides,
        profile_name=profile_name,
    )
    builder.save(profile, args.output)

    summary = {
        "output": str(Path(args.output).expanduser().resolve()),
        "profile_name": profile.name,
        "frames": int(profile.feature_matrix.shape[0]),
        "feature_size": int(profile.feature_matrix.shape[1]),
        "slides": len(profile.slides),
        "profile_version": profile.metadata["profile_version"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
