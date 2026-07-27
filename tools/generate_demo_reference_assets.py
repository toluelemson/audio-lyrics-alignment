from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic demo WAV and slide cues for ONNX "
            "reference-profile tests."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="demo_assets",
        help="Directory where the demo WAV and slides JSON will be written.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=32.0,
        help="Total demo audio duration in seconds.",
    )
    return parser.parse_args()


def _sine_segment(
    sample_rate: int,
    duration_seconds: float,
    frequency_hz: float,
    amplitude: float,
    phase_offset: float = 0.0,
) -> np.ndarray:
    sample_count = int(round(sample_rate * duration_seconds))
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = amplitude * np.sin(
        (2.0 * math.pi * frequency_hz * time) + phase_offset,
        dtype=np.float32,
    )
    return np.asarray(waveform, dtype=np.float32)


def _build_demo_waveform(sample_rate: int, duration_seconds: float) -> np.ndarray:
    section_count = 8
    segment_seconds = duration_seconds / section_count
    segments = [
        _sine_segment(sample_rate, segment_seconds, 220.00, 0.45, phase_offset=0.0),
        _sine_segment(sample_rate, segment_seconds, 246.94, 0.42, phase_offset=0.3),
        _sine_segment(sample_rate, segment_seconds, 261.63, 0.40, phase_offset=0.6),
        _sine_segment(sample_rate, segment_seconds, 293.66, 0.38, phase_offset=0.9),
        _sine_segment(sample_rate, segment_seconds, 329.63, 0.41, phase_offset=1.2),
        _sine_segment(sample_rate, segment_seconds, 349.23, 0.39, phase_offset=1.5),
        _sine_segment(sample_rate, segment_seconds, 392.00, 0.43, phase_offset=1.8),
        _sine_segment(sample_rate, segment_seconds, 440.00, 0.37, phase_offset=2.1),
    ]
    envelope = np.linspace(0.85, 1.0, num=sum(len(segment) for segment in segments))
    waveform = np.concatenate(segments).astype(np.float32)
    waveform = waveform[: envelope.shape[0]] * envelope.astype(np.float32)
    return np.asarray(waveform, dtype=np.float32)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32_767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _write_slides(path: Path, duration_seconds: float) -> None:
    section_count = 8
    interval = duration_seconds / section_count
    section_names = (
        "Intro",
        "Verse 1",
        "Pre-Chorus",
        "Chorus",
        "Verse 2",
        "Bridge",
        "Chorus 2",
        "Outro",
    )
    slides = [
        {
            "slide_number": index + 1,
            "section": name,
            "lyrics": f"Demo line {index + 1}",
            "reference_timestamp": round(interval * index, 3),
        }
        for index, name in enumerate(section_names)
    ]
    path.write_text(json.dumps(slides, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be greater than zero")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "onnx_demo_song.wav"
    slides_path = output_dir / "onnx_demo_slides.json"

    waveform = _build_demo_waveform(TARGET_SAMPLE_RATE, args.duration_seconds)
    _write_wav(audio_path, waveform, TARGET_SAMPLE_RATE)
    _write_slides(slides_path, args.duration_seconds)

    print(
        json.dumps(
            {
                "audio": str(audio_path),
                "duration_seconds": args.duration_seconds,
                "sample_rate": TARGET_SAMPLE_RATE,
                "slides": str(slides_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
