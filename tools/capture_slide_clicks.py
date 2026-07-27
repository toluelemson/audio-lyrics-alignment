from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture slide-change click timings in the terminal. "
            "Start the reference song, then press Enter for each slide change."
        ),
    )
    parser.add_argument(
        "--slides",
        required=True,
        help=(
            "Path to a slide script JSON file containing slide_number, section, "
            "and lyrics for each slide."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the captured slide cue JSON file with click_timestamp values.",
    )
    parser.add_argument(
        "--countdown-seconds",
        type=int,
        default=3,
        help="Countdown before capture starts.",
    )
    return parser.parse_args()


def _capture_timestamps(script_cues) -> tuple[float, ...]:
    print("Start the reference song now.", flush=True)
    print("Press Enter each time the next slide should appear.", flush=True)
    print("Press Ctrl-C to cancel.\n", flush=True)
    started_at = monotonic()
    captured: list[float] = []

    for index, cue in enumerate(script_cues, start=1):
        print(
            f"[{index}/{len(script_cues)}] Slide {cue.slide_number} | {cue.section}",
            flush=True,
        )
        first_line = cue.lyrics.splitlines()[0]
        print(f"  {first_line}", flush=True)
        input("  Press Enter at the slide change... ")
        captured.append(round(monotonic() - started_at, 3))
        print(f"  Captured at {captured[-1]:.3f}s\n", flush=True)
    return tuple(captured)


def main() -> None:
    from lyrics_aligner.application.click_capture import (
        build_captured_slide_cues,
        load_slide_script,
        write_captured_slide_cues,
    )

    args = _parse_args()
    script_cues = load_slide_script(args.slides)

    if args.countdown_seconds > 0:
        for seconds_remaining in range(args.countdown_seconds, 0, -1):
            print(f"Starting capture in {seconds_remaining}...", flush=True)
            sleep(1.0)

    click_timestamps = _capture_timestamps(script_cues)
    captured_cues = build_captured_slide_cues(script_cues, click_timestamps)
    write_captured_slide_cues(args.output, captured_cues)

    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "slides": len(captured_cues),
                "first_click_timestamp": captured_cues[0].click_timestamp,
                "last_click_timestamp": captured_cues[-1].click_timestamp,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
