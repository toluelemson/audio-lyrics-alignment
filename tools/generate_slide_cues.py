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
        description="Generate a slide cue JSON scaffold from plain lyrics text.",
    )
    parser.add_argument(
        "--lyrics",
        required=True,
        help="Path to a plain-text lyrics file with section headings such as '**Verse 1**'.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the generated slide cue JSON file.",
    )
    parser.add_argument(
        "--initial-timestamp",
        type=float,
        default=0.0,
        help="Reference timestamp assigned to the first generated slide.",
    )
    parser.add_argument(
        "--timestamp-step",
        type=float,
        default=20.0,
        help="Placeholder spacing in seconds between generated slides.",
    )
    return parser.parse_args()


def main() -> None:
    from lyrics_aligner.application.slide_template import (
        SlideTemplateConfig,
        build_slide_template,
    )

    args = _parse_args()
    lyrics_path = Path(args.lyrics).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    cues = build_slide_template(
        lyrics_path.read_text(encoding="utf-8"),
        SlideTemplateConfig(
            initial_timestamp=args.initial_timestamp,
            timestamp_step=args.timestamp_step,
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            [
                {
                    "slide_number": cue.slide_number,
                    "section": cue.section,
                    "lyrics": cue.lyrics,
                    "click_timestamp": cue.reference_timestamp,
                }
                for cue in cues
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "slides": len(cues),
                "first_section": cues[0].section,
                "last_section": cues[-1].section,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
