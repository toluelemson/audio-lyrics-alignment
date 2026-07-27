from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _print_help() -> None:
    print(
        "\n".join(
            [
                "Operator commands:",
                "  manual on       Enable manual override",
                "  manual off      Disable manual override",
                "  manual toggle   Toggle manual override",
                "  jump <number>   Jump to a grouped slide number",
                "  status          Print current runtime snapshot",
                "  help            Show this help",
                "  quit            Stop the runtime",
            ]
        ),
        flush=True,
    )


def _print_status(runtime) -> None:
    snapshot = runtime._build_status_snapshot()  # noqa: SLF001 - local operator tool
    print(
        json.dumps(
            {
                "tracking_state": snapshot.tracking_state,
                "manual_override_active": snapshot.manual_override_active,
                "last_match_reference_timestamp": (
                    None
                    if snapshot.last_match is None
                    else snapshot.last_match.reference_timestamp
                ),
                "last_slide_number": (
                    None
                    if snapshot.last_slide_command is None
                    else snapshot.last_slide_command.slide_number
                ),
                "queue_size": snapshot.queue_size,
                "command_queue_size": snapshot.command_queue_size,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    from lyrics_aligner.main import (
        _config_from_args,
        _log_runtime_health,
        _parse_args,
        _validate_config,
        build_runtime,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = _config_from_args(_parse_args())
    _validate_config(config)
    logger = logging.getLogger("operator-console")
    runtime = build_runtime(config, logger)

    result: dict[str, object] = {"report": None, "error": None}

    def run_runtime() -> None:
        try:
            result["report"] = runtime.run()
        except Exception as error:  # pragma: no cover - surfaced to operator
            result["error"] = error

    worker = Thread(target=run_runtime, name="runtime-worker", daemon=True)
    worker.start()

    print("Runtime started with operator controls.", flush=True)
    _print_help()

    try:
        while worker.is_alive():
            try:
                raw = input("> ").strip()
            except EOFError:
                runtime.stop()
                break
            if not raw:
                continue
            if raw == "help":
                _print_help()
                continue
            if raw == "status":
                _print_status(runtime)
                continue
            if raw == "quit":
                runtime.stop()
                break
            if raw == "manual on":
                runtime.enable_manual_override()
                print("Manual override enabled.", flush=True)
                continue
            if raw == "manual off":
                runtime.disable_manual_override()
                print("Manual override disabled.", flush=True)
                continue
            if raw == "manual toggle":
                active = runtime.toggle_manual_override()
                print(
                    f"Manual override {'enabled' if active else 'disabled'}.",
                    flush=True,
                )
                continue
            if raw.startswith("jump "):
                _, _, value = raw.partition(" ")
                try:
                    slide_number = int(value)
                except ValueError:
                    print("Slide number must be an integer.", flush=True)
                    continue
                try:
                    command = runtime.jump_to_slide(slide_number)
                except ValueError as error:
                    print(str(error), flush=True)
                    continue
                print(
                    f"Jumped to slide {command.slide_number} at "
                    f"{command.reference_timestamp:.2f}s.",
                    flush=True,
                )
                continue
            print("Unknown command. Type 'help' for available commands.", flush=True)
    finally:
        runtime.stop()
        worker.join()

    if result["error"] is not None:
        raise RuntimeError("runtime failed") from result["error"]  # pragma: no cover

    report = result["report"]
    if report is None:
        raise RuntimeError("runtime did not produce a report")
    _log_runtime_health(logger, config, report)


if __name__ == "__main__":
    main()
