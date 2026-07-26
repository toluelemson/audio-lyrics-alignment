from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Runtime Control</title>
  <style>
    :root {
      --bg: #f3efe7;
      --panel: #fffaf2;
      --ink: #191714;
      --muted: #6c655d;
      --line: #d7c8b5;
      --accent: #a34a27;
      --accent-2: #245a3c;
      --warn: #8d2424;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7df 0, transparent 26rem),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }
    .shell {
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.5rem);
      line-height: 0.95;
    }
    .sub {
      color: var(--muted);
      max-width: 44rem;
      line-height: 1.5;
      margin-bottom: 22px;
    }
    .grid {
      display: grid;
      grid-template-columns: 0.95fr 1.05fr;
      gap: 20px;
    }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(75, 53, 29, 0.08);
    }
    .status {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.6);
    }
    .label {
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.72rem;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .value {
      font-size: 1.1rem;
      font-weight: 700;
    }
    form {
      display: grid;
      gap: 14px;
    }
    .field {
      display: grid;
      gap: 6px;
    }
    .field label {
      color: var(--muted);
      font-size: 0.9rem;
    }
    input, select, button, textarea {
      font: inherit;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.78);
      color: var(--ink);
    }
    .buttons, .operator {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 12px 16px;
      cursor: pointer;
    }
    .primary { background: var(--accent); color: #fff7ef; }
    .secondary { background: #eadfce; color: var(--ink); }
    .success { background: var(--accent-2); color: #f3fff7; }
    .danger { background: var(--warn); color: #fff1f1; }
    .banner {
      min-height: 1.4rem;
      margin-top: 10px;
      color: var(--accent-2);
    }
    .banner.error { color: var(--warn); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.88rem;
      line-height: 1.45;
    }
    .logs {
      max-height: 28rem;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.66);
    }
    .inline {
      display: grid;
      grid-template-columns: 1fr 120px;
      gap: 10px;
      align-items: end;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .status { grid-template-columns: 1fr; }
      .inline { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <h1>Runtime Control UI</h1>
    <p class="sub">Start the one-song runtime from the browser, monitor live alignment state, and use manual operator controls without the terminal.</p>
    <div class="grid">
      <section class="card">
        <div class="status">
          <div class="stat"><div class="label">Runtime</div><div class="value" id="runtimeState">stopped</div></div>
          <div class="stat"><div class="label">Tracking</div><div class="value" id="trackingState">idle</div></div>
          <div class="stat"><div class="label">Position</div><div class="value" id="positionState">none</div></div>
          <div class="stat"><div class="label">Slide</div><div class="value" id="slideState">none</div></div>
          <div class="stat"><div class="label">Confidence</div><div class="value" id="confidenceState">0.00</div></div>
          <div class="stat"><div class="label">Manual</div><div class="value" id="manualState">auto</div></div>
        </div>

        <form id="startForm">
          <div class="field">
            <label for="device">Input Device</label>
            <select id="device" name="device"></select>
          </div>
          <div class="field">
            <label for="referenceProfilePath">Reference Profile Path</label>
            <input id="referenceProfilePath" name="referenceProfilePath" value="profiles/amazing-grace">
          </div>
          <div class="field">
            <label for="featureModelPath">Feature Model Path</label>
            <input id="featureModelPath" name="featureModelPath" value="models/wav2vec2-base.onnx">
          </div>
          <div class="field">
            <label for="silenceThreshold">Silence Threshold RMS</label>
            <input id="silenceThreshold" name="silenceThreshold" type="number" step="0.001" value="0.005">
          </div>
          <div class="field">
            <label for="matchDebugLogging">Match Debug Logging</label>
            <select id="matchDebugLogging" name="matchDebugLogging">
              <option value="false" selected>Off</option>
              <option value="true">On</option>
            </select>
          </div>
          <div class="buttons">
            <button class="primary" type="submit">Start Runtime</button>
            <button class="danger" type="button" id="stopButton">Stop Runtime</button>
            <button class="secondary" type="button" id="refreshButton">Refresh Status</button>
          </div>
        </form>

        <div class="operator" style="margin-top:18px">
          <button class="secondary" type="button" id="manualOnButton">Manual On</button>
          <button class="secondary" type="button" id="manualOffButton">Manual Off</button>
          <button class="secondary" type="button" id="manualToggleButton">Manual Toggle</button>
        </div>

        <div class="inline" style="margin-top:14px">
          <div class="field">
            <label for="jumpSlide">Jump To Slide</label>
            <input id="jumpSlide" name="jumpSlide" type="number" min="1" placeholder="3">
          </div>
          <button class="success" type="button" id="jumpButton">Jump</button>
        </div>

        <div class="banner" id="banner"></div>
      </section>

      <section class="card">
        <div class="label" style="margin-bottom:10px">Latest Snapshot</div>
        <pre id="snapshotView">Waiting for runtime status.</pre>
        <div class="label" style="margin:18px 0 10px">Recent Logs</div>
        <div class="logs"><pre id="logsView">No logs yet.</pre></div>
      </section>
    </div>
  </div>
  <script>
    const state = { payload: null };
    const runtimeState = document.getElementById("runtimeState");
    const trackingState = document.getElementById("trackingState");
    const positionState = document.getElementById("positionState");
    const slideState = document.getElementById("slideState");
    const confidenceState = document.getElementById("confidenceState");
    const manualState = document.getElementById("manualState");
    const device = document.getElementById("device");
    const banner = document.getElementById("banner");
    const snapshotView = document.getElementById("snapshotView");
    const logsView = document.getElementById("logsView");
    const stopButton = document.getElementById("stopButton");
    const refreshButton = document.getElementById("refreshButton");
    const manualOnButton = document.getElementById("manualOnButton");
    const manualOffButton = document.getElementById("manualOffButton");
    const manualToggleButton = document.getElementById("manualToggleButton");
    const jumpButton = document.getElementById("jumpButton");

    function setBanner(message, type = "") {
      banner.textContent = message || "";
      banner.className = type ? `banner ${type}` : "banner";
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ error: "Request failed" }));
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return response.json();
    }

    function render() {
      const payload = state.payload;
      if (!payload) return;
      const snapshot = payload.snapshot;
      runtimeState.textContent = payload.running ? "running" : "stopped";
      trackingState.textContent = snapshot ? snapshot.tracking_state || "idle" : "idle";
      positionState.textContent = snapshot && snapshot.last_match
        ? `${snapshot.last_match.reference_timestamp.toFixed(2)}s`
        : "none";
      slideState.textContent = snapshot && snapshot.last_slide_command
        ? String(snapshot.last_slide_command.slide_number)
        : "none";
      confidenceState.textContent = snapshot && snapshot.last_match
        ? snapshot.last_match.confidence.toFixed(2)
        : "0.00";
      manualState.textContent = snapshot && snapshot.manual_override_active ? "manual" : "auto";
      snapshotView.textContent = JSON.stringify(snapshot, null, 2);
      logsView.textContent = (payload.logs || []).join("\\n") || "No logs yet.";
    }

    async function loadStatus() {
      state.payload = await api("/api/status");
      render();
    }

    async function loadDevices() {
      const payload = await api("/api/devices");
      device.innerHTML = "";
      payload.devices.forEach((item) => {
        const option = document.createElement("option");
        option.value = String(item.id);
        option.textContent = `${item.id}: ${item.name}`;
        if (item.default) option.selected = true;
        device.appendChild(option);
      });
    }

    document.getElementById("startForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        state.payload = await api("/api/start", {
          method: "POST",
          body: JSON.stringify({
            input_device: Number(device.value),
            reference_profile_path: document.getElementById("referenceProfilePath").value,
            feature_model_path: document.getElementById("featureModelPath").value,
            silence_threshold_rms: Number(document.getElementById("silenceThreshold").value),
            match_debug_logging: document.getElementById("matchDebugLogging").value === "true",
          }),
        });
        setBanner("Runtime started.");
        render();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    stopButton.addEventListener("click", async () => {
      try {
        state.payload = await api("/api/stop", { method: "POST", body: "{}" });
        setBanner("Runtime stopped.");
        render();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    refreshButton.addEventListener("click", () => {
      loadStatus().catch((error) => setBanner(error.message, "error"));
    });

    manualOnButton.addEventListener("click", async () => {
      try {
        state.payload = await api("/api/manual/on", { method: "POST", body: "{}" });
        setBanner("Manual override enabled.");
        render();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    manualOffButton.addEventListener("click", async () => {
      try {
        state.payload = await api("/api/manual/off", { method: "POST", body: "{}" });
        setBanner("Manual override disabled.");
        render();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    manualToggleButton.addEventListener("click", async () => {
      try {
        state.payload = await api("/api/manual/toggle", { method: "POST", body: "{}" });
        setBanner("Manual override toggled.");
        render();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    jumpButton.addEventListener("click", async () => {
      const value = Number(document.getElementById("jumpSlide").value);
      if (!Number.isInteger(value) || value <= 0) {
        setBanner("Jump slide must be a positive integer.", "error");
        return;
      }
      try {
        state.payload = await api("/api/jump", {
          method: "POST",
          body: JSON.stringify({ slide_number: value }),
        });
        setBanner(`Jumped to slide ${value}.`);
        render();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    loadDevices()
      .then(loadStatus)
      .then(render)
      .catch((error) => setBanner(error.message, "error"));
    setInterval(() => {
      loadStatus().catch(() => {});
    }, 750);
  </script>
</body>
</html>
"""


@dataclass(frozen=True, slots=True)
class InputDeviceInfo:
    id: int
    name: str
    default: bool


class StatusCollector:
    def __init__(self) -> None:
        self._snapshot = None
        self._lock = Lock()

    def update(self, snapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def snapshot(self):
        with self._lock:
            return self._snapshot


class LogBuffer(logging.Handler):
    def __init__(self, limit: int = 200) -> None:
        super().__init__()
        self._limit = limit
        self._messages: list[str] = []
        self._lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        with self._lock:
            self._messages.append(message)
            if len(self._messages) > self._limit:
                self._messages = self._messages[-self._limit :]

    def messages(self) -> list[str]:
        with self._lock:
            return list(self._messages)


class RuntimeManager:
    def __init__(self) -> None:
        self._runtime = None
        self._thread: Thread | None = None
        self._report = None
        self._error: str | None = None
        self._lock = Lock()
        self.status_collector = StatusCollector()
        self.log_buffer = LogBuffer()
        self.log_buffer.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        self.logger = logging.getLogger("runtime-control-web")
        self.logger.setLevel(logging.INFO)
        if not any(isinstance(handler, LogBuffer) for handler in self.logger.handlers):
            self.logger.addHandler(self.log_buffer)

    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self, config) -> None:
        from lyrics_aligner.application.runtime import ManualOverrideController
        from lyrics_aligner.main import build_runtime

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("runtime is already running")
            self._report = None
            self._error = None
            runtime = build_runtime(
                config,
                self.logger,
                manual_override_controller=ManualOverrideController(),
                status_observer=self.status_collector,
            )
            self._runtime = runtime

            def run_runtime() -> None:
                try:
                    report = runtime.run()
                    with self._lock:
                        self._report = report
                except Exception as error:  # pragma: no cover
                    self.logger.exception("Runtime failed")
                    with self._lock:
                        self._error = str(error)

            self._thread = Thread(target=run_runtime, name="runtime-web", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            runtime = self._runtime
            thread = self._thread
        if runtime is not None:
            runtime.stop()
        if thread is not None:
            thread.join(timeout=5.0)

    def runtime(self):
        with self._lock:
            return self._runtime

    def report(self):
        with self._lock:
            return self._report

    def error(self) -> str | None:
        with self._lock:
            return self._error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a simple browser UI for running the operator runtime.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


def _input_devices() -> list[InputDeviceInfo]:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except Exception:
        return []
    devices = sd.query_devices()
    default_input_index = sd.default.device[0]
    output: list[InputDeviceInfo] = []
    for index, raw in enumerate(devices):
        max_input_channels = int(raw.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        output.append(
            InputDeviceInfo(
                id=index,
                name=str(raw.get("name", f"Device {index}")),
                default=index == default_input_index,
            )
        )
    return output


def _snapshot_to_dict(snapshot) -> dict[str, object] | None:
    if snapshot is None:
        return None
    payload = asdict(snapshot)
    return payload


def main() -> None:
    from lyrics_aligner.config import AppConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    manager = RuntimeManager()

    class Handler(BaseHTTPRequestHandler):
        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _status_payload(self) -> dict[str, object]:
            snapshot = manager.status_collector.snapshot()
            return {
                "running": manager.running(),
                "snapshot": _snapshot_to_dict(snapshot),
                "report": None if manager.report() is None else asdict(manager.report()),
                "error": manager.error(),
                "logs": manager.log_buffer.messages()[-80:],
            }

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/status":
                self._write_json(self._status_payload(), HTTPStatus.OK)
                return
            if parsed.path == "/api/devices":
                self._write_json(
                    {
                        "devices": [asdict(device) for device in _input_devices()],
                    },
                    HTTPStatus.OK,
                )
                return
            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/start":
                    payload = self._read_json()
                    config = replace(
                        AppConfig.from_env(),
                        audio_source="microphone",
                        input_device=int(payload.get("input_device", 0)),
                        reference_profile_path=str(payload.get("reference_profile_path", "profiles/amazing-grace")),
                        feature_model_path=str(payload.get("feature_model_path", "models/wav2vec2-base.onnx")),
                        presentation_mode="logging",
                        silence_threshold_rms=float(payload.get("silence_threshold_rms", 0.005)),
                        match_debug_logging=bool(payload.get("match_debug_logging", False)),
                    )
                    manager.start(config)
                    self._write_json(self._status_payload(), HTTPStatus.OK)
                    return
                if parsed.path == "/api/stop":
                    self._read_json()
                    manager.stop()
                    self._write_json(self._status_payload(), HTTPStatus.OK)
                    return
                if parsed.path == "/api/manual/on":
                    self._read_json()
                    runtime = manager.runtime()
                    if runtime is None:
                        raise ValueError("runtime is not running")
                    runtime.enable_manual_override()
                    self._write_json(self._status_payload(), HTTPStatus.OK)
                    return
                if parsed.path == "/api/manual/off":
                    self._read_json()
                    runtime = manager.runtime()
                    if runtime is None:
                        raise ValueError("runtime is not running")
                    runtime.disable_manual_override()
                    self._write_json(self._status_payload(), HTTPStatus.OK)
                    return
                if parsed.path == "/api/manual/toggle":
                    self._read_json()
                    runtime = manager.runtime()
                    if runtime is None:
                        raise ValueError("runtime is not running")
                    runtime.toggle_manual_override()
                    self._write_json(self._status_payload(), HTTPStatus.OK)
                    return
                if parsed.path == "/api/jump":
                    payload = self._read_json()
                    runtime = manager.runtime()
                    if runtime is None:
                        raise ValueError("runtime is not running")
                    slide_number = int(payload.get("slide_number", 0))
                    runtime.jump_to_slide(slide_number)
                    self._write_json(self._status_payload(), HTTPStatus.OK)
                    return
            except Exception as error:
                self._write_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args) -> None:
            del format, args

        def _write_json(self, payload: dict[str, object], status: HTTPStatus) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {"host": args.host, "port": args.port, "url": f"http://{args.host}:{args.port}/"},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
        server.server_close()


if __name__ == "__main__":
    main()
