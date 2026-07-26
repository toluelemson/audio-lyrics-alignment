from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
  <title>Line Click Capture</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: #fffaf1;
      --ink: #1d1b18;
      --muted: #6f665d;
      --line: #d8cbb8;
      --accent: #a54b2a;
      --accent-strong: #7f3317;
      --ok: #256b3f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff6de 0, transparent 28rem),
        linear-gradient(180deg, #f7f1e8 0%, var(--bg) 100%);
    }
    .shell {
      max-width: 960px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      margin-bottom: 24px;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.75rem;
      color: var(--muted);
      margin-bottom: 8px;
    }
    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.6rem);
      line-height: 0.95;
    }
    .sub {
      max-width: 42rem;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.5;
    }
    .grid {
      display: grid;
      gap: 20px;
      grid-template-columns: 1.1fr 0.9fr;
    }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(75, 53, 29, 0.08);
    }
    .status {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.9rem;
      color: var(--muted);
      background: rgba(255,255,255,0.72);
    }
    .timer {
      font-size: clamp(2.2rem, 6vw, 4rem);
      margin: 6px 0 16px;
      color: var(--accent-strong);
      font-variant-numeric: tabular-nums;
    }
    .stage {
      display: grid;
      gap: 14px;
      margin-bottom: 18px;
    }
    .stage-panel {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
      background: rgba(255,255,255,0.68);
    }
    .stage-panel.previous,
    .stage-panel.next {
      background: rgba(255,255,255,0.5);
    }
    .stage-panel.current {
      background: linear-gradient(135deg, #fff0e5 0%, #fffaf5 100%);
      border-color: color-mix(in srgb, var(--accent) 55%, var(--line));
      box-shadow: 0 0 0 3px rgba(165, 75, 42, 0.1);
    }
    .stage-panel.clickable,
    .list li.clickable {
      cursor: pointer;
    }
    .stage-panel.clickable:hover,
    .list li.clickable:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 22px rgba(75, 53, 29, 0.12);
    }
    .stage-label {
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-size: 0.72rem;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .stage-meta {
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 8px;
    }
    .stage-lyrics {
      white-space: pre-wrap;
      line-height: 1.45;
      margin: 0;
    }
    .stage-panel.current .stage-lyrics {
      font-size: 1.2rem;
      min-height: 10rem;
    }
    .stage-panel.previous .stage-lyrics,
    .stage-panel.next .stage-lyrics {
      font-size: 0.98rem;
      color: #4f4841;
      min-height: 4.5rem;
    }
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
    }
    button {
      border: 0;
      border-radius: 14px;
      padding: 14px 18px;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.45; cursor: default; transform: none; }
    .primary { background: var(--accent); color: #fff7f0; }
    .secondary { background: #eadfce; color: var(--ink); }
    .success { background: var(--ok); color: #f3fff7; }
    .caption {
      color: var(--muted);
      margin-top: 10px;
      font-size: 0.9rem;
    }
    .list {
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 10px;
      max-height: 34rem;
      overflow: auto;
    }
    .list li {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.65);
    }
    .list li.done {
      border-color: color-mix(in srgb, var(--ok) 40%, var(--line));
      background: color-mix(in srgb, #eef9f1 70%, white);
    }
    .list li.current {
      border-color: color-mix(in srgb, var(--accent) 55%, var(--line));
      background: linear-gradient(135deg, #fff2e8 0%, #fffaf4 100%);
      box-shadow: 0 0 0 3px rgba(165, 75, 42, 0.12);
    }
    .list-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 6px;
      font-size: 0.95rem;
    }
    .stamp {
      color: var(--accent-strong);
      font-variant-numeric: tabular-nums;
    }
    audio { width: 100%; margin-top: 6px; }
    .banner {
      margin-top: 14px;
      min-height: 1.5rem;
      color: var(--ok);
    }
    .banner.error { color: #9b1d1d; }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="eyebrow">Reference Prep</div>
      <h1>Capture Line Change Clicks</h1>
      <p class="sub">Start the song, click <strong>Start Capture</strong>, then press
      <strong>Mark Current Line</strong>, click the current line itself, or hit the space bar each time the lyrics should change.</p>
    </div>
    <div class="grid">
      <section class="card">
        <div class="status">
          <div class="pill" id="progressPill">0 / 0 captured</div>
          <div class="pill" id="nextPill">Next: none</div>
        </div>
        <audio id="audio" controls preload="metadata"></audio>
        <div class="timer" id="timer">00:00.000</div>
        <div class="stage">
          <div class="stage-panel previous">
            <div class="stage-label">Previous</div>
            <div class="stage-meta" id="previousMeta">No previous slide yet</div>
            <pre class="stage-lyrics" id="previousLyrics">Waiting to start.</pre>
          </div>
          <div class="stage-panel current" id="currentPanel">
            <div class="stage-label">Current Line</div>
            <div class="stage-meta" id="currentMeta">Waiting to start</div>
            <pre class="stage-lyrics" id="currentLyrics">Load complete.</pre>
          </div>
          <div class="stage-panel next">
            <div class="stage-label">Next Up</div>
            <div class="stage-meta" id="upcomingMeta">No next slide yet</div>
            <pre class="stage-lyrics" id="upcomingLyrics">Capture will advance here automatically.</pre>
          </div>
        </div>
        <div class="controls">
          <button class="primary" id="startButton">Start Capture</button>
          <button class="primary" id="markButton" disabled>Mark Current Line</button>
          <button class="secondary" id="undoButton" disabled>Undo Last Click</button>
          <button class="secondary" id="resetButton">Reset</button>
          <button class="success" id="finishButton" disabled>Finish Capture</button>
        </div>
        <div class="caption">Keyboard: <strong>space</strong> marks the current line. <strong>Backspace</strong> undoes the last click.</div>
        <div class="banner" id="banner"></div>
      </section>
      <aside class="card">
        <h2 style="margin-top:0">Captured Lines</h2>
        <ol class="list" id="capturedList"></ol>
      </aside>
    </div>
  </div>
  <script>
    const bootstrapSession = __BOOTSTRAP_SESSION__;
    const state = {
      session: bootstrapSession,
      timerHandle: null,
    };

    const audio = document.getElementById("audio");
    const progressPill = document.getElementById("progressPill");
    const nextPill = document.getElementById("nextPill");
    const timer = document.getElementById("timer");
    const previousMeta = document.getElementById("previousMeta");
    const previousLyrics = document.getElementById("previousLyrics");
    const currentMeta = document.getElementById("currentMeta");
    const currentLyrics = document.getElementById("currentLyrics");
    const currentPanel = document.getElementById("currentPanel");
    const upcomingMeta = document.getElementById("upcomingMeta");
    const upcomingLyrics = document.getElementById("upcomingLyrics");
    const banner = document.getElementById("banner");
    const capturedList = document.getElementById("capturedList");
    const startButton = document.getElementById("startButton");
    const markButton = document.getElementById("markButton");
    const undoButton = document.getElementById("undoButton");
    const resetButton = document.getElementById("resetButton");
    const finishButton = document.getElementById("finishButton");
    if (bootstrapSession.audio_available) {
      audio.src = "/audio";
      audio.style.display = "block";
    } else {
      audio.style.display = "none";
    }

    function formatSeconds(value) {
      const totalMs = Math.max(0, Math.round(value * 1000));
      const minutes = String(Math.floor(totalMs / 60000)).padStart(2, "0");
      const seconds = String(Math.floor((totalMs % 60000) / 1000)).padStart(2, "0");
      const millis = String(totalMs % 1000).padStart(3, "0");
      return `${minutes}:${seconds}.${millis}`;
    }

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

    function replaySlideIndex() {
      if (!state.session || audio.style.display === "none") {
        return null;
      }
      const captured = state.session.captured_cues;
      if (!captured.length) {
        return null;
      }
      let index = -1;
      for (let i = 0; i < captured.length; i += 1) {
        if (audio.currentTime + 0.001 >= captured[i].click_timestamp) {
          index = i;
          continue;
        }
        break;
      }
      return index;
    }

    async function captureCurrentLine() {
      state.session = await api("/api/mark", {
        method: "POST",
        body: JSON.stringify({ elapsed: currentCaptureTimestamp() }),
      });
      setBanner("Line click captured.");
      render();
      updateTimer();
    }

    function render() {
      if (!state.session) {
        return;
      }
      const captured = state.session.captured_cues;
      const total = state.session.script_cues.length;
      const next = state.session.next_cue;
      const replayIndex = replaySlideIndex();
      const replayEnabled = replayIndex !== null && (state.session.finished || captured.length === total);
      const captureIndex = captured.length;
      const activeIndex = replayEnabled ? replayIndex : captureIndex;
      const previous = replayEnabled
        ? (activeIndex > 0 ? captured[activeIndex - 1] : null)
        : (captured.length > 0 ? captured[captured.length - 1] : null);
      const currentCue = replayEnabled
        ? (activeIndex >= 0 ? state.session.script_cues[activeIndex] : null)
        : next;
      const upcoming = replayEnabled
        ? state.session.script_cues[activeIndex + 1] || null
        : (next ? state.session.script_cues[captured.length + 1] || null : null);
      progressPill.textContent = `${captured.length} / ${total} captured`;
      nextPill.textContent = replayEnabled
        ? (upcoming ? `Next: ${upcoming.section} · Line ${upcoming.line_number}` : "Next: complete")
        : (next ? `Next: ${next.section} · Line ${next.line_number}` : "Next: complete");
      currentMeta.textContent = replayEnabled
        ? (currentCue ? `Replay · ${currentCue.section} · Line ${currentCue.line_number} of ${currentCue.line_count}` : "Replay · Waiting for first saved cue")
        : (next ? `${next.section} · Line ${next.line_number} of ${next.line_count}` : "All lines captured");
      currentLyrics.textContent = replayEnabled
        ? (currentCue ? currentCue.lyrics : "No slide is active yet.")
        : (next ? next.lyrics : "Capture complete.");
      previousMeta.textContent = previous
        ? `${previous.section} · Line ${previous.line_number} · ${formatSeconds(previous.click_timestamp)}`
        : "No previous line yet";
      previousLyrics.textContent = previous ? previous.lyrics : "Waiting to start.";
      upcomingMeta.textContent = upcoming
        ? `${upcoming.section} · Line ${upcoming.line_number} of ${upcoming.line_count}`
        : "No next line yet";
      upcomingLyrics.textContent = upcoming
        ? upcoming.lyrics
        : "Capture will advance here automatically.";
      markButton.disabled = !state.session.started || state.session.complete || state.session.finished;
      undoButton.disabled = captured.length === 0;
      finishButton.disabled = captured.length !== total || state.session.finished;
      currentPanel.classList.toggle("clickable", !markButton.disabled);

      capturedList.innerHTML = "";
      let activeItem = null;
      state.session.script_cues.forEach((cue, index) => {
        const item = document.createElement("li");
        if (index < captured.length) item.classList.add("done");
        const isCurrentItem = replayEnabled ? index === activeIndex : index === captureIndex && index < total;
        if (isCurrentItem) {
          item.classList.add("current");
          activeItem = item;
        }
        if (!replayEnabled && isCurrentItem && !markButton.disabled) {
          item.classList.add("clickable");
          item.addEventListener("click", () => {
            captureCurrentLine().catch((error) => setBanner(error.message, "error"));
          });
        }
        const stamp = index < captured.length
          ? formatSeconds(captured[index].click_timestamp)
          : "pending";
        item.innerHTML = `
          <div class="list-top">
            <strong>${cue.section} · Line ${cue.line_number} of ${cue.line_count}</strong>
            <span class="stamp">${stamp}</span>
          </div>
          <div>${cue.lyrics}</div>
        `;
        capturedList.appendChild(item);
      });
      if (activeItem) {
        activeItem.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }

    function updateTimer() {
      if (!state.session || !state.session.started) {
        timer.textContent = "00:00.000";
        return;
      }
      const elapsed = audio.style.display !== "none"
        ? audio.currentTime || 0
        : Math.max(0, (Date.now() / 1000) - state.session.started_at_epoch_seconds);
      timer.textContent = formatSeconds(elapsed);
    }

    function syncTimerLoop() {
      clearInterval(state.timerHandle);
      state.timerHandle = setInterval(updateTimer, 50);
      updateTimer();
    }

    async function loadSession() {
      state.session = await api("/api/session");
      if (state.session.audio_available) {
        audio.src = "/audio";
        audio.style.display = "block";
      } else {
        audio.style.display = "none";
      }
      render();
      updateTimer();
    }

    function currentCaptureTimestamp() {
      if (audio.style.display !== "none") {
        return Number(audio.currentTime || 0);
      }
      if (!state.session || !state.session.started_at_epoch_seconds) {
        return 0;
      }
      return Math.max(0, (Date.now() / 1000) - state.session.started_at_epoch_seconds);
    }

    startButton.addEventListener("click", async () => {
      try {
        state.session = await api("/api/start", { method: "POST", body: "{}" });
        setBanner("Capture started.");
        syncTimerLoop();
        render();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    markButton.addEventListener("click", async () => {
      try {
        await captureCurrentLine();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    currentPanel.addEventListener("click", async () => {
      if (markButton.disabled) return;
      try {
        await captureCurrentLine();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    undoButton.addEventListener("click", async () => {
      try {
        state.session = await api("/api/undo", { method: "POST", body: "{}" });
        setBanner("Last click removed.");
        render();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    resetButton.addEventListener("click", async () => {
      try {
        state.session = await api("/api/reset", { method: "POST", body: "{}" });
        setBanner("Capture reset.");
        render();
        updateTimer();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    finishButton.addEventListener("click", async () => {
      try {
        state.session = await api("/api/finish", { method: "POST", body: "{}" });
        setBanner(`Capture finished and saved to ${state.session.saved_output_path}`);
        render();
        updateTimer();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    ["timeupdate", "play", "pause", "seeked", "loadedmetadata"].forEach((eventName) => {
      audio.addEventListener(eventName, () => {
        render();
        updateTimer();
      });
    });

    window.addEventListener("keydown", async (event) => {
      if (event.code === "Backspace") {
        if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) return;
        event.preventDefault();
        if (undoButton.disabled) return;
        undoButton.click();
        return;
      }
      if (event.code !== "Space") return;
      if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) return;
      event.preventDefault();
      if (markButton.disabled) return;
      markButton.click();
    });

    render();
    updateTimer();
    syncTimerLoop();
    loadSession().catch((error) => setBanner(error.message, "error"));
  </script>
</body>
</html>
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a simple local browser UI for capturing slide click timings.",
    )
    parser.add_argument(
        "--slides",
        required=True,
        help="Path to the slide script JSON file with slide_number, section, and lyrics.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the captured slide cue JSON file with click_timestamp values.",
    )
    parser.add_argument(
        "--audio",
        help="Optional audio file to expose to the browser for manual playback.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on.",
    )
    return parser.parse_args()


def _session_payload(session, output_path: Path, audio_path: Path | None) -> dict[str, object]:
    next_index = session.current_index()
    next_cue = (
        session.script_cues[next_index]
        if next_index < len(session.script_cues)
        else None
    )
    return {
        "script_cues": [
            {
                "slide_number": cue.slide_number,
                "section": cue.section,
                "lyrics": cue.lyrics,
                "line_number": cue.line_number,
                "line_count": cue.line_count,
            }
            for cue in session.script_cues
        ],
        "captured_cues": [
            {
                "slide_number": cue.slide_number,
                "section": cue.section,
                "lyrics": cue.lyrics,
                "line_number": cue.line_number,
                "line_count": cue.line_count,
                "click_timestamp": cue.click_timestamp,
            }
            for cue in session.captured_cues()
        ],
        "next_cue": (
            None
            if next_cue is None
            else {
                "slide_number": next_cue.slide_number,
                "section": next_cue.section,
                "lyrics": next_cue.lyrics,
                "line_number": next_cue.line_number,
                "line_count": next_cue.line_count,
            }
        ),
        "started": session.started_at is not None,
        "complete": session.is_complete(),
        "finished": session.finished,
        "started_at_epoch_seconds": session.started_at_wall_seconds,
        "saved_output_path": str(output_path),
        "audio_available": audio_path is not None,
    }


def _parse_range_header(value: str, size: int) -> tuple[int, int] | None:
    if not value.startswith("bytes="):
        return None
    range_spec = value.removeprefix("bytes=").strip()
    if "," in range_spec or "-" not in range_spec:
        return None
    start_text, end_text = range_spec.split("-", 1)
    if not start_text and not end_text:
        return None
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            return None
        if suffix_length >= size:
            return (0, size - 1)
        return (size - suffix_length, size - 1)
    start = int(start_text)
    if start < 0 or start >= size:
        return None
    if not end_text:
        return (start, size - 1)
    end = int(end_text)
    if end < start:
        return None
    return (start, min(end, size - 1))


def main() -> None:
    from lyrics_aligner.application.click_capture import (
        ClickCaptureSession,
        load_slide_script,
        write_captured_slide_cues,
    )

    args = _parse_args()
    script_cues = load_slide_script(args.slides)
    session = ClickCaptureSession(script_cues)
    output_path = Path(args.output).expanduser().resolve()
    audio_path = None if args.audio is None else Path(args.audio).expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def _read_json_body(self) -> dict[str, object]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}
            raw_body = self.rfile.read(content_length)
            if not raw_body:
                return {}
            payload = json.loads(raw_body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._write_html(HTML)
                return
            if parsed.path == "/api/session":
                self._write_json(
                    _session_payload(session, output_path, audio_path),
                    HTTPStatus.OK,
                )
                return
            if parsed.path == "/audio":
                if audio_path is None or not audio_path.exists():
                    self._write_json({"error": "audio file is not available"}, HTTPStatus.NOT_FOUND)
                    return
                self._write_file(audio_path)
                return
            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/start":
                    self._read_json_body()
                    session.start()
                    self._write_json(
                        _session_payload(session, output_path, audio_path),
                        HTTPStatus.OK,
                    )
                    return
                if parsed.path == "/api/mark":
                    payload = self._read_json_body()
                    elapsed = payload.get("elapsed")
                    if elapsed is not None and not isinstance(elapsed, (int, float)):
                        raise ValueError("elapsed must be a number")
                    session.mark_current(
                        elapsed=None if elapsed is None else float(elapsed),
                    )
                    self._write_json(
                        _session_payload(session, output_path, audio_path),
                        HTTPStatus.OK,
                    )
                    return
                if parsed.path == "/api/undo":
                    self._read_json_body()
                    session.undo_last()
                    self._write_json(
                        _session_payload(session, output_path, audio_path),
                        HTTPStatus.OK,
                    )
                    return
                if parsed.path == "/api/reset":
                    self._read_json_body()
                    session.reset()
                    self._write_json(
                        _session_payload(session, output_path, audio_path),
                        HTTPStatus.OK,
                    )
                    return
                if parsed.path == "/api/save":
                    self._read_json_body()
                    write_captured_slide_cues(str(output_path), session.captured_cues())
                    self._write_json(
                        _session_payload(session, output_path, audio_path),
                        HTTPStatus.OK,
                    )
                    return
                if parsed.path == "/api/finish":
                    self._read_json_body()
                    captured_cues = session.finish()
                    write_captured_slide_cues(str(output_path), captured_cues)
                    self._write_json(
                        _session_payload(session, output_path, audio_path),
                        HTTPStatus.OK,
                    )
                    return
            except ValueError as error:
                self._write_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args) -> None:
            del format, args

        def _write_html(self, content: str) -> None:
            bootstrap = json.dumps(_session_payload(session, output_path, audio_path))
            rendered = content.replace("__BOOTSTRAP_SESSION__", bootstrap)
            data = rendered.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_json(self, payload: dict[str, object], status: HTTPStatus) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_file(self, path: Path) -> None:
            content_type, _ = mimetypes.guess_type(path.name)
            file_size = path.stat().st_size
            range_header = self.headers.get("Range")
            byte_range = None
            if range_header is not None:
                try:
                    byte_range = _parse_range_header(range_header, file_size)
                except ValueError:
                    byte_range = None
            if byte_range is None:
                start = 0
                end = file_size - 1
                status = HTTPStatus.OK
            else:
                start, end = byte_range
                status = HTTPStatus.PARTIAL_CONTENT
            content_length = end - start + 1

            self.send_response(status)
            self.send_header(
                "Content-Type",
                content_type or "application/octet-stream",
            )
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{end}/{file_size}",
                )
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(content_length))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "slides": len(script_cues),
                "output": str(output_path),
                "audio": None if audio_path is None else str(audio_path),
                "url": f"http://{args.host}:{args.port}/",
            },
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
        server.server_close()


if __name__ == "__main__":
    main()
