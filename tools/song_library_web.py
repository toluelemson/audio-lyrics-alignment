from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sys
import wave
from dataclasses import asdict, dataclass, replace
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread
from time import sleep
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LIBRARY_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Song Library</title>
  <style>
    :root {
      --bg: #f5efe5;
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
    .shell { max-width: 1100px; margin: 0 auto; padding: 24px; }
    .hero { margin-bottom: 24px; }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.75rem; color: var(--muted); margin-bottom: 8px; }
    h1 { margin: 0 0 10px; font-size: clamp(2.2rem, 4vw, 4rem); line-height: 0.95; }
    .sub { max-width: 42rem; color: var(--muted); line-height: 1.5; }
    .grid { display: grid; gap: 20px; grid-template-columns: 0.95fr 1.05fr; }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(75, 53, 29, 0.08);
    }
    form { display: grid; gap: 14px; }
    .field { display: grid; gap: 6px; }
    .field label { color: var(--muted); font-size: 0.92rem; }
    input, textarea, button { font: inherit; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.78);
      color: var(--ink);
    }
    textarea { min-height: 14rem; resize: vertical; }
    button {
      border: 0;
      border-radius: 14px;
      padding: 14px 18px;
      cursor: pointer;
    }
    .primary { background: var(--accent); color: #fff7f0; }
    .songs { display: grid; gap: 12px; }
    .song {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.66);
    }
    .song-top { display: flex; justify-content: space-between; gap: 12px; align-items: start; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 0.88rem;
      color: var(--muted);
      background: rgba(255,255,255,0.74);
    }
    .song-meta { color: var(--muted); font-size: 0.92rem; margin-top: 6px; }
    .song-actions { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
    .song-actions a,
    .song-actions button {
      display: inline-block;
      padding: 11px 14px;
      border-radius: 12px;
      text-decoration: none;
      background: #eadfce;
      color: var(--ink);
      border: 0;
    }
    .song-actions button.danger {
      background: #f2d8cf;
      color: #7d2410;
    }
    .banner { min-height: 1.5rem; margin-top: 10px; color: var(--ok); }
    .banner.error { color: #9b1d1d; }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="eyebrow">Song Library</div>
      <h1>Capture First. Align After.</h1>
      <p class="sub">Create a song with a title, save the audio from YouTube or anywhere, then add lyrics and align them after the song is captured.</p>
    </div>
    <div class="grid">
      <section class="card">
        <h2 style="margin-top:0">Create Song</h2>
        <form id="addSongForm">
          <div class="field">
            <label for="title">Song Title</label>
            <input id="title" name="title" placeholder="Amazing Grace" required>
          </div>
          <div class="field">
            <label for="audio">Optional Local Audio</label>
            <input id="audio" name="audio" type="file" accept="audio/*">
          </div>
          <button class="primary" type="submit">Create Song</button>
        </form>
        <div class="banner" id="banner"></div>
      </section>
      <section class="card">
        <h2 style="margin-top:0">Songs</h2>
        <div class="songs" id="songsView"></div>
      </section>
    </div>
  </div>
  <script>
    const state = { songs: __SONGS__ };
    const songsView = document.getElementById("songsView");
    const banner = document.getElementById("banner");

    function setBanner(message, type = "") {
      banner.textContent = message || "";
      banner.className = type ? `banner ${type}` : "banner";
    }
    function confirmDanger(title, description) {
      return window.confirm(`${title}\n\n${description}`);
    }

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ error: "Request failed" }));
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return response.json();
    }

    function renderSongs() {
      songsView.innerHTML = "";
      if (!state.songs.length) {
        songsView.textContent = "No songs yet. Add the first one on the left.";
        return;
      }
      state.songs.forEach((song) => {
        const article = document.createElement("article");
        article.className = "song";
        article.innerHTML = `
          <div class="song-top">
            <div>
              <div style="font-size:1.35rem; font-weight:700">${song.title}</div>
              <div class="song-meta">Updated ${new Date(song.updated_at).toLocaleString()}</div>
            </div>
            <div class="pill">${song.status}</div>
          </div>
          <div class="song-actions">
            <a href="/songs/${song.song_id}">Open</a>
            <button class="danger" type="button" data-delete-song="${song.song_id}" data-song-title="${song.title}">Delete</button>
          </div>
        `;
        songsView.appendChild(article);
      });
      songsView.querySelectorAll("[data-delete-song]").forEach((button) => {
        button.addEventListener("click", async () => {
          const songTitle = button.dataset.songTitle || "this song";
          const confirmed = confirmDanger(
            `Delete ${songTitle}?`,
            "This removes the song, lyrics timing work, built profile, and all stored audio files. This cannot be undone.",
          );
          if (!confirmed) return;
          try {
            const payload = await api(`/api/songs/${button.dataset.deleteSong}/delete`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: "{}",
            });
            state.songs = payload.songs;
            renderSongs();
            setBanner("Song deleted.");
          } catch (error) {
            setBanner(error.message, "error");
          }
        });
      });
    }

    document.getElementById("addSongForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const formData = new FormData();
      formData.set("title", document.getElementById("title").value);
      const audioInput = document.getElementById("audio");
      if (audioInput.files && audioInput.files.length) {
        formData.set("audio", audioInput.files[0]);
      }
      try {
        const payload = await api("/api/songs", { method: "POST", body: formData });
        state.songs = payload.songs;
        renderSongs();
        event.target.reset();
        window.location.href = `/songs/${payload.song.song_id}`;
      } catch (error) {
        setBanner(error.message, "error");
      }
    });

    renderSongs();
  </script>
</body>
</html>
"""

SONG_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Song Workflow</title>
  <style>
    :root {
      --bg: #f5efe5;
      --panel: #fffaf1;
      --ink: #1d1b18;
      --muted: #6f665d;
      --line: #d8cbb8;
      --accent: #a54b2a;
      --ok: #256b3f;
      --warn: #9b1d1d;
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
    .shell { max-width: 980px; margin: 0 auto; padding: 24px; }
    .back { color: var(--muted); text-decoration: none; }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.75rem; color: var(--muted); margin: 14px 0 8px; }
    h1 { margin: 0 0 10px; font-size: clamp(2.2rem, 4vw, 4rem); line-height: 0.95; }
    .sub { max-width: 42rem; color: var(--muted); line-height: 1.5; }
    .hero-row { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.92rem;
      color: var(--muted);
      background: rgba(255,255,255,0.72);
    }
    .checklist {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 22px 0;
    }
    .check {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.68);
    }
    .check strong { display: block; margin-bottom: 4px; font-size: 0.95rem; }
    .check span { color: var(--muted); font-size: 0.88rem; line-height: 1.35; }
    .check.done {
      border-color: color-mix(in srgb, var(--ok) 38%, var(--line));
      background: color-mix(in srgb, #eef9f1 74%, white);
    }
    .grid { display: grid; gap: 18px; }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(75, 53, 29, 0.08);
    }
    .card h2 { margin: 0 0 8px; }
    .card p { margin: 0; color: var(--muted); line-height: 1.5; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
    .inline-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 16px; }
    button, input, textarea, select, a.action {
      font: inherit;
    }
    button, a.action {
      border: 0;
      border-radius: 14px;
      padding: 14px 18px;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.82);
      color: var(--ink);
    }
    textarea { min-height: 16rem; resize: vertical; margin-top: 16px; }
    .primary { background: var(--accent); color: #fff7f0; }
    .secondary { background: #eadfce; color: var(--ink); }
    .success { background: var(--ok); color: #f3fff7; }
    .danger { background: #f2d8cf; color: #7d2410; }
    .ghost { background: rgba(255,255,255,0.78); color: var(--ink); border: 1px solid var(--line); }
    .hidden { display: none !important; }
    .banner { min-height: 1.5rem; margin-top: 12px; color: var(--ok); }
    .banner.error { color: var(--warn); }
    .small { font-size: 0.9rem; color: var(--muted); }
    .step-state { margin-top: 10px; font-size: 0.95rem; color: var(--muted); }
    .audio-meta { margin-top: 12px; color: var(--muted); font-size: 0.92rem; }
    .advanced-tools {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.72);
      overflow: hidden;
    }
    .advanced-tools summary {
      list-style: none;
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 700;
      background: rgba(248, 240, 229, 0.9);
    }
    .advanced-tools summary::-webkit-details-marker { display: none; }
    .advanced-body { padding: 14px 16px 16px; }
    @media (max-width: 900px) {
      .checklist { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 640px) {
      .checklist { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <a class="back" href="/songs/">Back To Library</a>
    <div class="eyebrow">Song Workflow</div>
    <div class="hero-row">
      <div>
        <h1>__SONG_TITLE__</h1>
        <p class="sub">Save the song first, then add lyrics, then align them. Only the next useful step stays emphasized.</p>
      </div>
      <div class="pill" id="statusPill">Loading</div>
    </div>
    <div class="checklist">
      <div class="check" id="checkAudio"><strong>Song saved</strong><span>Reference audio is ready.</span></div>
      <div class="check" id="checkLyrics"><strong>Lyrics added</strong><span>Lyrics text exists for this song.</span></div>
      <div class="check" id="checkTiming"><strong>Timing aligned</strong><span>Line cues were saved.</span></div>
      <div class="check" id="checkProfile"><strong>Profile built</strong><span>Live runtime is ready.</span></div>
    </div>
    <div class="grid">
      <section class="card">
        <h2>Step 1: Capture Song</h2>
        <p>Play from YouTube, a local file, Spotify, or anywhere. If you already have a local audio file, attach it here instead.</p>
        <div class="step-state" id="captureState">No saved song audio yet.</div>
        <div class="inline-row">
          <select id="captureDeviceSelect" style="min-width:260px"></select>
          <button class="secondary" id="refreshCaptureDevicesButton" type="button">Refresh Inputs</button>
        </div>
        <div class="actions">
          <button class="primary" id="startListeningButton" type="button">Start Listening</button>
          <button class="success hidden" id="stopListeningButton" type="button">Stop And Save</button>
          <button class="secondary" id="attachAudioButton" type="button">Attach Local Audio</button>
          <button class="danger hidden" id="deleteAudioButton" type="button">Delete Audio</button>
          <a class="action ghost hidden" id="downloadAudioLink" href="#" download>Download Audio</a>
        </div>
        <input id="audioManagerInput" type="file" accept="audio/*" style="display:none">
        <div class="audio-meta" id="audioMeta">No audio saved yet.</div>
      </section>
      <section class="card">
        <h2>Step 2: Add Lyrics</h2>
        <p>Paste or edit the lyrics after the song is saved. This prepares the line list for alignment.</p>
        <textarea id="lyricsInput" placeholder="Paste lyrics here after the song is captured."></textarea>
        <div class="actions">
          <button class="primary" id="saveLyricsButton" type="button">Add Lyrics</button>
        </div>
        <div class="small">Saving lyrics clears old timings and profile data for this song.</div>
      </section>
      <section class="card">
        <h2>Step 3: Align Lyrics</h2>
        <p>Open the alignment view only after audio and lyrics are both ready.</p>
        <div class="step-state" id="alignState">Lyrics are needed before alignment can start.</div>
        <div class="actions">
          <a class="action primary hidden" id="alignLink" href="/songs/__SONG_ID__/prepare">Align Lyrics</a>
        </div>
        <details class="advanced-tools">
          <summary>Edit Timing Details</summary>
          <div class="advanced-body">
            <p class="small">Use this only for clip start/end, nudging, reset, or starting from a selected line after you open the alignment page.</p>
          </div>
        </details>
      </section>
      <section class="card hidden" id="profileCard">
        <h2>Build Profile</h2>
        <p>Build the runtime profile only after the audio and saved timings are both ready.</p>
        <div class="actions">
          <button class="primary" id="buildProfileButton" type="button">Build Profile</button>
          <a class="action success hidden" id="runLiveLink" href="/songs/__SONG_ID__/live">Run Live</a>
        </div>
      </section>
    </div>
    <div class="banner" id="banner"></div>
  </div>
  <script>
    const apiBase = "__API_BASE__";
    const state = { song: __SONG_STATE__ };
    const statusPill = document.getElementById("statusPill");
    const checkAudio = document.getElementById("checkAudio");
    const checkLyrics = document.getElementById("checkLyrics");
    const checkTiming = document.getElementById("checkTiming");
    const checkProfile = document.getElementById("checkProfile");
    const captureState = document.getElementById("captureState");
    const captureDeviceSelect = document.getElementById("captureDeviceSelect");
    const refreshCaptureDevicesButton = document.getElementById("refreshCaptureDevicesButton");
    const startListeningButton = document.getElementById("startListeningButton");
    const stopListeningButton = document.getElementById("stopListeningButton");
    const attachAudioButton = document.getElementById("attachAudioButton");
    const deleteAudioButton = document.getElementById("deleteAudioButton");
    const audioManagerInput = document.getElementById("audioManagerInput");
    const downloadAudioLink = document.getElementById("downloadAudioLink");
    const audioMeta = document.getElementById("audioMeta");
    const lyricsInput = document.getElementById("lyricsInput");
    const saveLyricsButton = document.getElementById("saveLyricsButton");
    const alignState = document.getElementById("alignState");
    const alignLink = document.getElementById("alignLink");
    const profileCard = document.getElementById("profileCard");
    const buildProfileButton = document.getElementById("buildProfileButton");
    const runLiveLink = document.getElementById("runLiveLink");
    const banner = document.getElementById("banner");

    function setBanner(message, type = "") {
      banner.textContent = message || "";
      banner.className = type ? `banner ${type}` : "banner";
    }
    function confirmDanger(title, description) {
      return window.confirm(`${title}\n\n${description}`);
    }
    async function api(path, options = {}) {
      const response = await fetch(`${apiBase}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ error: "Request failed" }));
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return response.json();
    }
    async function apiMultipart(path, formData) {
      const response = await fetch(`${apiBase}${path}`, { method: "POST", body: formData });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ error: "Request failed" }));
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return response.json();
    }
    function selectedCaptureDeviceId() {
      if (captureDeviceSelect.disabled) return null;
      const value = Number(captureDeviceSelect.value);
      return Number.isInteger(value) ? value : null;
    }
    async function loadDevices() {
      const payload = await api("/devices");
      captureDeviceSelect.innerHTML = "";
      if (!payload.devices || !payload.devices.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No input devices found";
        captureDeviceSelect.appendChild(option);
        captureDeviceSelect.disabled = true;
        return;
      }
      payload.devices.forEach((item) => {
        const option = document.createElement("option");
        option.value = String(item.id);
        option.textContent = `${item.id}: ${item.name}`;
        if (item.default) option.selected = true;
        captureDeviceSelect.appendChild(option);
      });
      captureDeviceSelect.disabled = false;
    }
    async function loadState() {
      state.song = await api("/state");
      render();
    }
    function renderCheck(node, done) {
      node.classList.toggle("done", Boolean(done));
    }
    function render() {
      const song = state.song;
      statusPill.textContent = song.status;
      lyricsInput.value = song.lyrics_text || "";
      renderCheck(checkAudio, song.has_audio);
      renderCheck(checkLyrics, song.has_lyrics);
      renderCheck(checkTiming, song.has_timings);
      renderCheck(checkProfile, song.has_profile);
      startListeningButton.classList.toggle("hidden", song.system_capture_active);
      stopListeningButton.classList.toggle("hidden", !song.system_capture_active);
      attachAudioButton.textContent = song.has_audio ? "Replace Local Audio" : "Attach Local Audio";
      deleteAudioButton.classList.toggle("hidden", !song.has_audio);
      downloadAudioLink.classList.toggle("hidden", !song.has_audio);
      if (song.has_audio) {
        downloadAudioLink.href = `${apiBase}/audio`;
      } else {
        downloadAudioLink.removeAttribute("href");
      }
      captureState.textContent = song.system_capture_active
        ? `Listening on ${song.system_capture_device_name || "selected input"} now. When the song ends, click Stop And Save.`
        : (song.has_audio ? "Song audio is saved and ready for lyrics alignment." : "No saved song audio yet.");
      audioMeta.textContent = song.has_audio
        ? "Saved audio is available for replay, alignment, and profile building."
        : "Use Start Listening for system playback capture, or attach a local audio file.";
      saveLyricsButton.textContent = song.has_lyrics ? "Save Lyrics" : "Add Lyrics";
      alignLink.classList.toggle("hidden", !(song.has_audio && song.has_lyrics));
      alignState.textContent = !song.has_audio
        ? "Save the song audio first."
        : (!song.has_lyrics
          ? "Lyrics are needed before alignment can start."
          : (song.has_timings ? "Lyrics have timings. Open alignment to review or edit." : "Audio and lyrics are ready. Start aligning."));
      profileCard.classList.toggle("hidden", !(song.has_audio && song.has_timings));
      runLiveLink.classList.toggle("hidden", !song.has_profile);
      buildProfileButton.classList.toggle("hidden", song.has_profile);
    }
    refreshCaptureDevicesButton.addEventListener("click", () => {
      loadDevices().then(() => setBanner("Input devices refreshed.")).catch((error) => setBanner(error.message, "error"));
    });
    startListeningButton.addEventListener("click", async () => {
      try {
        await api("/start-listening", {
          method: "POST",
          body: JSON.stringify({ input_device: selectedCaptureDeviceId() }),
        });
        await loadState();
        setBanner("Listening started. Play the song, then click Stop And Save.");
      } catch (error) {
        setBanner(error.message, "error");
      }
    });
    stopListeningButton.addEventListener("click", async () => {
      try {
        await api("/stop-listening", { method: "POST", body: "{}" });
        await loadState();
        setBanner("Song saved. Next: add lyrics.", "success");
      } catch (error) {
        setBanner(error.message, "error");
      }
    });
    attachAudioButton.addEventListener("click", () => {
      if (state.song.has_audio) {
        const confirmed = confirmDanger(
          "Replace audio?",
          "This replaces the current audio file, clears the built profile, and resets any saved clip range back to the full song.",
        );
        if (!confirmed) return;
      }
      audioManagerInput.click();
    });
    audioManagerInput.addEventListener("change", async () => {
      if (!audioManagerInput.files || !audioManagerInput.files.length) return;
      const formData = new FormData();
      formData.set("audio", audioManagerInput.files[0]);
      try {
        await apiMultipart("/audio", formData);
        audioManagerInput.value = "";
        await loadState();
        setBanner("Audio saved. Next: add lyrics.");
      } catch (error) {
        setBanner(error.message, "error");
      }
    });
    deleteAudioButton.addEventListener("click", async () => {
      const confirmed = confirmDanger(
        "Delete this song audio?",
        "Timings stay, but the stored audio, saved clip audio, and built profile will be removed.",
      );
      if (!confirmed) return;
      try {
        await api("/audio-delete", { method: "POST", body: "{}" });
        await loadState();
        setBanner("Audio deleted.");
      } catch (error) {
        setBanner(error.message, "error");
      }
    });
    saveLyricsButton.addEventListener("click", async () => {
      try {
        await api("/lyrics", {
          method: "POST",
          body: JSON.stringify({ lyrics: lyricsInput.value }),
        });
        await loadState();
        setBanner("Lyrics saved. Next: align lyrics.");
      } catch (error) {
        setBanner(error.message, "error");
      }
    });
    buildProfileButton.addEventListener("click", async () => {
      try {
        await api("/build-profile", { method: "POST", body: "{}" });
        await loadState();
        setBanner("Profile built. Live runtime is ready.");
      } catch (error) {
        setBanner(error.message, "error");
      }
    });
    loadDevices().then(loadState).catch((error) => setBanner(error.message, "error"));
  </script>
</body>
</html>
"""

LYRICS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Edit Lyrics</title>
  <style>
    :root {
      --bg: #f5efe5;
      --panel: #fffaf1;
      --ink: #1d1b18;
      --muted: #6f665d;
      --line: #d8cbb8;
      --accent: #a54b2a;
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
    .shell { max-width: 900px; margin: 0 auto; padding: 24px; }
    .back { color: var(--muted); text-decoration: none; }
    h1 { margin: 14px 0 10px; font-size: clamp(2rem, 4vw, 3.3rem); line-height: 0.95; }
    .sub { color: var(--muted); line-height: 1.5; max-width: 42rem; }
    .card {
      margin-top: 24px;
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(75, 53, 29, 0.08);
    }
    textarea {
      width: 100%;
      min-height: 24rem;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.82);
      color: var(--ink);
      font: inherit;
      resize: vertical;
    }
    .actions { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
    button, a.action {
      border: 0;
      border-radius: 14px;
      padding: 14px 18px;
      font: inherit;
      cursor: pointer;
      text-decoration: none;
    }
    .primary { background: var(--accent); color: #fff7f0; }
    .secondary { background: #eadfce; color: var(--ink); }
    .banner { min-height: 1.5rem; margin-top: 12px; color: var(--ok); }
    .banner.error { color: #9b1d1d; }
  </style>
</head>
<body>
  <div class="shell">
    <a class="back" href="/songs/">Back To Library</a>
    <h1>Edit Lyrics: __SONG_TITLE__</h1>
    <p class="sub">Saving lyrics regenerates the slide scaffold and clears existing timings and profile data for this song, so you can prepare it again cleanly.</p>
    <section class="card">
      <textarea id="lyricsInput">__LYRICS__</textarea>
      <div class="actions">
        <button class="primary" id="saveButton" type="button">Save Lyrics</button>
        <a class="action secondary" href="/songs/__SONG_ID__/prepare">Go To Prepare Song</a>
      </div>
      <div class="banner" id="banner"></div>
    </section>
  </div>
  <script>
    const apiBase = "__API_BASE__";
    const lyricsInput = document.getElementById("lyricsInput");
    const banner = document.getElementById("banner");
    function setBanner(message, type = "") {
      banner.textContent = message || "";
      banner.className = type ? `banner ${type}` : "banner";
    }
    async function api(path, options = {}) {
      const response = await fetch(`${apiBase}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ error: "Request failed" }));
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return response.json();
    }
    document.getElementById("saveButton").addEventListener("click", async () => {
      try {
        await api("/lyrics", {
          method: "POST",
          body: JSON.stringify({ lyrics: lyricsInput.value }),
        });
        setBanner("Lyrics saved. Timings and profile were cleared for re-prep.");
      } catch (error) {
        setBanner(error.message, "error");
      }
    });
  </script>
</body>
</html>
"""

PREPARE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prepare Song</title>
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
    .shell { max-width: 960px; margin: 0 auto; padding: 24px; }
    .hero { margin-bottom: 24px; }
    .back { color: var(--muted); text-decoration: none; }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.75rem; color: var(--muted); margin: 14px 0 8px; }
    h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.6rem); line-height: 0.95; }
    .sub { max-width: 42rem; color: var(--muted); font-size: 1rem; line-height: 1.5; }
    .grid { display: grid; gap: 20px; grid-template-columns: 1.1fr 0.9fr; }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(75, 53, 29, 0.08);
    }
    .status { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.9rem;
      color: var(--muted);
      background: rgba(255,255,255,0.72);
    }
    .timer { font-size: clamp(2.2rem, 6vw, 4rem); margin: 6px 0 16px; color: var(--accent-strong); font-variant-numeric: tabular-nums; }
    .stage { display: grid; gap: 14px; margin-bottom: 18px; }
    .stage-panel {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
      background: rgba(255,255,255,0.68);
    }
    .stage-panel.current { background: linear-gradient(135deg, #fff0e5 0%, #fffaf5 100%); border-color: color-mix(in srgb, var(--accent) 55%, var(--line)); box-shadow: 0 0 0 3px rgba(165, 75, 42, 0.1); }
    .stage-panel.clickable, .list-entry.clickable { cursor: pointer; }
    .stage-panel.clickable:hover, .list-entry.clickable:hover { transform: translateY(-1px); box-shadow: 0 10px 22px rgba(75, 53, 29, 0.12); }
    .stage-label { text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.72rem; color: var(--muted); margin-bottom: 8px; }
    .stage-meta { color: var(--muted); font-size: 0.95rem; margin-bottom: 8px; }
    .stage-lyrics { white-space: pre-wrap; line-height: 1.45; margin: 0; }
    .stage-panel.current .stage-lyrics { font-size: 1.2rem; min-height: 8rem; }
    .controls { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px; }
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
    .caption { color: var(--muted); margin-top: 10px; font-size: 0.9rem; }
    .workflow {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 0 0 16px;
    }
    .workflow-step {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.7);
    }
    .workflow-step strong {
      display: block;
      margin-bottom: 4px;
      font-size: 0.95rem;
    }
    .workflow-step span {
      display: block;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.35;
    }
    .workflow-step.active {
      border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
      background: linear-gradient(135deg, #fff2e8 0%, #fffaf4 100%);
      box-shadow: 0 0 0 3px rgba(165, 75, 42, 0.08);
    }
    .clip-bar {
      display: grid;
      gap: 8px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.68);
      margin-top: 10px;
    }
    .clip-labels {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 0.88rem;
      font-variant-numeric: tabular-nums;
    }
    .clip-range {
      width: 100%;
      accent-color: var(--accent);
      margin: 0;
    }
    .list { margin: 0; padding: 0; list-style: none; display: grid; gap: 10px; max-height: 34rem; overflow: auto; }
    .list li { padding: 0; border: 0; background: transparent; }
    .list-entry {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.65);
      text-align: left;
      color: inherit;
      display: block;
    }
    .list-entry.done { border-color: color-mix(in srgb, var(--ok) 40%, var(--line)); background: color-mix(in srgb, #eef9f1 70%, white); }
    .list-entry.current { border-color: color-mix(in srgb, var(--accent) 55%, var(--line)); background: linear-gradient(135deg, #fff2e8 0%, #fffaf4 100%); box-shadow: 0 0 0 3px rgba(165, 75, 42, 0.12); }
    .list-entry.selected {
      border-color: color-mix(in srgb, var(--ok) 52%, var(--line));
      background: linear-gradient(135deg, #eef9f1 0%, #fbfffc 100%);
      box-shadow: 0 0 0 3px rgba(37, 107, 63, 0.14);
    }
    .list-entry:focus-visible { outline: 3px solid rgba(165, 75, 42, 0.22); outline-offset: 2px; }
    .list-top { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 6px; font-size: 0.95rem; }
    .stamp { color: var(--accent-strong); font-variant-numeric: tabular-nums; }
    .timeline-feedback {
      margin: 0 0 12px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      color: var(--muted);
      min-height: 3.8rem;
      line-height: 1.4;
    }
    .timeline-feedback strong { color: var(--ink); display: block; margin-bottom: 4px; }
    .timeline-feedback.success {
      border-color: color-mix(in srgb, var(--ok) 40%, var(--line));
      background: color-mix(in srgb, #eef9f1 72%, white);
      color: var(--ok);
    }
    .timeline-feedback.error {
      border-color: color-mix(in srgb, #9b1d1d 35%, var(--line));
      background: color-mix(in srgb, #fff0f0 72%, white);
      color: #9b1d1d;
    }
    .save-reminder {
      margin-top: 14px;
      padding: 12px 14px;
      border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--line));
      border-radius: 14px;
      background: linear-gradient(135deg, #fff2e8 0%, #fffaf4 100%);
      color: var(--accent-strong);
      line-height: 1.4;
      display: none;
    }
    .save-reminder strong { display: block; color: var(--ink); margin-bottom: 4px; }
    .save-reminder.active { display: block; }
    .step-banner {
      margin: 0 0 14px;
      padding: 12px 14px;
      border: 1px solid color-mix(in srgb, var(--ok) 28%, var(--line));
      border-radius: 14px;
      background: linear-gradient(135deg, #eef9f1 0%, #fbfffc 100%);
      color: var(--ok);
      line-height: 1.45;
      display: none;
    }
    .step-banner strong { color: var(--ink); }
    .step-banner.active { display: block; }
    .advanced-tools {
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.72);
      overflow: hidden;
    }
    .advanced-tools summary {
      list-style: none;
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 700;
      background: rgba(248, 240, 229, 0.9);
    }
    .advanced-tools summary::-webkit-details-marker {
      display: none;
    }
    .advanced-tools[open] summary {
      border-bottom: 1px solid var(--line);
    }
    .advanced-body {
      padding: 14px 16px 16px;
    }
    .advanced-copy {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
      margin: 0 0 10px;
    }
    audio { display: none; }
    .player-shell {
      margin-top: 6px;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 14px 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(250,245,238,0.92) 100%);
      box-shadow: 0 10px 26px rgba(75, 53, 29, 0.08);
    }
    .player-shell.no-audio {
      background: linear-gradient(180deg, rgba(255,250,241,0.95) 0%, rgba(248,240,229,0.95) 100%);
    }
    .player-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .player-kicker {
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.72rem;
      color: var(--muted);
    }
    .player-state {
      font-size: 0.9rem;
      color: var(--muted);
    }
    .player-row {
      display: grid;
      gap: 12px;
    }
    .player-main {
      display: grid;
      grid-template-columns: auto auto minmax(0, 1fr) auto;
      align-items: center;
      gap: 12px;
    }
    .player-actions {
      justify-self: start;
    }
    .audio-menu {
      position: relative;
    }
    .audio-menu summary {
      list-style: none;
    }
    .audio-menu summary::-webkit-details-marker {
      display: none;
    }
    .audio-menu[open] summary {
      background: color-mix(in srgb, var(--accent) 12%, #eadfce);
    }
    .audio-menu-panel {
      position: absolute;
      right: 0;
      top: calc(100% + 8px);
      min-width: 13rem;
      display: grid;
      gap: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,250,241,0.98);
      box-shadow: 0 16px 28px rgba(75, 53, 29, 0.14);
      z-index: 5;
    }
    .audio-menu-panel a,
    .audio-menu-panel button {
      width: 100%;
      text-align: left;
    }
    .icon-button {
      min-width: 5.5rem;
      padding: 12px 16px;
      font-weight: 700;
    }
    .time-readout {
      min-width: 8rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      font-size: 0.92rem;
      white-space: nowrap;
    }
    .player-seek {
      min-width: 0;
      width: 100%;
      accent-color: var(--accent);
      margin: 0;
    }
    .player-speed {
      min-width: 7rem;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.9);
      color: var(--ink);
    }
    .player-download {
      display: inline-flex;
      align-items: center;
      justify-content: flex-start;
      padding: 12px 16px;
      border-radius: 14px;
      background: #eadfce;
      color: var(--ink);
      text-decoration: none;
    }
    .audio-menu-panel .secondary,
    .player-download {
      padding: 12px 14px;
      border-radius: 12px;
    }
    .banner { margin-top: 14px; min-height: 1.5rem; color: var(--ok); }
    .banner.error { color: #9b1d1d; }
    @media (max-width: 980px) {
      .workflow {
        grid-template-columns: 1fr;
      }
      .player-main {
        grid-template-columns: auto auto 1fr;
      }
      .player-speed {
        grid-column: 1 / -1;
        justify-self: start;
      }
    }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
      .player-header {
        align-items: flex-start;
        flex-direction: column;
      }
      .player-main {
        grid-template-columns: 1fr;
      }
      .icon-button,
      .time-readout,
      .player-speed {
        justify-self: start;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <a class="back" href="/songs/">Back To Library</a>
    <div class="hero">
      <div class="eyebrow">Prepare Song</div>
      <h1>__SONG_TITLE__</h1>
      <p class="sub">Start capture, click the current lyric line as the song moves, then save.</p>
    </div>
    <div class="grid">
      <section class="card">
        <div class="workflow" id="workflowSteps">
          <div class="workflow-step active" id="workflowAudio">
            <strong>1. Choose audio</strong>
            <span>Upload audio here or keep using system playback.</span>
          </div>
          <div class="workflow-step" id="workflowCapture">
            <strong>2. Start and click lines</strong>
            <span>Press Start, then click each lyric when it begins.</span>
          </div>
          <div class="workflow-step" id="workflowSave">
            <strong>3. Save timings</strong>
            <span>Save when the pass is done so replay uses these cues.</span>
          </div>
        </div>
        <div class="step-banner" id="stepBanner"><strong>How to use this:</strong> Press <strong>Start</strong>, click each lyric line when it begins, then click <strong>Save Timings</strong>.</div>
        <div class="status">
          <div class="pill" id="progressPill">0 / 0 captured</div>
          <div class="pill" id="nextPill">Next: none</div>
        </div>
        <div class="player-shell" id="playerShell">
          <div class="player-header">
            <div class="player-kicker">Reference Audio</div>
            <div class="player-state" id="playerStateLabel">No local audio loaded</div>
          </div>
          <div class="player-row">
            <div class="player-main">
              <button class="secondary icon-button" id="playToggleButton" type="button">Play</button>
              <div class="time-readout" id="playerTimeReadout">00:00.000 / 00:00.000</div>
              <input class="player-seek" id="playerSeekRange" type="range" min="0" max="0" step="0.01" value="0">
              <select class="player-speed" id="playerSpeedSelect">
                <option value="0.75">0.75x</option>
                <option value="1" selected>1.0x</option>
                <option value="1.25">1.25x</option>
                <option value="1.5">1.5x</option>
              </select>
            </div>
            <div class="player-actions">
              <details class="audio-menu" id="audioMenu">
                <summary class="secondary" id="audioMenuButton">Audio</summary>
                <div class="audio-menu-panel">
                  <a class="player-download" id="downloadAudioLink" href="#" download>Download Audio</a>
                  <button class="secondary" id="manageAudioButton" type="button">Attach Audio</button>
                  <button class="secondary" id="deleteAudioButton" type="button" style="display:none">Delete Audio</button>
                </div>
              </details>
              <input id="audioManagerInput" type="file" accept="audio/*" style="display:none">
            </div>
          </div>
        </div>
        <audio id="audio" preload="metadata"></audio>
        <div class="save-reminder active" id="externalAudioNote" style="display:none"><strong>System playback mode</strong>No local file was uploaded for this song. Start the song from your computer, browser, YouTube, Spotify, or another app, then click lines here as it runs.</div>
        <div class="controls" id="liveFeedControls" style="display:none; margin-top:12px">
          <select id="captureDeviceSelect" style="min-width:260px"></select>
          <button class="secondary" id="refreshCaptureDevicesButton" type="button">Refresh Inputs</button>
        </div>
        <div class="timer" id="timer">00:00.000</div>
        <details class="advanced-tools" id="advancedTools">
          <summary>More Tools</summary>
          <div class="advanced-body">
            <p class="advanced-copy">Use these only when you need to start from the middle, fine-tune timings, or set a clip window.</p>
            <div class="controls" style="margin-top:0">
              <button class="secondary" id="startFromSelectedButton">Start At Selected Line</button>
              <button class="secondary" id="toggleSlideBreakButton" disabled>Start New Slide Here</button>
              <button class="secondary" id="lineSlidesButton">One Line Per Slide</button>
              <button class="secondary" id="undoButton" disabled>Undo Last</button>
              <button class="secondary" id="resetButton">Clear All</button>
              <button class="secondary" id="nudgeBack50" disabled>Earlier</button>
              <button class="secondary" id="nudgeForward50" disabled>Later</button>
            </div>
            <div class="controls" id="clipControls" style="margin-top:12px">
              <button class="secondary" id="setClipStartButton" type="button">Set Start</button>
              <button class="secondary" id="setClipEndButton" type="button">Set End</button>
              <button class="success" id="saveClipButton" type="button">Save Clip</button>
              <button class="secondary" id="clearClipButton" type="button">Use Whole Song</button>
              <button class="secondary" id="resetSavedTimingsButton" type="button">Reset Timings</button>
            </div>
            <div class="clip-bar" id="clipBar">
              <div class="clip-labels">
                <span id="clipStartLabel">Start 00:00.000</span>
                <span id="clipEndLabel">End full song</span>
              </div>
              <input class="clip-range" id="clipStartRange" type="range" min="0" max="0" step="0.01" value="0">
              <input class="clip-range" id="clipEndRange" type="range" min="0" max="0" step="0.01" value="0">
            </div>
            <div class="caption" id="clipSummary">Clip: full song</div>
          </div>
        </details>
        <div class="stage">
          <div class="stage-panel current" id="currentPanel">
            <div class="stage-label" id="currentLabel">Click This Line Now</div>
            <div class="stage-meta" id="currentMeta">Waiting to start</div>
            <pre class="stage-lyrics" id="currentLyrics">Load complete.</pre>
          </div>
        </div>
        <div class="controls" id="primaryControls">
          <button class="primary" id="startButton">Start</button>
          <button class="success" id="saveButton" disabled>Save Timings</button>
        </div>
        <div class="save-reminder" id="saveReminder"><strong>Save reminder</strong>Click <strong>Save Timings</strong> before leaving or starting over.</div>
        <div class="caption" id="modeCaption">Press Start, click the lyric line when it begins, then save. After saving, clicking a saved line jumps the audio there for review.</div>
        <div class="banner" id="banner"></div>
      </section>
      <aside class="card">
        <h2 style="margin-top:0">Line Timeline</h2>
        <div class="timeline-feedback" id="timelineFeedback" aria-live="polite"><strong>Timeline feedback</strong>During training, the line you click turns green. In review, clicking a saved line jumps the reference audio to that spot.</div>
        <ol class="list" id="capturedList"></ol>
      </aside>
    </div>
  </div>
  <script>
    const bootstrapSession = __BOOTSTRAP_SESSION__;
    const apiBase = "__API_BASE__";
    const state = {
      session: bootstrapSession,
      timerHandle: null,
      selectedCueIndex: null,
      pendingCueIndex: null,
      clipDraftStartSeconds: bootstrapSession.clip_start_seconds || 0,
      clipDraftEndSeconds: bootstrapSession.clip_end_seconds,
      clipDraftDirty: false,
      clipPreviewActive: false,
      clipPreviewWasPaused: false,
    };
    const audio = document.getElementById("audio");
    const playerShell = document.getElementById("playerShell");
    const playToggleButton = document.getElementById("playToggleButton");
    const playerTimeReadout = document.getElementById("playerTimeReadout");
    const playerSeekRange = document.getElementById("playerSeekRange");
    const playerSpeedSelect = document.getElementById("playerSpeedSelect");
    const audioMenu = document.getElementById("audioMenu");
    const audioMenuButton = document.getElementById("audioMenuButton");
    const downloadAudioLink = document.getElementById("downloadAudioLink");
    const playerStateLabel = document.getElementById("playerStateLabel");
    const manageAudioButton = document.getElementById("manageAudioButton");
    const deleteAudioButton = document.getElementById("deleteAudioButton");
    const audioManagerInput = document.getElementById("audioManagerInput");
    const progressPill = document.getElementById("progressPill");
    const nextPill = document.getElementById("nextPill");
    const workflowAudio = document.getElementById("workflowAudio");
    const workflowCapture = document.getElementById("workflowCapture");
    const workflowSave = document.getElementById("workflowSave");
    const timer = document.getElementById("timer");
    const currentMeta = document.getElementById("currentMeta");
    const currentLyrics = document.getElementById("currentLyrics");
    const currentLabel = document.getElementById("currentLabel");
    const currentPanel = document.getElementById("currentPanel");
    const banner = document.getElementById("banner");
    const timelineFeedback = document.getElementById("timelineFeedback");
    const capturedList = document.getElementById("capturedList");
    const startButton = document.getElementById("startButton");
    const startFromSelectedButton = document.getElementById("startFromSelectedButton");
    const toggleSlideBreakButton = document.getElementById("toggleSlideBreakButton");
    const lineSlidesButton = document.getElementById("lineSlidesButton");
    const undoButton = document.getElementById("undoButton");
    const resetButton = document.getElementById("resetButton");
    const saveButton = document.getElementById("saveButton");
    const nudgeBack50 = document.getElementById("nudgeBack50");
    const nudgeForward50 = document.getElementById("nudgeForward50");
    const saveReminder = document.getElementById("saveReminder");
    const stepBanner = document.getElementById("stepBanner");
    const externalAudioNote = document.getElementById("externalAudioNote");
    const liveFeedControls = document.getElementById("liveFeedControls");
    const captureDeviceSelect = document.getElementById("captureDeviceSelect");
    const refreshCaptureDevicesButton = document.getElementById("refreshCaptureDevicesButton");
    const modeCaption = document.getElementById("modeCaption");
    const primaryControls = document.getElementById("primaryControls");
    const advancedTools = document.getElementById("advancedTools");
    const clipControls = document.getElementById("clipControls");
    const clipSummary = document.getElementById("clipSummary");
    const setClipStartButton = document.getElementById("setClipStartButton");
    const setClipEndButton = document.getElementById("setClipEndButton");
    const saveClipButton = document.getElementById("saveClipButton");
    const clearClipButton = document.getElementById("clearClipButton");
    const resetSavedTimingsButton = document.getElementById("resetSavedTimingsButton");
    const clipBar = document.getElementById("clipBar");
    const clipStartLabel = document.getElementById("clipStartLabel");
    const clipEndLabel = document.getElementById("clipEndLabel");
    const clipStartRange = document.getElementById("clipStartRange");
    const clipEndRange = document.getElementById("clipEndRange");
    function clipStartSeconds() {
      return Number(state.session?.clip_start_seconds || 0);
    }
    function clipEndSeconds() {
      const value = state.session?.clip_end_seconds;
      return value === null || value === undefined ? null : Number(value);
    }
    function draftClipStartSeconds() {
      return Number(state.clipDraftStartSeconds || 0);
    }
    function draftClipEndSeconds() {
      return state.clipDraftEndSeconds === null || state.clipDraftEndSeconds === undefined
        ? null
        : Number(state.clipDraftEndSeconds);
    }
    function currentAudioAbsoluteSeconds() {
      return Number(audio.currentTime || 0);
    }
    function toClipRelative(absoluteSeconds) {
      return Math.max(0, absoluteSeconds - clipStartSeconds());
    }
    function toClipAbsolute(relativeSeconds) {
      return clipStartSeconds() + relativeSeconds;
    }
    function syncClipDraftFromSession() {
      if (state.clipDraftDirty) return;
      state.clipDraftStartSeconds = clipStartSeconds();
      state.clipDraftEndSeconds = clipEndSeconds();
    }
    function audioDurationSeconds() {
      return Number.isFinite(audio.duration) ? Number(audio.duration || 0) : 0;
    }
    function renderPlayerBar() {
      const hasAudio = Boolean(state.session && state.session.audio_available);
      const currentSeconds = hasAudio ? Number(audio.currentTime || 0) : 0;
      const durationSeconds = hasAudio ? audioDurationSeconds() : 0;
      playerShell.classList.toggle("no-audio", !hasAudio);
      audioMenuButton.textContent = hasAudio ? "Audio" : "Add Audio";
      playToggleButton.style.display = hasAudio ? "inline-block" : "none";
      playerSeekRange.style.display = hasAudio ? "block" : "none";
      playerSpeedSelect.style.display = hasAudio ? "inline-block" : "none";
      downloadAudioLink.style.display = hasAudio ? "inline-flex" : "none";
      playToggleButton.textContent = hasAudio && !audio.paused ? "Pause" : "Play";
      playToggleButton.disabled = !hasAudio;
      playerSeekRange.disabled = !hasAudio;
      playerSeekRange.max = String(durationSeconds || 0);
      playerSeekRange.value = String(Math.min(currentSeconds, durationSeconds || 0));
      playerTimeReadout.textContent = `${formatSeconds(currentSeconds)} / ${formatSeconds(durationSeconds)}`;
      playerStateLabel.textContent = hasAudio
        ? (audio.paused ? "Ready to review and edit timings" : "Playing reference audio")
        : "No local audio loaded";
      if (hasAudio) {
        downloadAudioLink.href = `${apiBase}/audio`;
      } else {
        downloadAudioLink.removeAttribute("href");
      }
    }
    function beginClipPreview() {
      if (!state.session?.audio_available) return false;
      if (!state.clipPreviewActive) {
        state.clipPreviewActive = true;
        state.clipPreviewWasPaused = audio.paused;
      }
      return true;
    }
    function previewClipPosition(seconds) {
      if (!beginClipPreview()) return;
      audio.currentTime = Math.max(0, seconds);
      if (audio.paused) {
        audio.play().catch(() => {});
      }
    }
    function endClipPreview() {
      if (!state.clipPreviewActive) return;
      if (state.clipPreviewWasPaused && !audio.paused) {
        audio.pause();
      }
      state.clipPreviewActive = false;
      state.clipPreviewWasPaused = false;
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
    function cueSummary(cue) {
      return `Slide ${cue.slide_number} · ${cue.section} · Line ${cue.line_number}`;
    }
    function setTimelineFeedback(title, message, type = "") {
      timelineFeedback.innerHTML = `<strong>${title}</strong>${message}`;
      timelineFeedback.className = type ? `timeline-feedback ${type}` : "timeline-feedback";
    }
    function isReviewMode() {
      return Boolean(state.session && state.session.finished);
    }
    function firstCapturedIndex() {
      if (!state.session) return null;
      const cues = state.session.cue_states || [];
      for (let index = 0; index < cues.length; index += 1) {
        if (cues[index].click_timestamp !== null) return index;
      }
      return null;
    }
    function firstCapturedCue() {
      const index = firstCapturedIndex();
      if (index === null || !state.session) return null;
      return state.session.cue_states[index] || null;
    }
    function syncReviewAudioPosition() {
      if (!state.session || !state.session.audio_available || !isReviewMode()) return;
      const cue = firstCapturedCue();
      if (!cue || cue.click_timestamp === null) return;
      const cueAbsoluteSeconds = toClipAbsolute(cue.click_timestamp);
      if (!audio.paused) return;
      if (Math.abs((audio.currentTime || 0) - cueAbsoluteSeconds) < 0.05) return;
      if ((audio.currentTime || 0) <= cueAbsoluteSeconds) {
        audio.currentTime = cueAbsoluteSeconds;
      }
    }
    function syncAudioUi() {
      const hasAudio = Boolean(state.session && state.session.audio_available);
      manageAudioButton.textContent = hasAudio ? "Replace Audio" : "Attach Audio";
      deleteAudioButton.style.display = hasAudio ? "inline-block" : "none";
      clipControls.style.display = hasAudio ? "flex" : "none";
      clipBar.style.display = hasAudio ? "grid" : "none";
      clipSummary.style.display = hasAudio ? "block" : "none";
      if (hasAudio) {
        if (!audio.src) audio.src = `${apiBase}/audio`;
        externalAudioNote.style.display = "none";
        liveFeedControls.style.display = "none";
        stepBanner.classList.remove("active");
      } else {
        externalAudioNote.style.display = "block";
        liveFeedControls.style.display = "flex";
        stepBanner.classList.add("active");
      }
      renderPlayerBar();
      if (!state.session || hasAudio) {
        externalAudioNote.innerHTML = "<strong>System playback mode</strong>No local file was uploaded for this song. Start the song from your computer, browser, YouTube, Spotify, or another app, then click lines here as it runs.";
        return;
      }
      if (state.session.system_capture_active) {
        const deviceName = state.session.system_capture_device_name || "loopback device";
        externalAudioNote.innerHTML = `<strong>Recording system audio</strong>Capture is running from <strong>${deviceName}</strong>. Click the current lyric line as the song moves, then click <strong>Save Timings</strong>.`;
        return;
      }
      if (isReviewMode()) {
        externalAudioNote.innerHTML = "<strong>System audio saved</strong>This song now has recorded reference audio. You can replay it here, keep adjusting timings, or build a profile.";
      }
    }
    function syncWorkflow() {
      const hasAudio = Boolean(state.session && state.session.audio_available);
      const training = isTrainingMode();
      const review = isReviewMode();
      workflowAudio.classList.toggle("active", !training && !review);
      workflowCapture.classList.toggle("active", training);
      workflowSave.classList.toggle("active", review || hasUnsavedTimings());
      if (!hasAudio) {
        workflowAudio.querySelector("span").textContent = "No upload needed. Play the song anywhere and click lines here.";
      } else {
        workflowAudio.querySelector("span").textContent = "Replay and edit against the uploaded reference audio.";
      }
    }
    async function loadCaptureDevices() {
      const payload = await api("/devices");
      captureDeviceSelect.innerHTML = "";
      if (!payload.devices || !payload.devices.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No input devices found";
        captureDeviceSelect.appendChild(option);
        captureDeviceSelect.disabled = true;
        return;
      }
      payload.devices.forEach((item) => {
        const option = document.createElement("option");
        option.value = String(item.id);
        option.textContent = `${item.id}: ${item.name}`;
        if (item.default) option.selected = true;
        captureDeviceSelect.appendChild(option);
      });
      captureDeviceSelect.disabled = false;
    }
    function selectedCaptureDeviceId() {
      if (captureDeviceSelect.disabled) return null;
      const value = Number(captureDeviceSelect.value);
      return Number.isInteger(value) ? value : null;
    }
    function hasUnsavedTimings() {
      return Boolean(state.session && state.session.started && !state.session.finished);
    }
    function cueStartsSlide(cues, index) {
      if (!cues || index === null || index === undefined || index < 0 || index >= cues.length) return false;
      if (index === 0) return true;
      return cues[index].slide_number !== cues[index - 1].slide_number;
    }
    function updateSaveReminder() {
      if (hasUnsavedTimings()) {
        saveReminder.classList.add("active");
        saveReminder.innerHTML = "<strong>Save reminder</strong>Click <strong>Save Timings</strong> before leaving or starting over.";
        return;
      }
      saveReminder.classList.remove("active");
    }
    function seekAudioToCue(cue) {
      if (!state.session?.audio_available || cue.click_timestamp === null) return;
      audio.currentTime = toClipAbsolute(cue.click_timestamp);
    }
    function bindPrimaryPress(button, handler) {
      button.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        handler();
      });
      button.addEventListener("click", (event) => {
        event.preventDefault();
      });
      button.addEventListener("dblclick", (event) => {
        event.preventDefault();
      });
      button.addEventListener("keydown", (event) => {
        if (event.code !== "Enter" && event.code !== "Space") return;
        event.preventDefault();
        handler();
      });
    }
    async function api(path, options = {}) {
      const response = await fetch(`${apiBase}${path}`, {
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
      if (!state.session || !state.session.audio_available || !state.session.finished) return null;
      const captured = state.session.captured_cues;
      if (!captured.length) return null;
      const clipRelativeTime = toClipRelative(audio.currentTime || 0);
      let index = -1;
      for (let i = 0; i < captured.length; i += 1) {
        if (clipRelativeTime + 0.001 >= captured[i].click_timestamp) {
          index = i;
          continue;
        }
        break;
      }
      return index >= 0 ? index : null;
    }
    function currentCaptureTimestamp() {
      if (state.session?.audio_available) return toClipRelative(audio.currentTime || 0);
      if (!state.session || !state.session.started_at_epoch_seconds) return 0;
      return Math.max(0, (Date.now() / 1000) - state.session.started_at_epoch_seconds);
    }
    function lastCapturedTimestamp() {
      if (!state.session) return 0;
      const captured = state.session.captured_cues || [];
      if (!captured.length) return 0;
      const lastCue = captured[captured.length - 1];
      return Number(lastCue.click_timestamp || 0);
    }
    async function captureCurrentLine() {
      const clickedIndex = state.session ? state.session.current_index : null;
      state.pendingCueIndex = clickedIndex;
      render();
      state.session = await api("/mark", { method: "POST", body: JSON.stringify({ elapsed: currentCaptureTimestamp() }) });
      state.pendingCueIndex = null;
      if (clickedIndex !== null) {
        const capturedCue = state.session.cue_states[clickedIndex];
        setTimelineFeedback(
          "Line captured",
          `${cueSummary(capturedCue)} saved at ${formatSeconds(capturedCue.click_timestamp)}.`,
          "success",
        );
      }
      state.selectedCueIndex = null;
      setBanner("");
      render();
      updateTimer();
    }
    async function anchorCueAtPlayback(cueIndex) {
      state.pendingCueIndex = cueIndex;
      render();
      state.session = await api("/anchor", {
        method: "POST",
        body: JSON.stringify({ cue_index: cueIndex, elapsed: currentCaptureTimestamp() }),
      });
      state.pendingCueIndex = null;
      state.selectedCueIndex = cueIndex;
      const anchoredCue = state.session.cue_states[cueIndex];
      setTimelineFeedback(
        "Timeline updated",
        `${cueSummary(anchoredCue)} set to ${formatSeconds(anchoredCue.click_timestamp)}.`,
        "success",
      );
      render();
      updateTimer();
    }
    async function startFromSelectedCue() {
      if (state.selectedCueIndex === null) throw new Error("Select a line first.");
      const selectedCue = state.session.cue_states[state.selectedCueIndex];
      state.pendingCueIndex = state.selectedCueIndex;
      setTimelineFeedback(
        "Starting from selected line",
        `${cueSummary(selectedCue)} is now the new training start point.`,
      );
      render();
      state.session = await api("/start-from", {
        method: "POST",
        body: JSON.stringify({
          cue_index: state.selectedCueIndex,
          elapsed: currentCaptureTimestamp(),
          input_device: selectedCaptureDeviceId(),
        }),
      });
      state.pendingCueIndex = null;
      const anchoredCue = state.session.cue_states[state.selectedCueIndex];
      setTimelineFeedback(
        "Training restarted here",
        `${cueSummary(anchoredCue)} is anchored at ${formatSeconds(anchoredCue.click_timestamp)}. Continue from this line forward.`,
        "success",
      );
      setBanner("");
      syncTimerLoop();
      render();
      updateTimer();
    }
    function isTrainingMode() {
      return Boolean(state.session && state.session.started && !state.session.complete && !state.session.finished);
    }
    function render() {
      if (!state.session) return;
      syncClipDraftFromSession();
      const cues = state.session.cue_states;
      const captured = cues.filter((cue) => cue.click_timestamp !== null);
      const total = cues.length;
      const next = state.session.next_cue;
      const replayIndex = replaySlideIndex();
      const reviewMode = isReviewMode();
      const replayEnabled = replayIndex !== null && reviewMode;
      const beforeFirstReplayCue = reviewMode && replayIndex === null;
      const captureIndex = state.session.current_index;
      const firstSavedIndex = firstCapturedIndex();
      const activeIndex = reviewMode
        ? (replayEnabled ? replayIndex : (firstSavedIndex ?? 0))
        : captureIndex;
      const selectedCue = state.selectedCueIndex === null ? null : cues[state.selectedCueIndex] || null;
      const keepSelectedCueVisible = Boolean(
        selectedCue
        && selectedCue.click_timestamp !== null
        && !reviewMode
      );
      const displayIndex = keepSelectedCueVisible
        ? state.selectedCueIndex
        : activeIndex;
      const currentCue = displayIndex >= 0 ? cues[displayIndex] || null : null;
      const upcoming = reviewMode
        ? (displayIndex >= 0 ? cues[displayIndex + 1] || null : null)
        : (next || null);
      const canCaptureCurrent = isTrainingMode();
      progressPill.textContent = `${captured.length} / ${total} captured`;
      if (reviewMode) {
        nextPill.textContent = upcoming
          ? `Next: ${upcoming.section} · Line ${upcoming.line_number}`
          : (state.session.complete ? "Next: complete" : "Next: none");
      } else {
        nextPill.textContent = upcoming
          ? `Next: ${upcoming.section} · Line ${upcoming.line_number}`
          : "Next: complete";
      }
      currentLabel.textContent = canCaptureCurrent ? "Click This Line Now" : (reviewMode ? "Review Line" : "Current Line");
      currentMeta.textContent = reviewMode
        ? (currentCue
          ? `Review · ${currentCue.section} · Line ${currentCue.line_number} of ${currentCue.line_count}${beforeFirstReplayCue ? " · before saved cue" : ""}`
          : "Review · No saved line yet")
        : (currentCue ? `${currentCue.section} · Line ${currentCue.line_number} of ${currentCue.line_count}` : (next ? `${next.section} · Line ${next.line_number} of ${next.line_count}` : "Capture complete."));
      currentLyrics.textContent = currentCue ? currentCue.lyrics : (next ? next.lyrics : "Capture complete.");
      const canStartFromSelected = Boolean(
        state.selectedCueIndex !== null
        && !canCaptureCurrent
        && state.pendingCueIndex === null
      );
      const canToggleSlideBreak = Boolean(
        state.selectedCueIndex !== null
        && state.selectedCueIndex > 0
        && state.pendingCueIndex === null
      );
      undoButton.disabled = captured.length === 0;
      saveButton.disabled = captured.length === 0 || state.session.finished;
      startFromSelectedButton.disabled = !canStartFromSelected;
      toggleSlideBreakButton.disabled = !canToggleSlideBreak;
      currentPanel.classList.toggle("clickable", canCaptureCurrent);
      const canNudge = Boolean(selectedCue && selectedCue.click_timestamp !== null && state.session.audio_available && audio.paused);
      [nudgeBack50, nudgeForward50].forEach((button) => { button.disabled = !canNudge; });
      updateSaveReminder();
      syncAudioUi();
      syncWorkflow();
      syncReviewAudioPosition();
      if (state.session.audio_available) {
        startButton.textContent = canCaptureCurrent ? "Capture Running" : (reviewMode ? "Start New Pass" : "Start");
      } else {
        startButton.textContent = canCaptureCurrent ? "Capture Running" : (reviewMode ? "Start New Pass" : "Start");
      }
      startFromSelectedButton.textContent = "Start At Selected Line";
      if (state.selectedCueIndex !== null && state.selectedCueIndex > 0) {
        toggleSlideBreakButton.textContent = cueStartsSlide(cues, state.selectedCueIndex)
          ? "Merge With Previous Slide"
          : "Start New Slide Here";
      } else {
        toggleSlideBreakButton.textContent = "Start New Slide Here";
      }
      const activeClipStart = draftClipStartSeconds();
      const activeClipEnd = draftClipEndSeconds();
      clipSummary.textContent = activeClipEnd === null
        ? `Clip: ${formatSeconds(activeClipStart)} to end of song`
        : `Clip: ${formatSeconds(activeClipStart)} to ${formatSeconds(activeClipEnd)}`;
      const durationSeconds = audioDurationSeconds();
      clipStartRange.max = String(durationSeconds);
      clipEndRange.max = String(durationSeconds);
      clipStartRange.value = String(Math.min(activeClipStart, durationSeconds));
      clipEndRange.value = String(Math.min(activeClipEnd === null ? durationSeconds : activeClipEnd, durationSeconds));
      clipStartLabel.textContent = `Start ${formatSeconds(activeClipStart)}`;
      clipEndLabel.textContent = activeClipEnd === null
        ? "End full song"
        : `End ${formatSeconds(activeClipEnd)}`;
      saveClipButton.disabled = !state.clipDraftDirty;
      clearClipButton.disabled = activeClipStart === 0 && activeClipEnd === null && !state.clipDraftDirty;
      resetSavedTimingsButton.disabled = !Boolean(state.session.saved_output_path);
      if (canCaptureCurrent) {
        startButton.style.display = "none";
        saveButton.style.display = "inline-block";
        primaryControls.style.justifyContent = "flex-start";
      } else if (reviewMode) {
        startButton.style.display = "inline-block";
        saveButton.style.display = "none";
        primaryControls.style.justifyContent = "flex-start";
      } else {
        startButton.style.display = "inline-block";
        saveButton.style.display = "none";
        primaryControls.style.justifyContent = "flex-start";
      }
      startButton.disabled = canCaptureCurrent;
      startFromSelectedButton.disabled = !canStartFromSelected;
      modeCaption.textContent = canCaptureCurrent
        ? "Click the lyric line when the song reaches it. Space also captures the current line."
        : (reviewMode
          ? "Press play to review from the first saved cue. Clicking a saved line jumps the audio there."
          : "Start capture, then click the current lyric line as the external song plays.");
      if (state.session.audio_available && advancedTools.open) {
        modeCaption.textContent = "Advanced tools are open. Use Set Start and Set End only if you want to train from a clipped section of the song.";
      }
      capturedList.innerHTML = "";
      let activeItem = null;
      let selectedItem = null;
      cues.forEach((cue, index) => {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "list-entry";
        if (cue.click_timestamp !== null) button.classList.add("done");
        const isCurrentItem = replayEnabled ? index === activeIndex : index === captureIndex && index < total;
        if (isCurrentItem) { button.classList.add("current"); activeItem = item; }
        if (state.selectedCueIndex === index || state.pendingCueIndex === index) {
          button.classList.add("selected");
          selectedItem = item;
        }
        if (cue.locked) {
          button.style.borderColor = "color-mix(in srgb, var(--accent) 55%, var(--line))";
        } else if (cue.drafted) {
          button.style.borderColor = "color-mix(in srgb, var(--ok) 35%, var(--line))";
        }
        const canTrainFromTimeline = !replayEnabled && canCaptureCurrent;
        if (canTrainFromTimeline) {
          button.classList.add("clickable");
          bindPrimaryPress(button, () => {
            if (state.pendingCueIndex !== null) return;
            state.selectedCueIndex = index;
            state.pendingCueIndex = index;
            setTimelineFeedback(
              "Saving line",
              `${cueSummary(cue)} is being saved at ${formatSeconds(currentCaptureTimestamp())}.`,
            );
            setBanner(`Saving ${cueSummary(cue)}...`);
            render();
            anchorCueAtPlayback(index)
              .then(() => setBanner(cue.click_timestamp === null ? "Line saved." : "Line re-timed and saved."))
              .catch((error) => {
                state.pendingCueIndex = null;
                setBanner(error.message, "error");
                render();
              });
          });
        } else if (cue.click_timestamp !== null) {
          button.classList.add("clickable");
          bindPrimaryPress(button, () => {
            if (state.pendingCueIndex !== null) return;
            state.selectedCueIndex = index;
            setBanner(`Jumping audio to ${cueSummary(cue)}...`);
            seekAudioToCue(cue);
            setTimelineFeedback(
              "Line selected",
              `${cueSummary(cue)} opened at ${formatSeconds(cue.click_timestamp)} and is ready for Earlier or Later.`,
            );
            setBanner("Audio jumped to selected line.");
            render();
          });
        } else {
          button.classList.add("clickable");
          bindPrimaryPress(button, () => {
            if (state.pendingCueIndex !== null) return;
            state.selectedCueIndex = index;
            setBanner(`Selected ${cueSummary(cue)}.`);
            setTimelineFeedback(
              "Line selected",
              `${cueSummary(cue)} is selected. Open More Tools to start training here or change where a slide begins.`,
            );
            setBanner("Line selected.");
            render();
          });
        }
        const stamp = cue.click_timestamp !== null ? formatSeconds(cue.click_timestamp) : "pending";
        button.innerHTML = `<div class="list-top"><strong>Slide ${cue.slide_number} · ${cue.section} · Line ${cue.line_number} of ${cue.line_count}</strong><span class="stamp">${stamp}</span></div><div>${cue.lyrics}</div>`;
        item.appendChild(button);
        capturedList.appendChild(item);
      });
      const priorityItem = selectedItem || activeItem;
      if (priorityItem) priorityItem.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    function updateTimer() {
      if (!state.session || !state.session.started) { timer.textContent = "00:00.000"; return; }
      const elapsed = state.session.audio_available
        ? toClipRelative(audio.currentTime || 0)
        : (state.session.finished
          ? lastCapturedTimestamp()
          : Math.max(0, (Date.now() / 1000) - state.session.started_at_epoch_seconds));
      timer.textContent = formatSeconds(elapsed);
    }
    function syncTimerLoop() {
      clearInterval(state.timerHandle);
      state.timerHandle = setInterval(updateTimer, 50);
      updateTimer();
    }
    startButton.addEventListener("click", async () => {
      try {
        if (state.session.audio_available) {
          audio.currentTime = clipStartSeconds();
        }
        state.session = await api("/start", {
          method: "POST",
          body: JSON.stringify({ input_device: selectedCaptureDeviceId() }),
        });
        state.selectedCueIndex = null;
        state.pendingCueIndex = null;
        setTimelineFeedback("Capture started", "Click the current line or press space when the lyrics change.");
        setBanner("");
        syncTimerLoop();
        render();
      } catch (error) { setBanner(error.message, "error"); }
    });
    startFromSelectedButton.addEventListener("click", async () => {
      try {
        await startFromSelectedCue();
      } catch (error) { setBanner(error.message, "error"); }
    });
    toggleSlideBreakButton.addEventListener("click", async () => {
      if (state.selectedCueIndex === null || state.selectedCueIndex <= 0) {
        setBanner("Select a line after the first line.", "error");
        return;
      }
      try {
        const wasSlideStart = cueStartsSlide(state.session.cue_states, state.selectedCueIndex);
        state.pendingCueIndex = state.selectedCueIndex;
        render();
        state.session = await api("/regroup", {
          method: "POST",
          body: JSON.stringify({ cue_index: state.selectedCueIndex }),
        });
        state.pendingCueIndex = null;
        const updatedCue = state.session.cue_states[state.selectedCueIndex];
        setTimelineFeedback(
          wasSlideStart ? "Slide merged" : "New slide created",
          wasSlideStart
            ? `${cueSummary(updatedCue)} now continues the previous slide.`
            : `${cueSummary(updatedCue)} now begins a new slide group.`,
          "success",
        );
        setBanner(wasSlideStart ? "Slide merged." : "New slide start saved.");
        render();
      } catch (error) {
        state.pendingCueIndex = null;
        setBanner(error.message, "error");
        render();
      }
    });
    lineSlidesButton.addEventListener("click", async () => {
      const confirmed = confirmDanger(
        "Use one line per slide?",
        "This rewrites the song so every lyric line becomes its own slide. Saved timings will be kept and remapped line by line. Rebuild the profile after this.",
      );
      if (!confirmed) return;
      try {
        state.pendingCueIndex = state.selectedCueIndex;
        render();
        state.session = await api("/line-slides", { method: "POST", body: "{}" });
        state.pendingCueIndex = null;
        state.selectedCueIndex = null;
        setTimelineFeedback(
          "Line-by-line slides enabled",
          "Each lyric line now has its own slide number. Rebuild the profile before testing live again.",
          "success",
        );
        setBanner("Song converted to one line per slide.");
        render();
      } catch (error) {
        state.pendingCueIndex = null;
        setBanner(error.message, "error");
        render();
      }
    });
    manageAudioButton.addEventListener("click", () => {
      const hasAudio = Boolean(state.session && state.session.audio_available);
      if (hasAudio) {
        const confirmed = confirmDanger(
          "Replace audio?",
          "This replaces the current audio file, clears the built profile, and resets any saved clip range back to the full song.",
        );
        if (!confirmed) return;
      }
      audioManagerInput.click();
    });
    audioManagerInput.addEventListener("change", async () => {
      if (!audioManagerInput.files || !audioManagerInput.files.length) return;
      const formData = new FormData();
      formData.set("audio", audioManagerInput.files[0]);
      try {
        await api("/audio", { method: "POST", body: formData });
        audioManagerInput.value = "";
        audio.src = "";
        state.session = await api("/session");
        state.selectedCueIndex = null;
        state.pendingCueIndex = null;
        state.clipDraftDirty = false;
        audioMenu.open = false;
        setBanner("Audio updated. Profile cleared and status synced.");
        render();
        updateTimer();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });
    deleteAudioButton.addEventListener("click", async () => {
      const confirmed = confirmDanger(
        "Delete this song audio?",
        "Timings stay, but the stored audio, saved clip audio, and built profile will be removed. Live playback review will no longer be available until audio is added again.",
      );
      if (!confirmed) return;
      try {
        await api("/audio/delete", {
          method: "POST",
          body: "{}",
        });
        audio.src = "";
        state.session = await api("/session");
        state.selectedCueIndex = null;
        state.pendingCueIndex = null;
        state.clipDraftDirty = false;
        audioMenu.open = false;
        setBanner("Audio deleted. Status synced.");
        render();
        updateTimer();
      } catch (error) {
        setBanner(error.message, "error");
      }
    });
    refreshCaptureDevicesButton.addEventListener("click", () => {
      loadCaptureDevices()
        .then(() => setBanner("Input devices refreshed."))
        .catch((error) => setBanner(error.message, "error"));
    });
    currentPanel.addEventListener("click", async () => {
      if (!state.session || !state.session.started || state.session.complete || state.session.finished) return;
      try { await captureCurrentLine(); } catch (error) { setBanner(error.message, "error"); }
    });
    undoButton.addEventListener("click", async () => {
      try {
        state.session = await api("/undo", { method: "POST", body: "{}" });
        state.pendingCueIndex = null;
        setTimelineFeedback("Last click removed", "The most recent saved line timing was removed.");
        setBanner("");
        render();
      } catch (error) { setBanner(error.message, "error"); }
    });
    resetButton.addEventListener("click", async () => {
      const confirmed = confirmDanger(
        "Clear all captured timings?",
        "This removes the current captured timing pass from the page. Saved clip settings stay, but the current line timing work will be cleared.",
      );
      if (!confirmed) return;
      try {
        state.session = await api("/reset", { method: "POST", body: "{}" });
        state.selectedCueIndex = null;
        state.pendingCueIndex = null;
        setTimelineFeedback("Capture reset", "All captured timings were cleared for this song.");
        setBanner("");
        render();
        updateTimer();
      } catch (error) { setBanner(error.message, "error"); }
    });
    saveButton.addEventListener("click", async () => {
      try {
        const wasExternalOnly = !state.session.audio_available;
        state.session = await api("/save", { method: "POST", body: "{}" });
        state.pendingCueIndex = null;
        setTimelineFeedback("Timings saved", `Saved to ${state.session.saved_output_path}.`, "success");
        if (wasExternalOnly && state.session.audio_available) {
          setBanner("Saved. System playback audio was attached for review.");
        } else {
          setBanner("Saved.");
        }
        render();
        updateTimer();
      } catch (error) { setBanner(error.message, "error"); }
    });
    setClipStartButton.addEventListener("click", () => {
      state.clipDraftStartSeconds = currentAudioAbsoluteSeconds();
      const draftEnd = draftClipEndSeconds();
      if (draftEnd !== null && draftEnd <= state.clipDraftStartSeconds) {
        state.clipDraftEndSeconds = null;
      }
      state.clipDraftDirty = true;
      render();
    });
    setClipEndButton.addEventListener("click", () => {
      const currentSeconds = currentAudioAbsoluteSeconds();
      if (currentSeconds <= draftClipStartSeconds()) {
        setBanner("Clip end must be after clip start.", "error");
        return;
      }
      state.clipDraftEndSeconds = currentSeconds;
      state.clipDraftDirty = true;
      render();
    });
    clearClipButton.addEventListener("click", () => {
      const hasCustomClip = draftClipStartSeconds() > 0 || draftClipEndSeconds() !== null;
      if (hasCustomClip) {
        const confirmed = confirmDanger(
          "Use whole song again?",
          "This clears the current clip start/end selection in the editor. Saved timings are not changed until you click Save Clip.",
        );
        if (!confirmed) return;
      }
      state.clipDraftStartSeconds = 0;
      state.clipDraftEndSeconds = null;
      state.clipDraftDirty = true;
      render();
    });
    clipStartRange.addEventListener("input", () => {
      const value = Number(clipStartRange.value || 0);
      state.clipDraftStartSeconds = value;
      const draftEnd = draftClipEndSeconds();
      if (draftEnd !== null && draftEnd <= value) {
        state.clipDraftEndSeconds = null;
      }
      state.clipDraftDirty = true;
      previewClipPosition(value);
      render();
    });
    clipEndRange.addEventListener("input", () => {
      const value = Number(clipEndRange.value || 0);
      if (value <= draftClipStartSeconds()) {
        state.clipDraftEndSeconds = null;
      } else {
        state.clipDraftEndSeconds = value;
      }
      state.clipDraftDirty = true;
      if (value > 0) {
        previewClipPosition(value);
      }
      render();
    });
    ["pointerup", "pointercancel", "touchend", "mouseup", "change"].forEach((eventName) => {
      clipStartRange.addEventListener(eventName, endClipPreview);
      clipEndRange.addEventListener(eventName, endClipPreview);
    });
    saveClipButton.addEventListener("click", async () => {
      const confirmed = confirmDanger(
        "Save this clip?",
        "This makes the selected clip the active timing window, clears the built profile, and re-syncs saved line timings into the new clip range. Lines outside the clip are removed.",
      );
      if (!confirmed) return;
      try {
        state.session = await api("/clip", {
          method: "POST",
          body: JSON.stringify({
            clip_start_seconds: draftClipStartSeconds(),
            clip_end_seconds: draftClipEndSeconds(),
          }),
        });
        state.clipDraftDirty = false;
        state.selectedCueIndex = null;
        state.pendingCueIndex = null;
        if (state.session.audio_available) {
          audio.currentTime = clipStartSeconds();
        }
        setBanner("Clip saved and timeline synced.");
        setTimelineFeedback("Timeline synced", "Saved line timings were shifted into the new clip window. Use Reset Timings if you want to clear them.", "success");
        render();
        updateTimer();
      } catch (error) { setBanner(error.message, "error"); }
    });
    resetSavedTimingsButton.addEventListener("click", async () => {
      const confirmed = confirmDanger(
        "Reset saved timings?",
        "This deletes all saved line timings for this song. The audio stays, but the timing timeline will be cleared.",
      );
      if (!confirmed) return;
      try {
        state.session = await api("/reset-saved", { method: "POST", body: "{}" });
        state.selectedCueIndex = null;
        state.pendingCueIndex = null;
        state.clipDraftDirty = false;
        setTimelineFeedback("Saved timings cleared", "All saved line timings were removed for this song.");
        setBanner("Saved timings reset.");
        render();
        updateTimer();
      } catch (error) { setBanner(error.message, "error"); }
    });
    async function nudgeSelected(deltaSeconds) {
      if (state.selectedCueIndex === null) { setBanner("Select a timed line first.", "error"); return; }
      try {
        state.session = await api("/nudge", {
          method: "POST",
          body: JSON.stringify({ cue_index: state.selectedCueIndex, delta_seconds: deltaSeconds }),
        });
        const adjustedCue = state.session.cue_states[state.selectedCueIndex];
        setTimelineFeedback(
          "Line adjusted",
          `${cueSummary(adjustedCue)} moved to ${formatSeconds(adjustedCue.click_timestamp)}.`,
          "success",
        );
      setBanner("");
        render();
      } catch (error) { setBanner(error.message, "error"); }
    }
    nudgeBack50.addEventListener("click", () => nudgeSelected(-0.05));
    nudgeForward50.addEventListener("click", () => nudgeSelected(0.05));
    playToggleButton.addEventListener("click", async () => {
      if (!state.session?.audio_available) return;
      try {
        if (audio.paused) {
          await audio.play();
        } else {
          audio.pause();
        }
        renderPlayerBar();
      } catch (error) {
        setBanner(error.message || "Could not control playback.", "error");
      }
    });
    playerSeekRange.addEventListener("input", () => {
      if (!state.session?.audio_available) return;
      audio.currentTime = Number(playerSeekRange.value || 0);
      renderPlayerBar();
      updateTimer();
    });
    playerSpeedSelect.addEventListener("change", () => {
      audio.playbackRate = Number(playerSpeedSelect.value || 1);
    });
    ["timeupdate", "play", "pause", "seeked", "loadedmetadata"].forEach((eventName) => {
      audio.addEventListener(eventName, () => {
        const endSeconds = clipEndSeconds();
        if (endSeconds !== null && (audio.currentTime || 0) > endSeconds) {
          audio.currentTime = endSeconds;
          if (!audio.paused) {
            audio.pause();
          }
        }
        if (eventName === "play" && hasUnsavedTimings()) {
          setBanner("Reminder: click Save Timings when you finish this pass.");
        }
        if (eventName === "loadedmetadata" && clipStartSeconds() > 0 && (audio.currentTime || 0) < clipStartSeconds()) {
          audio.currentTime = clipStartSeconds();
        }
        render();
        updateTimer();
        renderPlayerBar();
      });
    });
    window.addEventListener("keydown", (event) => {
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
      if (!state.session || !state.session.started || state.session.complete || state.session.finished) return;
      currentPanel.click();
    });
    audio.playbackRate = Number(playerSpeedSelect.value || 1);
    loadCaptureDevices().catch(() => {});
    syncAudioUi();
    render();
    updateTimer();
    syncTimerLoop();
  </script>
</body>
</html>
"""

LIVE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Run Live</title>
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
    .shell { max-width: 1100px; margin: 0 auto; padding: 24px; }
    .back { color: var(--muted); text-decoration: none; }
    h1 { margin: 14px 0 10px; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 0.95; }
    .sub { color: var(--muted); max-width: 44rem; line-height: 1.5; margin-bottom: 22px; }
    .grid { display: grid; grid-template-columns: 0.95fr 1.05fr; gap: 20px; }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(75, 53, 29, 0.08);
    }
    .status { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
    .stat { border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: rgba(255,255,255,0.6); }
    .label { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; color: var(--muted); margin-bottom: 6px; }
    .value { font-size: 1.1rem; font-weight: 700; }
    form, .operator { display: grid; gap: 14px; }
    .field { display: grid; gap: 6px; }
    .field label { color: var(--muted); font-size: 0.9rem; }
    input, select, button { font: inherit; }
    input, select { width: 100%; border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; background: rgba(255,255,255,0.78); color: var(--ink); }
    .buttons { display: flex; flex-wrap: wrap; gap: 10px; }
    button { border: 0; border-radius: 12px; padding: 12px 16px; cursor: pointer; }
    .primary { background: var(--accent); color: #fff7ef; }
    .secondary { background: #eadfce; color: var(--ink); }
    .success { background: var(--accent-2); color: #f3fff7; }
    .danger { background: var(--warn); color: #fff1f1; }
    .banner { min-height: 1.4rem; margin-top: 10px; color: var(--accent-2); }
    .banner.error { color: var(--warn); }
    .signal-panel {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.68);
    }
    .signal-panel.warn {
      border-color: color-mix(in srgb, var(--warn) 45%, var(--line));
      background: #fff1ee;
    }
    .signal-panel.ok {
      border-color: color-mix(in srgb, var(--accent-2) 40%, var(--line));
      background: #eef8f1;
    }
    .signal-panel strong {
      display: block;
      margin-bottom: 6px;
    }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.88rem; line-height: 1.45; }
    .logs { max-height: 24rem; overflow: auto; border: 1px solid var(--line); border-radius: 14px; padding: 14px; background: rgba(255,255,255,0.66); }
    .targets { display: grid; gap: 10px; margin-top: 18px; max-height: 22rem; overflow: auto; }
    .target { border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; background: rgba(255,255,255,0.66); text-align: left; }
    .target.active { border-color: color-mix(in srgb, var(--accent) 55%, var(--line)); box-shadow: 0 0 0 3px rgba(163, 74, 39, 0.12); background: linear-gradient(135deg, #fff1e6 0%, #fffaf4 100%); }
    .target-meta { color: var(--muted); font-size: 0.88rem; margin-bottom: 6px; }
    details.debug { margin-top: 18px; }
    details.debug summary { cursor: pointer; color: var(--muted); }
    details.debug[open] summary { margin-bottom: 10px; }
    .debug-actions { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 12px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } .status { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <a class="back" href="/songs/">Back To Library</a>
    <h1>Run Live: __SONG_TITLE__</h1>
    <p class="sub">Start runtime, play the song, and click a lyric line when you need to re-anchor live tracking. The click sets the current lyric position, and the runtime follows timing locally from that point.</p>
    <div class="grid">
      <section class="card">
        <div class="status">
          <div class="stat"><div class="label">Runtime</div><div class="value" id="runtimeState">stopped</div></div>
          <div class="stat"><div class="label">Mode</div><div class="value" id="modeState">live</div></div>
          <div class="stat"><div class="label">Tracking</div><div class="value" id="trackingState">idle</div></div>
          <div class="stat"><div class="label">Rate</div><div class="value" id="rateState">1.000x</div></div>
          <div class="stat"><div class="label">Position</div><div class="value" id="positionState">none</div></div>
          <div class="stat"><div class="label">Matched Line</div><div class="value" id="lineState">none</div></div>
          <div class="stat"><div class="label">Shown Line</div><div class="value" id="shownLineState">none</div></div>
          <div class="stat"><div class="label">Confidence</div><div class="value" id="confidenceState">0.00</div></div>
          <div class="stat"><div class="label">Vocal</div><div class="value" id="vocalState">unknown</div></div>
          <div class="stat"><div class="label">Hold</div><div class="value" id="holdState">none</div></div>
        </div>
        <form id="startForm">
          <div class="field">
            <label for="runtimeMode">Run Mode</label>
            <select id="runtimeMode" name="runtimeMode">
              <option value="live_audio_inference" selected>Live Audio Inference</option>
              <option value="timing_only">Timing Only</option>
              <option value="capture_timings">Capture Timings</option>
            </select>
            <div class="caption" style="margin-top:6px">Use live inference for normal tracking. Switch to timing only when you want the current line anchor to drive the clock. Capture Timings opens the training page.</div>
          </div>
        <div class="field">
          <label for="device">Input Device</label>
          <select id="device" name="device"></select>
          <div class="caption" id="deviceHint" style="margin-top:6px">Use a loopback device like BlackHole for song playback matching. The laptop microphone is only for ambient mic tests.</div>
          <div class="caption" id="signalHint" style="margin-top:6px">Start the song after runtime begins and confirm this device shows a moving signal.</div>
          <div class="signal-panel" id="signalPanel">
            <strong>Waiting for signal</strong>
            <div id="signalPanelBody">Start runtime, then play the same song through the selected loopback input.</div>
          </div>
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
            <button class="secondary" type="button" id="applyModeButton">Apply Mode</button>
            <button class="danger" type="button" id="stopButton">Stop Runtime</button>
            <button class="secondary" type="button" id="refreshButton">Refresh Status</button>
          </div>
        </form>
        <div class="banner" id="banner"></div>
        <div class="label" style="margin:18px 0 10px">Lyric Lines</div>
        <div class="caption" style="margin-top:0">If lyrics drift, click the correct line once to set a new live anchor. After that, the runtime follows timing near that point instead of jumping freely across the whole song.</div>
        <div class="targets" id="targetsView"></div>
      </section>
      <section class="card">
        <details class="debug">
          <summary>Debug Details</summary>
          <div class="label" style="margin-bottom:10px">Latest Snapshot</div>
          <pre id="snapshotView">Waiting for runtime status.</pre>
          <div class="debug-actions">
            <button class="secondary" type="button" id="copyLogsButton">Copy Logs</button>
            <button class="secondary" type="button" id="copyTailButton">Copy Tail</button>
          </div>
          <div class="label" style="margin:18px 0 10px">Recent Logs</div>
          <div class="logs"><pre id="logsView">No logs yet.</pre></div>
        </details>
      </section>
    </div>
  </div>
  <script>
    const apiBase = "__API_BASE__";
    const state = {
      payload: null,
      pendingTargetSlideNumber: null,
    };
    const runtimeState = document.getElementById("runtimeState");
    const modeState = document.getElementById("modeState");
    const trackingState = document.getElementById("trackingState");
    const rateState = document.getElementById("rateState");
    const positionState = document.getElementById("positionState");
    const lineState = document.getElementById("lineState");
    const shownLineState = document.getElementById("shownLineState");
    const confidenceState = document.getElementById("confidenceState");
    const vocalState = document.getElementById("vocalState");
    const holdState = document.getElementById("holdState");
    const device = document.getElementById("device");
    const deviceHint = document.getElementById("deviceHint");
    const signalHint = document.getElementById("signalHint");
    const signalPanel = document.getElementById("signalPanel");
    const signalPanelBody = document.getElementById("signalPanelBody");
    const banner = document.getElementById("banner");
    const snapshotView = document.getElementById("snapshotView");
    const logsView = document.getElementById("logsView");
    const copyLogsButton = document.getElementById("copyLogsButton");
    const copyTailButton = document.getElementById("copyTailButton");
    const targetsView = document.getElementById("targetsView");
    const stopButton = document.getElementById("stopButton");
    const refreshButton = document.getElementById("refreshButton");
    const runtimeMode = document.getElementById("runtimeMode");
    const applyModeButton = document.getElementById("applyModeButton");
    function setBanner(message, type = "") {
      banner.textContent = message || "";
      banner.className = type ? `banner ${type}` : "banner";
    }
    async function api(path, options = {}) {
      const response = await fetch(`${apiBase}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ error: "Request failed" }));
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return response.json();
    }
    async function copyText(text) {
      if (!text) return false;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
          return true;
        }
      } catch (error) {}
      const helper = document.createElement("textarea");
      helper.value = text;
      helper.setAttribute("readonly", "readonly");
      helper.style.position = "absolute";
      helper.style.left = "-9999px";
      document.body.appendChild(helper);
      helper.select();
      try {
        const copied = document.execCommand("copy");
        document.body.removeChild(helper);
        return copied;
      } catch (error) {
        document.body.removeChild(helper);
        return false;
      }
    }
    function currentLogsText() {
      return String(logsView.textContent || "").trim();
    }
    function currentTailText(limit = 20) {
      const lines = currentLogsText().split("\\n").filter(Boolean);
      return lines.slice(-limit).join("\\n");
    }
    function selectedDeviceLooksLikeLoopback() {
      const selected = device.options[device.selectedIndex];
      if (!selected) return false;
      const label = String(selected.textContent || "").toLowerCase();
      return ["blackhole", "loopback", "soundflower", "stereo mix", "vb-audio"].some((token) => label.includes(token));
    }
    function setSignalPanel(level, title, body) {
      signalPanel.className = level ? `signal-panel ${level}` : "signal-panel";
      signalPanel.innerHTML = `<strong>${title}</strong><div id="signalPanelBody">${body}</div>`;
    }
    function modeLabel(mode) {
      if (mode === "timing_only") return "timing only";
      if (mode === "capture_timings") return "capture";
      return "live";
    }
    function bindPrimaryPress(button, handler) {
      button.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        handler();
      });
      button.addEventListener("click", (event) => {
        event.preventDefault();
      });
      button.addEventListener("dblclick", (event) => {
        event.preventDefault();
      });
      button.addEventListener("keydown", (event) => {
        if (event.code !== "Enter" && event.code !== "Space") return;
        event.preventDefault();
        handler();
      });
    }
    function render() {
      const payload = state.payload;
      if (!payload) return;
      const snapshot = payload.snapshot;
      const metrics = snapshot ? (snapshot.metrics || {}) : {};
      const usingLoopbackInput = selectedDeviceLooksLikeLoopback();
      const chunksReceived = Number(metrics.chunks_received || 0);
      const silentChunks = Number(metrics.silent_chunks || 0);
      const featureFramesProcessed = Number(metrics.feature_frames_processed || 0);
      const lowConfidenceMatches = Number(metrics.low_confidence_matches || 0);
      const acceptedMatches = Number(metrics.accepted_matches || 0);
      const slideTriggersSent = Number(metrics.slide_triggers_sent || 0);
      const totalRms = snapshot ? Number(snapshot.rms || 0) : 0;
      const activeMode = payload.live_tracking_mode || (snapshot ? snapshot.runtime_mode : null) || "live_audio_inference";
      runtimeState.textContent = payload.running ? "running" : "stopped";
      modeState.textContent = modeLabel(activeMode);
      trackingState.textContent = snapshot ? snapshot.tracking_state || "idle" : "idle";
      rateState.textContent = snapshot ? `${Number(snapshot.timeline_rate_estimate || 1).toFixed(3)}x` : "1.000x";
      positionState.textContent = snapshot && snapshot.last_match ? `${snapshot.last_match.reference_timestamp.toFixed(2)}s` : "none";
      lineState.textContent = snapshot && snapshot.matched_slide_command ? String(snapshot.matched_slide_command.slide_number) : "none";
      shownLineState.textContent = snapshot && snapshot.last_slide_command ? String(snapshot.last_slide_command.slide_number) : "none";
      confidenceState.textContent = snapshot && snapshot.last_match ? snapshot.last_match.confidence.toFixed(2) : "0.00";
      vocalState.textContent = snapshot ? (snapshot.no_vocal_detected ? "no vocal" : "voiced/ok") : "unknown";
      holdState.textContent = snapshot && snapshot.alignment_hold_active ? (snapshot.alignment_hold_reason || "hold") : "none";
      snapshotView.textContent = JSON.stringify(snapshot, null, 2);
      logsView.textContent = (payload.logs || []).join("\\n") || "No logs yet.";
      runtimeMode.value = payload.running ? activeMode : (runtimeMode.value || activeMode);
      applyModeButton.disabled = !payload.running;
      deviceHint.textContent = usingLoopbackInput
        ? "Loopback input selected. This is the right choice for matching the saved song playback."
        : "For real song matching, switch to BlackHole or another loopback input. The laptop microphone will usually not track the saved song reliably.";
      if (!payload.running) {
        signalHint.textContent = "Start the song after runtime begins and confirm this device shows a moving signal.";
        setSignalPanel(
          "",
          "Waiting for signal",
          "Start runtime, then play the same song through the selected loopback input.",
        );
      } else if (chunksReceived >= 20 && silentChunks === chunksReceived && featureFramesProcessed === 0 && totalRms === 0) {
        signalHint.textContent = "No audio is reaching this input. BlackHole is selected, but the song is not routed into it yet.";
        setSignalPanel(
          "warn",
          "No audio detected",
          "The runtime is alive, but this input is receiving silence. Route the song output into BlackHole or your loopback device, then refresh or restart runtime.",
        );
      } else if (!usingLoopbackInput && featureFramesProcessed > 0) {
        signalHint.textContent = "This is microphone or ambient input, not direct song playback. Switch to BlackHole or another loopback input.";
        setSignalPanel(
          "warn",
          "Microphone input selected",
          "Audio is reaching the matcher, but this input is not the right source for reliable song tracking. Use BlackHole or another loopback device for the saved song playback.",
        );
      } else if (
        featureFramesProcessed >= 300 &&
        slideTriggersSent === 0 &&
        lowConfidenceMatches > Math.max(acceptedMatches * 8, 120) &&
        !usingLoopbackInput
      ) {
        signalHint.textContent = "This looks like microphone input or room audio, not clean song playback. Switch to BlackHole or another loopback input.";
        setSignalPanel(
          "warn",
          "Wrong input for matching",
          "Audio is reaching the matcher, but it looks like microphone or ambient sound instead of direct song playback. Use BlackHole or another loopback device and play the same saved song feed.",
        );
      } else if (
        featureFramesProcessed >= 500 &&
        slideTriggersSent === 0 &&
        lowConfidenceMatches > Math.max(acceptedMatches * 10, 200)
      ) {
        signalHint.textContent = "Audio is reaching the matcher, but it does not match the saved reference strongly enough yet.";
        setSignalPanel(
          "warn",
          "Source does not match saved reference",
          "The runtime is hearing audio, but not locking onto the saved song. Use the same song/version, start closer to the saved section, and prefer direct loopback playback over room sound.",
        );
      } else if (chunksReceived >= 20 && silentChunks >= Math.max(10, chunksReceived * 0.9) && featureFramesProcessed === 0) {
        signalHint.textContent = "This input is almost silent. Check Multi-Output / BlackHole routing and make sure the song is actually playing.";
        setSignalPanel(
          "warn",
          "Signal is too weak",
          "A little audio is reaching the input, but not enough to match reliably. Check Mac output routing and make sure the song is playing into the loopback device.",
        );
      } else if (featureFramesProcessed > 0) {
        if (activeMode === "timing_only") {
          signalHint.textContent = "Timing only mode is active. Audio is not used for matching until you switch back to live inference.";
          setSignalPanel(
            "ok",
            "Timing only mode",
            "The runtime is following the saved lyric timing from the current anchor. Click a line to set or correct the anchor, or switch back to live inference to use audio matching again.",
          );
        } else {
          signalHint.textContent = "Audio is reaching the matcher. If tracking still does not lock, the issue is matching quality rather than device routing.";
          setSignalPanel(
            "ok",
            "Audio detected",
            "The matcher is receiving usable signal. If lyrics still do not move, the next issue is profile quality or song mismatch, not routing.",
          );
        }
      } else if (activeMode === "timing_only") {
        signalHint.textContent = "Timing only mode is active. Click a line once to set the current timing anchor.";
        setSignalPanel(
          "ok",
          "Waiting for timing anchor",
          "This mode ignores live matching. Click the correct lyric line once, and the runtime will continue using saved timing from there.",
        );
      } else {
        signalHint.textContent = "Signal is present but not stable yet. Let the song play a little longer so the matcher can search.";
        setSignalPanel(
          "",
          "Listening for a stable match",
          "Some signal is present. Let the song play a bit longer so the matcher can lock onto the saved reference.",
        );
      }
      targetsView.innerHTML = "";
      const slideTargets = payload.slide_targets || [];
      if (!slideTargets.length) {
        targetsView.textContent = "Build a profile first to load jump targets.";
        return;
      }
      slideTargets.forEach((target) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "target";
        const activeSlide = state.pendingTargetSlideNumber
          || (snapshot && snapshot.last_slide_command ? snapshot.last_slide_command.slide_number : null)
          || (snapshot && snapshot.matched_slide_command ? snapshot.matched_slide_command.slide_number : null);
        if (activeSlide === target.slide_number) button.classList.add("active");
        button.innerHTML = `<div class="target-meta">Line ${target.slide_number} · ${target.reference_timestamp.toFixed(2)}s</div><div><strong>${target.section}</strong></div><div>${target.lyrics}</div>`;
        bindPrimaryPress(button, async () => {
          if (state.pendingTargetSlideNumber !== null) return;
          state.pendingTargetSlideNumber = target.slide_number;
          setBanner(`Moving lyrics to line ${target.slide_number}...`);
          render();
          try {
            state.payload = await api("/jump", { method: "POST", body: JSON.stringify({ slide_number: target.slide_number }) });
            state.pendingTargetSlideNumber = null;
            setBanner(`Moved lyrics to line ${target.slide_number}.`);
            render();
          } catch (error) {
            state.pendingTargetSlideNumber = null;
            setBanner(error.message, "error");
            render();
          }
        });
        targetsView.appendChild(button);
      });
    }
    async function loadStatus() { state.payload = await api("/status"); render(); }
    async function loadDevices() {
      const payload = await api("/devices");
      device.innerHTML = "";
      payload.devices.forEach((item) => {
        const option = document.createElement("option");
        option.value = String(item.id);
        option.textContent = `${item.id}: ${item.name}`;
        if (item.default) option.selected = true;
        device.appendChild(option);
      });
      if (device.options.length && device.selectedIndex < 0) {
        device.selectedIndex = 0;
      }
      render();
    }
    device.addEventListener("change", () => render());
    document.getElementById("startForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (runtimeMode.value === "capture_timings") {
        window.location.href = window.location.pathname.replace(/\\/live$/, "/prepare");
        return;
      }
      try {
        state.payload = await api("/start", {
          method: "POST",
          body: JSON.stringify({
            live_tracking_mode: runtimeMode.value,
            input_device: Number(device.value),
            silence_threshold_rms: Number(document.getElementById("silenceThreshold").value),
            match_debug_logging: document.getElementById("matchDebugLogging").value === "true",
          }),
        });
        setBanner(
          runtimeMode.value === "timing_only"
            ? "Runtime started in timing only mode. Click a line to set the first anchor."
            : "Runtime started."
        );
        render();
      } catch (error) { setBanner(error.message, "error"); }
    });
    applyModeButton.addEventListener("click", async () => {
      if (!state.payload || !state.payload.running) return;
      if (runtimeMode.value === "capture_timings") {
        window.location.href = window.location.pathname.replace(/\\/live$/, "/prepare");
        return;
      }
      try {
        state.payload = await api("/mode", {
          method: "POST",
          body: JSON.stringify({ live_tracking_mode: runtimeMode.value }),
        });
        setBanner(
          runtimeMode.value === "timing_only"
            ? "Timing only mode is active."
            : "Live audio inference resumed."
        );
        render();
      } catch (error) { setBanner(error.message, "error"); }
    });
    stopButton.addEventListener("click", async () => {
      try { state.payload = await api("/stop", { method: "POST", body: "{}" }); setBanner("Runtime stopped."); render(); } catch (error) { setBanner(error.message, "error"); }
    });
    refreshButton.addEventListener("click", () => { loadStatus().catch((error) => setBanner(error.message, "error")); });
    copyLogsButton.addEventListener("click", async () => {
      const logs = currentLogsText();
      const ok = await copyText(logs);
      setBanner(ok ? "Copied logs." : "Could not copy logs.", ok ? "success" : "error");
    });
    copyTailButton.addEventListener("click", async () => {
      const tail = currentTailText(20);
      const ok = await copyText(tail);
      setBanner(ok ? "Copied last 20 log lines." : "Could not copy log tail.", ok ? "success" : "error");
    });
    loadDevices().then(loadStatus).then(render).catch((error) => setBanner(error.message, "error"));
    setInterval(() => { loadStatus().catch(() => {}); }, 750);
  </script>
</body>
</html>
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a simple local song library UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--library-root", default="library")
    return parser.parse_args()


def _song_lyrics_text(song_library, song_id: str) -> str:
    return song_library.lyrics_text(song_id)


def _song_has_lyrics(song_library, song_id: str) -> bool:
    return bool(_song_lyrics_text(song_library, song_id).strip())


def _song_status(song_library, record) -> str:
    has_audio = bool(record.reference_audio_filename)
    has_lyrics = _song_has_lyrics(song_library, record.song_id)
    has_timings = bool(record.timings_filename)
    has_profile = bool(record.profile_directory)
    if not has_audio:
        return "Needs audio"
    if not has_lyrics:
        return "Needs lyrics"
    if not has_timings:
        return "Ready to align"
    if not has_profile:
        return "Ready to build profile"
    return "Ready for live"


def _songs_payload(song_library, records: tuple) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for record in records:
        has_audio = bool(record.reference_audio_filename)
        has_lyrics = _song_has_lyrics(song_library, record.song_id)
        has_timings = bool(record.timings_filename)
        has_profile = bool(record.profile_directory)
        payload.append(
            {
                "song_id": record.song_id,
                "title": record.title,
                "status": _song_status(song_library, record),
                "updated_at": record.updated_at,
                "has_audio": has_audio,
                "has_lyrics": has_lyrics,
                "has_timings": has_timings,
                "has_profile": has_profile,
                "can_build_profile": bool(has_timings and has_audio),
                "can_run_live": has_profile,
            }
        )
    return payload


def _song_state_payload(song_library, song_id: str, *, system_capture_active: bool, system_capture_device_name: str | None) -> dict[str, object]:
    record = song_library.get_song(song_id)
    lyrics_text = _song_lyrics_text(song_library, song_id)
    has_audio = bool(record.reference_audio_filename)
    has_lyrics = bool(lyrics_text.strip())
    has_timings = bool(record.timings_filename)
    has_profile = bool(record.profile_directory)
    return {
        "song_id": record.song_id,
        "title": record.title,
        "status": _song_status(song_library, record),
        "lyrics_text": lyrics_text,
        "has_audio": has_audio,
        "has_lyrics": has_lyrics,
        "has_timings": has_timings,
        "has_profile": has_profile,
        "updated_at": record.updated_at,
        "system_capture_active": system_capture_active,
        "system_capture_device_name": system_capture_device_name,
    }


def _saved_slide_targets(song_library, song_id: str) -> list[dict[str, object]]:
    timings_path = song_library.timings_path(song_id)
    if timings_path.exists():
        content = json.loads(timings_path.read_text(encoding="utf-8"))
        if isinstance(content, list):
            targets: list[dict[str, object]] = []
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                slide_number = entry.get("slide_number")
                section = entry.get("section")
                lyrics = entry.get("lyrics")
                click_timestamp = entry.get("click_timestamp")
                reference_timestamp = entry.get("reference_timestamp")
                timestamp = None
                if isinstance(click_timestamp, (int, float)):
                    timestamp = float(click_timestamp)
                elif isinstance(reference_timestamp, (int, float)):
                    timestamp = float(reference_timestamp)
                if isinstance(slide_number, bool) or not isinstance(slide_number, int):
                    continue
                if not isinstance(section, str) or not section.strip():
                    continue
                if not isinstance(lyrics, str) or not lyrics.strip():
                    continue
                if timestamp is None:
                    continue
                targets.append(
                    {
                        "slide_number": slide_number,
                        "section": section.strip(),
                        "lyrics": lyrics.strip(),
                        "reference_timestamp": timestamp,
                    }
                )
            if targets:
                return targets

    song = song_library.get_song(song_id)
    if not song.profile_directory:
        return []
    profile_path = song_library.profile_path(song_id) / "profile.json"
    if not profile_path.exists():
        return []
    content = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        return []
    slides = content.get("slides")
    if not isinstance(slides, list):
        return []
    targets: list[dict[str, object]] = []
    seen_slide_numbers: set[int] = set()
    for entry in slides:
        if not isinstance(entry, dict):
            continue
        slide_number = entry.get("slide_number")
        section = entry.get("section")
        lyrics = entry.get("lyrics")
        reference_timestamp = entry.get("reference_timestamp")
        if isinstance(slide_number, bool) or not isinstance(slide_number, int):
            continue
        if not isinstance(section, str) or not section.strip():
            continue
        if not isinstance(lyrics, str) or not lyrics.strip():
            continue
        if not isinstance(reference_timestamp, (int, float)):
            continue
        if slide_number in seen_slide_numbers:
            continue
        seen_slide_numbers.add(slide_number)
        targets.append(
            {
                "slide_number": slide_number,
                "section": section.strip(),
                "lyrics": lyrics.strip(),
                "reference_timestamp": float(reference_timestamp),
            }
        )
    return targets


def _slide_entries_from_script_cues(script_cues) -> list[dict[str, object]]:
    if not script_cues:
        return []
    entries: list[dict[str, object]] = []
    start_index = 0
    current_slide_number = script_cues[0].slide_number
    for index in range(1, len(script_cues) + 1):
        at_group_end = index == len(script_cues) or script_cues[index].slide_number != current_slide_number
        if not at_group_end:
            continue
        grouped_cues = script_cues[start_index:index]
        entries.append(
            {
                "slide_number": current_slide_number,
                "section": grouped_cues[0].section,
                "lyrics": "\n".join(cue.lyrics for cue in grouped_cues),
            }
        )
        if index < len(script_cues):
            start_index = index
            current_slide_number = script_cues[index].slide_number
    return entries


def _line_slide_entries_from_script_cues(script_cues) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index, cue in enumerate(script_cues, start=1):
        entries.append(
            {
                "slide_number": index,
                "section": cue.section,
                "lyrics": cue.lyrics,
            }
        )
    return entries


def _retarget_script_cues(script_cues, cue_index: int):
    if cue_index <= 0 or cue_index >= len(script_cues):
        raise ValueError("cue_index must point to a line after the first line")
    breakpoints = {
        index
        for index in range(1, len(script_cues))
        if script_cues[index].slide_number != script_cues[index - 1].slide_number
    }
    if cue_index in breakpoints:
        breakpoints.remove(cue_index)
    else:
        breakpoints.add(cue_index)
    updated_cues = []
    next_breakpoints = sorted(breakpoints)
    breakpoint_cursor = 0
    slide_number = 1
    for index, cue in enumerate(script_cues):
        if index > 0 and breakpoint_cursor < len(next_breakpoints) and next_breakpoints[breakpoint_cursor] == index:
            slide_number += 1
            breakpoint_cursor += 1
        updated_cues.append(
            type(cue)(
                slide_number=slide_number,
                section=cue.section,
                lyrics=cue.lyrics,
                line_number=cue.line_number,
                line_count=cue.line_count,
            )
        )
    return tuple(updated_cues)


def _line_by_line_script_cues(script_cues):
    updated_cues = []
    for index, cue in enumerate(script_cues, start=1):
        updated_cues.append(
            type(cue)(
                slide_number=index,
                section=cue.section,
                lyrics=cue.lyrics,
                line_number=1,
                line_count=1,
            )
        )
    return tuple(updated_cues)


def _rewrite_saved_timings_for_script(song_library, song_id: str, script_cues, cue_states) -> None:
    payload: list[dict[str, object]] = []
    for cue, cue_state in zip(script_cues, cue_states, strict=True):
        timestamp = cue_state.get("click_timestamp")
        if not isinstance(timestamp, (int, float)):
            continue
        payload.append(
            {
                "slide_number": cue.slide_number,
                "section": cue.section,
                "lyrics": cue.lyrics,
                "line_number": cue.line_number,
                "line_count": cue.line_count,
                "click_timestamp": float(timestamp),
            }
        )
    timings_path = song_library.timings_path(song_id)
    if payload:
        timings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        song_library.update_timings(song_id)
    elif timings_path.exists():
        timings_path.unlink()
        song_library.reset_timings(song_id)


def _parse_multipart(headers: dict[str, str], body: bytes) -> dict[str, tuple[str | None, bytes]]:
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("request must be multipart/form-data")
    parsed = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    output: dict[str, tuple[str | None, bytes]] = {}
    for part in parsed.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        output[name] = (filename, payload)
    return output


def _session_payload(
    session,
    output_path: Path,
    audio_path: Path | None,
    song_record,
    *,
    system_capture_active: bool = False,
    system_capture_device_name: str | None = None,
) -> dict[str, object]:
    next_index = session.current_index()
    next_cue = session.script_cues[next_index] if next_index < len(session.script_cues) else None
    return {
        "cue_states": list(session.cue_states()),
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
        "current_index": session.current_index(),
        "finished": session.finished,
        "started_at_epoch_seconds": session.started_at_wall_seconds,
        "saved_output_path": str(output_path),
        "audio_available": audio_path is not None and audio_path.exists(),
        "clip_start_seconds": song_record.clip_start_seconds,
        "clip_end_seconds": song_record.clip_end_seconds,
        "system_capture_active": system_capture_active,
        "system_capture_device_name": system_capture_device_name,
    }


def _load_prepare_session(song_library, song_id: str):
    from lyrics_aligner.application.click_capture import ClickCaptureSession, detect_onset_candidates, load_slide_script

    slides_path = song_library.slides_path(song_id)
    audio_path = song_library.audio_path(song_id)
    timings_path = song_library.timings_path(song_id)
    onset_candidates: tuple[float, ...] = ()
    reference_audio_path = song_library.reference_audio_path(song_id)
    if reference_audio_path is not None and reference_audio_path.suffix.lower() == ".wav":
        try:
            onset_candidates = detect_onset_candidates(str(reference_audio_path))
        except (OSError, ValueError):
            onset_candidates = ()
    script_cues = load_slide_script(str(slides_path))
    session = ClickCaptureSession(script_cues, onset_candidates=onset_candidates)
    if timings_path.exists():
        content = json.loads(timings_path.read_text(encoding="utf-8"))
        if isinstance(content, list) and content:
            session.start()
            for fallback_index, entry in enumerate(content):
                if not isinstance(entry, dict):
                    continue
                timestamp = entry.get("click_timestamp")
                if isinstance(timestamp, (int, float)):
                    cue_index = _find_saved_cue_index(script_cues, entry, fallback_index)
                    session.set_anchor(cue_index, elapsed=float(timestamp))
            session.stop()
    return session


def _find_saved_cue_index(script_cues, entry: dict[str, object], fallback_index: int) -> int:
    slide_number = entry.get("slide_number")
    section = entry.get("section")
    lyrics = entry.get("lyrics")
    line_number = entry.get("line_number")
    for index, cue in enumerate(script_cues):
        if isinstance(slide_number, int) and cue.slide_number != slide_number:
            continue
        if isinstance(section, str) and cue.section != section:
            continue
        if isinstance(lyrics, str) and cue.lyrics != lyrics:
            continue
        if isinstance(line_number, int) and cue.line_number != line_number:
            continue
        return index
    return fallback_index


@dataclass(frozen=True, slots=True)
class InputDeviceInfo:
    id: int
    name: str
    default: bool


@dataclass(frozen=True, slots=True)
class LoopbackCaptureConfig:
    device_id: int
    device_name: str
    sample_rate: int
    channels: int


class SystemPlaybackRecorder:
    def __init__(self, config: LoopbackCaptureConfig, output_path: Path) -> None:
        self._config = config
        self._output_path = output_path
        self._lock = Lock()
        self._stop_requested = Event()
        self._ready = Event()
        self._thread: Thread | None = None
        self._wave_handle: wave.Wave_write | None = None
        self._error: str | None = None

    @property
    def device_name(self) -> str:
        return self._config.device_name

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._ready.clear()
        self._error = None
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = Thread(target=self._run, name="system-playback-recorder", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)
        if self._error is not None:
            raise ValueError(self._error)

    def stop(self) -> bytes:
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._error is not None:
            raise ValueError(self._error)
        if not self._output_path.exists():
            return b""
        payload = self._output_path.read_bytes()
        self._output_path.unlink()
        return payload

    def discard(self) -> None:
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._output_path.exists():
            self._output_path.unlink()

    def _run(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-untyped]

            with wave.open(str(self._output_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(self._config.sample_rate)
                self._wave_handle = handle
                with sd.InputStream(
                    samplerate=self._config.sample_rate,
                    channels=self._config.channels,
                    dtype="float32",
                    device=self._config.device_id,
                    callback=self._on_audio,
                ):
                    self._ready.set()
                    while not self._stop_requested.is_set():
                        sleep(0.05)
        except Exception as error:
            self._error = f"System playback recording failed: {error}"
            self._ready.set()
        finally:
            with self._lock:
                self._wave_handle = None

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del frames, time_info, status
        if self._stop_requested.is_set():
            return
        samples = np.asarray(indata, dtype=np.float32)
        mono = np.mean(samples, axis=1) if samples.ndim == 2 else samples
        pcm16 = np.clip(mono, -1.0, 1.0)
        payload = (pcm16 * 32767.0).astype(np.int16).tobytes()
        with self._lock:
            if self._wave_handle is not None:
                self._wave_handle.writeframes(payload)


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
    def __init__(self, logger_name: str) -> None:
        self._runtime = None
        self._thread: Thread | None = None
        self._report = None
        self._error: str | None = None
        self._slide_targets: list[dict[str, object]] = []
        self._live_tracking_mode = "live_audio_inference"
        self._lock = Lock()
        self.status_collector = StatusCollector()
        self.log_buffer = LogBuffer()
        self.log_buffer.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        self.logger = logging.getLogger(logger_name)
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
            self._live_tracking_mode = getattr(
                config,
                "live_tracking_mode",
                "live_audio_inference",
            )
            runtime = build_runtime(
                config,
                self.logger,
                manual_override_controller=ManualOverrideController(),
                status_observer=self.status_collector,
            )
            self._runtime = runtime
            seen_slide_numbers: set[int] = set()
            self._slide_targets = []
            for command in runtime.operator_slide_targets():
                if command.slide_number in seen_slide_numbers:
                    continue
                seen_slide_numbers.add(command.slide_number)
                self._slide_targets.append(
                    {
                        "slide_number": command.slide_number,
                        "section": command.section,
                        "lyrics": command.lyrics,
                        "reference_timestamp": command.reference_timestamp,
                    }
                )

            def run_runtime() -> None:
                try:
                    report = runtime.run()
                    with self._lock:
                        self._report = report
                except Exception as error:  # pragma: no cover
                    self.logger.exception("Runtime failed")
                    with self._lock:
                        self._error = str(error)

            self._thread = Thread(target=run_runtime, name="song-library-runtime", daemon=True)
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

    def live_tracking_mode(self) -> str:
        with self._lock:
            runtime = self._runtime
            if runtime is not None:
                return runtime.live_tracking_mode()
            return self._live_tracking_mode

    def set_live_tracking_mode(self, mode: str) -> str:
        with self._lock:
            runtime = self._runtime
            self._live_tracking_mode = mode
        if runtime is None:
            return mode
        updated_mode = runtime.set_live_tracking_mode(mode)
        with self._lock:
            self._live_tracking_mode = updated_mode
        return updated_mode

    def report(self):
        with self._lock:
            return self._report

    def error(self) -> str | None:
        with self._lock:
            return self._error

    def slide_targets(self) -> list[dict[str, object]]:
        with self._lock:
            return list(self._slide_targets)


def _input_devices() -> list[InputDeviceInfo]:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except Exception:
        return []
    devices = sd.query_devices()
    preferred_system_device = _preferred_system_playback_device()
    preferred_input_index = None if preferred_system_device is None else preferred_system_device.device_id
    default_input_index = preferred_input_index if preferred_input_index is not None else sd.default.device[0]
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


def _preferred_system_playback_device() -> LoopbackCaptureConfig | None:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except Exception:
        return None
    devices = sd.query_devices()
    preferred_tokens = ("blackhole", "loopback", "soundflower", "stereo mix", "vb-audio")
    for index, raw in enumerate(devices):
        max_input_channels = int(raw.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        device_name = str(raw.get("name", f"Device {index}"))
        sample_rate = int(float(raw.get("default_samplerate", 44_100)))
        config = LoopbackCaptureConfig(
            device_id=index,
            device_name=device_name,
            sample_rate=sample_rate,
            channels=min(max_input_channels, 2),
        )
        normalized_name = device_name.lower()
        if any(token in normalized_name for token in preferred_tokens):
            return config
    return None


def _capture_device_config(device_id: int | None) -> LoopbackCaptureConfig | None:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except Exception:
        return None
    if device_id is None:
        return _preferred_system_playback_device()
    devices = sd.query_devices()
    if device_id < 0 or device_id >= len(devices):
        raise ValueError("selected input device is out of range")
    raw = devices[device_id]
    max_input_channels = int(raw.get("max_input_channels", 0))
    if max_input_channels <= 0:
        raise ValueError("selected device does not support audio input")
    return LoopbackCaptureConfig(
        device_id=device_id,
        device_name=str(raw.get("name", f"Device {device_id}")),
        sample_rate=int(float(raw.get("default_samplerate", 44_100))),
        channels=min(max_input_channels, 2),
    )


def _snapshot_to_dict(snapshot) -> dict[str, object] | None:
    if snapshot is None:
        return None
    return asdict(snapshot)


AUTO_PROFILE_CLIP_TAIL_SECONDS = 2.0
AUTO_PROFILE_CLIP_FULL_AUDIO_MARGIN_SECONDS = 5.0
AUTO_PROFILE_CLIP_MIN_START_SECONDS = 0.5


def _profile_has_coarse_artifacts(profile_path: Path) -> bool:
    return all(
        (profile_path / name).exists()
        for name in (
            "profile.json",
            "reference_features.npy",
            "coarse_signatures.npy",
            "coarse_timestamps.npy",
            "coarse_frame_indexes.npy",
        )
    )


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _timing_timestamps_seconds(path: Path) -> list[float]:
    if not path.exists():
        return []
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        return []
    timestamps: list[float] = []
    for entry in content:
        if not isinstance(entry, dict):
            continue
        timestamp = entry.get("click_timestamp")
        if not isinstance(timestamp, (int, float)) or timestamp < 0:
            continue
        timestamps.append(float(timestamp))
    return sorted(timestamps)


def _suggest_profile_clip_range(
    audio_path: Path,
    timings_path: Path,
) -> tuple[float, float] | None:
    if audio_path.suffix.lower() != ".wav" or not audio_path.exists():
        return None
    timestamps = _timing_timestamps_seconds(timings_path)
    if not timestamps:
        return None
    audio_duration_seconds = _wav_duration_seconds(audio_path)
    clip_start_seconds = timestamps[0]
    clip_end_seconds = min(
        audio_duration_seconds,
        timestamps[-1] + AUTO_PROFILE_CLIP_TAIL_SECONDS,
    )
    if clip_end_seconds <= clip_start_seconds:
        return None
    if (
        clip_start_seconds <= AUTO_PROFILE_CLIP_MIN_START_SECONDS
        and clip_end_seconds >= audio_duration_seconds - AUTO_PROFILE_CLIP_FULL_AUDIO_MARGIN_SECONDS
    ):
        return None
    return round(clip_start_seconds, 3), round(clip_end_seconds, 3)


def _build_reference_profile_for_song(song_library, song_id: str) -> None:
    from lyrics_aligner.adapters.features import OnnxFeatureExtractor, OnnxFeatureExtractorConfig
    from lyrics_aligner.application.reference_builder import ReferenceBuilderConfig, ReferenceProfileBuilder

    song = song_library.get_song(song_id)
    if not song.timings_filename:
        raise ValueError("prepare and save timings before building a profile")
    source_reference_audio_path = song_library.source_reference_audio_path(song_id)
    if (
        source_reference_audio_path is not None
        and not song.clip_reference_audio_filename
    ):
        suggested_clip_range = _suggest_profile_clip_range(
            source_reference_audio_path,
            song_library.timings_path(song_id),
        )
        if suggested_clip_range is not None:
            clip_start_seconds, clip_end_seconds = suggested_clip_range
            song = song_library.update_clip(
                song_id,
                clip_start_seconds=clip_start_seconds,
                clip_end_seconds=clip_end_seconds,
            )
    reference_audio_path = song_library.reference_audio_path(song_id)
    if reference_audio_path is None or not reference_audio_path.exists():
        raise ValueError("upload reference audio before building a profile")
    model_path = ROOT / "models" / "wav2vec2-base.onnx"
    if not model_path.exists():
        raise ValueError(f"feature model not found: {model_path}")

    builder = ReferenceProfileBuilder(
        feature_extractor=OnnxFeatureExtractor(
            OnnxFeatureExtractorConfig(
                model_path=str(model_path),
                sample_rate=16_000,
                append_pitch_feature=True,
            )
        ),
        config=ReferenceBuilderConfig(sample_rate=16_000, block_size=4_096),
    )
    profile = builder.build(
        audio_path=str(reference_audio_path),
        slides_path=str(song_library.timings_path(song_id)),
        profile_name=song.title,
        audio_role="mixed",
    )
    builder.save(profile, str(song_library.profile_path(song_id)))
    song_library.update_profile_directory(song_id)


def main() -> None:
    from lyrics_aligner.application.song_library import SongLibrary
    from lyrics_aligner.application.click_capture import load_slide_script, write_captured_slide_cues
    from lyrics_aligner.config import AppConfig

    args = _parse_args()
    song_library = SongLibrary(str(Path(args.library_root).expanduser().resolve()))
    sessions: dict[str, object] = {}
    runtime_managers: dict[str, RuntimeManager] = {}
    system_recorders: dict[str, SystemPlaybackRecorder] = {}

    def prepare_session(song_id: str):
        session = sessions.get(song_id)
        if session is None:
            session = _load_prepare_session(song_library, song_id)
            sessions[song_id] = session
        return session

    def stop_system_recorder(song_id: str, *, keep_audio: bool) -> bool:
        recorder = system_recorders.pop(song_id, None)
        if recorder is None:
            return False
        if not keep_audio:
            recorder.discard()
            return False
        audio_bytes = recorder.stop()
        if not audio_bytes:
            return False
        song_library.attach_audio(
            song_id,
            audio_filename="system_playback.wav",
            audio_bytes=audio_bytes,
        )
        return True

    def ensure_system_recorder(song_id: str, *, device_id: int | None = None) -> SystemPlaybackRecorder | None:
        recorder = system_recorders.get(song_id)
        if recorder is not None:
            return recorder
        config = _capture_device_config(device_id)
        if config is None:
            raise ValueError(
                "No input device found for system playback capture. Route audio to BlackHole or another loopback device first."
            )
        recorder = SystemPlaybackRecorder(
            config,
            song_library.song_path(song_id) / ".system_playback_capture.wav",
        )
        recorder.start()
        system_recorders[song_id] = recorder
        return recorder

    def system_capture_state(song_id: str) -> tuple[bool, str | None]:
        recorder = system_recorders.get(song_id)
        if recorder is None:
            return False, None
        return True, recorder.device_name

    def runtime_manager(song_id: str) -> RuntimeManager:
        manager = runtime_managers.get(song_id)
        if manager is None:
            manager = RuntimeManager(f"song-library-runtime-{song_id}")
            runtime_managers[song_id] = manager
        return manager

    class Handler(BaseHTTPRequestHandler):
        def _read_body(self) -> bytes:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return b""
            return self.rfile.read(content_length)

        def _write_json(self, payload: dict[str, object], status: HTTPStatus) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_html(self, content: str) -> None:
            data = content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location: str) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", location)
            self.end_headers()

        def _write_file(self, path: Path) -> None:
            content_type, _ = mimetypes.guess_type(path.name)
            file_size = path.stat().st_size
            range_header = self.headers.get("Range", "").strip()
            start = 0
            end = file_size - 1
            status = HTTPStatus.OK

            if range_header.startswith("bytes="):
                try:
                    spec = range_header.removeprefix("bytes=")
                    start_text, end_text = spec.split("-", 1)
                    if start_text:
                        start = int(start_text)
                    if end_text:
                        end = int(end_text)
                    if start < 0 or end < start or start >= file_size:
                        raise ValueError
                    end = min(end, file_size - 1)
                    status = HTTPStatus.PARTIAL_CONTENT
                except ValueError:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return

            content_length = (end - start) + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            with path.open("rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(content_length))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            if parsed.path in {"/", "/songs", "/songs/"}:
                self._write_html(
                    LIBRARY_HTML.replace(
                        "__SONGS__",
                        json.dumps(_songs_payload(song_library, song_library.list_songs())),
                    )
                )
                return
            if len(parts) == 2 and parts[0] == "songs":
                song = song_library.get_song(parts[1])
                page = (
                    SONG_HTML.replace("__SONG_TITLE__", song.title)
                    .replace("__SONG_ID__", song.song_id)
                    .replace(
                        "__SONG_STATE__",
                        json.dumps(
                            _song_state_payload(
                                song_library,
                                song.song_id,
                                system_capture_active=system_capture_state(song.song_id)[0],
                                system_capture_device_name=system_capture_state(song.song_id)[1],
                            )
                        ),
                    )
                    .replace("__API_BASE__", f"/songs/{parts[1]}/song-api")
                )
                self._write_html(page)
                return
            if len(parts) == 3 and parts[0] == "songs" and parts[2] == "prepare":
                if not _song_has_lyrics(song_library, parts[1]):
                    self._redirect(f"/songs/{parts[1]}")
                    return
                song = song_library.get_song(parts[1])
                session = prepare_session(parts[1])
                page = (
                    PREPARE_HTML.replace("__SONG_TITLE__", song.title)
                    .replace(
                        "__BOOTSTRAP_SESSION__",
                        json.dumps(
                            _session_payload(
                                session,
                                song_library.timings_path(parts[1]),
                                song_library.audio_path(parts[1]),
                                song,
                                system_capture_active=system_capture_state(parts[1])[0],
                                system_capture_device_name=system_capture_state(parts[1])[1],
                            )
                        ),
                    )
                    .replace("__API_BASE__", f"/songs/{parts[1]}/api")
                )
                self._write_html(page)
                return
            if len(parts) == 3 and parts[0] == "songs" and parts[2] == "lyrics":
                song = song_library.get_song(parts[1])
                lyrics = song_library.lyrics_text(parts[1])
                page = (
                    LYRICS_HTML.replace("__SONG_TITLE__", song.title)
                    .replace("__LYRICS__", lyrics)
                    .replace("__SONG_ID__", song.song_id)
                    .replace("__API_BASE__", f"/songs/{parts[1]}/api")
                )
                self._write_html(page)
                return
            if len(parts) == 3 and parts[0] == "songs" and parts[2] == "live":
                song = song_library.get_song(parts[1])
                if not song.profile_directory:
                    self._write_json({"error": "build the profile before running live"}, HTTPStatus.BAD_REQUEST)
                    return
                page = (
                    LIVE_HTML.replace("__SONG_TITLE__", song.title)
                    .replace("__API_BASE__", f"/songs/{parts[1]}/live-api")
                )
                self._write_html(page)
                return
            if len(parts) == 4 and parts[0] == "songs" and parts[2] == "api" and parts[3] == "session":
                song_id = parts[1]
                session = prepare_session(song_id)
                song = song_library.get_song(song_id)
                self._write_json(
                    _session_payload(
                        session,
                        song_library.timings_path(song_id),
                        song_library.audio_path(song_id),
                        song,
                        system_capture_active=system_capture_state(song_id)[0],
                        system_capture_device_name=system_capture_state(song_id)[1],
                    ),
                    HTTPStatus.OK,
                )
                return
            if len(parts) == 4 and parts[0] == "songs" and parts[2] == "api" and parts[3] == "audio":
                audio_path = song_library.audio_path(parts[1])
                if audio_path is None or not audio_path.exists():
                    self._write_json({"error": "song does not have uploaded audio"}, HTTPStatus.NOT_FOUND)
                    return
                self._write_file(audio_path)
                return
            if len(parts) == 4 and parts[0] == "songs" and parts[2] == "api" and parts[3] == "devices":
                self._write_json({"devices": [asdict(device) for device in _input_devices()]}, HTTPStatus.OK)
                return
            if len(parts) == 4 and parts[0] == "songs" and parts[2] == "song-api" and parts[3] == "state":
                self._write_json(
                    _song_state_payload(
                        song_library,
                        parts[1],
                        system_capture_active=system_capture_state(parts[1])[0],
                        system_capture_device_name=system_capture_state(parts[1])[1],
                    ),
                    HTTPStatus.OK,
                )
                return
            if len(parts) == 4 and parts[0] == "songs" and parts[2] == "song-api" and parts[3] == "audio":
                audio_path = song_library.audio_path(parts[1])
                if audio_path is None or not audio_path.exists():
                    self._write_json({"error": "song does not have uploaded audio"}, HTTPStatus.NOT_FOUND)
                    return
                self._write_file(audio_path)
                return
            if len(parts) == 4 and parts[0] == "songs" and parts[2] == "song-api" and parts[3] == "devices":
                self._write_json({"devices": [asdict(device) for device in _input_devices()]}, HTTPStatus.OK)
                return
            if len(parts) == 4 and parts[0] == "songs" and parts[2] == "live-api" and parts[3] == "status":
                manager = runtime_manager(parts[1])
                slide_targets = manager.slide_targets() or _saved_slide_targets(song_library, parts[1])
                self._write_json(
                    {
                        "running": manager.running(),
                        "live_tracking_mode": manager.live_tracking_mode(),
                        "snapshot": _snapshot_to_dict(manager.status_collector.snapshot()),
                        "report": None if manager.report() is None else asdict(manager.report()),
                        "error": manager.error(),
                        "slide_targets": slide_targets,
                        "logs": manager.log_buffer.messages()[-80:],
                    },
                    HTTPStatus.OK,
                )
                return
            if len(parts) == 4 and parts[0] == "songs" and parts[2] == "live-api" and parts[3] == "devices":
                self._write_json({"devices": [asdict(device) for device in _input_devices()]}, HTTPStatus.OK)
                return
            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            try:
                if parsed.path == "/api/songs":
                    fields = _parse_multipart(dict(self.headers.items()), self._read_body())
                    title = fields.get("title", (None, b""))[1].decode("utf-8").strip()
                    lyrics = fields.get("lyrics", (None, b""))[1].decode("utf-8").strip()
                    audio_filename, audio_bytes = fields.get("audio", ("", b""))
                    record = song_library.create_song(
                        title=title,
                        lyrics_text=lyrics,
                        audio_filename=audio_filename,
                        audio_bytes=audio_bytes,
                    )
                    sessions.pop(record.song_id, None)
                    self._write_json(
                        {
                            "song": {"song_id": record.song_id},
                            "songs": _songs_payload(song_library, song_library.list_songs()),
                        },
                        HTTPStatus.OK,
                    )
                    return
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "songs" and parts[3] == "build-profile":
                    song_id = parts[2]
                    _build_reference_profile_for_song(song_library, song_id)
                    self._write_json(
                        {"songs": _songs_payload(song_library, song_library.list_songs())},
                        HTTPStatus.OK,
                    )
                    return
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "songs" and parts[3] == "audio":
                    song_id = parts[2]
                    stop_system_recorder(song_id, keep_audio=False)
                    manager = runtime_managers.pop(song_id, None)
                    if manager is not None and manager.running():
                        manager.stop()
                    fields = _parse_multipart(dict(self.headers.items()), self._read_body())
                    audio_filename, audio_bytes = fields.get("audio", ("", b""))
                    song_library.attach_audio(
                        song_id,
                        audio_filename=audio_filename,
                        audio_bytes=audio_bytes,
                    )
                    sessions.pop(song_id, None)
                    self._write_json(
                        {"songs": _songs_payload(song_library, song_library.list_songs())},
                        HTTPStatus.OK,
                    )
                    return
                if len(parts) == 5 and parts[0] == "api" and parts[1] == "songs" and parts[3] == "audio" and parts[4] == "delete":
                    song_id = parts[2]
                    stop_system_recorder(song_id, keep_audio=False)
                    manager = runtime_managers.pop(song_id, None)
                    if manager is not None and manager.running():
                        manager.stop()
                    sessions.pop(song_id, None)
                    song_library.delete_audio(song_id)
                    self._write_json(
                        {"songs": _songs_payload(song_library, song_library.list_songs())},
                        HTTPStatus.OK,
                    )
                    return
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "songs" and parts[3] == "delete":
                    song_id = parts[2]
                    stop_system_recorder(song_id, keep_audio=False)
                    manager = runtime_managers.pop(song_id, None)
                    if manager is not None and manager.running():
                        manager.stop()
                    sessions.pop(song_id, None)
                    song_library.delete_song(song_id)
                    self._write_json(
                        {"songs": _songs_payload(song_library, song_library.list_songs())},
                        HTTPStatus.OK,
                    )
                    return
                if len(parts) == 4 and parts[0] == "songs" and parts[2] == "song-api":
                    song_id = parts[1]
                    action = parts[3]
                    if action == "audio":
                        stop_system_recorder(song_id, keep_audio=False)
                        manager = runtime_managers.pop(song_id, None)
                        if manager is not None and manager.running():
                            manager.stop()
                        fields = _parse_multipart(dict(self.headers.items()), self._read_body())
                        audio_filename, audio_bytes = fields.get("audio", ("", b""))
                        song_library.attach_audio(
                            song_id,
                            audio_filename=audio_filename,
                            audio_bytes=audio_bytes,
                        )
                        sessions.pop(song_id, None)
                    else:
                        body = self._read_body()
                        payload = {} if not body else json.loads(body.decode("utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError("request body must be a JSON object")
                        if action == "start-listening":
                            input_device = payload.get("input_device")
                            if input_device is not None and (
                                isinstance(input_device, bool) or not isinstance(input_device, int)
                            ):
                                raise ValueError("input_device must be an integer or null")
                            ensure_system_recorder(song_id, device_id=input_device)
                        elif action == "stop-listening":
                            if not stop_system_recorder(song_id, keep_audio=True):
                                raise ValueError("no active listening session to stop")
                            sessions.pop(song_id, None)
                        elif action == "audio-delete":
                            stop_system_recorder(song_id, keep_audio=False)
                            manager = runtime_managers.pop(song_id, None)
                            if manager is not None and manager.running():
                                manager.stop()
                            sessions.pop(song_id, None)
                            song_library.delete_audio(song_id)
                        elif action == "lyrics":
                            lyrics = payload.get("lyrics")
                            if not isinstance(lyrics, str):
                                raise ValueError("lyrics must be a string")
                            stop_system_recorder(song_id, keep_audio=False)
                            song_library.update_lyrics(song_id, lyrics)
                            sessions.pop(song_id, None)
                        elif action == "build-profile":
                            _build_reference_profile_for_song(song_library, song_id)
                        else:
                            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                            return
                    self._write_json(
                        _song_state_payload(
                            song_library,
                            song_id,
                            system_capture_active=system_capture_state(song_id)[0],
                            system_capture_device_name=system_capture_state(song_id)[1],
                        ),
                        HTTPStatus.OK,
                    )
                    return
                if len(parts) == 4 and parts[0] == "songs" and parts[2] == "api":
                    song_id = parts[1]
                    session = prepare_session(song_id)
                    action = parts[3]
                    body = self._read_body()
                    payload = {} if not body else json.loads(body.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("request body must be a JSON object")
                    if action == "start":
                        input_device = payload.get("input_device")
                        if input_device is not None and (
                            isinstance(input_device, bool) or not isinstance(input_device, int)
                        ):
                            raise ValueError("input_device must be an integer or null")
                        ensure_system_recorder(song_id, device_id=input_device)
                        session.start()
                    elif action == "clip":
                        clip_start_seconds = payload.get("clip_start_seconds")
                        clip_end_seconds = payload.get("clip_end_seconds")
                        if not isinstance(clip_start_seconds, (int, float)):
                            raise ValueError("clip_start_seconds must be a number")
                        if clip_end_seconds is not None and not isinstance(clip_end_seconds, (int, float)):
                            raise ValueError("clip_end_seconds must be a number or null")
                        stop_system_recorder(song_id, keep_audio=False)
                        song_library.update_clip(
                            song_id,
                            clip_start_seconds=float(clip_start_seconds),
                            clip_end_seconds=None if clip_end_seconds is None else float(clip_end_seconds),
                        )
                        sessions.pop(song_id, None)
                        session = prepare_session(song_id)
                    elif action == "start-from":
                        input_device = payload.get("input_device")
                        if input_device is not None and (
                            isinstance(input_device, bool) or not isinstance(input_device, int)
                        ):
                            raise ValueError("input_device must be an integer or null")
                        ensure_system_recorder(song_id, device_id=input_device)
                        cue_index = payload.get("cue_index")
                        elapsed = payload.get("elapsed")
                        if isinstance(cue_index, bool) or not isinstance(cue_index, int):
                            raise ValueError("cue_index must be an integer")
                        if not isinstance(elapsed, (int, float)):
                            raise ValueError("elapsed must be a number")
                        session.start_from(cue_index, elapsed=float(elapsed))
                        write_captured_slide_cues(str(song_library.timings_path(song_id)), session.captured_cues())
                        song_library.update_timings(song_id)
                    elif action == "mark":
                        elapsed = payload.get("elapsed")
                        session.mark_current(elapsed=None if elapsed is None else float(elapsed))
                        write_captured_slide_cues(str(song_library.timings_path(song_id)), session.captured_cues())
                        song_library.update_timings(song_id)
                    elif action == "anchor":
                        cue_index = payload.get("cue_index")
                        elapsed = payload.get("elapsed")
                        if isinstance(cue_index, bool) or not isinstance(cue_index, int):
                            raise ValueError("cue_index must be an integer")
                        session.set_anchor(cue_index, elapsed=None if elapsed is None else float(elapsed))
                        write_captured_slide_cues(str(song_library.timings_path(song_id)), session.captured_cues())
                        song_library.update_timings(song_id)
                    elif action == "undo":
                        session.undo_last()
                        write_captured_slide_cues(str(song_library.timings_path(song_id)), session.captured_cues())
                        song_library.update_timings(song_id)
                    elif action == "reset":
                        stop_system_recorder(song_id, keep_audio=False)
                        session.reset()
                    elif action == "reset-saved":
                        stop_system_recorder(song_id, keep_audio=False)
                        song_library.reset_timings(song_id)
                        sessions.pop(song_id, None)
                        session = prepare_session(song_id)
                    elif action == "nudge":
                        cue_index = payload.get("cue_index")
                        delta_seconds = payload.get("delta_seconds")
                        if isinstance(cue_index, bool) or not isinstance(cue_index, int):
                            raise ValueError("cue_index must be an integer")
                        if not isinstance(delta_seconds, (int, float)):
                            raise ValueError("delta_seconds must be a number")
                        session.nudge_cue(cue_index, float(delta_seconds))
                        write_captured_slide_cues(str(song_library.timings_path(song_id)), session.captured_cues())
                        song_library.update_timings(song_id)
                    elif action == "regroup":
                        cue_index = payload.get("cue_index")
                        if isinstance(cue_index, bool) or not isinstance(cue_index, int):
                            raise ValueError("cue_index must be an integer")
                        updated_script_cues = _retarget_script_cues(session.script_cues, cue_index)
                        song_library.replace_slides(
                            song_id,
                            _slide_entries_from_script_cues(updated_script_cues),
                        )
                        refreshed_script_cues = load_slide_script(str(song_library.slides_path(song_id)))
                        if len(refreshed_script_cues) != len(session.script_cues):
                            raise ValueError("regrouped script cue count changed unexpectedly")
                        _rewrite_saved_timings_for_script(
                            song_library,
                            song_id,
                            refreshed_script_cues,
                            session.cue_states(),
                        )
                        sessions.pop(song_id, None)
                        session = prepare_session(song_id)
                    elif action == "line-slides":
                        updated_script_cues = _line_by_line_script_cues(session.script_cues)
                        song_library.replace_slides(
                            song_id,
                            _line_slide_entries_from_script_cues(updated_script_cues),
                        )
                        _rewrite_saved_timings_for_script(
                            song_library,
                            song_id,
                            updated_script_cues,
                            session.cue_states(),
                        )
                        sessions.pop(song_id, None)
                        session = prepare_session(song_id)
                    elif action == "save":
                        session.stop()
                        write_captured_slide_cues(str(song_library.timings_path(song_id)), session.captured_cues())
                        song_library.update_timings(song_id)
                        stop_system_recorder(song_id, keep_audio=True)
                    elif action == "lyrics":
                        lyrics = payload.get("lyrics")
                        if not isinstance(lyrics, str):
                            raise ValueError("lyrics must be a string")
                        stop_system_recorder(song_id, keep_audio=False)
                        song_library.update_lyrics(song_id, lyrics)
                        sessions.pop(song_id, None)
                    else:
                        self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                        return
                    self._write_json(
                        _session_payload(
                            session,
                            song_library.timings_path(song_id),
                            song_library.audio_path(song_id),
                            song_library.get_song(song_id),
                            system_capture_active=system_capture_state(song_id)[0],
                            system_capture_device_name=system_capture_state(song_id)[1],
                        ),
                        HTTPStatus.OK,
                    )
                    return
                if len(parts) in (4, 5) and parts[0] == "songs" and parts[2] == "live-api":
                    song_id = parts[1]
                    manager = runtime_manager(song_id)
                    action = "/".join(parts[3:])
                    body = self._read_body()
                    payload = {} if not body else json.loads(body.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("request body must be a JSON object")
                    if action == "start":
                        song = song_library.get_song(song_id)
                        if not song.profile_directory:
                            raise ValueError("build the profile before running live")
                        profile_path = song_library.profile_path(song_id)
                        if not _profile_has_coarse_artifacts(profile_path):
                            _build_reference_profile_for_song(song_library, song_id)
                        live_tracking_mode = str(
                            payload.get("live_tracking_mode", "live_audio_inference")
                        )
                        config = replace(
                            AppConfig.from_env(),
                            audio_source="microphone",
                            input_device=int(payload.get("input_device", 0)),
                            reference_profile_path=str(song_library.profile_path(song_id)),
                            feature_model_path=str(ROOT / "models" / "wav2vec2-base.onnx"),
                            live_tracking_mode=live_tracking_mode,
                            presentation_mode="logging",
                            block_size=4_096,
                            audio_queue_capacity=32,
                            presentation_command_queue_capacity=16,
                            match_confidence_threshold=0.7,
                            match_ambiguity_distance_margin=0.015,
                            match_confirmation_count=2,
                            tracking_search_miss_patience=2,
                            tracking_miss_patience=0,
                            tracking_match_window_seconds=6.0,
                            recovery_match_window_seconds=8.0,
                            tracking_max_forward_jump_seconds=0.5,
                            tracking_max_backward_jump_seconds=0.25,
                            tracking_expected_position_tolerance_seconds=4.0,
                            tracking_anchored_timeline_tolerance_seconds=1.0,
                            tracking_search_min_duration_seconds=0.1,
                            tracking_recovery_min_duration_seconds=0.1,
                            tracking_recovery_confidence_threshold=0.65,
                            slide_lookahead_seconds=0.2,
                            slide_trigger_cooldown_seconds=0.2,
                            slide_consecutive_match_count=1,
                            slide_max_emit_lag_seconds=1.5,
                            silence_threshold_rms=float(payload.get("silence_threshold_rms", 0.005)),
                            match_debug_logging=bool(payload.get("match_debug_logging", False)),
                        )
                        manager.start(config)
                    elif action == "stop":
                        manager.stop()
                    elif action == "mode":
                        runtime = manager.runtime()
                        if runtime is None:
                            raise ValueError("runtime is not running")
                        live_tracking_mode = str(
                            payload.get("live_tracking_mode", "live_audio_inference")
                        )
                        manager.set_live_tracking_mode(live_tracking_mode)
                    elif action == "manual/on":
                        runtime = manager.runtime()
                        if runtime is None:
                            raise ValueError("runtime is not running")
                        runtime.enable_manual_override()
                    elif action == "manual/off":
                        runtime = manager.runtime()
                        if runtime is None:
                            raise ValueError("runtime is not running")
                        runtime.disable_manual_override()
                    elif action == "manual/toggle":
                        runtime = manager.runtime()
                        if runtime is None:
                            raise ValueError("runtime is not running")
                        runtime.toggle_manual_override()
                    elif action == "jump":
                        runtime = manager.runtime()
                        if runtime is None:
                            raise ValueError("runtime is not running")
                        slide_number = int(payload.get("slide_number", 0))
                        runtime.jump_to_slide(
                            slide_number,
                            realign_tracker=True,
                            record_correction=True,
                        )
                    else:
                        self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                        return
                    self._write_json(
                        {
                            "running": manager.running(),
                            "live_tracking_mode": manager.live_tracking_mode(),
                            "snapshot": _snapshot_to_dict(manager.status_collector.snapshot()),
                            "report": None if manager.report() is None else asdict(manager.report()),
                            "error": manager.error(),
                            "slide_targets": manager.slide_targets(),
                            "logs": manager.log_buffer.messages()[-80:],
                        },
                        HTTPStatus.OK,
                    )
                    return
            except ValueError as error:
                self._write_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args) -> None:
            del format, args

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "host": args.host,
                "library_root": str(Path(args.library_root).expanduser().resolve()),
                "port": args.port,
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
