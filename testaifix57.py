from flask import Flask, Response, jsonify, redirect, render_template_string, request, session, url_for
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import csv
import cv2
import json
import math
import numpy as np
import threading
import time


# ================== CONFIG ==================
BASE_DIR = Path(__file__).resolve().parent

CLASSIFY_MODEL_PATH = "/home/pi/train_test/pytrain/runs/classify/board_cls/weights/best.pt"
DETECT_MODEL_PATH = "/home/pi/train_test/pytrain/runs/detect/components/weights/best.pt"

REFERENCE_MAP = {
    "M1": "/home/pi/mach1.csv",
    "M2": "/home/pi/mach2.csv",
}

BOARD_LABEL_ALIASES = {
    "MACH_1": "M1",
    "MACH1": "M1",
    "BOARD_1": "M1",
    "BOARD1": "M1",
    "M1": "M1",
    "MACH_2": "M2",
    "MACH2": "M2",
    "BOARD_2": "M2",
    "BOARD2": "M2",
    "M2": "M2",
}

LOGIN_USERNAME = "admin"
LOGIN_PASSWORD = "123456"
SECRET_KEY = "aoi-pcb-dashboard-secret"

CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_V4L2
STREAM_WIDTH = 640
STREAM_HEIGHT = 480

PCB_WIDTH_MM = 62.0
PCB_HEIGHT_MM = 49.2
IMG_WIDTH = 640
IMG_HEIGHT = 480
TOLERANCE_MM = 2.0

CLASSIFY_CONF = 0.25
DETECT_CONF = 0.45
INFERENCE_WIDTH = 320
INFERENCE_HEIGHT = 240
PRESENCE_TIMEOUT = 1.2
SESSION_MIN_FRAMES = 5
MAX_HISTORY = 200
HISTORY_FILE = BASE_DIR / "aoi_history.jsonl"
STREAM_JPEG_QUALITY = 68
BOARD_CONFIRM_FRAMES = 1
BOARD_HOLD_SECONDS = 2.0

scale_x_mm = PCB_WIDTH_MM / IMG_WIDTH
scale_y_mm = PCB_HEIGHT_MM / IMG_HEIGHT

WEBAOI_PAGE = (BASE_DIR / "webaoi1.html").read_text(encoding="utf-8")

SOCKET_IO_STUB = r"""
window.io = function () {
  const listeners = {};

  function normalizeStatus(data) {
    const stats = data.stats || {};
    let total = 0;
    let ok = 0;
    let fail = 0;

    Object.values(stats).forEach((item) => {
      total += item.total || 0;
      ok += item.ok || 0;
      fail += item.ng || 0;
    });

    const verdict = (data.current && data.current.verdict) || "WAIT";
    const status =
      verdict === "OK" ? "OK" :
      verdict === "NG" || verdict === "TIMEOUT" ? "FAIL" :
      "WAITING";

    return { total, ok, fail, status };
  }

  async function tick() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) {
        return;
      }
      const payload = await res.json();
      const normalized = normalizeStatus(payload);
      normalized.current = payload.current || {};
      normalized.history = payload.history || [];
      normalized.stats = payload.stats || {};
      normalized.system_message = payload.system_message || "";
      normalized.camera_fps = payload.camera_fps || 0;
      normalized.ai_fps = payload.ai_fps || 0;
      if (listeners.update) {
        listeners.update(normalized);
      }
    } catch (err) {}
  }

  setTimeout(tick, 200);
  setInterval(tick, 900);

  return {
    on(event, callback) {
      listeners[event] = callback;
    }
  };
};
"""

WEBAOI_RUNTIME_PATCH = r"""
<script>
(function () {
  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function verdictBadge(verdict) {
    const v = String(verdict || "WAIT").toUpperCase();
    if (v === "OK") return '<span style="color:#16a34a;font-weight:800">OK</span>';
    if (v === "NG" || v === "FAIL" || v === "TIMEOUT") return '<span style="color:#dc2626;font-weight:800">FAIL</span>';
    return '<span style="color:#f59e0b;font-weight:800">WAIT</span>';
  }

  function issueHtml(issue) {
    if (!issue) return "";
    return '<div style="padding:6px 0;border-top:1px dashed rgba(148,163,184,.25)">' + esc(issue) + "</div>";
  }

  function renderCurrentPanel(data) {
    const box = document.getElementById("dashAiResult");
    const content = document.getElementById("dashAiContent");
    if (!box || !content) return;

    const current = (data && data.current) || {};
    const issues = Array.isArray(current.issues) ? current.issues : [];
    const boardType = current.board_type || current.board_candidate || "--";
    const verdict = current.verdict || "WAIT";
    const counts = `
      <div style="display:grid;grid-template-columns:repeat(4,minmax(68px,1fr));gap:8px;margin:10px 0 12px">
        <div style="padding:8px;border:1px solid rgba(148,163,184,.18);border-radius:6px;background:#fff"><div style="font-size:10px;color:#64748b">OK</div><div style="font-weight:800">${current.ok_count || 0}</div></div>
        <div style="padding:8px;border:1px solid rgba(148,163,184,.18);border-radius:6px;background:#fff"><div style="font-size:10px;color:#64748b">Sai tọa độ</div><div style="font-weight:800">${current.mismatch_count || 0}</div></div>
        <div style="padding:8px;border:1px solid rgba(148,163,184,.18);border-radius:6px;background:#fff"><div style="font-size:10px;color:#64748b">Thiếu</div><div style="font-weight:800">${current.missing_count || 0}</div></div>
        <div style="padding:8px;border:1px solid rgba(148,163,184,.18);border-radius:6px;background:#fff"><div style="font-size:10px;color:#64748b">Thừa</div><div style="font-weight:800">${current.extra_count || 0}</div></div>
      </div>
    `;
    const meta = `
      <div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px">
        <div><span style="color:#64748b">Mạch:</span> <strong>${esc(boardType)}</strong></div>
        <div><span style="color:#64748b">Kết luận:</span> ${verdictBadge(verdict)}</div>
        <div><span style="color:#64748b">Cam:</span> ${Number(data.camera_fps || 0).toFixed(1)} fps</div>
        <div><span style="color:#64748b">AI:</span> ${Number(data.ai_fps || 0).toFixed(1)} fps</div>
      </div>
    `;
    const issueBlock = issues.length
      ? '<div style="margin-top:8px"><div style="font-size:11px;letter-spacing:1px;color:#64748b;margin-bottom:4px">CHI TIẾT LỖI</div>' + issues.map(issueHtml).join("") + "</div>"
      : '<div style="margin-top:8px;color:#16a34a;font-weight:700">Không có lỗi, mạch đang khớp với file chuẩn.</div>';

    box.style.display = "block";
    content.innerHTML = meta + counts + issueBlock;
  }

  function renderHistoryFromBackend(data) {
    const body = document.getElementById("historyBody");
    const count = document.getElementById("histCount");
    if (!body || !count || !Array.isArray(data.history)) return;

    count.innerText = data.history.length + " records";
    body.innerHTML = data.history.slice(0, 100).map((item) => {
      const verdict = (item.verdict || "WAIT").toUpperCase();
      const status = verdict === "OK" ? "OK" : "FAIL";
      const issues = Array.isArray(item.issues) && item.issues.length
        ? esc(item.issues.slice(0, 2).join(" | "))
        : "";
      return `<tr>
        <td style="font-size:11px">${esc(item.timestamp || "")}</td>
        <td><span class="badge badge-${status === "OK" ? "ok" : "fail"}">${status}</span></td>
        <td>${esc(item.board_type || "--")}</td>
        <td style="color:var(--ok);font-weight:600">${item.ok_count || 0}</td>
        <td style="color:var(--fail);font-weight:600">${(item.missing_count || 0) + (item.extra_count || 0) + (item.mismatch_count || 0)}</td>
        <td style="font-size:11px;color:var(--muted);max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${issues}</td>
      </tr>`;
    }).join("");
  }

  function enableMjpeg() {
    const mainImg = document.getElementById("resultImage");
    if (mainImg) {
      mainImg.src = "/video_feed";
      mainImg.removeAttribute("onload");
      mainImg.style.opacity = "1";
    }
    const extImg = document.getElementById("extCamLive");
    if (extImg) {
      extImg.src = "/video_feed";
      extImg.style.display = "block";
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    enableMjpeg();
    const originalRefresh = window.refreshExtCamLive;
    window.refreshExtCamLive = function () {
      const liveImg = document.getElementById("extCamLive");
      if (liveImg && liveImg.style.display !== "none" && liveImg.src.indexOf("/video_feed") === -1) {
        liveImg.src = "/video_feed";
      }
      if (typeof originalRefresh === "function") {
        return;
      }
    };

    const originalAddHistory = window.addHistory;
    window.addHistory = function (data) {
      if (data && data.history && data.current) {
        renderCurrentPanel(data);
        renderHistoryFromBackend(data);
        return;
      }
      if (typeof originalAddHistory === "function") {
        originalAddHistory(data);
      }
    };

    const oldIo = window.io;
    if (typeof oldIo === "function") {
      const wrapped = oldIo();
      const oldOn = wrapped.on.bind(wrapped);
      wrapped.on = function (event, callback) {
        if (event !== "update") return oldOn(event, callback);
        return oldOn(event, function (data) {
          enableMjpeg();
          renderCurrentPanel(data);
          renderHistoryFromBackend(data);
          callback(data);
        });
      };
      window.io = function () { return wrapped; };
    }
  });
})();
</script>
"""


LOGIN_PAGE = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AOI PCB Login</title>
  <style>
    :root {
      --bg: #08111d;
      --panel: rgba(10, 20, 34, 0.88);
      --line: rgba(126, 220, 255, 0.24);
      --accent: #67d8ff;
      --text: #eef8ff;
      --muted: #9ab5c7;
      --danger: #ff7373;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: Consolas, monospace;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(103, 216, 255, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(255, 115, 115, 0.12), transparent 24%),
        linear-gradient(180deg, #050913, #091423 65%, #050b12);
    }
    .panel {
      width: min(460px, calc(100vw - 32px));
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 28px;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
    }
    h1 {
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 30px;
    }
    .sub {
      color: var(--muted);
      line-height: 1.7;
      margin-bottom: 24px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
    }
    input {
      width: 100%;
      border-radius: 16px;
      border: 1px solid rgba(126, 220, 255, 0.18);
      background: rgba(3, 10, 19, 0.95);
      color: var(--text);
      padding: 14px;
      font-size: 16px;
      margin-bottom: 16px;
      outline: none;
    }
    button {
      width: 100%;
      border: 0;
      border-radius: 16px;
      padding: 14px;
      font-size: 16px;
      font-family: inherit;
      color: white;
      cursor: pointer;
      background: linear-gradient(135deg, #66ddff, #2a88ff);
    }
    .error {
      margin-top: 14px;
      color: var(--danger);
      min-height: 20px;
    }
    .hint {
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
    }
  </style>
</head>
<body>
  <form class="panel" method="post">
    <h1>AOI PCB Control</h1>
    <div class="sub">
      Đăng nhập để vào dashboard giám sát phân loại mạch, phát hiện linh kiện và thống kê lỗi theo thời gian thực.
    </div>
    <label for="username">Tài khoản</label>
    <input id="username" name="username" autocomplete="username" required>
    <label for="password">Mật khẩu</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">Đăng nhập</button>
    <div class="error">{{ error }}</div>
    <div class="hint">Mặc định trong code: {{ default_user }} / {{ default_pass }}</div>
  </form>
</body>
</html>
"""


LEGACY_APP_PAGE = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AOI PCB Dashboard</title>
  <style>
    :root {
      --bg: #071019;
      --panel: rgba(10, 22, 35, 0.92);
      --panel-2: rgba(6, 17, 28, 0.94);
      --line: rgba(105, 207, 255, 0.18);
      --accent: #65d9ff;
      --accent-2: #78ffbd;
      --text: #eefaff;
      --muted: #8fadc3;
      --danger: #ff7b7b;
      --warn: #ffc36d;
      --ok: #79ffb4;
      --shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Consolas, monospace;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(101, 217, 255, 0.15), transparent 25%),
        radial-gradient(circle at bottom right, rgba(121, 255, 180, 0.08), transparent 25%),
        linear-gradient(180deg, #03070d, #071019 55%, #04080d);
    }
    .layout {
      display: grid;
      grid-template-columns: 250px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      padding: 22px 16px;
      border-right: 1px solid var(--line);
      background: rgba(5, 12, 20, 0.88);
      position: sticky;
      top: 0;
      height: 100vh;
    }
    .brand {
      padding: 14px;
      border-radius: 20px;
      background: linear-gradient(160deg, rgba(31, 74, 118, 0.7), rgba(8, 21, 35, 0.95));
      border: 1px solid rgba(101, 217, 255, 0.18);
      margin-bottom: 18px;
      box-shadow: var(--shadow);
    }
    .brand .title {
      color: var(--accent);
      font-size: 22px;
      margin-bottom: 6px;
    }
    .brand .sub {
      color: var(--muted);
      line-height: 1.6;
      font-size: 13px;
    }
    .nav button, .logout {
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 16px;
      padding: 12px 14px;
      margin-bottom: 10px;
      cursor: pointer;
      font-family: inherit;
      color: var(--text);
      background: rgba(11, 23, 37, 0.92);
    }
    .nav button.active {
      border-color: rgba(101, 217, 255, 0.32);
      background: rgba(22, 46, 70, 0.94);
      color: var(--accent);
    }
    .logout {
      display: inline-block;
      text-decoration: none;
      margin-top: 12px;
      border-color: rgba(255, 123, 123, 0.2);
    }
    .main {
      padding: 22px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .headline {
      font-size: 28px;
      color: var(--accent);
    }
    .statusPill {
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(9, 20, 33, 0.9);
      color: var(--muted);
    }
    .section {
      display: none;
    }
    .section.active {
      display: block;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }
    .panel h3 {
      margin: 0 0 14px;
      color: var(--accent);
      font-size: 18px;
    }
    .videoWrap {
      overflow: hidden;
      border-radius: 18px;
      border: 1px solid rgba(101, 217, 255, 0.15);
      background: #03070d;
    }
    .videoWrap img {
      width: 100%;
      display: block;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .card {
      padding: 14px;
      border-radius: 18px;
      background: var(--panel-2);
      border: 1px solid rgba(101, 217, 255, 0.12);
    }
    .label {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }
    .value {
      font-size: 24px;
      color: var(--text);
      word-break: break-word;
    }
    .value.ok { color: var(--ok); }
    .value.ng { color: var(--danger); }
    .value.warn { color: var(--warn); }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(101, 217, 255, 0.08);
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 400;
    }
    .badge {
      padding: 4px 10px;
      border-radius: 999px;
      display: inline-block;
      font-size: 12px;
    }
    .badge.ok { background: rgba(121, 255, 180, 0.15); color: var(--ok); }
    .badge.ng { background: rgba(255, 123, 123, 0.15); color: var(--danger); }
    .issueList {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .issue {
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(255, 195, 109, 0.12);
      color: var(--warn);
      border: 1px solid rgba(255, 195, 109, 0.18);
    }
    .historyItem {
      padding: 14px;
      border-radius: 18px;
      background: var(--panel-2);
      border: 1px solid rgba(101, 217, 255, 0.1);
      margin-bottom: 12px;
    }
    .historyTop {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .muted {
      color: var(--muted);
    }
    .configCode {
      border-radius: 18px;
      padding: 16px;
      background: rgba(4, 10, 16, 0.95);
      color: #cfefff;
      overflow: auto;
      border: 1px solid rgba(101, 217, 255, 0.08);
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar {
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .cards { grid-template-columns: 1fr; }
      .headline { font-size: 22px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">
        <div class="title">AOI PCB</div>
        <div class="sub">Phân loại mạch, đối chiếu tọa độ mm, giám sát realtime và lưu lịch sử lỗi.</div>
      </div>
      <div class="nav">
        <button class="tabBtn active" data-tab="dashboard">Dashboard Live</button>
        <button class="tabBtn" data-tab="stats">Thống kê</button>
        <button class="tabBtn" data-tab="history">Lịch sử</button>
        <button class="tabBtn" data-tab="config">Cấu hình</button>
      </div>
      <a class="logout" href="/logout">Đăng xuất</a>
    </aside>

    <main class="main">
      <div class="topbar">
        <div class="headline">Realtime AOI PCB Dashboard</div>
        <div class="statusPill" id="systemStatus">Đang tải trạng thái...</div>
      </div>

      <section class="section active" id="tab-dashboard">
        <div class="grid">
          <div>
            <div class="panel">
              <h3>Video realtime từ camera Pi</h3>
              <div class="videoWrap">
                <img src="/video_feed" alt="Live video stream">
              </div>
            </div>
          </div>

          <div>
            <div class="panel">
              <h3>Trạng thái hiện tại</h3>
              <div class="cards">
                <div class="card">
                  <div class="label">Loại mạch hiện tại</div>
                  <div class="value" id="boardType">--</div>
                </div>
                <div class="card">
                  <div class="label">Xác suất phân loại</div>
                  <div class="value" id="boardConf">--</div>
                </div>
                <div class="card">
                  <div class="label">Phán quyết hiện tại</div>
                  <div class="value" id="verdict">--</div>
                </div>
                <div class="card">
                  <div class="label">Tổng linh kiện detect</div>
                  <div class="value" id="detectedCount">0</div>
                </div>
                <div class="card">
                  <div class="label">Camera</div>
                  <div class="value" id="cameraState">--</div>
                </div>
                <div class="card">
                  <div class="label">AI xử lý</div>
                  <div class="value" id="aiState">--</div>
                </div>
              </div>
            </div>

            <div class="panel">
              <h3>Chi tiết đối chiếu</h3>
              <div class="cards">
                <div class="card">
                  <div class="label">OK</div>
                  <div class="value ok" id="okCount">0</div>
                </div>
                <div class="card">
                  <div class="label">Sai tọa độ</div>
                  <div class="value warn" id="mismatchCount">0</div>
                </div>
                <div class="card">
                  <div class="label">Thiếu</div>
                  <div class="value ng" id="missingCount">0</div>
                </div>
                <div class="card">
                  <div class="label">Thừa</div>
                  <div class="value ng" id="extraCount">0</div>
                </div>
              </div>
              <div style="margin-top:14px;" class="issueList" id="issueList"></div>
            </div>
          </div>
        </div>
      </section>

      <section class="section" id="tab-stats">
        <div class="panel">
          <h3>Thống kê theo loại mạch</h3>
          <table>
            <thead>
              <tr>
                <th>Loại mạch</th>
                <th>Tổng</th>
                <th>Đúng</th>
                <th>Sai</th>
                <th>Thiếu</th>
                <th>Thừa</th>
                <th>Sai tọa độ</th>
              </tr>
            </thead>
            <tbody id="statsTable"></tbody>
          </table>
        </div>
      </section>

      <section class="section" id="tab-history">
        <div class="panel">
          <h3>Lịch sử kiểm tra</h3>
          <div id="historyList"></div>
        </div>
      </section>

      <section class="section" id="tab-config">
        <div class="panel">
          <h3>Cấu hình đang dùng</h3>
          <div class="configCode" id="configView"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const tabButtons = document.querySelectorAll(".tabBtn");
    const sections = document.querySelectorAll(".section");

    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        tabButtons.forEach((b) => b.classList.remove("active"));
        sections.forEach((s) => s.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(`tab-${button.dataset.tab}`).classList.add("active");
      });
    });

    function badge(verdict) {
      return verdict === "OK"
        ? '<span class="badge ok">OK</span>'
        : '<span class="badge ng">NG</span>';
    }

    function issueHtml(items) {
      if (!items.length) {
        return '<span class="muted">Không có lỗi đang hoạt động.</span>';
      }
      return items.map((item) => `<span class="issue">${item}</span>`).join("");
    }

    function renderStats(stats) {
      const rows = Object.keys(stats).sort().map((name) => {
        const item = stats[name];
        return `
          <tr>
            <td>${name}</td>
            <td>${item.total}</td>
            <td>${item.ok}</td>
            <td>${item.ng}</td>
            <td>${item.missing}</td>
            <td>${item.extra}</td>
            <td>${item.mismatch}</td>
          </tr>
        `;
      }).join("");

      document.getElementById("statsTable").innerHTML = rows || '<tr><td colspan="7" class="muted">Chưa có dữ liệu.</td></tr>';
    }

    function renderHistory(history) {
      const html = history.map((item) => `
        <div class="historyItem">
          <div class="historyTop">
            <div><strong>${item.board_type}</strong> ${badge(item.verdict)}</div>
            <div class="muted">${item.timestamp}</div>
          </div>
          <div class="muted">Phân loại: ${item.classification_confidence}% | Detect: ${item.detected_count} | OK: ${item.ok_count}</div>
          <div style="margin-top:10px;" class="issueList">${issueHtml(item.issues)}</div>
        </div>
      `).join("");

      document.getElementById("historyList").innerHTML = html || '<div class="muted">Chưa có lịch sử.</div>';
    }

    function renderConfig(config) {
      document.getElementById("configView").textContent = JSON.stringify(config, null, 2);
    }

    async function refresh() {
      try {
        const res = await fetch("/api/status");
        if (!res.ok) {
          return;
        }
        const data = await res.json();

        document.getElementById("systemStatus").textContent = data.system_message;
        document.getElementById("boardType").textContent = data.current.board_type || "--";
        document.getElementById("boardConf").textContent = data.current.classification_confidence ? `${data.current.classification_confidence}%` : "--";
        document.getElementById("verdict").textContent = data.current.verdict || "--";
        document.getElementById("detectedCount").textContent = data.current.detected_count;
        document.getElementById("cameraState").textContent = data.camera_alive ? `LIVE ${data.camera_fps} fps` : "MẤT KHUNG";
        document.getElementById("aiState").textContent = data.ai_alive ? `RUN ${data.ai_fps} fps` : "ĐANG DỪNG";
        document.getElementById("okCount").textContent = data.current.ok_count;
        document.getElementById("mismatchCount").textContent = data.current.mismatch_count;
        document.getElementById("missingCount").textContent = data.current.missing_count;
        document.getElementById("extraCount").textContent = data.current.extra_count;
        document.getElementById("issueList").innerHTML = issueHtml(data.current.issues);

        renderStats(data.stats);
        renderHistory(data.history);
        renderConfig(data.config);
      } catch (err) {}
    }

    setInterval(refresh, 700);
    refresh();
  </script>
</body>
</html>
"""


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_mm(value):
    text = str(value).strip().lower().replace("mm", "")
    return float(text)


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def normalize_board_label(label):
    return str(label).strip().upper().replace(" ", "")


def normalize_component_label(label):
    return str(label).strip().upper().replace(" ", "")


def resolve_board_label(label):
    normalized = normalize_board_label(label)
    return BOARD_LABEL_ALIASES.get(normalized, normalized)


def load_reference(path):
    data = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label") or row.get("Label") or row.get("name") or row.get("Name")
            x_value = row.get("x") or row.get("X")
            y_value = row.get("y") or row.get("Y")
            if label is None or x_value is None or y_value is None:
                continue
            data.append({
                "label": normalize_component_label(label),
                "x": parse_mm(x_value),
                "y": parse_mm(y_value),
            })
    return data


def safe_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def classify_board(model, frame):
    results = model(frame, imgsz=IMG_WIDTH, conf=CLASSIFY_CONF, verbose=False)
    best_detection = None

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_index = int(safe_float(box.cls[0]))
            conf = safe_float(box.conf[0]) if box.conf is not None else 0.0
            label = resolve_board_label(model.names[cls_index])

            detection = {
                "label": label,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }

            if best_detection is None or detection["confidence"] > best_detection["confidence"]:
                best_detection = detection

    if best_detection is None:
        return None, 0.0, None

    return best_detection["label"], best_detection["confidence"], best_detection


def detect_components(model, frame):
    results = model(frame, imgsz=IMG_WIDTH, conf=DETECT_CONF, verbose=False)
    detections = []
    frame_h, frame_w = frame.shape[:2]
    current_scale_x_mm = PCB_WIDTH_MM / frame_w
    current_scale_y_mm = PCB_HEIGHT_MM / frame_h

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_index = int(safe_float(box.cls[0]))
            conf = safe_float(box.conf[0]) if box.conf is not None else 0.0
            label = normalize_component_label(model.names[cls_index])

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            x_mm = cx * current_scale_x_mm
            y_mm = PCB_HEIGHT_MM - (cy * current_scale_y_mm)

            detections.append({
                "label": label,
                "confidence": conf,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
            })

    return detections


def compare_components(detected, reference):
    details = []
    used_detection_indices = set()

    for ref in reference:
        same_label_candidates = []
        for idx, det in enumerate(detected):
            if normalize_component_label(det["label"]) != normalize_component_label(ref["label"]):
                continue
            d = distance(ref["x"], ref["y"], det["x_mm"], det["y_mm"])
            same_label_candidates.append((d, idx, det))

        if not same_label_candidates:
            details.append({
                "type": "MISSING",
                "label": ref["label"],
                "expected": [ref["x"], ref["y"]],
                "actual": None,
                "distance_mm": None,
            })
            continue

        same_label_candidates.sort(key=lambda item: item[0])
        matched = False

        for d, idx, det in same_label_candidates:
            if idx in used_detection_indices:
                continue

            if d <= TOLERANCE_MM:
                used_detection_indices.add(idx)
                details.append({
                    "type": "OK",
                    "label": ref["label"],
                    "expected": [ref["x"], ref["y"]],
                    "actual": [det["x_mm"], det["y_mm"]],
                    "distance_mm": d,
                })
                matched = True
                break

            used_detection_indices.add(idx)
            details.append({
                "type": "MISMATCH",
                "label": ref["label"],
                "expected": [ref["x"], ref["y"]],
                "actual": [det["x_mm"], det["y_mm"]],
                "distance_mm": d,
            })
            matched = True
            break

        if not matched:
            details.append({
                "type": "MISSING",
                "label": ref["label"],
                "expected": [ref["x"], ref["y"]],
                "actual": None,
                "distance_mm": None,
            })

    for idx, det in enumerate(detected):
        if idx in used_detection_indices:
            continue
        details.append({
            "type": "EXTRA",
            "label": det["label"],
            "expected": None,
            "actual": [det["x_mm"], det["y_mm"]],
            "distance_mm": None,
        })

    ok_count = sum(1 for item in details if item["type"] == "OK")
    missing_count = sum(1 for item in details if item["type"] == "MISSING")
    extra_count = sum(1 for item in details if item["type"] == "EXTRA")
    mismatch_count = sum(1 for item in details if item["type"] == "MISMATCH")
    verdict = "OK" if missing_count == 0 and extra_count == 0 and mismatch_count == 0 and len(reference) > 0 else "NG"

    issues = []
    for item in details:
        if item["type"] == "OK":
            continue
        if item["type"] == "MISSING":
            issues.append(f"Thiếu {item['label']} tại ({item['expected'][0]:.1f}, {item['expected'][1]:.1f}) mm")
        elif item["type"] == "EXTRA":
            issues.append(f"Thừa {item['label']} tại ({item['actual'][0]:.1f}, {item['actual'][1]:.1f}) mm")
        else:
            issues.append(
                f"Sai tọa độ {item['label']} | chuẩn ({item['expected'][0]:.1f}, {item['expected'][1]:.1f}) mm"
                f" | AI ({item['actual'][0]:.1f}, {item['actual'][1]:.1f}) mm"
            )

    return {
        "verdict": verdict,
        "details": details,
        "ok_count": ok_count,
        "missing_count": missing_count,
        "extra_count": extra_count,
        "mismatch_count": mismatch_count,
        "issues": issues,
    }


class AOIEngine:
    def __init__(self):
        self.classifier = YOLO(CLASSIFY_MODEL_PATH)
        self.detector = YOLO(DETECT_MODEL_PATH)

        self.references = {resolve_board_label(board_type): load_reference(path) for board_type, path in REFERENCE_MAP.items()}
        self.history = deque(maxlen=MAX_HISTORY)
        self.stats = defaultdict(lambda: {
            "total": 0,
            "ok": 0,
            "ng": 0,
            "missing": 0,
            "extra": 0,
            "mismatch": 0,
        })

        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.latest_frame = None
        self.latest_jpeg = None
        self.running = True
        self.last_frame_time = 0.0
        self.last_inference_time = 0.0
        self.frame_counter = 0
        self.inference_counter = 0
        self.system_message = "Đang khởi động hệ thống AI..."
        self.latest_detections = []
        self.latest_board_detection = None

        self.current_result = {
            "board_type": None,
            "board_candidate": None,
            "board_detection": None,
            "classification_confidence": 0,
            "verdict": "WAIT",
            "detected_count": 0,
            "ok_count": 0,
            "missing_count": 0,
            "extra_count": 0,
            "mismatch_count": 0,
            "issues": [],
            "details": [],
            "updated_at": now_text(),
        }

        self.active_session = None
        self.board_votes = deque(maxlen=10)
        self.confirmed_board_type = None
        self.last_board_confirm_time = 0.0
        self.cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)

        dummy = np.zeros((INFERENCE_HEIGHT, INFERENCE_WIDTH, 3), dtype=np.uint8)
        self.classifier(dummy, imgsz=INFERENCE_WIDTH, verbose=False)
        self.detector(dummy, imgsz=INFERENCE_WIDTH, verbose=False)

        self._load_history()

    def _load_history(self):
        if not HISTORY_FILE.exists():
            return

        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.history.append(item)
            stats = self.stats[item["board_type"]]
            stats["total"] += 1
            if item["verdict"] == "OK":
                stats["ok"] += 1
            else:
                stats["ng"] += 1
            stats["missing"] += item.get("missing_count", 0)
            stats["extra"] += item.get("extra_count", 0)
            stats["mismatch"] += item.get("mismatch_count", 0)

    def _save_history_item(self, item):
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def start(self):
        threading.Thread(target=self._camera_loop, daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()

    def _camera_loop(self):
        if not self.cap.isOpened():
            self.system_message = "Không mở được camera."
            return

        self.system_message = "Camera đã sẵn sàng, đang chờ mạch."

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            with self.frame_lock:
                self.latest_frame = frame
                self.last_frame_time = time.time()
                self.frame_counter += 1

            with self.state_lock:
                snapshot = dict(self.current_result)
                detections = [dict(item) for item in self.latest_detections]

            self._build_overlay(frame, snapshot, detections)

    def _finalize_active_session(self, reason):
        if not self.active_session:
            return

        if self.active_session["frame_count"] < SESSION_MIN_FRAMES:
            self.active_session = None
            return

        snapshot = self.active_session["snapshot"]
        board_type = snapshot["board_type"] or "UNKNOWN"

        history_item = {
            "timestamp": now_text(),
            "board_type": board_type,
            "classification_confidence": snapshot["classification_confidence"],
            "verdict": snapshot["verdict"],
            "detected_count": snapshot["detected_count"],
            "ok_count": snapshot["ok_count"],
            "missing_count": snapshot["missing_count"],
            "extra_count": snapshot["extra_count"],
            "mismatch_count": snapshot["mismatch_count"],
            "issues": snapshot["issues"],
            "reason": reason,
        }

        self.history.appendleft(history_item)
        self._save_history_item(history_item)

        stats = self.stats[board_type]
        stats["total"] += 1
        if snapshot["verdict"] == "OK":
            stats["ok"] += 1
        else:
            stats["ng"] += 1
        stats["missing"] += snapshot["missing_count"]
        stats["extra"] += snapshot["extra_count"]
        stats["mismatch"] += snapshot["mismatch_count"]

        self.active_session = None

    def _build_overlay(self, frame, snapshot, detections):
        overlay = frame.copy()

        board_detection = snapshot.get("board_detection")
        if board_detection and board_detection.get("bbox"):
            x1, y1, x2, y2 = board_detection["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 120), 3)
            board_text = f"BOARD {board_detection.get('label', '--')} ({board_detection.get('confidence', 0.0) * 100:.1f}%)"
            cv2.putText(overlay, board_text, (x1, max(28, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (78, 221, 255), 2)
            cx, cy = det["center"]
            cv2.circle(overlay, (cx, cy), 4, (0, 90, 255), -1)
            text = f"{det['label']} ({det['x_mm']:.1f}, {det['y_mm']:.1f}) mm"
            cv2.putText(overlay, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        top_lines = [
            f"Board: {snapshot['board_type'] or '--'}",
            f"Candidate: {snapshot.get('board_candidate') or '--'}",
            f"Cls conf: {snapshot['classification_confidence']}%",
            f"Verdict: {snapshot['verdict']}",
            f"OK {snapshot['ok_count']} | Mismatch {snapshot['mismatch_count']} | Missing {snapshot['missing_count']} | Extra {snapshot['extra_count']}",
            f"Camera fps: {self.get_camera_fps():.1f} | AI fps: {self.get_ai_fps():.1f}",
        ]

        y = 24
        for line in top_lines:
            cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (105, 235, 255), 2)
            y += 24

        for issue in snapshot["issues"][:6]:
            cv2.putText(overlay, issue[:88], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 180, 255), 1)
            y += 20

        ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
        if ok:
            with self.state_lock:
                self.latest_jpeg = encoded.tobytes()

    def _resolve_confirmed_board(self, raw_board_type, cls_conf):
        now = time.time()

        if raw_board_type and cls_conf >= CLASSIFY_CONF:
            candidate = resolve_board_label(raw_board_type)
            if candidate in self.references:
                self.board_votes.append(candidate)

        if self.board_votes:
            counts = {}
            for label in self.board_votes:
                counts[label] = counts.get(label, 0) + 1

            best_label, best_count = max(counts.items(), key=lambda item: item[1])
            if best_count >= BOARD_CONFIRM_FRAMES:
                self.confirmed_board_type = best_label
                self.last_board_confirm_time = now

        if self.confirmed_board_type and now - self.last_board_confirm_time <= BOARD_HOLD_SECONDS:
            return self.confirmed_board_type

        if self.confirmed_board_type and now - self.last_board_confirm_time > BOARD_HOLD_SECONDS:
            self.confirmed_board_type = None
            self.board_votes.clear()

        return None

    def _inference_loop(self):
        while self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            raw_board_type, cls_conf, board_detection = classify_board(self.classifier, frame)
            board_type = self._resolve_confirmed_board(raw_board_type, cls_conf)

            detections = detect_components(self.detector, frame)
            reference = self.references.get(board_type, [])
            comparison = compare_components(detections, reference) if board_type else {
                "verdict": "WAIT_BOARD_TYPE",
                "details": [],
                "ok_count": 0,
                "missing_count": 0,
                "extra_count": 0,
                "mismatch_count": 0,
                "issues": [],
            }

            snapshot = {
                "board_type": board_type,
                "board_candidate": raw_board_type,
                "board_detection": board_detection,
                "classification_confidence": round(cls_conf * 100, 1),
                "verdict": comparison["verdict"],
                "detected_count": len(detections),
                "ok_count": comparison["ok_count"],
                "missing_count": comparison["missing_count"],
                "extra_count": comparison["extra_count"],
                "mismatch_count": comparison["mismatch_count"],
                "issues": comparison["issues"],
                "details": comparison["details"],
                "updated_at": now_text(),
            }

            if board_type:
                self.system_message = f"Checking {board_type} | {snapshot['verdict']} | {snapshot['updated_at']}"
                if self.active_session and self.active_session["board_type"] != board_type:
                    self._finalize_active_session("board_switch")

                if not self.active_session:
                    self.active_session = {
                        "board_type": board_type,
                        "first_seen": time.time(),
                        "last_seen": time.time(),
                        "frame_count": 1,
                        "snapshot": snapshot,
                    }
                else:
                    self.active_session["last_seen"] = time.time()
                    self.active_session["frame_count"] += 1
                    self.active_session["snapshot"] = snapshot
            else:
                if self.active_session and time.time() - self.active_session["last_seen"] > PRESENCE_TIMEOUT:
                    self._finalize_active_session("board_left")
                debug_label = raw_board_type or "--"
                self.system_message = f"Waiting board classify | cls={debug_label} | conf={cls_conf:.2f}"

            with self.state_lock:
                self.current_result = snapshot
                self.latest_detections = detections
                self.latest_board_detection = board_detection
                self.last_inference_time = time.time()
                self.inference_counter += 1
            time.sleep(0.01)

    def get_camera_fps(self):
        now = time.time()
        age = now - self.last_frame_time
        if age <= 0 or age > 2.0:
            return 0.0
        return min(30.0, 1.0 / max(age, 1e-6))

    def get_ai_fps(self):
        now = time.time()
        age = now - self.last_inference_time
        if age <= 0 or age > 2.0:
            return 0.0
        return min(30.0, 1.0 / max(age, 1e-6))

    def get_status(self):
        with self.state_lock:
            current = dict(self.current_result)
            history = list(self.history)
            stats = {key: value.copy() for key, value in self.stats.items()}
            message = self.system_message
            last_inference_time = self.last_inference_time
            last_frame_time = self.last_frame_time

        return {
            "system_message": message,
            "camera_alive": (time.time() - last_frame_time) < 2.0 if last_frame_time else False,
            "ai_alive": (time.time() - last_inference_time) < 2.0 if last_inference_time else False,
            "camera_fps": round(self.get_camera_fps(), 1),
            "ai_fps": round(self.get_ai_fps(), 1),
            "current": current,
            "history": history,
            "stats": stats,
            "config": {
                "classify_model_path": CLASSIFY_MODEL_PATH,
                "detect_model_path": DETECT_MODEL_PATH,
                "reference_map": REFERENCE_MAP,
                "camera_index": CAMERA_INDEX,
                "stream_size": [STREAM_WIDTH, STREAM_HEIGHT],
                "pcb_size_mm": [PCB_WIDTH_MM, PCB_HEIGHT_MM],
                "tolerance_mm": TOLERANCE_MM,
                "classify_conf_threshold": CLASSIFY_CONF,
                "detect_conf_threshold": DETECT_CONF,
            },
        }

    def mjpeg_stream(self):
        while True:
            with self.state_lock:
                frame = self.latest_jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


APP_PAGE = LEGACY_APP_PAGE


app = Flask(__name__)
app.secret_key = SECRET_KEY
engine = AOIEngine()
engine.start()


def is_logged_in():
    return session.get("logged_in", False)


@app.route("/login", methods=["GET", "POST"])
def login():
    if is_logged_in():
        return redirect(url_for("index"))

    error = ""
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("index"))
        error = "Sai tài khoản hoặc mật khẩu."

    return render_template_string(
        LOGIN_PAGE,
        error=error,
        default_user=LOGIN_USERNAME,
        default_pass=LOGIN_PASSWORD,
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/login.html")
def login_html():
    return redirect(url_for("login"))


@app.route("/")
def index():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template_string(APP_PAGE)


@app.route("/socket.io/socket.io.js")
def socketio_stub():
    return Response(SOCKET_IO_STUB, mimetype="application/javascript")


@app.route("/uploads/result.jpg")
def latest_result_image():
    with engine.state_lock:
        frame = engine.latest_jpeg
    if frame is None:
        return ("", 204)
    return Response(frame, mimetype="image/jpeg")


@app.route("/video_feed")
def video_feed():
    if not is_logged_in():
        return redirect(url_for("login"))
    return Response(engine.mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    if not is_logged_in():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(engine.get_status())


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        engine.stop()
