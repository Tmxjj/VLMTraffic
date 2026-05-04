"""
7_eval_dataset_viewer.py - Eval 结果本地可视化前端

用法：
    python scripts/7_eval_dataset_viewer.py

可选参数：
    python scripts/7_eval_dataset_viewer.py \
        --eval_path data/eval/JiNan/anon_3_4_jinan_real/qwen3-vl-8b \
        --scenario JiNan \
        --host 127.0.0.1 \
        --port 8503

说明：
1. 只读查看，不修改评测结果。
2. 展示每个路口每个决策步的模型输入图像、重建 Prompt、模型响应。
3. 按路口展示 phase / green duration 的时间序列分布。
4. 仅依赖 Python 标准库。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.prompt_builder import PromptBuilder


DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "JiNan" / "anon_3_4_jinan_real" / "qwen3-vl-8b"
APPROACH_ORDER = ["N", "E", "S", "W"]
ACTION_JSON_RE = re.compile(r"Action\s*:?\s*(\{.*?\})", re.IGNORECASE | re.DOTALL)
ACTION_KV_RE = re.compile(
    r"[\"']?phase(?:_id)?[\"']?\s*[:=]\s*(\d+).*?[\"']?duration[\"']?\s*[:=]\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
PROMPT_SECTION_RE = re.compile(
    r"\[User Prompt\]\s*(.*?)\s*(?=\[Model Response\]|\Z)",
    re.DOTALL,
)
MODEL_RESPONSE_SECTION_RE = re.compile(
    r"\[Model Response\]\s*(.*?)\s*(?=\[Thinking Process\]|\Z)",
    re.DOTALL,
)
THINKING_SECTION_RE = re.compile(
    r"\[Thinking Process\]\s*(.*?)\s*\Z",
    re.DOTALL,
)


class ViewerData:
    def __init__(self, eval_path: Path, scenario: str, junctions: Dict[str, Any], samples: Dict[str, Dict[int, Any]]) -> None:
        self.eval_path = eval_path
        self.scenario = scenario
        self.junctions = junctions
        self.samples = samples


VIEWER_DATA: Optional[ViewerData] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval dataset local web viewer")
    parser.add_argument("--eval_path", type=str, default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--scenario", type=str, default="JiNan")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8503, help="监听端口；设为 0 时自动分配")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_step_from_name(path: Path) -> Optional[int]:
    try:
        return int(path.name)
    except ValueError:
        return None


def parse_response(response_text: str) -> Dict[str, Any]:
    prompt_match = PROMPT_SECTION_RE.search(response_text or "")
    model_response_match = MODEL_RESPONSE_SECTION_RE.search(response_text or "")
    thinking_match = THINKING_SECTION_RE.search(response_text or "")

    prompt_text = prompt_match.group(1).strip() if prompt_match else ""
    model_response_text = model_response_match.group(1).strip() if model_response_match else (response_text or "").strip()
    thought_text = thinking_match.group(1).strip() if thinking_match else model_response_text

    action = {"phase": None, "duration": None}
    match = ACTION_JSON_RE.search(model_response_text)
    if match:
        try:
            payload = json.loads(match.group(1))
            action["phase"] = int(payload.get("phase", payload.get("phase_id")))
            action["duration"] = int(payload.get("duration"))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if action["phase"] is None or action["duration"] is None:
        kv_match = ACTION_KV_RE.search(model_response_text)
        if kv_match:
            action["phase"] = int(kv_match.group(1))
            action["duration"] = int(kv_match.group(2))

    thought = thought_text.strip()
    if "Action:" in thought:
        thought = thought.split("Action:", 1)[0].strip()
    return {
        "phase": action["phase"],
        "duration": action["duration"],
        "prompt": prompt_text,
        "thought": thought,
        "model_response": model_response_text.strip(),
        "raw": (response_text or "").strip(),
    }


def sort_images(step_dir: Path, with_watermark: bool) -> List[Path]:
    suffix = ".png"
    images: Dict[str, Path] = {}
    for path in step_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        if with_watermark and path.name.endswith("_no_watermark.png"):
            continue
        if not with_watermark and not path.name.endswith("_no_watermark.png"):
            continue
        stem = path.stem.replace("_no_watermark", "")
        direction = stem.rsplit("_", 1)[-1]
        if direction in APPROACH_ORDER:
            images[direction] = path
    return [images[d] for d in APPROACH_ORDER if d in images]


def load_render_info(render_dir: Path) -> Dict[int, Dict[str, Any]]:
    render_map: Dict[int, Dict[str, Any]] = {}
    if not render_dir.exists():
        return render_map
    for path in sorted(render_dir.glob("render_*.json")):
        step_text = path.stem.replace("render_", "", 1)
        try:
            step = int(step_text)
        except ValueError:
            continue
        try:
            render_map[step] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return render_map


def build_sample(eval_path: Path, scenario: str, jid: str, step: int, step_dir: Path, render_state: Dict[str, Any]) -> Dict[str, Any]:
    response_path = step_dir / "response.txt"
    response_text = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
    parsed_response = parse_response(response_text)
    tls_state = (render_state.get("tls") or {}).get(jid) or {}
    current_phase_id = int(tls_state.get("this_phase_index", 0))
    prompt = parsed_response["prompt"] or PromptBuilder.build_decision_prompt(
        current_phase_id=current_phase_id,
        scenario_name=scenario,
        neighbor_messages="",
    )
    images = sort_images(step_dir, with_watermark=True)
    raw_images = sort_images(step_dir, with_watermark=False)
    return {
        "junction_id": jid,
        "step": step,
        "phase": parsed_response["phase"],
        "duration": parsed_response["duration"],
        "current_phase": current_phase_id,
        "image_count": len(images),
        "input_images": [str(path.relative_to(eval_path)) for path in images],
        "raw_images": [str(path.relative_to(eval_path)) for path in raw_images],
        "prompt": prompt,
        "thought": parsed_response["thought"],
        "response": parsed_response["model_response"],
        "response_full_text": parsed_response["raw"],
        "tls_state": {
            "can_perform_action": bool(tls_state.get("can_perform_action", False)),
            "this_phase_index": current_phase_id,
            "delta_time": tls_state.get("delta_time"),
            "phase2movements": tls_state.get("phase2movements", {}),
            "movement_ids": tls_state.get("movement_ids", []),
            "jam_length_vehicle": tls_state.get("jam_length_vehicle", []),
        },
    }


def load_eval_data(eval_path: Path, scenario: str) -> ViewerData:
    render_map = load_render_info(eval_path / "render_info")
    samples: Dict[str, Dict[int, Any]] = {}
    junctions: Dict[str, Any] = {}

    for jid_dir in sorted(p for p in eval_path.iterdir() if p.is_dir() and p.name != "render_info"):
        jid = jid_dir.name
        step_dirs = []
        for child in jid_dir.iterdir():
            if not child.is_dir():
                continue
            step = parse_step_from_name(child)
            if step is None:
                continue
            step_dirs.append((step, child))

        step_dirs.sort(key=lambda item: item[0])
        sample_map: Dict[int, Any] = {}
        for step, step_dir in step_dirs:
            render_state = render_map.get(step, {})
            sample = build_sample(eval_path, scenario, jid, step, step_dir, render_state)
            sample_map[step] = sample

        timeline = [
            {"step": sample["step"], "phase": sample["phase"], "duration": sample["duration"], "current_phase": sample["current_phase"]}
            for sample in sample_map.values()
        ]
        phase_hist: Dict[str, int] = {}
        duration_hist: Dict[str, int] = {}
        for item in timeline:
            if item["phase"] is not None:
                phase_hist[str(item["phase"])] = phase_hist.get(str(item["phase"]), 0) + 1
            if item["duration"] is not None:
                duration_hist[str(item["duration"])] = duration_hist.get(str(item["duration"]), 0) + 1

        samples[jid] = sample_map
        junctions[jid] = {
            "junction_id": jid,
            "steps": [step for step, _ in step_dirs],
            "sample_count": len(step_dirs),
            "timeline": timeline,
            "phase_hist": phase_hist,
            "duration_hist": duration_hist,
        }

    return ViewerData(eval_path=eval_path, scenario=scenario, junctions=junctions, samples=samples)


def get_default_selection(viewer_data: ViewerData) -> Dict[str, Any]:
    if not viewer_data.junctions:
        return {"junction_id": "", "step": None}
    jid = next(iter(viewer_data.junctions))
    steps = viewer_data.junctions[jid]["steps"]
    return {"junction_id": jid, "step": steps[0] if steps else None}


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eval Viewer</title>
  <style>
    :root {
      --bg: #f3efe6;
      --panel: #fffaf0;
      --card: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #ddd2bd;
      --accent: #b45309;
      --accent-soft: #f59e0b;
      --accent-deep: #7c2d12;
      --good: #047857;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Source Han Sans SC", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(245,158,11,0.16), transparent 28%),
        linear-gradient(180deg, #f8f4eb 0%, var(--bg) 100%);
    }
    .app {
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: rgba(255,250,240,0.88);
      backdrop-filter: blur(8px);
      padding: 20px 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    .brand {
      margin-bottom: 18px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }
    .brand h1 {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.1;
    }
    .brand p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }
    .section-title {
      margin: 18px 0 10px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 700;
    }
    .select, .search {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      font: inherit;
    }
    .junction-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .junction-btn {
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 14px;
      padding: 12px;
      text-align: left;
      cursor: pointer;
    }
    .junction-btn.active {
      border-color: var(--accent);
      box-shadow: 0 10px 24px rgba(180,83,9,0.12);
      transform: translateY(-1px);
    }
    .junction-btn strong {
      display: block;
      margin-bottom: 4px;
      font-size: 14px;
    }
    .junction-btn span {
      color: var(--muted);
      font-size: 12px;
    }
    .content {
      padding: 24px;
    }
    .hero {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    .hero h2 {
      margin: 0;
      font-size: 28px;
    }
    .hero p {
      margin: 6px 0 0;
      color: var(--muted);
    }
    .step-nav {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .step-nav button {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 12px;
      padding: 9px 12px;
      cursor: pointer;
      font: inherit;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }
    .card {
      grid-column: span 12;
      background: rgba(255,253,248,0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 10px 40px rgba(31,41,55,0.05);
    }
    .card h3 {
      margin: 0 0 12px;
      font-size: 16px;
    }
    .meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .pill {
      border-radius: 999px;
      padding: 6px 10px;
      background: #fff;
      border: 1px solid var(--line);
      font-size: 12px;
    }
    .images {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    figure {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
    }
    figure img {
      width: 100%;
      display: block;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      background: #f4f4f4;
    }
    figure figcaption {
      padding: 10px 12px;
      font-size: 12px;
      color: var(--muted);
      border-top: 1px solid var(--line);
    }
    .text-block {
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.6;
      padding: 14px;
      border-radius: 14px;
      background: #fff;
      border: 1px solid var(--line);
      max-height: 420px;
      overflow: auto;
    }
    .split-2 { grid-column: span 6; }
    .split-4 { grid-column: span 4; }
    .split-8 { grid-column: span 8; }
    .timeline-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .timeline-table th, .timeline-table td {
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
    }
    .timeline-table tbody tr {
      cursor: pointer;
    }
    .timeline-table tbody tr.active {
      background: rgba(245,158,11,0.12);
    }
    svg.chart {
      width: 100%;
      height: 260px;
      background:
        linear-gradient(180deg, rgba(245,158,11,0.06), transparent),
        #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
    }
    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: -1px;
      background: currentColor;
    }
    .empty {
      color: var(--muted);
      padding: 20px 0;
    }
    @media (max-width: 1080px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { position: static; height: auto; border-right: none; border-bottom: 1px solid var(--line); }
      .split-2, .split-4, .split-6, .split-8 { grid-column: span 12; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <h1>Eval Viewer</h1>
        <p id="dataset-meta"></p>
      </div>
      <div class="section-title">路口</div>
      <input id="junction-search" class="search" placeholder="搜索 intersection id" />
      <div id="junction-list" class="junction-list"></div>
    </aside>
    <main class="content">
      <div class="hero">
        <div>
          <h2 id="hero-title">-</h2>
          <p id="hero-subtitle">-</p>
        </div>
        <div class="step-nav">
          <button id="prev-step">上一条</button>
          <select id="step-select" class="select"></select>
          <button id="next-step">下一条</button>
        </div>
      </div>

      <section class="grid">
        <div class="card split-8">
          <h3>模型输入图像</h3>
          <div id="sample-meta" class="meta"></div>
          <div id="input-images" class="images"></div>
        </div>

        <div class="card split-4">
          <h3>动作摘要</h3>
          <div id="action-summary" class="meta"></div>
          <div id="tls-summary" class="text-block"></div>
        </div>

        <div class="card split-6">
          <h3>重建 Prompt</h3>
          <div id="prompt-text" class="text-block"></div>
        </div>

        <div class="card split-6">
          <h3>模型响应</h3>
          <div id="response-text" class="text-block"></div>
        </div>

        <div class="card split-6">
          <h3>Phase 时间序列</h3>
          <svg id="phase-chart" class="chart" viewBox="0 0 760 260"></svg>
          <div class="legend">
            <span style="color:#b45309">模型预测下一 phase</span>
            <span style="color:#0f766e">当前执行 phase</span>
          </div>
        </div>

        <div class="card split-6">
          <h3>绿灯时长时间序列</h3>
          <svg id="duration-chart" class="chart" viewBox="0 0 760 260"></svg>
          <div class="legend">
            <span style="color:#7c3aed">预测 duration</span>
          </div>
        </div>

        <div class="card">
          <h3>该路口决策序列</h3>
          <table class="timeline-table">
            <thead>
              <tr><th>Step</th><th>Predicted Next Phase</th><th>Duration</th><th>Current Active Phase</th></tr>
            </thead>
            <tbody id="timeline-body"></tbody>
          </table>
        </div>
      </section>
    </main>
  </div>

  <script>
    const state = {
      meta: null,
      junctionId: null,
      step: null,
      junctionMap: {},
      junctionDetailCache: new Map(),
      sampleCache: new Map(),
    };

    function esc(text) {
      return String(text ?? "").replace(/[&<>"]/g, (ch) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[ch]));
    }

    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function loadMeta() {
      state.meta = await fetchJson("/api/meta");
      state.junctionMap = Object.fromEntries(state.meta.junctions.map((j) => [j.junction_id, j]));
      document.getElementById("dataset-meta").textContent = `${state.meta.scenario} | ${state.meta.eval_path}`;
      renderJunctionList("");
      const initialJunction = state.meta.default_selection.junction_id;
      const initialStep = state.meta.default_selection.step;
      if (initialJunction) {
        await selectJunction(initialJunction, initialStep);
      }
    }

    async function loadJunctionDetail(junctionId) {
      let detail = state.junctionDetailCache.get(junctionId);
      if (!detail) {
        detail = await fetchJson(`/api/junction?junction_id=${encodeURIComponent(junctionId)}`);
        state.junctionDetailCache.set(junctionId, detail);
      }
      return detail;
    }

    function renderJunctionList(keyword) {
      const box = document.getElementById("junction-list");
      const normalized = keyword.trim().toLowerCase();
      const items = state.meta.junctions.filter((item) => item.junction_id.toLowerCase().includes(normalized));
      box.innerHTML = items.map((item) => `
        <button class="junction-btn ${item.junction_id === state.junctionId ? "active" : ""}" data-jid="${esc(item.junction_id)}">
          <strong>${esc(item.junction_id)}</strong>
          <span>${item.sample_count} steps</span>
        </button>
      `).join("");
      box.querySelectorAll(".junction-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const jid = btn.getAttribute("data-jid");
          const firstStep = state.junctionMap[jid]?.steps?.[0] ?? null;
          await selectJunction(jid, firstStep);
        });
      });
    }

    async function selectJunction(junctionId, preferredStep) {
      state.junctionId = junctionId;
      renderJunctionList(document.getElementById("junction-search").value);
      const junction = state.junctionMap[junctionId];
      const detail = await loadJunctionDetail(junctionId);
      document.getElementById("hero-title").textContent = junctionId;
      document.getElementById("hero-subtitle").textContent = `共 ${detail.sample_count} 个决策样本`;
      const select = document.getElementById("step-select");
      select.innerHTML = detail.steps.map((step) => `<option value="${step}">${step}</option>`).join("");
      const targetStep = preferredStep ?? detail.steps[0] ?? null;
      if (targetStep !== null) {
        select.value = String(targetStep);
        await selectStep(targetStep);
      }
      renderTimeline(detail.timeline);
      drawCharts(detail.timeline);
    }

    async function selectStep(step) {
      state.step = Number(step);
      document.getElementById("step-select").value = String(step);
      const cacheKey = `${state.junctionId}:${step}`;
      let sample = state.sampleCache.get(cacheKey);
      if (!sample) {
        sample = await fetchJson(`/api/sample?junction_id=${encodeURIComponent(state.junctionId)}&step=${step}`);
        state.sampleCache.set(cacheKey, sample);
      }
      renderSample(sample);
      highlightActiveRow();
    }

    function renderSample(sample) {
      const meta = document.getElementById("sample-meta");
      meta.innerHTML = `
        <span class="pill">Step ${sample.step}</span>
        <span class="pill">${sample.image_count} 张输入图</span>
        <span class="pill">Current Active Phase ${sample.current_phase}</span>
        <span class="pill">Predicted Next Phase ${sample.phase ?? "-"}</span>
        <span class="pill">Duration ${sample.duration ?? "-"}s</span>
      `;

      const inputImages = document.getElementById("input-images");
      inputImages.innerHTML = sample.input_images.length ? sample.input_images.map((path, index) => `
        <figure>
          <img src="/file/${encodeURIComponent(path)}" alt="${esc(path)}" />
          <figcaption>Input ${index + 1}: ${esc(path.split("/").slice(-1)[0])}</figcaption>
        </figure>
      `).join("") : `<div class="empty">当前样本没有可展示的模型输入图像。</div>`;

      document.getElementById("prompt-text").textContent = sample.prompt || "";
      document.getElementById("response-text").textContent = sample.response || "";
      document.getElementById("tls-summary").textContent = JSON.stringify(sample.tls_state, null, 2);
      document.getElementById("action-summary").innerHTML = `
        <span class="pill">predicted_next_phase=${sample.phase ?? "-"}</span>
        <span class="pill">current_active_phase=${sample.current_phase ?? "-"}</span>
        <span class="pill">duration=${sample.duration ?? "-"}s</span>
        <span class="pill">can_perform_action=${sample.tls_state.can_perform_action}</span>
      `;
    }

    function renderTimeline(timeline) {
      const tbody = document.getElementById("timeline-body");
      tbody.innerHTML = timeline.map((item) => `
        <tr data-step="${item.step}">
          <td>${item.step}</td>
          <td>${item.phase ?? "-"}</td>
          <td>${item.duration ?? "-"}s</td>
          <td>${item.current_phase}</td>
        </tr>
      `).join("");
      tbody.querySelectorAll("tr").forEach((tr) => {
        tr.addEventListener("click", async () => {
          await selectStep(Number(tr.getAttribute("data-step")));
        });
      });
      highlightActiveRow();
    }

    function highlightActiveRow() {
      document.querySelectorAll("#timeline-body tr").forEach((tr) => {
        tr.classList.toggle("active", Number(tr.getAttribute("data-step")) === state.step);
      });
    }

    function drawCharts(timeline) {
      drawLineChart(document.getElementById("phase-chart"), timeline, [
        { key: "phase", color: "#b45309", label: "pred" },
        { key: "current_phase", color: "#0f766e", label: "current" },
      ], 0, Math.max(3, ...timeline.map((item) => item.phase ?? 0), ...timeline.map((item) => item.current_phase ?? 0)));
      drawBarChart(document.getElementById("duration-chart"), timeline, "duration", "#7c3aed");
    }

    function drawLineChart(svg, timeline, seriesList, minY, maxY) {
      const W = 760, H = 260, pad = {l: 44, r: 20, t: 18, b: 34};
      const innerW = W - pad.l - pad.r, innerH = H - pad.t - pad.b;
      const x = (i) => pad.l + (timeline.length <= 1 ? innerW / 2 : innerW * i / (timeline.length - 1));
      const y = (v) => pad.t + innerH - ((v - minY) / Math.max(1, maxY - minY)) * innerH;
      const grid = [];
      for (let i = minY; i <= maxY; i++) {
        grid.push(`<line x1="${pad.l}" y1="${y(i)}" x2="${W - pad.r}" y2="${y(i)}" stroke="#eadfcb" />`);
        grid.push(`<text x="10" y="${y(i) + 4}" font-size="11" fill="#6b7280">${i}</text>`);
      }
      const xTicks = [];
      const tickCount = Math.min(6, timeline.length);
      for (let i = 0; i < tickCount; i++) {
        const idx = Math.round(i * (timeline.length - 1) / Math.max(1, tickCount - 1));
        xTicks.push(`<text x="${x(idx)}" y="${H - 10}" text-anchor="middle" font-size="11" fill="#6b7280">${timeline[idx]?.step ?? ""}</text>`);
      }
      const lines = seriesList.map((series) => {
        const points = timeline.map((item, idx) => `${x(idx)},${y(item[series.key] ?? minY)}`).join(" ");
        const circles = timeline.map((item, idx) => {
          const selected = item.step === state.step;
          return `<circle cx="${x(idx)}" cy="${y(item[series.key] ?? minY)}" r="${selected ? 5 : 3.2}" fill="${series.color}" opacity="${selected ? 1 : 0.85}" />`;
        }).join("");
        return `<polyline fill="none" stroke="${series.color}" stroke-width="3" points="${points}" />${circles}`;
      }).join("");
      svg.innerHTML = `
        <rect x="0" y="0" width="${W}" height="${H}" rx="14" fill="transparent" />
        ${grid.join("")}
        <line x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}" stroke="#cfbfa4" />
        <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H - pad.b}" stroke="#cfbfa4" />
        ${xTicks.join("")}
        ${lines}
      `;
    }

    function drawBarChart(svg, timeline, key, color) {
      const W = 760, H = 260, pad = {l: 44, r: 20, t: 18, b: 34};
      const innerW = W - pad.l - pad.r, innerH = H - pad.t - pad.b;
      const values = timeline.map((item) => item[key] ?? 0);
      const maxY = Math.max(40, ...values);
      const barWidth = Math.max(6, innerW / Math.max(1, timeline.length) - 4);
      const x = (i) => pad.l + i * innerW / Math.max(1, timeline.length) + 2;
      const y = (v) => pad.t + innerH - (v / maxY) * innerH;
      const grid = [];
      for (let i = 0; i <= 4; i++) {
        const val = Math.round(maxY * i / 4);
        const yy = y(val);
        grid.push(`<line x1="${pad.l}" y1="${yy}" x2="${W - pad.r}" y2="${yy}" stroke="#eadfcb" />`);
        grid.push(`<text x="10" y="${yy + 4}" font-size="11" fill="#6b7280">${val}</text>`);
      }
      const xTicks = [];
      const tickCount = Math.min(6, timeline.length);
      for (let i = 0; i < tickCount; i++) {
        const idx = Math.round(i * (timeline.length - 1) / Math.max(1, tickCount - 1));
        xTicks.push(`<text x="${x(idx) + barWidth / 2}" y="${H - 10}" text-anchor="middle" font-size="11" fill="#6b7280">${timeline[idx]?.step ?? ""}</text>`);
      }
      const bars = timeline.map((item, idx) => {
        const value = item[key] ?? 0;
        const selected = item.step === state.step;
        return `<rect x="${x(idx)}" y="${y(value)}" width="${barWidth}" height="${H - pad.b - y(value)}" rx="6" fill="${color}" opacity="${selected ? 0.92 : 0.72}" />`;
      }).join("");
      svg.innerHTML = `
        <rect x="0" y="0" width="${W}" height="${H}" rx="14" fill="transparent" />
        ${grid.join("")}
        <line x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}" stroke="#cfbfa4" />
        <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H - pad.b}" stroke="#cfbfa4" />
        ${xTicks.join("")}
        ${bars}
      `;
    }

    document.getElementById("junction-search").addEventListener("input", (e) => {
      renderJunctionList(e.target.value);
    });
    document.getElementById("step-select").addEventListener("change", async (e) => {
      await selectStep(Number(e.target.value));
    });
    document.getElementById("prev-step").addEventListener("click", async () => {
      const steps = state.junctionMap[state.junctionId]?.steps ?? [];
      const idx = steps.indexOf(state.step);
      if (idx > 0) await selectStep(steps[idx - 1]);
    });
    document.getElementById("next-step").addEventListener("click", async () => {
      const steps = state.junctionMap[state.junctionId]?.steps ?? [];
      const idx = steps.indexOf(state.step);
      if (idx >= 0 && idx < steps.length - 1) await selectStep(steps[idx + 1]);
    });

    loadMeta().catch((err) => {
      document.body.innerHTML = `<pre style="padding:24px;">${esc(err.message)}</pre>`;
    });
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "EvalViewer/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            text_response(self, HTML_PAGE)
            return
        if parsed.path == "/api/meta":
            self.handle_meta()
            return
        if parsed.path == "/api/junction":
            self.handle_junction(parsed.query)
            return
        if parsed.path == "/api/sample":
            self.handle_sample(parsed.query)
            return
        if parsed.path.startswith("/file/"):
            self.handle_file(parsed.path)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def handle_meta(self) -> None:
        assert VIEWER_DATA is not None
        default_selection = get_default_selection(VIEWER_DATA)
        payload = {
            "scenario": VIEWER_DATA.scenario,
            "eval_path": str(VIEWER_DATA.eval_path.relative_to(PROJECT_ROOT)),
            "junctions": [
                {
                    "junction_id": item["junction_id"],
                    "steps": item["steps"],
                    "sample_count": item["sample_count"],
                }
                for item in VIEWER_DATA.junctions.values()
            ],
            "default_selection": default_selection,
        }
        json_response(self, payload)

    def handle_junction(self, query: str) -> None:
        assert VIEWER_DATA is not None
        params = parse_qs(query)
        jid = (params.get("junction_id") or [""])[0]
        junction = VIEWER_DATA.junctions.get(jid)
        if junction is None:
            json_response(self, {"error": "junction not found"}, HTTPStatus.NOT_FOUND)
            return
        payload = {
            "junction_id": junction["junction_id"],
            "steps": junction["steps"],
            "sample_count": junction["sample_count"],
            "timeline": junction["timeline"],
            "phase_hist": junction["phase_hist"],
            "duration_hist": junction["duration_hist"],
        }
        json_response(self, payload)

    def handle_sample(self, query: str) -> None:
        assert VIEWER_DATA is not None
        params = parse_qs(query)
        jid = (params.get("junction_id") or [""])[0]
        step_text = (params.get("step") or [""])[0]
        try:
            step = int(step_text)
        except ValueError:
            json_response(self, {"error": "invalid step"}, HTTPStatus.BAD_REQUEST)
            return
        sample = VIEWER_DATA.samples.get(jid, {}).get(step)
        if sample is None:
            json_response(self, {"error": "sample not found"}, HTTPStatus.NOT_FOUND)
            return
        json_response(self, sample)

    def handle_file(self, path_text: str) -> None:
        assert VIEWER_DATA is not None
        rel_path = Path(unquote(path_text[len("/file/") :]))
        file_path = (VIEWER_DATA.eval_path / rel_path).resolve()
        eval_root = VIEWER_DATA.eval_path.resolve()
        if eval_root not in file_path.parents or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        data = file_path.read_bytes()
        mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    global VIEWER_DATA
    args = parse_args()
    eval_path = resolve_path(args.eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"eval_path not found: {eval_path}")
    VIEWER_DATA = load_eval_data(eval_path=eval_path, scenario=args.scenario)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    actual_host, actual_port = server.server_address[:2]
    print(f"Eval viewer running at http://{actual_host}:{actual_port}")
    print(f"Eval path: {eval_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
