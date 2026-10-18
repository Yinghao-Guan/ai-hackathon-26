"""The stage panel: two dsh sessions side by side, reused vs recomputed prompt tokens per step, real TTFT.

serve/openai_server.py serves it: GET /panel is the page (HTML, CSS and JS inline, so it works offline),
which polls GET /panel/data twice a second; POST /panel/reset starts a new round. The data are the
server's own --log JSONL, plus the request it is computing right now.

A dsh session is the run of a model id's requests that carry tools (dsh's title request has none) and
continue one task: a request starts a new session when its first user message (the task) differs from
the previous one's, or when it has no more messages than the previous one (a fresh conversation).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

LABELS = {"rolekv-reuse": "RoleKV · shared KV", "rolekv-noreuse": "Baseline · full prefill"}


def task_of(messages: Sequence[Mapping]) -> str:
    """The first user message's text: the task a dsh session was started with."""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return (content or "").strip()[:300]
    return ""


def read_log(path: Path | None, offset: int = 0) -> list[dict]:
    """Request records from byte `offset` on; a line still being written (not valid JSON yet) is skipped."""
    if path is None or not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(offset)
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def dsh_sessions(records: Sequence[Mapping]) -> dict[str, list[list[dict]]]:
    """Per model id, its dsh sessions in log order, each a list of steps (requests with tools)."""
    by_model: dict[str, list[list[dict]]] = {}
    for r in records:
        if not r.get("tools"):
            continue  # dsh's session-title request, or a client that is not an agent
        sessions = by_model.setdefault(r["model"], [])
        last = sessions[-1][-1] if sessions else None
        if last is None or r.get("task", "") != last.get("task", "") or r["messages"] <= last["messages"]:
            sessions.append([dict(r)])
        else:
            sessions[-1].append(dict(r))
    return by_model


def session_stats(steps: Sequence[Mapping]) -> dict:
    """Totals over the finished steps: reuse share, recomputed tokens, TTFT mean and median."""
    done = [s for s in steps if not s.get("running")]
    prompt = sum(s["prompt_tokens"] for s in done)
    reused = sum(s["reused_tokens"] for s in done)
    ttfts = [s["ttft_ms"] for s in done]
    return {
        "steps": len(done),
        "prompt": prompt,
        "reused": reused,
        "computed": prompt - reused,
        "reuse_share": reused / prompt if prompt else None,
        "ttft_mean_ms": statistics.fmean(ttfts) if ttfts else None,
        "ttft_median_ms": statistics.median(ttfts) if ttfts else None,
    }


def _step(i: int, r: Mapping) -> dict:
    return {
        "step": i,
        "prompt": r["prompt_tokens"],
        "reused": r.get("reused_tokens", 0),
        "computed": r["prompt_tokens"] - r.get("reused_tokens", 0),
        "ttft_ms": r.get("ttft_ms"),
        "calls": [c["name"] for c in r.get("tool_calls", [])],
        "running": bool(r.get("running")),
        "elapsed_s": r.get("elapsed_s"),
    }


def panel_data(records: Sequence[Mapping], models: Sequence[str], inflight: Mapping | None = None) -> dict:
    """The latest dsh session of each model id, ready to draw. `inflight` is the request being computed."""
    rows = list(records)
    if inflight and inflight.get("tools"):
        rows.append({**inflight, "running": True})
    sessions = dsh_sessions(rows)
    columns = []
    for model in models:
        steps = sessions.get(model, [[]])[-1]
        columns.append({
            "model": model,
            "label": LABELS.get(model, model),
            "task": steps[0].get("task", "") if steps else "",
            "steps": [_step(i, r) for i, r in enumerate(steps, start=1)],
            "stats": session_stats(steps),
        })
    biggest = max((s["prompt"] for c in columns for s in c["steps"]), default=0)
    return {"columns": columns, "max_prompt": biggest}


PANEL_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoleKV KV Panel</title>
<style>
:root {
  --bg: #0a0f1e; --card: #141c33; --line: #2a3557; --text: #f8fafc; --muted: #a5b1cf;
  --reused: #38bdf8; --computed: #fb923c; --run: #64748b; --good: #4ade80; --bad: #f87171;
  font-size: clamp(14px, 1.15vw, 26px);
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header { display: flex; align-items: center; gap: 1.5rem; padding: 0.9rem 1.4rem; border-bottom: 2px solid var(--line); }
h1 { font-size: 1.6rem; margin: 0; flex: 1; letter-spacing: 0.01em; }
.legend { display: flex; gap: 1.2rem; font-size: 1.1rem; color: var(--muted); }
.sw { display: inline-block; width: 1.1rem; height: 1.1rem; border-radius: 3px; vertical-align: -0.15rem; margin-right: 0.4rem; }
button { font: inherit; font-size: 1.05rem; padding: 0.45rem 1rem; border-radius: 8px; border: 2px solid var(--line);
  background: var(--card); color: var(--text); cursor: pointer; }
button:hover { border-color: var(--muted); }
#status { width: 0.8rem; height: 0.8rem; border-radius: 50%; background: var(--good); }
#status.off { background: var(--bad); }
main { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; padding: 1.2rem 1.4rem; height: calc(100% - 4.6rem); }
.col { background: var(--card); border: 2px solid var(--line); border-radius: 14px; padding: 1.1rem 1.3rem;
  display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.name { font-size: 1.7rem; font-weight: 750; }
.model { font-size: 1rem; color: var(--muted); font-family: ui-monospace, Menlo, monospace; }
.task { font-size: 1.05rem; color: var(--muted); margin-top: 0.35rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin: 1rem 0 0.9rem; }
.stat { background: #0e1528; border-radius: 10px; padding: 0.6rem 0.8rem; }
.stat .k { font-size: 0.9rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.stat .v { font-size: 2.5rem; font-weight: 800; line-height: 1.1; font-variant-numeric: tabular-nums; }
.stat .s { font-size: 1rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.v.reused { color: var(--reused); } .v.computed { color: var(--computed); }
.rows { flex: 1; display: flex; flex-direction: column; gap: 0.5rem; overflow: hidden; justify-content: flex-start; }
.row { display: grid; grid-template-columns: 3.2rem 1fr 6.6rem; align-items: center; gap: 0.7rem; }
.row .n { font-size: 1.25rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.bar { height: 2.3rem; display: flex; border-radius: 6px; overflow: hidden; background: #0e1528; }
.seg { height: 100%; display: flex; align-items: center; justify-content: center; overflow: hidden; white-space: nowrap;
  font-size: 1rem; font-weight: 700; color: #0a0f1e; font-variant-numeric: tabular-nums; }
.seg.reused { background: var(--reused); } .seg.computed { background: var(--computed); }
.seg.run { background: repeating-linear-gradient(-45deg, var(--run) 0 14px, #475569 14px 28px);
  background-size: 40px 40px; animation: slide 1s linear infinite; color: var(--text); }
@keyframes slide { to { background-position: 40px 0; } }
.ttft { font-size: 1.6rem; font-weight: 800; text-align: right; font-variant-numeric: tabular-nums; }
.ttft.run { color: var(--muted); font-size: 1.2rem; }
.empty { color: var(--muted); font-size: 1.3rem; margin-top: 1rem; }
.foot { font-size: 0.95rem; color: var(--muted); margin-top: 0.6rem; }
</style>
</head>
<body>
<header>
  <h1>Prompt tokens per dsh step: reused vs recomputed</h1>
  <div class="legend">
    <span><span class="sw" style="background:var(--reused)"></span>reused KV</span>
    <span><span class="sw" style="background:var(--computed)"></span>recomputed</span>
    <span>right: server TTFT</span>
  </div>
  <button id="reset" title="New round: show only requests logged from now on">New round</button>
  <span id="status" title="data feed"></span>
</header>
<main id="cols"></main>
<script>
const qs = new URLSearchParams(location.search);
const models = [qs.get("left") || "rolekv-reuse", qs.get("right") || "rolekv-noreuse"];
const maxRows = Math.max(1, parseInt(qs.get("rows") || "10", 10));
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const num = n => n == null ? "–" : Math.round(n).toLocaleString("en-US");
const sec = ms => ms == null ? "–" : (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + " s";
const pct = x => x == null ? "–" : Math.round(x * 100) + "%";

function column(c, maxPrompt) {
  const st = c.stats;
  const shown = c.steps.slice(-maxRows);
  const rows = shown.map(s => {
    const w = t => maxPrompt ? (100 * t / maxPrompt).toFixed(2) + "%" : "0";
    if (s.running) {
      return `<div class="row"><div class="n">${s.step}</div>
        <div class="bar"><div class="seg run" style="width:${w(s.prompt)}">computing ${num(s.prompt)} tok</div></div>
        <div class="ttft run">${s.elapsed_s == null ? "…" : s.elapsed_s.toFixed(1) + " s…"}</div></div>`;
    }
    const label = (t, word, frac) => frac > 0.16 ? `${num(t)} ${word}` : "";
    return `<div class="row"><div class="n">${s.step}</div>
      <div class="bar" title="${num(s.reused)} reused + ${num(s.computed)} recomputed = ${num(s.prompt)} tokens">
        <div class="seg reused" style="width:${w(s.reused)}">${label(s.reused, "reused", s.reused / maxPrompt)}</div>
        <div class="seg computed" style="width:${w(s.computed)}">${label(s.computed, "new", s.computed / maxPrompt)}</div>
      </div>
      <div class="ttft">${sec(s.ttft_ms)}</div></div>`;
  }).join("");
  const hidden = c.steps.length - shown.length;
  return `<section class="col">
    <div class="name">${esc(c.label)}</div>
    <div class="model">${esc(c.model)}</div>
    <div class="task">${c.task ? esc(c.task) : "&nbsp;"}</div>
    <div class="stats">
      <div class="stat"><div class="k">reused</div><div class="v reused">${pct(st.reuse_share)}</div>
        <div class="s">${num(st.reused)} of ${num(st.prompt)} tok</div></div>
      <div class="stat"><div class="k">recomputed</div><div class="v computed">${num(st.computed)}</div>
        <div class="s">tokens, ${st.steps} step${st.steps == 1 ? "" : "s"}</div></div>
      <div class="stat"><div class="k">TTFT mean</div><div class="v">${sec(st.ttft_mean_ms)}</div>
        <div class="s">median ${sec(st.ttft_median_ms)}</div></div>
    </div>
    <div class="rows">${rows || '<div class="empty">Waiting for a dsh session on this model…</div>'}</div>
    <div class="foot">${hidden > 0 ? `${hidden} earlier step${hidden == 1 ? "" : "s"} not shown · ` : ""}TTFT measured by the server</div>
  </section>`;
}

let last = "";
async function tick() {
  try {
    const r = await fetch("/panel/data?models=" + encodeURIComponent(models.join(",")), {cache: "no-store"});
    const d = await r.json();
    const key = JSON.stringify(d);
    if (key !== last) {
      last = key;
      document.getElementById("cols").innerHTML = d.columns.map(c => column(c, d.max_prompt)).join("");
    }
    document.getElementById("status").className = "";
  } catch (e) {
    document.getElementById("status").className = "off";
  }
}
document.getElementById("reset").onclick = async () => { await fetch("/panel/reset", {method: "POST"}); last = ""; tick(); };
tick();
setInterval(tick, 400);
</script>
</body>
</html>
"""
