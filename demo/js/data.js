// Event-log loading shared by the dashboard and the presentation stage.
import { parseJSONL, isolateDataset, analyze } from "./metrics.js";

const REAL_DEFAULT = "data/events.jsonl";
const REAL_SOURCES = "data/sources.json";
const MOCK_DEFAULT = "mock/mock_events.jsonl";

async function sha256Hex(text) {
  if (!globalThis.crypto?.subtle) return null; // not a secure context; skip the checksum, keep the line count
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * The manifest demo/collect_logs.py writes next to the real log: which result files it was built from.
 * Returns null if there is none, else {sources, machine, stale} where stale explains a mismatch with the
 * log actually loaded (a hand-edited or partially copied events.jsonl).
 */
async function loadProvenance(eventsText) {
  let manifest;
  try {
    const res = await fetch(REAL_SOURCES, { cache: "no-store" });
    if (!res.ok) return null;
    manifest = await res.json();
  } catch { return null; }
  if (!Array.isArray(manifest?.sources)) return null;
  const text = eventsText.replace(/\r\n/g, "\n"); // the collector hashes with LF line endings
  const lines = text.split("\n").filter((l) => l.trim()).length;
  const hash = await sha256Hex(text);
  let stale = null;
  if (lines !== manifest.events?.lines) stale = `sources.json lists ${manifest.events?.lines} lines, events.jsonl has ${lines}`;
  else if (hash && hash !== manifest.events?.sha256) stale = "events.jsonl does not match the checksum in sources.json";
  return { sources: manifest.sources, machine: manifest.machine, stale };
}

async function fetchText(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.text();
}

/** Replace the active dataset completely. Nothing from a previous file survives. */
export function ingest(text, source) {
  const parsed = parseJSONL(text);
  const dataset = isolateDataset(parsed);
  return { source, parsed, dataset, analysis: dataset.mode === "empty" ? null : analyze(dataset) };
}

export function failure(message) {
  return {
    source: { kind: "error", label: message },
    parsed: { turns: [], tasks: [], errors: [], accepted: 0, rejected: 0, nonEmpty: 0 },
    dataset: { mode: "empty", turns: [], tasks: [], notes: [] },
    analysis: null,
  };
}

/** ?log=<path>, then the real default log, then the labelled mock fixture. */
export async function loadDefault() {
  const param = new URLSearchParams(location.search).get("log");
  if (param) {
    try { return ingest(await fetchText(param), { kind: "url", label: param }); }
    catch (e) { return failure(`Could not load ?log=${param} (${e.message}).`); }
  }
  try {
    const text = await fetchText(REAL_DEFAULT);
    if (text.trim()) return ingest(text, { kind: "default-real", label: REAL_DEFAULT, provenance: await loadProvenance(text) });
  } catch { /* no real default log yet: fall back to mock below */ }
  try {
    return ingest(await fetchText(MOCK_DEFAULT), {
      kind: "fallback-mock", label: MOCK_DEFAULT,
      reason: `No real log at demo/${REAL_DEFAULT}, so the mock fixture is shown.`,
    });
  } catch (e) {
    return failure(`Could not load the mock fixture (${e.message}). Serve this folder over HTTP (python -m http.server) or load a file.`);
  }
}

export function readFile(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(ingest(String(reader.result), { kind: "file", label: file.name }));
    reader.onerror = () => resolve(failure(`Could not read ${file.name}.`));
    reader.readAsText(file);
  });
}
