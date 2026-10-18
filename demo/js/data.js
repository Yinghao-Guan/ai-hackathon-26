// Event-log loading shared by the dashboard and the presentation stage.
import { parseJSONL, isolateDataset, analyze } from "./metrics.js";

const REAL_DEFAULT = "data/events.jsonl";
const MOCK_DEFAULT = "mock/mock_events.jsonl";

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
    if (text.trim()) return ingest(text, { kind: "default-real", label: REAL_DEFAULT });
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
