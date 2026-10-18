// Validate an event log with the dashboard's own rules (js/metrics.js), so the collector and the page
// never disagree about what counts as a valid record.
//
//   node demo/check_log.mjs demo/data/events.jsonl
//
// Prints a JSON summary. Exits 1 if any line is rejected, any accounting check fails, or the log is not
// real data (empty, or only mock- runs).
import { readFileSync } from "node:fs";
import { parseJSONL, isolateDataset, analyze } from "./js/metrics.js";

const path = process.argv[2];
if (!path) {
  console.error("usage: node demo/check_log.mjs <events.jsonl>");
  process.exit(2);
}
const parsed = parseJSONL(readFileSync(path, "utf8"));
const dataset = isolateDataset(parsed);
const analysis = dataset.mode === "empty" ? null : analyze(dataset);
const auditIssues = analysis ? [...analysis.audits.values()].reduce((n, a) => n + a.issues.length, 0) : 0;
const h = analysis?.headline;
const summary = {
  accepted: parsed.accepted,
  rejected: parsed.rejected,
  errors: parsed.errors.slice(0, 5),
  mode: dataset.mode,
  notes: dataset.notes,
  runs: analysis ? analysis.runs.size : 0,
  auditIssues,
  headline: h?.ok ? { ratio: +h.ratio.toFixed(3), pairedTasks: h.n } : { unavailable: h?.reason ?? "no data" },
};
console.log(JSON.stringify(summary));
process.exit(parsed.rejected === 0 && auditIssues === 0 && dataset.mode === "real" && dataset.notes.length === 0 ? 0 : 1);
