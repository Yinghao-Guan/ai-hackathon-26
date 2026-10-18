"""Build the dashboard's real event log from the eval and bench results.

    python demo/collect_logs.py                 # newest 4090 eval run + the 4090 cost sweep
    python demo/collect_logs.py --check         # exit 1 if demo/data/ is stale or invalid
    python demo/collect_logs.py --eval eval/results/eval-1.5b-4090-202610180231.jsonl --sweep none

Writes demo/data/events.jsonl (the source logs concatenated, line for line) and demo/data/sources.json
(which files went in: path, line count, sha256 with LF line endings, the commit that last changed each one, and the run_ids it
holds). The dashboard shows sources.json under "Data validation", so every number on the page traces back
to a logged run.

Refuses mock- runs, a run_id that appears in two sources, and machines other than --machine (timings are
only comparable on one machine). Then validates the result with the dashboard's own rules
(demo/check_log.mjs, needs Node); without Node it says so and skips that step.

Stdlib only, like the rest of demo/. The output has no timestamps, so rebuilding from the same inputs is
byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "demo" / "data"
EVAL_DIR = ROOT / "eval" / "results"
BENCH_DIR = ROOT / "bench" / "results"
CHECKER = ROOT / "demo" / "check_log.mjs"


class CollectError(Exception):
    """A source log that must not reach the dashboard."""


def newest_eval(machine: str, eval_dir: Path = EVAL_DIR) -> Path | None:
    """The eval log for `machine` with the latest timestamp (eval-<model>-<machine>-<YYYYMMDDHHMM>.jsonl)."""
    pattern = re.compile(rf"^eval-.+-{re.escape(machine)}-(\d{{12}})\.jsonl$")
    found = [(m.group(1), p) for p in eval_dir.glob("eval-*.jsonl") if (m := pattern.match(p.name))]
    return max(found)[1] if found else None


def default_sweep(machine: str, bench_dir: Path = BENCH_DIR) -> Path | None:
    path = bench_dir / f"cost_sweep_{machine}.jsonl"
    return path if path.exists() else None


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def last_commit(path: Path) -> str | None:
    """Short hash of the last commit that touched `path`, or None outside git / for untracked files."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(path)], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def read_source(path: Path, machine: str) -> dict:
    """Read one source log, check it, and describe it for sources.json."""
    # Hash with LF line endings: git may check the same file out as CRLF on Windows, and the manifest
    # must not change with the platform.
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    runs: dict[str, None] = {}
    for i, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise CollectError(f"{rel(path)}:{i}: malformed JSON ({e.msg})") from None
        run_id = rec.get("run_id") if isinstance(rec, dict) else None
        if not isinstance(run_id, str):
            raise CollectError(f"{rel(path)}:{i}: no run_id")
        if run_id.startswith("mock-"):
            raise CollectError(f"{rel(path)}:{i}: mock run {run_id!r} cannot go into the real log")
        if rec.get("machine") != machine:
            raise CollectError(f"{rel(path)}:{i}: machine {rec.get('machine')!r}, expected {machine!r}; "
                               "timings are only comparable on one machine (see --machine)")
        runs.setdefault(run_id)
    return {
        "path": rel(path),
        "lines": len(lines),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "last_commit": last_commit(path),
        "run_ids": list(runs),
        "_lines": lines,
    }


def collect(sources: list[Path], machine: str) -> tuple[str, dict]:
    """Return (events.jsonl text, sources.json object) for the given source logs."""
    if not sources:
        raise CollectError("no source logs; pass --eval and/or --sweep")
    described, owner = [], {}
    for path in sources:
        if not path.exists():
            raise CollectError(f"{rel(path)} does not exist")
        src = read_source(path, machine)
        for run_id in src["run_ids"]:
            if run_id in owner:
                raise CollectError(f"run_id {run_id!r} is in both {owner[run_id]} and {src['path']}; "
                                   "it would be counted twice")
            owner[run_id] = src["path"]
        described.append(src)
    lines = [ln for src in described for ln in src.pop("_lines")]
    events = "".join(f"{ln}\n" for ln in lines)
    manifest = {
        "generated_by": "demo/collect_logs.py",
        "machine": machine,
        "events": {"path": "demo/data/events.jsonl", "lines": len(lines),
                   "sha256": hashlib.sha256(events.encode("utf-8")).hexdigest()},
        "sources": described,
    }
    return events, manifest


def manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, indent=2) + "\n"


def validate(events_path: Path) -> tuple[bool, str]:
    """Run the dashboard's validator. Returns (ok, message); ok is True with a note if Node is missing."""
    node = shutil.which("node")
    if node is None:
        return True, "node not found; skipped the dashboard validation (run `node demo/check_log.mjs` later)"
    result = subprocess.run([node, str(CHECKER), str(events_path)], capture_output=True, text=True)
    return result.returncode == 0, (result.stdout.strip() or result.stderr.strip())


def resolve_sources(args: argparse.Namespace) -> list[Path]:
    def pick(values: list[str] | None, default: Path | None) -> list[Path]:
        if values is None:
            return [default] if default else []
        if values == ["none"]:
            return []
        return [Path(v) if Path(v).is_absolute() else ROOT / v for v in values]

    return pick(args.eval, newest_eval(args.machine)) + pick(args.sweep, default_sweep(args.machine))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", default="4090", choices=("4090", "4070", "mac"))
    parser.add_argument("--eval", nargs="+", metavar="JSONL",
                        help="eval logs to include; default: the newest eval/results/eval-*-<machine>-*.jsonl; 'none' for none")
    parser.add_argument("--sweep", nargs="+", metavar="JSONL",
                        help="cost-sweep logs to include; default: bench/results/cost_sweep_<machine>.jsonl; 'none' for none")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR, help="where events.jsonl and sources.json go")
    parser.add_argument("--check", action="store_true",
                        help="don't write; exit 1 if the files in --out-dir differ from a fresh build")
    args = parser.parse_args(argv)

    try:
        events, manifest = collect(resolve_sources(args), args.machine)
    except CollectError as e:
        print(f"collect_logs: {e}", file=sys.stderr)
        return 1

    events_path, manifest_path = args.out_dir / "events.jsonl", args.out_dir / "sources.json"
    names = ", ".join(s["path"] for s in manifest["sources"])
    if args.check:
        stale = [p.name for p, want in ((events_path, events), (manifest_path, manifest_text(manifest)))
                 if not p.exists() or p.read_text(encoding="utf-8") != want]
        if stale:
            print(f"collect_logs: {', '.join(stale)} out of date with {names}; run python demo/collect_logs.py",
                  file=sys.stderr)
            return 1
        ok, message = validate(events_path)
        print(message)
        return 0 if ok else 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps the files byte-identical across platforms (no CRLF on Windows).
    events_path.write_text(events, encoding="utf-8", newline="\n")
    manifest_path.write_text(manifest_text(manifest), encoding="utf-8", newline="\n")
    print(f"wrote {rel(events_path)} ({manifest['events']['lines']} lines) from {names}")
    ok, message = validate(events_path)
    print(message)
    if not ok:
        print("collect_logs: the dashboard would reject or flag this log; see the summary above", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
