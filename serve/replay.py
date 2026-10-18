"""Replay recorded harness traffic against the server, and summarize request logs.

    # Send every dumped request (in order) again, as model rolekv-noreuse, to a running server:
    uv run --no-sync python -m serve.replay send tmp/serve_logs/dumps --model rolekv-noreuse

    # Alternate two role adapters request by request (a server started with --adapter-dir):
    uv run --no-sync python -m serve.replay send tmp/serve_logs/dumps --main-only --model rolekv-coder,rolekv-reviewer

    # Totals for one or more request logs (the server's --log JSONL):
    uv run --no-sync python -m serve.replay summary tmp/serve_logs/reuse.jsonl tmp/serve_logs/noreuse.jsonl

A replay sends the exact prompts the harness sent, so both KV policies see the same traffic even when
greedy decoding would drift between live runs. The prompts are fixed; only the replies are regenerated.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def send(dump_dir: Path, models: list[str], url: str, *, main_only: bool = False) -> None:
    """Send each dumped request in order; request i goes to models[i % len(models)] (alternate roles)."""
    pairs = [(path, json.loads(path.read_text())) for path in sorted(dump_dir.glob("*.request.json"))]
    if main_only:  # drop side requests (dsh's session title) that carry no tools
        pairs = [(path, body) for path, body in pairs if body.get("tools")]
    for i, (path, body) in enumerate(pairs):
        model = models[i % len(models)]
        body["model"] = model
        body["stream"] = False
        body.pop("stream_options", None)
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as resp:
            usage = json.loads(resp.read())["usage"]
        print(f"{path.name} {model}: prompt={usage['prompt_tokens']} "
              f"cached={usage['prompt_tokens_details']['cached_tokens']}", flush=True)


def summarize(rows: list[dict]) -> dict:
    calls = [c for r in rows for c in r["tool_calls"]]
    total = lambda key: sum(r[key] for r in rows)  # noqa: E731
    prompt = total("prompt_tokens")
    return {
        "requests": len(rows),
        "prompt_tokens": prompt,
        "mean_prompt_tokens": round(prompt / max(1, len(rows))),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in rows), default=0),
        "mean_head_tokens": round(total("head_tokens") / max(1, len(rows))),
        "reused_tokens": total("reused_tokens"),
        "computed_tokens": total("computed_tokens"),
        "reused_share": round(total("reused_tokens") / max(1, prompt), 3),
        "decode_tokens": total("decode_tokens"),
        "ttft_s": round(total("ttft_ms") / 1000, 1),
        "decode_s": round(total("decode_ms") / 1000, 1),
        "wall_s": round(total("wall_ms") / 1000, 1),
        "replies_with_calls": sum(1 for r in rows if r["tool_calls"]),
        "parse_modes": {m: sum(1 for r in rows if r["parse_mode"] == m) for m in ("tagged", "bare", "text")},
        "tool_calls": len(calls),
        "tool_calls_valid": sum(1 for c in calls if c["known"] and not c["missing"]),
        "tagged_blocks_unparsed": sum(len(r["parse_errors"]) for r in rows),
        "truncated": sum(1 for r in rows if r["finish"] == "length"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("dump_dir", type=Path)
    s.add_argument("--model", default="rolekv-noreuse", help="model id, or a comma-separated list to alternate")
    s.add_argument("--main-only", action="store_true", help="skip requests without tools (session titles)")
    s.add_argument("--url", default="http://127.0.0.1:8765/v1/chat/completions")
    m = sub.add_parser("summary")
    m.add_argument("logs", type=Path, nargs="+")
    args = p.parse_args()
    if args.cmd == "send":
        send(args.dump_dir, args.model.split(","), args.url, main_only=args.main_only)
    else:
        for log in args.logs:
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            print(log, json.dumps(summarize(rows), indent=1))


if __name__ == "__main__":
    main()
