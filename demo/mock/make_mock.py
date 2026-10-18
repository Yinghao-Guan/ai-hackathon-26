"""Deterministic MOCK event-log generator for the RoleKV demo.

Writes demo/mock/mock_events.jsonl in the exact docs/CONTRACT.md §3 schema. Every run_id starts with
"mock-" so the UI can keep these synthetic numbers out of real-data calculations.

The numbers are NOT experimental results. They only exercise the demo: token accounting follows the
runtime rules in docs/TRACKS.md (A2 re-prefills the whole history each turn; one-cache methods prefill
only new tokens; A5-H1 recomputes header + body with base weights at each handoff), while pass rates
and timings are invented.

Usage:  python demo/mock/make_mock.py
"""

import json
import random
from pathlib import Path

SEED = 0
OUT = Path(__file__).with_name("mock_events.jsonl")

SYSTEM_TOKENS = 52   # SYSTEM segment
HEADER_TOKENS = 9    # ROLE header, e.g. <|im_start|>assistant\n<|role:coder|>\n
KV_MB_PER_TOKEN = 0.0273  # ~Qwen2.5-1.5B: 28 layers x 2 (K,V) x 2 kv-heads x 128 dims x 2 bytes

# Mock pass probabilities. Deliberately not all favourable to A5.
PASS_P = {"A0": 0.42, "A1": 0.56, "A2": 0.62, "A3": 0.36, "A5-H1": 0.58, "A5-H2": 0.51}

# (method, machine, ctx_k, n_tasks). Mirrors PLAN.md: timed A2 / A5-H1 on the 4090, the rest on the 4070.
RUNS = [
    ("A2", "4090", 3, 40), ("A5-H1", "4090", 3, 40),
    ("A0", "4070", 3, 40), ("A1", "4070", 3, 40), ("A3", "4070", 3, 40), ("A5-H2", "4070", 3, 40),
    ("A2", "4090", 1, 20), ("A5-H1", "4090", 1, 20),
    ("A2", "4090", 4, 20), ("A5-H1", "4090", 4, 20),
    ("A2", "4090", 8, 20), ("A5-H1", "4090", 8, 20),
]

# ms per prefill token, fixed overhead, ms per decode token
TIMING = {"4090": (0.065, 11.0, 24.0), "4070": (0.14, 18.0, 41.0)}

REVISE_NOTES = [
    "REVISE: the function returns None for an empty list; return 0 instead.",
    "REVISE: off-by-one in the loop bound; the last element is skipped.",
    "REVISE: the visible test expects a tuple, not a list.",
    "REVISE: handle the <empty> input case before indexing.",
    "REVISE: integer division is needed here; the test compares with an int.",
]


def run_id(method, ctx_k):
    slug = method.lower().replace("-", "")
    return f"mock-{slug}-1.5b-ctx{ctx_k}k"


def plan_session(rng, task_rng, passed):
    """Return the turn plan [(role, body_tokens, tool_tokens_after)] for one session."""
    # Failing sessions use more REVISE rounds; at most 2 per PLAN.md.
    revisions = task_rng.choice([0, 0, 1]) if passed else task_rng.choice([1, 2, 2])
    plan = []
    for r in range(revisions + 1):
        plan.append(("coder", rng.randint(70, 190), rng.randint(25, 140)))
        last = r == revisions
        reviewer_body = 4 if (last and passed) else rng.randint(14, 36)
        plan.append(("reviewer", reviewer_body, 0))
    return plan


def simulate(method, machine, ctx_tokens, plan, task_id, rid, rng):
    """Token / timing accounting for one session. Returns (turn_records, task_record_fields)."""
    ms_per_tok, overhead, ms_per_dec = TIMING[machine]
    one_cache = method != "A2"
    history = SYSTEM_TOKENS + ctx_tokens  # tokens before the first header
    cache = 0                             # tokens currently in the (single) cache
    pending = history                     # tokens appended but not yet in the cache
    turns, peak_kv, gpu_ms = [], 0.0, 0.0
    for i, (role, body, tool) in enumerate(plan):
        if one_cache:
            prefill = pending + HEADER_TOKENS
            reused = cache
        else:
            # A2: a fresh cache per turn, prefill everything with this role's LoRA.
            prefill = history + HEADER_TOKENS
            reused = 0
        handoff = i < len(plan) - 1
        canon = HEADER_TOKENS + body if (method == "A5-H1" and handoff) else 0
        ttft = overhead + prefill * ms_per_tok * rng.uniform(0.95, 1.08)
        turn_ms = ttft + body * ms_per_dec * rng.uniform(0.95, 1.05) + canon * ms_per_tok
        cache_after = (reused + prefill + body) if one_cache else (prefill + body)
        kv_mb = round(cache_after * KV_MB_PER_TOKEN, 1)
        peak_kv = max(peak_kv, kv_mb)
        gpu_ms += turn_ms
        if role == "coder":
            text = f"```python\ndef solve_{task_id}(xs):\n    ...\n```"
        else:
            text = "APPROVE" if body == 4 else rng.choice(REVISE_NOTES)
        turns.append({
            "type": "turn", "run_id": rid, "machine": machine, "method": method, "task_id": task_id,
            "turn": i, "role": role, "prefill_tokens": prefill, "reused_tokens": reused,
            "canon_tokens": canon, "decode_tokens": body, "ttft_ms": round(ttft, 1),
            "turn_ms": round(turn_ms, 1), "kv_mb": kv_mb, "text": text,
        })
        history += HEADER_TOKENS + body + tool
        cache = cache_after
        pending = tool  # the harness appends the TOOL segment after a coder turn
    totals = {
        "n_turns": len(turns),
        # Contract: total actual prefill compute, i.e. sum of (prefill + canon) over turns.
        "prefill_tokens_total": sum(t["prefill_tokens"] + t["canon_tokens"] for t in turns),
        "decode_tokens_total": sum(t["decode_tokens"] for t in turns),
        "gpu_s": round(gpu_ms / 1000, 2),
        "peak_kv_mb": peak_kv,
    }
    return turns, totals


def main():
    rng = random.Random(SEED)
    # Eval task ids from the MBPP test-split range; one difficulty draw per task is shared by all
    # methods so paired comparisons are correlated, as with real runs on the same problems.
    task_ids = sorted(rng.sample(range(11, 511), 40))
    difficulty = {t: rng.random() for t in task_ids}
    sweep_ids = task_ids[::2]

    lines = []
    for method, machine, ctx_k, n in RUNS:
        rid = run_id(method, ctx_k)
        ids = task_ids if n == len(task_ids) else sweep_ids
        for task_id in ids:
            # Same per-task context size and session shape across methods at a given ctx size.
            task_rng = random.Random(f"{SEED}-{task_id}-{ctx_k}")
            ctx_tokens = int(ctx_k * 1000 * task_rng.uniform(0.95, 1.08))
            passed = difficulty[task_id] < PASS_P[method]
            shape_rng = random.Random(f"{SEED}-{task_id}-{ctx_k}-{passed}")
            plan = plan_session(shape_rng, task_rng, passed)
            turns, totals = simulate(method, machine, ctx_tokens, plan, task_id, rid,
                                     random.Random(f"{SEED}-{rid}-{task_id}"))
            lines.extend(turns)
            lines.append({
                "type": "task", "run_id": rid, "machine": machine, "method": method,
                "task_id": task_id, "ctx_tokens": ctx_tokens, "passed": passed, **totals,
            })

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(lines)} records to {OUT}")


if __name__ == "__main__":
    main()
