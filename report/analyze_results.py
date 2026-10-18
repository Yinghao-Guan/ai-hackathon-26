"""Audit committed logs and regenerate the report's numbers, tables and plot.

Run from any directory with Python 3.10+; uses only the standard library.
No models, generated programs, or MBPP tests are executed.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
EVAL = ROOT / "eval/results/eval-1.5b-4090-202610180231.jsonl"
SWEEP = ROOT / "bench/results/cost_sweep_4090.jsonl"
METHODS = ("A0", "A1", "A2", "A3", "A5-H1", "A5-H2")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(rows):
    tasks, turns = {}, defaultdict(list)
    for row in rows:
        key = row["run_id"], row["task_id"]
        if row["type"] == "task":
            assert key not in tasks, f"Duplicate task {key}"
            tasks[key] = row
        else:
            assert row["type"] == "turn"
            turns[key].append(row)
    assert tasks.keys() == turns.keys(), "Orphaned task/turn rows"
    for key, task in tasks.items():
        seq = turns[key]
        assert len(seq) == task["n_turns"]
        assert [t["turn"] for t in seq] == list(range(len(seq)))
        assert all(t["method"] == task["method"] for t in seq)
        assert all(t["role"] == ("coder" if i % 2 == 0 else "reviewer")
                   for i, t in enumerate(seq))
        assert sum(t["prefill_tokens"] + t["canon_tokens"] for t in seq) == task["prefill_tokens_total"]
        assert sum(t["canon_tokens"] for t in seq) == task["canon_tokens_total"]
        assert sum(t["decode_tokens"] for t in seq) == task["decode_tokens_total"]
        assert math.isclose(sum(t["turn_ms"] for t in seq) / 1000, task["gpu_s"], abs_tol=1e-8)
        assert all(t["turn_ms"] >= t["ttft_ms"] > 0 for t in seq)
    return tasks, turns


def wilson(passed, n):
    z = 1.959963984540054
    p = passed / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def analyze():
    tasks, turns = validate(read_rows(EVAL))
    method_tasks = {m: {t["task_id"]: t for t in tasks.values() if t["method"] == m} for m in METHODS}
    ids = list(method_tasks["A2"])
    assert all(len(method_tasks[m]) == 100 and set(method_tasks[m]) == set(ids) for m in METHODS)
    assert all(method_tasks[m][i]["ctx_tokens"] == method_tasks["A2"][i]["ctx_tokens"]
               for m in METHODS for i in ids)
    assert all(t["machine"] == "4090" for t in tasks.values())
    training_ids = {r["task_id"] for r in read_rows(ROOT / "data/trajectories.jsonl")}
    assert len(training_ids) == 280 and not training_ids.intersection(ids)
    summary = {}
    for method in METHODS:
        group = list(method_tasks[method].values())
        seq = [t for task in group for t in turns[(task["run_id"], task["task_id"])]]
        review = [t for t in seq if t["role"] == "reviewer"]
        second = [t for t in review if t["turn"] == 3]
        approved = [t for t in second if t["verdict"] == "APPROVE"]
        # A second-review APPROVE ends the session, so final-code scoring applies
        # to exactly the code just approved; this is not an intermediate score.
        assert all(method_tasks[method][t["task_id"]]["n_turns"] == 4 for t in approved)
        passed = sum(t["passed"] for t in group)
        summary[method] = {
            "n": len(group), "passed": passed, "wilson_95": wilson(passed, len(group)),
            "mean_prefill_tokens": stats.mean(t["prefill_tokens_total"] for t in group),
            "mean_canon_tokens": stats.mean(t["canon_tokens_total"] for t in group),
            "median_switch_ttft_ms": stats.median(t["ttft_ms"] for t in seq if t["turn"] >= 1),
            "mean_model_turn_s": stats.mean(t["gpu_s"] for t in group),
            "mean_turns": stats.mean(t["n_turns"] for t in group),
            "mean_generated_tokens": stats.mean(t["decode_tokens_total"] for t in group),
            "review_verdicts": dict(Counter(t["verdict"] for t in review)),
            "second_reviews": len(second), "second_approvals": len(approved),
            "failed_second_approvals": sum(not method_tasks[method][t["task_id"]]["passed"] for t in approved),
            "truncated_turns": dict(Counter(t["role"] for t in seq if t["truncated"])),
        }
    paired = Counter((method_tasks["A2"][i]["passed"], method_tasks["A5-H1"][i]["passed"]) for i in ids)
    a2_only, h1_only = paired[True, False], paired[False, True]
    discordant = a2_only + h1_only
    paired_result = {
        "both_pass": paired[True, True], "only_A2": a2_only, "only_H1": h1_only,
        "neither_pass": paired[False, False],
        "exact_mcnemar_p": min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(a2_only, h1_only) + 1)) / 2**discordant),
    }
    example = {"task_id": 135, "selection": "Illustrates unchanged code followed by a false second approval; not representative."}
    for method in ("A2", "A5-H1"):
        task = method_tasks[method][example["task_id"]]
        seq = turns[(task["run_id"], task["task_id"])]
        excerpts = [re.search(r"return ([^\n]+)", t["text"]).group(1)
                    if t["role"] == "coder" else t["verdict"] for t in seq]
        example[method] = {"passed": task["passed"], "prefill_tokens": task["prefill_tokens_total"],
                           "canon_tokens": task["canon_tokens_total"], "excerpts_in_turn_order": excerpts}
    assert example["A2"]["passed"] and not example["A5-H1"]["passed"]
    assert example["A2"]["excerpts_in_turn_order"] == ["n * (2 * n - 1)", "APPROVE"]
    assert example["A5-H1"]["excerpts_in_turn_order"] == ["n * (2 * n * n - 1)", "REVISE", "n * (2 * n * n - 1)", "APPROVE"]
    sweep_tasks, sweep_turns = validate(read_rows(SWEEP))
    groups = defaultdict(list)
    for task in sweep_tasks.values():
        groups[task["ctx_tokens"], task["method"]].append(task)
    sweep = []
    for (ctx, method), group in sorted(groups.items()):
        seq = [t for task in group for t in sweep_turns[(task["run_id"], task["task_id"])]]
        assert len(group) == 5
        assert {t["decode_tokens_total"] for t in group} == {197}
        assert all(t["n_turns"] == 4 for t in group)
        assert len({t["prefill_tokens_total"] for t in group}) == 1
        sweep.append({
            "ctx_tokens": ctx, "method": method, "repeats": len(group),
            "prefill_tokens": group[0]["prefill_tokens_total"], "canon_tokens": group[0]["canon_tokens_total"],
            "generated_tokens": 197,
            "median_switch_ttft_ms": stats.median(t["ttft_ms"] for t in seq if t["turn"] >= 1),
            "median_model_turn_s": stats.median(t["gpu_s"] for t in group),
            "min_model_turn_s": min(t["gpu_s"] for t in group),
            "max_model_turn_s": max(t["gpu_s"] for t in group),
            "median_decode_ms_per_token": stats.median((t["turn_ms"] - t["ttft_ms"]) / (t["decode_tokens"] - 1)
                                                       for t in seq if t["decode_tokens"] > 1),
        })
    training = {}
    paths = [EVAL, SWEEP, ROOT / "data/trajectories.jsonl",
             ROOT / "data/sft_coder.jsonl", ROOT / "data/sft_reviewer.jsonl"]
    for path in sorted((ROOT / "train/results/1.5b").glob("*/train_run.json")):
        record = json.loads(path.read_text())
        assert record["status"] == "passed" and not record["gate_failures"]
        assert record["updates"] == record["planned_updates"]
        for file in record["data"]["files"]:
            assert digest(ROOT / file["path"]) == file["sha256_lf"]
        training[record["adapter"]] = {k: record[k] for k in (
            "hyperparameters", "updates", "trainable_params", "model_revision", "versions", "git",
            "status", "reload_err", "pre_header_diff", "train_wall_s", "peak_gpu_mb")}
        paths.append(path)
    return {
        "audit_date": "2026-10-18", "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "evaluation": summary, "paired_A2_H1": paired_result, "fixed_trace": sweep, "training": training,
        "illustrative_case": example,
        "eval_task_ids_in_log_order": ids,
        "context_tokens": {"min": min(t["ctx_tokens"] for t in method_tasks["A2"].values()),
                           "median": stats.median(t["ctx_tokens"] for t in method_tasks["A2"].values()),
                           "max": max(t["ctx_tokens"] for t in method_tasks["A2"].values())},
        "checks": "Unique paired task sets disjoint from training trajectories; matching contexts; contiguous turns; prefill/canon/decode sums; turn-time sums; training gates and input hashes.",
        "scope": "Offline log aggregation only; no fresh GPU measurements or execution of generated code.",
    }


def write_tables(result):
    directory = REPORT / "tables"
    directory.mkdir(exist_ok=True)
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"Method & Pass / 100 & Prefill tokens & Canon tokens & Switch TTFT (ms) & Model time (s) & Turns & Generated tokens \\", r"\midrule"]
    for method, row in result["evaluation"].items():
        lines.append(f"{method} & {row['passed']} & {row['mean_prefill_tokens']:,.0f} & {row['mean_canon_tokens']:,.0f} & "
                     f"{row['median_switch_ttft_ms']:.1f} & {row['mean_model_turn_s']:.2f} & {row['mean_turns']:.2f} & {row['mean_generated_tokens']:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (directory / "eval_summary.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Method & Reviews & Approvals & Failed / approved \\", r"\midrule"]
    for method, row in result["evaluation"].items():
        denominator = row["second_approvals"]
        frac = f"{row['failed_second_approvals']} / {denominator}" if denominator else "--"
        lines.append(f"{method} & {row['second_reviews']} & {denominator} & {frac} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (directory / "reviewer_summary.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{@{}ll@{}}"]
    labels = ("Coder 1", "Reviewer 1", "Coder 2", "Reviewer 2")
    for label, excerpt in zip(labels, result["illustrative_case"]["A5-H1"]["excerpts_in_turn_order"]):
        lines.append(label + r" & \texttt{" + excerpt + r"} \\")
    lines.append(r"\end{tabular}")
    (directory / "case_study.tex").write_text("\n".join(lines) + "\n")


def write_plot(result):
    # Native PGFPlots vector figure; no plotting dependency beyond TeX Live.
    lines = [r"\begin{tikzpicture}", r"\begin{groupplot}[",
             r"group style={group size=3 by 1,horizontal sep=1.1cm},",
             r"width=0.255\textwidth,height=4.1cm,scale only axis,",
             r"tick label style={font=\scriptsize},label style={font=\footnotesize},",
             r"title style={font=\footnotesize},xmin=0.8,xmax=8.5,xtick={1,2,4,8},",
             r"xlabel={Shared context (1,000 tokens)},grid=major,grid style={black!8},",
             r"legend style={font=\footnotesize,draw=none,at={(0.5,1.25)},anchor=south,legend columns=3},",
             r"legend cell align=left]"]
    panels = [("(a) Prefill volume", "Prefill tokens (1,000)", "prefill_tokens", 1000),
              ("(b) Role-switch latency", "Switch TTFT (ms)", "median_switch_ttft_ms", 1),
              ("(c) Model-turn time", "Model time per session (s)", "median_model_turn_s", 1)]
    styles = {"A2": "blue!65!black,mark=square*", "A5-H1": "orange!85!black,mark=*",
              "A5-H2": "black!55,dashed,mark=triangle*"}
    for index, (title, ylabel, field, divisor) in enumerate(panels):
        options = f"title={{{title}}},ylabel={{{ylabel}}}"
        if index < 2:
            options += ",ymin=0"
        else:
            options += ",ymin=6.3,ymax=8.7"
        lines.append(r"\nextgroupplot[" + options + "]")
        for method, style in styles.items():
            group = [r for r in result["fixed_trace"] if r["method"] == method]
            if index == 2:
                lines.append(r"\addplot+[thick," + style + r",error bars/.cd,y dir=both,y explicit] coordinates {")
                for row in group:
                    y = row[field]
                    lines.append(f"({row['ctx_tokens']/1000:.3f},{y:.6f}) += (0,{row['max_model_turn_s']-y:.6f}) -= (0,{y-row['min_model_turn_s']:.6f})")
            else:
                lines.append(r"\addplot+[thick," + style + "] coordinates {")
                for row in group:
                    lines.append(f"({row['ctx_tokens']/1000:.3f},{row[field]/divisor:.6f})")
            lines.append("};")
            if index == 1:
                lines.append(r"\addlegendentry{" + method + "}")
    lines += [r"\end{groupplot}", r"\end{tikzpicture}"]
    (REPORT / "figures/cost_scaling.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    result = analyze()
    (REPORT / "results_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_tables(result)
    write_plot(result)
    print("Audited 600 eval tasks and 60 fixed-trace sessions; regenerated report tables and plot.")
    print("A2 vs H1:", result["paired_A2_H1"])
