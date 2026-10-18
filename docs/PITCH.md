# Pitch and demo video: RoleKV (2 minutes)

> Audience: hackathon judges who know what an LLM is but not necessarily what a KV cache is.
> `[[Double brackets]]` = fill from validated real logs only. If real results aren't ready, keep the
> placeholder wording in the "If results are not ready" box; never read a mock number aloud as a result.

## Timed outline

| Time | Beat | On screen |
|---|---|---|
| 0:00–0:20 | The problem: role switches re-read everything | Session replay, A2 rows |
| 0:20–0:45 | The idea: Activated LoRA + shared cache; H1 handoff | Session replay, A2 vs A5-H1 rows |
| 0:45–1:20 | Live demo: replay → cost vs quality → context sweep | All three tabs |
| 1:20–1:45 | Real results | The finding sentence at the top |
| 1:45–2:00 | Caveats and what's next | Cost vs quality table, then logo |

## Script (~290 words, ~2:00 at a natural pace)

**0:00 — Problem**

> Coding agents often split work into roles: one model writes code, another reviews it. A cheap way to build
> that is one small model with a small adapter per role. The catch: an adapter changes how the model reads
> *every* token. So each time the Coder hands over to the Reviewer, the model throws away its memory of the
> session, the task, the code and the test output, and reads it all again. Here, every bar is that re-read.

**0:20 — Idea**

> RoleKV trains each role as an *Activated* LoRA. It only switches on after its role header. Everything
> before that is read with the plain base model, so both roles share one memory, one KV cache. At a handoff we
> re-read just the previous role's short answer with base weights. We call that canonicalization, and it keeps
> the cache exactly what each role was trained on.

**0:45 — Demo**

> Same task, same session. In each turn, the top bar is per-role LoRAs: everything recomputed, in coral. The
> bar under it is RoleKV: almost everything reused, in cyan. Only new tokens and this small violet canonicalization get computed.
>
> Cheap doesn't help if the code gets worse, so here's every method: cost across, pass rate up.
>
> And as the shared context grows, there's more to avoid re-reading. This is the saving at each context size.

**1:20 — Results**

> In the same four-turn session, RoleKV computed about four times fewer tokens than per-role LoRAs, at
> every context size from one to eight thousand tokens. The first token after a role switch arrived in about
> 50 milliseconds whatever the context size. Per-role LoRAs went from about 90 to almost 600 milliseconds.
>
> And quality: on 100 MBPP problems, with about 1,800 tokens of shared context, RoleKV passed 61, and
> per-role LoRAs passed 59. That's no measurable drop, at a third of the prefill compute and three times faster
> role switches.
>
> *(If a judge asks about the other baselines:)* One shared LoRA, or simply reusing the mismatched cache,
> scored about the same here. What RoleKV adds is that the sharing is exact by construction, and every role
> keeps its own adapter. Our honest weak spot: on a second review, the aLoRA Reviewer tends to approve, because
> our training data never had a second review.

**1:45 — Caveats**

> This is a 1.5-billion-parameter model on function-level tasks, one run per method. Next: more roles, bigger
> models, real repositories. RoleKV makes the cost of every role switch measurable.

### If results are not ready

Replace the 1:20 block with:

> Our final runs are still finishing. Everything you just saw is the real pipeline on labelled mock data, and
> the numbers will come straight from the logs, including the canonicalization cost.

Keep the MOCK DATA banner visible on screen while saying it.

## Shot list

| # | Time | View | Action |
|---|---|---|---|
| 1 | 0:00 | Session replay | Load the page on the real log. Press Play replay so the bars grow turn by turn; keep attention on the coral A2 rows. |
| 2 | 0:20 | Session replay | Press Play replay again and pause on a reviewer turn, where the A5-H1 bar is mostly cyan under a full coral A2 bar. Hover a bar to show the tooltip with computed / reused / recomputed-at-handoff. |
| 3 | 0:45 | Session replay | Point the cursor at a violet segment ("recomputed at handoff"). |
| 4 | 0:58 | Cost vs quality | Switch tab; hover the A2 point, then the A5-H1 point. |
| 5 | 1:10 | Context sweep | Switch tab; hover the largest-context point. |
| 6 | 1:20 | Finding | Scroll to the top; zoom on the finding sentence (ratio, tokens per task, n, both pass rates). |
| 7 | 1:45 | Cost vs quality table | Show the run / machine column briefly (where the numbers come from). |
| 8 | 1:55 | Title | RoleKV wordmark + repo URL. |

## Recording checklist

- [ ] `demo/data/events.jsonl` holds the validated real log; banner says **Real data**; no "Accounting check failed" warning under the finding.
- [ ] Data validation shows 0 rejected lines and a clean accounting audit for the A2 and A5-H1 runs.
- [ ] Task chosen for the replay is noted (task_id: `[[id]]`). Pick it for clarity, and say if it's typical.
- [ ] Browser at 1440×900, zoom 100%, OS reduced-motion off (so the replay animates).
- [ ] Numbers spoken in 1:20 match the finding sentence exactly; run_ids recorded in README Results.
