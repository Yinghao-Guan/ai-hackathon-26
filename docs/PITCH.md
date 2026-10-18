# Pitch and demo video: RoleKV (2 minutes)

> Audience: hackathon judges who know what an LLM is but not necessarily what a KV cache is.
> `[[Double brackets]]` = fill from validated real logs only. If real results aren't ready, keep the
> placeholder wording in the "If results are not ready" box; never read a mock number aloud as a result.

## Timed outline

| Time | Beat | On screen |
|---|---|---|
| 0:00–0:20 | The problem: role switches re-read everything | Session Replay, A2 panel only |
| 0:20–0:45 | The idea: Activated LoRA + shared cache; H1 handoff | Session Replay, both panels |
| 0:45–1:20 | Live demo: replay → cost vs quality → context sweep | All three tabs |
| 1:20–1:45 | Real results | Headline metrics strip |
| 1:45–2:00 | Caveats and what's next | Cost vs Quality table, then logo |

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

> Same task, same session. On the left, every turn recomputes everything, in coral. On the right, almost
> everything is reused, in cyan. Only new tokens and this small violet canonicalization get computed.
>
> Cheap doesn't help if the code gets worse, so here's every method: cost across, pass rate up.
>
> And as the shared context grows, there's more to avoid re-reading. This is the saving at each context size.

**1:20 — Results**

> On `[[N]]` MBPP problems with a 3,000-token shared context, the aLoRA agent computed `[[X.XX]]` times fewer
> prefill tokens than per-role LoRAs, `[[YY]]` percent fewer, with a pass rate of `[[A5-H1 %]]` versus
> `[[A2 %]]`. `[[If the gap is >0: "That's a [[d]]-point drop we're still working on." If ≈0: "with no
> measurable drop on this set."]]`

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
| 1 | 0:00 | Session Replay | Load the page on the real log. Crop to the A2 panel; press ▶ Replay so coral bars grow turn by turn. |
| 2 | 0:20 | Session Replay | Widen to both panels. Reset, then ▶ Replay; pause on the turn where the A5-H1 bar is mostly cyan. Hover a turn to show the tooltip with prefill / reused / canon. |
| 3 | 0:45 | Session Replay | Point the cursor at a violet segment ("canonicalized at handoff"). |
| 4 | 0:58 | Cost vs Quality | Switch tab; hover the A2 point, then the A5-H1 point. |
| 5 | 1:10 | Context Sweep | Switch tab; hover the largest-context point. |
| 6 | 1:20 | Header metrics | Zoom on the four metric cards: ratio, saved/task, both pass rates with n. |
| 7 | 1:45 | Cost vs Quality table | Show the run / machine column briefly (where the numbers come from). |
| 8 | 1:55 | Title | RoleKV wordmark + repo URL. |

## Recording checklist

- [ ] `demo/data/events.jsonl` holds the validated real log; banner says **REAL DATA**; no `CHECK` tag on the headline.
- [ ] Data validation shows 0 rejected lines and a clean accounting audit for the A2 and A5-H1 runs.
- [ ] Task chosen for the replay is noted (task_id: `[[id]]`). Pick it for clarity, and say if it's typical.
- [ ] Browser at 1440×900, zoom 100%, OS reduced-motion off (so the replay animates).
- [ ] Numbers spoken in 1:20 match the metric cards exactly; run_ids recorded in README Results.
