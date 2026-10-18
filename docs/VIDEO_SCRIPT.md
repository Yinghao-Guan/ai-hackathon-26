# Demo video script: RoleKV (about 2:20)

> For the Devpost video. It uses the current UI: the dark 3D homepage, the dark live demo with the trained
> adapters on the 4090, and DeepSeek Harness on the Mac. The live judging pitch stays in [PITCH.md](PITCH.md).
> Every number below comes from `README.md` › Results, `bench/results/cost_sweep_4090.md` or
> `serve/README.md`. `[[Double brackets]]` = read it off the screen during the take; never say a number that
> is not on screen or in those files.

## Timed outline

| Time | Beat | Screen | Where to record |
|---|---|---|---|
| 0:00–0:10 | Hook | Homepage, 3D scene | Any machine |
| 0:10–0:35 | Problem: every role switch re-reads the session | Homepage replay, left slab (A2) | Any machine |
| 0:35–1:00 | Idea: Activated LoRA, one shared cache, exact handoff | Homepage replay, right slab (RoleKV) | Any machine |
| 1:00–1:30 | Live: task 454 with the real 1.5B adapters | `demo/live/` | **4090**, Windows browser |
| 1:30–1:55 | Results over 100 tasks and the context sweep | Dashboard: Cost vs quality, Context sweep | Any machine |
| 1:55–2:10 | A real agent harness on the same cache | dsh web + stage panel side by side | **Mac** |
| 2:10–2:20 | Honest limits and close | Homepage title | Any machine |

## Narration (~330 words)

**0:00 — Hook**

> This is a KV cache: a language model's memory of everything it has read. RoleKV lets a coding agent switch
> roles without throwing that memory away.

**0:10 — Problem**

> Agents often split work into roles. A Coder writes, a Reviewer checks. A cheap way to build that is one small
> model with a LoRA adapter per role. But an adapter changes how the model reads every token. So at every
> handoff, the next role re-reads the whole session: the task, the code, the test output. On the left, that's
> the coral re-read, turn after turn.

**0:35 — Idea**

> RoleKV trains each role as an Activated LoRA. It only switches on after its own role header, so everything
> before that is read by the plain base model, and both roles share one cache. At a handoff we re-read just the
> previous role's short answer with base weights. That's the violet slice. It keeps the cache exactly what each
> role was trained on. It's exact, not an approximation.

**1:00 — Live**

> Here it is live, on a laptop RTX 4090, with our trained 1.5-billion-parameter adapters. Same MBPP task, both
> methods side by side. The Coder writes a function, the test fails, the Reviewer asks for a fix, the Coder
> fixes it, and the Reviewer approves. Per-role LoRAs recompute everything each turn. RoleKV computes only what's
> new: `[[ratio on the pill]]` times fewer tokens on this task.

**1:30 — Results**

> Across 100 MBPP tasks, RoleKV computed 3.2 times fewer tokens per task. The median role switch took
> 45 milliseconds instead of 139. It solved 61 tasks; per-role LoRAs solved 59. And the saving grows with
> context: at 8,000 tokens it's 3.9 times fewer, and a switch stays near 50 milliseconds while per-role LoRAs
> climb to almost 600.

**1:55 — Agent harness**

> The same cache sits behind an OpenAI-compatible server, so a real agent harness runs on it. This is DeepSeek
> Harness. On its real traffic, 97 percent of prompt tokens come from the cache, and each step starts in about
> a tenth of a second instead of three.

**2:10 — Limits and close**

> Honest limits: one 1.5B model, function-level tasks, one run per method. Simpler baselines scored about the
> same here, and our Reviewer tends to approve on a second review. What RoleKV adds is exact sharing while
> every role keeps its own adapter. RoleKV: switch roles, keep the cache.

## Wording rules (from the v1 freeze in STATUS.md)

- Say "solved 61 versus 59" or "similar quality". Do not say "no drop", "the same quality" or "pass@1": the
  protocol is multi-turn with all-assert feedback, and equivalence is not established.
- Do not say RoleKV is the cheapest method. A1 (one shared LoRA) and A3 (reusing the mismatched cache) passed
  61 and 62 with slightly fewer tokens. The claim is exact sharing with per-role adapters.
- In the dsh segment, do not say the agent completes tasks. The 1.5B gets the first tool call right and then
  loops; stop it after step 2.

## Shot list

| # | Time | Screen | Action |
|---|---|---|---|
| 1 | 0:00 | Homepage `demo/index.html` | Start on the hero. Drag the 3D scene slightly so it moves. |
| 2 | 0:10 | Homepage HUD | Press Replay. Keep the cursor near the left (A2) slab and its tape as it re-reads each turn. |
| 3 | 0:35 | Homepage HUD | Move to the right (RoleKV) slab. Pause on a Reviewer turn and point at the violet handoff segment. |
| 4 | 1:00 | Live page on the 4090 | Context 30. Task #454. Press Run both. Let all 4 turns play: REVISE on turn 2, APPROVE on turn 4. |
| 5 | 1:20 | Live page | Hold on the ratio pill and the totals line ("All turns played · A2 vs RoleKV tokens computed"). |
| 6 | 1:30 | Dashboard, Cost vs quality | Hover the A2 point, then the A5-H1 point. |
| 7 | 1:45 | Dashboard, Context sweep | Hover the 8k point. |
| 8 | 1:55 | Mac: dsh web left, stage panel right | Run "Create is_prime.py …" once with "shared KV" and once with "full prefill every request"; stop each after step 2. The panel shows both side by side. |
| 9 | 2:10 | Homepage | Hero title, then the repo URL. |

## Before recording

- [ ] The 4090's UI fixes are pushed to `peter`, and the 4090 runs that code: `server.py` with `adapters/1.5b`
      on CUDA, port 8001, opened in a normal Windows browser with hardware graphics.
- [ ] Do a full dry run of task 454 first. Use only a take where the turns are coder → REVISE → coder → APPROVE.
      In the eval, task 454 computed 8,922 tokens with A2 vs 2,361 with RoleKV (3.8×); the live pill should
      be close.
- [ ] The smoke-adapter banner must not be visible in any shot.
- [ ] Mac dsh shot: the RoleKV server (`serve/dsh/server.sh`) and dsh web (`serve/dsh/web.sh`) are running,
      and the panel is cleared with "New round". Use a clean `DSH_HOME` so the sidebar has no old runs. Never
      click the DeepSeek models in dsh's model menu.
- [ ] 1920×1080, browser zoom 100%, notifications off, OS reduced-motion off (so the 3D animates).
- [ ] Record the voice separately and cut the screen to it; speed up waits between turns rather than cutting
      inside a turn.
- [ ] Upload unlisted to YouTube and put the URL in `DEVPOST_DRAFT.md` and on Devpost.
