# Demo video script: RoleKV (about 2:20)

> For the Devpost video, recorded on the Mac (M4). It uses the current UI: the dark 3D homepage, the dark live
> demo with the trained 1.5B adapters, the dashboard, and DeepSeek Harness. The live judging pitch stays in
> [PITCH.md](PITCH.md). Every number below comes from `README.md` › Results, `bench/results/cost_sweep_4090.md`
> or `serve/README.md`. `[[Double brackets]]` = read it off the screen during the take; never say a number
> that is not on screen or in those files.

## Read-aloud script

Read the plain text. Bracketed lines are cues, not narration.

---

[Homepage hero. Drag the 3D scene a little.]

This is a KV cache: a language model's memory of everything it has read. RoleKV lets a coding agent switch
roles without throwing that memory away.

[Press Replay. Stay on the left, coral block.]

Agents often split work into roles. A Coder writes, a Reviewer checks. A cheap way to build that is one small
model with a LoRA adapter per role. But an adapter changes how the model reads every token. So at every
handoff, the next role re-reads the whole session: the task, the code, the test output. On the left, that's the
coral re-read, turn after turn.

[Move to the right, cyan block. Point at the violet slice on a Reviewer turn.]

RoleKV trains each role as an Activated LoRA. It only switches on after its own role header, so everything
before that is read by the plain base model, and both roles share one cache. At a handoff we re-read just the
previous role's short answer with base weights. That's the violet slice. It keeps the cache exactly what each
role was trained on. It's exact, not an approximation.

[Live page. Task 454, context 30. Press Run both.]

Here it is live on this laptop, with our trained 1.5-billion-parameter adapters. Same coding task, both
methods side by side. The Coder writes a function, the test fails, the Reviewer asks for a fix, the Coder
fixes it, and the Reviewer approves. Per-role LoRAs recompute everything each turn. RoleKV computes only what's
new: [[ratio on the pill]] times fewer tokens on this task.

[Dashboard, Cost vs quality, then Context sweep.]

We measured it on 100 coding tasks on an RTX 4090. RoleKV computed 3.2 times fewer tokens per task. The median
role switch took 45 milliseconds instead of 139. It solved 61 tasks; per-role LoRAs solved 59. And the saving
grows with context: at 8,000 tokens it's 3.9 times fewer, and a switch stays near 50 milliseconds while
per-role LoRAs climb to almost 600.

[dsh web on the left, the stage panel on the right.]

The same cache sits behind an OpenAI-compatible server, so a real agent harness runs on it. This is DeepSeek
Harness. On its real traffic, 97 percent of prompt tokens come from the cache, and each step starts in about a
tenth of a second instead of three.

[Homepage hero, then the repo URL.]

Honest limits: one 1.5B model, function-level tasks, one run per method. Simpler baselines scored about the
same here, and our Reviewer tends to approve on a second review. What RoleKV adds is exact sharing while every
role keeps its own adapter. RoleKV: switch roles, keep the cache.

---

## Timed outline

| Time | Beat | Screen |
|---|---|---|
| 0:00–0:10 | Hook | Homepage, 3D scene |
| 0:10–0:35 | Problem: every role switch re-reads the session | Homepage replay, left block (A2) |
| 0:35–1:00 | Idea: Activated LoRA, one shared cache, exact handoff | Homepage replay, right block (RoleKV) |
| 1:00–1:30 | Live: task 454 with the trained 1.5B adapters | Live page |
| 1:30–1:55 | Results over 100 tasks (4090) and the context sweep | Dashboard: Cost vs quality, Context sweep |
| 1:55–2:10 | A real agent harness on the same cache | dsh web + stage panel side by side |
| 2:10–2:20 | Honest limits and close | Homepage hero |

## Wording rules (from the v1 freeze in STATUS.md)

- Say "solved 61 versus 59" or "similar quality". Do not say "no drop", "the same quality" or "pass@1": the
  protocol is multi-turn with all-assert feedback, and equivalence is not established.
- Do not say RoleKV is the cheapest method. A1 (one shared LoRA) and A3 (reusing the mismatched cache) passed
  61 and 62 with slightly fewer tokens. The claim is exact sharing with per-role adapters.
- The 100-task numbers and the sweep are 4090 results; say so. The live run is on the Mac, so do not read its
  TTFT as a 4090 number.
- In the dsh segment, do not say the agent completes tasks. The 1.5B gets the first tool call right and then
  loops; stop it after step 2.

## Shot list

| # | Screen | Action |
|---|---|---|
| 1 | Homepage `/` → `index.html` | Start on the hero. Drag the 3D scene slightly. |
| 2 | Homepage HUD | Press Replay. Keep the cursor near the left (A2) block and its tape as it re-reads each turn. |
| 3 | Homepage HUD | Move to the right (RoleKV) block. Pause on a Reviewer turn and point at the violet handoff slice. |
| 4 | Live page `live/` | Context 30, task #454, Run both. Let all 4 turns play: REVISE on turn 2, APPROVE on turn 4. Speed up the waits in editing. |
| 5 | Live page | Hold on the ratio pill and the totals line ("All turns played · … vs … tokens computed"). |
| 6 | Dashboard, Cost vs quality | Hover the A2 point, then the A5-H1 point. |
| 7 | Dashboard, Context sweep | Hover the 8k point. |
| 8 | dsh web left, stage panel right | Run "Create is_prime.py …" with "shared KV", stop after step 2; New session, "full prefill every request", same task, stop after step 2. |
| 9 | Homepage | Hero title, then the repo URL. |

## Before recording

- [ ] The trained adapters are on the Mac (`alora_coder`, `alora_reviewer`, `lora_coder`, `lora_reviewer` from
      the 4090's `adapters/1.5b`), and the live server runs them: no "Smoke adapters" banner anywhere.
- [ ] Dry-run task 454 once. Use only a take where the turns are coder → REVISE → coder → APPROVE. In the 4090
      eval, task 454 computed 8,922 tokens with A2 vs 2,361 with RoleKV (3.8×); the pill should be close.
- [ ] dsh: the RoleKV server and dsh web are running with a clean `DSH_HOME`, the panel is cleared with
      "New round", and the "rolekv" preset is selected. Never click the DeepSeek models in dsh's model menu.
- [ ] 1920×1080 (or the full Retina screen), browser zoom 100%, Do Not Disturb on, Reduce Motion off.
- [ ] Record the voice separately and cut the screen to it. Speed up waits between turns rather than cutting
      inside a turn.
- [ ] Upload unlisted to YouTube and put the URL in `DEVPOST_DRAFT.md` and on Devpost.
