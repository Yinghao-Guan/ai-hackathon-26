# RoleKV behind DeepSeek Harness: feasibility spike

Question: can DeepSeek Harness (dsh, github.com/deepseek-ai/deepseek-harness, `@deepseek-ai/dsh` 0.2.0-rc.2
from npm) run on our model through a RoleKV-backed OpenAI-compatible server, and how much prefill does
server-side KV reuse save on dsh's real traffic?

Setup: the Mac (Apple M4, 24 GB, mps, bf16), base Qwen/Qwen2.5-Coder-1.5B-Instruct with no adapters, greedy
decoding, 2026-10-18. The raw logs are under `tmp/` (gitignored). The numbers below are copied from them.

## Verdict: partly

- **The plumbing works.** dsh runs headless against `serve/openai_server.py` as a custom `openai-completions`
  provider. It needs only a profile patch: no dsh code changes and no new Python dependencies.
  - In the greedy runs the 1.5B made 133 tool calls. All 133 named a real tool and had every required argument.
  - dsh executed the 132 it received (the last arrived as dsh was stopped): 126 completed and 6 returned tool
    errors. All 6 were `edit` precondition errors: old and new string identical, or editing before reading.
- **Only because the server parses leniently.** None of the 133 replies used Qwen's own
  `<tool_call>...</tool_call>` tags. Every one was a JSON call in a ```` ```json ```` fence, or bare JSON.
  With `--strict` (tags only), the base 1.5B gives dsh no tool calls at all.
- **No task succeeded.** The first action is usually right:
  - is_prime.py was correct in every run that wrote it.
  - `python3 --version` ran and returned `Python 3.13.0`.
  - The `add` bug was fixed in 2 of 3 attempts.

  But the model does not finish a task:
  - With the history rendered as it generated it, the model repeats its last call until the 180-300 s timeout.
  - With the history re-rendered canonically, it stops after about 3 calls without giving the answer.
  - It never ran the check it was asked for.
  - Both times it fixed `add` it then rewrote calc.py and broke `mul`.
  - With dsh's default 24 tools it called `create_goal`, then printed the code as text.
  - Sampling at temperature 0.7 did not help: one run answered in prose with no tool call; another read
    `/usr/bin/python3` 108 times.
- **Server-side KV reuse pays off on dsh's traffic, and the saving grows with the session.**
  - Same 33 prompts with and without reuse: 4,899 vs 163,398 computed prompt tokens (97% reused).
  - Total TTFT 6.8 s vs 176.6 s. At the 8.2k-token last step, 0.16 s vs 9.7 s.
  - Live, with the same 180 s budget on the same task, dsh got through 32 steps with reuse and 20 without.
- **Role adapters work on the server side; nothing upstream is ready.**
  - `rolekv-coder` / `rolekv-reviewer` model ids serve aLoRAs that share the base-model prefix. On dsh's real
    prompts, with a role switch on every request, 94.8% of prompt tokens are reused.
  - Our adapters are trained only on our 2-role MBPP format, not on dsh's.
  - dsh has no built-in Coder/Reviewer alternation, so the role switch needs a subagent or a server-side rule
    (see "RoleKV role adapters inside dsh").

The stop rule (no valid tool call within an hour) never applied: the first dsh-accepted call came on the first
real request, about 15 minutes into the spike.

## What was built

- **`serve/openai_server.py`**: stdlib HTTP server on 127.0.0.1, backed by `rolekv.cache.KVSession`. It serves
  `POST /v1/chat/completions` (SSE streaming, which dsh requires, and plain JSON) and `GET /v1/models`.
  - **Prompt rendering:** the tokenizer's chat template with the request's tools,
    `apply_chat_template(messages, tools=..., add_generation_prompt=True)`.
  - **Tool-call parsing:** replies become OpenAI `tool_calls`. Parsed forms are Qwen's `<tool_call>{json}</tool_call>`
    blocks and, in lenient mode, a reply that is only a JSON call naming one of the request's tools, bare or in a
    ```` ```json ```` fence. Lenient is the default; `--strict` turns it off.
  - **KV policy by model id.**
    - `rolekv-reuse` keeps up to 4 sessions and continues the one with the longest common token prefix: crop to
      the prefix, then extend.
    - `rolekv-noreuse` prefills the whole prompt on every request.
    - A side request that shares under half of a session forks it instead of cropping it. dsh's session-title
      request is one.
  - **History as generated.** Earlier tool calls are rendered as the text the model generated for them, looked up
    by tool-call id, so their KV stays reusable. `--canonical-history` re-renders them with the template instead.
  - **Role adapters.** With `--adapter-dir`, the server also serves `rolekv-coder`, `rolekv-reviewer` and their
    `-noreuse` twins.
  - **Usage reporting.** `usage.prompt_tokens_details.cached_tokens` carries the reused count. dsh's client (pi-ai)
    reads it as `cacheReadTokens`, so dsh's own `--json` step events show the saving.
  - **Logging.** One JSONL line per request (`--log`) with:
    - prompt and head (system + tools) tokens;
    - reused and computed tokens;
    - TTFT, decode tokens and decode time, wall time;
    - how the reply parsed, and per call whether the tool exists and the required arguments are present.

    `--dump-dir` keeps every request and response body.
  - **Sampling.** `--temperature` turns on sampling; greedy is the default.
- **`serve/replay.py`**: replays dumped dsh requests against any model id (or alternating ids) and summarizes
  logs. A replay gives both policies the exact same prompts.
- **`serve/dsh/`**:
  - `cordis.patch.yml`: the dsh profile patch (provider, default model, telemetry off, permissions).
  - `minimal.patch.yml`: an overlay that keeps 7 tools.
  - `server.sh`: (re)starts the server for one run.
  - `run.sh`: runs one headless task in a throwaway `tmp/ws/<name>/`, with a timeout.
- **`tests/test_openai_server.py`** (29 tests; the whole suite passes):
  - parsing, streaming hold-back, message conversion and history-as-generated rendering;
  - prefix reuse checked against a full prefill (fp32, CPU, 0.5B dev model with random adapters), plus forking
    and the session bound;
  - role switches that share only base-model KV, matching a full aLoRA forward;
  - the HTTP API.

## Results

### What dsh sends

| | dsh default (24 tools) | `minimal.patch.yml` (7 tools) |
|---|---|---|
| tools | bash, create_goal, edit, exit_plan_mode, get_goal, glob, grep, interrupt_agent, job_kill, job_list, job_output, list_agents, read, read_image, send_message, skill, subagent, subagent_fork, todo_write, update_goal, web_fetch, web_search, workflow, write | bash, edit, glob, grep, read, read_image, write |
| system prompt | 555 tokens | 217 tokens |
| tool schemas (Qwen renders them into the system message) | 4,774 tokens (workflow 866, bash 537, edit 272, ...) | 1,635 tokens |
| first request (system + tools + task + dsh's runtime-context message) | 5,487 tokens | 2,000-2,015 tokens |
| first-request TTFT on the M4 | 8.3 s | 1.9 s |

- **Requests per step.** One request per agent step, plus one session-title request at the start (about 140
  tokens, no tools, `max_tokens` 64).
- **Fields on every request.** `stream: true`, `stream_options.include_usage`, `store: false`, `max_tokens`
  (with our compat switch) and the tools. There is never a `tool_choice` or `temperature`.
- **Growth per step.** Each step appends the assistant call and the tool result: 135-250 tokens per step with
  the minimal set.
- **Repeat reminders.** dsh's repeat-tool reminder (thresholds 3, 5 and 8 in the default config) adds about
  200 tokens when the model keeps repeating a call.
- **Session length.** The longest greedy session reached 9,141 tokens after 45 steps (r4), far inside the
  32k window.

### Runs (base 1.5B, greedy, minimal tools unless noted)

Tasks:
- **T1:** "Create is_prime.py with a function is_prime(n) that returns True for prime numbers, then run a quick
  check with python3."
- **T2:** "Use bash to run python3 --version and tell me which Python version is installed."
- **T3:** "calc.py has a bug: add(2, 3) returns -1 instead of 5. Fix it, then verify with python3.", with a
  seeded `calc.py` whose `add` returns `a - b`.

| run | task | model id | history | dsh steps | tool calls (valid) | dsh tool errors | outcome | reused share | computed prompt tokens | TTFT after step 1 (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | T1, default 24 tools | rolekv-reuse | canonical | 2 | 1 (1) | 0 | fail: `create_goal`, then code as text | 49% | 5,793 | 315 ms |
| r2 | T1 | rolekv-reuse | canonical | 4 | 3 (3) | 0 | partial: correct file, check not run | 70% | 2,860 | 282 ms |
| r3 | T1 | rolekv-reuse | as generated | 10 | 10 (10) | 0 | partial: correct file, `write` repeated, stopped by hand at 76 s | 90% | 3,304 | 186 ms |
| r6 | T1 | rolekv-noreuse | as generated | 16 | 16 (16) | 0 | partial: correct file, `write` repeated, 180 s timeout | 0% | 64,399 | 4,141 ms |
| r4 | T2 | rolekv-reuse | as generated | 45 | 45 (45) | 0 | partial: right command and output, repeated 45x, no answer, 300 s timeout | 99% | 3,683 | 97 ms |
| r9 | T2 | rolekv-reuse | canonical | 4 | 3 (3) | 0 | fail: ran it 3x, then "let's try a different approach" | 71% | 2,585 | 187 ms |
| r5 | T3 | rolekv-reuse | as generated | 32 | 32 (32) | 2 | fail: fixed `add`, broke `mul`, looped edit/write/read, 180 s timeout | 97% | 4,899 | 148 ms |
| r7 | T3 | rolekv-noreuse | as generated | 20 | 20 (20) | 2 | fail: fixed `add`, broke `mul`, 180 s timeout | 0% | 78,012 | 4,054 ms |
| r8 | T3 | rolekv-reuse | canonical | 5 | 3 (3) | 2 | fail: read the file, then empty replies (dsh `EMPTY_RESPONSE`) | 77% | 2,678 | 173 ms |

"Valid" means the tool exists and every required argument is present. All 133 calls came back fenced or bare,
never in `<tool_call>` tags. Sampling runs: r10 (T3, T=0.7) answered in prose with no tool call, and r11
(T2, T=0.7) read `/usr/bin/python3` 108 times.

How the history is rendered changes both reuse and behaviour:
- **Canonical.** dsh sends the calls back as OpenAI `tool_calls`. The template then renders them as `<tool_call>`
  blocks, which are not the tokens the model generated. Each step recomputes the previous reply (about 150-220
  tokens), and reuse lands at 70-77%.
- **As generated.** Rendered the way the model wrote it, a step computes only the tool result and the next
  header: 28-130 tokens, or about 280 when dsh adds a reminder. Reuse lands at 90-99%.

Neither mode makes the 1.5B finish: as-generated history loops, and canonical history stops early.

### Reuse on identical traffic (replay)

r5's 33 requests (T3, as-generated history), replayed with the same prompts under `rolekv-noreuse`. The
replay used `--max-tokens 1`, which is exact for TTFT because that policy keeps no cache. All 33 prompt
lengths matched the live run.

| | reuse (live r5) | no reuse (replay) |
|---|---|---|
| prompt tokens | 163,398 | 163,398 |
| computed prompt tokens | **4,899** | **163,398** |
| total TTFT, 33 requests | **6.8 s** | **176.6 s** |
| request 1, the first step (2,013 tokens) | 1.90 s | 1.95 s |
| request 10 (3,585 tokens) | 0.13 s | 3.49 s |
| request 20 (5,611 tokens) | 0.18 s | 6.08 s |
| request 33, the last step (8,223 tokens) | 0.16 s | 9.71 s |

**Live, same 180 s budget (r5 vs r7).** With reuse dsh completed 32 steps and without it 20.
- Reuse: TTFT took 6.5 s of the 178 s; decode took the rest (3,622 tokens at about 21 tokens/s with HF's
  per-token loop on mps).
- No reuse: prefill took 78.9 s (44%) of the budget.

The share grows with context. dsh's default 24-tool profile starts at 5.5k tokens, so it gains more per step
than the minimal profile measured here.

**Money.**
- **Locally:** the saving is GPU time, about 170 s of prefill avoided on one 33-step session.
- **Hosted API:** the server reports reused tokens as `cached_tokens`, which dsh and OpenAI-style billing treat
  as cache reads. The saving is reused tokens × (uncached − cached input price): 158,499 of 163,398 input tokens
  on this session.
- I did not attach vendor prices, because they vary.

### Role adapters (throwaway, untrained for tool use)

- **Adapters.** Two 0.5B aLoRAs: `train/train_role_adapter.py --kind alora --model Qwen/Qwen2.5-Coder-0.5B-Instruct
  --max-steps 2 --grad-accum 1 --device mps`, 16 s each, written to `tmp/adapters-0.5b-throwaway/` and not
  committed. The real 1.5B adapters are on the 4090 and were not used.
- **Role-switch replay.** r5's 32 main prompts, alternating `rolekv-coder` and `rolekv-reviewer` on every request,
  `--max-tokens 1`:
  - with reuse: 154,936 of 163,492 prompt tokens reused (94.8%), total TTFT 4.4 s;
  - with the `-noreuse` ids (per-role full prefill): 0 reused, total TTFT 60.1 s.
- **Live dsh run** with model `rolekv-coder` on T2: dsh received 14 `bash` calls, and all 14 failed dsh's
  argument check (`timeoutMs: 0`). That is expected for an adapter never trained on dsh's tools.

## How to run it

```sh
# 1. dsh, locally (no global install)
mkdir -p tmp/dsh-install && (cd tmp/dsh-install && npm init -y && npm install @deepseek-ai/dsh@0.2.0-rc.2)

# 2. One server per run (loads the 1.5B on mps; log, dumps and raw_calls.jsonl go to tmp/serve_logs/<name>/),
#    then one headless task in tmp/ws/<name>/; --minimal keeps 7 tools
serve/dsh/server.sh demo-reuse
serve/dsh/run.sh demo-reuse rolekv-reuse --minimal "Create is_prime.py with a function is_prime(n), then run a quick check with python3."
serve/dsh/server.sh demo-noreuse
serve/dsh/run.sh demo-noreuse rolekv-noreuse --minimal "Create is_prime.py with a function is_prime(n), then run a quick check with python3."
uv run --no-sync python -m serve.replay summary tmp/serve_logs/demo-reuse/requests.jsonl tmp/serve_logs/demo-noreuse/requests.jsonl

# 3. The same prompts under the other policy: copy the run's raw_calls.jsonl so earlier calls render the same way
mkdir -p tmp/serve_logs/demo-replay && cp tmp/serve_logs/demo-reuse/raw_calls.jsonl tmp/serve_logs/demo-replay/
serve/dsh/server.sh demo-replay
uv run --no-sync python -m serve.replay send tmp/serve_logs/demo-reuse/dumps --model rolekv-noreuse --max-tokens 1

# 4. Role adapters: alora_coder/ and alora_reviewer/ in a directory, any base model they were trained on
serve/dsh/server.sh roles --model Qwen/Qwen2.5-Coder-0.5B-Instruct --adapter-dir tmp/adapters-0.5b-throwaway
uv run --no-sync python -m serve.replay send tmp/serve_logs/demo-reuse/dumps --main-only --model rolekv-coder,rolekv-reviewer --max-tokens 1
```

What `run.sh` sets up:
- **Home:** `DSH_HOME=tmp/dsh-home` (dsh's default is `~/.dsh`); `serve/dsh/cordis.patch.yml` is copied into the
  headless profile.
- **Telemetry:** off (`DSH_TELEMETRY_MODE=DISABLED`, `DSH_TELEMETRY_DISABLED=1`).
- **Timeout:** dsh is stopped after `DSH_TIMEOUT` seconds (default 180), because headless has no step limit.
- **Sandbox:** dsh's `workspace-write` Seatbelt sandbox stays on, so bash and file writes are confined to the
  workspace.
- **Approval:** the policy is `never` through a custom `workspace-write-auto` preset. In-workspace actions need no
  approval and escalations fail. dsh refuses to start its permission plugin when the composed defaults match no
  preset.

Other settings in the patch:
- `compat` makes the system prompt go out as `system` (not `developer`) and the output cap as `max_tokens`.
- AGENTS.md / CLAUDE.md injection is off (`agent-instructions.maxBytes: 0`).
- The compaction headroom is 2048. dsh's default of 65,536 exceeds a 32k window and silently disables compaction.
  No run here reached the window.

## RoleKV role adapters inside dsh

### What the server already does

`--adapter-dir DIR` loads `alora_coder/` and `alora_reviewer/`. For a role id, the server:
1. Renders the conversation without the generation prompt.
2. Appends that role's header, `<|im_start|>assistant\n<|role:coder|>\n`, which is also the aLoRA invocation
   sequence.
3. Runs the aLoRA from the header on.

Every earlier position keeps the base model's KV, so a request for the other role reuses the whole shared
prefix, as `rolekv.runtime`'s A5 does.

Reuse stops at the previous turn's header, even though the next request repeats its `<|im_start|>assistant`
tokens: the adapter computed those, so their KV is not base KV. dsh sends that turn back as a plain assistant
message without the role tag, so the turn is recomputed with base weights. That is A5-H1's canonical KV,
obtained without an explicit crop. The tests check both the reuse point and the logits against a full aLoRA
forward.

### What dsh needs

dsh runs one model per agent, so the role switch has to come from somewhere:

1. **A reviewer subagent (configuration only).**
   - `@deepseek-ai/dsh-tool-subagent` takes `agentOptions: {provider, model}`, and its `fork` provider hands the
     child the parent's conversation. A second instance, for example `toolName: review` with `provider: fork` and
     `agentOptions: {provider: rolekv, model: rolekv-reviewer}`, sends the Reviewer a request whose prefix is the
     Coder's session.
   - Not tried. The fork's system prompt and tool list must match the parent's for the prefix to match, because
     Qwen renders tools inside the system message.
   - The Coder model would also have to decide to call `review`.
2. **Role picked by the server.**
   - One model id (say `rolekv-auto`) whose server rule picks the role per request. For example: Reviewer right
     after a `bash` result that ran tests, Coder otherwise.
   - No dsh changes, and the demo controls the switch. The cost is that dsh does not know roles exist.
3. **A dsh plugin that switches the model per step.** The most control and the most work.

### What the adapters need

Our adapters learned our 2-role MBPP format: one `python` block, or APPROVE / REVISE, after a 7-line system
prompt. Inside dsh the same weights face a 1.5k-5k-token dsh system prompt and tool schemas, must answer with
tool calls, and must stop. They need dsh-format data:

- **Data.**
  - Run dsh on the MBPP tasks with a teacher: the DeepSeek API through dsh's own provider, or a bigger local Qwen
    behind this server.
  - Keep this server's request/response dumps, or dsh's session logs under `$DSH_HOME/sessions/`.
  - Each assistant step becomes one SFT row: context = the rendered request; target = the reply in Qwen's
    `<tool_call>` format plus `<|im_end|>`.
  - Coder rows are steps that write, edit or run code. Reviewer rows are the step after a test run, which accepts
    or asks for a fix.
  - Keep only sessions whose hidden tests pass, as for the current data.
- **Training.**
  - `train/train_role_adapter.py` already takes `segments` plus a target. A dsh row is one segment (the rendered
    request) plus the role header, so only a converter is missing.
  - `--max-length` must cover the prompt: about 2-8k tokens with the minimal tools, 5.5k+ with the default 24.
  - One run per role. Base the 1.5B runs on the 4090.
- **What the data must teach**, from these runs:
  - Qwen's tag format: the base model writes fenced JSON.
  - Stopping after success: the base model repeats its last call.
  - Running the check it was asked for.
  - Editing only what is broken: it rewrote calc.py and broke `mul`.

## Recommendation for the hackathon demo

1. **Show the saving with dsh's real traffic, not with dsh finishing tasks.**
   - Run the same dsh and the same prompt against `rolekv-reuse` and `rolekv-noreuse` side by side, each with the
     same time budget.
   - Next to each run, show the server's per-request line: prompt, reused and computed tokens, TTFT.
   - dsh's own `--json` step events show `cacheReadTokens` too.
   - Back it with the replay table above: identical prompts, 97% fewer computed tokens, and a TTFT gap growing to
     60x at 8k tokens.
2. **Show the role switch with the replay.** Use `--model rolekv-coder,rolekv-reviewer` on recorded dsh traffic:
   94.8% reuse with a role switch on every request, vs per-role full prefill. Say plainly that the adapters behind
   it are placeholders.
3. **Do not promise task completion with the base 1.5B in dsh.** If the demo needs a finished task, there are
   three options:
   - Pick a one-step task (the is_prime file write works every time), and stop dsh after it.
   - Train the role adapters on dsh traces (previous section).
   - Put a stronger Qwen behind the same server: it is model-agnostic, and a 7B fits on the 24 GB Mac in bf16,
     slowly.
4. **Measure the headline numbers on the 4090 when it is free.** These are M4/mps numbers, and decode there is
   slow, about 21 tokens/s.

## Decisions taken unattended

- **Branch.** The permission classifier refused `git reset --hard origin/peter`, so I created branch `dsh-spike`
  from `origin/peter` instead. The worktree's original branch (`claude/upbeat-antonelli-53306f`, at main) is
  unchanged. Only `dsh-spike` is pushed.
- **Lenient tool-call parsing by default** (`--strict` to turn it off), with the parse mode logged, so validity
  can be reported both ways.
- **History as generated by default:** better reuse. Canonical history is one flag away; it does not fix the
  model either.
- **Greedy decoding by default**, for repeatability. I added `--temperature` only to test whether sampling breaks
  the loops; it does not.
- **Reuse is capped at the adapter boundary.** Without the cap, a few tokens of Coder-adapter KV would carry into
  a Reviewer request.
- **dsh settings:**
  - telemetry off;
  - sandbox kept on, approval `never` via a custom preset;
  - AGENTS.md injection off;
  - compaction headroom lowered;
  - dsh's title request kept, because it is real traffic and it exercises the fork path;
  - 180-300 s per run, killed by a watchdog.
- **Minimal toolset** by disabling plugin rows, because the headless profile has no preset roster: goals, plan
  mode, jobs, subagents, skills, todos, web, workflows.
- **Install and placement.** dsh is installed with npm into `tmp/dsh-install`, with no global install; DSH_HOME,
  workspaces, logs, dumps and the throwaway adapters all live under `tmp/` (gitignored). I did not use the 4090
  or any remote machine.
