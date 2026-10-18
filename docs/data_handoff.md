# SFT data handoff (Member B → Member A)

Member A training should consume the committed files under `data/` as specified in [CONTRACT.md](CONTRACT.md) §2.

## Files

| File | Purpose |
|------|---------|
| `data/sft_coder.jsonl` | Coder turns: `segments` prefix + `target` body (loss on body tokens only) |
| `data/sft_reviewer.jsonl` | Reviewer turns: same schema |
| `data/trajectories.jsonl` | Full multi-turn sessions (debugging / inspection; training uses the SFT splits) |
| `data/README.md` | Verified trajectory count and SHA-256 hashes |

Current snapshot (regenerate `data/README.md` after re-running `build_dataset.py`):

- **280** verified trajectories
- **338** coder rows, **280** reviewer rows (extra coder rows come from repair trajectories with two coder turns)

## Row schema

Each line is one JSON object:

```json
{"task_id": 11, "role": "coder", "segments": ["...", "...", "<|im_start|>assistant\n<|role:coder|>\n"], "target": "...<|im_end|>\n"}
```

- Tokenize **each** string in `segments` separately, then the `target` string separately; concatenate token ids (see CONTRACT §1).
- Import role header token ids from `rolekv/roles.py` when that module is available on the Lead branch; do not re-tokenize a joined prompt string.

## Environment

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Training stack (Member A): install torch, transformers, peft==0.21.2, etc. separately
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
```

`PYTHONPATH` must include the repository root so imports like `rolekv.tasks` and `train.train_role_adapter` resolve.

## Regenerating data (Member B)

Teacher generation requires `ANTHROPIC_API_KEY` in `.env` (see `.env.example`). Example:

```bash
PYTHONPATH=. python train/build_dataset.py --limit 20   # spend check
PYTHONPATH=. python train/build_dataset.py --limit 500  # full pass (adjust as needed)
```

Outputs overwrite `data/trajectories.jsonl`, `data/sft_*.jsonl`, and `data/README.md`.

## Quick validation

```bash
pip install pytest
ROLEKV_SKIP_MBPP=1 PYTHONPATH=. pytest tests/test_member_b.py -q
```

Remove `ROLEKV_SKIP_MBPP=1` once `datasets` is installed to run the MBPP leakage check.
