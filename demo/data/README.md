# Real event logs

Put the real eval log here as `events.jsonl` (CONTRACT.md §3 schema). The dashboard loads it automatically and
switches from mock to real mode. Multiple runs can share one file:

```bash
cat eval/logs/*.jsonl > demo/data/events.jsonl
```

Never put `mock-` run_ids in a real log; the dashboard drops them when real runs are present.
