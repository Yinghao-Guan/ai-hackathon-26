# Real event logs

`events.jsonl` and `sources.json` are generated; don't edit them by hand. Rebuild them after a new eval or
cost sweep:

```bash
python demo/collect_logs.py           # newest 4090 eval + the 4090 cost sweep
python demo/collect_logs.py --check   # exit 1 if these files are stale or invalid
```

`sources.json` records which result files went in (path, lines, sha256, last commit, run_ids). The dashboard
shows it under Data validation and warns if `events.jsonl` no longer matches it. Options and rules are in
[../README.md](../README.md#refresh-the-real-log-after-a-new-eval-or-sweep).
