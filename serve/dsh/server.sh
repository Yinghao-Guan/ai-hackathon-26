#!/bin/sh
# (Re)start the RoleKV server in the background, logging to tmp/serve_logs/<name>/.
#   serve/dsh/server.sh <name> [server args...]
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NAME=$1
shift
L="$ROOT/tmp/serve_logs/$NAME"
mkdir -p "$L"
pkill -f "serve.openai_server" 2>/dev/null || true
sleep 1
cd "$ROOT"
nohup uv run --no-sync python -m serve.openai_server --log "$L/requests.jsonl" --dump-dir "$L/dumps" --raw-store "$L/raw_calls.jsonl" "$@" > "$L/server.out" 2>&1 &
i=0
until curl -s 127.0.0.1:8765/health >/dev/null 2>&1; do
  i=$((i + 1)); [ $i -gt 120 ] && { echo "server did not start"; tail "$L/server.out"; exit 1; }
  sleep 1
done
echo "server up, logging to $L"
