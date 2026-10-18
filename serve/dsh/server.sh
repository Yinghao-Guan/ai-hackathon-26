#!/bin/sh
# (Re)start the RoleKV server in the background, logging to tmp/serve_logs/<name>/.
#   [ROLEKV_PORT=8765] serve/dsh/server.sh <name> [server args...]
# Only a server on the same port is stopped first.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NAME=$1
shift
PORT="${ROLEKV_PORT:-8765}"
L="$ROOT/tmp/serve_logs/$NAME"
mkdir -p "$L"
pkill -f "serve.openai_server --port $PORT " 2>/dev/null || true
sleep 1
cd "$ROOT"
nohup uv run --no-sync python -m serve.openai_server --port "$PORT" --log "$L/requests.jsonl" --dump-dir "$L/dumps" --raw-store "$L/raw_calls.jsonl" "$@" > "$L/server.out" 2>&1 &
i=0
until curl -s "127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  i=$((i + 1)); [ $i -gt 120 ] && { echo "server did not start"; tail "$L/server.out"; exit 1; }
  sleep 1
done
echo "server up on http://127.0.0.1:$PORT/v1, logging to $L"
