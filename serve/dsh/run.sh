#!/bin/sh
# Run one dsh headless task against the local RoleKV server in a fresh throwaway workspace.
#   serve/dsh/run.sh <name> <model id> [--minimal] "task"
# Writes tmp/ws/<name>/ (the workspace), tmp/ws/<name>.events.jsonl and tmp/ws/<name>.stderr.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NAME=$1 MODEL=$2
shift 2
PATCH=""
if [ "${1:-}" = "--minimal" ]; then PATCH="--patch $ROOT/serve/dsh/minimal.patch.yml"; shift; fi
DSH="${DSH:-$ROOT/tmp/dsh-install/node_modules/.bin/dsh}"
export DSH_HOME="${DSH_HOME:-$ROOT/tmp/dsh-home}" DSH_TELEMETRY_MODE=DISABLED DSH_TELEMETRY_DISABLED=1
export ROLEKV_API_KEY=local ROLEKV_MODEL="$MODEL"
mkdir -p "$DSH_HOME/profiles/headless" "$ROOT/tmp/ws/$NAME"
cp "$ROOT/serve/dsh/cordis.patch.yml" "$DSH_HOME/profiles/headless/cordis.patch.yml"
cd "$ROOT/tmp/ws/$NAME"
start=$(date +%s)
set +e
# shellcheck disable=SC2086
"$DSH" --profile headless $PATCH --json "$@" > "../$NAME.events.jsonl" 2> "../$NAME.stderr" &
pid=$!
# Headless has no step limit and a small model can repeat a tool call forever: stop it after DSH_TIMEOUT s.
( sleep "${DSH_TIMEOUT:-180}"; kill "$pid" 2>/dev/null && echo "dsh killed after ${DSH_TIMEOUT:-180}s" ) &
watchdog=$!
wait "$pid"
code=$?
kill "$watchdog" 2>/dev/null
echo "dsh exit=$code wall_s=$(( $(date +%s) - start ))"
ls -la
