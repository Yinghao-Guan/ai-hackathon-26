#!/bin/sh
# Start `dsh web` against the local RoleKV server, in the background, with the workspace tmp/ws/<name>/.
#   [ROLEKV_PORT=8765] [DSH_WEB_PORT=8790] serve/dsh/web.sh <name>
# Pick the "rolekv" agent preset (7 tools) in the composer; see web.patch.yml.
# Uses DSH_HOME=tmp/dsh-home (profile "web"), telemetry off; the log goes to tmp/ws/<name>.web.log.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NAME=$1
shift
PATCH="--patch $ROOT/serve/dsh/web.patch.yml"
DSH="${DSH:-$ROOT/tmp/dsh-install/node_modules/.bin/dsh}"
WEB_PORT="${DSH_WEB_PORT:-8790}"
export DSH_HOME="${DSH_HOME:-$ROOT/tmp/dsh-home}" DSH_TELEMETRY_MODE=DISABLED DSH_TELEMETRY_DISABLED=1
export ROLEKV_API_KEY=local ROLEKV_MODEL="${ROLEKV_MODEL:-rolekv-reuse}"
export ROLEKV_BASE_URL="http://127.0.0.1:${ROLEKV_PORT:-8765}/v1"
mkdir -p "$DSH_HOME/profiles/web" "$ROOT/tmp/ws/$NAME"
cp "$ROOT/serve/dsh/cordis.patch.yml" "$DSH_HOME/profiles/web/cordis.patch.yml"
pkill -f "dsh --profile web .*--port $WEB_PORT" 2>/dev/null || true
sleep 1
cd "$ROOT/tmp/ws/$NAME"
# shellcheck disable=SC2086
nohup "$DSH" --profile web $PATCH --host 127.0.0.1 --port "$WEB_PORT" --no-open > "../$NAME.web.log" 2>&1 &
i=0
until grep -q "http://" "../$NAME.web.log" 2>/dev/null; do
  i=$((i + 1)); [ $i -gt 60 ] && { echo "dsh web did not print a URL"; tail "../$NAME.web.log"; exit 1; }
  sleep 1
done
grep "http://" "../$NAME.web.log" | head -3
