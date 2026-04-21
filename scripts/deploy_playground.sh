#!/usr/bin/env bash
# One-command deploy of the GUI grounding API + playground SPA to a
# RunPod GPU pod. Assumes:
#   - The pod already has /workspace persisted from prior runs (model
#     at /dev/shm/models/gui-g2-3b OR /workspace/models/gui-g2-3b).
#   - SSH key at $KEY (default ~/.ssh/id_ed25519).
#   - The pod's HTTP port (default 8000) is exposed via the RunPod
#     proxy URL https://<RUNPOD_POD_ID>-<port>.proxy.runpod.net.
#
# Usage:
#   POD_HOST=root@1.2.3.4 POD_PORT=22001 bash scripts/deploy_playground.sh
#   POD_HOST=root@1.2.3.4 POD_PORT=22001 KEY=~/.ssh/id_ed25519 \
#     INTERNAL_PORT=8000 bash scripts/deploy_playground.sh
#
set -euo pipefail

POD_HOST="${POD_HOST:?set POD_HOST=root@<host>}"
POD_PORT="${POD_PORT:?set POD_PORT=<ssh-port>}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
INTERNAL_PORT="${INTERNAL_PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-/dev/shm/models/gui-g2-3b}"
MAX_PIXELS="${MAX_PIXELS:-12845056}"

REMOTE_DIR="/workspace/serve"
SSH="ssh -i $KEY -p $POD_PORT $POD_HOST"
SCP="scp -i $KEY -P $POD_PORT"
RSYNC="rsync -avz --delete -e \"ssh -i $KEY -p $POD_PORT\""

echo ">>> [1/6] Checking pod reachability"
if ! eval $SSH 'echo alive' >/dev/null 2>&1; then
  echo "ERROR: cannot SSH to $POD_HOST:$POD_PORT with $KEY" >&2
  exit 1
fi

echo ">>> [2/6] Checking dist build is fresh"
DIST="$(cd "$(dirname "$0")/../playground" && pwd)/dist"
if [ ! -f "$DIST/index.html" ]; then
  echo "ERROR: $DIST/index.html missing. Run: cd playground && npm ci && npm run build" >&2
  exit 1
fi

echo ">>> [3/6] Verifying model on pod"
eval $SSH "test -d $MODEL_PATH" || {
  echo "ERROR: model not found at $MODEL_PATH on pod. Either:" >&2
  echo "   - pass MODEL_PATH=/workspace/models/gui-g2-3b" >&2
  echo "   - or copy first: cp -r /workspace/models/gui-g2-3b /dev/shm/models/" >&2
  exit 1
}

echo ">>> [4/6] Syncing code + dist to $REMOTE_DIR"
eval $SSH "mkdir -p $REMOTE_DIR/src $REMOTE_DIR/playground/dist"

# Server-side files: just the modules the server imports
eval $SCP "$(cd "$(dirname "$0")/.." && pwd)/src/inference.py $POD_HOST:$REMOTE_DIR/src/"
eval $SCP "$(cd "$(dirname "$0")/.." && pwd)/src/server.py $POD_HOST:$REMOTE_DIR/src/"
eval $SCP "$(cd "$(dirname "$0")/.." && pwd)/src/cursor_ccf.py $POD_HOST:$REMOTE_DIR/src/"
eval $SCP "$(cd "$(dirname "$0")/.." && pwd)/requirements-server.txt $POD_HOST:$REMOTE_DIR/"

eval $RSYNC "$DIST/" "$POD_HOST:$REMOTE_DIR/playground/dist/"

echo ">>> [5/6] Installing server deps + launching uvicorn"
eval $SSH "set -e
  cd $REMOTE_DIR
  pip install -q -r requirements-server.txt 2>&1 | tail -3
  pkill -f 'uvicorn src.server' 2>/dev/null || true
  pkill -f 'src/server.py' 2>/dev/null || true
  sleep 1
  cd $REMOTE_DIR/src
  MODEL_PATH=$MODEL_PATH MAX_PIXELS=$MAX_PIXELS \
    nohup python3 -m uvicorn server:app --host 0.0.0.0 --port $INTERNAL_PORT \
    > /workspace/server.log 2>&1 &
  echo \$! > /workspace/server.pid
  sleep 2
  echo 'Server PID:' \$(cat /workspace/server.pid)
"

echo ">>> [6/6] Polling /health (model load takes ~60s)"
HEALTH_HOST="$(echo $POD_HOST | sed 's/^[^@]*@//')"
HEALTH_URL_DIRECT="http://${HEALTH_HOST}:${INTERNAL_PORT}/health"
PROXY_HINT="https://<RUNPOD_POD_ID>-${INTERNAL_PORT}.proxy.runpod.net"

for i in $(seq 1 90); do
  if eval $SSH "curl -sf http://localhost:${INTERNAL_PORT}/health" >/tmp/health.json 2>/dev/null; then
    echo
    echo "===================================================================="
    echo "DEPLOY OK"
    echo
    cat /tmp/health.json
    echo
    echo
    echo "Public URL pattern:  $PROXY_HINT"
    echo "Direct URL (if reachable):  $HEALTH_URL_DIRECT"
    echo
    echo "To test:"
    echo "  curl \$URL/health"
    echo "  open \$URL/    (the playground SPA)"
    echo
    echo "Logs:    ssh -i $KEY -p $POD_PORT $POD_HOST tail -f /workspace/server.log"
    echo "Stop:    ssh -i $KEY -p $POD_PORT $POD_HOST kill \\\$(cat /workspace/server.pid)"
    echo "===================================================================="
    exit 0
  fi
  if [ $((i % 10)) -eq 0 ]; then
    echo "  still loading... ($((i*2))s elapsed)"
  fi
  sleep 2
done

echo "ERROR: /health didn't respond in 180s. Tail logs:" >&2
eval $SSH "tail -40 /workspace/server.log"
exit 1
