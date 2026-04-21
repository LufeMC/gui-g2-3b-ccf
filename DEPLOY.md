# Deploying the GUI Grounding API + Playground

The FastAPI server in `src/server.py` serves both the `/v1/ground` API and the
built playground SPA from a single origin. One command deploys the whole thing.

## Prerequisites

- A RunPod GPU pod (or any SSH-reachable Linux box with an NVIDIA GPU).
- The pod has the model under either:
  - `/dev/shm/models/gui-g2-3b` (preferred, fast RAM disk), or
  - `/workspace/models/gui-g2-3b` (slower NFS).
- Pod has Python 3.10+, CUDA 12.x, and PyTorch already installed.
- Local machine: SSH key for the pod, `playground/dist/` already built
  (`cd playground && npm ci && npm run build`).

## One-command deploy

```bash
POD_HOST=root@<host> POD_PORT=<ssh-port> bash scripts/deploy_playground.sh
```

Optional overrides:

```bash
KEY=~/.ssh/id_ed25519 \
INTERNAL_PORT=8000 \
MODEL_PATH=/workspace/models/gui-g2-3b \
MAX_PIXELS=12845056 \
POD_HOST=root@1.2.3.4 POD_PORT=22001 \
bash scripts/deploy_playground.sh
```

The script:

1. Verifies SSH reachability and that the model exists at `MODEL_PATH`.
2. Syncs `src/{inference,server,cursor_ccf}.py`, `requirements-server.txt`,
   and `playground/dist/` to `/workspace/serve/` on the pod.
3. Installs server-side deps.
4. Kills any prior `uvicorn` and launches a new one in the background.
5. Polls `/health` for up to 3 minutes; prints the public URL pattern when ready.

## Public URL

RunPod exposes container ports via:

```
https://<RUNPOD_POD_ID>-<INTERNAL_PORT>.proxy.runpod.net
```

Get `RUNPOD_POD_ID` from the RunPod console (or `echo $RUNPOD_POD_ID` inside
the pod). Open the URL in a browser to use the playground; hit
`/health` to verify the API.

## Verifying

```bash
URL=https://<RUNPOD_POD_ID>-8000.proxy.runpod.net
curl -s $URL/health | jq .
# {"status":"ok","model":"/dev/shm/models/gui-g2-3b","modes_supported":["fast","accurate"],"version":"0.2.0",...}

# Quick prediction smoke test (replace with your own base64 image):
B64=$(base64 -i screenshot.png | tr -d '\n')
curl -s -X POST $URL/v1/ground \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$B64\",\"instruction\":\"close button\",\"mode\":\"fast\"}"
```

## Local dev against a deployed backend

If you're iterating on the playground SPA locally and want it to call the
deployed API instead of the mock, create `playground/.env.local`:

```
VITE_API_URL=https://<RUNPOD_POD_ID>-8000.proxy.runpod.net
```

Then `cd playground && npm run dev`.

## Modes

The API exposes two `mode`s:

| Mode | Forward passes | Latency (H100) | Behavior |
|---|---|---|---|
| `fast` (default) | 2 (CCF coarse + refined) | ~1-2s | Single prediction. Confidence is moderate (0.75) since there's no agreement signal. |
| `accurate` | 6 logical passes (1 greedy CCF + 3 sampled CCF + 1 native + 1 0.75x) | ~4-6s | Largest-cluster centroid across all passes. Returns real `agreement_px` for confidence. |

Honest accuracy ceiling for single-shot grounding: roughly 89.2% (greedy) →
~91.4% (CCF, our `fast`) → ~95-96% (`accurate` with input perturbation +
self-consistency). Higher numbers (e.g. 99% per step) require a verifier loop
that lives above this engine, in an agent layer that observes UI state changes.

## Stopping the server

```bash
ssh -i ~/.ssh/id_ed25519 -p $POD_PORT $POD_HOST \
  "kill \$(cat /workspace/server.pid)"
```

## Troubleshooting

- **`/health` 503 or model not loaded**: tail `/workspace/server.log` on the
  pod. Most likely cause is the model path doesn't exist; pass
  `MODEL_PATH=/workspace/models/gui-g2-3b` if `/dev/shm/` was wiped.
- **OOM during `accurate` mode**: lower `MAX_PIXELS` to `8294400` (4K-ish).
- **CORS errors in browser**: the server already sets `allow_origins=*`. If
  you're behind a corporate proxy, verify the proxy isn't stripping headers.
- **Pod restarts -> backend dies**: re-run the deploy script. State that
  matters (HF model card, GitHub repo) survives independently.
