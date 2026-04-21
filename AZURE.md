# Deploying on Azure Container Apps Serverless GPU

The model + playground are deployed as a single container on Azure Container
Apps with the `Consumption-GPU-NC24-A100` workload profile (scale-to-zero).
This is the cheapest "always-available" path for a 3B VLM demo on Azure
because:

- No GPU charges while idle (scales to zero replicas).
- ~6s warm response (fast mode) on a 1.5MP screenshot.
- ~$0.004 per request after the first cold start.

For low-traffic marketing demo (10-100 visits/day): **~$2-25/mo** total.

## Architecture

```
Visitor (browser)
    --> https://<app>.eastus.azurecontainerapps.io/
        --> SPA index.html (built playground)
            --> POST /v1/ground { image, instruction, mode }
                --> uvicorn -> GroundingEngine.predict()
                    --> CCF coarse + refined (or 6-pass accurate)
                    --> JSON { x, y, confidence, latency_ms, mode, n_passes, agreement_px }
```

A100 GPU is overkill for a 3B model but it's the only Container Apps GPU
profile that hits sub-10s. T4 is ~5x slower (30-180s/request) at the same
cost while active, so A100 is actually cheaper per useful request.

## Live URL

`https://guigrounding.whiteplant-27564a0e.eastus.azurecontainerapps.io`

- `/` -- playground SPA
- `/health` -- liveness/readiness
- `/v1/ground` -- POST `{image: base64, instruction: str, mode: "fast"|"accurate"}`

## Cost levers

| Setting | Default | Cost impact |
|---|---|---|
| `min-replicas` | 0 | 0 = scale-to-zero (cheapest); 1 = always 1 warm replica (~$2.40/hr -> $1700/mo) |
| `max-replicas` | 1 | Hard cap on concurrent replicas |
| `MAX_PIXELS` env | 1000000 | Lower = faster + cheaper per request, but can hurt accuracy on tiny icons |
| Workload profile | `gpu-a100` | Switch to `gpu-t4` to halve $/active-hour, but inference becomes 3-5x slower |

## Manual operations

### Deploy a new image
```bash
cd <repo>
az acr build --registry guigroundingacr1776761055 --image guigrounding:vN --file Dockerfile .
az containerapp update --name guigrounding --resource-group guigrounding-rg \
  --image guigroundingacr1776761055.azurecr.io/guigrounding:vN
```

### Watch logs
```bash
az containerapp logs show --name guigrounding --resource-group guigrounding-rg --follow
```

### Force scale-to-zero (for cost troubleshooting)
```bash
az containerapp update --name guigrounding --resource-group guigrounding-rg \
  --max-replicas 0
# ...wait 30s...
az containerapp update --name guigrounding --resource-group guigrounding-rg \
  --max-replicas 1
```

### Pricing verification
```bash
az consumption usage list --start-date $(date -v-7d +%Y-%m-%d) --end-date $(date +%Y-%m-%d) \
  --query "[?contains(meterName, 'GPU')].{date:date, meter:meterName, qty:quantity, cost:pretaxCost}" \
  -o table
```

### Tear down everything
```bash
az group delete --name guigrounding-rg --yes
```

## Cold-start optimization (if needed)

The container downloads the 6GB model from Hugging Face on every cold start
(~30s). To skip this, mount an Azure Files share at `/cache`:

```bash
# 1. Create a storage account + file share
az storage account create -g guigrounding-rg -n guigroundingst$(date +%s) \
  --sku Standard_LRS --kind StorageV2
az storage share create -n model-cache --account-name <storage-account>

# 2. Add the storage to the env
az containerapp env storage set -g guigrounding-rg -n guigrounding-env \
  --storage-name hf-cache --account-name <storage-account> \
  --share-name model-cache --access-mode ReadWrite

# 3. Add a volume to the container app
az containerapp update -g guigrounding-rg -n guigrounding \
  --container-name guigrounding \
  --add-volume '{"name":"hf-cache-vol","storageType":"AzureFile","storageName":"hf-cache"}' \
  --add-volume-mount '{"volumeName":"hf-cache-vol","mountPath":"/cache"}'
```

After this, cold starts drop from ~90s to ~60s (model download skipped on
warm restarts; first cold start still pays the download).

## Why Sponsorship subscriptions can't use Spot VMs

Microsoft Azure Sponsorship subs explicitly disallow Spot pricing. The error
surfaces as `InvalidTemplateDeployment` with no inner detail. Container Apps
serverless GPU is the closest equivalent: pay-per-use with scale-to-zero.

## Performance numbers (A100, vLLM v0.19.1, MAX_PIXELS=800000, COARSE_MAX_PIXELS=600000, max_new_tokens=16)

| Mode | Forward passes | Server time (warm) | Wall time (warm) | Cost / request |
|---|---|---|---|---|
| `fast` (sync) | 2 (CCF coarse + refined) | **~240-360ms** | **~700-900ms** | ~$0.0003 |
| `fast` (streaming, coarse-to-first-dot) | 1 (coarse only emitted first) | ~120-200ms | **~600-800ms perceived** | ~$0.0003 |
| `accurate` | 6-7 (CCF + 3 sampled CCF + native + 0.75x, batched) | **~900ms** | **~1.6s** | ~$0.0007 |

vLLM via the official `vllm/vllm-openai:v0.19.1` image -- not pip-installed -- delivers the speedup on its own:
- LM decode: 80ms/tok (HF eager) -> ~20ms/tok (vLLM paged attention + CUDA graphs)
- 7 sequential passes (HF) -> 3 batched calls (vLLM continuous batching) for accurate mode

The streaming variant uses Server-Sent Events (`/v1/ground/stream`) to ship the coarse prediction as soon as it's ready, then the refined prediction. The playground renders a tentative dot at ~600ms and snaps to the refined dot at ~800ms. Server work is identical to sync; the perceived latency is what changes.

## How we got here (lineage)

| Tag | Backend | Fast wall | Accurate wall | Notes |
|---|---|---|---|---|
| v3 | HF (default `MAX_PIXELS=12M`) | ~6s | ~31s | Initial Container Apps deployment, naive HF transformers |
| v5 | HF + smaller `MAX_PIXELS` + `max_new_tokens=16` | ~2s | ~8s | Three quick env-var tweaks |
| v10 | **vLLM v0.19.1** (official image) | ~0.9s | ~1.6s | Sub-1s wall time achieved |
| v11 | vLLM + SSE streaming endpoint + playground UI streaming | ~0.6s perceived first dot | ~1.6s | Coarse-to-first-dot UX trick |

### Why the previous vLLM attempt (v6-v9) failed

Earlier we tried `pip install vllm==0.7.3` and `0.8.5` and hit a chain of compat errors (`Qwen2Tokenizer.all_special_tokens_extended`, `KeyError: 'qwen2_5_vl'`, `Qwen2VLImageProcessor.min_pixels`). The fix was to use the **official vLLM Docker image** instead of pip-installing -- the upstream image pins (vllm, transformers, flash-attn, xformers, qwen-vl-utils) to a tested combination. Building on top of `FROM vllm/vllm-openai:v0.19.1` skips every compat issue.

To reproduce the deployment:

```bash
# 1. Mirror upstream vLLM image into ACR (avoids Docker Hub rate limits)
az acr import --name guigroundingacr1776761055 \
  --source docker.io/vllm/vllm-openai:v0.19.1 \
  --image vllm-openai:v0.19.1

# 2. Build our app on top
az acr build --registry guigroundingacr1776761055 \
  --image guigrounding:v11 --file Dockerfile .

# 3. Deploy
az containerapp update --name guigrounding --resource-group guigrounding-rg \
  --image guigroundingacr1776761055.azurecr.io/guigrounding:v11
```

## Streaming endpoint contract

`POST /v1/ground/stream` returns `text/event-stream`. Two events emitted per request:

```
event: coarse
data: {"stage": "coarse", "x": 0.51, "y": 0.42, "latency_ms": 200}

event: refined
data: {"stage": "refined", "x": 0.512, "y": 0.420, "confidence": 0.99, "latency_ms": 360, "mode": "fast", "n_passes": 2, "agreement_px": 1.5}
```

The `coarse` event arrives ~150ms before `refined` and is meant for showing a tentative UI marker. Use the `refined` event for the canonical answer. Always emits both events even on failure (with confidence=0 if both passes failed to parse).
