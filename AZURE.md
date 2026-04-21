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

## Performance numbers (A100, MAX_PIXELS=800000, COARSE_MAX_PIXELS=600000, max_new_tokens=16)

| Mode | Forward passes | Wall time (warm) | Cost / request | Confidence |
|---|---|---|---|---|
| `fast` | 2 (CCF coarse + refined) | **~1.5-2s** | ~$0.0013 | 0.99 if CCF parse succeeds |
| `accurate` | 7 (CCF + 3 sampled CCF + native + 0.75x) | **~7-9s** | ~$0.005 | Real agreement-based |

Got there from a baseline of 6s / 31s with three changes:
1. `max_new_tokens 32 -> 16` (bbox is ~12 tokens; capped output cuts ~50% generation time)
2. `COARSE_MAX_PIXELS 1.5M -> 600k` env var (coarse pass dominates; less pixels = less vision encode work)
3. `MAX_PIXELS 12M -> 800k` env var (refined crop already small; this caps the worst case)

To push further (sub-1s), the next step is **vLLM with paged attention + continuous batching**. ~2-3x speedup on Qwen2.5-VL but a meaningful re-engineer.

## vLLM attempt (deferred, April 2026)

We tried swapping the HF transformers backend for vLLM (versions 0.7.3, 0.8.5) targeting sub-1s fast mode and ~2s accurate mode via batching the 7-pass self-consistency loop into 3 batched calls. Build + deploy worked; **vLLM/transformers compat for Qwen2.5-VL bit us hard**:

| Attempt | Stack | Failure |
|---|---|---|
| v6 | `vllm==0.7.3` + auto-bundled transformers (4.49+) | `Qwen2Tokenizer has no attribute all_special_tokens_extended` |
| v7 | `vllm==0.7.3 + transformers==4.48.3` | `KeyError: 'qwen2_5_vl'` -- model_type only added in 4.49+ |
| v8 | `vllm==0.8.5` (auto transformers) | Same `all_special_tokens_extended` error in newer engine |
| v9 | `vllm==0.8.5` + monkeypatch for the missing attr | `'Qwen2VLImageProcessor' object has no attribute 'min_pixels'` -- yet another version mismatch in the multimodal init path |

After 4 build/deploy iterations and ~15 minutes per cycle, we rolled back to v5 (HF transformers, 1.5-2s warm). The HF backend is "good enough" for the demo; vLLM would need a known-good pinned combination of `vllm`, `transformers`, AND `qwen-vl-utils`, which moves with each vLLM release. Worth revisiting once vLLM ships a stable Qwen2.5-VL recipe (their docs reference one but it's currently underspecified for version pins).

**Things to try next time** (in priority order):

1. Use vLLM's official Docker image (`vllm/vllm-openai:vX.Y.Z`) instead of building our own -- they pin the entire stack.
2. Try the brand-new vLLM (>=0.9.x) which has reportedly fixed the multimodal processor compat issues.
3. Accept the cold-start hit and pre-warm CUDA graphs for our specific image shapes (the variable image sizes mean vLLM can't fully reuse graphs anyway).
4. Skip vLLM entirely; integrate `torch.compile()` on the HF model for ~30% speedup with much less compat risk.

The committed code (the v5 HF backend) is what's actually running. The vLLM-rewrite branch was discarded after rollback to keep the deployable surface single-track.
