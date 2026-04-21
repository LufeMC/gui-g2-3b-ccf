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

## Performance numbers (A100, MAX_PIXELS=1000000)

| Mode | Forward passes | Wall time (warm) | Confidence |
|---|---|---|---|
| `fast` | 2 (CCF coarse + refined) | ~6-10s | 0.99 if CCF parse succeeds |
| `accurate` | 7 (CCF + 3 sampled CCF + native + 0.75x) | ~30-35s | Real agreement-based |
