# GUI-G2-3B + CCF inference server, packaged for Azure Container Apps
# serverless GPU (T4 / Consumption-GPU-NC8as-T4 workload profile).
#
# The model is NOT baked into the image; it's downloaded from
# Hugging Face on first start into a writable cache directory. This
# keeps the image small (~3GB instead of ~9GB) and makes redeploys
# fast. Cold-start trade-off: first request after the container scales
# from zero waits ~60-90s for model download + load.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    MODEL_PATH=inclusionAI/GUI-G2-3B \
    MAX_PIXELS=12845056 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && pip install \
        "torch==2.4.1" "torchvision==0.19.1" \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements-server.txt /app/requirements-server.txt
RUN pip install -r /app/requirements-server.txt

WORKDIR /app

COPY src/inference.py src/server.py src/cursor_ccf.py /app/src/
COPY playground/dist /app/playground/dist

# Make the cache dir writable (HF_HOME is set above)
RUN mkdir -p /cache/huggingface && chmod -R 777 /cache

EXPOSE 8000

# Container Apps probes /health; the server returns "loading" until
# the model finishes loading, then "ok". We use the same image entry
# point on either workload profile.
WORKDIR /app/src
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
