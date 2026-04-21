# GUI-G2-3B + CCF inference server, vLLM backend on the official
# vLLM image (avoids the version-compat hell of pip-installing vllm).
#
# We mirror the upstream image into our ACR ahead of time to avoid
# Docker Hub rate limits during ACR builds:
#   az acr import --name <acr> \
#     --source docker.io/vllm/vllm-openai:v0.19.1 \
#     --image vllm-openai:v0.19.1
FROM guigroundingacr1776761055.azurecr.io/vllm-openai:v0.19.1

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    MODEL_PATH=inclusionAI/GUI-G2-3B \
    MAX_PIXELS=800000 \
    COARSE_MAX_PIXELS=600000 \
    PORT=8000

# Our small overlay -- vLLM image already has torch + transformers +
# flash-attn pinned, so we only need the FastAPI bits.
RUN pip install --no-cache-dir \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "python-multipart>=0.0.9" \
        "Pillow>=10.0.0"

WORKDIR /app

COPY src/inference.py src/server.py src/cursor_ccf.py /app/src/
COPY playground/dist /app/playground/dist
COPY tests/test_build_smoke.py /app/tests/test_build_smoke.py

# Build-time smoke: catches missing deps and import-time errors before
# we ship a broken image. Actual model load is GPU-only and gets
# validated post-deploy via /health.
RUN python3 /app/tests/test_build_smoke.py

# Make the cache dir writable; HF_HOME is set above
RUN mkdir -p /cache/huggingface && chmod -R 777 /cache

EXPOSE 8000

# Override the upstream vllm OpenAI server entrypoint with our FastAPI
ENTRYPOINT []
WORKDIR /app/src
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
