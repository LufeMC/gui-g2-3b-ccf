"""FastAPI server for GUI grounding.

Serves the grounding API and the playground frontend from a single process.

Usage:
    uvicorn src.server:app --host 0.0.0.0 --port 8000
    python src/server.py --model inclusionAI/GUI-G2-3B --port 8000
    python src/server.py --model ./models/gui-g2-3b --adapter ./checkpoints/...

Environment variables:
    MODEL_PATH    Path or HF id of the base Qwen2.5-VL model
    ADAPTER_PATH  Optional LoRA adapter path
    MAX_PIXELS    Max pixel budget for the processor
    WAITLIST_PATH Where to append /v1/waitlist submissions
"""

import argparse
import base64
import json
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from inference import GroundingEngine

engine: Optional[GroundingEngine] = None

MODEL_PATH = os.environ.get("MODEL_PATH", "inclusionAI/GUI-G2-3B")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "12845056"))
WAITLIST_PATH = os.environ.get("WAITLIST_PATH", "data/waitlist.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    adapter = ADAPTER_PATH if ADAPTER_PATH else None
    engine = GroundingEngine(
        model_path=MODEL_PATH,
        adapter_path=adapter,
        max_pixels=MAX_PIXELS,
    )
    yield
    engine = None


app = FastAPI(
    title="GUI Grounding API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GroundRequest(BaseModel):
    image: str
    instruction: str
    mode: Optional[Literal["fast", "accurate"]] = "fast"


class GroundResponse(BaseModel):
    x: float
    y: float
    confidence: float
    latency_ms: int
    mode: str
    n_passes: int
    agreement_px: float


class WaitlistRequest(BaseModel):
    email: str
    company: Optional[str] = None
    useCase: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter: Optional[str]
    modes_supported: list[str]
    version: str


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if engine is not None else "loading",
        model=MODEL_PATH,
        adapter=ADAPTER_PATH or None,
        modes_supported=["fast", "accurate"],
        version=app.version,
    )


@app.post("/v1/ground", response_model=GroundResponse)
async def ground(req: GroundRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data)
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="Instruction cannot be empty")

    mode = req.mode or "fast"
    result = engine.predict(image, req.instruction.strip(), mode=mode)

    return GroundResponse(
        x=result.x,
        y=result.y,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        mode=result.mode,
        n_passes=result.n_passes,
        agreement_px=result.agreement_px,
    )


@app.post("/v1/waitlist")
async def waitlist(req: WaitlistRequest):
    entry = {
        "email": req.email,
        "company": req.company,
        "useCase": req.useCase,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = Path(WAITLIST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if path.exists():
        with open(path) as f:
            entries = json.load(f)
    entries.append(entry)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    return {"status": "ok"}


# Serve the built playground SPA (and its assets) from the same origin.
playground_dist = Path(__file__).parent.parent / "playground" / "dist"
if playground_dist.is_dir():
    app.mount("/assets", StaticFiles(directory=playground_dist / "assets"), name="assets")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        # Don't shadow API routes
        if path.startswith(("v1/", "health", "docs", "openapi.json")):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = playground_dist / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(playground_dist / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI Grounding Server")
    parser.add_argument("--model", type=str, default="inclusionAI/GUI-G2-3B")
    parser.add_argument("--adapter", type=str, default="")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-pixels", type=int, default=12_845_056)
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model
    os.environ["ADAPTER_PATH"] = args.adapter
    os.environ["MAX_PIXELS"] = str(args.max_pixels)

    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=args.port, workers=1)
