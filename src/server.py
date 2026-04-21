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
import asyncio
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
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from inference import GroundingEngine

engine: Optional[GroundingEngine] = None

MODEL_PATH = os.environ.get("MODEL_PATH", "inclusionAI/GUI-G2-3B")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "12845056"))
COARSE_MAX_PIXELS = int(os.environ.get("COARSE_MAX_PIXELS", "1500000"))
WAITLIST_PATH = os.environ.get("WAITLIST_PATH", "data/waitlist.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    adapter = ADAPTER_PATH if ADAPTER_PATH else None
    engine = GroundingEngine(
        model_path=MODEL_PATH,
        adapter_path=adapter,
        max_pixels=MAX_PIXELS,
        coarse_max_pixels=COARSE_MAX_PIXELS,
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


def _decode_request_image(req: "GroundRequest") -> Image.Image:
    try:
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data)
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.post("/v1/ground", response_model=GroundResponse)
async def ground(req: GroundRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="Instruction cannot be empty")

    image = _decode_request_image(req)
    mode = req.mode or "fast"
    # Run blocking inference off the asyncio thread so the event loop
    # stays responsive (Container Apps multiplexes other healthchecks
    # on the same loop).
    result = await asyncio.to_thread(
        engine.predict, image, req.instruction.strip(), mode,
    )

    return GroundResponse(
        x=result.x,
        y=result.y,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        mode=result.mode,
        n_passes=result.n_passes,
        agreement_px=result.agreement_px,
    )


@app.post("/v1/ground/stream")
async def ground_stream(req: GroundRequest):
    """Server-Sent Events endpoint: emits the coarse CCF prediction as
    soon as it's ready (~300-500ms warm), then the refined prediction
    when the second pass completes (~600-800ms warm). The playground
    uses this to render a tentative dot at coarse latency and snap to
    the refined dot at full latency, so the UX feels sub-1s even when
    actual server work is closer to 1s.

    Always emits two events: `coarse` then `refined`. On parse failure
    in the coarse pass we still emit a `refined` event with confidence=0
    so the client knows to stop waiting."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="Instruction cannot be empty")

    image = _decode_request_image(req)
    instruction = req.instruction.strip()

    async def generator():
        # Stage 1: coarse pass
        coarse = await asyncio.to_thread(
            engine.predict_coarse_only, image, instruction,
        )
        if coarse is not None:
            payload = {
                "stage": "coarse",
                "x": coarse.x,
                "y": coarse.y,
                "latency_ms": coarse.coarse_latency_ms,
            }
            yield f"event: coarse\ndata: {json.dumps(payload)}\n\n"

        # Stage 2: refined pass (always emitted)
        refined = await asyncio.to_thread(
            engine.predict_refined_from_coarse, image, instruction, coarse,
        )
        payload = {
            "stage": "refined",
            "x": refined.x,
            "y": refined.y,
            "confidence": refined.confidence,
            "latency_ms": refined.latency_ms,
            "mode": refined.mode,
            "n_passes": refined.n_passes,
            "agreement_px": refined.agreement_px,
        }
        yield f"event: refined\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            # Tell intermediaries (Azure proxy, browser, CDN) not to
            # buffer or compress; SSE depends on flushing per chunk.
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't shadow API routes (POSTs to /v1/* are handled above; GETs to
        # /v1/* would have nothing to render anyway).
        if full_path.startswith(("v1/", "health", "docs", "openapi.json")):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = playground_dist / full_path
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
