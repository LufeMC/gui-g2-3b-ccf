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
import hashlib
import ipaddress
import json
import os
import smtplib
import ssl
import threading
import time
import urllib.request
from collections import deque
from contextlib import asynccontextmanager
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
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

# SMTP for /v1/waitlist email notifications. All optional -- if SMTP_HOST
# is unset we just persist to disk and skip the email (useful for local
# dev). Credentials should come from Container Apps secrets, never from
# checked-in files.
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587") or "587")
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "")
WAITLIST_NOTIFY_TO = os.environ.get("WAITLIST_NOTIFY_TO", "")

# Notify on /v1/ground* requests too. Use the same SMTP relay as waitlist.
# Throttled to one email per IP per NOTIFY_GROUND_THROTTLE_S so a visitor
# doing 20 predictions only emails us once per session, not 20 times.
NOTIFY_GROUND = os.environ.get("NOTIFY_GROUND", "true").lower() in ("1", "true", "yes")
NOTIFY_GROUND_THROTTLE_S = int(os.environ.get("NOTIFY_GROUND_THROTTLE_S", "1800"))
GROUND_NOTIFY_TO = os.environ.get("GROUND_NOTIFY_TO", "") or WAITLIST_NOTIFY_TO

# In-memory per-IP last-notified timestamp. Resets on container restart,
# which is fine -- the worst case is a duplicate email after a redeploy.
# For a multi-replica setup this would need to move to Redis; we're at
# min/max=1 replicas so a Python dict is the right call.
_LAST_NOTIFIED_AT: dict[str, float] = {}

# Bounded ring buffer of recent /v1/ground* requests for the /v1/stats
# endpoint. Each entry: (epoch_ts, ip_hash, mode, streaming, ok).
# Cap at 50k entries (~2 weeks at 1500 req/day) to bound memory.
# IP is hashed before storage so the buffer can't be exfiltrated as a
# visitor list -- /v1/stats only ever returns counts.
_RECENT_GROUND: "deque[tuple[float, str, str, bool, bool]]" = deque(maxlen=50_000)
_STATS_LOCK = threading.Lock()
_STARTED_AT = time.time()


def _ip_hash(ip: str) -> str:
    """Stable but non-reversible IP identifier for unique-visitor counts.
    Salted with the process start so different deployments can't be
    cross-referenced even if logs leak."""
    h = hashlib.sha256(f"{_STARTED_AT}:{ip}".encode()).hexdigest()
    return h[:16]


def _record_ground(ip: str, mode: str, streaming: bool, ok: bool) -> None:
    with _STATS_LOCK:
        _RECENT_GROUND.append((time.time(), _ip_hash(ip), mode, streaming, ok))


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


@app.get("/v1/stats")
async def stats():
    """Public counters of recent /v1/ground* activity.

    Only counts are exposed -- IPs are hashed before storage and never
    leave the process. Useful as a curlable real-time pulse during a
    launch:

        curl https://<host>/v1/stats | jq
    """
    now = time.time()
    cutoff_24h = now - 86400
    cutoff_1h = now - 3600

    with _STATS_LOCK:
        all_entries = list(_RECENT_GROUND)

    def summarize(entries):
        if not entries:
            return {
                "total": 0, "unique_visitors": 0,
                "fast": 0, "accurate": 0,
                "streaming": 0, "successful": 0,
            }
        return {
            "total": len(entries),
            "unique_visitors": len({e[1] for e in entries}),
            "fast": sum(1 for e in entries if e[2] == "fast"),
            "accurate": sum(1 for e in entries if e[2] == "accurate"),
            "streaming": sum(1 for e in entries if e[3]),
            "successful": sum(1 for e in entries if e[4]),
        }

    last_24h = [e for e in all_entries if e[0] >= cutoff_24h]
    last_1h = [e for e in all_entries if e[0] >= cutoff_1h]
    last_request_at = (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(all_entries[-1][0]))
        if all_entries else None
    )

    return {
        "now_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "uptime_seconds": int(now - _STARTED_AT),
        "buffer_size": len(all_entries),
        "last_request_at": last_request_at,
        "last_1h": summarize(last_1h),
        "last_24h": summarize(last_24h),
        "all_time_in_buffer": summarize(all_entries),
    }


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
async def ground(req: GroundRequest, request: Request):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="Instruction cannot be empty")

    image = _decode_request_image(req)
    mode = req.mode or "fast"
    instruction = req.instruction.strip()
    visitor_ip = _client_ip(request)

    # Notify (off-thread, throttled per IP) so we get a real-time email
    # the first time each visitor exercises the API in a session.
    asyncio.create_task(asyncio.to_thread(
        _maybe_notify_ground,
        visitor_ip,
        instruction,
        mode,
        False,
        request.headers.get("user-agent", ""),
    ))

    # Run blocking inference off the asyncio thread so the event loop
    # stays responsive (Container Apps multiplexes other healthchecks
    # on the same loop).
    result = await asyncio.to_thread(engine.predict, image, instruction, mode)

    # Record for /v1/stats (a successful prediction returns confidence > 0).
    _record_ground(visitor_ip, mode, streaming=False, ok=result.confidence > 0)

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
async def ground_stream(req: GroundRequest, request: Request):
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
    visitor_ip = _client_ip(request)

    # Notify (off-thread, throttled per IP)
    asyncio.create_task(asyncio.to_thread(
        _maybe_notify_ground,
        visitor_ip,
        instruction,
        "fast",
        True,
        request.headers.get("user-agent", ""),
    ))

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

        # Record only after the refined pass so /v1/stats counts each
        # streaming session as a single prediction (matches the sync
        # endpoint's semantics).
        _record_ground(visitor_ip, "fast", streaming=True, ok=refined.confidence > 0)

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


def _send_email(to: str, subject: str, body: str) -> Optional[str]:
    """Send a plain-text email via SMTP. Returns None on success or an
    error string on failure. Never raises.

    Uses SMTP STARTTLS (port 587/2525) by default; if SMTP_PORT is 465
    we use implicit SSL. SMTP2GO + most providers support both."""
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_FROM and to):
        return "smtp not configured"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to
    msg.set_content(body)

    try:
        if SMTP_PORT == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx, timeout=10) as s:
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
                s.ehlo()
                # SMTP2GO and most modern relays support STARTTLS on 587/2525.
                if s.has_extn("starttls"):
                    ctx = ssl.create_default_context()
                    s.starttls(context=ctx)
                    s.ehlo()
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        return None
    except Exception as e:
        return f"smtp error: {type(e).__name__}: {e}"


def _send_waitlist_notification(entry: dict) -> Optional[str]:
    body = "\n".join([
        f"New waitlist signup at {entry.get('timestamp', '')}",
        "",
        f"Email:    {entry.get('email', '')}",
        f"Company:  {entry.get('company') or '-'}",
        f"Use case: {entry.get('useCase') or '-'}",
    ])
    return _send_email(
        to=WAITLIST_NOTIFY_TO,
        subject=f"GUI-G2-3B waitlist: {entry.get('email', 'unknown')}",
        body=body,
    )


# Reserved / non-routable / our-own ranges that can never be a real
# visitor. Catches RFC 5737 TEST-NET, RFC 6598 carrier-grade NAT (the
# Container Apps gateway sits in 100.x.x.x), and a few defensive picks
# in case Python's is_private misses something on the running version.
_TEST_NETS = [
    ipaddress.ip_network("192.0.2.0/24"),     # TEST-NET-1
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),   # TEST-NET-3
    ipaddress.ip_network("100.64.0.0/10"),    # CGN (Container Apps gateway lives here)
]


def _is_test_or_private_ip(ip: str) -> bool:
    """Return True for any IP that can't be a real visitor: malformed,
    unknown, RFC 5737 TEST-NET, private (RFC 1918), CGN (100.64/10),
    loopback, link-local, or the 1.2.3.4 / 5.6.7.8 sentinels we use
    in our own smoke tests."""
    if ip in ("", "unknown", "1.2.3.4", "5.6.7.8"):
        return True
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    if (addr.is_private or addr.is_loopback or
            addr.is_link_local or addr.is_multicast or addr.is_unspecified):
        return True
    return any(addr in net for net in _TEST_NETS)


def _lookup_geo(ip: str) -> dict:
    """Best-effort city/country/ISP lookup via ip-api.com (free tier:
    no key, ~45 req/min, HTTP only). Returns an empty dict on any
    failure (network, parse, rate limit) -- never raises."""
    try:
        url = (
            f"http://ip-api.com/json/{ip}"
            "?fields=status,country,countryCode,regionName,city,isp,org,query"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "gui-grounding-notify/1"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        if data.get("status") == "success":
            return data
    except Exception:
        pass
    return {}


def _client_ip(req: Request) -> str:
    """Best-effort real visitor IP. uvicorn is started with
    --proxy-headers --forwarded-allow-ips '*' (see Dockerfile CMD), so
    request.client.host is already the X-Forwarded-For client when we're
    behind the Container Apps gateway. We additionally check the header
    directly as a fallback."""
    fwd = req.headers.get("x-forwarded-for", "")
    if fwd:
        # X-Forwarded-For is a comma-separated list; the leftmost is the
        # original client.
        return fwd.split(",")[0].strip()
    return req.client.host if req.client else "unknown"


def _maybe_notify_ground(
    ip: str,
    instruction: str,
    mode: str,
    streaming: bool,
    user_agent: str,
) -> None:
    """Fire a 'someone tried the playground' email IF the IP is real
    AND hasn't been notified about in NOTIFY_GROUND_THROTTLE_S seconds.

    Skipped silently for our own smoke-test IPs (TEST-NET, private,
    1.2.3.4 / 5.6.7.8) -- we don't want to spam our inbox while
    iterating on the API.

    Called from the request handler off-thread so it doesn't add
    latency. Geo lookup is a separate ~100ms HTTP call to ip-api.com,
    safe in this thread context."""
    if not NOTIFY_GROUND or not GROUND_NOTIFY_TO:
        return
    if _is_test_or_private_ip(ip):
        print(f"[ground-notify] {ip}: skipped (test/private IP)")
        return

    now = time.time()
    last = _LAST_NOTIFIED_AT.get(ip, 0.0)
    if now - last < NOTIFY_GROUND_THROTTLE_S:
        return
    _LAST_NOTIFIED_AT[ip] = now

    geo = _lookup_geo(ip)
    location_parts = [p for p in (geo.get("city"), geo.get("regionName"),
                                  geo.get("country")) if p]
    location = ", ".join(location_parts)
    isp = geo.get("isp") or geo.get("org") or "unknown ISP"

    # Subject leads with location + ISP so you can triage from the
    # notification preview without opening the email.
    short_instr = instruction[:50] + ("..." if len(instruction) > 50 else "")
    if location_parts:
        subj_loc = f"{location_parts[0]} - {isp}"
    else:
        subj_loc = ip
    subj = f"GUI grounding hit ({subj_loc}): \"{short_instr}\""

    body = "\n".join([
        "Someone just used the live playground.",
        "",
        f"Instruction: {instruction!r}",
        f"Mode:        {mode}{' (streaming)' if streaming else ''}",
        "",
        f"IP:          {ip}",
        f"Location:    {location or 'unknown'}",
        f"ISP / Org:   {isp}" + (
            f" / {geo.get('org')}" if geo.get("org") and geo.get("org") != isp else ""
        ),
        f"User-Agent:  {user_agent[:200] or 'unknown'}",
        f"Time:        {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now))}",
        "",
        f"(Throttled: only one email per IP per {NOTIFY_GROUND_THROTTLE_S // 60} min, "
        f"so this visitor's subsequent calls in this session will be silent.)",
    ])
    err = _send_email(GROUND_NOTIFY_TO, subj, body)
    if err:
        print(f"[ground-notify] {ip}: skipped/failed: {err}")
    else:
        print(f"[ground-notify] {ip}: notified {GROUND_NOTIFY_TO} ({location or 'no geo'})")


@app.post("/v1/waitlist")
async def waitlist(req: WaitlistRequest):
    entry = {
        "email": req.email,
        "company": req.company,
        "useCase": req.useCase,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Persist to disk first so we never lose a signup, even if the SMTP
    # call later fails.
    path = Path(WAITLIST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if path.exists():
        try:
            with open(path) as f:
                entries = json.load(f)
        except Exception:
            entries = []  # corrupted; we'll overwrite
    entries.append(entry)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)

    # Fire the email off the request thread so the client doesn't wait
    # on SMTP. Errors are logged but don't fail the request.
    async def _notify():
        err = await asyncio.to_thread(_send_waitlist_notification, entry)
        if err:
            print(f"[waitlist] notification skipped/failed: {err}")
        else:
            print(f"[waitlist] notified {WAITLIST_NOTIFY_TO} for {entry['email']}")

    asyncio.create_task(_notify())
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
