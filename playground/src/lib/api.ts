export type GroundingMode = "fast" | "accurate";
export type GroundingStage = "coarse" | "refined";

export interface GroundingResult {
  x: number;
  y: number;
  confidence: number;
  latency_ms: number;
  mode: GroundingMode;
  n_passes: number;
  agreement_px: number;
  stage?: GroundingStage; // present when result came from the streaming endpoint
}

export interface CoarseStageEvent {
  stage: "coarse";
  x: number;
  y: number;
  latency_ms: number;
}

export interface RefinedStageEvent {
  stage: "refined";
  x: number;
  y: number;
  confidence: number;
  latency_ms: number;
  mode: GroundingMode;
  n_passes: number;
  agreement_px: number;
}

export type GroundingStreamEvent = CoarseStageEvent | RefinedStageEvent;

const API_BASE = import.meta.env.VITE_API_URL ?? "";

// When VITE_API_URL is empty AND we're served from a non-localhost
// origin, we assume the FastAPI server is on the same origin and call
// it via relative paths. When served from localhost without an explicit
// API URL, we fall back to a mock response so frontend devs aren't
// blocked on the backend.
function shouldMock(): boolean {
  if (API_BASE) return false;
  if (typeof window === "undefined") return true;
  const host = window.location.hostname;
  return host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0";
}

async function postWithRetry(
  path: string,
  body: unknown,
  maxRetries = 1,
): Promise<Response> {
  let lastErr: unknown = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.status >= 500 && attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, 500));
        continue;
      }
      return res;
    } catch (err) {
      lastErr = err;
      if (attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, 500));
        continue;
      }
    }
  }
  throw lastErr ?? new Error("Network error");
}

export async function groundImage(
  imageBase64: string,
  instruction: string,
  mode: GroundingMode = "fast",
): Promise<GroundingResult> {
  if (shouldMock()) {
    return mockGrounding(mode);
  }

  const res = await postWithRetry("/v1/ground", {
    image: imageBase64,
    instruction,
    mode,
  });

  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

/**
 * Streaming variant of groundImage. Calls onEvent twice:
 *   1. With a "coarse" event as soon as the coarse CCF pass returns
 *      (~250-500ms server time). Use this to render a tentative dot.
 *   2. With a "refined" event when the second CCF pass completes
 *      (~700-900ms). Use this to snap to the final dot.
 *
 * Only used in fast mode -- accurate mode runs through the regular
 * /v1/ground endpoint.
 */
export async function groundImageStream(
  imageBase64: string,
  instruction: string,
  onEvent: (event: GroundingStreamEvent) => void,
): Promise<void> {
  if (shouldMock()) {
    return mockGroundingStream(onEvent);
  }

  const res = await fetch(`${API_BASE}/v1/ground/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageBase64, instruction, mode: "fast" }),
  });

  if (!res.ok || !res.body) throw new Error(`Stream API error: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by \n\n. Each message has lines like:
    //   event: coarse
    //   data: {"x": ...}
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const dataLine = block.split("\n").find((l) => l.startsWith("data: "));
      if (!dataLine) continue;
      try {
        const payload = JSON.parse(dataLine.slice(6)) as GroundingStreamEvent;
        onEvent(payload);
      } catch {
        // ignore malformed line
      }
    }
  }
}

async function mockGroundingStream(
  onEvent: (event: GroundingStreamEvent) => void,
): Promise<void> {
  // Simulate the perceived behavior: coarse at ~400ms, refined at ~900ms
  await new Promise((r) => setTimeout(r, 350 + Math.random() * 100));
  const cx = 0.45 + Math.random() * 0.1;
  const cy = 0.45 + Math.random() * 0.1;
  onEvent({ stage: "coarse", x: cx, y: cy, latency_ms: 350 });
  await new Promise((r) => setTimeout(r, 450 + Math.random() * 150));
  onEvent({
    stage: "refined",
    x: cx + (Math.random() - 0.5) * 0.02,
    y: cy + (Math.random() - 0.5) * 0.02,
    confidence: 0.95,
    latency_ms: 850,
    mode: "fast",
    n_passes: 2,
    agreement_px: 4.2,
  });
}

export async function joinWaitlist(data: {
  email: string;
  company?: string;
  useCase?: string;
}): Promise<void> {
  if (shouldMock()) {
    await new Promise((r) => setTimeout(r, 400));
    return;
  }

  const res = await postWithRetry("/v1/waitlist", data);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

async function mockGrounding(mode: GroundingMode): Promise<GroundingResult> {
  const baseLatency = mode === "accurate" ? 4500 : 1200;
  const latency = baseLatency + Math.random() * 400;
  await new Promise((r) => setTimeout(r, latency));
  return {
    x: Math.round((0.2 + Math.random() * 0.6) * 1000) / 1000,
    y: Math.round((0.2 + Math.random() * 0.6) * 1000) / 1000,
    confidence: Math.round((0.65 + Math.random() * 0.3) * 1000) / 1000,
    latency_ms: Math.round(latency),
    mode,
    n_passes: mode === "accurate" ? 6 : 2,
    agreement_px: mode === "accurate" ? Math.round(Math.random() * 12 * 10) / 10 : 0,
  };
}
