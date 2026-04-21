export type GroundingMode = "fast" | "accurate";

export interface GroundingResult {
  x: number;
  y: number;
  confidence: number;
  latency_ms: number;
  mode: GroundingMode;
  n_passes: number;
  agreement_px: number;
}

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
