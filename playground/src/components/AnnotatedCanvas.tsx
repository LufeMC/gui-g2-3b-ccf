import { useRef, useState, useEffect, useCallback } from "react";
import { ImageIcon } from "lucide-react";
import { GroundingDot } from "./GroundingDot";
import { pingHealth, type GroundingResult } from "@/lib/api";

interface AnnotatedCanvasProps {
  image: string | null;
  result: GroundingResult | null;
  loading: boolean;
}

// We show a different loading message after this long because anything
// past it almost certainly means a cold start (warm requests are
// sub-1s, even with network).
const COLD_START_THRESHOLD_MS = 2500;

export function AnnotatedCanvas({ image, result, loading }: AnnotatedCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [imgBounds, setImgBounds] = useState<{
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);
  const [coldStart, setColdStart] = useState(false);

  // If the request takes more than COLD_START_THRESHOLD_MS without
  // returning a coarse-stage result, switch the loading copy to the
  // cold-start explainer + optionally check /health to confirm the
  // backend is still booting (vs. a network failure).
  useEffect(() => {
    if (!loading) {
      setColdStart(false);
      return;
    }
    const t = window.setTimeout(() => {
      setColdStart(true);
      // Best-effort check; we don't actually use the result, but firing
      // the request now also helps the cold-start replica wake up if
      // it hadn't been pinged in a while.
      void pingHealth();
    }, COLD_START_THRESHOLD_MS);
    return () => window.clearTimeout(t);
  }, [loading]);

  const updateBounds = useCallback(() => {
    const container = containerRef.current;
    const img = imgRef.current;
    if (!container || !img || !img.naturalWidth) return;

    const cr = container.getBoundingClientRect();
    const ir = img.getBoundingClientRect();

    setImgBounds({
      left: ir.left - cr.left,
      top: ir.top - cr.top,
      width: ir.width,
      height: ir.height,
    });
  }, []);

  useEffect(() => {
    // Re-measure on every layout change of either the container or the
    // image itself. This catches:
    //   - dialog zoom-in animation finishing after onLoad fired
    //   - sibling pane resize (e.g. ResultCard expanding)
    //   - DevTools opening/closing (was relying on window resize alone)
    //   - flexbox/grid reflow after image natural size is known
    // Without this the dot can land off the image until the user does
    // *something* that triggers a window resize.
    updateBounds();
    window.addEventListener("resize", updateBounds);

    const ro = new ResizeObserver(() => updateBounds());
    if (containerRef.current) ro.observe(containerRef.current);
    if (imgRef.current) ro.observe(imgRef.current);

    // Also re-measure across the next animation frame (catches the
    // post-zoom-in steady state) and after a short delay (catches any
    // last reflow from font/asset loads).
    const raf = requestAnimationFrame(updateBounds);
    const timeout = window.setTimeout(updateBounds, 250);

    return () => {
      window.removeEventListener("resize", updateBounds);
      ro.disconnect();
      cancelAnimationFrame(raf);
      window.clearTimeout(timeout);
    };
  }, [updateBounds, image]);

  return (
    <div
      ref={containerRef}
      className="relative bg-bg-card border border-border rounded-[var(--radius-lg)] shadow-sm h-full flex items-center justify-center"
    >
      {!image ? (
        <div className="text-center p-8">
          <div className="w-[52px] h-[52px] mx-auto mb-3.5 rounded-[14px] bg-bg-elevated border border-border flex items-center justify-center">
            <ImageIcon className="w-[22px] h-[22px] text-text-tertiary" />
          </div>
          <p className="text-sm text-text-tertiary leading-relaxed">
            Upload an image or pick an
            <br />
            example to get started
          </p>
        </div>
      ) : (
        <>
          <img
            ref={imgRef}
            src={image}
            alt="Screenshot"
            className="max-w-full max-h-full object-contain rounded-[var(--radius-lg)]"
            onLoad={updateBounds}
          />
          {result && imgBounds && (
            <GroundingDot
              x={result.x}
              y={result.y}
              confidence={result.confidence}
              imgBounds={imgBounds}
              tentative={result.stage === "coarse"}
            />
          )}
        </>
      )}

      {/*
        Two loading states:
          - Full-screen spinner: no result yet (waiting for coarse pass).
          - Tiny "Refining..." pill: we have a coarse dot already, just
            waiting for the refined snap. Doesn't cover the dot so the
            user sees their answer immediately.
      */}
      {loading && !result && (
        <div className="absolute inset-0 flex items-center justify-center bg-bg-card/85 backdrop-blur-sm rounded-[var(--radius-lg)] p-6">
          <div className="flex flex-col items-center gap-3 max-w-[320px] text-center">
            <div className="w-[18px] h-[18px] border-2 border-border border-t-accent rounded-full animate-[spin_0.7s_linear_infinite]" />
            {coldStart ? (
              <>
                <div className="font-mono text-[0.82rem] font-semibold text-text-primary">
                  Spinning up GPU&hellip;
                </div>
                <div className="font-mono text-[0.7rem] leading-relaxed text-text-secondary">
                  This takes ~90 seconds the first time after idle. We scale to
                  zero between sessions to keep this demo open and free.
                </div>
                <a
                  href="https://github.com/LufeMC/gui-g2-3b-ccf"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-mono text-[0.66rem] underline text-text-tertiary hover:text-accent"
                >
                  The model is open source &mdash; run it always-on yourself
                </a>
              </>
            ) : (
              <div className="font-mono text-[0.82rem] text-text-secondary">
                Running inference&hellip;
              </div>
            )}
          </div>
        </div>
      )}
      {loading && result?.stage === "coarse" && (
        <div className="absolute top-3 right-3 px-2.5 py-1 rounded-full bg-bg-card/90 border border-border shadow-sm flex items-center gap-1.5 font-mono text-[0.65rem] text-text-secondary">
          <div className="w-[10px] h-[10px] border-2 border-border border-t-accent rounded-full animate-[spin_0.7s_linear_infinite]" />
          Refining&hellip;
        </div>
      )}
    </div>
  );
}
