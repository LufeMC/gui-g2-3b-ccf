import { useRef, useState, useEffect, useCallback } from "react";
import { ImageIcon } from "lucide-react";
import { GroundingDot } from "./GroundingDot";
import type { GroundingResult } from "@/lib/api";

interface AnnotatedCanvasProps {
  image: string | null;
  result: GroundingResult | null;
  loading: boolean;
}

export function AnnotatedCanvas({ image, result, loading }: AnnotatedCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [imgBounds, setImgBounds] = useState<{
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);

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
    updateBounds();
    window.addEventListener("resize", updateBounds);
    return () => window.removeEventListener("resize", updateBounds);
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
            />
          )}
        </>
      )}

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-bg-card/80 backdrop-blur-sm rounded-[var(--radius-lg)]">
          <div className="flex items-center gap-2 font-mono text-[0.78rem] text-text-secondary">
            <div className="w-[15px] h-[15px] border-2 border-border border-t-accent rounded-full animate-[spin_0.7s_linear_infinite]" />
            Running inference&hellip;
          </div>
        </div>
      )}
    </div>
  );
}
