import { cn } from "@/lib/utils";

interface GroundingDotProps {
  x: number;
  y: number;
  confidence: number;
  imgBounds: { left: number; top: number; width: number; height: number };
  /**
   * If true, render the dot in a "tentative" style -- slightly lighter
   * and faster pulse -- to convey that this is the streaming coarse
   * prediction and the refined version is still loading.
   */
  tentative?: boolean;
}

export function GroundingDot({ x, y, confidence, imgBounds, tentative }: GroundingDotProps) {
  const isLow = confidence < 0.8;

  const pixelLeft = imgBounds.left + x * imgBounds.width;
  const pixelTop = imgBounds.top + y * imgBounds.height;
  const showBelow = y < 0.15;

  return (
    <div
      className={cn(
        "absolute w-[18px] h-[18px] rounded-full -translate-x-1/2 -translate-y-1/2 pointer-events-none z-10 transition-all duration-200",
        tentative ? "opacity-70" : "opacity-100",
      )}
      style={{ left: `${pixelLeft}px`, top: `${pixelTop}px` }}
    >
      <span
        className={cn(
          "absolute inset-0 rounded-full",
          tentative
            ? "animate-[pulse_0.9s_ease-in-out_infinite]"
            : "animate-[pulse_1.8s_ease-in-out_infinite]",
          isLow
            ? "bg-accent-orange shadow-[0_0_0_3px_rgba(230,138,25,0.3)]"
            : "bg-accent shadow-[0_0_0_3px_rgba(230,59,25,0.3)]",
        )}
      />
      <span className="absolute inset-[4px] rounded-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.2)]" />
      <span
        className="absolute left-1/2 -translate-x-1/2 bg-text-primary text-bg font-mono text-[0.66rem] font-medium px-2.5 py-1 rounded-md whitespace-nowrap shadow-md"
        style={{
          bottom: showBelow ? "auto" : "calc(100% + 8px)",
          top: showBelow ? "calc(100% + 8px)" : "auto",
        }}
      >
        ({x}, {y}){tentative ? " · coarse" : ""}
        <span
          className="absolute left-1/2 -translate-x-1/2 border-4 border-transparent"
          style={
            showBelow
              ? { bottom: "100%", borderBottomColor: "var(--color-text-primary)" }
              : { top: "100%", borderTopColor: "var(--color-text-primary)" }
          }
        />
      </span>
    </div>
  );
}
