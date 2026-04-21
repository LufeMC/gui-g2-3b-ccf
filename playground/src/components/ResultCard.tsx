import type { GroundingResult } from "@/lib/api";
import { cn } from "@/lib/utils";

interface ResultCardProps {
  result: GroundingResult | null;
}

export function ResultCard({ result }: ResultCardProps) {
  const isHighConf = result ? result.confidence >= 0.8 : true;

  return (
    <div
      className={cn(
        "flex flex-col sm:flex-row gap-3 bg-bg-card border border-border rounded-[var(--radius-lg)] p-[14px] sm:p-[18px] shadow-sm transition-opacity duration-200",
        result ? "opacity-100" : "opacity-0 pointer-events-none",
      )}
    >
      <pre className="flex-1 bg-text-primary text-[#E8E6E0] rounded-[var(--radius-base)] p-3 px-3.5 font-mono text-[0.72rem] sm:text-[0.78rem] leading-[1.7] overflow-x-auto whitespace-pre m-0">
        {result
          ? `{\n  "x": ${result.x},\n  "y": ${result.y},\n  "confidence": ${result.confidence},\n  "latency_ms": ${result.latency_ms},\n  "mode": "${result.mode}",\n  "n_passes": ${result.n_passes},\n  "agreement_px": ${result.agreement_px}\n}`
          : `{\n  "x": 0,\n  "y": 0,\n  "confidence": 0,\n  "latency_ms": 0\n}`}
      </pre>
      <div className="grid grid-cols-2 sm:grid-cols-1 gap-2 sm:min-w-[140px]">
        <div className="bg-bg-elevated rounded-[var(--radius-base)] p-3 px-3.5 border border-border">
          <div className="font-mono text-[0.62rem] font-semibold tracking-wider uppercase text-text-tertiary mb-0.5">
            Confidence
          </div>
          <div
            className={cn(
              "font-mono text-[1.05rem] font-bold",
              isHighConf ? "text-green" : "text-accent-orange",
            )}
          >
            {result ? `${(result.confidence * 100).toFixed(1)}%` : "\u2014"}
          </div>
          {result && result.mode === "accurate" && (
            <div className="font-mono text-[0.6rem] text-text-tertiary mt-0.5">
              agreement {result.agreement_px}px / {result.n_passes} passes
            </div>
          )}
        </div>
        <div className="bg-bg-elevated rounded-[var(--radius-base)] p-3 px-3.5 border border-border">
          <div className="font-mono text-[0.62rem] font-semibold tracking-wider uppercase text-text-tertiary mb-0.5">
            Latency
          </div>
          <div className="font-mono text-[1.05rem] font-bold text-text-primary">
            {result ? `${result.latency_ms}ms` : "\u2014"}
          </div>
          {result && (
            <div className="font-mono text-[0.6rem] text-text-tertiary mt-0.5">
              {result.mode} mode
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
