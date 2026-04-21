import { useCallback, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { X, ArrowRight } from "lucide-react";
import { useGrounding } from "@/hooks/useGrounding";
import { ImageUpload } from "./ImageUpload";
import { ExamplePicker } from "./ExamplePicker";
import { AnnotatedCanvas } from "./AnnotatedCanvas";
import { ResultCard } from "./ResultCard";

interface PlaygroundDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onOpenWaitlist: () => void;
}

export function PlaygroundDialog({ open, onOpenChange, onOpenWaitlist }: PlaygroundDialogProps) {
  const g = useGrounding();
  const [showCta, setShowCta] = useState(false);

  if (g.result && !showCta) setShowCta(true);

  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!next) {
        g.reset();
        setShowCta(false);
      }
      onOpenChange(next);
    },
    [g, onOpenChange],
  );

  return (
    <Dialog.Root open={open} onOpenChange={handleOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-[rgba(26,25,23,0.35)] backdrop-blur-[6px] data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=open]:fade-in data-[state=closed]:fade-out duration-200" />
        <Dialog.Content
          className="fixed inset-0 z-50 flex items-center justify-center p-6 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=open]:fade-in data-[state=open]:zoom-in-[0.98] data-[state=open]:slide-in-from-bottom-3 data-[state=closed]:fade-out data-[state=closed]:zoom-out-[0.98] data-[state=closed]:slide-out-to-bottom-3 duration-300"
          aria-describedby={undefined}
          onClick={(e) => { if (e.target === e.currentTarget) handleOpenChange(false); }}
        >
          <div className="relative bg-bg border border-border rounded-[20px] shadow-[0_12px_48px_rgba(26,25,23,0.12),0_24px_64px_rgba(26,25,23,0.06)] w-[min(1140px,calc(100vw-48px))] md:h-[min(720px,calc(100vh-48px))] max-h-[calc(100vh-48px)] overflow-hidden max-md:overflow-y-auto p-6 md:p-8 flex flex-col" onClick={(e) => e.stopPropagation()}>
            <div className="mb-5 shrink-0">
              <div className="flex items-center justify-between gap-3 mb-1">
                <div className="font-mono text-[0.7rem] font-semibold tracking-[0.1em] uppercase text-accent">
                  Playground
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => onOpenWaitlist()}
                    className={`inline-flex items-center gap-1.5 py-1.5 px-3 md:py-2 md:px-4 rounded-[var(--radius-base)] font-mono text-[0.72rem] md:text-[0.78rem] font-semibold bg-accent text-white cursor-pointer transition-all hover:bg-accent-hover hover:-translate-y-px hover:shadow-md ${showCta ? "opacity-100" : "opacity-0 pointer-events-none"}`}
                  >
                    Get API Access
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                  <Dialog.Close className="w-8 h-8 md:w-9 md:h-9 rounded-[var(--radius-base)] border border-border bg-bg-card flex items-center justify-center cursor-pointer hover:bg-bg-elevated hover:border-border-strong shadow-sm transition-all">
                    <X className="w-4 h-4 text-text-secondary" />
                  </Dialog.Close>
                </div>
              </div>
              <div className="font-mono text-[1.15rem] md:text-[1.35rem] font-bold tracking-tight mb-0.5">
                Try GUI Grounding
              </div>
              <p className="text-text-secondary text-sm">
                Upload a screenshot, describe the element, and watch the model find it.
              </p>
            </div>

            <div className="grid md:grid-cols-[1fr_1.15fr] grid-cols-1 gap-5 md:flex-1 md:min-h-0">
              {/* Left column */}
              <div className="flex flex-col gap-3.5 md:min-h-0">
                <div className="bg-bg-card border border-border rounded-[var(--radius-lg)] p-[18px] shadow-sm flex flex-col md:flex-1 md:min-h-0">
                  <div className="md:flex-1 md:min-h-0 min-h-[180px]">
                    <ImageUpload image={g.image} onImageSelected={g.setImage} />
                  </div>
                  <div className="flex flex-col sm:flex-row gap-2.5 mt-3.5 shrink-0">
                    <input
                      type="text"
                      value={g.instruction}
                      onChange={(e) => g.setInstruction(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && g.canGround && g.runGrounding()}
                      placeholder='e.g. "click the search bar"'
                      className="flex-1 py-[11px] px-3.5 rounded-[var(--radius-base)] border border-border font-mono text-[0.82rem] bg-white text-text-primary outline-none transition-colors placeholder:text-text-tertiary focus:border-accent"
                    />
                    <button
                      onClick={g.runGrounding}
                      disabled={!g.canGround || g.loading}
                      className="sm:shrink-0 inline-flex items-center justify-center gap-2 py-[11px] px-6 rounded-[var(--radius-base)] font-mono text-sm font-semibold bg-accent text-white cursor-pointer transition-all hover:bg-accent-hover hover:-translate-y-px hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none"
                    >
                      Ground
                    </button>
                  </div>
                  <div className="flex items-center gap-2 mt-2.5 shrink-0">
                    <span className="font-mono text-[0.65rem] font-semibold tracking-wider uppercase text-text-tertiary">
                      Mode
                    </span>
                    <div className="inline-flex rounded-[var(--radius-base)] border border-border overflow-hidden bg-bg-elevated">
                      <button
                        onClick={() => g.setMode("fast")}
                        className={`py-1 px-3 font-mono text-[0.7rem] font-semibold transition-colors cursor-pointer ${
                          g.mode === "fast"
                            ? "bg-accent text-white"
                            : "bg-transparent text-text-secondary hover:bg-bg-card"
                        }`}
                        title="CCF refinement, single sample. ~1-2s on H100."
                      >
                        Fast
                      </button>
                      <button
                        onClick={() => g.setMode("accurate")}
                        className={`py-1 px-3 font-mono text-[0.7rem] font-semibold transition-colors cursor-pointer ${
                          g.mode === "accurate"
                            ? "bg-accent text-white"
                            : "bg-transparent text-text-secondary hover:bg-bg-card"
                        }`}
                        title="CCF + 4 perturbed samples + multi-resolution agreement. ~4-6s. Higher accuracy, real confidence."
                      >
                        Accurate
                      </button>
                    </div>
                    <span className="font-mono text-[0.65rem] text-text-tertiary">
                      {g.mode === "fast"
                        ? "~1s, single CCF pass"
                        : "~5s, multi-pass self-consistency"}
                    </span>
                  </div>
                </div>

                <div className="shrink-0">
                  <ExamplePicker activeExample={g.activeExample} onSelect={g.loadExample} />
                </div>
              </div>

              {/* Right column */}
              <div className="flex flex-col gap-3 md:min-h-0">
                <div className="md:flex-1 md:min-h-0 min-h-[240px]">
                  <AnnotatedCanvas image={g.image} result={g.result} loading={g.loading} />
                </div>
                <div className="shrink-0">
                  <ResultCard result={g.result} />
                </div>
              </div>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
