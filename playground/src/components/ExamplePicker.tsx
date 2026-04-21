import { EXAMPLES, type Example } from "@/lib/examples";
import { cn } from "@/lib/utils";

interface ExamplePickerProps {
  activeExample: Example | null;
  onSelect: (example: Example) => void;
}

export function ExamplePicker({ activeExample, onSelect }: ExamplePickerProps) {
  return (
    <div>
      <div className="font-mono text-[0.68rem] font-semibold tracking-wider uppercase text-text-tertiary mb-2">
        Quick Examples
      </div>
      <div className="grid grid-cols-2 gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.name}
            onClick={() => onSelect(ex)}
            className={cn(
              "bg-bg-elevated border border-border rounded-[var(--radius-base)]",
              "p-2.5 px-3 cursor-pointer transition-all flex gap-2.5 items-center text-left",
              "hover:border-accent hover:bg-accent-soft",
              activeExample?.name === ex.name &&
                "border-accent bg-accent-soft shadow-[0_0_0_3px_rgba(230,59,25,0.08)]",
            )}
          >
            <div
              className="w-9 h-9 rounded-md shrink-0 flex items-center justify-center text-base"
              style={{ background: `${ex.color}15`, color: ex.color }}
            >
              {ex.emoji}
            </div>
            <div className="min-w-0">
              <div className="text-[0.72rem] font-semibold text-text-primary truncate">
                {ex.name}
              </div>
              <div className="font-mono text-[0.65rem] text-text-secondary truncate">
                "{ex.instruction}"
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
