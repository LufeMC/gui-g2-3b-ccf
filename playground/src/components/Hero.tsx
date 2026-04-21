import { Play } from "lucide-react";

interface HeroProps {
  onOpenPlayground: () => void;
  onOpenWaitlist: () => void;
}

export function Hero({ onOpenPlayground, onOpenWaitlist }: HeroProps) {
  return (
    <section className="h-screen flex flex-col items-center justify-center text-center px-8 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_30%,rgba(230,59,25,0.04)_0%,transparent_50%),radial-gradient(circle_at_80%_70%,rgba(230,138,25,0.03)_0%,transparent_50%)] pointer-events-none" />
      <div
        className="absolute inset-0 opacity-40 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(var(--color-border) 1px, transparent 1px), linear-gradient(90deg, var(--color-border) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
          maskImage: "radial-gradient(ellipse 70% 60% at 50% 50%, black 20%, transparent 70%)",
          WebkitMaskImage:
            "radial-gradient(ellipse 70% 60% at 50% 50%, black 20%, transparent 70%)",
        }}
      />

      {/* Badge */}
      <div className="relative inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-bg-card border border-border font-mono text-xs font-medium text-text-secondary tracking-wide mb-8 shadow-sm animate-[fade-up_0.6s_ease_both]">
        <span className="w-[7px] h-[7px] rounded-full bg-green shadow-[0_0_0_3px_var(--color-green-soft)]" />
        Open Source &middot; 3B Model
      </div>

      {/* Headline */}
      <h1 className="relative font-mono text-[clamp(2.4rem,5.5vw,4.2rem)] font-bold leading-[1.1] tracking-tighter mb-5 animate-[fade-up_0.6s_ease_0.1s_both]">
        GUI Grounding
        <br />
        as a <span className="text-accent">Service</span>
      </h1>

      {/* Subheadline */}
      <p className="relative text-[clamp(1rem,2vw,1.25rem)] text-text-secondary max-w-[560px] leading-relaxed mb-10 animate-[fade-up_0.6s_ease_0.2s_both]">
        Point. Click. Done.
        <br />
        <strong className="text-text-primary font-semibold">89.2% accuracy</strong> on
        ScreenSpot-v2 with a{" "}
        <strong className="text-text-primary font-semibold">3B model</strong>.
      </p>

      {/* CTAs */}
      <div className="relative flex gap-3 animate-[fade-up_0.6s_ease_0.3s_both]">
        <button
          onClick={onOpenPlayground}
          className="inline-flex items-center gap-2 py-3 px-6 rounded-[var(--radius-base)] font-mono text-sm font-semibold bg-text-primary text-bg cursor-pointer transition-all hover:bg-[#333230] hover:-translate-y-px hover:shadow-md"
        >
          <Play className="w-3.5 h-3.5" />
          Try it
        </button>
        <button
          onClick={onOpenWaitlist}
          className="inline-flex items-center gap-2 py-3 px-6 rounded-[var(--radius-base)] font-mono text-sm font-semibold bg-bg-card text-text-primary border border-border-strong cursor-pointer transition-all hover:border-text-secondary hover:-translate-y-px hover:shadow-sm"
        >
          Join Waitlist
        </button>
      </div>

      {/* Footer */}
      <div className="absolute bottom-6 font-mono text-[0.7rem] text-text-tertiary">
        Grounding Playground
      </div>
    </section>
  );
}
