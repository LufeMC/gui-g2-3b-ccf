import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { joinWaitlist } from "@/lib/api";
import { toast } from "sonner";

interface WaitlistDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function WaitlistDialog({ open, onOpenChange }: WaitlistDialogProps) {
  const [email, setEmail] = useState("");
  const [company, setCompany] = useState("");
  const [useCase, setUseCase] = useState("");
  const [submitting, setSubmitting] = useState(false);

  function handleOpenChange(next: boolean) {
    if (!next) {
      setEmail("");
      setCompany("");
      setUseCase("");
    }
    onOpenChange(next);
  }

  async function handleSubmit() {
    if (!email.trim() || !email.includes("@")) {
      toast.error("Please enter a valid email");
      return;
    }

    setSubmitting(true);
    try {
      await joinWaitlist({ email, company, useCase });
      toast.success("You're on the list — we'll be in touch!");
      setEmail("");
      setCompany("");
      setUseCase("");
      setTimeout(() => onOpenChange(false), 800);
    } catch {
      toast.error("Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={handleOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-[rgba(26,25,23,0.35)] backdrop-blur-[6px] data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=open]:fade-in data-[state=closed]:fade-out duration-200" />
        <Dialog.Content
          className="fixed inset-0 z-50 flex items-center justify-center p-6 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=open]:fade-in data-[state=open]:zoom-in-[0.98] data-[state=open]:slide-in-from-bottom-3 data-[state=closed]:fade-out data-[state=closed]:zoom-out-[0.98] data-[state=closed]:slide-out-to-bottom-3 duration-300"
          aria-describedby={undefined}
          onClick={(e) => { if (e.target === e.currentTarget) handleOpenChange(false); }}
        >
          <div className="relative bg-bg border border-border rounded-[20px] shadow-[0_12px_48px_rgba(26,25,23,0.12),0_24px_64px_rgba(26,25,23,0.06)] w-[min(460px,calc(100vw-48px))] p-8" onClick={(e) => e.stopPropagation()}>
            <Dialog.Close className="absolute top-4 right-4 z-10 w-9 h-9 rounded-[var(--radius-base)] border border-border bg-bg-card flex items-center justify-center cursor-pointer hover:bg-bg-elevated hover:border-border-strong shadow-sm transition-all">
              <X className="w-4 h-4 text-text-secondary" />
            </Dialog.Close>

            <div className="mb-6 pr-10">
              <div className="font-mono text-[0.7rem] font-semibold tracking-[0.1em] uppercase text-accent mb-1">
                Early Access
              </div>
              <div className="font-mono text-[1.35rem] font-bold tracking-tight mb-0.5">
                Get API Access
              </div>
              <p className="text-text-secondary text-sm">
                We're onboarding design partners now. Drop your email and we'll reach out.
              </p>
            </div>

            <div className="flex flex-col gap-3.5">
              <div>
                <label className="block font-mono text-[0.7rem] font-semibold tracking-wider uppercase text-text-secondary mb-1.5">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  className="w-full py-[11px] px-3.5 rounded-[var(--radius-base)] border border-border font-sans text-[0.9rem] bg-white text-text-primary outline-none transition-colors focus:border-accent"
                />
              </div>
              <div>
                <label className="block font-mono text-[0.7rem] font-semibold tracking-wider uppercase text-text-secondary mb-1.5">
                  Company{" "}
                  <span className="text-text-tertiary font-normal normal-case tracking-normal">
                    (optional)
                  </span>
                </label>
                <input
                  type="text"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                  placeholder="Acme Inc."
                  className="w-full py-[11px] px-3.5 rounded-[var(--radius-base)] border border-border font-sans text-[0.9rem] bg-white text-text-primary outline-none transition-colors focus:border-accent"
                />
              </div>
              <div>
                <label className="block font-mono text-[0.7rem] font-semibold tracking-wider uppercase text-text-secondary mb-1.5">
                  Use case{" "}
                  <span className="text-text-tertiary font-normal normal-case tracking-normal">
                    (optional)
                  </span>
                </label>
                <textarea
                  value={useCase}
                  onChange={(e) => setUseCase(e.target.value)}
                  placeholder="Briefly describe how you'd use GUI grounding..."
                  className="w-full py-[11px] px-3.5 rounded-[var(--radius-base)] border border-border font-sans text-[0.9rem] bg-white text-text-primary outline-none transition-colors focus:border-accent resize-y min-h-[72px]"
                />
              </div>
              <button
                onClick={handleSubmit}
                disabled={submitting}
                className="w-full mt-1 inline-flex items-center justify-center gap-2 py-3 px-6 rounded-[var(--radius-base)] font-mono text-sm font-semibold bg-accent text-white cursor-pointer transition-all hover:bg-accent-hover hover:-translate-y-px hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {submitting ? "Submitting..." : "Request Access"}
              </button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
