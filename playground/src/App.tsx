import { useEffect, useState } from "react";
import { Toaster } from "sonner";
import { Hero } from "@/components/Hero";
import { PlaygroundDialog } from "@/components/PlaygroundDialog";
import { WaitlistDialog } from "@/components/WaitlistDialog";
import { pingHealth } from "@/lib/api";

export default function App() {
  const [playgroundOpen, setPlaygroundOpen] = useState(false);
  const [waitlistOpen, setWaitlistOpen] = useState(false);

  // Warm-up ping: fire as soon as the SPA mounts so any cold-start
  // cycle begins while the visitor is still reading the hero. By the
  // time they click "Try playground" -> upload an image -> type an
  // instruction, the model is already up. Result is intentionally
  // ignored -- we're only here for the side effect of waking the
  // serverless GPU.
  useEffect(() => {
    void pingHealth();
  }, []);

  return (
    <>
      <Hero
        onOpenPlayground={() => setPlaygroundOpen(true)}
        onOpenWaitlist={() => setWaitlistOpen(true)}
      />
      <PlaygroundDialog
        open={playgroundOpen}
        onOpenChange={setPlaygroundOpen}
        onOpenWaitlist={() => setWaitlistOpen(true)}
      />
      <WaitlistDialog open={waitlistOpen} onOpenChange={setWaitlistOpen} />
      <Toaster
        position="bottom-center"
        toastOptions={{
          style: {
            fontFamily: "var(--font-mono)",
            fontSize: "0.82rem",
            fontWeight: 500,
          },
        }}
      />
    </>
  );
}
