import { useState } from "react";
import { Toaster } from "sonner";
import { Hero } from "@/components/Hero";
import { PlaygroundDialog } from "@/components/PlaygroundDialog";
import { WaitlistDialog } from "@/components/WaitlistDialog";

export default function App() {
  const [playgroundOpen, setPlaygroundOpen] = useState(false);
  const [waitlistOpen, setWaitlistOpen] = useState(false);

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
