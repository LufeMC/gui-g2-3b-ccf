import { useState, useCallback } from "react";
import {
  groundImage,
  groundImageStream,
  type GroundingMode,
  type GroundingResult,
} from "@/lib/api";
import type { Example } from "@/lib/examples";

interface GroundingState {
  image: string | null;
  instruction: string;
  result: GroundingResult | null;
  loading: boolean;
  activeExample: Example | null;
  mode: GroundingMode;
}

const INITIAL_STATE: GroundingState = {
  image: null,
  instruction: "",
  result: null,
  loading: false,
  activeExample: null,
  mode: "fast",
};

export function useGrounding() {
  const [state, setState] = useState<GroundingState>(INITIAL_STATE);

  const setImage = useCallback((image: string) => {
    setState((s) => ({
      ...s,
      image,
      result: null,
      activeExample: null,
    }));
  }, []);

  const setInstruction = useCallback((instruction: string) => {
    setState((s) => ({ ...s, instruction }));
  }, []);

  const setMode = useCallback((mode: GroundingMode) => {
    setState((s) => ({ ...s, mode }));
  }, []);

  const loadExample = useCallback((example: Example) => {
    setState((s) => ({
      ...s,
      image: example.svg,
      instruction: example.instruction,
      result: null,
      activeExample: example,
    }));
  }, []);

  const runGrounding = useCallback(async () => {
    if (!state.image || !state.instruction.trim()) return;

    setState((s) => ({ ...s, loading: true, result: null }));

    try {
      // Fast mode: stream so the UI can render the coarse dot at
      // ~400ms before the refined pass completes (~900ms). Accurate
      // mode goes through the regular sync endpoint -- there's no
      // useful intermediate stage to show for the 6-pass cluster.
      if (state.mode === "fast") {
        await groundImageStream(state.image, state.instruction, (event) => {
          if (event.stage === "coarse") {
            setState((s) => ({
              ...s,
              loading: true, // still loading; refined pass coming
              result: {
                x: event.x,
                y: event.y,
                confidence: 0.5, // tentative -- refined will overwrite
                latency_ms: event.latency_ms,
                mode: "fast",
                n_passes: 1,
                agreement_px: 0,
                stage: "coarse",
              },
            }));
          } else {
            setState((s) => ({
              ...s,
              loading: false,
              result: {
                x: event.x,
                y: event.y,
                confidence: event.confidence,
                latency_ms: event.latency_ms,
                mode: event.mode,
                n_passes: event.n_passes,
                agreement_px: event.agreement_px,
                stage: "refined",
              },
            }));
          }
        });
      } else {
        const result = await groundImage(
          state.image,
          state.instruction,
          state.mode,
        );
        setState((s) => ({ ...s, result, loading: false }));
      }
    } catch {
      setState((s) => ({ ...s, loading: false }));
    }
  }, [state.image, state.instruction, state.mode]);

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  const canGround = Boolean(state.image && state.instruction.trim());

  return {
    ...state,
    setImage,
    setInstruction,
    setMode,
    loadExample,
    runGrounding,
    canGround,
    reset,
  };
}
