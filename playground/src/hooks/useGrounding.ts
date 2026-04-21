import { useState, useCallback } from "react";
import { groundImage, type GroundingMode, type GroundingResult } from "@/lib/api";
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
      const result = await groundImage(
        state.image,
        state.instruction,
        state.mode,
      );
      setState((s) => ({ ...s, result, loading: false }));
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
