"""Policy interface for the cursor environment.

Phase 2 introduced `CursorPolicy` + `MockCursorPolicy` for driving the
rollout with string completions only. Phase 3 extends this with a richer
policy surface that also exposes per-token log-probs -- the raw material
GRPO needs for its clipped surrogate loss.

Two policy surfaces coexist:
- `CursorPolicy`           (Phase 2): `generate(image, instruction, step) -> str`
- `LogprobCursorPolicy`    (Phase 3): adds `generate_with_logprobs(...) -> CompletionOutput`
                           and `teacher_force_logprobs(completions, ref_only=bool) -> list[Tensor]`

The real VLM implementation (Phase 3b, GPU-only) lives in
[src/cursor_vlm_policy.py](src/cursor_vlm_policy.py). The mock
`MockLogprobPolicy` below keeps everything else unit-testable on CPU.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, Tuple

import torch
from PIL import Image


class CursorPolicy(Protocol):
    """Something that can produce an action string given the current env state."""

    def generate(
        self,
        image: Image.Image,
        instruction: str,
        step_index: int,
    ) -> str:
        ...


@dataclass
class PolicyCall:
    """Record of one invocation of a policy's generate method. Used by MockCursorPolicy for assertions in tests."""
    instruction: str
    step_index: int
    image_size: Tuple[int, int]


@dataclass
class MockCursorPolicy:
    """Replays a list of scripts (one per trajectory).

    Each script is a list of strings to return on successive `generate`
    calls for a single trajectory. Call `next_trajectory()` between
    trajectories. `calls` tracks all invocations for inspection in tests.
    """
    scripts: List[List[str]]
    fallback: str = "<action>stop</action>"
    _traj: int = field(default=0, init=False)
    _step: int = field(default=0, init=False)
    calls: List[PolicyCall] = field(default_factory=list, init=False)

    def next_trajectory(self) -> None:
        """Advance to the next script. Wraps silently if exhausted."""
        self._traj += 1
        self._step = 0

    def reset(self) -> None:
        self._traj = 0
        self._step = 0
        self.calls.clear()

    def generate(
        self,
        image: Image.Image,
        instruction: str,
        step_index: int,
    ) -> str:
        self.calls.append(
            PolicyCall(
                instruction=instruction,
                step_index=step_index,
                image_size=image.size,
            )
        )
        if self._traj >= len(self.scripts):
            return self.fallback
        script = self.scripts[self._traj]
        if self._step >= len(script):
            self._step += 1
            return self.fallback
        out = script[self._step]
        self._step += 1
        return out


# ---------------------------------------------------------------------------
# Phase 3: richer policy surface with per-token log-probs
# ---------------------------------------------------------------------------


@dataclass
class CompletionOutput:
    """One completion plus the metadata the training loop needs.

    `token_ids` and `logprobs_at_generation` are aligned: one entry per
    generated token. The VLM policy fills `token_ids` from the tokenizer;
    the mock policy uses deterministic fake IDs.

    `replay_data` is an opaque per-implementation bundle used by
    `teacher_force_logprobs` to re-forward the same (prompt+completion)
    through the model. Mock policies leave it as None; VLM policies stash
    `input_ids`, `pixel_values`, `image_grid_thw`, and `prompt_length`.
    """
    text: str
    token_ids: List[int]
    logprobs_at_generation: List[float]
    replay_data: Optional[Any] = None


class LogprobCursorPolicy(Protocol):
    """Policy that supports GRPO-style training.

    Any implementation must still satisfy the Phase 2 CursorPolicy
    protocol via a `generate()` method so the existing rollout driver
    works unchanged.
    """

    def generate(self, image: Image.Image, instruction: str, step_index: int) -> str:
        ...

    def generate_with_logprobs(
        self,
        image: Image.Image,
        instruction: str,
        step_index: int,
    ) -> CompletionOutput:
        ...

    def teacher_force_logprobs(
        self,
        completions: Sequence[CompletionOutput],
        ref_only: bool = False,
    ) -> List[torch.Tensor]:
        """Recompute per-token log-probs for `completions` under the current policy.

        If `ref_only=True`, use the reference policy (e.g. LoRA disabled).
        Returns a list of 1D tensors, one per completion, same length as
        that completion's `token_ids`.
        """
        ...


class MockLogprobPolicy:
    """Deterministic mock that satisfies LogprobCursorPolicy.

    Drives the rollout with scripted completions (same as MockCursorPolicy)
    and returns fake but reproducible token IDs + log-probs. A single
    learnable scalar `offset` is injected into the "current policy"
    log-probs so the training loop has something to backprop through.

    - `generate_with_logprobs` returns fake IDs/logprobs captured at
      "generation time" (detached baseline).
    - `teacher_force_logprobs(ref_only=False)` returns `baseline + offset`
      so gradients flow back to `offset`.
    - `teacher_force_logprobs(ref_only=True)` returns the baseline
      (detached) -- the "reference policy" is just the initial mock.
    """

    def __init__(
        self,
        scripts: List[List[str]],
        fallback: str = "<action>stop</action>",
        baseline_logprob: float = -1.0,
    ):
        self.scripts = scripts
        self.fallback = fallback
        self.baseline_logprob = baseline_logprob
        # Single learnable parameter -- enables backprop without a real model.
        self.offset = torch.nn.Parameter(torch.tensor(0.0))
        self._traj = 0
        self._step = 0
        self.calls: List[PolicyCall] = []

    def parameters(self):
        """Mimic nn.Module so the training loop can pass us to AdamW."""
        return [self.offset]

    def next_trajectory(self) -> None:
        self._traj += 1
        self._step = 0

    def reset(self) -> None:
        self._traj = 0
        self._step = 0
        self.calls.clear()

    # ---- CursorPolicy surface ----

    def generate(self, image: Image.Image, instruction: str, step_index: int) -> str:
        return self.generate_with_logprobs(image, instruction, step_index).text

    def generate_with_logprobs(
        self,
        image: Image.Image,
        instruction: str,
        step_index: int,
    ) -> CompletionOutput:
        self.calls.append(
            PolicyCall(
                instruction=instruction,
                step_index=step_index,
                image_size=image.size,
            )
        )
        if self._traj >= len(self.scripts):
            text = self.fallback
        else:
            script = self.scripts[self._traj]
            if self._step >= len(script):
                text = self.fallback
            else:
                text = script[self._step]
        self._step += 1

        # Fake tokenization: one id per character. Deterministic + recoverable.
        token_ids = [ord(c) for c in text]
        logprobs = [self.baseline_logprob for _ in token_ids]
        return CompletionOutput(
            text=text, token_ids=token_ids, logprobs_at_generation=logprobs,
        )

    # ---- LogprobCursorPolicy surface ----

    def teacher_force_logprobs(
        self,
        completions: Sequence[CompletionOutput],
        ref_only: bool = False,
    ) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for c in completions:
            n = len(c.token_ids)
            base = torch.full((n,), self.baseline_logprob, dtype=torch.float32)
            if ref_only:
                out.append(base)  # detached, no grad
            else:
                # Attach grad through `self.offset` so backward reaches the param.
                out.append(base + self.offset)
        return out
