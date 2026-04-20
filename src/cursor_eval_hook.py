"""Lightweight in-training validation eval for the cursor GRPO loop.

Without this hook a 250-step training run is a 3-hour black box: we don't
know whether the policy is learning until the run is done and we point
`eval.py` at the saved checkpoint. The hook runs greedy rollouts on a
small held-out cursor val set every `eval_every` steps and reports:

  - accuracy        : final cursor position inside the target bbox
  - parse_rate      : trajectories that emitted at least one valid move
  - mean_steps      : average steps until stop or max_steps
  - num_samples     : how many samples were evaluated

Greedy = `do_sample=False`. This trades a tiny bit of policy realism for
deterministic numbers across runs so we can compare step-N to step-(N+25)
without sampling noise. The full ScreenSpot-v2 eval at the end of training
uses the same greedy generation, so the in-training acc tracks the metric
we care about.

The eval is small on purpose (50-100 samples, ~30-90s on GPU). We're
trading off coverage for frequency: a 50-sample acc with std ~5pp is
plenty to spot regressions early.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from cursor_actions import parse_action
from cursor_dataset import CursorSample
from cursor_env import CursorEnv


@dataclass
class CursorEvalResult:
    """Summary of one validation pass."""
    step: int
    num_samples: int
    accuracy: float
    parse_rate: float
    mean_steps: float
    mean_steps_when_stopped_cleanly: float


def _hit(final_xy, target_bbox) -> bool:
    """True if (x, y) lies inside [x1, y1, x2, y2] (inclusive)."""
    x, y = final_xy
    x1, y1, x2, y2 = target_bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _greedy_rollout(
    sample: CursorSample,
    policy: Any,
    max_steps: int,
    repeat_tol: int,
):
    """Run a single greedy trajectory. Returns (final_xy, num_steps, made_valid_move, stopped_cleanly)."""
    env = CursorEnv(sample.image, max_steps=max_steps, repeat_tol=repeat_tol)
    made_valid_move = False
    stopped_cleanly = False
    step_index = 0

    # Many policies (especially MockLogprobPolicy) advance scripts on
    # generate. We use generate() not generate_with_logprobs() so we
    # don't pay the logprob bookkeeping cost during eval.
    while True:
        image = env.render()
        text = policy.generate(image, sample.instruction, step_index)
        action = parse_action(text)
        result = env.step(action)
        if action.kind == "move" and result.was_valid:
            made_valid_move = True
        if result.done:
            if result.reason in ("stopped", "stopped_without_move"):
                stopped_cleanly = True
            return env.position, result.step_index, made_valid_move, stopped_cleanly
        step_index += 1


def evaluate_cursor_policy(
    policy: Any,
    samples: Sequence[CursorSample],
    *,
    step: int = 0,
    max_steps: int = 4,
    repeat_tol: int = 4,
) -> CursorEvalResult:
    """Greedy-evaluate `policy` on `samples`. Returns a `CursorEvalResult`.

    Toggles `policy.cfg.do_sample = False` during eval if the policy has
    that field, then restores it. Same for `temperature`. Mock policies
    don't expose `cfg` and are unaffected.
    """
    if not samples:
        return CursorEvalResult(
            step=step, num_samples=0, accuracy=0.0,
            parse_rate=0.0, mean_steps=0.0,
            mean_steps_when_stopped_cleanly=0.0,
        )

    cfg = getattr(policy, "cfg", None)
    saved_do_sample = getattr(cfg, "do_sample", None) if cfg else None
    saved_temperature = getattr(cfg, "temperature", None) if cfg else None
    if cfg is not None:
        if saved_do_sample is not None:
            cfg.do_sample = False
        if saved_temperature is not None:
            cfg.temperature = 1.0  # any value is ignored when do_sample=False

    correct = 0
    valid_moves = 0
    total_steps = 0
    clean_stop_steps_sum = 0
    clean_stops = 0

    try:
        for sample in samples:
            final_xy, n_steps, made_valid, stopped_cleanly = _greedy_rollout(
                sample, policy, max_steps=max_steps, repeat_tol=repeat_tol,
            )
            total_steps += n_steps
            if made_valid:
                valid_moves += 1
            if stopped_cleanly:
                clean_stops += 1
                clean_stop_steps_sum += n_steps
            if made_valid and _hit(final_xy, sample.target_bbox):
                correct += 1

            advance = getattr(policy, "next_trajectory", None)
            if callable(advance):
                advance()
    finally:
        if cfg is not None:
            if saved_do_sample is not None:
                cfg.do_sample = saved_do_sample
            if saved_temperature is not None:
                cfg.temperature = saved_temperature

    n = len(samples)
    return CursorEvalResult(
        step=step,
        num_samples=n,
        accuracy=correct / n,
        parse_rate=valid_moves / n,
        mean_steps=total_steps / n,
        mean_steps_when_stopped_cleanly=(
            clean_stop_steps_sum / clean_stops if clean_stops else 0.0
        ),
    )
