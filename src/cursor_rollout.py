"""Multi-step rollout driver.

Given a base image, an instruction, a target bounding box, and a policy,
generates N independent trajectories. Each trajectory is a rollout through
the CursorEnv until the env signals done (stop action, invalid action, or
max_steps reached).

All orchestration -- env stepping, policy invocation, reward computation --
lives here. Env and policy are decoupled via the CursorPolicy Protocol, so
Phase 3 can drop in a real VLM-backed policy without touching this module.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from PIL import Image

from cursor_actions import parse_action
from cursor_env import CursorEnv
from cursor_policy import CursorPolicy
from cursor_rewards import (
    DEFAULT_PENALTY_WEIGHT,
    DEFAULT_REPEAT_TOL,
    Bbox,
    RewardBreakdown,
    Trajectory,
    trajectory_reward,
)

DEFAULT_MAX_STEPS = 4
DEFAULT_NUM_TRAJECTORIES = 6


@dataclass
class RolloutResult:
    """All data produced by a single call to `run_rollout`."""
    trajectories: List[Trajectory] = field(default_factory=list)
    rewards: List[RewardBreakdown] = field(default_factory=list)
    # Raw completion strings per step per trajectory. Phase 3 will pair these
    # with tokenizer output + logprobs for the policy gradient computation.
    completions: List[List[str]] = field(default_factory=list)


def run_single_trajectory(
    env: CursorEnv,
    policy: CursorPolicy,
    instruction: str,
) -> Tuple[Trajectory, List[str]]:
    """Drive `env` with `policy` until `step.done`.

    Returns the assembled Trajectory and the list of raw completion strings
    (one per policy invocation). The env is consumed -- callers should pass
    a fresh env per trajectory.
    """
    trajectory = Trajectory()
    completions: List[str] = []
    step_index = 0

    while True:
        image = env.render()
        completion = policy.generate(image, instruction, step_index)
        completions.append(completion)

        action = parse_action(completion)
        trajectory.actions.append(action)

        result = env.step(action)
        trajectory.steps = result.step_index

        if action.kind == "move" and result.was_valid:
            # env.position reflects the clamped coordinate; prefer it over
            # the raw action coords so rewards see what actually happened.
            assert result.position is not None
            trajectory.positions.append(result.position)

        if result.done:
            trajectory.final_position = env.position
            trajectory.stopped_via = _classify_termination(result.reason)
            break

        step_index += 1

    return trajectory, completions


def run_rollout(
    image: Image.Image,
    instruction: str,
    target_bbox: Bbox,
    policy: CursorPolicy,
    num_trajectories: int = DEFAULT_NUM_TRAJECTORIES,
    max_steps: int = DEFAULT_MAX_STEPS,
    repeat_tol: int = DEFAULT_REPEAT_TOL,
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
) -> RolloutResult:
    """Sample `num_trajectories` independent trajectories from `policy`.

    A fresh CursorEnv is built per trajectory so history doesn't leak. If
    the policy is a MockCursorPolicy, `next_trajectory()` is called between
    rollouts so scripts advance in sync.
    """
    result = RolloutResult()
    image_size = image.size

    for _ in range(num_trajectories):
        env = CursorEnv(image, max_steps=max_steps, repeat_tol=repeat_tol)
        trajectory, completions = run_single_trajectory(env, policy, instruction)
        reward = trajectory_reward(
            trajectory,
            target_bbox=target_bbox,
            image_size=image_size,
            penalty_weight=penalty_weight,
            repeat_tol=repeat_tol,
        )

        result.trajectories.append(trajectory)
        result.rewards.append(reward)
        result.completions.append(completions)

        # Advance scripted policies between trajectories. Real VLM policies
        # can implement a no-op `next_trajectory` (the Protocol doesn't
        # require it; we only call it if present).
        advance = getattr(policy, "next_trajectory", None)
        if callable(advance):
            advance()

    return result


def _classify_termination(env_reason: str) -> str:
    """Map CursorEnv's reason strings to the Trajectory schema."""
    if env_reason in ("stopped", "stopped_without_move"):
        return "stop"
    if env_reason == "invalid_action":
        return "invalid"
    if env_reason == "max_steps":
        return "max_steps"
    # Fallback: preserve whatever the env reported.
    return env_reason
