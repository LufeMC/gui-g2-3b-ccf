"""Tests for the multi-step rollout driver.

All tests use MockCursorPolicy -- no real model, no GPU. The goal is to
validate the orchestration logic (env <-> policy <-> reward) end-to-end.
"""

import os
import sys

import pytest
from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_env import CursorEnv  # noqa: E402
from cursor_policy import MockCursorPolicy  # noqa: E402
from cursor_rewards import RewardBreakdown, Trajectory  # noqa: E402
from cursor_rollout import run_rollout, run_single_trajectory  # noqa: E402


IMAGE_W, IMAGE_H = 400, 400
BBOX = (180, 180, 220, 220)


@pytest.fixture
def blank_image() -> Image.Image:
    return Image.new("RGB", (IMAGE_W, IMAGE_H), (255, 255, 255))


# ---- Single trajectory ----

def test_single_trajectory_stop_action(blank_image):
    policy = MockCursorPolicy([[
        "<action>move(200, 200)</action>",
        "<action>stop</action>",
    ]])
    env = CursorEnv(blank_image, max_steps=4)
    traj, completions = run_single_trajectory(env, policy, "click the target")

    assert traj.stopped_via == "stop"
    assert traj.final_position == (200, 200)
    assert traj.positions == [(200, 200)]
    assert traj.steps == 2  # move + stop
    assert len(completions) == 2


def test_single_trajectory_max_steps(blank_image):
    policy = MockCursorPolicy([[
        "<action>move(10, 10)</action>",
        "<action>move(20, 20)</action>",
        "<action>move(30, 30)</action>",
    ]])
    env = CursorEnv(blank_image, max_steps=3)
    traj, completions = run_single_trajectory(env, policy, "go")

    assert traj.stopped_via == "max_steps"
    assert traj.final_position == (30, 30)
    assert len(traj.positions) == 3
    assert len(completions) == 3


def test_single_trajectory_invalid_action(blank_image):
    policy = MockCursorPolicy([[
        "<action>move(50, 50)</action>",
        "this is not a valid action",
    ]])
    env = CursorEnv(blank_image, max_steps=4)
    traj, completions = run_single_trajectory(env, policy, "go")

    assert traj.stopped_via == "invalid"
    assert traj.final_position == (50, 50)
    assert len(completions) == 2  # even the invalid one is captured


def test_single_trajectory_stop_only(blank_image):
    """Stop before any move -> trajectory has no positions."""
    policy = MockCursorPolicy([["<action>stop</action>"]])
    env = CursorEnv(blank_image, max_steps=4)
    traj, completions = run_single_trajectory(env, policy, "go")

    assert traj.stopped_via == "stop"
    assert traj.final_position is None
    assert traj.positions == []
    assert len(completions) == 1


# ---- Multi-trajectory rollout ----

def test_rollout_runs_n_trajectories(blank_image):
    scripts = [
        ["<action>move(200, 200)</action>", "<action>stop</action>"],
        ["<action>move(100, 100)</action>", "<action>stop</action>"],
        ["<action>move(300, 300)</action>", "<action>stop</action>"],
    ]
    policy = MockCursorPolicy(scripts)
    result = run_rollout(
        image=blank_image,
        instruction="click",
        target_bbox=BBOX,
        policy=policy,
        num_trajectories=3,
        max_steps=4,
    )

    assert len(result.trajectories) == 3
    assert len(result.rewards) == 3
    assert len(result.completions) == 3


def test_rollout_advances_policy_between_trajectories(blank_image):
    scripts = [
        ["<action>move(200, 200)</action>", "<action>stop</action>"],
        ["<action>move(100, 100)</action>", "<action>stop</action>"],
    ]
    policy = MockCursorPolicy(scripts)
    result = run_rollout(
        image=blank_image,
        instruction="click",
        target_bbox=BBOX,
        policy=policy,
        num_trajectories=2,
        max_steps=4,
    )

    # Each trajectory should use its own script
    assert result.trajectories[0].final_position == (200, 200)
    assert result.trajectories[1].final_position == (100, 100)


def test_rollout_passes_rendered_images_per_step(blank_image):
    """The policy sees a fresh rendered image each step (not just the original)."""
    policy = MockCursorPolicy([[
        "<action>move(200, 200)</action>",
        "<action>move(210, 210)</action>",
        "<action>stop</action>",
    ]])
    run_rollout(
        image=blank_image,
        instruction="click",
        target_bbox=BBOX,
        policy=policy,
        num_trajectories=1,
        max_steps=4,
    )

    # Policy was called 3 times for the single trajectory
    assert len(policy.calls) == 3
    # All images have the same size as the base
    for call in policy.calls:
        assert call.image_size == (IMAGE_W, IMAGE_H)
    # Step indices progress
    assert [c.step_index for c in policy.calls] == [0, 1, 2]


def test_rollout_collects_completions(blank_image):
    scripts = [
        ["<action>move(200, 200)</action>", "<action>stop</action>"],
    ]
    policy = MockCursorPolicy(scripts)
    result = run_rollout(
        image=blank_image,
        instruction="click",
        target_bbox=BBOX,
        policy=policy,
        num_trajectories=1,
        max_steps=4,
    )

    # Raw completion strings preserved verbatim for Phase 3 logprob use
    assert result.completions[0] == scripts[0]


def test_rollout_computes_rewards(blank_image):
    # Traj 1: lands on center -> total 1.0
    # Traj 2: lands outside, stops -> negative penalties + partial position
    scripts = [
        ["<action>move(200, 200)</action>", "<action>stop</action>"],
        ["<action>move(0, 0)</action>", "<action>stop</action>"],
    ]
    policy = MockCursorPolicy(scripts)
    result = run_rollout(
        image=blank_image,
        instruction="click",
        target_bbox=BBOX,
        policy=policy,
        num_trajectories=2,
        max_steps=4,
    )

    assert all(isinstance(r, RewardBreakdown) for r in result.rewards)
    assert result.rewards[0].total == pytest.approx(1.0)
    # Traj 2 should be lower than traj 1
    assert result.rewards[1].total < result.rewards[0].total
    # And should have false_stop triggered
    assert result.rewards[1].false_stop == pytest.approx(-0.2)


def test_rollout_returns_trajectory_type(blank_image):
    policy = MockCursorPolicy([["<action>stop</action>"]])
    result = run_rollout(
        image=blank_image,
        instruction="x",
        target_bbox=BBOX,
        policy=policy,
        num_trajectories=1,
        max_steps=4,
    )
    assert isinstance(result.trajectories[0], Trajectory)


def test_rollout_fresh_env_per_trajectory(blank_image):
    """State from one trajectory must not leak into the next."""
    scripts = [
        # Traj 1 visits (200, 200) and stops
        ["<action>move(200, 200)</action>", "<action>stop</action>"],
        # Traj 2 visits (200, 200) too -- should NOT be flagged as repeated
        # because repeated_position is scoped to a single trajectory.
        ["<action>move(200, 200)</action>", "<action>stop</action>"],
    ]
    policy = MockCursorPolicy(scripts)
    result = run_rollout(
        image=blank_image,
        instruction="click",
        target_bbox=BBOX,
        policy=policy,
        num_trajectories=2,
        max_steps=4,
    )
    assert result.rewards[1].repeated_position == 0.0
