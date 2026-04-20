"""Unit tests for the trajectory reward function.

All tests build Trajectory objects by hand so the reward logic is
exercised independently of the rollout driver and environment.
"""

import math
import os
import sys

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_actions import Action  # noqa: E402
from cursor_rewards import (  # noqa: E402
    RewardBreakdown,
    Trajectory,
    trajectory_reward,
)


IMAGE_W, IMAGE_H = 400, 400
IMAGE_SIZE = (IMAGE_W, IMAGE_H)
# bbox centered at (200, 200), 40x40
BBOX = (180, 180, 220, 220)
BBOX_CENTER = (200, 200)


def _traj(moves, stopped_via="stop", invalid=False):
    """Helper: build a Trajectory from a list of (x, y) moves."""
    actions = []
    positions = list(moves)
    for x, y in moves:
        actions.append(Action(kind="move", x=x, y=y))
    if stopped_via == "stop":
        actions.append(Action(kind="stop"))
    elif stopped_via == "invalid":
        actions.append(Action(kind="invalid", raw="garbage"))
    # max_steps case: no extra action appended
    final_position = positions[-1] if positions else None
    return Trajectory(
        actions=actions,
        positions=positions,
        final_position=final_position,
        stopped_via=stopped_via,
        steps=len(actions),
    )


# ---- Position reward ----

def test_position_reward_inside_bbox():
    traj = _traj([(200, 200)])  # dead center
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.position == pytest.approx(1.0)


def test_position_reward_just_inside_edge():
    traj = _traj([(181, 181)])  # just inside
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.position == pytest.approx(1.0)


def test_position_reward_outside_bbox_scales_with_distance():
    # Far corner (0,0) -> dist to (200,200) = sqrt(80000) ≈ 282.8
    # diag = sqrt(400^2 + 400^2) ≈ 565.7
    # reward ≈ 1 - 282.8/565.7 ≈ 0.5
    traj = _traj([(0, 0)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert 0.4 < r.position < 0.6


def test_position_reward_with_no_moves():
    """If the trajectory has no move actions, position reward should be 0."""
    traj = Trajectory(
        actions=[Action(kind="stop")],
        positions=[],
        final_position=None,
        stopped_via="stop",
        steps=1,
    )
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.position == 0.0


def test_position_reward_never_negative():
    # Place bbox such that max distance > diagonal; check floor at 0
    # A trajectory at (0,0) with target way off-image isn't realistic, but
    # we still guarantee max(0, ...) semantics.
    far_bbox = (395, 395, 399, 399)
    traj = _traj([(0, 0)])
    r = trajectory_reward(traj, far_bbox, IMAGE_SIZE)
    assert r.position >= 0.0


# ---- False stop penalty ----

def test_false_stop_penalty_when_outside():
    traj = _traj([(50, 50)], stopped_via="stop")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_stop == pytest.approx(-0.2)


def test_no_false_stop_when_inside():
    traj = _traj([(200, 200)], stopped_via="stop")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_stop == 0.0


def test_no_false_stop_when_terminated_by_max_steps():
    traj = _traj([(50, 50)], stopped_via="max_steps")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    # max_steps is not "stop" -> no false_stop penalty
    assert r.false_stop == 0.0


def test_no_false_stop_when_invalid():
    traj = _traj([(50, 50)], stopped_via="invalid")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_stop == 0.0


# ---- False move penalty ----

def test_false_move_penalty_inside_then_outside():
    # Step 1 inside, step 2 outside -> penalty
    traj = _traj([(200, 200), (50, 50)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_move == pytest.approx(-0.2)


def test_no_false_move_when_always_outside():
    traj = _traj([(50, 50), (60, 60)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_move == 0.0


def test_no_false_move_when_always_inside():
    traj = _traj([(195, 195), (205, 205)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_move == 0.0


def test_no_false_move_with_single_move():
    traj = _traj([(200, 200)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_move == 0.0


# ---- False direction penalty ----

def test_false_direction_penalty_when_moving_away():
    # First move at dist ~100, second at dist ~200 (farther)
    traj = _traj([(100, 200), (0, 200)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_direction == pytest.approx(-0.2)


def test_no_false_direction_when_moving_closer():
    traj = _traj([(0, 200), (100, 200)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_direction == 0.0


def test_no_false_direction_with_single_move():
    traj = _traj([(50, 50)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_direction == 0.0


# ---- Repeated position penalty ----

def test_repeated_position_penalty():
    traj = _traj([(100, 100), (100, 100)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.repeated_position == pytest.approx(-0.2)


def test_repeated_position_within_tolerance():
    traj = _traj([(100, 100), (103, 102)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE, repeat_tol=5)
    assert r.repeated_position == pytest.approx(-0.2)


def test_no_repeat_outside_tolerance():
    traj = _traj([(100, 100), (150, 150)])
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE, repeat_tol=5)
    assert r.repeated_position == 0.0


# ---- Totals & bounds ----

def test_all_penalties_compound():
    # inside -> outside (false_move + false_direction)
    # stop while outside (false_stop)
    # revisit within tolerance (repeated)
    traj = _traj([(200, 200), (200, 202), (0, 0)], stopped_via="stop")
    # step1 (200,200) inside. step2 (200,202) still inside, within tol of step1 -> repeated
    # step3 (0, 0) outside. inside->outside = false_move. also dist increased = false_direction.
    # stop outside bbox -> false_stop
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.false_stop == pytest.approx(-0.2)
    assert r.false_move == pytest.approx(-0.2)
    assert r.false_direction == pytest.approx(-0.2)
    assert r.repeated_position == pytest.approx(-0.2)


def test_reward_lower_bound():
    # Construct the worst-case trajectory and check >= -0.8
    traj = _traj([(200, 200), (200, 202), (0, 0)], stopped_via="stop")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.total >= -0.8 - 1e-6
    # Also: sum of penalties never below -0.8
    penalties = r.false_stop + r.false_move + r.false_direction + r.repeated_position
    assert penalties >= -0.8 - 1e-6


def test_reward_upper_bound():
    # Best case: single move to center, stop there
    traj = _traj([(200, 200)], stopped_via="stop")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.total == pytest.approx(1.0)
    assert r.total <= 1.0 + 1e-6


def test_total_equals_sum_of_components():
    traj = _traj([(180, 180), (100, 100)], stopped_via="stop")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    expected = (
        r.position + r.false_stop + r.false_move
        + r.false_direction + r.repeated_position
    )
    assert r.total == pytest.approx(expected)


def test_breakdown_is_dataclass_like():
    traj = _traj([(200, 200)], stopped_via="stop")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert isinstance(r, RewardBreakdown)
    assert hasattr(r, "position")
    assert hasattr(r, "total")


# ---- Phase 6: stop bonus + time penalty (new shaping terms) ----


def test_stop_bonus_zero_by_default():
    """stop_bonus_when_inside defaults to 0 -> no behavioral change."""
    traj = _traj([(200, 200)], stopped_via="stop")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.stop_bonus == 0.0
    # Total still matches the pre-Phase-6 expectation
    assert r.total == pytest.approx(1.0)


def test_stop_bonus_fires_only_on_stop_inside():
    traj = _traj([(200, 200)], stopped_via="stop")
    r = trajectory_reward(
        traj, BBOX, IMAGE_SIZE,
        stop_bonus_when_inside=0.3,
    )
    assert r.stop_bonus == pytest.approx(0.3)
    # Position 1.0 + bonus 0.3 = 1.3, all other components zero
    assert r.total == pytest.approx(1.3)


def test_stop_bonus_does_not_fire_on_stop_outside():
    """STOP outside bbox: false_stop fires, stop_bonus does NOT."""
    traj = _traj([(50, 50)], stopped_via="stop")
    r = trajectory_reward(
        traj, BBOX, IMAGE_SIZE,
        stop_bonus_when_inside=0.3,
    )
    assert r.stop_bonus == 0.0
    assert r.false_stop == pytest.approx(-0.2)


def test_stop_bonus_does_not_fire_on_max_steps_termination():
    """Even if final position is inside, max_steps termination is NOT a stop."""
    traj = _traj([(200, 200)], stopped_via="max_steps")
    r = trajectory_reward(
        traj, BBOX, IMAGE_SIZE,
        stop_bonus_when_inside=0.3,
    )
    assert r.stop_bonus == 0.0


def test_stop_bonus_does_not_fire_when_no_position():
    """Stopping without ever moving -> no final position -> no bonus."""
    traj = Trajectory(
        actions=[Action(kind="stop")],
        positions=[],
        final_position=None,
        stopped_via="stop",
        steps=1,
    )
    r = trajectory_reward(
        traj, BBOX, IMAGE_SIZE,
        stop_bonus_when_inside=0.5,
    )
    assert r.stop_bonus == 0.0


def test_time_penalty_zero_by_default():
    """time_penalty_per_step defaults to 0 -> no behavioral change."""
    traj = _traj([(200, 200)], stopped_via="stop")
    r = trajectory_reward(traj, BBOX, IMAGE_SIZE)
    assert r.time_penalty == 0.0


def test_time_penalty_scales_linearly_with_steps():
    # 1 move + 1 stop = 2 actions
    traj_short = _traj([(200, 200)], stopped_via="stop")
    # 4 moves + 1 stop = 5 actions
    traj_long = _traj([(200, 200), (200, 201), (200, 202), (200, 203)], stopped_via="stop")
    r_short = trajectory_reward(
        traj_short, BBOX, IMAGE_SIZE, time_penalty_per_step=0.05,
    )
    r_long = trajectory_reward(
        traj_long, BBOX, IMAGE_SIZE, time_penalty_per_step=0.05,
    )
    assert r_short.time_penalty == pytest.approx(-0.05 * 2)
    assert r_long.time_penalty == pytest.approx(-0.05 * 5)


def test_one_step_solution_strictly_beats_max_steps():
    """The headline behavior the reward fix needs to enable.

    Phase 5: max_steps trajectory had reward >= 1-step trajectory ->
    model never stopped. Phase 6: with stop_bonus 0.3 + time_penalty
    0.05, the 1-move-then-stop trajectory must score strictly higher
    than the 4-move max_steps trajectory even when both end inside.
    """
    one_step = _traj([(200, 200)], stopped_via="stop")  # 2 actions
    # Four moves all inside, no STOP (max_steps termination)
    four_steps = Trajectory(
        actions=[Action(kind="move", x=200, y=200)] * 4,
        positions=[(200, 200)] * 4,
        final_position=(200, 200),
        stopped_via="max_steps",
        steps=4,
    )
    r1 = trajectory_reward(
        one_step, BBOX, IMAGE_SIZE,
        stop_bonus_when_inside=0.3, time_penalty_per_step=0.05,
    )
    r4 = trajectory_reward(
        four_steps, BBOX, IMAGE_SIZE,
        stop_bonus_when_inside=0.3, time_penalty_per_step=0.05,
    )
    assert r1.total > r4.total, (
        f"1-step ({r1.total:.3f}) should beat max-steps ({r4.total:.3f}) "
        f"with new rewards; otherwise Phase 6 won't actually make the model stop"
    )


def test_total_equals_sum_with_new_components():
    """Total should include stop_bonus and time_penalty in the sum."""
    traj = _traj([(200, 200)], stopped_via="stop")
    r = trajectory_reward(
        traj, BBOX, IMAGE_SIZE,
        stop_bonus_when_inside=0.3, time_penalty_per_step=0.05,
    )
    expected = (
        r.position + r.false_stop + r.false_move
        + r.false_direction + r.repeated_position
        + r.stop_bonus + r.time_penalty
    )
    assert r.total == pytest.approx(expected)


def test_negative_stop_bonus_is_clamped_to_zero():
    """Defensive: callers passing negative bonus shouldn't accidentally double-penalize."""
    traj = _traj([(200, 200)], stopped_via="stop")
    r = trajectory_reward(
        traj, BBOX, IMAGE_SIZE,
        stop_bonus_when_inside=-0.5,  # nonsense input
    )
    assert r.stop_bonus == 0.0


def test_negative_time_penalty_is_clamped_to_zero():
    traj = _traj([(200, 200)], stopped_via="stop")
    r = trajectory_reward(
        traj, BBOX, IMAGE_SIZE,
        time_penalty_per_step=-0.1,  # nonsense input
    )
    assert r.time_penalty == 0.0
