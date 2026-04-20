"""Trajectory-level reward for GUI-Cursor style training.

The reward combines a position-based score (how close the final cursor
position is to the target bounding box) with four trajectory penalties
that discourage pathological search behaviour. Matches GUI-Cursor paper
(arXiv:2509.21552), Section 2.3.

All coordinates live in the same pixel space as the image that was
rendered for the model -- no coordinate conversion happens here.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from cursor_actions import Action

Bbox = Tuple[int, int, int, int]  # x1, y1, x2, y2
Point = Tuple[int, int]

DEFAULT_PENALTY_WEIGHT = 0.2
DEFAULT_REPEAT_TOL = 5

# New in Phase 6: defaults are 0 here to keep `trajectory_reward` itself
# backward-compatible (the existing 21 unit tests assume the original
# reward shape). The training loop explicitly passes the production
# values (0.3 / 0.05) via TrainConfig.
DEFAULT_STOP_BONUS_WHEN_INSIDE = 0.0
DEFAULT_TIME_PENALTY_PER_STEP = 0.0


@dataclass
class Trajectory:
    """A full rollout: the sequence of model actions and cursor positions."""
    actions: List[Action] = field(default_factory=list)
    positions: List[Point] = field(default_factory=list)
    final_position: Optional[Point] = None
    stopped_via: str = ""  # "stop" | "invalid" | "max_steps"
    steps: int = 0


@dataclass
class RewardBreakdown:
    """Per-component reward values for analysis and logging."""
    position: float
    false_stop: float
    false_move: float
    false_direction: float
    repeated_position: float
    total: float
    # New in Phase 6: positive shaping terms that fix the
    # "model never stops, always uses max_steps=4" pathology we
    # diagnosed in Phase 5. Both default to 0 in the existing
    # `trajectory_reward` to keep the original tests valid; the
    # trainer passes non-zero values via TrainConfig.
    stop_bonus: float = 0.0
    time_penalty: float = 0.0


def _bbox_center(bbox: Bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _inside_bbox(pos: Point, bbox: Bbox) -> bool:
    x, y = pos
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _dist_to_center(pos: Point, bbox: Bbox) -> float:
    cx, cy = _bbox_center(bbox)
    dx = pos[0] - cx
    dy = pos[1] - cy
    return math.sqrt(dx * dx + dy * dy)


def _image_diagonal(image_size: Tuple[int, int]) -> float:
    w, h = image_size
    return math.sqrt(w * w + h * h)


def _position_reward(
    final_pos: Optional[Point],
    bbox: Bbox,
    image_size: Tuple[int, int],
) -> float:
    if final_pos is None:
        return 0.0
    if _inside_bbox(final_pos, bbox):
        return 1.0
    diag = _image_diagonal(image_size)
    if diag <= 0:
        return 0.0
    dist = _dist_to_center(final_pos, bbox)
    return max(0.0, 1.0 - dist / diag)


def _false_stop_penalty(
    trajectory: Trajectory,
    bbox: Bbox,
    weight: float,
) -> float:
    if trajectory.stopped_via != "stop":
        return 0.0
    if trajectory.final_position is None:
        # stop without any prior move; treat as a stop outside bbox
        return -weight
    if _inside_bbox(trajectory.final_position, bbox):
        return 0.0
    return -weight


def _false_move_penalty(
    positions: List[Point],
    bbox: Bbox,
    weight: float,
) -> float:
    """-weight if any step had cursor inside bbox and a later step has it outside."""
    seen_inside = False
    for pos in positions:
        if seen_inside and not _inside_bbox(pos, bbox):
            return -weight
        if _inside_bbox(pos, bbox):
            seen_inside = True
    return 0.0


def _false_direction_penalty(
    positions: List[Point],
    bbox: Bbox,
    weight: float,
) -> float:
    """-weight if distance to bbox center ever increases between consecutive moves."""
    for i in range(1, len(positions)):
        prev = _dist_to_center(positions[i - 1], bbox)
        curr = _dist_to_center(positions[i], bbox)
        if curr > prev:
            return -weight
    return 0.0


def _repeated_position_penalty(
    positions: List[Point],
    weight: float,
    tol: int,
) -> float:
    """-weight if any move lands within `tol` pixels of an earlier visited point."""
    for i, curr in enumerate(positions):
        for prev in positions[:i]:
            if abs(curr[0] - prev[0]) <= tol and abs(curr[1] - prev[1]) <= tol:
                return -weight
    return 0.0


def _stop_bonus(
    trajectory: Trajectory,
    bbox: Bbox,
    bonus: float,
) -> float:
    """+bonus when STOP is called and final cursor is INSIDE the bbox.

    Counterweight to the false-stop penalty. Together they make a
    correct early stop strictly better than running max_steps with
    the same final position.
    """
    if bonus <= 0.0:
        return 0.0
    if trajectory.stopped_via != "stop":
        return 0.0
    if trajectory.final_position is None:
        return 0.0
    if not _inside_bbox(trajectory.final_position, bbox):
        return 0.0
    return bonus


def _time_penalty(
    trajectory: Trajectory,
    per_step: float,
) -> float:
    """-per_step * trajectory.steps. Linearly penalizes longer trajectories.

    Counts every action emitted (moves AND the terminating stop), so a
    1-move-then-stop trajectory pays 2*per_step while a max_steps=4
    trajectory pays 4*per_step. With the production per_step=0.05 and
    a position reward of 1.0, a 1-step solution scores 1.0 + bonus -
    0.10 vs 1.0 - 0.20 for the max_steps version -- 1-step strictly wins.
    """
    if per_step <= 0.0:
        return 0.0
    return -per_step * float(trajectory.steps)


def trajectory_reward(
    trajectory: Trajectory,
    target_bbox: Bbox,
    image_size: Tuple[int, int],
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
    repeat_tol: int = DEFAULT_REPEAT_TOL,
    stop_bonus_when_inside: float = DEFAULT_STOP_BONUS_WHEN_INSIDE,
    time_penalty_per_step: float = DEFAULT_TIME_PENALTY_PER_STEP,
) -> RewardBreakdown:
    """Compute the full reward for a trajectory.

    The original GUI-Cursor reward shape (position + 4 penalties) made
    the model never stop early in Phase 5: stopping outside the bbox
    cost -0.2, but stopping correctly cost 0, while running max_steps
    also cost 0 -- so it was always strictly safer to run all 4 steps.
    Phase 6 adds two optional shaping terms (default 0 to preserve the
    original behavior, set to 0.3/0.05 by the trainer) that fix this:

    - `stop_bonus_when_inside`: +bonus for STOP when inside the bbox
    - `time_penalty_per_step`: -per_step * trajectory.steps

    Worst-case lower bound (with default zeros): position - 0.8.
    With production shaping: position - 0.8 - 4 * time_penalty_per_step.
    """
    position = _position_reward(trajectory.final_position, target_bbox, image_size)
    false_stop = _false_stop_penalty(trajectory, target_bbox, penalty_weight)
    false_move = _false_move_penalty(trajectory.positions, target_bbox, penalty_weight)
    false_direction = _false_direction_penalty(
        trajectory.positions, target_bbox, penalty_weight
    )
    repeated = _repeated_position_penalty(
        trajectory.positions, penalty_weight, repeat_tol
    )
    stop_bonus = _stop_bonus(trajectory, target_bbox, stop_bonus_when_inside)
    time_penalty = _time_penalty(trajectory, time_penalty_per_step)
    total = (
        position + false_stop + false_move + false_direction + repeated
        + stop_bonus + time_penalty
    )

    return RewardBreakdown(
        position=position,
        false_stop=false_stop,
        false_move=false_move,
        false_direction=false_direction,
        repeated_position=repeated,
        total=total,
        stop_bonus=stop_bonus,
        time_penalty=time_penalty,
    )
