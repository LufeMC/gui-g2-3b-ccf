"""CCF wrapper for the cursor-movement policy.

Mirrors the GUI-Cursor paper's inference-time CCF exactly: run a coarse
cursor rollout on the full image, crop around the final cursor position,
run a refined rollout on the crop, and translate the refined cursor back
to original-image coordinates.

This module is pure orchestration -- it delegates the actual model calls
to a LogprobCursorPolicy, so it's testable on Mac with MockLogprobPolicy
just like the training loop.
"""

from typing import Optional

from PIL import Image

from cursor_ccf import (
    CCFConfig,
    CCFResult,
    compute_crop_window,
    map_crop_to_orig,
    _should_skip_ccf,
)
from cursor_env import CursorEnv
from cursor_rewards import Trajectory

DEFAULT_MAX_STEPS_PER_STAGE = 3


def _rollout_final_position(
    env: CursorEnv,
    policy,
    instruction: str,
) -> Optional[tuple]:
    """Drive `env` with `policy.generate_with_logprobs` and return the
    final cursor position (or None if the rollout never placed a cursor).
    """
    # Delay-import to avoid a hard dependency on the trainer module.
    from cursor_actions import parse_action

    step_index = 0
    while True:
        image = env.render()
        comp = policy.generate_with_logprobs(image, instruction, step_index)
        action = parse_action(comp.text)
        result = env.step(action)
        if result.done:
            return env.position
        step_index += 1


def ccf_predict_cursor(
    policy,
    image: Image.Image,
    instruction: str,
    config: Optional[CCFConfig] = None,
    max_steps_per_stage: int = DEFAULT_MAX_STEPS_PER_STAGE,
) -> Optional[CCFResult]:
    """Two-stage cursor rollout: coarse on full image, refined on crop.

    Returns CCFResult with final coordinates in the ORIGINAL image's
    pixel space. Falls back to coarse if the refined rollout produces
    no valid cursor position (matching `fallback_to_coarse_on_invalid`).
    """
    if config is None:
        config = CCFConfig()

    coarse_env = CursorEnv(image, max_steps=max_steps_per_stage)
    coarse_pos = _rollout_final_position(coarse_env, policy, instruction)
    if coarse_pos is None:
        return None

    # Advance scripted mock policies between stages; no-op for real VLMs.
    advance = getattr(policy, "next_trajectory", None)
    if callable(advance):
        advance()

    if _should_skip_ccf(image.size, config):
        return CCFResult(
            x=float(coarse_pos[0]),
            y=float(coarse_pos[1]),
            stage="coarse",
            coarse_xy=(float(coarse_pos[0]), float(coarse_pos[1])),
            crop_window=None,
        )

    crop_window = compute_crop_window(
        center=coarse_pos,
        image_size=image.size,
        zoom_factor=config.zoom_factor,
        min_crop_side=config.min_crop_side,
    )
    cropped = image.crop(crop_window)

    refined_env = CursorEnv(cropped, max_steps=max_steps_per_stage)
    refined_pos = _rollout_final_position(refined_env, policy, instruction)

    if callable(advance):
        advance()

    if refined_pos is None:
        if not config.fallback_to_coarse_on_invalid:
            return None
        return CCFResult(
            x=float(coarse_pos[0]),
            y=float(coarse_pos[1]),
            stage="fallback",
            coarse_xy=(float(coarse_pos[0]), float(coarse_pos[1])),
            crop_window=crop_window,
        )

    orig_x, orig_y = map_crop_to_orig(refined_pos[0], refined_pos[1], crop_window)
    return CCFResult(
        x=orig_x,
        y=orig_y,
        stage="refined",
        coarse_xy=(float(coarse_pos[0]), float(coarse_pos[1])),
        crop_window=crop_window,
    )
