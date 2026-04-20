"""Group-normalized advantage computation for GRPO.

Given a flat list of trajectory rewards organized as `num_prompts * group_size`,
normalize within each group (the N trajectories sharing one prompt):

    A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

If a group has zero variance (all trajectories got the same reward), its
advantages are set to zero rather than NaN -- there's no learning signal
from such a group and the optimizer should just pass it by.
"""

import statistics
from typing import List


def group_advantages(
    rewards: List[float],
    group_size: int,
    eps: float = 1e-6,
) -> List[float]:
    """Normalize rewards within contiguous groups of size `group_size`.

    Args:
        rewards: flat list of length `num_prompts * group_size`.
        group_size: number of trajectories per prompt (>= 2 for meaningful std).
        eps: added to std for numerical stability.

    Returns:
        List of advantages, same length and order as `rewards`.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if len(rewards) % group_size != 0:
        raise ValueError(
            f"len(rewards)={len(rewards)} is not divisible by group_size={group_size}"
        )

    advantages = [0.0] * len(rewards)
    for group_start in range(0, len(rewards), group_size):
        group = rewards[group_start : group_start + group_size]
        mean = sum(group) / group_size
        if group_size > 1:
            variance = sum((r - mean) ** 2 for r in group) / group_size
            std = variance ** 0.5
        else:
            std = 0.0

        if std < eps:
            # Zero-variance group: no signal, give zero advantage to all members.
            for i, _ in enumerate(group):
                advantages[group_start + i] = 0.0
        else:
            for i, r in enumerate(group):
                advantages[group_start + i] = (r - mean) / (std + eps)

    return advantages


def _sample_stats(values: List[float]) -> tuple:
    """Return (mean, std) using population variance. Exposed for tests."""
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance ** 0.5
