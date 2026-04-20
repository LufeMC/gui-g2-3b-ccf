"""Tests for group_advantages."""

import math
import os
import sys

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_advantages import group_advantages  # noqa: E402


def test_single_group_normalizes_to_zero_mean():
    rewards = [0.0, 1.0, 2.0, 3.0]
    advs = group_advantages(rewards, group_size=4)
    assert len(advs) == 4
    assert math.isclose(sum(advs) / len(advs), 0.0, abs_tol=1e-6)


def test_multiple_groups_normalized_independently():
    # Two groups with very different scales
    rewards = [0.0, 1.0,   # group 0: mean=0.5
               100.0, 200.0]  # group 1: mean=150
    advs = group_advantages(rewards, group_size=2)
    # Group 0: lower reward -> negative advantage, higher -> positive
    assert advs[0] < 0
    assert advs[1] > 0
    # Group 1 same pattern
    assert advs[2] < 0
    assert advs[3] > 0
    # Magnitudes equal per group (symmetric around mean)
    assert math.isclose(abs(advs[0]), abs(advs[1]), abs_tol=1e-6)
    assert math.isclose(abs(advs[2]), abs(advs[3]), abs_tol=1e-6)


def test_zero_variance_group_returns_zeros():
    rewards = [0.5, 0.5, 0.5, 0.5]
    advs = group_advantages(rewards, group_size=4)
    assert all(a == 0.0 for a in advs)
    # Critically: no NaN
    assert all(not math.isnan(a) for a in advs)


def test_preserves_order():
    rewards = [1.0, 3.0, 2.0, 4.0]
    advs = group_advantages(rewards, group_size=2)
    # Within group 0: rewards[0] < rewards[1] -> advs[0] < advs[1]
    assert advs[0] < advs[1]
    # Within group 1: rewards[2] < rewards[3] -> advs[2] < advs[3]
    assert advs[2] < advs[3]


def test_normalization_has_unit_std():
    rewards = [0.0, 1.0, 2.0, 3.0]
    advs = group_advantages(rewards, group_size=4)
    mean = sum(advs) / 4
    variance = sum((a - mean) ** 2 for a in advs) / 4
    std = variance ** 0.5
    assert math.isclose(std, 1.0, rel_tol=1e-3)


def test_group_size_1_returns_zero():
    """A group of one has no std; advantage should be 0."""
    rewards = [0.5, 1.0, 2.0]
    advs = group_advantages(rewards, group_size=1)
    assert all(a == 0.0 for a in advs)


def test_mismatched_length_raises():
    with pytest.raises(ValueError, match="divisible"):
        group_advantages([1.0, 2.0, 3.0], group_size=2)


def test_nonpositive_group_size_raises():
    with pytest.raises(ValueError):
        group_advantages([1.0, 2.0], group_size=0)


def test_empty_rewards():
    assert group_advantages([], group_size=4) == []
