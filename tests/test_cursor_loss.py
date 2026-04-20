"""Tests for grpo_loss. All tensors are small CPU tensors -- no GPU needed."""

import math
import os
import sys

import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_loss import grpo_loss  # noqa: E402


def _lps(*values, requires_grad=False):
    """Tiny helper: build a 1D tensor of log-probs."""
    t = torch.tensor(list(values), dtype=torch.float32)
    if requires_grad:
        t.requires_grad_(True)
    return t


def test_identical_logprobs_zero_policy_loss():
    """If new == old, ratio=1 and the surrogate equals the advantage with sign flipped."""
    new = _lps(-1.0, -2.0, -3.0, requires_grad=True)
    old = _lps(-1.0, -2.0, -3.0)
    ref = _lps(-1.0, -2.0, -3.0)

    out = grpo_loss(new, old, ref, advantage=0.5)
    # ratio=1, so per-token policy loss = -advantage (uniform)
    assert math.isclose(out["policy_loss"].item(), -0.5, abs_tol=1e-5)
    # KL = 0
    assert math.isclose(out["kl_loss"].item(), 0.0, abs_tol=1e-5)


def test_kl_term_is_zero_when_new_equals_ref():
    new = _lps(-1.0, -2.0, requires_grad=True)
    old = _lps(-1.5, -2.5)  # different from new -> nonzero policy but we ignore
    ref = _lps(-1.0, -2.0)
    out = grpo_loss(new, old, ref, advantage=0.0)
    assert math.isclose(out["kl_loss"].item(), 0.0, abs_tol=1e-5)


def test_kl_term_positive_when_ref_differs():
    new = _lps(-1.0, -1.0, requires_grad=True)
    old = _lps(-1.0, -1.0)
    ref = _lps(-0.5, -0.5)  # ref assigns higher prob to these tokens
    out = grpo_loss(new, old, ref, advantage=0.0, kl_beta=1.0)
    assert out["kl_loss"].item() > 0.0


def test_positive_advantage_with_matching_logprobs_gives_negative_policy_loss():
    """Loss is minimized when we increase prob of good trajectories;
    with ratio=1 and +adv, policy_loss = -adv (negative)."""
    new = _lps(-1.0, requires_grad=True)
    old = _lps(-1.0)
    ref = _lps(-1.0)
    out = grpo_loss(new, old, ref, advantage=1.0)
    assert out["policy_loss"].item() < 0.0


def test_negative_advantage_with_matching_logprobs_gives_positive_policy_loss():
    new = _lps(-1.0, requires_grad=True)
    old = _lps(-1.0)
    ref = _lps(-1.0)
    out = grpo_loss(new, old, ref, advantage=-1.0)
    assert out["policy_loss"].item() > 0.0


def test_clipping_bounds_ratio_with_positive_advantage():
    """If ratio blows up but advantage is positive, clip kicks in."""
    # new is much higher logprob than old -> ratio = exp(1) ~ 2.72 (way above 1.2)
    new = _lps(-1.0, requires_grad=True)
    old = _lps(-2.0)
    ref = _lps(-1.0)
    out = grpo_loss(new, old, ref, advantage=1.0, clip_eps=0.2)
    # clipped surrogate = 1.2 * adv, unclipped = 2.72 * adv
    # min(2.72, 1.2) * 1.0 = 1.2
    # policy_loss = -1.2
    assert math.isclose(out["policy_loss"].item(), -1.2, abs_tol=1e-4)
    # clip_frac should be 1.0 (the single token was clipped)
    assert math.isclose(out["clip_frac"].item(), 1.0, abs_tol=1e-6)


def test_mask_excludes_tokens():
    new = _lps(-1.0, -1.0, -1.0, requires_grad=True)
    old = _lps(-1.0, -1.0, -1.0)
    ref = _lps(-1.0, -1.0, -1.0)
    mask = torch.tensor([1.0, 0.0, 1.0])

    out = grpo_loss(new, old, ref, advantage=1.0, completion_mask=mask)
    # Only 2 tokens contribute
    assert math.isclose(out["n_effective"].item(), 2.0, abs_tol=1e-6)


def test_empty_mask_yields_zero_loss():
    new = _lps(-1.0, -1.0, requires_grad=True)
    old = _lps(-1.0, -1.0)
    ref = _lps(-1.0, -1.0)
    mask = torch.tensor([0.0, 0.0])

    out = grpo_loss(new, old, ref, advantage=1.0, completion_mask=mask)
    assert out["loss"].item() == 0.0
    # Should not error, should be safe to backprop
    # (Won't actually update anything, but shouldn't NaN either.)


def test_loss_requires_grad_and_backprops():
    new = _lps(-1.0, -2.0, requires_grad=True)
    old = _lps(-1.0, -2.0)
    ref = _lps(-1.0, -2.0)

    out = grpo_loss(new, old, ref, advantage=0.5)
    assert out["loss"].requires_grad
    out["loss"].backward()
    # Gradient should flow back to new
    assert new.grad is not None
    assert new.grad.shape == new.shape


def test_shape_mismatch_raises():
    new = _lps(-1.0, -2.0)
    old = _lps(-1.0, -2.0, -3.0)
    ref = _lps(-1.0, -2.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        grpo_loss(new, old, ref, advantage=0.0)


def test_ref_shape_mismatch_raises():
    new = _lps(-1.0, -2.0)
    old = _lps(-1.0, -2.0)
    ref = _lps(-1.0, -2.0, -3.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        grpo_loss(new, old, ref, advantage=0.0)


def test_total_loss_includes_kl():
    new = _lps(-1.0, -1.0, requires_grad=True)
    old = _lps(-1.0, -1.0)
    ref = _lps(-2.0, -2.0)
    beta = 0.5
    out = grpo_loss(new, old, ref, advantage=0.0, kl_beta=beta)
    # Policy loss is 0 (zero advantage), so total should equal beta * kl_loss
    expected = beta * out["kl_loss"].item()
    assert math.isclose(out["loss"].item(), expected, abs_tol=1e-5)


def test_returned_components_are_detached():
    new = _lps(-1.0, -1.0, requires_grad=True)
    old = _lps(-1.0, -1.0)
    ref = _lps(-1.0, -1.0)

    out = grpo_loss(new, old, ref, advantage=0.5)
    # Components we report for logging shouldn't carry grad
    assert not out["policy_loss"].requires_grad
    assert not out["kl_loss"].requires_grad
    assert not out["clip_frac"].requires_grad
    # But the total loss does
    assert out["loss"].requires_grad
