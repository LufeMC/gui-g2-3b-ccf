"""GRPO loss for multi-step cursor training.

Implements the PPO-style clipped surrogate objective with an explicit KL
penalty term (the standard RLHF formulation used by TRL, VLM-R1, and the
GUI-Cursor / GUI-G2 papers).

Inputs are per-token log-probabilities for ONE trajectory:
    logprobs_new  -- under the current (being-trained) policy, with grad
    logprobs_old  -- captured at generation time, detached (no grad)
    logprobs_ref  -- under the reference policy (LoRA disabled), detached

All three are 1D tensors of the same length T (total completion tokens
across all steps of the trajectory).

The trajectory's scalar advantage is broadcast across all T tokens, and
each token's contribution is gated by `completion_mask` so we can zero
out padding or BOS/EOS positions that the tokenizer emitted but the
policy didn't actually generate.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

DEFAULT_CLIP_EPS = 0.2
DEFAULT_KL_BETA = 0.04


def grpo_loss(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
    logprobs_ref: torch.Tensor,
    advantage: float,
    completion_mask: Optional[torch.Tensor] = None,
    clip_eps: float = DEFAULT_CLIP_EPS,
    kl_beta: float = DEFAULT_KL_BETA,
) -> Dict[str, torch.Tensor]:
    """Compute the GRPO loss for one trajectory.

    Args:
        logprobs_new: [T] log-probs under current policy (requires grad).
        logprobs_old: [T] log-probs captured at generation time (detached).
        logprobs_ref: [T] log-probs under the frozen reference policy (detached).
        advantage: scalar advantage for this trajectory, broadcast across tokens.
        completion_mask: [T] of 0/1, True where the token should contribute.
            Defaults to all ones if None.
        clip_eps: PPO clip range epsilon.
        kl_beta: weight for KL penalty.

    Returns:
        Dict with keys:
          - loss           (scalar tensor)
          - policy_loss    (scalar tensor, pre-KL)
          - kl_loss        (scalar tensor, pre-beta multiplication)
          - clip_frac      (scalar tensor, fraction of tokens clipped)
          - n_effective    (scalar tensor, sum of mask used for averaging)
    """
    if logprobs_new.shape != logprobs_old.shape:
        raise ValueError(
            f"logprob shape mismatch: new {logprobs_new.shape} vs old {logprobs_old.shape}"
        )
    if logprobs_new.shape != logprobs_ref.shape:
        raise ValueError(
            f"logprob shape mismatch: new {logprobs_new.shape} vs ref {logprobs_ref.shape}"
        )

    if completion_mask is None:
        completion_mask = torch.ones_like(logprobs_new)
    completion_mask = completion_mask.to(dtype=logprobs_new.dtype)

    n_effective = completion_mask.sum()
    if n_effective.item() == 0:
        zero = torch.zeros((), dtype=logprobs_new.dtype, device=logprobs_new.device)
        return {
            "loss": zero.clone().requires_grad_(logprobs_new.requires_grad),
            "policy_loss": zero.clone(),
            "kl_loss": zero.clone(),
            "clip_frac": zero.clone(),
            "n_effective": n_effective,
        }

    # PPO clipped surrogate
    log_ratio = logprobs_new - logprobs_old.detach()
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    # minus sign because we're MINIMIZING a loss but MAXIMIZING the surrogate.
    per_token_policy_loss = -torch.minimum(unclipped, clipped)
    policy_loss = (per_token_policy_loss * completion_mask).sum() / n_effective

    # K3 unbiased KL estimator from http://joschu.net/blog/kl-approx.html
    # D_KL(new || ref) ~= exp(ref - new) - (ref - new) - 1
    ref_minus_new = (logprobs_ref.detach() - logprobs_new)
    kl_per_token = torch.exp(ref_minus_new) - ref_minus_new - 1.0
    kl_loss = (kl_per_token * completion_mask).sum() / n_effective

    total = policy_loss + kl_beta * kl_loss

    # Fraction of tokens that would have been clipped
    was_clipped = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).to(
        completion_mask.dtype
    )
    clip_frac = (was_clipped * completion_mask).sum() / n_effective

    return {
        "loss": total,
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "clip_frac": clip_frac.detach(),
        "n_effective": n_effective.detach(),
    }
