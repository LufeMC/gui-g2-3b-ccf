"""Tests for MockLogprobPolicy and CompletionOutput (Phase 3)."""

import os
import sys

import pytest
import torch
from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_policy import (  # noqa: E402
    CompletionOutput,
    MockLogprobPolicy,
)


@pytest.fixture
def blank_image():
    return Image.new("RGB", (100, 100), (255, 255, 255))


def test_generate_returns_scripted_text(blank_image):
    policy = MockLogprobPolicy([["<action>stop</action>"]])
    text = policy.generate(blank_image, "instr", step_index=0)
    assert text == "<action>stop</action>"


def test_generate_with_logprobs_returns_completion_output(blank_image):
    policy = MockLogprobPolicy([["hello"]])
    out = policy.generate_with_logprobs(blank_image, "instr", step_index=0)
    assert isinstance(out, CompletionOutput)
    assert out.text == "hello"
    assert len(out.token_ids) == 5  # one "token" per char
    assert len(out.logprobs_at_generation) == 5


def test_teacher_force_attaches_grad(blank_image):
    policy = MockLogprobPolicy([["abc"]])
    comp = policy.generate_with_logprobs(blank_image, "i", 0)
    lps = policy.teacher_force_logprobs([comp])
    assert len(lps) == 1
    assert lps[0].shape[0] == 3
    assert lps[0].requires_grad


def test_teacher_force_ref_only_does_not_require_grad(blank_image):
    policy = MockLogprobPolicy([["abc"]])
    comp = policy.generate_with_logprobs(blank_image, "i", 0)
    lps_ref = policy.teacher_force_logprobs([comp], ref_only=True)
    assert not lps_ref[0].requires_grad


def test_advancing_trajectories_rotates_scripts(blank_image):
    scripts = [["first"], ["second"]]
    policy = MockLogprobPolicy(scripts)

    out1 = policy.generate_with_logprobs(blank_image, "i", 0)
    assert out1.text == "first"

    policy.next_trajectory()
    out2 = policy.generate_with_logprobs(blank_image, "i", 0)
    assert out2.text == "second"


def test_fallback_when_script_exhausted(blank_image):
    policy = MockLogprobPolicy([["only one"]], fallback="<action>stop</action>")
    policy.generate_with_logprobs(blank_image, "i", 0)
    # Second call on the same trajectory exhausts the script
    out = policy.generate_with_logprobs(blank_image, "i", 1)
    assert out.text == "<action>stop</action>"


def test_parameters_includes_offset():
    policy = MockLogprobPolicy([["x"]])
    params = list(policy.parameters())
    assert len(params) == 1
    assert params[0] is policy.offset


def test_loss_backprops_to_offset(blank_image):
    """End-to-end-ish: generate, teacher-force, compute loss, backprop."""
    from cursor_loss import grpo_loss

    policy = MockLogprobPolicy([["hi"]])
    comp = policy.generate_with_logprobs(blank_image, "i", 0)

    old_lps = torch.tensor(comp.logprobs_at_generation, dtype=torch.float32)
    new_lps = policy.teacher_force_logprobs([comp])[0]
    ref_lps = policy.teacher_force_logprobs([comp], ref_only=True)[0]

    result = grpo_loss(new_lps, old_lps, ref_lps, advantage=1.0)
    result["loss"].backward()

    assert policy.offset.grad is not None
    assert policy.offset.grad.abs().item() > 0
