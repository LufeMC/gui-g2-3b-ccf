"""Mac-side tests for the VLM policy module.

These cover:
  - The module imports cleanly even without transformers installed.
  - Pure helpers (smart_resize, rescale_action_coords, build_chat_messages)
    produce the right values.
  - Instantiating VLMCursorPolicy without deps raises a clear error.

The real forward/backward paths are GPU-only and get exercised in Phase 3b.
"""

import os
import sys

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import cursor_vlm_policy as vlm_policy  # noqa: E402
from cursor_actions import parse_action  # noqa: E402


def test_module_imports_cleanly():
    """This test passing means the file imports without raising,
    even when transformers isn't installed on Mac."""
    assert hasattr(vlm_policy, "VLMCursorPolicy")
    assert hasattr(vlm_policy, "VLMPolicyConfig")
    assert hasattr(vlm_policy, "build_chat_messages")


def test_smart_resize_preserves_aspect_ratio_roughly():
    h, w = vlm_policy.smart_resize(1000, 2000)
    # Multiple of 28, within min/max pixel budget
    assert h % 28 == 0
    assert w % 28 == 0
    assert h * w <= vlm_policy.DEFAULT_MAX_PIXELS
    # Aspect ratio roughly 1:2
    assert 1.7 < w / h < 2.3


def test_smart_resize_clamps_tiny_images_up():
    h, w = vlm_policy.smart_resize(10, 10)
    assert h * w >= vlm_policy.DEFAULT_MIN_PIXELS


def test_rescale_action_coords_for_move():
    # Original image 1000x500, processed to 500x250 -> scale 2x.
    text = "<action>move(100, 50)</action>"
    out = vlm_policy.rescale_action_coords(
        text, orig_size=(1000, 500), processed_size=(500, 250)
    )
    parsed = parse_action(out)
    assert parsed.kind == "move"
    assert parsed.x == 200
    assert parsed.y == 100


def test_rescale_passes_through_stop():
    text = "<action>stop</action>"
    out = vlm_policy.rescale_action_coords(
        text, orig_size=(1000, 500), processed_size=(500, 250)
    )
    assert out == text


def test_rescale_passes_through_invalid():
    text = "garbage text"
    out = vlm_policy.rescale_action_coords(
        text, orig_size=(100, 100), processed_size=(100, 100)
    )
    assert out == text


def test_rescale_handles_zero_processed_size():
    text = "<action>move(1, 2)</action>"
    out = vlm_policy.rescale_action_coords(
        text, orig_size=(100, 100), processed_size=(0, 0)
    )
    assert out == text  # returns unchanged rather than dividing by zero


def test_build_chat_messages_shape():
    messages = vlm_policy.build_chat_messages(
        image_placeholder="<IMG>",
        instruction="click settings",
        step_index=0,
        history=[],
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_content = messages[1]["content"]
    # user content is a list with image + text
    assert len(user_content) == 2
    assert user_content[0]["type"] == "image"
    assert user_content[0]["image"] == "<IMG>"
    assert user_content[1]["type"] == "text"
    assert "click settings" in user_content[1]["text"]


def test_build_chat_messages_with_history():
    messages = vlm_policy.build_chat_messages(
        image_placeholder="X",
        instruction="click x",
        step_index=2,
        history=[(100, 200), (150, 180)],
    )
    user_text = messages[1]["content"][1]["text"]
    assert "(100, 200)" in user_text
    assert "(150, 180)" in user_text


def test_instantiation_without_deps_raises_clear_error(monkeypatch):
    """If transformers isn't importable, VLMCursorPolicy should say so clearly."""
    monkeypatch.setattr(vlm_policy, "_VLM_AVAILABLE", False)
    cfg = vlm_policy.VLMPolicyConfig(base_model_path="/nonexistent")
    with pytest.raises(ImportError, match="transformers"):
        vlm_policy.VLMCursorPolicy(cfg)


# ---- bucket_by_grid: pure logic, no torch model needed ----


def _mk_completion(thw_tuple):
    """Build a CompletionOutput with a stub replay carrying image_grid_thw."""
    import torch
    from cursor_policy import CompletionOutput
    replay = {
        "image_grid_thw": torch.tensor([list(thw_tuple)]),
    }
    return CompletionOutput(
        text="<action>stop</action>",
        token_ids=[1, 2, 3],
        logprobs_at_generation=[-1.0, -1.0, -1.0],
        replay_data=replay,
    )


def test_bucket_by_grid_groups_identical_shapes():
    a = _mk_completion((1, 30, 40))
    b = _mk_completion((1, 30, 40))
    c = _mk_completion((1, 60, 80))
    buckets = vlm_policy._bucket_by_grid([a, b, c])
    # Two distinct shapes -> two buckets
    assert len(buckets) == 2
    sizes = sorted(len(b) for b in buckets)
    assert sizes == [1, 2]


def test_bucket_by_grid_preserves_indices():
    a = _mk_completion((1, 30, 40))
    b = _mk_completion((1, 60, 80))
    c = _mk_completion((1, 30, 40))
    buckets = vlm_policy._bucket_by_grid([a, b, c])
    flat = sorted(idx for b in buckets for idx in b)
    assert flat == [0, 1, 2]


def test_bucket_by_grid_falls_back_for_missing_replay():
    from cursor_policy import CompletionOutput
    bad = CompletionOutput(
        text="<action>stop</action>",
        token_ids=[1],
        logprobs_at_generation=[-1.0],
        replay_data=None,
    )
    good = _mk_completion((1, 30, 40))
    buckets = vlm_policy._bucket_by_grid([bad, good])
    # The bad sample becomes its own singleton bucket
    assert any(b == [0] for b in buckets)
    assert any(b == [1] for b in buckets)


def test_bucket_by_grid_handles_empty_input():
    assert vlm_policy._bucket_by_grid([]) == []
