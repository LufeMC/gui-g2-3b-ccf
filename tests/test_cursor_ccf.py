"""Tests for the CCF inference wrapper."""

import os
import sys

import pytest
from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_ccf import (  # noqa: E402
    CCFConfig,
    CCFResult,
    ccf_predict_bbox,
    classify_instruction,
    compute_crop_window,
    map_crop_to_orig,
)


IMAGE_W, IMAGE_H = 1000, 800


# ---- compute_crop_window ---------------------------------------------------


def test_crop_window_centered_on_point():
    left, top, right, bottom = compute_crop_window(
        center=(500, 400), image_size=(IMAGE_W, IMAGE_H), zoom_factor=2.0,
    )
    # 2x zoom -> target 500x400 around (500, 400)
    assert right - left == 500
    assert bottom - top == 400
    assert left == 250
    assert top == 200


def test_crop_window_shifts_at_left_edge():
    left, top, right, bottom = compute_crop_window(
        center=(5, 400), image_size=(IMAGE_W, IMAGE_H), zoom_factor=2.0,
    )
    # Crop cannot go negative -> shifted so left == 0
    assert left == 0
    assert right - left == 500  # target width preserved
    # Still centered vertically
    assert top == 200


def test_crop_window_shifts_at_top_edge():
    left, top, right, bottom = compute_crop_window(
        center=(500, 10), image_size=(IMAGE_W, IMAGE_H), zoom_factor=2.0,
    )
    assert top == 0
    assert bottom - top == 400  # preserved


def test_crop_window_shifts_at_bottom_right_corner():
    left, top, right, bottom = compute_crop_window(
        center=(995, 795), image_size=(IMAGE_W, IMAGE_H), zoom_factor=2.0,
    )
    # Shifted to keep full size inside image
    assert right == IMAGE_W
    assert bottom == IMAGE_H
    assert right - left == 500
    assert bottom - top == 400


def test_crop_window_respects_min_side():
    """Tiny zoom factor on tiny image still gives at least min_crop_side."""
    left, top, right, bottom = compute_crop_window(
        center=(50, 50),
        image_size=(80, 80),  # smaller than min_crop_side
        zoom_factor=10.0,
        min_crop_side=112,
    )
    # Clamped to image bounds but never smaller than image itself
    assert right - left == 80
    assert bottom - top == 80


def test_crop_window_small_image_clamps_to_image_size():
    left, top, right, bottom = compute_crop_window(
        center=(200, 200),
        image_size=(400, 400),
        zoom_factor=1.0,  # target would be full image
    )
    assert left == 0 and top == 0
    assert right == 400 and bottom == 400


def test_crop_window_rejects_invalid_image_size():
    with pytest.raises(ValueError):
        compute_crop_window((10, 10), (0, 100), zoom_factor=2.0)


def test_crop_window_rejects_nonpositive_zoom():
    with pytest.raises(ValueError):
        compute_crop_window((10, 10), (100, 100), zoom_factor=0.0)


# ---- map_crop_to_orig ------------------------------------------------------


def test_map_crop_to_orig_identity():
    x, y = map_crop_to_orig(42.0, 17.0, (0, 0, 200, 100))
    assert x == 42.0
    assert y == 17.0


def test_map_crop_to_orig_with_offset():
    x, y = map_crop_to_orig(10.0, 20.0, (100, 200, 300, 400))
    assert x == 110.0
    assert y == 220.0


# ---- ccf_predict_bbox ------------------------------------------------------


def _make_large_image():
    # Above default min_pixels_for_ccf (200K)
    return Image.new("RGB", (IMAGE_W, IMAGE_H), (255, 255, 255))


def _make_small_image():
    return Image.new("RGB", (200, 200), (255, 255, 255))


def test_ccf_returns_none_when_coarse_fails():
    def predict(img, instr):
        return None, "garbage"

    result = ccf_predict_bbox(predict, _make_large_image(), "x", CCFConfig())
    assert result is None


def test_ccf_skips_crop_on_small_image():
    def predict(img, instr):
        return (100.0, 100.0), "ok"

    result = ccf_predict_bbox(predict, _make_small_image(), "x", CCFConfig())
    assert result is not None
    assert result.stage == "coarse"
    assert result.crop_window is None
    assert result.x == 100.0 and result.y == 100.0


def test_ccf_uses_refined_when_valid():
    """Mock predict returns different xy on full image vs crop.

    First call: full image, returns (500, 400).
    Second call: crop of size 500x400 starting at (250, 200), returns (10, 20)
    in crop space, which maps back to (260, 220) in original space.
    """
    calls = []

    def predict(img, instr):
        calls.append(img.size)
        if len(calls) == 1:
            return (500.0, 400.0), "coarse"
        return (10.0, 20.0), "refined"

    result = ccf_predict_bbox(
        predict, _make_large_image(), "x", CCFConfig(zoom_factor=2.0)
    )
    assert result is not None
    assert result.stage == "refined"
    assert result.coarse_xy == (500.0, 400.0)
    # crop maps (10, 20) -> (250 + 10, 200 + 20) = (260, 220)
    assert result.x == 260.0
    assert result.y == 220.0
    # Predict was called twice: once on full, once on crop
    assert len(calls) == 2
    assert calls[0] == (IMAGE_W, IMAGE_H)
    assert calls[1] == (500, 400)  # crop dimensions


def test_ccf_falls_back_on_invalid_refined():
    call_count = [0]

    def predict(img, instr):
        call_count[0] += 1
        if call_count[0] == 1:
            return (500.0, 400.0), "coarse"
        return None, "bad refined"

    result = ccf_predict_bbox(predict, _make_large_image(), "x", CCFConfig())
    assert result is not None
    assert result.stage == "fallback"
    assert result.x == 500.0  # coarse retained
    assert result.y == 400.0
    assert result.crop_window is not None  # we did attempt the crop


def test_ccf_returns_none_on_invalid_refined_when_fallback_disabled():
    call_count = [0]

    def predict(img, instr):
        call_count[0] += 1
        if call_count[0] == 1:
            return (500.0, 400.0), "coarse"
        return None, "bad"

    config = CCFConfig(fallback_to_coarse_on_invalid=False)
    result = ccf_predict_bbox(predict, _make_large_image(), "x", config)
    assert result is None


def test_ccf_result_has_expected_fields():
    def predict(img, instr):
        return (100.0, 100.0), "ok"

    result = ccf_predict_bbox(predict, _make_small_image(), "x", CCFConfig())
    assert isinstance(result, CCFResult)
    assert hasattr(result, "x")
    assert hasattr(result, "y")
    assert hasattr(result, "stage")
    assert hasattr(result, "coarse_xy")
    assert hasattr(result, "crop_window")


# ---- Phase 9: type-aware classifier + gated CCF ----------------------------


def test_classify_instruction_icon_keywords():
    assert classify_instruction("the close icon") == "icon"
    assert classify_instruction("the menu button") == "icon"
    assert classify_instruction("the back arrow") == "icon"
    assert classify_instruction("settings icon at top right") == "icon"


def test_classify_instruction_text_keywords():
    assert classify_instruction("click the address bar") == "text"
    assert classify_instruction("the search field") == "text"
    assert classify_instruction("the page heading") == "text"
    assert classify_instruction("the username input") == "text"


def test_classify_instruction_quoted_strings_force_text():
    # Quoted labels almost always reference text by exact match, even
    # when the surrounding sentence contains an icon-keyword like "button".
    assert classify_instruction('the "Submit" button') == "text"
    assert classify_instruction("click 'Sign in'") == "text"


def test_classify_instruction_short_default_is_icon():
    # Short instructions with no keyword hits default to icon (heuristic
    # that worked across SS-v2 manual inspection: short = visual element).
    assert classify_instruction("X") == "icon"
    assert classify_instruction("the gear") == "icon"


def test_classify_instruction_long_default_is_ambiguous():
    # Long instruction with no keyword in either list -> ambiguous (so we
    # default to refining, since unsure-about-text isn't grounds to skip).
    long_instr = (
        "the element near the corner that the user interacts with often"
    )
    assert classify_instruction(long_instr) == "ambiguous"


def test_classify_instruction_handles_empty_string():
    assert classify_instruction("") == "ambiguous"


def test_classify_instruction_keyword_count_breaks_tie():
    # Two text keywords vs one icon keyword -> text wins
    s = "the search field heading button"  # 2 text (field, heading) vs 1 icon (button)
    assert classify_instruction(s) == "text"


def test_ccf_skips_refinement_on_text_classification():
    """When the classifier returns 'text', refined predictor must NOT be called."""
    calls = []

    def predict(img, instr):
        calls.append(img.size)
        # Coarse pass returns a valid point. If refinement runs, calls will be 2.
        return (500.0, 400.0), "coarse"

    def stub_classifier(_instruction):
        return "text"

    cfg = CCFConfig(
        zoom_factor=2.0,
        instruction_classifier_fn=stub_classifier,
    )
    result = ccf_predict_bbox(predict, _make_large_image(), "the page heading", cfg)
    assert result is not None
    assert result.stage == "coarse_text_gate"
    assert result.x == 500.0 and result.y == 400.0
    # Only ONE predictor call: coarse. No refined pass because gate fired.
    assert len(calls) == 1


def test_ccf_runs_refinement_on_icon_classification():
    """Classifier returns 'icon' -> normal CCF flow with refined call."""
    call_count = [0]

    def predict(img, instr):
        call_count[0] += 1
        if call_count[0] == 1:
            return (500.0, 400.0), "coarse"
        return (10.0, 20.0), "refined"

    def stub_classifier(_instruction):
        return "icon"

    cfg = CCFConfig(
        zoom_factor=2.0,
        instruction_classifier_fn=stub_classifier,
    )
    result = ccf_predict_bbox(predict, _make_large_image(), "the close icon", cfg)
    assert result is not None
    assert result.stage == "refined"
    assert call_count[0] == 2


def test_ccf_runs_refinement_on_ambiguous_classification():
    """Ambiguous targets get the benefit of the doubt -> refine."""
    call_count = [0]

    def predict(img, instr):
        call_count[0] += 1
        return ((500.0, 400.0) if call_count[0] == 1 else (10.0, 20.0)), "ok"

    def stub_classifier(_instruction):
        return "ambiguous"

    cfg = CCFConfig(
        zoom_factor=2.0,
        instruction_classifier_fn=stub_classifier,
    )
    result = ccf_predict_bbox(predict, _make_large_image(), "x", cfg)
    assert result is not None
    assert result.stage == "refined"
    assert call_count[0] == 2


def test_ccf_no_classifier_preserves_phase4_behavior():
    """instruction_classifier_fn=None is the Phase 4 default: always refine."""
    call_count = [0]

    def predict(img, instr):
        call_count[0] += 1
        return ((500.0, 400.0) if call_count[0] == 1 else (10.0, 20.0)), "ok"

    cfg = CCFConfig(zoom_factor=2.0)  # no classifier
    result = ccf_predict_bbox(predict, _make_large_image(), "the page heading", cfg)
    assert result is not None
    assert result.stage == "refined"
    assert call_count[0] == 2


# ---- ccf_predict_cursor (integration with MockLogprobPolicy) ---------------


def test_ccf_cursor_wrapper_basic():
    from cursor_ccf_cursor import ccf_predict_cursor
    from cursor_policy import MockLogprobPolicy

    # Two trajectories' worth of scripts: coarse stage and refined stage.
    # Both move to (100, 100) and stop. On a 200x200 image, CCF skips the
    # crop (below min_pixels_for_ccf), so only the coarse script runs.
    scripts = [
        ["<action>move(100, 100)</action>", "<action>stop</action>"],
        ["<action>move(50, 50)</action>", "<action>stop</action>"],
    ]
    policy = MockLogprobPolicy(scripts)
    image = Image.new("RGB", (200, 200), (255, 255, 255))
    result = ccf_predict_cursor(policy, image, "click", CCFConfig())

    assert result is not None
    assert result.stage == "coarse"
    assert result.x == 100.0 and result.y == 100.0


def test_ccf_cursor_uses_refined_on_large_image():
    from cursor_ccf_cursor import ccf_predict_cursor
    from cursor_policy import MockLogprobPolicy

    # Large image -> CCF engages. Coarse at center, refined at crop origin.
    # With zoom=2.0, crop at (500,400)->(250,200,750,600). Refined moves to
    # (10, 20) in crop space -> (260, 220) in original.
    scripts = [
        ["<action>move(500, 400)</action>", "<action>stop</action>"],
        ["<action>move(10, 20)</action>", "<action>stop</action>"],
    ]
    policy = MockLogprobPolicy(scripts)
    image = Image.new("RGB", (IMAGE_W, IMAGE_H), (255, 255, 255))
    result = ccf_predict_cursor(policy, image, "click", CCFConfig(zoom_factor=2.0))

    assert result is not None
    assert result.stage == "refined"
    assert result.coarse_xy == (500.0, 400.0)
    assert result.x == 260.0
    assert result.y == 220.0


def test_ccf_cursor_returns_none_when_coarse_never_moves():
    from cursor_ccf_cursor import ccf_predict_cursor
    from cursor_policy import MockLogprobPolicy

    # Policy stops immediately with no move -> no cursor position recorded
    scripts = [["<action>stop</action>"]]
    policy = MockLogprobPolicy(scripts)
    image = Image.new("RGB", (IMAGE_W, IMAGE_H), (255, 255, 255))
    result = ccf_predict_cursor(policy, image, "click", CCFConfig())
    assert result is None
