"""Integration tests for CCF wiring into the eval pipeline.

These tests exercise `make_ccf_eval_adapter` -- the thin shim between
the CCF core module and eval.py's call shape -- using a stubbed base
predictor. We never import eval.py itself (that would pull in torch /
transformers / qwen_vl_utils), so these still run cleanly on Mac.
"""

import os
import sys

from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_ccf import CCFConfig, make_ccf_eval_adapter  # noqa: E402


LARGE_IMG = (1000, 800)  # above default min_pixels_for_ccf
SMALL_IMG = (200, 200)   # below default


class _StubPredictor:
    """Mimics eval.py's `predict_gui_g2(model, processor, image, instruction)`
    -> ((x, y), raw). Returns `full_xy` on the FIRST call and `crop_xy` on
    subsequent calls -- CCF always does the coarse pass first, so we can
    test the two-stage flow without caring about image-size heuristics."""

    def __init__(self, full_xy, crop_xy, full_raw="coarse", crop_raw="refined"):
        self.full_xy = full_xy
        self.crop_xy = crop_xy
        self.full_raw = full_raw
        self.crop_raw = crop_raw
        self.calls = []  # list of (image_size, instruction)

    def __call__(self, model, processor, image, instruction):
        self.calls.append((image.size, instruction))
        if len(self.calls) == 1:
            xy, raw = self.full_xy, self.full_raw
        else:
            xy, raw = self.crop_xy, self.crop_raw
        if xy is None:
            return (None, None), raw
        return xy, raw


def _mk_image(size):
    return Image.new("RGB", size, (255, 255, 255))


def test_adapter_runs_coarse_only_on_small_image():
    stub = _StubPredictor(full_xy=(100, 100), crop_xy=(50, 50))
    adapter = make_ccf_eval_adapter(
        model=None, processor=None, base_predict_fn=stub,
        config=CCFConfig(),
    )
    (x, y), tag = adapter(None, None, _mk_image(SMALL_IMG), "click")
    assert tag == "[ccf:coarse]"
    assert x == 100.0 and y == 100.0
    # Only ONE call: the coarse one, because image is below min_pixels_for_ccf
    assert len(stub.calls) == 1
    assert stub.calls[0][0] == SMALL_IMG


def test_adapter_refines_on_large_image():
    # Coarse at (500, 400) -> crop 500x400 starting at (250, 200)
    # Refined at (10, 20) on crop -> maps to (260, 220) original
    stub = _StubPredictor(full_xy=(500, 400), crop_xy=(10, 20))
    adapter = make_ccf_eval_adapter(None, None, stub, CCFConfig(zoom_factor=2.0))
    (x, y), tag = adapter(None, None, _mk_image(LARGE_IMG), "click")
    assert tag == "[ccf:refined]"
    assert x == 260.0 and y == 220.0
    assert len(stub.calls) == 2
    # First call full image, second call crop
    assert stub.calls[0][0] == LARGE_IMG
    assert stub.calls[1][0] == (500, 400)


def test_adapter_falls_back_when_refined_fails():
    stub = _StubPredictor(full_xy=(500, 400), crop_xy=None)
    adapter = make_ccf_eval_adapter(None, None, stub, CCFConfig())
    (x, y), tag = adapter(None, None, _mk_image(LARGE_IMG), "click")
    assert tag == "[ccf:fallback]"
    assert x == 500.0 and y == 400.0
    assert len(stub.calls) == 2


def test_adapter_returns_parse_fail_when_coarse_fails():
    stub = _StubPredictor(full_xy=None, crop_xy=None)
    adapter = make_ccf_eval_adapter(None, None, stub, CCFConfig())
    (x, y), tag = adapter(None, None, _mk_image(LARGE_IMG), "click")
    assert tag == "[ccf:parse_fail]"
    assert x is None and y is None
    # Only the coarse call; no refinement attempted
    assert len(stub.calls) == 1


def test_adapter_passes_instruction_to_both_stages():
    stub = _StubPredictor(full_xy=(500, 400), crop_xy=(10, 10))
    adapter = make_ccf_eval_adapter(None, None, stub, CCFConfig())
    adapter(None, None, _mk_image(LARGE_IMG), "click the settings icon")
    assert all(call[1] == "click the settings icon" for call in stub.calls)


def test_adapter_respects_custom_zoom_factor():
    stub = _StubPredictor(full_xy=(500, 400), crop_xy=(5, 5))
    # zoom_factor=4 -> target crop 250x200 around (500, 400)
    # -> left=375, top=300, so crop(0..250, 0..200) maps (5, 5) -> (380, 305)
    adapter = make_ccf_eval_adapter(None, None, stub, CCFConfig(zoom_factor=4.0))
    (x, y), tag = adapter(None, None, _mk_image(LARGE_IMG), "click")
    assert tag == "[ccf:refined]"
    assert x == 380.0 and y == 305.0
    # Crop size matches zoom_factor=4
    assert stub.calls[1][0] == (250, 200)


def test_adapter_tag_format_is_parseable_by_eval_stats():
    """The eval loop parses "[ccf:xxx]" tags to update ccf_stats. Double-check
    that every emitted stage produces the expected prefix/suffix."""
    import re
    tag_pattern = re.compile(
        r"^\[ccf:(coarse|coarse_text_gate|refined|fallback|parse_fail)\]$"
    )

    stub = _StubPredictor(full_xy=(100, 100), crop_xy=(50, 50))
    adapter = make_ccf_eval_adapter(None, None, stub, CCFConfig())
    for img_size in (SMALL_IMG, LARGE_IMG):
        _, tag = adapter(None, None, _mk_image(img_size), "x")
        assert tag_pattern.match(tag), f"unexpected tag format: {tag!r}"


# ---- Phase 9: type-aware gate integration ---------------------------------


def test_adapter_skips_refinement_on_text_gate():
    """With the text gate enabled, a text-targeted instruction should
    trigger the gate -> only the coarse predictor call happens."""
    stub = _StubPredictor(full_xy=(500, 400), crop_xy=(10, 20))

    def stub_classifier(_instruction):
        return "text"

    cfg = CCFConfig(
        zoom_factor=2.0,
        instruction_classifier_fn=stub_classifier,
    )
    adapter = make_ccf_eval_adapter(None, None, stub, cfg)
    (x, y), tag = adapter(None, None, _mk_image(LARGE_IMG), "the page heading")
    assert tag == "[ccf:coarse_text_gate]"
    assert x == 500.0 and y == 400.0
    assert len(stub.calls) == 1  # ONLY the coarse call, refinement skipped


def test_adapter_runs_refinement_on_icon_classified_with_gate_enabled():
    """Even with the gate enabled, icon-classified instructions still refine."""
    stub = _StubPredictor(full_xy=(500, 400), crop_xy=(10, 20))

    def stub_classifier(_instruction):
        return "icon"

    cfg = CCFConfig(
        zoom_factor=2.0,
        instruction_classifier_fn=stub_classifier,
    )
    adapter = make_ccf_eval_adapter(None, None, stub, cfg)
    (x, y), tag = adapter(None, None, _mk_image(LARGE_IMG), "the menu icon")
    assert tag == "[ccf:refined]"
    assert x == 260.0 and y == 220.0  # refined coords mapped back
    assert len(stub.calls) == 2
