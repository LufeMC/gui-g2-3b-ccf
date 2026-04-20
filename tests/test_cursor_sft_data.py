"""Tests for the SFT warmstart data module.

Covers the pure-logic helpers: bbox center, action formatting,
processed-space rescaling, JSONL loading with bad records, and the
batch iterator. The trainer (train_cursor_sft.py) is GPU-only and
exercised in the actual SFT run.
"""

import json
import os
import sys

import pytest
from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_actions import parse_action  # noqa: E402
from cursor_sft_data import (  # noqa: E402
    SFTExample,
    STOP_ACTION,
    bbox_center,
    format_move_action,
    gold_completion_in_processed_space,
    iter_sft_batches,
    load_sft_jsonl,
    make_sft_example,
)


def _mk_white(w=200, h=200):
    return Image.new("RGB", (w, h), (255, 255, 255))


def test_bbox_center_simple():
    assert bbox_center((0, 0, 100, 200)) == (50, 100)


def test_bbox_center_floor():
    # (1+4)//2 = 2, (3+8)//2 = 5
    assert bbox_center((1, 3, 4, 8)) == (2, 5)


def test_format_move_action_roundtrips_via_parser():
    action = format_move_action(123, 456)
    parsed = parse_action(action)
    assert parsed.kind == "move"
    assert parsed.x == 123
    assert parsed.y == 456


def test_format_move_action_handles_floats_via_int_cast():
    action = format_move_action(123.7, 456.2)
    parsed = parse_action(action)
    assert parsed.x == 123
    assert parsed.y == 456


def test_stop_action_parses():
    parsed = parse_action(STOP_ACTION)
    assert parsed.kind == "stop"


def test_make_sft_example_basic():
    img = _mk_white(400, 300)
    ex = make_sft_example(img, "click x", (50, 50, 150, 150))
    assert isinstance(ex, SFTExample)
    assert ex.target_xy == (100, 100)
    assert ex.image_size == (400, 300)
    assert ex.instruction == "click x"


def test_gold_completion_no_resize_means_identity():
    img = _mk_white(400, 300)
    ex = make_sft_example(img, "click x", (40, 60, 60, 80))
    # bbox center is (50, 70). With proc_size == orig_size, no rescaling.
    completion = gold_completion_in_processed_space(ex, processed_size=(400, 300))
    parsed = parse_action(completion)
    assert parsed.kind == "move"
    assert parsed.x == 50
    assert parsed.y == 70


def test_gold_completion_scales_when_processor_resizes():
    img = _mk_white(400, 300)
    ex = make_sft_example(img, "click x", (40, 60, 60, 80))
    # Proc image is half the size in each dim -> coords halve.
    completion = gold_completion_in_processed_space(ex, processed_size=(200, 150))
    parsed = parse_action(completion)
    assert parsed.x == 25
    assert parsed.y == 35


def test_gold_completion_rejects_zero_processed_size():
    img = _mk_white(400, 300)
    ex = make_sft_example(img, "click x", (0, 0, 10, 10))
    with pytest.raises(ValueError):
        gold_completion_in_processed_space(ex, processed_size=(0, 100))


def test_load_sft_jsonl_skips_missing_image(tmp_path):
    p = tmp_path / "data.jsonl"
    img_real = tmp_path / "real.png"
    _mk_white(50, 50).save(img_real)
    with open(p, "w") as f:
        f.write(json.dumps({
            "img_path": str(img_real),
            "instruction": "ok",
            "abs_box": [0, 0, 10, 10],
        }) + "\n")
        f.write(json.dumps({
            "img_path": str(tmp_path / "missing.png"),
            "instruction": "skip me",
            "abs_box": [0, 0, 10, 10],
        }) + "\n")
    examples = load_sft_jsonl(str(p))
    assert len(examples) == 1
    assert examples[0].instruction == "ok"


def test_load_sft_jsonl_skips_degenerate_bbox(tmp_path):
    p = tmp_path / "data.jsonl"
    img = tmp_path / "img.png"
    _mk_white(50, 50).save(img)
    with open(p, "w") as f:
        f.write(json.dumps({
            "img_path": str(img),
            "instruction": "ok",
            "abs_box": [0, 0, 10, 10],
        }) + "\n")
        f.write(json.dumps({
            "img_path": str(img),
            "instruction": "zero w",
            "abs_box": [10, 10, 10, 20],
        }) + "\n")
        f.write(json.dumps({
            "img_path": str(img),
            "instruction": "zero h",
            "abs_box": [10, 10, 20, 10],
        }) + "\n")
    examples = load_sft_jsonl(str(p))
    assert len(examples) == 1


def test_load_sft_jsonl_skips_missing_fields(tmp_path):
    p = tmp_path / "data.jsonl"
    img = tmp_path / "img.png"
    _mk_white(50, 50).save(img)
    with open(p, "w") as f:
        f.write(json.dumps({
            "img_path": str(img),
            "instruction": "ok",
            "abs_box": [0, 0, 10, 10],
        }) + "\n")
        f.write(json.dumps({"img_path": str(img), "abs_box": [0, 0, 1, 1]}) + "\n")
        f.write(json.dumps({"instruction": "no img", "abs_box": [0, 0, 1, 1]}) + "\n")
        f.write(json.dumps({"img_path": str(img), "instruction": "no bbox"}) + "\n")
    examples = load_sft_jsonl(str(p))
    assert len(examples) == 1


def test_load_sft_jsonl_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_sft_jsonl("/nonexistent.jsonl")


def test_iter_sft_batches_partial_last_batch():
    examples = [
        make_sft_example(_mk_white(), f"x{i}", (0, 0, 10, 10))
        for i in range(7)
    ]
    batches = list(iter_sft_batches(examples, batch_size=3))
    assert [len(b) for b in batches] == [3, 3, 1]


def test_iter_sft_batches_exact_division():
    examples = [
        make_sft_example(_mk_white(), f"x{i}", (0, 0, 10, 10))
        for i in range(6)
    ]
    batches = list(iter_sft_batches(examples, batch_size=3))
    assert [len(b) for b in batches] == [3, 3]


def test_iter_sft_batches_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        list(iter_sft_batches([], batch_size=0))


def test_iter_sft_batches_handles_empty():
    assert list(iter_sft_batches([], batch_size=4)) == []
