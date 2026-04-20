"""Smoke tests for scripts/prep_cursor_train.py helpers.

These are pure-logic tests (bbox math, stratification, instruction
classification). The full script requires RICO + Playwright data which
only exists on the GPU pod, so we don't run main() here.
"""

import json
import os
import random
import sys

import pytest
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import prep_cursor_train as prep  # noqa: E402


def test_normalize_bbox_to_abs_basic():
    bbox = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    result = prep._normalize_bbox_to_abs(bbox, viewport_w=1000, viewport_h=500)
    assert result == [100, 100, 400, 300]


def test_normalize_bbox_to_abs_rounds_correctly():
    bbox = {"x": 0.1234, "y": 0.5678, "w": 0.0321, "h": 0.0654}
    result = prep._normalize_bbox_to_abs(bbox, viewport_w=1440, viewport_h=900)
    assert all(isinstance(v, int) for v in result)
    assert result[2] > result[0] and result[3] > result[1]


def test_looks_like_icon_obvious_cases():
    assert prep._looks_like_icon("the close icon") is True
    assert prep._looks_like_icon("the menu button") is True
    assert prep._looks_like_icon("the 'About us' link") is False
    assert prep._looks_like_icon("the page heading") is False


def test_looks_like_icon_short_default():
    # Short generic strings -> icon
    assert prep._looks_like_icon("the X") is True


def test_looks_like_icon_long_default():
    # Long descriptive strings without obvious cues -> text
    long = "the very long descriptive element that goes on and on and on"
    assert prep._looks_like_icon(long) is False


def test_load_icons_parses_jsonl(tmp_path):
    p = tmp_path / "icons.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({
            "img_path": "/some/path.jpg",
            "instruction": "the back icon",
            "icon_label": "ICON_BACK",
            "abs_box": [10, 20, 30, 40],
        }) + "\n")
        f.write(json.dumps({
            "img_path": "/x.png",
            "instruction": "the menu",
            "icon_label": "ICON_MENU",
            "abs_box": [5, 5, 15, 15],
        }) + "\n")
    records = prep.load_icons(str(p))
    assert len(records) == 2
    assert records[0]["source"] == "icons"
    assert records[0]["element_type"] == "icon"
    assert records[0]["abs_box"] == [10, 20, 30, 40]


def test_load_playwright_skips_missing_png(tmp_path):
    json_path = tmp_path / "page1.json"
    with open(json_path, "w") as f:
        json.dump({
            "viewport": {"width": 1000, "height": 800},
            "elements": [
                {"instruction": "the X icon", "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}},
            ],
        }, f)
    # No corresponding PNG -> page is skipped
    records = prep.load_playwright(str(tmp_path), samples_per_page=1)
    assert records == []


def test_load_playwright_loads_valid_page(tmp_path):
    page_id = "page1"
    json_path = tmp_path / f"{page_id}.json"
    png_path = tmp_path / f"{page_id}.png"
    Image.new("RGB", (1000, 800), (255, 255, 255)).save(png_path)
    with open(json_path, "w") as f:
        json.dump({
            "viewport": {"width": 1000, "height": 800},
            "elements": [
                {
                    "instruction": "the search button",
                    "bbox": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.05},
                },
                {
                    "instruction": "the about link",
                    "bbox": {"x": 0.7, "y": 0.5, "w": 0.1, "h": 0.05},
                },
            ],
        }, f)
    records = prep.load_playwright(str(tmp_path), samples_per_page=2)
    assert len(records) == 2
    for r in records:
        assert r["source"] == "playwright"
        assert os.path.exists(r["img_path"])
        assert len(r["abs_box"]) == 4


def test_load_playwright_skips_degenerate_bbox(tmp_path):
    page_id = "page1"
    json_path = tmp_path / f"{page_id}.json"
    png_path = tmp_path / f"{page_id}.png"
    Image.new("RGB", (1000, 800), (255, 255, 255)).save(png_path)
    with open(json_path, "w") as f:
        json.dump({
            "viewport": {"width": 1000, "height": 800},
            "elements": [
                {"instruction": "zero-size", "bbox": {"x": 0.5, "y": 0.5, "w": 0, "h": 0}},
                {"instruction": "valid one", "bbox": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.05}},
            ],
        }, f)
    records = prep.load_playwright(str(tmp_path), samples_per_page=10)
    # Only the valid one survives
    assert len(records) == 1
    assert records[0]["instruction"] == "valid one"


def test_load_playwright_skips_long_or_multiline_instructions(tmp_path):
    page_id = "page1"
    json_path = tmp_path / f"{page_id}.json"
    png_path = tmp_path / f"{page_id}.png"
    Image.new("RGB", (1000, 800), (255, 255, 255)).save(png_path)
    with open(json_path, "w") as f:
        json.dump({
            "viewport": {"width": 1000, "height": 800},
            "elements": [
                {
                    "instruction": "x" * 250,  # too long
                    "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
                {
                    "instruction": "has\nnewline",
                    "bbox": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1},
                },
                {
                    "instruction": "good one",
                    "bbox": {"x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1},
                },
            ],
        }, f)
    records = prep.load_playwright(str(tmp_path), samples_per_page=10)
    assert len(records) == 1
    assert records[0]["instruction"] == "good one"


def test_stratified_split_respects_val_size():
    rng = random.Random(0)
    records = [
        {"source": "icons", "element_type": "icon"} for _ in range(60)
    ] + [
        {"source": "playwright", "element_type": "icon"} for _ in range(20)
    ] + [
        {"source": "playwright", "element_type": "text"} for _ in range(20)
    ]
    train, val = prep.stratified_split(records, val_size=10, rng=rng)
    assert len(val) == 10
    assert len(train) + len(val) == 100


def test_stratified_split_keeps_at_least_one_per_stratum():
    """Even small strata get represented in val."""
    rng = random.Random(0)
    records = [
        {"source": "icons", "element_type": "icon"} for _ in range(100)
    ] + [
        {"source": "playwright", "element_type": "text"} for _ in range(5)
    ]
    train, val = prep.stratified_split(records, val_size=10, rng=rng)
    val_strata = set((r["source"], r["element_type"]) for r in val)
    assert ("icons", "icon") in val_strata
    assert ("playwright", "text") in val_strata


def test_write_jsonl_roundtrips(tmp_path):
    records = [
        {"img_path": "/a.png", "instruction": "x", "abs_box": [1, 2, 3, 4]},
        {"img_path": "/b.png", "instruction": "y", "abs_box": [5, 6, 7, 8]},
    ]
    out = tmp_path / "out.jsonl"
    prep.write_jsonl(str(out), records)
    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["instruction"] == "x"
