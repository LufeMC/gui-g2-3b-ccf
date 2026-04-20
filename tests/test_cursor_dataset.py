"""Tests for the CursorDataset loader."""

import json
import os
import sys

import pytest
from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_dataset import CursorDataset, CursorSample  # noqa: E402


def _write_fixture(tmp_path, records, image_size=(100, 100)):
    """Create a JSONL with records + optional image files on disk."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    jsonl = tmp_path / "samples.jsonl"

    with open(jsonl, "w") as f:
        for i, rec in enumerate(records):
            img_name = rec.get("_img_name", f"img_{i}.png")
            if rec.pop("_make_image", True):
                Image.new("RGB", image_size, (255, 255, 255)).save(img_dir / img_name)
            rec["img_path"] = str(img_dir / img_name)
            rec.pop("_img_name", None)
            f.write(json.dumps(rec) + "\n")

    return jsonl


def test_loads_valid_records(tmp_path):
    records = [
        {"instruction": "click settings", "abs_box": [10, 20, 30, 40]},
        {"instruction": "click search", "abs_box": [50, 60, 70, 80]},
    ]
    jsonl = _write_fixture(tmp_path, records)

    ds = CursorDataset(str(jsonl))
    assert len(ds) == 2

    sample = ds[0]
    assert isinstance(sample, CursorSample)
    assert sample.instruction == "click settings"
    assert sample.target_bbox == (10, 20, 30, 40)
    assert sample.image.size == (100, 100)


def test_preserves_extra_keys_in_meta(tmp_path):
    records = [{
        "instruction": "x",
        "abs_box": [1, 2, 3, 4],
        "icon_label": "ICON_SETTINGS",
        "screen_id": "abc123",
    }]
    jsonl = _write_fixture(tmp_path, records)

    sample = CursorDataset(str(jsonl))[0]
    assert sample.meta["icon_label"] == "ICON_SETTINGS"
    assert sample.meta["screen_id"] == "abc123"
    assert "img_path" in sample.meta


def test_skips_missing_images(tmp_path):
    records = [
        {"instruction": "a", "abs_box": [1, 2, 3, 4]},
        {"instruction": "b", "abs_box": [1, 2, 3, 4], "_make_image": False},
        {"instruction": "c", "abs_box": [1, 2, 3, 4]},
    ]
    jsonl = _write_fixture(tmp_path, records)

    ds = CursorDataset(str(jsonl))
    assert len(ds) == 2
    assert ds.skipped_count == 1


def test_max_samples_truncation(tmp_path):
    records = [{"instruction": f"task {i}", "abs_box": [1, 2, 3, 4]} for i in range(5)]
    jsonl = _write_fixture(tmp_path, records)

    ds = CursorDataset(str(jsonl), max_samples=3)
    assert len(ds) == 3


def test_missing_required_field_raises(tmp_path):
    records = [{"instruction": "no bbox"}]
    jsonl = _write_fixture(tmp_path, records)
    with pytest.raises(ValueError, match="abs_box"):
        CursorDataset(str(jsonl))


def test_missing_jsonl_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CursorDataset(str(tmp_path / "nope.jsonl"))


def test_sample_returns_distinct_items(tmp_path):
    records = [{"instruction": f"t{i}", "abs_box": [1, 2, 3, 4]} for i in range(5)]
    jsonl = _write_fixture(tmp_path, records)
    ds = CursorDataset(str(jsonl))

    import random as _rand
    rng = _rand.Random(42)
    sampled = ds.sample(3, rng=rng)
    assert len(sampled) == 3
    # Distinct instructions means distinct samples
    instrs = [s.instruction for s in sampled]
    assert len(set(instrs)) == 3


def test_sample_requires_enough_records(tmp_path):
    records = [{"instruction": "a", "abs_box": [1, 2, 3, 4]}]
    jsonl = _write_fixture(tmp_path, records)
    with pytest.raises(ValueError, match="Cannot sample"):
        CursorDataset(str(jsonl)).sample(5)
