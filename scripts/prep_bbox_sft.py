"""Build the bbox-format SFT dataset for the Phase 6 Tier 3 pivot.

Combines three sources into one JSONL where each record has the GUI-G2
native schema (bbox in absolute pixels, NOT cursor actions):

    {"img_path": "/abs/path.png",
     "instruction": "click the back icon",
     "abs_box": [x1, y1, x2, y2],
     "source": "icons"|"playwright"|"ricosca"|"widget_captioning",
     "element_type": "icon"|"text"}

Sources:
  - data/icon_mining/hard_icons.jsonl       -- 1,367 hard RICO icons
  - data/cleaned_subset/                    -- 685 Playwright web pages
  - data/os-atlas/mobile_domain/ricosca.json
  - data/os-atlas/mobile_domain/widget_captioning.json

The OS-Atlas data (RICOSCA + widget captioning) gives us a much wider
range of mobile grounding instructions than the icon-only RICO mining
produced. Combined, the dataset is ~5-7k samples vs Phase 5's 635.
That's the lever this pivot is pulling: more diverse data, native
schema, no cursor reformulation.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional


def load_icons(path: str) -> List[Dict]:
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.append({
                "img_path": r["img_path"],
                "instruction": r["instruction"],
                "abs_box": [int(v) for v in r["abs_box"]],
                "source": "icons",
                "element_type": "icon",
            })
    return out


def load_ricosca(
    json_path: str,
    rico_image_dir: str,
    max_samples: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """Load RICOSCA. bbox is normalized [x1, y1, x2, y2]; img sizes vary
    per screenshot so we open each image to convert to absolute pixels.
    """
    rng = rng or random.Random(0)
    with open(json_path) as f:
        records = json.load(f)

    rng.shuffle(records)
    out = []
    skipped = 0
    from PIL import Image

    for r in records:
        if max_samples and len(out) >= max_samples:
            break
        fname = r.get("img_filename")
        if not fname:
            skipped += 1
            continue
        img_path = os.path.join(rico_image_dir, fname)
        if not os.path.exists(img_path):
            skipped += 1
            continue
        instruction = (r.get("instruction") or "").strip()
        if not instruction or len(instruction) > 200 or "\n" in instruction:
            skipped += 1
            continue
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            skipped += 1
            continue
        bbox_norm = r.get("bbox")
        if not bbox_norm or len(bbox_norm) != 4:
            skipped += 1
            continue
        x1 = int(round(bbox_norm[0] * W))
        y1 = int(round(bbox_norm[1] * H))
        x2 = int(round(bbox_norm[2] * W))
        y2 = int(round(bbox_norm[3] * H))
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue
        out.append({
            "img_path": img_path,
            "instruction": instruction,
            "abs_box": [x1, y1, x2, y2],
            "source": "ricosca",
            "element_type": _classify_element_type(instruction),
        })

    if skipped:
        print(f"[ricosca] Skipped {skipped} records")
    return out


def load_widget_captioning(
    json_path: str,
    rico_image_dir: str,
    max_samples: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """Same schema as ricosca."""
    return load_ricosca(json_path, rico_image_dir, max_samples, rng)


def load_playwright(
    cleaned_dir: str,
    samples_per_page: int = 1,
    max_samples: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """Sample N elements per Playwright page, convert to abs_box."""
    rng = rng or random.Random(0)
    out: List[Dict] = []
    for fname in sorted(os.listdir(cleaned_dir)):
        if not fname.endswith(".json"):
            continue
        json_path = os.path.join(cleaned_dir, fname)
        png_path = json_path.replace(".json", ".png")
        if not os.path.exists(png_path):
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                page = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        viewport = page.get("viewport", {})
        vw = int(viewport.get("width", 1440))
        vh = int(viewport.get("height", 900))
        elements = page.get("elements", [])
        if not elements:
            continue

        chosen = rng.sample(elements, min(samples_per_page, len(elements)))
        for el in chosen:
            instruction = (el.get("instruction") or "").strip()
            if not instruction or len(instruction) > 200 or "\n" in instruction:
                continue
            bb = el.get("bbox")
            if not bb or "x" not in bb:
                continue
            x1 = int(round(bb["x"] * vw))
            y1 = int(round(bb["y"] * vh))
            x2 = int(round((bb["x"] + bb["w"]) * vw))
            y2 = int(round((bb["y"] + bb["h"]) * vh))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append({
                "img_path": os.path.abspath(png_path),
                "instruction": instruction,
                "abs_box": [x1, y1, x2, y2],
                "source": "playwright",
                "element_type": _classify_element_type(instruction),
            })
        if max_samples and len(out) >= max_samples:
            break
    return out


def _classify_element_type(instruction: str) -> str:
    s = instruction.lower()
    icon_terms = ("icon", "button", "logo", "image", "svg", "arrow", "menu", "close", "search button", "click", "select")
    text_terms = ("link", "text", "label", "heading", "title", "input", "field", "type")
    if any(t in s for t in icon_terms):
        return "icon"
    if any(t in s for t in text_terms):
        return "text"
    return "icon" if len(instruction) < 25 else "text"


def write_jsonl(path: str, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icons", type=str,
                        default="/workspace/data/icon_mining/hard_icons.jsonl")
    parser.add_argument("--playwright-dir", type=str,
                        default="/workspace/data/cleaned_subset")
    parser.add_argument("--ricosca", type=str,
                        default="/workspace/data/os-atlas/mobile_domain/ricosca.json")
    parser.add_argument("--widget-captioning", type=str,
                        default="/workspace/data/os-atlas/mobile_domain/widget_captioning.json")
    parser.add_argument("--rico-image-dir", type=str,
                        default="/workspace/data/os-atlas/mobile_domain/rico_images/combined")
    parser.add_argument("--ricosca-samples", type=int, default=2500)
    parser.add_argument("--widget-samples", type=int, default=1500)
    parser.add_argument("--playwright-samples-per-page", type=int, default=1)
    parser.add_argument("--out", type=str, default="/workspace/data/bbox_sft_train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading icons from {args.icons}")
    icons = load_icons(args.icons)
    print(f"  {len(icons)} icon samples")

    print(f"Loading playwright from {args.playwright_dir}")
    playwright = load_playwright(args.playwright_dir, args.playwright_samples_per_page, rng=rng)
    print(f"  {len(playwright)} playwright samples")

    print(f"Loading ricosca from {args.ricosca} (max {args.ricosca_samples})")
    ricosca = load_ricosca(args.ricosca, args.rico_image_dir, args.ricosca_samples, rng=rng)
    print(f"  {len(ricosca)} ricosca samples")

    print(f"Loading widget_captioning from {args.widget_captioning} (max {args.widget_samples})")
    widget = load_widget_captioning(args.widget_captioning, args.rico_image_dir, args.widget_samples, rng=rng)
    print(f"  {len(widget)} widget_captioning samples")

    all_records = icons + playwright + ricosca + widget
    rng.shuffle(all_records)
    print(f"Total: {len(all_records)} records")

    write_jsonl(args.out, all_records)
    print(f"Wrote {args.out}")

    # Distribution summary
    from collections import Counter
    by_source = Counter(r["source"] for r in all_records)
    by_type = Counter(r["element_type"] for r in all_records)
    print(f"By source: {dict(by_source)}")
    print(f"By type:   {dict(by_type)}")


if __name__ == "__main__":
    main()
