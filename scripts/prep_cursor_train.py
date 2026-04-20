"""Curate the cursor training/validation JSONL files for Phase 5.

Input sources:
- data/icon_mining/hard_icons.jsonl    -- 1,367 hard mobile icons (RICO)
- data/cleaned/*.json + *.png           -- Playwright synthetic web grounding

Output:
- data/cursor_train.jsonl    -- main training pool
- data/cursor_val.jsonl      -- ~50 held-out samples for in-training eval

Schema for both output files (one JSON object per line):
    {"img_path": "/abs/path/to.png",
     "instruction": "click the back navigation icon",
     "abs_box": [x1, y1, x2, y2],
     "source": "icons"|"playwright",
     "element_type": "icon"|"text"}

Notes:
- We DON'T touch ScreenSpot-v2 here -- we evaluate on its full 1,272 samples
  for marketing, so any sample we train on would contaminate the eval. The
  GUI-Cursor plan originally suggested a SS-v2 train slice, but ScreenSpot-v2
  has no separate train split.
- Playwright bboxes are normalized; we convert to absolute pixels using each
  page's viewport.
- We sample at most `playwright_per_page` elements from each Playwright page
  to keep the per-page diversity high (otherwise cluttered pages dominate).
- Stratify the val set: ~half icons, half web text, mirror the train mix.
"""

import argparse
import json
import os
import random
from typing import Dict, Iterable, List


def _abs_path(*parts: str) -> str:
    """Resolve relative paths against the gui-grounding repo root."""
    return os.path.normpath(os.path.join(*parts))


def load_icons(path: str) -> List[Dict]:
    """Load hard_icons.jsonl into our standard schema."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append({
                "img_path": r["img_path"],
                "instruction": r["instruction"],
                "abs_box": [int(v) for v in r["abs_box"]],
                "source": "icons",
                "element_type": "icon",
            })
    return out


def _normalize_bbox_to_abs(
    bbox_norm: Dict,
    viewport_w: int,
    viewport_h: int,
) -> List[int]:
    """Playwright bbox: {x, y, w, h} in 0-1 normalized -> [x1, y1, x2, y2] in pixels."""
    x = bbox_norm["x"] * viewport_w
    y = bbox_norm["y"] * viewport_h
    w = bbox_norm["w"] * viewport_w
    h = bbox_norm["h"] * viewport_h
    return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]


def _looks_like_icon(instruction: str) -> bool:
    """Cheap heuristic for tagging Playwright elements as icon vs text."""
    s = instruction.lower()
    icon_terms = ("icon", "button", "logo", "image", "svg", "arrow", "menu", "close", "search button")
    text_terms = ("link", "text", "label", "heading", "title", "input", "field")
    if any(t in s for t in icon_terms):
        return True
    if any(t in s for t in text_terms):
        return False
    # Generic fallback: short labels are usually icon-like, long labels text-like
    return len(instruction) < 25


def load_playwright(
    cleaned_dir: str,
    samples_per_page: int = 1,
    rng: random.Random = None,
) -> List[Dict]:
    """Load the Playwright synthetic dataset, sampling N elements per page."""
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
            # Some Playwright-scraped pages have non-UTF8 metadata; skip silently.
            continue

        viewport = page.get("viewport", {})
        vw = int(viewport.get("width", 1440))
        vh = int(viewport.get("height", 900))
        elements = page.get("elements", [])
        if not elements:
            continue

        # Sample without replacement up to samples_per_page elements.
        k = min(samples_per_page, len(elements))
        chosen = rng.sample(elements, k)
        for el in chosen:
            instruction = el.get("instruction", "").strip()
            if not instruction:
                continue
            # Some Playwright instructions have embedded newlines / massive
            # truncated SVG paths; skip the obviously broken ones.
            if len(instruction) > 200 or "\n" in instruction:
                continue
            try:
                abs_box = _normalize_bbox_to_abs(el["bbox"], vw, vh)
            except (KeyError, TypeError):
                continue
            # Sanity-check bbox is non-degenerate
            x1, y1, x2, y2 = abs_box
            if x2 <= x1 or y2 <= y1:
                continue
            out.append({
                "img_path": os.path.abspath(png_path),
                "instruction": instruction,
                "abs_box": abs_box,
                "source": "playwright",
                "element_type": "icon" if _looks_like_icon(instruction) else "text",
            })

    return out


def stratified_split(
    records: List[Dict],
    val_size: int,
    rng: random.Random,
) -> tuple:
    """Carve out a stratified val set without replacement.

    We stratify by (source, element_type) so the val set mirrors the train
    distribution. Returns (train_records, val_records).
    """
    by_strat: Dict[tuple, List[Dict]] = {}
    for r in records:
        key = (r["source"], r["element_type"])
        by_strat.setdefault(key, []).append(r)

    val_records: List[Dict] = []
    train_records: List[Dict] = []

    total = sum(len(v) for v in by_strat.values())
    for key, group in by_strat.items():
        rng.shuffle(group)
        # Allocate val proportionally; round at 1 so every stratum gets at
        # least one val sample.
        n_val = max(1, int(round(val_size * len(group) / total)))
        n_val = min(n_val, len(group))
        val_records.extend(group[:n_val])
        train_records.extend(group[n_val:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    # If we over-allocated to val, trim back to val_size.
    if len(val_records) > val_size:
        train_records.extend(val_records[val_size:])
        val_records = val_records[:val_size]
    return train_records, val_records


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icons", type=str,
                        default="data/icon_mining/hard_icons.jsonl")
    parser.add_argument("--playwright-dir", type=str, default="data/cleaned")
    parser.add_argument("--samples-per-page", type=int, default=1,
                        help="How many elements to pull from each Playwright page.")
    parser.add_argument("--max-playwright", type=int, default=None,
                        help="Cap on Playwright samples (after per-page sampling).")
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--out-train", type=str, default="data/cursor_train.jsonl")
    parser.add_argument("--out-val", type=str, default="data/cursor_val.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    icons = load_icons(args.icons)
    print(f"Loaded {len(icons)} icon samples from {args.icons}")

    playwright = load_playwright(
        args.playwright_dir,
        samples_per_page=args.samples_per_page,
        rng=rng,
    )
    if args.max_playwright and len(playwright) > args.max_playwright:
        rng.shuffle(playwright)
        playwright = playwright[:args.max_playwright]
    print(f"Loaded {len(playwright)} playwright samples from {args.playwright_dir}")

    all_records = icons + playwright
    print(f"Total before split: {len(all_records)}")

    train_records, val_records = stratified_split(
        all_records, val_size=args.val_size, rng=rng,
    )
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    write_jsonl(args.out_train, train_records)
    write_jsonl(args.out_val, val_records)

    # Distribution summary
    def _counts(rs):
        c = {}
        for r in rs:
            k = (r["source"], r["element_type"])
            c[k] = c.get(k, 0) + 1
        return c

    print(f"Train distribution: {_counts(train_records)}")
    print(f"Val distribution:   {_counts(val_records)}")
    print(f"Wrote {args.out_train} and {args.out_val}")


if __name__ == "__main__":
    main()
