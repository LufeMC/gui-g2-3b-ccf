"""Render the 2x2 CCF-vs-baseline comparison image for the HF model card.

Reads a JSONL produced by scripts/find_demo_samples.py, filters to icon
samples where the baseline missed and CCF hit, picks 4 visually diverse
ones, and composites a single PNG with annotated crops.

This is local-only (no GPU). All it needs is the JSONL + the screenshot
files referenced by `img_filename`.
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


GT_COLOR = (59, 130, 246)      # blue-500
BASELINE_COLOR = (239, 68, 68)  # red-500
CCF_COLOR = (34, 197, 94)       # green-500
TITLE_BG = (17, 24, 39)         # gray-900
TILE_BG = (255, 255, 255)


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def filter_candidates(records: List[Dict], require_icon: bool = True) -> List[Dict]:
    out = []
    for r in records:
        if require_icon and r.get("data_type") != "icon":
            continue
        if r.get("baseline_hit"):
            continue
        if not r.get("ccf_hit"):
            continue
        if not r.get("baseline_xy") or not r.get("ccf_xy"):
            continue
        out.append(r)
    return out


def pick_diverse(records: List[Dict], n: int = 4, seed: int = 7) -> List[Dict]:
    """Spread across splits, prefer unique screenshots, prefer short
    instructions (so caption isn't truncated). Falls back to any if pool is small."""
    if len(records) <= n:
        return records[:n]
    rng = random.Random(seed)

    # Score: shorter instruction = better, unique image = better
    scored = []
    for r in records:
        instr_len = len(r.get("instruction", ""))
        score = -instr_len  # shorter wins; tie-broken by jitter below
        score += rng.random() * 0.5
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])

    chosen: List[Dict] = []
    used_imgs = set()
    used_splits: Dict[str, int] = {}

    # First pass: enforce one per split (when possible) and unique image
    for _, r in scored:
        if len(chosen) >= n:
            break
        split = r.get("split", "?")
        img = r.get("img_filename")
        if img in used_imgs:
            continue
        if used_splits.get(split, 0) >= 2:
            continue
        chosen.append(r)
        used_imgs.add(img)
        used_splits[split] = used_splits.get(split, 0) + 1

    # Top up if needed (unique-image constraint relaxed last)
    if len(chosen) < n:
        for _, r in scored:
            if len(chosen) >= n:
                break
            if r in chosen:
                continue
            if r.get("img_filename") in used_imgs:
                continue
            chosen.append(r)
            used_imgs.add(r.get("img_filename"))
    if len(chosen) < n:
        for _, r in scored:
            if len(chosen) >= n:
                break
            if r in chosen:
                continue
            chosen.append(r)

    return chosen[:n]


def _bbox_xywh_to_xyxy(b) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return (x, y, x + w, y + h)


def _try_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_x(draw: ImageDraw.ImageDraw, xy, color, size: int = 14, width: int = 4):
    x, y = xy
    halo = (255, 255, 255)
    halo_w = width + 4
    draw.line([(x - size, y - size), (x + size, y + size)], fill=halo, width=halo_w)
    draw.line([(x - size, y + size), (x + size, y - size)], fill=halo, width=halo_w)
    draw.line([(x - size, y - size), (x + size, y + size)], fill=color, width=width)
    draw.line([(x - size, y + size), (x + size, y - size)], fill=color, width=width)
    r = 5
    draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=halo)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _draw_check(draw: ImageDraw.ImageDraw, xy, color, size: int = 14, width: int = 4):
    x, y = xy
    halo = (255, 255, 255)
    halo_w = width + 4
    pts1 = [(x - size, y), (x - 2, y + size - 2)]
    pts2 = [(x - 2, y + size - 2), (x + size, y - size + 2)]
    draw.line(pts1, fill=halo, width=halo_w)
    draw.line(pts2, fill=halo, width=halo_w)
    draw.line(pts1, fill=color, width=width)
    draw.line(pts2, fill=color, width=width)
    r = 5
    draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=halo)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    xy,
    text: str,
    font: ImageFont.ImageFont,
    fill=(255, 255, 255),
    stroke=(0, 0, 0),
):
    x, y = xy
    pad = 6
    try:
        bbox = draw.textbbox((x, y), text, font=font, stroke_width=3)
    except Exception:
        bbox = (x, y, x + 8 * len(text), y + 16)
    tx0, ty0, tx1, ty1 = bbox
    draw.rectangle([tx0 - pad, ty0 - pad, tx1 + pad, ty1 + pad], fill=(0, 0, 0, 200))
    draw.text((x, y), text, fill=fill, font=font, stroke_fill=stroke, stroke_width=3)


def _legend(font: ImageFont.ImageFont, width: int) -> Image.Image:
    h = 56
    img = Image.new("RGB", (width, h), TITLE_BG)
    draw = ImageDraw.Draw(img)
    items = [
        (GT_COLOR, "Ground truth"),
        (BASELINE_COLOR, "Baseline (X)"),
        (CCF_COLOR, "CCF (check)"),
    ]
    cursor_x = 20
    y = h // 2
    swatch = 18
    gap = 12
    for color, label in items:
        draw.rectangle(
            [cursor_x, y - swatch // 2, cursor_x + swatch, y + swatch // 2],
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
        cursor_x += swatch + gap
        try:
            tb = draw.textbbox((cursor_x, y), label, font=font, anchor="lm")
        except Exception:
            tb = (cursor_x, y - 8, cursor_x + 100, y + 8)
        draw.text((cursor_x, y), label, fill=(229, 231, 235), font=font, anchor="lm")
        cursor_x = tb[2] + 24
    return img


def _pick_caption_corner(
    annotations_local: List[Tuple[float, float]],
    crop_w: int,
    crop_h: int,
) -> Tuple[Tuple[int, int], str]:
    """Pick the corner farthest from any annotation point.

    Returns (xy_anchor, alignment_hint). Alignment is "tl" | "tr" | "bl" | "br".
    """
    margin = 18
    corners = {
        "tl": (margin, margin),
        "tr": (crop_w - margin, margin),
        "bl": (margin, crop_h - margin),
        "br": (crop_w - margin, crop_h - margin),
    }
    best_key, best_score = "tl", -1.0
    for key, (cx, cy) in corners.items():
        # Maximize the minimum distance to any annotation point
        worst = min(math.hypot(cx - ax, cy - ay) for (ax, ay) in annotations_local)
        if worst > best_score:
            best_score = worst
            best_key = key
    return corners[best_key], best_key


def _draw_caption_at(
    crop: Image.Image,
    draw: ImageDraw.ImageDraw,
    anchor: Tuple[int, int],
    align: str,
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    pad_x, pad_y = 10, 6
    try:
        tb = draw.textbbox((0, 0), text, font=font, stroke_width=3)
    except Exception:
        tb = (0, 0, 8 * len(text), 18)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]

    ax, ay = anchor
    if align.endswith("r"):
        x0 = ax - tw - pad_x
    else:
        x0 = ax
    if align.startswith("b"):
        y0 = ay - th - pad_y * 2
    else:
        y0 = ay

    crop_w, crop_h = crop.size
    x0 = max(4, min(crop_w - tw - pad_x * 2 - 4, x0))
    y0 = max(4, min(crop_h - th - pad_y * 2 - 4, y0))

    draw.rectangle(
        [x0, y0, x0 + tw + pad_x * 2, y0 + th + pad_y * 2],
        fill=(0, 0, 0, 220),
    )
    draw.text((x0 + pad_x, y0 + pad_y - tb[1]), text,
              fill=(255, 255, 255), font=font,
              stroke_fill=(0, 0, 0), stroke_width=2)


def render_one(
    rec: Dict,
    image_dir: str,
    crop_size: int = 480,
    out_size: int = 480,
    font: Optional[ImageFont.ImageFont] = None,
) -> Image.Image:
    img_path = os.path.join(image_dir, rec["img_filename"])
    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    bx, by, bw, bh = rec["bbox"]
    gt_x0, gt_y0, gt_x1, gt_y1 = bx, by, bx + bw, by + bh
    bb_cx = bx + bw / 2.0
    bb_cy = by + bh / 2.0

    px = rec.get("baseline_xy") or [bb_cx, bb_cy]
    cxy = rec.get("ccf_xy") or [bb_cx, bb_cy]

    # Compute the minimum bounding box that contains GT, baseline X, and
    # CCF check, plus padding for the marker glyph + caption space.
    marker_pad = 28  # half-extent of an X / check glyph
    pts_x = [gt_x0, gt_x1, px[0] - marker_pad, px[0] + marker_pad,
             cxy[0] - marker_pad, cxy[0] + marker_pad]
    pts_y = [gt_y0, gt_y1, px[1] - marker_pad, px[1] + marker_pad,
             cxy[1] - marker_pad, cxy[1] + marker_pad]
    span_w = max(pts_x) - min(pts_x)
    span_h = max(pts_y) - min(pts_y)

    # Square side: tight to the spread plus generous breathing room for
    # surrounding UI context, but bounded by the requested crop_size and
    # original image dimensions.
    breathing = 80
    needed = max(span_w, span_h) + breathing * 2
    side = int(min(min(W, H), max(needed, crop_size * 0.55)))
    side = max(side, 200)  # never absurdly tiny

    cx_pred_avg = (min(pts_x) + max(pts_x)) / 2.0
    cy_pred_avg = (min(pts_y) + max(pts_y)) / 2.0
    half = side / 2.0
    x0 = int(max(0, min(W - side, cx_pred_avg - half)))
    y0 = int(max(0, min(H - side, cy_pred_avg - half)))
    x1 = int(x0 + side)
    y1 = int(y0 + side)
    crop = image.crop((x0, y0, x1, y1))

    if font is None:
        font = _try_font(20)

    draw = ImageDraw.Draw(crop, "RGBA")

    # Annotations in crop-local coords
    gt_local = (gt_x0 - x0, gt_y0 - y0, gt_x1 - x0, gt_y1 - y0)
    px_local = (px[0] - x0, px[1] - y0)
    cxy_local = (cxy[0] - x0, cxy[1] - y0)

    draw.rectangle(gt_local, outline=GT_COLOR, width=4)
    _draw_x(draw, px_local, BASELINE_COLOR, size=16, width=5)
    _draw_check(draw, cxy_local, CCF_COLOR, size=16, width=5)

    instr = rec.get("instruction", "").strip()
    if len(instr) > 60:
        instr = instr[:57] + "..."
    caption = f'"{instr}"'

    crop_w, crop_h = crop.size
    annotations_for_caption = [
        ((gt_local[0] + gt_local[2]) / 2.0, (gt_local[1] + gt_local[3]) / 2.0),
        px_local,
        cxy_local,
    ]
    anchor, align = _pick_caption_corner(annotations_for_caption, crop_w, crop_h)
    _draw_caption_at(crop, draw, anchor, align, caption, font)

    if crop.size != (out_size, out_size):
        crop = crop.resize((out_size, out_size), Image.LANCZOS)
    return crop


def composite_grid(
    tiles: List[Image.Image],
    title: str,
    subtitle: str,
    pad: int = 16,
    bg=(243, 244, 246),
) -> Image.Image:
    assert len(tiles) == 4
    tile_w, tile_h = tiles[0].size
    title_font = _try_font(34)
    subtitle_font = _try_font(20)
    legend_font = _try_font(20)

    grid_w = tile_w * 2 + pad * 3
    grid_h = tile_h * 2 + pad * 3
    title_h = 100
    legend_h = 56
    total_h = title_h + grid_h + legend_h
    total_w = grid_w

    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, total_w, title_h], fill=TITLE_BG)
    draw.text((24, 16), title, fill=(249, 250, 251), font=title_font)
    draw.text((24, 60), subtitle, fill=(156, 163, 175), font=subtitle_font)

    legend = _legend(legend_font, total_w)
    canvas.paste(legend, (0, title_h))

    grid_top = title_h + legend_h
    positions = [
        (pad, grid_top + pad),
        (pad * 2 + tile_w, grid_top + pad),
        (pad, grid_top + pad * 2 + tile_h),
        (pad * 2 + tile_w, grid_top + pad * 2 + tile_h),
    ]
    for tile, pos in zip(tiles, positions):
        # white frame around each tile for separation against the gray bg
        draw.rectangle(
            [pos[0] - 2, pos[1] - 2, pos[0] + tile_w + 2, pos[1] + tile_h + 2],
            outline=(229, 231, 235),
            width=2,
        )
        canvas.paste(tile, pos)

    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="JSONL from find_demo_samples.py")
    parser.add_argument("--image-dir", required=True,
                        help="Directory containing the screenshot PNGs (img_filename)")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=520,
                        help="Initial crop window in source pixels (will auto-grow if needed)")
    parser.add_argument("--tile-size", type=int, default=480,
                        help="Tile output size in pixels (each of the 4 tiles)")
    parser.add_argument("--allow-non-icon", action="store_true",
                        help="If we can't find 4 icon candidates, fall back to any data_type")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-only", action="store_true",
                        help="Print the chosen img_filenames and exit (no rendering). "
                             "Used to drive a targeted scp of just the needed screenshots.")
    args = parser.parse_args()

    print(f"Loading {args.results}")
    records = load_jsonl(args.results)
    print(f"  {len(records)} total records")

    candidates = filter_candidates(records, require_icon=True)
    print(f"  {len(candidates)} icon candidates (baseline_miss + ccf_hit)")
    if len(candidates) < args.n and args.allow_non_icon:
        candidates = filter_candidates(records, require_icon=False)
        print(f"  fallback to any data_type: {len(candidates)} candidates")
    if not candidates:
        raise SystemExit("ERROR: no baseline-miss + CCF-hit candidates available")

    chosen = pick_diverse(candidates, n=args.n, seed=args.seed)
    print(f"Picked {len(chosen)} samples:")
    for c in chosen:
        print(f"  - [{c.get('split')}/{c.get('data_type')}] "
              f"{c['img_filename']}  '{c['instruction'][:50]}'  stage={c.get('ccf_stage')}")

    if args.print_only:
        # Machine-readable: one img_filename per line on stderr-free stdout suffix
        print("---PICKS---")
        for c in chosen:
            print(c["img_filename"])
        return

    font = _try_font(20)
    tiles = [render_one(c, args.image_dir,
                        crop_size=args.crop_size, out_size=args.tile_size,
                        font=font) for c in chosen]

    grid = composite_grid(
        tiles,
        title="GUI-G2-3B + CCF: refining icons the base model misses",
        subtitle="Real ScreenSpot-v2 samples. Blue = ground truth, red X = baseline (missed), green check = CCF (hit).",
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    grid.save(args.out, optimize=True)
    print(f"Wrote {args.out}  ({grid.size[0]}x{grid.size[1]} px, "
          f"{os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
