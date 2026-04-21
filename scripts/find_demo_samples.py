"""Find ScreenSpot-v2 samples where GUI-G2-3B baseline misses but CCF hits.

For each sample we run:
  1. Baseline: a single greedy forward pass via predict_gui_g2 from src/eval.py
  2. CCF: ccf_predict_bbox wrapping the same predictor (with coarse downsizing,
     same config we ship)

We log per-sample: the predicted (x, y) for both modes, whether each landed
inside the ground-truth bbox, the CCF stage (coarse / refined / fallback /
text_gate), and the basic sample metadata. The output JSONL is the input
to scripts/render_comparison.py which picks 4 diverse cases.

This script is intentionally NOT in src/ -- it's a one-shot tool for
generating marketing artifacts, not part of the published inference path.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm


def _add_src_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, "..", "src"))
    if src not in sys.path:
        sys.path.insert(0, src)


def find_image_dir(data_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(data_dir, "screenspotv2_image"),
        os.path.join(data_dir, "images"),
        data_dir,
    ]
    for d in candidates:
        if os.path.isdir(d) and any(f.endswith(".png") for f in os.listdir(d)):
            return d
    return None


def load_screenspot_v2(data_dir: str, splits: List[str]) -> List[Dict]:
    out = []
    for split in splits:
        path = os.path.join(data_dir, f"screenspot_{split}_v2.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for ann in json.load(f):
                ann["split"] = split
                out.append(ann)
    return out


def stratified_sample(
    annotations: List[Dict],
    max_samples: int,
    seed: int = 42,
) -> List[Dict]:
    """Same logic as src/eval.py's _stratified_sample. Same seed (42) means
    we hit the same stratum the Phase 7/9 evals used."""
    if max_samples >= len(annotations):
        return annotations
    rng = random.Random(seed)
    by_strat: Dict = {}
    for a in annotations:
        key = (a.get("split", "?"), a.get("data_type", "?"))
        by_strat.setdefault(key, []).append(a)
    total = len(annotations)
    out: List[Dict] = []
    for group in by_strat.values():
        rng.shuffle(group)
        n = max(1, int(round(max_samples * len(group) / total)))
        out.extend(group[: min(n, len(group))])
    rng.shuffle(out)
    return out[:max_samples]


def is_inside(x: Optional[float], y: Optional[float], bbox_xywh) -> bool:
    if x is None or y is None:
        return False
    bx, by, bw, bh = bbox_xywh
    return bx <= x <= bx + bw and by <= y <= by + bh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="ScreenSpot-v2 root (containing screenspot_*_v2.json + screenspotv2_image/)")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--splits", nargs="+",
                        default=["desktop", "mobile", "web"])
    parser.add_argument("--max-samples", type=int, default=300,
                        help="Cap (stratified by split + data_type). Use 0 for full set.")
    parser.add_argument("--icon-only", action="store_true",
                        help="Drop text-typed samples after stratification (we only need icon "
                             "candidates for the comparison image, but stratified sampling is "
                             "easier to reason about with both types)")
    parser.add_argument("--coarse-max-pixels", type=int, default=1_500_000)
    parser.add_argument("--zoom-factor", type=float, default=2.0)
    parser.add_argument("--attn-impl", default="flash_attention_2")
    args = parser.parse_args()

    _add_src_to_path()
    from cursor_ccf import CCFConfig, ccf_predict_bbox  # noqa: E402
    from eval import load_model, predict_gui_g2  # noqa: E402

    print(f"Loading ScreenSpot-v2 from {args.data_dir}")
    annotations = load_screenspot_v2(args.data_dir, args.splits)
    print(f"  {len(annotations)} total samples across {args.splits}")

    if args.max_samples and args.max_samples < len(annotations):
        annotations = stratified_sample(annotations, args.max_samples)
        print(f"  stratified down to {len(annotations)}")

    if args.icon_only:
        annotations = [a for a in annotations if a.get("data_type") == "icon"]
        print(f"  icon-only filter: {len(annotations)} remaining")

    image_dir = find_image_dir(args.data_dir)
    if not image_dir:
        sys.exit(f"ERROR: no image directory in {args.data_dir}")

    print(f"Loading model from {args.base_model}")
    model, processor = load_model(
        None, base_model_name=args.base_model, attn_impl=args.attn_impl,
    )

    ccf_cfg = CCFConfig(
        zoom_factor=args.zoom_factor,
        coarse_max_pixels=args.coarse_max_pixels,
        instruction_classifier_fn=None,  # We want raw CCF here, not gated
    )

    out: List[Dict] = []
    skipped = 0
    n_baseline_hit = 0
    n_ccf_hit = 0
    n_flipped = 0  # baseline missed, CCF hit
    t0 = time.time()

    for s in tqdm(annotations, desc="Eval", unit="sample"):
        img_path = os.path.join(image_dir, s["img_filename"])
        if not os.path.exists(img_path):
            skipped += 1
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        # Baseline
        try:
            (bx, by), _ = predict_gui_g2(model, processor, image, s["instruction"])
        except Exception:
            bx = by = None

        # CCF
        def predict(img, instr):
            (x, y), raw = predict_gui_g2(model, processor, img, instr)
            if x is None:
                return None, raw
            return (float(x), float(y)), raw

        try:
            ccf_result = ccf_predict_bbox(predict, image, s["instruction"], ccf_cfg)
            if ccf_result is None:
                cx = cy = None
                stage = "fail"
            else:
                cx, cy = ccf_result.x, ccf_result.y
                stage = ccf_result.stage
        except Exception:
            cx = cy = None
            stage = "error"

        bbox = s["bbox"]
        baseline_hit = is_inside(bx, by, bbox)
        ccf_hit = is_inside(cx, cy, bbox)

        if baseline_hit:
            n_baseline_hit += 1
        if ccf_hit:
            n_ccf_hit += 1
        if not baseline_hit and ccf_hit:
            n_flipped += 1

        out.append({
            "img_filename": s["img_filename"],
            "instruction": s["instruction"],
            "bbox": bbox,
            "split": s.get("split"),
            "data_type": s.get("data_type"),
            "image_size": list(image.size),
            "baseline_xy": [float(bx), float(by)] if bx is not None else None,
            "baseline_hit": baseline_hit,
            "ccf_xy": [float(cx), float(cy)] if cx is not None else None,
            "ccf_hit": ccf_hit,
            "ccf_stage": stage,
        })

    elapsed = (time.time() - t0) / 60.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone in {elapsed:.1f}min. Wrote {len(out)} records to {args.out}")
    print(f"  Baseline hit:           {n_baseline_hit}/{len(out)} = {n_baseline_hit/max(len(out),1):.1%}")
    print(f"  CCF hit:                {n_ccf_hit}/{len(out)} = {n_ccf_hit/max(len(out),1):.1%}")
    print(f"  Baseline miss + CCF hit: {n_flipped}/{len(out)} = {n_flipped/max(len(out),1):.1%}")
    if skipped:
        print(f"  Skipped (missing/bad image): {skipped}")


if __name__ == "__main__":
    main()
