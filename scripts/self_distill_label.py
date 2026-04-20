"""Self-distillation labelling for the Phase 7 probe.

Reads (img_path, instruction) records from an input JSONL, runs
GUI-G2-3B greedy inference per sample to produce the model's own
prediction, and writes a new JSONL with the teacher's prediction
as the gold label.

The premise being tested: every Phase 5/6 fine-tune regressed because
ground-truth labels pulled the model OFF its prior. Self-distilled
labels are EXACTLY the model's existing capability surface, so SFT on
them should be capability-preserving (loss can drop without accuracy
degrading). If this works, a real 7B teacher distillation is justified.

`predict_gui_g2` returns a CENTER point (cx, cy). We wrap it in a tiny
synthetic bbox `[cx-8, cy-8, cx+8, cy+8]` so train_bbox_sft.py's
existing gold-completion code (which expects a bbox and uses its center)
plugs in unchanged. The 16x16 size is small enough that the bbox
center is dominated by the teacher's prediction, not the synthetic side.
"""

import argparse
import json
import os
import random
import sys
import time

from PIL import Image
from tqdm import tqdm


def _add_src_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, "..", "src"))
    if src not in sys.path:
        sys.path.insert(0, src)


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "img_path" in r and "instruction" in r:
                out.append(r)
    return out


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="JSONL with img_path + instruction (abs_box ignored)")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--base-model", required=True,
                        help="Path to GUI-G2-3B (the teacher)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with self-distilled labels")
    parser.add_argument("--max-pixels", type=int, default=1_003_520)
    parser.add_argument("--attn-impl", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bbox-half-side", type=int, default=8,
                        help="Half-side of the synthetic bbox around the "
                             "teacher's center prediction. 8 -> 16x16 bbox.")
    args = parser.parse_args()

    _add_src_to_path()
    # Defer import so this script imports cleanly on Mac for syntax checks.
    from eval import load_model, predict_gui_g2  # noqa: E402

    print(f"Loading input from {args.input}")
    samples = load_jsonl(args.input)
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    samples = samples[: args.num_samples]
    print(f"Selected {len(samples)} samples for labelling")

    print(f"Loading teacher from {args.base_model}")
    model, processor = load_model(
        None, base_model_name=args.base_model,
        attn_impl=args.attn_impl,
    )

    out = []
    skipped = 0
    parse_fail = 0
    img_missing = 0
    t0 = time.time()
    for s in tqdm(samples, desc="Self-distilling", unit="sample"):
        img_path = s["img_path"]
        if not os.path.exists(img_path):
            img_missing += 1
            skipped += 1
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue
        try:
            (cx, cy), _ = predict_gui_g2(model, processor, image, s["instruction"])
        except Exception as exc:
            print(f"  [skip] {img_path}: {exc}")
            skipped += 1
            continue
        if cx is None:
            parse_fail += 1
            skipped += 1
            continue
        h = args.bbox_half_side
        # Clamp to image bounds so the SFT trainer's coordinate rescaling
        # doesn't produce out-of-range numbers the processor will reject.
        W, H = image.size
        x1 = max(0, int(cx - h))
        y1 = max(0, int(cy - h))
        x2 = min(W - 1, int(cx + h))
        y2 = min(H - 1, int(cy + h))
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue
        out.append({
            "img_path": img_path,
            "instruction": s["instruction"],
            "abs_box": [x1, y1, x2, y2],
            "source": "self_distill",
            "element_type": s.get("element_type", "icon"),
            "teacher_xy": [int(cx), int(cy)],
        })

    elapsed = time.time() - t0
    write_jsonl(args.output, out)
    print(
        f"\nDone in {elapsed/60:.1f}min. Labelled {len(out)} / {len(samples)} "
        f"(skipped {skipped}: {img_missing} missing img, {parse_fail} parse fail)"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
