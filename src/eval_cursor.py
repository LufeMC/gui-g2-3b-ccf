"""ScreenSpot-v2 evaluation for the trained cursor-movement policy.

Flow per sample:
  1. Load PIL screenshot + instruction
  2. Run a greedy cursor rollout (max 4 steps) using VLMCursorPolicy
  3. Take the FINAL cursor position as the prediction
  4. Optionally wrap with CCF (coarse rollout on full image -> crop ->
     refined rollout on crop -> map back)
  5. Score: hit if final (x, y) is inside the target bbox

This is a sibling to src/eval.py (which evaluates bbox-style models).
We don't merge them because the rollout interface is fundamentally
different from a single forward pass.
"""

import argparse
import json
import os
import sys
from collections import Counter

from PIL import Image
from tqdm import tqdm

from cursor_actions import parse_action
from cursor_ccf import CCFConfig
from cursor_ccf_cursor import ccf_predict_cursor
from cursor_env import CursorEnv


def find_image_dir(data_dir):
    candidates = [
        os.path.join(data_dir, "screenspotv2_image"),
        os.path.join(data_dir, "images"),
        data_dir,
    ]
    for d in candidates:
        if os.path.isdir(d) and any(f.endswith(".png") for f in os.listdir(d)):
            return d
    return None


def greedy_cursor_predict(policy, image, instruction, max_steps=4):
    """Run a single greedy cursor rollout. Returns ((x, y), raw_text_summary).

    Returns (None, None), tag if no valid move was emitted.
    """
    env = CursorEnv(image, max_steps=max_steps)
    made_valid_move = False
    raw_segments = []
    step_index = 0
    while True:
        rendered = env.render()
        text = policy.generate(rendered, instruction, step_index)
        raw_segments.append(text)
        action = parse_action(text)
        result = env.step(action)
        if action.kind == "move" and result.was_valid:
            made_valid_move = True
        if result.done:
            break
        step_index += 1

    advance = getattr(policy, "next_trajectory", None)
    if callable(advance):
        advance()

    if not made_valid_move or env.position is None:
        return (None, None), "[cursor:no_valid_move]"
    return env.position, f"[cursor:steps={env.steps}]"


def load_policy(base_model, adapter_path, attn_impl, max_pixels):
    """Load a VLMCursorPolicy with the given LoRA adapter."""
    from cursor_vlm_policy import VLMCursorPolicy, VLMPolicyConfig
    cfg = VLMPolicyConfig(
        base_model_path=base_model,
        warmstart_adapter_path=adapter_path,
        max_pixels=max_pixels,
        attn_implementation=attn_impl,
        # Greedy at eval time. Setting do_sample=False; temperature ignored.
        do_sample=False,
        temperature=1.0,
    )
    return VLMCursorPolicy(cfg)


def _stratified_sample(annotations, max_samples, seed=42):
    """Pick `max_samples` annotations preserving (split, data_type) proportions.

    Used by the Phase 6 checkpoint screen so a 200-sample run is a fair
    proxy for the full 1272-sample eval -- icons/text and desktop/mobile/web
    stay in roughly the same ratio they have in the full set.
    """
    import random
    if max_samples is None or max_samples >= len(annotations):
        return annotations
    rng = random.Random(seed)
    by_strat = {}
    for a in annotations:
        key = (a.get("split", "unknown"), a.get("data_type", "unknown"))
        by_strat.setdefault(key, []).append(a)
    total = len(annotations)
    out = []
    for key, group in by_strat.items():
        rng.shuffle(group)
        # Allocate proportionally; round up so small strata still appear.
        n = max(1, int(round(max_samples * len(group) / total)))
        n = min(n, len(group))
        out.extend(group[:n])
    rng.shuffle(out)
    # If proportional rounding overshot, trim back to max_samples.
    if len(out) > max_samples:
        out = out[:max_samples]
    return out


def eval_screenspot_cursor(
    policy,
    data_dir,
    split="all",
    use_ccf=False,
    zoom_factor=2.0,
    coarse_max_pixels=None,
    max_steps=4,
    results_out=None,
    max_samples=None,
):
    splits = []
    if split in ("all", "desktop"):
        splits.append(("desktop", os.path.join(data_dir, "screenspot_desktop_v2.json")))
    if split in ("all", "mobile"):
        splits.append(("mobile", os.path.join(data_dir, "screenspot_mobile_v2.json")))
    if split in ("all", "web"):
        splits.append(("web", os.path.join(data_dir, "screenspot_web_v2.json")))

    image_dir = find_image_dir(data_dir)
    if not image_dir:
        print(f"ERROR: no image directory in {data_dir}")
        return 0.0

    all_annotations = []
    for split_name, json_path in splits:
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            anns = json.load(f)
        for a in anns:
            a["split"] = split_name
        all_annotations.extend(anns)
        print(f"  {split_name}: {len(anns)} samples")

    if max_samples is not None and max_samples < len(all_annotations):
        all_annotations = _stratified_sample(all_annotations, max_samples)
        print(f"  Sampled down to {len(all_annotations)} (stratified)")
    print(f"  Total: {len(all_annotations)} samples\n")

    results = {"correct": 0, "wrong": 0, "parse_fail": 0}
    by_type = Counter()
    by_type_correct = Counter()
    by_split = Counter()
    by_split_correct = Counter()
    ccf_cfg = (
        CCFConfig(
            zoom_factor=zoom_factor,
            coarse_max_pixels=coarse_max_pixels,
        )
        if use_ccf else None
    )
    if use_ccf:
        print(f"  CCF enabled: zoom={zoom_factor} coarse_max_pixels={coarse_max_pixels}")

    pbar = tqdm(all_annotations, desc="Evaluating", unit="sample")
    for i, sample in enumerate(pbar):
        filename = sample["img_filename"]
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")

        if use_ccf:
            ccf_result = ccf_predict_cursor(
                policy, image, sample["instruction"],
                config=ccf_cfg, max_steps_per_stage=max_steps,
            )
            if ccf_result is None:
                pred_x, pred_y = None, None
            else:
                pred_x, pred_y = ccf_result.x, ccf_result.y
        else:
            (pred_x, pred_y), _raw = greedy_cursor_predict(
                policy, image, sample["instruction"], max_steps=max_steps,
            )

        el_type = sample.get("data_type", "unknown")
        split_name = sample.get("split", "unknown")
        by_type[el_type] += 1
        by_split[split_name] += 1

        if pred_x is None:
            results["parse_fail"] += 1
            continue
        bbox_x, bbox_y, bbox_w, bbox_h = sample["bbox"]
        hit = (bbox_x <= pred_x <= bbox_x + bbox_w
               and bbox_y <= pred_y <= bbox_y + bbox_h)
        if hit:
            results["correct"] += 1
            by_type_correct[el_type] += 1
            by_split_correct[split_name] += 1
        else:
            results["wrong"] += 1

        total_so_far = sum(results.values())
        acc = results["correct"] / total_so_far if total_so_far else 0
        pbar.set_postfix(acc=f"{acc:.1%}", pf=results["parse_fail"])

        if results_out and total_so_far > 0 and total_so_far % 25 == 0:
            _dump(results_out, total_so_far, len(all_annotations),
                  results, by_split, by_split_correct,
                  by_type, by_type_correct)

    total = sum(results.values())
    accuracy = results["correct"] / total if total else 0
    print(f"\n{'='*60}")
    print(f"RESULTS: {accuracy:.1%} ({results['correct']}/{total})")
    print(f"  Parse failures: {results['parse_fail']}")
    print("\nBy platform:")
    for s in sorted(by_split.keys()):
        s_acc = by_split_correct[s] / by_split[s] if by_split[s] else 0
        print(f"  {s}: {s_acc:.1%} ({by_split_correct[s]}/{by_split[s]})")
    print("\nBy element type:")
    for t in sorted(by_type.keys()):
        t_acc = by_type_correct[t] / by_type[t] if by_type[t] else 0
        print(f"  {t}: {t_acc:.1%} ({by_type_correct[t]}/{by_type[t]})")

    if results_out:
        _dump(results_out, total, len(all_annotations),
              results, by_split, by_split_correct,
              by_type, by_type_correct)
    return accuracy


def _dump(path, processed, total, results, by_split, by_split_correct, by_type, by_type_correct):
    payload = {
        "processed": processed,
        "total": total,
        "results": dict(results),
        "running_accuracy": results["correct"] / max(processed, 1),
        "by_split": {
            s: {"n": by_split[s], "correct": by_split_correct[s],
                "acc": by_split_correct[s] / max(by_split[s], 1)}
            for s in by_split
        },
        "by_type": {
            t: {"n": by_type[t], "correct": by_type_correct[t],
                "acc": by_type_correct[t] / max(by_type[t], 1)}
            for t in by_type
        },
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="Path to cursor LoRA adapter")
    parser.add_argument("--data", required=True, help="ScreenSpot-v2 data dir")
    parser.add_argument("--split", default="all", choices=["all", "desktop", "mobile", "web"])
    parser.add_argument("--ccf", action="store_true")
    parser.add_argument("--zoom-factor", type=float, default=2.0)
    parser.add_argument("--coarse-max-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=1_003_520)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--attn-impl", default="flash_attention_2")
    parser.add_argument("--results-out", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap eval at N samples (stratified by split + element type) "
                             "for fast checkpoint screening. Default: full eval.")
    args = parser.parse_args()

    print(f"Loading policy: base={args.base_model} adapter={args.adapter}")
    policy = load_policy(args.base_model, args.adapter, args.attn_impl, args.max_pixels)

    accuracy = eval_screenspot_cursor(
        policy,
        args.data,
        split=args.split,
        use_ccf=args.ccf,
        zoom_factor=args.zoom_factor,
        coarse_max_pixels=args.coarse_max_pixels,
        max_steps=args.max_steps,
        results_out=args.results_out,
        max_samples=args.max_samples,
    )
    print(f"\nFinal accuracy: {accuracy:.1%}")
