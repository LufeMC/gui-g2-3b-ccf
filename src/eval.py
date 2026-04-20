"""
Evaluate a GUI grounding model on ScreenSpot-v2 benchmark.

Supports two prompt modes:
  --prompt-mode gui-g2    Uses GUI-G2's bounding box prompt (default for GUI-G2 models)
  --prompt-mode point     Uses simple point coordinate prompt (for SFT models)

Usage:
    python src/eval.py --base-model ./models/gui-g2-3b --data ./data/screenspot-v2
    python src/eval.py --ckpt ./checkpoints/step-3750 --prompt-mode point
    python src/eval.py --base-model ./models/gui-g2-3b --split desktop
"""

import argparse
import torch
torch.backends.cudnn.enabled = False
import json
import os
import re
import math
import numpy as np
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image
from collections import Counter
from tqdm import tqdm

from cursor_ccf import CCFConfig, classify_instruction, make_ccf_eval_adapter

MIN_PIXELS = 3136               # Paper default (4 * 28 * 28)
MAX_PIXELS = 12845056           # Paper default -- use full resolution on H100
IMAGE_FACTOR = 28

GUI_G2_PROMPT = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].'
POINT_PROMPT = 'Click on: {}. Reply with ONLY the click point coordinates as (x, y).'


def smart_resize(height, width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Qwen2.5-VL's image resize: ensures dimensions are multiples of 28."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    return h_bar, w_bar


def parse_bbox(text):
    """Extract [x1,y1,x2,y2] from model output and return center point."""
    patterns = [
        r'\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',
        r'\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            x1 = float(match.group(1))
            y1 = float(match.group(2))
            x2 = float(match.group(3))
            y2 = float(match.group(4))
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            return cx, cy
    return None, None


def parse_point(text):
    """Extract (x, y) from model output like '(227, 128)'."""
    patterns = [
        r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)',
        r'(\d+\.?\d*),\s*(\d+\.?\d*)',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None


def load_model(
    checkpoint_path,
    base_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    use_cpu=False,
    attn_impl=None,
):
    dtype = torch.float32 if use_cpu else torch.bfloat16
    device_map = "cpu" if use_cpu else "auto"

    if attn_impl is None:
        attn_impl = "eager" if use_cpu else "sdpa"
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name, torch_dtype=dtype, device_map=device_map,
        attn_implementation=attn_impl,
    )
    if checkpoint_path:
        model = PeftModel.from_pretrained(base, checkpoint_path, device_map=device_map)
        model.merge_adapter()
        print(f"  LoRA merged in memory (files untouched on disk)")
    else:
        model = base
    model.eval()

    processor = AutoProcessor.from_pretrained(
        base_model_name, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    return model, processor


def find_image_dir(data_dir):
    """Find the directory containing extracted ScreenSpot-v2 images."""
    candidates = [
        os.path.join(data_dir, "screenspotv2_image"),
        os.path.join(data_dir, "images"),
        data_dir,
    ]
    for d in candidates:
        if os.path.isdir(d) and any(f.endswith(".png") for f in os.listdir(d)):
            return d
    return None


def predict_gui_g2(model, processor, image, instruction):
    """GUI-G2 style: bbox prompt, process_vision_info, image_grid_thw normalization."""
    import time as _time
    timings = {}
    t0 = _time.time()
    orig_w, orig_h = image.size

    prompt = GUI_G2_PROMPT.format(instruction)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    t1 = _time.time()
    image_inputs, video_inputs = process_vision_info(messages)
    t2 = _time.time()
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    t3 = _time.time()

    n_vision_tokens = int(inputs['image_grid_thw'][0].prod().item())
    n_input_tokens = int(inputs.input_ids.shape[1])

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t4 = _time.time()

    trimmed = output[0][inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        [trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    t5 = _time.time()

    timings = {
        "tpl": t1 - t0,
        "vis_info": t2 - t1,
        "proc": t3 - t2,
        "gen": t4 - t3,
        "dec": t5 - t4,
        "total": t5 - t0,
    }
    if os.environ.get("EVAL_PROFILE") == "1":
        print(
            f"  [PROF] img={image.size} vis_tok={n_vision_tokens} "
            f"in_tok={n_input_tokens} "
            f"tpl={timings['tpl']:.2f}s visinfo={timings['vis_info']:.2f}s "
            f"proc={timings['proc']:.2f}s gen={timings['gen']:.2f}s "
            f"dec={timings['dec']:.2f}s total={timings['total']:.2f}s",
            flush=True,
        )

    abs_cx, abs_cy = parse_bbox(response)
    if abs_cx is None:
        return (None, None), response

    input_h = inputs['image_grid_thw'][0][1].item() * 14
    input_w = inputs['image_grid_thw'][0][2].item() * 14

    norm_x = abs_cx / input_w
    norm_y = abs_cy / input_h

    orig_x = norm_x * orig_w
    orig_y = norm_y * orig_h

    return (orig_x, orig_y), response


def predict_point(model, processor, image, instruction):
    """Point-coordinate style prompt (for SFT models)."""
    orig_w, orig_h = image.size
    resized_h, resized_w = smart_resize(orig_h, orig_w)
    resized_image = image.resize((resized_w, resized_h))

    prompt = POINT_PROMPT.format(instruction)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": resized_image},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=[resized_image], return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    response = processor.tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    abs_x, abs_y = parse_point(response)
    if abs_x is None:
        return (None, None), response

    orig_x = abs_x * orig_w / resized_w
    orig_y = abs_y * orig_h / resized_h

    return (orig_x, orig_y), response


def _predict_gui_g2_sampled(model, processor, image, instruction, temperature=0.7):
    """Single GUI-G2 prediction with stochastic sampling."""
    orig_w, orig_h = image.size

    prompt = GUI_G2_PROMPT.format(instruction)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=128,
            do_sample=True, temperature=temperature,
        )

    trimmed = output[0][inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        [trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    abs_cx, abs_cy = parse_bbox(response)
    if abs_cx is None:
        return None, None

    input_h = inputs['image_grid_thw'][0][1].item() * 14
    input_w = inputs['image_grid_thw'][0][2].item() * 14

    norm_x = abs_cx / input_w
    norm_y = abs_cy / input_h

    return norm_x * orig_w, norm_y * orig_h


def predict_gui_g2_zoom(model, processor, image, instruction,
                        num_samples=3, temperature=0.7,
                        zoom_factor=2.0, spread_pct=0.02):
    """Multi-sample prediction with zoom-in for uncertain cases.

    1. Generate num_samples predictions with stochastic sampling
    2. If predictions agree (spread < spread_pct of image diagonal) → return median
    3. If uncertain → crop around median, re-predict on zoomed crop, return best
    """
    orig_w, orig_h = image.size
    diag = math.sqrt(orig_w ** 2 + orig_h ** 2)
    threshold = spread_pct * diag

    predictions = []
    for _ in range(num_samples):
        px, py = _predict_gui_g2_sampled(model, processor, image, instruction, temperature)
        if px is not None:
            predictions.append((px, py))

    if len(predictions) == 0:
        return predict_gui_g2(model, processor, image, instruction)

    if len(predictions) == 1:
        return predictions[0], "zoom:single-sample"

    xs = [p[0] for p in predictions]
    ys = [p[1] for p in predictions]
    med_x = float(np.median(xs))
    med_y = float(np.median(ys))
    spread = math.sqrt(float(np.var(xs)) + float(np.var(ys)))

    if spread < threshold:
        return (med_x, med_y), f"zoom:confident spread={spread:.1f}<{threshold:.1f}"

    crop_w = int(orig_w / zoom_factor)
    crop_h = int(orig_h / zoom_factor)
    left = max(0, int(med_x) - crop_w // 2)
    top = max(0, int(med_y) - crop_h // 2)
    right = min(orig_w, left + crop_w)
    bottom = min(orig_h, top + crop_h)

    cropped = image.crop((left, top, right, bottom))

    zoom_preds = []
    for _ in range(num_samples):
        px, py = _predict_gui_g2_sampled(model, processor, cropped, instruction, temperature)
        if px is not None:
            zoom_preds.append((px, py))

    if len(zoom_preds) < 2:
        return (med_x, med_y), f"zoom:fallback spread={spread:.1f}"

    zxs = [p[0] for p in zoom_preds]
    zys = [p[1] for p in zoom_preds]
    zoom_med_x = float(np.median(zxs))
    zoom_med_y = float(np.median(zys))
    zoom_spread = math.sqrt(float(np.var(zxs)) + float(np.var(zys)))

    full_x = left + zoom_med_x
    full_y = top + zoom_med_y

    if zoom_spread < spread:
        return (full_x, full_y), f"zoom:used spread={spread:.1f}->{zoom_spread:.1f}"
    else:
        return (med_x, med_y), f"zoom:orig-better spread={spread:.1f}->{zoom_spread:.1f}"


def _dump_progress(
    path, processed, total, results, by_split, by_split_correct,
    by_type, by_type_correct, ccf_stats, use_ccf,
):
    """Write a single-line atomic JSON snapshot of running stats."""
    payload = {
        "processed": processed,
        "total": total,
        "results": dict(results),
        "running_accuracy": (
            results["correct"] / max(processed, 1)
        ),
        "by_split": {
            s: {
                "n": by_split[s],
                "correct": by_split_correct[s],
                "acc": by_split_correct[s] / max(by_split[s], 1),
            }
            for s in by_split
        },
        "by_type": {
            t: {
                "n": by_type[t],
                "correct": by_type_correct[t],
                "acc": by_type_correct[t] / max(by_type[t], 1),
            }
            for t in by_type
        },
    }
    if use_ccf:
        payload["ccf_stats"] = dict(ccf_stats)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _stratified_sample(annotations, max_samples, seed=42):
    """Pick `max_samples` annotations preserving (split, data_type) proportions.

    Same logic as src/eval_cursor.py's helper -- duplicated here so eval.py
    has no dependency on the cursor-specific module. With seed=42 and
    identical input, both copies return the SAME 200-sample subset, which
    is what lets baseline-vs-probe runs hit the same SS-v2 stratum.
    """
    import random as _r
    if max_samples is None or max_samples >= len(annotations):
        return annotations
    rng = _r.Random(seed)
    by_strat = {}
    for a in annotations:
        key = (a.get("split", "unknown"), a.get("data_type", "unknown"))
        by_strat.setdefault(key, []).append(a)
    total = len(annotations)
    out = []
    for key, group in by_strat.items():
        rng.shuffle(group)
        n = max(1, int(round(max_samples * len(group) / total)))
        n = min(n, len(group))
        out.extend(group[:n])
    rng.shuffle(out)
    if len(out) > max_samples:
        out = out[:max_samples]
    return out


def eval_screenspot(
    model,
    processor,
    data_dir,
    split="all",
    prompt_mode="gui-g2",
    use_zoom=False,
    use_ccf=False,
    zoom_factor=2.0,
    min_pixels_for_ccf=CCFConfig().min_pixels_for_ccf,
    coarse_max_pixels=None,
    results_out=None,
    max_samples=None,
    ccf_type_gate=False,
):
    """Evaluate on ScreenSpot-v2. Bbox format is [x, y, w, h] in absolute pixels."""
    if use_zoom and use_ccf:
        raise ValueError("--zoom and --ccf are mutually exclusive")

    if use_ccf and prompt_mode == "gui-g2":
        ccf_config = CCFConfig(
            zoom_factor=zoom_factor,
            min_pixels_for_ccf=min_pixels_for_ccf,
            coarse_max_pixels=coarse_max_pixels,
            instruction_classifier_fn=(
                classify_instruction if ccf_type_gate else None
            ),
        )
        predict_fn = make_ccf_eval_adapter(
            model, processor, predict_gui_g2, ccf_config,
        )
        coarse_str = (
            f"coarse_max_pixels={coarse_max_pixels}"
            if coarse_max_pixels else "coarse=full_res"
        )
        gate_str = " + type-gate" if ccf_type_gate else ""
        print(
            f"  Prompt mode: {prompt_mode} + CCF{gate_str} "
            f"(zoom={zoom_factor}, min_pixels={min_pixels_for_ccf}, {coarse_str})"
        )
    elif use_zoom and prompt_mode == "gui-g2":
        predict_fn = predict_gui_g2_zoom
        print(f"  Prompt mode: {prompt_mode} + ZOOM (multi-sample + crop)")
    elif prompt_mode == "gui-g2":
        predict_fn = predict_gui_g2
        print(f"  Prompt mode: {prompt_mode}")
    else:
        predict_fn = predict_point
        print(f"  Prompt mode: {prompt_mode}")

    splits = []
    if split in ("all", "desktop"):
        splits.append(("desktop", os.path.join(data_dir, "screenspot_desktop_v2.json")))
    if split in ("all", "mobile"):
        splits.append(("mobile", os.path.join(data_dir, "screenspot_mobile_v2.json")))
    if split in ("all", "web"):
        splits.append(("web", os.path.join(data_dir, "screenspot_web_v2.json")))

    image_dir = find_image_dir(data_dir)
    if not image_dir:
        print(f"ERROR: No image directory found in {data_dir}")
        print("Extract screenspotv2_image.zip first.")
        return 0.0
    print(f"  Image directory: {image_dir}")

    all_annotations = []
    for split_name, json_path in splits:
        if not os.path.exists(json_path):
            print(f"  Warning: {json_path} not found, skipping")
            continue
        with open(json_path) as f:
            annotations = json.load(f)
        for a in annotations:
            a["split"] = split_name
        all_annotations.extend(annotations)
        print(f"  {split_name}: {len(annotations)} samples")

    if max_samples is not None and max_samples < len(all_annotations):
        all_annotations = _stratified_sample(all_annotations, max_samples)
        print(f"  Sampled down to {len(all_annotations)} (stratified)")
    print(f"  Total: {len(all_annotations)} samples\n")

    results = {"correct": 0, "wrong": 0, "parse_fail": 0}
    zoom_stats = {"confident": 0, "zoomed": 0, "fallback": 0}
    ccf_stats = {"coarse": 0, "refined": 0, "fallback": 0, "coarse_text_gate": 0}
    by_type = Counter()
    by_type_correct = Counter()
    by_split = Counter()
    by_split_correct = Counter()

    pbar = tqdm(all_annotations, desc="Evaluating", unit="sample")
    for i, sample in enumerate(pbar):
        filename = sample["img_filename"]
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue

        # CCF does two forward passes per sample with different image sizes,
        # which fragments CUDA's allocator. Periodic empty_cache keeps
        # throughput roughly constant instead of decaying over the run.
        if use_ccf and i > 0 and i % 25 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        image = Image.open(img_path).convert("RGB")
        (pred_x, pred_y), raw = predict_fn(model, processor, image, sample["instruction"])

        el_type = sample.get("data_type", "unknown")
        split_name = sample.get("split", "unknown")
        by_type[el_type] += 1
        by_split[split_name] += 1

        if use_zoom and isinstance(raw, str) and raw.startswith("zoom:"):
            if "confident" in raw:
                zoom_stats["confident"] += 1
            elif "used" in raw:
                zoom_stats["zoomed"] += 1
            else:
                zoom_stats["fallback"] += 1

        if use_ccf and isinstance(raw, str) and raw.startswith("[ccf:"):
            # raw is like "[ccf:refined]" or "[ccf:coarse]" or "[ccf:fallback]"
            tag = raw[5:-1] if raw.endswith("]") else raw[5:]
            if tag in ccf_stats:
                ccf_stats[tag] += 1

        if pred_x is None:
            results["parse_fail"] += 1
            if (i + 1) <= 10 or results["parse_fail"] <= 5:
                print(f"  PARSE FAIL [{i+1}]: {raw[:100]}")
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

        total_so_far = results["correct"] + results["wrong"] + results["parse_fail"]
        acc = results["correct"] / total_so_far if total_so_far else 0
        pbar.set_postfix(acc=f"{acc:.1%}", pf=results["parse_fail"])

        # Periodic checkpoint of all running stats so a pod death doesn't
        # discard hours of eval. Cheap (single JSON dump every 25 samples).
        if results_out and total_so_far > 0 and total_so_far % 25 == 0:
            _dump_progress(
                results_out, total_so_far, len(all_annotations),
                results, by_split, by_split_correct,
                by_type, by_type_correct, ccf_stats, use_ccf,
            )

    total = sum(results.values())
    accuracy = results["correct"] / total if total else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {accuracy:.1%} ({results['correct']}/{total})")
    print(f"  Parse failures: {results['parse_fail']}")

    if use_zoom:
        ztotal = sum(zoom_stats.values())
        print(f"\nZoom stats ({ztotal} samples):")
        print(f"  Confident (no zoom needed): {zoom_stats['confident']} ({zoom_stats['confident']/max(ztotal,1):.0%})")
        print(f"  Zoomed (used crop): {zoom_stats['zoomed']} ({zoom_stats['zoomed']/max(ztotal,1):.0%})")
        print(f"  Fallback (zoom didn't help): {zoom_stats['fallback']} ({zoom_stats['fallback']/max(ztotal,1):.0%})")

    if use_ccf:
        ctotal = sum(ccf_stats.values())
        print(f"\nCCF stats ({ctotal} samples):")
        print(f"  Coarse only (small image):     {ccf_stats['coarse']} ({ccf_stats['coarse']/max(ctotal,1):.0%})")
        print(f"  Coarse via text gate (Phase 9):{ccf_stats['coarse_text_gate']} ({ccf_stats['coarse_text_gate']/max(ctotal,1):.0%})")
        print(f"  Refined (used crop):           {ccf_stats['refined']} ({ccf_stats['refined']/max(ctotal,1):.0%})")
        print(f"  Fallback (refined invalid):    {ccf_stats['fallback']} ({ccf_stats['fallback']/max(ctotal,1):.0%})")

    print(f"\nBy platform:")
    for s in sorted(by_split.keys()):
        s_acc = by_split_correct[s] / by_split[s] if by_split[s] else 0
        print(f"  {s}: {s_acc:.1%} ({by_split_correct[s]}/{by_split[s]})")

    print(f"\nBy element type:")
    for t in sorted(by_type.keys()):
        t_acc = by_type_correct[t] / by_type[t] if by_type[t] else 0
        print(f"  {t}: {t_acc:.1%} ({by_type_correct[t]}/{by_type[t]})")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval on ScreenSpot-v2")
    parser.add_argument("--ckpt", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model name")
    parser.add_argument("--data", type=str, default="./data/screenspot-v2", help="ScreenSpot-v2 data dir")
    parser.add_argument("--split", type=str, default="all", choices=["all", "desktop", "mobile", "web"])
    parser.add_argument("--prompt-mode", type=str, default="gui-g2", choices=["gui-g2", "point"],
                        help="gui-g2: bbox prompt matching GUI-G2 training; point: simple (x,y) prompt")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument("--zoom", action="store_true",
                        help="Enable multi-sample + zoom-in pipeline for uncertain predictions")
    parser.add_argument("--ccf", action="store_true",
                        help="Enable CCF (greedy coarse + refined crop). Mutually exclusive with --zoom.")
    parser.add_argument("--zoom-factor", type=float, default=2.0,
                        help="CCF zoom factor (crop = image / zoom_factor). Default 2.0.")
    parser.add_argument("--min-pixels-for-ccf", type=int,
                        default=CCFConfig().min_pixels_for_ccf,
                        help="Images with fewer pixels skip the refinement pass.")
    parser.add_argument("--coarse-max-pixels", type=int, default=None,
                        help="Downsize images above this pixel count for the coarse "
                             "pass. Speeds up wall time on big screenshots; refined "
                             "pass still uses native resolution. ~1500000 is a good "
                             "value for ScreenSpot-v2 (~2x faster, same accuracy).")
    parser.add_argument("--results-out", type=str, default=None,
                        help="Periodically dump running stats to this JSON path "
                             "so we can recover partial results if the run dies.")
    parser.add_argument("--attn-impl", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention backend. sdpa works everywhere; "
                             "flash_attention_2 requires flash-attn installed.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap eval at N samples (stratified by split + "
                             "data_type). Same seed (42) means baseline and "
                             "probe runs hit the same samples for a paired "
                             "comparison.")
    parser.add_argument("--ccf-type-gate", action="store_true",
                        help="Phase 9: enable instruction-classifier-based "
                             "gating. Skips CCF refinement when the "
                             "instruction looks like a text target. "
                             "Recovers Phase 4's text -2.3pp regression "
                             "while keeping the icon +2.2pp win.")
    args = parser.parse_args()

    if args.ccf and args.zoom:
        parser.error("--ccf and --zoom are mutually exclusive")

    print(f"Loading model (base: {args.base_model}, checkpoint: {args.ckpt or 'none'})...")
    model, processor = load_model(
        args.ckpt,
        base_model_name=args.base_model,
        use_cpu=args.cpu,
        attn_impl=args.attn_impl,
    )

    tag = "  + CCF" if args.ccf else ("  + ZOOM" if args.zoom else "")
    print(f"Evaluating on ScreenSpot-v2 ({args.split}){tag}...")
    accuracy = eval_screenspot(
        model, processor, args.data,
        split=args.split,
        prompt_mode=args.prompt_mode,
        use_zoom=args.zoom,
        use_ccf=args.ccf,
        zoom_factor=args.zoom_factor,
        min_pixels_for_ccf=args.min_pixels_for_ccf,
        coarse_max_pixels=args.coarse_max_pixels,
        results_out=args.results_out,
        max_samples=args.max_samples,
        ccf_type_gate=args.ccf_type_gate,
    )
