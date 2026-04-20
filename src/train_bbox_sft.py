"""Bbox-format SFT for GUI-G2-3B (Phase 6 Tier 3 pivot).

Trains the model to produce its NATIVE output -- the bbox string
"[x1, y1, x2, y2]" -- given the GUI-G2 prompt format. No cursor schema,
no rollouts, no GRPO. This is the "do less, optimize what already works"
fallback after both cursor GRPO attempts failed to beat baseline.

Mirrors src/train_cursor_sft.py exactly except:
  - prompt: GUI-G2's `"Outline the position corresponding to the
    instruction: {}. The output should be only [x1,y1,x2,y2]."`
  - gold completion: `"[x1, y1, x2, y2]"` in the processor's processed
    pixel space (so the model and inference are in the same coord frame)
  - target rescaling reuses src/cursor_sft_data.py's logic for converting
    original-pixel bbox -> processed-pixel bbox

Eval is via the existing src/eval.py (gui-g2 prompt mode + CCF).
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image


GUI_G2_PROMPT = (
    "Outline the position corresponding to the instruction: {}. "
    "The output should be only [x1,y1,x2,y2]."
)


@dataclass
class BboxExample:
    image: Image.Image
    instruction: str
    abs_box: tuple  # (x1, y1, x2, y2) in original image pixels
    image_size: tuple  # (w, h) of original


@dataclass
class BboxSFTConfig:
    base_model_path: str
    train_data: str
    output_dir: str = "./checkpoints/bbox-sft"
    epochs: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.0
    max_pixels: int = 1_003_520
    min_pixels: int = 3136
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    attn_implementation: str = "sdpa"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    save_every: int = 0  # 0 = save only at end of each epoch
    log_every: int = 50
    seed: int = 42
    max_samples: Optional[int] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class StepMetrics:
    step: int
    epoch: int
    loss: float
    completion_tokens: int
    wall_seconds: float


@dataclass
class SFTState:
    metrics: List[StepMetrics] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    completed_examples: int = 0


def _dump_progress(path: str, state: SFTState, total: int) -> None:
    payload = {
        "completed_examples": state.completed_examples,
        "total_examples_planned": total,
        "checkpoints": list(state.checkpoints),
        "last_metric": state.metrics[-1].__dict__ if state.metrics else None,
        "metrics_history": [m.__dict__ for m in state.metrics[-200:]],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_bbox_jsonl(path: str) -> List[BboxExample]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            img_path = r.get("img_path")
            instruction = r.get("instruction")
            bbox = r.get("abs_box")
            if not img_path or not instruction or not bbox or len(bbox) != 4:
                skipped += 1
                continue
            if not os.path.exists(img_path):
                skipped += 1
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox)
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                continue
            out.append(BboxExample(
                image=image,
                instruction=instruction,
                abs_box=(x1, y1, x2, y2),
                image_size=image.size,
            ))
    if skipped:
        print(f"[bbox-sft] Skipped {skipped} records from {path}")
    return out


def _bbox_in_processed_space(
    example: BboxExample,
    processed_size: tuple,
) -> str:
    """Build the gold completion '[x1, y1, x2, y2]' in processed-pixel space.

    Matches what GUI-G2-3B emits at generate-time so SFT and inference
    live in the same coord frame.
    """
    orig_w, orig_h = example.image_size
    proc_w, proc_h = processed_size
    if proc_w <= 0 or proc_h <= 0:
        raise ValueError(f"processed_size must be positive, got {processed_size}")
    x1, y1, x2, y2 = example.abs_box
    px1 = int(round(x1 * proc_w / orig_w))
    py1 = int(round(y1 * proc_h / orig_h))
    px2 = int(round(x2 * proc_w / orig_w))
    py2 = int(round(y2 * proc_h / orig_h))
    return f"[{px1},{py1},{px2},{py2}]"


def _build_inputs_and_labels(processor, example, device, dtype):
    """Tokenize one example with masked LM CE labels covering only the gold completion."""
    from qwen_vl_utils import process_vision_info  # type: ignore

    user_text = GUI_G2_PROMPT.format(example.instruction)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": example.image},
            {"type": "text", "text": user_text},
        ]}
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    proc_h = int(prompt_inputs["image_grid_thw"][0][1].item() * 14)
    proc_w = int(prompt_inputs["image_grid_thw"][0][2].item() * 14)
    gold = _bbox_in_processed_space(example, (proc_w, proc_h))

    full_text = prompt_text + gold
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_ids = full_inputs["input_ids"][0]
    full_len = int(input_ids.shape[0])
    if full_len <= prompt_len:
        return None
    labels = torch.full((full_len,), -100, dtype=torch.long)
    labels[prompt_len:] = input_ids[prompt_len:]
    return {
        "input_ids": full_inputs["input_ids"].to(device),
        "attention_mask": full_inputs["attention_mask"].to(device),
        "pixel_values": full_inputs["pixel_values"].to(device, dtype=dtype),
        "image_grid_thw": full_inputs["image_grid_thw"].to(device),
        "labels": labels.unsqueeze(0).to(device),
        "completion_tokens": full_len - prompt_len,
    }


def train_bbox_sft(cfg: BboxSFTConfig, examples) -> SFTState:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        cfg.torch_dtype, torch.float32
    )

    print(f"Loading processor from {cfg.base_model_path}")
    processor = AutoProcessor.from_pretrained(
        cfg.base_model_path,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        padding_side="left",
    )
    if not hasattr(processor, "pad_token_id"):
        processor.pad_token_id = processor.tokenizer.pad_token_id
    if not hasattr(processor, "eos_token_id"):
        processor.eos_token_id = processor.tokenizer.eos_token_id

    print(f"Loading model from {cfg.base_model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.base_model_path,
        torch_dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        device_map=cfg.device_map,
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"LoRA trainable parameters: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.output_dir, exist_ok=True)
    progress_path = os.path.join(cfg.output_dir, "progress.json")
    state = SFTState()

    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb  # type: ignore
            wandb_run = wandb.init(
                project=cfg.wandb_project, name=cfg.wandb_run_name,
                config={k: v for k, v in cfg.__dict__.items()},
                reinit=True,
            )
            print(f"[wandb] {cfg.wandb_project} / {wandb_run.name}")
        except Exception as exc:
            print(f"[wandb] init failed ({exc})")

    device = next(model.parameters()).device
    total_examples = cfg.epochs * len(examples)
    print(f"Starting bbox SFT: {len(examples)} examples * {cfg.epochs} epochs = {total_examples}")

    global_step = 0
    for epoch in range(cfg.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i, idx in enumerate(order):
            t0 = time.time()
            ex = examples[idx]
            try:
                packed = _build_inputs_and_labels(processor, ex, device, dtype)
            except Exception as exc:
                print(f"  [skip] {idx}: {exc}")
                continue
            if packed is None:
                continue
            optimizer.zero_grad()
            outputs = model(
                input_ids=packed["input_ids"],
                attention_mask=packed["attention_mask"],
                pixel_values=packed["pixel_values"],
                image_grid_thw=packed["image_grid_thw"],
                labels=packed["labels"],
            )
            loss = outputs.loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()

            metrics = StepMetrics(
                step=global_step, epoch=epoch,
                loss=float(loss.item()),
                completion_tokens=int(packed["completion_tokens"]),
                wall_seconds=time.time() - t0,
            )
            state.metrics.append(metrics)
            state.completed_examples += 1
            global_step += 1

            if cfg.log_every and global_step % cfg.log_every == 0:
                window = state.metrics[-cfg.log_every:]
                avg_loss = sum(m.loss for m in window) / len(window)
                avg_wall = sum(m.wall_seconds for m in window) / len(window)
                print(
                    f"epoch {epoch} step {global_step:>5} "
                    f"loss {avg_loss:.4f} tokens {metrics.completion_tokens} "
                    f"wall {avg_wall:.2f}s"
                )

            if wandb_run is not None:
                wandb_run.log({
                    "sft/loss": metrics.loss,
                    "sft/completion_tokens": metrics.completion_tokens,
                    "sft/wall_seconds": metrics.wall_seconds,
                    "sft/epoch": epoch,
                }, step=global_step)

            if cfg.save_every and global_step % cfg.save_every == 0:
                ckpt = os.path.join(cfg.output_dir, f"step-{global_step}")
                model.save_pretrained(ckpt)
                processor.save_pretrained(ckpt)
                state.checkpoints.append(ckpt)

            _dump_progress(progress_path, state, total_examples)

        ckpt = os.path.join(cfg.output_dir, f"epoch-{epoch + 1}")
        model.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)
        state.checkpoints.append(ckpt)
        print(f"Saved epoch checkpoint to {ckpt}")

    final_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    state.checkpoints.append(final_path)

    summary_path = os.path.join(cfg.output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "config": {k: v for k, v in cfg.__dict__.items()},
            "num_examples": len(examples),
            "num_epochs": cfg.epochs,
            "completed_examples": state.completed_examples,
            "final_loss": state.metrics[-1].loss if state.metrics else None,
            "checkpoints": state.checkpoints,
        }, f, indent=2)

    print(f"bbox SFT complete. Final checkpoint at {final_path}")
    if wandb_run is not None:
        wandb_run.finish()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="./checkpoints/bbox-sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-pixels", type=int, default=1_003_520)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    examples = load_bbox_jsonl(args.data)
    if args.max_samples and len(examples) > args.max_samples:
        random.Random(args.seed).shuffle(examples)
        examples = examples[: args.max_samples]
    print(f"Loaded {len(examples)} bbox SFT examples from {args.data}")

    cfg = BboxSFTConfig(
        base_model_path=args.base_model,
        train_data=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        max_pixels=args.max_pixels,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha if args.lora_alpha else args.lora_r * 2,
        attn_implementation=args.attn_impl,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        max_samples=args.max_samples,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    train_bbox_sft(cfg, examples)


if __name__ == "__main__":
    main()
