"""LoRA-only SFT warmstart for the cursor action schema.

This is the cheap pre-training pass that teaches GUI-G2-3B (and vanilla
Qwen2.5-VL-3B-Instruct) to emit `<action>move(x, y)</action>` instead
of the bbox strings GUI-G2 was trained on. After this finishes, the
GRPO trainer in [src/train_cursor_grpo.py] can take over with a model
that already produces the right schema, so its rollouts produce
parseable signal from step 0.

Design choices:
- Single forward+backward per example (no batching). At ~2k samples
  and bf16, one epoch finishes in well under an hour on H100/H200.
  We can revisit batching if the run is the bottleneck.
- LoRA only (rank configurable; default 64). Same target modules as
  the GRPO policy in [src/cursor_vlm_policy.py] so the adapter is
  drop-in compatible as a warmstart.
- Loss is masked LM CE: only the gold completion tokens contribute,
  not the prompt or vision tokens. Standard SFT.
- Gold completion is `<action>move(proc_x, proc_y)</action>` where
  (proc_x, proc_y) are in the processor's post-smart_resize space.
  This matches what `model.generate` natively emits, so SFT and
  inference live in the same coordinate frame.
- Periodic progress JSON dump for the same reason as the GRPO trainer:
  if the pod dies mid-run, we don't lose the metric history.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

from cursor_prompt import SYSTEM_PROMPT, build_user_prompt
from cursor_sft_data import (
    SFTExample,
    gold_completion_in_processed_space,
    load_sft_jsonl,
)


@dataclass
class SFTConfig:
    base_model_path: str
    train_data: str
    output_dir: str = "./checkpoints/sft-cursor"
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
    log_every: int = 25
    seed: int = 42
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class SFTStepMetrics:
    step: int
    epoch: int
    loss: float
    completion_tokens: int
    wall_seconds: float


@dataclass
class SFTState:
    metrics: List[SFTStepMetrics] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    completed_examples: int = 0


def _dump_progress(path: str, state: SFTState, total: int) -> None:
    payload = {
        "completed_examples": state.completed_examples,
        "total_examples_planned": total,
        "checkpoints": list(state.checkpoints),
        "last_metric": (
            state.metrics[-1].__dict__ if state.metrics else None
        ),
        "metrics_history": [m.__dict__ for m in state.metrics[-200:]],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _build_inputs_and_labels(
    processor,
    example: SFTExample,
    device: torch.device,
    dtype: torch.dtype,
):
    """Tokenize one example into model inputs + a labels tensor that
    masks everything except the gold completion tokens.

    Returns a dict of model kwargs (input_ids, attention_mask, pixel_values,
    image_grid_thw, labels) all on `device`.
    """
    from qwen_vl_utils import process_vision_info  # type: ignore

    user_text = build_user_prompt(example.instruction, step_index=0, history=[])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example.image},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # Render the prompt (without the answer) so we know where the
    # completion starts. We then append the gold completion text and
    # tokenize the whole thing -- this guarantees the prompt prefix is
    # bit-identical to what the model would see at generate time.
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

    # Compute processed image dims so the gold completion lives in the
    # same coord space the model emits at generate time.
    proc_h = int(prompt_inputs["image_grid_thw"][0][1].item() * 14)
    proc_w = int(prompt_inputs["image_grid_thw"][0][2].item() * 14)
    gold = gold_completion_in_processed_space(example, processed_size=(proc_w, proc_h))

    # Tokenize prompt+completion as a single string so token boundaries
    # match the prompt-only tokenization at the prefix.
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

    # Sanity: the prompt prefix should be unchanged. If for some reason
    # the tokenizer merges across the boundary, fall back to truncating.
    if full_len <= prompt_len:
        # Nothing to learn; degenerate case.
        return None

    labels = torch.full((full_len,), -100, dtype=torch.long)
    labels[prompt_len:] = input_ids[prompt_len:]
    completion_tokens = full_len - prompt_len

    return {
        "input_ids": full_inputs["input_ids"].to(device),
        "attention_mask": full_inputs["attention_mask"].to(device),
        "pixel_values": full_inputs["pixel_values"].to(device, dtype=dtype),
        "image_grid_thw": full_inputs["image_grid_thw"].to(device),
        "labels": labels.unsqueeze(0).to(device),
        "completion_tokens": completion_tokens,
    }


def train_sft(cfg: SFTConfig, examples: Sequence[SFTExample]) -> SFTState:
    """Run an SFT epoch on the given examples. Returns SFTState."""
    # GPU-only imports; defer so the file imports cleanly on Mac.
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

    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    progress_path = os.path.join(cfg.output_dir, "progress.json")
    state = SFTState()

    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb  # type: ignore
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={k: v for k, v in cfg.__dict__.items()},
                reinit=True,
            )
            print(f"[wandb] logging to project {cfg.wandb_project} run {wandb_run.name}")
        except Exception as exc:
            print(f"[wandb] failed to init ({exc}); continuing without it")
            wandb_run = None

    device = next(model.parameters()).device
    total_examples = cfg.epochs * len(examples)
    print(
        f"Starting SFT: {len(examples)} examples * {cfg.epochs} epochs "
        f"= {total_examples} steps"
    )

    global_step = 0
    for epoch in range(cfg.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)

        for i, idx in enumerate(order):
            t0 = time.time()
            example = examples[idx]

            try:
                packed = _build_inputs_and_labels(processor, example, device, dtype)
            except Exception as exc:
                print(f"  [skip] example {idx}: {exc}")
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
                print(f"  [skip] non-finite loss at step {global_step}")
                continue
            loss.backward()
            optimizer.step()

            metrics = SFTStepMetrics(
                step=global_step,
                epoch=epoch,
                loss=float(loss.item()),
                completion_tokens=int(packed["completion_tokens"]),
                wall_seconds=time.time() - t0,
            )
            state.metrics.append(metrics)
            state.completed_examples += 1
            global_step += 1

            if cfg.log_every and global_step % cfg.log_every == 0:
                # Average the last `log_every` losses for a smoother signal.
                window = state.metrics[-cfg.log_every:]
                avg_loss = sum(m.loss for m in window) / len(window)
                avg_wall = sum(m.wall_seconds for m in window) / len(window)
                print(
                    f"epoch {epoch} step {global_step:>5} "
                    f"loss {avg_loss:.4f} "
                    f"tokens {metrics.completion_tokens} "
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

        # End-of-epoch checkpoint
        ckpt = os.path.join(cfg.output_dir, f"epoch-{epoch + 1}")
        model.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)
        state.checkpoints.append(ckpt)
        print(f"Saved epoch checkpoint to {ckpt}")

    # Always save a "final" symlink-equivalent so eval scripts have a
    # stable path to consume.
    final_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    state.checkpoints.append(final_path)

    summary_path = os.path.join(cfg.output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": {k: v for k, v in cfg.__dict__.items()},
                "num_examples": len(examples),
                "num_epochs": cfg.epochs,
                "completed_examples": state.completed_examples,
                "final_loss": (
                    state.metrics[-1].loss if state.metrics else None
                ),
                "checkpoints": state.checkpoints,
            },
            f,
            indent=2,
        )
    print(f"SFT complete. Final checkpoint at {final_path}")
    if wandb_run is not None:
        wandb_run.finish()
    return state


def main():  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="./checkpoints/sft-cursor")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-pixels", type=int, default=1_003_520)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="Defaults to lora_r * 2")
    parser.add_argument("--attn-impl", type=str, default="sdpa")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Step interval; 0 means save only at epoch end.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    examples = load_sft_jsonl(args.data)
    if args.max_samples and len(examples) > args.max_samples:
        random.Random(args.seed).shuffle(examples)
        examples = examples[: args.max_samples]
    print(f"Loaded {len(examples)} SFT examples from {args.data}")

    cfg = SFTConfig(
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
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    train_sft(cfg, examples)


if __name__ == "__main__":  # pragma: no cover
    main()
