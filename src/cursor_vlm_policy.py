"""Real VLM-backed policy for GUI-Cursor training and inference.

GPU-only at runtime. The file imports cleanly on Mac because we guard the
heavy dependencies (transformers, qwen_vl_utils, peft) and only fail with
a clear error when `VLMCursorPolicy` is actually instantiated.

The policy satisfies the `LogprobCursorPolicy` protocol from
[src/cursor_policy.py](src/cursor_policy.py):

  - generate(image, instruction, step_index) -> str
  - generate_with_logprobs(...) -> CompletionOutput
  - teacher_force_logprobs(completions, ref_only) -> list[Tensor]
  - save_checkpoint(path)

Reference policy = this same model with its LoRA adapter disabled via
PEFT's `disable_adapter()` context manager. No separate frozen copy, so
we don't pay 2x VRAM.

Coordinate handling: the caller passes an image in native pixel space. The
processor applies Qwen2.5-VL's `smart_resize` internally, and the model
emits coordinates in the processed pixel space. We rescale them back to
native pixels before returning so the env sees coords in the same space
as the target bbox.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from cursor_actions import parse_action
from cursor_policy import CompletionOutput
from cursor_prompt import SYSTEM_PROMPT, build_user_prompt

try:  # GPU-only imports
    from transformers import (  # type: ignore
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )
    from qwen_vl_utils import process_vision_info  # type: ignore
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
    _VLM_AVAILABLE = True
    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as _err:
    _VLM_AVAILABLE = False
    _IMPORT_ERROR = _err


IMAGE_FACTOR = 28
DEFAULT_MIN_PIXELS = 3136
DEFAULT_MAX_PIXELS = 1_003_520  # 1280 * 28 * 28


def _require_vlm_deps() -> None:
    if not _VLM_AVAILABLE:
        raise ImportError(
            "VLMCursorPolicy requires transformers, peft, and qwen_vl_utils. "
            f"Original import error: {_IMPORT_ERROR}"
        )


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Tuple[int, int]:
    """Mirror of Qwen2.5-VL's smart_resize so we can convert coords."""
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


def rescale_action_coords(
    raw_text: str,
    orig_size: Tuple[int, int],
    processed_size: Tuple[int, int],
) -> str:
    """Rescale any move(x, y) in `raw_text` from processed space to native."""
    action = parse_action(raw_text)
    if action.kind != "move" or action.x is None or action.y is None:
        return raw_text

    orig_w, orig_h = orig_size
    proc_w, proc_h = processed_size
    if proc_w == 0 or proc_h == 0:
        return raw_text

    x = int(round(action.x * orig_w / proc_w))
    y = int(round(action.y * orig_h / proc_h))
    return f"<action>move({x}, {y})</action>"


@dataclass
class VLMPolicyConfig:
    base_model_path: str
    min_pixels: int = DEFAULT_MIN_PIXELS
    max_pixels: int = DEFAULT_MAX_PIXELS
    attn_implementation: str = "sdpa"  # flash_attention_2 optional
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 32
    temperature: float = 1.0
    do_sample: bool = True
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # If set, load an existing LoRA adapter (e.g. an SFT warmstart) instead
    # of creating a fresh one. The reference policy is still "this LoRA
    # disabled", which gives back the underlying base model -- so the KL
    # term in GRPO measures drift from the base, not from the warmstart.
    # That's intentional: we want GRPO to refine the SFT init while staying
    # in the neighborhood of the base model's general capability.
    warmstart_adapter_path: Optional[str] = None


class VLMCursorPolicy:
    """Qwen2.5-VL-backed policy with LoRA adapter.

    Only instantiate on CUDA. Methods raise clear ImportErrors otherwise.
    """

    def __init__(self, cfg: VLMPolicyConfig):
        _require_vlm_deps()
        self.cfg = cfg

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            cfg.torch_dtype, torch.float32
        )
        self._dtype = dtype

        self.processor = AutoProcessor.from_pretrained(
            cfg.base_model_path,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
            padding_side="left",
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.base_model_path,
            torch_dtype=dtype,
            attn_implementation=cfg.attn_implementation,
            device_map=cfg.device_map,
        )

        if cfg.warmstart_adapter_path:
            # Load an existing adapter (typically the SFT warmstart). The
            # adapter is loaded as TRAINABLE so GRPO can update it; this is
            # the default for `is_trainable=True`.
            print(f"[VLMCursorPolicy] Loading warmstart adapter from {cfg.warmstart_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                cfg.warmstart_adapter_path,
                is_trainable=True,
            )
            self._has_lora = True
        elif cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_target_modules),
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self._has_lora = True
        else:
            self._has_lora = False

        self.model.eval()  # rollout runs in eval mode; teacher-force switches to train

        # Expose pad/eos on processor so downstream TRL-ish code works.
        if not hasattr(self.processor, "pad_token_id"):
            self.processor.pad_token_id = self.processor.tokenizer.pad_token_id
        if not hasattr(self.processor, "eos_token_id"):
            self.processor.eos_token_id = self.processor.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def next_trajectory(self) -> None:
        return None

    def generate(
        self,
        image: Image.Image,
        instruction: str,
        step_index: int,
    ) -> str:
        return self.generate_with_logprobs(image, instruction, step_index).text

    @torch.no_grad()
    def generate_with_logprobs(
        self,
        image: Image.Image,
        instruction: str,
        step_index: int,
    ) -> CompletionOutput:
        _require_vlm_deps()

        orig_size = image.size
        messages = self._build_messages(image, instruction, step_index, history=[])

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[1]

        self.model.eval()
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        seq = output.sequences[0]  # [prompt_len + T]
        new_token_ids = seq[prompt_len:].tolist()
        # Drop trailing EOS/PAD for cleanliness
        eos_id = self.processor.tokenizer.eos_token_id
        if new_token_ids and new_token_ids[-1] == eos_id:
            new_token_ids = new_token_ids[:-1]

        # Per-token logprobs under the sampling distribution at generation time.
        logprobs: List[float] = []
        scores = output.scores  # tuple of [1, V] tensors, one per generated position
        for i, tok_id in enumerate(new_token_ids):
            if i >= len(scores):
                break
            score = scores[i][0]  # [V]
            log_probs = torch.log_softmax(score.float(), dim=-1)
            logprobs.append(log_probs[tok_id].item())

        raw_text = self.processor.tokenizer.decode(
            new_token_ids, skip_special_tokens=True,
        )

        # Model outputs coordinates in the processed (post smart_resize) space.
        input_h = int(inputs["image_grid_thw"][0][1].item() * 14)
        input_w = int(inputs["image_grid_thw"][0][2].item() * 14)
        rescaled = rescale_action_coords(
            raw_text, orig_size=orig_size, processed_size=(input_w, input_h),
        )

        replay_data = {
            "input_ids": seq.detach().cpu(),  # full prompt + completion
            "pixel_values": inputs["pixel_values"].detach().cpu(),
            "image_grid_thw": inputs["image_grid_thw"].detach().cpu(),
            "attention_mask": output.sequences.new_ones(seq.shape).cpu(),
            "prompt_length": prompt_len,
        }

        return CompletionOutput(
            text=rescaled,
            token_ids=new_token_ids,
            logprobs_at_generation=logprobs,
            replay_data=replay_data,
        )

    def teacher_force_logprobs(
        self,
        completions: Sequence[CompletionOutput],
        ref_only: bool = False,
    ) -> List[torch.Tensor]:
        """Recompute per-token log-probs for each completion under the
        current (or reference) policy.

        Returns one 1D tensor per completion, aligned with `token_ids`.

        Implementation: completions are bucketed by `image_grid_thw` (so
        batched samples share a vision-token count) and each bucket is
        forwarded as a single right-padded batch. With the default Phase 5
        config (`prompts_per_step=4 * trajectories_per_prompt=6 * max_steps=4`)
        this collapses ~96 sequential forwards per optimizer step into a
        handful of batched forwards. See Phase 3 reflection in
        [.cursor/plans/gui-cursor_3b_replication_e314760e.plan.md] for why
        this was the dominant scaling bottleneck.
        """
        _require_vlm_deps()
        if not completions:
            return []

        self.model.train(mode=not ref_only)
        ctx = torch.no_grad() if ref_only else torch.enable_grad()
        adapter_ctx = (
            self.model.disable_adapter()
            if (ref_only and self._has_lora)
            else _null_context()
        )

        # Per-completion result list, filled in by bucket processing below.
        # We preserve the input order so callers can zip with their own state.
        results: List[Optional[torch.Tensor]] = [None] * len(completions)

        with adapter_ctx, ctx:
            for bucket_indices in _bucket_by_grid(completions):
                bucket_completions = [completions[i] for i in bucket_indices]
                bucket_logprobs = self._teacher_force_batch(bucket_completions)
                for i, lp in zip(bucket_indices, bucket_logprobs):
                    results[i] = lp

        if not ref_only:
            self.model.eval()  # default back to eval for the next rollout

        # mypy/runtime sanity: every slot got filled
        return [lp if lp is not None else torch.zeros(0) for lp in results]

    def save_checkpoint(self, path: str) -> None:
        _require_vlm_deps()
        os.makedirs(path, exist_ok=True)
        save_fn = getattr(self.model, "save_pretrained", None)
        if save_fn is None:
            raise RuntimeError("Wrapped model does not support save_pretrained")
        save_fn(path)
        self.processor.save_pretrained(path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        image: Image.Image,
        instruction: str,
        step_index: int,
        history: Sequence[Tuple[int, int]],
    ) -> List[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": build_user_prompt(
                            instruction, step_index, list(history),
                        ),
                    },
                ],
            },
        ]

    def _teacher_force_batch(
        self, completions: Sequence[CompletionOutput]
    ) -> List[torch.Tensor]:
        """Forward `completions` as a single right-padded batch.

        All completions in the batch must share the same `image_grid_thw`
        (i.e. their image inputs decompose into the same number of vision
        patches). Caller is responsible for bucketing.

        Returns one 1D tensor per completion, aligned with `token_ids`.
        """
        if not completions:
            return []

        # Validate replay data and pull tensors.
        replays = []
        for comp in completions:
            replay = comp.replay_data
            if not isinstance(replay, dict):
                raise RuntimeError(
                    "teacher_force_logprobs requires replay_data on CompletionOutput"
                )
            replays.append(replay)

        # LEFT-pad input_ids and attention_mask to the longest sequence in
        # the batch. Qwen2.5-VL with flash_attention_2 requires left padding:
        # right padding makes the causal attention mask attend to PAD tokens
        # in some samples and `_update_causal_mask` raises a ValueError.
        # Left padding keeps real tokens at the end of the sequence so the
        # causal mask is correct for all samples in the batch.
        pad_id = self.processor.tokenizer.pad_token_id
        seq_lens = [int(r["input_ids"].shape[0]) for r in replays]
        max_len = max(seq_lens)
        batch_size = len(completions)

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        # Per-sample left-pad offset so we can recover prompt/completion
        # positions later: real tokens occupy [pad_offset, max_len).
        pad_offsets: List[int] = []

        for i, r in enumerate(replays):
            L = seq_lens[i]
            offset = max_len - L
            pad_offsets.append(offset)
            input_ids[i, offset:] = r["input_ids"]
            attention_mask[i, offset:] = r["attention_mask"]

        # Concat pixel_values along the patch dim (Qwen2.5-VL stores
        # pixel_values as [total_patches, patch_dim] across the batch and
        # uses image_grid_thw to disambiguate which patches belong to which
        # sample). Concat image_grid_thw as [B, 3].
        pixel_values = torch.cat(
            [r["pixel_values"] for r in replays], dim=0
        ).to(self.model.device, dtype=self._dtype)
        image_grid_thw = torch.cat(
            [r["image_grid_thw"] for r in replays], dim=0
        ).to(self.model.device)

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        logits = outputs.logits  # [B, max_len, V]

        results: List[torch.Tensor] = []
        for i, comp in enumerate(completions):
            completion_len = len(comp.token_ids)
            prompt_length = int(replays[i]["prompt_length"])
            if completion_len == 0:
                results.append(
                    torch.zeros(0, device=logits.device, dtype=logits.dtype)
                )
                continue
            # With LEFT padding, the real tokens for sample i live at
            # positions [pad_offsets[i], max_len). The prompt tokens are
            # the first `prompt_length` of those, and completion tokens
            # follow. Logits at position t predict input_ids[t+1], so to
            # score completion tokens we read logits one position earlier.
            offset = pad_offsets[i]
            prompt_end = offset + prompt_length  # first completion-token absolute position
            start = prompt_end - 1
            end = start + completion_len
            sliced = logits[i, start:end, :]  # [T, V]
            log_probs = torch.log_softmax(sliced.float(), dim=-1)
            targets = input_ids[i, prompt_end : prompt_end + completion_len]
            gathered = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
            results.append(gathered)

        return results


# ---------------------------------------------------------------------------
# Mac-testable chat message builder (kept for Phase 3's test suite)
# ---------------------------------------------------------------------------


def build_chat_messages(
    image_placeholder: Any,
    instruction: str,
    step_index: int,
    history: Sequence[Tuple[int, int]],
) -> List[dict]:
    """Return a list of chat messages matching Qwen2.5-VL's expected format."""
    user_text = build_user_prompt(instruction, step_index, list(history))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_placeholder},
                {"type": "text", "text": user_text},
            ],
        },
    ]


class _null_context:
    """A no-op context manager used when adapter-disable isn't needed."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _bucket_by_grid(
    completions: Sequence[CompletionOutput],
) -> List[List[int]]:
    """Group completion indices by their `image_grid_thw` signature.

    All completions in a returned bucket share the same vision-token shape
    so their `pixel_values` can be safely concatenated for a batched
    forward pass. Completions with no replay data, or with an unexpected
    image_grid_thw shape, are returned as singleton buckets so the batched
    forward path falls back gracefully to size-1 batches.
    """
    buckets: Dict[Any, List[int]] = {}
    fallback_index_buckets: List[List[int]] = []

    for i, comp in enumerate(completions):
        replay = comp.replay_data
        if not isinstance(replay, dict) or "image_grid_thw" not in replay:
            fallback_index_buckets.append([i])
            continue
        thw = replay["image_grid_thw"]
        try:
            key = tuple(thw.flatten().tolist())
        except AttributeError:
            fallback_index_buckets.append([i])
            continue
        buckets.setdefault(key, []).append(i)

    return list(buckets.values()) + fallback_index_buckets
