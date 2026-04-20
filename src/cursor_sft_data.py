"""SFT warmstart data for the cursor action schema.

Phase 3 surfaced that GUI-G2-3B doesn't natively emit our
`<action>move(x, y)</action>` / `<action>stop</action>` schema -- it's
been GRPO-trained to output `[x1, y1, x2, y2]` bbox strings, so most
rollouts return unparseable text and the GRPO learning signal
collapses. This module builds the supervised-fine-tuning data we use
to teach the model the right schema before kicking off GRPO.

The transformation is intentionally simple:

  grounding sample ((image, instruction, target_bbox))
  -> SFT example   ((image, instruction, target_center_xy))

The trainer is responsible for turning the target center into the
gold completion string in the model's processed-pixel space (the
space `model.generate` actually emits). Keeping that step out of this
module means cursor_sft_data.py has no torch/transformers dependency
and stays unit-testable on Mac.
"""

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from PIL import Image


@dataclass
class SFTExample:
    """One supervised-fine-tuning example.

    `target_xy` is in the ORIGINAL image's pixel space. The trainer
    rescales it into processed pixel space (post smart_resize) before
    tokenizing the gold completion, mirroring what the model actually
    emits at inference time.
    """
    image: Image.Image
    instruction: str
    target_xy: Tuple[int, int]
    image_size: Tuple[int, int]  # (w, h) of the original image


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Return the integer center of a [x1, y1, x2, y2] bounding box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def format_move_action(x: int, y: int) -> str:
    """Format a move action exactly the way our parser expects.

    Must match the schema in [src/cursor_actions.py] and [src/cursor_prompt.py].
    """
    return f"<action>move({int(x)}, {int(y)})</action>"


STOP_ACTION = "<action>stop</action>"


def make_sft_example(
    image: Image.Image,
    instruction: str,
    target_bbox: Tuple[int, int, int, int],
) -> SFTExample:
    """Convert a grounding sample into a 1-step SFT example.

    The target is the bbox center, which is what we want the model to
    move to in a single greedy step. Multi-step refinement is GRPO's
    job; SFT just teaches the schema and the rough localization.
    """
    cx, cy = bbox_center(target_bbox)
    return SFTExample(
        image=image,
        instruction=instruction,
        target_xy=(cx, cy),
        image_size=image.size,
    )


def load_sft_jsonl(path: str) -> List[SFTExample]:
    """Load an SFT dataset from the same JSONL schema we use elsewhere.

    Each line: {"img_path", "instruction", "abs_box": [x1, y1, x2, y2], ...}.
    Records with missing images or degenerate bboxes are skipped with a
    warning so a partially-available dataset still works.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out: List[SFTExample] = []
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
            image = Image.open(img_path).convert("RGB")
            out.append(make_sft_example(image, instruction, (x1, y1, x2, y2)))

    if skipped:
        print(f"[cursor_sft_data] Skipped {skipped} records from {path}")
    return out


def gold_completion_in_processed_space(
    example: SFTExample,
    processed_size: Tuple[int, int],
) -> str:
    """Build the gold completion string in the processed-pixel space.

    `processed_size` is the (w, h) of the image AFTER the processor's
    smart_resize. The model's `generate` emits coords in this space, so
    SFT must teach it the same. The trainer obtains processed_size from
    `image_grid_thw` * patch_size (typically 14).
    """
    orig_w, orig_h = example.image_size
    proc_w, proc_h = processed_size
    if proc_w <= 0 or proc_h <= 0:
        raise ValueError(f"processed_size must be positive, got {processed_size}")

    cx, cy = example.target_xy
    proc_x = int(round(cx * proc_w / orig_w))
    proc_y = int(round(cy * proc_h / orig_h))
    return format_move_action(proc_x, proc_y)


def iter_sft_batches(
    examples: Iterable[SFTExample],
    batch_size: int,
) -> Iterable[List[SFTExample]]:
    """Yield successive batches of `batch_size` examples."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: List[SFTExample] = []
    for ex in examples:
        batch.append(ex)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
