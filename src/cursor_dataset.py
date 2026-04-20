"""Dataset loader for GUI-Cursor training.

Reads a JSONL file where each line has at minimum:
    {
      "img_path": "/path/to/screenshot.png",
      "instruction": "click the settings icon",
      "abs_box": [x1, y1, x2, y2]
    }

Extra keys are kept under `meta` for logging. Missing image files are
skipped with a warning at load time so a partially-available dataset
still works.
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class CursorSample:
    """One training example for the cursor environment."""
    image: Image.Image
    instruction: str
    target_bbox: Tuple[int, int, int, int]
    meta: Dict[str, Any] = field(default_factory=dict)


class CursorDataset:
    """In-memory index over a JSONL of grounding samples.

    Images are opened lazily on `__getitem__` so construction is fast
    even for datasets with tens of thousands of entries.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_samples: Optional[int] = None,
        require_existing_images: bool = True,
    ):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(jsonl_path)

        self._records: List[Dict[str, Any]] = []
        skipped = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "img_path" not in record:
                    raise ValueError(f"Record missing img_path: {record}")
                if "instruction" not in record:
                    raise ValueError(f"Record missing instruction: {record}")
                if "abs_box" not in record:
                    raise ValueError(f"Record missing abs_box: {record}")

                if require_existing_images and not os.path.exists(record["img_path"]):
                    skipped += 1
                    continue

                self._records.append(record)
                if max_samples is not None and len(self._records) >= max_samples:
                    break

        self._skipped = skipped

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> CursorSample:
        record = self._records[index]
        image = Image.open(record["img_path"]).convert("RGB")
        bbox = tuple(int(v) for v in record["abs_box"])
        if len(bbox) != 4:
            raise ValueError(f"abs_box must be 4 ints, got {record['abs_box']}")
        meta = {k: v for k, v in record.items() if k not in ("instruction", "abs_box")}
        return CursorSample(
            image=image,
            instruction=record["instruction"],
            target_bbox=bbox,  # type: ignore[arg-type]
            meta=meta,
        )

    def sample(self, n: int, rng: Optional[random.Random] = None) -> List[CursorSample]:
        """Return `n` randomly-sampled CursorSamples without replacement."""
        if n > len(self._records):
            raise ValueError(f"Cannot sample {n} from dataset of size {len(self)}")
        chooser = rng or random
        indices = chooser.sample(range(len(self._records)), n)
        return [self[i] for i in indices]

    @property
    def skipped_count(self) -> int:
        """Number of records skipped during load due to missing images."""
        return self._skipped
