"""GUI grounding inference engine, vLLM backend (vLLM >= 0.19).

Two production modes:
  - "fast" (default): Cursor-Centric Focusing -- one coarse + one refined
    pass. Two sequential vLLM calls, ~500-700ms total on a warm A100.
  - "accurate": batched multi-pass self-consistency. Greedy CCF + 3
    sampled CCF + 1 native + 1 0.75x. Exploits vLLM's continuous
    batching to run all 4 coarse passes as ONE batched call, then all 4
    refined passes as ONE batched call, plus one batched call of
    {native, downsampled}. ~2-3s on a warm A100, with a real
    agreement-based confidence.

Why vLLM: HF transformers' eager decode is ~80ms/token; vLLM's paged
attention + CUDA graphs do ~20-25ms/token. For our short bbox outputs
(~12 tokens), that's a 3-4x improvement on the LM half. Vision encode
time stays the same since both stacks call the same ViT.

Why we run vLLM as a library (not its OpenAI server): we need custom
mode logic (CCF wrapper for fast, batched self-consistency for
accurate), agreement-based confidence, and consistent normalized [0,1]
coordinate outputs. Wrapping our FastAPI around `vllm.LLM` is cleaner
than monkeypatching their OpenAI server.

Why we ship on top of `vllm/vllm-openai:vX` rather than pip-installing
vllm: vLLM has a tightly-pinned matrix of (torch, transformers, flash-
attn, xformers, qwen-vl-utils) and getting any one wrong breaks
multimodal init. The official image is the only reliable source.
"""

import math
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

# Heavy imports (vllm) are deferred to GroundingEngine.__init__ so the
# helper functions and dataclasses in this module can be imported (and
# unit-tested) on machines that don't have a CUDA stack installed.

GUI_G2_USER_PROMPT = (
    "Outline the position corresponding to the instruction: {}. "
    "The output should be only [x1,y1,x2,y2]."
)

# Qwen2.5-VL chat template for a single user turn with one image. We
# build the prompt string by hand so we don't have to load the HF
# processor's tokenizer just to apply the chat template.
QWEN_VL_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}"
    "<|im_end|>\n<|im_start|>assistant\n"
)

DEFAULT_MIN_PIXELS = 3136
DEFAULT_MAX_PIXELS = 800_000
DEFAULT_COARSE_MAX_PIXELS = 600_000
# Bbox output is "[x1,y1,x2,y2]" -- ~12 tokens. Capping at 16 keeps
# decode short.
DEFAULT_MAX_NEW_TOKENS = 16
# Patch factor for Qwen2.5-VL's vision tower (2 spatial * 14 patch).
QWEN_VL_PATCH_FACTOR = 28

# Agreement-based confidence thresholds.
AGREEMENT_FULL_PX = 5.0
AGREEMENT_FLOOR_PX = 100.0
CONFIDENCE_FLOOR = 0.2
CLUSTER_RADIUS_PX = 30.0

# CCF crop sizing.
DEFAULT_ZOOM_FACTOR = 2.0
DEFAULT_MIN_CROP_SIDE = 112
DEFAULT_MIN_PIXELS_FOR_CCF = 200_000


@dataclass
class CoarseResult:
    """First-pass output from CCF; emitted by the streaming endpoint
    before the refined pass completes so the UI can show a tentative
    dot while it waits."""
    x: float            # normalized [0, 1] in the original image's space
    y: float
    abs_x: float        # original-image pixel-space (kept for refined_from_coarse)
    abs_y: float
    raw_response: str
    coarse_latency_ms: int


@dataclass
class GroundingResult:
    x: float            # normalized [0, 1] in original image coords
    y: float
    confidence: float   # [0.2, 0.99], based on cross-pass agreement
    latency_ms: int
    mode: str           # "fast" | "accurate"
    n_passes: int       # how many model forwards were used
    agreement_px: float # median pairwise distance across passes (0 = perfect)
    raw_response: str
    passes_xy: List[Tuple[float, float]] = field(default_factory=list)
    coarse_xy: Optional[Tuple[float, float]] = None  # for streaming clients


# ---------- pure-Python helpers (testable without GPU) -----------------

def _parse_bbox_center(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract [x1,y1,x2,y2] from model output, return center (cx, cy)."""
    patterns = [
        r"\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]",
        r"\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            x1 = float(match.group(1))
            y1 = float(match.group(2))
            x2 = float(match.group(3))
            y2 = float(match.group(4))
            return (x1 + x2) / 2, (y1 + y2) / 2
    return None, None


def _median_pairwise_distance(points: List[Tuple[float, float]]) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    dists: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dists.append(math.hypot(dx, dy))
    dists.sort()
    mid = len(dists) // 2
    if len(dists) % 2 == 1:
        return dists[mid]
    return (dists[mid - 1] + dists[mid]) / 2.0


def _confidence_from_agreement(median_px: float, n_passes: int) -> float:
    if n_passes <= 1:
        return 0.75
    if median_px <= AGREEMENT_FULL_PX:
        return 0.99
    if median_px >= AGREEMENT_FLOOR_PX:
        return max(CONFIDENCE_FLOOR, 0.4)
    span = AGREEMENT_FLOOR_PX - AGREEMENT_FULL_PX
    frac = (median_px - AGREEMENT_FULL_PX) / span
    conf = 0.99 - frac * (0.99 - 0.4)
    return max(CONFIDENCE_FLOOR, conf)


def _largest_cluster_centroid(
    points: List[Tuple[float, float]],
    radius_px: float = CLUSTER_RADIUS_PX,
) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
    if not points:
        return (0.0, 0.0), []
    if len(points) == 1:
        return points[0], list(points)

    n = len(points)
    visited = [False] * n
    best_cluster: List[int] = []

    for seed in range(n):
        if visited[seed]:
            continue
        cluster = [seed]
        visited[seed] = True
        i = 0
        while i < len(cluster):
            cur = cluster[i]
            cx, cy = points[cur]
            for j in range(n):
                if visited[j]:
                    continue
                dx = points[j][0] - cx
                dy = points[j][1] - cy
                if math.hypot(dx, dy) <= radius_px:
                    visited[j] = True
                    cluster.append(j)
            i += 1
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    chosen = [points[i] for i in best_cluster]
    cx = sum(p[0] for p in chosen) / len(chosen)
    cy = sum(p[1] for p in chosen) / len(chosen)
    return (cx, cy), chosen


def _smart_resize(
    h: int,
    w: int,
    factor: int = QWEN_VL_PATCH_FACTOR,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Tuple[int, int]:
    """qwen_vl_utils.smart_resize parity. Returns (new_h, new_w), each
    a multiple of `factor`, with total pixels in [min_pixels, max_pixels]."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def _resize_to_pixel_budget(
    image: Image.Image,
    max_pixels: int,
    min_pixels: int = DEFAULT_MIN_PIXELS,
) -> Tuple[Image.Image, float, float]:
    """Pre-resize image to the smart-resize target. Returns (resized,
    scale_x, scale_y) where scale_* multiplies a coordinate in
    resized-image space to recover original-image space."""
    w, h = image.size
    new_h, new_w = _smart_resize(h, w, max_pixels=max_pixels, min_pixels=min_pixels)
    if (new_w, new_h) != image.size:
        resized = image.resize((new_w, new_h), Image.BILINEAR)
    else:
        resized = image
    scale_x = w / new_w if new_w else 1.0
    scale_y = h / new_h if new_h else 1.0
    return resized, scale_x, scale_y


def _compute_crop_window(
    center: Tuple[float, float],
    image_size: Tuple[int, int],
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
    min_crop_side: int = DEFAULT_MIN_CROP_SIDE,
) -> Tuple[int, int, int, int]:
    img_w, img_h = image_size
    target_w = max(min_crop_side, int(img_w / zoom_factor))
    target_h = max(min_crop_side, int(img_h / zoom_factor))
    target_w = min(target_w, img_w)
    target_h = min(target_h, img_h)

    cx = int(round(center[0]))
    cy = int(round(center[1]))

    left = cx - target_w // 2
    top = cy - target_h // 2
    if left < 0:
        left = 0
    elif left + target_w > img_w:
        left = img_w - target_w
    if top < 0:
        top = 0
    elif top + target_h > img_h:
        top = img_h - target_h
    return left, top, left + target_w, top + target_h


def _should_skip_ccf(image_size: Tuple[int, int]) -> bool:
    return image_size[0] * image_size[1] < DEFAULT_MIN_PIXELS_FOR_CCF


def _defensive_downscale(image: Image.Image, max_input_px: int = 1_500_000) -> Image.Image:
    w, h = image.size
    if w * h <= max_input_px:
        return image
    scale = (max_input_px / (w * h)) ** 0.5
    return image.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.LANCZOS,
    )


# ---------- vLLM-backed engine ----------------------------------------

class GroundingEngine:
    """Loads a vLLM Qwen2.5-VL once; serves predictions in two modes."""

    def __init__(
        self,
        model_path: str = "inclusionAI/GUI-G2-3B",
        adapter_path: Optional[str] = None,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        coarse_max_pixels: int = DEFAULT_COARSE_MAX_PIXELS,
        device: str = "auto",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
    ):
        # Deferred heavy import.
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.coarse_max_pixels = coarse_max_pixels

        if adapter_path:
            print(f"WARNING: vLLM LoRA adapter not auto-loaded; ignoring {adapter_path}")

        print(f"Loading vLLM model: {model_path}")
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=gpu_memory_utilization,
            disable_log_stats=True,
            enforce_eager=False,  # CUDA graphs ON
        )

        self._greedy = SamplingParams(
            max_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=0.0,
        )
        self._sampled = SamplingParams(
            max_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=0.4,
            top_p=0.95,
        )
        print("vLLM model ready.")

    # ---------- low-level vLLM call helpers ----------------------------

    @staticmethod
    def _build_prompt(instruction: str) -> str:
        return QWEN_VL_TEMPLATE.format(GUI_G2_USER_PROMPT.format(instruction))

    def _vllm_batch(
        self,
        images_resized: List[Image.Image],
        instruction: str,
        sampling,
    ) -> List[str]:
        """One vLLM batched call. Returns N raw text completions in input order."""
        prompt_text = self._build_prompt(instruction)
        prompts = [
            {"prompt": prompt_text, "multi_modal_data": {"image": im}}
            for im in images_resized
        ]
        outs = self.llm.generate(prompts, sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def _predict_in_orig_space(
        self,
        image: Image.Image,
        instruction: str,
        sampling,
        pixel_budget: Optional[int] = None,
    ) -> Tuple[Optional[Tuple[float, float]], str]:
        budget = pixel_budget if pixel_budget is not None else self.max_pixels
        resized, sx, sy = _resize_to_pixel_budget(image, budget, self.min_pixels)
        outs = self._vllm_batch([resized], instruction, sampling)
        text = outs[0]
        cx, cy = _parse_bbox_center(text)
        if cx is None:
            return None, text
        return (cx * sx, cy * sy), text

    def _predict_batch_in_orig_space(
        self,
        images: List[Image.Image],
        instruction: str,
        sampling,
        pixel_budget: Optional[int] = None,
    ) -> Tuple[List[Optional[Tuple[float, float]]], List[str]]:
        budget = pixel_budget if pixel_budget is not None else self.max_pixels
        prepared = [_resize_to_pixel_budget(im, budget, self.min_pixels) for im in images]
        resized_imgs = [p[0] for p in prepared]
        scales = [(p[1], p[2]) for p in prepared]
        texts = self._vllm_batch(resized_imgs, instruction, sampling)

        results: List[Optional[Tuple[float, float]]] = []
        for text, (sx, sy) in zip(texts, scales):
            cx, cy = _parse_bbox_center(text)
            if cx is None:
                results.append(None)
            else:
                results.append((cx * sx, cy * sy))
        return results, texts

    # ---------- streaming-friendly factored fast mode -----------------

    def predict_coarse_only(
        self, image: Image.Image, instruction: str,
    ) -> Optional[CoarseResult]:
        """Just the coarse CCF pass. Used by /v1/ground/stream so the
        client can render a tentative dot before the refined pass."""
        image = _defensive_downscale(image)
        start = time.perf_counter()
        coarse_xy, raw = self._predict_in_orig_space(
            image, instruction, self._greedy,
            pixel_budget=self.coarse_max_pixels,
        )
        elapsed = int((time.perf_counter() - start) * 1000)
        if coarse_xy is None:
            return None
        in_w, in_h = image.size
        return CoarseResult(
            x=round(coarse_xy[0] / in_w if in_w else 0.5, 4),
            y=round(coarse_xy[1] / in_h if in_h else 0.5, 4),
            abs_x=coarse_xy[0],
            abs_y=coarse_xy[1],
            raw_response=raw,
            coarse_latency_ms=elapsed,
        )

    def predict_refined_from_coarse(
        self,
        image: Image.Image,
        instruction: str,
        coarse: Optional[CoarseResult],
    ) -> GroundingResult:
        """Run the refined pass given a known coarse prediction. If the
        coarse pass already failed (coarse is None), returns a minimal
        unsuccessful result without re-running the model."""
        image = _defensive_downscale(image)
        in_w, in_h = image.size

        if coarse is None:
            return GroundingResult(
                x=0.5, y=0.5, confidence=0.0, latency_ms=0,
                mode="fast", n_passes=1, agreement_px=0.0, raw_response="",
            )

        coarse_xy = (coarse.abs_x, coarse.abs_y)
        if _should_skip_ccf(image.size):
            return GroundingResult(
                x=coarse.x, y=coarse.y, confidence=0.75,
                latency_ms=coarse.coarse_latency_ms,
                mode="fast", n_passes=1, agreement_px=0.0,
                raw_response=coarse.raw_response,
                passes_xy=[coarse_xy],
                coarse_xy=coarse_xy,
            )

        start = time.perf_counter()
        crop_window = _compute_crop_window(coarse_xy, image.size)
        cropped = image.crop(crop_window)
        refined_local, raw = self._predict_in_orig_space(
            cropped, instruction, self._greedy, pixel_budget=self.max_pixels,
        )
        refined_ms = int((time.perf_counter() - start) * 1000)
        total_ms = coarse.coarse_latency_ms + refined_ms

        if refined_local is None:
            # Fall back to coarse rather than failing
            return GroundingResult(
                x=coarse.x, y=coarse.y, confidence=0.75,
                latency_ms=total_ms,
                mode="fast", n_passes=2, agreement_px=0.0,
                raw_response=raw,
                passes_xy=[coarse_xy],
                coarse_xy=coarse_xy,
            )

        orig_x = crop_window[0] + refined_local[0]
        orig_y = crop_window[1] + refined_local[1]
        norm_x = max(0.0, min(1.0, orig_x / in_w if in_w else 0.5))
        norm_y = max(0.0, min(1.0, orig_y / in_h if in_h else 0.5))
        passes = [coarse_xy, (orig_x, orig_y)]
        median_px = _median_pairwise_distance(passes)
        confidence = _confidence_from_agreement(median_px, n_passes=2)

        return GroundingResult(
            x=round(norm_x, 4),
            y=round(norm_y, 4),
            confidence=round(confidence, 3),
            latency_ms=total_ms,
            mode="fast",
            n_passes=2,
            agreement_px=round(median_px, 1),
            raw_response=raw,
            passes_xy=passes,
            coarse_xy=coarse_xy,
        )

    # ---------- public predict (sync) ---------------------------------

    def predict(
        self,
        image: Image.Image,
        instruction: str,
        mode: str = "fast",
    ) -> GroundingResult:
        if mode not in ("fast", "accurate"):
            raise ValueError(f"unknown mode: {mode!r}")
        image = _defensive_downscale(image)

        if mode == "fast":
            coarse = self.predict_coarse_only(image, instruction)
            return self.predict_refined_from_coarse(image, instruction, coarse)

        # Accurate: batched multi-pass self-consistency
        return self._accurate(image, instruction)

    # ---------- accurate mode: batched self-consistency ---------------

    def _accurate(
        self, image: Image.Image, instruction: str,
    ) -> GroundingResult:
        start = time.perf_counter()
        in_w, in_h = image.size

        # 1 greedy CCF coarse + 3 sampled CCF coarse, batched into 2 calls
        greedy_xy_list, _greedy_texts = self._predict_batch_in_orig_space(
            [image], instruction, self._greedy,
            pixel_budget=self.coarse_max_pixels,
        )
        sampled_xys, sampled_texts = self._predict_batch_in_orig_space(
            [image] * 3, instruction, self._sampled,
            pixel_budget=self.coarse_max_pixels,
        )
        coarse_xys = greedy_xy_list + sampled_xys
        last_raw = (sampled_texts[-1] if sampled_texts else "")
        valid_coarse = [xy for xy in coarse_xys if xy is not None]

        if not valid_coarse:
            return GroundingResult(
                x=0.5, y=0.5, confidence=0.0,
                latency_ms=int((time.perf_counter() - start) * 1000),
                mode="accurate", n_passes=4, agreement_px=0.0, raw_response=last_raw,
            )

        # Refined passes on per-coarse crops, batched
        crop_windows = [_compute_crop_window(xy, image.size) for xy in valid_coarse]
        crops = [image.crop(w) for w in crop_windows]

        if crops:
            greedy_refined_list, _ = self._predict_batch_in_orig_space(
                crops[:1], instruction, self._greedy, pixel_budget=self.max_pixels,
            )
            sampled_refined_list, sampled_refined_texts = self._predict_batch_in_orig_space(
                crops[1:], instruction, self._sampled, pixel_budget=self.max_pixels,
            )
            refined_local = greedy_refined_list + sampled_refined_list
            if sampled_refined_texts:
                last_raw = sampled_refined_texts[-1]
        else:
            refined_local = []

        refined_global: List[Tuple[float, float]] = []
        for window, local_xy in zip(crop_windows, refined_local):
            if local_xy is None:
                continue
            refined_global.append((window[0] + local_xy[0], window[1] + local_xy[1]))

        # Multi-resolution: native + 0.75x downsample, batched
        small = image.resize(
            (max(1, int(image.size[0] * 0.75)), max(1, int(image.size[1] * 0.75))),
            Image.LANCZOS,
        )
        sx_small = image.size[0] / small.size[0]
        sy_small = image.size[1] / small.size[1]
        multires_xys, multires_texts = self._predict_batch_in_orig_space(
            [image, small], instruction, self._greedy, pixel_budget=self.max_pixels,
        )
        if multires_texts:
            last_raw = multires_texts[-1]

        all_passes: List[Tuple[float, float]] = list(refined_global)
        if multires_xys[0] is not None:
            all_passes.append(multires_xys[0])
        if multires_xys[1] is not None:
            all_passes.append((multires_xys[1][0] * sx_small, multires_xys[1][1] * sy_small))

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if not all_passes:
            return GroundingResult(
                x=0.5, y=0.5, confidence=0.0,
                latency_ms=elapsed_ms, mode="accurate",
                n_passes=4 + 2, agreement_px=0.0, raw_response=last_raw,
            )

        centroid, voters = _largest_cluster_centroid(all_passes)
        median_px = _median_pairwise_distance(voters) if len(voters) > 1 else 0.0
        confidence = _confidence_from_agreement(median_px, n_passes=len(voters))

        norm_x = max(0.0, min(1.0, centroid[0] / in_w if in_w else 0.5))
        norm_y = max(0.0, min(1.0, centroid[1] / in_h if in_h else 0.5))

        return GroundingResult(
            x=round(norm_x, 4),
            y=round(norm_y, 4),
            confidence=round(confidence, 3),
            latency_ms=elapsed_ms,
            mode="accurate",
            n_passes=4 + 2,
            agreement_px=round(median_px, 1),
            raw_response=last_raw,
            passes_xy=voters,
        )
