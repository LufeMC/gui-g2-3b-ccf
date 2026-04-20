"""Cursor-Centric Focusing (CCF) inference wrapper.

Generalizes the GUI-Cursor paper's CCF technique to any grounding model:
run once on the full screenshot to get a coarse prediction, crop a
window centered on that prediction, run again on the crop, and map the
refined coordinate back to the original image's pixel space.

Why this helps: a small icon that occupies ~30 pixels in a 1920x1080
screenshot becomes ~60 pixels in a 2x crop. Qwen2.5-VL sees more
per-element tokens, making small-target grounding noticeably easier.

Design choices locked in:
- Greedy-only on both passes. Our earlier stochastic-sampling zoom
  regressed because temperature noise corrupted already-correct
  predictions. Never do that again.
- Single refinement pass. Iterative narrowing (> 1 refinement) gives
  diminishing returns and compounds coordinate-mapping errors.
- Fall back to coarse if the refined pass fails to parse -- we never
  return None if the coarse pass succeeded.
- Crop shifts near the boundary instead of shrinking, so the model
  always sees a fixed-size region at its native resolution budget.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PIL import Image

Bbox = Tuple[int, int, int, int]  # (left, top, right, bottom)
Point = Tuple[float, float]

DEFAULT_ZOOM_FACTOR = 2.0
DEFAULT_MIN_PIXELS_FOR_CCF = 200_000  # ~450x450; below this, crop won't help
DEFAULT_MIN_CROP_SIDE = 112  # 4 * 28, Qwen2.5-VL's smallest patch grid
DEFAULT_COARSE_MAX_PIXELS: Optional[int] = None  # None == use original


# Phase 9: type-aware classifier keyword lists. Tuned by inspecting the
# Phase 4 ScreenSpot-v2 results -- text-targeted instructions that
# regressed when refined had patterns like "address bar", "search field",
# "page heading", quoted strings; icon-targeted ones that benefitted had
# "icon", "button", "menu", short verb-noun patterns.
ICON_KEYWORDS = (
    " icon", " button", " logo", " image", " menu", " arrow",
    " plus", " minus", " back", " home", " settings", " trash",
    " delete", " edit", " send", " download", " upload", " share",
    " filter", " sort", " heart", " star", " bell", " camera",
    " phone", " mail", " calendar", " clock", " refresh", " play",
    " pause", " stop", " volume", " mute", " expand", " collapse",
    " hamburger", " kebab", " dots", " more", " plus icon",
    " gear", " cog", " checkbox", " toggle", " switch",
)
TEXT_KEYWORDS = (
    " link", " label", " heading", " title", " field", " input",
    " text", " bar", " address bar", " search bar", " search field",
    " text box", " textbox", " textarea", " text area", " paragraph",
    " caption", " sentence", " phrase", " word ",
)


def classify_instruction(instruction: str) -> str:
    """Classify a grounding instruction as targeting an icon, text, or ambiguous.

    Returns one of "icon" | "text" | "ambiguous". Used by the Phase 9
    type-aware CCF gate to decide whether to run the refined pass --
    text targets are usually wide enough that the coarse model already
    nails them, while CCF refinement introduces small drift that misses.

    Heuristic in three stages:
      1. Quoted strings ("Submit", "Login") almost always reference a
         text label by exact match -> "text".
      2. Keyword counts: more icon hits than text hits -> "icon", and
         vice versa.
      3. Tie-break by length: short instructions are usually icon-like
         ("the X icon"), longer descriptive ones are usually text.
    """
    if not instruction:
        return "ambiguous"
    s = " " + instruction.lower() + " "  # pad so " icon" matches at edges
    if '"' in instruction or "'" in instruction:
        return "text"
    icon_hits = sum(1 for k in ICON_KEYWORDS if k in s)
    text_hits = sum(1 for k in TEXT_KEYWORDS if k in s)
    if text_hits > icon_hits:
        return "text"
    if icon_hits > text_hits:
        return "icon"
    # Tie / no hits: short = icon, long = text
    n = len(instruction.strip())
    if n < 25:
        return "icon"
    return "ambiguous"


@dataclass
class CCFConfig:
    """Tuning knobs for CCF."""
    zoom_factor: float = DEFAULT_ZOOM_FACTOR
    min_pixels_for_ccf: int = DEFAULT_MIN_PIXELS_FOR_CCF
    min_crop_side: int = DEFAULT_MIN_CROP_SIDE
    fallback_to_coarse_on_invalid: bool = True
    # When set, the coarse pass downsizes any image whose pixel count
    # exceeds this value. Predicted (x, y) is scaled back to the
    # original-image space before the crop window is computed. This is
    # how the GUI-Cursor paper actually runs CCF -- the coarse pass
    # just needs to localize; the refined pass uses native resolution
    # on the cropped region. Saves ~50% wall time on 1920x1080 inputs.
    coarse_max_pixels: Optional[int] = DEFAULT_COARSE_MAX_PIXELS
    # Phase 9: when set, called with the instruction string. If it
    # returns "text", we skip the refinement pass and return the coarse
    # prediction (with stage="coarse_text_gate"). The Phase 4 result
    # showed CCF refinement helps icons (+2.2pp) but hurts text (-2.3pp);
    # gating by instruction type recovers the text loss without giving
    # up the icon win. None disables gating (Phase 4 behavior).
    instruction_classifier_fn: Optional[Callable[[str], str]] = None


@dataclass
class CCFResult:
    """What CCF returns. Never None if the coarse pass succeeded."""
    x: float
    y: float
    stage: str  # "coarse" | "refined" | "fallback"
    coarse_xy: Point
    crop_window: Optional[Bbox]  # None when we skipped cropping


# Type alias for the caller-provided prediction function. Takes an image
# and an instruction, returns ((x, y) or None, raw_text_for_logging).
PredictFn = Callable[[Image.Image, str], Tuple[Optional[Point], str]]


def compute_crop_window(
    center: Point,
    image_size: Tuple[int, int],
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
    min_crop_side: int = DEFAULT_MIN_CROP_SIDE,
) -> Bbox:
    """Return a (left, top, right, bottom) window of target dimensions.

    The window is centered on `center` when possible. If `center` sits
    near an edge the window shifts rather than shrinks, so the model
    always sees a full-size crop. Dimensions floor at `min_crop_side`.
    """
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"invalid image size {image_size}")
    if zoom_factor <= 0:
        raise ValueError(f"zoom_factor must be positive, got {zoom_factor}")

    target_w = max(min_crop_side, int(img_w / zoom_factor))
    target_h = max(min_crop_side, int(img_h / zoom_factor))
    # Clamp target to image bounds so we never try to crop bigger than the image.
    target_w = min(target_w, img_w)
    target_h = min(target_h, img_h)

    cx = int(round(center[0]))
    cy = int(round(center[1]))

    left = cx - target_w // 2
    top = cy - target_h // 2

    # Shift window inward if it spills past the image bounds.
    if left < 0:
        left = 0
    elif left + target_w > img_w:
        left = img_w - target_w
    if top < 0:
        top = 0
    elif top + target_h > img_h:
        top = img_h - target_h

    return left, top, left + target_w, top + target_h


def map_crop_to_orig(
    x_on_crop: float,
    y_on_crop: float,
    crop_window: Bbox,
) -> Point:
    """Translate crop-pixel coords to original-image pixel coords."""
    left, top, _right, _bottom = crop_window
    return left + float(x_on_crop), top + float(y_on_crop)


def _should_skip_ccf(image_size: Tuple[int, int], config: CCFConfig) -> bool:
    w, h = image_size
    return (w * h) < config.min_pixels_for_ccf


def _maybe_downsize_for_coarse(
    image: Image.Image, max_pixels: Optional[int]
) -> Tuple[Image.Image, float]:
    """Return (downsized_image, scale_factor).

    `scale_factor` is the multiplier that converts a coordinate in the
    DOWNSIZED image's pixel space back into the ORIGINAL image's pixel
    space. For an image already smaller than `max_pixels`, scale is 1.0.
    """
    if max_pixels is None:
        return image, 1.0
    w, h = image.size
    px = w * h
    if px <= max_pixels:
        return image, 1.0
    # Preserve aspect ratio. scale_down < 1.0 shrinks the image.
    import math
    scale_down = math.sqrt(max_pixels / px)
    new_w = max(1, int(round(w * scale_down)))
    new_h = max(1, int(round(h * scale_down)))
    downsized = image.resize((new_w, new_h), Image.BILINEAR)
    # Scale UP: coarse coords are in downsized space; multiply by this to
    # recover original-space coords.
    scale_up = w / new_w  # equivalently h / new_h up to rounding
    return downsized, scale_up


def ccf_predict_bbox(
    predict_fn: PredictFn,
    image: Image.Image,
    instruction: str,
    config: Optional[CCFConfig] = None,
) -> Optional[CCFResult]:
    """Run CCF around a bbox/point predictor.

    `predict_fn` must return ((x, y) or None, raw_text). Coordinates are
    in the passed image's pixel space.

    Returns a `CCFResult` or `None` if even the coarse pass failed to
    parse. When the refined pass fails, we return a `CCFResult` with
    stage="fallback" using the coarse prediction (provided the config
    has `fallback_to_coarse_on_invalid=True`, which is the default).
    """
    if config is None:
        config = CCFConfig()

    # Coarse pass: optionally downsize the image so we don't spend
    # 10+ seconds on a 1920x1080 forward pass when all we need is a
    # rough localization. Predicted coords come back in downsized
    # pixel space and we scale them up.
    coarse_image, coarse_scale = _maybe_downsize_for_coarse(
        image, config.coarse_max_pixels
    )
    coarse_xy_raw, _coarse_raw = predict_fn(coarse_image, instruction)
    if coarse_xy_raw is None:
        return None
    coarse_xy = (
        float(coarse_xy_raw[0]) * coarse_scale,
        float(coarse_xy_raw[1]) * coarse_scale,
    )

    # Skip refinement on already-small images -- CCF adds latency without
    # helping when the model is already seeing the target at near-native res.
    if _should_skip_ccf(image.size, config):
        return CCFResult(
            x=coarse_xy[0],
            y=coarse_xy[1],
            stage="coarse",
            coarse_xy=coarse_xy,
            crop_window=None,
        )

    # Phase 9 type-aware gate: if the instruction looks like a text
    # target, skip refinement -- text bboxes are wide and the coarse
    # prediction lands inside them; CCF refinement introduces sub-bbox
    # drift that misses (Phase 4 saw text -2.3pp from this).
    if config.instruction_classifier_fn is not None:
        target_type = config.instruction_classifier_fn(instruction)
        if target_type == "text":
            return CCFResult(
                x=coarse_xy[0],
                y=coarse_xy[1],
                stage="coarse_text_gate",
                coarse_xy=coarse_xy,
                crop_window=None,
            )

    crop_window = compute_crop_window(
        center=coarse_xy,
        image_size=image.size,
        zoom_factor=config.zoom_factor,
        min_crop_side=config.min_crop_side,
    )
    cropped = image.crop(crop_window)

    refined_xy, _refined_raw = predict_fn(cropped, instruction)
    if refined_xy is None:
        if not config.fallback_to_coarse_on_invalid:
            return None
        return CCFResult(
            x=coarse_xy[0],
            y=coarse_xy[1],
            stage="fallback",
            coarse_xy=coarse_xy,
            crop_window=crop_window,
        )

    orig_x, orig_y = map_crop_to_orig(refined_xy[0], refined_xy[1], crop_window)
    return CCFResult(
        x=orig_x,
        y=orig_y,
        stage="refined",
        coarse_xy=coarse_xy,
        crop_window=crop_window,
    )


def make_ccf_eval_adapter(model, processor, base_predict_fn, config: Optional[CCFConfig] = None):
    """Adapt an eval-style predictor (takes model + processor + image + instr,
    returns ((x, y), raw_text)) into a CCF-wrapped version with the same signature.

    Used by src/eval.py's --ccf flag. Extracted here so tests can exercise the
    wrapper without importing torch / transformers.

    The wrapped function returns ((x, y), tag) where tag is one of
    "[ccf:coarse]", "[ccf:refined]", "[ccf:fallback]", or "[ccf:parse_fail]".
    """
    if config is None:
        config = CCFConfig()

    def wrapped(model_unused, processor_unused, image: Image.Image, instruction: str):
        def inner(img, instr):
            (x, y), raw = base_predict_fn(model, processor, img, instr)
            if x is None:
                return None, raw
            return (float(x), float(y)), raw

        result = ccf_predict_bbox(inner, image, instruction, config)
        if result is None:
            return (None, None), "[ccf:parse_fail]"
        return (result.x, result.y), f"[ccf:{result.stage}]"

    return wrapped
