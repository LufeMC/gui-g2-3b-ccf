"""Programmatically generate a standard arrow cursor sprite.

The sprite is a 24x24 transparent RGBA image with a classic pointer-arrow
silhouette. The hotspot (tip of the arrow) is at pixel (0, 0), so when
pasting onto a base image we simply align the sprite's top-left with the
target (x, y) coordinate.

Design goals:
- Visually distinct against any background (white fill + black outline)
- No external sprite files -- fully generated via PIL
- Small enough to minimize occlusion but large enough for the VLM to see
"""

from PIL import Image, ImageDraw

CURSOR_POINTS = [
    (0, 0),
    (0, 16),
    (4, 12),
    (7, 20),
    (9, 19),
    (6, 11),
    (12, 11),
]

SPRITE_SIZE = 24


def build_cursor_sprite() -> Image.Image:
    """Return a 24x24 RGBA image with a white-filled, black-outlined arrow.

    The arrow's tip (hotspot) sits at pixel (0, 0) of the returned image.
    """
    sprite = Image.new("RGBA", (SPRITE_SIZE, SPRITE_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.polygon(CURSOR_POINTS, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
    # Thicken outline by drawing the boundary a second time.
    draw.line(CURSOR_POINTS + [CURSOR_POINTS[0]], fill=(0, 0, 0, 255), width=2)
    return sprite
