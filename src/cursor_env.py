"""Cursor environment for GUI-Cursor style multi-step grounding.

The environment wraps a single screenshot and keeps track of where the
model has moved its virtual cursor. Each call to `step` applies an action
(move or stop) and returns the resulting rendered image plus metadata.

This module is pure Python + PIL -- no torch / transformers imports -- so
it can be developed and unit-tested on a laptop without GPU access.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image

from cursor_actions import Action
from cursor_sprite import build_cursor_sprite

DEFAULT_MAX_STEPS = 4
DEFAULT_REPEAT_TOL = 5


@dataclass
class StepResult:
    """Outcome of a single environment step."""
    image: Image.Image
    position: Optional[Tuple[int, int]]
    done: bool
    reason: str
    step_index: int
    was_valid: bool


class CursorEnv:
    """Interactive environment where a VLM moves a virtual cursor.

    Usage:
        env = CursorEnv(screenshot, max_steps=4)
        while not result.done:
            image = env.render()
            action = model_predict(image, instruction)
            result = env.step(action)
        final_position = env.position
    """

    def __init__(
        self,
        image: Image.Image,
        max_steps: int = DEFAULT_MAX_STEPS,
        repeat_tol: int = DEFAULT_REPEAT_TOL,
    ):
        self.image = image.convert("RGB")
        self.w, self.h = self.image.size
        self.max_steps = max_steps
        self.repeat_tol = repeat_tol
        self._sprite = build_cursor_sprite()
        self._sprite_w, self._sprite_h = self._sprite.size
        self.reset()

    def reset(self) -> None:
        """Clear state so the env can be reused for a fresh rollout."""
        self.position: Optional[Tuple[int, int]] = None
        self.history: List[Tuple[int, int]] = []
        self.steps: int = 0
        self.stopped: bool = False

    def clamp(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp coordinates to valid image bounds [0, w-1] x [0, h-1]."""
        cx = max(0, min(int(x), self.w - 1))
        cy = max(0, min(int(y), self.h - 1))
        return cx, cy

    def is_repeated(self, x: int, y: int, tol: Optional[int] = None) -> bool:
        """True if (x, y) is within `tol` pixels of any previously visited point."""
        tolerance = self.repeat_tol if tol is None else tol
        for px, py in self.history:
            if abs(px - x) <= tolerance and abs(py - y) <= tolerance:
                return True
        return False

    def render(self) -> Image.Image:
        """Return a fresh image with the cursor drawn at the current position.

        If no move has been made yet (`position is None`), returns an
        untouched copy of the original screenshot.
        """
        if self.position is None:
            return self.image.copy()

        base = self.image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

        cx, cy = self.position
        # Paste the sprite so its hotspot (0, 0) lands on (cx, cy).
        paste_x = cx
        paste_y = cy
        overlay.paste(self._sprite, (paste_x, paste_y), self._sprite)

        composed = Image.alpha_composite(base, overlay).convert("RGB")
        return composed

    def step(self, action: Action) -> StepResult:
        """Apply an action and return the updated state.

        Done conditions (any one triggers termination):
          - action is `stop`
          - action is `invalid`
          - `max_steps` moves have been consumed
        """
        if self.stopped:
            return StepResult(
                image=self.render(),
                position=self.position,
                done=True,
                reason="already_stopped",
                step_index=self.steps,
                was_valid=False,
            )

        self.steps += 1
        step_index = self.steps

        if action.kind == "invalid":
            self.stopped = True
            return StepResult(
                image=self.render(),
                position=self.position,
                done=True,
                reason="invalid_action",
                step_index=step_index,
                was_valid=False,
            )

        if action.kind == "stop":
            self.stopped = True
            reason = "stopped_without_move" if self.position is None else "stopped"
            return StepResult(
                image=self.render(),
                position=self.position,
                done=True,
                reason=reason,
                step_index=step_index,
                was_valid=self.position is not None,
            )

        # kind == "move"
        assert action.x is not None and action.y is not None
        new_pos = self.clamp(action.x, action.y)
        self.history.append(new_pos)
        self.position = new_pos

        at_limit = self.steps >= self.max_steps
        if at_limit:
            self.stopped = True

        return StepResult(
            image=self.render(),
            position=new_pos,
            done=at_limit,
            reason="max_steps" if at_limit else "moved",
            step_index=step_index,
            was_valid=True,
        )
