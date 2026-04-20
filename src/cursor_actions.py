"""Parse model-emitted action strings for the cursor environment.

The model is prompted to emit one of two formats:

    <action>move(420, 315)</action>
    <action>stop</action>

Anything else (malformed syntax, missing tags, wrong keyword) is treated as
an invalid action. The caller decides how to penalize invalid actions; this
module just reports the parse result.
"""

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

ActionKind = Literal["move", "stop", "invalid"]

_ACTION_BLOCK_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_MOVE_RE = re.compile(
    r"^move\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*$",
    re.IGNORECASE,
)
_STOP_RE = re.compile(r"^stop\s*$", re.IGNORECASE)


@dataclass
class Action:
    """Structured representation of a model action."""
    kind: ActionKind
    x: Optional[int] = None
    y: Optional[int] = None
    raw: str = field(default="")


def parse_action(text: str) -> Action:
    """Extract the action from a model completion.

    Returns an Action with kind="move" | "stop" | "invalid".
    For invalid inputs, the raw text is preserved for debugging / logging.
    """
    if not text:
        return Action(kind="invalid", raw=text or "")

    block_match = _ACTION_BLOCK_RE.search(text)
    if not block_match:
        return Action(kind="invalid", raw=text)

    inner = block_match.group(1).strip()

    move_match = _MOVE_RE.match(inner)
    if move_match:
        try:
            x = int(round(float(move_match.group(1))))
            y = int(round(float(move_match.group(2))))
        except (ValueError, OverflowError):
            return Action(kind="invalid", raw=text)
        return Action(kind="move", x=x, y=y, raw=text)

    if _STOP_RE.match(inner):
        return Action(kind="stop", raw=text)

    return Action(kind="invalid", raw=text)
