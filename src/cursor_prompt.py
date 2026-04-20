"""Prompt templates for the GUI-Cursor policy.

These are pure strings; tokenization and chat-template application happen
inside the VLM policy. The system prompt tells the model it is controlling
a virtual cursor, explains the action schema, and (per the GUI-Cursor
paper) skips chain-of-thought. The user prompt restates the target and
optionally includes the cursor movement history so the model can reason
about where it has already been.
"""

from typing import List, Tuple

SYSTEM_PROMPT = (
    "You are a GUI agent controlling a virtual cursor on a screenshot. "
    "Your goal is to move the cursor so that its tip lands on the target "
    "element described by the user. The current cursor position (if any) "
    "is rendered as a white arrow on the image.\n\n"
    "At each step, emit exactly one action using this schema:\n"
    "  <action>move(x, y)</action>   -- move the cursor to absolute pixel "
    "coordinates (x, y)\n"
    "  <action>stop</action>          -- commit the current position as "
    "your final answer\n\n"
    "Do not explain your reasoning. Output only the action tag. Stop as "
    "soon as the cursor is on the target."
)


def build_user_prompt(
    instruction: str,
    step_index: int,
    history: List[Tuple[int, int]],
) -> str:
    """Build the user-facing prompt for a single step.

    Args:
        instruction: the grounding instruction, e.g. "click the settings icon".
        step_index: 0-based index of this step within the trajectory.
        history: list of (x, y) cursor positions from previous moves.

    Returns:
        A user-prompt string. For step 0 with empty history, this is just
        the target description. For later steps, it also lists prior moves.
    """
    lines = [f"Target: {instruction.strip()}"]

    if step_index > 0 and history:
        lines.append("")
        lines.append("Previous cursor positions:")
        for i, (x, y) in enumerate(history, start=1):
            lines.append(f"  step {i}: ({x}, {y})")

    lines.append("")
    lines.append("Output your next action.")
    return "\n".join(lines)
