"""Tests for the prompt templates."""

import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_prompt import SYSTEM_PROMPT, build_user_prompt  # noqa: E402


def test_system_prompt_describes_action_schema():
    assert "<action>move(x, y)</action>" in SYSTEM_PROMPT
    assert "<action>stop</action>" in SYSTEM_PROMPT


def test_system_prompt_forbids_thinking():
    # Paper finds no-thinking works best for grounding; prompt should say so.
    lowered = SYSTEM_PROMPT.lower()
    assert "do not explain" in lowered or "only the action" in lowered


def test_user_prompt_contains_instruction():
    prompt = build_user_prompt("click the settings icon", step_index=0, history=[])
    assert "click the settings icon" in prompt


def test_user_prompt_step_0_has_no_history_section():
    prompt = build_user_prompt("x", step_index=0, history=[])
    assert "Previous cursor positions" not in prompt


def test_user_prompt_with_history():
    prompt = build_user_prompt(
        "click search",
        step_index=2,
        history=[(100, 200), (150, 180)],
    )
    assert "Previous cursor positions" in prompt
    assert "(100, 200)" in prompt
    assert "(150, 180)" in prompt


def test_user_prompt_always_prompts_for_action():
    prompt = build_user_prompt("x", step_index=0, history=[])
    assert "Output your next action" in prompt


def test_user_prompt_history_ordered():
    prompt = build_user_prompt(
        "x",
        step_index=3,
        history=[(10, 20), (30, 40), (50, 60)],
    )
    # Should list steps in order
    p_first = prompt.index("(10, 20)")
    p_second = prompt.index("(30, 40)")
    p_third = prompt.index("(50, 60)")
    assert p_first < p_second < p_third


def test_user_prompt_empty_history_after_step_0():
    """If step_index > 0 but history is empty (e.g., after invalid action),
    we gracefully skip the history section."""
    prompt = build_user_prompt("x", step_index=1, history=[])
    assert "Previous cursor positions" not in prompt
    assert "Output your next action" in prompt
