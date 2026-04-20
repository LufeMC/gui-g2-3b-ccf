"""Unit tests for the Phase 1 cursor environment modules.

Covers:
- Cursor sprite rendering onto the base image
- Action parser: move / stop / invalid
- CursorEnv state transitions: move, stop, max_steps, invalid
- Coordinate clamping and repeated-position detection

All tests use a synthetic 400x400 white image; no model or GPU required.
"""

import os
import sys

import pytest
from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_actions import Action, parse_action  # noqa: E402
from cursor_env import CursorEnv  # noqa: E402


IMAGE_W, IMAGE_H = 400, 400


@pytest.fixture
def blank_image() -> Image.Image:
    return Image.new("RGB", (IMAGE_W, IMAGE_H), (255, 255, 255))


# ---- Sprite / render tests ----

def test_render_cursor_at_position(blank_image):
    env = CursorEnv(blank_image)
    env.step(Action(kind="move", x=100, y=100))
    out = env.render()
    # Some pixel near the hotspot should now be non-white (black outline).
    px = out.getpixel((100, 100))
    assert px != (255, 255, 255), "Cursor not drawn at expected position"
    # Far-away region should still be white.
    far = out.getpixel((10, 10))
    assert far == (255, 255, 255), "Cursor unexpectedly leaked to distant pixel"


def test_render_near_edge(blank_image):
    """Cursor placed at the far corner must not raise and must clip cleanly."""
    env = CursorEnv(blank_image)
    env.step(Action(kind="move", x=IMAGE_W - 2, y=IMAGE_H - 2))
    out = env.render()
    assert out.size == (IMAGE_W, IMAGE_H)


def test_render_before_any_move(blank_image):
    """Rendering before any step should return the untouched image."""
    env = CursorEnv(blank_image)
    out = env.render()
    # Every pixel should still be white.
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((200, 200)) == (255, 255, 255)


# ---- Action parser tests ----

@pytest.mark.parametrize(
    "text,expected_x,expected_y",
    [
        ("<action>move(420, 315)</action>", 420, 315),
        ("<action>move(  10 ,  20  )</action>", 10, 20),
        ("<action>MOVE(5, 5)</action>", 5, 5),
        ("prefix text <action>move(420.0, 315.7)</action> suffix", 420, 316),
        ("<action>\n  move(100, 200)\n</action>", 100, 200),
    ],
)
def test_parse_move(text, expected_x, expected_y):
    action = parse_action(text)
    assert action.kind == "move"
    assert action.x == expected_x
    assert action.y == expected_y


@pytest.mark.parametrize(
    "text",
    [
        "<action>stop</action>",
        "<action>STOP</action>",
        "<action>  stop  </action>",
    ],
)
def test_parse_stop(text):
    action = parse_action(text)
    assert action.kind == "stop"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "<action>click(1, 2)</action>",
        "<action>move(a, b)</action>",
        "<action>move(1)</action>",
        "move(1, 2)",  # missing tags
    ],
)
def test_parse_invalid(text):
    action = parse_action(text)
    assert action.kind == "invalid"
    assert action.raw == text


# ---- Env state transitions ----

def test_step_move_updates_position(blank_image):
    env = CursorEnv(blank_image)
    result = env.step(Action(kind="move", x=50, y=60))
    assert result.position == (50, 60)
    assert result.done is False
    assert result.was_valid is True
    assert env.history == [(50, 60)]
    assert env.steps == 1


def test_step_stop_sets_done(blank_image):
    env = CursorEnv(blank_image)
    env.step(Action(kind="move", x=50, y=60))
    result = env.step(Action(kind="stop"))
    assert result.done is True
    assert result.reason == "stopped"
    assert env.stopped is True


def test_step_stop_without_move_is_invalid(blank_image):
    env = CursorEnv(blank_image)
    result = env.step(Action(kind="stop"))
    assert result.done is True
    assert result.reason == "stopped_without_move"
    assert result.was_valid is False


def test_step_max_steps_sets_done(blank_image):
    env = CursorEnv(blank_image, max_steps=2)
    env.step(Action(kind="move", x=10, y=10))
    result = env.step(Action(kind="move", x=20, y=20))
    assert result.done is True
    assert result.reason == "max_steps"
    assert env.stopped is True


def test_step_invalid_action_terminates(blank_image):
    env = CursorEnv(blank_image)
    result = env.step(Action(kind="invalid", raw="garbage"))
    assert result.done is True
    assert result.reason == "invalid_action"
    assert result.was_valid is False


def test_step_after_stop_is_noop(blank_image):
    env = CursorEnv(blank_image)
    env.step(Action(kind="move", x=10, y=10))
    env.step(Action(kind="stop"))
    result = env.step(Action(kind="move", x=99, y=99))
    assert result.done is True
    assert result.reason == "already_stopped"
    assert env.position == (10, 10), "Position must not change after stop"


# ---- Utility behaviors ----

def test_is_repeated(blank_image):
    env = CursorEnv(blank_image, max_steps=5, repeat_tol=5)
    env.step(Action(kind="move", x=100, y=100))
    assert env.is_repeated(102, 103) is True
    assert env.is_repeated(150, 150) is False


def test_clamp_coords(blank_image):
    env = CursorEnv(blank_image)
    assert env.clamp(-5, -5) == (0, 0)
    assert env.clamp(IMAGE_W + 50, IMAGE_H + 50) == (IMAGE_W - 1, IMAGE_H - 1)
    assert env.clamp(100, 100) == (100, 100)


def test_move_out_of_bounds_is_clamped(blank_image):
    env = CursorEnv(blank_image)
    result = env.step(Action(kind="move", x=10_000, y=-5))
    assert result.position == (IMAGE_W - 1, 0)


def test_reset_clears_state(blank_image):
    env = CursorEnv(blank_image)
    env.step(Action(kind="move", x=10, y=10))
    env.step(Action(kind="stop"))
    env.reset()
    assert env.position is None
    assert env.history == []
    assert env.stopped is False
    assert env.steps == 0
