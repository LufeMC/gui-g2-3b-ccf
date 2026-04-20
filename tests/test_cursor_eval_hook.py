"""Tests for the in-training validation eval hook."""

import os
import sys

from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_dataset import CursorSample  # noqa: E402
from cursor_eval_hook import (  # noqa: E402
    CursorEvalResult,
    evaluate_cursor_policy,
)
from cursor_policy import MockLogprobPolicy  # noqa: E402


def _mk_sample(bbox, instruction="click target"):
    return CursorSample(
        image=Image.new("RGB", (200, 200), (255, 255, 255)),
        instruction=instruction,
        target_bbox=bbox,
    )


def test_evaluate_returns_zero_for_empty_samples():
    policy = MockLogprobPolicy([["<action>stop</action>"]])
    result = evaluate_cursor_policy(policy, [], step=0)
    assert isinstance(result, CursorEvalResult)
    assert result.num_samples == 0
    assert result.accuracy == 0.0


def test_evaluate_perfect_accuracy_when_all_hit():
    samples = [
        _mk_sample(bbox=(90, 90, 110, 110)),
        _mk_sample(bbox=(40, 40, 60, 60)),
    ]
    scripts = [
        ["<action>move(100, 100)</action>", "<action>stop</action>"],
        ["<action>move(50, 50)</action>", "<action>stop</action>"],
    ]
    policy = MockLogprobPolicy(scripts)
    result = evaluate_cursor_policy(policy, samples, step=10)
    assert result.num_samples == 2
    assert result.accuracy == 1.0
    assert result.parse_rate == 1.0
    assert result.step == 10


def test_evaluate_zero_accuracy_when_all_miss():
    samples = [
        _mk_sample(bbox=(90, 90, 110, 110)),
        _mk_sample(bbox=(40, 40, 60, 60)),
    ]
    scripts = [
        ["<action>move(0, 0)</action>", "<action>stop</action>"],
        ["<action>move(0, 0)</action>", "<action>stop</action>"],
    ]
    policy = MockLogprobPolicy(scripts)
    result = evaluate_cursor_policy(policy, samples, step=0)
    assert result.accuracy == 0.0
    assert result.parse_rate == 1.0  # they emitted valid moves, just wrong ones


def test_evaluate_parse_rate_drops_when_garbage_emitted():
    samples = [_mk_sample(bbox=(90, 90, 110, 110))]
    # Script returns an unparseable string. CursorEnv treats invalid actions
    # as "invalid_action" and ends the trajectory; no valid move was emitted.
    scripts = [["this is not a valid action"]]
    policy = MockLogprobPolicy(scripts)
    result = evaluate_cursor_policy(policy, samples, step=0)
    assert result.parse_rate == 0.0
    assert result.accuracy == 0.0


def test_evaluate_handles_mixed_outcomes():
    samples = [
        _mk_sample(bbox=(90, 90, 110, 110)),  # will be hit
        _mk_sample(bbox=(40, 40, 60, 60)),    # will be missed
    ]
    scripts = [
        ["<action>move(100, 100)</action>", "<action>stop</action>"],
        ["<action>move(0, 0)</action>", "<action>stop</action>"],
    ]
    policy = MockLogprobPolicy(scripts)
    result = evaluate_cursor_policy(policy, samples, step=0)
    assert result.accuracy == 0.5
    assert result.parse_rate == 1.0


def test_evaluate_restores_policy_sample_settings():
    """The eval should toggle do_sample to False and restore on exit."""
    class _StubPolicy:
        class _Cfg:
            do_sample = True
            temperature = 1.5
        cfg = _Cfg()
        _calls = []

        def generate(self, image, instruction, step_index):
            # Capture sampling state at generate-time
            self._calls.append((self.cfg.do_sample, self.cfg.temperature))
            return "<action>stop</action>"

        def next_trajectory(self):
            pass

    policy = _StubPolicy()
    samples = [_mk_sample(bbox=(0, 0, 1, 1))]
    evaluate_cursor_policy(policy, samples, step=0)
    # During eval, do_sample should have been False
    assert policy._calls[0][0] is False
    # After eval, original settings restored
    assert policy.cfg.do_sample is True
    assert policy.cfg.temperature == 1.5


def test_evaluate_advances_mock_trajectory_between_samples():
    """Without next_trajectory(), the second sample would re-use the
    first script. The hook must call it so each sample sees a fresh script.
    """
    samples = [
        _mk_sample(bbox=(90, 90, 110, 110)),
        _mk_sample(bbox=(40, 40, 60, 60)),
    ]
    scripts = [
        ["<action>move(100, 100)</action>", "<action>stop</action>"],
        ["<action>move(50, 50)</action>", "<action>stop</action>"],
    ]
    policy = MockLogprobPolicy(scripts)
    result = evaluate_cursor_policy(policy, samples, step=0)
    # Both should hit their respective bboxes
    assert result.accuracy == 1.0
