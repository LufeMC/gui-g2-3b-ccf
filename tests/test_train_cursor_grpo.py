"""End-to-end training-loop tests with MockLogprobPolicy."""

import json
import os
import sys

import pytest
from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cursor_dataset import CursorDataset  # noqa: E402
from cursor_policy import MockLogprobPolicy  # noqa: E402
from train_cursor_grpo import TrainConfig, train  # noqa: E402


def _make_dataset(tmp_path, num_samples=2):
    """Build a tiny on-disk dataset of white images + arbitrary bboxes."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    jsonl = tmp_path / "samples.jsonl"
    with open(jsonl, "w") as f:
        for i in range(num_samples):
            img_path = img_dir / f"img_{i}.png"
            Image.new("RGB", (200, 200), (255, 255, 255)).save(img_path)
            rec = {
                "img_path": str(img_path),
                "instruction": f"click target {i}",
                "abs_box": [90, 90, 110, 110],
            }
            f.write(json.dumps(rec) + "\n")
    return CursorDataset(str(jsonl))


def _make_policy(num_trajectories):
    """Scripted policy: every trajectory moves somewhere and stops.

    We build a bank of `num_trajectories` scripts with varying behaviors so
    each trajectory within a group has a different reward (avoiding zero
    variance).
    """
    # Alternate between a clean hit and a near-miss to create reward variance.
    scripts = []
    for i in range(num_trajectories):
        if i % 2 == 0:
            scripts.append([
                "<action>move(100, 100)</action>",  # center of bbox
                "<action>stop</action>",
            ])
        else:
            scripts.append([
                "<action>move(50, 50)</action>",  # outside bbox
                "<action>stop</action>",
            ])
    return MockLogprobPolicy(scripts)


def test_runs_two_steps_without_error(tmp_path):
    dataset = _make_dataset(tmp_path, num_samples=2)
    # 2 prompts * 2 trajectories = 4 scripts needed
    policy = _make_policy(num_trajectories=4)
    cfg = TrainConfig(
        num_steps=2,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=str(tmp_path / "ckpts"),
        save_every=0,
        log_every=0,
    )
    state = train(policy, dataset, cfg)
    assert len(state.metrics) == 2
    for m in state.metrics:
        assert m.num_trajectories == 4


def test_optimizer_actually_updates_parameters(tmp_path):
    dataset = _make_dataset(tmp_path, num_samples=2)
    policy = _make_policy(num_trajectories=4)
    initial_offset = policy.offset.detach().clone()

    cfg = TrainConfig(
        num_steps=2,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        lr=1e-2,  # big enough to see movement on fake logprobs
        # Pin to the Phase 5 reward shape (no stop_bonus, no time_penalty)
        # so we're testing the optimizer + grad-flow plumbing, not the
        # reward shape itself. The new shaping terms create symmetric
        # gradients that exactly cancel for the alternating hit/miss
        # MockLogprobPolicy used here, which would mask grad-flow bugs
        # behind reward symmetry.
        stop_bonus_when_inside=0.0,
        time_penalty_per_step=0.0,
        output_dir=str(tmp_path / "ckpts"),
        save_every=0,
        log_every=0,
    )
    train(policy, dataset, cfg)
    assert not policy.offset.detach().equal(initial_offset), (
        "AdamW should have moved offset; if this fails, grad isn't flowing"
    )


def test_loss_is_finite(tmp_path):
    dataset = _make_dataset(tmp_path, num_samples=2)
    policy = _make_policy(num_trajectories=4)
    cfg = TrainConfig(
        num_steps=2,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=str(tmp_path / "ckpts"),
        save_every=0,
        log_every=0,
    )
    state = train(policy, dataset, cfg)
    for m in state.metrics:
        assert m.loss == m.loss  # not NaN
        assert abs(m.loss) < 1e3  # not exploded


def test_reward_stats_are_collected(tmp_path):
    dataset = _make_dataset(tmp_path, num_samples=2)
    policy = _make_policy(num_trajectories=4)
    cfg = TrainConfig(
        num_steps=1,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=str(tmp_path / "ckpts"),
        save_every=0,
        log_every=0,
    )
    state = train(policy, dataset, cfg)
    m = state.metrics[0]
    # Good + bad trajectories -> std > 0, mean in (0, 1)
    assert m.reward_std > 0
    assert 0.0 < m.reward_mean < 1.0


def test_checkpoints_written_at_save_interval(tmp_path):
    dataset = _make_dataset(tmp_path, num_samples=2)
    policy = _make_policy(num_trajectories=4)
    output_dir = str(tmp_path / "ckpts")
    cfg = TrainConfig(
        num_steps=2,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=output_dir,
        save_every=1,
        log_every=0,
    )
    state = train(policy, dataset, cfg)
    assert len(state.checkpoints) == 2
    for ckpt in state.checkpoints:
        assert os.path.isdir(ckpt)
        # Mock policy drops a marker file
        assert os.path.exists(os.path.join(ckpt, "MOCK_CHECKPOINT"))


def test_train_summary_written(tmp_path):
    dataset = _make_dataset(tmp_path, num_samples=2)
    policy = _make_policy(num_trajectories=4)
    output_dir = str(tmp_path / "ckpts")
    cfg = TrainConfig(
        num_steps=1,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=output_dir,
        save_every=0,
        log_every=0,
    )
    train(policy, dataset, cfg)
    summary_path = os.path.join(output_dir, "train_summary.json")
    assert os.path.exists(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary["num_steps"] == 1
    assert "config" in summary


def test_progress_json_is_written_each_step(tmp_path):
    dataset = _make_dataset(tmp_path, num_samples=2)
    policy = _make_policy(num_trajectories=4)
    output_dir = str(tmp_path / "ckpts")
    cfg = TrainConfig(
        num_steps=3,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=output_dir,
        save_every=0,
        log_every=0,
    )
    train(policy, dataset, cfg)
    progress_path = os.path.join(output_dir, "progress.json")
    assert os.path.exists(progress_path)
    with open(progress_path) as f:
        progress = json.load(f)
    assert progress["num_steps_completed"] == 3
    assert progress["num_steps_planned"] == 3
    assert progress["step"] == 2  # 0-indexed
    assert len(progress["metrics_history"]) == 3
    assert progress["last_metrics"]["step"] == 2


def test_in_training_eval_hook_runs_at_interval(tmp_path):
    """When eval_every and val_dataset are set, eval_history fills up."""
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    dataset = _make_dataset(tmp_path, num_samples=2)
    val_dataset = _make_dataset(val_dir, num_samples=2)
    # Need scripts for both rollout (4 trajectories * 2 steps) AND eval (2 samples).
    # MockLogprobPolicy wraps to fallback when scripts are exhausted, so be safe.
    policy = _make_policy(num_trajectories=12)
    output_dir = str(tmp_path / "ckpts")
    cfg = TrainConfig(
        num_steps=2,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=output_dir,
        save_every=0,
        log_every=0,
        eval_every=1,  # eval after every step
    )
    state = train(policy, dataset, cfg, val_dataset=val_dataset)
    # Two steps + eval_every=1 => two eval entries
    assert len(state.eval_history) == 2
    for ev in state.eval_history:
        assert "accuracy" in ev
        assert "parse_rate" in ev
        assert ev["num_samples"] == 2

    # Eval history is also serialized into progress.json
    with open(os.path.join(output_dir, "progress.json")) as f:
        progress = json.load(f)
    assert len(progress["eval_history"]) == 2


def test_in_training_eval_disabled_when_eval_every_is_zero(tmp_path):
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    dataset = _make_dataset(tmp_path, num_samples=2)
    val_dataset = _make_dataset(val_dir, num_samples=2)
    policy = _make_policy(num_trajectories=8)
    cfg = TrainConfig(
        num_steps=2,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=str(tmp_path / "ckpts"),
        save_every=0,
        log_every=0,
        eval_every=0,
    )
    state = train(policy, dataset, cfg, val_dataset=val_dataset)
    assert state.eval_history == []


def test_progress_json_is_atomic(tmp_path):
    """Killing the process between writes should never produce a torn JSON file.

    We can't actually kill in-process, so we verify the no-tmp-leftover
    invariant: after a normal completion, no progress.json.tmp should exist.
    """
    dataset = _make_dataset(tmp_path, num_samples=2)
    policy = _make_policy(num_trajectories=4)
    output_dir = str(tmp_path / "ckpts")
    cfg = TrainConfig(
        num_steps=2,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=output_dir,
        save_every=0,
        log_every=0,
    )
    train(policy, dataset, cfg)
    assert not os.path.exists(os.path.join(output_dir, "progress.json.tmp"))


def test_log_callback_fires(tmp_path):
    dataset = _make_dataset(tmp_path, num_samples=2)
    policy = _make_policy(num_trajectories=4)
    cfg = TrainConfig(
        num_steps=3,
        prompts_per_step=2,
        trajectories_per_prompt=2,
        max_steps_per_trajectory=4,
        output_dir=str(tmp_path / "ckpts"),
        save_every=0,
        log_every=1,
    )
    recorded = []
    train(policy, dataset, cfg, log_callback=lambda m: recorded.append(m))
    assert len(recorded) == 3
