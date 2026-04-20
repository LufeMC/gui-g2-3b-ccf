"""Multi-step GRPO training loop for GUI-Cursor.

Works with any policy that satisfies the LogprobCursorPolicy protocol --
the MockLogprobPolicy for Mac-side testing, and the VLMCursorPolicy (Phase 3b)
for real training on GPU.

Pipeline per optimizer step:
  1. Sample `prompts_per_step` examples from the dataset
  2. For each: roll out `trajectories_per_prompt` trajectories, collecting
     CompletionOutput per step so we have token IDs + generation-time logprobs
  3. Compute trajectory rewards and group-normalized advantages
  4. For each trajectory, teacher-force logprobs under the current policy
     and the reference policy, then compute the GRPO loss with KL penalty
  5. Backprop + optimizer step
  6. Every `save_every` steps, save a checkpoint (if policy supports it)
  7. Every `log_every` steps, print + optionally log to W&B
"""

import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from cursor_actions import parse_action
from cursor_advantages import group_advantages
from cursor_dataset import CursorDataset, CursorSample
from cursor_env import CursorEnv
from cursor_loss import DEFAULT_CLIP_EPS, DEFAULT_KL_BETA, grpo_loss
from cursor_policy import CompletionOutput
from cursor_rewards import (
    DEFAULT_PENALTY_WEIGHT,
    DEFAULT_REPEAT_TOL,
    Trajectory,
    trajectory_reward,
)


@dataclass
class TrainConfig:
    """Knobs for a single `train()` invocation."""
    num_steps: int = 100
    prompts_per_step: int = 4
    trajectories_per_prompt: int = 4
    max_steps_per_trajectory: int = 4
    lr: float = 1e-6
    kl_beta: float = DEFAULT_KL_BETA
    clip_eps: float = DEFAULT_CLIP_EPS
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT
    repeat_tol: int = DEFAULT_REPEAT_TOL
    # Phase 6 reward shaping (the actual training defaults; the
    # reward function itself defaults to 0 to keep its existing tests
    # green). Set to 0 to revert to the Phase 5 reward shape.
    stop_bonus_when_inside: float = 0.3
    time_penalty_per_step: float = 0.05
    output_dir: str = "./checkpoints/gui-cursor"
    save_every: int = 50
    log_every: int = 1
    eval_every: int = 0  # 0 disables in-training eval
    seed: int = 42
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class StepMetrics:
    """Summary numbers for a single optimizer step; easy to serialize."""
    step: int
    loss: float
    policy_loss: float
    kl_loss: float
    clip_frac: float
    reward_mean: float
    reward_std: float
    advantage_mean: float
    wall_seconds: float
    num_trajectories: int
    trajectories_stopped_cleanly: int


@dataclass
class TrainState:
    """In-memory record of what happened during training."""
    metrics: List[StepMetrics] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    eval_history: List[Dict[str, Any]] = field(default_factory=list)


def _classify_termination(env_reason: str) -> str:
    if env_reason in ("stopped", "stopped_without_move"):
        return "stop"
    if env_reason == "invalid_action":
        return "invalid"
    if env_reason == "max_steps":
        return "max_steps"
    return env_reason


def run_trajectory_with_logprobs(
    env: CursorEnv,
    policy: Any,
    instruction: str,
) -> Tuple[Trajectory, List[CompletionOutput]]:
    """Drive an env to completion using policy.generate_with_logprobs.

    Returns the trajectory (for reward computation) and the per-step
    CompletionOutput list (for teacher-forcing and loss computation).
    """
    trajectory = Trajectory()
    completions: List[CompletionOutput] = []
    step_index = 0

    while True:
        image = env.render()
        comp = policy.generate_with_logprobs(image, instruction, step_index)
        completions.append(comp)

        action = parse_action(comp.text)
        trajectory.actions.append(action)

        result = env.step(action)
        trajectory.steps = result.step_index

        if action.kind == "move" and result.was_valid:
            assert result.position is not None
            trajectory.positions.append(result.position)

        if result.done:
            trajectory.final_position = env.position
            trajectory.stopped_via = _classify_termination(result.reason)
            break

        step_index += 1

    return trajectory, completions


def _gather_trajectory_logprobs(
    policy: Any,
    completions: Sequence[CompletionOutput],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher-force new/ref logprobs for one trajectory; also stack old ones.

    All three tensors are aligned on the same device/dtype so loss math
    doesn't trip over CPU/GPU or bf16/fp32 mismatches.
    """
    new_per_step = policy.teacher_force_logprobs(completions, ref_only=False)
    ref_per_step = policy.teacher_force_logprobs(completions, ref_only=True)

    new_lps = torch.cat(new_per_step) if new_per_step else torch.zeros(0)
    ref_lps = torch.cat(ref_per_step).detach() if ref_per_step else torch.zeros(0)

    target_device = new_lps.device
    target_dtype = new_lps.dtype
    ref_lps = ref_lps.to(device=target_device, dtype=target_dtype)

    old_lps = torch.cat(
        [
            torch.tensor(
                c.logprobs_at_generation,
                dtype=target_dtype,
                device=target_device,
            )
            for c in completions
        ]
    ).detach()

    return new_lps, old_lps, ref_lps


def _dump_progress(path: str, state: "TrainState", config: "TrainConfig") -> None:
    """Atomically write a JSON snapshot of training state.

    Mirrors what we did for the CCF eval -- if the pod dies mid-run, we
    can read this file to recover (a) where we were and (b) the metric
    history. Written every step because the payload is tiny (<10 KB even
    after 250 steps) and the cost of losing data is much larger than the
    cost of writing it.
    """
    payload = {
        "step": state.metrics[-1].step if state.metrics else -1,
        "num_steps_completed": len(state.metrics),
        "num_steps_planned": config.num_steps,
        "checkpoints": list(state.checkpoints),
        "last_metrics": (
            state.metrics[-1].__dict__ if state.metrics else None
        ),
        "metrics_history": [m.__dict__ for m in state.metrics],
        "eval_history": list(state.eval_history),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _save_checkpoint(policy: Any, path: str) -> bool:
    """Best-effort checkpoint save. Returns True if the policy actually saved."""
    os.makedirs(path, exist_ok=True)
    save_fn = getattr(policy, "save_checkpoint", None)
    if callable(save_fn):
        save_fn(path)
        return True
    # Mock policies don't implement save_checkpoint. Drop a marker file so
    # tests and the log still see that a save event happened.
    marker = os.path.join(path, "MOCK_CHECKPOINT")
    with open(marker, "w") as f:
        f.write("mock policy, no weights persisted\n")
    return False


def train(
    policy: Any,
    dataset: CursorDataset,
    config: TrainConfig,
    log_callback: Optional[Callable[[StepMetrics], None]] = None,
    val_dataset: Optional[CursorDataset] = None,
) -> TrainState:
    """Run `config.num_steps` optimizer steps. Returns an in-memory TrainState.

    If `val_dataset` is provided AND `config.eval_every > 0`, run a greedy
    in-training eval every `eval_every` steps and append the result to
    `state.eval_history` (also surfaced in the progress JSON).
    """
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    params = list(policy.parameters())
    if not params:
        raise ValueError("policy.parameters() is empty; cannot optimize")
    optimizer = torch.optim.AdamW(params, lr=config.lr)

    os.makedirs(config.output_dir, exist_ok=True)
    state = TrainState()
    progress_path = os.path.join(config.output_dir, "progress.json")

    # Optional W&B integration. We import lazily so the file still works
    # if wandb isn't installed (e.g. during local Mac unit tests).
    wandb_run = None
    if config.wandb_project:
        try:
            import wandb  # type: ignore
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={k: v for k, v in config.__dict__.items()},
                reinit=True,
            )
            print(f"[wandb] logging to project {config.wandb_project} run {wandb_run.name}")
        except Exception as exc:
            print(f"[wandb] failed to init ({exc}); continuing without it")
            wandb_run = None

    for step_idx in range(config.num_steps):
        t0 = time.time()
        batch = dataset.sample(config.prompts_per_step, rng=rng)

        all_trajectories: List[Trajectory] = []
        all_completions: List[List[CompletionOutput]] = []
        all_rewards: List[float] = []

        for sample in batch:
            for _ in range(config.trajectories_per_prompt):
                env = CursorEnv(
                    sample.image,
                    max_steps=config.max_steps_per_trajectory,
                    repeat_tol=config.repeat_tol,
                )
                traj, completions = run_trajectory_with_logprobs(
                    env, policy, sample.instruction
                )
                reward = trajectory_reward(
                    traj,
                    target_bbox=sample.target_bbox,
                    image_size=sample.image.size,
                    penalty_weight=config.penalty_weight,
                    repeat_tol=config.repeat_tol,
                    stop_bonus_when_inside=config.stop_bonus_when_inside,
                    time_penalty_per_step=config.time_penalty_per_step,
                )
                all_trajectories.append(traj)
                all_completions.append(completions)
                all_rewards.append(reward.total)

                advance = getattr(policy, "next_trajectory", None)
                if callable(advance):
                    advance()

        advantages = group_advantages(all_rewards, config.trajectories_per_prompt)

        optimizer.zero_grad()

        # Per-trajectory backward (gradient accumulation). Doing it this
        # way means we never hold more than ONE trajectory's compute graph
        # in GPU memory at once -- the previous approach stacked all
        # `prompts_per_step * trajectories_per_prompt` graphs and OOM'd
        # at the default config on a 143GB H200. The optimizer.step()
        # at the end applies the accumulated gradient.
        n_traj_with_loss = sum(1 for c in all_completions if c)
        loss_scale = 1.0 / max(n_traj_with_loss, 1)

        per_traj_metrics: List[Dict[str, float]] = []
        for completions, adv in zip(all_completions, advantages):
            if not completions:
                continue
            new_lps, old_lps, ref_lps = _gather_trajectory_logprobs(policy, completions)
            if new_lps.numel() == 0:
                continue
            loss_dict = grpo_loss(
                logprobs_new=new_lps,
                logprobs_old=old_lps,
                logprobs_ref=ref_lps,
                advantage=adv,
                clip_eps=config.clip_eps,
                kl_beta=config.kl_beta,
            )
            # Scale + backward NOW so the graph for this trajectory can be
            # freed before we build the next one. Gradients accumulate in
            # the policy's parameters via .grad fields.
            (loss_dict["loss"] * loss_scale).backward()

            per_traj_metrics.append({
                "loss": float(loss_dict["loss"].item()),
                "policy_loss": float(loss_dict["policy_loss"].item()),
                "kl_loss": float(loss_dict["kl_loss"].item()),
                "clip_frac": float(loss_dict["clip_frac"].item()),
            })

        if per_traj_metrics:
            optimizer.step()
            loss_val = statistics.fmean(d["loss"] for d in per_traj_metrics)
            policy_loss_val = statistics.fmean(d["policy_loss"] for d in per_traj_metrics)
            kl_val = statistics.fmean(d["kl_loss"] for d in per_traj_metrics)
            clip_val = statistics.fmean(d["clip_frac"] for d in per_traj_metrics)
        else:
            loss_val = policy_loss_val = kl_val = clip_val = 0.0

        reward_std = (
            statistics.pstdev(all_rewards) if len(all_rewards) > 1 else 0.0
        )
        metrics = StepMetrics(
            step=step_idx,
            loss=loss_val,
            policy_loss=policy_loss_val,
            kl_loss=kl_val,
            clip_frac=clip_val,
            reward_mean=statistics.fmean(all_rewards) if all_rewards else 0.0,
            reward_std=reward_std,
            advantage_mean=statistics.fmean(advantages) if advantages else 0.0,
            wall_seconds=time.time() - t0,
            num_trajectories=len(all_trajectories),
            trajectories_stopped_cleanly=sum(
                1 for t in all_trajectories if t.stopped_via == "stop"
            ),
        )
        state.metrics.append(metrics)

        if config.log_every and step_idx % config.log_every == 0:
            print(
                f"step {metrics.step:>4} "
                f"loss {metrics.loss:+.4f} "
                f"policy {metrics.policy_loss:+.4f} "
                f"kl {metrics.kl_loss:+.4f} "
                f"reward_mean {metrics.reward_mean:+.3f} "
                f"reward_std {metrics.reward_std:.3f} "
                f"clip {metrics.clip_frac:.2f} "
                f"wall {metrics.wall_seconds:.2f}s"
            )
            if log_callback is not None:
                log_callback(metrics)

        if wandb_run is not None:
            wandb_run.log({
                "train/loss": metrics.loss,
                "train/policy_loss": metrics.policy_loss,
                "train/kl_loss": metrics.kl_loss,
                "train/clip_frac": metrics.clip_frac,
                "train/reward_mean": metrics.reward_mean,
                "train/reward_std": metrics.reward_std,
                "train/advantage_mean": metrics.advantage_mean,
                "train/wall_seconds": metrics.wall_seconds,
                "train/trajectories_stopped_cleanly": metrics.trajectories_stopped_cleanly,
            }, step=step_idx)

        if config.save_every and (step_idx + 1) % config.save_every == 0:
            ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{step_idx + 1}")
            _save_checkpoint(policy, ckpt_dir)
            state.checkpoints.append(ckpt_dir)

        # In-training validation eval. We do this AFTER the optimizer step
        # so the printed eval reflects the latest weights.
        if (
            config.eval_every
            and val_dataset is not None
            and (step_idx + 1) % config.eval_every == 0
        ):
            from cursor_eval_hook import evaluate_cursor_policy
            val_samples = [val_dataset[i] for i in range(len(val_dataset))]
            eval_result = evaluate_cursor_policy(
                policy,
                val_samples,
                step=step_idx,
                max_steps=config.max_steps_per_trajectory,
                repeat_tol=config.repeat_tol,
            )
            state.eval_history.append(eval_result.__dict__)
            print(
                f"  [eval@{step_idx + 1}] "
                f"acc {eval_result.accuracy:.1%} "
                f"parse {eval_result.parse_rate:.1%} "
                f"steps {eval_result.mean_steps:.2f} "
                f"(n={eval_result.num_samples})"
            )
            if wandb_run is not None:
                wandb_run.log({
                    "eval/accuracy": eval_result.accuracy,
                    "eval/parse_rate": eval_result.parse_rate,
                    "eval/mean_steps": eval_result.mean_steps,
                }, step=step_idx)

        # Dump every step. JSON serialization of ~10 floats per step is
        # negligible vs the seconds-long forward pass we just ran.
        _dump_progress(progress_path, state, config)

    # Write a small run summary for post-hoc inspection.
    summary_path = os.path.join(config.output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": config.__dict__,
                "num_steps": len(state.metrics),
                "final_reward_mean": (
                    state.metrics[-1].reward_mean if state.metrics else None
                ),
                "checkpoints": state.checkpoints,
            },
            f,
            indent=2,
        )

    if wandb_run is not None:
        wandb_run.finish()

    return state


def main():  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Train GUI-Cursor via multi-step GRPO."
    )
    parser.add_argument("--policy", choices=["mock", "vlm"], required=True)
    parser.add_argument("--base-model", type=str, default=None,
                        help="Path to base VLM (required if --policy vlm)")
    parser.add_argument("--data", type=str, required=True,
                        help="JSONL file with img_path/instruction/abs_box")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--trajectories-per-prompt", type=int, default=4)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-beta", type=float, default=DEFAULT_KL_BETA)
    parser.add_argument("--clip-eps", type=float, default=DEFAULT_CLIP_EPS)
    parser.add_argument("--stop-bonus", type=float, default=0.3,
                        help="Phase 6: bonus reward when STOP fires inside the bbox. "
                             "Counterweights the false-stop penalty so 1-step "
                             "solutions are strictly preferred.")
    parser.add_argument("--time-penalty", type=float, default=0.05,
                        help="Phase 6: per-step penalty (-time_penalty * trajectory.steps). "
                             "Makes shorter trajectories strictly preferred.")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/gui-cursor")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run greedy validation eval every N steps. 0 disables.")
    parser.add_argument("--val-data", type=str, default=None,
                        help="JSONL of held-out cursor val samples for in-training eval.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=1_003_520)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--attn-impl", type=str, default="sdpa")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmstart-adapter", type=str, default=None,
                        help="Path to an SFT adapter to warm-start GRPO from.")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Optional W&B project name. If unset, no W&B logging.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Optional W&B run name. Defaults to W&B's auto-generated.")
    args = parser.parse_args()

    dataset = CursorDataset(args.data, max_samples=args.max_samples)
    print(f"Loaded {len(dataset)} samples from {args.data}")

    if args.policy == "mock":
        from cursor_policy import MockLogprobPolicy
        # Repeat a sensible scripted rollout across all trajectories.
        script = [
            "<action>move(100, 100)</action>",
            "<action>stop</action>",
        ]
        num_scripts = args.prompts_per_step * args.trajectories_per_prompt * args.num_steps
        policy = MockLogprobPolicy([script] * num_scripts)
    else:
        if not args.base_model:
            parser.error("--base-model is required when --policy=vlm")
        from cursor_vlm_policy import VLMCursorPolicy, VLMPolicyConfig
        cfg = VLMPolicyConfig(
            base_model_path=args.base_model,
            max_pixels=args.max_pixels,
            attn_implementation=args.attn_impl,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            warmstart_adapter_path=args.warmstart_adapter,
        )
        policy = VLMCursorPolicy(cfg)
        if args.warmstart_adapter:
            print(f"Loaded VLM from {args.base_model} with SFT adapter {args.warmstart_adapter}")
        else:
            print(f"Loaded VLM from {args.base_model} with fresh LoRA rank {args.lora_r}")

    config = TrainConfig(
        num_steps=args.num_steps,
        prompts_per_step=args.prompts_per_step,
        trajectories_per_prompt=args.trajectories_per_prompt,
        max_steps_per_trajectory=args.max_steps_per_trajectory,
        lr=args.lr,
        kl_beta=args.kl_beta,
        clip_eps=args.clip_eps,
        stop_bonus_when_inside=args.stop_bonus,
        time_penalty_per_step=args.time_penalty,
        output_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    val_dataset = None
    if args.val_data and args.eval_every > 0:
        val_dataset = CursorDataset(args.val_data)
        print(f"Loaded {len(val_dataset)} val samples from {args.val_data}")

    train(policy, dataset, config, val_dataset=val_dataset)
    print(f"Training complete. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
