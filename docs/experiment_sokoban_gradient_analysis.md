# Sokoban Gradient Analysis Runs

This doc covers the helper scripts for the Sokoban top-p=0.9 gradient-analysis experiments.

For the internal execution order, metric definitions, and plotting workflow, see [guide_gradient_analysis.md](guide_gradient_analysis.md).

## Scripts Overview

| Script | Purpose | When to use |
|--------|---------|-------------|
| `run_sokoban_ppo_filter_grad_analysis.sh` | Train Sokoban with periodic gradient-analysis passes | Run this first to produce training logs and checkpoints |
| `run_sokoban_ppo_filter_grad_analysis_probe_ckpt.sh` | Resume a saved checkpoint and run one analysis-only probe | Run this after the training script when you want to inspect a specific checkpoint |

Both scripts run Sokoban with `Qwen2.5-3B`, reward-variance top-p filtering at `0.9`, and a separate gradient-analysis batch of `128x16`.

---

## Recommended Workflow

1. Start with `run_sokoban_ppo_filter_grad_analysis.sh`.
- This is the script that actually trains the policy, runs periodic gradient analysis, and writes the checkpoint layout that the probe script expects by default.

2. Choose a saved `global_step_*` checkpoint.
- The probe helper defaults to `global_step_101` under the checkpoint directory layout produced by the training script.
- If your run saved a different step, pass `--checkpoint-step` or `--resume-from-path`.

3. Run `run_sokoban_ppo_filter_grad_analysis_probe_ckpt.sh`.
- This reloads the checkpoint, runs one gradient-analysis pass, optionally performs a validation first, and then exits.

---

## 1. Periodic Training + Analysis (`run_sokoban_ppo_filter_grad_analysis.sh`)

Trains Sokoban and inserts gradient-analysis passes during training.

Goal:
- Follow the filtered Sokoban setup while logging gradient-analysis metrics at a fixed cadence on a larger analysis batch.

Key Details:
- Validation runs once before training and then every `10` steps.
- Gradient analysis runs every `50` steps. With the default `101` steps, the trigger points are `1`, `51`, and `101`.
- The normal training batch is `8` env groups x `16` samples.
- Gradient analysis uses a separate batch of `128` env groups x `16` samples.
- The run continues after analysis because `trainer.exit_after_gradient_analysis=False`.
- This script uses `top_p=0.9`, `rollout_filter_top_p_prob_mode=linear`, `rollout_filter_type=largest`, `rollout_filter_metric=reward_variance`, and `rollout_filter_include_zero=False`.
- `--algo PPO` selects `algorithm.adv_estimator=gae` and `actor_rollout_ref.actor.loss_agg_mode=token-mean`.
- `--algo GRPO` selects `algorithm.adv_estimator=grpo`, `algorithm.norm_adv_by_std_in_grpo=True`, and `actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean`.
- The training helper keeps `actor_rollout_ref.actor.use_kl_loss=False`, so it is meant for filtered training with periodic analysis rather than a KL-regularized sweep.

Examples:
```bash
# Default PPO run
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis.sh

# GRPO run on four GPUs
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis.sh --algo GRPO --gpus 0,1,2,3

# Short smoke test
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis.sh --steps 5 --gpus 0,1,2,3
```

Options:
- `--algo NAME` (`PPO` or `GRPO`; default: `PPO`)
- `--steps` (default: `101`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpu-memory-utilization` (default: `0.3`)
- `--ray-num-cpus` (default: `16`)
- `--ppo-micro-batch-size-per-gpu` (default: `4`)
- `--log-prob-micro-batch-size-per-gpu` (default: `4`)
- `--save-freq` (default: `100`)

Outputs:
- Per-run log: `logs/gradient_analysis_sokoban_Qwen2.5-3B/<exp_name>.log`
- Checkpoints: `model_saving/gradient_analysis/sokoban/<ALGO>/filter/<exp_name>/`
- W&B project: `ragen_gradient_analysis`

---

## 2. Checkpoint Probe (`run_sokoban_ppo_filter_grad_analysis_probe_ckpt.sh`)

Resumes a saved checkpoint and runs one gradient-analysis-only probe.

Goal:
- Inspect one checkpoint without continuing the normal training run.

Key Details:
- The script resumes from an existing `global_step_*` directory with `trainer.resume_mode=resume_path`.
- It runs in probe mode with `trainer.gradient_analysis_only=True`.
- It exits after the analysis pass because `trainer.exit_after_gradient_analysis=True`.
- By default it does not run validation first; add `--with-val` if you want a pre-probe validation.
- It uses the same Sokoban task, model, filter setup, and analysis batch shape as the training helper.
- Unlike the training helper, this probe sets `actor_rollout_ref.actor.use_kl_loss=True` together with `kl_loss_coef=0.001` and `entropy_coeff=0.001`, so the checkpoint probe explicitly logs KL and entropy gradient components.
- If `--resume-from-path` is given, that exact checkpoint directory is used. Otherwise the script resolves `<checkpoint-root>/global_step_<checkpoint-step>`.

Examples:
```bash
# Probe the default checkpoint layout produced by the training helper
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis_probe_ckpt.sh

# Probe a specific saved step with validation
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis_probe_ckpt.sh \
  --checkpoint-step 51 \
  --with-val \
  --gpus 0,1,2,3

# Probe an exact checkpoint path
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis_probe_ckpt.sh \
  --resume-from-path model_saving/gradient_analysis/sokoban/PPO/filter/<exp_name>/global_step_101 \
  --gpus 0,1,2,3
```

Options:
- `--algo NAME` (`PPO` or `GRPO`; default: `PPO`)
- `--checkpoint-step` (default: `101`)
- `--checkpoint-root DIR` (default: derived from the training helper's checkpoint layout)
- `--resume-from-path DIR` (exact `global_step_*` directory; overrides root + step resolution)
- `--with-val` (flag; default: off)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpu-memory-utilization` (default: `0.3`)
- `--ray-num-cpus` (default: `16`)
- `--ppo-micro-batch-size-per-gpu` (default: `4`)
- `--log-prob-micro-batch-size-per-gpu` (default: `4`)

Outputs:
- Per-run log: `logs/gradient_analysis_probe_sokoban_Qwen2.5-3B/<exp_name>.log`
- Probe output dir: `model_saving/gradient_analysis_probe/sokoban/<ALGO>/filter/<exp_name>/`
- W&B project: `ragen_gradient_analysis_probe`

---

## Common Notes

- Shared fixed setup:
  - config: `_2_sokoban`
  - model: `Qwen/Qwen2.5-3B`
  - training batch: `es_manager.train.env_groups=8`, `es_manager.train.group_size=16`
  - analysis batch: `trainer.gradient_analysis_env_groups=128`, `trainer.gradient_analysis_group_size=16`
  - `trainer.gradient_analysis_log_prefilter=True`
  - `actor_rollout_ref.rollout.gradient_analysis_num_buckets=6`
  - `actor_rollout_ref.rollout.gradient_analysis_bucket_mode=quantile`
- Shared rollout filter setup:
  - `actor_rollout_ref.rollout.rollout_filter_value=0.9`
  - `actor_rollout_ref.rollout.rollout_filter_strategy=top_p`
  - `actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=linear`
  - `actor_rollout_ref.rollout.rollout_filter_type=largest`
  - `actor_rollout_ref.rollout.rollout_filter_metric=reward_variance`
  - `actor_rollout_ref.rollout.rollout_filter_include_zero=False`
- GPU behavior:
  - if `--gpus` is omitted, the scripts try to auto-detect GPUs with `nvidia-smi`
  - if auto-detection fails, they fall back to `0,1,2,3,4,5,6,7`
- Directory relationship:
  - the training helper writes checkpoints under `model_saving/gradient_analysis/...`
  - the probe helper reads from that layout by default and writes its own outputs under `model_saving/gradient_analysis_probe/...`
- If you need the meaning of bucket metrics, prefilter logging, or the plotting commands after the run finishes, use [guide_gradient_analysis.md](guide_gradient_analysis.md).
