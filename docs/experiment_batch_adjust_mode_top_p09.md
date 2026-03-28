# Batch-Adjust Mode Runs Under Linear Top-p 0.9

This doc covers the comparison script:
- `scripts/runs/run_batch_adjust_mode_top_p09.sh`

For the underlying rollout-filter and pad/loss-scaling logic, see [guide_rollout_filtering.md](guide_rollout_filtering.md) and [guide_filtering_and_loss_scaling.md](guide_filtering_and_loss_scaling.md).

## Script Overview

| Script | Purpose | Variables |
|--------|---------|-----------|
| `run_batch_adjust_mode_top_p09.sh` | Compare batch-adjust behavior under fixed linear top-p `0.9` filtering | `copy`, `delete`, `pad` |

The script runs `Qwen2.5-3B` across 5 tasks (`sokoban`, `frozenlake`, `webshop`, `metamathqa`, `countdown`) and 4 algorithms (`PPO`, `DAPO`, `GRPO`, `DrGRPO`).

---

## Batch-Adjust Modes (`run_batch_adjust_mode_top_p09.sh`)

Fixes reward-variance filtering to `linear top-p 0.9` and compares how the trainer handles non-divisible filtered batches.

```bash
bash scripts/runs/run_batch_adjust_mode_top_p09.sh --steps 400
```

Goal:
- Isolate the effect of `copy`, `delete`, and `pad` when filtering keeps an irregular number of trajectories.

Key Details:
- The filter is fixed to:
  - `rollout_filter_strategy=top_p`
  - `rollout_filter_value=0.9`
  - `rollout_filter_top_p_prob_mode=linear`
  - `rollout_filter_type=largest`
  - `rollout_filter_metric=reward_variance`
  - `rollout_filter_include_zero=False`
  - `rollout_filter_selection_eps=0.01`
- `copy` duplicates samples to reach the next valid divisor.
- `delete` drops samples to reach the previous valid divisor.
- `pad` keeps the real batch intact and pads only the optimizer batch later with zero-loss rows.
- The script still accepts `add` as a legacy alias, but it maps to `copy`.
- In the current trainer, if `delete` removes all samples for a step, that step is skipped and logged as `train/empty_after_adjust`; training stops early after 5 consecutive such steps.

Options:
- `--steps` (default: `400`)
- `--tasks` (comma list; default: `sokoban,frozenlake,webshop,metamathqa,countdown`)
- `--algos` (comma list; default: `PPO,DAPO,GRPO,DrGRPO`)
- `--batch-adjust-modes` (comma list; `copy`, `delete`, `pad`, `add`, or `all`; default: `all`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--cooldown` (seconds; default: `30`)
- `--gpu-memory-utilization` (default: `0.3`)
- `--ray-num-cpus` (default: `16`)
- `--save-freq` (default: `-1`)
- `--ppo-micro-batch-size-per-gpu` (default: `4`)
- `--log-prob-micro-batch-size-per-gpu` (default: `4`)

Examples:
```bash
# Run the full default comparison
bash scripts/runs/run_batch_adjust_mode_top_p09.sh

# Compare copy/delete/pad for GRPO on Sokoban using 2 GPUs per experiment
bash scripts/runs/run_batch_adjust_mode_top_p09.sh \
  --steps 200 \
  --tasks sokoban \
  --algos GRPO \
  --batch-adjust-modes copy,delete,pad \
  --gpus-per-exp 2 \
  --gpus 0,1,2,3,4,5
```

Outputs:
- Per-task logs: `logs/batch_adjust_mode_top_p09_<task>_Qwen2.5-3B/`
- Per-run result summaries: `logs/batch_adjust_mode_top_p09_<task>_Qwen2.5-3B/<exp_name>.result`
- Summary log: `logs/batch_adjust_mode_top_p09_Qwen2.5-3B.log`
- Checkpoints: `model_saving/batch_adjust_mode_top_p09_Qwen2.5-3B/<task>/<algo>/<mode>/<exp_name>/`
- W&B project: `ragen_release_batch_adjust_top_p09`

---

## Common Notes

- Shared fixed setup:
  - model: `Qwen/Qwen2.5-3B`
  - tasks: `sokoban,frozenlake,webshop,metamathqa,countdown`
  - algorithms: `PPO,DAPO,GRPO,DrGRPO`
  - `trainer.total_training_steps=400` by default
  - `trainer.save_freq=-1`
  - `trainer.logger=['console','wandb']`
  - `trainer.val_before_train=True`
  - `trainer.gradient_analysis_mode=False`
  - `trainer.log_group_rv_table=True`
  - `actor_rollout_ref.actor.use_kl_loss=False`
  - `actor_rollout_ref.actor.kl_loss_type=low-var-kl`
  - `actor_rollout_ref.actor.kl_loss_coef=0.001`
  - `actor_rollout_ref.actor.entropy_coeff=0.001`
  - `actor_rollout_ref.actor.entropy_from_logits_with_chunking=True`
  - `actor_rollout_ref.actor.filter_loss_scaling=none`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`
  - `critic.ppo_micro_batch_size_per_gpu=4`
  - `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4`
  - `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4`
- Algorithm mapping:
  - `PPO`: `algorithm.adv_estimator=gae`, `loss_agg_mode=token-mean`
  - `DAPO`: `algorithm.adv_estimator=gae`, `loss_agg_mode=token-mean`, plus asymmetric clip settings and KL-in-reward disabled
  - `GRPO`: `algorithm.adv_estimator=grpo`, `algorithm.norm_adv_by_std_in_grpo=True`, `loss_agg_mode=seq-mean-token-mean`
  - `DrGRPO`: `algorithm.adv_estimator=grpo`, `algorithm.norm_adv_by_std_in_grpo=False`, `loss_agg_mode=seq-mean-token-sum`
- Task-specific note:
  - For `frozenlake`, the script additionally sets `custom_envs.CoordFrozenLake.env_config.success_rate=1.0`
- GPU scheduling behavior:
  - If `--gpus` is omitted, the script tries to auto-detect GPUs with `nvidia-smi`
  - If auto-detection fails, it falls back to `0,1,2,3,4,5,6,7`
  - `--gpus-per-exp` controls how many GPUs each experiment gets
  - When you provide multiple GPU groups, the script launches one worker queue per group and runs experiments in parallel
- Result files:
  - Each run writes both a raw `.log` file and a one-line `.result` summary with task, algo, adjust mode, batch settings, timing, GPU label, and final status
