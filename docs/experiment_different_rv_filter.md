# Different Reward-Variance Filter Runs

This doc covers the comparison script:
- `scripts/runs/run_different_rv_filter_comparison.sh`

For the underlying rollout-filter logic, see [guide_rollout_filtering.md](guide_rollout_filtering.md).

## Script Overview

| Script | Purpose | Variables |
|--------|---------|-----------|
| `run_different_rv_filter_comparison.sh` | Compare reward-variance filter choices | `nofilter`, `linear_top_p_09`, `top_k_025` |

The script runs `Qwen2.5-3B` across 5 tasks (`sokoban`, `frozenlake`, `webshop`, `metamathqa`, `countdown`) and 4 algorithms (`PPO`, `DAPO`, `GRPO`, `DrGRPO`).

---

## Different RV Filters (`run_different_rv_filter_comparison.sh`)

Compares several reward-variance filtering schemes while keeping the rest of the training recipe aligned.

```bash
bash scripts/runs/run_different_rv_filter_comparison.sh --steps 400
```

Goal:
- Isolate the effect of different reward-variance filter strategies and keep/drop rules.

Key Details:
- All conditions use `rollout_filter_metric=reward_variance`.
- `nofilter` is implemented as `top_p=1.0` with `include_zero=True`; it keeps all groups while still going through the filter code path.
- `linear_top_p_09` uses `top_p=0.9`, `rollout_filter_top_p_prob_mode=linear`, `include_zero=False`, `selection_eps=0.01`.
- `top_k_025` uses `top_k=0.25`, `largest`, `include_zero=True`.
- The script keeps `actor_rollout_ref.actor.filter_loss_scaling=none`, so comparisons reflect filtering plus the algorithm's native loss aggregation rather than a separate kept-ratio correction.

Options:
- `--steps` (default: `400`)
- `--tasks` (comma list; default: `sokoban,frozenlake,webshop,metamathqa,countdown`)
- `--algos` (comma list; default: `PPO,DAPO,GRPO,DrGRPO`)
- `--filters` (comma list; `nofilter`, `linear_top_p_09`, `top_k_025`, or `all`; default: `all`)
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
bash scripts/runs/run_different_rv_filter_comparison.sh

# Compare RV filters for GRPO on Sokoban using 2 GPUs per experiment
bash scripts/runs/run_different_rv_filter_comparison.sh \
  --steps 200 \
  --tasks sokoban \
  --algos GRPO \
  --filters nofilter,linear_top_p_09,top_k_025 \
  --gpus-per-exp 2 \
  --gpus 0,1,2,3,4,5
```

Outputs:
- Per-task logs: `logs/different_rv_filter_<task>_Qwen2.5-3B/`
- Per-run result summaries: `logs/different_rv_filter_<task>_Qwen2.5-3B/<exp_name>.result`
- Summary log: `logs/different_rv_filter_Qwen2.5-3B.log`
- Checkpoints: `model_saving/different_rv_filter_Qwen2.5-3B/<task>/<algo>/<filter>/<exp_name>/`
- W&B project: `ragen_release_different_rv_filter`

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
  - Each run writes both a raw `.log` file and a one-line `.result` summary with task, algo, filter condition, batch settings, timing, GPU label, and final status
