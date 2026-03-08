# FrozenLake Slipper Sweep Runs

This doc covers the experiment script for the FrozenLake slipper-rate sweep.

## Scripts Overview

| Script | Purpose | Variables |
|--------|---------|-----------|
| `run_frozen_lake_slipper_rate_sweep.sh` | Sweep FrozenLake stochasticity while comparing `filter` vs `nofilter` | `slipper_rate` (`100,50,20,10,5,2,0` by default), `filter`/`nofilter` |

The script runs FrozenLake with `Qwen2.5-3B`, `GAE`.

---

## 1. FrozenLake Slipper Sweep (`run_frozen_lake_slipper_rate_sweep.sh`)

Tracks how `filter` and `nofilter` success rates change as FrozenLake stochasticity varies via `slipper_rate`, using project `ragen_release_frozenlake_slipper_rate_sweep`.

Goal:
- Test whether RV-style filtering remains helpful as FrozenLake transition randomness changes

Key Details:
- `slipper_rate` is normalized to a ratio in `[0, 1]`, and the environment is configured with `success_rate = 1 - slipper_rate`
- Default comparison modes are both `filter` and `nofilter`
- This script explicitly fixes `rollout_filter_top_p_prob_mode=softmax`
- Mode mapping:
  - `filter`: `top_p=0.9` by default and `rollout_filter_include_zero=False`
  - `nofilter`: `top_p=1.0` by default and `rollout_filter_include_zero=True`

```bash
bash scripts/runs/run_frozen_lake_slipper_rate_sweep.sh
```

Options:
- `--steps` (default: `400`)
- `--slipper-rate` (comma list; accepts `100,50,20,10,5,2,0`, `1.0,0.5,...`, or `%`-suffixed values)
- `--filter-modes` (comma list; `filter`, `nofilter`, or both; default: both)
- `--filter-top-p` (default: `0.9`)
- `--nofilter-top-p` (default: `1.0`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--ray-num-cpus` (default: `16`)
- `--cooldown` (default: `30`)
- `--gpu-memory-utilization` (default: `0.5`)
- `--save-freq` (default: `-1`)

Examples:
```bash
# Run the full default sweep
bash scripts/runs/run_frozen_lake_slipper_rate_sweep.sh

# Run only `nofilter` on a custom subset of slipper rates
bash scripts/runs/run_frozen_lake_slipper_rate_sweep.sh --slipper-rate 50,20,5 --filter-modes nofilter --gpus 0 --cooldown 30 --ray-num-cpus 8

# Run one `filter` and one `nofilter` 50%-slipper experiment on 4xH100 each
bash scripts/runs/run_frozen_lake_slipper_rate_sweep.sh --slipper-rate 50 --gpus-per-exp 4 --gpus 0,1,2,3,4,5,6,7
```

Outputs:
- Per-run logs: `logs/frozenlake_slipper_rate_sweep_Qwen2.5-3B/<mode>/slip<label>/`
- Summary log: `logs/frozenlake_slipper_rate_sweep_Qwen2.5-3B.log`

---

## Common Notes

- Shared setup:
  - Config: `_3_frozen_lake`
  - Model: `Qwen/Qwen2.5-3B`
  - `algorithm.adv_estimator=gae`
  - `trainer.total_training_steps=400`
  - `trainer.save_freq=-1`
  - `trainer.logger=['console','wandb']`
  - `trainer.val_before_train=True`
  - `actor_rollout_ref.actor.loss_agg_mode=token-mean`
  - `actor_rollout_ref.actor.use_kl_loss=False`
  - `actor_rollout_ref.actor.kl_loss_type=low-var-kl`
  - `actor_rollout_ref.actor.kl_loss_coef=0`
  - `actor_rollout_ref.actor.entropy_coeff=0`
  - `actor_rollout_ref.actor.entropy_from_logits_with_chunking=True`
  - `actor_rollout_ref.actor.filter_loss_scaling=none`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=32`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`
  - `critic.ppo_mini_batch_size=32`
  - `critic.ppo_micro_batch_size_per_gpu=4`
  - `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8`
  - `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8`
  - `actor_rollout_ref.rollout.rollout_filter_strategy=top_p`
  - `actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=softmax`
  - `actor_rollout_ref.rollout.rollout_filter_type=largest`
  - `actor_rollout_ref.rollout.rollout_filter_metric=reward_variance`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.5`
  - `actor_rollout_ref.actor.checkpoint.save_contents=[model]`
  - `critic.checkpoint.save_contents=[model]`
- Input and naming conventions:
  - `slipper_rate` accepts `50`, `0.5`, and `50%` as equivalent inputs
  - Experiment labels use `slip<label>` with compact decimal formatting
  - Examples: `50% -> slip0p5`, `2% -> slip0p02`
- Comparison protocol:
  - Each slipper rate is run under both `filter` and `nofilter` unless `--filter-modes` restricts the set
  - With `--gpus-per-exp 4`, a 4-GPU list runs one experiment at a time; an 8-GPU list can run one `filter` and one `nofilter` experiment in parallel
- Base-config inheritance:
  - `algorithm.kl_ctrl.kl_coef` is not overridden in this script
