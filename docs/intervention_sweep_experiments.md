# Intervention Sweep Runs

This doc covers the experiment scripts for the intervention sweep experiments.

## Scripts Overview

| Script | Purpose | Variables |
|--------|---------|-----------|
| `run_top_p_sweep.sh` | Sweep RV-filter strength | `rollout_filter_value` (`1.0,0.98,0.95,0.9,0.8,0.6,0.4,nofilter`) |
| `run_kl_sweep.sh` | Sweep KL regularization | `kl_loss_coef` (`0,0.001,0.003,0.01,0.03,0.1`) |
| `run_entropy_sweep.sh` | Sweep entropy regularization | `entropy_coeff` (`0,0.001,0.003,0.01,0.03,0.1`) |

All three scripts run Sokoban with `Qwen2.5-3B`, `GAE`.

---

## 1. Top-p Sweep (`run_top_p_sweep.sh`)

Scans `actor_rollout_ref.rollout.rollout_filter_value` on Sokoban.

Goal:
- Isolate the effect of RV-filter strength while keeping KL and entropy lightly enabled at `0.001`

Key Details:
- Reward-variance filtering scores each env group by the standard deviation of rollout rewards within the group
- Selection uses `top_p`, `largest`, `reward_variance`, and explicitly fixes `rollout_filter_top_p_prob_mode=softmax`
- Filtered groups are dropped as whole groups, and this sweep keeps `filter_loss_scaling=none`
- `1.0` and `nofilter` are different conditions: `1.0` still uses `include_zero=False`, while `nofilter` sets `include_zero=True`

Options:
- `--steps` (default: `400`)
- `--rollout_filter_value` (comma list; default: `1.0,0.98,0.95,0.9,0.8,0.6,0.4,nofilter`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--ray-num-cpus` (default: `16`)
- `--gpu-memory-utilization` (default: `0.5`)
- `--save-freq` (default: `-1`)

Examples:
```bash
# Run the full default sweep
bash scripts/runs/run_top_p_sweep.sh

# Run one `0.9` point and one `nofilter` point on 4xH100 each
bash scripts/runs/run_top_p_sweep.sh --rollout_filter_value 0.9,nofilter --gpus-per-exp 4 --gpus 0,1,2,3,4,5,6,7
```

Outputs:
- Per-value logs: `logs/top_p_sweep_Qwen2.5-3B/<value_label>/`
- Summary log: `logs/top_p_sweep_Qwen2.5-3B.log`

---

## 2. KL Sweep (`run_kl_sweep.sh`)

Scans `actor_rollout_ref.actor.kl_loss_coef` on Sokoban.

Goal:
- Isolate the effect of KL regularization while fixing entropy to `0` and keeping RV-filter effectively off with `rollout_filter_value=1`

Key Details:
- KL is computed token-wise between the current policy and a frozen reference policy worker
- This sweep uses the actor KL loss, not reward-level KL shaping
- When `kl_loss_coef=0`, the script also sets `use_kl_loss=False`, so the ref-policy forward pass is skipped
- Increasing `kl_loss_coef` penalizes drift from the reference policy more strongly

Options:
- `--steps` (default: `400`)
- `--kl-values` (comma list; default: `0,0.001,0.003,0.01,0.03,0.1`)
- `--rollout_filter_include_zero` (bool; default: `True`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--ray-num-cpus` (default: `16`)
- `--gpu-memory-utilization` (default: `0.5`)
- `--save-freq` (default: `-1`)

Examples:
```bash
# Run the full default sweep
bash scripts/runs/run_kl_sweep.sh

# Run two KL points on 4xH100 each, with zero-variance groups excluded
bash scripts/runs/run_kl_sweep.sh --kl-values 0,0.01 --rollout_filter_include_zero False --gpus-per-exp 4 --gpus 0,1,2,3,4,5,6,7
```

Outputs:
- Per-value logs: `logs/kl_sweep_Qwen2.5-3B/<filter_tag>/<value_label>/`
- Summary log: `logs/kl_sweep_Qwen2.5-3B.log`

---

## 3. Entropy Sweep (`run_entropy_sweep.sh`)

Scans `actor_rollout_ref.actor.entropy_coeff` on Sokoban.

Goal:
- Isolate the effect of entropy regularization while fixing KL to `0` and keeping RV-filter effectively off with `rollout_filter_value=1`

Key Details:
- Entropy is computed token-wise over the full vocabulary on response tokens
- Aggregation is `token-mean`, so the sweep compares average token-level exploration pressure
- The entropy term enters the actor loss with a negative sign, so larger `entropy_coeff` encourages more exploration
- The script keeps `entropy_from_logits_with_chunking=True`, so large-vocabulary entropy is computed in a memory-friendly way

```bash
bash scripts/runs/run_entropy_sweep.sh
```

Options:
- `--steps` (default: `400`)
- `--entropy-values` (comma list; default: `0,0.001,0.003,0.01,0.03,0.1`)
- `--rollout_filter_include_zero` (bool; default: `True`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--ray-num-cpus` (default: `16`)
- `--gpu-memory-utilization` (default: `0.5`)
- `--save-freq` (default: `-1`)

Examples:
```bash
# Run the full default sweep
bash scripts/runs/run_entropy_sweep.sh

# Run two entropy points on 4xH100 each, with zero-variance groups excluded
bash scripts/runs/run_entropy_sweep.sh --entropy-values 0,0.01 --rollout_filter_include_zero False --gpus-per-exp 4 --gpus 0,1,2,3,4,5,6,7
```

Outputs:
- Per-value logs: `logs/entropy_sweep_Qwen2.5-3B/<filter_tag>/<value_label>/`
- Summary log: `logs/entropy_sweep_Qwen2.5-3B.log`

---

## Common Notes

- Comparability protocol:
  - The three Sokoban sweeps change only one intervention axis at a time
  - Top-p sweep scans RV-filter while fixing `use_kl_loss=True`, `kl_loss_coef=0.001`, and `entropy_coeff=0.001`
  - KL sweep scans `kl_loss_coef` while keeping entropy off and filtering off
  - Entropy sweep scans `entropy_coeff` while keeping KL off and filtering off
- Training budget and early stopping:
  - Each condition runs for at most `400` PPO steps with `8` train env groups and `16` rollouts per group
  - Runs may stop early if reward variance collapses for long enough or if validation success stays below the failure threshold for repeated validations
  - Early stopping is part of the comparison protocol: if a setting stops early, that run is treated as a failed training regime rather than a fully budgeted run
- Shared setup across all three sweeps:
  - Config: `_2_sokoban`
  - Model: `Qwen/Qwen2.5-3B`
  - `algorithm.adv_estimator=gae`
  - `trainer.total_training_steps=400`
  - `trainer.save_freq=-1`
  - `trainer.logger=['console','wandb']`
  - `trainer.val_before_train=True`
  - `actor_rollout_ref.actor.filter_loss_scaling=none`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=32`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`
  - `critic.ppo_mini_batch_size=32`
  - `critic.ppo_micro_batch_size_per_gpu=4`
  - `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8`
  - `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8`
  - `es_manager.train.env_groups=8`, `es_manager.train.group_size=16`
  - `es_manager.val.env_groups=512`, `es_manager.val.group_size=1`
- Rollout filter settings used by these sweeps:
  - `rollout_filter_strategy=top_p`
  - `rollout_filter_top_p_prob_mode=softmax`
  - `rollout_filter_type=largest`
  - `rollout_filter_metric=reward_variance`
- Top-p sweep uses two distinct `top_p=1.0` conditions:
  - `1.0`: `rollout_filter_value=1.0`, `rollout_filter_include_zero=False`
  - `nofilter`: `rollout_filter_value=1.0`, `rollout_filter_include_zero=True`
- KL sweep and Entropy sweep default to `rollout_filter_include_zero=True`; if you pass `--rollout_filter_include_zero False`, logs are written under `filter_zero/` instead of `nofilter/`
- You can run a single sweep point on `4xH100` by setting `--gpus-per-exp 4` and passing a 4-GPU list, or run two sweep points in parallel by passing an 8-GPU list
