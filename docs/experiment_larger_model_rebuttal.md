# Larger-Model Rebuttal Runs

This doc covers the experiment script for the larger-model rebuttal runs.

## Script Overview

| Script | Purpose | Variables |
|--------|---------|-----------|
| `run_larger_model_rebuttal.sh` | Run the larger-model rebuttal setting | current defaults: `Qwen2.5-14B`, `sokoban`, `filter`/`nofilter` |

---

## 1. Larger-Model Rebuttal (`run_larger_model_rebuttal.sh`)

Runs PPO larger-model rebuttal experiments.

```bash
bash scripts/runs/run_larger_model_rebuttal.sh --steps 400
```

Current defaults in the script:
- task: `sokoban`
- model: `Qwen2.5-14B`
- GPUs per experiment: `4`
- rollout `gpu_memory_utilization`: `0.3`

Options:
- `--steps` (default: `400`)
- `--tasks` (comma list; current default: `sokoban`)
- `--models` (comma list; current default: `Qwen2.5-14B`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `4`)
- `--cooldown` (seconds; default: `30`)
- `--gpu-memory-utilization` (default: `0.3`)
- `--ray-num-cpus` (default: `16`)
- `--save-freq` (default: `-1`)
- `--filters` (comma list; `filter`, `nofilter`, or `all`; default: `all`)

Examples:
```bash
# Run both `filter` and `nofilter` on 8 GPUs, 4 GPUs per experiment
bash scripts/runs/run_larger_model_rebuttal.sh \
  --steps 400 \
  --tasks sokoban \
  --models Qwen2.5-14B \
  --gpus-per-exp 4 \
  --gpus 0,1,2,3,4,5,6,7 \
  --filters all

# Quick smoke test on a single 4-GPU slot
bash scripts/runs/run_larger_model_rebuttal.sh \
  --steps 10 \
  --tasks sokoban \
  --models Qwen2.5-14B \
  --gpus 0,1,2,3 \
  --filters filter
```

Outputs:
- Per-task logs: `logs/diff_size_<task>/`
- Summary log: `logs/diff_size_PPO.log`
- Checkpoints: `model_saving/diff_size/<task>/<model>/<filter>/`

---

## Common Notes

- This script currently writes into the same `diff_size` output roots used by the main size sweep.
- PPO is fixed; this script does not sweep algorithms.
- Rebuttal-specific train overrides include:
  - `es_manager.train.group_size=8`
  - `collapse_detection.first_turn_enabled=False`
  - `collapse_detection.multi_turn_enabled=False`
- Performance-related overrides include:
  - `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
  - `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1`
  - `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1`
  - `critic.ppo_micro_batch_size_per_gpu=1`
  - gradient checkpointing enabled
- Effective rollout filter config for this script:
  - `rollout_filter_strategy=top_p`
  - `rollout_filter_top_p_prob_mode=linear`
  - `rollout_filter_type=largest`
  - `rollout_filter_metric=reward_variance`
  - `rollout_filter_selection_eps=0.01`
- Filter mode mapping:
  - `filter`: `top_p=0.9`, `include_zero=False`
  - `nofilter`: `top_p=1.0`, `include_zero=True`
- The `wandb` project name is `ragen_rebuttal_larger_models`.
- Validation before training is enabled: `trainer.val_before_train=True`.
