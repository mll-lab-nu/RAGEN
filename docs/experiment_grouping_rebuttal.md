# Grouping Rebuttal Runs

This doc covers the experiment script for the grouping rebuttal runs.

## Script Overview

| Script | Purpose | Variables |
|--------|---------|-----------|
| `run_different_grouping_rebuttal.sh` | Compare different `env_groups × group_size` layouts | `8x16`, `128x1`, `64x2`, `32x4`; `filter`/`nofilter` |

---

## 1. Grouping Rebuttal (`run_different_grouping_rebuttal.sh`)

Runs PPO grouping-layout rebuttal experiments on Sokoban with Qwen2.5-3B.

```bash
bash scripts/runs/run_different_grouping_rebuttal.sh --steps 400
```

Current defaults in the script:
- task: `sokoban`
- model: `Qwen2.5-3B`
- group layouts: `8x16`, `128x1`, `64x2`, `32x4`
- GPUs per experiment: `1`

Options:
- `--steps` (default: `400`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--cooldown` (seconds; default: `30`)
- `--gpu-memory-utilization` (default: `0.3`)
- `--ray-num-cpus` (default: `16`)
- `--save-freq` (default: `-1`)
- `--filters` (comma list; `filter`, `nofilter`, or `all`; default: `all`)
- `--group-layouts` (comma list; default: `all`)
- `--batch-sizes` is still accepted as a backward-compatible alias for `--group-layouts`

Examples:
```bash
# Run all layouts with both `filter` and `nofilter`
bash scripts/runs/run_different_grouping_rebuttal.sh --steps 400

# Run two layouts with `nofilter` only on two GPUs
bash scripts/runs/run_different_grouping_rebuttal.sh \
  --steps 400 \
  --filters nofilter \
  --group-layouts 8x16,128x1 \
  --gpus 0,1

# Quick single-layout smoke test
bash scripts/runs/run_different_grouping_rebuttal.sh \
  --steps 20 \
  --filters filter \
  --group-layouts 64x2 \
  --gpus 0
```

Outputs:
- Per-run logs: `logs/diff_grouping_sokoban_Qwen2.5-3B/`
- Summary log: `logs/diff_grouping_Qwen2.5-3B.log`
- Checkpoints: `model_saving/diff_grouping_Qwen2.5-3B/sokoban/<filter>/<layout>/`

---

## Common Notes

- This is a grouping-layout ablation, not a batch-size ablation.
- The default layouts all keep the same total number of samples:
  - `8x16 = 128`
  - `128x1 = 128`
  - `64x2 = 128`
  - `32x4 = 128`
- PPO is fixed; this script does not sweep algorithms.
- Task and model are fixed:
  - task: `sokoban`
  - model: `Qwen2.5-3B`
- Effective rollout filter config for this script:
  - `rollout_filter_strategy=top_p`
  - `rollout_filter_top_p_prob_mode=softmax`
  - `rollout_filter_type=largest`
  - `rollout_filter_metric=reward_variance`
  - `rollout_filter_include_zero=True`
- Filter mode mapping:
  - `filter`: `top_p=0.9`
  - `nofilter`: `top_p=1.0`
- Because `include_zero=True`, `nofilter` (`top_p=1.0`) keeps all groups and behaves as effective no-filtering.
- The `wandb` project name is `ragen_rebuttal_grouping`.
- Validation before training is enabled: `trainer.val_before_train=True`.
