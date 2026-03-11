# WebShop Release Runs

This doc covers the current WebShop release runner.

## Overview

Use [`scripts/runs/run_webshop_release_combos.sh`](../scripts/runs/run_webshop_release_combos.sh) for the main WebShop release experiments.

Fixed setup:
- **Task**: WebShop
- **Config**: `_6_webshop`
- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Algorithm**: `GRPO`
- **Train batch**: `16` env groups x `8` samples
- **Validation batch**: `256` env groups x `1` sample
- **W&B project**: `main_webshop`

The runner compares four filter modes while keeping the rest of the setup fixed:

| Filter | Strategy | Value | Prob Mode | Description |
|--------|----------|-------|-----------|-------------|
| `topk25` | `top_k` | `0.25` | `linear` | Keep top 25% of groups by reward variance |
| `topp09` | `top_p` | `0.9` | `linear` | Keep groups covering 90% cumulative reward variance |
| `topp095` | `top_p` | `0.95` | `linear` | Keep groups covering 95% cumulative reward variance |
| `nofilter` | `top_p` | `1.0` | `linear` | Keep all groups |

All four modes use:
- `rollout_filter_type=largest`
- `rollout_filter_metric=reward_variance`
- `rollout_filter_include_zero=True`

## Setup

For WebShop, use the original base setup path:

```bash
bash scripts/setup_ragen.sh
bash scripts/setup_webshop.sh
```

Do not use `setup_ragen_release.sh` as the primary WebShop setup path.

Notes:
- `setup_webshop.sh` now treats the Google Drive full-data download as best-effort. A failure there does not mean the Python environment setup failed.
- The runner already pins `micro_batch_size_per_gpu=1` and `log_prob_micro_batch_size_per_gpu=1` to reduce WebShop rollout/log-prob memory pressure.

## Main Commands

### 4 GPUs

Run one experiment on 4 GPUs:

```bash
bash scripts/runs/run_webshop_release_combos.sh \
  --steps 400 \
  --gpus 0,1,2,3 \
  --gpus-per-exp 4 \
  --filters nofilter
```

Other filter modes:

```bash
bash scripts/runs/run_webshop_release_combos.sh \
  --steps 400 \
  --gpus 0,1,2,3 \
  --gpus-per-exp 4 \
  --filters topk25
```

```bash
bash scripts/runs/run_webshop_release_combos.sh \
  --steps 400 \
  --gpus 0,1,2,3 \
  --gpus-per-exp 4 \
  --filters topp09
```

```bash
bash scripts/runs/run_webshop_release_combos.sh \
  --steps 400 \
  --gpus 0,1,2,3 \
  --gpus-per-exp 4 \
  --filters topp095
```

### 8 GPUs

Run one experiment on 8 GPUs:

```bash
bash scripts/runs/run_webshop_release_combos.sh \
  --steps 400 \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus-per-exp 8 \
  --filters nofilter
```

You can swap `nofilter` for `topk25`, `topp09`, or `topp095`.

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | `100` | Total training steps |
| `--gpus LIST` | auto-detect | Comma-separated GPU IDs |
| `--gpus-per-exp N` | `1` | GPUs assigned to one experiment |
| `--save-freq N` | `100` | Checkpoint save frequency |
| `--gpu-memory-utilization V` | `0.3` | vLLM rollout memory fraction |
| `--filters LIST` | `all` | `topk25`, `topp09`, `topp095`, `nofilter`, or `all` |
| `--cooldown N` | `30` | Delay before reusing the same GPU slot |

## Outputs

- Logs: `logs/webshop_release_combos/<exp_name>.log`
- Per-run summaries: `logs/webshop_release_combos/<exp_name>.result`
- Checkpoints: `model_saving/webshop_release_combos/<model>/<algo>/<filter>/<exp_name>/`
- W&B project: `main_webshop`

Experiment names follow:

```text
webshop-release-GRPO-<filter-suffix>-Qwen2.5-3B-Instruct-16x8
```

Examples:
- `webshop-release-GRPO-topk25-Qwen2.5-3B-Instruct-16x8`
- `webshop-release-GRPO-topp09-linear-Qwen2.5-3B-Instruct-16x8`
- `webshop-release-GRPO-topp095-linear-Qwen2.5-3B-Instruct-16x8`
- `webshop-release-GRPO-nofilter-Qwen2.5-3B-Instruct-16x8`

## Related Files

- [`scripts/runs/run_webshop_release_combos.sh`](../scripts/runs/run_webshop_release_combos.sh)
- [`scripts/runs/run_webshop_small_combos.sh`](../scripts/runs/run_webshop_small_combos.sh)
- [`scripts/runs/README_webshop_small_combos.md`](../scripts/runs/README_webshop_small_combos.md)
