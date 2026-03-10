# Search Environment Experiments

This doc covers the experiment scripts for the Search (HotpotQA + Dense Retrieval) environment.

## Overview

All experiments use:
- **Task**: SearchQA (HotpotQA multi-hop QA with Wikipedia dense retrieval)
- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Algorithm**: PPO (`algorithm.adv_estimator=gae`)
- **Config**: `_10_search`

The sweep compares three rollout filtering strategies while keeping all other hyperparameters fixed.

| Experiment | Filter Strategy | Filter Value | Effective Batch | Description |
|-----------|----------------|-------------|----------------|-------------|
| No Filter | `top_p` | `1.0` | 128 | Baseline: all rollout groups kept |
| TopK 0.25 | `top_k` | `0.25` | 32 | Keep top 25% groups by reward variance |
| TopP 0.9 | `top_p` | `0.9` | ~115 | Keep groups covering 90% cumulative reward variance |

---

## Prerequisites

### 1. Prepare data

```bash
# HotpotQA train/val parquet
python scripts/prepare_search_data.py

# Wikipedia corpus + FAISS index (~74GB)
python scripts/download_search_index.py
```

### 2. Start retrieval server

The retrieval server provides dense retrieval over ~21M Wikipedia passages using E5-base-v2 + FAISS.

```bash
python scripts/retrieval/server.py \
    --data_dir ./search_data/prebuilt_indices \
    --port 8000 --host 127.0.0.1 \
    --device cuda:0 --gpu_memory_limit_mb 6144
```

**Important**: We recommend running the retrieval server on a **dedicated GPU** not used by training, or on CPU. Sharing a GPU with vLLM rollout and training causes CUDA OOM errors due to memory contention between processes.

---

## Experiment Scripts

All experiments use `scripts/runs/run_search_benchmark.sh`.

### Experiment 1: PPO + No Filter (baseline)

No filtering — all rollout groups are used for training.

```bash
bash scripts/runs/run_search_benchmark.sh \
    --algos PPO \
    --filter-strategy top_p --filter-value 1.0 \
    --gpus 0,1,2,3,4,5,6,7 --gpus-per-exp 8 \
    --micro-batch 4 --mini-batch 64 \
    --gpu-memory-utilization 0.65 \
    --save-freq 20 --steps 200 \
    --retrieval-port 8000
```

### Experiment 2: PPO + TopK=0.25

Keep only the top 25% of rollout groups ranked by reward variance.

```bash
bash scripts/runs/run_search_benchmark.sh \
    --algos PPO \
    --filter-strategy top_k --filter-value 0.25 \
    --gpus 0,1,2,3,4,5,6,7 --gpus-per-exp 8 \
    --micro-batch 4 --mini-batch 32 \
    --gpu-memory-utilization 0.65 \
    --save-freq 20 --steps 200 \
    --retrieval-port 8000
```

Note: `mini-batch` is reduced to 32 because effective batch after filtering is `16 groups * 8 group_size * 0.25 = 32`. The `ppo_mini_batch_size` must not exceed this value.

### Experiment 3: PPO + TopP=0.9

Keep rollout groups covering the top 90% cumulative reward variance (softmax-weighted).

```bash
bash scripts/runs/run_search_benchmark.sh \
    --algos PPO \
    --filter-strategy top_p --filter-value 0.9 \
    --gpus 0,1,2,3,4,5,6,7 --gpus-per-exp 8 \
    --micro-batch 4 --mini-batch 64 \
    --gpu-memory-utilization 0.65 \
    --save-freq 20 --steps 200 \
    --retrieval-port 8000
```

---

## W&B Runs

Project: [`cuhksz-gc/ragen_search_benchmark`](https://wandb.ai/cuhksz-gc/ragen_search_benchmark)

| Experiment | Run ID | Link |
|-----------|--------|------|
| PPO + No Filter | `2sbt8952` | [wandb](https://wandb.ai/cuhksz-gc/ragen_search_benchmark/runs/2sbt8952) |
| PPO + TopK=0.25 | `2h5c7kbb` | [wandb](https://wandb.ai/cuhksz-gc/ragen_search_benchmark/runs/2h5c7kbb) |
| PPO + TopP=0.9 | `tbgx0lpt` | [wandb](https://wandb.ai/cuhksz-gc/ragen_search_benchmark/runs/tbgx0lpt) |

---

## Shared Config

```yaml
# config/_10_search.yaml overrides
micro_batch_size_per_gpu: 4
ppo_mini_batch_size: 32-64  # depends on filter setting

agent_proxy:
  max_turn: 5
  max_actions_per_turn: 1

actor_rollout_ref:
  rollout:
    max_model_len: 5000       # TopK=0.25 experiment used 4000
    max_num_batched_tokens: 5000  # TopK=0.25 experiment used 4000
    gpu_memory_utilization: 0.65
    temperature: 1
  actor:
    use_kl_loss: False
    kl_loss_coef: 0.001
    entropy_coeff: 0.001
    loss_agg_mode: token-mean
    filter_loss_scaling: none

es_manager:
  train:
    env_groups: 16
    group_size: 8    # 16 * 8 = 128 rollouts per step
  val:
    env_groups: 256

collapse_detection:
  compute_freq: 999  # effectively disabled

trainer:
  total_training_steps: 200
  save_freq: 20
  val_before_train: True
  logger: ['console', 'wandb']
```

---

## Common Notes

- **Retrieval server GPU deployment**: Place the E5 retrieval server on a **dedicated GPU** not used by training. Co-locating with training on the same GPU causes CUDA OOM due to memory contention between vLLM, training, and the E5 server process. Do not use CPU mode — during rollout, hundreds of environments issue concurrent retrieval requests (256 env groups can produce 1000+ requests), and CPU cannot keep up.
- **mini-batch size adjustment**: When using aggressive filtering (e.g., `top_k=0.25`), reduce `ppo_mini_batch_size` so it does not exceed `env_groups * group_size * filter_value`. Otherwise training fails with an assertion error.
- **max_model_len**: Default is 5000 (in `_10_search.yaml`). The TopK=0.25 experiment used 4000 to save KV cache memory; the No Filter and TopP=0.9 experiments use the default 5000.
- **Checkpoint size**: Each checkpoint is ~35GB (model + optimizer, 8 FSDP shards). With `save_freq=20` and 200 steps, expect 10 checkpoints (~350GB). Monitor disk usage and delete old checkpoints as needed.
