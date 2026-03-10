# Search Environment (HotpotQA + Dense Retrieval)

A multi-turn search environment for training LLM agents on multi-hop question answering. The agent interacts with a retrieval server to search Wikipedia and answer questions from HotpotQA.

## Overview

The agent receives a question and can take two types of actions:
- `search[query]` — retrieve relevant Wikipedia passages via dense retrieval (E5 + FAISS)
- `finish[answer]` — submit a final answer

Reward is computed using F1 / exact match against the HotpotQA ground truth.

## Components

| Component | Description |
|-----------|-------------|
| `env.py` | Gym environment: parses actions, calls retrieval server, computes reward |
| `config.py` | Dataclass config (`SearchEnvConfig`) |
| `reward.py` | F1 / EM reward computation |
| `retrieval_client.py` | HTTP client for the retrieval server |
| `scripts/retrieval/server.py` | Flask server: E5 encoder + FAISS index over Wikipedia |

## Setup

### 1. Prepare data

```bash
# Download HotpotQA → data/search/{train,val}.parquet
python scripts/prepare_search_data.py

# Download Wikipedia corpus + FAISS index (~74GB) → search_data/prebuilt_indices/
python scripts/download_search_index.py
```

### 2. Start the retrieval server

The retrieval server provides dense retrieval over ~21M Wikipedia passages using E5-base-v2 embeddings and a FAISS Flat index.

```bash
python scripts/retrieval/server.py \
    --data_dir ./search_data/prebuilt_indices \
    --port 8000 --host 127.0.0.1 \
    --device cuda:0 --gpu_memory_limit_mb 6144
```

Loading the 61GB FAISS index takes 2-5 minutes. Verify with:

```bash
curl http://127.0.0.1:8000/health
```

**Important: GPU deployment recommendation**

We recommend running the retrieval server on a **dedicated GPU** separate from training. In our experiments, placing the E5 server on the same GPU as training (e.g., GPU 0) caused CUDA OOM errors — vLLM rollout and training both compete for GPU memory, squeezing out the retrieval server process.

Run the server on a GPU not used by training (e.g., `--device cuda:7`, train on GPUs 0-6). During rollout, hundreds of environments issue concurrent retrieval requests (e.g., 256 env groups can produce 1000+ requests). Running on CPU cannot keep up with this concurrency and causes timeouts. A dedicated GPU with `threading.Lock` serialization handles this load reliably.

### 3. Run training

```bash
# PPO, no filtering (baseline)
bash scripts/runs/run_search_benchmark.sh \
    --algos PPO \
    --gpus 0,1,2,3,4,5,6,7 --gpus-per-exp 8

# PPO, top_k filtering (keep top 25% by reward variance)
bash scripts/runs/run_search_benchmark.sh \
    --algos PPO --filter-strategy top_k --filter-value 0.25 \
    --gpus 0,1,2,3,4,5,6,7 --gpus-per-exp 8

# PPO, top_p filtering (keep 90% by reward variance)
bash scripts/runs/run_search_benchmark.sh \
    --algos PPO --filter-strategy top_p --filter-value 0.9 \
    --gpus 0,1,2,3,4,5,6,7 --gpus-per-exp 8
```

Key training parameters (pass via `run_search_benchmark.sh` flags):

| Flag | Description | Default |
|------|-------------|---------|
| `--algos` | Algorithm: PPO or GRPO | PPO |
| `--filter-strategy` | Rollout filter strategy: `top_p`, `top_k`, etc. | `top_p` |
| `--filter-value` | Filter value (1.0 = no filtering) | `1.0` |
| `--gpus` | Comma-separated GPU IDs | auto-detect |
| `--gpus-per-exp` | GPUs per experiment | 1 |
| `--gpu-memory-utilization` | vLLM KV cache memory fraction | 0.6 |
| `--micro-batch` | Micro batch size per GPU | config default |
| `--mini-batch` | PPO mini batch size | config default |
| `--save-freq` | Checkpoint save frequency | -1 (disabled) |
| `--steps` | Total training steps | 200 |
| `--retrieval-port` | Retrieval server port | 8000 |

## Config

The search environment config is at `config/_10_search.yaml`. Key settings:

```yaml
micro_batch_size_per_gpu: 4
ppo_mini_batch_size: 32

agent_proxy:
  max_turn: 5              # up to 5 search rounds
  max_actions_per_turn: 1  # one action per response

actor_rollout_ref:
  rollout:
    max_model_len: 5000
    max_num_batched_tokens: 5000

es_manager:
  train:
    env_groups: 16
    group_size: 8           # 16 × 8 = 128 samples per batch
```

Note: when using `top_k` filtering with a small value (e.g., 0.25), the effective batch size after filtering is `env_groups × group_size × filter_value`. Ensure `ppo_mini_batch_size` does not exceed this value, or training will fail with an assertion error.
