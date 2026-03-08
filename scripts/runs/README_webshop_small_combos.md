# Webshop Small Combos — Running Guide

## Combos

The script defines 4 model×algorithm combinations:

| Combo | Model | Algorithm |
|-------|-------|-----------|
| 1 | Qwen2.5-3B-Instruct | PPO |
| 2 | Qwen2.5-3B-Instruct | GRPO |
| 3 | Qwen2.5-7B-Instruct | PPO |
| 4 | Llama-3.2-3B-Instruct | PPO |

Use `--combos` to select which ones to run (1-indexed, comma-separated). Default: all.

## Filter Modes

- `--filters filter` → top_p=0.9 (rollout filtering enabled)
- `--filters nofilter` → top_p=1.0 (no filtering)
- `--filters all` → runs both (default)

## Multi-GPU for One Experiment

Use `--gpus-per-exp N` to assign multiple GPUs to a single experiment. The GPUs listed in `--gpus` are split into groups of N.

Example: 2 GPUs per experiment on GPUs 0-3:
```bash
bash scripts/runs/run_webshop_small_combos.sh --gpus 0,1,2,3 --gpus-per-exp 2 --combos 3
# Creates 2 slots: [0,1] and [2,3]
```

## Example: Running All Experiments (8 GPUs)

Run these in separate terminals (disjoint GPUs, safe to run in parallel):

```bash
# Terminal 1 — 7B filter (GPUs 0,1)
bash scripts/runs/run_webshop_small_combos.sh --steps 200 --gpus 0,1 --gpus-per-exp 2 --filters filter --combos 3

# Terminal 2 — 7B nofilter (GPUs 2,3)
bash scripts/runs/run_webshop_small_combos.sh --steps 200 --gpus 2,3 --gpus-per-exp 2 --filters nofilter --combos 3

# Terminal 3 — 3B Qwen PPO nofilter (GPU 4)
bash scripts/runs/run_webshop_small_combos.sh --steps 200 --gpus 4 --filters nofilter --combos 1

# Terminal 4 — 3B Qwen GRPO nofilter (GPU 5)
bash scripts/runs/run_webshop_small_combos.sh --steps 200 --gpus 5 --filters nofilter --combos 2

# Terminal 5 — 3B Llama PPO nofilter (GPU 6)
bash scripts/runs/run_webshop_small_combos.sh --steps 200 --gpus 6 --filters nofilter --combos 4
```

## Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | 400 | Training steps |
| `--gpus LIST` | auto-detect | Comma-separated GPU IDs |
| `--gpus-per-exp N` | 1 | GPUs per experiment |
| `--combos LIST` | all | Which combos to run (1-4) |
| `--filters LIST` | all | filter, nofilter, or all |
| `--save-freq N` | 200 | Checkpoint save frequency |
| `--cooldown N` | 30 | Seconds between runs on same GPU slot |
| `--gpu-memory-utilization V` | 0.2 | vLLM rollout GPU memory fraction |

## Output

- Logs: `logs/webshop_small_combos/<name>.log`
- Results: `logs/webshop_small_combos/<name>.result`
- Checkpoints: `model_saving/webshop_small_combos/<model>/<algo>/<filter>/<name>/`
- Wandb project: `main_webshop`
