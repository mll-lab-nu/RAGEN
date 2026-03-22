# Main Table Runs

This doc covers the experiment scripts for the main performance table.

## Scripts Overview

| Script | Purpose | Variables |
|--------|---------|-----------|
| `run_main_table_diff_algo.sh` | Compare algorithms | PPO, DAPO, GRPO, DrGRPO; `filter`/`nofilter` |
| `run_main_table_diff_size.sh` | Compare model sizes | 0.5B, 1.5B, 3B, 7B; `filter`/`nofilter` |
| `run_main_table_diff_model.sh` | Compare model types | Instruct, Reasoning; `filter`/`nofilter` |

All scripts run experiments across 5 tasks (sokoban, frozenlake, webshop, metamathqa, countdown) with filter/nofilter settings.

---

## 1. Different Algorithms (`run_main_table_diff_algo.sh`)

Compares PPO/DAPO/GRPO/DrGRPO using Qwen2.5-3B.

```bash
bash scripts/runs/run_main_table_diff_algo.sh --steps 400
```

Options:
- `--steps` (default: `400`)
- `--tasks` (comma list; default: `sokoban,frozenlake,webshop,metamathqa,countdown`)
- `--algos` (comma list; default: `PPO,DAPO,GRPO,DrGRPO`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--cooldown` (seconds; default: `30`)
- `--gpu-memory-utilization` (default: `0.3`)
- `--filters` (comma list; `filter`, `nofilter`, or `all`; default: `all`)

Examples:
```bash
# Run one `filter` and one `nofilter` PPO experiment on 4xH100 each
bash scripts/runs/run_main_table_diff_algo.sh --steps 400 --tasks sokoban --gpus-per-exp 4 --gpu-memory-utilization 0.3 --filters all --gpus 0,1,2,3,4,5,6,7 --algos PPO
```

Outputs:
- Per-task logs: `logs/diff_algo_<task>_Qwen2.5-3B/`
- Summary log: `logs/diff_algo_Qwen2.5-3B.log`

---

## 2. Different Model Sizes (`run_main_table_diff_size.sh`)

Compares Qwen2.5 models of different sizes using PPO.

```bash
bash scripts/runs/run_main_table_diff_size.sh --steps 400
```

Options:
- `--steps` (default: `400`)
- `--tasks` (comma list; default: `sokoban,frozenlake,webshop,metamathqa,countdown`)
- `--models` (comma list; default: `Qwen2.5-0.5B,Qwen2.5-1.5B,Qwen2.5-3B,Qwen2.5-7B`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--cooldown` (seconds; default: `30`)
- `--gpu-memory-utilization` (default: `0.3`)
- `--filters` (comma list; `filter`, `nofilter`, or `all`; default: `all`)

Examples:
```bash
# Run a single 1.5B/filter experiment on 4xH100
bash scripts/runs/run_main_table_diff_size.sh --steps 400 --tasks sokoban --gpus-per-exp 4 --gpu-memory-utilization 0.3 --filters filter --gpus 0,1,2,3 --models Qwen2.5-1.5B

# Quick test with smallest model
bash scripts/runs/run_main_table_diff_size.sh --steps 5 --models Qwen2.5-0.5B --tasks sokoban
```

Outputs:
- Per-task logs: `logs/diff_size_<task>/`
- Summary log: `logs/diff_size_PPO.log`

---

## 3. Different Model Types (`run_main_table_diff_model.sh`)

Compares different model types (Instruct, Reasoning) using PPO.

```bash
bash scripts/runs/run_main_table_diff_model.sh --steps 400
```

Options:
- `--steps` (default: `400`)
- `--tasks` (comma list; default: `sokoban,frozenlake,webshop,metamathqa,countdown`)
- `--models` (comma list; default: `Qwen2.5-3B-Instruct`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--cooldown` (seconds; default: `30`)
- `--gpu-memory-utilization` (default: `0.3`)
- `--filters` (comma list; `filter`, `nofilter`, or `all`; default: `all`)

Examples:
```bash
# Run one `filter` and one `nofilter` Llama-3.2-3B-Instruct experiment on 4xH100 each
bash scripts/runs/run_main_table_diff_model.sh --steps 400 --tasks sokoban --gpus-per-exp 4 --gpu-memory-utilization 0.5 --filters all --gpus 0,1,2,3,4,5,6,7 --models=meta-llama/Llama-3.2-3B-Instruct
```

Outputs:
- Per-task logs: `logs/diff_model_<task>/`
- Summary log: `logs/diff_model_PPO.log`

---

## Common Notes

- Effective rollout filter config for main-table runs:
  - `rollout_filter_strategy=top_p`
  - `rollout_filter_top_p_prob_mode=softmax`
  - `rollout_filter_type=largest`
  - `rollout_filter_metric=reward_variance`
  - `rollout_filter_include_zero=True`
- Filter mode mapping:
  - `filter`: `top_p=0.9`, `include_zero=True`
  - `nofilter`: `top_p=1.0`, `include_zero=True`
- Because `include_zero=True`, `nofilter` (`top_p=1.0`) keeps all groups; it does not disable the filter code path, but it is effectively "no filtering" for the batch
- You can run a single experiment on `4xH100` by setting `--gpus-per-exp 4` and passing a 4-GPU list, or run one `filter` and one `nofilter` experiment in parallel by passing an 8-GPU list
