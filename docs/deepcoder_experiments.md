# DeepCoder Experiment Runs

## Command Snippets Overview

| Condition | Purpose | Variables |
|--------|---------|-----------|
| `GRPO top-p 1.0` | Full-retention baseline under linear top-p filtering | `rollout_filter_strategy=top_p`, `rollout_filter_value=1`, `rollout_filter_include_zero=True` |
| `GRPO top-p 0.9` | Stronger reward-variance filtering with adaptive retention | `rollout_filter_strategy=top_p`, `rollout_filter_value=0.9`, `rollout_filter_include_zero=False` |
| `GRPO top-k 0.25` | Fixed-budget filtering that keeps the top 25% of train groups | `rollout_filter_strategy=top_k`, `rollout_filter_value=0.25`, `rollout_filter_include_zero=True` |

All three command snippets run DeepCoder with `Qwen/Qwen2.5-Coder-7B`, `GRPO`, and single-turn code generation.

---

## 1. Top-p 1.0 (`Qwen/Qwen2.5-Coder-7B-200-GRPO-top-p-1`)

Uses linear top-p filtering with `rollout_filter_value=1`.

Goal:
- Establish a full-retention baseline while keeping the same reward-variance ranking machinery as the filtered runs

Key Details:
- Filtering uses `top_p`, `largest`, `reward_variance`, and `rollout_filter_top_p_prob_mode=linear`
- `rollout_filter_value=1` with `rollout_filter_include_zero=True` keeps the full train-group pool under linear top-p selection, so this is the closest thing to a no-filter baseline in `deepcoder_lines`
- `actor_rollout_ref.actor.use_ref=False` and `actor_rollout_ref.actor.use_kl_loss=False` remove reference-policy KL from training
- The run budget is `200` training steps with checkpoints every `20` steps

Source:
- `docs/deepcoder_lines`, lines `1-89`

Outputs:
- W&B project: `deepcoder_RAGEN_final_3`
- Run name: `Qwen/Qwen2.5-Coder-7B-200-GRPO-top-p-1`

---

## 2. Top-p 0.9 (`Qwen/Qwen2.5-Coder-7B-200-GRPO-top-p-0.9`)

Uses the same linear top-p filter, but keeps only the highest-variance groups whose score mass reaches `0.9`.

Goal:
- Increase reward-variance filtering strength while keeping the rest of the GRPO setup fixed

Key Details:
- Filtering again uses `top_p`, `largest`, `reward_variance`, and `rollout_filter_top_p_prob_mode=linear`
- `rollout_filter_value=0.9` makes retention adaptive: the number of kept groups depends on how reward variance is distributed across the `16` train groups
- `rollout_filter_include_zero=False` excludes zero-variance groups from selection
- Because `rollout_filter_type=largest`, the filter prioritizes groups with the highest within-group reward variance

Source:
- `docs/deepcoder_lines`, lines `93-181`

Outputs:
- W&B project: `deepcoder_RAGEN_final_3`
- Run name: `Qwen/Qwen2.5-Coder-7B-200-GRPO-top-p-0.9`

---

## 3. Top-k 0.25 (`Qwen/Qwen2.5-Coder-7B-200-GRPO-top-k-0.25`)

Switches from adaptive top-p filtering to fixed-fraction top-k filtering.

Goal:
- Compare adaptive top-p filtering against a fixed keep-top-25% regime

Key Details:
- `rollout_filter_strategy=top_k` with `rollout_filter_value=0.25` keeps `int(0.25 * 16) = 4` train groups per step
- With `es_manager.train.group_size=8`, this corresponds to at most `32` kept rollouts per training step after filtering
- `rollout_filter_include_zero=True` means zero-variance groups are still part of the ranking pool, but only the top `4` groups survive
- `rollout_filter_type=largest` means those `4` groups are chosen by highest reward variance

Source:
- `docs/deepcoder_lines`, lines `185-273`

Outputs:
- W&B project: `deepcoder_RAGEN_final_3`
- Run name: `Qwen/Qwen2.5-Coder-7B-200-GRPO-top-k-0.25`

---

## Common Notes

- Source format:
  - `docs/deepcoder_lines` is a collection of three standalone bash snippets, not a parameterized sweep script
  - The file defines both `USE_GRPO` and `USE_PPO`, but all three `python train.py` commands actually expand `$USE_GRPO`
- Shared setup across all three conditions:
  - Config: `_10_deepcoder`
  - Model: `Qwen/Qwen2.5-Coder-7B`
  - `algorithm.adv_estimator=grpo`
  - `agent_proxy.reward_normalization.method=identity`
  - `trainer.total_training_steps=200`
  - `ppo_mini_batch_size=32`
  - `micro_batch_size_per_gpu=1`
  - `es_manager.train.env_groups=16`, `es_manager.train.group_size=8`
  - `es_manager.val.env_groups=256`, `es_manager.val.group_size=1`
  - `trainer.n_gpus_per_node=8`
  - `system.CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"`
  - `actor_rollout_ref.rollout.tensor_model_parallel_size=4`
  - `agent_proxy.max_turn=1`
  - `actor_rollout_ref.actor.use_ref=False`
  - `actor_rollout_ref.rollout.rollout_filter_type=largest`
  - `actor_rollout_ref.rollout.rollout_filter_metric=reward_variance` by default from `config/base.yaml`
  - `actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=linear`
  - `actor_rollout_ref.rollout.rollout_filter_empty_stop_steps=0`
  - `actor_rollout_ref.rollout.max_model_len=10000`
  - `actor_rollout_ref.rollout.max_num_batched_tokens=10000`
  - `actor_rollout_ref.rollout.response_length=4000`
  - `agent_proxy.fail_on_prompt_too_long=True`
  - `lora.rank=0`, `lora.alpha=64`, `lora.target_modules=all-linear`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.6`
  - `trainer.save_freq=20`
  - `trainer.validation_steps=1`
  - `trainer.val_before_train=True`
  - `trainer.test_freq=10`
  - `collapse_detection.first_turn_enabled=False`
  - `collapse_detection.multi_turn_enabled=False`
  - `trainer.resume_mode=disable`
- Logging and artifacts:
  - Default local log dir remains `results/`
  - Default logger remains `['console', 'wandb']`
  - Checkpoints are saved every `20` steps
