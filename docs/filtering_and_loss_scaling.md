# Filtering Strategies and Loss Scaling in RAGEN

## Overview
This document details the advanced filtering strategies and the loss scaling mechanism implemented to stabilize Reinforcement Learning (RL) training, particularly when using aggressive filtering techniques in the GRPO/PPO loop.

Note: the short guide for the current `top_p`, `top_k`, and no-filter variants lives in [docs/rollout_filtering.md](./rollout_filtering.md).

## 1. Rolling Filter Strategies (`rollout_filter_strategy`)
We have implemented three strategies to filter rollout groups based on their rewards/scores.

### `top_p` (Nucleus Sampling)
-   **Description**: Selects the smallest set of groups whose **cumulative probability** (derived from the softmax of scores) exceeds the threshold `value`.
-   **Configuration**:
    ```yaml
    actor_rollout_ref:
      rollout:
        rollout_filter_strategy: top_p
        rollout_filter_value: 0.5  # Keep top cumulative 50% probability mass
    ```
-   **Behavior**:
    -   Scores are converted to logits (negated if `rollout_filter_type: smallest`).
    -   Softmax is applied to get probabilities.
    -   Groups are sorted by probability.
    -   Groups are selected until the cumulative sum $\ge$ `value`.
    -   **Constraint**: Always keeps at least one group.

### `top_k`
-   **Description**: Selects the top fraction `value` (e.g., 0.5 for 50%) of groups.
-   **Configuration**:
    ```yaml
    actor_rollout_ref:
      rollout:
        rollout_filter_strategy: top_k
        rollout_filter_value: 0.5  # Keep top 50% groups
    ```

### `top_k_abs`
-   **Description**: Selects specifically the top `k` groups with the highest (or lowest) scores.
-   **Configuration**:
    ```yaml
    actor_rollout_ref:
      rollout:
        rollout_filter_strategy: top_k_abs
        rollout_filter_value: 4  # Keep top 4 groups
    ```
-   **Behavior**: A simple sorting and slicing operation. Useful for guaranteeing a fixed batch size of "good" examples.


### `min_p`
-   **Description**: Selects groups whose score is at least a fraction `value` of the maximum score in the batch.
-   **Behavior**:
    -   **`largest`**: Keeps groups where $\text{score} \ge \text{max\_score} \cdot \text{value}$.
    -   **`smallest`**: Keeps groups where $\text{score} \le \text{min\_score} / \text{value}$.
-   **Configuration**:
    ```yaml
    actor_rollout_ref:
      rollout:
        rollout_filter_strategy: min_p
        rollout_filter_value: 0.8  # Keep groups with score >= 0.8 * max_score
    ```

### Other Parameters
-   **`rollout_filter_metric`**: `reward_variance` (default), `reward`, `reward_sum`, `entropy`, `entropy_variance`, or `length`.
-   **`rollout_filter_type`**: `largest` (default) or `smallest`. Determines if we want high or low scores.
-   **`rollout_filter_include_zero`**: If `True`, groups with zero score are candidates for filtering. If `False`, they are excluded or handled differently depending on the specific logic (often used to ensure we don't train on complete failures).

---

## 2. Filter Loss Scaling (`filter_loss_scaling`)
Aggressive filtering (e.g., `top_p=0.2`) can result in keeping only a small fraction of the generated prompts. This can lead to high variance in gradients. To mitigate this, we implemented loss scaling.

### Concept
We scale the PPO policy loss (and potentially KL/entropy components depending on the implementation) by a factor derived from the **kept ratio**:
$$ \text{ratio} = \frac{N_{\text{kept}}}{N_{\text{total}}} $$

### Configuration
Controlled via `actor_rollout_ref.actor.filter_loss_scaling`:

1.  **`none`** (Default): No scaling.
    $$ \mathcal{L}_{\text{final}} = \mathcal{L}_{\text{ppo}} $$

2.  **`linear`**: Scales linearly with the kept ratio.
    $$ \mathcal{L}_{\text{final}} = \mathcal{L}_{\text{ppo}} \times \text{ratio} $$
    -   *Intuition*: If we only keep 10% of the data, we scale the update down by 10% to prevent over-fitting to this small subset.

3.  **`sqrt`**: Scales by the square root of the kept ratio.
    $$ \mathcal{L}_{\text{final}} = \mathcal{L}_{\text{ppo}} \times \sqrt{\text{ratio}} $$
    -   *Intuition*: A milder dampening than linear.

### Implementation Details
-   **Trainer**: The kept ratio is calculated in `ragen/trainer/agent_trainer.py`.
-   **Loss Scaling**: The scaling is applied directly to the **advantages** in `ragen/trainer/agent_trainer.py` (after `compute_advantage`).
    ```python
    if filter_loss_scaling == "linear":
        batch.batch["advantages"] *= filter_kept_ratio
    ```
    This effectively scales the policy gradient updates.

---

## 3. Reward Variance Early Stopping
To prevent training on collapsed or uninformative rollout groups, we implemented an early stopping mechanism based on reward variance.

### Concept
The trainer monitors the reward standard deviation (`rollout/in_group_reward_std`) at the successful training-step level.

1.  **Baseline Generation**: During the first 10 successful training steps, the trainer calculates the average reward variance ($V_{base}$).
2.  **Monitoring**: A sliding window of the last 10 successful training steps is maintained (starts after baseline is ready).
3.  **Stopping Condition**: If all 10 consecutive step variances are less than 10% of $V_{base}$, training is stopped.
    $$ \forall i \in \{1 \dots 10\}: V_i < 0.1 \times V_{base} \implies \text{Stop Training} $$

### Implementation
-   **Baseline**: Average of `rollout/in_group_reward_std` for `global_steps` 1-10.
-   **Sliding Window**: Uses a `collections.deque(maxlen=10)` to track the most recent successful training steps.
-   **Metric**: Logs `early_stopped/reward_variance_collapse: 1.0` when triggered.

### 2. Success-Based Early Stopping
To prevent wasting compute on environments where the model is failing to learn, we implemented an early stopping mechanism based on validation success rates.

- **Condition**: If the success rate for a specific environment (e.g., `val-env/CoordSokoban/success`) remains below **1% (0.01)** for **5 consecutive** validation steps, the training is stopped.
- **Metric**: Logs `early_stopped/low_validation_success: 1.0` when triggered.

---

A unified script `run_filtering_final.sh` is provided to run the validated set of filtering experiments.

### Usage
```bash
# Run experiments across available GPUs (e.g., 2 GPUs per experiment)
bash run_filtering_final.sh 2
```

### Features
-   **PPO Focused**: All experiments in this suite use the PPO algorithm.
-   **400 Steps**: Standardized training length.
-   **Auto-Scheduling**: Automatically detects available GPUs and distributes experiments.
-   **Metric Coverage**: Covers `reward_variance`, `entropy`, `entropy_variance`, and `length`.
-   **Automatic Skip**: Tracks progress in `filter_final_donelist.txt` to avoid redundant runs.

---

## 5. Legacy Scripts
The following scripts were used during the initial exploration phase:
- `run_filtering_exps.sh`: Initial algorithm comparison (GRPO vs PPO).
- `run_filtering_multigpu.sh`: Batch execution with comma-separated arguments.
- `run_filtering_pergpu.sh`: Parallel execution on individual GPUs.

### Directory Structure
Results are saved to `results/` with subdirectories named after the experiment:
`results/[DATE]_soko_3b_[ALGO]_[STRATEGY]_[VALUE]_inc[TRUE/FALSE]/`

---

## 5. Code References
-   **Filtering Logic**: `ragen/trainer/rollout_filter.py`
-   **Trainer Integration**: `ragen/trainer/agent_trainer.py`
-   **Early Stopping Logic**: `RayAgentTrainer` in `ragen/trainer/agent_trainer.py`
-   **Loss Scaling Implementation**: `verl/verl/workers/actor/dp_actor.py` (specifically `DataParallelPPOActor.update_policy`)
-   **Configuration**: `config/base.yaml` and `verl/verl/workers/config/actor.py`

---

## 6. Troubleshooting

### `AssertionError: old_log_probs` Collision
If you use `rollout_filter_metric=entropy`, you might encounter an `AssertionError` during the `batch.union` operation in `agent_trainer.py`.

-   **Cause**: The `EntropyRolloutFilter` recomputes log probabilities to calculate entropy and returns them in the `DataProto`. The trainer also recomputes log probabilities for the PPO update. `DataProto.union` rejects keys that already exist if they are not the exact same tensor instance.
-   **Resolution**: The filter has been updated to only include the `entropys` key and prune the redundant `old_log_probs` before unioning with the main batch.
