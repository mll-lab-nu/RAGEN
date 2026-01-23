# Filtering Strategies and Loss Scaling in RAGEN

## Overview
This document details the advanced filtering strategies and the loss scaling mechanism implemented to stabilize Reinforcement Learning (RL) training, particularly when using aggressive filtering techniques in the GRPO/PPO loop.

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
-   **Description**: Selects the top `k` groups with the highest (or lowest) scores.
-   **Configuration**:
    ```yaml
    actor_rollout_ref:
      rollout:
        rollout_filter_strategy: top_k
        rollout_filter_value: 4  # Keep top 4 groups
    ```
-   **Behavior**: A simple sorting and slicing operation.

### `min_p`
-   **Description**: Selects groups whose score is at least a fraction `value` of the maximum score in the batch.
-   **Configuration**:
    ```yaml
    actor_rollout_ref:
      rollout:
        rollout_filter_strategy: min_p
        rollout_filter_value: 0.8  # Keep groups with score >= 0.8 * max_score
    ```

### Other Parameters
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
The trainer monitors the reward standard deviation (`rollout/in_group_reward_std`) across all rollout attempts (successful steps and retries).

1.  **Baseline Generation**: During the first 10 successful steps, the trainer calculates the average reward variance ($V_{base}$).
2.  **Monitoring**: A sliding window of the last 10 rollout attempts is maintained.
3.  **Stopping Condition**: If all 10 consecutive attempts have a reward variance less than 10% of $V_{base}$, training is stopped.
    $$ \forall i \in \{1 \dots 10\}: V_i < 0.1 \times V_{base} \implies \text{Stop Training} $$

### Implementation
-   **Baseline**: Average of `rollout/in_group_reward_std` for `global_steps` 1-10.
-   **Sliding Window**: Uses a `collections.deque(maxlen=10)` to track the most recent attempts across multiple global steps if retries occur.
-   **Metric**: Logs `train/early_stopped: 1.0` when triggered.

---

## 4. Running Experiments
A unified script `run_filtering_exps.sh` is provided to run grid search experiments.

### Usage
```bash
# Run all experiments (GRPO and PPO)
./run_filtering_exps.sh all

# Run only GRPO experiments
./run_filtering_exps.sh grpo

# Run only PPO experiments
./run_filtering_exps.sh ppo
```

### Features
-   **Date Prefix**: All experiment names are prefixed with `MMDD_` (e.g., `0120_`) to organize runs by date.
-   **Baseline**: Automatically runs a "No Filtering" baseline first.
-   **Grid Search**: Iterates over combinations of:
    -   `rollout_filter_strategy`: `top_p`, `top_k`, `min_p`
    -   `rollout_filter_value`:
        - `top_p`: 0.7, 0.85 (Nucleus Sampling)
        - `top_k`: 4, 6 (Keep top 50%, 75% of 8 groups)
        - `min_p`: 0.5, 0.8 (Keep >50%, >80% of max score)
    -   `rollout_filter_include_zero`: `False` (default for grid)

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
