# Rollout Filtering Guide

## What is Rollout Filtering?

In RAGEN, each training step generates multiple rollouts per prompt (a **group**). The within-group **reward variance (RV)** measures how much the model's responses differ in quality for that prompt:

- **High RV** — the model sometimes succeeds and sometimes fails → strong learning signal
- **Low RV** — all responses receive similar rewards → noisy gradient (low SNR)

**SNR-Adaptive Filtering** discards low-variance groups before the policy gradient update, keeping only prompts that provide meaningful signal. This reduces gradient noise and mitigates reasoning collapse during training.

<p align="center"><img src="../public/top_p.png" width="800px" alt="SNR-Adaptive Filtering (Top-p) pipeline" /></p>
<p align="center"><em>Top-p filtering pipeline: (1) sample rollouts and compute rewards, (2) compute within-prompt reward variance, (3) rank by RV and apply Top-p threshold — low-variance prompts are discarded.</em></p>

All config keys live under `actor_rollout_ref.rollout` in [config/base.yaml](../config/base.yaml).

## Quick Recommendation

| Goal | Config |
|---|---|
| No filtering (default) | `rollout_filter_value=1.0`, `rollout_filter_include_zero=True` |
| **Top-p Linear (recommended)** | `rollout_filter_value=0.9`, `rollout_filter_top_p_prob_mode=linear`, `rollout_filter_include_zero=False`, `rollout_filter_selection_eps=0.01` |
| Top-p Softmax | `rollout_filter_value=0.9`, `rollout_filter_top_p_prob_mode=softmax`, `rollout_filter_include_zero=False` |
| Top-k Fractional | `rollout_filter_strategy=top_k`, `rollout_filter_value=0.25`, `rollout_filter_type=largest`, `rollout_filter_include_zero=True` |

## Config Parameters

| Parameter | Description |
|---|---|
| `rollout_filter_strategy` | Selection strategy: `top_p`, `top_k`, `top_k_abs`, `min_p` |
| `rollout_filter_value` | Threshold value — meaning depends on strategy (see below) |
| `rollout_filter_type` | `largest` (keep high-RV groups) or `smallest` (keep low-RV groups) |
| `rollout_filter_include_zero` | Whether to keep groups with zero reward variance |
| `rollout_filter_top_p_prob_mode` | Top-p score aggregation: `linear` (score-sum rule) or `softmax` (probability mass) |
| `rollout_filter_selection_eps` | Epsilon for the linear top-p threshold (default `0.01`) |
| `rollout_filter_metric` | What to compute per group: `reward_variance` (default), `reward`, `reward_sum`, `entropy`, `entropy_variance`, `length` |
| `rollout_filter_empty_stop_steps` | Early-stop after this many consecutive steps with 0 kept samples (default `5`) |

## Filtering Strategies

### 1. No Filter

Keep all groups. Use this as a baseline.

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_p
    rollout_filter_value: 1.0
    rollout_filter_include_zero: True
```

With `value=1.0` and `include_zero=True`, the filter is effectively disabled — all groups pass through.

### 2. Top-p Linear (Recommended)

Keep the highest-RV groups whose cumulative score reaches a fraction of the total score.

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_p
    rollout_filter_value: 0.9
    rollout_filter_top_p_prob_mode: linear
    rollout_filter_include_zero: False
    rollout_filter_selection_eps: 0.01
```

How it works:
1. Remove zero-RV groups (`include_zero=False`)
2. Sort remaining groups by score (descending)
3. Compute threshold: `top_p * sum(scores) - eps`
4. Accumulate scores from the top until the threshold is reached
5. If the threshold cannot be reached, the step is skipped (`empty_after_filter`)

Tuning:
- `value=0.9`: recommended default
- Lower `value` → more aggressive filtering (fewer groups kept)
- `eps=0.01`: recommended default; larger eps rejects near-zero-RV batches more easily

### 3. Top-p Softmax

Nucleus-style selection based on softmax probability mass over group scores.

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_p
    rollout_filter_value: 0.9
    rollout_filter_top_p_prob_mode: softmax
    rollout_filter_include_zero: False
```

How it works:
1. Remove zero-RV groups (`include_zero=False`)
2. Convert scores to probabilities: `probs = softmax(scores)`
3. Sort by probability (descending) and accumulate until cumulative mass reaches `top_p`

Tuning:
- `value=0.9`: recommended default
- `0.95–0.98`: mild filtering
- `0.6–0.8`: aggressive filtering

### 4. Top-k Fractional

Keep a fixed fraction of groups ranked by score.

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_k
    rollout_filter_value: 0.25
    rollout_filter_type: largest
    rollout_filter_include_zero: True
```

How it works:
1. Compute `k = int(value * num_groups)` (at least 1)
2. Keep the top-k groups by score
3. With `include_zero=True`, zero-RV groups remain as candidates

`value=0.25` means "keep about 25% of groups". Example: with 8 groups, keeps `int(0.25 * 8) = 2` groups.

For a fixed absolute count instead of a fraction, use `strategy=top_k_abs`.

## Code References

- Filter logic: [ragen/trainer/rollout_filter.py](../ragen/trainer/rollout_filter.py)
- Trainer integration: [ragen/trainer/agent_trainer.py](../ragen/trainer/agent_trainer.py)
- Experiment scripts: [scripts/runs/run_filtering_final.sh](../scripts/runs/run_filtering_final.sh)
