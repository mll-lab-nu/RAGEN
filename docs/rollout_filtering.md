# Rollout Filtering Short Guide

This note only covers the four common settings used in current RAGEN runs.

Relevant config keys live under `actor_rollout_ref.rollout` in [config/base.yaml](../config/base.yaml).

## 1. No Filter

Use this when you want true no-filter behavior.

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_p
    rollout_filter_value: 1.0
    rollout_filter_include_zero: True
```

Notes:

- `include_zero=True` is the important part.
- With `top_p=1.0`, the filter keeps all groups.
- In this setting, `rollout_filter_top_p_prob_mode` does not matter.
- If you want guaranteed no filtering, prefer this setting over `top_k`.

## 2. Top-p Softmax

Use this when you want the old behavior.

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_p
    rollout_filter_value: 0.9
    rollout_filter_top_p_prob_mode: softmax
    rollout_filter_include_zero: False
```

Behavior:

- zero-score groups are removed first because `include_zero=False`
- remaining group scores are turned into

```text
probs = softmax(scores)
```

- the filter keeps the smallest prefix whose cumulative probability mass reaches `top_p`

How to choose `rollout_filter_value`:

- `0.9`: recommended default
- `0.95` to `0.98`: mild filtering
- `0.6` to `0.8`: aggressive filtering
- lower `top_p` means fewer groups kept

## 3. Top-p Linear

Use this when you want the new stricter RV-mass rule.

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_p
    rollout_filter_value: 0.9
    rollout_filter_top_p_prob_mode: linear
    rollout_filter_include_zero: False
    rollout_filter_selection_eps: 0.01
```

Behavior:

- zero-score groups are removed first because `include_zero=False`
- scores are sorted from large to small
- threshold is computed as

```text
threshold = top_p * sum(scores) - eps
```

- scores are accumulated from the front until the threshold is reached
- if the threshold cannot be reached, the step becomes `empty_after_filter`

How to choose the variables:

- `rollout_filter_value=0.9`: recommended default
- smaller `top_p`: more aggressive filtering
- `rollout_filter_selection_eps=0.01`: recommended default
- smaller `eps`: less aggressive emptying
- larger `eps`: easier to reject near-zero-RV batches

## 4. Top-k Fractional

Use this when you want to keep a fixed fraction of groups.

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_k
    rollout_filter_value: 0.25
    rollout_filter_type: largest
    rollout_filter_include_zero: True
```

Behavior:

- with `include_zero=True`, zero-score groups stay in the candidate set
- the code computes

```text
k = int(top_k * num_groups)
k = min(k, num_candidate_groups)
k = max(k, 1)
```

- `num_groups` is the total group count for the current batch
- with `include_zero=True`, `num_candidate_groups = num_groups`
- `largest` keeps the highest-score groups
- `smallest` keeps the lowest-score groups
- zero-score groups can still be selected if they fall inside the top-`k` ranking

What `top_k=0.25` means:

- it is a fraction, not an absolute count
- equivalently, it means "keep about 25% of the groups"
- example: with `num_groups=8`, it keeps `int(0.25 * 8) = 2` groups
- example: with `num_groups=7`, it keeps `int(0.25 * 7) = 1` group
- because the code uses `int(...)`, this is floor rounding, so small batches can be a bit more aggressive than the nominal ratio
- if you want a fixed absolute count instead, use `top_k_abs`

How to choose `rollout_filter_value`:

- `0.25`: recommended setting for the current Top-k runs
- `0.5`: milder filtering
- smaller values: more aggressive filtering
- use `largest` for the standard "keep best groups" setup
- use `smallest` only when you intentionally want the reverse selection

## Quick Recommendation

- want no filtering: `top_p=1.0`, `include_zero=True`
- want the old top-p behavior: `top_p_prob_mode=softmax`, `include_zero=False`, start with `top_p=0.9`
- want the new stricter RV filter: `top_p_prob_mode=linear`, `include_zero=False`, start with `top_p=0.9`, `eps=0.01`
- want the new Top-k setup: `strategy=top_k`, `value=0.25`, `type=largest`, `include_zero=True`

## Code References

- filter logic: [ragen/trainer/rollout_filter.py](../ragen/trainer/rollout_filter.py)
- trainer handling for empty filtered steps: [ragen/trainer/agent_trainer.py](../ragen/trainer/agent_trainer.py)
- validated Top-k experiment settings: [run_filtering_final.sh](../run_filtering_final.sh)
