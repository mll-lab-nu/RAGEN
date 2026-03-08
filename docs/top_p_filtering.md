# Top-p Filtering Short Guide

This note only covers the three common settings used in current RAGEN runs.

Relevant config keys live under `actor_rollout_ref.rollout` in [config/base.yaml](/work/hdd/bfea/cgui/RAGEN/config/base.yaml).

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

## Quick Recommendation

- want no filtering: `top_p=1.0`, `include_zero=True`
- want the old top-p behavior: `top_p_prob_mode=softmax`, `include_zero=False`, start with `top_p=0.9`
- want the new stricter RV filter: `top_p_prob_mode=linear`, `include_zero=False`, start with `top_p=0.9`, `eps=0.01`

## Code References

- filter logic: [ragen/trainer/rollout_filter.py](/work/hdd/bfea/cgui/RAGEN/ragen/trainer/rollout_filter.py)
- trainer handling for empty filtered steps: [ragen/trainer/agent_trainer.py](/work/hdd/bfea/cgui/RAGEN/ragen/trainer/agent_trainer.py)
