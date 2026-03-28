# Batch-Adjust Modes and Loss Scaling Guide

This guide explains:
- how to configure the three batch-adjust modes: `copy`, `delete`, and `pad`
- how they interact with rollout filtering
- how the two loss-scaling mechanisms in RAGEN work

For the rollout-filtering side itself, see [guide_rollout_filtering.md](./guide_rollout_filtering.md).

## What Problem Does Batch Adjustment Solve?

After rollout filtering, the trainer may keep an irregular number of trajectories. That filtered batch is often **not divisible** by the training layout required by PPO.

In the current trainer, the key divisibility constraint is driven by:
- `es_manager.train.env_groups`
- `actor_rollout_ref.actor.ppo_mini_batch_size`
- `trainer.n_gpus_per_node`

For `copy` and `delete`, the trainer adjusts the **real training batch** to be divisible by:

```text
lcm(env_groups, ppo_mini_batch_size, n_gpus_per_node)
```

For `pad`, the trainer keeps the real filtered batch intact and pads only the later optimizer batch.

All batch-adjust config lives under `agent_proxy` in [config/base.yaml](../config/base.yaml).

## Quick Recommendation

| Goal | Config |
|---|---|
| Keep all real trajectories and preserve trajectory-level weighting under GRPO/DrGRPO | `batch_adjust_mode=pad` |
| Simple baseline that never drops kept samples | `batch_adjust_mode=copy` |
| Avoid duplicated samples and accept occasional skipped steps | `batch_adjust_mode=delete` |

Practical note:
- In most current experiment scripts, `actor_rollout_ref.actor.filter_loss_scaling=none`
- If you use `pad` together with `seq-mean-token-mean` or `seq-mean-token-sum`, RAGEN applies an additional **pad-specific seq loss scaling** automatically

## Config Parameters

| Parameter | Description |
|---|---|
| `agent_proxy.batch_adjust_mode` | One of `copy`, `delete`, `pad` |
| `agent_proxy.batch_adjust_empty_stop_steps` | Early-stop threshold for consecutive `delete -> empty batch` steps (default `5`) |
| `actor_rollout_ref.actor.filter_loss_scaling` | One of `none`, `linear`, `sqrt` |
| `actor_rollout_ref.actor.loss_agg_mode` | Loss aggregation mode: `token-mean`, `seq-mean-token-sum`, `seq-mean-token-mean`, `seq-mean-token-sum-norm` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | PPO mini-batch size used for optimizer batching |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | Per-GPU micro-batch size used inside actor updates |
| `trainer.n_gpus_per_node` | Number of GPUs participating in the trainer |

Important constraint:
- `batch_adjust_mode=pad` does **not** support `use_dynamic_bsz=True` for actor or critic

## 1. `copy`

`copy` duplicates some kept samples so that the adjusted batch becomes divisible by the required divisor.

```yaml
agent_proxy:
  batch_adjust_mode: copy
```

How it works:
1. Rollout filtering keeps a subset of trajectories.
2. If the filtered batch size is already divisible, nothing changes.
3. Otherwise, the trainer samples some existing rows and appends duplicates.
4. The duplicated batch is then used for reward/model bookkeeping, advantage computation, and optimization.

Properties:
- Never discards kept trajectories
- Never produces an empty batch
- Can over-represent some trajectories because duplicated rows participate in training normally

Use this when:
- you want the simplest non-empty behavior after filtering
- you prefer duplicated samples over dropped samples or optimizer-only padding

## 2. `delete`

`delete` removes some kept samples so that the adjusted batch becomes divisible by the required divisor.

```yaml
agent_proxy:
  batch_adjust_mode: delete
  batch_adjust_empty_stop_steps: 5
```

How it works:
1. Rollout filtering keeps a subset of trajectories.
2. If the filtered batch size is already divisible, nothing changes.
3. Otherwise, the trainer randomly drops the remainder.
4. The reduced batch is then used for the rest of training.

Properties:
- Never duplicates trajectories
- Can discard some filtered-in samples
- Can produce an empty batch if the filtered batch size is smaller than the divisor

Current empty-batch behavior:
- If `delete` reduces the batch to `0`, the trainer skips that step
- It logs:
  - `train/empty_after_adjust=1`
  - `train/consecutive_empty_after_adjust_steps`
- If this happens for `batch_adjust_empty_stop_steps` consecutive steps, training stops with:
  - `early_stopped/empty_after_adjust_steps=1`

Use this when:
- you want to avoid duplicated samples
- you are okay with occasionally losing data or skipping steps under aggressive filtering

## 3. `pad`

`pad` keeps the filtered batch intact and pads only the optimizer batch later with zero-loss rows.

```yaml
agent_proxy:
  batch_adjust_mode: pad
```

How it works:
1. Rollout filtering keeps a subset of trajectories.
2. Reward computation, reference log-prob, old log-prob, values, and advantages are computed on the **real batch only**.
3. Right before actor/critic updates, the trainer pads the batch to the next `ppo_mini_batch_size` multiple.
4. The padded rows are marked by `pad_mask`.
5. Training-related tensors on padded rows are zeroed out.

The pad helper currently zeros these keys on padded rows:
- `response_mask`
- `loss_mask`
- `advantages`
- `returns`
- `token_level_scores`
- `token_level_rewards`
- `old_log_probs`
- `ref_log_prob`
- `values`

Properties:
- Preserves the real filtered batch exactly
- Avoids both dropping and fully training on duplicate samples
- Requires extra logic so seq-level losses keep the intended weighting

Use this when:
- you want to preserve the kept trajectories exactly
- you care about trajectory-level weighting under `GRPO` / `DrGRPO`

## Loss Scaling: Two Different Mechanisms

RAGEN currently has **two distinct scaling ideas** that are easy to confuse:

1. `filter_loss_scaling`
- scales the update based on the **kept ratio after filtering**

2. pad-specific seq loss scaling
- fixes the weighting of **padded seq-mean actor losses**

They solve different problems.

## 4. Filter Loss Scaling (`filter_loss_scaling`)

This is the more general scaling knob. It is controlled by:

```yaml
actor_rollout_ref:
  actor:
    filter_loss_scaling: none   # none, linear, sqrt
```

The trainer computes:

```text
filter_kept_ratio = N_kept / N_total
```

and applies scaling by modifying the advantages:

- `none`

```text
advantages <- advantages
```

- `linear`

```text
advantages <- advantages * filter_kept_ratio
```

- `sqrt`

```text
advantages <- advantages * sqrt(filter_kept_ratio)
```

Interpretation:
- `none`: no dampening after filtering
- `linear`: the most conservative option
- `sqrt`: a milder dampening

Implementation detail:
- This scaling is applied in the trainer after advantage computation and before actor update
- It is independent of whether batch adjustment is `copy`, `delete`, or `pad`

In many current experiment scripts, this is explicitly set to:

```yaml
actor_rollout_ref:
  actor:
    filter_loss_scaling: none
```

## 5. Pad-Specific Seq Loss Scaling

This second scaling mechanism is used only for `batch_adjust_mode=pad`.

### Why Is It Needed?

With `pad`, padded rows have zero masks and zero advantages, so they should not contribute real learning signal. That is enough for `token-mean`, but not always enough for sequence-level aggregation.

For `seq-mean-token-mean` and `seq-mean-token-sum`, the actor loss is computed at the **trajectory level**. If the last micro-batch contains a mixture of real rows and padded rows, we still want each real trajectory to keep its original contribution relative to the configured PPO mini-batch size.

### Current Rule

For each actor micro-batch, RAGEN computes:

```text
seq_loss_scale = real_rows_in_micro_batch / ppo_micro_batch_size_per_gpu
```

This scaling is applied to:
- policy loss
- entropy loss
- KL loss

but only when:
- `batch_adjust_mode=pad`, and
- `loss_agg_mode` is one of:
  - `seq-mean-token-sum`
  - `seq-mean-token-mean`

### Why Micro-Batch Ratio Instead of Mini-Batch Ratio?

Because the actor update already uses gradient accumulation across micro-batches.

If:
- `ppo_mini_batch_size = 32`
- `ppo_micro_batch_size_per_gpu = 4`
- the last real batch contains `18` trajectories

then the padded optimizer batch becomes `32`, and the 8 micro-batches contain real counts like:

```text
[4, 4, 4, 4, 2, 0, 0, 0]
```

The partial micro-batch gets scaled by:

```text
2 / 4
```

After gradient accumulation, the total effective weight becomes:

```text
18 / 32
```

which is exactly the intended trajectory-level weighting.

### What About `token-mean`?

`token-mean` does **not** need this extra seq scaling.

Reason:
- padded rows have `response_mask=0` and `loss_mask=0`
- therefore they contribute to neither the numerator nor the denominator of the masked mean

So for `token-mean`, the helper returns `1.0`.

### What About `seq-mean-token-sum-norm`?

Currently, the pad-specific seq scaling is **not** applied to `seq-mean-token-sum-norm`.

That mode currently falls back to scale `1.0`.

## Recommended Settings

### A. Simple baseline

```yaml
agent_proxy:
  batch_adjust_mode: copy

actor_rollout_ref:
  actor:
    filter_loss_scaling: none
```

### B. No duplicated samples

```yaml
agent_proxy:
  batch_adjust_mode: delete
  batch_adjust_empty_stop_steps: 5

actor_rollout_ref:
  actor:
    filter_loss_scaling: none
```

### C. Preserve real trajectories under GRPO/DrGRPO

```yaml
agent_proxy:
  batch_adjust_mode: pad

actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-mean   # or seq-mean-token-sum
    filter_loss_scaling: none
```

This is the setting where the pad-specific seq scaling matters.

## Useful Metrics

When debugging batch adjustment, these metrics are the most informative:

- `train/real_batch_size`
- `train/batch_size`
- `train/optimizer_batch_size`
- `train/pad_count`
- `train/pad_ratio`
- `rollout/empty_after_filter`
- `rollout/consecutive_empty_after_filter_steps`
- `train/empty_after_adjust`
- `train/consecutive_empty_after_adjust_steps`
- `early_stopped/empty_after_adjust_steps`

Interpretation:
- `real_batch_size`: filtered batch before optimizer-only padding
- `optimizer_batch_size`: batch actually sent into actor/critic update
- `pad_count`: how many padded rows were added in `pad` mode
- `empty_after_adjust`: whether `delete` reduced the step to zero rows

## Code References

- Batch-adjust config validation: [train.py](../train.py)
- Real-batch adjustment and skip logic: [ragen/trainer/agent_trainer.py](../ragen/trainer/agent_trainer.py)
- Pad helper: [ragen/trainer/pad_batch.py](../ragen/trainer/pad_batch.py)
- Actor pad-specific seq scaling: [ragen/workers/actor/dp_actor.py](../ragen/workers/actor/dp_actor.py)
- Tests: [tests/test_pad_batch.py](../tests/test_pad_batch.py)
