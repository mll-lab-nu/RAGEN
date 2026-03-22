# Gradient Analysis Walkthrough

This document explains how gradient analysis works in the current RAGEN codebase, which arguments control it, which W&B metrics it writes, how to run it from scratch or from a checkpoint, and how to turn a finished W&B run into local plots.

Normal training now enables gradient analysis by default:
- `trainer.gradient_analysis_mode=True`
- `trainer.gradient_analysis_every=50`
- by default it reuses the training batch unless separate analysis batch overrides are set

## Quickstart

### Run a periodic analysis job from scratch

```bash
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis.sh \
  --gpus 0,1,2,3,4,5,6,7
```

That helper script:
- trains for `101` steps
- runs validation before training and then every `10` steps
- runs gradient analysis on steps `1`, `51`, and `101`
- uses the main-table training batch (`8x16`) but a separate analysis batch (`128x16`)

### Run one analysis-only job

```bash
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis.sh \
  --steps 1 \
  --gpus 0,1,2,3,4,5,6,7
```

Then set:
- `trainer.gradient_analysis_every=1`
- `trainer.exit_after_gradient_analysis=True`

### Plot the finished run

```bash
python gradient_analysis/plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id> \
  --step 1
```

Default output directory:

```text
gradient_analysis_outputs/<run_name>_<run_id>/
```

## What It Does

Gradient analysis is a reporting path that probes the actor on reward-variance buckets without taking an optimizer step.

The current execution order inside [agent_trainer.py](../ragen/trainer/agent_trainer.py) is:

1. generate rollouts
2. apply rollout filtering
3. compute rewards and advantages
4. update critic if enabled
5. if the current step matches the gradient-analysis cadence:
   - split the batch into reward-variance buckets
   - run actor backward passes for task / entropy / KL on each bucket
   - log gradient-analysis metrics
6. if `trainer.exit_after_gradient_analysis=True`, exit here
7. otherwise continue to the normal actor update

So the important boundary is:
- gradient analysis runs after rollout, filtering, reward computation, and critic update
- it runs before the normal actor update
- with `exit_after_gradient_analysis=True`, the run exits before actor update, checkpoint save, and post-step validation

When `trainer.gradient_analysis_env_groups` or `trainer.gradient_analysis_group_size` is set:
- training still uses the normal training batch
- the analysis step generates a second, separate rollout batch just for gradient analysis
- that separate batch is filtered, scored, bucketed, and probed without affecting the training update batch

## How Bucketing Works

Bucketing is implemented in [rollout_filter.py](../ragen/trainer/rollout_filter.py).

### Source Signal

- The rollout filter computes in-group reward standard deviation
- It broadcasts that value to each sample as `batch.batch["reward_std"]`
- Gradient analysis then splits the filtered batch using this `reward_std`

This means the buckets are built from the same reward-variance signal used by rollout filtering, not from a separate offline computation.

### Bucket Modes

Two bucket modes are supported:

1. `quantile` (default)
- Controlled by `gradient_analysis_num_buckets`
- Groups are sorted by group-level `reward_std`
- They are split into equal-percentage buckets
- If the filtered batch contains fewer groups than the requested bucket count, the effective bucket count is reduced so each bucket still contains at least one group
- Bucket names are `bucket_1` to `bucket_N`
- `bucket_1` is lowest reward variance, `bucket_N` is highest reward variance

2. `fixed_rv`
- Uses fixed reward-variance intervals:
  - `bucket_1`: `[0, 1)`
  - `bucket_2`: `[1, 2)`
  - `bucket_3`: `[2, 3)`
  - `bucket_4`: `[3, 4)`
  - `bucket_5`: `[4, 5)`
  - `bucket_6`: `[5, +inf)`

### Special `all` Bucket

Gradient analysis always includes an `all` bucket in addition to the real variance buckets.

`all` means:
- the whole filtered batch
- no bucket subsetting

This is the bridge between the bucketed metrics and the top-level actor metrics.

### DP Safety

If a bucket size is not divisible by `trainer.n_gpus_per_node`, the reporter drops the remainder before calling actor update. If a bucket would become empty after that adjustment, it is skipped.

## How The Analysis Is Computed

The reporting loop is in [gradient_reporter.py](../ragen/trainer/gradient_reporter.py).

For each bucket:

1. create a sub-batch
2. set two meta flags:
   - `skip_optimizer_step=True`
   - `grad_component_analysis=True`
3. call `trainer.actor_rollout_wg.update_actor(sub_batch)`
4. inside the actor, run three backward passes:
   - task policy loss
   - entropy term
   - KL term
5. record gradient norms and losses
6. zero gradients and move to the next bucket

The component-wise backward path is implemented in [dp_actor.py](../ragen/workers/actor/dp_actor.py).

Important detail:
- this path does not call `optimizer.step()`
- it is analysis-only probing of the current actor state

## Arguments

### New Hydra Args

These are the gradient-analysis-specific trainer overrides:

1. `trainer.gradient_analysis_mode=True`
- enables the feature
- default behavior in `config/base.yaml`: enabled
- set `trainer.gradient_analysis_mode=False` to disable it

2. `trainer.gradient_analysis_every=<N>`
- run analysis every `N` training steps
- trigger condition is:
  - `(global_steps - 1) % gradient_analysis_every == 0`
- default behavior in `config/base.yaml`: `50`
- so `gradient_analysis_every=1` means every step
- `gradient_analysis_every=50` means steps `1, 51, 101, ...`

3. `trainer.exit_after_gradient_analysis=True`
- analysis-only mode
- after the selected analysis step finishes, log metrics and exit immediately
- exit happens before:
  - actor update
  - checkpoint save
  - post-step validation
- default behavior in `config/base.yaml`: `False`
- it does not suppress `val_before_train`

4. `trainer.gradient_analysis_env_groups=<N>`
- optional
- if set, gradient analysis uses a separate rollout batch with this many groups
- training keeps using `es_manager.train.env_groups`
- default behavior in `config/base.yaml`: `null` (reuse training batch)

5. `trainer.gradient_analysis_group_size=<N>`
- optional
- if set, gradient analysis uses a separate rollout batch with this group size
- training keeps using `es_manager.train.group_size`
- default behavior in `config/base.yaml`: `null` (reuse training batch)

6. `actor_rollout_ref.rollout.gradient_analysis_num_buckets=<N>`
- number of quantile buckets
- default is `6`

7. `actor_rollout_ref.rollout.gradient_analysis_bucket_mode=quantile|fixed_rv`
- chooses the bucketing rule
- default is `quantile`

### Existing Training Args That Matter

These are not new, but they materially affect the analysis:

- `es_manager.train.env_groups`
  - number of prompt groups
  - more groups gives a finer reward-variance ranking

- `es_manager.train.group_size`
  - number of rollouts per group
  - affects how stable each group reward-variance estimate is

- `trainer.n_gpus_per_node`
  - affects DP-safe bucket trimming

- rollout filter args such as:
  - `actor_rollout_ref.rollout.rollout_filter_value`
  - `actor_rollout_ref.rollout.rollout_filter_strategy`
  - `actor_rollout_ref.rollout.rollout_filter_metric`
  - `actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode`
  - `actor_rollout_ref.rollout.rollout_filter_include_zero`

The current Sokoban PPO helper runner uses:
- training batch: `env_groups=8`, `group_size=16`
- analysis batch: `gradient_analysis_env_groups=128`, `gradient_analysis_group_size=16`
- `rollout_filter_strategy=top_p`
- `rollout_filter_value=0.9`
- `rollout_filter_metric=reward_variance`
- `rollout_filter_top_p_prob_mode=softmax`

## W&B Metrics

There are two layers of metrics:

1. bucket-prefixed analysis metrics
2. top-level actor metrics copied from the `all` bucket

### Bucket-Prefixed Metrics

These are written under:
- `grad_norm/all/...`
- `grad_norm/bucket_1/...`
- `grad_norm/bucket_2/...`
- ...

#### Bucket Size / Coverage

- `grad_norm/<bucket>/sample_count`
- `grad_norm/<bucket>/sample_pct`

#### Reward-Variance Stats

- `grad_norm/<bucket>/reward_std_mean`
- `grad_norm/<bucket>/reward_std_min`
- `grad_norm/<bucket>/reward_std_max`
- `grad_norm/<bucket>/group_rv_count`
- `grad_norm/<bucket>/group_rv_table`

`group_rv_table` is a `wandb.Table` with columns:
- `bucket`
- `group_id`
- `reward_std`

#### Gradient Norms

- `grad_norm/<bucket>/task`
- `grad_norm/<bucket>/entropy`
- `grad_norm/<bucket>/kl`

#### Normalized Gradient Norms

- `grad_norm/<bucket>/per_sample/task`
- `grad_norm/<bucket>/per_sample/entropy`
- `grad_norm/<bucket>/per_sample/kl`
- `grad_norm/<bucket>/per_token/task`
- `grad_norm/<bucket>/per_token/entropy`
- `grad_norm/<bucket>/per_token/kl`

#### Losses

- `grad_norm/<bucket>/loss/policy`
- `grad_norm/<bucket>/loss/entropy`
- `grad_norm/<bucket>/loss/kl`
- `grad_norm/<bucket>/loss/total`

### Top-Level Metrics Copied From `all`

When the bucket name is `all`, the reporter also writes the actor metrics back to top level:

- `actor/loss/policy`
- `actor/loss/entropy`
- `actor/loss/kl`
- `actor/loss/total`
- `actor/grad_norm/task`
- `actor/grad_norm/entropy`
- `actor/grad_norm/kl`

If `exit_after_gradient_analysis=True`, the trainer also logs:

- `trainer/exited_after_gradient_analysis = 1.0`

## Commands

### 1. From Scratch With The Helper Script

The helper runner is:
- [run_sokoban_ppo_filter_grad_analysis.sh](../scripts/runs/run_sokoban_ppo_filter_grad_analysis.sh)

It is fixed to:
- task: `sokoban`
- algo: `PPO`
- filter: `top_p=0.9`
- model: `Qwen2.5-3B`
- env_groups: `32`
- group_size: `16`
- gradient analysis: once on step 1
- `exit_after_gradient_analysis=True`
- one initial validation before training
- no periodic validation afterwards

Example on 8 GPUs:

```bash
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis.sh \
  --steps 1 \
  --gpus 0,1,2,3,4,5,6,7
```

Key defaults inside that script:
- `trainer.project_name=ragen_gradient_analysis`
- `env_groups=32`
- `group_size=16`
- `trainer.gradient_analysis_every=1`
- `trainer.exit_after_gradient_analysis=True`
- `actor_rollout_ref.rollout.gradient_analysis_num_buckets=6`
- `actor_rollout_ref.rollout.gradient_analysis_bucket_mode=quantile`

### 2. Direct `train.py` Usage

Minimal pattern when you want to override the global defaults:

```bash
python train.py ... \
  trainer.gradient_analysis_mode=True \
  trainer.gradient_analysis_every=1 \
  trainer.exit_after_gradient_analysis=True \
  actor_rollout_ref.rollout.gradient_analysis_num_buckets=6 \
  actor_rollout_ref.rollout.gradient_analysis_bucket_mode=quantile
```

### 3. Resume From A Checkpoint And Probe It Once

Checkpoint resume should point to the `global_step_<N>` directory, not the nested `actor/` directory.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --config-name _2_sokoban \
  model_path=Qwen/Qwen2.5-3B \
  trainer.project_name=ragen_gradient_analysis \
  trainer.experiment_name=sokoban-PPO-filter-topp09-grad-from-ckpt100 \
  trainer.default_local_dir=model_saving/gradient_analysis/sokoban/PPO/filter/sokoban-PPO-filter-topp09-grad-from-ckpt100 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/ABS/PATH/TO/your_run/global_step_100 \
  trainer.total_training_steps=101 \
  trainer.save_freq=-1 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=8 \
  ray_kwargs.ray_init.num_cpus=16 \
  system.CUDA_VISIBLE_DEVICES="'0,1,2,3,4,5,6,7'" \
  es_manager.train.env_groups=32 \
  es_manager.train.group_size=16 \
  es_manager.train.env_configs.n_groups="[32]" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_type=low-var-kl \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.filter_loss_scaling=none \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.rollout_filter_value=0.9 \
  actor_rollout_ref.rollout.rollout_filter_strategy=top_p \
  actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=softmax \
  actor_rollout_ref.rollout.rollout_filter_type=largest \
  actor_rollout_ref.rollout.rollout_filter_metric=reward_variance \
  actor_rollout_ref.rollout.rollout_filter_include_zero=True \
  algorithm.adv_estimator=gae \
  trainer.gradient_analysis_mode=True \
  trainer.gradient_analysis_every=1 \
  trainer.exit_after_gradient_analysis=True \
  actor_rollout_ref.rollout.gradient_analysis_num_buckets=6 \
  actor_rollout_ref.rollout.gradient_analysis_bucket_mode=quantile
```

Why `total_training_steps=101` for a `global_step_100` checkpoint:
- the trainer resumes at step 100
- one more training iteration is enough to trigger one analysis pass
- `exit_after_gradient_analysis=True` then exits before actor update

## Practical Notes

- `exit_after_gradient_analysis=True` is not a full “no training code at all” mode.
  - rollout generation, filtering, reward computation, and critic update still happen before the analysis point
  - what it prevents is the normal actor update and everything after it

- If you want the cleanest checkpoint inspection, keep the resumed run aligned with the original training geometry:
  - same `env_groups`
  - same `group_size`
  - same rollout filter settings

- The analysis buckets are built on the filtered batch, not on the raw unfiltered rollout batch.

## Plotting After The Run

The plotting entry point is:
- [plot_gradient_analysis.py](../gradient_analysis/plot_gradient_analysis.py)

### List Available Analysis Steps

Before plotting, you can ask the script which steps in the run actually contain bucket metrics:

```bash
python gradient_analysis/plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id> \
  --list-steps
```

### Plot All Available Analysis Steps

```bash
python gradient_analysis/plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id>
```

### Plot A Specific Step

```bash
python gradient_analysis/plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id> \
  --step 1
```

You can also request multiple steps:

```bash
python gradient_analysis/plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id> \
  --step 1 11 21
```

### Choose An Output Directory

```bash
python gradient_analysis/plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id> \
  --step 1 \
  --output-dir gradient_analysis_outputs/my_run_step1
```

### Files The Plot Script Produces

For each selected step, it writes:

- `gradient_analysis_summary_step_<N>.png`
- `gradient_analysis_plots_step_<N>.png`
- `gradient_analysis_loss_plots_step_<N>.png`
- `gradient_analysis_reward_std_step_<N>.png`
- `gradient_analysis_normed_grads_step_<N>.png`
- `gradient_analysis_metrics_step_<N>.json`
- `gradient_analysis_bucket_rv_table_step_<N>.csv`

Recommended reading order:

1. `gradient_analysis_summary_step_<N>.png`
2. `gradient_analysis_plots_step_<N>.png`
3. `gradient_analysis_metrics_step_<N>.json`

### Paper-Style Multi-Step Figure

If you already exported per-step `metrics.json` files and want the fixed 3-step comparison figure, use:

```bash
python gradient_analysis/plot_icml_steps.py \
  --mode ppo \
  --step0-dir /path/to/step0 \
  --step20-dir /path/to/step20 \
  --step40-dir /path/to/step40 \
  --out gradient_analysis_outputs/ppo_step0_20_40.png
```

The plotting-only README is here:
- [gradient_analysis/README.md](../gradient_analysis/README.md)
