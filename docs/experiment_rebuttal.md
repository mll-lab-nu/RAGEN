# Rebuttal Experiments

Experiments added for the RAGEN-V2 rebuttal. All build on top of `train.py` with Hydra config overrides.

---

## Exp 4 — Quartile Ablation (R2-Q1, R3-W1, R3-Q2)

**Goal:** Show that reward variance causally drives training quality.
Sort all prompt groups by RV each step, split into 4 quartiles, train 4 separate runs each using only one quartile for gradient updates. Expected outcome: performance degrades monotonically Q1 → Q4.

**New config keys** (added in `config/base.yaml`):
| Key | Type | Description |
|-----|------|-------------|
| `rollout_filter_strategy` | str | Set to `percentile_range` |
| `rollout_filter_type` | str | `largest` = Q1 is highest RV |
| `rollout_filter_percentile_low` | float | Lower bound of kept band, 0.0–1.0 (inclusive) |
| `rollout_filter_percentile_high` | float | Upper bound of kept band, 0.0–1.0 (exclusive) |

**4 runs:**

```bash
# Q1 — top 25% RV (highest signal)
python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=percentile_range \
  actor_rollout_ref.rollout.rollout_filter_type=largest \
  actor_rollout_ref.rollout.rollout_filter_percentile_low=0.0 \
  actor_rollout_ref.rollout.rollout_filter_percentile_high=0.25

# Q2 — 25–50% RV
python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=percentile_range \
  actor_rollout_ref.rollout.rollout_filter_type=largest \
  actor_rollout_ref.rollout.rollout_filter_percentile_low=0.25 \
  actor_rollout_ref.rollout.rollout_filter_percentile_high=0.5

# Q3 — 50–75% RV
python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=percentile_range \
  actor_rollout_ref.rollout.rollout_filter_type=largest \
  actor_rollout_ref.rollout.rollout_filter_percentile_low=0.5 \
  actor_rollout_ref.rollout.rollout_filter_percentile_high=0.75

# Q4 — bottom 25% RV (lowest signal)
python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=percentile_range \
  actor_rollout_ref.rollout.rollout_filter_type=largest \
  actor_rollout_ref.rollout.rollout_filter_percentile_low=0.75 \
  actor_rollout_ref.rollout.rollout_filter_percentile_high=1.0
```

**Note:** `percentile_range` works with any existing `rollout_filter_metric` (default: `reward_variance`). The `(low, high)` band is applied after zero-exclusion if `rollout_filter_include_zero=False`.

---

## Exp 5 — Trajectory-level Filtering (R3-Q3, R4-Q3)

**Goal:** Isolate whether the benefit of RV-filtering comes from signal quality (SNR) vs. prompt selection bias (dropping "hard" prompts).

This filter keeps **all prompts** but selects `keep_ratio × G` trajectories per prompt to maximize within-group reward variance. If the result matches prompt-level filtering, the benefit is from signal quality, not prompt selection.

**Solver:** For each group of G trajectories, enumerate all `k+1` possible "bottom-j + top-(k-j)" splits and pick the one with maximum variance. This is provably optimal — the variance-maximizing subset is always at the extremes.

**New config keys:**
| Key | Type | Description |
|-----|------|-------------|
| `rollout_filter_strategy` | str | Set to `within_group` |
| `rollout_filter_value` | float | Keep ratio per group (e.g. `0.5` = keep 8/16 trajectories) |

```bash
# Trajectory-level filtering, keep 50% of trajectories per group
python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=within_group \
  actor_rollout_ref.rollout.rollout_filter_value=0.5

# Compare against prompt-level baseline (same compute budget: 50% of data enters update)
python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=top_k \
  actor_rollout_ref.rollout.rollout_filter_type=largest \
  actor_rollout_ref.rollout.rollout_filter_value=0.5
```

**Control:** Both runs use the same total number of trajectories entering the update step (`num_groups × G × 0.5`). Prompt-level keeps half the groups with all their trajectories; trajectory-level keeps all groups with half their trajectories.
