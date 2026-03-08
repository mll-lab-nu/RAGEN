# WebShop Regression Investigation (ICML/RAGEN vs RAGEN)

Context: WebShop experiments run from `/Users/deimos/Desktop/ICML/RAGEN` appear worse than `/Users/deimos/Desktop/RAGEN`.
This doc records the diffs found so far and likely causes.

## High-impact suspects

1) Hydra default search path
- ICML adds `hydra.searchpath: pkg://verl.trainer.config` in `_6_webshop.yaml`.
- RAGEN does not.
- This can change which `ppo_trainer` defaults are loaded and therefore optimizer/loss/rollout defaults.
- Files:
  - `/Users/deimos/Desktop/ICML/RAGEN/config/_6_webshop.yaml`
  - `/Users/deimos/Desktop/RAGEN/config/_6_webshop.yaml`
### Item 1 detail: Hydra/ppo_trainer resolution
- On this machine, neither repo has a real `ppo_trainer.yaml` file in-tree.
  - RAGEN has a broken symlink: `/Users/deimos/Desktop/RAGEN/config/ppo_trainer.yaml -> ../verl/verl/trainer/config/ppo_trainer.yaml` (target missing).
  - ICML relies on `hydra.searchpath` to find `pkg://verl.trainer.config`.
- That means resolved defaults depend on the *external verl package* present in the runtime environment.
- If two environments have different `verl` versions, the resolved defaults (optimizer, loss agg, GRPO knobs) will differ silently.
- Action to confirm: in the actual training environment, dump the resolved config for the same command in both repos and diff.

2) Rollout filtering default strength
- ICML base defaults: `rollout_filter_value: 0.9` and `gpu_memory_utilization: 0.3`.
- RAGEN base defaults: `rollout_filter_value: 0.25` and `gpu_memory_utilization: 0.5`.
- This changes how many groups are kept and the effective training signal quality.
- Files:
  - `/Users/deimos/Desktop/ICML/RAGEN/config/base.yaml` (actor_rollout_ref.rollout)
  - `/Users/deimos/Desktop/RAGEN/config/base.yaml` (actor_rollout_ref.rollout)
### Item 2 detail: filtering strength + vLLM utilization
- With `rollout_filter_strategy=top_p`, the value behaves like a kept ratio.
  - ICML default `0.9` keeps ~90% of groups.
  - RAGEN default `0.25` keeps ~25% of groups.
- This is a big shift in the training distribution and variance of advantages.
- `gpu_memory_utilization` also differs (0.3 vs 0.5), which can change throughput and batch composition.
- Action: fix `actor_rollout_ref.rollout.rollout_filter_value` to the same value in both repos and re-run a short baseline.

3) GRPO loss aggregation / advantage knobs
- ICML explicitly sets `loss_agg_mode: token-mean` and exposes `norm_adv_by_std_in_grpo`, `soft_advantage_reweight`, `zero_task_advantage` in base config.
- RAGEN base config omits these (falls back to `ppo_trainer` defaults).
- This can materially change the GRPO update.
- Files:
  - `/Users/deimos/Desktop/ICML/RAGEN/config/base.yaml`
  - `/Users/deimos/Desktop/RAGEN/config/base.yaml`
### Item 3 detail: loss aggregation + GRPO normalization
- ICML hard-codes `loss_agg_mode=token-mean` in base config; RAGEN inherits from `ppo_trainer`.
- ICML code supports soft advantage reweighting and `zero_task_advantage`, RAGEN does not.
  - See `/Users/deimos/Desktop/ICML/RAGEN/ragen/trainer/agent_trainer.py` and `/Users/deimos/Desktop/ICML/RAGEN/ragen/trainer/core_algos.py`.
- Even if the defaults are off, a hidden `ppo_trainer` default could change GRPO behavior in RAGEN.
- Action: explicitly set `loss_agg_mode` and `norm_adv_by_std_in_grpo` in the run command for both repos.

4) Validation grouping
- ICML: `es_manager.val.group_size = 1`.
- RAGEN: `es_manager.val.group_size = 16`.
- This impacts reported validation metrics (not necessarily training), which can look like a regression.
- Files:
  - `/Users/deimos/Desktop/ICML/RAGEN/config/base.yaml`
  - `/Users/deimos/Desktop/RAGEN/config/base.yaml`
### Item 4 detail: validation metric comparability
- Different `val.group_size` changes the effective number of unique prompts and can bias reported success/reward.
- This can make ICML look worse even if training is identical.
- Action: set `es_manager.val.group_size=1` (or 16) consistently for a fair comparison.

5) WebShop env seeding
- RAGEN passes `seed=self._seed` into `SimServer`.
- ICML does not pass `seed`.
- This changes goal sampling determinism and possibly the train distribution.
- Files:
  - `/Users/deimos/Desktop/ICML/RAGEN/ragen/env/webshop/env.py`
  - `/Users/deimos/Desktop/RAGEN/ragen/env/webshop/env.py`
### Item 5 detail: SimServer seed
- ICML initializes `SimServer(...)` without a seed; RAGEN passes `seed=self._seed`.
- This changes goal ordering and reproducibility, which can shift reward distribution.
- Action: align seeding (add/remove `seed=self._seed`) and re-run a short baseline.

## Moderate-impact suspects

6) Rollout filter semantics changed in code
- ICML `top_k` means fraction of groups (value * num_groups). RAGEN `top_k` means absolute count.
- `min_p` logic also changed (ICML supports smallest/largest; RAGEN uses max_score threshold only).
- If you use `top_k` or `min_p`, the filtered set differs a lot.
- Files:
  - `/Users/deimos/Desktop/ICML/RAGEN/ragen/trainer/rollout_filter.py`
  - `/Users/deimos/Desktop/RAGEN/ragen/trainer/rollout_filter.py`
### Item 6 detail: filtering semantics drift
- ICML supports `top_k_abs` and uses `top_k` as a *fraction*.
- RAGEN removed `top_k_abs` and interprets `top_k` as an *absolute* count.
- `min_p` changed from smallest/largest semantics to a single max-score threshold.
- Action: avoid `top_k`/`min_p` until semantics are aligned; prefer `top_p` with explicit ratio.

7) Memory manager reset in rollouts
- ICML resets memory managers per rollout, and conditionally includes collapse data in rollouts.
- RAGEN does not reset memory managers and always formulates rollouts without collapse data.
- Likely small for WebShop (SimpleMemory), but could affect multi-turn prompt buildup.
- Files:
  - `/Users/deimos/Desktop/ICML/RAGEN/ragen/llm_agent/agent_proxy.py`
  - `/Users/deimos/Desktop/RAGEN/ragen/llm_agent/ctx_manager.py`
### Item 7 detail: memory manager behavior
- ICML introduces `ragen/llm_agent/memory/*` and calls `ctx_manager.reset_memory_managers()` each rollout.
- RAGEN has no memory manager; it directly builds history text.
- The SimpleMemory formatting matches the old logic, so effect is likely minor, but rollout resets might still influence multi-turn prompts.
- Action: if testing, force `context_window_mode=full` to bypass single/limited turn memory formatting.

8) Collapse detection / soft reweight paths (ICML only)
- ICML adds collapse detection and optional soft advantage reweighting.
- If enabled, it can change training dynamics or compute budget.
- Files:
  - `/Users/deimos/Desktop/ICML/RAGEN/ragen/trainer/agent_trainer.py`
  - `/Users/deimos/Desktop/ICML/RAGEN/ragen/trainer/core_algos.py`
### Item 8 detail: collapse metrics + soft reweighting
- ICML computes collapse metrics every N steps (base config enables it) and can inject extra compute and metadata.
- ICML also has optional soft advantage reweighting based on reward variance; RAGEN does not.
- These can alter training speed and (if enabled) advantage scaling.
- Action: temporarily disable collapse detection (`collapse_detection.*`) and ensure `soft_advantage_reweight=false` for parity.

## Next steps to confirm root cause

A) Resolve full Hydra config for each run
- Dump the resolved config (with overrides) for the same command in both repos and diff them.
- This will confirm whether `ppo_trainer` defaults differ.

B) Isolate rollout filtering
- Fix `rollout_filter_value` to the same value in both repos.
- If results converge, filtering strength is likely the culprit.

C) Fix WebShop seeding
- Align seeding behavior (pass `seed` into `SimServer` or not) and re-run a short baseline.

D) Control validation grouping
- Set `es_manager.val.group_size=1` in both repos for apples-to-apples validation.

E) Compare GRPO loss aggregation
- Pin `loss_agg_mode` and `norm_adv_by_std_in_grpo` explicitly in both repos.

---

Notes:
- The `ppo_trainer.yaml` symlink in RAGEN points to `../verl/verl/trainer/config/ppo_trainer.yaml`, but the target path is missing in this checkout. This makes the Hydra resolution path especially important to verify.

## Merge notes: `origin/xjin-webshop-revert-align` into `main`

This branch was merged onto `main` with one manual content conflict and several non-conflicting integration decisions reviewed by hand.

### Manual conflict resolution

File: `ragen/trainer/agent_trainer.py`

- Kept `main`'s newer early-stop structure:
  - typed metrics via `early_stopped/<reason>`
  - empty-after-filter step tracking
  - validation-success early stopping
- Added the branch's WebShop-specific reward table logging:
  - `ragen/trainer/rollout_filter.py` now emits `rollout/_reward_matrix`
  - `agent_trainer.py` converts that matrix into `rollout/reward_table` when `wandb` is available
- Dropped the branch's older duplicate reward-variance-stop block because `main` already supersedes it with step-level handling after successful filtering.

### Auto-merge decisions reviewed and adjusted

- Kept the branch's intended WebShop changes:
  - reverted WebShop environment behavior in `ragen/env/webshop/env.py`
  - updated WebShop instruction/max actions in `config/envs.yaml`
  - bumped `max_model_len` / `max_num_batched_tokens` to `15000` in `config/_6_webshop.yaml`
  - added `scripts/runs/run_webshop.sh`, `scripts/runs/run_webshop_small_combos.sh`, and checkpoint resharding patch files
- Preserved `main` where the branch introduced broader repo regressions unrelated to the WebShop fix:
  - kept the spatial submodule entry in `.gitmodules`
  - kept spatial install in `scripts/setup_ragen.sh`
  - kept `config/base.yaml` default `trainer.project_name: ragen_profiling`

### Follow-up note

`train_webshop_low_filter25.sh` now launches the revert-alignment top-p=0.9 setup even though the filename still mentions `low_filter25`. The file was left in place and annotated rather than renamed during the merge so existing references do not break.
