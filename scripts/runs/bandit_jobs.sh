#!/bin/bash
# Experiments: Bandit 3B base PPO/GRPO contrast (normal vs StarPO-S) with entropy and instruct ablations.
# Args: 400 steps, lr_actor=1e-6, lr_critic=1e-5, micro_batch=1, env tags=[Bandit] with BanditTest validation; StarPO-S disables reference, optional entropy/filter tweaks.

# set -u -o pipefail
set +e

GPUS=(0 1 2 3 4 5 6 7)
TOTAL_GPUS=${#GPUS[@]}
gpu_idx=0

maybe_flush() {
  local needed=$1
  if (( gpu_idx + needed > TOTAL_GPUS )); then
    wait
    gpu_idx=0
    sleep 10
  fi
}

init_singleton() {
  local tag=${1:-$(basename "$0")}
  local dir="/blob/v-zihanwang/tmp"
  mkdir -p "$dir"
  export SGL_FILE="${dir}/${tag}.lock"

  local ts
  ts=$(date +%s)

  if [[ -f "$SGL_FILE" ]]; then
    local last_modified
    last_modified=$(stat -c %Y "$SGL_FILE")
    if (( ts - last_modified < 60 )); then
      echo "[singleton] newer process already active (lock updated $(date -d @$last_modified)). exiting."
      exit 0
    fi
  fi

  echo "$ts" > "$SGL_FILE"
  touch -d "@$ts" "$SGL_FILE"

  export SGL_TS="$ts"

  echo "[singleton] init: file=$SGL_FILE ts=$SGL_TS"
}

check_singleton() {
  if [[ -z "${SGL_FILE:-}" || -z "${SGL_TS:-}" ]]; then
    echo "[singleton] check: env not initialized (SGL_FILE/SGL_TS empty) -> exiting."
    exit 0
  fi

  if [[ ! -f "$SGL_FILE" ]]; then
    echo "[singleton] check: lock file missing -> taken over by another script. exiting."
    exit 0
  fi

  local mtime
  mtime=$(stat -c %Y "$SGL_FILE" 2>/dev/null || echo 0)

  if [[ "$mtime" != "$SGL_TS" ]]; then
    echo "[singleton] check: lock updated (was $SGL_TS, now $mtime). exiting."
    exit 0
  fi
}

wait_sleep_reset_check() {
  wait
  sleep 15
  gpu_idx=0
  check_singleton
}

launch_bandit() {
  local run_name=$1
  local think=$2
  local algo=$3
  local mode=$4
  local n_gpus=${5:-2}
  local total_training_steps=${6:-200}
  shift 6
  local overrides=("$@")

  maybe_flush ${n_gpus}

  local estimator
  if [[ "$algo" == "ppo" ]]; then
    estimator="gae"
  else
    estimator="$algo"
  fi

  local assigned=(${GPUS[@]:$gpu_idx:$n_gpus})
  local visible=""
  for id in "${assigned[@]}"; do
    if [[ -n "$visible" ]]; then
      visible+="," 
    fi
    visible+="$id"
  done
  gpu_idx=$((gpu_idx + n_gpus))

  local storage_args=(
    "trainer.default_local_dir=/blob/v-zihanwang/ragen_checkpoints/${run_name}"
    "trainer.max_actor_ckpt_to_keep=1"
    "trainer.max_critic_ckpt_to_keep=1"
  )

  local mode_overrides=()
  case "$mode" in
    normal)
      mode_overrides=(
        "algorithm.kl_ctrl.kl_coef=0.001"
        "actor_rollout_ref.actor.clip_ratio_high=0.20"
        "actor_rollout_ref.rollout.rollout_filter_ratio=1"
        "actor_rollout_ref.actor.use_ref=True"
      )
      ;;
    s)
      mode_overrides=(
        "actor_rollout_ref.actor.use_ref=False"
        "algorithm.kl_ctrl.kl_coef=0.0"
        "actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
      )
      ;;
    det)
      mode_overrides=(
        "algorithm.kl_ctrl.kl_coef=0.001"
        "actor_rollout_ref.actor.clip_ratio_high=0.20"
        "actor_rollout_ref.rollout.rollout_filter_ratio=1"
        "actor_rollout_ref.actor.use_ref=True"
        "agent_proxy.max_turn=1"
        "agent_proxy.max_actions_per_turn=1"
        "custom_envs.Bandit.max_actions_per_traj=1"
        "+custom_envs.Bandit.env_config.hi_arm_loscore=0.25"
        "+custom_envs.Bandit.env_config.hi_arm_hiscore=0.25"
      )
      ;;
    *)
      echo "[bandit_jobs] Unknown mode: $mode" >&2
      return 1
      ;;
  esac

  local base_args=(
    "system.CUDA_VISIBLE_DEVICES=\"${visible}\""
    "trainer.n_gpus_per_node=${n_gpus}"
    "trainer.experiment_name=${run_name}"
    "trainer.total_training_steps=${total_training_steps}"
    "trainer.save_freq=50"
    "model_path=Qwen/Qwen2.5-3B"
    "lora.rank=0"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "critic.optim.lr=1e-5"
    "micro_batch_size_per_gpu=1"
    "algorithm.adv_estimator=${estimator}"
    "agent_proxy.enable_think=${think}"
    "agent_proxy.max_turn=1"
    "agent_proxy.max_actions_per_turn=1"
    "es_manager.train.env_configs.tags=[Bandit]"
    "es_manager.val.env_configs.tags=[Bandit,BanditTest]"
    "es_manager.val.env_configs.n_groups=[32,32]"
    "es_manager.val.env_groups=64"
  )

  local log_dir=$(echo "${storage_args[0]}" | cut -d'=' -f2)
  mkdir -p "$log_dir"

  echo "=== Running ${run_name} on GPUs ${visible} ==="
  CUDA_VISIBLE_DEVICES="${visible}" \
  WANDB_RUN_ID=${run_name} \
  python train.py \
    "${base_args[@]}" \
    "${mode_overrides[@]}" \
    "${storage_args[@]}" \
    "${overrides[@]}" \
    2>&1 | tee -a "$log_dir/log.log" &

  sleep 5
}

kl_coef_overrides=(
        "algorithm.kl_ctrl.kl_coef=0.001"
        "actor_rollout_ref.actor.use_ref=True"
)

entropy_filter_overrides=(
  "actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
  "actor_rollout_ref.rollout.rollout_filter_metric=entropy"
)

entvar_filter_overrides=(
  "actor_rollout_ref.rollout.rollout_filter_metric=entropy_variance"
)

instruct_overrides=("model_path=Qwen/Qwen2.5-3B-Instruct")

init_singleton "$(basename "${BASH_SOURCE[0]}")" # create a lock file with the name of the script

# launch_bandit "bandit_3b_base_ppo_think_s_entvarfilter" True ppo s 8 400 "${entvar_filter_overrides[@]}"

# launch_bandit "bandit_3b_base_grpo_think_normal_1" True grpo normal 8 200

launch_bandit "bandit_3b_base_ppo_think_s_2" True ppo s 8 400

# launch_bandit "bandit_3b_base_ppo_think_normal_2" True ppo normal 4 200
# launch_bandit "bandit_3b_base_ppo_nothink_normal_2" False ppo normal 4 200

wait_sleep_reset_check

# launch_bandit "bandit_3b_base_ppo_think_s" True ppo s 4
# launch_bandit "bandit_3b_base_ppo_think_det" True ppo det 4

# wait_sleep_reset_check

# launch_bandit "bandit_3b_base_ppo_think_s_klcoef0.001" True ppo s 4 "${kl_coef_overrides[@]}"
# launch_bandit "bandit_3b_base_ppo_think_s_entropyfilter" True ppo s 4 "${entropy_filter_overrides[@]}"

# wait_sleep_reset_check

# launch_bandit "bandit_3b_instruct_ppo_think_s" True ppo s 4 "${instruct_overrides[@]}"
# launch_bandit "bandit_3b_base_grpo_nothink_normal" False grpo normal 4 200

# wait_sleep_reset_check

# launch_bandit "bandit_3b_base_grpo_nothink_normal" False grpo normal 4
# launch_bandit "bandit_3b_base_ppo_think_s" True ppo s 8 400
