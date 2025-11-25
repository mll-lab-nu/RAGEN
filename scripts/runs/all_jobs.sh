#!/bin/bash
# Unified launcher for RAGEN experiments with per-environment presets.
# Usage example:
#   launch sokoban "run_name" True ppo s 8 800 "${extra_overrides[@]}"

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

wait_sleep_reset() {
  wait
  sleep 15
  gpu_idx=0
}

launch() {
  if (( $# < 5 )); then
    echo "Usage: launch <env> <run_name> <think> <algo> <mode> [n_gpus] [total_steps] [extra_overrides...]" >&2
    return 1
  fi

  local env=${1,,}; shift
  local run_name=$1; shift
  local think=$1; shift
  local algo=$1; shift
  local mode=${1,,}; shift

  local n_gpus=2
  if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then
    n_gpus=$1; shift
  fi

  local total_training_steps=200
  if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then
    total_training_steps=$1; shift
  fi

  local overrides=("$@")

  maybe_flush "${n_gpus}"

  local estimator
  if [[ "$algo" == "ppo" ]]; then
    estimator="gae"
  else
    estimator="$algo"
  fi

  local assigned=(${GPUS[@]:$gpu_idx:$n_gpus})
  local visible=""
  for id in "${assigned[@]}"; do
    [[ -n "$visible" ]] && visible+=","
    visible+="$id"
  done
  gpu_idx=$((gpu_idx + n_gpus))

  local storage_args=(
    "trainer.default_local_dir=/blob/v-zihanwang/ragen_checkpoints/${run_name}"
    "trainer.max_actor_ckpt_to_keep=1"
    "trainer.max_critic_ckpt_to_keep=1"
  )

  local common_args=(
    "model_path=Qwen/Qwen2.5-3B"
    "lora.rank=0"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "critic.optim.lr=1e-5"
    "micro_batch_size_per_gpu=1"
    "algorithm.adv_estimator=${estimator}"
    "agent_proxy.enable_think=${think}"
  )

  local env_base_args=()
  local mode_overrides=()
  local det_extra=()

  apply_common_mode_overrides() {
    case "$mode" in
      normal)
        mode_overrides+=(
          "algorithm.kl_ctrl.kl_coef=0.001"
          "actor_rollout_ref.actor.clip_ratio_high=0.20"
          "actor_rollout_ref.rollout.rollout_filter_ratio=1"
          "actor_rollout_ref.actor.use_ref=True"
        )
        ;;
      s)
        mode_overrides+=(
          "actor_rollout_ref.actor.use_ref=False"
          "algorithm.kl_ctrl.kl_coef=0.0"
          "actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
        )
        ;;
      det)
        mode_overrides+=(
          "algorithm.kl_ctrl.kl_coef=0.001"
          "actor_rollout_ref.actor.clip_ratio_high=0.20"
          "actor_rollout_ref.rollout.rollout_filter_ratio=1"
          "actor_rollout_ref.actor.use_ref=True"
        )
        mode_overrides+=("${det_extra[@]}")
        ;;
      void)
        mode_overrides+=(
          "actor_rollout_ref.actor.use_ref=False"
          "algorithm.kl_ctrl.kl_coef=0.0"
        )
        ;;
      *)
        return 1
        ;;
    esac
  }

  case "$env" in
    bandit)
      env_base_args=(
        "${common_args[@]}"
        "trainer.save_freq=50"
        "agent_proxy.max_turn=1"
        "agent_proxy.max_actions_per_turn=1"
        "es_manager.train.env_configs.tags=[Bandit]"
        "es_manager.val.env_configs.tags=[Bandit,BanditTest]"
        "es_manager.val.env_configs.n_groups=[32,32]"
        "es_manager.val.env_groups=64"
      )
      det_extra=(
        "agent_proxy.max_turn=1"
        "agent_proxy.max_actions_per_turn=1"
        "custom_envs.Bandit.max_actions_per_traj=1"
        "+custom_envs.Bandit.env_config.hi_arm_loscore=0.25"
        "+custom_envs.Bandit.env_config.hi_arm_hiscore=0.25"
      )
      apply_common_mode_overrides || { echo "[launch] Unknown mode '$mode' for bandit" >&2; return 1; }
      ;;

    frozenlake)
      env_base_args=(
        "${common_args[@]}"
        "trainer.save_freq=50"
        "es_manager.train.env_configs.tags=[CoordFrozenLake]"
        "es_manager.val.env_configs.tags=[CoordFrozenLake]"
      )
      det_extra=(
        "agent_proxy.max_turn=1"
        "agent_proxy.max_actions_per_turn=10"
        "+custom_envs.CoordFrozenLake.max_actions_per_traj=10"
        "+custom_envs.CoordFrozenLake.env_config.is_slippery=False"
      )
      apply_common_mode_overrides || { echo "[launch] Unknown mode '$mode' for frozenlake" >&2; return 1; }
      ;;

    sokoban)
      env_base_args=(
        "${common_args[@]}"
        "trainer.save_freq=50"
        "es_manager.train.env_configs.tags=[CoordSokoban]"
        "es_manager.val.env_configs.tags=[CoordSokoban]"
      )
      det_extra=(
        "agent_proxy.max_turn=1"
        "agent_proxy.max_actions_per_turn=10"
        "custom_envs.CoordSokoban.max_actions_per_traj=10"
      )
      apply_common_mode_overrides || { echo "[launch] Unknown mode '$mode' for sokoban" >&2; return 1; }
      ;;

    webshop)
      env_base_args=(
        "${common_args[@]}"
        "trainer.save_freq=25"
        "agent_proxy.max_turn=8"
        "agent_proxy.max_actions_per_turn=1"
        "actor_rollout_ref.rollout.max_model_len=15000"
        "actor_rollout_ref.rollout.max_num_batched_tokens=15000"
        "es_manager.train.env_configs.tags=[WebShop]"
        "es_manager.val.env_configs.tags=[WebShop]"
      )
      if [[ "$mode" != "s" ]]; then
        echo "[launch] Only StarPO-S (mode 's') is supported for webshop" >&2
        return 1
      fi
      mode_overrides=(
        "actor_rollout_ref.actor.use_ref=False"
        "algorithm.kl_ctrl.kl_coef=0.0"
        "actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
      )
      ;;

    *)
      echo "[launch] Unknown environment: $env" >&2
      return 1
      ;;
  esac

  local base_args=(
    "system.CUDA_VISIBLE_DEVICES=\"${visible}\""
    "trainer.n_gpus_per_node=${n_gpus}"
    "trainer.experiment_name=${run_name}"
    "trainer.total_training_steps=${total_training_steps}"
  )
  base_args+=("${env_base_args[@]}")

  local log_dir=$(echo "${storage_args[0]}" | cut -d'=' -f2)
  mkdir -p "$log_dir"

  echo "=== Running ${run_name} (${env}) on GPUs ${visible} ==="
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

entvar_filter_overrides=(
  "actor_rollout_ref.rollout.rollout_filter_metric=entropy_variance"
)

filter_ratio_0_75_overrides=(
  "actor_rollout_ref.rollout.rollout_filter_ratio=0.75"
)

launch sokoban "sokoban_coord_3b_base_ppo_think_s_entvarfilter" True ppo s 8 800 "${entvar_filter_overrides[@]}"
wait_sleep_reset

launch frozenlake "frozenlake_coord_3b_base_ppo_think_rolloutfilterratio0.75" True ppo void 8 800 "${filter_ratio_0_75_overrides[@]}"
wait_sleep_reset