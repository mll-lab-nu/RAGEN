#!/bin/bash
# Experiments: WebShop 3B StarPO-S sweeps (base vs instruct, entropy/n-gram filtering, entropy ablation).
# Args: 400 steps, lr_actor=1e-6, lr_critic=1e-5, micro_batch=1, actor rollout max_len=15000, env tags=WebShop, StarPO-S disables reference.

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

wait_sleep_reset() {
  wait
  sleep 15
  gpu_idx=0
}

launch_webshop_s() {
  local run_name=$1
  local n_gpus=${2:-4}
  local total_training_steps=${3:-200}
  shift 3
  local overrides=("$@")

  maybe_flush ${n_gpus}

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

  local base_args=(
    "system.CUDA_VISIBLE_DEVICES=\"${visible}\""
    "trainer.n_gpus_per_node=${n_gpus}"
    "trainer.experiment_name=${run_name}"
    "trainer.total_training_steps=${total_training_steps}"
    "trainer.save_freq=25"
    "model_path=Qwen/Qwen2.5-3B"
    "lora.rank=0"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "critic.optim.lr=1e-5"
    "micro_batch_size_per_gpu=1"
    "algorithm.adv_estimator=gae"
    "agent_proxy.enable_think=True"
    "agent_proxy.max_turn=8"
    "agent_proxy.max_actions_per_turn=1"
    "actor_rollout_ref.actor.use_ref=False"
    "algorithm.kl_ctrl.kl_coef=0.0"
    "actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
    "actor_rollout_ref.rollout.max_model_len=15000"
    "actor_rollout_ref.rollout.max_num_batched_tokens=15000"
    "es_manager.train.env_configs.tags=[WebShop]"
    "es_manager.val.env_configs.tags=[WebShop]"
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
  "actor_rollout_ref.rollout.rollout_filter_metric=entropy"
)

entvar_filter_overrides=(
  "actor_rollout_ref.rollout.rollout_filter_metric=entropy_variance"
)

launch_webshop_s "webshop_3b_base_ppo_think_s_entvarfilter" 8 400 "${entvar_filter_overrides[@]}"
wait_sleep_reset

# launch_webshop_s "webshop_3b_base_ppo_think_s" 8 400
# wait_sleep_reset

# launch_webshop_s "webshop_3b_base_ppo_think_s_entropyfilter" 8 400 "${entropy_filter_overrides[@]}"
# wait_sleep_reset

# launch_webshop_s "webshop_3b_base_ppo_think_s_klcoef0.001" 8 400 "${kl_coef_overrides[@]}"
# wait_sleep_reset
