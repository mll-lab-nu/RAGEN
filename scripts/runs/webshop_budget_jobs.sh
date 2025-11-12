#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-3B-Instruct"
PROJ="budget_main"
BASE_DIR="/blob/v-zihanwang/budget_checkpoints"
DEVICES="0,1,2,3,4,5,6,7"

run_experiment() {
  local turns=$1
  local exp_name="webshop_starpos_grpo_3b_small_max_${turns}turns"
  local out_dir="${BASE_DIR}/${exp_name}"

  if [[ "$turns" -ge 7 ]]; then
    local max_len=15000
    local max_tok=15000
  else
    local max_len=10000
    local max_tok=10000
  fi

  echo "=== Running ${exp_name} ==="
  mkdir -p "${BASE_DIR}/${exp_name}"

  CUDA_VISIBLE_DEVICES="${DEVICES}" \
  WANDB_RUN_ID=${exp_name} \
  python train.py --config-name _6_webshop ${USE_PPO:-} \
    model_path="${MODEL}" \
    actor_rollout_ref.rollout.rollout_filter_ratio=1 \
    trainer.project_name="${PROJ}" \
    micro_batch_size_per_gpu=1 \
    trainer.experiment_name="${exp_name}" \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups='[2]' \
    es_manager.val.env_groups=64 es_manager.val.group_size=8 es_manager.val.env_configs.n_groups='[64]' \
    system.CUDA_VISIBLE_DEVICES="${DEVICES}" trainer.n_gpus_per_node=8 actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=200 \
    trainer.save_freq=50 \
    agent_proxy.max_turn="${turns}" \
    actor_rollout_ref.rollout.max_model_len="${max_len}" actor_rollout_ref.rollout.max_num_batched_tokens="${max_tok}" \
    trainer.default_local_dir="${out_dir}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    custom_envs.WebShop.max_actions_per_traj="${turns}" \
    actor_rollout_ref.actor.use_ref=False \
    trainer.nnodes=1
}

main() {
  run_experiment 3
  run_experiment 4
  run_experiment 5
  run_experiment 6
  run_experiment 7
}

main "$@"
