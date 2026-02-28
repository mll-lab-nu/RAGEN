#!/usr/bin/env bash
set -e

cd /workspace/RAGEN_temp
export PYTHONPATH="$PWD:$PWD/verl"

python train.py --config-name _11_deepcoder \
    model_path="Qwen/Qwen2.5-3B-Instruct"
    trainer.project_name=deepcoder_P1 \
    trainer.experiment_name=deepcoder_100turns_test \
    trainer.total_training_steps=100 \
    ppo_mini_batch_size=4 \
    micro_batch_size_per_gpu=1 \
    es_manager.train.env_groups=32 es_manager.train.group_size=1 es_manager.train.env_configs.n_groups=[32] \
    es_manager.val.env_groups=1 es_manager.val.group_size=1 es_manager.val.env_configs.n_groups=[1] \
    system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.save_freq=100 trainer.validation_steps=1 trainer.val_before_train=True \
    trainer.test_freq=20 \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
    trainer.nnodes=1 \
    agent_proxy.max_turn=1 \
    actor_rollout_ref.actor.use_ref=False \
    actor_rollout_ref.rollout.max_model_len=25000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=25000 \
    actor_rollout_ref.rollout.response_length=8000 \
    lora.rank=0 lora.alpha=64 lora.target_modules=all-linear \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    trainer.resume_mode=disable