#!/usr/bin/env bash
set -e

cd /workspace/RAGEN
export PYTHONPATH="$PWD:$PWD/verl"
export NCCL_DEBUG=WARN

USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" 

python train.py --config-name _10_deepcoder $USE_PPO\
    model_path="Qwen/Qwen2.5-3B-Instruct" \
    trainer.project_name=deepcoder_RAGEN_final \
    trainer.experiment_name=deepcoder_3binstructppo_200turns_test_3_filter \
    trainer.total_training_steps=200 \
    actor_rollout_ref.nccl_timeout=120 \
    ppo_mini_batch_size=4 \
    micro_batch_size_per_gpu=2 \
    es_manager.train.env_groups=16 es_manager.train.group_size=8 es_manager.train.env_configs.tags=["DeepCoder"] es_manager.train.env_configs.n_groups=[16] \
    es_manager.val.env_groups=128 es_manager.val.group_size=1 es_manager.val.env_configs.tags=["DeepCoder"] es_manager.val.env_configs.n_groups=[128] \
    system.CUDA_VISIBLE_DEVICES=\"0,1\" trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.save_freq=20 trainer.validation_steps=1 trainer.val_before_train=True \
    trainer.test_freq=10 \
    actor_rollout_ref.nccl_timeout=120 \
    actor_rollout_ref.rollout.rollout_filter_value=0.9 \
    actor_rollout_ref.rollout.rollout_filter_strategy=top_p \
    actor_rollout_ref.rollout.rollout_filter_type=largest \
    actor_rollout_ref.rollout.rollout_filter_include_zero=False \
    actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=linear \
    trainer.nnodes=1 \
    agent_proxy.max_turn=1 \
    actor_rollout_ref.actor.use_ref=False \
    actor_rollout_ref.rollout.max_model_len=6000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=6000 \
    actor_rollout_ref.rollout.response_length=5000 \
    lora.rank=0 lora.alpha=64 lora.target_modules=all-linear \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    trainer.resume_mode=disable