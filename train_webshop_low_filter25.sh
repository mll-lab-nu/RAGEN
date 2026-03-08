#!/bin/bash
# Historical filename: this script now runs the revert-alignment WebShop top_p=0.9 setup.
USE_LIMITED_MULTI_TURN="agent_proxy.context_window_mode=limited_multi_turn agent_proxy.max_context_window=3"

MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop \
    algorithm.adv_estimator=grpo \
    $USE_LIMITED_MULTI_TURN \
    trainer.project_name="main_webshop" \
    trainer.experiment_name=webshop3b_top_p90_wndw3_grpo_revert_alignment \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    es_manager.train.env_groups=4 es_manager.train.group_size=8 es_manager.train.env_configs.n_groups=[4] \
    es_manager.val.env_configs.n_groups=[2] \
    es_manager.val.env_groups=2 \
    es_manager.val.group_size=4 \
    system.CUDA_VISIBLE_DEVICES=\"0,1\" trainer.n_gpus_per_node=2 \
    trainer.default_local_dir=/mnt/permanent/xjin/20260118/webshop3b_top_p90_wndw3_grpo_revert_alignment \
    trainer.nnodes=1 micro_batch_size_per_gpu=1 \
    ppo_mini_batch_size=64 \
    agent_proxy.max_turn=15 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.rollout_filter_strategy=top_p \
    actor_rollout_ref.rollout.rollout_filter_value=0.9 \
    actor_rollout_ref.rollout.rollout_filter_include_zero=true \
    actor_rollout_ref.rollout.rollout_filter_type=largest \
    actor_rollout_ref.rollout.rollout_filter_metric=reward_variance \
    collapse_detection.first_turn_enabled=False \
    collapse_detection.multi_turn_enabled=False
