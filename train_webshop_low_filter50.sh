USE_LIMITED_MULTI_TURN="agent_proxy.context_window_mode=limited_multi_turn agent_proxy.max_context_window=3"

MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.rollout_filter_lower_ratio=0.5 \
    actor_rollout_ref.rollout.rollout_filter_include_zero=False \
    $USE_LIMITED_MULTI_TURN \
    trainer.experiment_name=webshop3b_lowfilter50_wndw3_grpo \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    es_manager.train.env_groups=16 es_manager.train.group_size=8 es_manager.train.env_configs.n_groups=[16] \
    system.CUDA_VISIBLE_DEVICES=\"2,3\" trainer.n_gpus_per_node=2 \
    trainer.default_local_dir=/mnt/permanent/xjin/20260118/webshop3b_lowfilter50_wndw3_grpo \
    trainer.nnodes=1 micro_batch_size_per_gpu=1 \
    ppo_mini_batch_size=64 \
    agent_proxy.max_turn=15
