# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo"
# USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"


# PPO Without history (Without rollout filter)
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-ppo-without-history agent_proxy.without_history=True $USE_PPO

# PPO Without history (With rollout filter)
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-ppo-without-history-rolloutfilter0.25 agent_proxy.without_history=True actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_PPO

# GRPO Without history (Without rollout filter)
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-grpo-without-history agent_proxy.without_history=True $USE_GRPO

# GRPO Without history (With rollout filter)
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-grpo-without-history-rolloutfilter0.25 agent_proxy.without_history=True actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_GRPO

# PPO With history (With rollout filter)
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-ppo-rolloutfilter0.25 agent_proxy.without_history=False actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_PPO


# TODO: Reward Nomalization? Need Test
