# TEST DIFFERENT CONTEXT WINDOW MODES FOR SOKOBAN
USE_GRPO_WITH_FILTER="actor_rollout_ref.rollout.rollout_filter_ratio=0.25 algorithm.adv_estimator=grpo"
USE_PPO_WITH_FILTER="actor_rollout_ref.rollout.rollout_filter_ratio=0.25 algorithm.adv_estimator=gae"

USE_FULL="agent_proxy.context_window_mode=full agent_proxy.max_context_window=-1"
USE_LIMITED_MULTI_TURN="agent_proxy.context_window_mode=limited_multi_turn agent_proxy.max_context_window=3"
USE_SINGLE_TURN="agent_proxy.context_window_mode=single_turn agent_proxy.max_context_window=3"
USE_WITHOUT_HISTORY="agent_proxy.context_window_mode=single_turn agent_proxy.max_context_window=1"

# FULL CONTEXT WINDOW
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-ppo-rolloutfilter0.25-full $USE_PPO_WITH_FILTER $USE_FULL
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-grpo-rolloutfilter0.25-full $USE_GRPO_WITH_FILTER $USE_FULL

# LIMITED MULTI-TURN CONTEXT WINDOW
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-ppo-rolloutfilter0.25-limited_multi_turn $USE_PPO_WITH_FILTER $USE_LIMITED_MULTI_TURN
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-grpo-rolloutfilter0.25-limited_multi_turn $USE_GRPO_WITH_FILTER $USE_LIMITED_MULTI_TURN

# SINGLE-TURN CONTEXT WINDOW
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-ppo-rolloutfilter0.25-single_turn $USE_PPO_WITH_FILTER $USE_SINGLE_TURN
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-grpo-rolloutfilter0.25-single_turn $USE_GRPO_WITH_FILTER $USE_SINGLE_TURN

# WITHOUT HISTORY
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-ppo-rolloutfilter0.25-without_history $USE_PPO_WITH_FILTER $USE_WITHOUT_HISTORY
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-grpo-rolloutfilter0.25-without_history $USE_GRPO_WITH_FILTER $USE_WITHOUT_HISTORY
