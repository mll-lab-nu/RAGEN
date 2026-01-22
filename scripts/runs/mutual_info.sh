#!/bin/bash
set -euo pipefail

export RAY_TMPDIR=/dev/shm/ray

NUM_SAMPLES=64

LOG_DIR="logs/sokoban_mi_runs"
mkdir -p "${LOG_DIR}"

################# First-Turn Only #################
# PPO no filter
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_first_only_ppo_nofilter" \
    trainer.save_freq=-1 \
    algorithm.adv_estimator=gae \
    collapse_detection.first_turn_enabled=true \
    actor_rollout_ref.rollout.rollout_filter_ratio=1 \
    collapse_detection.multi_turn_enabled=false \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_first_only_ppo_nofilter.log" || true

# # PPO with filter
# python train.py --config-name "_2_sokoban" \
#     trainer.experiment_name="sokoban_mi_first_only_ppo_filter" \
#     trainer.save_freq=-1 \
#     algorithm.adv_estimator=gae \
#     collapse_detection.first_turn_enabled=true \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
#     collapse_detection.multi_turn_enabled=false \
#     2>&1 | tee "${LOG_DIR}/sokoban_mi_first_only_ppo_filter.log" || true

# # GRPO no filter
# python train.py --config-name "_2_sokoban" \
#     trainer.experiment_name="sokoban_mi_first_only_grpo_nofilter" \
#     trainer.save_freq=-1 \
#     algorithm.adv_estimator=grpo \
#     collapse_detection.first_turn_enabled=true \
#     actor_rollout_ref.rollout.rollout_filter_ratio=1 \
#     collapse_detection.multi_turn_enabled=false \
#     2>&1 | tee "${LOG_DIR}/sokoban_mi_first_only_grpo_nofilter.log" || true

# # GRPO with filter
# python train.py --config-name "_2_sokoban" \
#     trainer.experiment_name="sokoban_mi_first_only_grpo_filter" \
#     trainer.save_freq=-1 \
#     algorithm.adv_estimator=grpo \
#     collapse_detection.first_turn_enabled=true \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
#     collapse_detection.multi_turn_enabled=false \
#     2>&1 | tee "${LOG_DIR}/sokoban_mi_first_only_grpo_filter.log" || true


################# First-Turn and Multi-Turn #################

# # PPO no filter
# python train.py --config-name "_2_sokoban" \
#     trainer.experiment_name="sokoban_mi${NUM_SAMPLES}_ppo_nofilter" \
#     trainer.save_freq=-1 \
#     algorithm.adv_estimator=gae \
#     collapse_detection.first_turn_enabled=true \
#     actor_rollout_ref.rollout.rollout_filter_ratio=1 \
#     collapse_detection.multi_turn_enabled=true \
#     collapse_detection.num_samples=${NUM_SAMPLES} \
#     2>&1 | tee "${LOG_DIR}/sokoban_mi_${NUM_SAMPLES}_ppo_nofilter.log" || true

# # PPO with filter
# python train.py --config-name "_2_sokoban" \
#     trainer.experiment_name="sokoban_mi${NUM_SAMPLES}_ppo_filter" \
#     trainer.save_freq=-1 \
#     algorithm.adv_estimator=gae \
#     collapse_detection.first_turn_enabled=true \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
#     collapse_detection.multi_turn_enabled=true \
#     collapse_detection.num_samples=${NUM_SAMPLES} \
#     2>&1 | tee "${LOG_DIR}/sokoban_mi_${NUM_SAMPLES}_ppo_filter.log" || true

# # GRPO no filter
# python train.py --config-name "_2_sokoban" \
#     trainer.experiment_name="sokoban_mi${NUM_SAMPLES}_grpo_nofilter" \
#     trainer.save_freq=-1 \
#     algorithm.adv_estimator=grpo \
#     collapse_detection.first_turn_enabled=true \
#     actor_rollout_ref.rollout.rollout_filter_ratio=1 \
#     collapse_detection.multi_turn_enabled=true \
#     collapse_detection.num_samples=${NUM_SAMPLES} \
#     2>&1 | tee "${LOG_DIR}/sokoban_mi_${NUM_SAMPLES}_grpo_nofilter.log" || true

# # GRPO with filter
# python train.py --config-name "_2_sokoban" \
#     trainer.experiment_name="sokoban_mi${NUM_SAMPLES}_grpo_filter" \
#     trainer.save_freq=-1 \
#     algorithm.adv_estimator=grpo \
#     collapse_detection.first_turn_enabled=true \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
#     collapse_detection.multi_turn_enabled=true \
#     collapse_detection.num_samples=${NUM_SAMPLES} \
#     2>&1 | tee "${LOG_DIR}/sokoban_mi_${NUM_SAMPLES}_grpo_filter.log" || true
