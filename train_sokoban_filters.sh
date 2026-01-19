#!/bin/bash

# Experiment script for Bandit, Sokoban, and Frozen Lake
# Testing different rollout filtering configurations

COMMON_FLAGS="trainer.total_training_steps=100 micro_batch_size_per_gpu=4 ppo_mini_batch_size=16 trainer.save_freq=-1 trainer.n_gpus_per_node=1 system.CUDA_VISIBLE_DEVICES=\"0\" algorithm.adv_estimator=grpo"

ENVS=("_1_bandit" "_2_sokoban" "_3_frozen_lake")

for env in "${ENVS[@]}"; do
    echo "Running experiments for environment: $env"

    # # Config 1: rollout_filter_ratio 0.5 + rollout_filter_include_zero true
    # echo "Config 1: Ratio 0.5, Include Zero True"
    # python train.py --config-name $env \
    #     trainer.experiment_name="${env#_}_ratio50_inc0" \
    #     actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
    #     actor_rollout_ref.rollout.rollout_filter_include_zero=True \
    #     $COMMON_FLAGS

    # # Config 2: rollout_filter_lower_ratio 0.25 + rollout_filter_include_zero false
    # echo "Config 2: Lower Ratio 0.25, Include Zero False"
    # EXP_NAME="${env#_}_low25_noinc0"
    # python train.py --config-name $env \
    #     trainer.experiment_name="${EXP_NAME}" \
    #     actor_rollout_ref.rollout.rollout_filter_lower_ratio=0.25 \
    #     actor_rollout_ref.rollout.rollout_filter_include_zero=False \
    #     $COMMON_FLAGS \
    #     trainer.default_local_dir=/mnt/permanent/xjin/20260118/${EXP_NAME}

    # Config 3: rollout_filter_lower_ratio 0.5 + rollout_filter_include_zero false
    echo "Config 3: Lower Ratio 0.5, Include Zero False"
    EXP_NAME="${env#_}_low50_noinc0"
    python train.py --config-name $env \
        trainer.experiment_name="${EXP_NAME}" \
        actor_rollout_ref.rollout.rollout_filter_lower_ratio=0.5 \
        actor_rollout_ref.rollout.rollout_filter_include_zero=False \
        $COMMON_FLAGS \
        trainer.default_local_dir=/mnt/permanent/xjin/20260118/${EXP_NAME}

done

echo "All experiments completed."
