#!/bin/bash

# usage: bash run_loss_scaling_exps.sh [grpo|ppo|all]

ALGO="${1:-grpo}" # default to grpo
DATE=$(date +%m%d)

COMMON_FLAGS="trainer.total_training_steps=100 micro_batch_size_per_gpu=4 ppo_mini_batch_size=16 trainer.save_freq=-1 trainer.n_gpus_per_node=1 system.CUDA_VISIBLE_DEVICES=\"0\" \
    algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8]"

ENV="_2_sokoban"
OUTPUT_DIR="/mnt/permanent/xjin/20260119_sokoban_filters"

mkdir -p $OUTPUT_DIR

# Define loss scaling strategies to iterate
# Format: "strategy suffix"
SCALING_STRATEGIES=(
    "none noscale"
    "linear linscale"
    "sqrt sqrtscale"
)

run_exps_for_algo() {
    local alg_name=$1
    local alg_flag=$2

    echo "========================================"
    echo "Starting Loss Scaling Experiments for: $alg_name"
    echo "========================================"

    # Grid Search for Scaling Strategies
    for scale_str in "${SCALING_STRATEGIES[@]}"; do
        read -r scaling_strategy scale_suffix <<< "$scale_str"
        
        # Fixed filtering parameters as requested
        filter_strategy="top_p"
        filter_value=0.9
        filter_suffix="topp90"
        
        EXP_NAME="soko_3b_${alg_name}_${filter_suffix}_${scale_suffix}"
        
        if [ -f "${OUTPUT_DIR}/${EXP_NAME}/DONE" ]; then
            echo "Skipping ${EXP_NAME} (Already Done)"
        else
            echo "Running Experiment: $EXP_NAME (Filter: $filter_strategy $filter_value, Scaling: $scaling_strategy)"
            mkdir -p "${OUTPUT_DIR}/${EXP_NAME}"
            
            timeout 2h python train.py --config-name $ENV \
                trainer.experiment_name="${EXP_NAME}" \
                actor_rollout_ref.rollout.rollout_filter_strategy="${filter_strategy}" \
                actor_rollout_ref.rollout.rollout_filter_value=${filter_value} \
                actor_rollout_ref.rollout.rollout_filter_type="largest" \
                actor_rollout_ref.rollout.rollout_filter_include_zero=False \
                actor_rollout_ref.actor.filter_loss_scaling="${scaling_strategy}" \
                $alg_flag \
                $COMMON_FLAGS \
                trainer.default_local_dir="${OUTPUT_DIR}/${EXP_NAME}"
            
            touch "${OUTPUT_DIR}/${EXP_NAME}/DONE"
        fi
    done
}

if [ "$ALGO" == "grpo" ]; then
    run_exps_for_algo "grpo" "algorithm.adv_estimator=grpo"
elif [ "$ALGO" == "ppo" ]; then
    run_exps_for_algo "ppo" "algorithm.adv_estimator=gae"
elif [ "$ALGO" == "all" ]; then
    run_exps_for_algo "grpo" "algorithm.adv_estimator=grpo"
    run_exps_for_algo "ppo" "algorithm.adv_estimator=gae"
else
    echo "Unknown algorithm argument: $ALGO"
    echo "Usage: bash run_loss_scaling_exps.sh [grpo|ppo|all]"
    exit 1
fi

echo "All requested loss scaling experiments completed."
