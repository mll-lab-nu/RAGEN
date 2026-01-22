#!/bin/bash

# usage: bash run_filtering_exps.sh [grpo|ppo|all]

ALGO="${1:-grpo}" # default to grpo
DATE=$(date +%m%d)

COMMON_FLAGS="trainer.total_training_steps=100 micro_batch_size_per_gpu=4 ppo_mini_batch_size=16 trainer.save_freq=-1 trainer.n_gpus_per_node=1 system.CUDA_VISIBLE_DEVICES=\"0\" \
    algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8]"

ENV="_2_sokoban"
OUTPUT_DIR="/mnt/permanent/xjin/20260120_sokoban_filters"

mkdir -p $OUTPUT_DIR

# Define configurations to iterate
# Format: "strategy value suffix"
CONFIGS=(
    "top_p 0.7 topp70"
    "top_p 0.9 topp90"
    "min_p 0.5 minp50"
    "min_p 0.8 minp80"
    "top_k 4 topk4"
    "top_k 6 topk6"
)

# Format: "type suffix"
TYPES=(
    "smallest small"
    "largest large"
)

# Format: "bool suffix"
INC_ZEROS=(
    "False noinc0"
)

# Format: "scaling suffix"
LOSS_SCALES=(
    "sqrt sqrtscale"
)

run_exps_for_algo() {
    local alg_name=$1
    local alg_flag=$2

    echo "========================================"
    echo "Starting experiments for: $alg_name"
    echo "========================================"

    # 1. Baseline: No Filtering
    # top_p = 1.0, include_zero = True
    EXP_NAME="soko_3b_${alg_name}_nofilter"
    mkdir -p "${OUTPUT_DIR}/${EXP_NAME}"
    if [ -f "${OUTPUT_DIR}/${EXP_NAME}/DONE" ]; then
        echo "Skipping ${EXP_NAME} (Already Done)"
    else
        echo "Running Baseline: $EXP_NAME (No Filtering)"
        timeout 1h python train.py --config-name $ENV \
            trainer.experiment_name="${EXP_NAME}" \
            actor_rollout_ref.rollout.rollout_filter_strategy="top_p" \
            actor_rollout_ref.rollout.rollout_filter_value=1.0 \
            actor_rollout_ref.rollout.rollout_filter_type="largest" \
            actor_rollout_ref.rollout.rollout_filter_include_zero=True \
            $alg_flag \
            $COMMON_FLAGS \
            trainer.default_local_dir="${OUTPUT_DIR}/${EXP_NAME}"
        
        touch "${OUTPUT_DIR}/${EXP_NAME}/DONE"
    fi

    # 2. Grid Search
    for config_str in "${CONFIGS[@]}"; do
        read -r strategy value stra_suffix <<< "$config_str"
        
        for type_str in "${TYPES[@]}"; do
            read -r ftype type_suffix <<< "$type_str"
            
            for inc_str in "${INC_ZEROS[@]}"; do
                read -r inc_bool inc_suffix <<< "$inc_str"
                
                for scale_str in "${LOSS_SCALES[@]}"; do
                    read -r scaling scale_suffix <<< "$scale_str"
                    
                    EXP_NAME="soko_3b_${alg_name}_${stra_suffix}_${type_suffix}_${inc_suffix}_${scale_suffix}"
                    if [ -f "${OUTPUT_DIR}/${EXP_NAME}/DONE" ]; then
                        echo "Skipping ${EXP_NAME} (Already Done)"
                    else
                        echo "Running Experiment: $EXP_NAME (Strategy: $strategy, Value: $value, Type: $ftype, IncludeZero: $inc_bool, Scaling: $scaling)"
                        
                        timeout 1h python train.py --config-name $ENV \
                            trainer.experiment_name="${EXP_NAME}" \
                            actor_rollout_ref.rollout.rollout_filter_strategy="${strategy}" \
                            actor_rollout_ref.rollout.rollout_filter_value=${value} \
                            actor_rollout_ref.rollout.rollout_filter_type="${ftype}" \
                            actor_rollout_ref.rollout.rollout_filter_include_zero=${inc_bool} \
                            actor_rollout_ref.actor.filter_loss_scaling="${scaling}" \
                            $alg_flag \
                            $COMMON_FLAGS \
                            trainer.default_local_dir="${OUTPUT_DIR}/${EXP_NAME}"
                        
                        touch "${OUTPUT_DIR}/${EXP_NAME}/DONE"
                    fi
                done
            done
        done
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
    echo "Usage: bash run_filtering_exps.sh [grpo|ppo|all]"
    exit 1
fi

echo "All requested experiments completed."
