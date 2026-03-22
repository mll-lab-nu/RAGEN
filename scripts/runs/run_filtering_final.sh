#!/bin/bash
set -euo pipefail

# usage: bash scripts/runs/run_filtering_final.sh [gpus_per_exp]
GPUS_PER_EXP="${1:-2}" 

# -----------------------
# GPU AUTO-DETECTION
# -----------------------
detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
  else
    echo 0
  fi
}

TOTAL_GPUS=$(detect_gpus)
if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected via nvidia-smi." >&2
    exit 1
fi
echo "INFO: Detected $TOTAL_GPUS GPUs."

# -----------------------
# Parallel GPU Management
# -----------------------
GPU_POOL_FIFO="/tmp/gpu_pool_$$"
mkfifo "$GPU_POOL_FIFO"
exec 3<>"$GPU_POOL_FIFO"
rm "$GPU_POOL_FIFO"

for ((i=0; i<TOTAL_GPUS; i++)); do
    echo "$i" >&3
done

# -----------------------
# Experiment List
# -----------------------
# Format: "ALGO METRIC STRATEGY VALUE TYPE INCLUDE_ZERO EXP_SUFFIX"
EXPS=(
    # Reward Variance - topP
    "ppo reward_variance top_p 1.0 largest False rv_tp1.0"
    "ppo reward_variance top_p 0.95 largest False rv_tp0.95"
    "ppo reward_variance top_p 0.9 largest False rv_tp0.9"
    "ppo reward_variance top_p 0.5 largest False rv_tp0.5"
    
    # Reward Variance - minP
    "ppo reward_variance min_p 0.1 largest False rv_mp0.1"
    "ppo reward_variance min_p 0.05 largest False rv_mp0.05"
    "ppo reward_variance min_p 0.2 largest False rv_mp0.2"
    
    # Reward Variance - topK (Fractional)
    "ppo reward_variance top_k 0.25 smallest False rv_tk0.25_rev"
    "ppo reward_variance top_k 0.5 smallest False rv_tk0.5_rev"
    "ppo reward_variance top_k 0.25 largest False rv_tk0.25"
    "ppo reward_variance top_k 0.5 largest False rv_tk0.5"
    
    # Entropy
    "ppo entropy top_p 0.9 largest False ent_tp0.9"
    
    # Entropy Variance
    "ppo entropy_variance top_p 0.9 largest False entv_tp0.9"
    
    # Length
    "ppo length top_p 0.9 largest False len_tp0.9"
    "ppo length top_p 0.9 smallest False len_tp0.9_rev"
)

# -----------------------
# Setup
# -----------------------
ENV="_2_sokoban"
OUTPUT_DIR="/mnt/permanent/xjin/20260126_filters_final"
mkdir -p "$OUTPUT_DIR"
DONE_LIST="filter_final_donelist.txt"
touch "$DONE_LIST"

COMMON_FLAGS_BASE="trainer.total_training_steps=400 micro_batch_size_per_gpu=4 ppo_mini_batch_size=32 trainer.save_freq=-1 \
    trainer.project_name=filtering_final \
    algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8]"

echo "========================================"
echo "Starting Final Filtering Experiments"
echo "Pool: $TOTAL_GPUS GPUs | Per-Exp: $GPUS_PER_EXP"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# -----------------------
# Execution Loop
# -----------------------
for exp_str in "${EXPS[@]}"; do
    read -r alg metric strategy value ftype inc_zero suffix <<< "$exp_str"
    
    exp_name="soko_3b_${alg}_${suffix}"
    
    if grep -q "^${exp_name}$" "$DONE_LIST"; then
        echo "Skipping ${exp_name} (Already Done)"
        continue
    fi

    # Acquire GPUs
    allocated_gpus=()
    for ((i=0; i<GPUS_PER_EXP; i++)); do
        read -u 3 gid
        allocated_gpus+=("$gid")
    done
    gpu_csv=$(IFS=,; echo "${allocated_gpus[*]}")

    (
        echo "Running: $exp_name on GPUs $gpu_csv"
        
        alg_flag="algorithm.adv_estimator=grpo"
        [ "$alg" == "ppo" ] && alg_flag="algorithm.adv_estimator=gae"

        if CUDA_VISIBLE_DEVICES="$gpu_csv" python train.py --config-name "$ENV" \
            trainer.experiment_name="${exp_name}" \
            actor_rollout_ref.rollout.rollout_filter_metric="${metric}" \
            actor_rollout_ref.rollout.rollout_filter_strategy="${strategy}" \
            actor_rollout_ref.rollout.rollout_filter_value="${value}" \
            actor_rollout_ref.rollout.rollout_filter_type="${ftype}" \
            actor_rollout_ref.rollout.rollout_filter_include_zero="${inc_zero}" \
            trainer.n_gpus_per_node="${GPUS_PER_EXP}" \
            system.CUDA_VISIBLE_DEVICES="\"${gpu_csv}\"" \
            $alg_flag \
            $COMMON_FLAGS_BASE \
            trainer.default_local_dir="${OUTPUT_DIR}/${exp_name}"; then
            
            echo "$exp_name" >> "$DONE_LIST"
        else
            echo "ERROR: $exp_name failed on GPUs $gpu_csv." >&2
        fi

        # Release GPUs
        for gid in "${allocated_gpus[@]}"; do
            echo "$gid" >&3
        done
    ) &
done

wait
exec 3>&-
echo "All requested experiments completed."
