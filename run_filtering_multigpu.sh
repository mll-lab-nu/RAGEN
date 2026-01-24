#!/bin/bash
set -euo pipefail

# usage: bash run_filtering_multigpu.sh [grpo|ppo|all] [reward_variance|entropy|entropy_variance]
ALGO="${1:-grpo}" # default to grpo
METRIC="${2:-reward_variance}" # default to reward_variance
EXP_NAME="final0123"
DONE_LIST="filter_exp_donelist.txt"
touch "$DONE_LIST"

# -----------------------
# GPU autodetect + CUDA_VISIBLE_DEVICES
# -----------------------
detect_gpus() {
  # If CUDA_VISIBLE_DEVICES is already set by scheduler/user, respect it.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Count entries (handles "0,1,2" and also UUID lists)
    local n
    n=$(python - <<'PY'
import os
s=os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
if not s:
  print(0)
else:
  # remove empties
  parts=[p for p in s.split(",") if p.strip()!=""]
  print(len(parts))
PY
)
    echo "INFO: Using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (count=${n})" >&2
    echo "${CUDA_VISIBLE_DEVICES}" "${n}"
    return
  fi

  # Otherwise detect via nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
    if [[ "${n}" -gt 0 ]]; then
      # Use GPUs 0..n-1
      local devs=""
      for ((i=0; i<n; i++)); do
        devs+="${i},"
      done
      devs="${devs%,}"
      echo "INFO: Detected ${n} GPU(s) via nvidia-smi. Setting CUDA_VISIBLE_DEVICES=${devs}" >&2
      echo "${devs}" "${n}"
      return
    fi
  fi

  # Fallback: no GPU
  echo "WARN: No GPUs detected and CUDA_VISIBLE_DEVICES not set. Using CPU-only (CUDA_VISIBLE_DEVICES=)" >&2
  echo "" "0"
}

read -r CUDA_DEVS NGPUS <<<"$(detect_gpus)"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVS}"

# If your training framework expects these Hydra args:
NGPUS_PER_NODE="${NGPUS}"
# If you want to cap at 1 GPU even if more are available, set:
# NGPUS_PER_NODE=1
# CUDA_VISIBLE_DEVICES="0"

# -----------------------
# Common flags helper
# -----------------------
get_common_flags() {
  local metric=$1
  echo "trainer.total_training_steps=400 micro_batch_size_per_gpu=4 ppo_mini_batch_size=32 trainer.save_freq=-1 \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} system.CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\" \
    algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.rollout_filter_metric=${metric} \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8]"
}

ENV="_2_sokoban"
OUTPUT_DIR="/mnt/permanent/xjin/20260120_sokoban_filters"
mkdir -p "$OUTPUT_DIR"

# Define configurations to iterate
CONFIGS=(
    "top_p 0.5 topp50"
    "top_p 0.7 topp70"
    "top_p 0.9 topp90"
    "top_p 0.95 topp95"
    "min_p 0.3 minp30"
    "min_p 0.5 minp50"
    "min_p 0.8 minp80"
    "min_p 0.95 minp95"
    "top_k 4 topk4"
    "top_k 6 topk6"
)

TYPES=(
    "smallest small"
    "largest large"
)

INC_ZEROS=(
    "False noinc0"
)

LOSS_SCALES=(
    "sqrt sqrtscale"
)

run_exps_for_algo() {
    local alg_name=$1
    local alg_flag=$2
    local metric=$3
    local common_flags
    common_flags=$(get_common_flags "$metric")

    echo "========================================"
    echo "Starting experiments for: $alg_name | Metric: $metric"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} | n_gpus_per_node=${NGPUS_PER_NODE}"
    echo "========================================"

    # 1. Baseline: No Filtering
    local base_exp_name="soko_3b_${alg_name}_${metric}_nofilter"
    mkdir -p "${OUTPUT_DIR}/${base_exp_name}"
    if grep -q "^${base_exp_name}$" "$DONE_LIST"; then
        echo "Skipping ${base_exp_name} (Already in done-list)"
    else
        echo "Running Baseline: $base_exp_name (No Filtering)"
        if python train.py --config-name "$ENV" \
            trainer.experiment_name="${base_exp_name}" \
            actor_rollout_ref.rollout.rollout_filter_strategy="top_p" \
            actor_rollout_ref.rollout.rollout_filter_value=1.0 \
            actor_rollout_ref.rollout.rollout_filter_type="largest" \
            actor_rollout_ref.rollout.rollout_filter_include_zero=True \
            $alg_flag \
            $common_flags \
            trainer.default_local_dir="${OUTPUT_DIR}/${base_exp_name}"; then
            
            echo "$base_exp_name" >> "$DONE_LIST"
        else
            echo "ERROR: Baseline $base_exp_name failed. Continuing..." >&2
        fi
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

                    local exp_name="soko_3b_${alg_name}_${metric}_${stra_suffix}_${type_suffix}_${inc_suffix}_${scale_suffix}"
                    mkdir -p "${OUTPUT_DIR}/${exp_name}"
                    if grep -q "^${exp_name}$" "$DONE_LIST"; then
                        echo "Skipping ${exp_name} (Already in done-list)"
                    else
                        echo "Running Experiment: $exp_name (Strategy: $strategy, Value: $value, Type: $ftype, IncludeZero: $inc_bool, Scaling: $scaling)"

                        if python train.py --config-name "$ENV" \
                            trainer.experiment_name="${exp_name}" \
                            actor_rollout_ref.rollout.rollout_filter_strategy="${strategy}" \
                            actor_rollout_ref.rollout.rollout_filter_value=${value} \
                            actor_rollout_ref.rollout.rollout_filter_type="${ftype}" \
                            actor_rollout_ref.rollout.rollout_filter_include_zero=${inc_bool} \
                            actor_rollout_ref.actor.filter_loss_scaling="${scaling}" \
                            $alg_flag \
                            $common_flags \
                            trainer.default_local_dir="${OUTPUT_DIR}/${exp_name}"; then
                            
                            echo "$exp_name" >> "$DONE_LIST"
                        else
                            echo "ERROR: Experiment $exp_name failed. Continuing..." >&2
                        fi
                    fi
                done
            done
        done
    done
}

IFS=',' read -ra ALGOS <<< "$ALGO"
IFS=',' read -ra METRICS <<< "$METRIC"

for m in "${METRICS[@]}"; do
    for a in "${ALGOS[@]}"; do
        if [ "$a" == "grpo" ]; then
            run_exps_for_algo "grpo" "algorithm.adv_estimator=grpo" "$m"
        elif [ "$a" == "ppo" ]; then
            run_exps_for_algo "ppo" "algorithm.adv_estimator=gae" "$m"
        elif [ "$a" == "all" ]; then
            run_exps_for_algo "grpo" "algorithm.adv_estimator=grpo" "$m"
            run_exps_for_algo "ppo" "algorithm.adv_estimator=gae" "$m"
        else
            echo "Unknown algorithm argument: $a"
            echo "Usage: bash run_filtering_multigpu.sh [grpo|ppo|all] [reward_variance|entropy|entropy_variance]"
            exit 1
        fi
    done
done

echo "All requested experiments completed."

