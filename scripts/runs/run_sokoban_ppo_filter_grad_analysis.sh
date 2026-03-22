#!/bin/bash
# Runner for Sokoban PPO/GRPO with top-p=0.9 filtering plus periodic
# gradient-analysis passes during training.
#
# This intentionally stays close to scripts/runs/run_main_table_diff_algo.sh:
# - task: sokoban
# - algo: PPO or GRPO
# - filter: top_p=0.9
# - model: Qwen2.5-3B
# - train batch: 8 groups x 16 samples (same as the main-table setup)
# - gradient-analysis batch: 128 groups x 16 samples
#
# Validation policy:
# - one validation before training
# - periodic validation every 10 steps
#
# Gradient-analysis policy:
# - run gradient analysis every 50 steps
# - current trainer trigger is step 1, 51, 101, ...
# - do not exit after the analysis step

set -euo pipefail

STEPS=101
SAVE_FREQ=100
GPU_MEMORY_UTILIZATION=0.3
RAY_NUM_CPUS=16
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
GPUS=()
GPUS_PROVIDED=false

MODEL_NAME="Qwen2.5-3B"
MODEL_PATH="Qwen/${MODEL_NAME}"
TASK="sokoban"
ALGO="PPO"
FILTER_LABEL="filter"
FILTER_VALUE="0.9"
GROUP_SIZE=16
ENV_GROUPS=8
ANALYSIS_GROUP_SIZE=16
ANALYSIS_ENV_GROUPS=128
CONFIG_NAME="_2_sokoban"

normalize_algo() {
    case "$1" in
        PPO|ppo) echo "PPO" ;;
        GRPO|grpo) echo "GRPO" ;;
        *)
            echo "Error: unsupported --algo '$1'. Supported values: PPO, GRPO" >&2
            exit 1
            ;;
    esac
}

get_algo_overrides() {
    case "$1" in
        PPO)
            echo "algorithm.adv_estimator=gae actor_rollout_ref.actor.loss_agg_mode=token-mean"
            ;;
        GRPO)
            echo "algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=True actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
            ;;
        *)
            echo "Error: unsupported algorithm '$1'" >&2
            exit 1
            ;;
    esac
}

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --algo NAME                 Training algorithm: PPO or GRPO (default: PPO)"
    echo "  --steps N                    Training steps (default: 101)"
    echo "  --gpus LIST                  Comma-separated GPU IDs (default: auto-detect)"
    echo "  --gpu-memory-utilization V   Rollout gpu_memory_utilization (default: 0.3)"
    echo "  --ray-num-cpus N             Max CPUs for ray.init (default: 16)"
    echo "  --ppo-micro-batch-size-per-gpu N      PPO micro batch size per GPU for actor/critic (default: 4)"
    echo "  --log-prob-micro-batch-size-per-gpu N log-prob micro batch size per GPU for ref/rollout (default: 4)"
    echo "  --save-freq N                Checkpoint save frequency (default: 100)"
    echo "  -h, --help                   Show this help"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --algo) ALGO="$(normalize_algo "$2")"; shift 2 ;;
        --algo=*) ALGO="$(normalize_algo "${1#*=}")"; shift ;;
        --steps) STEPS="$2"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        --gpus) IFS=',' read -r -a GPUS <<< "$2"; GPUS_PROVIDED=true; shift 2 ;;
        --gpus=*) IFS=',' read -r -a GPUS <<< "${1#*=}"; GPUS_PROVIDED=true; shift ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --gpu-memory-utilization=*) GPU_MEMORY_UTILIZATION="${1#*=}"; shift ;;
        --ray-num-cpus) RAY_NUM_CPUS="$2"; shift 2 ;;
        --ray-num-cpus=*) RAY_NUM_CPUS="${1#*=}"; shift ;;
        --ppo-micro-batch-size-per-gpu) PPO_MICRO_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
        --ppo-micro-batch-size-per-gpu=*) PPO_MICRO_BATCH_SIZE_PER_GPU="${1#*=}"; shift ;;
        --log-prob-micro-batch-size-per-gpu) LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
        --log-prob-micro-batch-size-per-gpu=*) LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${1#*=}"; shift ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --save-freq=*) SAVE_FREQ="${1#*=}"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [ "$GPUS_PROVIDED" = false ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && [ "$GPU_COUNT" -gt 0 ]; then
            GPUS=()
            for ((i=0; i<GPU_COUNT; i++)); do
                GPUS+=("$i")
            done
        fi
    fi
    if [ ${#GPUS[@]} -eq 0 ]; then
        echo "Warning: failed to auto-detect GPUs, falling back to 0-7" >&2
        GPUS=(0 1 2 3 4 5 6 7)
    fi
fi

if ! [[ "$RAY_NUM_CPUS" =~ ^[0-9]+$ ]] || [ "$RAY_NUM_CPUS" -lt 1 ]; then
    echo "Error: --ray-num-cpus must be a positive integer" >&2
    exit 1
fi
if ! [[ "$PPO_MICRO_BATCH_SIZE_PER_GPU" =~ ^[0-9]+$ ]] || [ "$PPO_MICRO_BATCH_SIZE_PER_GPU" -lt 1 ]; then
    echo "Error: --ppo-micro-batch-size-per-gpu must be a positive integer" >&2
    exit 1
fi
if ! [[ "$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" =~ ^[0-9]+$ ]] || [ "$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" -lt 1 ]; then
    echo "Error: --log-prob-micro-batch-size-per-gpu must be a positive integer" >&2
    exit 1
fi

ALGO_OVERRIDES=$(get_algo_overrides "$ALGO")
read -r -a ALGO_ARGS <<< "$ALGO_OVERRIDES"

GPU_LIST=$(IFS=,; echo "${GPUS[*]}")
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: no GPUs available" >&2
    exit 1
fi

EXP_NAME="${TASK}-${ALGO}-${FILTER_LABEL}-topp09-${MODEL_NAME}-train${ENV_GROUPS}x${GROUP_SIZE}-analysis${ANALYSIS_ENV_GROUPS}x${ANALYSIS_GROUP_SIZE}-grad-every50"
LOG_DIR="logs/gradient_analysis_${TASK}_${MODEL_NAME}"
CHECKPOINT_DIR="model_saving/gradient_analysis/${TASK}/${ALGO}/${FILTER_LABEL}/${EXP_NAME}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.log"

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "=== Gradient Analysis Runner: $(date) ===" | tee "$LOG_PATH"
echo "task=${TASK} algo=${ALGO} filter=top_p:${FILTER_VALUE} model=${MODEL_NAME} steps=${STEPS} gpus=${GPU_LIST}" | tee -a "$LOG_PATH"
echo "train_group_size=${GROUP_SIZE} train_env_groups=${ENV_GROUPS} analysis_group_size=${ANALYSIS_GROUP_SIZE} analysis_env_groups=${ANALYSIS_ENV_GROUPS} gradient_analysis_every=50 exit_after_gradient_analysis=False" | tee -a "$LOG_PATH"
echo "ppo_micro_batch_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} log_prob_micro_batch_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" | tee -a "$LOG_PATH"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" python train.py --config-name "${CONFIG_NAME}" \
    model_path="${MODEL_PATH}" \
    trainer.project_name="ragen_gradient_analysis" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_training_steps="${STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.logger="['console','wandb']" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
    system.CUDA_VISIBLE_DEVICES="'${GPU_LIST}'" \
    es_manager.train.env_groups="${ENV_GROUPS}" \
    es_manager.train.group_size="${GROUP_SIZE}" \
    es_manager.train.env_configs.n_groups="[${ENV_GROUPS}]" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.filter_loss_scaling=none \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.rollout_filter_value="${FILTER_VALUE}" \
    actor_rollout_ref.rollout.rollout_filter_strategy=top_p \
    actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=linear \
    actor_rollout_ref.rollout.rollout_filter_type=largest \
    actor_rollout_ref.rollout.rollout_filter_metric=reward_variance \
    actor_rollout_ref.rollout.rollout_filter_include_zero=False \
    actor_rollout_ref.actor.checkpoint.save_contents=[model] \
    critic.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    critic.checkpoint.save_contents=[model] \
    trainer.gradient_analysis_mode=True \
    trainer.gradient_analysis_every=50 \
    trainer.gradient_analysis_env_groups="${ANALYSIS_ENV_GROUPS}" \
    trainer.gradient_analysis_group_size="${ANALYSIS_GROUP_SIZE}" \
    trainer.gradient_analysis_log_prefilter=True \
    trainer.exit_after_gradient_analysis=False \
    actor_rollout_ref.rollout.gradient_analysis_num_buckets=6 \
    actor_rollout_ref.rollout.gradient_analysis_bucket_mode=quantile \
    "${ALGO_ARGS[@]}" \
    2>&1 | tee -a "$LOG_PATH"
