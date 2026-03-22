#!/bin/bash
# Search Benchmark: models × algorithms × filter/no-filter on SearchQA (HotpotQA + Dense Retrieval)
# Models: Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Llama-3.2-3B-Instruct
# Algorithms: PPO, GRPO
# Filtering rule: filter => top_p=0.9, nofilter => top_p=1.0

set -euo pipefail

# Activate ragen conda environment
eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate ragen 2>/dev/null || true

# Use user-writable datasets cache to avoid permission conflicts with root-owned lock files
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.hf_cache/datasets}"

# Defaults
STEPS=200
MODEL_NAMES=("Qwen2.5-3B-Instruct")
ALGORITHMS=("PPO")
SAVE_FREQ=-1
FILTER_STRATEGY="top_p"
FILTER_VALUE=""

# GPU settings
GPUS=()
GPUS_PROVIDED=false
GPUS_PER_EXP=1
COOLDOWN_SECONDS=30
GPU_MEMORY_UTILIZATION=0.6
TENSOR_PARALLEL_SIZE=1
MICRO_BATCH_SIZE=""
MINI_BATCH_SIZE=""
COLLAPSE_FREQ=""
RETRIEVAL_PORT=""
declare -A GPU_LABELS

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --steps N             Training steps (default: 400)"
    echo "  --models LIST         Comma-separated model names (default: Qwen2.5-3B-Instruct)"
    echo "  --algos LIST          Comma-separated algorithms: PPO,GRPO (default: PPO)"
    echo "  --gpus LIST           Comma-separated GPU IDs (default: auto-detect)"
    echo "  --gpus-per-exp N      GPUs per experiment (default: 1)"
    echo "  --cooldown SECONDS    Cooldown between runs on the same GPU group (default: 30)"
    echo "  --gpu-memory-utilization V  Rollout gpu_memory_utilization (default: 0.6)"
    echo "  --tp N                Tensor parallel size for vLLM rollout (default: 1)"
    echo "  --micro-batch N       micro_batch_size_per_gpu override"
    echo "  --mini-batch N        ppo_mini_batch_size override"
    echo "  --collapse-freq N     collapse_detection.compute_freq override"
    echo "  --save-freq N         Checkpoint save frequency (default: -1 to disable saving)"
    echo "  --retrieval-port N    Retrieval server port (overrides default 8000 in config)"
    echo "  --filter-strategy S   Rollout filter strategy: top_p, top_k, etc. (default: top_p)"
    echo "  --filter-value V     Rollout filter value (default: 1.0 for no filtering)"
    echo "  -h, --help            Show this help"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        --models) IFS=',' read -r -a MODEL_NAMES <<< "$2"; shift 2 ;;
        --models=*) IFS=',' read -r -a MODEL_NAMES <<< "${1#*=}"; shift ;;
        --algos) IFS=',' read -r -a ALGORITHMS <<< "$2"; shift 2 ;;
        --algos=*) IFS=',' read -r -a ALGORITHMS <<< "${1#*=}"; shift ;;
        --gpus) IFS=',' read -r -a GPUS <<< "$2"; GPUS_PROVIDED=true; shift 2 ;;
        --gpus=*) IFS=',' read -r -a GPUS <<< "${1#*=}"; GPUS_PROVIDED=true; shift ;;
        --gpus-per-exp) GPUS_PER_EXP="$2"; shift 2 ;;
        --gpus-per-exp=*) GPUS_PER_EXP="${1#*=}"; shift ;;
        --cooldown) COOLDOWN_SECONDS="$2"; shift 2 ;;
        --cooldown=*) COOLDOWN_SECONDS="${1#*=}"; shift ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --gpu-memory-utilization=*) GPU_MEMORY_UTILIZATION="${1#*=}"; shift ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --save-freq=*) SAVE_FREQ="${1#*=}"; shift ;;
        --tp) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --tp=*) TENSOR_PARALLEL_SIZE="${1#*=}"; shift ;;
        --micro-batch) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --micro-batch=*) MICRO_BATCH_SIZE="${1#*=}"; shift ;;
        --mini-batch) MINI_BATCH_SIZE="$2"; shift 2 ;;
        --mini-batch=*) MINI_BATCH_SIZE="${1#*=}"; shift ;;
        --collapse-freq) COLLAPSE_FREQ="$2"; shift 2 ;;
        --collapse-freq=*) COLLAPSE_FREQ="${1#*=}"; shift ;;
        --retrieval-port) RETRIEVAL_PORT="$2"; shift 2 ;;
        --retrieval-port=*) RETRIEVAL_PORT="${1#*=}"; shift ;;
        --filter-strategy) FILTER_STRATEGY="$2"; shift 2 ;;
        --filter-strategy=*) FILTER_STRATEGY="${1#*=}"; shift ;;
        --filter-value) FILTER_VALUE="$2"; shift 2 ;;
        --filter-value=*) FILTER_VALUE="${1#*=}"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# Default filter value: 1.0 (no filtering) if not specified
if [ -z "$FILTER_VALUE" ]; then
    FILTER_VALUE=1.0
fi

# Derive a short filter label for experiment naming / logs
if [ "$FILTER_VALUE" = "1.0" ] || [ "$FILTER_VALUE" = "1" ]; then
    FILTER_LABEL="nofilter"
else
    FILTER_LABEL="${FILTER_STRATEGY}${FILTER_VALUE}"
fi

# Fixed: search task config
CONFIG="_9_search"

# Map model names to HuggingFace paths
get_model_path() {
    if [[ "$1" == *"/"* ]]; then
        echo "$1"
        return
    fi
    case "$1" in
        Qwen2.5-3B-Instruct) echo "Qwen/Qwen2.5-3B-Instruct" ;;
        Qwen2.5-7B-Instruct) echo "Qwen/Qwen2.5-7B-Instruct" ;;
        Llama-3.2-3B-Instruct) echo "meta-llama/Llama-3.2-3B-Instruct" ;;
        *) echo "Qwen/$1" ;;
    esac
}

# Algorithm-specific overrides
get_algo_overrides() {
    case "$1" in
        PPO)
            echo "algorithm.adv_estimator=gae actor_rollout_ref.actor.loss_agg_mode=token-mean"
            ;;
        GRPO)
            echo "algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=True actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Auto-detect GPUs
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

if ! [[ "$GPUS_PER_EXP" =~ ^[0-9]+$ ]] || [ "$GPUS_PER_EXP" -lt 1 ]; then
    echo "Error: --gpus-per-exp must be a positive integer"
    exit 1
fi
if (( ${#GPUS[@]} < GPUS_PER_EXP )); then
    echo "Error: --gpus-per-exp (${GPUS_PER_EXP}) exceeds available GPUs (${#GPUS[@]})"
    exit 1
fi
if (( ${#GPUS[@]} % GPUS_PER_EXP != 0 )); then
    echo "Error: GPU count (${#GPUS[@]}) must be divisible by --gpus-per-exp (${GPUS_PER_EXP})"
    exit 1
fi

GPU_GROUPS=()
for ((i=0; i<${#GPUS[@]}; i+=GPUS_PER_EXP)); do
    group="${GPUS[$i]}"
    for ((j=1; j<GPUS_PER_EXP; j++)); do
        group+=",${GPUS[$((i+j))]}"
    done
    GPU_GROUPS+=("$group")
done
NUM_SLOTS=${#GPU_GROUPS[@]}

short_gpu_name() {
    local name="$1"
    local cleaned
    cleaned=$(echo "$name" | sed -E 's/^NVIDIA //; s/^Tesla //; s/^GeForce //; s/^Quadro //; s/^RTX //')
    if [[ "$cleaned" =~ (B[0-9]{2,3}|H[0-9]{2,3}|A[0-9]{2,3}|L[0-9]{2,3}|V100|T4|P100|K80) ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi
    echo "${cleaned%% *}"
}

get_gpu_label() {
    local gpu_id="$1"
    if [ -n "${GPU_LABELS[$gpu_id]+x}" ]; then
        echo "${GPU_LABELS[$gpu_id]}"
        return
    fi
    local name=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu_id" 2>/dev/null | head -1)
    fi
    if [ -z "$name" ]; then
        GPU_LABELS[$gpu_id]="1xGPU${gpu_id}"
        echo "${GPU_LABELS[$gpu_id]}"
        return
    fi
    local short
    short=$(short_gpu_name "$name")
    GPU_LABELS[$gpu_id]="1x${short}"
    echo "${GPU_LABELS[$gpu_id]}"
}

get_gpu_model_label() {
    local models=()
    local id label model
    for id in "${GPUS[@]}"; do
        label=$(get_gpu_label "$id")
        model="${label#1x}"
        models+=("$model")
    done
    local unique_models=()
    local m found
    for m in "${models[@]}"; do
        found=false
        for u in "${unique_models[@]}"; do
            if [ "$u" = "$m" ]; then
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            unique_models+=("$m")
        fi
    done
    if [ ${#unique_models[@]} -eq 1 ]; then
        echo "${unique_models[0]}"
    else
        echo "mixed"
    fi
}

get_gpu_label_for_list() {
    local gpu_list="$1"
    IFS=',' read -r -a ids <<< "$gpu_list"
    local count=${#ids[@]}
    if [ "$count" -eq 0 ]; then
        echo "0xGPU"
        return
    fi
    local first_model
    first_model="$(get_gpu_label "${ids[0]}")"
    first_model="${first_model#1x}"
    local id model
    for id in "${ids[@]:1}"; do
        model="$(get_gpu_label "$id")"
        model="${model#1x}"
        if [ "$model" != "$first_model" ]; then
            echo "${count}xmixed"
            return
        fi
    done
    echo "${count}x${first_model}"
}

GPU_MODEL_LABEL=$(get_gpu_model_label)
GPU_LOG_LABEL="${GPUS_PER_EXP}x${GPU_MODEL_LABEL}"
LOG_DIR="logs/search_benchmark"
LOG_FILE="logs/search_benchmark.log"
RESULT_ROOT="logs/search_benchmark"
CHECKPOINT_ROOT="model_saving/search_benchmark"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_ROOT"
mkdir -p "$CHECKPOINT_ROOT"

echo "=== Search Benchmark Runner: $(date) ===" | tee "$LOG_FILE"
echo "Models: ${MODEL_NAMES[*]} | Algos: ${ALGORITHMS[*]} | Filter: ${FILTER_STRATEGY}:${FILTER_VALUE} | Steps: ${STEPS} | GPU per exp: ${GPUS_PER_EXP}x${GPU_MODEL_LABEL}" | tee -a "$LOG_FILE"
echo "GPUS: ${GPUS[*]} | groups: ${GPU_GROUPS[*]} | cooldown=${COOLDOWN_SECONDS}s" | tee -a "$LOG_FILE"

run_experiment() {
    local model_name=$1
    local algo=$2
    local gpu_list=$3

    local model_path
    model_path=$(get_model_path "$model_name")

    local filter_strategy="$FILTER_STRATEGY"
    local filter_value="$FILTER_VALUE"

    local common_overrides=(
        "actor_rollout_ref.actor.use_kl_loss=False"
        "actor_rollout_ref.actor.kl_loss_type=low-var-kl"
        "actor_rollout_ref.actor.kl_loss_coef=0.001"
        "actor_rollout_ref.actor.entropy_coeff=0.001"
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True"
        "actor_rollout_ref.actor.filter_loss_scaling=none"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE}"
    )

    if [ -n "$MICRO_BATCH_SIZE" ]; then
        common_overrides+=("micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}")
    fi
    if [ -n "$MINI_BATCH_SIZE" ]; then
        common_overrides+=("ppo_mini_batch_size=${MINI_BATCH_SIZE}")
    fi
    if [ -n "$COLLAPSE_FREQ" ]; then
        common_overrides+=("collapse_detection.compute_freq=${COLLAPSE_FREQ}")
    fi

    local checkpoint_overrides=(
        "actor_rollout_ref.actor.checkpoint.save_contents=[model]"
        "critic.checkpoint.save_contents=[model]"
    )

    local retrieval_overrides=()
    if [ -n "$RETRIEVAL_PORT" ]; then
        retrieval_overrides+=("+custom_envs.SearchQA.env_config.retrieval_server_url=http://127.0.0.1:${RETRIEVAL_PORT}")
    fi

    local algo_overrides
    algo_overrides=$(get_algo_overrides "$algo")
    read -r -a algo_args <<< "$algo_overrides"

    local name="search-${algo}-${FILTER_LABEL}-${model_name}"
    local log_path="${LOG_DIR}/${name}.log"
    local checkpoint_dir="${CHECKPOINT_ROOT}/${model_name}/${algo}/${FILTER_LABEL}/${name}"
    local gpus_per_exp
    IFS=',' read -r -a gpu_ids <<< "$gpu_list"
    gpus_per_exp=${#gpu_ids[@]}

    # Limit Ray CPU workers to avoid spawning hundreds of idle workers (default: 8 per GPU)
    local ray_num_cpus=$((gpus_per_exp * 8))
    common_overrides+=("ray_kwargs.ray_init.num_cpus=${ray_num_cpus}")

    mkdir -p "${checkpoint_dir}"
    START=$(date +%s)
    CUDA_VISIBLE_DEVICES="${gpu_list}" python train.py --config-name "$CONFIG" \
        model_path="${model_path}" \
        trainer.project_name="ragen_search_benchmark" \
        trainer.total_training_steps="${STEPS}" \
        trainer.experiment_name="${name}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.default_local_dir="${checkpoint_dir}" \
        trainer.logger="['console','wandb']" \
        trainer.val_before_train=True \
        trainer.n_gpus_per_node="${gpus_per_exp}" \
        system.CUDA_VISIBLE_DEVICES="'${gpu_list}'" \
        actor_rollout_ref.rollout.rollout_filter_strategy="${filter_strategy}" \
        actor_rollout_ref.rollout.rollout_filter_value="${filter_value}" \
        "${common_overrides[@]}" \
        "${checkpoint_overrides[@]}" \
        "${retrieval_overrides[@]}" \
        "${algo_args[@]}" \
        2>&1 | tee "$log_path"
    EXIT_CODE=${PIPESTATUS[0]}
    END=$(date +%s)

    TOTAL_TIME=$((END - START))
    timing_values=()
    mapfile -t timing_values < <(
        python - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

def last(pattern, text):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""

try:
    text = Path(sys.argv[1]).read_text(errors="ignore")
except Exception:
    text = ""

patterns = [
    r"timing_s/train_total[:\s]+([\d.]+)",
    r"timing_s/eval_total[:\s]+([\d.]+)",
    r"timing_s/total[:\s]+([\d.]+)",
]

for pattern in patterns:
    print(last(pattern, text))
PY
    )
    TRAIN_TIME_RAW="${timing_values[0]:-}"
    EVAL_TIME_RAW="${timing_values[1]:-}"
    TOTAL_TIME_RAW="${timing_values[2]:-}"
    TRAIN_TIME=$([ -n "$TRAIN_TIME_RAW" ] && printf "%.2f" "$TRAIN_TIME_RAW" || echo "N/A")
    EVAL_TIME=$([ -n "$EVAL_TIME_RAW" ] && printf "%.2f" "$EVAL_TIME_RAW" || echo "N/A")
    TOTAL_TIME_METRIC=$([ -n "$TOTAL_TIME_RAW" ] && printf "%.2f" "$TOTAL_TIME_RAW" || echo "N/A")

    local status="success"
    local error_line=""
    if [ $EXIT_CODE -ne 0 ]; then
        status="fail"
        error_line=$(tail -2 "$log_path" | tr '\n' ' ')
    fi

    local gpu_label
    gpu_label=$(get_gpu_label_for_list "$gpu_list")
    local summary_line="task=search | algo=${algo} | filter=${FILTER_LABEL} | model=${model_name} | steps=${STEPS} | strategy=${filter_strategy}:${filter_value} | train_time=${TRAIN_TIME}s | eval_time=${EVAL_TIME}s | total_time=${TOTAL_TIME_METRIC}s | wall_time=${TOTAL_TIME}s | gpu=${gpu_label} | status=${status}"
    echo "${summary_line}" > "${LOG_DIR}/${name}.result"
    echo "${summary_line}" | tee -a "$LOG_FILE"
    if [ "$status" = "fail" ]; then
        echo "  error: ${error_line}" | tee -a "$LOG_FILE"
    fi
    return 0
}

EXPERIMENTS=()
GROUP_LABELS=()
CURRENT_GROUP=""

set_group() {
    CURRENT_GROUP="$1"
    GROUP_LABELS+=("$1")
}

add_experiment() {
    local model_name=$1
    local algo=$2
    EXPERIMENTS+=("${CURRENT_GROUP}|${model_name}|${algo}")
}

# Build experiment list: models × algos (filter is a global setting)
for model_name in "${MODEL_NAMES[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
        set_group "${model_name} / ${algo}"
        add_experiment "$model_name" "$algo"
    done
done

QUEUE_FILE=$(mktemp -t ragen_search_bench_queue.XXXXXX)
QUEUE_LOCK="${QUEUE_FILE}.lock"
echo 0 > "$QUEUE_FILE"
USE_FLOCK=false
QUEUE_LOCK_DIR="${QUEUE_LOCK}.d"
MAIN_PID=$$

cleanup_queue() {
    if [ "$$" -ne "$MAIN_PID" ]; then
        return
    fi
    rm -f "$QUEUE_FILE" "$QUEUE_LOCK"
    rmdir "$QUEUE_LOCK_DIR" 2>/dev/null || true
}
trap cleanup_queue EXIT

if command -v flock >/dev/null 2>&1; then
    USE_FLOCK=true
fi

next_experiment_index() {
    local idx
    if [ "$USE_FLOCK" = true ]; then
        flock -x "$QUEUE_LOCK_FD"
        idx=$(cat "$QUEUE_FILE")
        if [ -z "$idx" ]; then
            idx=0
        fi
        if (( idx >= ${#EXPERIMENTS[@]} )); then
            flock -u "$QUEUE_LOCK_FD"
            echo -1
            return
        fi
        echo $((idx + 1)) > "$QUEUE_FILE"
        flock -u "$QUEUE_LOCK_FD"
        echo "$idx"
        return
    fi

    while ! mkdir "$QUEUE_LOCK_DIR" 2>/dev/null; do
        sleep 0.05
    done
    idx=$(cat "$QUEUE_FILE")
    if [ -z "$idx" ]; then
        idx=0
    fi
    if (( idx >= ${#EXPERIMENTS[@]} )); then
        rmdir "$QUEUE_LOCK_DIR"
        echo -1
        return
    fi
    echo $((idx + 1)) > "$QUEUE_FILE"
    rmdir "$QUEUE_LOCK_DIR"
    echo "$idx"
}

run_queue_for_slot() {
    local gpu_list=$1
    if [ "$USE_FLOCK" = true ]; then
        exec {QUEUE_LOCK_FD}>"$QUEUE_LOCK"
    fi
    while true; do
        local idx
        idx=$(next_experiment_index)
        if [ "$idx" -lt 0 ]; then
            break
        fi
        local exp="${EXPERIMENTS[$idx]}"
        IFS='|' read -r exp_group model_name algo <<< "$exp"
        run_experiment "$model_name" "$algo" "$gpu_list" || true
        if [ "$COOLDOWN_SECONDS" -gt 0 ]; then
            sleep "$COOLDOWN_SECONDS"
        fi
    done
    if [ "$USE_FLOCK" = true ]; then
        exec {QUEUE_LOCK_FD}>&-
    fi
}

pids=()
for idx in "${!GPU_GROUPS[@]}"; do
    run_queue_for_slot "${GPU_GROUPS[$idx]}" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

{
    echo ""
    echo "=== Grouped Summary ==="
    echo "GPU per exp: ${GPUS_PER_EXP}x${GPU_MODEL_LABEL} | Steps: ${STEPS}"
    for group_label in "${GROUP_LABELS[@]}"; do
        echo "=== ${group_label} ==="
        for exp in "${EXPERIMENTS[@]}"; do
            IFS='|' read -r exp_group model_name algo <<< "$exp"
            if [ "$exp_group" != "$group_label" ]; then
                continue
            fi
            name="search-${algo}-${FILTER_LABEL}-${model_name}"
            if [ -f "${LOG_DIR}/${name}.result" ]; then
                cat "${LOG_DIR}/${name}.result"
            else
                echo "task=search | algo=${algo} | filter=${FILTER_LABEL} | model=${model_name} | status=missing"
            fi
        done
    done
} | tee -a "$LOG_FILE"
