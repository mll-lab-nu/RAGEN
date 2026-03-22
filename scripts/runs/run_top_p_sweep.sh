#!/bin/bash
# Top-p sweep for sokoban using GAE with the requested filter overrides.
set -euo pipefail

# Defaults
STEPS=400
MODEL_NAME="Qwen2.5-3B"
MODEL_PATH="Qwen/${MODEL_NAME}"
PROJECT_NAME="ragen_release_top_p_sweep"
CONFIG_NAME="_2_sokoban"
SAVE_FREQ=-1
ROLL_FILTER_VALUES="1.0,0.98,0.95,0.9,0.8,0.6,0.4,nofilter"
GPUS=()
GPUS_PROVIDED=false
GPUS_PER_EXP=1
COOLDOWN_SECONDS=0
GPU_MEMORY_UTILIZATION=0.5
RAY_NUM_CPUS=16

usage() {
    cat <<'EOF'
Usage: $0 [options]
Options:
  --steps N                     Training steps (default: 400)
  --rollout_filter_value LIST   Comma-separated top_p values (default: 1.0,0.98,0.95,0.9,0.8,0.6,0.4,nofilter)
  --gpus LIST                   Comma-separated GPU IDs (auto-detected if omitted)
  --gpus-per-exp N              GPUs per experiment (default: 1)
  --ray-num-cpus N              Max CPUs per task for ray.init (default: 16)
  --gpu-memory-utilization V    GPU memory utilization for rollouts (default: 0.5)
  --save-freq N                 Checkpoint save frequency (default: -1)
  -h, --help                    Show this help message
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        --rollout_filter_value) ROLL_FILTER_VALUES="$2"; shift 2 ;;
        --rollout_filter_value=*) ROLL_FILTER_VALUES="${1#*=}"; shift ;;
        --gpus) IFS=',' read -r -a GPUS <<< "$2"; GPUS_PROVIDED=true; shift 2 ;;
        --gpus=*) IFS=',' read -r -a GPUS <<< "${1#*=}"; GPUS_PROVIDED=true; shift ;;
        --gpus-per-exp) GPUS_PER_EXP="$2"; shift 2 ;;
        --gpus-per-exp=*) GPUS_PER_EXP="${1#*=}"; shift ;;
        --ray-num-cpus) RAY_NUM_CPUS="$2"; shift 2 ;;
        --ray-num-cpus=*) RAY_NUM_CPUS="${1#*=}"; shift ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --gpu-memory-utilization=*) GPU_MEMORY_UTILIZATION="${1#*=}"; shift ;;
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
if ! [[ "$RAY_NUM_CPUS" =~ ^[0-9]+$ ]] || [ "$RAY_NUM_CPUS" -lt 1 ]; then
    echo "Error: --ray-num-cpus must be a positive integer"
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

declare -A GPU_LABELS
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
LOG_FILE="logs/top_p_sweep_${MODEL_NAME}.log"
RESULT_ROOT="logs/top_p_sweep_${MODEL_NAME}"
CHECKPOINT_ROOT="model_saving/top_p_sweep_${MODEL_NAME}"

mkdir -p logs
mkdir -p "$RESULT_ROOT"
mkdir -p "$CHECKPOINT_ROOT"

echo "=== Top-p Sweep Runner (${MODEL_NAME}): $(date) ===" | tee "$LOG_FILE"
echo "Values: ${ROLL_FILTER_VALUES} | Steps: ${STEPS} | GPUs per exp: ${GPU_LOG_LABEL}" | tee -a "$LOG_FILE"
echo "Groups: ${GPU_GROUPS[*]} | GPU memory util: ${GPU_MEMORY_UTILIZATION} | ray_num_cpus: ${RAY_NUM_CPUS} | save_freq: ${SAVE_FREQ}" | tee -a "$LOG_FILE"

parse_rollout_values() {
    IFS=',' read -r -a raw <<< "$ROLL_FILTER_VALUES"
    ROLL_VALUES=()
    for token in "${raw[@]}"; do
        token="${token// /}"
        if [ -n "$token" ]; then
            ROLL_VALUES+=("$token")
        fi
    done
    if [ ${#ROLL_VALUES[@]} -eq 0 ]; then
        echo "Error: no rollout_filter_value entries provided" >&2
        exit 1
    fi
}

parse_rollout_values
EXPERIMENTS=("${ROLL_VALUES[@]}")

run_experiment() {
    local value="$1"
    local gpu_list="$2"

    local include_zero="False"
    local filter_value="$value"
    if [ "$value" = "nofilter" ]; then
        include_zero="True"
        filter_value="1.0"
    fi
    local safe_label="${value//./}"
    safe_label="${safe_label,,}"

    local name="sokoban_top_p_${safe_label}-${MODEL_NAME}"
    local task_dir="${RESULT_ROOT}/${safe_label}"
    local log_path="${task_dir}/${name}.log"
    local checkpoint_dir="${CHECKPOINT_ROOT}/${safe_label}/${name}"
    local gpus_per_exp
    IFS=',' read -r -a gpu_ids <<< "$gpu_list"
    gpus_per_exp=${#gpu_ids[@]}

    mkdir -p "$task_dir"
    mkdir -p "$checkpoint_dir"

    START=$(date +%s)
    CUDA_VISIBLE_DEVICES="${gpu_list}" python train.py --config-name "$CONFIG_NAME" \
        model_path="${MODEL_PATH}" \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${name}" \
        trainer.total_training_steps="${STEPS}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.default_local_dir="${checkpoint_dir}" \
        trainer.logger="['console','wandb']" \
        trainer.val_before_train=True \
        trainer.n_gpus_per_node="${gpus_per_exp}" \
        ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
        system.CUDA_VISIBLE_DEVICES="'${gpu_list}'" \
        algorithm.adv_estimator=gae \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
        actor_rollout_ref.actor.filter_loss_scaling=none \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        critic.ppo_mini_batch_size=32 \
        critic.ppo_micro_batch_size_per_gpu=4 \
        ppo_mini_batch_size=32 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.rollout_filter_strategy=top_p \
        actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=softmax \
        actor_rollout_ref.rollout.rollout_filter_value=${filter_value} \
        actor_rollout_ref.rollout.rollout_filter_include_zero=${include_zero} \
        actor_rollout_ref.rollout.rollout_filter_type=largest \
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
        actor_rollout_ref.rollout.rollout_filter_metric=reward_variance \
        es_manager.train.env_groups=8 \
        es_manager.train.group_size=16 \
        es_manager.train.env_configs.n_groups='[8]' \
        es_manager.val.env_groups=512 \
        es_manager.val.group_size=1 \
        es_manager.val.env_configs.n_groups='[512]' \
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
    local summary_line="value=${value} | include_zero=${include_zero} | filter_value=${filter_value} | train_time=${TRAIN_TIME}s | eval_time=${EVAL_TIME}s | total_time=${TOTAL_TIME_METRIC}s | wall_time=${TOTAL_TIME}s | gpu=${gpu_label} | status=${status}"
    echo "${summary_line}" > "${task_dir}/${name}.result"
    echo "${summary_line}" | tee -a "$LOG_FILE"
    if [ "$status" = "fail" ]; then
        echo "  error: ${error_line}" | tee -a "$LOG_FILE"
    fi
}

EXPERIMENT_COUNT=${#EXPERIMENTS[@]}
if [ $EXPERIMENT_COUNT -eq 0 ]; then
    echo "No experiments to run" >&2
    exit 1
fi

QUEUE_FILE=$(mktemp -t ragen_top_p_queue.XXXXXX)
echo 0 > "$QUEUE_FILE"
QUEUE_LOCK="${QUEUE_FILE}.lock"
QUEUE_LOCK_DIR="${QUEUE_LOCK}.d"
USE_FLOCK=false
MAIN_PID=$$

cleanup_queue() {
    if [ "$MAIN_PID" != "$$" ]; then
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
    local gpu_list="$1"
    if [ "$USE_FLOCK" = true ]; then
        exec {QUEUE_LOCK_FD}>"$QUEUE_LOCK"
    fi
    while true; do
        local idx
        idx=$(next_experiment_index)
        if [ "$idx" -lt 0 ]; then
            break
        fi
        local value="${EXPERIMENTS[$idx]}"
        run_experiment "$value" "$gpu_list" || true
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
    echo "=== Top-p Sweep Summary ==="
    echo "Project: ${PROJECT_NAME} | Steps: ${STEPS} | GPU per exp: ${GPU_LOG_LABEL}"
    for val in "${EXPERIMENTS[@]}"; do
        safe_label="${val//./p}"
        safe_label="${safe_label,,}"
        name="sokoban_top_p_${safe_label}-${MODEL_NAME}"
        task_dir="${RESULT_ROOT}/${safe_label}"
        if [ -f "${task_dir}/${name}.result" ]; then
            cat "${task_dir}/${name}.result"
        else
            echo "value=${val} | status=missing"
        fi
    done
} | tee -a "$LOG_FILE"
