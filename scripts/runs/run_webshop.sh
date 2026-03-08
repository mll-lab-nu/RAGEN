#!/bin/bash
# Main Table: Different algorithms (PPO/DAPO/GRPO/DrGRPO) × tasks × filter/no-filter
# Model: Qwen2.5-3B only
# Filtering rule: filter => top_p=0.9, nofilter => top_p=1.0

set -euo pipefail

# Defaults
STEPS=400
MODEL_NAME="Qwen2.5-3B"
TASKS=("webshop")
ALGORITHMS=("GRPO")
MODEL_PATH=""
SAVE_FREQ=-1
FILTER_MODES=("filter" "nofilter")
FILTERS_OPTION="all"
SELECTED_FILTERS=("${FILTER_MODES[@]}")

# GPU settings
GPUS=()
GPUS_PROVIDED=false
GPUS_PER_EXP=4
COOLDOWN_SECONDS=30
GPU_MEMORY_UTILIZATION=0.3
declare -A GPU_LABELS

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --steps N             Training steps (default: 400)"
    echo "  --tasks LIST          Comma-separated tasks (default: sokoban,frozenlake,webshop,metamathqa,countdown)"
    echo "  --algos LIST          Comma-separated algorithms (default: PPO,DAPO,GRPO,DrGRPO)"
    echo "  --gpus LIST           Comma-separated GPU IDs (default: auto-detect)"
    echo "  --gpus-per-exp N      GPUs per experiment (default: 1)"
    echo "  --cooldown SECONDS    Cooldown between runs on the same GPU group (default: 30)"
    echo "  --gpu-memory-utilization V  Rollout gpu_memory_utilization (default: 0.3)"
    echo "  --save-freq N         Checkpoint save frequency (default: -1 to disable saving)"
    echo "  --filters LIST        Comma-separated filter modes (filter,nofilter,all). Default: all"
    echo "  -h, --help            Show this help"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        --tasks) IFS=',' read -r -a TASKS <<< "$2"; shift 2 ;;
        --tasks=*) IFS=',' read -r -a TASKS <<< "${1#*=}"; shift ;;
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
        --filters) FILTERS_OPTION="$2"; shift 2 ;;
        --filters=*) FILTERS_OPTION="${1#*=}"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

MODEL_PATH="Qwen/${MODEL_NAME}"

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

# Pre-fetch GPU names to avoid slow repeated nvidia-smi calls
declare -A GPU_NAMES_CACHE
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Detecting GPUs..." >&2
    mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
    for i in "${!GPU_NAMES[@]}"; do
        GPU_NAMES_CACHE[$i]="${GPU_NAMES[$i]}"
    done
fi

get_gpu_label() {
    local gpu_id="$1"
    if [ -n "${GPU_LABELS[$gpu_id]+x}" ]; then
        echo "${GPU_LABELS[$gpu_id]}"
        return
    fi
    local name="${GPU_NAMES_CACHE[$gpu_id]:-}"
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
LOG_FILE="logs/webshop_lora_${MODEL_NAME}.log"
RESULT_ROOT="logs"
CHECKPOINT_ROOT="model_saving/webshop_lora_${MODEL_NAME}"

mkdir -p logs
mkdir -p "$RESULT_ROOT"
mkdir -p "$CHECKPOINT_ROOT"

echo "Initializing experiments..."
echo "=== Perf Table Runner for ${MODEL_NAME}: $(date) ===" | tee "$LOG_FILE"
echo "Tasks: ${TASKS[*]} | Steps: ${STEPS} | GPU per exp: ${GPUS_PER_EXP}x${GPU_MODEL_LABEL}" | tee -a "$LOG_FILE"
echo "GPUS: ${GPUS[*]} | groups: ${GPU_GROUPS[*]} | cooldown=${COOLDOWN_SECONDS}s" | tee -a "$LOG_FILE"

get_config_for_task() {
    case "$1" in
        countdown) echo "_4_countdown" ;;
        sokoban) echo "_2_sokoban" ;;
        frozenlake) echo "_3_frozen_lake" ;;
        webshop) echo "_6_webshop" ;;
        webshop_full) echo "webshop_full" ;;
        metamathqa) echo "_5_metamathqa" ;;
        *) echo "" ;;
    esac
}

get_algo_overrides() {
    case "$1" in
        PPO)
            echo "algorithm.adv_estimator=gae actor_rollout_ref.actor.loss_agg_mode=token-mean"
            ;;
        DAPO)
            # DAPO here = PPO + higher clip + no KL + token-level loss.
            echo "algorithm.adv_estimator=gae actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0"
            ;;
        GRPO)
            echo "algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=True actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
            ;;
        DrGRPO)
            echo "algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=False actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum"
            ;;
        *)
            echo ""
            ;;
    esac
}

run_experiment() {
    local task=$1
    local algo=$2
    local filter=$3
    local config=$4
    local gpu_list=$5

    local filter_value
    if [ "$filter" = "filter" ]; then
        filter_value=0.9
    else
        filter_value=1.0
    fi
    local filter_strategy="top_p"

    local common_overrides=(
        "actor_rollout_ref.actor.use_kl_loss=False"
        "actor_rollout_ref.actor.kl_loss_type=low-var-kl"
        "actor_rollout_ref.actor.kl_loss_coef=0.001"
        "actor_rollout_ref.actor.entropy_coeff=0.001"
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True"
        "actor_rollout_ref.actor.entropy_checkpointing=True"
        "actor_rollout_ref.actor.use_remove_padding=True"
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2"
        "actor_rollout_ref.actor.filter_loss_scaling=none"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
        # "lora.rank=16"
        # "lora.alpha=32"
        # "lora.target_modules=all-linear"
        "agent_proxy.context_window_mode=limited_multi_turn"
        "agent_proxy.max_context_window=3"
    )

    local env_overrides=()
    if [ "$task" = "frozenlake" ]; then
        env_overrides+=("custom_envs.CoordFrozenLake.env_config.success_rate=1.0")
    fi

    local checkpoint_overrides=(
        "actor_rollout_ref.actor.checkpoint.save_contents=[model]"
        "critic.checkpoint.save_contents=[model]"
    )

    local algo_overrides
    algo_overrides=$(get_algo_overrides "$algo")
    read -r -a algo_args <<< "$algo_overrides"

    local name="${task}-${algo}-${filter}-${MODEL_NAME}-lookback3"
    local task_dir="${RESULT_ROOT}/diff_algo_${task}_${MODEL_NAME}"
    local log_path="${task_dir}/${name}.log"
    local checkpoint_dir="${CHECKPOINT_ROOT}/${task}/${algo}/${filter}/${name}"
    local gpus_per_exp
    IFS=',' read -r -a gpu_ids <<< "$gpu_list"
    gpus_per_exp=${#gpu_ids[@]}

    mkdir -p "$task_dir"
    mkdir -p "${checkpoint_dir}"
    echo "[Command] CUDA_VISIBLE_DEVICES=\"${gpu_list}\" python train.py --config-name \"$config\" \\
        model_path=\"${MODEL_PATH}\" \\
        trainer.project_name=\"main_webshop\" \\
        trainer.total_training_steps=\"${STEPS}\" \\
        trainer.experiment_name=\"${name}\" \\
        trainer.save_freq=\"${SAVE_FREQ}\" \\
        trainer.default_local_dir=\"${checkpoint_dir}\" \\
        trainer.logger=\"['console','wandb']\" \\
        trainer.val_before_train=True \\
        trainer.n_gpus_per_node=\"${gpus_per_exp}\" \\
        system.CUDA_VISIBLE_DEVICES=\"'${gpu_list}'\" \\
        actor_rollout_ref.rollout.rollout_filter_strategy=\"${filter_strategy}\" \\
        actor_rollout_ref.rollout.rollout_filter_value=\"${filter_value}\" \\
        ${common_overrides[*]} \\
        ${env_overrides[*]} \\
        ${checkpoint_overrides[*]} \\
        ${algo_args[*]}"

    START=$(date +%s)
    CUDA_VISIBLE_DEVICES="${gpu_list}" python train.py --config-name "$config" \
        model_path="${MODEL_PATH}" \
        trainer.project_name="main_webshop" \
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
        "${env_overrides[@]}" \
        "${checkpoint_overrides[@]}" \
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
    local summary_line="task=${task} | algo=${algo} | filter=${filter} | model=${MODEL_NAME} | steps=${STEPS} | filter=${filter_strategy}:${filter_value} | train_time=${TRAIN_TIME}s | eval_time=${EVAL_TIME}s | total_time=${TOTAL_TIME_METRIC}s | wall_time=${TOTAL_TIME}s | gpu=${gpu_label} | status=${status}"
    echo "${summary_line}" > "${task_dir}/${name}.result"
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
    local task=$1
    local algo=$2
    local filter=$3
    local config=$4
    EXPERIMENTS+=("${CURRENT_GROUP}|${task}|${algo}|${filter}|${config}")
}

resolve_filter_selection() {
    local raw="$1"
    if [ -z "$raw" ] || [ "$raw" = "all" ]; then
        SELECTED_FILTERS=("${FILTER_MODES[@]}")
        return
    fi
    IFS=',' read -r -a candidates <<< "$raw"
    SELECTED_FILTERS=()
    for candidate in "${candidates[@]}"; do
        candidate="${candidate// /}"
        case "$candidate" in
            filter|nofilter)
                SELECTED_FILTERS+=("$candidate")
                ;;
            "")
                continue
                ;;
            *)
                echo "Unknown filter mode: $candidate" >&2
                exit 1
                ;;
        esac
    done
    if [ ${#SELECTED_FILTERS[@]} -eq 0 ]; then
        echo "No valid filters selected via --filters" >&2
        exit 1
    fi
}

resolve_filter_selection "$FILTERS_OPTION"

for algo in "${ALGORITHMS[@]}"; do
    set_group "Algorithm: ${algo}"
    for task in "${TASKS[@]}"; do
        config=$(get_config_for_task "$task")
        if [ -z "$config" ]; then
            echo "Unknown task: $task" >&2
            exit 1
        fi
        for filter in "${SELECTED_FILTERS[@]}"; do
            add_experiment "$task" "$algo" "$filter" "$config"
        done
    done
done

QUEUE_FILE=$(mktemp -t ragen_main_table_queue.XXXXXX)
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
        IFS='|' read -r exp_group task algo filter config <<< "$exp"
        run_experiment "$task" "$algo" "$filter" "$config" "$gpu_list" || true
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
echo "GPU per exp: ${GPUS_PER_EXP}x${GPU_MODEL_LABEL} | Model: ${MODEL_NAME} | Steps: ${STEPS}"
    for group_label in "${GROUP_LABELS[@]}"; do
        echo "=== ${group_label} ==="
        for exp in "${EXPERIMENTS[@]}"; do
            IFS='|' read -r exp_group task algo filter config <<< "$exp"
            if [ "$exp_group" != "$group_label" ]; then
                continue
            fi
            name="${task}-${algo}-${filter}-${MODEL_NAME}"
            task_dir="${RESULT_ROOT}/diff_algo_${task}_${MODEL_NAME}"
            if [ -f "${task_dir}/${name}.result" ]; then
                cat "${task_dir}/${name}.result"
            else
                echo "task=${task} | algo=${algo} | filter=${filter} | model=${MODEL_NAME} | status=missing"
            fi
        done
    done
} | tee -a "$LOG_FILE"
