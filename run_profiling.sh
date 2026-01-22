#!/bin/bash
# RAGEN Profiling Script - 10 steps per run
# 4 algo configs (GRPO/PPO × with/without filtering) × 3 tasks × 2 thinking modes = 24 runs

# Common settings
STEPS=10
MODEL_SIZE="${1:-3B}"
MODEL_PATH="Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"

LOG_FILE="profiling_results_${MODEL_SIZE}.log"

echo "=== RAGEN Profiling for $MODEL_SIZE: $(date) ===" | tee $LOG_FILE


# Algorithm parameters
GRPO_FILTER="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
GRPO_NO_FILTER="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.rollout_filter_ratio=1"
PPO_FILTER="algorithm.adv_estimator=gae actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
PPO_NO_FILTER="algorithm.adv_estimator=gae actor_rollout_ref.rollout.rollout_filter_ratio=1"

run_experiment() {
    local task=$1
    local algo=$2
    local filter=$3
    local think=$4
    local config=$5
    local think_label="yes"
    local think_name="thinking"
    if [ "$think" = "False" ]; then
        think_label="no"
        think_name="nothink"
    fi
    local name="${task}-${algo}-${filter}-${MODEL_SIZE}-${think_name}"

    echo "task=${task} | algo=${algo} | filter=${filter} | model=${MODEL_SIZE} | thinking=${think_label} | steps=${STEPS} | status=running" | tee -a $LOG_FILE

    START=$(date +%s)
    python train.py --config-name $config \
        model_path="${MODEL_PATH}" \
        trainer.total_training_steps=${STEPS} \
        trainer.experiment_name=${name} \
        trainer.logger="['console']" \
        trainer.val_before_train=False \
        trainer.save_freq=-1 \
        agent_proxy.enable_think=${think} \
        ${@:6} 2>&1 | tee "logs/${name}.log"
    EXIT_CODE=${PIPESTATUS[0]}  # 获取 python 的退出码，而不是 tee 的
    END=$(date +%s)

    TOTAL_TIME=$((END - START))

    # Extract timing from log (if available) and format to 2 decimal places
    TRAIN_TIME_RAW=$(grep -oP 'timing_s/train_total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
    EVAL_TIME_RAW=$(grep -oP 'timing_s/eval_total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
    TOTAL_TIME_RAW=$(grep -oP 'timing_s/total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
    COLLAPSE_TIME_RAW=$(grep -oP 'timing_s/collapse_total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
    TRAIN_TIME=$([ -n "$TRAIN_TIME_RAW" ] && printf "%.2f" "$TRAIN_TIME_RAW" || echo "N/A")
    EVAL_TIME=$([ -n "$EVAL_TIME_RAW" ] && printf "%.2f" "$EVAL_TIME_RAW" || echo "N/A")
    TOTAL_TIME_METRIC=$([ -n "$TOTAL_TIME_RAW" ] && printf "%.2f" "$TOTAL_TIME_RAW" || echo "N/A")
    COLLAPSE_TIME=$([ -n "$COLLAPSE_TIME_RAW" ] && printf "%.2f" "$COLLAPSE_TIME_RAW" || echo "N/A")

    if [ $EXIT_CODE -eq 0 ]; then
        STATUS="success"
    else
        STATUS="fail"
        ERROR=$(tail -2 "logs/${name}.log" | tr '\n' ' ')
    fi

    # Update log with results
    echo "task=${task} | algo=${algo} | filter=${filter} | model=${MODEL_SIZE} | thinking=${think_label} | steps=${STEPS} | train_time=${TRAIN_TIME}s | eval_time=${EVAL_TIME}s | collapse_time=${COLLAPSE_TIME}s | total_time=${TOTAL_TIME_METRIC}s | wall_time=${TOTAL_TIME}s | gpu=1xB200 | status=${STATUS}" | tee -a $LOG_FILE
    [ "$STATUS" = "fail" ] && echo "  error: ${ERROR}" | tee -a $LOG_FILE
}

mkdir -p logs

# Group 1: GRPO with filtering
echo "=== Group 1: GRPO with filtering ===" | tee -a $LOG_FILE
run_experiment bandit GRPO filter0.5 True _1_bandit $GRPO_FILTER
run_experiment sokoban GRPO filter0.5 True _2_sokoban $GRPO_FILTER
run_experiment frozenlake GRPO filter0.5 True _3_frozen_lake $GRPO_FILTER

# Group 2: GRPO without filtering
echo "=== Group 2: GRPO without filtering ===" | tee -a $LOG_FILE
run_experiment bandit GRPO no_filter True _1_bandit $GRPO_NO_FILTER
run_experiment sokoban GRPO no_filter True _2_sokoban $GRPO_NO_FILTER
run_experiment frozenlake GRPO no_filter True _3_frozen_lake $GRPO_NO_FILTER

# Group 3: PPO with filtering
echo "=== Group 3: PPO with filtering ===" | tee -a $LOG_FILE
run_experiment bandit PPO filter0.5 True _1_bandit $PPO_FILTER
run_experiment sokoban PPO filter0.5 True _2_sokoban $PPO_FILTER
run_experiment frozenlake PPO filter0.5 True _3_frozen_lake $PPO_FILTER

# Group 4: PPO without filtering
echo "=== Group 4: PPO without filtering ===" | tee -a $LOG_FILE
run_experiment bandit PPO no_filter True _1_bandit $PPO_NO_FILTER
run_experiment sokoban PPO no_filter True _2_sokoban $PPO_NO_FILTER
run_experiment frozenlake PPO no_filter True _3_frozen_lake $PPO_NO_FILTER

# Group 5: GRPO with filtering (no-think)
echo "=== Group 5: GRPO with filtering (no-think) ===" | tee -a $LOG_FILE
run_experiment bandit GRPO filter0.5 False _1_bandit $GRPO_FILTER
run_experiment sokoban GRPO filter0.5 False _2_sokoban $GRPO_FILTER
run_experiment frozenlake GRPO filter0.5 False _3_frozen_lake $GRPO_FILTER

# Group 6: GRPO without filtering (no-think)
echo "=== Group 6: GRPO without filtering (no-think) ===" | tee -a $LOG_FILE
run_experiment bandit GRPO no_filter False _1_bandit $GRPO_NO_FILTER
run_experiment sokoban GRPO no_filter False _2_sokoban $GRPO_NO_FILTER
run_experiment frozenlake GRPO no_filter False _3_frozen_lake $GRPO_NO_FILTER

# Group 7: PPO with filtering (no-think)
echo "=== Group 7: PPO with filtering (no-think) ===" | tee -a $LOG_FILE
run_experiment bandit PPO filter0.5 False _1_bandit $PPO_FILTER
run_experiment sokoban PPO filter0.5 False _2_sokoban $PPO_FILTER
run_experiment frozenlake PPO filter0.5 False _3_frozen_lake $PPO_FILTER

# Group 8: PPO without filtering (no-think)
echo "=== Group 8: PPO without filtering (no-think) ===" | tee -a $LOG_FILE
run_experiment bandit PPO no_filter False _1_bandit $PPO_NO_FILTER
run_experiment sokoban PPO no_filter False _2_sokoban $PPO_NO_FILTER
run_experiment frozenlake PPO no_filter False _3_frozen_lake $PPO_NO_FILTER

echo "=== Profiling Completed: $(date) ===" | tee -a $LOG_FILE
