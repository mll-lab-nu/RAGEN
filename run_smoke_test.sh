#!/bin/bash
# Smoke Test Script for B200
# PPO + 3B + Thinking Sokoban, 50 steps
# Group 1: KL sweep (4 filter ratios × 4 KL values = 16 runs), entropy=0
# Group 2: Entropy sweep (4 filter ratios × 4 entropy values = 16 runs), KL=0
# Total: 32 experiments

# Common settings
STEPS=50
MODEL_SIZE="3B"
MODEL_PATH="Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"
CONFIG="_2_sokoban"

# Batch settings: prompt_batch_size=8, samples_per_prompt=16
ENV_GROUPS=8
GROUP_SIZE=16

LOG_FILE="smoke_test_results_${MODEL_SIZE}.log"

# Collapse detection settings
COLLAPSE_FIRST_TURN=true
COLLAPSE_MULTI_TURN=false
COLLAPSE_NUM_SAMPLES="all"

echo "=== Smoke Test for $MODEL_SIZE on B200: $(date) ===" | tee $LOG_FILE
echo "Config: PPO + Thinking Sokoban + ${STEPS} steps" | tee -a $LOG_FILE
echo "Batch: env_groups=${ENV_GROUPS}, group_size=${GROUP_SIZE}" | tee -a $LOG_FILE
echo "Collapse: first_turn=${COLLAPSE_FIRST_TURN}, multi_turn=${COLLAPSE_MULTI_TURN}, num_samples=${COLLAPSE_NUM_SAMPLES}" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Filter ratios (TopK 100/80/60/40)
FILTER_RATIOS=(1.0 0.8 0.6 0.4)
FILTER_NAMES=("topk100" "topk80" "topk60" "topk40")

# KL coefficients
KL_COEFFS=(0.001 0.003 0.01 0.03)

# Entropy coefficients
ENTROPY_COEFFS=(0.001 0.003 0.01 0.03)

run_experiment() {
    local name=$1
    local filter_ratio=$2
    local kl_coef=$3
    local entropy_coef=$4

    # Determine if KL loss should be enabled
    local use_kl_loss="False"
    if [ "$kl_coef" != "0" ]; then
        use_kl_loss="True"
    fi

    echo "Running: ${name} | filter=${filter_ratio} | kl=${kl_coef} | entropy=${entropy_coef} | collapse=first:${COLLAPSE_FIRST_TURN},multi:${COLLAPSE_MULTI_TURN},samples:${COLLAPSE_NUM_SAMPLES} | status=running" | tee -a $LOG_FILE

    START=$(date +%s)
    python train.py --config-name $CONFIG \
        model_path="${MODEL_PATH}" \
        trainer.total_training_steps=${STEPS} \
        trainer.experiment_name=${name} \
        trainer.logger="['console','wandb']" \
        trainer.val_before_train=False \
        trainer.save_freq=-1 \
        agent_proxy.enable_think=True \
        algorithm.adv_estimator=gae \
        collapse_detection.first_turn_enabled=${COLLAPSE_FIRST_TURN} \
        collapse_detection.multi_turn_enabled=${COLLAPSE_MULTI_TURN} \
        collapse_detection.num_samples=${COLLAPSE_NUM_SAMPLES} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=${entropy_coef} \
        actor_rollout_ref.rollout.rollout_filter_ratio=${filter_ratio} \
        es_manager.train.env_groups=${ENV_GROUPS} \
        es_manager.train.group_size=${GROUP_SIZE} \
        2>&1 | tee "logs/${name}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    END=$(date +%s)

    TOTAL_TIME=$((END - START))

    # Extract timing from log
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

    echo "${name} | filter=${filter_ratio} | kl=${kl_coef} | entropy=${entropy_coef} | collapse=first:${COLLAPSE_FIRST_TURN},multi:${COLLAPSE_MULTI_TURN},samples:${COLLAPSE_NUM_SAMPLES} | train_time=${TRAIN_TIME}s | eval_time=${EVAL_TIME}s | collapse_time=${COLLAPSE_TIME}s | total_time=${TOTAL_TIME_METRIC}s | wall_time=${TOTAL_TIME}s | gpu=1xB200 | status=${STATUS}" | tee -a $LOG_FILE
    [ "$STATUS" = "fail" ] && echo "  error: ${ERROR}" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
}

mkdir -p logs

# ============================================================
# Group 1: KL sweep (entropy=0)
# ============================================================
echo "=== Group 1: KL sweep (entropy=0) ===" | tee -a $LOG_FILE
echo "Filter ratios: ${FILTER_RATIOS[*]}" | tee -a $LOG_FILE
echo "KL coeffs: ${KL_COEFFS[*]}" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

for i in "${!FILTER_RATIOS[@]}"; do
    filter_ratio=${FILTER_RATIOS[$i]}
    filter_name=${FILTER_NAMES[$i]}
    for kl_coef in "${KL_COEFFS[@]}"; do
        kl_name=$(echo $kl_coef | sed 's/\.//g')
        exp_name="smoke-ppo-sokoban-${filter_name}-kl${kl_name}-ent0"
        run_experiment "$exp_name" "$filter_ratio" "$kl_coef" "0"
    done
done

# ============================================================
# Group 2: Entropy sweep (KL=0)
# ============================================================
echo "=== Group 2: Entropy sweep (KL=0) ===" | tee -a $LOG_FILE
echo "Filter ratios: ${FILTER_RATIOS[*]}" | tee -a $LOG_FILE
echo "Entropy coeffs: ${ENTROPY_COEFFS[*]}" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

for i in "${!FILTER_RATIOS[@]}"; do
    filter_ratio=${FILTER_RATIOS[$i]}
    filter_name=${FILTER_NAMES[$i]}
    for entropy_coef in "${ENTROPY_COEFFS[@]}"; do
        ent_name=$(echo $entropy_coef | sed 's/\.//g')
        exp_name="smoke-ppo-sokoban-${filter_name}-kl0-ent${ent_name}"
        run_experiment "$exp_name" "$filter_ratio" "0" "$entropy_coef"
    done
done

echo "=== Smoke Test Completed: $(date) ===" | tee -a $LOG_FILE

# Summary
echo "" | tee -a $LOG_FILE
echo "=== Summary ===" | tee -a $LOG_FILE
TOTAL_RUNS=$(grep -c "status=success\|status=fail" $LOG_FILE || echo 0)
SUCCESS_RUNS=$(grep -c "status=success" $LOG_FILE || echo 0)
FAIL_RUNS=$(grep -c "status=fail" $LOG_FILE || echo 0)
echo "Total: $TOTAL_RUNS | Success: $SUCCESS_RUNS | Failed: $FAIL_RUNS" | tee -a $LOG_FILE
