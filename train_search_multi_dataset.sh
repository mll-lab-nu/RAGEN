#!/usr/bin/env bash

set -euo pipefail

# Configuration
PROJ="search_multi_dataset"
# Use environment variables with sensible defaults
BASE_DIR="${RAGEN_CHECKPOINTS_DIR:-${HOME}/RAGEN/checkpoints}"
DEVICES="0,1,2,3,4,5,6,7"

# Set required environment variables with configurable paths
CACHE_DIR="${RAGEN_CACHE_DIR:-${HOME}/.cache/huggingface}"
TMP_DIR="${RAGEN_TMP_DIR:-${TMPDIR:-/tmp}}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_DIR}}"
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export TMPDIR="${TMP_DIR}"

# Training configurations
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=False"
USE_PPO="algorithm.adv_estimator=gae"  # by default
USE_BASE="algorithm.kl_ctrl.kl_coef=0.0 actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# Dataset names - these are used directly (no tag mapping needed)
VALID_DATASETS=("nq" "triviaqa" "popqa" "web_questions" "hotpotqa" "2wikimultihopqa" "musique" "eli5")

# Test/evaluation datasets - all datasets for evaluation
TEST_DATASETS=("nq" "triviaqa" "popqa" "hotpotqa" "2wikimultihopqa" "musique")

# Common parameters for 2 GPU setup
# Using tensor_parallel_size=1 to avoid multi-GPU communication issues during vLLM init
NUM_GPUS=8
TENSOR_PARALLEL=1  # Changed from 2 to 1 to avoid NCCL communication issues
ENV_GROUPS=4  # Increased to meet ppo_mini_batch_size requirement (4 * 8 * 1 = 32 >= 32)
GROUP_SIZE=8

# Function to capitalize dataset name for display
# Note: For Hydra compatibility, tags starting with numbers are converted (e.g., "2wikimultihopqa" -> "TwoWikiMultihopQA")
capitalize_dataset_name() {
  local name=$1
  case "$name" in
    "nq") echo "NQ" ;;
    "2wikimultihopqa") echo "TwoWikiMultihopQA" ;;
    "web_questions") echo "WebQuestions" ;;
    "musique") echo "MuSiQue" ;;
    "eli5") echo "ELI5" ;;
    *) echo "${name^}" ;;
  esac
}

# Function to convert dataset name list to tags and n_groups
# Usage: convert_datasets_to_tags [groups_per_dataset] dataset1 dataset2 ...
# If groups_per_dataset is not provided, calculates as ENV_GROUPS / num_datasets
convert_datasets_to_tags() {
  local groups_per_dataset
  local datasets
  
  # Check if first argument is a number (groups_per_dataset)
  if [[ "$1" =~ ^[0-9]+$ ]]; then
    groups_per_dataset=$1
    shift
    datasets=("$@")
  else
    # Use default calculation based on ENV_GROUPS
    datasets=("$@")
    groups_per_dataset=$((ENV_GROUPS / ${#datasets[@]}))
  fi
  
  local tags=()
  local n_groups=()
  
  for dataset in "${datasets[@]}"; do
    local valid=false
    for valid_ds in "${VALID_DATASETS[@]}"; do
      if [[ "$dataset" == "$valid_ds" ]]; then
        valid=true
        break
      fi
    done
    if [[ "$valid" == false ]]; then
      echo "Error: Unknown dataset name: $dataset. Available: ${VALID_DATASETS[*]}" >&2
      exit 1
    fi
    local tag=$(capitalize_dataset_name "$dataset")
    tags+=("$tag")
    n_groups+=("$groups_per_dataset")
  done
  
  local tags_str=$(IFS=,; echo "${tags[*]}")
  local n_groups_str=$(IFS=,; echo "${n_groups[*]}")
  echo "${tags_str}|${n_groups_str}"
}

# Helper function to format tag path for Hydra
format_tag_path() {
  local tag=$1
  local suffix=$2
  echo "custom_envs.${tag}.${suffix}"
}

# Function to set up base Search config (required for dynamic config creation)
# This must be called before setting individual dataset configs
setup_base_search_config() {
  local turns=$1
  local cmd=""
  
  # Set up base Search config that will be used as template for all search datasets
  # Note: Most config is set in envs.yaml to avoid Hydra parsing issues with quotes
  # We override the values that need to change based on turns (no + prefix since they exist in envs.yaml)
  cmd+="custom_envs.Search.max_actions_per_traj=${turns} "
  cmd+="custom_envs.Search.env_config.max_steps=${turns} "
  
  echo "$cmd"
}

# Function to set max_turn and server config for a list of datasets
# Note: We skip individual dataset configs and rely on dynamic config creation
# The base Search config in envs.yaml + dynamic config creation handles everything
# This avoids Hydra parsing issues with dataset names starting with numbers
set_datasets_config() {
  # With dynamic config creation, we don't need to set individual dataset configs
  # The base Search config + dynamic config creation will handle it
  # We can optionally set dataset-specific overrides here if needed, but it's not required
  echo ""
}

# Function to run a single experiment
run_experiment() {
  local train_dataset=$1  # Single dataset name like "nq" or "hotpotqa"
  local turns=$2
  local model_path=$3
  local algorithm=$4  # "ppo" or "grpo"
  
  # Determine config name - use base config since search configs may not exist
  # We'll override everything via command line
  local config_name="base"
  
  # Determine algorithm flags
  local algo_flags
  if [[ "$algorithm" == "grpo" ]]; then
    algo_flags="$USE_GRPO $USE_BASE"
  else
    algo_flags="$USE_PPO $USE_BASE"
  fi
  
  # Set max_len based on turns
  # With tensor_parallel_size=1 and 8 GPUs, we have more memory available
  # max_model_len must be > prompt length (vLLM requires strict inequality)
  # We add buffer to account for prompt + response tokens
  local max_len
  local max_tok
  if [[ "$turns" -ge 7 ]]; then
    max_len=2500   # Increased to leave room for prompts (2000) + response tokens
    max_tok=2560   # Increased accordingly
  else
    max_len=2000   # Increased to leave room for prompts + response tokens
    max_tok=2048   # Increased accordingly
  fi
  
  # Create experiment name
  local exp_name="${train_dataset}-turn${turns}-qwen3b-base-${algorithm}"
  local out_dir="${BASE_DIR}/${exp_name}"
  local log_file="${exp_name}.log"
  
  # Use dataset names directly in tags (dynamic config creation will handle it)
  # Alternatively, you can use capitalized tags - both work
  local train_tag="$train_dataset"  # Use dataset name directly
  
  # Calculate total env_groups for validation (512 per dataset)
  local val_instances_per_dataset=512
  local total_val_groups=$((${#TEST_DATASETS[@]} * val_instances_per_dataset))
  
  # Convert test datasets to tags with 512 groups per dataset
  # Use dataset names directly instead of capitalized tags
  local test_tags=()
  local test_n_groups=()
  for test_ds in "${TEST_DATASETS[@]}"; do
    test_tags+=("$test_ds")  # Use dataset name directly
    test_n_groups+=("${val_instances_per_dataset}")
  done
  local test_tags_str=$(IFS=,; echo "${test_tags[*]}")
  local test_n_groups_str=$(IFS=,; echo "${test_n_groups[*]}")
  
  # Build command for all datasets (train + test), deduplicated
  local all_datasets=("$train_dataset")
  for test_ds in "${TEST_DATASETS[@]}"; do
    # Only add if not already in the list (avoid duplicates)
    local found=false
    for existing_ds in "${all_datasets[@]}"; do
      if [[ "$test_ds" == "$existing_ds" ]]; then
        found=true
        break
      fi
    done
    if [[ "$found" == false ]]; then
      all_datasets+=("$test_ds")
    fi
  done
  
  # Set up base Search config first (required for dynamic config creation)
  local base_search_config=$(setup_base_search_config $turns)
  local all_datasets_config=$(set_datasets_config $turns "${all_datasets[@]}")
  
  CUDA_VISIBLE_DEVICES="${DEVICES}" \
  WANDB_RUN_ID="${exp_name}" \
  python train.py --config-name "${config_name}" \
    system.CUDA_VISIBLE_DEVICES=\"${DEVICES}\" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.experiment_name="${exp_name}" \
    trainer.project_name="${PROJ}" \
    trainer.default_local_dir="${out_dir}" \
    +ray.num_gpus=${NUM_GPUS} \
    +ray.num_cpus=32 \
    ${algo_flags} \
    agent_proxy.max_turn=${turns} \
    ${base_search_config} \
    ${all_datasets_config} \
    actor_rollout_ref.rollout.max_model_len=${max_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_tok} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    es_manager.train.env_groups=${ENV_GROUPS} \
    es_manager.train.group_size=${GROUP_SIZE} \
    es_manager.train.env_configs.tags=[${train_tag}] \
    es_manager.train.env_configs.n_groups=[${ENV_GROUPS}] \
    es_manager.val.env_groups=${total_val_groups} \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags=[${test_tags_str}] \
    es_manager.val.env_configs.n_groups=[${test_n_groups_str}] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL} \
    model_path="${model_path}" \
    2>&1 | tee "${log_file}"
}

# Function to check if search server is running
check_search_server() {
  local server_url="http://127.0.0.1:8000"
  if command -v curl &> /dev/null; then
    # Try to connect to the server (check if it's responding)
    # We check the root or retrieve endpoint, not /health which may not exist
    if curl -s --connect-timeout 2 "${server_url}/" > /dev/null 2>&1 || \
       curl -s --connect-timeout 2 "${server_url}/retrieve" > /dev/null 2>&1; then
      echo "✓ Search server appears to be running at ${server_url}"
      return 0
    else
      echo "⚠ Warning: Could not connect to search server at ${server_url}"
      echo "  The search environment requires a running search server."
      echo "  Please start the search server before running training."
      echo "  Example: cd Search-R1 && bash retrieval_launch.sh"
      read -p "Continue anyway? (y/n) " -n 1 -r
      echo
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
      fi
      return 1
    fi
  else
    echo "⚠ Warning: curl not found, cannot check search server status"
    echo "  Please ensure the search server is running at ${server_url}"
    return 1
  fi
}

# Main function
main() {
  # Check if search server is running
  check_search_server
  
  local model_path="Qwen/Qwen2.5-3B"
  local algorithm="ppo"
  
  echo "Starting search environment training..."
  echo "Model: ${model_path}"
  echo "Algorithm: ${algorithm}"
  echo "Training datasets: ${VALID_DATASETS[*]}"
  echo "Evaluation datasets: ${TEST_DATASETS[*]}"
  echo ""
  
  # You can easily comment out any experiment you don't want to run
  
  # NQ experiments
  run_experiment "nq" 8 "$model_path" "$algorithm"
  run_experiment "nq" 6 "$model_path" "$algorithm"
  
  # HotpotQA experiments
  run_experiment "hotpotqa" 8 "$model_path" "$algorithm"
  run_experiment "hotpotqa" 6 "$model_path" "$algorithm"
}

main "$@"
