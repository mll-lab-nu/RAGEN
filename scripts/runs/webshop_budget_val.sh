# exp_name=qwen_2_5_3b_instruct
# exp_name=webshop_starpos_grpo_3b_small_max_3turns_2
# exp_name=webshop_starpos_grpo_3b_small_max_4turns_2
# exp_name=webshop_starpos_grpo_3b_small_max_5turns_2
# exp_name=webshop_starpos_grpo_3b_small_max_6turns_2
exp_name=webshop_starpos_grpo_3b_small_max_7turns_2

if [ "$exp_name" = "qwen_2_5_3b_instruct" ]; then
    echo "No resume needed for $exp_name"
    resume_args=""
else
    resume_args="trainer.resume_mode=resume_path \
    trainer.resume_from_path=/blob/v-zihanwang/budget_checkpoints/$exp_name/global_step_100/qwen_actor_merged"
    # if path not exist, do merging
    if [ ! -d /blob/v-zihanwang/budget_checkpoints/$exp_name/global_step_100/qwen_actor_merged ]; then
        echo "Merging $exp_name"
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir /blob/v-zihanwang/budget_checkpoints/$exp_name/global_step_100/actor \
            --target_dir /blob/v-zihanwang/budget_checkpoints/$exp_name/global_step_100/qwen_actor_merged
    else
        echo "Merged checkpoint already exists for $exp_name"
    fi
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_RUN_ID=${exp_name} \
python -m ragen.llm_agent.ap_webshop \
    trainer.project_name=budget_eval \
    trainer.experiment_name=$exp_name \
    es_manager.val.env_configs.tags=[WebShop] \
    es_manager.val.env_configs.n_groups=[64] \
    es_manager.val.env_groups=64 \
    es_manager.val.group_size=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" \
    trainer.n_gpus_per_node=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 $resume_args \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=15000 \
    "$@"