"""
FSDP PPO Trainer with Ray-based single controller.
Adapted from the excellently written verl implementation.
"""

import os
import uuid
import ray
import torch
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from pprint import pprint
from copy import deepcopy

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from ragen.trainer.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


from verl.trainer.ppo.ray_trainer import ResourcePoolManager, compute_response_mask, apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer

import torch
from verl.utils.torch_functional import masked_mean

from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.utils import GenerationsLogger
from ragen.trainer.rollout_filter import build_rollout_filter


def adjust_batch(batch: DataProto, size_divisor: int, mode: str = "copy") -> DataProto:
    """
    Adjust batch size to be divisible by size_divisor.

    Args:
        batch: The DataProto batch to adjust
        size_divisor: The number that batch size should be divisible by
        mode: "copy" to duplicate samples, "delete" to remove samples

    Returns:
        Adjusted DataProto with batch size divisible by size_divisor
    """
    bs = len(batch.batch) if hasattr(batch.batch, '__len__') else batch.batch.batch_size[0]
    remainder = bs % size_divisor

    if remainder == 0:
        return batch

    if mode == "delete":
        # Remove samples to make it divisible
        remove_indices = np.random.choice(bs, remainder, replace=False)
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool)
        if batch.batch is not None:
            tensor_data = batch.batch[keep_mask_tensor]
        else:
            tensor_data = None

        non_tensor_data = {}
        if batch.non_tensor_batch is not None:
            for key, val in batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    non_tensor_data[key] = val[keep_mask]
                elif isinstance(val, list):
                    non_tensor_data[key] = [v for v, m in zip(val, keep_mask) if m]
                else:
                    non_tensor_data[key] = val

        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=batch.meta_info)

    elif mode == "copy":
        # Duplicate samples to make it divisible
        to_add = size_divisor - remainder
        dup_indices = np.random.choice(bs, to_add, replace=False)

        # Create duplicated batch using TensorDict concat
        dup_indices_tensor = torch.tensor(dup_indices, dtype=torch.long)
        if batch.batch is not None:
            dup_tensor_data = batch.batch[dup_indices_tensor]
            # Use TensorDict's cat method
            from tensordict import TensorDict
            tensor_data = TensorDict.cat([batch.batch, dup_tensor_data], dim=0)
        else:
            tensor_data = None

        non_tensor_data = {}
        if batch.non_tensor_batch is not None:
            for key, val in batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    dup_val = val[dup_indices]
                    non_tensor_data[key] = np.concatenate([val, dup_val], axis=0)
                elif isinstance(val, list):
                    dup_val = [val[i] for i in dup_indices]
                    non_tensor_data[key] = val + dup_val
                else:
                    non_tensor_data[key] = val

        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=batch.meta_info)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'copy' or 'delete'.")

    return adjusted_batch


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, bi_level_gae=False, high_level_gamma=1.0):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        if bi_level_gae:
            advantages, returns = core_algos.compute_bi_level_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                loss_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
                high_level_gamma=high_level_gamma,
            )
        else:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        # Pass episode_ids for deduplication in without_history mode
        episode_ids = data.non_tensor_batch.get("episode_ids", None)
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            episode_ids=episode_ids,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


class RayAgentTrainer(VerlRayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        # do not use the original val logger, but use this here
        self.generations_logger = GenerationsLogger()

        
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        assert self.config.trainer.total_training_steps is not None, "must determine total training steps"
        total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
        # val_start = 100000
        # self.train_seeds = [seed for seed in range(0, self.config.trainer.total_training_steps * 1000, 1000)]
        # self.val_seeds = [seed for seed in range(val_start, val_start + self.config.trainer.validation_steps)]

    def init_agent_proxy(self):
        self.agent_proxy = LLMAgentProxy(
            config=self.config,
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer
        )

    def _maybe_log_generations(self, inputs, outputs, scores, _type="val"):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.generations_to_log_to_wandb[_type]

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.generations_logger.log(self.config.trainer.logger, samples, self.global_steps, _type)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        env_metric_dict = {}
        for step in range(self.config.trainer.validation_steps):
            
            meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            import time
            start_time = time.time()
            test_batch = self.agent_proxy.rollout(test_gen_batch, val=True)
            end_time = time.time()
            print(f"validation generation time: {end_time - start_time} seconds")
            for key, value in test_batch.meta_info["metrics"].items():
                if "val-env/" + key not in env_metric_dict:
                    env_metric_dict["val-env/" + key] = []
                env_metric_dict["val-env/" + key].append(value)

            # Store original inputs and outputs
            batch_size = test_batch.batch["input_ids"].shape[0]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

            # Handle without_history mode: group messages by episode
            without_history = getattr(self.config.agent_proxy, "without_history", False)
            if without_history and "messages_list" in test_batch.non_tensor_batch:
                # Group samples by episode_id to reconstruct episodes
                episode_ids = test_batch.non_tensor_batch["episode_ids"]
                messages_list = test_batch.non_tensor_batch["messages_list"]

                # Find unique episodes and their samples
                unique_groups = []
                group_to_indices = {}
                for i, eid in enumerate(episode_ids):
                    if eid not in group_to_indices:
                        unique_groups.append(eid)
                        group_to_indices[eid] = []
                    group_to_indices[eid].append(i)

                # Create grouped outputs
                grouped_inputs = []
                grouped_outputs = []
                for gid in unique_groups:
                    indices = group_to_indices[gid]
                    # Combine all messages from this episode
                    episode_output = ""
                    for idx in indices:
                        msgs = messages_list[idx]
                        for msg in msgs:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            episode_output += f"[{role}]\n{content}\n\n"
                    grouped_inputs.append("")
                    grouped_outputs.append(episode_output.strip())

                sample_inputs.extend(grouped_inputs)
                sample_outputs.extend(grouped_outputs)
            else:
                input_texts = ["" for _ in range(batch_size)]
                sample_inputs.extend(input_texts)
                sample_outputs.extend(output_texts)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()

            # Group scores by episode if without_history mode
            if without_history and "messages_list" in test_batch.non_tensor_batch:
                grouped_scores = []
                for gid in unique_groups:
                    indices = group_to_indices[gid]
                    # Use the first turn's score (all turns have same episode reward)
                    episode_score = scores[indices[0]]
                    grouped_scores.append(episode_score)
                sample_scores.extend(grouped_scores)
                reward_extra_infos_dict["reward"].extend(grouped_scores)
            else:
                sample_scores.extend(scores)
                reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # Get data sources and group if needed
            data_sources_batch = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            if without_history and "messages_list" in test_batch.non_tensor_batch:
                # Group data sources by episode
                grouped_data_sources = [data_sources_batch[group_to_indices[gid][0]] for gid in unique_groups]
                data_source_lst.append(grouped_data_sources)
            else:
                data_source_lst.append(data_sources_batch)

        self._maybe_log_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, _type="val")

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = reduce_metrics(env_metric_dict)

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        super().init_workers()

        # create rollout filter
        rollout_cfg = self.config.actor_rollout_ref.rollout
        rollout_metric = getattr(rollout_cfg, "rollout_filter_metric", "reward_variance")
        self.rollout_filter = build_rollout_filter(
            ratio=rollout_cfg.rollout_filter_ratio,
            filter_type=rollout_cfg.rollout_filter_type,
            num_groups=self.config.es_manager.train.env_groups,
            group_size=self.config.es_manager.train.group_size,
            metric=rollout_metric,
            compute_log_prob=self.actor_rollout_wg.compute_log_prob,
        )


    def _save_checkpoint(self):
        """ 
        Different from VerlRayPPOTrainer, we have no dataloader so we won"t save it. Other logic is the same.
        """
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
         to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        def _process_batch_for_logging(batch):
            inputs_raw = batch.batch["input_ids"]
            inputs = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in inputs_raw]
            outputs = [""] * len(inputs)
            scores = batch.batch["rm_scores"].sum(-1).cpu().tolist()

            # Group by episode if without_history mode
            without_history = getattr(self.config.agent_proxy, "without_history", False)
            if without_history and "messages_list" in batch.non_tensor_batch:
                episode_ids = batch.non_tensor_batch["episode_ids"]
                messages_list = batch.non_tensor_batch["messages_list"]

                # Find unique episodes and their samples
                unique_groups = []
                group_to_indices = {}
                for i, eid in enumerate(episode_ids):
                    if eid not in group_to_indices:
                        unique_groups.append(eid)
                        group_to_indices[eid] = []
                    group_to_indices[eid].append(i)

                # Create grouped outputs
                grouped_inputs = []
                grouped_outputs = []
                grouped_scores = []
                for gid in unique_groups:
                    indices = group_to_indices[gid]
                    # Combine all messages from this episode
                    episode_output = ""
                    for idx in indices:
                        msgs = messages_list[idx]
                        for msg in msgs:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            episode_output += f"[{role}]\n{content}\n\n"
                    grouped_inputs.append("")
                    grouped_outputs.append(episode_output.strip())
                    grouped_scores.append(scores[indices[0]])

                return grouped_inputs, grouped_outputs, grouped_scores

            return inputs, outputs, scores

        import time
        self.start_time = time.time()
        for step in range(self.total_training_steps):
            # metrics = {}
            timing_raw = {}

            batch: DataProto = DataProto()
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw):
                    batch = self.agent_proxy.rollout(batch, val=False)

                    # Filter first, then adjust batch size
                    batch, metrics = self.rollout_filter.filter(batch)

                    # Adjust batch size to be divisible by num_groups, ppo_mini_batch_size, and n_gpus
                    num_groups = self.config.es_manager.train.env_groups
                    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    n_gpus = self.config.trainer.n_gpus_per_node
                    size_divisor = np.lcm.reduce([num_groups, ppo_mini_batch_size, n_gpus])
                    adjust_mode = getattr(self.config.agent_proxy, "batch_adjust_mode", "copy")
                    batch = adjust_batch(batch, size_divisor, mode=adjust_mode)

                    # Record batch and mini-batch statistics
                    batch_size = batch.batch["input_ids"].shape[0]
                    num_mini_batches = batch_size // ppo_mini_batch_size
                    metrics.update({
                        "train/batch_size": batch_size,
                        "train/num_mini_batches": num_mini_batches,
                    })
                    metrics.update({"train/" + key: value for key, value in batch.meta_info["metrics"].items()})

                    inputs, outputs, scores = _process_batch_for_logging(batch)
                    # self._maybe_log_generations(inputs=inputs, outputs=outputs, scores=scores, _type="train")

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # TODO: check if this is correct. Not tested yer
                    logger.log("[NotImplemented] REMAX implementation is not tested yet in RAGEN. Exiting.")
                    exit()
                    with marked_timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.agent_proxy.rollout(gen_baseline_batch, val=False)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            # dtype=object)
                # repeat to align with repeated responses in rollout
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                # batch = batch.union(gen_batch_output)

                # NOTE reward normalization already done in ctx_manager, so set group size = 1 here
                # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            # dtype=object)
                
                # NOTE: do not do reward normalization in ctx_manager, so we need to do it here
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["group_ids"]

                # batch.batch["response_mask"] = compute_response_mask(batch)
                batch.batch["response_mask"] = batch.batch["loss_mask"]
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if self.use_rm:
                    with marked_timer("reward", timing_raw):
                    # compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # recompute old_log_probs

                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        from verl.utils.debug.metrics import calculate_debug_metrics

                        metrics.update(calculate_debug_metrics(batch))

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                        avg_ref_log_prob = masked_mean(ref_log_prob.batch["ref_log_prob"], batch.batch["response_mask"])
                        metrics.update({"rollout/ref_log_prob": avg_ref_log_prob})

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    print(f"{list(reward_extra_infos_dict.keys())=}")
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, multi_turn=True)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process

                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        multi_turn=True,
                        high_level_gamma=self.config.algorithm.high_level_gamma,
                        bi_level_gae=self.config.algorithm.bi_level_gae,
                    )

                ##### A very different setting, just here for testing: Can I normalize the advantages to have a mean of 0?
                if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO and self.config.grpo_advantage_length_weight:
                    response_mask = batch.batch["response_mask"]
                    advantages = batch.batch["advantages"]
                    response_relative_lengths = (torch.sum(response_mask, dim=-1) + 1e-6) / torch.sum(response_mask, dim=-1).float().mean()
                    advantages = advantages / response_relative_lengths.unsqueeze(-1) 
                    batch.batch["advantages"] = advantages

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = True
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw):
                        print(batch.batch.keys())
                        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with marked_timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with marked_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # add another timing metric: total time
            metrics.update({"timing_s/total": time.time() - self.start_time})
            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
