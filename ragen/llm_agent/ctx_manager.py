"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
from itertools import zip_longest
import logging

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

def get_special_tokens(tokenizer: AutoTokenizer):
    if "qwen" in tokenizer.name_or_path.lower():
        special_token = tokenizer.encode("<|im_start|>")[0]
        reward_token = tokenizer.encode("<|im_end|>")[0]
    elif "llama-3" in tokenizer.name_or_path.lower():
        special_token = 128006
        reward_token = 128009
    else:
        raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
    return special_token, reward_token

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False, enable_response_mask: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    special_token, reward_token = get_special_tokens(tokenizer)
    
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    if enable_response_mask:
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    else:
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
        if "qwen" in tokenizer.name_or_path.lower():
            # for Qwen, there is a "\n" between special token and reward token, so we shift this to make sure reward is assigned to the last token of a turn
            score_tensor = score_tensor.roll(shifts=1, dims=-1)
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    score_tensor = score_tensor[:, 1:] # remove the first token
    loss_mask = loss_mask[:, :-1].float() # remove the last token
    response_mask = response_mask[:, :-1].float() # remove the last token

    return score_tensor, loss_mask, response_mask



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()

    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script.")

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue

            self._check_env_installed(env_config.env_type)
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k,v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            observation_format = env_config_new.get("observation_format", "grid")
            if observation_format == "grid" and env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if observation_format == "coord":
                coord_hint = (
                    "\nStates are provided as coordinate lists using zero-based indexing "
                    "with the format (row, col) for each entity type: Walls, Targets, Boxes, "
                    "Boxes on target, Player, and Player on target when applicable."
                )
                env_instruction += coord_hint
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group
            
        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_response(self, response: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

                
            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions
        
    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        
        # apply penalty pre-normalization
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor([env_output.get("penalty", 0) for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]: # the group size > 1
            for group, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor

    def _build_independent_turn_samples(self, env_outputs: List[Dict]) -> DataProto:
        """
        Build independent samples for each turn when without_history=True.
        Each turn becomes a separate training sample with [system, user, assistant].
        """
        llm_input_texts = []
        messages_list = []
        env_ids = []
        group_ids = []
        episode_ids = []  # Unique ID for each episode
        episode_rewards = []  # Total episode reward for each sample
        uid_list = []  # For GRPO grouping

        # First pass: collect episode-level rewards for normalization
        episode_raw_rewards = []
        episode_group_ids_for_norm = []
        episode_penalties = []
        for env_output in env_outputs:
            if 'state' in env_output['history'][-1]:
                history = env_output['history'][:-1]
            else:
                history = env_output['history']
            total_reward = sum(turn.get('reward', 0.0) for turn in history)
            episode_raw_rewards.append(total_reward)
            episode_group_ids_for_norm.append(env_output["group_id"])
            episode_penalties.append(env_output.get("penalty", 0))

        # Apply reward normalization at episode level
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method

        if grouping == "state":
            group_tags = episode_group_ids_for_norm
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")

        # Build normalization function
        if method == "mean_std":
            norm_func = lambda x: (x - x.mean()) / (x.std() + 1e-6) if x.std().abs() > 1e-6 else torch.zeros_like(x)
        elif method == "mean":
            norm_func = lambda x: (x - x.mean())
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean()) / (x.std() + 1e-6) if x.std().abs() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # Apply groupwise normalization to episode rewards
        episode_rewards_tensor = torch.tensor(episode_raw_rewards, dtype=torch.float32)
        episode_penalties_tensor = torch.tensor(episode_penalties, dtype=torch.float32)
        normalized_episode_rewards = episode_rewards_tensor + episode_penalties_tensor

        group2index = {}
        for i, tag in enumerate(group_tags):
            if tag not in group2index:
                group2index[tag] = []
            group2index[tag].append(i)

        if len(group2index) < len(env_outputs):  # group size > 1
            for group, indices in group2index.items():
                indices_tensor = torch.tensor(indices)
                normalized_episode_rewards[indices_tensor] = norm_func(normalized_episode_rewards[indices_tensor])

        # Second pass: build turn samples with normalized rewards
        for episode_idx, env_output in enumerate(env_outputs):
            if 'state' in env_output['history'][-1]:
                history = env_output['history'][:-1]  # Remove last state-only entry
            else:
                history = env_output['history']

            # Use normalized episode reward
            normalized_reward = normalized_episode_rewards[episode_idx].item()

            for turn_idx, turn in enumerate(history):
                if 'llm_response' not in turn:
                    continue

                # Build independent [system, user, assistant] for this turn
                messages = [
                    {"role": "system", "content": "You're a helpful assistant. "},
                    {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
                ]

                # Add turn info to user message
                actual_turn = turn_idx + 1
                FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                messages[-1]["content"] += f"\nTurn {actual_turn}:\n"
                messages[-1]["content"] += f"State:\n{turn['state']}\nYou have {turn['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"

                # Add assistant response
                messages.append({"role": "assistant", "content": turn["llm_response"]})

                # Apply chat template
                text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

                llm_input_texts.append(text)
                messages_list.append(messages)
                env_ids.append(env_output["env_id"])
                group_ids.append(env_output["group_id"])
                episode_ids.append(episode_idx)
                episode_rewards.append(normalized_reward)
                uid_list.append(env_output.get("uid", env_output["env_id"]))

        # Tokenize all samples
        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        # Build score tensor - put episode reward at last token of each sample
        score_tensor = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.float32)
        for i, reward in enumerate(episode_rewards):
            score_tensor[i, -1] = reward

        # Build loss_mask and response_mask for independent samples
        # For independent samples, we use a simplified mask: only assistant tokens
        loss_mask = torch.zeros_like(score_tensor)
        response_mask = torch.zeros_like(score_tensor)

        special_token, _ = get_special_tokens(self.tokenizer)
        for i in range(input_ids.shape[0]):
            # Find assistant turn (the last turn in each sample)
            turn_starts = (input_ids[i] == special_token).nonzero(as_tuple=True)[0]
            if len(turn_starts) >= 3:  # system, user, assistant
                assistant_start = turn_starts[2].item()
                # Shift by 1 for loss computation
                if assistant_start > 0:
                    loss_mask[i, assistant_start-1:] = 1
                    response_mask[i, assistant_start-1:] = 1

        # Remove padding from masks
        for i in range(input_ids.shape[0]):
            valid_len = attention_mask[i].sum().item()
            if valid_len < input_ids.shape[1]:
                pad_len = input_ids.shape[1] - valid_len
                loss_mask[i, :pad_len-1] = 0
                response_mask[i, :pad_len-1] = 0

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # Build DataProto
        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:],
            "loss_mask": loss_mask,
            "rm_scores": score_tensor,
            "original_rm_scores": score_tensor.clone(),
        }, batch_size=input_ids.shape[0])

        llm_inputs.non_tensor_batch = {
            "env_ids": np.array(env_ids, dtype=int),
            "group_ids": np.array(group_ids, dtype=int),
            "episode_ids": np.array(episode_ids, dtype=int),
            "messages_list": np.array(messages_list, dtype=object),
            "uid": np.array(uid_list, dtype=object),
        }

        # Compute metrics
        metrics = {}
        for env_output in env_outputs:
            for key, value in env_output.get("metrics", {}).items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

        mean_metrics = {
            key: np.sum(value) / self.env_nums.get(key.split("/")[0], 1)
            for key, value in metrics.items()
        }
        for key, values in metrics.items():
            if not isinstance(values, list):
                continue
            prefix, suffix = key.split("/", 1)
            non_zero_values = [v for v in values if v != 0]
            if non_zero_values:
                non_zero_key = f"{prefix}/non-zero/{suffix}"
                mean_metrics[non_zero_key] = np.mean(non_zero_values)

        mean_metrics["response_length"] = response_length
        llm_inputs.meta_info = {"metrics": mean_metrics}

        return llm_inputs

    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        # Handle without_history mode for training
        without_history = getattr(self.config.agent_proxy, "without_history", False)
        if without_history and prepare_for_update:
            return self._build_independent_turn_samples(env_outputs)

        llm_input_texts = []
        messages_list = [] # for api calling
        for env_output in env_outputs:
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            
            max_k = getattr(self.config.agent_proxy, "max_context_window", None)
            if max_k is not None and isinstance(max_k, int) and max_k > 0:
                env_output['history'] = env_output['history'][-max_k:]
            
            max_model_len = getattr(self.config.actor_rollout_ref.rollout, "max_model_len", None)
            history = env_output['history']
            truncated = False

            # Handle without_history mode for sampling
            without_history = getattr(self.config.agent_proxy, "without_history", False)
            turn_offset = 0
            if without_history and not prepare_for_update:
                # For sampling: only use current turn but keep actual turn number
                turn_offset = len(history) - 1
                history = history[-1:] if history else []

            while True:
                messages = [
                    {"role": "system", "content": "You're a helpful assistant. "}, 
                    {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
                ]

                for idx, content in enumerate(history):
                    actual_turn = idx + 1 + turn_offset
                    messages[-1]["content"] += f"\nTurn {actual_turn}:\n"
                    if "state" in content:
                        FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                        LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                        messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                    if "llm_response" in content:
                        messages.append({"role": "assistant", "content": content["llm_response"]})
                    if "reward" in content and not (prepare_for_update and idx == len(history) - 1):
                        messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})


                # NOTE: this assertion is important for loss mask computation        
                assert all(msg["role"] == "assistant" for msg in messages[2::2])

                text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
                if not prepare_for_update:
                    if self.config.agent_proxy.enable_think:
                        text_with_prompt = text + "<think>"
                    else:
                        text_with_prompt = text + "<answer>"
                else:
                    text_with_prompt = text

                if max_model_len is None:
                    break
                token_len = len(self.tokenizer(text_with_prompt, add_special_tokens=False)["input_ids"])
                if token_len <= max_model_len:
                    break
                if len(history) <= 1:
                    logging.error(
                        "The model in env %s has exceeded max_model_len %d in one turn (%d tokens). Please try to increase max_model_len.",
                        env_output["env_id"], max_model_len, token_len,
                    )
                    break
                history = history[1:]
                truncated = True

            if truncated:
                env_output['history'] = history
                logging.warning(
                    "Truncated history for env %s to keep prompt within %d tokens (current %d).",
                    env_output["env_id"], max_model_len, token_len,
                )

            llm_input_texts.append(text_with_prompt)
            messages_list.append(messages)

        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # We have truncated previously, truncation in tokenizer may cause issues.
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        if prepare_for_update:
            scores = [[i.get('reward', 0.0) for i in env_output['history']] for env_output in env_outputs]
            score_tensor, loss_mask, response_mask = get_masks_and_scores(input_ids, self.tokenizer, scores, use_turn_scores=self.config.agent_proxy.use_turn_scores, enable_response_mask=self.config.enable_response_mask)

            normalized_score_tensor = score_tensor
            if not self.config.agent_proxy.use_turn_scores:
                normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
        }, batch_size=input_ids.shape[0])

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor # remove the first token
            llm_inputs.batch["original_rm_scores"] = score_tensor # remove the first token
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=int),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=int),
            "messages_list": np.array(messages_list, dtype=object),
        }

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            mean_metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }
            for key, values in metrics.items():
                if not isinstance(values, list):
                    continue
                prefix, suffix = key.split("/", 1)
                non_zero_values = [v for v in values if v != 0]
                if non_zero_values:  # Avoid division by zero
                    non_zero_key = f"{prefix}/non-zero/{suffix}"
                    mean_metrics[non_zero_key] = np.mean(non_zero_values)
            metrics = mean_metrics
            metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
        responses = ["<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response for response in responses] # The LLM generation does not include <think> tags. Add them back here.
            
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions = self._parse_response(response)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs

    



@hydra.main(version_base = None, config_path = "../../config", config_name = "base")
def main(config):
    import json
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    print("ctx_manager prefix", ctx_manager.prefix_lookup)
    # batch_list = [
    #     {
    #         "env_ids": 0,
    #         "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
    #     },
    #     {
    #         "env_ids": 1,
    #         "chat_response": "<think> 456. </think><answer> 789 </answer><think> 10123 </think><answer> 11111 </answer>",
    #     }
    # ]
    # ctx_manager.action_sep_lookup = {
    #     0: "|",
    #     1: ";"
    # }
    # for item in batch_list:
    #     item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    # batch_dict = collate_fn(batch_list)
    # batch = DataProto.from_single_dict(batch_dict)
    # env_inputs = ctx_manager.get_env_inputs(batch)
    # print(env_inputs)
    


    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5, "actions_left": 2},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 0,
            "metrics": {}
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 3", "reward": 0.3, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 1,
            "metrics": {}
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(env_prompt)
    formulate_rollouts_rst= ctx_manager.formulate_rollouts(env_outputs)
    print(formulate_rollouts_rst)

if __name__ == "__main__":
    main()
    
