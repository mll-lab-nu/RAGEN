"""
This is the environment state manager for the LLM agent.
author: Pingyue Zhang
date: 2025-03-30
"""
import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import random
import numpy as np
import logging

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers, all_seed
register_resolvers()

# Dataset name to Search environment tag mapping
# This allows using dataset names (like 'nq', 'hotpotqa') directly instead of requiring separate config entries
SEARCH_DATASET_NAMES = {
    'nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', 
    '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa', 'eli5'
}

# Mapping from capitalized tag names to dataset names
TAG_TO_DATASET = {
    'NQ': 'nq',
    'TriviaQA': 'triviaqa',
    'PopQA': 'popqa',
    'WebQuestions': 'web_questions',
    'HotpotQA': 'hotpotqa',
    '2WikiMultihopQA': '2wikimultihopqa',
    'MuSiQue': 'musique',
    'Bamboogle': 'bamboogle',
    'StrategyQA': 'strategyqa',
    'ELI5': 'eli5',
}

def normalize_to_dataset_name(name: str) -> str:
    """Normalize a tag or dataset name to lowercase dataset name"""
    # Check if it's already a dataset name
    name_lower = name.lower()
    if name_lower in SEARCH_DATASET_NAMES:
        return name_lower
    # Check if it's a capitalized tag
    if name in TAG_TO_DATASET:
        return TAG_TO_DATASET[name]
    # Try lowercase version
    if name_lower in SEARCH_DATASET_NAMES:
        return name_lower
    return name_lower  # Return lowercase as fallback

def is_search_dataset_name(name: str) -> bool:
    """Check if a name is a Search dataset name (case-insensitive)"""
    name_lower = name.lower()
    if name_lower in SEARCH_DATASET_NAMES:
        return True
    # Also check capitalized tags
    if name in TAG_TO_DATASET:
        return True
    return False

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        seed_cfg = getattr(self.sys_config, "seed", None)
        if seed_cfg is not None:
            self.base_seed = seed_cfg.get(mode, None)
        else:
            self.base_seed = None
        self.seed_counter = 0
        self._init_envs()
        self.rollout_cache = None
        self._executors: Dict[str, ThreadPoolExecutor] = {}
        self._executors_shutdown = False
        self._register_parallel_executors()
        atexit.register(self._shutdown_executors)

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

    def _get_or_create_env_config(self, tag_or_dataset: str):
        """Get environment config from custom_envs, or create dynamically for Search datasets"""
        # Check if tag exists in custom_envs
        if tag_or_dataset in self.sys_config.custom_envs:
            return self.sys_config.custom_envs[tag_or_dataset]
        
        # If not found, check if it's a Search dataset name
        if is_search_dataset_name(tag_or_dataset):
            # Use the base Search config as template
            if 'Search' not in self.sys_config.custom_envs:
                raise ValueError(f"Base 'Search' config not found in custom_envs. Cannot create Search environment for dataset '{tag_or_dataset}'")
            
            base_search_config = self.sys_config.custom_envs['Search']
            # Create a new config dict with data_source overridden
            from dataclasses import is_dataclass, asdict
            from omegaconf import DictConfig, OmegaConf
            import copy
            
            # Convert base_search_config to dict, handling OmegaConf, dataclass, or dict
            if isinstance(base_search_config, dict):
                base_config_dict = copy.deepcopy(base_search_config)
            elif isinstance(base_search_config, DictConfig):
                base_config_dict = copy.deepcopy(OmegaConf.to_container(base_search_config, resolve=True))
            elif is_dataclass(base_search_config):
                base_config_dict = copy.deepcopy(asdict(base_search_config))
            else:
                # Fallback: try to convert using vars() or dict()
                try:
                    base_config_dict = copy.deepcopy(dict(base_search_config) if hasattr(base_search_config, '__iter__') else vars(base_search_config))
                except (TypeError, ValueError) as e:
                    raise TypeError(f"base_search_config must be a dict, OmegaConf DictConfig, or dataclass instance, got {type(base_search_config)}: {e}")
            
            new_config = base_config_dict
            
            # Normalize to dataset name (handles both 'nq' and 'NQ')
            dataset_name = normalize_to_dataset_name(tag_or_dataset)
            
            # Override data_source
            if new_config.get('env_config') is None:
                new_config['env_config'] = {}
            new_config['env_config']['data_source'] = dataset_name
            
            # Set max_steps and max_actions_per_traj based on dataset (only if not already set via Hydra overrides)
            # Default values: multi-hop datasets need more steps
            multi_hop_datasets = {'hotpotqa', '2wikimultihopqa', 'musique'}
            if dataset_name in multi_hop_datasets:
                default_max_actions = 6
                default_max_steps = 6
            else:
                default_max_actions = 4
                default_max_steps = 4
            
            # Only set defaults if not already configured (allows Hydra overrides to take precedence)
            if 'max_actions_per_traj' not in new_config or new_config.get('max_actions_per_traj') is None:
                new_config['max_actions_per_traj'] = default_max_actions
            
            if new_config.get('env_config') and ('max_steps' not in new_config['env_config'] or new_config['env_config'].get('max_steps') is None):
                new_config['env_config']['max_steps'] = default_max_steps
            
            # Create a simple object to hold the config (similar to OmegaConf structure)
            class DynamicEnvConfig:
                def __init__(self, config_dict):
                    self.env_type = config_dict.get('env_type', 'search')
                    self.max_actions_per_traj = config_dict.get('max_actions_per_traj', 4)
                    self.env_config = config_dict.get('env_config', {})
                    self.env_instruction = config_dict.get('env_instruction', '')
                    self.max_tokens = config_dict.get('max_tokens', 500)
                    # Store all other attributes
                    for k, v in config_dict.items():
                        if not hasattr(self, k):
                            setattr(self, k, v)
            
            return DynamicEnvConfig(new_config)
        
        raise ValueError(f"Environment tag or dataset name '{tag_or_dataset}' not found in custom_envs and is not a valid Search dataset name")

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self._get_or_create_env_config(tag)
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    # Handle dict, OmegaConf DictConfig, and dataclass env_config
                    from omegaconf import DictConfig, OmegaConf
                    from dataclasses import is_dataclass, asdict
                    
                    if isinstance(cfg_template.env_config, dict):
                        env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                    elif isinstance(cfg_template.env_config, DictConfig):
                        # Convert OmegaConf DictConfig to dict
                        env_config_dict = OmegaConf.to_container(cfg_template.env_config, resolve=True)
                        env_config = REGISTERED_ENV_CONFIGS[env_class](**env_config_dict)
                    elif is_dataclass(cfg_template.env_config):
                        env_config = REGISTERED_ENV_CONFIGS[env_class](**asdict(cfg_template.env_config))
                    else:
                        # Fallback: try to convert to dict using vars() or get attributes
                        try:
                            env_config_dict = dict(cfg_template.env_config) if hasattr(cfg_template.env_config, '__iter__') else vars(cfg_template.env_config)
                            env_config = REGISTERED_ENV_CONFIGS[env_class](**env_config_dict)
                        except (TypeError, ValueError) as e:
                            raise TypeError(f"env_config must be a dict, OmegaConf DictConfig, or dataclass instance, got {type(cfg_template.env_config)}: {e}")
                
                # Pass mode to environment config (for Search environment to load correct data split)
                # This matches Search-R1's approach of using separate train/test splits
                if hasattr(env_config, 'mode'):
                    env_config.mode = self.mode
                
                env_obj = REGISTERED_ENVS[env_class](env_config)
                parallel_friendly = bool(getattr(cfg_template, "parallel_friendly", False))
                max_workers = int(getattr(cfg_template, "max_workers", 1) or 1)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj,
                        'parallel_friendly': parallel_friendly, 'max_workers': max_workers}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def _register_parallel_executors(self):
        tag_seen: Dict[str, dict] = {}
        for entry in self.envs:
            tag = entry["tag"]
            cfg = {
                "parallel_friendly": entry.get("parallel_friendly", False),
                "max_workers": entry.get("max_workers", 1),
            }
            if tag in tag_seen:
                assert tag_seen[tag] == cfg, f"Inconsistent config for tag {tag}: {tag_seen[tag]} vs {cfg}"
            else:
                tag_seen[tag] = cfg

        for tag, cfg in tag_seen.items():
            parallel_friendly = cfg.get('parallel_friendly', False)
            max_workers = cfg.get('max_workers', 1)
            if parallel_friendly and max_workers > 1:
                self._executors[tag] = ThreadPoolExecutor(max_workers=max_workers)

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        Uses sampling without replacement to ensure unique question selection per dataset.
        """
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        # reset all environments
        if seed is None:
            if self.mode == "train":
                if self.base_seed is not None:
                    seed = self.base_seed + self.seed_counter
                    self.seed_counter += self.env_groups
                else:
                    seed = random.randint(0, 1000000)
            else:
                seed = 123 if self.base_seed is None else self.base_seed
        else:
            if self.mode == "train" and self.base_seed is not None:
                self.seed_counter = seed - self.base_seed + 1
        
        # Group environments by tag (dataset) for sampling without replacement
        tag_to_envs = {}
        for entry in envs:
            tag = entry['tag']
            if tag not in tag_to_envs:
                tag_to_envs[tag] = []
            tag_to_envs[tag].append(entry)
        
        # Prepare data indices for each environment (sampling without replacement per tag)
        env_data_indices = {}  # env_id -> data_idx or None
        
        # Sample without replacement for each tag
        with all_seed(seed):
            for tag, tag_envs in tag_to_envs.items():
                # Check if this is a Search environment (has 'data' attribute)
                first_env = tag_envs[0]['env']
                if hasattr(first_env, 'data'):
                    try:
                        dataset_size = len(first_env.data)
                    except:
                        dataset_size = None
                else:
                    dataset_size = None
                
                if dataset_size is not None and dataset_size > 0:
                    # Sample without replacement for Search environments
                    num_envs = len(tag_envs)
                    if num_envs <= dataset_size:
                        # We have enough unique samples: sample without replacement
                        indices = random.sample(range(dataset_size), num_envs)
                    else:
                        # More environments than samples: cycle through with random start
                        # This ensures even distribution while allowing repeats when necessary
                        base_indices = list(range(dataset_size))
                        random.shuffle(base_indices)
                        # Repeat the list to cover all environments
                        indices = (base_indices * ((num_envs // dataset_size) + 1))[:num_envs]
                    
                    # Assign indices to environments
                    for env_entry, data_idx in zip(tag_envs, indices):
                        env_data_indices[env_entry['env_id']] = data_idx
                else:
                    # Not a Search environment or dataset size unknown: no data_idx
                    for env_entry in tag_envs:
                        env_data_indices[env_entry['env_id']] = None
        
        # Reset all environments with appropriate parameters
        seeds = _expand_seed(seed)
        for seed_val, entry in zip(seeds, envs):
            data_idx = env_data_indices.get(entry['env_id'])
            if data_idx is not None:
                # Search environment: use data_idx for sampling without replacement
                entry['env'].reset(seed=seed_val, mode=self.mode, data_idx=data_idx)
            else:
                # Other environment: use original method
                entry['env'].reset(seed=seed_val, mode=self.mode)
            entry['status'] = EnvStatus(seed=seed_val)

        # update rollout cache
        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env['env'].render())
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            
        self.rollout_cache = rollout_cache
        return rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True # TODO check terminated definition in gymnasium
                status.truncated = not turn_info.get('success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
            })
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        envs = self.envs
        env_outputs = []

        def _process_env_input(env_input: Dict) -> Dict:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            # execute actions in envs
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env, valid_actions[:actions_left_before])
            no_manager_action = len(valid_actions) == 0
            penalty_delta = 0.0
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                penalty_delta = self.sys_config.es_manager.format_penalty
            if no_manager_action:
                turn_info = dict(turn_info)
                turn_info['manager_invalid_action'] = True

            status, history = _log_env_state(entry['status'], self.rollout_cache[env_id]['history'], entry['env'].render(), entry['max_actions_per_traj'], executed_actions, valid_actions, acc_reward, turn_done, turn_info, env_input)
            if no_manager_action and history:
                history[-1]['manager_invalid_action'] = True
            if status.num_actions >= entry['max_actions_per_traj'] and not turn_done:
                status.truncated = True
                status.terminated = True
                turn_done = True

            return {
                'env_id': env_id,
                'status': status,
                'history': history,
                'turn_done': turn_done,
                'penalty_delta': penalty_delta,
            }

        results: List[Optional[Dict]] = [None] * len(all_env_inputs)
        tag2items: Dict[str, List[tuple]] = {}
        for idx, env_input in enumerate(all_env_inputs):
            entry = envs[env_input['env_id']]
            tag2items.setdefault(entry['tag'], []).append((idx, env_input))

        for tag, items in tag2items.items():
            sample_entry = envs[items[0][1]['env_id']]
            parallel_friendly = sample_entry.get('parallel_friendly', False)
            max_workers = sample_entry.get('max_workers', 1)
            executor = self._executors.get(tag) if parallel_friendly and max_workers > 1 else None
            if executor is None or len(items) == 1:
                for idx, env_input in items:
                    results[idx] = _process_env_input(env_input)
            else:
                futures = {executor.submit(_process_env_input, env_input): idx for idx, env_input in items}
                for future, idx in futures.items():
                    results[idx] = future.result()

        for result in results:
            if result is None:
                continue
            env_id = result['env_id']
            entry = envs[env_id]
            if result['penalty_delta']:
                self.rollout_cache[env_id]["penalty"] += result['penalty_delta']
            self.rollout_cache[env_id]['history'] = result['history']
            entry['status'] = result['status']
            if not result['turn_done']:
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    def get_rollout_states(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            custom_metric = {}
            # Metrics that should use the last turn's value instead of averaging
            LAST_TURN_METRICS = ['accuracy', 'rouge', 'format_reward', 'total_reward']
            
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if k == 'success' or k == 'reward_type':
                        continue
                    # Skip non-numeric values (e.g., strings that can't be converted to float)
                    try:
                        float_val = float(v)
                    except (ValueError, TypeError):
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    custom_metric[k].append(float_val)
            for k, v in custom_metric.items():
                # For metrics like accuracy, format_reward, total_reward, use the last turn's value
                # (these are only meaningful at the final answer turn)
                if k in LAST_TURN_METRICS and len(v) > 0:
                    env_metric[k] = v[-1]  # Use last turn's value
                # TODO: Move TURN_LVL_METRICS into the environment
                elif "webshop" not in cache['tag'].lower() or ("webshop" in cache['tag'].lower() and k in TURN_LVL_METRICS):
                    env_metric[k] = np.sum(v) / (len(cache['history']) - 1) # NOTE: exclude the last observation
                else:
                    env_metric['traj_sum/' + k] = np.sum(v)


            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            
            # Extract data_source from environment config (for Search environments and others that have it)
            if hasattr(entry['config'], 'data_source'):
                cache['data_source'] = entry['config'].data_source
            elif hasattr(entry['env'], 'config') and hasattr(entry['env'].config, 'data_source'):
                cache['data_source'] = entry['env'].config.data_source
            else:
                # Fallback: try to infer from tag (e.g., SearchNQ -> nq, SearchTriviaQA -> triviaqa)
                tag = entry['tag']
                # Map common Search environment tags to dataset names
                tag_to_dataset = {
                    'SearchNQ': 'nq',
                    'SearchTriviaQA': 'triviaqa',
                    'SearchPopQA': 'popqa',
                    'SearchHotpotQA': 'hotpotqa',
                    'Search2WikiMultihopQA': '2wikimultihopqa',
                    'SearchMuSiQue': 'musique',
                    'Search': 'unknown',  # Generic Search tag
                }
                if tag in tag_to_dataset:
                    cache['data_source'] = tag_to_dataset[tag]
                elif tag.lower().startswith('search'):
                    # Generic fallback: SearchX -> x (lowercase)
                    dataset_name = tag[6:].lower() if len(tag) > 6 else 'unknown'
                    cache['data_source'] = dataset_name
                else:
                    cache['data_source'] = 'unknown'
            
            if entry['tag'] == "MetamathQA":
                cache['correct_answer'] = entry['env'].correct_answer

        # calculate pass@k where k is the group size
        group_success = {}
        for entry, cache in zip(envs, rollout_cache):
            key = (entry['tag'], entry['group_id'])
            success_val = cache['metrics'].get(f"{entry['tag']}/success", 0.0)
            group_success.setdefault(key, []).append(success_val)

        for (tag, gid), succ_list in group_success.items():
            pass_success = float(any(succ_list))
            for entry, cache in zip(envs, rollout_cache):
                if entry['tag'] == tag and entry['group_id'] == gid:
                    cache['metrics'][f"{tag}/pass@{self.group_size}"] = pass_success
        return rollout_cache




    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {} # append state to history
        if isinstance(next_state, str): # text state
            entry['state'] = next_state
        else: # multimodal state
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if isinstance(state, str): # text state
            return state
        elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()
        self._shutdown_executors()

    def _shutdown_executors(self):
        if getattr(self, "_executors_shutdown", False):
            return
        self._executors_shutdown = True
        executors = getattr(self, "_executors", None)
        if not executors:
            return
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self):
        self._shutdown_executors()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
