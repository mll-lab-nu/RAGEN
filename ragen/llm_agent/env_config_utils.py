"""
Shared utilities for environment configuration management.
Consolidates duplicate logic for creating environment configs dynamically.
"""
from typing import Any, Dict
from dataclasses import is_dataclass, asdict
from omegaconf import DictConfig, OmegaConf
import copy

from ragen.llm_agent.es_manager import is_search_dataset_name, normalize_to_dataset_name


def get_or_create_env_config(tag_or_dataset: str, custom_envs: Dict[str, Any]) -> Any:
    """Get environment config from custom_envs, or create dynamically for Search datasets.
    
    This function consolidates the duplicate logic from EnvStateManager and ContextManager.
    
    Args:
        tag_or_dataset: The environment tag or dataset name
        custom_envs: Dictionary of custom environment configurations
        
    Returns:
        Environment configuration object
        
    Raises:
        ValueError: If tag not found and not a valid Search dataset name
        TypeError: If base_search_config cannot be converted to dict
    """
    # Check if tag exists in custom_envs
    if tag_or_dataset in custom_envs:
        return custom_envs[tag_or_dataset]
    
    # If not found, check if it's a Search dataset name
    if is_search_dataset_name(tag_or_dataset):
        # Use the base Search config as template
        if 'Search' not in custom_envs:
            raise ValueError(f"Base 'Search' config not found in custom_envs. Cannot create Search environment for dataset '{tag_or_dataset}'")
        
        base_search_config = custom_envs['Search']
        
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
                base_config_dict = copy.deepcopy(
                    dict(base_search_config) if hasattr(base_search_config, '__iter__') 
                    else vars(base_search_config)
                )
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"base_search_config must be a dict, OmegaConf DictConfig, or dataclass instance, "
                    f"got {type(base_search_config)}: {e}"
                )
        
        new_config = base_config_dict
        
        # Normalize to dataset name (handles both 'nq' and 'NQ')
        dataset_name = normalize_to_dataset_name(tag_or_dataset)
        
        # Override data_source
        if new_config.get('env_config') is None:
            new_config['env_config'] = {}
        new_config['env_config']['data_source'] = dataset_name
        
        # Set max_steps and max_actions_per_traj based on dataset (only if not already set)
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
        
        if new_config.get('env_config') and (
            'max_steps' not in new_config['env_config'] 
            or new_config['env_config'].get('max_steps') is None
        ):
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
    
    raise ValueError(
        f"Environment tag or dataset name '{tag_or_dataset}' not found in custom_envs "
        f"and is not a valid Search dataset name"
    )

