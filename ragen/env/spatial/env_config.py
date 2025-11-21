from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import ListConfig, OmegaConf
import os
import json
import numpy as np

from vagen.env.base.base_env_config import BaseEnvConfig
from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType


@dataclass
class SpatialGymConfig(BaseEnvConfig):
    """
    Configuration for the SpatialGym environment.

    Parameters:
        exp_type: Exploration type ('passive', 'active')
        perspective: Perspective of exploration ('ego' or 'allo')
        eval_tasks: List of evaluation tasks with their configurations
        max_exp_steps: Maximum exploration steps for active exploration
        render_mode: Rendering mode (currently only 'text' supported)
    """

    # common configuration
    env_name: str = field(default="SpatialGym", init=False)
    max_actions_per_step: int = field(default=1, init=False)    
    prompt_format: str = field(default="free_think", init=False)
    action_sep: str = field(default="|", init=False)
    image_size: Tuple[int, int] = field(default=(300, 300), init=False)

    # Environment specific configuration
    name: str = 'unnamed_env'
    render_mode: str = field(default="vision")

    # Room configuration (minimal additions from RAGEN)
    room_size: List[int] = field(default_factory=lambda: [10, 10])
    n_objects: int = 3
    level: int = 0
    main: int = 6

    # Field of view and base directory
    field_of_view: int = field(default=90, init=False)
    data_dir: str = os.path.join(os.path.dirname(__file__), "room_data/")
    # Exploration configuration
    exp_type: str = 'passive'
    perspective: str = 'ego'
    max_exp_steps: int = 20
    kwargs: Dict = None
    proxy_agent: str = 'scout'
    # Evaluation configuration
    # Each eval task entry supports {task_type, task_kwargs}
    eval_tasks: List[Dict[str, Any]] = field(default_factory=lambda: [{"task_type": "rot", "task_kwargs": {}}])

    prompt_config: Dict[str, Any] = field(default_factory=lambda: {"topdown": False, "oblique": False, "type": "shorter"})

    calculate_information_gain: bool = False

    def config_id(self) -> str:
        eval_task_str = ", ".join([f"{task['task_type']}" for task in self.eval_tasks])
        return f"SpatialGymConfig(mode={self.render_mode},format={self.prompt_format},eval_tasks={eval_task_str})"

    def generate_seeds(self, size, seed=0, n_candidate = 20000):
        ks = self.kwargs or {}
        start = int(ks.get('seed_start', 0))
        end = ks.get('seed_end')
        if end is None:
            return [start + i for i in range(size)]
        end = int(end)
        count = max(0, min(size, end - start + 1))
        return [start + i for i in range(count)]

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate room size (new from RAGEN)
        assert self.room_size[0] > 0 and self.room_size[1] > 0, "room_size must be positive"
        self._validate_exp_type()
        self._validate_field_of_view()
        self._validate_eval_tasks()

    def _validate_exp_type(self):
        """Validate exp_type parameter."""
        valid_exp_types = ["passive", "active"]
        if self.exp_type not in valid_exp_types:
            raise ValueError(f"exp_type must be one of {valid_exp_types}")

    def _validate_field_of_view(self):
        """Validate field_of_view parameter."""
        assert self.field_of_view == 90, "Field of view must be 90 for spatial gym"

    def _validate_eval_tasks(self):
        """Validate eval_tasks parameter."""
        valid_eval_tasks = EvalTaskType.get_short_names()
        assert len(self.eval_tasks) == 1, "Only one evaluation task is supported"

        if isinstance(self.eval_tasks, ListConfig):
            self.eval_tasks = OmegaConf.to_container(self.eval_tasks, resolve=True)

        if isinstance(self.eval_tasks, np.ndarray):
            self.eval_tasks = self.eval_tasks.tolist()


        if not self.eval_tasks:
            raise ValueError("eval_tasks must be non-empty")

        for i, task in enumerate(self.eval_tasks):
            if not isinstance(task, dict) or 'task_type' not in task:
                raise ValueError("Each eval_task must be a dict with 'task_type' key")

            task_type = task['task_type']
            if task_type not in valid_eval_tasks:
                raise ValueError(f"task_type '{task_type}' must be one of {valid_eval_tasks}")
            # validate task_kwargs if present
            if 'task_kwargs' in task and task['task_kwargs'] is not None:
                assert isinstance(task['task_kwargs'], dict), "task_kwargs must be a dict"

    def get_room_config(self) -> Dict[str, Any]:
        """Get configuration for room generation (updated from RAGEN)."""
        return {
            'room_size': self.room_size,
            'n_objects': self.n_objects,
            'level': self.level,
            'main': self.main,
        }

    def get_observation_config(self) -> Dict[str, Any]:
        return {
            'field_of_view': self.field_of_view,
            'prompt_config': self.prompt_config,
            'render_mode': self.render_mode,
            'exp_type': self.exp_type,
            "proxy_agent": self.proxy_agent
        }        
    def get_model_config(self) -> Dict[str, Any]:
        return  self.kwargs['model_config']

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # Specific config (spatial-specific parameters)
        specific_config = {
            'name': self.name,
            'room_size': self.room_size,  # New from RAGEN
            'n_objects': self.n_objects,  # New from RAGEN
            'level': self.level,  # New from RAGEN
            'main': self.main,  # New from RAGEN
            'exp_type': self.exp_type,
            'perspective': self.perspective,  # VAGEN specific
            'eval_tasks': self.eval_tasks,
            'max_exp_steps': self.max_exp_steps,
            'calculate_information_gain': self.calculate_information_gain,
            'output_dir': self.kwargs['output_dir'],
            'image_size': self.image_size,
            'prompt_config': self.prompt_config,
            'observation_config': self.get_observation_config(),
            'model_config': self.get_model_config(),
            'field_of_view': self.field_of_view,
        }

        # Common config (inherited from BaseEnvConfig)
        common_config = {
            'format_reward': self.format_reward,
            'special_token_list': self.special_token_list,
            'image_placeholder': self.image_placeholder,
        }

        return {
            **specific_config,
            **common_config
        }




if __name__ == "__main__":
    config = SpatialGymConfig(eval_tasks=[{"task_type": "rot", "task_kwargs": {"turn_direction": "clockwise"}}])
    print(config)
    print(config.to_dict())
    print(config.config_id())
    print(config.eval_tasks)
