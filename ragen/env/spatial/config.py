from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from ragen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType

@dataclass
class SpatialGymConfig:
    """
    Configuration for the SpatialGym environment.
    """
    # Environment specific configuration
    name: str = 'unnamed_env'
    render_mode: str = "text"

    # Room configuration
    room_size: List[int] = field(default_factory=lambda: [10, 10])
    n_objects: int = 3
    level: int = 0
    main: int = 6

    # Exploration configuration
    max_exp_steps: int = 10
    
    # Evaluation configuration
    eval_tasks: List[str] = field(default_factory=lambda: [
        "dir", "rot", "rot_dual", "pov", "bwd_pov", 
        "e2a", "fwd_loc", "bwd_loc", "fwd_fov", "bwd_nav"
    ])

    prompt_config: Dict[str, Any] = field(default_factory=lambda: {"topdown": False, "oblique": False, "type": "shorter"})

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate room size
        assert self.room_size[0] > 0 and self.room_size[1] > 0, "room_size must be positive"
        self._validate_eval_tasks()
        assert self.render_mode == 'text', "Only text render mode is supported in RAGEN"

    def _validate_eval_tasks(self):
        """Validate eval_tasks parameter."""
        valid_eval_tasks = EvalTaskType.get_short_names()
        
        if not self.eval_tasks:
            raise ValueError("eval_tasks must be non-empty")

        for task_name in self.eval_tasks:
            if not isinstance(task_name, str):
                raise ValueError(f"eval_tasks must be a list of strings, got {type(task_name)}")
            if task_name not in valid_eval_tasks:
                raise ValueError(f"task_type '{task_name}' must be one of {valid_eval_tasks}")
