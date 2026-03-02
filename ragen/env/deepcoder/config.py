from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepCoderEnvConfig:
    render_mode: str = "text"
    max_steps: int = 1
    invalid_action_score: float = 0.0
    dataset_path: Optional[str] = None
    cache_dir: Optional[str] = None
    split: str = "test"
