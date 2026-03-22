from dataclasses import dataclass
import gymnasium as gym
from ragen.env.base import BaseEnvConfig


@dataclass
class Game2048EnvConfig(BaseEnvConfig):
    size: int = 4
    two_prob: float = 0.9
    render_mode: str = "text"
    action_lookup: dict = None
    use_log_reward: bool = True 

    def __post_init__(self):
        if self.action_lookup is None:
            self.action_lookup = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
        
        self.invalid_act = 0 
        self.invalid_act_score = -0.1
        self.observation_format = "grid"
        
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4)