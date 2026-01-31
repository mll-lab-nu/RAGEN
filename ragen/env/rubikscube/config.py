from dataclasses import dataclass
import gymnasium as gym
from ragen.env.base import BaseEnvConfig

@dataclass
class RubiksCube2x2Config(BaseEnvConfig):
    render_mode: str = "text"
    # 打乱的步数
    scramble_depth: int = 5 
    # 最大步数
    max_steps: int = 20
    action_lookup: dict = None

    def __post_init__(self):
        if self.action_lookup is None:
            # 12个基础动作：U, U', D, D', L, L', R, R', F, F', B, B'
            # 这里的 U=Up, D=Down, L=Left, R=Right, F=Front, B=Back
            # Prime (') 表示逆时针
            self.action_lookup = {
                1: "U",  2: "U'",
                3: "D",  4: "D'",
                5: "L",  6: "L'",
                7: "R",  8: "R'",
                9: "F",  10: "F'",
                11: "B", 12: "B'"
            }
        self.invalid_act = 0
        self.invalid_act_score = 0.0
        # 动作空间：1到12
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(12, start=1)