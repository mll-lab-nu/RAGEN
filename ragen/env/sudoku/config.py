from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass

@dataclass
class SudokuEnvConfig:
    render_mode: str = "text"
    board_size: int = 4                   # 4 或 9
    max_steps: int = 30                   # 动作步数上限
    difficulty: str = "easy"              # easy / medium / hard
 