import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Dict, List
from ragen.env.base import BaseDiscreteActionEnv
from .config import Game2048EnvConfig
from ragen.utils import all_seed


class Game2048Env(BaseDiscreteActionEnv, gym.Env):
    def __init__(self, config: Game2048EnvConfig | None = None):
        BaseDiscreteActionEnv.__init__(self)
        self.config = config or Game2048EnvConfig()
        self.size = self.config.size
        self.two_prob = self.config.two_prob
        self.render_mode = self.config.render_mode
        self.ACTION_LOOKUP = self.config.action_lookup
        
        self.action_space = self.config.ACTION_SPACE
        self.ACTION_SPACE = self.action_space
        
        self.observation_space = gym.spaces.Text(max_length=4096)

        self.grid = None
        self.score = 0
        self.rng = np.random.default_rng()

    def reset(self, seed=None, mode=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)
        with all_seed(seed):
            self.rng = np.random.default_rng(seed)
            self.grid = np.zeros((self.size, self.size), dtype=np.int64)
            self.score = 0
            self._spawn_tile()
            self._spawn_tile()
        
        info = {
            "grid": self.grid.copy(), 
            "score": self.score,
            "action_mask": self._get_action_mask(),
            "raw_reward": 0.0  # 初始化
        }
        return self.render(), info

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        assert action in self.ACTION_LOOKUP, f"Invalid action: {action}"
        
        reward_merge, moved = self._move(action)
        
        terminated = False
        
        # === 准备 Info ===
        info = {
            "action_is_effective": moved,
            "action_is_valid": True,
            "success": False,
        }

        # === 计算 Raw Reward ===
        raw_reward = 0.0
        if not moved:
            info["action_is_valid"] = False 
            raw_reward = self.config.invalid_act_score
        else:
            self._spawn_tile()
            self.score += reward_merge # Score 始终记录原始分数
            raw_reward = float(reward_merge)
            
            terminated = self._is_terminal()
            if terminated:
                info["max_tile"] = np.max(self.grid)
                if np.max(self.grid) >= 2048:
                    info["success"] = True
        final_reward = raw_reward
        if self.config.use_log_reward:
            if raw_reward > 0:
                final_reward = float(np.log2(raw_reward + 1.0)) * 0.1
            else:
                final_reward = 0.0 

        # === 更新 Info ===
        info["grid"] = self.grid.copy()
        info["action_mask"] = self._get_action_mask()
        info["score"] = self.score            # 累计原始分数
        info["raw_reward"] = raw_reward       # 单步原始分数 (供 Logger 使用)
        
        next_obs = self.render()
        done = terminated
        
        return next_obs, final_reward, done, info

    def render(self, mode: str | None = None) -> str:
        lines: List[str] = []
        lines.append("Current 2048 Grid:")
        for i in range(self.size):
            row = ", ".join(str(int(x)) for x in self.grid[i])
            lines.append(f"Row {i+1}: [{row}]")
        
        valid_actions = self._valid_actions()
        actions_text = ", ".join(f"{a}({self.ACTION_LOOKUP[a]})" for a in valid_actions) if valid_actions else "None"
        
        lines.append("")
        lines.append(f"Valid Actions: {actions_text}.")
        lines.append("Goal: Merge same numbers to reach 2048.")
        lines.append(f"Current Score: {self.score}") # 显示分数
        lines.append("What is your next move?")
        return "\n".join(lines)

    def close(self):
        pass

    def get_all_actions(self):
        return list(self.ACTION_LOOKUP.keys())

    # ---- 2048 helpers (保持不变) ----
    def _spawn_tile(self):
        empties = list(zip(*np.where(self.grid == 0)))
        if not empties:
            return
        r, c = empties[self.rng.integers(0, len(empties))]
        value = 2 if self.rng.random() < self.two_prob else 4
        self.grid[r, c] = value

    def _valid_actions(self) -> List[int]:
        valid = []
        for a in self.ACTION_LOOKUP.keys():
            tmp = self.grid.copy()
            merged, moved = self._move_sim(tmp, a)
            if moved:
                valid.append(a)
        return valid
        
    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(4, dtype=bool)
        valid_indices = self._valid_actions()
        for idx in valid_indices:
            mask[idx] = True
        return mask

    def _is_terminal(self) -> bool:
        if np.any(self.grid == 0):
            return False
        for a in self.ACTION_LOOKUP.keys():
            tmp = self.grid.copy()
            _, moved = self._move_sim(tmp, a)
            if moved:
                return False
        return True

    def _move(self, action: int) -> Tuple[int, bool]:
        grid = self.grid
        reward, moved = self._move_sim(grid, action)
        return reward, moved

    def _move_sim(self, grid: np.ndarray, action: int) -> Tuple[int, bool]:
        if action == 0: return self._compress_and_merge(grid, axis=0, reverse=False)
        if action == 1: return self._compress_and_merge(grid, axis=1, reverse=True)
        if action == 2: return self._compress_and_merge(grid, axis=0, reverse=True)
        if action == 3: return self._compress_and_merge(grid, axis=1, reverse=False)
        raise ValueError(f"Unknown action {action}")

    def _compress_and_merge(self, grid: np.ndarray, axis: int, reverse: bool) -> Tuple[int, bool]:
        reward = 0
        moved_any = False
        size = grid.shape[axis]
        other = 1 - axis
        for fixed in range(grid.shape[other]):
            if axis == 0: line = grid[:, fixed]
            else: line = grid[fixed, :]
            if reverse: line = line[::-1]
            
            vals = line[line != 0].tolist()
            merged = []
            i = 0
            while i < len(vals):
                if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                    m = vals[i] * 2
                    reward += m
                    merged.append(m)
                    i += 2
                else:
                    merged.append(vals[i])
                    i += 1
            
            merged += [0] * (size - len(merged))
            new_line = np.array(merged, dtype=np.int64)
            if reverse: new_line = new_line[::-1]
            
            if axis == 0:
                before_segment = grid[:, fixed].copy()
                grid[:, fixed] = new_line
                if not np.array_equal(before_segment, new_line): moved_any = True
            else:
                before_segment = grid[fixed, :].copy()
                grid[fixed, :] = new_line
                if not np.array_equal(before_segment, new_line): moved_any = True
                    
        return reward, moved_any

if __name__ == "__main__":
    def demo(seed: int = 42, steps: int = 700):
        env = Game2048Env()
        obs, info = env.reset(seed=seed)
        print(f"Start Info: Score={info['score']}, RawReward={info['raw_reward']}")
        print(f"obs:{obs}")
        key_map = {
            'w': 0, # 上
            'd': 1, # 右
            's': 2, # 下
            'a': 3, # 左
            'q': -1 # 退出
        }
        
        for t in range(steps):
            valid_input = False
            a = 0
            while not valid_input:
                user_input = input(f"\n[Step {t+1}] 请输入动作 (w/a/s/d): ").strip().lower()
                
                if user_input in key_map:
                    a = key_map[user_input]
                    if a == -1:
                        print("手动退出游戏")
                        return
                    valid_input = True
                else:
                    print("输入无效，请重新输入 (w/a/s/d)")
            # 注意: 这里的 reward 是正则化后的
            obs, reward, done, info = env.step(int(a))
            # import pdb;pdb.set_trace()
            print(f"obs:{obs}")
            action_name = env.ACTION_LOOKUP[int(a)]
            print(f"\nStep {t+1}, Action: {a} ({action_name})")
            print(f"Log Reward: {reward:.4f} | Raw Reward: {info['raw_reward']} | Total Score: {info['score']}")
            if done:
                print("Game Over")
                break
    demo()