import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import gymnasium as gym
import numpy as np
from ragen.env.base import BaseLanguageBasedEnv
# import pdb; pdb.set_trace()
import os
from ragen.env.sudoku.config import SudokuEnvConfig



# -----------------------------
# Sudoku Environment
# -----------------------------
class SudokuEnv(BaseLanguageBasedEnv, gym.Env):
    """
    语言交互版 Sudoku 环境：
    - reset(): 随机生成唯一解题目 + 保存完整解
    - step(action): action 支持 "row,col,value" 或 "place v at row R col C"
    - render(): 文本展示
    兼容 4x4 / 9x9；board_size 必须为完全平方数（b*b）。
    """

    metadata = {"render_modes": ["text", "text_with_coordinates"]}

    def __init__(self, config: Optional[SudokuEnvConfig] = None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else SudokuEnvConfig()

        self.board_size: int = self.config.board_size
        self.max_steps: int = self.config.max_steps
        self.difficulty: str = self.config.difficulty
        self.render_mode: str = os.environ.get('RENDER_MODE')

        # 状态
        self.board: Optional[np.ndarray] = None
        self.solution: Optional[np.ndarray] = None
        self.initial_board: Optional[np.ndarray] = None
        self.current_step: int = 0
        self.render_cache: Optional[str] = None

        # 校验
        b = self._box_size()  # 会 assert
        assert b * b == self.board_size

    # --------- Helpers: size / render ---------
    def _box_size(self) -> int:
        """Return sub-box size, e.g., 2 for 4x4, 3 for 9x9."""
        b = int(math.isqrt(self.board_size))
        assert b * b == self.board_size, "board_size must be a perfect square (e.g., 4, 9, 16)"
        return b

    def _render_board_sep(self) -> str:
        N = self.board_size
        b = self._box_size()
        line = "-" * (N * 2 + (b - 1) * 2 + 5)
        result = f"Sudoku Board ({N}x{N}):\n{line}\n"
        for i in range(N):
            if i % b == 0 and i != 0:
                result += line + "\n"
            row_str = "| "
            for j in range(N):
                if j % b == 0 and j != 0:
                    row_str += "| "
                cell = self.board[i, j]
                row_str += str(cell) if cell != 0 else "."
                row_str += " "
            row_str += "|\n"
            result += row_str
        result += line + "\n"
        result += f"Step: {self.current_step}/{self.max_steps}\n"
        result += f"Difficulty: {self.difficulty}\n"
        return result
    def _render_board(self) -> str:
        N = self.board_size
        b = self._box_size()
        # line = "-" * (N * 2 + (b - 1) * 2 + 5)
        # result = f"Sudoku Board ({N}x{N}):\n{line}\n"
        result=''
        for i in range(N):
            # if i % b == 0 and i != 0:
                # result += line + "\n"
            row_str = "| "
            for j in range(N):
                cell = self.board[i, j]
                row_str += str(cell) if cell != 0 else "."
                row_str += " "
            result += row_str
        # result += line + "\n"
        # result += f"Step: {self.current_step}/{self.max_steps}\n"
        # result += f"Difficulty: {self.difficulty}\n"
        return result

    # def find_coordinates(self, game_state: str, character: str):
    #     """Utility to locate the first occurrence of a character in the textual game state and return its (row, col)."""
    #     rows = game_state.split('\n')
    #     for i, row in enumerate(rows):
    #         if character in row:
    #             return (i, row.index(character))
    #     raise ValueError(f"Character {character} not found in game state {game_state}")
    def find_coordinates(self, game_state: str, character: str):
        """
        Return the (row, col) of the first occurrence of `character` in the game_state string.
        The game_state is a string representation of the board, with rows separated by '|'.
        """
        # Remove leading/trailing whitespace and split by '|'
        rows = [row.strip() for row in game_state.strip().split('|') if row.strip()]
        for i, row in enumerate(rows):
            # Split row into cells by whitespace
            cells = row.split()
            for j, cell in enumerate(cells):
                if cell == character:
                    return (i, j)
        raise ValueError(f"Character {character} not found in game state {game_state}")
    def find_all_coordinates(self, game_state: str, character: str):
        """
        Return a list of (row, col) for every occurrence of `character` in the game_state string.
        The game_state is a string representation of the board, with rows separated by '|'.
        """
        coords = []
        # Remove leading/trailing whitespace and split by '|'
        rows = [row.strip() for row in game_state.strip().split('|') if row.strip()]
        for i, row in enumerate(rows):
            # Split row into cells by whitespace
            cells = row.split()
            for j, cell in enumerate(cells):
                if cell == character:
                    coords.append((i, j))
        return coords
    # def find_all_coordinates(self, game_state: str, character: str):
    #     """Return a list of (row, col) for every occurrence of `character` in `game_state`."""
    #     rows = game_state.split('\n')
    #     coords = []
    #     for i, row in enumerate(rows):
    #         for j, ch in enumerate(row):
    #             if ch == character:
    #                 coords.append((i, j))
    #     return coords

    # --------- Parsing / Validity ---------
    def _parse_action(self, action: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        支持：
          - "row,col,value"
          - "row col value"
          - "place V at row R col C"
        返回 0-indexed (row, col, value)；失败返回 (None, None, None)
        """
        s = (action or "").lower().strip()
        try:
            if "row" in s and "col" in s:
                parts = s.replace(",", " ").split()
                row_idx = parts.index("row") + 1
                col_idx = parts.index("col") + 1
                # 找 value（既可能在开头 "place 3 at row 2 col 1"，也可能在末尾）
                value = None
                for i, p in enumerate(parts):
                    if p.isdigit() and i not in (row_idx, col_idx):
                        value = int(p)
                        break
                row = int(parts[row_idx]) - 1
                col = int(parts[col_idx]) - 1
                return row, col, value
            # "r,c,v" 或 "r c v"
            parts = s.replace(",", " ").split()
            if len(parts) >= 3 and all(p.lstrip("-").isdigit() for p in parts[:3]):
                r, c, v = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])
                return r, c, v
        except Exception:
            pass
        return None, None, None

    def _is_action_valid(self, row: Optional[int], col: Optional[int], value: Optional[int]) -> bool:
        # import pdb; pdb.set_trace()
        if row is None or col is None or value is None:
            return False
        N = self.board_size
        if not (0 <= row < N and 0 <= col < N):
            return False
        if not (1 <= value <= N):
            return False
        # 初始非空格不允许改
        if self.initial_board[row, col] != 0:
            return False
        return True

    def _is_valid_move(self, board: np.ndarray, row: int, col: int, num: int) -> bool:
        """是否满足数独约束（行/列/宫）。"""
        N = self.board_size
        if num in board[row, :]:
            return False
        if num in board[:, col]:
            return False
        b = self._box_size()
        sr, sc = (row // b) * b, (col // b) * b
        if num in board[sr:sr + b, sc:sc + b]:
            return False
        return True

    def _is_board_complete(self) -> bool:
        return np.all(self.board != 0)

    def _is_board_correct(self) -> bool:
        """是否为一个完整有效解。"""
        N = self.board_size
        need = set(range(1, N + 1))
        # rows
        for i in range(N):
            row = list(self.board[i])
            if 0 in row or set(row) != need:
                return False
        # cols
        for j in range(N):
            col = list(self.board[:, j])
            if 0 in col or set(col) != need:
                return False
        # boxes
        b = self._box_size()
        for bi in range(0, N, b):
            for bj in range(0, N, b):
                box = list(self.board[bi:bi + b, bj:bj + b].ravel())
                if 0 in box or set(box) != need:
                    return False
        return True

    # --------- Generator: full board + unique puzzle ---------
    def _generate_full_board(self, rng: random.Random) -> np.ndarray:
        """
        生成一张完整合法解（基底模板 + 打乱行/列 + 数字置换），O(1) 无回溯。
        对 N=b*b 有效（如 4、9）。
        """
        N = self.board_size
        b = self._box_size()
        base = np.fromfunction(lambda r, c: ((r * b + (r // b) + c) % N) + 1,
                               (N, N), dtype=int).astype(int)

        # 随机打乱 band/stack 以及 band/stack 内部
        bands = list(range(b))
        rng.shuffle(bands)
        rows: List[int] = []
        for g in bands:
            rr = list(range(g * b, (g + 1) * b))
            rng.shuffle(rr)
            rows += rr

        stacks = list(range(b))
        rng.shuffle(stacks)
        cols: List[int] = []
        for g in stacks:
            cc = list(range(g * b, (g + 1) * b))
            rng.shuffle(cc)
            cols += cc

        grid = base[rows][:, cols]

        # 数字标签随机置换 1..N
        nums = list(range(1, N + 1))
        rng.shuffle(nums)
        mapping = np.array(nums)
        solved = mapping[grid - 1]
        return solved

    def _count_solutions(self, board: np.ndarray, limit: int = 2) -> int:
        """回溯计数（到达 limit 立即停止），用于唯一解检测。"""
        N = self.board_size
        b = self._box_size()

        def find_min_choice_cell(bd: np.ndarray):
            best = None
            best_opts = None
            for i in range(N):
                for j in range(N):
                    if bd[i, j] == 0:
                        used = set(bd[i, :]) | set(bd[:, j]) | set(
                            bd[(i // b) * b:(i // b + 1) * b, (j // b) * b:(j // b + 1) * b].ravel()
                        )
                        opts = [v for v in range(1, N + 1) if v not in used]
                        if best is None or len(opts) < len(best_opts):
                            best, best_opts = (i, j), opts
                            if len(best_opts) <= 1:
                                return best, best_opts
            return best, best_opts

        bd = board.copy()
        count = 0

        def dfs():
            nonlocal count
            if count >= limit:
                return
            cell, opts = find_min_choice_cell(bd)
            if cell is None:
                count += 1
                return
            i, j = cell
            for v in opts:
                if self._is_valid_move(bd, i, j, v):
                    bd[i, j] = v
                    dfs()
                    if count >= limit:
                        break
                    bd[i, j] = 0

        dfs()
        return count

    def _has_unique_solution(self, puzzle: np.ndarray) -> bool:
        return self._count_solutions(puzzle, limit=2) == 1

    def _clues_by_difficulty(self) -> int:
        """
        返回“保留的非零格数”（越小越难）。
        4x4: easy 11, medium 9, hard 6
        9x9: easy 36, medium 30, hard 24
        其他 N: 用经验比例
        """
        N = self.board_size
        N2 = N * N
        if N == 4:
            table = {"easy": 10, "medium": 9, "hard": 6}
        elif N == 9:
            table = {"easy": 36, "medium": 30, "hard": 24}
        else:
            table = {"easy": int(0.50 * N2), "medium": int(0.40 * N2), "hard": int(0.33 * N2)}
        return table.get(self.difficulty, table["easy"])

    def _make_puzzle_from_solution(self, solved: np.ndarray, rng: random.Random, target_clues: int) -> np.ndarray:
        """
        从完整解挖空，保持唯一解；尽量达到 target_clues。
        """
        N = self.board_size
        puzzle = solved.copy()
        cells = [(i, j) for i in range(N) for j in range(N)]
        rng.shuffle(cells)

        filled = N * N
        for (i, j) in cells:
            if filled <= target_clues:
                break
            keep = puzzle[i, j]
            puzzle[i, j] = 0
            if self._has_unique_solution(puzzle):
                filled -= 1
            else:
                puzzle[i, j] = keep

        # 如果没达到目标，额外尝试几轮
        attempts = 0
        while filled > target_clues and attempts < 8:
            attempts += 1
            rng.shuffle(cells)
            for (i, j) in cells:
                if filled <= target_clues:
                    break
                if puzzle[i, j] != 0:
                    keep = puzzle[i, j]
                    puzzle[i, j] = 0
                    if self._has_unique_solution(puzzle):
                        filled -= 1
                    else:
                        puzzle[i, j] = keep
        return puzzle

    def _random_puzzle(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        solved = self._generate_full_board(rng)
        clues = self._clues_by_difficulty()
        puzzle = self._make_puzzle_from_solution(solved, rng, target_clues=clues)
        return puzzle, solved

    # --------- Public API: reset / step / render ---------
    def reset(self, seed: Optional[int] = None):
        """
        保持你之前的返回约定：只返回 obs（文本）。
        如需 gymnasium 正式签名 (obs, info)，可改为 return self.render_cache, {}
        """
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        puzzle, solved = self._random_puzzle(seed)
        self.initial_board = puzzle
        self.board = puzzle.copy()
        self.solution = solved
        self.current_step = 0
        self.render_cache = self._render_board()
        return self.render_cache

    def step(self, action: str):
        """
        返回 (obs, reward, done, info) 以兼容你现有管线。
        """
        self.current_step += 1
        # import pdb; pdb.set_trace()
        row, col, value = self._parse_action(action)

        # 非法动作（解析失败/越界/改初始格/值域非法）
        if not self._is_action_valid(row, col, value):
            reward = 0.0
            done = self.current_step >= self.max_steps
            info = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                
            }
            next_obs = self._render_board()
            
            self.render_cache = next_obs
            return next_obs, reward, done, info

        # 合法但违反数独规则
        if not self._is_valid_move(self.board, row, col, value):
            reward = 0.0
            done = self.current_step >= self.max_steps
            info = {
                "action_is_effective": False,
                "action_is_valid": True,
                "success": False,
                
            }
            next_obs = self._render_board()
            self.render_cache = next_obs
            return next_obs, reward, done, info

        # 执行动作
        self.board[row, col] = value

        # 终局检测
        if self._is_board_complete():
            if self._is_board_correct():
                reward = 1.0
                done = True
                info = {
                    "action_is_effective": True,
                    "action_is_valid": True,
                    "success": True,
                    
                }
                next_obs = self._render_board()
            else:
                reward = 0.0
                done = self.current_step >= self.max_steps
                info = {
                    "action_is_effective": True,
                    "action_is_valid": True,
                    "success": False,
                    
                }
                next_obs = self._render_board()
        else:
            reward = 0.0
            done = self.current_step >= self.max_steps
            info = {
                "action_is_effective": True,
                "action_is_valid": True,
                "success": False,
                
            }
            next_obs = self._render_board()

        self.render_cache = next_obs
        return next_obs, reward, done, info

    def render(self, mode: str = "text"):
        mode = os.environ.get('RENDER_MODE')
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            return self.render_cache
        elif render_mode == 'text_with_coordinates':
            # 获取基本的游戏状态
            game_state = self.render_cache
            
            # 对于数独，我们需要找到所有空位置（用"."表示）的坐标
            try:
                empty_positions = self.find_all_coordinates(game_state, '.')
                if empty_positions:
                    # 取第一个空位置作为示例
                    # first_empty, second_empty = empty_positions[0], empty_positions[1]
                    # first_empty, second_empty, third_empty, forth_empty = empty_positions[0], empty_positions[1], empty_positions[2], empty_positions[3]
                    # game_state += f"\nEmpty positions (.) are at: {empty_positions[:5]}{'...' if len(empty_positions) > 5 else ''}"
                    # game_state += f"\nFirst empty position is at ({first_empty[0]},{first_empty[1]})."
                    # game_state += f"\nFirst empty position is at ({first_empty[0]},{first_empty[1]})."
                    # game_state += f"\nSecond empty position is at ({second_empty[0]},{second_empty[1]})."
                    # game_state += f"\nEmpty positions to be filled are at ({first_empty[0]+1},{first_empty[1]+1}) and ({second_empty[0]+1},{second_empty[1]+1})."
                    if len(empty_positions) == 1:
                        first_empty = empty_positions[0]
                        game_state += f"\nEmpty position to be filled is at ({first_empty[0]+1},{first_empty[1]+1})."
                    elif len(empty_positions) > 1:
                        coords_str = ", ".join(f"({r+1},{c+1})" for r, c in empty_positions)
                        game_state += f"\nEmpty positions to be filled are at {coords_str}."
                    # game_state += f"\nEmpty positions to be filled are at ({first_empty[0]+1},{first_empty[1]+1}),({second_empty[0]+1},{second_empty[1]+1}),({third_empty[0]+1},{third_empty[1]+1}),({forth_empty[0]+1},{forth_empty[1]+1})."
    
                    # print(game_state)
                else:
                    game_state += "\nNo empty positions found (board might be complete)."
            except Exception as e:
                game_state += f"\nCould not find coordinates: {str(e)}"
            
            return game_state
        else:
            raise ValueError(f"Invalid mode: {render_mode}")

    def close(self):
        pass


if __name__ == "__main__":
    # 测试数独环境
    config = SudokuEnvConfig(board_size=4, max_steps=20, difficulty="easy")
    env = SudokuEnv(config)
    
    print("=== Testing Sudoku Environment ===")
    print("Initial state:")
    obs = env.reset(seed=42)
    print(obs)
    
    print("\n=== Testing text_with_coordinates mode ===")
    print(env.render(mode='text_with_coordinates'))
    
    print("\n=== Testing step ===")
    # 测试一个有效动作
    obs, reward, done, info = env.step("1,1,2")
    print(f"Action: 1,1,2")
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    print("\n=== Testing invalid action ===")
    obs, reward, done, info = env.step("invalid")
    print(f"Action: invalid")
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
