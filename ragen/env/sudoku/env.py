import gymnasium as gym
import numpy as np
import re
from typing import Tuple, Dict, Any
from ragen.env.base import BaseLanguageBasedEnv
from .config import SudokuEnvConfig
from .utils import (
    generate_sudoku_puzzle,
    is_valid_placement,
    get_valid_numbers,
    find_conflicts,
    is_solved,
    format_grid_simple,
    format_grid_with_conflicts,
)


class SudokuEnv(BaseLanguageBasedEnv, gym.Env):
    """
    Sudoku environment with enhanced feedback for better exploration efficiency.

    Key features to address low exploration efficiency:
    1. Clear feedback on valid vs invalid moves
    2. Conflict detection and visualization
    3. Valid number suggestions for each cell
    4. Detailed action validation info
    """

    def __init__(self, config=None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else SudokuEnvConfig()
        self.grid_size = self.config.grid_size
        self.max_steps = self.config.max_steps

        # State variables
        self.current_grid = None
        self.initial_grid = None
        self.solution_grid = None
        self.num_steps = 0
        self.render_cache = None
        self.last_action_feedback = ""

        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'

    def reset(self, seed=None, mode=None):
        """Reset the environment with a new Sudoku puzzle."""
        gym.Env.reset(self, seed=seed)

        # Generate a new puzzle
        self.initial_grid, self.solution_grid = generate_sudoku_puzzle(
            grid_size=self.grid_size,
            difficulty=self.config.difficulty,
            seed=seed
        )
        self.current_grid = self.initial_grid.copy()
        self.num_steps = 0
        self.last_action_feedback = ""

        return self.render()

    def parse_action(self, action: str) -> Tuple[bool, int, int, int, str]:
        """
        Parse action string into (success, row, col, number, error_msg).

        Supported formats:
        - "place 5 at row 2 col 3"
        - "place 5 at (2,3)"
        - "place 5 at 2,3"
        - "5 at 2,3"
        - "(2,3,5)"
        - "2,3,5"

        Returns:
            Tuple of (success, row, col, number, error_message)
        """
        action = action.strip().lower()

        # Try different patterns
        patterns = [
            r'place\s+(\d+)\s+at\s+row\s+(\d+)\s+col\s+(\d+)',
            r'place\s+(\d+)\s+at\s+\((\d+),\s*(\d+)\)',
            r'place\s+(\d+)\s+at\s+(\d+),\s*(\d+)',
            r'(\d+)\s+at\s+(\d+),\s*(\d+)',
            r'\((\d+),\s*(\d+),\s*(\d+)\)',
            r'(\d+),\s*(\d+),\s*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, action)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    # Determine if first number is the value or row
                    # For "place NUM at ROW,COL", first is number
                    if 'place' in action or 'at' in action:
                        num, row, col = map(int, groups)
                    else:
                        # For "ROW,COL,NUM", assume positional format
                        row, col, num = map(int, groups)

                    # Convert to 0-indexed
                    row -= 1
                    col -= 1

                    # Validate ranges
                    if not (0 <= row < self.grid_size):
                        return False, -1, -1, -1, f"Row {row+1} is out of range (1-{self.grid_size})"
                    if not (0 <= col < self.grid_size):
                        return False, -1, -1, -1, f"Column {col+1} is out of range (1-{self.grid_size})"
                    if not (1 <= num <= self.grid_size):
                        return False, -1, -1, -1, f"Number {num} is out of range (1-{self.grid_size})"

                    return True, row, col, num, ""

        return False, -1, -1, -1, f"Could not parse action: '{action}'. Expected format: 'place 5 at row 2 col 3' or '2,3,5'"

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.num_steps += 1

        # Parse the action
        success, row, col, num, error_msg = self.parse_action(action)

        if not success:
            self.last_action_feedback = f"❌ Invalid action format: {error_msg}"
            reward = self.config.invalid_action_score
            info = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "error": error_msg
            }
            return self.render(), reward, False, info

        # Check if cell is modifiable (not part of initial puzzle)
        if self.initial_grid[row, col] != 0:
            self.last_action_feedback = f"❌ Cannot modify initial cell at ({row+1},{col+1})"
            reward = self.config.invalid_action_score
            info = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "error": "Cannot modify initial cells"
            }
            return self.render(), reward, False, info

        # Check if placement is valid according to Sudoku rules
        if not is_valid_placement(self.current_grid, row, col, num):
            # Get the specific conflict reason
            conflicts = []
            if num in self.current_grid[row, :]:
                conflicts.append(f"row {row+1}")
            if num in self.current_grid[:, col]:
                conflicts.append(f"column {col+1}")

            # Check box
            box_size = int(np.sqrt(self.grid_size))
            box_row = (row // box_size) * box_size
            box_col = (col // box_size) * box_size
            if num in self.current_grid[box_row:box_row+box_size, box_col:box_col+box_size]:
                conflicts.append(f"box ({box_row//box_size+1},{box_col//box_size+1})")

            conflict_str = ", ".join(conflicts)
            self.last_action_feedback = f"❌ Invalid placement: {num} conflicts with {conflict_str}"

            # Get valid numbers for this cell
            valid_nums = get_valid_numbers(self.current_grid, row, col)
            if valid_nums:
                self.last_action_feedback += f"\n   Valid numbers for ({row+1},{col+1}): {sorted(valid_nums)}"

            reward = self.config.invalid_action_score
            info = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "error": f"Number {num} conflicts with {conflict_str}",
                "valid_numbers": sorted(valid_nums)
            }
            return self.render(), reward, False, info

        # Place the number
        old_value = self.current_grid[row, col]
        self.current_grid[row, col] = num

        # Check if the placement is correct according to solution
        correct_placement = (num == self.solution_grid[row, col])

        if correct_placement:
            self.last_action_feedback = f"✓ Correct! Placed {num} at ({row+1},{col+1})"
            reward = self.config.correct_placement_score
        else:
            self.last_action_feedback = f"⚠ Placed {num} at ({row+1},{col+1}) - Valid but not optimal"
            reward = self.config.correct_placement_score * 0.5

        # Check if puzzle is solved
        solved = is_solved(self.current_grid)
        if solved:
            self.last_action_feedback += "\n🎉 Congratulations! Puzzle solved!"
            reward += self.config.completion_bonus

        # Check for max steps
        done = solved or (self.num_steps >= self.max_steps)

        info = {
            "action_is_effective": True,
            "action_is_valid": True,
            "success": solved,
            "correct_placement": correct_placement,
            "steps_remaining": self.max_steps - self.num_steps,
            "cells_filled": np.count_nonzero(self.current_grid),
            "cells_remaining": np.count_nonzero(self.current_grid == 0)
        }

        return self.render(), reward, done, info

    def render(self) -> str:
        """
        Render the current state with enhanced feedback.

        The render function provides:
        1. Current grid state with visual distinction between:
           - Initial cells [N]
           - User-placed valid cells N
           - Conflicting cells *N*
           - Empty cells .
        2. Last action feedback
        3. Current conflicts (if any)
        4. Valid numbers for empty cells (if enabled)
        """
        if self.config.render_format == "simple":
            return self._render_simple()
        elif self.config.render_format == "detailed":
            return self._render_detailed()
        else:  # "with_feedback"
            return self._render_with_feedback()

    def _render_simple(self) -> str:
        """Simple grid rendering."""
        return format_grid_simple(self.current_grid)

    def _render_detailed(self) -> str:
        """Detailed rendering with conflicts highlighted."""
        conflicts = find_conflicts(self.current_grid, self.initial_grid)
        grid_str = format_grid_with_conflicts(self.current_grid, self.initial_grid, conflicts)

        output = ["=" * 50]
        output.append("SUDOKU PUZZLE")
        output.append("=" * 50)
        output.append(grid_str)
        output.append("")
        output.append("Legend: [N]=initial cell, N=user-placed, *N*=conflict, .=empty")

        if self.last_action_feedback:
            output.append("")
            output.append(self.last_action_feedback)

        return "\n".join(output)

    def _render_with_feedback(self) -> str:
        """Full rendering with conflicts and valid numbers."""
        conflicts = find_conflicts(self.current_grid, self.initial_grid)
        grid_str = format_grid_with_conflicts(self.current_grid, self.initial_grid, conflicts)

        output = ["=" * 50]
        output.append("SUDOKU PUZZLE")
        output.append("=" * 50)
        output.append(grid_str)
        output.append("")
        output.append("Legend: [N]=initial cell, N=user-placed, *N*=conflict, .=empty")

        if self.last_action_feedback:
            output.append("")
            output.append(self.last_action_feedback)

        # Show conflicts if any
        if self.config.show_conflicts:
            all_conflicts = set(conflicts['row'] + conflicts['col'] + conflicts['box'])
            if all_conflicts:
                output.append("")
                output.append("⚠ CONFLICTS DETECTED:")
                for r, c in sorted(all_conflicts):
                    output.append(f"  - Cell ({r+1},{c+1}): {self.current_grid[r,c]}")

        # Show valid numbers for empty cells
        if self.config.show_valid_numbers:
            empty_cells = list(zip(*np.where(self.current_grid == 0)))
            if empty_cells and len(empty_cells) <= 10:  # Only show for first 10 empty cells
                output.append("")
                output.append("💡 VALID NUMBERS FOR EMPTY CELLS:")
                for row, col in empty_cells[:10]:
                    if self.initial_grid[row, col] == 0:  # Only show for non-initial cells
                        valid = get_valid_numbers(self.current_grid, row, col)
                        if valid:
                            output.append(f"  - ({row+1},{col+1}): {sorted(valid)}")

        # Show statistics
        cells_filled = np.count_nonzero(self.current_grid)
        cells_total = self.grid_size * self.grid_size
        initial_filled = np.count_nonzero(self.initial_grid)
        output.append("")
        output.append(f"Progress: {cells_filled}/{cells_total} cells filled ({initial_filled} initial, {cells_filled - initial_filled} placed)")
        output.append(f"Steps: {self.num_steps}/{self.max_steps}")

        return "\n".join(output)

    def close(self):
        """Clean up resources."""
        pass


if __name__ == "__main__":
    # Test the environment
    config = SudokuEnvConfig(
        grid_size=9,
        difficulty="easy",
        render_format="with_feedback"
    )
    env = SudokuEnv(config)

    print("Testing Sudoku Environment")
    print("=" * 50)

    obs = env.reset(seed=42)
    print(obs)
    print("\n")

    # Test some actions
    test_actions = [
        "place 5 at row 1 col 1",  # This might be valid or invalid depending on puzzle
        "1,2,3",  # Positional format
        "place 9 at (3,3)",  # Another format
    ]

    for action in test_actions:
        print(f"\nAction: {action}")
        obs, reward, done, info = env.step(action)
        print(obs)
        print(f"Reward: {reward}, Done: {done}")
        print(f"Info: {info}")

        if done:
            break
