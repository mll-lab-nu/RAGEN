from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SudokuEnvConfig:
    """Configuration for Sudoku environment with enhanced feedback."""
    grid_size: int = 9  # Standard 9x9 Sudoku
    max_steps: int = 81  # Maximum number of steps (one per cell)
    difficulty: str = "easy"  # Difficulty level: easy, medium, hard
    render_mode: str = "text"
    show_conflicts: bool = True  # Show row/column/box conflicts in render
    show_valid_numbers: bool = True  # Show valid numbers for each empty cell
    show_candidates: bool = False  # Show all candidate numbers for empty cells
    render_format: str = "detailed"  # "simple", "detailed", "with_feedback"

    # Scoring
    correct_placement_score: float = 1.0
    invalid_action_score: float = -0.1  # Penalty for invalid placements
    completion_bonus: float = 10.0  # Bonus for solving the puzzle

    def __post_init__(self):
        if self.grid_size not in {4, 9, 16}:
            raise ValueError(f"Unsupported grid_size: {self.grid_size}. Must be 4, 9, or 16.")
        if self.render_format not in {"simple", "detailed", "with_feedback"}:
            raise ValueError(f"Unsupported render_format: {self.render_format}")
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"Unsupported difficulty: {self.difficulty}")
