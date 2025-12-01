#!/usr/bin/env python3
"""Simple test script for Sudoku environment."""

import sys
sys.path.insert(0, '/home/user/RAGEN')

import numpy as np
from ragen.env.sudoku.env import SudokuEnv
from ragen.env.sudoku.config import SudokuEnvConfig

def test_sudoku():
    """Test the Sudoku environment with enhanced feedback."""
    print("=" * 70)
    print("Testing Sudoku Environment with Enhanced Feedback")
    print("=" * 70)
    print()

    # Create environment with enhanced feedback
    config = SudokuEnvConfig(
        grid_size=9,
        difficulty="easy",
        render_format="with_feedback",
        show_conflicts=True,
        show_valid_numbers=True,
    )

    env = SudokuEnv(config)

    # Reset environment
    print("Resetting environment with seed=42...")
    obs = env.reset(seed=42)
    print(obs)
    print()

    # Find an empty cell and get its valid numbers
    empty_cells = list(zip(*np.where(env.current_grid == 0)))
    if empty_cells:
        row, col = empty_cells[0]
        print(f"\nTesting valid placement at first empty cell ({row+1},{col+1})...")

        # Get valid numbers for this cell
        from ragen.env.sudoku.utils import get_valid_numbers
        valid_nums = get_valid_numbers(env.current_grid, row, col)
        print(f"Valid numbers for ({row+1},{col+1}): {sorted(valid_nums)}")

        if valid_nums:
            # Test placing a valid number
            valid_num = sorted(valid_nums)[0]
            action = f"place {valid_num} at row {row+1} col {col+1}"
            print(f"\nAction: {action}")
            obs, reward, done, info = env.step(action)
            print(f"\nReward: {reward}")
            print(f"Info: {info}")
            print(f"\nObservation after action:")
            print(obs)
            print()

            # Test placing an INVALID number (that conflicts)
            if len(empty_cells) > 1:
                row2, col2 = empty_cells[1]
                # Try to place a number that's already in the row
                invalid_num = env.current_grid[row2, :][env.current_grid[row2, :] != 0]
                if len(invalid_num) > 0:
                    invalid_num = int(invalid_num[0])
                    action_invalid = f"place {invalid_num} at row {row2+1} col {col2+1}"
                    print(f"\n--- Testing INVALID placement ---")
                    print(f"Action: {action_invalid}")
                    obs, reward, done, info = env.step(action_invalid)
                    print(f"\nReward: {reward} (should be negative)")
                    print(f"Info: {info}")
                    print(f"\nObservation after invalid action:")
                    print(obs)

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)

if __name__ == "__main__":
    test_sudoku()
