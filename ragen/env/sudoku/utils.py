import numpy as np
import random
from typing import Set, Tuple, List, Dict


def get_box_size(grid_size: int) -> int:
    """Get the box size for a given grid size."""
    return int(np.sqrt(grid_size))


def get_box_index(row: int, col: int, grid_size: int) -> Tuple[int, int]:
    """Get the box indices for a given cell."""
    box_size = get_box_size(grid_size)
    return row // box_size, col // box_size


def is_valid_placement(grid: np.ndarray, row: int, col: int, num: int) -> bool:
    """
    Check if placing a number at (row, col) is valid according to Sudoku rules.

    Returns:
        True if placement is valid, False otherwise
    """
    grid_size = grid.shape[0]
    box_size = get_box_size(grid_size)

    # Check if number already exists in row
    if num in grid[row, :]:
        return False

    # Check if number already exists in column
    if num in grid[:, col]:
        return False

    # Check if number already exists in box
    box_row, box_col = get_box_index(row, col, grid_size)
    box_start_row = box_row * box_size
    box_start_col = box_col * box_size
    if num in grid[box_start_row:box_start_row + box_size, box_start_col:box_start_col + box_size]:
        return False

    return True


def get_valid_numbers(grid: np.ndarray, row: int, col: int) -> Set[int]:
    """
    Get all valid numbers that can be placed at (row, col).

    Returns:
        Set of valid numbers (1 to grid_size)
    """
    if grid[row, col] != 0:
        return set()

    grid_size = grid.shape[0]
    all_numbers = set(range(1, grid_size + 1))

    # Remove numbers in same row
    all_numbers -= set(grid[row, :]) - {0}

    # Remove numbers in same column
    all_numbers -= set(grid[:, col]) - {0}

    # Remove numbers in same box
    box_size = get_box_size(grid_size)
    box_row, box_col = get_box_index(row, col, grid_size)
    box_start_row = box_row * box_size
    box_start_col = box_col * box_size
    box_numbers = grid[box_start_row:box_start_row + box_size, box_start_col:box_start_col + box_size]
    all_numbers -= set(box_numbers.flatten()) - {0}

    return all_numbers


def find_conflicts(grid: np.ndarray, initial_grid: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
    """
    Find all conflicts in the current grid state.

    Returns:
        Dictionary with 'row', 'col', 'box' keys mapping to lists of conflicting cells
    """
    conflicts = {'row': [], 'col': [], 'box': []}
    grid_size = grid.shape[0]
    box_size = get_box_size(grid_size)

    # Check row conflicts
    for i in range(grid_size):
        row = grid[i, :]
        for num in range(1, grid_size + 1):
            positions = np.where(row == num)[0]
            if len(positions) > 1:
                for pos in positions:
                    conflicts['row'].append((i, pos))

    # Check column conflicts
    for j in range(grid_size):
        col = grid[:, j]
        for num in range(1, grid_size + 1):
            positions = np.where(col == num)[0]
            if len(positions) > 1:
                for pos in positions:
                    conflicts['col'].append((pos, j))

    # Check box conflicts
    for box_row in range(box_size):
        for box_col in range(box_size):
            start_row = box_row * box_size
            start_col = box_col * box_size
            box = grid[start_row:start_row + box_size, start_col:start_col + box_size]
            for num in range(1, grid_size + 1):
                positions = np.argwhere(box == num)
                if len(positions) > 1:
                    for pos in positions:
                        conflicts['box'].append((start_row + pos[0], start_col + pos[1]))

    # Remove duplicates
    conflicts['row'] = list(set(conflicts['row']))
    conflicts['col'] = list(set(conflicts['col']))
    conflicts['box'] = list(set(conflicts['box']))

    return conflicts


def is_solved(grid: np.ndarray) -> bool:
    """Check if the Sudoku puzzle is completely solved."""
    grid_size = grid.shape[0]

    # Check if all cells are filled
    if np.any(grid == 0):
        return False

    # Check if there are any conflicts
    conflicts = find_conflicts(grid, grid)
    return len(conflicts['row']) == 0 and len(conflicts['col']) == 0 and len(conflicts['box']) == 0


def generate_sudoku_puzzle(grid_size: int = 9, difficulty: str = "easy", seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Sudoku puzzle with a unique solution.

    Args:
        grid_size: Size of the grid (4, 9, or 16)
        difficulty: Difficulty level ("easy", "medium", "hard")
        seed: Random seed for reproducibility

    Returns:
        Tuple of (puzzle_grid, solution_grid) where puzzle_grid has some cells filled
        and solution_grid is the complete solution
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create a solved grid first
    solution_grid = np.zeros((grid_size, grid_size), dtype=int)

    # Fill the grid using backtracking
    def fill_grid(grid):
        empty_cells = list(zip(*np.where(grid == 0)))
        if not empty_cells:
            return True

        row, col = empty_cells[0]
        numbers = list(range(1, grid_size + 1))
        random.shuffle(numbers)

        for num in numbers:
            if is_valid_placement(grid, row, col, num):
                grid[row, col] = num
                if fill_grid(grid):
                    return True
                grid[row, col] = 0

        return False

    fill_grid(solution_grid)

    # Create puzzle by removing numbers based on difficulty
    puzzle_grid = solution_grid.copy()
    cells_to_remove = {
        "easy": int(grid_size * grid_size * 0.4),    # Remove 40% of cells
        "medium": int(grid_size * grid_size * 0.5),  # Remove 50% of cells
        "hard": int(grid_size * grid_size * 0.6),    # Remove 60% of cells
    }

    num_to_remove = cells_to_remove.get(difficulty, cells_to_remove["easy"])
    cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(cells)

    for i in range(num_to_remove):
        row, col = cells[i]
        puzzle_grid[row, col] = 0

    return puzzle_grid, solution_grid


def format_grid_simple(grid: np.ndarray) -> str:
    """Format the grid as a simple string representation."""
    grid_size = grid.shape[0]
    box_size = get_box_size(grid_size)
    lines = []

    for i, row in enumerate(grid):
        if i > 0 and i % box_size == 0:
            lines.append("-" * (grid_size * 2 + box_size - 1))

        row_str = ""
        for j, val in enumerate(row):
            if j > 0 and j % box_size == 0:
                row_str += "| "
            row_str += (str(val) if val != 0 else ".") + " "
        lines.append(row_str.rstrip())

    return "\n".join(lines)


def format_grid_with_conflicts(grid: np.ndarray, initial_grid: np.ndarray,
                                 conflicts: Dict[str, List[Tuple[int, int]]]) -> str:
    """Format the grid with conflict markers."""
    grid_size = grid.shape[0]
    box_size = get_box_size(grid_size)
    lines = []

    # Collect all conflicting cells
    all_conflicts = set(conflicts['row'] + conflicts['col'] + conflicts['box'])

    for i, row in enumerate(grid):
        if i > 0 and i % box_size == 0:
            lines.append("-" * (grid_size * 3 + box_size - 1))

        row_str = ""
        for j, val in enumerate(row):
            if j > 0 and j % box_size == 0:
                row_str += "| "

            if val == 0:
                row_str += " . "
            elif initial_grid[i, j] != 0:
                # Initial cell (immutable)
                row_str += f"[{val}]"
            elif (i, j) in all_conflicts:
                # Conflict cell
                row_str += f"*{val}*"
            else:
                # User-placed cell (valid)
                row_str += f" {val} "

        lines.append(row_str.rstrip())

    return "\n".join(lines)
