import numpy as np
from typing import Dict, List, Optional, Tuple
from gymnasium.utils import seeding

CoordDict = Dict[str, List[Tuple[int, int]]]


def _to_tuple_list(array: np.ndarray, index_origin: int = 0) -> List[Tuple[int, int]]:
    if array.size == 0:
        return []
    return [(int(r) + index_origin, int(c) + index_origin) for r, c in array]


def collect_entity_coordinates(desc: np.ndarray, player_pos: Tuple[int, int], index_origin: int = 0) -> CoordDict:
    """Collect coordinates for key FrozenLake entities."""

    coords: CoordDict = {}

    start_coords = _to_tuple_list(np.argwhere(desc == b"S"), index_origin)
    if start_coords:
        coords["start"] = start_coords

    goal_coords = _to_tuple_list(np.argwhere(desc == b"G"), index_origin)
    if goal_coords:
        coords["goal"] = goal_coords

    hole_coords = _to_tuple_list(np.argwhere(desc == b"H"), index_origin)
    if hole_coords:
        coords["holes"] = hole_coords

    player_row, player_col = player_pos
    player_tile = desc[player_row, player_col]
    player_coord = (player_row + index_origin, player_col + index_origin)

    if player_tile == b"G":
        coords["player_on_goal"] = [player_coord]
    elif player_tile == b"H":
        coords["player_in_hole"] = [player_coord]
    else:
        coords["player"] = [player_coord]

    return coords


def format_coordinate_render(entity_coords: CoordDict, board_shape: Tuple[int, int], index_origin: int = 0) -> str:
    """Format FrozenLake entities into a coordinate description."""

    rows, cols = board_shape
    origin_str = "zero-indexed" if index_origin == 0 else f"origin at {index_origin}"
    lines = [f"Board size: {rows} rows x {cols} cols ({origin_str})."]

    ordered_labels = [
        ("start", "Start"),
        ("goal", "Goal"),
        ("player", "Player"),
        ("player_on_goal", "Player on goal"),
        ("player_in_hole", "Player in hole"),
        ("holes", "Holes"),
    ]

    for key, label in ordered_labels:
        coords = entity_coords.get(key, [])
        if not coords:
            continue
        coord_str = ", ".join(f"({r}, {c})" for r, c in coords)
        lines.append(f"{label}: {coord_str}")

    return "\n".join(lines)

def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    start_r, start_c = np.where(np.array(board) == "S")
    frontier.append((start_r[0], start_c[0]))

    while frontier:
        r, c = frontier.pop()
        if (r, c) not in discovered:
            discovered.add((r, c))
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                r_new, c_new = r + dr, c + dc
                if 0 <= r_new < max_size and 0 <= c_new < max_size:
                    if board[r_new][c_new] == "G":
                        return True
                    if board[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    """
    Generates a random valid map with a path from start (S) to goal (G).
    Args:
        size: The size of the map.
        p: The probability of generating a hole (H).
        seed: The seed for the random number generator.
    Returns:
        A list of strings representing the map.
    """
    np_random, _ = seeding.np_random(seed)

    while True:
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        start_r, start_c = np_random.integers(size, size=2)
        goal_r, goal_c = np_random.integers(size, size=2)

        if (start_r, start_c) != (goal_r, goal_c):
            board[start_r][start_c], board[goal_r][goal_c] = "S", "G"
            if is_valid(board, size):
                break

    return ["".join(row) for row in board]
