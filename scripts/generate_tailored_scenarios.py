#!/usr/bin/env python3
import os
import sys
import numpy as np
import imageio
from gym_sokoban.envs.render_utils import room_to_rgb

REVERSE_LOOKUP = {
    '#': 0, '_': 1, 'O': 2, '√': 3, 'X': 4, 'P': 5, 'S': 6,
    ' ': 1, '.': 2, '*': 3, '$': 4, '@': 5, '+': 6,
}

def text_to_room(text_grid):
    lines = [line.strip() for line in text_grid.strip().split('\n') if line.strip()]
    if not lines: raise ValueError("Empty grid")
    height = len(lines)
    width = max(len(line) for line in lines)
    room = np.zeros((height, width), dtype=np.uint8)
    room_structure = np.zeros((height, width), dtype=np.uint8)
    for i, line in enumerate(lines):
        line = line.ljust(width, '#') # Pad with walls if needed
        for j, char in enumerate(line):
            val = REVERSE_LOOKUP.get(char, 1)
            room[i, j] = val
            if val == 0: room_structure[i, j] = 0
            elif val in [2, 3, 6]: room_structure[i, j] = 2
            else: room_structure[i, j] = 1
    return room, room_structure

def save_scenario(name, grid):
    output_dir = "scenarios_tailored"
    os.makedirs(output_dir, exist_ok=True)
    room, struct = text_to_room(grid)
    img = room_to_rgb(room, struct)
    
    img_path = os.path.join(output_dir, f"{name}.png")
    imageio.imwrite(img_path, img)
    
    txt_path = os.path.join(output_dir, f"{name}.txt")
    with open(txt_path, 'w') as f:
        f.write(grid.strip())
    
    print(f"Generated {name}")

# --- DEFINING CORRECTED SCENARIOS ---

# 1. Template Collapse 1: Easy to get stuck, but solvable.
# A simple corridor where pushing the wrong way is fatal.
# Solution: P goes down, around, pushes Left.
# If P pushes Right, X hits wall. Stuck.
template_collapse_1 = """
######
#O_X_#
#_##_#
#_P__#
######
######
"""

# 2. Template Collapse 2: Hard to get stuck (Open).
# Wide open room.
template_collapse_2 = """
######
#____#
#_P__#
#_X__#
#_O__#
######
"""

# 3. Echo Trap 1: Just a generic solvable level.
echo_trap_1 = """
######
#_OX_#
#_#__#
#_P__#
######
######
"""

# 4. Echo Trap 2: Different generic level.
echo_trap_2 = """
######
#____#
#XP__#
#O####
#____#
######
"""

# 5. Strategic Compression 1: "Go up twice and left once".
# P(4,3) -> U(3,3) -> U(2,3) -> L(2,2) [Pushes Box to 2,1 Target]
stragetic_compression_1 = """
######
#____#
#OX__#
#____#
#__P_#
######
"""

# 6. Strategic Compression 2: "Go down twice then done".
# P(1,3). Box(2,3). Target(4,3).
# D -> Push(3,3). D -> Push(4,3) [Target]. Done.
stragetic_compression_2 = """
######
#__P_#
#__X_#
#____#
#__O_#
######
"""
# Note: '.' is empty floor but visually nice to remember it's path. Treated as floor.

# 7. True Diverse Reasoning 1: "Move up agent twice...", "I see two boxes..."
# Two boxes clearly visible. P needs to move up.
true_diverse_1 = """
######
#_O_O#
#_X_X#
#____#
#_P__#
######
"""

# 8. True Diverse Reasoning 2: "I see box is on top of...", "Move the box down then..."
true_diverse_2 = """
######
#__P_#
#__X_#
#__O_#
#____#
######
"""

SCENARIOS = {
    "TemplateCollapse1": template_collapse_1,
    "TemplateCollapse2": template_collapse_2,
    "EchoTrap1": echo_trap_1,
    "EchoTrap2": echo_trap_2,
    "StrategicCompression1": stragetic_compression_1,
    "StrategicCompression2": stragetic_compression_2,
    "TrueDiverseReasoning1": true_diverse_1,
    "TrueDiverseReasoning2": true_diverse_2,
}

if __name__ == "__main__":
    for name, grid in SCENARIOS.items():
        save_scenario(name, grid)
