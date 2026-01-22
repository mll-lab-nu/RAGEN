#!/usr/bin/env python3
import sys
import numpy as np
import imageio
from gym_sokoban.envs.render_utils import room_to_rgb

# Mapping from characters to GymSokobanEnv internal IDs
# 0: Wall, 1: Floor, 2: Target, 3: Box on target, 4: Box, 5: Player, 6: Player on target
REVERSE_LOOKUP = {
    # RAGEN formats
    '#': 0,
    '_': 1,
    'O': 2,
    '√': 3,
    'X': 4,
    'P': 5,
    'S': 6,
    
    # Standard Sokoban formats (added for convenience)
    ' ': 1,
    '.': 2,
    '*': 3,
    '$': 4,
    '@': 5,
    '+': 6,
}

def text_to_room(text_grid):
    lines = [line.strip() for line in text_grid.strip().split('\n') if line.strip()]
    if not lines:
        raise ValueError("Empty grid")
    
    height = len(lines)
    width = max(len(line) for line in lines)
    
    room = np.zeros((height, width), dtype=np.uint8)
    room_structure = np.zeros((height, width), dtype=np.uint8)
    
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            val = REVERSE_LOOKUP.get(char, 1) # Default to floor if unknown
            room[i, j] = val
            
            # Structure identifies walls(0), floors(1), and targets(2)
            if val == 0:
                room_structure[i, j] = 0
            elif val in [2, 3, 6]:
                room_structure[i, j] = 2
            else:
                room_structure[i, j] = 1
                
    return room, room_structure

def render_to_file(text_grid, output_path):
    room, room_structure = text_to_room(text_grid)
    img_rgb = room_to_rgb(room, room_structure)
    imageio.imwrite(output_path, img_rgb)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_sokoban.py <text_grid_raw_or_file> [output_path]")
        sys.exit(1)
        
    input_str = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "sokoban_render.png"
    
    # Check if input is a file
    try:
        with open(input_str, 'r') as f:
            text_grid = f.read()
    except:
        text_grid = input_str
        
    render_to_file(text_grid, output_path)
