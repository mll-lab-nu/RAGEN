#!/usr/bin/env python3
"""
Debug script to inspect rollout data structure.

Usage:
    python scripts/debug_rollout.py <path_to_pkl_file>
"""

import sys
import json
from pathlib import Path
from verl import DataProto


def inspect_rollout(pkl_path: str):
    """Inspect the structure of a rollout PKL file."""
    print(f"Loading: {pkl_path}")
    data = DataProto.load_from_disk(pkl_path)

    total = len(data)
    print(f"\nTotal trajectories: {total}")

    if total == 0:
        print("No trajectories found!")
        return

    # Inspect first trajectory
    print("\n" + "="*80)
    print("INSPECTING FIRST TRAJECTORY (index 0)")
    print("="*80)

    item = data[0]

    # Check meta_info
    print("\n--- meta_info ---")
    if item.meta_info:
        for k, v in item.meta_info.items():
            print(f"  {k}: {type(v)} = {v if not isinstance(v, (list, dict)) or len(str(v)) < 100 else f'{type(v)} (length {len(v)})'}")
    else:
        print("  (None)")

    # Check batch
    print("\n--- batch ---")
    if item.batch is not None:
        try:
            print(f"  Type: {type(item.batch)}")
            if hasattr(item.batch, 'keys'):
                for k in item.batch.keys():
                    try:
                        v = item.batch[k]
                        print(f"  {k}: {type(v)} shape={getattr(v, 'shape', 'N/A')}")
                    except Exception as e:
                        print(f"  {k}: Error accessing - {e}")
        except Exception as e:
            print(f"  Error inspecting batch: {e}")
    else:
        print("  (None)")

    # Check non_tensor_batch
    print("\n--- non_tensor_batch ---")
    ntb = item.non_tensor_batch
    if ntb:
        for k, v in ntb.items():
            if k == 'history':
                print(f"  history: list with {len(v)} entries")
                if len(v) > 0:
                    print(f"    First entry keys: {list(v[0].keys())}")
                    print(f"    First entry: {json.dumps(v[0], indent=4, default=str)[:500]}...")
            elif k == 'metrics':
                print(f"  metrics: {type(v)}")
                if isinstance(v, dict):
                    for mk, mv in v.items():
                        print(f"    {mk}: {mv}")
            else:
                val_str = str(v) if len(str(v)) < 100 else f"{type(v)} (length {len(v) if hasattr(v, '__len__') else 'N/A'})"
                print(f"  {k}: {type(v)} = {val_str}")
    else:
        print("  (None)")

    # Inspect a few more trajectories
    print("\n" + "="*80)
    print("SUMMARY OF ALL TRAJECTORIES")
    print("="*80)

    histories_found = 0
    non_empty_histories = 0

    for idx in range(min(5, total)):
        item = data[idx]
        ntb = item.non_tensor_batch or {}
        history = ntb.get('history', [])

        if history is not None:
            histories_found += 1
            if len(history) > 0:
                non_empty_histories += 1

        print(f"Traj {idx}: history={len(history) if history else 0} entries")

    print(f"\nOut of {total} trajectories:")
    print(f"  - {histories_found} have 'history' field")
    print(f"  - {non_empty_histories} have non-empty history")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_rollout.py <path_to_pkl_file>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    if not Path(pkl_path).exists():
        print(f"Error: File not found: {pkl_path}")
        sys.exit(1)

    inspect_rollout(pkl_path)
