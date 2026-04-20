"""
compare_seeds.py
────────────────
Evaluates the new (Challenger) homography seeds against the old (Champion) seeds.
Keeps whichever matrix has the lower condition number.
"""

import numpy as np
import shutil
from pathlib import Path

def get_condition_number(npz_path):
    """Loads an .npz file, finds the 3x3 homography matrix, and returns its condition number."""
    if not npz_path.exists():
        return float('inf')
    try:
        data = np.load(npz_path)
        # Find the 3x3 matrix in the saved arrays
        for key in data.files:
            if data[key].shape == (3, 3):
                return np.linalg.cond(data[key])
    except Exception as e:
        print(f"  [Error loading {npz_path.name}: {e}]")
    return float('inf')

def run_tournament(backup_dir: str, active_dir: str):
    backup_path = Path(backup_dir)
    active_path = Path(active_dir)
    
    print("\n  goalX — Seed Homography Tournament")
    print("  " + "─" * 40)

    # Find all backup seeds
    old_seeds = list(backup_path.glob("homography_data_193_seed_mid_*.npz"))
    
    if not old_seeds:
        print("  ⚠ No backup seeds found. Ensure you moved the old files to the backup directory.")
        return

    for old_file in sorted(old_seeds):
        new_file = active_path / old_file.name
        
        old_cond = get_condition_number(old_file)
        new_cond = get_condition_number(new_file)
        
        frame_name = old_file.stem.split('_')[-1]
        
        print(f"  Frame {frame_name}:")
        print(f"     Old (Champion)   : {old_cond:,.0f}")
        print(f"     New (Challenger) : {new_cond:,.0f}")
        
        if new_cond < old_cond:
            print("     🏆 Challenger wins! Keeping the new seed.\n")
            # The new file is already in the active directory, so we do nothing.
        else:
            print("     🛡️ Champion defends! Restoring the old seed.\n")
            # Overwrite the bad new file with the stable old file
            shutil.copy(old_file, new_file)

if __name__ == "__main__":
    run_tournament(backup_dir="data/backup_seeds", active_dir="data")
