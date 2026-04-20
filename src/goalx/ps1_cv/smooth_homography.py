"""
smooth_homography.py
────────────────────
Applies weighted temporal smoothing directly to homography matrix elements.
Uses Sharma match distances as inverse-confidence weights so high-confidence
frames act as anchors and uncertain frames get blended with neighbours.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def smooth_homographies(
    h_csv: str,
    distances_file: str,
    out_csv: str,
    window: int = 15,
    min_dist_weight: float = 0.1,
):
    print("\n  goalX — Homography Matrix Smoother")
    print("  " + "─" * 40)

    h_df = pd.read_csv(h_csv).sort_values('frame_id').reset_index(drop=True)
    frame_col = 'frame_id'
    h_cols = [c for c in h_df.columns if c != frame_col][:9]

    # Load match distances directly from the .npy file
    dist_path = Path(distances_file)
    if dist_path.suffix == '.npy':
        distances = np.load(dist_path)
    else:
        print("  ⚠  Did not receive a .npy file — using uniform weights")
        distances = np.ones(len(h_df))

    # Convert distances to weights: lower distance = higher weight
    max_dist = np.percentile(distances, 90)  # cap at 90th percentile
    distances_capped = np.clip(distances, 0, max_dist)
    weights = 1.0 - (distances_capped / max_dist)
    weights = np.clip(weights, min_dist_weight, 1.0)

    H_matrices = h_df[h_cols].values.astype(np.float64)
    n_frames = len(H_matrices)
    H_smooth = np.zeros_like(H_matrices)

    half_w = window // 2
    
    for i in range(n_frames):
        lo = max(0, i - half_w)
        hi = min(n_frames, i + half_w + 1)
        
        window_H = H_matrices[lo:hi]
        window_w = weights[lo:hi]
        
        # Normalize weights in window
        w_sum = window_w.sum()
        if w_sum < 1e-9:
            H_smooth[i] = H_matrices[i]
            continue
        
        w_norm = window_w / w_sum
        
        # Weighted average of each H element
        H_avg = np.zeros(9)
        for j, (h_row, w) in enumerate(zip(window_H, w_norm)):
            H_avg += h_row * w
        
        # Renormalize so H[2,2] = 1
        if abs(H_avg[8]) > 1e-10:
            H_avg = H_avg / H_avg[8]
        
        H_smooth[i] = H_avg

    # Write back to CSV
    result_df = h_df.copy()
    result_df[h_cols] = H_smooth
    result_df.to_csv(out_csv, index=False)

    # Report quality improvement
    orig_conds = [np.linalg.cond(H_matrices[i].reshape(3,3)) for i in range(n_frames)]
    smooth_conds = [np.linalg.cond(H_smooth[i].reshape(3,3)) for i in range(n_frames)]
    
    print(f"  Condition numbers:")
    print(f"     Before  — mean: {np.mean(orig_conds):.0f}  median: {np.median(orig_conds):.0f}")
    print(f"     After   — mean: {np.mean(smooth_conds):.0f}  median: {np.median(smooth_conds):.0f}")
    print(f"  Frames usable (cond < 100k):")
    print(f"     Before: {sum(c < 100000 for c in orig_conds)}/750")
    print(f"     After:  {sum(c < 100000 for c in smooth_conds)}/750")
    print(f"  ✅  Saved → {out_csv}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h-csv", required=True, help="Sharma MRF homography CSV")
    p.add_argument("--distances", required=True, help="match_distances.npy file from HOG matcher")
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=15)
    args = p.parse_args()
    smooth_homographies(args.h_csv, args.distances, args.out, args.window)
