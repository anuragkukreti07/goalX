"""
build_interpolated_homography.py
─────────────────────────────────
Builds a per-frame homography CSV by using manually-picked seed .npz files
as hard geometric anchors and linearly interpolating H matrices between them.

This bypasses the Sharma HOG-matching pipeline entirely for downstream
tracking, using only the verified seed calibrations as ground truth.
Linear interpolation of H elements is valid for slow broadcast camera pans.
"""

import argparse
import glob
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_seeds(seed_pattern: str) -> dict[int, np.ndarray]:
    """
    Load all seed .npz files matching the pattern and extract (frame_id → H).
    Frame ID is parsed from the filename — supports formats like:
        homography_data_193_seed1.npz       → frame from inside npz
        homography_data_193_seed_mid_100.npz → frame 100 from filename
        gt_000300.npz                        → frame 300 from filename
    """
    seeds = {}
    paths = sorted(glob.glob(seed_pattern))
    
    if not paths:
        raise FileNotFoundError(f"No seed files found matching: {seed_pattern}")
    
    for path in paths:
        stem = Path(path).stem
        
        # Try to extract frame number from filename
        # Matches patterns like: seed_mid_100, 000300, seed_100
        frame_match = re.search(r'(\d{3,6})(?:\.npz)?$', stem)
        
        data = np.load(path, allow_pickle=True)
        H = data['H'].astype(np.float32)
        
        if frame_match:
            frame_id = int(frame_match.group(1))
        else:
            # For seeds named like homography_data_193_seed1.npz,
            # we don't know the frame — skip with warning
            print(f"  ⚠  Cannot determine frame_id from {stem} — skipping")
            continue
        
        cond = np.linalg.cond(H)
        if cond > 1_000_000:
            print(f"  ⚠  Seed at frame {frame_id} has cond={cond:.0f} — skipping (poor calibration)")
            continue
        
        seeds[frame_id] = H
        print(f"  ✔  Loaded seed frame {frame_id:04d}  (cond={cond:.0f}, inliers={int(np.sum(data.get('status', [0])))})")
    
    return seeds


def interpolate_homographies(seeds: dict[int, np.ndarray], 
                              n_frames: int,
                              first_frame: int = 1) -> dict[int, np.ndarray]:
    """
    For every frame in [first_frame, first_frame + n_frames - 1],
    compute H by linearly interpolating between the two nearest seed anchors.
    
    Frames before the first seed or after the last seed are clamped
    to the nearest anchor (no extrapolation — extrapolation of H is unreliable).
    """
    sorted_frames = sorted(seeds.keys())
    
    if len(sorted_frames) < 2:
        raise ValueError("Need at least 2 seed anchors to interpolate. Add more seeds.")
    
    result = {}
    frame_ids = range(first_frame, first_frame + n_frames)
    
    for fid in frame_ids:
        if fid in seeds:
            # Exact anchor — use it directly
            result[fid] = seeds[fid]
            continue
        
        # Find bracketing anchors
        prev_anchors = [f for f in sorted_frames if f <= fid]
        next_anchors = [f for f in sorted_frames if f > fid]
        
        if not prev_anchors:
            # Before first anchor — clamp
            result[fid] = seeds[sorted_frames[0]]
        elif not next_anchors:
            # After last anchor — clamp
            result[fid] = seeds[sorted_frames[-1]]
        else:
            f0 = prev_anchors[-1]
            f1 = next_anchors[0]
            H0 = seeds[f0].astype(np.float64)
            H1 = seeds[f1].astype(np.float64)
            
            # Linear interpolation parameter t ∈ [0, 1]
            t = (fid - f0) / (f1 - f0)
            H_interp = (1 - t) * H0 + t * H1
            
            # Renormalize so H[2,2] = 1 (standard convention)
            if H_interp[2, 2] != 0:
                H_interp /= H_interp[2, 2]
            
            result[fid] = H_interp.astype(np.float32)
    
    return result


def evaluate_coverage(seeds: dict, n_frames: int, first_frame: int = 1):
    """Print a coverage report showing max gap between anchors."""
    sorted_anchors = sorted(seeds.keys())
    last_frame = first_frame + n_frames - 1
    
    print(f"\n  Coverage report:")
    print(f"  {'─' * 50}")
    
    # Gap before first anchor
    if sorted_anchors[0] > first_frame:
        gap = sorted_anchors[0] - first_frame
        print(f"  ⚠  Clamped region  : frames {first_frame}–{sorted_anchors[0]-1} ({gap} frames)")
    
    # Gaps between anchors
    max_gap = 0
    for i in range(len(sorted_anchors) - 1):
        gap = sorted_anchors[i+1] - sorted_anchors[i]
        max_gap = max(max_gap, gap)
        status = "✔" if gap <= 75 else "⚠" if gap <= 150 else "✖"
        print(f"  {status}  Interpolated gap : frames {sorted_anchors[i]}–{sorted_anchors[i+1]} ({gap} frames)")
    
    # Gap after last anchor
    if sorted_anchors[-1] < last_frame:
        gap = last_frame - sorted_anchors[-1]
        print(f"  ⚠  Clamped region  : frames {sorted_anchors[-1]+1}–{last_frame} ({gap} frames)")
    
    print(f"\n  Total anchors    : {len(sorted_anchors)}")
    print(f"  Max interp gap   : {max_gap} frames  ({'✔ good' if max_gap <= 75 else '⚠ add more seeds for gaps > 75'})")


def save_csv(h_dict: dict[int, np.ndarray], out_path: str):
    """Save H matrices to CSV in the same format as homographies_mrf.csv."""
    rows = []
    for fid in sorted(h_dict.keys()):
        H = h_dict[fid].ravel()
        rows.append({
            'frame_id': fid,
            'h00': H[0], 'h01': H[1], 'h02': H[2],
            'h10': H[3], 'h11': H[4], 'h12': H[5],
            'h20': H[6], 'h21': H[7], 'h22': H[8],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run(seed_pattern, n_frames, out_csv, first_frame=1):
    print("\n  goalX — Anchor-Based Homography Interpolator")
    print("  " + "─" * 40)
    
    seeds = load_seeds(seed_pattern)
    print(f"\n  Loaded {len(seeds)} valid anchor frames.")
    
    evaluate_coverage(seeds, n_frames, first_frame)
    
    print(f"\n  Interpolating {n_frames} frames ...")
    h_dict = interpolate_homographies(seeds, n_frames, first_frame)
    
    save_csv(h_dict, out_csv)
    print(f"\n  ✅  Saved {len(h_dict)} per-frame H matrices → {out_csv}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build per-frame homography CSV from seed anchors via linear interpolation."
    )
    p.add_argument("--seeds", required=True,
                   help="Glob pattern for seed .npz files, e.g. 'data/homography_data_193_seed_mid_*.npz'")
    p.add_argument("--n-frames", type=int, required=True,
                   help="Total number of frames in the sequence")
    p.add_argument("--out", required=True,
                   help="Output CSV path")
    p.add_argument("--first-frame", type=int, default=1,
                   help="First frame number in sequence (default: 1)")
    args = p.parse_args()
    
    run(args.seeds, args.n_frames, args.out, args.first_frame)
