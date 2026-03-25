"""
smooth_tracks.py
────────────────
Applies per-track rolling-mean smoothing to (pitch_x, pitch_y) coordinates
produced by project_tracks.py, removing Kalman-filter jitter and
single-frame "teleportation" artefacts before event logic runs.

Input
─────
  --projected   projected_tracks.csv  (from project_tracks.py)
                Required columns: frame_id, track_id, pitch_x, pitch_y

  --window      Rolling-window size in frames  (default: 5)
                Larger → smoother but more lag.  5-7 works well for 25 fps.

  --max-gap     Maximum frame gap allowed inside one track before the window
                is reset.  Prevents smoothing across ID-switch boundaries.
                (default: 10)

Output
──────
  <out-dir>/smoothed_tracks.csv
      All original columns preserved, plus:
        smooth_x  — smoothed pitch_x
        smooth_y  — smoothed pitch_y

Usage
─────
  python3 src/goalx/ps1_cv/smooth_tracks.py \
      --projected  outputs/projected/projected_tracks.csv \
      --out-dir    outputs/smoothed \
      --window     5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --- CONFIG ---
MAX_SPEED_PX = 50.0  # Maximum allowed displacement (px) per frame. Clamps teleportation glitches.


# ─────────────────────────────────────────────────────────────────
#  Core smoothing logic
# ─────────────────────────────────────────────────────────────────

def smooth_tracks(
    df: pd.DataFrame,
    window: int = 5,
    max_gap: int = 10,
) -> pd.DataFrame:
    """
    Apply a per-track rolling mean over (pitch_x, pitch_y).

    Parameters
    ──────────
    df       : projected_tracks DataFrame
    window   : rolling window size (frames)
    max_gap  : if two consecutive frames for a track_id are > max_gap apart,
               the rolling window resets (treats it as a new track segment).

    Returns
    ───────
    DataFrame with added columns: smooth_x, smooth_y
    """
    df = df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)

    # --- PHYSICS CLAMP: Kill Teleportation Glitches ---
    # Calculate frame-to-frame displacement (raw speed)
    df["dx"] = df.groupby("track_id")["pitch_x"].diff()
    df["dy"] = df.groupby("track_id")["pitch_y"].diff()
    df["raw_speed"] = np.hypot(df["dx"], df["dy"])
    
    # If speed > MAX_SPEED_PX, set to NaN so the rolling smoother ignores it 
    # and doesn't drag the valid track across the screen.
    df.loc[df["raw_speed"] > MAX_SPEED_PX, ["pitch_x", "pitch_y"]] = np.nan

    smooth_x = np.full(len(df), np.nan, dtype=np.float64)
    smooth_y = np.full(len(df), np.nan, dtype=np.float64)

    for tid, group in df.groupby("track_id"):
        idx    = group.index.to_numpy()
        frames = group["frame_id"].to_numpy()
        xs     = group["pitch_x"].to_numpy(dtype=np.float64)
        ys     = group["pitch_y"].to_numpy(dtype=np.float64)

        # Build a "segment_id" that resets the window whenever there is a
        # large frame gap (e.g. player left the camera view and came back
        # under a different track ID that later got merged).
        gaps    = np.diff(frames, prepend=frames[0])
        segment = np.cumsum(gaps > max_gap)

        for seg_id in np.unique(segment):
            mask    = segment == seg_id
            seg_idx = idx[mask]
            seg_x   = xs[mask]
            seg_y   = ys[mask]

            # min_periods=1 → no NaN at the start of short segments
            # Pandas correctly ignores the NaNs we injected via the physics clamp
            sx = pd.Series(seg_x).rolling(window, min_periods=1, center=False).mean()
            sy = pd.Series(seg_y).rolling(window, min_periods=1, center=False).mean()
            smooth_x[seg_idx] = sx.values
            smooth_y[seg_idx] = sy.values

    df = df.copy()
    df["smooth_x"] = smooth_x
    df["smooth_y"] = smooth_y
    
    # Cleanup temporary columns used for clamping
    df = df.drop(columns=["dx", "dy", "raw_speed"])
    
    return df


# ─────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────

def run(
    projected_csv: str,
    out_dir:       str,
    window:        int = 5,
    max_gap:       int = 10,
) -> pd.DataFrame:

    projected_csv = Path(projected_csv)
    out_dir       = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not projected_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {projected_csv}")

    print(f"\n  goalX — Trajectory Smoother")
    print(f"  {'─' * 40}")

    df = pd.read_csv(projected_csv)
    required = {"frame_id", "track_id", "pitch_x", "pitch_y"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    n_tracks = df["track_id"].nunique()
    n_frames = df["frame_id"].nunique()
    print(f"  ✔  Loaded {len(df)} rows  |  {n_tracks} tracks  |  {n_frames} frames")
    print(f"  ⚙️  Rolling window = {window} frames  |  max_gap = {max_gap} frames")

    # Keep a copy of the raw dataframe to accurately calculate "Before" jitter
    raw_df = df.copy()
    
    smoothed_df = smooth_tracks(df, window=window, max_gap=max_gap)

    # ── Jitter report ──────────────────────────────────────────────────────
    raw_disp = np.sqrt(
        raw_df.groupby("track_id")["pitch_x"].diff() ** 2 +
        raw_df.groupby("track_id")["pitch_y"].diff() ** 2
    )
    sm_disp = np.sqrt(
        smoothed_df.groupby("track_id")["smooth_x"].diff() ** 2 +
        smoothed_df.groupby("track_id")["smooth_y"].diff() ** 2
    )
    print(f"\n  📊 Jitter Report (frame-to-frame displacement, px):")
    print(f"     Before:  mean={raw_disp.mean():.2f}  "
          f"p95={raw_disp.quantile(0.95):.2f}  max={raw_disp.max():.2f}")
    print(f"     After:   mean={sm_disp.mean():.2f}  "
          f"p95={sm_disp.quantile(0.95):.2f}  max={sm_disp.max():.2f}")

    out_csv = out_dir / "smoothed_tracks.csv"
    smoothed_df.to_csv(out_csv, index=False)
    print(f"\n  💾 Saved → {out_csv}")
    print(f"\n  ✅  Smoothing complete.")
    print(f"      Next step: extract_events.py --tracks {out_csv}\n")

    return smoothed_df


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Smooth projected player trajectories via rolling mean."
    )
    p.add_argument("--projected", required=True,
                   help="projected_tracks.csv from project_tracks.py")
    p.add_argument("--out-dir",   default="outputs/smoothed",
                   help="Output directory  (default: outputs/smoothed)")
    p.add_argument("--window",    type=int, default=5,
                   help="Rolling window size in frames  (default: 5)")
    p.add_argument("--max-gap",   type=int, default=10,
                   help="Max frame gap before window resets  (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        projected_csv = args.projected,
        out_dir       = args.out_dir,
        window        = args.window,
        max_gap       = args.max_gap,
    )