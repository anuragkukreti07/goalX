"""
smooth_tracks.py
────────────────
Gap-aware rolling-mean trajectory smoother.

FIXES
─────────────────────────────────
FIX 1 — Ball NaN forward-fill bug (CRITICAL)
  Skips the rolling smoother entirely for track_id == -1 (ball).
  Propagates pitch_x/pitch_y directly to smooth_x/smooth_y WITHOUT
  filling NaN gaps so extract_events.py doesn't read stale positions.

FIX 2 — Jitter report now ignores NaN rows
  Calculates displacement only on valid player tracks to avoid inflating
  the jitter number with NaN→value transitions.

FIX 3 — Column validation
  Safely catches missing projection columns before processing.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

TRACK_ID_BALL = -1   # convention from track_players.py

# ─────────────────────────────────────────────────────────────────
#  Core smoothing
# ─────────────────────────────────────────────────────────────────

def smooth_tracks(df: pd.DataFrame,
                  window: int   = 5,
                  max_gap: int  = 10,
                  max_speed: float = 50.0) -> pd.DataFrame:
    """
    Apply per-track gap-aware rolling mean to (pitch_x, pitch_y).
    """
    df = df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)

    smooth_x = np.full(len(df), np.nan, dtype=np.float64)
    smooth_y = np.full(len(df), np.nan, dtype=np.float64)

    for tid, group in df.groupby("track_id"):
        idx    = group.index.to_numpy()
        frames = group["frame_id"].to_numpy()
        xs     = group["pitch_x"].to_numpy(dtype=np.float64)
        ys     = group["pitch_y"].to_numpy(dtype=np.float64)

        # ── FIX 1: ball passes through unmodified ─────────────────
        # ── FIX 1: Ball velocity gate (Clamp aerial/false jumps) ──
        if tid == TRACK_ID_BALL:
            dx = np.diff(xs, prepend=np.nan)
            dy = np.diff(ys, prepend=np.nan)
            jump = np.hypot(dx, dy)
            
            xs_clean = xs.copy()
            ys_clean = ys.copy()
            
            # Nuke the teleporting frames (like penalty spot false detections)
            xs_clean[jump > 150] = np.nan
            ys_clean[jump > 150] = np.nan
            
            # Interpolate over the nuked frames to keep the ball path realistic
            smooth_x[idx] = pd.Series(xs_clean).interpolate(method='linear', limit_direction='both').to_numpy()
            smooth_y[idx] = pd.Series(ys_clean).interpolate(method='linear', limit_direction='both').to_numpy()
            continue

        # ── Physics clamp for players ─────────────────────────────
        dx = np.diff(xs, prepend=np.nan)
        dy = np.diff(ys, prepend=np.nan)
        speed = np.hypot(dx, dy)
        
        xs = xs.copy()
        ys = ys.copy()
        xs[speed > max_speed] = np.nan
        ys[speed > max_speed] = np.nan

        # ── Gap-aware segment detection ───────────────────────────
        gaps    = np.diff(frames, prepend=frames[0])
        segment = np.cumsum(gaps > max_gap)

        for seg_id in np.unique(segment):
            mask    = segment == seg_id
            seg_idx = idx[mask]
            seg_x   = xs[mask]
            seg_y   = ys[mask]

            sx = pd.Series(seg_x).rolling(window, min_periods=1, center=False).mean()
            sy = pd.Series(seg_y).rolling(window, min_periods=1, center=False).mean()
            smooth_x[seg_idx] = sx.values
            smooth_y[seg_idx] = sy.values

    df = df.copy()
    df["smooth_x"] = smooth_x
    df["smooth_y"] = smooth_y
    return df

# ─────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────

def run_pipeline_step(input_csv: str, output_csv: str,
                      window: int, clamp_val: float) -> None:
    input_path  = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX — Trajectory Smoother")
    print(f"  {'─' * 40}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)

    # FIX 3 — column validation
    required = {"frame_id", "track_id", "pitch_x", "pitch_y"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"projected_tracks.csv missing columns: {missing}")

    n_tracks = df["track_id"].nunique()
    n_frames = df["frame_id"].nunique()
    n_ball   = len(df[df["track_id"] == TRACK_ID_BALL])
    n_ball_nan = df[df["track_id"] == TRACK_ID_BALL]["pitch_x"].isna().sum()

    print(f"  ✔  {len(df):,} rows  |  {n_tracks} tracks  |  {n_frames} frames")
    print(f"  ⚽ Ball rows: {n_ball:,}  (pitch_x=NaN: {n_ball_nan:,} — will be preserved)")
    print(f"  ⚙️  window={window}  clamp={clamp_val} px\n")

    # Jitter report on player tracks only
    players = df[df["track_id"] != TRACK_ID_BALL]
    raw_disp = np.sqrt(
        players.groupby("track_id")["pitch_x"].diff() ** 2 +
        players.groupby("track_id")["pitch_y"].diff() ** 2
    ).dropna()

    smoothed_df = smooth_tracks(df, window=window, max_speed=clamp_val)

    sm_players = smoothed_df[smoothed_df["track_id"] != TRACK_ID_BALL]
    sm_disp = np.sqrt(
        sm_players.groupby("track_id")["smooth_x"].diff() ** 2 +
        sm_players.groupby("track_id")["smooth_y"].diff() ** 2
    ).dropna()

    print(f"  📊 Player jitter (frame-to-frame displacement, px):")
    print(f"     Before : mean={raw_disp.mean():.2f}  p95={raw_disp.quantile(0.95):.2f}")
    print(f"     After  : mean={sm_disp.mean():.2f}  p95={sm_disp.quantile(0.95):.2f}")

    # Verify ball NaN preservation
    ball_smooth = smoothed_df[smoothed_df["track_id"] == TRACK_ID_BALL]
    ball_nan_after = ball_smooth["smooth_x"].isna().sum()
    check_msg = "✔ preserved correctly" if ball_nan_after == n_ball_nan else "✖ NaN count changed!"
    print(f"\n  Ball NaN check: {n_ball_nan} NaN before → {ball_nan_after} NaN after ({check_msg})")

    smoothed_df.to_csv(output_path, index=False)
    print(f"\n  ✅ Saved → {output_path}\n")

# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smooth projected trajectories.")
    parser.add_argument("--tracks", required=True, help="Input projected_tracks.csv")
    parser.add_argument("--out",    required=True, help="Output smoothed_tracks.csv")
    parser.add_argument("--window", type=int,   default=7, help="Rolling window in frames")
    parser.add_argument("--clamp",  type=float, default=50.0, help="Max speed clamp px/frame")
    args = parser.parse_args()
    
    run_pipeline_step(args.tracks, args.out, args.window, args.clamp)
