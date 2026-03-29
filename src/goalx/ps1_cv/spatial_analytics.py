"""
spatial_analytics.py
────────────────────
Computes kinematic statistics (distance, speed, acceleration) from
smoothed pitch-space trajectories.

Uses smooth_x / smooth_y from smoothed_tracks.csv for noise-free
displacement — do NOT feed raw projected_tracks.csv here (the jitter
produces artificially inflated speed values).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

FPS              = 25
PIXELS_PER_METRE = 10.0    # must match draw_pitch.py SCALE
MAX_SPEED_MS     = 12.0    # 43 km/h — Usain Bolt ceiling; cap teleportation
MIN_FRAMES       = 50      # ignore tracks shorter than 2 seconds


def run_analytics(
    tracks_csv: str,
    out_csv:    str,
    fps:        float = FPS,
    px_per_m:   float = PIXELS_PER_METRE,
    min_frames: int   = MIN_FRAMES,
    top_n:      int   = 10,
) -> pd.DataFrame:

    tracks_csv = Path(tracks_csv)
    out_csv    = Path(out_csv)

    print(f"\n  goalX — Spatial Analytics")
    print(f"  {'─' * 40}")

    if not tracks_csv.exists():
        raise FileNotFoundError(
            f"smoothed_tracks.csv not found: {tracks_csv}\n"
            f"Run smooth_tracks.py first."
        )

    df = pd.read_csv(tracks_csv)
    
    # Adapt to whatever coordinates are available
    coord_col_x = "smooth_x" if "smooth_x" in df.columns else "pitch_x"
    coord_col_y = "smooth_y" if "smooth_y" in df.columns else "pitch_y"

    print(f"  ✔  Loaded {len(df)} rows from {tracks_csv.name}")
    print(f"  ⚙️  Using coords: {coord_col_x}/{coord_col_y}  |  "
          f"{px_per_m} px/m  |  {fps} fps")

    df = df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)

    # ── Per-frame kinematics ──────────────────────────────────────
    df["dx_px"] = df.groupby("track_id")[coord_col_x].diff().fillna(0)
    df["dy_px"] = df.groupby("track_id")[coord_col_y].diff().fillna(0)
    df["dist_px"] = np.hypot(df["dx_px"], df["dy_px"])
    df["dist_m"]  = df["dist_px"] / px_per_m

    # Cap teleportation  (> max_speed = impossible displacement in one frame)
    max_dist_per_frame = MAX_SPEED_MS / fps        # metres per frame
    df.loc[df["dist_m"] > max_dist_per_frame, "dist_m"] = 0.0

    df["speed_ms"]  = df["dist_m"] * fps
    df["speed_kmh"] = df["speed_ms"] * 3.6

    # Acceleration (Δspeed / Δt, capped at physical max ~10 m/s²)
    df["accel_ms2"] = (
        df.groupby("track_id")["speed_ms"]
          .diff()
          .fillna(0)
          .abs() * fps
    )
    df.loc[df["accel_ms2"] > 10.0, "accel_ms2"] = 0.0

    # ── Per-track aggregation ─────────────────────────────────────
    
    # Calculate total dist, top speed, avg speed, peak accel, frames active
    stats = df.groupby("track_id").agg(
        total_dist_m   = ("dist_m",   "sum"),
        top_speed_kmh  = ("speed_kmh","max"),
        avg_speed_kmh  = ("speed_kmh","mean"),
        peak_accel_ms2 = ("accel_ms2","max"),
        frames_active  = ("frame_id", "count"),
    ).reset_index()

    stats["time_active_s"] = stats["frames_active"] / fps

    # ── Filter short tracks ───────────────────────────────────────
    active = stats[stats["frames_active"] >= min_frames].copy()
    active = active.sort_values("total_dist_m", ascending=False).reset_index(drop=True)

    # ── Console summary ───────────────────────────────────────────
    print(f"\n  📊 {len(active)} active tracks  (≥{min_frames} frames each)")
    print(f"  {'─' * 55}")
    print(f"  {'Track':>6}  {'Dist (m)':>9}  {'Top km/h':>9}  "
          f"{'Avg km/h':>9}  {'Time (s)':>9}")
    print(f"  {'─' * 55}")
    for _, row in active.head(top_n).iterrows():
        print(f"  {int(row['track_id']):>6}  "
              f"{row['total_dist_m']:>9.1f}  "
              f"{row['top_speed_kmh']:>9.1f}  "
              f"{row['avg_speed_kmh']:>9.1f}  "
              f"{row['time_active_s']:>9.1f}")

    # ── Save ──────────────────────────────────────────────────────
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    active.to_csv(out_csv, index=False)
    print(f"\n  💾 Saved → {out_csv}")
    print(f"  ✅  Analytics complete.\n")

    return active


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute kinematic stats from smoothed trajectories."
    )
    # Aligned with orchestrator expectations
    p.add_argument("--tracks",      required=True, help="smoothed_tracks.csv")
    p.add_argument("--out",         required=True, help="Output CSV path")
    
    p.add_argument("--fps",         type=float, default=FPS)
    p.add_argument("--px-per-m",    type=float, default=PIXELS_PER_METRE,
                   help=f"Pixels per metre in the pitch map  (default: {PIXELS_PER_METRE})")
    p.add_argument("--min-frames",  type=int,   default=MIN_FRAMES,
                   help="Minimum frames for a track to be included")
    p.add_argument("--top",         type=int,   default=10,
                   help="Number of players to print in leaderboard")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_analytics(
        tracks_csv = args.tracks,
        out_csv    = args.out,
        fps        = args.fps,
        px_per_m   = args.px_per_m,
        min_frames = args.min_frames,
        top_n      = args.top,
    )