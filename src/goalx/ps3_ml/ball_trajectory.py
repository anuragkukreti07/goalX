"""
ball_trajectory.py  —  PS3 Step 2
────────────────────────────────────
Physics-based ball trajectory interpolation.

Problem
───────
SAHI detects the ball in ~60–80% of frames. The remaining frames are gaps
where the ball is occluded by players, blurred by motion, or outside the
camera view. Passing these raw detections to event extraction causes
kinematic spikes (instantaneous teleportation) that trigger false shot events.

Solution
─────────
Model the ball trajectory as piecewise polynomial segments:
  - Within each possession phase, fit a degree-2 polynomial to the detected
    x(t) and y(t) sequences (quadratic motion = constant-acceleration model).
  - Interpolate missing frames within each segment.
  - Reject interpolations that cross the pitch boundary or imply physically
    impossible speeds (> 40 m/s ≈ 144 km/h).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

PITCH_SCALE    = 10.0          # px/m (must match draw_pitch.py)
MAX_BALL_SPEED = 40.0          # m/s  (absolute physical maximum)
MAX_GAP        = 25            # frames: interpolate gaps shorter than this
SEGMENT_BREAK  = 50            # frames: gaps longer than this → new segment
POLY_DEGREE    = 2             # quadratic polynomial (constant acceleration)
MIN_SEGMENT_LEN = 6            # minimum detected frames to fit a polynomial
PITCH_W_PX     = 1050          # canvas width (px)
PITCH_H_PX     = 680           # canvas height (px)
TRACK_ID_BALL  = -1            # convention from track_players.py

_DARK_BG = "#0e1117"
_SURFACE = "#1a1d23"
_BORDER  = "#2e3140"
_TEXT    = "#c8c8d0"
_ACCENT  = "#5b8dee"
_AMBER   = "#f5a623"

plt.rcParams.update({
    "figure.facecolor": _DARK_BG,
    "axes.facecolor":   _SURFACE,
    "axes.edgecolor":   _BORDER,
    "axes.labelcolor":  _TEXT,
    "text.color":       _TEXT,
    "grid.color":       _BORDER,
    "xtick.color":      "#9090a0",
    "ytick.color":      "#9090a0",
})


# ─────────────────────────────────────────────────────────────────
#  Helper: speed in m/s between consecutive frames
# ─────────────────────────────────────────────────────────────────

def _speed_ms(x: np.ndarray, y: np.ndarray,
              frame_ids: np.ndarray, fps: float = 25.0) -> np.ndarray:
    """Returns (N-1,) array of speeds in m/s between consecutive detections."""
    dt = np.diff(frame_ids.astype(float)) / fps
    dx = np.diff(x) / PITCH_SCALE
    dy = np.diff(y) / PITCH_SCALE
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return dist / np.maximum(dt, 1e-6)


# ─────────────────────────────────────────────────────────────────
#  Segment splitter
# ─────────────────────────────────────────────────────────────────

def _split_into_segments(ball_df: pd.DataFrame,
                          fps: float = 25.0
                          ) -> list[pd.DataFrame]:
    """
    Split the raw ball detections into continuous segments.
    A segment boundary is placed wherever:
      - Frame gap > SEGMENT_BREAK
      - Instantaneous speed > MAX_BALL_SPEED m/s (bad detection)
    """
    if ball_df.empty:
        return []

    df   = ball_df.sort_values("frame_id").reset_index(drop=True)
    segs: list[pd.DataFrame] = []
    start = 0

    for i in range(1, len(df)):
        gap = int(df.at[i, "frame_id"]) - int(df.at[i - 1, "frame_id"])

        if gap > SEGMENT_BREAK:
            segs.append(df.iloc[start:i].copy())
            start = i
            continue

        # Speed check
        dt   = gap / fps
        dx   = (df.at[i, "pitch_x"] - df.at[i-1, "pitch_x"]) / PITCH_SCALE
        dy   = (df.at[i, "pitch_y"] - df.at[i-1, "pitch_y"]) / PITCH_SCALE
        spd  = np.sqrt(dx**2 + dy**2) / max(dt, 1e-6)
        if spd > MAX_BALL_SPEED:
            segs.append(df.iloc[start:i].copy())
            start = i

    segs.append(df.iloc[start:].copy())
    return [s for s in segs if len(s) >= 2]


# ─────────────────────────────────────────────────────────────────
#  Polynomial interpolator
# ─────────────────────────────────────────────────────────────────

def _fit_and_interpolate(seg: pd.DataFrame) -> pd.DataFrame:
    """
    Fit degree-2 polynomials to x(t) and y(t) for one segment.
    Interpolate all integer frame IDs within the segment's range,
    skipping gaps longer than MAX_GAP.
    """
    t   = seg["frame_id"].values.astype(float)
    x   = seg["pitch_x"].values.astype(float)
    y   = seg["pitch_y"].values.astype(float)

    t_min, t_max = int(t.min()), int(t.max())
    all_frames   = np.arange(t_min, t_max + 1, dtype=float)

    detected_set = set(t.astype(int))

    # Fit polynomials — fall back to degree 1 for very short segments
    deg = min(POLY_DEGREE, len(t) - 1)
    if deg < 1:
        return seg.assign(interp=False, conf=seg.get("conf", 1.0))[
            ["frame_id", "pitch_x", "pitch_y", "interp", "conf"]
        ]

    # Use numpy polyfit
    px = np.polyfit(t, x, deg)
    py = np.polyfit(t, y, deg)

    x_dense = np.polyval(px, all_frames)
    y_dense = np.polyval(py, all_frames)

    x_dense = np.clip(x_dense, 0, PITCH_W_PX)
    y_dense = np.clip(y_dense, 0, PITCH_H_PX)

    rows = []
    # Optimization: index by frame_id for fast lookup
    seg_indexed = seg.set_index("frame_id")
    
    for i, fi in enumerate(all_frames):
        fid = int(fi)
        is_detected = fid in detected_set

        if is_detected:
            row_actual = seg_indexed.loc[fid]
            # Handle potential duplicate frames safely
            if isinstance(row_actual, pd.DataFrame):
                row_actual = row_actual.iloc[0]
                
            rows.append({
                "frame_id": fid,
                "pitch_x":  float(row_actual["pitch_x"]),
                "pitch_y":  float(row_actual["pitch_y"]),
                "interp":   False,
                "conf":     float(row_actual.get("conf", 1.0)),
            })
        else:
            prev_dets = [d for d in detected_set if d < fid]
            next_dets = [d for d in detected_set if d > fid]

            if not prev_dets or not next_dets:
                continue   # edge extrapolation — skip

            gap = min(fid - max(prev_dets), min(next_dets) - fid)
            if gap > MAX_GAP:
                continue   # gap too large — don't fabricate

            if i > 0 and rows:
                prev = rows[-1]
                dt   = (fid - prev["frame_id"]) / 25.0
                dx   = (x_dense[i] - prev["pitch_x"]) / PITCH_SCALE
                dy   = (y_dense[i] - prev["pitch_y"]) / PITCH_SCALE
                spd  = np.sqrt(dx**2 + dy**2) / max(dt, 1e-6)
                if spd > MAX_BALL_SPEED:
                    continue   # physically impossible — skip

            rows.append({
                "frame_id": fid,
                "pitch_x":  float(x_dense[i]),
                "pitch_y":  float(y_dense[i]),
                "interp":   True,
                "conf":     0.5,   # lower confidence for interpolated
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────

def make_trajectory_figure(raw_df: pd.DataFrame,
                           interp_df: pd.DataFrame,
                           out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ball trajectory interpolation", fontsize=13,
                 color="#e8e8f0", weight="bold")

    # ── Top-down pitch view ────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#1a3a1a")
    ax.set_xlim(0, PITCH_W_PX);  ax.set_ylim(PITCH_H_PX, 0)

    ax.scatter(raw_df["pitch_x"], raw_df["pitch_y"],
               s=6, c=_ACCENT, alpha=0.6, zorder=3, label="Detected")

    interped = interp_df[interp_df["interp"]]
    ax.scatter(interped["pitch_x"], interped["pitch_y"],
               s=4, c=_AMBER, alpha=0.5, zorder=2, label="Interpolated")

    ax.set_title("Top-down trajectory view")
    ax.set_xlabel("pitch_x (px)");  ax.set_ylabel("pitch_y (px)")
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.2, zorder=0)

    # ── Frame timeline ─────────────────────────────────────────
    ax = axes[1]
    frames_all = interp_df.sort_values("frame_id")
    ax.plot(frames_all["frame_id"], frames_all["pitch_x"],
            color=_ACCENT, lw=1, alpha=0.8, label="x (detected)")
    ax.plot(frames_all[frames_all["interp"]]["frame_id"],
            frames_all[frames_all["interp"]]["pitch_x"],
            color=_AMBER, lw=0, marker=".", ms=3, alpha=0.8,
            label="x (interpolated)")
    ax.set_title("Ball x-coordinate over time")
    ax.set_xlabel("frame_id");  ax.set_ylabel("pitch_x (px)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Trajectory figure → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class BallTrajectoryInterpolator:
    def __init__(self, ball_csv: Path, tracks_csv: Path,
                 out_dir: Path, fps: float = 25.0):
        self.ball_csv   = ball_csv
        self.tracks_csv = tracks_csv
        self.out_dir    = out_dir
        self.fps        = fps

    def _load_ball(self) -> pd.DataFrame:
        df = pd.read_csv(self.ball_csv)

        if "frame_id" not in df.columns and "frame" in df.columns:
            df = df.rename(columns={"frame": "frame_id"})

        if "track_id" in df.columns:
            ball = df[df["track_id"] == TRACK_ID_BALL].copy()
        elif "class_id" in df.columns:
            ball = df[df["class_id"] == 32].copy()
        else:
            raise ValueError("Cannot identify ball rows — "
                             "need 'track_id' == -1 or 'class_id' == 32")

        if "pitch_x" not in ball.columns:
            raise ValueError("❌ Error: Input CSV must contain 'pitch_x' and 'pitch_y'. "
                             "You provided an image-space CSV (x1, y1). Please pass "
                             "'outputs/projected_tracks.csv' or 'outputs/smoothed_tracks.csv' "
                             "to the --ball argument instead.")

        ball = ball.sort_values("frame_id").reset_index(drop=True)
        print(f"  Ball detections loaded: {len(ball):,} rows "
              f"across {ball['frame_id'].nunique():,} frames")
        return ball

    def run(self) -> None:
        print(f"\n  goalX PS3 — Ball Trajectory Interpolator")
        print(f"  {'─'*46}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        raw_ball = self._load_ball()

        segments = _split_into_segments(raw_ball, self.fps)
        print(f"  Detected {len(segments)} continuous segments.")

        all_interp: list[pd.DataFrame] = []
        report_lines = [
            "Ball trajectory interpolation report",
            "=" * 50, "",
            f"Source: {self.ball_csv}",
            f"Total raw detections: {len(raw_ball):,}",
            f"Segments: {len(segments)}",
            "",
        ]

        for i, seg in enumerate(segments, 1):
            interp = _fit_and_interpolate(seg)
            n_real  = int((~interp["interp"]).sum())
            n_interp = int(interp["interp"].sum())
            coverage = f"{n_real}/{n_real + n_interp}"
            frames   = f"{int(seg['frame_id'].min())}–{int(seg['frame_id'].max())}"

            report_lines.append(
                f"  Segment {i:3d}:  frames {frames:>14}  "
                f"  detected={n_real:4d}  interpolated={n_interp:4d}  "
                f"({coverage})"
            )
            all_interp.append(interp)

        # FIX: Guard against empty lists if no valid segments exist
        if not all_interp:
            print("  ⚠️  No valid ball segments to interpolate. Exiting cleanly.")
            return

        result_df = (pd.concat(all_interp, ignore_index=True)
                     .sort_values("frame_id")
                     .drop_duplicates("frame_id"))

        n_raw_total    = int((~result_df["interp"]).sum())
        n_interp_total = int(result_df["interp"].sum())
        fill_rate = n_interp_total / max(len(result_df), 1)

        report_lines += [
            "",
            f"Total output frames : {len(result_df):,}",
            f"Real detections     : {n_raw_total:,}",
            f"Interpolated        : {n_interp_total:,}  ({fill_rate:.1%})",
        ]

        out_csv = self.out_dir / "interpolated_ball.csv"
        result_df.to_csv(out_csv, index=False)
        print(f"\n  ✔  Interpolated CSV → {out_csv}")
        print(f"  Frames: {len(result_df):,}  "
              f"(real={n_raw_total:,}  interp={n_interp_total:,}  "
              f"fill={fill_rate:.1%})")

        rpt_path = self.out_dir / "trajectory_report.txt"
        rpt_path.write_text("\n".join(report_lines))
        print(f"  ✔  Report → {rpt_path}")

        make_trajectory_figure(raw_ball, result_df,
                               self.out_dir / "trajectory_plot.png")

        print(f"\n  ✅  Done.\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Physics-based ball trajectory interpolation (goalX PS3)."
    )
    p.add_argument("--ball",    default="outputs/smoothed_tracks.csv",
                   help="Tracks CSV containing pitch coordinates for the ball")
    p.add_argument("--tracks",  default="",
                   help="Projected tracks CSV (optional, for context)")
    p.add_argument("--out-dir", default="outputs/ball_trajectory")
    p.add_argument("--fps",     type=float, default=25.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    BallTrajectoryInterpolator(
        ball_csv   = Path(args.ball),
        tracks_csv = Path(args.tracks) if args.tracks else Path(""),
        out_dir    = Path(args.out_dir),
        fps        = args.fps,
    ).run()