"""
pitch_control.py
────────────────
Computes how much of the pitch each team "controls" on every frame using
Voronoi tessellation.  Each player owns the region of the pitch they are
closest to — summing those regions by team gives a continuous pitch-control
percentage that feeds directly into the Clutch Score model.

Method
──────
1. For each frame, collect smooth_x / smooth_y positions for all players.
2. Build a Voronoi diagram using scipy.spatial.Voronoi.
3. Clip each infinite/finite Voronoi cell to the pitch rectangle using Shapely.
4. Sum cell areas by team.  home_pct = home_area / total_area.
5. Export a per-frame CSV and an optional zone-grid heatmap CSV.

Dependencies
────────────
  pip install scipy shapely --break-system-packages

Input
─────
  --tracks   smoothed_tracks.csv
  --teams    team_assignments.csv
  --out-dir  Output directory

Output
──────
  <out-dir>/pitch_control.csv
      Columns: frame_id, home_pct, away_pct, contested_pct

  <out-dir>/control_heatmap.csv   (optional, --heatmap flag)
      Grid of average control values across all frames.

Usage
─────
  python3 src/goalx/ps1_cv/pitch_control.py \\
      --tracks   outputs/smoothed/smoothed_tracks.csv \\
      --teams    outputs/teams/team_assignments.csv \\
      --out-dir  outputs/pitch_control \\
      --pitch-w  1050 --pitch-h 680
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon, MultiPolygon
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

PITCH_W        = 1050
PITCH_H        = 680
HEATMAP_COLS   = 20     # grid resolution for optional heatmap
HEATMAP_ROWS   = 13
SAMPLE_EVERY   = 1      # process every Nth frame (set >1 to speed up)


# ─────────────────────────────────────────────────────────────────
#  Voronoi pitch control
# ─────────────────────────────────────────────────────────────────

def _voronoi_control(
    positions:   np.ndarray,   # (N, 2) — pitch-canvas pixel coords
    team_labels: np.ndarray,   # (N,) — 0=home, 1=away, -1=other
    pitch_w:     float,
    pitch_h:     float,
) -> tuple[float, float]:
    """
    Voronoi pitch control for two teams.

    Returns (home_fraction, away_fraction) summing to ≤1.0.
    The remainder (1 - home - away) is area owned by 'other' players (refs etc.)
    """
    if len(positions) < 4:
        # Not enough points for a meaningful Voronoi — assume equal split
        return 0.5, 0.5

    pitch_rect = Polygon([
        (0, 0), (pitch_w, 0), (pitch_w, pitch_h), (0, pitch_h)
    ])

    # Add 4 far-away mirror points to close all infinite Voronoi regions
    margin = max(pitch_w, pitch_h) * 4
    guards = np.array([
        [-margin, -margin],
        [pitch_w + margin, -margin],
        [pitch_w + margin, pitch_h + margin],
        [-margin, pitch_h + margin],
    ])
    all_pts = np.vstack([positions, guards])
    vor     = Voronoi(all_pts)

    team_areas = {0: 0.0, 1: 0.0}

    for i, pt_idx in enumerate(range(len(positions))):
        region_idx = vor.point_region[pt_idx]
        region     = vor.regions[region_idx]

        if not region or -1 in region:
            continue

        verts   = vor.vertices[region]
        cell    = Polygon(verts)
        clipped = cell.intersection(pitch_rect)

        if clipped.is_empty:
            continue

        area = clipped.area
        t    = int(team_labels[i])
        if t in team_areas:
            team_areas[t] += area

    total = pitch_rect.area
    return team_areas[0] / total, team_areas[1] / total


# ─────────────────────────────────────────────────────────────────
#  Heatmap aggregation
# ─────────────────────────────────────────────────────────────────

def _build_heatmap(
    positions:   np.ndarray,
    team_labels: np.ndarray,
    pitch_w:     float,
    pitch_h:     float,
    cols:        int,
    rows:        int,
) -> np.ndarray:
    """
    Returns a (rows, cols) grid where each cell value is:
       +1  if home player is closest to cell centre
       -1  if away player is closest
        0  if no players
    """
    cell_w = pitch_w / cols
    cell_h = pitch_h / rows
    grid   = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            dists = np.hypot(positions[:, 0] - cx, positions[:, 1] - cy)
            nearest = int(np.argmin(dists))
            t = int(team_labels[nearest])
            grid[r, c] = 1.0 if t == 0 else (-1.0 if t == 1 else 0.0)

    return grid


# ─────────────────────────────────────────────────────────────────
#  Main entry-point
# ─────────────────────────────────────────────────────────────────

def compute_pitch_control(
    tracks_csv: str,
    teams_csv:  str,
    out_dir:    str,
    pitch_w:    int = PITCH_W,
    pitch_h:    int = PITCH_H,
    heatmap:    bool = False,
    sample_every: int = SAMPLE_EVERY,
) -> pd.DataFrame:

    if not _DEPS_OK:
        raise ImportError(
            "pitch_control.py requires scipy and shapely.\n"
            "  pip install scipy shapely --break-system-packages"
        )

    tracks_csv = Path(tracks_csv)
    teams_csv  = Path(teams_csv)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX — Pitch Control (Voronoi)")
    print(f"  {'─' * 40}")

    tracks = pd.read_csv(tracks_csv)
    teams  = pd.read_csv(teams_csv)

    merged = tracks.merge(teams[["track_id", "team"]], on="track_id", how="left")
    merged["team"] = merged["team"].fillna("other")

    team_to_int = {"home": 0, "away": 1, "other": -1}
    merged["team_int"] = merged["team"].map(team_to_int).fillna(-1).astype(int)

    all_frames  = sorted(merged["frame_id"].unique())
    proc_frames = all_frames[::sample_every]
    print(f"  ✔  Processing {len(proc_frames)} / {len(all_frames)} frames  "
          f"(sample_every={sample_every})")

    records      = []
    heatmap_acc  = np.zeros((HEATMAP_ROWS, HEATMAP_COLS), dtype=np.float32)
    heatmap_cnt  = 0

    for fid in tqdm(proc_frames, desc="Computing Voronoi control"):
        fdata = merged[merged["frame_id"] == fid].dropna(subset=["smooth_x", "smooth_y"])

        if len(fdata) < 4:
            records.append({"frame_id": fid, "home_pct": 50.0,
                             "away_pct": 50.0, "contested_pct": 0.0})
            continue

        positions   = fdata[["smooth_x", "smooth_y"]].values.astype(float)
        team_labels = fdata["team_int"].values

        h_frac, a_frac = _voronoi_control(positions, team_labels, pitch_w, pitch_h)
        contested      = max(0.0, 1.0 - h_frac - a_frac)

        records.append({
            "frame_id":      fid,
            "home_pct":      round(h_frac * 100, 2),
            "away_pct":      round(a_frac * 100, 2),
            "contested_pct": round(contested * 100, 2),
        })

        if heatmap:
            g = _build_heatmap(positions, team_labels, pitch_w, pitch_h,
                               HEATMAP_COLS, HEATMAP_ROWS)
            heatmap_acc += g
            heatmap_cnt += 1

    df = pd.DataFrame(records)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n  📊 Average pitch control:")
    print(f"     Home : {df['home_pct'].mean():.1f}%")
    print(f"     Away : {df['away_pct'].mean():.1f}%")

    out_csv = out_dir / "pitch_control.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  💾 Saved → {out_csv}")

    if heatmap and heatmap_cnt > 0:
        avg_grid = heatmap_acc / heatmap_cnt
        rows_flat = []
        for r in range(HEATMAP_ROWS):
            for c in range(HEATMAP_COLS):
                rows_flat.append({"row": r, "col": c, "control": round(float(avg_grid[r, c]), 3)})
        hm_df  = pd.DataFrame(rows_flat)
        hm_csv = out_dir / "control_heatmap.csv"
        hm_df.to_csv(hm_csv, index=False)
        print(f"  💾 Heatmap → {hm_csv}")

    print(f"  ✅  Pitch control complete.")
    print(f"      Next step: clutch_score.py\n")
    return df


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-frame Voronoi pitch control percentages."
    )
    p.add_argument("--tracks",       required=True)
    p.add_argument("--teams",        required=True)
    p.add_argument("--out-dir",      default="outputs/pitch_control")
    p.add_argument("--pitch-w",      type=int, default=PITCH_W)
    p.add_argument("--pitch-h",      type=int, default=PITCH_H)
    p.add_argument("--heatmap",      action="store_true",
                   help="Also export a spatial heatmap grid CSV")
    p.add_argument("--sample-every", type=int, default=SAMPLE_EVERY,
                   help="Process every Nth frame to speed up (default: 1)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compute_pitch_control(
        tracks_csv   = args.tracks,
        teams_csv    = args.teams,
        out_dir      = args.out_dir,
        pitch_w      = args.pitch_w,
        pitch_h      = args.pitch_h,
        heatmap      = args.heatmap,
        sample_every = args.sample_every,
    )