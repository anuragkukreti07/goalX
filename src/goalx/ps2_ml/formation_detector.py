# """
# formation_detector.py
# ─────────────────────
# Detects the tactical formation (e.g. "4-4-2", "4-3-3") of each team on
# every frame by clustering player pitch positions along the depth axis.

# Method
# ──────
# 1. For each frame, select players belonging to a team (from team_assignments.csv).
# 2. Project their smooth_x or smooth_y coordinate onto the "depth" axis
#    (the axis that runs from goal to goal — configurable via --depth-axis).
# 3. Apply K-Means with K=3 (3 lines: defence, midfield, attack).
# 4. Count players per line → formation string  e.g. "4-3-3".
# 5. Apply a rolling majority-vote window to smooth out single-frame noise.

# Input
# ─────
#   --tracks   smoothed_tracks.csv  (from smooth_tracks.py)
#   --teams    team_assignments.csv  (from team_classifier.py)
#   --out-dir  Output directory

# Output
# ──────
#   <out-dir>/formations.csv
#       Columns: frame_id, home_formation, away_formation

# Usage
# ─────
#   python3 src/goalx/ps1_cv/formation_detector.py \\
#       --tracks    outputs/smoothed/smoothed_tracks.csv \\
#       --teams     outputs/teams/team_assignments.csv \\
#       --out-dir   outputs/formations
# """

# import argparse
# from collections import Counter
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from tqdm import tqdm


# # ─────────────────────────────────────────────────────────────────
# #  CONFIG
# # ─────────────────────────────────────────────────────────────────

# N_LINES          = 3     # formation lines: defence / midfield / attack
# MIN_PLAYERS      = 5     # skip frame if fewer than this many players visible
# SMOOTHING_WINDOW = 25    # rolling majority-vote window (frames)
# DEPTH_AXIS       = "x"   # "x" or "y" — the axis from goal to goal on the pitch


# # ─────────────────────────────────────────────────────────────────
# #  Formation detection for one team in one frame
# # ─────────────────────────────────────────────────────────────────

# def _formation_for_frame(
#     depth_positions: np.ndarray,
#     n_lines:         int = N_LINES,
# ) -> str:
#     """
#     Given 1-D depth positions of one team's players, cluster into n_lines
#     and return a formation string like "4-3-3" (back line first).

#     If there are fewer players than lines, returns "unknown".
#     """
#     n = len(depth_positions)
#     if n < n_lines:
#         return "unknown"

#     k  = min(n_lines, n)
#     km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
#     labels = km.fit_predict(depth_positions.reshape(-1, 1))

#     # Sort cluster IDs by their centroid depth (ascending = defensive)
#     centroid_order = np.argsort(km.cluster_centers_.flatten())
#     counts = [int(np.sum(labels == c)) for c in centroid_order]

#     return "-".join(map(str, counts))


# # ─────────────────────────────────────────────────────────────────
# #  Rolling majority-vote smoother
# # ─────────────────────────────────────────────────────────────────

# def _majority_smooth(series: pd.Series, window: int) -> pd.Series:
#     """Replace each value with the most common value in a rolling window (string-safe)."""
#     vals = series.tolist()
#     out = []
#     half = window // 2
#     n = len(vals)
    
#     for i in range(n):
#         start = max(0, i - half)
#         end = min(n, i + half + 1)
#         window_slice = vals[start:end]
        
#         counter = Counter(v for v in window_slice if v != "unknown")
#         if counter:
#             out.append(counter.most_common(1)[0][0])
#         else:
#             out.append("unknown")
            
#     return pd.Series(out, index=series.index)


# # ─────────────────────────────────────────────────────────────────
# #  Main entry-point
# # ─────────────────────────────────────────────────────────────────

# def detect_formations(
#     tracks_csv: str,
#     teams_csv:  str,
#     out_dir:    str,
#     depth_axis: str = DEPTH_AXIS,
#     n_lines:    int = N_LINES,
#     window:     int = SMOOTHING_WINDOW,
# ) -> pd.DataFrame:

#     tracks_csv = Path(tracks_csv)
#     teams_csv  = Path(teams_csv)
#     out_dir    = Path(out_dir)
#     out_dir.parent.mkdir(parents=True, exist_ok=True)

#     print(f"\n  goalX — Formation Detector")
#     print(f"  {'─' * 40}")

#     # ── Load ─────────────────────────────────────────────────────
#     tracks = pd.read_csv(tracks_csv)
#     teams  = pd.read_csv(teams_csv)

#     if depth_axis not in ("x", "y"):
#         raise ValueError("--depth-axis must be 'x' or 'y'")
#     depth_col = f"smooth_{depth_axis}"

#     required = {"frame_id", "track_id", depth_col}
#     if missing := required - set(tracks.columns):
#         raise ValueError(f"smoothed_tracks.csv missing columns: {missing}")

#     # Merge team labels onto tracks
#     merged = tracks.merge(teams[["track_id", "team"]], on="track_id", how="left")
#     merged["team"] = merged["team"].fillna("unknown")

#     home_ids  = set(teams.loc[teams["team"] == "home",  "track_id"])
#     away_ids  = set(teams.loc[teams["team"] == "away",  "track_id"])
#     n_frames  = merged["frame_id"].nunique()

#     print(f"  ✔  {n_frames} frames  |  home tracks: {len(home_ids)}  |  away tracks: {len(away_ids)}")
#     print(f"  ⚙️  depth_axis={depth_axis}  n_lines={n_lines}  smooth_window={window}")

#     # ── Per-frame detection ───────────────────────────────────────
#     records = []
#     for fid in tqdm(sorted(merged["frame_id"].unique()), desc="Detecting formations"):
#         fdata = merged[merged["frame_id"] == fid]

#         home_pos = fdata.loc[fdata["track_id"].isin(home_ids), depth_col].dropna().values
#         away_pos = fdata.loc[fdata["track_id"].isin(away_ids), depth_col].dropna().values

#         home_form = _formation_for_frame(home_pos, n_lines) if len(home_pos) >= MIN_PLAYERS else "unknown"
#         away_form = _formation_for_frame(away_pos, n_lines) if len(away_pos) >= MIN_PLAYERS else "unknown"

#         records.append({"frame_id": fid, "home_formation": home_form, "away_formation": away_form})

#     df = pd.DataFrame(records)

#     # ── Temporal smoothing ────────────────────────────────────────
#     df["home_formation"] = _majority_smooth(df["home_formation"], window)
#     df["away_formation"] = _majority_smooth(df["away_formation"], window)

#     # ── Summary ───────────────────────────────────────────────────
#     print(f"\n  📊 Most common formations detected:")
#     for col, label in [("home_formation", "Home"), ("away_formation", "Away")]:
#         top = df[df[col] != "unknown"][col].value_counts().head(3)
#         if not top.empty:
#             top_str = "  |  ".join(f"{k} ({v}f)" for k, v in top.items())
#             print(f"     {label:<5}: {top_str}")

#     out_csv = out_dir / "formations.csv"
#     df.to_csv(out_csv, index=False)
#     print(f"\n  💾 Saved → {out_csv}")
#     print(f"  ✅  Formation detection complete.")
#     print(f"      Next step: pitch_control.py\n")
#     return df


# # ─────────────────────────────────────────────────────────────────
# #  CLI
# # ─────────────────────────────────────────────────────────────────

# def _parse_args():
#     p = argparse.ArgumentParser(
#         description="Detect team formations from smoothed pitch positions."
#     )
#     p.add_argument("--tracks",      required=True,
#                    help="smoothed_tracks.csv from smooth_tracks.py")
#     p.add_argument("--teams",       required=True,
#                    help="team_assignments.csv from team_classifier.py")
#     p.add_argument("--out-dir",     default="outputs/formations")
#     p.add_argument("--depth-axis",  default=DEPTH_AXIS, choices=["x", "y"],
#                    help="Pitch axis running from goal to goal  (default: x)")
#     p.add_argument("--n-lines",     type=int, default=N_LINES,
#                    help=f"Number of formation lines  (default: {N_LINES})")
#     p.add_argument("--window",      type=int, default=SMOOTHING_WINDOW,
#                    help=f"Temporal smoothing window in frames  (default: {SMOOTHING_WINDOW})")
#     return p.parse_args()


# # if __name__ == "__main__":
# #     args = _parse_args()
# #     detect_formations(
# #         tracks_csv = args.tracks,
# #         teams_csv  = args.teams,
# #         out_dir    = args.out_dir,
# #         depth_axis = args.depth_axis,
# #         n_lines    = args.n_lines,
# #         window     = args.window,
# #     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="goalX Formation Detector")
    
#     # Arguments sent by run_goalx.py
#     parser.add_argument("--tracks", required=True)
#     parser.add_argument("--teams", required=True)
#     parser.add_argument("--out", required=True)
    
#     args = parser.parse_args()
    
#     # Calling the function with the aligned arguments
#     detect_formations(
#         tracks_csv = args.tracks,
#         teams_csv = args.teams,
#         out_file_path = args.out
#     )


"""
formation_detector.py
─────────────────────
Detects the tactical formation (e.g. "4-4-2", "4-3-3") of each team on
every frame by clustering player pitch positions along the depth axis.

Method
──────
1. For each frame, select players belonging to a team (from team_assignments.csv).
2. Project their smooth_x or smooth_y coordinate onto the "depth" axis
   (the axis that runs from goal to goal — configurable via --depth-axis).
3. Apply K-Means with K=3 (3 lines: defence, midfield, attack).
4. Count players per line → formation string  e.g. "4-3-3".
5. Apply a rolling majority-vote window to smooth out single-frame noise.

Input
─────
  --tracks   smoothed_tracks.csv  (from smooth_tracks.py)
  --teams    team_assignments.csv  (from team_classifier.py)
  --out      Output file path (e.g. outputs/formations.csv)

Output
──────
  formations.csv
      Columns: frame_id, home_formation, away_formation
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

N_LINES          = 3     # formation lines: defence / midfield / attack
MIN_PLAYERS      = 5     # skip frame if fewer than this many players visible
SMOOTHING_WINDOW = 25    # rolling majority-vote window (frames)
DEPTH_AXIS       = "x"   # "x" or "y" — the axis from goal to goal on the pitch


# ─────────────────────────────────────────────────────────────────
#  Formation detection for one team in one frame
# ─────────────────────────────────────────────────────────────────

def _formation_for_frame(
    depth_positions: np.ndarray,
    n_lines:         int = N_LINES,
) -> str:
    """
    Given 1-D depth positions of one team's players, cluster into n_lines
    and return a formation string like "4-3-3" (back line first).

    If there are fewer players than lines, returns "unknown".
    """
    n = len(depth_positions)
    if n < n_lines:
        return "unknown"

    k  = min(n_lines, n)
    km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
    labels = km.fit_predict(depth_positions.reshape(-1, 1))

    # Sort cluster IDs by their centroid depth (ascending = defensive)
    centroid_order = np.argsort(km.cluster_centers_.flatten())
    counts = [int(np.sum(labels == c)) for c in centroid_order]

    return "-".join(map(str, counts))


# ─────────────────────────────────────────────────────────────────
#  Rolling majority-vote smoother
# ─────────────────────────────────────────────────────────────────

def _majority_smooth(series: pd.Series, window: int) -> pd.Series:
    """Replace each value with the most common value in a rolling window (string-safe)."""
    vals = series.tolist()
    out = []
    half = window // 2
    n = len(vals)
    
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_slice = vals[start:end]
        
        counter = Counter(v for v in window_slice if v != "unknown")
        if counter:
            out.append(counter.most_common(1)[0][0])
        else:
            out.append("unknown")
            
    return pd.Series(out, index=series.index)


# ─────────────────────────────────────────────────────────────────
#  Main entry-point
# ─────────────────────────────────────────────────────────────────

def detect_formations(
    tracks_csv:    str,
    teams_csv:     str,
    out_file_path: str,  # Logic Fix: Renamed to match the CLI call
    depth_axis:    str = DEPTH_AXIS,
    n_lines:       int = N_LINES,
    window:        int = SMOOTHING_WINDOW,
) -> pd.DataFrame:

    tracks_csv = Path(tracks_csv)
    teams_csv  = Path(teams_csv)
    
    # Logic Fix: Correctly handle path as a file, not a directory
    out_path = Path(out_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX — Formation Detector")
    print(f"  {'─' * 40}")

    # ── Load ─────────────────────────────────────────────────────
    tracks = pd.read_csv(tracks_csv)
    teams  = pd.read_csv(teams_csv)

    if depth_axis not in ("x", "y"):
        raise ValueError("--depth-axis must be 'x' or 'y'")
    depth_col = f"smooth_{depth_axis}"

    required = {"frame_id", "track_id", depth_col}
    if missing := required - set(tracks.columns):
        raise ValueError(f"smoothed_tracks.csv missing columns: {missing}")

    # Merge team labels onto tracks
    merged = tracks.merge(teams[["track_id", "team"]], on="track_id", how="left")
    merged["team"] = merged["team"].fillna("unknown")

    home_ids  = set(teams.loc[teams["team"] == "home",  "track_id"])
    away_ids  = set(teams.loc[teams["team"] == "away",  "track_id"])
    n_frames  = merged["frame_id"].nunique()

    print(f"  ✔  {n_frames} frames  |  home tracks: {len(home_ids)}  |  away tracks: {len(away_ids)}")
    print(f"  ⚙️  depth_axis={depth_axis}  n_lines={n_lines}  smooth_window={window}")

    # ── Per-frame detection ───────────────────────────────────────
    records = []
    for fid in tqdm(sorted(merged["frame_id"].unique()), desc="Detecting formations"):
        fdata = merged[merged["frame_id"] == fid]

        home_pos = fdata.loc[fdata["track_id"].isin(home_ids), depth_col].dropna().values
        away_pos = fdata.loc[fdata["track_id"].isin(away_ids), depth_col].dropna().values

        home_form = _formation_for_frame(home_pos, n_lines) if len(home_pos) >= MIN_PLAYERS else "unknown"
        away_form = _formation_for_frame(away_pos, n_lines) if len(away_pos) >= MIN_PLAYERS else "unknown"

        records.append({"frame_id": fid, "home_formation": home_form, "away_formation": away_form})

    df = pd.DataFrame(records)

    # ── Temporal smoothing ────────────────────────────────────────
    df["home_formation"] = _majority_smooth(df["home_formation"], window)
    df["away_formation"] = _majority_smooth(df["away_formation"], window)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n  📊 Most common formations detected:")
    for col, label in [("home_formation", "Home"), ("away_formation", "Away")]:
        top = df[df[col] != "unknown"][col].value_counts().head(3)
        if not top.empty:
            top_str = "  |  ".join(f"{k} ({v}f)" for k, v in top.items())
            print(f"     {label:<5}: {top_str}")

    # Logic Fix: Save directly to out_path file
    df.to_csv(out_path, index=False)
    print(f"\n  💾 Saved → {out_path}")
    print(f"  ✅  Formation detection complete.")
    print(f"      Next step: pitch_control.py\n")
    return df


# ─────────────────────────────────────────────────────────────────
#  CLI Aligned with run_goalx.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="goalX Formation Detector")
    
    # Orchestrator-compatible arguments
    parser.add_argument("--tracks", required=True, help="Input smoothed_tracks.csv")
    parser.add_argument("--teams",  required=True, help="Input team_assignments.csv")
    parser.add_argument("--out",    required=True, help="Output formations.csv")
    
    args = parser.parse_args()
    
    # Calling the function with keyword arguments that match the definition
    detect_formations(
        tracks_csv    = args.tracks,
        teams_csv     = args.teams,
        out_file_path = args.out
    )