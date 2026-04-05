"""
formation_detector.py
─────────────────────
Detects the tactical formation (e.g. "4-4-2", "4-3-3") of each team on
every frame by clustering player pitch positions along the depth axis.

FIXES
─────────────────────────────────
FIX 1 — Minimum player count guard (CRITICAL)
  Raised MIN_PLAYERS to 7 and added a MAX_PER_LINE cap of 6.
  Any formation line with > 6 players is marked "invalid" and replaced 
  with "unknown" before temporal smoothing.

FIX 2 — Formation validity check
  Filters out strings like "2-9-1" that cannot exist in football.
  Invalid formations revert to "unknown" so the temporal smoother
  can fill them from neighbouring valid frames.

FIX 3 — Informative Summary
  Prints per-team unique formation count and warns if >10 unique
  formations are detected, which indicates bad team labels.
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

N_LINES          = 3      # formation lines: defence / midfield / attack
MIN_PLAYERS      = 7      # need ≥7 for stable clustering
MAX_PER_LINE     = 6      # more than 6 in one line = physically impossible
SMOOTHING_WINDOW = 25     # rolling majority-vote window (frames)
DEPTH_AXIS       = "x"    # "x" or "y" — the axis from goal to goal on the pitch

# Teams that contribute to formation analysis (skip ball / uncertain / unknown)
VALID_TEAMS = {"home", "away"}


# ─────────────────────────────────────────────────────────────────
#  Formation detection for one team in one frame
# ─────────────────────────────────────────────────────────────────

def _formation_for_frame(
    depth_positions: np.ndarray,
    n_lines:         int = N_LINES,
) -> str:
    """
    Given 1-D depth positions, cluster into n_lines and return a 
    formation string like "4-3-3". Returns "unknown" for invalid shapes.
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

    # Reject physically impossible formations
    if any(c > MAX_PER_LINE for c in counts):
        return "unknown"

    return "-".join(map(str, counts))


# ─────────────────────────────────────────────────────────────────
#  Rolling majority-vote smoother
# ─────────────────────────────────────────────────────────────────

def _majority_smooth(series: pd.Series, window: int) -> pd.Series:
    """Replace each value with the most common value in a rolling window."""
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
    out_file_path: str,
    depth_axis:    str = DEPTH_AXIS,
    n_lines:       int = N_LINES,
    window:        int = SMOOTHING_WINDOW,
) -> pd.DataFrame:

    tracks_csv = Path(tracks_csv)
    teams_csv  = Path(teams_csv)
    out_path   = Path(out_file_path)
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

    # Merge team labels onto tracks (keep only valid teams)
    merged = tracks.merge(teams[["track_id", "team"]], on="track_id", how="left")
    merged["team"] = merged["team"].fillna("unknown")

    home_ids  = set(teams.loc[teams["team"] == "home",  "track_id"])
    away_ids  = set(teams.loc[teams["team"] == "away",  "track_id"])
    n_frames  = merged["frame_id"].nunique()

    print(f"  ✔  {n_frames} frames  |  home tracks: {len(home_ids)}  |  away tracks: {len(away_ids)}")

    if len(home_ids) < MIN_PLAYERS or len(away_ids) < MIN_PLAYERS:
        print(f"\n  ⚠  Fewer than {MIN_PLAYERS} tracks for one or both teams.")
        print(f"     Formations will show 'unknown' often.")
        print(f"     Cause: team classifier assigned most players to 'other' or 'uncertain'.")

    print(f"  ⚙️  depth_axis={depth_axis}  n_lines={n_lines}  smooth_window={window}")

    # ── Per-frame detection ───────────────────────────────────────
    records = []
    for fid in tqdm(sorted(merged["frame_id"].unique()), desc="  Detecting formations"):
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
    print(f"\n  📊 Formation summary:")
    for col, label in [("home_formation", "Home"), ("away_formation", "Away")]:
        n_unknown = (df[col] == "unknown").sum()
        n_unique  = df[df[col] != "unknown"][col].nunique()
        top3      = df[df[col] != "unknown"][col].value_counts().head(3)
        top_str   = "  |  ".join(f"{k}({v}f)" for k, v in top3.items()) if not top3.empty else "–"
        print(f"     {label}: {n_unique} unique  |  unknown={n_unknown}f  |  top: {top_str}")

    if df["home_formation"].nunique() > 10 or df["away_formation"].nunique() > 10:
        print(f"\n  ⚠  High formation variability detected (>10 unique values).")
        print(f"     This usually means the team classifier labels are noisy.")

    df.to_csv(out_path, index=False)
    print(f"\n  💾 Saved → {out_path}")
    print(f"  ✅  Formation detection complete.\n")
    return df


# ─────────────────────────────────────────────────────────────────
#  CLI Aligned with run_goalx.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="goalX Formation Detector")
    
    parser.add_argument("--tracks", required=True, help="Input smoothed_tracks.csv")
    parser.add_argument("--teams",  required=True, help="Input team_assignments.csv")
    parser.add_argument("--out",    required=True, help="Output formations.csv")
    
    args = parser.parse_args()
    
    detect_formations(
        tracks_csv    = args.tracks,
        teams_csv     = args.teams,
        out_file_path = args.out
    )