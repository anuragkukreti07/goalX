# """
# clutch_score.py  —  PS2 begins
# ──────────────────────────────
# Computes a composite "Clutch Score" for every shot event detected by
# extract_events.py.  The score combines three independent signals:

#   1. xG (expected goals)
#      Position-based logistic model using distance and angle to goal.
#      Higher xG = more dangerous shot location.

#   2. Pressure Score
#      How many opponents were within PRESSURE_RADIUS of the shooter at the
#      moment of the shot.  More pressure under which a shot is taken →
#      higher clutch significance.

#   3. Pitch Control Context
#      Fraction of the pitch controlled by the shooter's team at the shot
#      moment.  A shot taken while the team is under spatial siege (low
#      control %) is weighted as more clutch than one from a comfortable
#      possession phase.

#   4. Temporal Weight  (proxy — uses frame_id normalised to [0, 1])
#      Later-game events carry more weight under the assumption that pressure
#      accumulates over time.  Replace with actual match clock when available.

#   Clutch Score = xG × (1 + pressure_bonus) × control_weight × temporal_weight

# Input
# ─────
#   --events    events.csv from extract_events.py
#   --tracks    smoothed_tracks.csv from smooth_tracks.py
#   --teams     team_assignments.csv from team_classifier.py
#   --control   pitch_control.csv from pitch_control.py
#   --out-dir   Output directory

#   Pitch geometry (must match draw_pitch.py canvas):
#   --pitch-w   Canvas width in pixels  (default: 1050)
#   --pitch-h   Canvas height in pixels (default: 680)

# Output
# ──────
#   <out-dir>/clutch_scores.csv
#       Columns: frame_id, track_id, x, y, xG, pressure_score,
#                control_context, temporal_weight, clutch_score

# Usage
# ─────
#   python3 src/goalx/ps1_cv/clutch_score.py \\
#       --events   outputs/events/events.csv \\
#       --tracks   outputs/smoothed/smoothed_tracks.csv \\
#       --teams    outputs/teams/team_assignments.csv \\
#       --control  outputs/pitch_control/pitch_control.csv \\
#       --out-dir  outputs/clutch
# """

# import argparse
# import math
# from pathlib import Path

# import numpy as np
# import pandas as pd


# # ─────────────────────────────────────────────────────────────────
# #  PITCH GEOMETRY  (pixels — must match draw_pitch.py)
# # ─────────────────────────────────────────────────────────────────

# PITCH_W = 1050
# PITCH_H = 680

# # Goal centres on the pitch canvas (x = goal line, y = mid-height)
# GOAL_LEFT  = (0,          PITCH_H / 2)
# GOAL_RIGHT = (PITCH_W,    PITCH_H / 2)
# GOAL_WIDTH = 73   # pixels — tune to your draw_pitch.py scale


# # ─────────────────────────────────────────────────────────────────
# #  xG MODEL  (position-based logistic approximation)
# # ─────────────────────────────────────────────────────────────────
# # Coefficients derived from the public literature on position-only xG models
# # (e.g. Caley 2015, Statsbomb).  Replace with trained coefficients when you
# # have labelled shot data.

# XG_INTERCEPT        = -1.56
# XG_COEF_DIST        = -0.0138   # per pixel (negative — further = lower xG)
# XG_COEF_ANGLE_SIN   =  0.88     # sin(angle) reward
# PRESSURE_RADIUS_PX  = 200       # px — opponent within this range = applying pressure


# def _nearest_goal(x: float, y: float) -> tuple[float, float]:
#     """Return the goal centre (left or right) closest to this pitch position."""
#     dist_l = math.hypot(x - GOAL_LEFT[0],  y - GOAL_LEFT[1])
#     dist_r = math.hypot(x - GOAL_RIGHT[0], y - GOAL_RIGHT[1])
#     return GOAL_LEFT if dist_l < dist_r else GOAL_RIGHT


# # def _shot_angle(x: float, y: float, gx: float, gy: float) -> float:
# #     """
# #     Shooting angle (radians) — the angle subtended by the goal mouth at the
# #     shooter's position.  Wider angle = easier shot.

# #     Approximation: angle = arctan(goal_width / 2 / dist_to_post_midpoint)
# #     """
# #     dist = math.hypot(x - gx, y - gy)
# #     if dist < 1:
# #         return math.pi / 2
# #     return math.atan2(GOAL_WIDTH / 2, dist)


# # def compute_xg(px: float, py: float) -> float:
# #     """
# #     Return expected-goals probability for a shot from (px, py).
# #     Output is in [0, 1].
# #     """
# #     gx, gy = _nearest_goal(px, py)
# #     dist   = math.hypot(px - gx, py - gy)
# #     angle  = _shot_angle(px, py, gx, gy)

# #     log_odds = (
# #         XG_INTERCEPT
# #         + XG_COEF_DIST * dist
# #         + XG_COEF_ANGLE_SIN * math.sin(angle)
# #     )
# #     return 1.0 / (1.0 + math.exp(-log_odds))


# # ─────────────────────────────────────────────────────────────────
# #  PRESSURE SCORE
# # ─────────────────────────────────────────────────────────────────

# def _pressure_at_frame(
#     frame_id:  int,
#     shooter_tid: int,
#     tracks:    pd.DataFrame,
#     teams:     pd.DataFrame,
# ) -> float:
#     """
#     Count opponents within PRESSURE_RADIUS_PX of the shooter on this frame.
#     Returns a normalised score in [0, 1] (capped at 5 opponents = 1.0).
#     """
#     fdata = tracks[tracks["frame_id"] == frame_id]
#     if fdata.empty:
#         return 0.0

#     shooter_rows = fdata[fdata["track_id"] == shooter_tid]
#     if shooter_rows.empty:
#         return 0.0

#     sx = float(shooter_rows.iloc[0]["smooth_x"])
#     sy = float(shooter_rows.iloc[0]["smooth_y"])

#     shooter_team = teams.loc[teams["track_id"] == shooter_tid, "team"]
#     shooter_team = shooter_team.values[0] if len(shooter_team) > 0 else "unknown"

#     # Count opponents (different team) within radius
#     opp_mask   = ~fdata["track_id"].isin(
#         teams.loc[teams["team"] == shooter_team, "track_id"]
#     )
#     opponents  = fdata[opp_mask]

#     if opponents.empty:
#         return 0.0

#     dists = np.hypot(
#         opponents["smooth_x"].values - sx,
#         opponents["smooth_y"].values - sy,
#     )
#     n_pressuring = int(np.sum(dists < PRESSURE_RADIUS_PX))
#     return min(1.0, n_pressuring / 5.0)   # normalise: 5 opponents = max pressure


# # ─────────────────────────────────────────────────────────────────
# #  CONTROL CONTEXT
# # ─────────────────────────────────────────────────────────────────

# def _control_at_frame(
#     frame_id:    int,
#     team:        str,
#     control_df:  pd.DataFrame,
# ) -> float:
#     """
#     Return the pitch control fraction for the shooting team at this frame.
#     Closer to 0 = team under pressure (spatially) = more clutch context.
#     We invert so that low control → high clutch weight.
#     """
#     row = control_df[control_df["frame_id"] == frame_id]
#     if row.empty:
#         return 0.5   # neutral default

#     col = "home_pct" if team == "home" else "away_pct"
#     pct = float(row.iloc[0].get(col, 50.0)) / 100.0

#     # Invert: shooting under siege (pct < 0.5) → clutch_weight > 1
#     # shooting from dominance (pct > 0.5) → clutch_weight < 1
#     return 1.0 + (0.5 - pct)   # range [0.5, 1.5]


# # ─────────────────────────────────────────────────────────────────
# #  CLUTCH SCORE COMPOSITE
# # ─────────────────────────────────────────────────────────────────

# def _temporal_weight(frame_id: int, max_frame: int) -> float:
#     """
#     Linear temporal weight: events near the end of the clip score higher.
#     Range [0.8, 1.5].  Replace with actual match minute when available.
#     """
#     t = frame_id / max(max_frame, 1)
#     return 0.8 + 0.7 * t


# def compute_clutch_scores(
#     events_csv:  str,
#     tracks_csv:  str,
#     teams_csv:   str,
#     control_csv: str,
#     out_dir:     str,
#     pitch_w:     int = PITCH_W,
#     pitch_h:     int = PITCH_H,
# ) -> pd.DataFrame:

#     global PITCH_W, PITCH_H, GOAL_LEFT, GOAL_RIGHT
#     PITCH_W     = pitch_w
#     PITCH_H     = pitch_h
#     GOAL_LEFT   = (0,       PITCH_H / 2)
#     GOAL_RIGHT  = (PITCH_W, PITCH_H / 2)

#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     print(f"\n  goalX — Clutch Score  (PS2)")
#     print(f"  {'─' * 40}")

#     events  = pd.read_csv(events_csv)
#     tracks  = pd.read_csv(tracks_csv)
#     teams   = pd.read_csv(teams_csv)
#     control = pd.read_csv(control_csv)

#     shots = events[events["event_type"] == "shot"].copy()
#     if shots.empty:
#         print("  ⚠️  No shot events found in events.csv.")
#         print("       Check extract_events.py thresholds or ball CSV quality.")
#         return pd.DataFrame()

#     print(f"  ✔  {len(shots)} shot events to score")

#     max_frame = int(events["frame_id"].max())
#     records   = []

#     for _, shot in shots.iterrows():
#         fid = int(shot["frame_id"])
#         tid = int(shot["track_id"])
#         px  = float(shot["x"])
#         py  = float(shot["y"])

#         # 1. xG
#         xg = compute_xg(px, py)

#         # 2. Pressure (how many opponents were close)
#         pressure = _pressure_at_frame(fid, tid, tracks, teams)
#         pressure_bonus = 0.5 * pressure   # up to +50 % boost

#         # 3. Pitch control context
#         team_label   = teams.loc[teams["track_id"] == tid, "team"]
#         team_label   = team_label.values[0] if len(team_label) > 0 else "unknown"
#         ctrl_weight  = _control_at_frame(fid, team_label, control)

#         # 4. Temporal weight
#         t_weight = _temporal_weight(fid, max_frame)

#         # 5. Composite Clutch Score
#         clutch = xg * (1.0 + pressure_bonus) * ctrl_weight * t_weight

#         records.append({
#             "frame_id":       fid,
#             "track_id":       tid,
#             "team":           team_label,
#             "x":              round(px, 1),
#             "y":              round(py, 1),
#             "xG":             round(xg, 4),
#             "pressure_score": round(pressure, 3),
#             "control_context":round(ctrl_weight, 3),
#             "temporal_weight":round(t_weight, 3),
#             "clutch_score":   round(clutch, 4),
#         })

#     df = pd.DataFrame(records).sort_values("clutch_score", ascending=False, ignore_index=True)

#     # ── Summary ───────────────────────────────────────────────────
#     print(f"\n  📊 Clutch Score summary:")
#     print(f"     Shots scored      : {len(df)}")
#     print(f"     Mean xG           : {df['xG'].mean():.3f}")
#     print(f"     Mean Clutch Score : {df['clutch_score'].mean():.3f}")
#     print(f"     Top Clutch Score  : {df['clutch_score'].max():.3f}  "
#           f"(frame {int(df.iloc[0]['frame_id'])}, track {int(df.iloc[0]['track_id'])})")

#     out_csv = out_dir / "clutch_scores.csv"
#     df.to_csv(out_csv, index=False)
#     print(f"\n  💾 Saved → {out_csv}")
#     print(f"  ✅  Clutch scoring complete.")
#     print(f"      Next step: tactical_radar.py --clutch {out_csv}\n")
#     return df


# # ─────────────────────────────────────────────────────────────────
# #  CLI
# # ─────────────────────────────────────────────────────────────────

# def _parse_args():
#     p = argparse.ArgumentParser(
#         description="Compute Clutch Score for each shot event."
#     )
#     p.add_argument("--events",  required=True)
#     p.add_argument("--tracks",  required=True)
#     p.add_argument("--teams",   required=True)
#     p.add_argument("--control", required=True,
#                    help="pitch_control.csv from pitch_control.py")
#     p.add_argument("--out-dir", default="outputs/clutch")
#     p.add_argument("--pitch-w", type=int, default=PITCH_W)
#     p.add_argument("--pitch-h", type=int, default=PITCH_H)
#     return p.parse_args()


# if __name__ == "__main__":
#     args = _parse_args()
#     compute_clutch_scores(
#         events_csv  = args.events,
#         tracks_csv  = args.tracks,
#         teams_csv   = args.teams,
#         control_csv = args.control,
#         out_dir     = args.out_dir,
#         pitch_w     = args.pitch_w,
#         pitch_h     = args.pitch_h,
#     )



"""
clutch_score.py  —  Contextual Event Scoring
──────────────────────────────────────────────
Calculates the 'Clutch' significance of shot events based on 
Expected Goals (xG), defensive pressure, and pitch control.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
#  PITCH & xG CONSTANTS (Corner View Optimized)
# ─────────────────────────────────────────────────────────────────

PITCH_W = 1050
PITCH_H = 680
GOAL_WIDTH = 73 

# Logistic regression coefficients for xG
XG_INTERCEPT      = -1.56
XG_COEF_DIST      = -0.0138   
XG_COEF_ANGLE_SIN = 0.88     
PRESSURE_RADIUS   = 200 # Pixels

# ─────────────────────────────────────────────────────────────────
#  Math Engine
# ─────────────────────────────────────────────────────────────────

def _nearest_goal(x, y):
    dist_l = math.hypot(x - 0, y - (PITCH_H / 2))
    dist_r = math.hypot(x - PITCH_W, y - (PITCH_H / 2))
    return (0, PITCH_H / 2) if dist_l < dist_r else (PITCH_W, PITCH_H / 2)

def compute_xg(px, py):
    gx, gy = _nearest_goal(px, py)
    dist = math.hypot(px - gx, py - gy)
    
    # Angle approximation
    if dist < 1: return 0.5
    angle = math.atan2(GOAL_WIDTH / 2, dist)

    log_odds = XG_INTERCEPT + (XG_COEF_DIST * dist) + (XG_COEF_ANGLE_SIN * math.sin(angle))
    return 1.0 / (1.0 + math.exp(-log_odds))

def _pressure_score(fid, tid, tracks, teams):
    if teams is None: return 0.2
    
    fdata = tracks[tracks["frame_id"] == fid]
    shooter = fdata[fdata["track_id"] == tid]
    if shooter.empty: return 0.0

    sx, sy = shooter.iloc[0]["smooth_x"], shooter.iloc[0]["smooth_y"]
    
    # Find team
    t_row = teams.loc[teams["track_id"] == tid, "team"]
    s_team = t_row.values[0] if not t_row.empty else "unknown"

    # Count opponents
    opps = fdata[~fdata["track_id"].isin(teams.loc[teams["team"] == s_team, "track_id"])]
    if opps.empty: return 0.0

    dists = np.hypot(opps["smooth_x"].values - sx, opps["smooth_y"].values - sy)
    return min(1.0, np.sum(dists < PRESSURE_RADIUS) / 5.0)

def _control_weight(fid, team, control_df):
    row = control_df[control_df["frame_id"] == fid]
    if row.empty: return 1.0
    
    col = "home_pct" if team == "home" else "away_pct"
    pct = float(row.iloc[0].get(col, 50.0)) / 100.0
    return 1.0 + (0.5 - pct) # Low control = High weight

# ─────────────────────────────────────────────────────────────────
#  Execution Logic
# ─────────────────────────────────────────────────────────────────

def run_scoring(events_in, tracks_in, control_in, out_file, teams_in=None):
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX — Clutch Score (Stage 10)")
    print(f"  {'─' * 40}")

    ev_df = pd.read_csv(events_in)
    tr_df = pd.read_csv(tracks_in)
    ct_df = pd.read_csv(control_in)
    tm_df = pd.read_csv(teams_in) if teams_in and Path(teams_in).exists() else None

    shots = ev_df[ev_df["event_type"] == "shot"].copy()
    if shots.empty:
        print("  ⚠ No shots to score.")
        pd.DataFrame(columns=["frame_id", "clutch_score"]).to_csv(out_path, index=False)
        return

    max_f = int(ev_df["frame_id"].max())
    results = []

    for _, s in shots.iterrows():
        fid, tid = int(s["frame_id"]), int(s["track_id"])
        
        # 1. xG
        xg = compute_xg(s["x"], s["y"])
        
        # 2. Pressure
        pres = _pressure_score(fid, tid, tr_df, tm_df)
        
        # 3. Control
        team = "unknown"
        if tm_df is not None:
            tr = tm_df.loc[tm_df["track_id"] == tid, "team"]
            team = tr.values[0] if not tr.empty else "unknown"
        ctrl = _control_weight(fid, team, ct_df)
        
        # 4. Time
        time_w = 0.8 + 0.7 * (fid / max_f)

        # 5. Result
        clutch = xg * (1.0 + (0.5 * pres)) * ctrl * time_w

        results.append({
            "frame_id": fid, "track_id": tid, "team": team,
            "xG": round(xg, 4), "clutch_score": round(clutch, 4)
        })

    final_df = pd.DataFrame(results).sort_values("clutch_score", ascending=False)
    final_df.to_csv(out_path, index=False)
    
    print(f"  ✔ Scored {len(final_df)} shots.")
    print(f"  💾 Saved → {out_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--pitch-control", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--teams", required=False) # Optional to prevent crash
    
    args = parser.parse_args()
    run_scoring(args.events, args.tracks, args.pitch_control, args.out, args.teams)
