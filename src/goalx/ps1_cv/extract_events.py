"""
extract_events.py
─────────────────
Scans smoothed player trajectories and ball positions to extract three
football event types:

  1. SHOT       — Kinematic detection: The ball accelerates rapidly (speed spike)
                  near a goal zone, and a player was in close proximity just before.

  2. POSSESSION — Hysteresis-based detection: The player closest to the ball.
                  A new player must be significantly closer to the ball to "steal"
                  possession, reducing flicker in contested duels.

  3. PRESSURE   — multiple opponents converge within a configurable radius
                  of the ball-carrier simultaneously.

Input
─────
  --tracks      smoothed_tracks.csv  (from smooth_tracks.py)
                Required columns: frame_id, track_id, smooth_x, smooth_y
                NOTE: The ball must be present in this file as track_id = -1

  --ball        (Deprecated) Left for CLI compatibility, but the ball is now
                extracted directly from --tracks to ensure coordinate alignment.

  --pitch-w     Pitch canvas width  in pixels  (default: 1050)
  --pitch-h     Pitch canvas height in pixels  (default: 680)

Output
──────
  <out-dir>/events.csv
      Columns: frame_id, event_type, track_id, x, y, detail

Usage
─────
  python3 src/goalx/ps1_cv/extract_events.py \
      --tracks  outputs/smoothed/smoothed_tracks.csv \
      --out-dir outputs/events
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
#  PITCH GEOMETRY  (pixels — match your draw_pitch.py canvas)
# ─────────────────────────────────────────────────────────────────

PITCH_W = 1050      # canvas width
PITCH_H = 680       # canvas height
FPS     = 25        # frames per second (for velocity conversion)

# Goal boxes: x-ranges at each end of the pitch
GOAL_ZONE_X_LEFT  = (0,   120)   # left-end goal area
GOAL_ZONE_X_RIGHT = (930, 1050)  # right-end goal area
GOAL_ZONE_Y       = (240, 440)   # vertical extent of goal mouth


# ─────────────────────────────────────────────────────────────────
#  EVENT THRESHOLDS
# ─────────────────────────────────────────────────────────────────

# SHOT: Kinematic ball acceleration
SHOT_BALL_SPEED_THRESH = 15.0  # Ball speed (px/frame) must exceed this
SHOT_PLAYER_PROX_PX    = 60    # Player must be within this distance 2 frames prior
SHOT_COOLDOWN          = 25    # Wait N frames before registering another shot

# POSSESSION: Hysteresis logic
POSSESSION_DIST_THRESHOLD = 120  # px — max distance to own the ball
POSSESSION_HYSTERESIS     = 30   # px — challenger must beat current owner by this margin
POSSESSION_MIN_DURATION   = 3    # frames — debounce to suppress flicker

# PRESSURE: multiple opponents within this radius of the ball-carrier
PRESSURE_RADIUS_PX   = 200       # px
PRESSURE_MIN_PLAYERS = 2         # at least N opponents converging


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _dist(ax, ay, bx, by) -> float:
    return float(np.hypot(ax - bx, ay - by))


# ─────────────────────────────────────────────────────────────────
#  1.  SHOT detection (Kinematic)
# ─────────────────────────────────────────────────────────────────

def extract_shots(
    players: pd.DataFrame,
    ball_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kinematic shot detector.
    Looks for frames where the ball velocity spikes significantly near a goal,
    and identifies the player who was closest to the ball right before the spike.
    """
    events = []
    if ball_df.empty:
        return pd.DataFrame(events)

    last_shot_frame = -999
    fast_ball_frames = ball_df[ball_df["b_speed"] > SHOT_BALL_SPEED_THRESH]

    for _, b_row in fast_ball_frames.iterrows():
        fid = int(b_row["frame_id"])
        
        if fid - last_shot_frame < SHOT_COOLDOWN:
            continue

        bx, by = b_row["smooth_x"], b_row["smooth_y"]

        # Was the ball near a goal zone during the speed spike?
        near_goal = (bx < GOAL_ZONE_X_LEFT[1] or bx > GOAL_ZONE_X_RIGHT[0])
        if not near_goal:
            continue

        # Check players 2 frames BEFORE the spike (the moment of the strike)
        shooter_frame = fid - 2
        frame_players = players[players["frame_id"] == shooter_frame]
        
        if frame_players.empty:
            continue

        dists = np.hypot(
            frame_players["smooth_x"].values - bx, 
            frame_players["smooth_y"].values - by
        )
        min_idx  = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        # If a player was close enough, they are the shooter
        if min_dist <= SHOT_PLAYER_PROX_PX:
            shooter_id = int(frame_players.iloc[min_idx]["track_id"])
            last_shot_frame = fid
            
            events.append(dict(
                frame_id   = fid,
                event_type = "shot",
                track_id   = shooter_id,
                x          = round(bx, 1),
                y          = round(by, 1),
                detail     = f"ball_speed={b_row['b_speed']:.1f}px/f",
            ))

    return pd.DataFrame(events)


# ─────────────────────────────────────────────────────────────────
#  2.  POSSESSION changes (Hysteresis)
# ─────────────────────────────────────────────────────────────────

def extract_possession(
    players: pd.DataFrame,
    ball_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hysteresis-based possession.
    A new player must be significantly closer to the ball to steal it.
    """
    if ball_df.empty:
        return pd.DataFrame()

    ball_dict = ball_df.set_index("frame_id")[["smooth_x", "smooth_y"]].to_dict("index")
    owner_log: list[dict] = []
    
    current_owner = None

    for fid in sorted(players["frame_id"].unique()):
        if fid not in ball_dict:
            current_owner = None
            continue

        bx = ball_dict[fid]["smooth_x"]
        by = ball_dict[fid]["smooth_y"]
        frame_players = players[players["frame_id"] == fid]

        if frame_players.empty:
            continue

        dists = np.hypot(
            frame_players["smooth_x"].values - bx,
            frame_players["smooth_y"].values - by,
        )
        min_idx      = int(np.argmin(dists))
        closest_dist = float(dists[min_idx])
        closest_id   = int(frame_players.iloc[min_idx]["track_id"])

        if closest_dist > POSSESSION_DIST_THRESHOLD:
            current_owner = None
            owner_id = -1
        else:
            # Hysteresis check
            if current_owner is not None and current_owner in frame_players["track_id"].values:
                owner_row = frame_players[frame_players["track_id"] == current_owner].iloc[0]
                owner_dist = _dist(owner_row["smooth_x"], owner_row["smooth_y"], bx, by)
                
                # Must beat the current owner by the hysteresis margin
                if closest_id != current_owner and closest_dist < (owner_dist - POSSESSION_HYSTERESIS):
                    current_owner = closest_id
            else:
                current_owner = closest_id
            
            owner_id = current_owner

        owner_log.append({"frame_id": fid, "track_id": owner_id, "x": bx, "y": by})

    if not owner_log:
        return pd.DataFrame()

    owner_df = pd.DataFrame(owner_log)
    events = []
    prev_owner    = None
    stable_owner  = None
    stable_since  = 0

    for _, row in owner_df.iterrows():
        tid = int(row["track_id"])

        if tid != prev_owner:
            stable_since = 1
            prev_owner   = tid
        else:
            stable_since += 1

        if stable_since == POSSESSION_MIN_DURATION and tid != stable_owner:
            stable_owner = tid
            events.append(dict(
                frame_id   = int(row["frame_id"]),
                event_type = "possession",
                track_id   = tid,
                x          = round(row["x"], 1),
                y          = round(row["y"], 1),
                detail     = "loose_ball" if tid == -1 else f"owner={tid}",
            ))

    return pd.DataFrame(events)


# ─────────────────────────────────────────────────────────────────
#  3.  PRESSURE zones
# ─────────────────────────────────────────────────────────────────

def extract_pressure(
    players: pd.DataFrame,
    ball_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fires when >= PRESSURE_MIN_PLAYERS opponents are near the ball carrier.
    """
    if ball_df.empty:
        return pd.DataFrame()

    ball_dict = ball_df.set_index("frame_id")[["smooth_x", "smooth_y"]].to_dict("index")
    events = []
    last_pressure_frame = -999   

    for fid in sorted(players["frame_id"].unique()):
        if fid not in ball_dict:
            continue
        if fid - last_pressure_frame < 10:
            continue

        bx = ball_dict[fid]["smooth_x"]
        by = ball_dict[fid]["smooth_y"]
        frame_players = players[players["frame_id"] == fid]
        
        if frame_players.empty:
            continue

        dists = np.hypot(
            frame_players["smooth_x"].values - bx,
            frame_players["smooth_y"].values - by,
        )

        carrier_idx = int(np.argmin(dists))
        carrier_tid = int(frame_players.iloc[carrier_idx]["track_id"])

        pressuring = np.sum(
            (dists < PRESSURE_RADIUS_PX) &
            (frame_players["track_id"].values != carrier_tid)
        )

        if pressuring >= PRESSURE_MIN_PLAYERS:
            last_pressure_frame = fid
            events.append(dict(
                frame_id   = fid,
                event_type = "pressure",
                track_id   = carrier_tid,
                x          = round(bx, 1),
                y          = round(by, 1),
                detail     = f"pressuring={pressuring}",
            ))

    return pd.DataFrame(events)


# ─────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────

def run(
    tracks_csv:  str,
    ball_csv:    Optional[str],
    out_dir:     str,
    pitch_w:     int = PITCH_W,
    pitch_h:     int = PITCH_H,
) -> pd.DataFrame:

    global PITCH_W, PITCH_H
    PITCH_W = pitch_w
    PITCH_H = pitch_h

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX — Advanced Event Extractor")
    print(f"  {'─' * 40}")

    # ── Load tracks ───────────────────────────────────────────────
    tracks = pd.read_csv(tracks_csv)
    required = {"frame_id", "track_id", "smooth_x", "smooth_y"}
    if missing := required - set(tracks.columns):
        raise ValueError(f"smoothed_tracks.csv missing columns: {missing}")

    # Separate players and projected ball
    players = tracks[tracks["track_id"] != -1].copy()
    ball_df = tracks[tracks["track_id"] == -1].copy()

    if ball_df.empty:
        print("  ⚠  No projected ball found (track_id = -1). Skipping events.")
        return pd.DataFrame()

    # Calculate Ball Kinematics
    ball_df = ball_df.sort_values("frame_id")
    ball_df["bdx"] = ball_df["smooth_x"].diff()
    ball_df["bdy"] = ball_df["smooth_y"].diff()
    ball_df["b_speed"] = np.hypot(ball_df["bdx"], ball_df["bdy"]).fillna(0)

    print(f"  ✔  Players : {len(players)} rows  |  {players['track_id'].nunique()} IDs")
    print(f"  ✔  Ball    : {len(ball_df)} rows  (Coordinate-aligned)")

    # ── Extract events ────────────────────────────────────────────
    print("\n  Extracting shots (Kinematic) …")
    shot_df = extract_shots(players, ball_df)

    print("  Extracting possession changes (Hysteresis) …")
    poss_df = extract_possession(players, ball_df)

    print("  Extracting pressure zones …")
    pres_df = extract_pressure(players, ball_df)

    # ── Combine ───────────────────────────────────────────────────
    all_events = pd.concat(
        [df for df in [shot_df, poss_df, pres_df] if not df.empty],
        ignore_index=True,
    )

    if all_events.empty:
        print("\n  ⚠  No events detected. Check thresholds.")
        return all_events

    all_events.sort_values("frame_id", inplace=True, ignore_index=True)

    # ── Save ──────────────────────────────────────────────────────
    out_csv = out_dir / "events.csv"
    all_events.to_csv(out_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────
    summary = all_events.groupby("event_type").size().to_dict()
    print(f"\n  📊 Event Summary:")
    for etype, count in sorted(summary.items()):
        print(f"     {etype:<12} : {count}")

    print(f"\n  💾 Saved → {out_csv}")
    print(f"\n  ✅  Event extraction complete.\n")

    return all_events


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Extract football events from smoothed player + ball tracks."
    )
    p.add_argument("--tracks",   required=True,
                   help="smoothed_tracks.csv from smooth_tracks.py")
    p.add_argument("--ball",     default="none",
                   help="(Deprecated) Ball is now extracted natively from --tracks")
    p.add_argument("--out-dir",  default="outputs/events",
                   help="Output directory  (default: outputs/events)")
    p.add_argument("--pitch-w",  type=int, default=PITCH_W,
                   help=f"Pitch canvas width in px  (default: {PITCH_W})")
    p.add_argument("--pitch-h",  type=int, default=PITCH_H,
                   help=f"Pitch canvas height in px  (default: {PITCH_H})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        tracks_csv = args.tracks,
        ball_csv   = args.ball,
        out_dir    = args.out_dir,
        pitch_w    = args.pitch_w,
        pitch_h    = args.pitch_h,
    )