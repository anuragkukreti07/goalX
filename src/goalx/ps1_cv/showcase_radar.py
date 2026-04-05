"""
showcase_radar.py
─────────────────
Renders a top-down tactical radar video from smoothed player + ball tracks,
annotated with detected events.

Improvements over v1
────────────────────
  • Team-coloured dots (reads team_assignments.csv if available)
  • Ball drawn at correct smoothed position
  • Event flash: circles pulse on the pitch at shot / pressure locations
  • Player trails (last TRAIL_FRAMES positions)
  • Correct frame_id column (was reading wrong column name before)
  • Event HUD always rendered even without team assignments
  • Out-Of-Bounds (OOB) filtering to prevent ghost trails on camera cuts
"""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

FPS         = 25
TRAIL_LEN   = 15      # frames of position history to show per player

# BGR colours
TEAM_COLORS = {
    "home":    (230,  60,  60),   # red-ish
    "away":    (60,  200,  60),   # green
    "other":   (200, 200,  60),   # yellow (refs)
    "unknown": (180, 180, 180),   # grey fallback
}
BALL_COLOR     = (0, 220, 255)    # orange
EVENT_COLORS   = {
    "shot":      (0,   80, 255),
    "possession":(255, 180,   0),
    "pressure":  (180,   0, 255),
}
EVENT_FLASH_FRAMES = 20   # how many frames an event marker stays visible


def _get_team_color(tid: int, team_map: dict) -> tuple:
    team = team_map.get(int(tid), "unknown")
    return TEAM_COLORS.get(team, TEAM_COLORS["unknown"])


def run_showcase(
    tracks_csv: str,
    events_csv: str,
    pitch_img:  str,
    out_vid:    str,
    teams_csv:  str = None,
    fps:        int = FPS,
) -> None:

    tracks_csv = Path(tracks_csv)
    events_csv = Path(events_csv)
    pitch_img  = Path(pitch_img)
    out_vid    = Path(out_vid)
    teams_csv  = Path(teams_csv) if teams_csv else None

    print(f"\n  goalX — Tactical Radar Showcase")
    print(f"  {'─' * 40}")

    # ── Load ──────────────────────────────────────────────────────
    if not tracks_csv.exists():
        raise FileNotFoundError(f"smoothed_tracks.csv not found: {tracks_csv}")
    if not pitch_img.exists():
        raise FileNotFoundError(f"Pitch map not found: {pitch_img}  — run draw_pitch.py first")

    tracks     = pd.read_csv(tracks_csv)
    pitch_base = cv2.imread(str(pitch_img))
    h, w       = pitch_base.shape[:2]

    events = pd.DataFrame()
    if events_csv.exists():
        events = pd.read_csv(events_csv)
        print(f"  ✔  Events: {len(events)} rows")

    # Team colour map
    team_map: dict[int, str] = {}
    if teams_csv and teams_csv.exists():
        teams_df = pd.read_csv(teams_csv)
        team_map = dict(zip(teams_df["track_id"].astype(int),
                            teams_df["team"].astype(str)))
        print(f"  ✔  Team assignments loaded for {len(team_map)} tracks")
    else:
        print(f"  ⚠️  No team_assignments.csv — all players shown in grey")

    # Pre-build event flash lookup:  frame → list of (x, y, event_type)
    flash_lookup: dict[int, list] = defaultdict(list)
    if not events.empty:
        for _, ev in events.iterrows():
            fid = int(ev["frame_id"])
            
            # --- FIX: Handle NaN coordinates gracefully ---
            ex_raw = ev.get("x", 0)
            ey_raw = ev.get("y", 0)
            
            if pd.isna(ex_raw) or pd.isna(ey_raw):
                continue # Skip drawing a flash if there are no physical coordinates
                
            ex = float(ex_raw)
            ey = float(ey_raw)
            et = str(ev.get("event_type", "shot"))
            
            # Register the event for the next EVENT_FLASH_FRAMES frames
            for offset in range(EVENT_FLASH_FRAMES):
                flash_lookup[fid + offset].append((int(ex), int(ey), et))

    # ── Video writer ──────────────────────────────────────────────
    out_vid.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    frames     = sorted(tracks["frame_id"].unique())
    trail_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LEN))

    print(f"  Rendering {len(frames)} frames …")
    for fid in tqdm(frames, desc="Rendering Radar", unit="frame"):
        img          = pitch_base.copy()
        frame_tracks = tracks[tracks["frame_id"] == fid]

        # ── 1. Draw player trails ──────────────────────────────────
        for _, row in frame_tracks.iterrows():
            tid = int(row["track_id"])
            sx = row.get("smooth_x")
            sy = row.get("smooth_y")
            
            # --- FIX: Skip NaN and out-of-canvas positions ---
            if pd.isna(sx) or pd.isna(sy):
                continue
            if not (0 <= sx <= w and 0 <= sy <= h):
                continue
                
            px = int(sx)
            py = int(sy)

            trail_hist[tid].append((px, py))
            if tid != -1:
                color = _get_team_color(tid, team_map)
                pts   = list(trail_hist[tid])
                for k in range(1, len(pts)):
                    alpha     = k / len(pts)
                    tc        = tuple(int(c * alpha * 0.7) for c in color)
                    cv2.line(img, pts[k - 1], pts[k], tc, 1, cv2.LINE_AA)

        # ── 2. Draw players and ball ───────────────────────────────
        for _, row in frame_tracks.iterrows():
            tid = int(row["track_id"])
            sx = row.get("smooth_x")
            sy = row.get("smooth_y")
            
            # --- FIX: Skip NaN and out-of-canvas positions ---
            if pd.isna(sx) or pd.isna(sy):
                continue
            if not (0 <= sx <= w and 0 <= sy <= h):
                continue
                
            px = int(sx)
            py = int(sy)

            if tid == -1:
                # Ball
                cv2.circle(img, (px, py), 7, (0, 0, 0), -1)
                cv2.circle(img, (px, py), 5, BALL_COLOR, -1)
            else:
                color = _get_team_color(tid, team_map)
                cv2.circle(img, (px, py), 9, (0, 0, 0), -1)   # shadow
                cv2.circle(img, (px, py), 8, color, -1)
                cv2.putText(img, str(tid), (px - 8, py - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # ── 3. Event flash markers ────────────────────────────────
        for (ex, ey, etype) in flash_lookup.get(fid, []):
            ec = EVENT_COLORS.get(etype, (255, 255, 255))
            cv2.circle(img, (ex, ey), 18, ec, 2, cv2.LINE_AA)
            cv2.putText(img, etype.upper(), (ex + 20, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ec, 1, cv2.LINE_AA)

        # ── 4. HUD ────────────────────────────────────────────────
        cv2.putText(img, f"Frame {fid:05d}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # Legend (top-left)
        for i, (team, color) in enumerate(TEAM_COLORS.items()):
            if team == "unknown":
                continue
            lx, ly = 10, 20 + i * 18
            cv2.circle(img, (lx + 6, ly), 5, color, -1)
            cv2.putText(img, team, (lx + 14, ly + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (240, 240, 240), 1, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f"\n  ✅  Radar video saved → {out_vid}")
    print(f"  Play with: vlc {out_vid}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Render top-down tactical radar video."
    )
    p.add_argument("--tracks", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--pitch",  required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--teams",  default=None, help="team_assignments.csv (optional)")
    p.add_argument("--fps",    type=int, default=FPS)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_showcase(
        tracks_csv = args.tracks,
        events_csv = args.events,
        pitch_img  = args.pitch,
        out_vid    = args.out,
        teams_csv  = args.teams,
        fps        = args.fps,
    )