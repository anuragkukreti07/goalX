"""
broadcast_overlay.py  —  "God Mode" broadcast overlay
──────────────────────────────────────────────────────
Projects the full analytical layer back onto the original broadcast video:
  • Team-coloured bounding boxes
  • Voronoi pitch-control polygons (warped from pitch pixels → image space
    using the inverse homography H⁻¹)
  • Live event HUD (shot / possession / pressure)
  • Player velocity vectors + motion trails
  • Minimap (2D pitch thumbnail) in the corner
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

# Team visual palette  (BGR)
TEAM_COLORS: dict[str, tuple] = {
    "home": (230,  60,  60),
    "away": (60,  200,  60),
    "other":(30,  220, 220),
    "ball": (0,   180, 255),
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
TRAIL_LEN = 20


# ─────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────

def _warp_points(pts_px: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """
    Warp pitch-pixel coordinates → image-pixel coordinates via H_inv.
    """
    pts    = pts_px.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    return warped.astype(np.int32)


def _sutherland_hodgman(poly: list, x0, y0, x1, y1) -> list:
    """Clip polygon against a single half-plane edge."""
    result = []
    n      = len(poly)
    for i in range(n):
        curr = poly[i]
        prev = poly[(i - 1) % n]

        def inside(p):
            return (x1 - x0) * (p[1] - y0) - (y1 - y0) * (p[0] - x0) >= 0

        if inside(curr):
            if not inside(prev):
                dx, dy = curr[0] - prev[0], curr[1] - prev[1]
                ex, ey = x1 - x0, y1 - y0
                denom  = dx * ey - dy * ex
                if abs(denom) > 1e-9:
                    t = ((x0 - prev[0]) * ey - (y0 - prev[1]) * ex) / denom
                    result.append([prev[0] + t * dx, prev[1] + t * dy])
            result.append(curr)
        elif inside(prev):
            dx, dy = curr[0] - prev[0], curr[1] - prev[1]
            ex, ey = x1 - x0, y1 - y0
            denom  = dx * ey - dy * ex
            if abs(denom) > 1e-9:
                t = ((x0 - prev[0]) * ey - (y0 - prev[1]) * ex) / denom
                result.append([prev[0] + t * dx, prev[1] + t * dy])
    return result


def _clip_to_pitch(vertices: np.ndarray,
                   pitch_w: float, pitch_h: float) -> np.ndarray | None:
    """Sutherland-Hodgman clip to pitch rectangle.  Returns None if empty."""
    poly = vertices.tolist()
    poly = _sutherland_hodgman(poly, 0,       0,       pitch_w, 0)
    poly = _sutherland_hodgman(poly, pitch_w, 0,       pitch_w, pitch_h)
    poly = _sutherland_hodgman(poly, pitch_w, pitch_h, 0,       pitch_h)
    poly = _sutherland_hodgman(poly, 0,       pitch_h, 0,       0)
    return np.array(poly, dtype=np.float32) if len(poly) >= 3 else None


def _build_voronoi_regions(
    positions_px: np.ndarray,   # (N, 2) — pitch-PIXEL coords
    pitch_w_px:   float,
    pitch_h_px:   float,
) -> list[np.ndarray | None]:
    """Build Voronoi regions in pitch-pixel space, clipped to pitch boundary."""
    n = len(positions_px)
    if n < 4:
        return [None] * n

    # Mirror guards to bound all infinite regions
    margin = max(pitch_w_px, pitch_h_px) * 3
    guards = np.array([
        [-margin,              pitch_h_px / 2],
        [pitch_w_px + margin,  pitch_h_px / 2],
        [pitch_w_px / 2,       -margin],
        [pitch_w_px / 2,       pitch_h_px + margin],
    ])
    all_pts = np.vstack([positions_px, guards])

    try:
        vor = Voronoi(all_pts)
    except Exception:
        return [None] * n

    regions = []
    for i in range(n):
        region_idx = vor.point_region[i]
        region     = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            regions.append(None)
            continue
        verts   = vor.vertices[region]
        clipped = _clip_to_pitch(verts, pitch_w_px, pitch_h_px)
        regions.append(clipped)

    return regions


# ─────────────────────────────────────────────────────────────────
#  HUD renderer
# ─────────────────────────────────────────────────────────────────

def _draw_hud(img: np.ndarray, frame_id: int,
              recent_events: pd.DataFrame,
              home_poss: float, away_poss: float) -> None:
    h, w     = img.shape[:2]
    panel_w  = 270
    panel_h  = 90 + max(0, len(recent_events) - 1) * 16

    overlay = img.copy()
    cv2.rectangle(overlay,
                  (w - panel_w - 10, 10),
                  (w - 10, 10 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    bar_x = w - panel_w
    bar_y = 18
    bar_w = panel_w - 16

    home_w = int(bar_w * max(0.0, min(1.0, home_poss)))
    cv2.rectangle(img, (bar_x, bar_y),
                  (bar_x + home_w, bar_y + 16), TEAM_COLORS["home"], -1)
    cv2.rectangle(img, (bar_x + home_w, bar_y),
                  (bar_x + bar_w, bar_y + 16), TEAM_COLORS["away"], -1)
    cv2.putText(img, f"H {home_poss*100:.0f}%",
                (bar_x + 4, bar_y + 13), FONT, 0.42, (255, 255, 255), 1)
    cv2.putText(img, f"A {away_poss*100:.0f}%",
                (bar_x + home_w + 4, bar_y + 13), FONT, 0.42, (255, 255, 255), 1)

    y = bar_y + 30
    for _, evt in recent_events.iterrows():
        label = f"{str(evt.get('event_type','?')).upper()}  f={int(evt.get('frame_id',0))}"
        cv2.putText(img, label, (bar_x, y), FONT, 0.4, (0, 220, 255), 1)
        y += 16

    cv2.putText(img, f"frame {frame_id:06d}",
                (w - panel_w, h - 10), FONT, 0.38, (150, 150, 150), 1)


def _draw_minimap(img: np.ndarray,
                  pitch_img: np.ndarray,
                  frame_tracks: pd.DataFrame,
                  team_map: dict[int, str],
                  pitch_w_px: float,
                  pitch_h_px: float) -> None:
    """
    Minimap in bottom-left corner.
    Coordinates are pitch PIXELS → scaled to minimap dimensions.
    """
    h, w   = img.shape[:2]
    map_w  = 190
    map_h  = 120
    margin = 10

    mini = cv2.resize(pitch_img, (map_w, map_h))

    for _, row in frame_tracks.iterrows():
        sx = row.get("smooth_x")
        sy = row.get("smooth_y")
        if sx is None or pd.isna(sx):
            continue

        # Scale from pitch pixels → minimap pixels
        mx = int((float(sx) / pitch_w_px) * map_w)
        my = int((float(sy) / pitch_h_px) * map_h)
        mx = int(np.clip(mx, 0, map_w - 1))
        my = int(np.clip(my, 0, map_h - 1))

        tid = int(row["track_id"])
        if tid == -1:
            cv2.circle(mini, (mx, my), 5, (255, 255, 255), -1)
            cv2.circle(mini, (mx, my), 4, TEAM_COLORS["ball"], -1)
        else:
            team  = team_map.get(tid, "away")
            color = TEAM_COLORS.get(team, (200, 200, 200))
            cv2.circle(mini, (mx, my), 4, color, -1)

    x0 = margin
    y0 = h - map_h - margin
    overlay = img.copy()
    overlay[y0:y0 + map_h, x0:x0 + map_w] = mini
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + map_w, y0 + map_h),
                  (200, 200, 200), 1)


# ─────────────────────────────────────────────────────────────────
#  Main overlay class
# ─────────────────────────────────────────────────────────────────

class BroadcastOverlay:
    def __init__(self, frames_dir, tracks_csv, teams_csv, events_csv,
                 pitch_control_csv, homography_npz, pitch_img_path,
                 out_path, fps):
        self.frames_dir        = Path(frames_dir)
        self.tracks_csv        = Path(tracks_csv)
        self.teams_csv         = Path(teams_csv)
        self.events_csv        = Path(events_csv)
        self.pitch_control_csv = Path(pitch_control_csv)
        self.homography_npz    = Path(homography_npz)
        self.pitch_img_path    = Path(pitch_img_path)
        self.out_path          = Path(out_path)
        self.fps               = fps

    def _load(self) -> None:
        data      = np.load(str(self.homography_npz))
        H         = data["H"].astype(np.float32)
        self.H_inv = np.linalg.inv(H)   # pitch pixels → image pixels

        self.tracks    = pd.read_csv(self.tracks_csv)
        self.events    = pd.read_csv(self.events_csv) \
                         if self.events_csv.exists() else pd.DataFrame()
        self.pitch_img = cv2.imread(str(self.pitch_img_path))
        if self.pitch_img is None:
            raise FileNotFoundError(f"Cannot read pitch image: {self.pitch_img_path}")

        self.pitch_h_px, self.pitch_w_px = self.pitch_img.shape[:2]

        teams_df       = pd.read_csv(self.teams_csv)
        self.team_map  = dict(zip(teams_df["track_id"].astype(int),
                                  teams_df["team"].astype(str)))

        self.pc: dict[int, dict] = {}
        if self.pitch_control_csv.exists():
            pc_df = pd.read_csv(self.pitch_control_csv)
            for _, row in pc_df.iterrows():
                self.pc[int(row["frame_id"])] = {
                    "home": float(row.get("home_pct", 50.0)) / 100.0,
                    "away": float(row.get("away_pct", 50.0)) / 100.0,
                }

        print(f"  Pitch canvas: {self.pitch_w_px}×{self.pitch_h_px} px")
        print(f"  H_inv computed. Tracks: {len(self.tracks)}")

    def _render_frame(self, img: np.ndarray, frame_id: int,
                      frame_tracks: pd.DataFrame,
                      trail_history: dict) -> np.ndarray:
        h_img, w_img = img.shape[:2]

        players = frame_tracks[
            (frame_tracks["track_id"] >= 0) &
            frame_tracks["smooth_x"].notna()
        ]

        # ── 1. Voronoi overlay (pitch-pixel space → H_inv → image) ──
        if len(players) >= 4:
            pos_px = players[["smooth_x", "smooth_y"]].values.astype(np.float32)
            regions = _build_voronoi_regions(
                pos_px, self.pitch_w_px, self.pitch_h_px
            )

            for idx, (_, row) in enumerate(players.iterrows()):
                poly_pitch = regions[idx]
                if poly_pitch is None:
                    continue

                # Warp Voronoi polygon from pitch-pixel space to image space
                poly_img = _warp_points(poly_pitch, self.H_inv)
                poly_img[:, 0] = np.clip(poly_img[:, 0], 0, w_img - 1)
                poly_img[:, 1] = np.clip(poly_img[:, 1], 0, h_img - 1)

                tid   = int(row["track_id"])
                team  = self.team_map.get(tid, "away")
                color = TEAM_COLORS.get(team, (128, 128, 128))

                overlay = img.copy()
                cv2.fillPoly(overlay, [poly_img], color)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
                cv2.polylines(img, [poly_img], True, color, 1, cv2.LINE_AA)

        # ── 2. Players — boxes + trails + velocity arrows ────────
        for _, row in players.iterrows():
            tid   = int(row["track_id"])
            team  = self.team_map.get(tid, "away")
            color = TEAM_COLORS.get(team, (200, 200, 200))

            # Image-space bounding box
            if all(c in row.index for c in ["x1", "y1", "x2", "y2"]) \
                    and not pd.isna(row["x1"]):
                x1 = int(np.clip(row["x1"], 0, w_img - 1))
                y1 = int(np.clip(row["y1"], 0, h_img - 1))
                x2 = int(np.clip(row["x2"], 0, w_img - 1))
                y2 = int(np.clip(row["y2"], 0, h_img - 1))
            else:
                # Fallback: project pitch-pixel foot through H_inv
                foot = np.array([[[row["smooth_x"], row["smooth_y"]]]],
                                dtype=np.float32)
                pt   = cv2.perspectiveTransform(foot, self.H_inv)[0, 0]
                cx, cy = int(pt[0]), int(pt[1])
                x1, y1, x2, y2 = cx - 20, cy - 60, cx + 20, cy

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{team[0].upper()}{tid}",
                        (x1, y1 - 6), FONT, 0.45, color, 2, cv2.LINE_AA)

            # Trail + velocity
            foot_pt = ((x1 + x2) // 2, y2)
            trail_history[tid].append(foot_pt)
            pts = list(trail_history[tid])
            for k in range(1, len(pts)):
                alpha   = k / len(pts)
                tc      = tuple(int(c * alpha * 0.8) for c in color)
                cv2.line(img, pts[k - 1], pts[k], tc,
                         max(1, int(alpha * 2)), cv2.LINE_AA)
            if len(pts) >= 5:
                dx = pts[-1][0] - pts[-5][0]
                dy = pts[-1][1] - pts[-5][1]
                end = (foot_pt[0] + int(dx * 1.5), foot_pt[1] + int(dy * 1.5))
                cv2.arrowedLine(img, foot_pt, end, color, 2,
                                tipLength=0.35, line_type=cv2.LINE_AA)

        # ── 3. Ball ───────────────────────────────────────────────
        ball_rows = frame_tracks[frame_tracks["track_id"] == -1]
        for _, brow in ball_rows.iterrows():
            if pd.isna(brow.get("smooth_x")):
                continue
            # Project ball foot from pitch pixels → image pixels
            bp  = np.array([[[brow["smooth_x"], brow["smooth_y"]]]],
                            dtype=np.float32)
            bpt = cv2.perspectiveTransform(bp, self.H_inv)[0, 0]
            bx  = int(np.clip(bpt[0], 0, w_img - 1))
            by  = int(np.clip(bpt[1], 0, h_img - 1))
            cv2.circle(img, (bx, by), 9, (255, 255, 255), -1)
            cv2.circle(img, (bx, by), 7, TEAM_COLORS["ball"], -1)

        # ── 4. HUD ────────────────────────────────────────────────
        pc     = self.pc.get(frame_id, {"home": 0.5, "away": 0.5})
        recent = self.events[
            (self.events["frame_id"] >= frame_id - 75) &
            (self.events["frame_id"] <= frame_id)
        ].tail(3) if not self.events.empty else pd.DataFrame()
        _draw_hud(img, frame_id, recent, pc["home"], pc["away"])

        # ── 5. Minimap ────────────────────────────────────────────
        _draw_minimap(img, self.pitch_img, frame_tracks, self.team_map,
                      self.pitch_w_px, self.pitch_h_px)

        return img

    def run(self) -> None:
        print(f"\n  goalX — Broadcast Overlay")
        print(f"  {'─' * 45}\n")

        self._load()

        frame_ids = sorted(self.tracks["frame_id"].unique())
        first_img = cv2.imread(
            str(self.frames_dir / f"{int(frame_ids[0]):06d}.jpg")
        )
        if first_img is None:
            raise FileNotFoundError(
                f"Cannot read first broadcast frame. Check: {self.frames_dir}"
            )
        h_img, w_img = first_img.shape[:2]

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps, (w_img, h_img),
        )

        trail_history: dict = defaultdict(lambda: deque(maxlen=TRAIL_LEN))

        for fid in tqdm(frame_ids, desc="Rendering overlay", unit="frame"):
            img_path = self.frames_dir / f"{int(fid):06d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            frame_tracks = self.tracks[self.tracks["frame_id"] == fid]
            img = self._render_frame(img, fid, frame_tracks, trail_history)
            writer.write(img)

        writer.release()
        print(f"\n  ✅  Overlay video → {self.out_path}")
        print(f"  Frames: {len(frame_ids)}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="God Mode broadcast overlay.")
    p.add_argument("--frames-dir",    required=True)
    p.add_argument("--tracks",        required=True)
    p.add_argument("--teams",         required=True)
    p.add_argument("--events",        default="outputs/events/events.csv")
    p.add_argument("--pitch-control", required=True)
    p.add_argument("--homography",    required=True)
    # FIX: Provide a default so it doesn't crash when run_goalx.py omits it
    p.add_argument("--pitch",         default="data/pitch_map.png")
    p.add_argument("--out",           default="outputs/broadcast_overlay.mp4")
    p.add_argument("--fps",           type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    BroadcastOverlay(
        frames_dir        = args.frames_dir,
        tracks_csv        = args.tracks,
        teams_csv         = args.teams,
        events_csv        = args.events,
        pitch_control_csv = args.pitch_control,
        homography_npz    = args.homography,
        pitch_img_path    = args.pitch,
        out_path          = args.out,
        fps               = args.fps,
    ).run()