"""
broadcast_overlay.py  —  "God Mode" broadcast overlay
──────────────────────────────────────────────────────
Projects the full analytical layer back onto the original broadcast video:
  • Team-colored bounding boxes (using pre-computed team labels)
  • Voronoi pitch-control polygons (warped from pitch space → image space
    using the inverse homography H⁻¹)
  • Live event HUD (shot / possession / pressure annotations)
  • Player velocity vectors (direction of movement)
  • Minimap (2D pitch thumbnail) in the corner

The key mathematical operation is cv2.perspectiveTransform with H_inv,
which takes a polygon defined in 2D pitch coordinates and warps it back
into the perspective of the camera — the inverse of what project_tracks
does.

Usage
─────
  python3 src/goalx/ps2_ml/broadcast_overlay.py \
      --frames-dir    data/raw_videos/tracking/test/SNMOT-116/img1 \
      --tracks        outputs/smoothed/smoothed_tracks.csv \
      --teams         outputs/teams/team_assignments.csv \
      --events        outputs/events/events.csv \
      --pitch-control outputs/pitch_control/pitch_control.csv \
      --homography    data/homography_data.npz \
      --pitch         data/pitch_map.png \
      --out           outputs/broadcast_overlay.mp4 \
      --fps           25
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

# FIFA standard pitch dimensions in metres (used for Voronoi boundary clipping)
PITCH_W_M = 105.0
PITCH_H_M = 68.0

# Team visual palette  (BGR)
TEAM_COLORS: dict[str, tuple] = {
    "home": (230,  60,  60),   # blue-ish
    "away": (60,  200,  60),   # green
    "ref":  (30,  220, 220),   # yellow
    "ball": (0,   180, 255),   # orange
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
TRAIL_LEN = 20


# ─────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────

def _warp_points(pts_2d: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """
    Project an array of 2D pitch-space points into image space.

    pts_2d  : (N, 2) float32 — coordinates in pitch pixel/metre space
    H_inv   : (3, 3) float32 — inverse homography matrix
    Returns : (N, 2) int32   — image pixel coordinates
    """
    pts = pts_2d.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    return warped.astype(np.int32)


def _clip_voronoi_region(vertices: np.ndarray,
                          pitch_w: float, pitch_h: float) -> np.ndarray | None:
    """
    Clip a Voronoi polygon to the pitch boundary using Sutherland-Hodgman.
    Returns clipped polygon or None if the region is entirely outside.
    """
    def _clip_edge(poly, x0, y0, x1, y1):
        """Clip polygon against one edge of the boundary."""
        result = []
        n = len(poly)
        for i in range(n):
            curr = poly[i]
            prev = poly[(i - 1) % n]
            def inside(p):
                return (x1 - x0) * (p[1] - y0) - (y1 - y0) * (p[0] - x0) >= 0
            if inside(curr):
                if not inside(prev):
                    # Compute intersection
                    dx, dy = curr[0]-prev[0], curr[1]-prev[1]
                    ex, ey = x1-x0, y1-y0
                    t = ((x0-prev[0])*ey - (y0-prev[1])*ex) / (dx*ey - dy*ex + 1e-9)
                    result.append([prev[0]+t*dx, prev[1]+t*dy])
                result.append(curr)
            elif inside(prev):
                dx, dy = curr[0]-prev[0], curr[1]-prev[1]
                ex, ey = x1-x0, y1-y0
                t = ((x0-prev[0])*ey - (y0-prev[1])*ex) / (dx*ey - dy*ex + 1e-9)
                result.append([prev[0]+t*dx, prev[1]+t*dy])
        return result

    poly = vertices.tolist()
    # Clip against all 4 pitch boundaries
    poly = _clip_edge(poly, 0, 0, pitch_w, 0)              # bottom
    poly = _clip_edge(poly, pitch_w, 0, pitch_w, pitch_h)  # right
    poly = _clip_edge(poly, pitch_w, pitch_h, 0, pitch_h)  # top
    poly = _clip_edge(poly, 0, pitch_h, 0, 0)              # left

    if len(poly) < 3:
        return None
    return np.array(poly, dtype=np.float32)


def _build_voronoi_regions(positions: np.ndarray,
                            pitch_w: float, pitch_h: float
                            ) -> list[np.ndarray | None]:
    """
    Build clipped Voronoi regions for a set of player positions in pitch space.

    Returns one clipped polygon per input position (None if outside pitch).
    """
    if len(positions) < 4:
        return [None] * len(positions)

    # Add boundary mirror points to avoid unbounded regions
    mirrored = np.vstack([
        positions,
        positions * [-1, 1] + [0, 0],
        positions * [1, -1] + [0, 0],
        positions * [-1, -1],
        [[pitch_w * 2, pitch_h / 2], [-pitch_w, pitch_h / 2],
         [pitch_w / 2, pitch_h * 2], [pitch_w / 2, -pitch_h]],
    ])

    try:
        vor = Voronoi(mirrored)
    except Exception:
        return [None] * len(positions)

    regions = []
    for i in range(len(positions)):
        region_idx = vor.point_region[i]
        region     = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            regions.append(None)
            continue
        verts = vor.vertices[region]
        clipped = _clip_voronoi_region(verts, pitch_w, pitch_h)
        regions.append(clipped)

    return regions


# ─────────────────────────────────────────────────────────────────
#  HUD renderer
# ─────────────────────────────────────────────────────────────────

def _draw_hud(img: np.ndarray, frame_id: int,
              events_at_frame: pd.DataFrame,
              home_poss: float, away_poss: float) -> None:
    """
    Draw the top-right info panel: possession bar + recent events.
    """
    h, w = img.shape[:2]
    panel_w, panel_h = 260, 90

    # Semi-transparent dark panel
    overlay = img.copy()
    cv2.rectangle(overlay, (w - panel_w - 10, 10),
                  (w - 10, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Possession bar
    bar_x = w - panel_w - 2
    bar_y = 18
    bar_w = panel_w - 16
    bar_h = 16
    home_w = int(bar_w * home_poss)
    away_w = bar_w - home_w

    cv2.rectangle(img, (bar_x, bar_y),
                  (bar_x + home_w, bar_y + bar_h), TEAM_COLORS["home"], -1)
    cv2.rectangle(img, (bar_x + home_w, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), TEAM_COLORS["away"], -1)

    cv2.putText(img, f"{home_poss*100:.0f}%",
                (bar_x + 4, bar_y + bar_h - 3),
                FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{away_poss*100:.0f}%",
                (bar_x + home_w + 4, bar_y + bar_h - 3),
                FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Recent events
    y_evt = bar_y + bar_h + 16
    for _, evt in events_at_frame.iterrows():
        label = f"{evt.get('event_type', '?').upper()}  t={int(evt.get('frame_id', 0))}"
        cv2.putText(img, label, (bar_x, y_evt),
                    FONT, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        y_evt += 16

    # Frame counter
    cv2.putText(img, f"frame {frame_id:06d}",
                (w - panel_w - 2, h - 10),
                FONT, 0.40, (160, 160, 160), 1, cv2.LINE_AA)


def _draw_minimap(img: np.ndarray, pitch_img: np.ndarray,
                  frame_tracks: pd.DataFrame,
                  team_map: dict[int, str]) -> None:
    """
    Draw a small 2D pitch minimap in the bottom-left corner with
    player dots color-coded by team.
    """
    h, w = img.shape[:2]
    map_w, map_h = 180, 110
    margin       = 10

    mini = cv2.resize(pitch_img, (map_w, map_h))

    for _, row in frame_tracks.iterrows():
        # Map smooth coordinates (meters) directly to the minimap dimensions
        if "smooth_x" not in row or pd.isna(row["smooth_x"]):
            continue
            
        mx = int((row["smooth_x"] / PITCH_W_M) * map_w)
        my = int((row["smooth_y"] / PITCH_H_M) * map_h)
        
        # Keep dots within bounds just in case
        mx = max(0, min(mx, map_w - 1))
        my = max(0, min(my, map_h - 1))
        
        tid  = int(row["track_id"])
        
        if tid == -1:
            color = TEAM_COLORS["ball"]
            cv2.circle(mini, (mx, my), 5, (255, 255, 255), -1) # White outline for ball
            cv2.circle(mini, (mx, my), 4, color, -1)
        else:
            team = team_map.get(tid, "away")
            color = TEAM_COLORS.get(team, (200, 200, 200))
            cv2.circle(mini, (mx, my), 4, color, -1)

    # Paste minimap with semi-transparent border
    x0 = margin
    y0 = h - map_h - margin
    overlay = img.copy()
    overlay[y0:y0 + map_h, x0:x0 + map_w] = mini
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + map_w, y0 + map_h),
                  (200, 200, 200), 1)


# ─────────────────────────────────────────────────────────────────
#  Main overlay class
# ─────────────────────────────────────────────────────────────────

class BroadcastOverlay:
    def __init__(self, frames_dir: Path, tracks_csv: Path,
                 teams_csv: Path, events_csv: Path,
                 pitch_control_csv: Path, homography_npz: Path,
                 pitch_img_path: Path, out_path: Path, fps: int):
        self.frames_dir       = frames_dir
        self.tracks_csv       = tracks_csv
        self.teams_csv        = teams_csv
        self.events_csv       = events_csv
        self.pitch_control_csv = pitch_control_csv
        self.homography_npz   = homography_npz
        self.pitch_img_path   = pitch_img_path
        self.out_path         = out_path
        self.fps              = fps

    # ──────────────────────────────────────────
    #  Load data
    # ──────────────────────────────────────────

    def _load(self):
        data = np.load(str(self.homography_npz))
        H     = data["H"].astype(np.float32)
        self.H_inv = np.linalg.inv(H)          # pitch → image

        self.tracks = pd.read_csv(self.tracks_csv)
        self.events = pd.read_csv(self.events_csv)
        self.pitch_img = cv2.imread(str(self.pitch_img_path))

        # Build track_id → team label map
        teams_df = pd.read_csv(self.teams_csv)
        self.team_map: dict[int, str] = dict(
            zip(teams_df["track_id"].astype(int),
                teams_df["team"].astype(str))
        )

        # Pitch control: frame_id → {home_poss, away_poss}
        self.pc = {}
        if self.pitch_control_csv.exists():
            pc_df = pd.read_csv(self.pitch_control_csv)
            for _, row in pc_df.iterrows():
                self.pc[int(row["frame_id"])] = {
                    "home": float(row.get("home_pct", 50.0)) / 100.0,
                    "away": float(row.get("away_pct", 50.0)) / 100.0,
                }

        # Read pitch image dimensions to get scale factor for Voronoi
        if self.pitch_img is not None:
            self.pitch_h_px, self.pitch_w_px = self.pitch_img.shape[:2]
        else:
            self.pitch_w_px, self.pitch_h_px = 1050, 680

        # Scale from pitch pixels → metres
        self.scale_x = PITCH_W_M / self.pitch_w_px
        self.scale_y = PITCH_H_M / self.pitch_h_px

        print(f"  H_inv computed. Pitch scale: "
              f"{self.scale_x:.4f} m/px × {self.scale_y:.4f} m/px")

    # ──────────────────────────────────────────
    #  Per-frame rendering
    # ──────────────────────────────────────────

    def _render_frame(self, img: np.ndarray, frame_id: int,
                      frame_tracks: pd.DataFrame,
                      trail_history: dict[int, deque]) -> np.ndarray:
        h_img, w_img = img.shape[:2]

        # ── 1. Voronoi overlay ─────────────────────────────────
        # Separate home and away tracks (exclude ball track_id = -1)
        players = frame_tracks[(frame_tracks["track_id"] >= 0) & (frame_tracks["smooth_x"].notna())]
        
        if len(players) >= 4 and "smooth_x" in players.columns:
            # Positions are already in meters, so just extract them
            pos_m = players[["smooth_x", "smooth_y"]].values.astype(np.float32)
            
            regions = _build_voronoi_regions(pos_m, PITCH_W_M, PITCH_H_M)

            for idx, (_, row) in enumerate(players.iterrows()):
                poly_m = regions[idx]
                if poly_m is None:
                    continue

                tid  = int(row["track_id"])
                team = self.team_map.get(tid, "away")

                # Scale metres back to pitch pixels for warping via H_inv
                poly_px = poly_m.copy()
                poly_px[:, 0] /= self.scale_x 
                poly_px[:, 1] /= self.scale_y 

                # Warp polygon from pitch space → image space using H_inv
                poly_img = _warp_points(poly_px, self.H_inv)

                # Clip to image bounds
                poly_img[:, 0] = np.clip(poly_img[:, 0], 0, w_img - 1)
                poly_img[:, 1] = np.clip(poly_img[:, 1], 0, h_img - 1)

                color = TEAM_COLORS.get(team, (128, 128, 128))

                # Semi-transparent fill
                overlay = img.copy()
                cv2.fillPoly(overlay, [poly_img], color)
                cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

                # Polygon border
                cv2.polylines(img, [poly_img], True, color, 1, cv2.LINE_AA)

        # ── 2. Player bounding boxes + trails ──────────────────
        for _, row in players.iterrows():
            tid  = int(row["track_id"])
            team = self.team_map.get(tid, "away")
            color = TEAM_COLORS.get(team, (200, 200, 200))

            # If we have image-space coordinates from original tracking
            if all(c in row.index for c in ["x1", "y1", "x2", "y2"]) and not pd.isna(row["x1"]):
                x1 = int(np.clip(row["x1"], 0, w_img - 1))
                y1 = int(np.clip(row["y1"], 0, h_img - 1))
                x2 = int(np.clip(row["x2"], 0, w_img - 1))
                y2 = int(np.clip(row["y2"], 0, h_img - 1))
            else:
                # Fall back: project pitch coords back to image (convert meters to pitch pixels first)
                foot = np.array([[[row["smooth_x"] / self.scale_x, row["smooth_y"] / self.scale_y]]], dtype=np.float32)
                img_pt = cv2.perspectiveTransform(foot, self.H_inv)[0, 0]
                cx, cy = int(img_pt[0]), int(img_pt[1])
                x1, y1, x2, y2 = cx - 20, cy - 60, cx + 20, cy

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Team label + ID
            label = f"{team[0].upper()}{tid}"
            cv2.putText(img, label, (x1, y1 - 6),
                        FONT, 0.5, color, 2, cv2.LINE_AA)

            # Velocity vector: arrow from foot to next predicted position
            foot_pt = ((x1 + x2) // 2, y2)
            trail_history[tid].append(foot_pt)
            pts = list(trail_history[tid])
            if len(pts) >= 2:
                # Draw trail
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    t_color = tuple(int(c * alpha) for c in color)
                    cv2.line(img, pts[k-1], pts[k], t_color,
                             max(1, int(alpha * 2)), cv2.LINE_AA)
                # Velocity arrow (last 5-frame displacement)
                if len(pts) >= 5:
                    dx = pts[-1][0] - pts[-5][0]
                    dy = pts[-1][1] - pts[-5][1]
                    end = (foot_pt[0] + int(dx * 1.5),
                           foot_pt[1] + int(dy * 1.5))
                    cv2.arrowedLine(img, foot_pt, end, color, 2,
                                    tipLength=0.35)

        # ── 3. Ball marker ─────────────────────────────────────
        ball_rows = frame_tracks[frame_tracks["track_id"] == -1]
        for _, brow in ball_rows.iterrows():
            if "smooth_x" in brow.index and not pd.isna(brow["smooth_x"]):
                # Convert meters back to pitch pixels for H_inv
                bp = np.array([[[brow["smooth_x"] / self.scale_x, brow["smooth_y"] / self.scale_y]]], np.float32)
                bimg = cv2.perspectiveTransform(bp, self.H_inv)[0, 0].astype(int)
                bx, by = int(np.clip(bimg[0], 0, w_img-1)), \
                         int(np.clip(bimg[1], 0, h_img-1))
                cv2.circle(img, (bx, by), 8, TEAM_COLORS["ball"], -1)
                cv2.circle(img, (bx, by), 9, (255, 255, 255), 1)

        # ── 4. HUD ─────────────────────────────────────────────
        pc     = self.pc.get(frame_id, {"home": 0.5, "away": 0.5})
        recent = self.events[
            (self.events["frame_id"] >= frame_id - 60) &
            (self.events["frame_id"] <= frame_id)
        ].tail(3)
        _draw_hud(img, frame_id, recent, pc["home"], pc["away"])

        # ── 5. Minimap ─────────────────────────────────────────
        if self.pitch_img is not None:
            _draw_minimap(img, self.pitch_img, frame_tracks, self.team_map)

        return img

    # ──────────────────────────────────────────
    #  Run
    # ──────────────────────────────────────────

    def run(self) -> None:
        print(f"\n  goalX — Broadcast Overlay (God Mode)")
        print(f"  {'─'*45}\n")

        self._load()

        frame_ids = sorted(self.tracks["frame_id"].unique())

        # Read image dimensions
        first_img = cv2.imread(
            str(self.frames_dir / f"{int(frame_ids[0]):06d}.jpg")
        )
        if first_img is None:
            raise FileNotFoundError(f"Cannot read first frame. Check path: {self.frames_dir}")
        h_img, w_img = first_img.shape[:2]

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.out_path), fourcc,
                                 self.fps, (w_img, h_img))

        trail_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=TRAIL_LEN)
        )

        for fid in tqdm(frame_ids, desc="Rendering overlay"):
            img_path = self.frames_dir / f"{int(fid):06d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            frame_tracks = self.tracks[self.tracks["frame_id"] == fid]
            img = self._render_frame(img, fid, frame_tracks, trail_history)
            writer.write(img)

        writer.release()
        print(f"\n  ✅  Overlay video saved → {self.out_path}")
        print(f"  Frames rendered: {len(frame_ids)}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="God Mode broadcast overlay for goalX."
    )
    p.add_argument("--frames-dir",    required=True)
    p.add_argument("--tracks",        required=True)
    p.add_argument("--teams",         required=True)
    p.add_argument("--events",        required=True)
    p.add_argument("--pitch-control", required=True)
    p.add_argument("--homography",    required=True)
    p.add_argument("--pitch",         required=True)
    p.add_argument("--out",           default="outputs/broadcast_overlay.mp4")
    p.add_argument("--fps",           type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    BroadcastOverlay(
        frames_dir       = Path(args.frames_dir),
        tracks_csv       = Path(args.tracks),
        teams_csv        = Path(args.teams),
        events_csv       = Path(args.events),
        pitch_control_csv = Path(args.pitch_control),
        homography_npz   = Path(args.homography),
        pitch_img_path   = Path(args.pitch),
        out_path         = Path(args.out),
        fps              = args.fps,
    ).run()