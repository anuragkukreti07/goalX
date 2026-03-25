"""
project_tracks.py
─────────────────
Applies the pre-computed Homography matrix H to project tracked player
positions from image space onto the 2D metric pitch canvas.

Input
─────
  --tracks       CSV produced by track_players.py
                 Expected columns: frame_id, track_id, x1, y1, x2, y2
                 (bounding box corners in image pixels)

  --homography   .npz produced by homography_picker.py
                 Must contain key 'H' — a (3,3) float32 matrix.

  --pitch        Path to the 2D pitch PNG (from draw_pitch.py)
  --out-dir      Directory to write annotated frames and projected CSV

Output
──────
  <out-dir>/projected_tracks.csv
      Columns: frame_id, track_id, img_x, img_y, pitch_x, pitch_y

  <out-dir>/frames/<frame_id>.jpg
      2D pitch canvas with projected player dots for each frame.

Usage
─────
  python3 src/goalx/ps1_cv/project_tracks.py \\
      --tracks      outputs/tracks.csv \\
      --homography  data/homography_data.npz \\
      --pitch       data/pitch_map.png \\
      --out-dir     outputs/projected
"""

import cv2
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
#  Colour palette  (track_id → BGR)
# ─────────────────────────────────────────────────────────────────

_PALETTE = [
    (255,  80,  80), (80,  255,  80), (80,  80,  255),
    (255, 200,  80), (200,  80, 255), (80,  200, 255),
    (255, 120, 200), (120, 255, 200), (200, 255, 120),
    (180, 180,  60), (60,  180, 180), (180,  60, 180),
]

def _color_for(track_id: int):
    return _PALETTE[int(track_id) % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────
#  Core projection function
# ─────────────────────────────────────────────────────────────────

def project_footprint(bbox: np.ndarray, H: np.ndarray) -> tuple[float, float]:
    """
    Project the bottom-centre of a bounding box through H.

    The bottom-centre is the best single-pixel approximation for a
    player's ground contact point, which is what homography maps
    correctly (homography is valid for the ground plane only).

    Parameters
    ──────────
    bbox : (4,) array  [x1, y1, x2, y2]  in image pixels
    H    : (3, 3) homography matrix

    Returns
    ───────
    (pitch_x, pitch_y) in pitch-canvas pixels
    """
    x1, y1, x2, y2 = bbox
    foot_x = (x1 + x2) / 2.0       # horizontal centre
    foot_y = float(y2)              # bottom of box = ground contact

    src = np.array([[[foot_x, foot_y]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)  # shape (1, 1, 2)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


# ─────────────────────────────────────────────────────────────────
#  Annotator
# ─────────────────────────────────────────────────────────────────

class TrackProjector:
    def __init__(self, tracks_path, homography_path, pitch_path, out_dir):
        self.tracks_path     = tracks_path
        self.homography_path = homography_path
        self.pitch_path      = pitch_path
        self.out_dir         = Path(out_dir)

        self.H:     np.ndarray | None = None
        self.df:    pd.DataFrame | None = None

    # ──────────────────────────────────────────
    #  Load
    # ──────────────────────────────────────────

    def _load_homography(self) -> None:
        data = np.load(self.homography_path)
        self.H = data["H"].astype(np.float32)
        inliers = int(np.sum(data["status"])) if "status" in data else "?"
        print(f"  ✔  Loaded H from {self.homography_path}  (RANSAC inliers: {inliers})")

    def _load_tracks(self) -> None:
        self.df = pd.read_csv(self.tracks_path)
        rename_map = {}
        if "frame" in self.df.columns: rename_map["frame"] = "frame_id"
        if "id" in self.df.columns: rename_map["id"] = "track_id"
        if rename_map:
            self.df.rename(columns=rename_map, inplace=True)

        required = {"frame_id", "track_id", "x1", "y1", "x2", "y2"}
        missing  = required - set(self.df.columns)
        
        if missing:
            # Print available columns so we know exactly what went wrong if it fails again
            raise ValueError(f"CSV is missing columns: {missing}\nAvailable columns in your CSV: {list(self.df.columns)}")
            
        print(f"  ✔  Loaded {len(self.df)} detections across "
              f"{self.df['frame_id'].nunique()} frames.")

    def _load_pitch_template(self) -> np.ndarray:
        img = cv2.imread(str(self.pitch_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read pitch: {self.pitch_path}")
        return img

    # ──────────────────────────────────────────
    #  Project all tracks
    # ──────────────────────────────────────────

    def _project_all(self) -> pd.DataFrame:
        """Vectorised projection of all bounding box footprints."""
        bboxes = self.df[["x1", "y1", "x2", "y2"]].values.astype(np.float32)

        # Foot points: (N, 2)
        foot_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        foot_y = bboxes[:, 3]
        feet   = np.stack([foot_x, foot_y], axis=1).reshape(-1, 1, 2)

        # Project all at once
        projected = cv2.perspectiveTransform(feet, self.H).reshape(-1, 2)

        result = self.df[["frame_id", "track_id"]].copy()
        result["img_x"]   = foot_x
        result["img_y"]   = foot_y
        result["pitch_x"] = projected[:, 0]
        result["pitch_y"] = projected[:, 1]

        return result

    # ──────────────────────────────────────────
    #  Visualise per-frame
    # ──────────────────────────────────────────

    def _draw_frame(self, frame_df: pd.DataFrame,
                    pitch_template: np.ndarray) -> np.ndarray:
        """
        Draw all projected player dots for a single frame onto a
        fresh copy of the pitch canvas.
        """
        canvas = pitch_template.copy()
        h_c, w_c = canvas.shape[:2]

        for _, row in frame_df.iterrows():
            px = int(round(row["pitch_x"]))
            py = int(round(row["pitch_y"]))

            # Skip projections that fall outside the canvas
            if not (0 <= px < w_c and 0 <= py < h_c):
                continue

            color = _color_for(row["track_id"])
            cv2.circle(canvas, (px, py), 8, color, -1)
            cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(canvas, str(int(row["track_id"])),
                        (px + 10, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2, cv2.LINE_AA)

        # Frame ID watermark (bottom-left)
        frame_id = int(frame_df.iloc[0]["frame_id"])
        cv2.putText(canvas, f"frame {frame_id:05d}",
                    (10, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        return canvas

    # ──────────────────────────────────────────
    #  Public entry-point
    # ──────────────────────────────────────────

    def run(self) -> None:
        print("\n  goalX — Track Projector")
        print("  " + "─" * 40)

        self._load_homography()
        self._load_tracks()

        pitch_template = self._load_pitch_template()

        # --- Project all detections ---
        print("  Projecting footprints through H …")
        projected_df = self._project_all()

        # --- Save projected CSV ---
        csv_out = self.out_dir / "projected_tracks.csv"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        projected_df.to_csv(csv_out, index=False)
        print(f"  💾 Saved projected CSV → {csv_out}")

        # --- Render per-frame annotated pitch images ---
        frames_dir = self.out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        unique_frames = sorted(projected_df["frame_id"].unique())
        print(f"  Rendering {len(unique_frames)} annotated pitch frames …")

        for fid in unique_frames:
            fdata  = projected_df[projected_df["frame_id"] == fid]
            canvas = self._draw_frame(fdata, pitch_template)
            out_p  = frames_dir / f"{int(fid):05d}.jpg"
            cv2.imwrite(str(out_p), canvas)

        print(f"  💾 Saved frames → {frames_dir}/")
        print(f"\n  ✅  Projection complete.")
        print(f"      Inspect projected_tracks.csv for (pitch_x, pitch_y) columns.")
        print(f"      Next step: event extraction from spatial trajectories.\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Project ByteTrack bounding boxes onto the 2D pitch."
    )
    p.add_argument("--tracks",
                   required=True,
                   help="CSV from track_players.py  (frame_id, track_id, x1, y1, x2, y2)")
    p.add_argument("--homography",
                   required=True,
                   help=".npz from homography_picker.py  (must contain key 'H')")
    p.add_argument("--pitch",
                   required=True,
                   help="2D pitch PNG from draw_pitch.py")
    p.add_argument("--out-dir",
                   default="outputs/projected",
                   help="Output directory  (default: outputs/projected)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    TrackProjector(
        tracks_path     = args.tracks,
        homography_path = args.homography,
        pitch_path      = args.pitch,
        out_dir         = args.out_dir,
    ).run()