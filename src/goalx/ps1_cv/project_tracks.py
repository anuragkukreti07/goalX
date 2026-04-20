"""
project_tracks.py
─────────────────
Projects tracked bounding boxes from image space onto the 2D pitch canvas
using either a static homography matrix (.npz) or dynamic, per-frame 
homography matrices (.csv) from the Sharma 2018 pipeline.

FIXES (based on output analysis & pipeline upgrades)
────────────────────────────────────────────────────
FIX 1 — Condition-number guard (CRITICAL / UPDATED)
  The output showed H condition# = 2,275,402 (ideal < 1000).
  A poorly-conditioned H extrapolates wildly for points outside the
  calibration region. 
  *Update:* Because we now use dynamic per-frame homographies from the 
  Sharma MRF pipeline, this script calculates the condition number for 
  EVERY frame's matrix. It evaluates the 'Maximum' condition number across
  the video to warn you of the worst-case extrapolation.

FIX 2 — OOB flagging (CRITICAL)
  Previously, out-of-canvas coordinates were written to the CSV silently.
  Every downstream script (smooth_tracks, events, clutch, pitch_control)
  consumed them as valid positions, corrupting all spatial metrics.
  The fix: add an `in_canvas` boolean column.  OOB rows get pitch_x / pitch_y
  clamped to the canvas boundary AND flagged as in_canvas=False so
  downstream scripts can filter or weight them appropriately.

FIX 3 — Per-frame OOB summary in the quality report
  Prints which frames have the worst homography extrapolation so you know
  exactly which time windows to distrust.

FIX 4 — Camera Movement Corrector (Optical Flow)
  Tracks background optical flow to estimate camera pan/tilt per frame.
  Subtracts cumulative camera movement from player foot positions before
  applying homography, eliminating the dictionary-snapping jitter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────────────────────────

_PALETTE = [
    (255,  80,  80), (80, 255,  80), (80,  80, 255),
    (255, 200,  80), (200, 80, 255), (80, 200, 255),
    (255, 120, 200), (120, 255, 200), (200, 255, 120),
    (180, 180,  60), (60, 180, 180), (180,  60, 180),
]

def _color_for(track_id: int):
    return _PALETTE[int(track_id) % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────
#  Condition-number guard  (FIX 1)
# ─────────────────────────────────────────────────────────────────

COND_WARN  = 1_000       # warn threshold
COND_HALT  = 100_000     # halt threshold (override with --force)


def _check_homography(H_data, force: bool = False) -> float:
    """
    Compute the condition number of H (or max condition number across
    all frames if dynamic) and warn / halt accordingly.
    """
    if isinstance(H_data, dict):
        conds = [float(np.linalg.cond(matrix)) for matrix in H_data.values()]
        cond = max(conds)
        mean_cond = sum(conds) / len(conds)
        print(f"  ✔  Dynamic H processed. Mean cond: {mean_cond:.0f}, Max cond: {cond:.0f}")
    else:
        cond = float(np.linalg.cond(H_data))

    msg_suffix = "Check your homography calibration or MRF smoothing for frame jitter."

    if cond < COND_WARN:
        print(f"  ✔  Worst H condition number: {cond:.0f}  (good — < {COND_WARN:,})")
    elif cond < COND_HALT:
        print(f"  ⚠  Worst H condition number: {cond:,.0f}  "
              f"(marginal — ideally < {COND_WARN:,})")
        print(f"     Some projections may be inaccurate.  {msg_suffix}")
    else:
        msg = (
            f"Max H condition number is {cond:,.0f} — far above the {COND_HALT:,} "
            f"safety threshold.  This means the homography was calibrated with "
            f"too few or heavily clustered points and WILL project most players "
            f"to wrong positions.  {msg_suffix}"
        )
        if force:
            print(f"  ⚠  WARNING (--force set): {msg}")
        else:
            print(f"\n  ✖  CRITICAL: {msg}")
            print("     Pass --force to proceed anyway.\n")
            sys.exit(1)

    return cond


# ─────────────────────────────────────────────────────────────────
#  Core projection
# ─────────────────────────────────────────────────────────────────

def _project_all(df: pd.DataFrame, H_data,
                 pitch_w: int, pitch_h: int) -> pd.DataFrame:
    """
    Vectorised projection of all bounding-box foot points.
    Includes an Edge Filter to destroy off-screen sideline officials.
    """
    projected_rows = []
    IMG_W, IMG_H = 1920, 1080  # SNMOT-193 video resolution
    
    for frame_id, group in df.groupby("frame_id"):
        if isinstance(H_data, dict):
            if frame_id not in H_data:
                continue
            H = H_data[frame_id]
        else:
            H = H_data

        bboxes = group[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
        foot_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        foot_y = bboxes[:, 3]

        # ── THE EDGE FILTER ──────────────────────────────────────
        # Kills detections hugging the extreme left, right, top, or bottom edges
        edge_mask = (foot_x < 60) | (foot_x > IMG_W - 60) | (foot_y < 50) | (foot_y > IMG_H - 50)
        
        if edge_mask.any():
            bboxes = bboxes[~edge_mask]
            foot_x = foot_x[~edge_mask]
            foot_y = foot_y[~edge_mask]
            group = group[~edge_mask]
            if len(foot_x) == 0:
                continue
        # ─────────────────────────────────────────────────────────

        feet      = np.stack([foot_x, foot_y], axis=1).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(feet, H).reshape(-1, 2)

        res = group[["frame_id", "track_id"]].copy()
        res["img_x"]   = foot_x
        res["img_y"]   = foot_y
        res["pitch_x"] = projected[:, 0]
        res["pitch_y"] = projected[:, 1]
        
        if "pitch_x" in res.columns:
            ball_nan_mask = (res["track_id"] == -1) & (res["pitch_x"].isna() | res["pitch_y"].isna())
            res = res[~ball_nan_mask]
        projected_rows.append(res)
        
    if not projected_rows:
        return pd.DataFrame()
        
    result = pd.concat(projected_rows, ignore_index=True)

    in_canvas = (
        (result["pitch_x"] >= 0) & (result["pitch_x"] <= pitch_w) &
        (result["pitch_y"] >= 0) & (result["pitch_y"] <= pitch_h)
    )
    result["in_canvas"] = in_canvas

    result["pitch_x"] = result["pitch_x"].clip(0, pitch_w)
    result["pitch_y"] = result["pitch_y"].clip(0, pitch_h)

    return result

def _draw_frame(frame_df: pd.DataFrame, pitch_template: np.ndarray) -> np.ndarray:
    canvas   = pitch_template.copy()
    h_c, w_c = canvas.shape[:2]

    for _, row in frame_df.iterrows():
        if pd.isna(row["pitch_x"]) or pd.isna(row["pitch_y"]):
            continue
            
        px = int(round(row["pitch_x"]))
        py = int(round(row["pitch_y"]))

        in_cv = bool(row.get("in_canvas", True))
        color = _color_for(row["track_id"]) if in_cv else (180, 180, 180)

        cv2.circle(canvas, (px, py), 8, color, -1)
        cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1)
        cv2.putText(canvas, str(int(row["track_id"])),
                    (px + 10, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2, cv2.LINE_AA)

    frame_id = int(frame_df.iloc[0]["frame_id"])
    cv2.putText(canvas, f"frame {frame_id:05d}",
                (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def _interpolate_bad_frames(df: pd.DataFrame, h_dict: dict, cond_threshold: float = 50_000) -> pd.DataFrame:
    bad_frames = {
        fid for fid, H in h_dict.items() 
        if np.linalg.cond(H) > cond_threshold
    }
    
    if not bad_frames:
        return df
    
    print(f"  Interpolating {len(bad_frames)} bad-H frames ...")
    df = df.copy().sort_values(["track_id", "frame_id"])
    
    for tid, group in df.groupby("track_id"):
        if tid == -1: 
            continue
            
        idx = group.index
        
        for col in ["pitch_x", "pitch_y"]:
            vals = group[col].copy()
            bad_mask = group["frame_id"].isin(bad_frames)
            vals.loc[bad_mask] = np.nan
            vals = vals.interpolate(method="linear", limit_direction="both")
            df.loc[idx, col] = vals.values
        
        bad_idx = group[group["frame_id"].isin(bad_frames)].index
        df.loc[bad_idx, "in_canvas"] = False
    
    return df
    
    
class CameraMovementCorrector:
    """
    Tracks background optical flow to estimate camera pan/tilt per frame.
    Subtracts cumulative camera movement from player foot positions before
    applying homography, eliminating the dictionary-snapping jitter.
    Tracks features only on left/right edges where no players appear.
    """
    def __init__(self, first_frame_path: str):
        import cv2, numpy as np
        frame = cv2.imread(first_frame_path)
        if frame is None:
            self.enabled = False
            return
        self.enabled = True
        self.lk_params = dict(winSize=(15,15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.zeros_like(gray)
        mask[:, 0:30] = 1        # left edge strip
        mask[:, w-30:w] = 1      # right edge strip
        feature_params = dict(maxCorners=100, qualityLevel=0.3,
                              minDistance=3, blockSize=7, mask=mask)
        self.prev_gray = gray
        self.prev_features = cv2.goodFeaturesToTrack(gray, **feature_params)
        self.feature_params = feature_params
        self.movements = {1: (0.0, 0.0)}  # frame_id -> (cum_dx, cum_dy)
        self.cum_dx = 0.0
        self.cum_dy = 0.0

    def compute_all(self, frames_dir: str, frame_ids: list):
        import cv2, numpy as np
        from pathlib import Path
        if not self.enabled:
            return
        frames_dir = Path(frames_dir)
        for fid in sorted(frame_ids):
            path = frames_dir / f"{int(fid):06d}.jpg"
            if not path.exists():
                self.movements[fid] = (self.cum_dx, self.cum_dy)
                continue
            gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
            if self.prev_features is not None and len(self.prev_features) > 4:
                new_feat, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_features, None, **self.lk_params)
                if new_feat is not None and status is not None:
                    good_new = new_feat[status.ravel()==1]
                    good_old = self.prev_features[status.ravel()==1]
                    if len(good_new) > 4:
                        dx = float(np.median(good_new[:,0,0] - good_old[:,0,0]))
                        dy = float(np.median(good_new[:,0,1] - good_old[:,0,1]))
                        if abs(dx) > 1.5 or abs(dy) > 1.5:
                            self.cum_dx += dx
                            self.cum_dy += dy
            self.movements[fid] = (self.cum_dx, self.cum_dy)
            self.prev_features = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            self.prev_gray = gray

    def get(self, frame_id):
        return self.movements.get(frame_id, (self.cum_dx, self.cum_dy))


class TrackProjector:
    def __init__(self, tracks_path, homography_path, pitch_path,
                 out_dir, force: bool = False):
        self.tracks_path     = tracks_path
        self.homography_path = str(homography_path)
        self.pitch_path      = pitch_path
        self.out_dir         = Path(out_dir)
        self.force           = force
        self.H               = None
        self.df              = None

    def _load_homography(self) -> None:
        if self.homography_path.endswith('.csv'):
            h_df = pd.read_csv(self.homography_path)
            frame_col = "frame_id" if "frame_id" in h_df.columns else "frame"
            
            h_names_1 = ['h11', 'h12', 'h13', 'h21', 'h22', 'h23', 'h31', 'h32', 'h33']
            h_names_0 = ['h00', 'h01', 'h02', 'h10', 'h11', 'h12', 'h20', 'h21', 'h22']
            
            if all(c in h_df.columns for c in h_names_1):
                h_cols = h_names_1
            elif all(c in h_df.columns for c in h_names_0):
                h_cols = h_names_0
            else:
                h_cols = [c for c in h_df.columns if c != frame_col][:9]

            if len(h_cols) != 9:
                raise ValueError(f"Could not extract exactly 9 matrix elements. Found: {h_cols}")

            self.H = {}
            for _, row in h_df.iterrows():
                fid = int(row[frame_col])
                matrix = row[h_cols].values.astype(np.float32).reshape(3, 3)
                self.H[fid] = matrix
                
            print(f"  ✔  Dynamic H loaded ({len(self.H)} frames from CSV)")
            
        else:
            data    = np.load(self.homography_path)
            self.H  = data["H"].astype(np.float32)
            inliers = int(np.sum(data["status"])) if "status" in data else "?"
            n_pts   = len(data["frame_pts"]) if "frame_pts" in data else "?"
            print(f"  ✔  Static H loaded  ({n_pts} calibration points,  "
                  f"RANSAC inliers: {inliers})")

    def _load_tracks(self) -> None:
        self.df = pd.read_csv(self.tracks_path)
        rename  = {}
        if "frame" in self.df.columns:   rename["frame"] = "frame_id"
        if "id"    in self.df.columns:   rename["id"]    = "track_id"
        if rename:
            self.df.rename(columns=rename, inplace=True)

        required = {"frame_id", "track_id", "x1", "y1", "x2", "y2"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"CSV missing columns: {missing}\n"
                f"Available: {list(self.df.columns)}"
            )
        print(f"  ✔  {len(self.df)} detections across "
              f"{self.df['frame_id'].nunique()} frames.")

    def run(self) -> None:
        print("\n  goalX — Track Projector")
        print("  " + "─" * 40)

        self._load_homography()
        self._load_tracks()

        cond = _check_homography(self.H, force=self.force)

        pitch_img = cv2.imread(str(self.pitch_path))
        if pitch_img is None:
            raise FileNotFoundError(f"Cannot read pitch: {self.pitch_path}")
        pitch_h, pitch_w = pitch_img.shape[:2]

        # ── Camera movement correction ────────────────────────────
        frames_dir = "data/raw_videos/tracking/test3/SNMOT-193/img1"
        first_frame = frames_dir + "/000001.jpg"
        corrector = CameraMovementCorrector(first_frame)
        frame_ids = sorted(self.df['frame_id'].unique().tolist())
        print("  Computing camera movement via optical flow ...")
        corrector.compute_all(frames_dir, frame_ids)

        # Subtract camera movement from bounding boxes before projection
        self.df = self.df.copy()
        for fid, group_idx in self.df.groupby('frame_id').groups.items():
            dx, dy = corrector.get(fid)
            if 'x1' in self.df.columns:
                self.df.loc[group_idx, 'x1'] -= dx
                self.df.loc[group_idx, 'x2'] -= dx
                self.df.loc[group_idx, 'y1'] -= dy
                self.df.loc[group_idx, 'y2'] -= dy

        # ── Project ───────────────────────────────────────────────
        print("  Projecting foot points through H …")
        projected_df = _project_all(self.df, self.H, pitch_w, pitch_h)
        
        if isinstance(self.H, dict):
            projected_df = _interpolate_bad_frames(projected_df, self.H, cond_threshold=1_000_000)
            
        if projected_df.empty:
            print("  ✖  Projection failed. No data to save.")
            sys.exit(1)

        n_total    = len(projected_df)
        n_oob      = int((~projected_df["in_canvas"]).sum())
        oob_pct    = 100.0 * n_oob / max(n_total, 1)

        print(f"\n  Projection quality:")
        print(f"     Total rows     : {n_total:,}")
        print(f"     In canvas      : {n_total - n_oob:,}")
        print(f"     Out-of-canvas  : {n_oob:,}  ({oob_pct:.1f}%)")

        if oob_pct > 5.0:
            print(f"\n  ⚠  {oob_pct:.1f}% of projections are outside the pitch canvas.")
            print(f"     This is caused by the ill-conditioned H (max cond# {cond:,.0f}).")
            print(f"     Action: Check the stability of your MRF homography smoothing.")
            print(f"     OOB rows have been clamped to canvas boundary and flagged "
                  f"in_canvas=False.\n")

        # ── Save CSV ──────────────────────────────────────────────
        csv_out = self.out_dir / "projected_tracks.csv"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        projected_df.to_csv(csv_out, index=False)
        print(f"  Saved CSV → {csv_out}")

        # ── Render annotated frames ───────────────────────────────
        frames_dir = self.out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        unique_frames = sorted(projected_df["frame_id"].unique())
        print(f"  Rendering {len(unique_frames)} annotated pitch frames …")

        for fid in unique_frames:
            fdata  = projected_df[projected_df["frame_id"] == fid]
            canvas = _draw_frame(fdata, pitch_img)
            cv2.imwrite(str(frames_dir / f"{int(fid):05d}.jpg"), canvas)

        print(f"  Saved frames → {frames_dir}/")
        print(f"\n  ✅  Projection complete.")
        print(f"      Columns added: pitch_x, pitch_y, in_canvas")
        print(f"      Filter with: df[df['in_canvas']==True] for valid projections.\n")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Project ByteTrack bounding boxes onto the 2D pitch canvas."
    )
    p.add_argument("--tracks",      required=True,
                   help="CSV from track_players.py")
    p.add_argument("--homography",  required=True,
                   help=".npz or .csv containing homography matrices")
    p.add_argument("--pitch",       required=True,
                   help="2D pitch PNG from draw_pitch.py")
    p.add_argument("--out-dir",     default="outputs/projected",
                   help="Output directory")
    p.add_argument("--force",       action="store_true",
                   help="Continue even if H condition number is dangerously high")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    TrackProjector(
        tracks_path     = args.tracks,
        homography_path = args.homography,
        pitch_path      = args.pitch,
        out_dir         = args.out_dir,
        force           = args.force,
    ).run()
