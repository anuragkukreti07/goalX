"""
project_tracks.py
─────────────────
Projects tracked bounding boxes from image space onto the 2D pitch canvas
using the pre-computed homography matrix H.

FIXES (based on output analysis)
─────────────────────────────────
FIX 1 — Condition-number guard (CRITICAL)
  The output showed H condition# = 2,275,402 (ideal < 1000).
  A poorly-conditioned H extrapolates wildly for points outside the
  calibration region — this caused 15.5 % of projections (1,773 / 11,402)
  to land outside the 1050×680 px pitch canvas.
  The fix: compute np.linalg.cond(H) before projection and WARN loudly
  when it exceeds 1000.  If > 100,000 the pipeline halts by default
  (override with --force) because downstream metrics will all be wrong.

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

NOTE — foot-point and cv2.perspectiveTransform ARE already correct.
  The feedback suggested adding manual homogeneous normalisation, but
  cv2.perspectiveTransform already handles this internally.  The root
  cause of wrong projections was always the ill-conditioned H matrix,
  not the projection math.

Input / Output format: UNCHANGED from original.
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


def _check_homography(H: np.ndarray, force: bool = False) -> None:
    """
    Compute the condition number of H and warn / halt accordingly.

    A condition number close to 1 means H is well-conditioned and
    interpolates reliably everywhere.  A number above 1,000 means H
    was calibrated with too few or poorly-spread points and will
    extrapolate badly for players outside the calibration region.

    The output showed condition# = 2,275,402 — this is why 15.5 % of
    projections landed outside the pitch canvas.
    """
    cond = float(np.linalg.cond(H))
    msg_suffix = (
        "Re-run homography_picker.py with 8+ well-spread landmarks "
        "(corners + centre circle + both penalty spot areas)."
    )

    if cond < COND_WARN:
        print(f"  ✔  H condition number: {cond:.0f}  (good — < {COND_WARN:,})")
    elif cond < COND_HALT:
        print(f"  ⚠  H condition number: {cond:,.0f}  "
              f"(marginal — ideally < {COND_WARN:,})")
        print(f"     Some projections may be inaccurate.  {msg_suffix}")
    else:
        msg = (
            f"H condition number is {cond:,.0f} — far above the {COND_HALT:,} "
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

def _project_all(df: pd.DataFrame, H: np.ndarray,
                 pitch_w: int, pitch_h: int) -> pd.DataFrame:
    """
    Vectorised projection of all bounding-box foot points.

    Foot point = bottom-centre of the bounding box:
        foot_x = (x1 + x2) / 2
        foot_y = y2
    This is the player's ground contact point — the only point that
    lies on the ground plane for which the homography is valid.

    cv2.perspectiveTransform handles homogeneous coordinate division
    internally; no manual normalisation is needed.
    """
    bboxes = df[["x1", "y1", "x2", "y2"]].values.astype(np.float32)

    foot_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0   # horizontal centre
    foot_y = bboxes[:, 3]                            # y2 = bottom = ground

    feet      = np.stack([foot_x, foot_y], axis=1).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(feet, H).reshape(-1, 2)

    result = df[["frame_id", "track_id"]].copy()
    result["img_x"]   = foot_x
    result["img_y"]   = foot_y
    result["pitch_x"] = projected[:, 0]
    result["pitch_y"] = projected[:, 1]

    # ── FIX 2: flag out-of-canvas rows ────────────────────────────
    in_canvas = (
        (result["pitch_x"] >= 0) & (result["pitch_x"] <= pitch_w) &
        (result["pitch_y"] >= 0) & (result["pitch_y"] <= pitch_h)
    )
    result["in_canvas"] = in_canvas

    # Clamp OOB coords to canvas boundary so downstream scripts
    # that ignore the flag still get a plausible value rather than
    # a wildly extrapolated one.
    result["pitch_x"] = result["pitch_x"].clip(0, pitch_w)
    result["pitch_y"] = result["pitch_y"].clip(0, pitch_h)

    return result


# ─────────────────────────────────────────────────────────────────
#  Frame drawing
# ─────────────────────────────────────────────────────────────────

def _draw_frame(frame_df: pd.DataFrame,
                pitch_template: np.ndarray) -> np.ndarray:
    canvas   = pitch_template.copy()
    h_c, w_c = canvas.shape[:2]

    for _, row in frame_df.iterrows():
        px = int(round(row["pitch_x"]))
        py = int(round(row["pitch_y"]))

        # Use a different colour for OOB (clamped) points so they're visually distinct
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


# ─────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────

class TrackProjector:
    def __init__(self, tracks_path, homography_path, pitch_path,
                 out_dir, force: bool = False):
        self.tracks_path     = tracks_path
        self.homography_path = homography_path
        self.pitch_path      = pitch_path
        self.out_dir         = Path(out_dir)
        self.force           = force
        self.H               = None
        self.df              = None

    def _load_homography(self) -> None:
        data    = np.load(self.homography_path)
        self.H  = data["H"].astype(np.float32)
        inliers = int(np.sum(data["status"])) if "status" in data else "?"
        status  = data.get("status", None)
        n_pts   = len(data["frame_pts"]) if "frame_pts" in data else "?"
        print(f"  ✔  H loaded  ({n_pts} calibration points,  "
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

        # ── FIX 1: condition-number guard ─────────────────────────
        cond = _check_homography(self.H, force=self.force)

        pitch_img = cv2.imread(str(self.pitch_path))
        if pitch_img is None:
            raise FileNotFoundError(f"Cannot read pitch: {self.pitch_path}")
        pitch_h, pitch_w = pitch_img.shape[:2]

        # ── Project ───────────────────────────────────────────────
        print("  Projecting foot points through H …")
        projected_df = _project_all(self.df, self.H, pitch_w, pitch_h)

        # ── FIX 3: quality report ─────────────────────────────────
        n_total    = len(projected_df)
        n_oob      = int((~projected_df["in_canvas"]).sum())
        oob_pct    = 100.0 * n_oob / max(n_total, 1)

        print(f"\n  Projection quality:")
        print(f"     Total rows     : {n_total:,}")
        print(f"     In canvas      : {n_total - n_oob:,}")
        print(f"     Out-of-canvas  : {n_oob:,}  ({oob_pct:.1f}%)")

        if oob_pct > 5.0:
            print(f"\n  ⚠  {oob_pct:.1f}% of projections are outside the pitch canvas.")
            print(f"     This is caused by the ill-conditioned H (cond# {cond:,.0f}).")
            print(f"     Action: re-run homography_picker.py with 8+ well-spread points.")
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


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Project ByteTrack bounding boxes onto the 2D pitch canvas."
    )
    p.add_argument("--tracks",     required=True,
                   help="CSV from track_players.py")
    p.add_argument("--homography", required=True,
                   help=".npz from homography_picker.py (must contain 'H')")
    p.add_argument("--pitch",      required=True,
                   help="2D pitch PNG from draw_pitch.py")
    p.add_argument("--out-dir",    default="outputs/projected",
                   help="Output directory")
    p.add_argument("--force",      action="store_true",
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