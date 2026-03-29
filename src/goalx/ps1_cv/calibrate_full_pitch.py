"""
calibrate_full_pitch.py
───────────────────────
Smart homography calibration tool designed specifically for full-pitch
tactical camera views where the entire field is visible.

Key difference from homography_picker.py
─────────────────────────────────────────
  homography_picker.py  — generic; YOU supply both image and pitch coordinates.
  calibrate_full_pitch  — the metric pitch coordinates are PRE-KNOWN from
                          FIFA regulations. You only click on the video frame.
                          The math side is looked up automatically.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────
#  FIFA Landmark Library
#  All coordinates in METRES relative to top-left corner of pitch.
#  Pitch = 105 m × 68 m  (FIFA regulation A-standard)
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Landmark:
    id:    int
    name:  str
    x_m:   float   # metres along length
    y_m:   float   # metres along width


_W = 105.0   # pitch length
_H = 68.0    # pitch width
_PA = 16.5   # penalty area depth
_PAW = 20.16 # penalty area half-width from centre
_GA = 5.5    # goal area depth
_GAW = 9.16  # goal area half-width
_PS = 11.0   # penalty spot distance
_CX, _CY = _W / 2, _H / 2   # centre of pitch

FIFA_LANDMARKS: list[Landmark] = [
    # ── Pitch corners ──────────────────────────────────────
    Landmark( 1, "Top-left corner",           0.0,      0.0),
    Landmark( 2, "Top-right corner",          _W,       0.0),
    Landmark( 3, "Bottom-right corner",       _W,       _H),
    Landmark( 4, "Bottom-left corner",        0.0,      _H),

    # ── Halfway line ───────────────────────────────────────
    Landmark( 5, "Halfway line top",          _CX,      0.0),
    Landmark( 6, "Centre spot",               _CX,      _CY),
    Landmark( 7, "Halfway line bottom",       _CX,      _H),

    # ── Left penalty area ──────────────────────────────────
    Landmark( 8, "Left penalty area top-left",   0.0,   _CY - _PAW),
    Landmark( 9, "Left penalty area top-right",  _PA,   _CY - _PAW),
    Landmark(10, "Left penalty area bot-right",  _PA,   _CY + _PAW),
    Landmark(11, "Left penalty area bot-left",   0.0,   _CY + _PAW),
    Landmark(12, "Left penalty spot",            _PS,   _CY),

    # ── Right penalty area ─────────────────────────────────
    Landmark(13, "Right penalty area top-right",  _W,         _CY - _PAW),
    Landmark(14, "Right penalty area top-left",   _W - _PA,   _CY - _PAW),
    Landmark(15, "Right penalty area bot-left",   _W - _PA,   _CY + _PAW),
    Landmark(16, "Right penalty area bot-right",  _W,         _CY + _PAW),
    Landmark(17, "Right penalty spot",            _W - _PS,   _CY),

    # ── Left goal area ─────────────────────────────────────
    Landmark(18, "Left goal area top-right",    _GA,   _CY - _GAW),
    Landmark(19, "Left goal area bot-right",    _GA,   _CY + _GAW),

    # ── Right goal area ────────────────────────────────────
    Landmark(20, "Right goal area top-left",    _W - _GA,   _CY - _GAW),
    Landmark(21, "Right goal area bot-left",    _W - _GA,   _CY + _GAW),
]

# Quick lookup: landmark id → Landmark
_LM_MAP: dict[int, Landmark] = {lm.id: lm for lm in FIFA_LANDMARKS}


# ─────────────────────────────────────────────────────────────────
#  Pitch pixel scale  (must match draw_pitch.py output)
# ─────────────────────────────────────────────────────────────────

PITCH_SCALE = 10.0   # pixels per metre  (105m → 1050px, 68m → 680px)


def _m_to_px(x_m: float, y_m: float) -> tuple[int, int]:
    """Convert metric pitch coordinates to pitch-canvas pixel coordinates."""
    return int(round(x_m * PITCH_SCALE)), int(round(y_m * PITCH_SCALE))


# ─────────────────────────────────────────────────────────────────
#  Reference sheet generator
# ─────────────────────────────────────────────────────────────────

def generate_reference_sheet(pitch_path: Path,
                             out_path: Path,
                             highlight_ids: list[int] | None = None) -> None:
    """
    Draw every landmark on the pitch canvas with its ID number.
    Save as a PNG reference image the user can study before calibrating.
    """
    img = cv2.imread(str(pitch_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read pitch map: {pitch_path}")

    h_img, w_img = img.shape[:2]
    # Add a right-side legend panel
    legend_w = 350
    canvas = np.full((h_img, w_img + legend_w, 3), 30, dtype=np.uint8)
    canvas[:, :w_img] = img

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55

    for lm in FIFA_LANDMARKS:
        px, py = _m_to_px(lm.x_m, lm.y_m)
        px = max(8, min(px, w_img - 8))
        py = max(8, min(py, h_img - 8))

        active = highlight_ids is None or lm.id in highlight_ids
        color  = (0, 220, 120) if active else (80, 80, 80)
        size   = 10 if active else 6

        cv2.circle(canvas, (px, py), size, color, -1)
        cv2.circle(canvas, (px, py), size + 1, (0, 0, 0), 1)
        cv2.putText(canvas, str(lm.id), (px + 12, py - 6),
                    font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend
    cv2.putText(canvas, "Landmark reference", (w_img + 10, 30),
                font, 0.65, (200, 200, 220), 2, cv2.LINE_AA)
    cv2.line(canvas, (w_img + 10, 40), (w_img + legend_w - 10, 40),
             (80, 80, 80), 1)

    for i, lm in enumerate(FIFA_LANDMARKS):
        y_leg = 62 + i * 22
        if y_leg > h_img - 10:
            break
        active = highlight_ids is None or lm.id in highlight_ids
        col    = (0, 220, 120) if active else (80, 80, 80)
        cv2.circle(canvas, (w_img + 18, y_leg - 4), 6, col, -1)
        label = f"{lm.id:2d}.  {lm.name}"
        cv2.putText(canvas, label, (w_img + 30, y_leg),
                    font, scale, (200, 200, 220), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"  ✔  Reference sheet saved → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Interactive calibrator
# ─────────────────────────────────────────────────────────────────

class FullPitchCalibrator:
    """
    Guides the user through clicking each selected landmark on the
    video frame. The corresponding metric coordinates are looked up
    automatically from the FIFA landmark library.
    """

    def __init__(self, frame_path: Path, pitch_path: Path,
                 out_path: Path, landmark_ids: list[int]):
        self.frame_path  = frame_path
        self.pitch_path  = pitch_path
        self.out_path    = out_path

        # Validate all requested IDs exist
        unknown = [i for i in landmark_ids if i not in _LM_MAP]
        if unknown:
            raise ValueError(f"Unknown landmark IDs: {unknown}. "
                             f"Valid range: 1–{max(_LM_MAP)}")
        if len(landmark_ids) < 4:
            raise ValueError("Need at least 4 landmarks. "
                             "6–10 recommended for full-pitch accuracy.")

        self.landmarks = [_LM_MAP[i] for i in landmark_ids]
        self.frame_pts: list[tuple[int, int]] = []
        self._current_click: tuple[int, int] | None = None
        self._win = ""

    # ──────────────────────────────────────────────────────────────

    def _load_clean(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return img

    def _cb_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_click = (x, y)

    def _draw_instructions(self, img: np.ndarray,
                            lm: Landmark, n_done: int, n_total: int,
                            pts_so_far: list) -> np.ndarray:
        """Overlay the current instruction HUD onto the frame."""
        canvas = img.copy()
        h, w   = canvas.shape[:2]

        # Dark banner at bottom
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas,
                    f"[{n_done+1}/{n_total}]  Click:  {lm.name}",
                    (16, h - 58), font, 0.8, (0, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"Metric coords:  ({lm.x_m:.1f} m,  {lm.y_m:.1f} m)",
                    (16, h - 30), font, 0.6, (160, 160, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    "Left-click = place   |   Backspace = undo   |   ESC = abort",
                    (16, h - 8), font, 0.48, (120, 120, 140), 1, cv2.LINE_AA)

        # Already-placed dots
        for i, pt in enumerate(pts_so_far):
            cv2.circle(canvas, pt, 7, (0, 200, 80), -1)
            cv2.circle(canvas, pt, 8, (255, 255, 255), 1)
            cv2.putText(canvas, str(self.landmarks[i].id),
                        (pt[0] + 10, pt[1] - 8), font, 0.6,
                        (0, 200, 80), 2, cv2.LINE_AA)

        return canvas

    def _collect_frame_points(self) -> bool:
        """
        Phase 1: step through each selected landmark and ask user to click it.
        Returns True on success, False on abort.
        """
        frame_orig = self._load_clean(self.frame_path)
        n = len(self.landmarks)

        self._win = "goalX — Full-Pitch Calibration"
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._win, self._cb_click)

        print(f"\n{'═'*60}")
        print(f"  CALIBRATION — {n} landmarks selected")
        print(f"  Click each landmark ON THE VIDEO FRAME in order.")
        print(f"  Backspace = undo last click    ESC = abort")
        print(f"{'═'*60}\n")

        i = 0
        while i < n:
            lm      = self.landmarks[i]
            display = self._draw_instructions(frame_orig, lm, i, n,
                                              self.frame_pts)
            cv2.imshow(self._win, display)

            self._current_click = None
            while True:
                key = cv2.waitKey(15) & 0xFF
                if key == 27:     # ESC — abort
                    cv2.destroyAllWindows()
                    return False
                if key == 8 and i > 0:  # Backspace — undo
                    self.frame_pts.pop()
                    i -= 1
                    print(f"  ↩  Undid landmark {self.landmarks[i].id}. "
                          f"Re-clicking: {self.landmarks[i].name}")
                    break
                if self._current_click is not None:
                    self.frame_pts.append(self._current_click)
                    print(f"  ✔  Landmark {lm.id}  ({lm.name})  "
                          f"→  pixel {self._current_click}")
                    i += 1
                    break

        cv2.destroyAllWindows()
        return True

    # ──────────────────────────────────────────────────────────────

    def _compute_H(self) -> np.ndarray | None:
        """
        Build src/dst point arrays and compute H via RANSAC findHomography.
        src = image pixel coordinates (clicked by user)
        dst = pitch canvas pixel coordinates (computed from metric coords)
        """
        src_pts = np.array(self.frame_pts, dtype=np.float32)

        # Convert metric coords → pitch canvas pixels
        dst_pts = np.array(
            [_m_to_px(lm.x_m, lm.y_m) for lm in self.landmarks],
            dtype=np.float32
        )

        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("  ❌  Homography computation failed.")
            print("      Likely cause: 3+ clicked points are collinear.")
            print("      Fix: choose landmarks spread across different regions.")
            return None

        inliers = int(np.sum(status))
        n       = len(self.frame_pts)
        print(f"\n  ✔  Homography computed.  "
              f"RANSAC inliers: {inliers}/{n}")

        if inliers < 4:
            print(f"  ⚠  Only {inliers} inliers. "
                  "Consider re-calibrating with more spread-out landmarks.")

        self._status = status
        self._src_pts = src_pts
        self._dst_pts = dst_pts
        return H

    # ──────────────────────────────────────────────────────────────

    def _sanity_check(self, H: np.ndarray) -> None:
        """
        TRUE sanity check: warpPerspective the frame into pitch space
        and blend with the pitch canvas. White field lines from the
        video should align exactly with white lines on the 2D map.
        """
        print("\n  🔍  Sanity check: warping video frame onto pitch…")
        print("      ✔ GOOD: grass lines sit on top of green pitch lines.")
        print("      ✖ BAD : twisted / skewed — wrong landmark order or position.")
        print("      Press any key to continue.\n")

        pitch_clean = self._load_clean(self.pitch_path)
        frame_clean = self._load_clean(self.frame_path)

        h_p, w_p = pitch_clean.shape[:2]
        warped   = cv2.warpPerspective(frame_clean, H, (w_p, h_p))
        blended  = cv2.addWeighted(pitch_clean, 0.55, warped, 0.45, 0)

        # Overlay projected dst points for reference
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (lm, dp) in enumerate(zip(self.landmarks, self._dst_pts)):
            px, py = int(dp[0]), int(dp[1])
            cv2.circle(blended, (px, py), 8, (0, 220, 255), -1)
            cv2.putText(blended, str(lm.id), (px + 10, py - 6),
                        font, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

        cv2.namedWindow("Sanity — do the field lines align?", cv2.WINDOW_NORMAL)
        cv2.imshow("Sanity — do the field lines align?", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ──────────────────────────────────────────────────────────────

    def _save(self, H: np.ndarray) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(self.out_path),
            H          = H,
            frame_pts  = self._src_pts,
            pitch_pts  = self._dst_pts,
            status     = self._status,
            landmark_ids = np.array([lm.id for lm in self.landmarks]),
        )
        print(f"  💾  Saved → {self.out_path}")
        print(f"        Contains: H, frame_pts, pitch_pts, status, landmark_ids\n")

    # ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        print(f"\n  goalX — Full-Pitch Calibrator")
        print(f"  {'─'*42}")
        print(f"  Frame     : {self.frame_path}")
        print(f"  Landmarks : {[lm.id for lm in self.landmarks]}\n")

        if not self._collect_frame_points():
            print("  Calibration aborted.")
            return

        H = self._compute_H()
        if H is None:
            return

        self._sanity_check(H)
        self._save(H)

        print("  ✅  Calibration complete.")
        print("      Next: python3 run_goalx.py --config config/goalx_config.yaml\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Full-pitch homography calibrator for goalX."
    )
    # Removed required=True to allow --list to bypass these checks natively
    p.add_argument("--frame",  help="Path to a single frame from the new full-pitch video")
    p.add_argument("--pitch",  help="Path to the 2D pitch map PNG (from draw_pitch.py)")
    p.add_argument("--out",    default="data/homography_data.npz",
                   help="Output .npz path")
    p.add_argument("--ref-out", default="data/landmarks_reference.png",
                   help="Output path for the landmark reference sheet")
    p.add_argument("--landmarks", type=int, nargs="+",
                   default=list(range(1, len(FIFA_LANDMARKS) + 1)),
                   help="Landmark IDs to use (default: all). "
                        "Specify fewer if some are off-camera.")
    p.add_argument("--no-ref", action="store_true",
                   help="Skip generating the reference sheet PNG")
    p.add_argument("--list",   action="store_true",
                   help="Print all available landmark IDs and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list:
        print("\n  Available FIFA pitch landmarks:\n")
        for lm in FIFA_LANDMARKS:
            print(f"  {lm.id:2d}.  {lm.name:<40}  "
                  f"({lm.x_m:.2f} m, {lm.y_m:.2f} m)")
        print()
        raise SystemExit(0)

    # Manual enforcement of required args if not using --list
    if not args.frame or not args.pitch:
        print("\n  ❌ Error: --frame and --pitch are required unless using --list.")
        print("  Usage: python3 src/goalx/ps1_cv/calibrate_full_pitch.py --frame <path> --pitch <path>\n")
        raise SystemExit(1)

    if not args.no_ref:
        generate_reference_sheet(
            pitch_path    = Path(args.pitch),
            out_path      = Path(args.ref_out),
            highlight_ids = args.landmarks,
        )
        print(f"\n  Open {args.ref_out} to see which numbered dots to click.")
        print(f"  Then press ENTER here to begin clicking on the video frame.")
        input("  Press ENTER to start…\n")

    FullPitchCalibrator(
        frame_path   = Path(args.frame),
        pitch_path   = Path(args.pitch),
        out_path     = Path(args.out),
        landmark_ids = args.landmarks,
    ).run()