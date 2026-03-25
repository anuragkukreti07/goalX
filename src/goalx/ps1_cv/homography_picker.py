"""
homography_picker.py
--------------------
A robust, interactive tool to compute the Homography matrix H that maps
image-space coordinates (broadcast video frame) to metric pitch coordinates
(2D tactical board).

Usage:
    python3 src/goalx/ps1_cv/homography_picker.py \
        --frame data/raw_videos/tracking/test/SNMOT-116/img1/000001.jpg \
        --pitch data/pitch_map.png \
        --out   data/homography_data.npz

Output:
    A .npz file containing:
        H          : (3, 3) float32  — the Homography matrix
        frame_pts  : (N, 2) float32  — clicked points on the video frame
        pitch_pts  : (N, 2) float32  — corresponding points on the 2D pitch
        status     : (N, 1) uint8    — RANSAC inlier mask (1 = inlier)
"""

import cv2
import numpy as np
import os
import argparse


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

COLORS = {
    "red":    (0,   0,   255),   # video frame points
    "blue":   (255, 0,   0),     # pitch points
    "yellow": (0,   255, 255),   # projected / sanity
    "green":  (0,   255, 0),
    "white":  (255, 255, 255),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.75
THICKNESS  = 2


def _draw_point(img: np.ndarray, pt: tuple, idx: int, color: tuple) -> None:
    """Draw a labelled circle on img in-place."""
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img, (x, y), 6, color, -1)
    cv2.circle(img, (x, y), 7, COLORS["white"], 1)          # thin white ring
    cv2.putText(img, str(idx), (x + 10, y - 10), FONT,
                FONT_SCALE, color, THICKNESS, cv2.LINE_AA)


def _draw_hud(img: np.ndarray, n_current: int, n_target: int,
              label: str, hint: str = "") -> None:
    """
    Overlay a semi-transparent HUD at the bottom of the image so the
    operator can always see how many points have been collected.
    """
    h, w = img.shape[:2]
    bar_h = 48
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    text = f"  {label}  |  Points: {n_current}/{n_target}  |  {hint}"
    cv2.putText(img, text, (10, h - 14), FONT, 0.6,
                COLORS["white"], 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
#  HomographyPicker
# ─────────────────────────────────────────────────────────────────

class HomographyPicker:
    """
    Interactive point-picker that computes H from N >= 4 correspondences.

    Session flow
    ─────────────
    Phase 1  →  Click N points on the video frame.
                Left-click  : add point
                Backspace   : undo last point
                Enter/Space : confirm and advance to Phase 2

    Phase 2  →  Click the same N points on the 2D pitch map.
                (Same undo shortcut available.)
                Completes automatically once N points are collected.

    Phase 3  →  Compute H via RANSAC findHomography.

    Phase 4  →  Sanity check: warpPerspective the video frame onto the
                pitch canvas and blend — visual alignment check.

    Phase 5  →  Persist results to .npz.
    """

    MIN_POINTS = 4

    def __init__(self, frame_path: str, pitch_path: str, out_path: str):
        self.frame_path = frame_path
        self.pitch_path = pitch_path
        self.out_path   = out_path

        self.frame_points: list[tuple] = []
        self.pitch_points: list[tuple] = []

        # Working copies for drawing — rebuilt on undo
        self._frame_canvas: np.ndarray | None = None
        self._pitch_canvas: np.ndarray | None = None
        self._win_name: str = ""

    # ──────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────

    def _load_clean(self, path: str) -> np.ndarray:
        """Load a fresh BGR copy of the image."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    def _redraw_frame_canvas(self) -> None:
        """Rebuild frame canvas from scratch to support undo."""
        self._frame_canvas = self._load_clean(self.frame_path)
        for i, pt in enumerate(self.frame_points):
            _draw_point(self._frame_canvas, pt, i + 1, COLORS["red"])
        n_target = max(self.MIN_POINTS, len(self.frame_points))
        _draw_hud(self._frame_canvas, len(self.frame_points), n_target,
                  "VIDEO FRAME",
                  "LClick=add  Backspace=undo  Enter=confirm")
        cv2.imshow(self._win_name, self._frame_canvas)

    def _redraw_pitch_canvas(self, n_target: int) -> None:
        """Rebuild pitch canvas from scratch to support undo."""
        self._pitch_canvas = self._load_clean(self.pitch_path)
        for i, pt in enumerate(self.pitch_points):
            _draw_point(self._pitch_canvas, pt, i + 1, COLORS["blue"])
        _draw_hud(self._pitch_canvas, len(self.pitch_points), n_target,
                  "PITCH MAP",
                  f"Click point {len(self.pitch_points)+1}  |  Backspace=undo")
        cv2.imshow(self._win_name, self._pitch_canvas)

    # ──────────────────────────────────────────
    #  Mouse callbacks
    # ──────────────────────────────────────────

    def _cb_frame(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.frame_points.append((x, y))
            self._redraw_frame_canvas()

    def _cb_pitch(self, event, x, y, flags, param):
        n_target: int = param
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pitch_points) < n_target:
                self.pitch_points.append((x, y))
                self._redraw_pitch_canvas(n_target)

    # ──────────────────────────────────────────
    #  Phase 1 — pick frame points
    # ──────────────────────────────────────────

    def _phase1_pick_frame(self) -> bool:
        self._win_name = "Phase 1 - Video Frame"
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        self._redraw_frame_canvas() 
        cv2.waitKey(10) 
        cv2.setMouseCallback(self._win_name, self._cb_frame)
        

        print("\n" + "═" * 60)
        print("  PHASE 1: Click landmarks on the VIDEO FRAME")
        print("  ─────────────────────────────────────────────")
        print("  • Minimum 4 points  (6-8 recommended for better accuracy)")
        print("  • Use line intersections: penalty box corners, goal line")
        print("  • Left-click  = add point")
        print("  • Backspace   = undo last point")
        print("  • Enter/Space = confirm and continue")
        print("═" * 60)

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == 8:  # Backspace — undo
                if self.frame_points:
                    self.frame_points.pop()
                    print(f"  ↩  Undid last point. ({len(self.frame_points)} remaining)")
                    self._redraw_frame_canvas()

            elif key in (13, 32):  # Enter or Space — confirm
                if len(self.frame_points) >= self.MIN_POINTS:
                    break
                else:
                    print(f"  ⚠  Need at least {self.MIN_POINTS} points. "
                          f"You have {len(self.frame_points)}.")

            elif key in (27, ord('q')):  # ESC / q — abort
                print("  Cancelled by user.")
                cv2.destroyAllWindows()
                return False

        cv2.destroyAllWindows()
        print(f"  ✔  Confirmed {len(self.frame_points)} frame points.\n")
        return True

    # ──────────────────────────────────────────
    #  Phase 2 — pick pitch points
    # ──────────────────────────────────────────

    def _phase2_pick_pitch(self) -> bool:
        n_target = len(self.frame_points)
        self._win_name = f"Phase 2 - Pitch Map  (click {n_target} matching points)"
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        self._redraw_pitch_canvas(n_target)
        cv2.waitKey(1)
        
        cv2.setMouseCallback(self._win_name, self._cb_pitch, n_target)
        print("═" * 60)
        print(f"  PHASE 2: Click the SAME {n_target} landmarks on the PITCH MAP")
        print("  ─────────────────────────────────────────────────────────")
        print("  • STRICT ORDER: Point 1 here must match Point 1 on the video")
        print("  • Backspace = undo last point")
        print("═" * 60)

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == 8:  # Backspace — undo
                if self.pitch_points:
                    self.pitch_points.pop()
                    print(f"  ↩  Undid last pitch point. ({len(self.pitch_points)} remaining)")
                    self._redraw_pitch_canvas(n_target)

            elif key in (27, ord('q')):
                print("  Cancelled by user.")
                cv2.destroyAllWindows()
                return False

            # Auto-advance once all target points are collected
            if len(self.pitch_points) == n_target:
                print(f"  ✔  All {n_target} pitch points collected. Computing matrix…\n")
                break

        cv2.destroyAllWindows()
        return True

    # ──────────────────────────────────────────
    #  Phase 3 — compute H
    # ──────────────────────────────────────────

    def _phase3_compute(self):
        src_pts = np.array(self.frame_points, dtype=np.float32)
        dst_pts = np.array(self.pitch_points, dtype=np.float32)

        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("  ❌ Homography computation failed.")
            print("     Likely cause: points are collinear (on the same line).")
            print("     Fix: spread your clicks so they form a quadrilateral.")
            return None, None, None, None

        inliers = int(np.sum(status))
        n       = len(self.frame_points)
        print(f"  ✔  Matrix computed.  RANSAC inliers: {inliers}/{n}")

        if inliers < self.MIN_POINTS:
            print(f"  ⚠  Only {inliers} inliers — matrix may be unreliable.")
            print("     Recommendation: re-run with better-spread points.")

        return H, status, src_pts, dst_pts

    # ──────────────────────────────────────────
    #  Phase 4 — sanity check (warp overlay)
    # ──────────────────────────────────────────

    def _phase4_sanity(self, H: np.ndarray) -> None:
        """
        True homography validation:
        Warp the video frame into the pitch coordinate system and
        blend it on top of a clean pitch canvas.

        If H is correct → the grass lines in the warped frame will align
        with the white lines on the 2D pitch map.

        If twisted / skewed → the click ORDER was inconsistent.
        """
        print("\n  🔍 Opening Sanity Check window…")
        print("     Green lines  = 2D pitch map (ground truth)")
        print("     Grass/player overlay = video frame warped by H")
        print("     ✔ Good result : grass lines sit on top of green lines")
        print("     ✖ Bad result  : field looks twisted or flipped")
        print("     Press any key to close.\n")

        # Always work from CLEAN images — never a marked-up canvas
        pitch_clean = self._load_clean(self.pitch_path)
        frame_clean = self._load_clean(self.frame_path)

        h_p, w_p = pitch_clean.shape[:2]

        # Warp the broadcast frame into the pitch plane
        warped = cv2.warpPerspective(frame_clean, H, (w_p, h_p))

        # Blend: 60% pitch map + 40% warped frame
        blended = cv2.addWeighted(pitch_clean, 0.6, warped, 0.4, 0)

        # Overlay the original pitch points so we can cross-check alignment
        for i, pt in enumerate(self.pitch_points):
            _draw_point(blended, pt, i + 1, COLORS["blue"])

        cv2.namedWindow("Sanity Check — Do the white lines align?",
                        cv2.WINDOW_NORMAL)
        cv2.imshow("Sanity Check — Do the white lines align?", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ──────────────────────────────────────────
    #  Phase 5 — persist
    # ──────────────────────────────────────────

    def _phase5_save(self, H, status, src_pts, dst_pts) -> None:
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        np.savez(
            self.out_path,
            H         = H,
            frame_pts = src_pts,
            pitch_pts = dst_pts,
            status    = status,
        )
        print(f"  💾 Saved to: {self.out_path}")
        print(f"     Keys: H, frame_pts, pitch_pts, status\n")

    # ──────────────────────────────────────────
    #  Public entry-point
    # ──────────────────────────────────────────

    def run(self) -> None:
        # Validate paths upfront
        for p in (self.frame_path, self.pitch_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Input not found: {p}")

        print("\n  goalX — Homography Picker")
        print("  " + "─" * 40)

        if not self._phase1_pick_frame():
            return
        if not self._phase2_pick_pitch():
            return

        H, status, src_pts, dst_pts = self._phase3_compute()
        if H is None:
            return

        self._phase4_sanity(H)
        self._phase5_save(H, status, src_pts, dst_pts)

        print("  ✅  Pipeline complete.")
        print("  Next step: run project_tracks.py with --homography data/homography_data.npz\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Interactive homography point-picker for goalX."
    )
    p.add_argument("--frame",  required=True,
                   help="Path to a single broadcast video frame (.jpg/.png)")
    p.add_argument("--pitch",  required=True,
                   help="Path to the 2D metric pitch map (from draw_pitch.py)")
    p.add_argument("--out",    default="data/homography_data.npz",
                   help="Output .npz path  (default: data/homography_data.npz)")
    p.add_argument("--min-pts", type=int, default=4,
                   help="Minimum correspondence points  (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    HomographyPicker.MIN_POINTS = args.min_pts
    picker = HomographyPicker(args.frame, args.pitch, args.out)
    picker.run()