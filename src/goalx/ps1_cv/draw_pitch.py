"""
draw_pitch.py
─────────────
Generates the canonical 2D metric pitch canvas used throughout the goalX
pipeline as the destination space for homography projection.

Scale contract  (critical — must not be changed without updating calibrate_full_pitch.py)
───────────────
  PITCH_SCALE = 10 px / metre
  Canvas size  = 1050 × 680 px  (105 m × 68 m)

  Every coordinate in the goalX pipeline is expressed in PITCH-CANVAS
  pixels. The homography H maps:
    image-space pixels  →  pitch-canvas pixels

  Because PITCH_SCALE is fixed and documented, converting to metric
  is always:  x_metres = pitch_x_px / PITCH_SCALE
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────
#  Scale contract  ← DO NOT CHANGE
# ─────────────────────────────────────────────────────────────────

PITCH_SCALE: float = 10.0   # pixels per metre

# FIFA regulation A-standard dimensions
PITCH_W_M: float = 105.0
PITCH_H_M: float = 68.0

# Canvas dimensions in pixels
CANVAS_W: int = int(PITCH_W_M * PITCH_SCALE)  # 1050
CANVAS_H: int = int(PITCH_H_M * PITCH_SCALE)  # 680

# Margin around the field on the canvas (purely aesthetic)
MARGIN: int = 30


# ─────────────────────────────────────────────────────────────────
#  Coordinate helpers
# ─────────────────────────────────────────────────────────────────

def _px(x_m: float) -> int:
    """Convert metres (from left/top of pitch) → canvas pixel (x or y)."""
    return int(round(x_m * PITCH_SCALE))


def _pt(x_m: float, y_m: float) -> tuple[int, int]:
    return _px(x_m), _px(y_m)


# ─────────────────────────────────────────────────────────────────
#  FIFA pitch geometry constants  (all in metres)
# ─────────────────────────────────────────────────────────────────

_W   = PITCH_W_M
_H   = PITCH_H_M
_CX  = _W / 2
_CY  = _H / 2

_PA  = 16.5    # penalty area depth
_PAW = 20.16   # penalty area half-width  (total 40.32 m)
_GA  = 5.5     # goal area depth
_GAW = 9.16    # goal area half-width
_PS  = 11.0    # penalty spot distance from goal line
_CR  = 9.15    # centre circle radius
_ARC = 9.15    # penalty arc radius (same as centre circle)
_CA  = 1.0     # corner arc radius


# ─────────────────────────────────────────────────────────────────
#  Pitch painter
# ─────────────────────────────────────────────────────────────────

# Colors (BGR)
_GRASS   = (34,  102,  34)    # dark green field
_LINE    = (255, 255, 255)    # white line
_SPOT    = (255, 255, 255)    # penalty & centre spots
_GRASS2  = (30,   95,  30)    # alternating stripe (subtle)


def _draw_pitch(canvas: np.ndarray,
                stripe: bool = True) -> None:
    """
    Draw all FIFA-standard pitch markings onto the canvas.
    All coordinates are computed from the metric constants above.
    """
    # ── Grass stripes  (10 m wide alternating) ─────────────────
    if stripe:
        for i in range(int(_W / 10) + 1):
            x1 = _px(i * 10)
            x2 = min(_px((i + 1) * 10), CANVAS_W)
            col = _GRASS2 if i % 2 == 0 else _GRASS
            cv2.rectangle(canvas, (x1, 0), (x2, CANVAS_H), col, -1)
    else:
        canvas[:] = _GRASS

    # ── Outer boundary ─────────────────────────────────────────
    cv2.rectangle(canvas, _pt(0, 0), _pt(_W, _H), _LINE, 2)

    # ── Halfway line ───────────────────────────────────────────
    cv2.line(canvas, _pt(_CX, 0), _pt(_CX, _H), _LINE, 2)

    # ── Centre circle + spot ───────────────────────────────────
    cv2.circle(canvas, _pt(_CX, _CY), _px(_CR), _LINE, 2)
    cv2.circle(canvas, _pt(_CX, _CY), 5, _SPOT, -1)

    # ── Left penalty area ──────────────────────────────────────
    cv2.rectangle(canvas,
                  _pt(0,   _CY - _PAW),
                  _pt(_PA, _CY + _PAW),
                  _LINE, 2)

    # ── Right penalty area ─────────────────────────────────────
    cv2.rectangle(canvas,
                  _pt(_W - _PA, _CY - _PAW),
                  _pt(_W,       _CY + _PAW),
                  _LINE, 2)

    # ── Left goal area ─────────────────────────────────────────
    cv2.rectangle(canvas,
                  _pt(0,   _CY - _GAW),
                  _pt(_GA, _CY + _GAW),
                  _LINE, 2)

    # ── Right goal area ────────────────────────────────────────
    cv2.rectangle(canvas,
                  _pt(_W - _GA, _CY - _GAW),
                  _pt(_W,       _CY + _GAW),
                  _LINE, 2)

    # ── Penalty spots ──────────────────────────────────────────
    cv2.circle(canvas, _pt(_PS,      _CY), 5, _SPOT, -1)
    cv2.circle(canvas, _pt(_W - _PS, _CY), 5, _SPOT, -1)

    # ── Penalty arcs  (only the part outside the penalty area) ─
    cv2.ellipse(canvas, _pt(_PS, _CY),
                (_px(_ARC), _px(_ARC)),
                0, -53, 53,    # only visible arc outside box
                _LINE, 2)

    cv2.ellipse(canvas, _pt(_W - _PS, _CY),
                (_px(_ARC), _px(_ARC)),
                0, 180 - 53, 180 + 53,
                _LINE, 2)

    # ── Corner arcs ────────────────────────────────────────────
    for cx_m, cy_m, start_angle in [
        (0,  0,   0),    # top-left
        (_W, 0,   90),   # top-right
        (_W, _H, 180),   # bottom-right
        (0,  _H, 270),   # bottom-left
    ]:
        cv2.ellipse(canvas, _pt(cx_m, cy_m),
                    (_px(_CA), _px(_CA)),
                    0, start_angle, start_angle + 90,
                    _LINE, 2)

    # ── Goals  (represented as thick rectangles) ───────────────
    goal_w = 7.32 / 2   # half goal width (total 7.32 m)
    goal_d = 2.44       # goal depth (2.44 m = post width visual proxy)

    cv2.rectangle(canvas,
                  _pt(-goal_d, _CY - goal_w),
                  _pt(0,       _CY + goal_w),
                  _LINE, 2)
    cv2.rectangle(canvas,
                  _pt(_W,          _CY - goal_w),
                  _pt(_W + goal_d, _CY + goal_w),
                  _LINE, 2)


# ─────────────────────────────────────────────────────────────────
#  Landmark overlay  (for calibration reference)
# ─────────────────────────────────────────────────────────────────

_LANDMARKS_META: list[tuple[int, str, float, float]] = [
    (1,  "Top-left corner",                0.0,       0.0),
    (2,  "Top-right corner",               _W,        0.0),
    (3,  "Bottom-right corner",            _W,        _H),
    (4,  "Bottom-left corner",             0.0,       _H),
    (5,  "Halfway top",                    _CX,       0.0),
    (6,  "Centre spot",                    _CX,       _CY),
    (7,  "Halfway bottom",                 _CX,       _H),
    (8,  "Left PA top-left",               0.0,       _CY - 20.16),
    (9,  "Left PA top-right",              _PA,       _CY - 20.16),
    (10, "Left PA bot-right",              _PA,       _CY + 20.16),
    (11, "Left PA bot-left",               0.0,       _CY + 20.16),
    (12, "Left penalty spot",              _PS,       _CY),
    (13, "Right PA top-right",             _W,        _CY - 20.16),
    (14, "Right PA top-left",              _W - _PA,  _CY - 20.16),
    (15, "Right PA bot-left",              _W - _PA,  _CY + 20.16),
    (16, "Right PA bot-right",             _W,        _CY + 20.16),
    (17, "Right penalty spot",             _W - _PS,  _CY),
    (18, "Left GA top-right",              _GA,       _CY - _GAW),
    (19, "Left GA bot-right",              _GA,       _CY + _GAW),
    (20, "Right GA top-left",              _W - _GA,  _CY - _GAW),
    (21, "Right GA bot-left",              _W - _GA,  _CY + _GAW),
]


def _draw_landmarks(canvas: np.ndarray) -> None:
    """Overlay numbered landmark dots onto the canvas."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for lid, name, xm, ym in _LANDMARKS_META:
        px, py = _pt(xm, ym)
        px = max(10, min(px, CANVAS_W - 10))
        py = max(10, min(py, CANVAS_H - 10))

        cv2.circle(canvas, (px, py), 9, (0, 220, 120), -1)
        cv2.circle(canvas, (px, py), 10, (0, 0, 0), 1)
        cv2.putText(canvas, str(lid),
                    (px + 12, py + 5), font, 0.55,
                    (255, 255, 200), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
#  Public API + CLI
# ─────────────────────────────────────────────────────────────────

def make_pitch(stripe: bool = True) -> np.ndarray:
    """Return a fresh BGR pitch canvas (CANVAS_W × CANVAS_H)."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = _GRASS
    _draw_pitch(canvas, stripe=stripe)
    return canvas


def make_landmark_reference(pitch_canvas: np.ndarray | None = None) -> np.ndarray:
    """Return a pitch canvas with numbered landmarks drawn on top."""
    canvas = pitch_canvas.copy() if pitch_canvas is not None else make_pitch()
    _draw_landmarks(canvas)
    return canvas


def run(out_path: Path,
        landmark_ref_path: Path | None = None,
        no_stripe: bool = False) -> None:

    print(f"\n  goalX — Pitch Map Generator")
    print(f"  {'─'*42}")
    print(f"  Scale  : {PITCH_SCALE:.0f} px/m")
    print(f"  Canvas : {CANVAS_W} × {CANVAS_H} px  "
          f"({PITCH_W_M:.0f} m × {PITCH_H_M:.0f} m)")

    pitch = make_pitch(stripe=not no_stripe)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), pitch)
    print(f"  ✔  Pitch map → {out_path}")

    if landmark_ref_path is not None:
        ref = make_landmark_reference(pitch)
        landmark_ref_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(landmark_ref_path), ref)
        print(f"  ✔  Landmark reference → {landmark_ref_path}")
        print(f"     Open this image to see which numbered points to click")
        print(f"     when running calibrate_full_pitch.py.")

    print(f"\n  Scale reference:")
    print(f"    pitch_x_pixels / {PITCH_SCALE:.0f}  =  x in metres")
    print(f"    pitch_y_pixels / {PITCH_SCALE:.0f}  =  y in metres\n")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate goalX 2D pitch map canvas."
    )
    p.add_argument("--out",          default="data/pitch_map.png",
                   help="Output path for the clean pitch PNG")
    p.add_argument("--landmark-ref", default=None,
                   help="Also write a numbered landmark reference image here")
    p.add_argument("--no-stripe",    action="store_true",
                   help="Disable alternating grass stripes")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        out_path          = Path(args.out),
        landmark_ref_path = Path(args.landmark_ref) if args.landmark_ref else None,
        no_stripe         = args.no_stripe,
    )