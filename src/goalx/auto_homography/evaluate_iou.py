"""
evaluate_iou.py
───────────────
Mean IOU evaluation for homography registration quality.

METRIC DEFINITION (following Sharma et al. §4.1.2)
────────────────────────────────────────────────────
For each frame:
  1. Project the frame boundary (image corners) onto the pitch canvas
     using the PREDICTED H.
  2. Project the same corners using the GROUND TRUTH H.
  3. Compute the polygon intersection over union (IOU) of the two
     projected quadrilaterals on the pitch canvas.
  4. Mean IOU over all evaluated frames = the reported accuracy.

WHY POLYGON IOU NOT PIXEL IOU
───────────────────────────────
Pixel-level IOU would require warping the entire frame at each evaluation,
which is O(N × H × W) and slow.  Polygon IOU from 4 projected corners
gives an identical result (both quadrilaterals represent the camera's
field of view on the pitch) and runs in microseconds per frame.

GROUND TRUTH FORMAT
────────────────────
You need at least 20–30 manually-labelled frames to get statistically
meaningful numbers.  These can come from:
  (a) homography_picker.py sessions saved to separate .npz files per frame
  (b) A CSV with columns: frame_id, h00..h22 (same as homographies.csv)

The script computes IOU for every frame_id that exists in BOTH the
predicted CSV and the ground truth CSV.

EXPECTED RESULTS (per Sharma Table 2)
───────────────────────────────────────
  SNMOT-193 (top-down broadcast)  : ~88% mean IOU  (Milestone 1 target)
  SNMOT-116 (corner camera)       : ~45% mean IOU  (Milestone 2: failure case)

  After player masking (Phase 2):
  SNMOT-116 with masking          : target >75% mean IOU  (thesis contribution)

CLI
───
  python -m goalx.sharma_2018.evaluate_iou \\
      --predicted   outputs/sharma_H/homographies.csv \\
      --ground-truth data/gt_homographies.csv \\
      --pitch        data/pitch_map.png \\
      --out-dir      outputs/eval/

  # To generate ground truth CSV from homography_picker .npz files:
  python -m goalx.sharma_2018.evaluate_iou --make-gt \\
      --gt-npz-dir data/gt_labels/ \\
      --out        data/gt_homographies.csv

Output
──────
  outputs/eval/
      iou_results.csv      — per-frame IOU values
      summary.json         — mean, median, std IOU + comparison info
      iou_histogram.png    — distribution plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
#  Polygon IOU
# ─────────────────────────────────────────────────────────────────

def _project_frame_corners(H: np.ndarray,
                             frame_w: int = 1280,
                             frame_h: int = 720) -> np.ndarray:
    """
    Project the 4 frame corners through H.
    Returns a (4, 2) float32 array in pitch-canvas coordinates.
    """
    corners = np.array([
        [0,       0],
        [frame_w, 0],
        [frame_w, frame_h],
        [0,       frame_h],
    ], dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)
    return projected.reshape(-1, 2)


def polygon_iou(quad1: np.ndarray, quad2: np.ndarray,
                canvas_w: int = 1050, canvas_h: int = 680) -> float:
    """
    Compute IOU of two projected quadrilaterals on the pitch canvas.

    Uses OpenCV's fillPoly on a binary canvas — simple, numerically robust,
    handles non-convex (degenerate) projections correctly.

    WHY RASTERIZE NOT SHAPELY: For small canvases (1050×680) rasterisation
    is faster than Shapely polygon intersection for convex quads and avoids
    the Shapely dependency.  The pixel-level IOU is numerically identical to
    the continuous polygon IOU for well-behaved projections.
    """
    mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask2 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    pts1 = quad1.astype(np.int32).reshape(-1, 1, 2)
    pts2 = quad2.astype(np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(mask1, [pts1], 255)
    cv2.fillPoly(mask2, [pts2], 255)

    intersection = int(np.logical_and(mask1, mask2).sum())
    union        = int(np.logical_or (mask1, mask2).sum())

    if union == 0:
        return 0.0
    return intersection / union


# ─────────────────────────────────────────────────────────────────
#  Ground truth helpers
# ─────────────────────────────────────────────────────────────────

def make_gt_csv(gt_npz_dir: Path, out_csv: Path) -> pd.DataFrame:
    """
    Build a ground-truth CSV from a directory of homography_picker .npz files.

    Each .npz file must contain key 'H' and the filename must be the frame_id
    (e.g., 000001.npz → frame_id=1).

    This is the format produced by running homography_picker.py once per
    manually-labelled frame with --out data/gt_labels/000001.npz etc.
    """
    rows = []
    for npz_path in sorted(gt_npz_dir.glob("*.npz")):
        data    = np.load(str(npz_path))
        H       = data["H"].astype(np.float32)
        frame_id = int(npz_path.stem)
        rows.append({
            "frame_id": frame_id,
            "h00": H[0, 0], "h01": H[0, 1], "h02": H[0, 2],
            "h10": H[1, 0], "h11": H[1, 1], "h12": H[1, 2],
            "h20": H[2, 0], "h21": H[2, 1], "h22": H[2, 2],
        })

    df = pd.DataFrame(rows).sort_values("frame_id", ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv), index=False)
    print(f"  Ground truth CSV: {len(df)} frames → {out_csv}")
    return df


def _h_from_row(row: pd.Series) -> np.ndarray:
    return np.array([
        [row["h00"], row["h01"], row["h02"]],
        [row["h10"], row["h11"], row["h12"]],
        [row["h20"], row["h21"], row["h22"]],
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────────────────────────

def run_evaluation(
    predicted_csv: Path,
    gt_csv:        Path,
    pitch_path:    Path,
    out_dir:       Path,
    frame_w:       int = 1280,
    frame_h:       int = 720,
    label:         str = "Sharma HOG",
) -> dict:
    """
    Compute per-frame and aggregate IOU metrics.

    Returns a dict with mean_iou, median_iou, std_iou.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pitch canvas dimensions from pitch PNG
    pitch_img = cv2.imread(str(pitch_path))
    if pitch_img is None:
        canvas_w, canvas_h = 1050, 680
    else:
        canvas_h, canvas_w = pitch_img.shape[:2]

    df_pred = pd.read_csv(str(predicted_csv))
    df_gt   = pd.read_csv(str(gt_csv))

    # Join on frame_id
    df_pred.set_index("frame_id", inplace=True)
    df_gt.set_index  ("frame_id", inplace=True)
    common_ids = df_pred.index.intersection(df_gt.index)

    if len(common_ids) == 0:
        raise ValueError(
            "No common frame_ids between predicted and ground truth CSVs. "
            "Check that frame_ids match."
        )

    print(f"\n  goalX / Sharma 2018 — IOU Evaluator")
    print(f"  {'─' * 40}")
    print(f"  Predicted  : {len(df_pred)} frames")
    print(f"  Ground truth: {len(df_gt)} frames")
    print(f"  Overlap    : {len(common_ids)} frames evaluated")

    iou_rows = []
    for fid in sorted(common_ids):
        H_pred = _h_from_row(df_pred.loc[fid])
        H_gt   = _h_from_row(df_gt.loc[fid])

        quad_pred = _project_frame_corners(H_pred, frame_w, frame_h)
        quad_gt   = _project_frame_corners(H_gt,   frame_w, frame_h)

        iou = polygon_iou(quad_pred, quad_gt, canvas_w, canvas_h)
        dist_pred = float(df_pred.loc[fid].get("match_distance", -1))

        iou_rows.append({
            "frame_id":       fid,
            "iou":            iou,
            "match_distance": dist_pred,
        })

    df_iou = pd.DataFrame(iou_rows)

    # ── Aggregate stats ───────────────────────────────────────────
    mean_iou   = float(df_iou["iou"].mean() * 100)
    median_iou = float(df_iou["iou"].median() * 100)
    std_iou    = float(df_iou["iou"].std() * 100)
    pct_over80 = float((df_iou["iou"] >= 0.80).mean() * 100)
    pct_over90 = float((df_iou["iou"] >= 0.90).mean() * 100)

    summary = {
        "label":       label,
        "n_frames":    len(df_iou),
        "mean_iou":    round(mean_iou,   2),
        "median_iou":  round(median_iou, 2),
        "std_iou":     round(std_iou,    2),
        "pct_over80":  round(pct_over80, 2),
        "pct_over90":  round(pct_over90, 2),
    }

    # Print results table
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  IOU Results — {label:<26}│")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Mean IOU      : {mean_iou:>6.1f}%                │")
    print(f"  │  Median IOU    : {median_iou:>6.1f}%                │")
    print(f"  │  Std dev       : {std_iou:>6.1f}%                │")
    print(f"  │  Frames ≥ 80%  : {pct_over80:>6.1f}%                │")
    print(f"  │  Frames ≥ 90%  : {pct_over90:>6.1f}%                │")
    print(f"  └─────────────────────────────────────────┘")

    # Context against paper benchmarks
    print(f"\n  Benchmark comparison:")
    print(f"    Paper (top-down broadcast) : ~91.4% mean IOU  [Table 2]")
    print(f"    Expected (SNMOT-193)       : ~88%   mean IOU  (Milestone 1)")
    print(f"    Expected (SNMOT-116 corner): ~45%   mean IOU  (Milestone 2)")
    print(f"    Your result ({label}): {mean_iou:.1f}%")

    if mean_iou >= 85:
        print(f"    → ✅  Milestone 1 achieved (≥85% on standard broadcast)")
    elif mean_iou <= 55:
        print(f"    → ✅  Milestone 2 proven (baseline fails on corner camera)")
    elif mean_iou >= 70:
        print(f"    → 🔶  Phase 2 showing improvement over baseline")

    # Save outputs
    iou_csv  = out_dir / "iou_results.csv"
    json_out = out_dir / "summary.json"
    df_iou.to_csv(str(iou_csv), index=False)
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)

    # Histogram plot (optional — requires matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df_iou["iou"] * 100, bins=20, range=(0, 100),
                color="#1D9E75", edgecolor="white", linewidth=0.5)
        ax.axvline(mean_iou, color="#D85A30", linestyle="--",
                   linewidth=1.5, label=f"Mean = {mean_iou:.1f}%")
        ax.axvline(88, color="#666", linestyle=":", linewidth=1,
                   label="Paper target (88%)")
        ax.set_xlabel("IOU (%)")
        ax.set_ylabel("Frames")
        ax.set_title(f"IOU Distribution — {label}")
        ax.legend()
        ax.set_xlim(0, 100)
        plt.tight_layout()
        hist_path = out_dir / "iou_histogram.png"
        plt.savefig(str(hist_path), dpi=120)
        plt.close()
        print(f"\n  Histogram → {hist_path}")
    except ImportError:
        pass

    print(f"\n  ✅  Results saved:")
    print(f"      {iou_csv}")
    print(f"      {json_out}\n")
    return summary


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute mean IOU for homography registration quality."
    )
    subparsers = p.add_subparsers(dest="command")

    # Evaluate command
    ev = subparsers.add_parser("eval", help="Run IOU evaluation")
    ev.add_argument("--predicted",    required=True,
                    help="homographies.csv from hog_matcher.py or mrf_smoother.py")
    ev.add_argument("--ground-truth", required=True,
                    help="CSV with columns frame_id, h00..h22 (manually labelled)")
    ev.add_argument("--pitch",        required=True,
                    help="pitch_map.png from draw_pitch.py")
    ev.add_argument("--out-dir",      default="outputs/eval")
    ev.add_argument("--frame-w",      type=int, default=1280)
    ev.add_argument("--frame-h",      type=int, default=720)
    ev.add_argument("--label",        default="Sharma HOG")

    # Make GT command
    gt = subparsers.add_parser("make-gt",
                               help="Build ground truth CSV from .npz label files")
    gt.add_argument("--npz-dir", required=True,
                    help="Directory of per-frame homography_picker .npz files")
    gt.add_argument("--out",     required=True,
                    help="Output CSV path")

    # Default to eval if no subcommand given
    p.add_argument("--predicted",    default=None)
    p.add_argument("--ground-truth", default=None)
    p.add_argument("--pitch",        default=None)
    p.add_argument("--out-dir",      default="outputs/eval")
    p.add_argument("--frame-w",      type=int, default=1280)
    p.add_argument("--frame-h",      type=int, default=720)
    p.add_argument("--label",        default="Sharma HOG")
    p.add_argument("--make-gt",      action="store_true")
    p.add_argument("--gt-npz-dir",   default=None)
    p.add_argument("--out",          default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.make_gt or (hasattr(args, "command") and args.command == "make-gt"):
        npz_dir  = args.gt_npz_dir or (args.npz_dir if hasattr(args, "npz_dir") else None)
        out_file = args.out
        if not npz_dir or not out_file:
            print("Usage: --make-gt --gt-npz-dir DIR --out CSV")
        else:
            make_gt_csv(Path(npz_dir), Path(out_file))
    else:
        run_evaluation(
            predicted_csv = Path(args.predicted),
            gt_csv        = Path(args.ground_truth),
            pitch_path    = Path(args.pitch),
            out_dir       = Path(args.out_dir),
            frame_w       = args.frame_w,
            frame_h       = args.frame_h,
            label         = args.label,
        )
