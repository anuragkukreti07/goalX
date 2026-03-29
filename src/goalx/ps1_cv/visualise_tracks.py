"""
visualise_tracks.py
───────────────────
Renders ByteTrack output as a broadcast-style video with coloured
bounding boxes, ID labels, and ball highlighting.

Bug fixes vs v1
───────────────
  • v1 read df['frame'] but track_players.py writes 'frame_id' → crash
  • Ball (track_id = -1) now rendered distinctly (orange circle)
  • get_color now uses a deterministic palette instead of random to give
    visually distinct stable colours per ID
  • Missing frame images are now properly skipped with a warning counter
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

CSV_PATH  = "data/tracking_SNMOT-116.csv"
IMG_DIR   = "data/raw_videos/tracking/test/SNMOT-116/img1/"
OUT_PATH  = "tracking_demo.mp4"
FPS       = 25

# ─────────────────────────────────────────────────────────────────
#  Colour palette — 20 visually distinct BGR colours for track IDs
# ─────────────────────────────────────────────────────────────────

_PALETTE = [
    (255,  60,  60), (60,  255,  60), (60,   60, 255),
    (255, 200,  60), (200,  60, 255), (60,  200, 255),
    (255, 120, 200), (120, 255, 200), (200, 255, 120),
    (180, 180,  60), (60,  180, 180), (180,  60, 180),
    (255, 150,  80), (150, 255,  80), (80,  150, 255),
    (255,  80, 150), (80,  255, 150), (150,  80, 255),
    (220, 220, 100), (100, 220, 220),
]


def _color_for(track_id: int) -> tuple:
    if track_id < 0:
        return (255, 255, 255)
    return _PALETTE[int(track_id) % len(_PALETTE)]


def run_viz(
    csv_path: str = CSV_PATH,
    img_dir:  str = IMG_DIR,
    out_path: str = OUT_PATH,
    fps:      int = FPS,
) -> None:

    csv_path = Path(csv_path)
    img_dir  = Path(img_dir)
    out_path = Path(out_path)

    print(f"\n  goalX — Track Visualiser")
    print(f"  {'─' * 40}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {img_dir}")

    df = pd.read_csv(csv_path)

    # ── Resolve the frame column (handle both naming conventions) ──
    if "frame_id" in df.columns:
        frame_col = "frame_id"
    elif "frame" in df.columns:
        frame_col = "frame"
    else:
        raise ValueError(f"CSV has neither 'frame_id' nor 'frame' column. "
                         f"Columns found: {list(df.columns)}")

    frames = sorted(df[frame_col].unique())
    print(f"  ✔  {len(df)} detections  |  {len(frames)} frames")

    # ── Determine video dimensions from first frame ────────────────
    first_path = img_dir / f"{int(frames[0]):06d}.jpg"
    first_img  = cv2.imread(str(first_path))
    if first_img is None:
        raise FileNotFoundError(f"Cannot read first frame: {first_path}")
    h_img, w_img = first_img.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_img, h_img)
    )

    skipped = 0
    for f_id in tqdm(frames, desc="Rendering Video", unit="frame"):
        img_path = img_dir / f"{int(f_id):06d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        f_df = df[df[frame_col] == f_id]

        for _, row in f_df.iterrows():
            tid = int(row["track_id"])

            # ── Clamp coords to frame bounds ──
            x1 = int(np.clip(row["x1"], 0, w_img - 1))
            y1 = int(np.clip(row["y1"], 0, h_img - 1))
            x2 = int(np.clip(row["x2"], 0, w_img - 1))
            y2 = int(np.clip(row["y2"], 0, h_img - 1))

            if tid == -1:
                # Ball — orange filled circle on foot-centre
                bx = (x1 + x2) // 2
                by = y2
                cv2.circle(img, (bx, by), 8, (0, 0, 0), -1)
                cv2.circle(img, (bx, by), 6, (0, 200, 255), -1)
                continue

            color = _color_for(tid)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # ID label with filled background
            label = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_y = max(y1, th + 4)
            cv2.rectangle(img,
                          (x1, label_y - th - 4),
                          (x1 + tw + 4, label_y),
                          color, -1)
            cv2.putText(img, label, (x1 + 2, label_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # Frame counter watermark
        cv2.putText(img, f"frame {int(f_id):06d}",
                    (10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2, cv2.LINE_AA)

        writer.write(img)

    writer.release()

    if skipped:
        print(f"\n  ⚠️  {skipped} frame images not found (sequence gaps).")
    print(f"\n  ✅  Video saved → {out_path}")
    print(f"  Play with: vlc {out_path}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Render ByteTrack output as annotated video."
    )
    p.add_argument("--csv",    default=CSV_PATH,
                   help="Tracking CSV from track_players.py")
    p.add_argument("--frames", default=IMG_DIR,
                   help="Directory of original .jpg frames")
    p.add_argument("--out",    default=OUT_PATH)
    p.add_argument("--fps",    type=int, default=FPS)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_viz(
        csv_path = args.csv,
        img_dir  = args.frames,
        out_path = args.out,
        fps      = args.fps,
    )