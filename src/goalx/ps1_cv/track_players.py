"""
track_players.py
────────────────
Links per-frame player detections into persistent track IDs using the
vendored ByteTrack algorithm.  Ball detections (class 32, track_id -1)
are passed through unchanged — ByteTrack only runs on players.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from goalx.ps1_cv.bytetrack.byte_tracker import BYTETracker

# ─────────────────────────────────────────────────────────────────
#  CONFIG  (overridden by CLI args)
# ─────────────────────────────────────────────────────────────────

# ByteTrack hyperparameters
TRACK_THRESH  = 0.50    # minimum detection confidence to init a track
MATCH_THRESH  = 0.80    # IoU threshold for track-detection association
TRACK_BUFFER  = 90      # frames to keep a lost track alive (3 s @ 25 fps)


# ─────────────────────────────────────────────────────────────────
#  Tracker args dataclass
# ─────────────────────────────────────────────────────────────────

class _TrackerArgs:
    def __init__(self, track_thresh, match_thresh, track_buffer):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.mot20        = False


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def run_tracking(
    input_csv:  str,
    output_csv: str,
    sample_img: str,
    track_thresh:  float = TRACK_THRESH,
    match_thresh:  float = MATCH_THRESH,
    track_buffer:  int   = TRACK_BUFFER,
) -> pd.DataFrame:

    input_csv  = Path(input_csv)
    output_csv = Path(output_csv)
    sample_img = Path(sample_img)

    # ── Validate inputs ───────────────────────────────────────────
    if not input_csv.exists():
        raise FileNotFoundError(f"Detection CSV not found: {input_csv}")
    if not sample_img.exists():
        raise FileNotFoundError(
            f"Sample image not found: {sample_img}\n"
            f"ByteTrack needs frame dimensions — provide any .jpg from the sequence."
        )

    print(f"\n  goalX — ByteTrack Player ID Linker")
    print(f"  {'─' * 40}")

    img = cv2.imread(str(sample_img))
    H, W = img.shape[:2]
    print(f"  Frame size: {W}×{H}")

    df = pd.read_csv(input_csv)
    
    # --- FIX 1: Automatic Frame Column Detection ---
    frame_col = "frame_id" if "frame_id" in df.columns else "frame"
    
    # --- FIX 2: Automatic Class Column Detection ---
    if "object_type" in df.columns:
        class_col, val_player, val_ball = "object_type", "player", "ball"
    elif "name" in df.columns:
        class_col, val_player, val_ball = "name", "person", "sports ball"
    elif "class_id" in df.columns:
        class_col, val_player, val_ball = "class_id", 0, 32
    elif "class" in df.columns:
        class_col, val_player, val_ball = "class", 0, 32
    else:
        raise ValueError(f"Could not find a valid class column in detections. Found: {df.columns.tolist()}")

    frames = sorted(df[frame_col].unique())
    print(f"  Detections: {len(df)}  |  frames: {len(frames)}")

    tracker = BYTETracker(
        _TrackerArgs(track_thresh, match_thresh, track_buffer)
    )

    tracked_rows: list[dict] = []

    for f_id in tqdm(frames, desc="Linking Player IDs", unit="frame"):
        frame_df = df[df[frame_col] == f_id]

        # ── Pass ball through (no tracking — one ball per frame) ──
        ball_df = frame_df[frame_df[class_col] == val_ball]
        for _, b in ball_df.iterrows():
            tracked_rows.append({
                "frame_id": int(f_id),  # Standardizing output to frame_id
                "track_id": -1,
                "x1": float(b["x1"]), "y1": float(b["y1"]),
                "x2": float(b["x2"]), "y2": float(b["y2"]),
                "class_id": 32,
                "conf":     float(b["conf"]),
            })

        # ── Run ByteTrack on player detections ────────────────────
        players_df = frame_df[frame_df[class_col] == val_player]
        if players_df.empty:
            continue

        dets = players_df[["x1", "y1", "x2", "y2", "conf"]].values.astype(np.float32)

        online_targets = tracker.update(
            dets,
            img_info=(H, W),
            img_size=(H, W),
        )

        for t in online_targets:
            tlwh = t.tlwh    # (top-left x, top-left y, w, h)
            tracked_rows.append({
                "frame_id": int(f_id),
                "track_id": int(t.track_id),
                "x1": float(tlwh[0]),
                "y1": float(tlwh[1]),
                "x2": float(tlwh[0] + tlwh[2]),
                "y2": float(tlwh[1] + tlwh[3]),
                "class_id": 0,
                "conf":     float(t.score),
            })

    result_df = pd.DataFrame(tracked_rows)
    
    # Safely create parent directories
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    n_players = result_df[result_df["class_id"] == 0]["track_id"].nunique()
    n_ball_f  = result_df[result_df["class_id"] == 32]["frame_id"].nunique()

    print(f"\n  📊 Tracking summary:")
    print(f"     Unique player IDs : {n_players}")
    print(f"     Ball frames       : {n_ball_f} / {len(frames)}")
    print(f"\n  💾 Saved → {output_csv}")
    print(f"  ✅  Tracking complete.")
    print(f"      Next step: homography_picker.py → project_tracks.py\n")

    return result_df


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Link detections into persistent track IDs via ByteTrack."
    )
    p.add_argument("--detections",  required=True, help="Detection CSV from detect_players_full.py")
    p.add_argument("--out",         required=True, help="Output tracking CSV")
    p.add_argument("--sample-img",  required=True, help="Any .jpg from the frame sequence")
    
    p.add_argument("--track-thresh", type=float, default=TRACK_THRESH)
    p.add_argument("--match-thresh", type=float, default=MATCH_THRESH)
    p.add_argument("--track-buffer", type=int,   default=TRACK_BUFFER)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_tracking(
        input_csv    = args.detections,
        output_csv   = args.out,
        sample_img   = args.sample_img,
        track_thresh = args.track_thresh,
        match_thresh = args.match_thresh,
        track_buffer = args.track_buffer,
    )
