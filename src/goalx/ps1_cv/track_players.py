import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

# Import OFFICIAL ByteTrack (vendored)
from goalx.ps1_cv.bytetrack.byte_tracker import BYTETracker

# ---------------- CONFIG ----------------
# 1. Point to the NEW joint detection file
INPUT_CSV = "outputs/detections_with_ball.csv"
OUTPUT_CSV = "data/tracking_SNMOT-116.csv"
SAMPLE_IMG = "data/raw_videos/tracking/test/SNMOT-116/img1/000001.jpg"

# ByteTrack hyperparameters
TRACK_THRESH = 0.20
MATCH_THRESH = 0.9
TRACK_BUFFER = 90
# ---------------------------------------

def run_tracking():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing {INPUT_CSV}")

    img = cv2.imread(SAMPLE_IMG)
    if img is None:
        raise FileNotFoundError("Sample image not found.")

    H, W, _ = img.shape
    df = pd.read_csv(INPUT_CSV)
    
    # detect_ball.py outputs the column as 'frame'
    frames = sorted(df["frame"].unique())

    class TrackerArgs:
        track_thresh = TRACK_THRESH
        match_thresh = MATCH_THRESH
        track_buffer = TRACK_BUFFER
        mot20 = False

    tracker = BYTETracker(TrackerArgs())
    tracked_rows = []

    for f_id in tqdm(frames, desc="Linking Player IDs"):
        frame_df = df[df["frame"] == f_id]

        # --- A. Bypass the ball (Don't track it, just pass it through) ---
        ball_df = frame_df[frame_df["object_type"] == "ball"]
        for _, b in ball_df.iterrows():
            tracked_rows.append({
                "frame_id": f_id,  # Rename to frame_id so project_tracks.py is happy
                "track_id": -1,    # Ball gets -1
                "x1": b["x1"],
                "y1": b["y1"],
                "x2": b["x2"],
                "y2": b["y2"],
                "class_id": 32,    # COCO Ball
                "conf": b["conf"]
            })

        # --- B. Run ByteTrack on Players ---
        players_df = frame_df[frame_df["object_type"] == "player"]
        if not players_df.empty:
            dets = players_df[["x1", "y1", "x2", "y2", "conf"]].values
            dets = dets.astype(np.float32)

            online_targets = tracker.update(
                dets,
                img_info=(H, W),
                img_size=(H, W)
            )

            for t in online_targets:
                tlwh = t.tlwh 
                tracked_rows.append({
                    "frame_id": f_id,  # Rename to frame_id
                    "track_id": t.track_id,
                    "x1": tlwh[0],
                    "y1": tlwh[1],
                    "x2": tlwh[0] + tlwh[2],
                    "y2": tlwh[1] + tlwh[3],
                    "class_id": 0,     # COCO Person
                    "conf": t.score
                })

    # --------- Save Results ---------
    result_df = pd.DataFrame(tracked_rows)
    result_df.to_csv(OUTPUT_CSV, index=False)

    n_players = result_df[result_df["class_id"] == 0]["track_id"].nunique()
    print("\n✅ Tracking complete!")
    print(f"📄 Saved: {OUTPUT_CSV}")
    print(f"🧍 Unique player IDs: {n_players}")

if __name__ == "__main__":
    run_tracking()