import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

# Import OFFICIAL ByteTrack (vendored)
from goalx.ps1_cv.bytetrack.byte_tracker import BYTETracker

# ---------------- CONFIG ----------------
INPUT_CSV = "data/detections_SNMOT-116_raw.csv"
OUTPUT_CSV = "data/tracking_SNMOT-116.csv"
SAMPLE_IMG = "data/raw_videos/tracking/test/SNMOT-116/img1/000001.jpg"

# ByteTrack hyperparameters (standard)
TRACK_THRESH = 0.20
MATCH_THRESH = 0.9
TRACK_BUFFER = 90
# ---------------------------------------


def run_tracking():
    # --------- Safety Checks ---------
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError("Detection CSV not found. Run Week 1 first.")

    img = cv2.imread(SAMPLE_IMG)
    if img is None:
        raise FileNotFoundError("Sample image not found.")

    H, W, _ = img.shape

    # --------- Load Detections ---------
    df = pd.read_csv(INPUT_CSV)
    frames = sorted(df["frame"].unique())

    # --------- Initialize Tracker ---------
    class TrackerArgs:
        track_thresh = TRACK_THRESH
        match_thresh = MATCH_THRESH
        track_buffer = TRACK_BUFFER
        mot20 = False

    tracker = BYTETracker(TrackerArgs())


    tracked_rows = []

    # --------- Tracking Loop ---------
    for frame_id in tqdm(frames, desc="Linking Player IDs"):
        frame_df = df[df["frame"] == frame_id]

        # ByteTrack expects: [x1, y1, x2, y2, score]
        dets = frame_df[["x1", "y1", "x2", "y2", "conf"]].values
        dets = dets.astype(np.float32)

        online_targets = tracker.update(
            dets,
            img_info=(H, W),
            img_size=(H, W)
        )

        for t in online_targets:
            tlwh = t.tlwh  # [x, y, w, h]

            tracked_rows.append({
                "frame": frame_id,
                "track_id": t.track_id,
                "x1": tlwh[0],
                "y1": tlwh[1],
                "x2": tlwh[0] + tlwh[2],
                "y2": tlwh[1] + tlwh[3],
                "class_id": 0,     # person (player-first)
                "conf": t.score
            })

    # --------- Save Results ---------
    result_df = pd.DataFrame(tracked_rows)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ Tracking complete!")
    print(f"📄 Saved: {OUTPUT_CSV}")
    print(f"🧍 Unique track IDs: {result_df['track_id'].nunique()}")


if __name__ == "__main__":
    run_tracking()




