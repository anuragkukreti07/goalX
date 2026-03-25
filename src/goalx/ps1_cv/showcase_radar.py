import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def run_showcase():
    # --- CONFIG ---
    TRACKS_CSV = "outputs/smoothed/smoothed_tracks.csv"
    EVENTS_CSV = "outputs/events/events.csv"
    PITCH_IMG  = "data/pitch_map.png"
    OUT_VID    = "outputs/ps1_tactical_radar.mp4"

    print("\n  goalX — Phase 1 Tactical Radar Showcase")
    print("  " + "─" * 40)

    # 1. Load Data
    if not os.path.exists(TRACKS_CSV) or not os.path.exists(PITCH_IMG):
        raise FileNotFoundError("Missing smoothed tracks or pitch map.")

    tracks = pd.read_csv(TRACKS_CSV)
    events = pd.read_csv(EVENTS_CSV) if os.path.exists(EVENTS_CSV) else pd.DataFrame()
    pitch_base = cv2.imread(PITCH_IMG)
    h, w, _ = pitch_base.shape

    out = cv2.VideoWriter(OUT_VID, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    frames = sorted(tracks["frame_id"].unique())

    for fid in tqdm(frames, desc="Rendering Radar Video"):
        img = pitch_base.copy()
        frame_tracks = tracks[tracks["frame_id"] == fid]

        # 2. Draw Players and Ball
        for _, row in frame_tracks.iterrows():
            tid = int(row["track_id"])
            
            # Skip rows where smoothing left NaNs (gaps in tracking)
            if pd.isna(row["smooth_x"]) or pd.isna(row["smooth_y"]):
                continue

            px, py = int(row["smooth_x"]), int(row["smooth_y"])

            # Defensive clamp to keep dots on the canvas
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))

            if tid == -1:
                # Draw Ball (Yellow with black border)
                cv2.circle(img, (px, py), 6, (0, 0, 0), -1)
                cv2.circle(img, (px, py), 4, (0, 255, 255), -1)
            else:
                # Draw Player (Red)
                cv2.circle(img, (px, py), 8, (0, 0, 255), -1)
                cv2.putText(img, str(tid), (px - 8, py - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 3. Draw Events HUD
        if not events.empty:
            frame_events = events[events["frame_id"] == fid]
            y_offset = 30
            
            # Add Frame Counter
            cv2.putText(img, f"Frame: {fid}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

            for _, ev in frame_events.iterrows():
                etype = str(ev["event_type"]).upper()
                tid = int(ev["track_id"])

                # HUD Text (Top Left)
                cv2.putText(img, f"> {etype} (Player {tid})", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
                
        # --- THE MISSING LINE ---
        out.write(img)

    out.release()
    print(f"\n  ✅ Showcase saved to: {OUT_VID}")
    print("  Play this file to see your Phase 1 intelligence system in action.")

if __name__ == "__main__":
    run_showcase()