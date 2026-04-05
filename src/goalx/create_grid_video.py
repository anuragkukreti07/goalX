"""
create_grid_video.py
────────────────────
Stitches multiple pipeline output videos into a single synchronized 2x2 grid
for thesis presentations and visual comparisons.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def create_grid(vid_paths: list[str], output_path: str, fps: int = 25):
    print(f"\n  goalX — Video Grid Builder")
    print(f"  {'─'*40}")
    
    # Open all video captures
    caps = [cv2.VideoCapture(v) for v in vid_paths]
    
    # Check if videos opened successfully
    for i, (cap, path) in enumerate(zip(caps, vid_paths)):
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video {i+1}: {path}")
        print(f"  [{i+1}] Loaded: {Path(path).name}")

    # Standardize each quadrant to 960x540 (makes a 1920x1080 final grid)
    QUAD_W, QUAD_H = 960, 540
    
    # Set up the Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (QUAD_W * 2, QUAD_H * 2))
    
    # Find the minimum frame count so we don't read past the shortest video
    frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    total_frames = min(frame_counts) if frame_counts else 0
    
    print(f"\n  Stitching {total_frames} frames into a 2x2 grid...")

    # Blank frame for empty quadrants (if you pass fewer than 4 videos)
    blank_frame = np.zeros((QUAD_H, QUAD_W, 3), dtype=np.uint8)

    with tqdm(total=total_frames, desc="Rendering Grid", unit="frame") as pbar:
        while True:
            frames = []
            all_good = True
            
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    all_good = False
                    break
                # Resize to fit the quadrant
                frames.append(cv2.resize(frame, (QUAD_W, QUAD_H)))
                
            if not all_good:
                break
                
            # Pad with blank frames if we have less than 4 videos
            while len(frames) < 4:
                frames.append(blank_frame)
                
            # Stitch: Top row, Bottom row, then vertically combine
            top_row = np.hstack((frames[0], frames[1]))
            bot_row = np.hstack((frames[2], frames[3]))
            grid_frame = np.vstack((top_row, bot_row))
            
            out.write(grid_frame)
            pbar.update(1)

    # Cleanup
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"\n  ✅ Grid video saved → {output_path}")
    print(f"     Play this at your defense!\n")

def _parse_args():
    p = argparse.ArgumentParser(description="Stitch videos into a 2x2 grid.")
    p.add_argument("--vid1", required=True, help="Top-Left video")
    p.add_argument("--vid2", required=True, help="Top-Right video")
    p.add_argument("--vid3", default="", help="Bottom-Left video")
    p.add_argument("--vid4", default="", help="Bottom-Right video")
    p.add_argument("--out", default="outputs/pipeline_comparison.mp4")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    vids = [v for v in [args.vid1, args.vid2, args.vid3, args.vid4] if v]
    create_grid(vids, args.out)