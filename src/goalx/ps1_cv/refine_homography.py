import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def refine_with_flow(h_csv, frames_dir, out_csv, cond_threshold=50000, max_flow_gap=30):
    print("\n  goalX — Optical Flow Homography Refinement")
    print("  " + "─" * 40)
    
    h_df = pd.read_csv(h_csv)
    frames_dir = Path(frames_dir)
    h_cols = [c for c in h_df.columns if c.startswith('h')][:9]
    
    if len(h_cols) != 9:
        h_cols = [c for c in h_df.columns if c != 'frame_id'][:9]
        
    good_frames = {}
    bad_frames = set()
    
    for _, row in h_df.iterrows():
        fid = int(row['frame_id'])
        H = row[h_cols].values.astype(np.float32).reshape(3,3)
        if np.linalg.cond(H) < cond_threshold:
            good_frames[fid] = H
        else:
            bad_frames.add(fid)
            
    print(f"  ✔  Found {len(good_frames)} stable anchors and {len(bad_frames)} unstable frames.")
    
    refined = dict(good_frames)
    frame_ids = sorted(h_df['frame_id'].astype(int).tolist())
    
    lk_params = dict(winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
    
    for fid in frame_ids:
        if fid not in bad_frames:
            continue
            
        prev_good = max((f for f in good_frames if f < fid), default=None)
        if prev_good is None or (fid - prev_good) > max_flow_gap:
            continue 
            
        H_prev = refined.get(prev_good)
        if H_prev is None: continue
            
        img1 = cv2.imread(str(frames_dir / f"{prev_good:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frames_dir / f"{fid:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None: continue
            
        pts1 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
        if pts1 is None or len(pts1) < 10: continue
            
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, pts1, None, **lk_params)
        good1 = pts1[status.ravel() == 1]
        good2 = pts2[status.ravel() == 1]
        
        if len(good1) < 8: continue
            
        H_flow, inliers = cv2.findHomography(good1, good2, cv2.RANSAC, 3.0)
        if H_flow is None or inliers.sum() < 8: continue
            
        H_refined = H_flow @ H_prev
        if np.linalg.cond(H_refined) < cond_threshold * 2:
            refined[fid] = H_refined.astype(np.float32)

    rows = []
    for _, row in h_df.iterrows():
        fid = int(row['frame_id'])
        if fid in refined and fid in bad_frames:
            H = refined[fid]
            new_row = {'frame_id': fid}
            for j, col in enumerate(h_cols):
                new_row[col] = H.ravel()[j]
            rows.append(new_row)
        else:
            rows.append(row.to_dict())
            
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    fixed = sum(1 for f in bad_frames if f in refined)
    print(f"  ✔  Successfully bridged {fixed}/{len(bad_frames)} broken frames using Optical Flow.")
    print(f"  ✅  Saved refined matrices to {out_csv}\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--h-csv", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    refine_with_flow(args.h_csv, args.frames, args.out)
