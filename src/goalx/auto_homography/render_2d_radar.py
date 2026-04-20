# render_2d_radar.py — replace the homography loading section with this:

import argparse, cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

def _h_from_row(row):
    return np.array([
        [row["h00"], row["h01"], row["h02"]],
        [row["h10"], row["h11"], row["h12"]],
        [row["h20"], row["h21"], row["h22"]],
    ], dtype=np.float32)

def render_radar(tracking_csv, homography_csv, pitch_map, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_tracks = pd.read_csv(tracking_csv)
    frame_col = "frame_id" if "frame_id" in df_tracks.columns else "frame"

    df_H = pd.read_csv(homography_csv)
    df_H.set_index("frame_id", inplace=True)

    pitch_img = cv2.imread(str(pitch_map))
    frames = sorted(df_tracks[frame_col].unique())
    print(f"  Rendering {len(frames)} frames...")

    for f_id in tqdm(frames):
        canvas = pitch_img.copy()
        if f_id in df_H.index:
            H = _h_from_row(df_H.loc[f_id])
            f_df = df_tracks[df_tracks[frame_col] == f_id]
            feet, is_ball = [], []
            for _, row in f_df.iterrows():
                feet.append([(row["x1"]+row["x2"])/2.0, float(row["y2"])])
                is_ball.append(int(row.get("track_id", 1)) == -1)
            if feet:
                pts = cv2.perspectiveTransform(
                    np.array([feet], dtype=np.float32), H)[0]
                for i, pt in enumerate(pts):
                    px, py = int(pt[0]), int(pt[1])
                    if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                        color = (0, 165, 255) if is_ball[i] else (0, 0, 255)
                        r = 8 if is_ball[i] else 6
                        cv2.circle(canvas, (px, py), r, color, -1)
        cv2.imwrite(str(out_dir / f"{int(f_id):06d}.jpg"), canvas)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracking", required=True)
    p.add_argument("--homography-csv", required=True)   # ← CSV not .npz
    p.add_argument("--pitch", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    render_radar(args.tracking, args.homography_csv, args.pitch, args.out_dir)