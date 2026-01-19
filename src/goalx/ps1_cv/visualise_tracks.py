import cv2
import pandas as pd
import os
import random
from tqdm import tqdm

def get_color(tid):
    """Generates a consistent color for a given Track ID."""
    random.seed(tid)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def run_viz():
    # 1. Load Data
    CSV_PATH = "data/tracking_SNMOT-116.csv"
    IMG_DIR = "data/raw_videos/tracking/test/SNMOT-116/img1/"
    OUT_PATH = "tracking_demo.mp4"
    
    if not os.path.exists(CSV_PATH):
        print(f" Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    frames = sorted(df['frame'].unique())
    
    # 2. Setup Video Writer
    first_img = cv2.imread(os.path.join(IMG_DIR, f"{frames[0]:06d}.jpg"))
    h_img, w_img, _ = first_img.shape
    out = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w_img, h_img))

    # 3. Render Frames
    for f_id in tqdm(frames, desc="Rendering Video"):
        img_path = os.path.join(IMG_DIR, f"{f_id:06d}.jpg")
        img = cv2.imread(img_path)
        if img is None: continue
        
        f_df = df[df['frame'] == f_id]
        
        for _, row in f_df.iterrows():
            tid = int(row['track_id'])
            color = get_color(tid)
            
            # DEFENSIVE CLAMPING
            x1 = max(0, min(int(row['x1']), w_img - 1))
            y1 = max(0, min(int(row['y1']), h_img - 1))
            x2 = max(0, min(int(row['x2']), w_img - 1))
            y2 = max(0, min(int(row['y2']), h_img - 1))
            
            # DRAWING
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label with ID
            label = f"ID:{tid}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + text_w, y1), color, -1) # Background
            cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        out.write(img)
    
    out.release()
    print(f"\n Done! Video saved as {OUT_PATH}")

if __name__ == "__main__":
    run_viz()