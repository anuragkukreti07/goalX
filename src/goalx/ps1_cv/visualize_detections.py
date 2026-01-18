import cv2
import pandas as pd
import os

# CONFIG
CSV_PATH = "data/detections_SNMOT-116_raw.csv"
IMG_DIR = "data/raw_videos/tracking/test/SNMOT-116/img1/"
FRAMES_TO_SAVE = [1, 350, 750] 

def verify_csv_data():
    df = pd.read_csv(CSV_PATH)
    
    for f_id in FRAMES_TO_SAVE:
        filename = f"{f_id:06d}.jpg"
        img_path = os.path.join(IMG_DIR, filename)
        
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping frame {f_id}, image not found.")
            continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        frame_data = df[df['frame'] == f_id]
        
        for _, row in frame_data.iterrows():
            # 1. DEFENSIVE CLAMPING
            # Ensures coordinates stay within [0, width-1] and [0, height-1]
            x1 = max(0, min(int(row['x1']), w - 1))
            y1 = max(0, min(int(row['y1']), h - 1))
            x2 = max(0, min(int(row['x2']), w - 1))
            y2 = max(0, min(int(row['y2']), h - 1))
            
            cls = int(row['class'])
            conf = row['conf']
            
            # Color: Green for players, Yellow for Ball
            color = (0, 255, 0) if cls == 0 else (0, 255, 255)
            # Label with confidence score (e.g., "Player 0.85")
            label = f"{'Player' if cls == 0 else 'Ball'} {conf:.2f}"
            
            # 2. DRAWING
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw a small filled background for the text to make it readable
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        out_name = f"vis_frame_{f_id:03d}.jpg"
        cv2.imwrite(out_name, img)
        print(f"✅ Saved verification with confidence: {out_name}")

if __name__ == "__main__":
    verify_csv_data()