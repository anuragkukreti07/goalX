import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# CONFIG
SEQ_PATH = "data/raw_videos/tracking/test/SNMOT-116/img1/"
OUTPUT_CSV = "data/detections_SNMOT-116_raw.csv"

def run_player_detection():
    # Load YOLOv8s - Small
    model = YOLO("yolov8s.pt") 
    model.fuse() # Speed optimization for inference
    
    # Safety Fix: Filter for only .jpg files
    frames = sorted([
        f for f in os.listdir(SEQ_PATH)
        if f.endswith(".jpg")
    ])
    
    detections = []

    for frame_name in tqdm(frames, desc="Detecting Players"):
        frame_path = os.path.join(SEQ_PATH, frame_name)
        
        # Frame Indexing Fix: Get frame number from filename (e.g., '000001' -> 1)
        frame_id = int(os.path.splitext(frame_name)[0])

        # imgsz=1280 is the 'High-Res' setting for tactical views
        results = model.predict(frame_path, imgsz=1280, conf=0.25, classes=[0, 32], verbose=False)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # MOT format: [frame, class, x1, y1, x2, y2, confidence]
            detections.append([frame_id, cls, x1, y1, x2, y2, conf])

    # Save to CSV
    df = pd.DataFrame(detections, columns=['frame', 'class', 'x1', 'y1', 'x2', 'y2', 'conf'])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Created: {OUTPUT_CSV} with {len(df)} entries.")

if __name__ == "__main__":
    run_player_detection()