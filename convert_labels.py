import glob
import os
import pandas as pd
import cv2

def convert_yolo_to_csv():
    print("Converting YOLO txt labels to pipeline CSV...")
    
    # We need the original image dimensions to convert YOLO's relative coords back to pixels
    img_path = 'data/raw_videos/tracking/test3/SNMOT-193/img1/000001.jpg'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {img_path} to get dimensions.")
    h_img, w_img = img.shape[:2]

    txt_files = glob.glob('runs/detect/predict2/labels/*.txt')
    if not txt_files:
        raise FileNotFoundError("No YOLO .txt files found in runs/detect/predict2/labels/")

    records = []
    for txt in txt_files:
        frame_id = int(os.path.basename(txt).split('.')[0])
        with open(txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Accept 5 or 6 columns
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id == 1:       # <--- Catch the custom Colab ball
                        cls_id = 32       # <--- Disguise it as a generic ball for the pipeline
                        
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    # Use confidence if available, otherwise default to 0.9
                    conf = float(parts[5]) if len(parts) >= 6 else 0.90
                    
                    # Convert to pixel bounding box
                    x1 = (cx - bw / 2) * w_img
                    y1 = (cy - bh / 2) * h_img
                    x2 = (cx + bw / 2) * w_img
                    y2 = (cy + bh / 2) * h_img
                    
                    records.append({
                        'frame_id': frame_id, 
                        'class_id': cls_id, 
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 
                        'conf': conf
                    })
                    
    if not records:
        raise ValueError("The text files were read, but no valid bounding boxes were found.")

    df = pd.DataFrame(records)
    df = df.sort_values(['frame_id', 'class_id']).reset_index(drop=True)
    
    os.makedirs('outputs_193', exist_ok=True)
    df.to_csv('outputs_193/detections_raw.csv', index=False)
    print(f"✅ Successfully saved {len(df)} detections to outputs_193/detections_raw.csv!")

if __name__ == "__main__":
    convert_yolo_to_csv()