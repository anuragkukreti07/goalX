'''import cv2
from ultralytics import YOLO
import os

# Updated path to match the extracted SoccerNet images
IMG_PATH = "data/raw_videos/tracking/test/SNMOT-116/img1/000535.jpg"

def run_test():
    if not os.path.exists(IMG_PATH):
        print(f"Error: Could not find image at {IMG_PATH}")
        return

    # Load the Nano model (fastest for your Inspiron)
    # It will download automatically
    model = YOLO("yolov8n.pt")

    # Run detection
    # imgsz=1280 is required for small player detection in tactical views
    results = model(IMG_PATH, imgsz=1280, conf=0.3)

    # Plot results on the image and save it
    res_plotted = results[0].plot()
    cv2.imwrite("test_detection_output.jpg", res_plotted)
    print("Success! Open 'test_detection_output.jpg' to see the AI in action.")

if __name__ == "__main__":
    run_test()'''


#Using sahi slicing


import cv2
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Paths
IMG_PATH = "data/raw_videos/tracking/test/SNMOT-116/img1/000535.jpg"
MODEL_PATH = "yolov8s.pt" # S-model is better than N

def run_sahi_test():
    # 1. Load the model via SAHI wrapper
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=MODEL_PATH,
        confidence_threshold=0.15, # Keep it low to catch the ball
        device="cpu" # Use "cuda:0" if you have a GPU
    )

    # 2. Run SLICED prediction
    # slice_height/width: size of each tile
    # overlap_ratio: ensures objects on the "cut line" aren't missed
    result = get_sliced_prediction(
        IMG_PATH,
        detection_model,
        slice_height=400,
        slice_width=400,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 3. Filter and Visualize
    # result.object_prediction_list contains all boxes
    # We only want classes 0 (person) and 32 (ball)
    result.export_visuals(export_dir=".", file_name="test_detection_sahi")
    
    # Check for ball in predictions
    ball_detected = any(pred.category.id == 32 for pred in result.object_prediction_list)
    
    print("-" * 30)
    if ball_detected:
        print("⚽ SUCCESS: Ball detected using SAHI slicing!")
    else:
        print("⚠️ Ball still missing. This is a great research point for your thesis.")

if __name__ == "__main__":
    run_sahi_test()