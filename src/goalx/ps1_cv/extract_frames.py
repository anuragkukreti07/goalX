import cv2
import os
from tqdm import tqdm

VIDEO_PATH = "data/raw_videos/match_01.mp4"
OUTPUT_DIR = "data/frames/match_01"
FPS = 25

def extract_frames(video_path, output_dir, fps):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_id = 0
    saved_id = 0

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                frame_path = os.path.join(
                    output_dir, f"{saved_id:06d}.jpg"
                )
                cv2.imwrite(frame_path, frame)
                saved_id += 1

            frame_id += 1
            pbar.update(1)

    cap.release()

if __name__ == "__main__":
    extract_frames(VIDEO_PATH, OUTPUT_DIR, FPS)
