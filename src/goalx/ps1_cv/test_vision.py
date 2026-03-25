"""
test_vision.py — Quick sanity-check for player/ball detection using SAHI slicing.

Two modes (set MODE below):
  "sahi"   – Sliced Adaptive Histogram Inference (catches small/distant players)
  "vanilla"– Plain YOLOv8 inference (faster, useful baseline comparison)
"""

import os
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_PATH   = "data/raw_videos/tracking/test/SNMOT-116/img1/000535.jpg"
MODEL_PATH = "yolov8s.pt"
DEVICE     = "cpu"          # "cuda:0" if you have a GPU
CONF       = 0.15           # Low threshold to catch the ball
MODE       = "vanilla"         # "sahi" | "vanilla"

# SAHI slice settings — 400 px tiles with 20 % overlap works well for 1080p broadcasts
SLICE_H     = 400
SLICE_W     = 400
OVERLAP     = 0.2

# Classes to keep: 0 = person, 32 = sports ball
KEEP_CLASSES = {0, 32}
# ──────────────────────────────────────────────────────────────────────────────


def run_sahi_test(img_path: str, model_path: str) -> None:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=CONF,
        device=DEVICE,
    )

    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=SLICE_H,
        slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
    )

    # Filter to only our target classes before exporting
    filtered = [
        p for p in result.object_prediction_list
        if p.category.id in KEEP_CLASSES
    ]
    # Mutate list in-place so export_visuals draws only filtered boxes
    result.object_prediction_list = filtered

    out_dir = Path(".")
    result.export_visuals(export_dir=str(out_dir), file_name="test_detection_sahi")

    n_players = sum(1 for p in filtered if p.category.id == 0)
    n_balls   = sum(1 for p in filtered if p.category.id == 32)

    print("-" * 40)
    print(f"👥  Players detected : {n_players}")
    if n_balls:
        print(f"⚽  Ball detected    : {n_balls}")
    else:
        print("⚠️  Ball not found — consider lowering CONF or using a finetuned model.")
    print(f"🖼️  Output saved     : test_detection_sahi.png")


def run_vanilla_test(img_path: str, model_path: str) -> None:
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(
        img_path,
        imgsz=1280,
        conf=CONF,
        classes=list(KEEP_CLASSES),
        verbose=False,
    )

    out_img = results[0].plot()
    out_path = "test_detection_vanilla.jpg"
    cv2.imwrite(out_path, out_img)

    boxes = results[0].boxes
    n_players = int((boxes.cls == 0).sum())
    n_balls   = int((boxes.cls == 32).sum())

    print("-" * 40)
    print(f"👥  Players detected : {n_players}")
    print(f"⚽  Ball detected    : {n_balls}")
    print(f"🖼️  Output saved     : {out_path}")


def main() -> None:
    img_path = Path(IMG_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    print(f"🔎  Mode: {MODE.upper()}  |  Model: {MODEL_PATH}  |  Image: {img_path.name}\n")

    if MODE == "sahi":
        run_sahi_test(str(img_path), MODEL_PATH)
    elif MODE == "vanilla":
        run_vanilla_test(str(img_path), MODEL_PATH)
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'sahi' or 'vanilla'.")


if __name__ == "__main__":
    main()