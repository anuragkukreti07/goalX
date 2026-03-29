"""
fine_tune_yolo.py  —  PS4 Step 5
──────────────────────────────────
Fine-tunes YOLOv8 on football-specific data to replace the generic COCO
model with domain-adapted weights.

Why this file exists
─────────────────────
The existing pipeline uses yolov8s.pt — a model trained on 80 COCO classes.
It was not trained on football-specific data. Problems this causes:

  1. Referees (black/yellow shirts) are often misclassified.
  2. Crowded penalty boxes cause high false-negative rates.
  3. The ball in full-pitch views (~8px) is rarely detected.
  4. Advertising boards with people printed on them trigger false positives.

Two training modes
───────────────────
  Mode A: Supervised (you have manually annotated YOLO-format labels)
  Mode B: Self-training (pseudo-labels from your existing pipeline)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional ultralytics import
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False
    print("  ⚠  ultralytics not found: pip install ultralytics")

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

# YOLO class IDs for our pipeline
CLS_PLAYER = 0   # person / player
CLS_BALL   = 32  # sports ball (COCO) → remapped to 1 in football dataset

# Pseudo-label confidence threshold
# Only use detections above this as pseudo-labels (reduces noise)
PSEUDO_CONF_THRESHOLD = 0.65


# ─────────────────────────────────────────────────────────────────
#  Pseudo-label builder (Mode B)
# ─────────────────────────────────────────────────────────────────

def bbox_to_yolo(x1: float, y1: float, x2: float, y2: float,
                  img_w: int, img_h: int) -> tuple[float, ...]:
    """Convert pixel bbox [x1,y1,x2,y2] to YOLO format [cx_n,cy_n,w_n,h_n]."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return (
        max(0.0, min(cx, 1.0)),
        max(0.0, min(cy, 1.0)),
        max(0.001, min(bw, 1.0)),
        max(0.001, min(bh, 1.0)),
    )


def build_pseudo_label_dataset(detections_csv: Path,
                                frames_dir: Path,
                                out_dir: Path,
                                conf_threshold: float = PSEUDO_CONF_THRESHOLD,
                                train_ratio: float = 0.85) -> Path:
    """
    Convert pipeline detections to a YOLO-format dataset for fine-tuning.

    Filters:
      - Confidence >= conf_threshold
      - Box area > 100 px² (removes tiny false positives)
      - Player class: remap COCO 0 → football 0
      - Ball class:   remap COCO 32 → football 1

    Returns the path to the generated dataset.yaml file.
    """
    print(f"\n  Building pseudo-label dataset from {detections_csv}")

    df = pd.read_csv(detections_csv)

    # Normalise column names
    if "frame_id" not in df.columns and "frame" in df.columns:
        df.rename(columns={"frame": "frame_id"}, inplace=True)
    if "class_id" not in df.columns and "class" in df.columns:
        df.rename(columns={"class": "class_id"}, inplace=True)

    # Filter high-confidence detections
    df = df[df["conf"] >= conf_threshold].copy()
    # Filter minimum box area
    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
    df = df[df["area"] > 100].copy()

    print(f"  Detections after filtering: {len(df):,}  "
          f"(conf≥{conf_threshold}, area>100px²)")

    # Remap class IDs: player=0, ball=1
    cls_remap = {0: 0, 32: 1}
    df = df[df["class_id"].isin(cls_remap)].copy()
    df["yolo_cls"] = df["class_id"].map(cls_remap)

    unique_frames = sorted(df["frame_id"].unique())
    n_train = int(len(unique_frames) * train_ratio)
    train_frames = set(unique_frames[:n_train])
    val_frames   = set(unique_frames[n_train:])

    print(f"  Frames: {len(unique_frames):,} total  "
          f"({n_train} train / {len(val_frames)} val)")

    # Create YOLO directory structure
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_written = 0
    for fid in tqdm(unique_frames, desc="Writing labels"):
        split = "train" if fid in train_frames else "val"

        # Find source image
        img_path = frames_dir / f"{int(fid):06d}.jpg"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        # Copy image
        dst_img = out_dir / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # Write label file
        frame_dets = df[df["frame_id"] == fid]
        label_lines = []
        for _, row in frame_dets.iterrows():
            cx, cy, bw, bh = bbox_to_yolo(
                row["x1"], row["y1"], row["x2"], row["y2"], w_img, h_img
            )
            label_lines.append(f"{int(row['yolo_cls'])} {cx:.6f} {cy:.6f} "
                                f"{bw:.6f} {bh:.6f}")

        if label_lines:
            lbl_path = (out_dir / "labels" / split /
                        img_path.with_suffix(".txt").name)
            lbl_path.write_text("\n".join(label_lines))
            n_written += 1

    print(f"  Written: {n_written:,} labeled frames")

    # Write dataset.yaml
    yaml_content = f"""# goalX football fine-tune dataset
# Auto-generated by fine_tune_yolo.py

path: {out_dir.resolve()}
train: images/train
val:   images/val

nc: 2
names:
  0: player
  1: ball
"""
    yaml_path = out_dir / "football_dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  ✔  Dataset YAML → {yaml_path}")

    return yaml_path


# ─────────────────────────────────────────────────────────────────
#  Training wrapper
# ─────────────────────────────────────────────────────────────────

def run_fine_tuning(model_wt: str,
                    data_yaml: Path,
                    out_dir: Path,
                    epochs: int,
                    imgsz: int,
                    batch: int) -> Path:
    """
    Launch Ultralytics fine-tuning with the goalX football dataset.
    Returns path to best trained weights.
    """
    if not _YOLO_OK:
        raise RuntimeError("ultralytics not installed: pip install ultralytics")

    print(f"\n  Launching fine-tuning:")
    print(f"    Base model : {model_wt}")
    print(f"    Dataset    : {data_yaml}")
    print(f"    Epochs     : {epochs}")
    print(f"    imgsz      : {imgsz}")
    print(f"    Batch      : {batch}")

    model = YOLO(model_wt)

    results = model.train(
        data    = str(data_yaml),
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        project = str(out_dir),
        name    = "football_finetune",
        exist_ok = True,
        # Learning rate warmup for stable fine-tuning
        warmup_epochs = 3,
        lr0     = 1e-4,    # lower LR than default (we're fine-tuning, not training from scratch)
        lrf     = 0.01,
        # Data augmentation appropriate for football broadcast footage
        hsv_h   = 0.015,   # slight hue shift (different broadcast colorimetry)
        hsv_s   = 0.7,     # saturation variation (weather / camera settings)
        hsv_v   = 0.4,     # value variation
        degrees = 5.0,     # small rotation (camera tilt)
        flipud  = 0.0,     # no vertical flip (grass is always below)
        fliplr  = 0.5,     # horizontal flip OK (both directions on pitch)
        mosaic  = 0.5,     # mosaic augmentation helps small objects
        copy_paste = 0.1,  # copy-paste augmentation for player overlaps
        verbose = False,
    )

    # Best weights path
    best_wt = out_dir / "football_finetune" / "weights" / "best.pt"
    if best_wt.exists():
        final_wt = out_dir / "yolov8_football.pt"
        shutil.copy2(best_wt, final_wt)
        print(f"\n  ✔  Best weights → {final_wt}")
        return final_wt
    else:
        print(f"  ⚠  best.pt not found at {best_wt}")
        return best_wt


# ─────────────────────────────────────────────────────────────────
#  Evaluation: mAP before vs after
# ─────────────────────────────────────────────────────────────────

def evaluate_model(model_wt: str, data_yaml: Path,
                    label: str, imgsz: int = 1280) -> dict:
    """Run YOLO validation and return mAP metrics."""
    if not _YOLO_OK:
        return {}
    model   = YOLO(model_wt)
    results = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False)
    return {
        "label":    label,
        "mAP50":    round(float(results.box.map50), 4),
        "mAP50-95": round(float(results.box.map),   4),
        "precision": round(float(results.box.mp),   4),
        "recall":    round(float(results.box.mr),   4),
    }


def make_comparison_report(before: dict, after: dict, out_path: Path) -> None:
    lines = [
        "goalX YOLOv8 Fine-Tune Report",
        "=" * 44,
        "",
        f"  {'Metric':<18} {'Before':>10}  {'After':>10}  {'Delta':>8}",
        f"  {'─'*44}",
    ]
    for key in ["mAP50", "mAP50-95", "precision", "recall"]:
        bv = before.get(key, 0)
        av = after.get(key, 0)
        d  = av - bv
        sign = "+" if d >= 0 else ""
        lines.append(f"  {key:<18} {bv:>10.4f}  {av:>10.4f}  {sign}{d:.4f}")

    lines += [
        "",
        f"  Base model : {before.get('label', 'N/A')}",
        f"  Fine-tuned : {after.get('label',  'N/A')}",
    ]
    out_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\n  ✔  Report → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class YOLOFineTuner:
    def __init__(self, **kw):
        self.kw = kw
        self.out_dir = Path(kw["out_dir"])

    def run(self):
        print(f"\n  goalX PS4 — YOLOv8 Fine-Tuner")
        print(f"  {'─'*44}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        model_wt = self.kw.get("model", "yolov8s.pt")

        # ── Determine data YAML ────────────────────────────────
        # FIX: Restructured if/elif logic to properly route Mode A and Mode B
        data_yaml = None
        
        if self.kw.get("pseudo_labels"):
            print("  Mode B: pseudo-label self-training")
            detections_csv = Path(self.kw["detections"])
            frames_dir     = Path(self.kw["frames_dir"])
            dataset_dir    = self.out_dir / "pseudo_dataset"
            data_yaml = build_pseudo_label_dataset(
                detections_csv, frames_dir, dataset_dir,
                conf_threshold=PSEUDO_CONF_THRESHOLD,
            )

        elif self.kw.get("data_dir"):
            print("  Mode A: supervised fine-tuning")
            data_dir  = Path(self.kw["data_dir"])
            data_yaml = data_dir / "football_dataset.yaml"
            if not data_yaml.exists():
                # Try to find any yaml in the directory
                yamls = list(data_dir.glob("*.yaml"))
                if not yamls:
                    raise FileNotFoundError(
                        f"No .yaml found in {data_dir}. "
                        "Create football_dataset.yaml following YOLO format."
                    )
                data_yaml = yamls[0]
            print(f"  Using dataset config: {data_yaml}")
            
        else:
            print("  ❌  Specify either --data-dir or --pseudo-labels")
            return

        if not data_yaml:
            print("  ❌  Failed to resolve dataset YAML.")
            return

        # ── Evaluate base model ────────────────────────────────
        print("\n  Evaluating base model (this may take a minute)…")
        before = evaluate_model(model_wt, data_yaml, label=model_wt,
                                 imgsz=self.kw.get("imgsz", 640))

        if before:
            print(f"  Base mAP50: {before['mAP50']:.4f}  "
                  f"Recall: {before['recall']:.4f}")

        # ── Fine-tune ──────────────────────────────────────────
        fine_tuned_wt = run_fine_tuning(
            model_wt  = model_wt,
            data_yaml = data_yaml,
            out_dir   = self.out_dir,
            epochs    = self.kw.get("epochs", 50),
            imgsz     = self.kw.get("imgsz", 640),
            batch     = self.kw.get("batch", 16),
        )

        # ── Evaluate fine-tuned model ──────────────────────────
        if fine_tuned_wt.exists():
            print("\n  Evaluating fine-tuned model…")
            after = evaluate_model(str(fine_tuned_wt), data_yaml,
                                    label=str(fine_tuned_wt),
                                    imgsz=self.kw.get("imgsz", 640))
            if after:
                print(f"  Fine-tuned mAP50: {after['mAP50']:.4f}  "
                      f"Recall: {after['recall']:.4f}")

            if before and after:
                make_comparison_report(
                    before, after,
                    self.out_dir / "fine_tune_report.txt"
                )

        print(f"\n  ✅  Fine-tuning complete → {self.out_dir}")
        print(f"  To use fine-tuned model in pipeline:")
        print(f"    Update goalx_config.yaml:  model.yolo_weights: "
              f"{self.out_dir}/yolov8_football.pt\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on football data (goalX PS4)."
    )
    p.add_argument("--model",        default="yolov8s.pt")
    p.add_argument("--out-dir",      default="models/fine_tuned")
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--imgsz",        type=int, default=640)
    p.add_argument("--batch",        type=int, default=16)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data-dir",     help="Mode A: supervised dataset directory")
    g.add_argument("--pseudo-labels",action="store_true",
                   help="Mode B: self-training from pipeline detections")

    p.add_argument("--detections",   default="outputs/detections_raw.csv",
                   help="Mode B: detections CSV")
    p.add_argument("--frames-dir",   default="data/raw_videos/SNMOT-116/img1",
                   help="Mode B: source frame directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    YOLOFineTuner(
        model        = args.model,
        out_dir      = args.out_dir,
        epochs       = args.epochs,
        imgsz        = args.imgsz,
        batch        = args.batch,
        data_dir     = args.data_dir,
        pseudo_labels = args.pseudo_labels,
        detections   = args.detections,
        frames_dir   = args.frames_dir,
    ).run()