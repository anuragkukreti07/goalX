"""
fine_tune_yolo.py  —  PS4 Step 5
──────────────────────────────────
Fine-tunes YOLOv8 on football-specific pseudo-label data to replace 
the generic COCO model with domain-adapted weights.

FIXES
─────────────────────────────────
FIX 1 — Default epochs raised to 50
  A proper fine-tune needs at least 50 epochs on a new domain.

FIX 2 — Default batch raised to 8
  batch=2 causes extremely noisy gradient estimates. batch≥8 is safe.

FIX 3 — Pseudo-label labels directory naming fixed
  Label file name is now derived from the image file stem, not a counter.

FIX 4 — Ball labels included in pseudo-dataset
  Lowered ball confidence threshold to 0.40 while keeping player 
  threshold at 0.65 to ensure ball labels make it into the dataset.

FIX 5 — Pre-flight warnings for short datasets
  Warns if fewer than 50 labeled frames are provided.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False
    print("  ⚠  ultralytics not found: pip install ultralytics")

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

CLS_PLAYER = 0
CLS_BALL   = 32

PLAYER_CONF_THRESHOLD = 0.65   # keep player value strict
BALL_CONF_THRESHOLD   = 0.40   # FIX 4: lower for ball (SAHI detects at ~0.10-0.40)


# ─────────────────────────────────────────────────────────────────
#  Pseudo-label builder
# ─────────────────────────────────────────────────────────────────

def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w;  cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w;      bh = (y2 - y1) / img_h
    return (max(0.0, min(cx, 1.0)), max(0.0, min(cy, 1.0)),
            max(0.001, min(bw, 1.0)), max(0.001, min(bh, 1.0)))


def build_pseudo_label_dataset(detections_csv: Path, frames_dir: Path,
                                 out_dir: Path, train_ratio: float = 0.85) -> Path:
    print(f"\n  Building pseudo-label dataset from {detections_csv}")

    df = pd.read_csv(detections_csv)
    if "frame_id" not in df.columns and "frame" in df.columns:
        df.rename(columns={"frame": "frame_id"}, inplace=True)
    if "class_id" not in df.columns and "object_type" in df.columns:
        df["class_id"] = df["object_type"].map({"player": 0, "ball": 32}).fillna(-1).astype(int)
    if "class_id" not in df.columns and "class" in df.columns:
        df.rename(columns={"class": "class_id"}, inplace=True)

    # FIX 4: separate conf thresholds for player vs ball
    df_player = df[(df["class_id"] == 0)  & (df["conf"] >= PLAYER_CONF_THRESHOLD)].copy()
    df_ball   = df[(df["class_id"] == 32) & (df["conf"] >= BALL_CONF_THRESHOLD)].copy()
    df = pd.concat([df_player, df_ball], ignore_index=True)

    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
    df = df[df["area"] > 100].copy()

    cls_remap = {0: 0, 32: 1}
    df = df[df["class_id"].isin(cls_remap)].copy()
    df["yolo_cls"] = df["class_id"].map(cls_remap)

    n_players = int((df["yolo_cls"] == 0).sum())
    n_balls   = int((df["yolo_cls"] == 1).sum())
    print(f"  Labels after filtering: {len(df):,}  (players={n_players:,}  balls={n_balls:,})")

    if n_balls == 0:
        print("  ⚠  No ball labels passed the threshold. Check detections.csv.")

    unique_frames = sorted(df["frame_id"].unique())
    
    # FIX 5: pre-flight warning
    if len(unique_frames) < 50:
        print(f"  ⚠  Only {len(unique_frames)} labeled frames — very likely to overfit.")
        
    n_train = int(len(unique_frames) * train_ratio)
    train_f = set(unique_frames[:n_train])
    val_f   = set(unique_frames[n_train:])

    print(f"  Frames: {len(unique_frames):,} ({n_train} train / {len(val_f)} val)")

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_written = 0
    for fid in tqdm(unique_frames, desc="Writing labels"):
        split    = "train" if fid in train_f else "val"
        img_path = frames_dir / f"{int(fid):06d}.jpg"
        
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h_img, w_img = img.shape[:2]
        dst_img = out_dir / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        frame_dets  = df[df["frame_id"] == fid]
        label_lines = []
        for _, row in frame_dets.iterrows():
            cx, cy, bw, bh = bbox_to_yolo(row["x1"], row["y1"], row["x2"], row["y2"], w_img, h_img)
            label_lines.append(f"{int(row['yolo_cls'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if label_lines:
            # FIX 3: label file name == image file stem
            lbl_path = out_dir / "labels" / split / img_path.with_suffix(".txt").name
            lbl_path.write_text("\n".join(label_lines))
            n_written += 1

    print(f"  Written: {n_written:,} labeled frames")

    yaml_content = (
        f"# goalX football fine-tune dataset\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: player\n"
        f"  1: ball\n"
    )
    yaml_path = out_dir / "football_dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  ✔  Dataset YAML → {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────
#  Training wrapper
# ─────────────────────────────────────────────────────────────────

def run_fine_tuning(model_wt: str, data_yaml: Path, out_dir: Path,
                    epochs: int, imgsz: int, batch: int) -> Path:
    if not _YOLO_OK:
        raise RuntimeError("ultralytics not installed: pip install ultralytics")

    print(f"\n  Fine-tuning:")
    print(f"    Base model : {model_wt}")
    print(f"    Dataset    : {data_yaml}")
    print(f"    Epochs     : {epochs}  (Default adjusted to 50)") 
    print(f"    Batch      : {batch}   (Default adjusted to 8)")   
    print(f"    imgsz      : {imgsz}")

    if epochs < 20:
        print(f"  ⚠  epochs={epochs} is very low. Recommend ≥50 for meaningful fine-tuning.")

    model   = YOLO(model_wt)
    results = model.train(
        data         = str(data_yaml),
        epochs       = epochs,
        imgsz        = imgsz,
        batch        = batch,
        project      = str(out_dir),
        name         = "football_finetune",
        exist_ok     = True,
        warmup_epochs= 3,
        lr0          = 1e-4,
        lrf          = 0.01,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        degrees      = 5.0,
        flipud       = 0.0,
        fliplr       = 0.5,
        mosaic       = 0.5,
        copy_paste   = 0.1,
        verbose      = False,
    )

    best_wt  = out_dir / "football_finetune" / "weights" / "best.pt"
    final_wt = out_dir / "yolov8_football.pt"
    
    if best_wt.exists():
        shutil.copy2(best_wt, final_wt)
        print(f"\n  ✔  Best weights → {final_wt}")
        return final_wt
    
    print(f"  ⚠  best.pt not found at {best_wt}")
    return best_wt


# ─────────────────────────────────────────────────────────────────
#  Evaluation: mAP before vs after
# ─────────────────────────────────────────────────────────────────

def evaluate_model(model_wt: str, data_yaml: Path, label: str, imgsz: int = 640) -> dict:
    if not _YOLO_OK:
        return {}
    model   = YOLO(model_wt)
    results = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False)
    return {
        "label":     label,
        "mAP50":     round(float(results.box.map50), 4),
        "mAP50-95":  round(float(results.box.map),   4),
        "precision": round(float(results.box.mp),    4),
        "recall":    round(float(results.box.mr),    4),
    }

def make_comparison_report(before: dict, after: dict, out_path: Path) -> None:
    lines = [
        "goalX YOLOv8 Fine-Tune Report", "=" * 44, "",
        f"  {'Metric':<18} {'Before':>10}  {'After':>10}  {'Delta':>8}",
        f"  {'─'*44}",
    ]
    for key in ["mAP50", "mAP50-95", "precision", "recall"]:
        bv = before.get(key, 0); av = after.get(key, 0); d = av - bv
        sign = "+" if d >= 0 else ""
        lines.append(f"  {key:<18} {bv:>10.4f}  {av:>10.4f}  {sign}{d:.4f}")

    lines += ["", f"  Base model : {before.get('label', 'N/A')}", f"  Fine-tuned : {after.get('label',  'N/A')}"]
    
    out_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\n  ✔  Report → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class YOLOFineTuner:
    def __init__(self, **kw):
        self.kw      = kw
        self.out_dir = Path(kw["out_dir"])

    def run(self):
        print(f"\n  goalX PS4 — YOLOv8 Fine-Tuner")
        print(f"  {'─'*44}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        model_wt = self.kw.get("model", "yolov8s.pt")

        if self.kw.get("pseudo_labels"):
            print("  Mode B: pseudo-label self-training")
            dataset_dir = self.out_dir / "pseudo_dataset"
            data_yaml   = build_pseudo_label_dataset(
                detections_csv = Path(self.kw["detections"]),
                frames_dir     = Path(self.kw["frames_dir"]),
                out_dir        = dataset_dir,
            )
        elif self.kw.get("data_dir"):
            print("  Mode A: supervised fine-tuning")
            data_dir  = Path(self.kw["data_dir"])
            yamls     = list(data_dir.glob("*.yaml"))
            data_yaml = yamls[0] if yamls else data_dir / "football_dataset.yaml"
            if not data_yaml.exists():
                raise FileNotFoundError(f"No .yaml in {data_dir}")
        else:
            print("  ❌  Specify --data-dir or --pseudo-labels"); return

        print("\n  Evaluating base model…")
        before = evaluate_model(model_wt, data_yaml, label=model_wt, imgsz=self.kw.get("imgsz", 640))
        if before:
            print(f"  Base mAP50: {before['mAP50']:.4f}  Recall: {before['recall']:.4f}")

        fine_tuned_wt = run_fine_tuning(
            model_wt  = model_wt,
            data_yaml = data_yaml,
            out_dir   = self.out_dir,
            epochs    = self.kw.get("epochs", 50),
            imgsz     = self.kw.get("imgsz", 640),
            batch     = self.kw.get("batch", 8),
        )

        if fine_tuned_wt.exists():
            print("\n  Evaluating fine-tuned model…")
            after = evaluate_model(str(fine_tuned_wt), data_yaml, label=str(fine_tuned_wt), imgsz=self.kw.get("imgsz", 640))
            if before and after:
                make_comparison_report(before, after, self.out_dir / "fine_tune_report.txt")

        print(f"\n  ✅  Fine-tuning complete → {self.out_dir}")
        print(f"  Update goalx_config.yaml: model.yolo_weights: {self.out_dir}/yolov8_football.pt\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on football data — goalX PS4.")
    p.add_argument("--model",    default="yolov8s.pt")
    p.add_argument("--out-dir",  default="models/fine_tuned")
    p.add_argument("--epochs",   type=int, default=50)   # FIX 1
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=8)    # FIX 2
    
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data-dir",      help="Mode A: supervised dataset dir")
    g.add_argument("--pseudo-labels", action="store_true", help="Mode B: self-training from pipeline detections")
    
    p.add_argument("--detections", default="outputs/detections_raw.csv")
    p.add_argument("--frames-dir", default="data/raw_videos/tracking/test/SNMOT-116/img1")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    YOLOFineTuner(
        model         = args.model,
        out_dir       = args.out_dir,
        epochs        = args.epochs,
        imgsz         = args.imgsz,
        batch         = args.batch,
        data_dir      = args.data_dir,
        pseudo_labels = args.pseudo_labels,
        detections    = args.detections,
        frames_dir    = args.frames_dir,
    ).run()