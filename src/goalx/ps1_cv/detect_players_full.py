"""
detect_players_full.py
──────────────────────
YOLOv8 player + ball detection over an image sequence.

Full-pitch upgrade
──────────────────
In a tactical full-pitch view a player is typically 15–25 px tall in a
1920×1080 frame. Standard single-pass inference on a downsampled frame
misses most of them.

The fix is SAHI — Slicing Aided Hyper Inference:
  1. Slice the frame into overlapping 640×640 patches.
  2. Run YOLO on each patch independently (players appear much larger).
  3. Merge all detections from all patches using NMS.
  4. Coordinate-remap back to full-frame pixel space.

If SAHI is not installed the script falls back to a built-in manual
slicer that works the same way but is slightly slower.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

_CLS_PERSON = 0
_CLS_BALL   = 32
_CLASSES     = [_CLS_PERSON, _CLS_BALL]

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _SAHI_AVAILABLE = True
except ImportError:
    _SAHI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
#  NMS helper  (used by built-in manual slicer)
# ─────────────────────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _nms(dets: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    if not dets:
        return []
    dets    = sorted(dets, key=lambda d: d["conf"], reverse=True)
    kept    = []
    supp    = set()
    for i, d in enumerate(dets):
        if i in supp:
            continue
        kept.append(d)
        ba = [d["x1"], d["y1"], d["x2"], d["y2"]]
        for j, e in enumerate(dets[i+1:], i+1):
            if j in supp or d["class_id"] != e["class_id"]:
                continue
            if _iou(ba, [e["x1"], e["y1"], e["x2"], e["y2"]]) > iou_thresh:
                supp.add(j)
    return kept


# ─────────────────────────────────────────────────────────────────
#  SAHI-backed detector  (preferred for full-pitch)
# ─────────────────────────────────────────────────────────────────

class SAHIDetector:
    def __init__(self, model_wt: str, conf: float,
                 slice_hw: int = 640, overlap: float = 0.2):
        print(f"  Loading SAHI detection model ({model_wt})…")
        self.model = AutoDetectionModel.from_pretrained(
            model_type           = "yolov8",
            model_path           = model_wt,
            confidence_threshold = conf,
            device               = "cpu",
        )
        self.conf     = conf
        self.slice_hw = slice_hw
        self.overlap  = overlap

    def predict(self, path: Path) -> list[dict]:
        result = get_sliced_prediction(
            image                       = str(path),
            detection_model             = self.model,
            slice_height                = self.slice_hw,
            slice_width                 = self.slice_hw,
            overlap_height_ratio        = self.overlap,
            overlap_width_ratio         = self.overlap,
            postprocess_type            = "NMS",
            postprocess_match_metric    = "IOU",
            postprocess_match_threshold = 0.5,
            verbose                     = 0,
        )
        rows = []
        for pred in result.object_prediction_list:
            cid = pred.category.id
            if cid not in _CLASSES:
                continue
            b = pred.bbox
            rows.append({"class_id": cid,
                         "x1": b.minx, "y1": b.miny,
                         "x2": b.maxx, "y2": b.maxy,
                         "conf": pred.score.value})
        return rows


# ─────────────────────────────────────────────────────────────────
#  Manual slicer  (fallback when SAHI not installed)
# ─────────────────────────────────────────────────────────────────

class ManualSlicedDetector:
    def __init__(self, model: YOLO, conf: float,
                 slice_hw: int = 640, overlap: float = 0.2):
        self.model    = model
        self.conf     = conf
        self.slice_hw = slice_hw
        self.overlap  = overlap

    def _slices(self, h: int, w: int) -> list[tuple]:
        stride = int(self.slice_hw * (1 - self.overlap))
        out    = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x2 = min(x + self.slice_hw, w)
                y2 = min(y + self.slice_hw, h)
                out.append((x, y, x2, y2))
                if x2 == w: break
                x += stride
            if y == y2 - self.slice_hw + stride or y2 == h: break
            y += stride
        return out

    def predict(self, path: Path) -> list[dict]:
        img = cv2.imread(str(path))
        if img is None:
            return []
        h, w = img.shape[:2]
        all_dets: list[dict] = []

        for sx1, sy1, sx2, sy2 in self._slices(h, w):
            crop   = img[sy1:sy2, sx1:sx2]
            result = self.model.predict(
                crop, imgsz=self.slice_hw,
                conf=self.conf, classes=_CLASSES, verbose=False,
            )
            for box in result[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_dets.append({
                    "class_id": int(box.cls[0]),
                    "x1":  x1 + sx1, "y1": y1 + sy1,
                    "x2":  x2 + sx1, "y2": y2 + sy1,
                    "conf": float(box.conf[0]),
                })

        return _nms(all_dets)


# ─────────────────────────────────────────────────────────────────
#  Standard batched detector  (broadcast / corner mode)
# ─────────────────────────────────────────────────────────────────

class BatchedDetector:
    def __init__(self, model: YOLO, conf: float,
                 imgsz: int = 1280, batch: int = 8):
        self.model = model
        self.conf  = conf
        self.imgsz = imgsz
        self.batch = batch

    def predict_batch(self, paths: list[Path]) -> list[list[dict]]:
        results = self.model.predict(
            [str(p) for p in paths],
            imgsz=self.imgsz, conf=self.conf,
            classes=_CLASSES, verbose=False,
        )
        out = []
        for r in results:
            frame_dets = []
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                frame_dets.append({
                    "class_id": int(box.cls[0]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(box.conf[0]),
                })
            out.append(frame_dets)
        return out


# ─────────────────────────────────────────────────────────────────
#  Auto mode detection
# ─────────────────────────────────────────────────────────────────

def _auto_detect_mode(seq_dir: Path) -> str:
    first = sorted(seq_dir.glob("*.jpg"))
    if not first:
        return "broadcast"
    img = cv2.imread(str(first[0]))
    if img is None:
        return "broadcast"
    _, w = img.shape[:2]
    return "fullpitch" if w >= 1600 else "broadcast"


# ─────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────

class PlayerDetector:
    def __init__(self, seq_dir: Path, out_csv: Path,
                 model_wt: str, mode: str,
                 imgsz: int, conf: float, batch: int,
                 slice_hw: int, overlap: float):
        self.seq_dir  = seq_dir
        self.out_csv  = out_csv
        self.model_wt = model_wt
        self.mode     = mode
        self.imgsz    = imgsz
        self.conf     = conf
        self.batch    = batch
        self.slice_hw = slice_hw
        self.overlap  = overlap

    def run(self) -> None:
        print(f"\n  goalX — Player Detector")
        print(f"  {'─'*42}")

        frames = sorted(self.seq_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"No .jpg files in {self.seq_dir}")

        actual_mode = self.mode
        if actual_mode == "auto":
            actual_mode = _auto_detect_mode(self.seq_dir)
            print(f"  Mode (auto-detected) : {actual_mode}")
        else:
            print(f"  Mode     : {actual_mode}")

        print(f"  Sequence : {self.seq_dir}  ({len(frames)} frames)")
        print(f"  Model    : {self.model_wt}  |  conf={self.conf}")

        all_rows: list[dict] = []
        t0 = time.perf_counter()

        # ── Full-pitch: sliced inference ───────────────────────
        if actual_mode == "fullpitch":
            print(f"  Slice    : {self.slice_hw}px  overlap={self.overlap}\n")

            if _SAHI_AVAILABLE:
                det = SAHIDetector(self.model_wt, self.conf,
                                   self.slice_hw, self.overlap)
                for fp in tqdm(frames, desc="Detecting (SAHI)"):
                    fid = int(fp.stem)
                    for d in det.predict(fp):
                        all_rows.append({"frame_id": fid, **d})
            else:
                print("  ⚠  SAHI not installed — using built-in slicer.")
                print("     pip install sahi  for better speed & accuracy.\n")
                model = YOLO(self.model_wt)
                model.fuse()
                det = ManualSlicedDetector(model, self.conf,
                                           self.slice_hw, self.overlap)
                for fp in tqdm(frames, desc="Detecting (manual slice)"):
                    fid = int(fp.stem)
                    for d in det.predict(fp):
                        all_rows.append({"frame_id": fid, **d})

        # ── Broadcast: fast batched inference ─────────────────
        else:
            print(f"  Batch    : {self.batch}  imgsz={self.imgsz}\n")
            model = YOLO(self.model_wt)
            model.fuse()
            bdet  = BatchedDetector(model, self.conf, self.imgsz, self.batch)
            batches = [frames[i:i+self.batch]
                       for i in range(0, len(frames), self.batch)]
            for bp in tqdm(batches, desc="Detecting"):
                for fp, dets in zip(bp, bdet.predict_batch(bp)):
                    fid = int(fp.stem)
                    for d in dets:
                        all_rows.append({"frame_id": fid, **d})

        # ── Save ──────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_rows,
                          columns=["frame_id","class_id","x1","y1","x2","y2","conf"])
        df.to_csv(self.out_csv, index=False)

        n_p = int((df["class_id"] == _CLS_PERSON).sum())
        n_b = int((df["class_id"] == _CLS_BALL).sum())
        print(f"\n  ✅  Done in {elapsed:.1f}s  ({len(frames)/elapsed:.1f} fps)")
        print(f"  Detections : {len(df):,}  (players={n_p:,}  balls={n_b:,})")
        print(f"  Output     : {self.out_csv}  "
              f"({self.out_csv.stat().st_size//1024} KB)\n")

        if n_p < len(frames) * 5:
            print("  ⚠  Suspiciously few players detected. Try:")
            print("     --conf 0.10  --mode fullpitch  --model yolov8m.pt\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8 player/ball detector — full-pitch aware."
    )
    p.add_argument("--seq",      required=True)
    p.add_argument("--out",      default="data/detections_raw.csv")
    p.add_argument("--model",    default="yolov8s.pt")
    p.add_argument("--mode",     default="auto",
                   choices=["auto","fullpitch","broadcast"])
    p.add_argument("--imgsz",    type=int,   default=1280)
    p.add_argument("--conf",     type=float, default=0.15)
    p.add_argument("--batch",    type=int,   default=8)
    p.add_argument("--slice-hw", type=int,   default=640)
    p.add_argument("--overlap",  type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    PlayerDetector(
        seq_dir  = Path(args.seq),
        out_csv  = Path(args.out),
        model_wt = args.model,
        mode     = args.mode,
        imgsz    = args.imgsz,
        conf     = args.conf,
        batch    = args.batch,
        slice_hw = args.slice_hw,
        overlap  = args.overlap,
    ).run()