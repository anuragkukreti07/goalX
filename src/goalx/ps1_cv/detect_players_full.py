"""
detect_players_full.py
──────────────────────
YOLOv8 player + ball detection over an image sequence.
 
Key improvements over v1
  • Batch inference   — sends BATCH_SIZE frames per model call instead of 1;
                        on a typical laptop this is ~4–6× faster.
  • CLI args          — no hardcoded paths; fully reusable for any sequence.
  • Canonical column  — output uses 'frame_id' (not 'frame') so it chains
                        cleanly into track_players.py and project_tracks.py.
  • Completion stats  — prints per-class counts, fps estimate, file size on exit.
  • Pathlib throughout — no raw os.path string juggling.
 
Usage
─────
  python3 src/goalx/ps1_cv/detect_players_full.py \\
      --seq    data/raw_videos/tracking/test/SNMOT-116/img1 \\
      --out    data/detections_SNMOT-116_raw.csv \\
      --model  yolov8s.pt \\
      --imgsz  1280 \\
      --conf   0.25 \\
      --batch  8
"""
 
import argparse
import time
from pathlib import Path
 
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
 
# COCO class indices we care about
_CLS_PERSON = 0
_CLS_BALL   = 32
_CLASSES     = [_CLS_PERSON, _CLS_BALL]
_CLASS_NAMES = {_CLS_PERSON: "player", _CLS_BALL: "ball"}
 
 
# ─────────────────────────────────────────────────────────────────
#  Core detector
# ─────────────────────────────────────────────────────────────────
 
class PlayerDetector:
    """
    Wraps a YOLOv8 model and runs batched inference over an image sequence.
 
    Parameters
    ──────────
    seq_dir  : directory containing zero-padded .jpg frames
    out_csv  : path to write the detections CSV
    model_wt : YOLO weight file or name (e.g. 'yolov8s.pt')
    imgsz    : inference resolution (1280 recommended for tactical views)
    conf     : detection confidence threshold
    batch    : number of frames per model.predict call
    """
 
    def __init__(self, seq_dir: Path, out_csv: Path,
                 model_wt: str = "yolov8s.pt",
                 imgsz: int = 1280, conf: float = 0.25,
                 batch: int = 8):
        self.seq_dir  = seq_dir
        self.out_csv  = out_csv
        self.model_wt = model_wt
        self.imgsz    = imgsz
        self.conf     = conf
        self.batch    = batch
 
    # ──────────────────────────────────────────
 
    def _collect_frames(self) -> list[Path]:
        frames = sorted(self.seq_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"No .jpg files found in {self.seq_dir}")
        return frames
 
    def _frame_id(self, path: Path) -> int:
        """Parse zero-padded filename → integer frame index."""
        return int(path.stem)
 
    def _parse_results(self, results, frame_paths: list[Path]) -> list[dict]:
        rows = []
        for result, fpath in zip(results, frame_paths):
            fid = self._frame_id(fpath)
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append({
                    "frame_id": fid,
                    "class_id": cls,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": conf,
                })
        return rows
 
    # ──────────────────────────────────────────
 
    def run(self) -> None:
        print(f"\n  goalX — Player Detector")
        print(f"  {'─'*40}")
        print(f"  Sequence : {self.seq_dir}")
        print(f"  Model    : {self.model_wt}  |  imgsz={self.imgsz}  conf={self.conf}")
        print(f"  Batch    : {self.batch} frames per call\n")
 
        frames = self._collect_frames()
        print(f"  Found {len(frames)} frames.")
 
        # Load and fuse model (fuse merges Conv+BN layers → faster inference)
        model = YOLO(self.model_wt)
        model.fuse()
 
        all_rows: list[dict] = []
        t0 = time.perf_counter()
 
        # ── Batched inference loop ──────────────────────────────
        batches = [frames[i:i + self.batch]
                   for i in range(0, len(frames), self.batch)]
 
        for batch_paths in tqdm(batches, desc="Detecting"):
            # model.predict accepts a list of paths — runs as a single batch
            results = model.predict(
                [str(p) for p in batch_paths],
                imgsz    = self.imgsz,
                conf     = self.conf,
                classes  = _CLASSES,
                verbose  = False,
            )
            all_rows.extend(self._parse_results(results, batch_paths))
 
        elapsed = time.perf_counter() - t0
 
        # ── Save ────────────────────────────────────────────────
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_rows, columns=[
            "frame_id", "class_id", "x1", "y1", "x2", "y2", "conf"
        ])
        df.to_csv(self.out_csv, index=False)
 
        # ── Summary ─────────────────────────────────────────────
        n_players = int((df["class_id"] == _CLS_PERSON).sum())
        n_balls   = int((df["class_id"] == _CLS_BALL).sum())
        fps       = len(frames) / elapsed
 
        print(f"\n  ✅  Detection complete")
        print(f"  Detections : {len(df):,}  "
              f"(players={n_players:,}  balls={n_balls:,})")
        print(f"  Speed      : {fps:.1f} fps  ({elapsed:.1f}s total)")
        print(f"  Output     : {self.out_csv}  "
              f"({self.out_csv.stat().st_size // 1024} KB)\n")
 
 
# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────
 
def _parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8 player/ball detector for goalX."
    )
    p.add_argument("--seq",   required=True,
                   help="Directory of zero-padded .jpg frames")
    p.add_argument("--out",   default="data/detections_raw.csv",
                   help="Output CSV path")
    p.add_argument("--model", default="yolov8s.pt",
                   help="YOLO weight file or name  (default: yolov8s.pt)")
    p.add_argument("--imgsz", type=int,   default=1280,
                   help="Inference resolution  (default: 1280)")
    p.add_argument("--conf",  type=float, default=0.25,
                   help="Confidence threshold  (default: 0.25)")
    p.add_argument("--batch", type=int,   default=8,
                   help="Frames per predict call  (default: 8)")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = _parse_args()
    PlayerDetector(
        seq_dir  = Path(args.seq),
        out_csv  = Path(args.out),
        model_wt = args.model,
        imgsz    = args.imgsz,
        conf     = args.conf,
        batch    = args.batch,
    ).run()
 