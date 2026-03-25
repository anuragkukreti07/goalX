"""
detect_ball.py
──────────────
Extends the batched player detection pipeline to jointly detect both players
(COCO class 0) and the sports ball (COCO class 32) in every frame.

Ball detections use SAHI sliced inference to overcome YOLO's difficulty with
the small-pixel-area ball.  Player detections run standard batched inference
at imgsz=1280 (same as detect_players_full.py) and are merged into one CSV.

Output CSV columns
──────────────────
  frame, object_type, track_id, x1, y1, x2, y2, conf
  object_type ∈ { "player", "ball" }
  track_id    = -1 for all ball rows (ball has no ByteTrack ID at this stage)

Usage
─────
  python3 src/goalx/ps1_cv/detect_ball.py \\
      --seq      data/raw_videos/tracking/test/SNMOT-116/img1/ \\
      --out-csv  outputs/detections_with_ball.csv \\
      --model    yolov8s.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# ── SAHI is optional: ball detection falls back to vanilla YOLO if unavailable
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _SAHI_AVAILABLE = True
except ImportError:
    _SAHI_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
#  CONFIG  (override via CLI args)
# ─────────────────────────────────────────────────────────────────

SEQ_PATH   = "data/raw_videos/tracking/test/SNMOT-116/img1/"
OUTPUT_CSV = "outputs/detections_with_ball.csv"
MODEL_PATH = "yolov8s.pt"
DEVICE     = "cpu"          # "cuda:0" if GPU available

# Player detection
PLAYER_BATCH  = 8           # frames per YOLO predict() call
PLAYER_IMGSZ  = 1280
PLAYER_CONF   = 0.25

# Ball detection (SAHI)
BALL_CONF       = 0.10      # lower threshold — ball is small and often uncertain
BALL_SLICE_H    = 400
BALL_SLICE_W    = 400
BALL_OVERLAP    = 0.2
BALL_BATCH      = 1         # SAHI processes one frame at a time


# ─────────────────────────────────────────────────────────────────
#  Player detection  (batched YOLO, same as detect_players_full.py)
# ─────────────────────────────────────────────────────────────────

def detect_players_batched(
    frames: list[Path],
    model: YOLO,
    batch_size: int = PLAYER_BATCH,
) -> list[dict]:
    rows: list[dict] = []

    for i in tqdm(range(0, len(frames), batch_size),
                  desc="🔍  Player detection", unit="batch"):
        batch = [str(p) for p in frames[i : i + batch_size]]
        results = model.predict(
            batch,
            imgsz=PLAYER_IMGSZ,
            conf=PLAYER_CONF,
            classes=[0],        # person only — ball handled separately
            verbose=False,
        )
        for path, result in zip(batch, results):
            frame_id = int(Path(path).stem)
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append(dict(
                    frame       = frame_id,
                    object_type = "player",
                    track_id    = -1,       # ByteTrack runs in track_players.py
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    conf        = float(box.conf[0]),
                ))
    return rows


# ─────────────────────────────────────────────────────────────────
#  Ball detection
#
#  Strategy A (preferred): SAHI sliced inference
#    Slices each frame into 400×400 tiles with 20 % overlap, runs YOLO on
#    each tile, then NMS-merges the results.  This dramatically improves
#    recall on small ball detections that YOLO misses at full resolution.
#
#  Strategy B (fallback if SAHI not installed): vanilla YOLO at imgsz=1280
#    Still useful — catches the ball in ~60 % of frames at close range.
# ─────────────────────────────────────────────────────────────────

def detect_ball_sahi(frames: list[Path], model_path: str) -> list[dict]:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type           = "ultralytics",
        model_path           = model_path,
        confidence_threshold = BALL_CONF,
        device               = DEVICE,
    )

    rows: list[dict] = []
    for frame_path in tqdm(frames, desc="⚽  Ball detection (SAHI)", unit="frame"):
        frame_id = int(frame_path.stem)

        result = get_sliced_prediction(
            str(frame_path),
            detection_model,
            slice_height         = BALL_SLICE_H,
            slice_width          = BALL_SLICE_W,
            overlap_height_ratio = BALL_OVERLAP,
            overlap_width_ratio  = BALL_OVERLAP,
            perform_standard_pred= True,    # also run full-frame pass
            postprocess_type     = "GREEDYNMM",
            postprocess_match_threshold = 0.5,
            verbose              = 0,
        )

        for pred in result.object_prediction_list:
            if pred.category.id != 32:
                continue
            b = pred.bbox
            rows.append(dict(
                frame       = frame_id,
                object_type = "ball",
                track_id    = -1,
                x1 = b.minx, y1 = b.miny,
                x2 = b.maxx, y2 = b.maxy,
                conf = pred.score.value,
            ))
    return rows


def detect_ball_vanilla(frames: list[Path], model: YOLO) -> list[dict]:
    """Fallback: single-pass YOLO ball detection (no SAHI)."""
    rows: list[dict] = []
    for frame_path in tqdm(frames, desc="⚽  Ball detection (vanilla)", unit="frame"):
        frame_id = int(frame_path.stem)
        results  = model.predict(
            str(frame_path),
            imgsz   = PLAYER_IMGSZ,
            conf    = BALL_CONF,
            classes = [32],
            verbose = False,
        )
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append(dict(
                frame       = frame_id,
                object_type = "ball",
                track_id    = -1,
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf        = float(box.conf[0]),
            ))
    return rows


# ─────────────────────────────────────────────────────────────────
#  Ball position consensus: keep only the highest-conf ball per frame
#
#  Rationale: there is exactly one ball.  Multiple detections per frame are
#  usually duplicate/overlapping tiles from SAHI.  Keeping the top-1 gives a
#  clean single (x, y) ball position per frame for downstream event logic.
# ─────────────────────────────────────────────────────────────────

def _keep_best_ball(ball_rows: list[dict]) -> list[dict]:
    if not ball_rows:
        return []
    df = pd.DataFrame(ball_rows)
    best = (
        df.sort_values("conf", ascending=False)
          .groupby("frame", sort=False)
          .first()
          .reset_index()
    )
    return best.to_dict("records")


# ─────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────

def run(
    seq_path:   str = SEQ_PATH,
    output_csv: str = OUTPUT_CSV,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:

    seq_path   = Path(seq_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(seq_path.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames in {seq_path}")

    print(f"\n  goalX — Joint Player + Ball Detector")
    print(f"  {'─' * 40}")
    print(f"  Frames  : {len(frames)}")
    print(f"  Model   : {model_path}")
    print(f"  SAHI    : {'enabled' if _SAHI_AVAILABLE else 'NOT installed — falling back to vanilla'}")

    # ── Player detection ─────────────────────────────────────────
    model = YOLO(model_path)
    model.fuse()
    player_rows = detect_players_batched(frames, model)

    # ── Ball detection ───────────────────────────────────────────
    if _SAHI_AVAILABLE:
        ball_rows = detect_ball_sahi(frames, model_path)
    else:
        ball_rows = detect_ball_vanilla(frames, model)

    ball_rows = _keep_best_ball(ball_rows)

    # ── Merge & save ─────────────────────────────────────────────
    all_rows = player_rows + ball_rows
    df = pd.DataFrame(all_rows, columns=[
        "frame", "object_type", "track_id", "x1", "y1", "x2", "y2", "conf"
    ])
    df.sort_values(["frame", "object_type"], inplace=True, ignore_index=True)
    df.to_csv(output_csv, index=False)

    n_players = (df["object_type"] == "player").sum()
    n_ball    = (df["object_type"] == "ball").sum()
    n_frames_with_ball = df[df["object_type"] == "ball"]["frame"].nunique()

    print(f"\n  📊 Detection summary:")
    print(f"     Players : {n_players} detections across {len(frames)} frames")
    print(f"     Ball    : {n_ball} detections across {n_frames_with_ball}/{len(frames)} frames "
          f"({100*n_frames_with_ball/len(frames):.1f}% recall)")
    print(f"\n  💾 Saved → {output_csv}")
    print(f"\n  ✅  Detection complete.")
    print(f"      Pipe this CSV into track_players.py, then project_tracks.py,")
    print(f"      then smooth_tracks.py, then extract_events.py.\n")

    return df


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Jointly detect players and ball for goalX pipeline."
    )
    p.add_argument("--seq",      default=SEQ_PATH,
                   help="Directory of frame .jpgs")
    p.add_argument("--out-csv",  default=OUTPUT_CSV,
                   help="Output CSV path")
    p.add_argument("--model",    default=MODEL_PATH,
                   help="YOLOv8 weights file")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(seq_path=args.seq, output_csv=args.out_csv, model_path=args.model)