"""
detect_ball.py
──────────────
Joint player + ball detection with dual-model support.

WHY DUAL-MODEL IS NEEDED
─────────────────────────
A fine-tuned football model (yolov8_football.pt) trained on pseudo-labels
from SNMOT-116 produces zero ball labels because the ball confidence in that
corner-view footage never exceeded the 0.65 pseudo-label threshold.
The fine-tuned model therefore learned players only.

If you pass the same model for both players and ball, ball recall = 0.

The fix: split detection responsibilities.
  --player-model  fine-tuned model  (class 0 = player)
  --ball-model    base yolov8s.pt   (class 32 = sports ball, SAHI tiling)

If only one model is given (--model), both players and ball are detected
from that single model — original behaviour preserved.

Output CSV columns (unchanged)
────────────────────────────────
  frame_id, object_type, track_id, x1, y1, x2, y2, conf
  object_type ∈ { "player", "ball" }
  track_id = -1 for all rows at this stage

Usage — single model (original):
  python3 src/goalx/ps1_cv/detec    t_ball.py \
      --seq     data/raw_videos/tracking/test3/SNMOT-193/img1/ \
      --out-csv outputs_193/detections_with_ball.csv \
      --model   yolov8s.pt

Usage — dual model (when fine-tuned model has 0 ball labels):
  python3 src/goalx/ps1_cv/detect_ball.py \
      --seq          data/raw_videos/tracking/test3/SNMOT-193/img1/ \
      --out-csv      outputs_193/detections_with_ball.csv \
      --player-model models/fine_tuned/yolov8_football.pt \
      --ball-model   yolov8s.pt
"""

import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _SAHI_AVAILABLE = True
except ImportError:
    _SAHI_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

SEQ_PATH      = "data/raw_videos/tracking/test3/SNMOT-193/img1/"
OUTPUT_CSV    = "outputs_193/detections_with_ball.csv"
MODEL_PATH    = "yolov8s.pt"

# Dynamic device selection for GPU speedup
DEVICE        = 'cuda:0' if torch.cuda.is_available() else 'cpu'

PLAYER_BATCH  = 8
PLAYER_IMGSZ  = 1280
PLAYER_CONF   = 0.25

BALL_CONF     = 0.05      # very low — ball is tiny, get FPs rather than miss it
BALL_SLICE_H  = 400
BALL_SLICE_W  = 400
BALL_OVERLAP  = 0.2
#BALL_MIN_AREA = 9        # px² — discard sub 3×3 noise
BALL_MIN_AREA = 25   # Eliminates sub-5x5 noise (player feet)
BALL_MAX_AREA = 2500 # Rejects large blobs (torsos, flags)

# ─────────────────────────────────────────────────────────────────
#  Player detection
# ─────────────────────────────────────────────────────────────────

def detect_players_batched(frames, model, player_class=0, batch_size=PLAYER_BATCH):
    rows = []
    # FP16 precision speeds up inference significantly if on GPU
    use_half = (DEVICE != 'cpu')
    
    for i in tqdm(range(0, len(frames), batch_size),
                  desc="Player detection", unit="batch"):
        batch   = [str(p) for p in frames[i:i + batch_size]]
        results = model.predict(batch, imgsz=PLAYER_IMGSZ,
                                conf=PLAYER_CONF,
                                device=DEVICE,
                                half=use_half,
                                classes=[player_class], verbose=False)
        for path, result in zip(batch, results):
            frame_id = int(Path(path).stem)
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append(dict(frame_id=frame_id, object_type="player",
                                 track_id=-1,
                                 x1=x1, y1=y1, x2=x2, y2=y2,
                                 conf=float(box.conf[0])))
    return rows


# ─────────────────────────────────────────────────────────────────
#  Ball detection  (SAHI preferred, vanilla fallback)
# ─────────────────────────────────────────────────────────────────

def detect_ball_sahi(frames, ball_model_path):
    """SAHI sliced ball detection. Always targets class 32 (COCO sports ball)."""
    detection_model = AutoDetectionModel.from_pretrained(
        model_type           = "ultralytics",  # Updated from 'yolov8' to match new SAHI API
        model_path           = ball_model_path,
        confidence_threshold = BALL_CONF,
        device               = DEVICE,
    )
    rows = []
    for fp in tqdm(frames, desc="Ball detection (SAHI)", unit="frame"):
        frame_id = int(fp.stem)
        result   = get_sliced_prediction(
            str(fp), detection_model,
            slice_height=BALL_SLICE_H, slice_width=BALL_SLICE_W,
            overlap_height_ratio=BALL_OVERLAP, overlap_width_ratio=BALL_OVERLAP,
            perform_standard_pred=True,
            postprocess_type="GREEDYNMM",
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        for pred in result.object_prediction_list:
            if pred.category.id != 32:
                continue
            b = pred.bbox
            rows.append(dict(frame_id=frame_id, object_type="ball",
                             track_id=-1,
                             x1=b.minx, y1=b.miny, x2=b.maxx, y2=b.maxy,
                             conf=pred.score.value))
    return rows


def detect_ball_vanilla(frames, model, ball_class=32):
    """Fallback when SAHI not installed."""
    rows = []
    use_half = (DEVICE != 'cpu')
    for fp in tqdm(frames, desc="Ball detection (vanilla)", unit="frame"):
        frame_id = int(fp.stem)
        results  = model.predict(str(fp), imgsz=PLAYER_IMGSZ,
                                 conf=BALL_CONF, device=DEVICE, half=use_half, 
                                 classes=[ball_class], verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append(dict(frame_id=frame_id, object_type="ball",
                             track_id=-1,
                             x1=x1, y1=y1, x2=x2, y2=y2,
                             conf=float(box.conf[0])))
    return rows


# ─────────────────────────────────────────────────────────────────
#  Deduplication
# ─────────────────────────────────────────────────────────────────

def _keep_best_ball(ball_rows):
    if not ball_rows:
        return []
    
    df = pd.DataFrame(ball_rows)
    # Calculate bounding box area
    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
    
    # Filter out impossible ball sizes
    df = df[(df["area"] >= BALL_MIN_AREA) & (df["area"] <= BALL_MAX_AREA)]
    df = df.drop(columns="area")
    
    if df.empty:
        return []
        
    # Group by frame and keep the one with the highest confidence
    best = (df.sort_values("conf", ascending=False)
              .groupby("frame_id", sort=False)
              .first()
              .reset_index())
              
    return best.to_dict("records")

# ─────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────

def run(seq_path="", output_csv="", model_path=MODEL_PATH,
        player_model="", ball_model=""):
    """
    player_model : fine-tuned model path for players (class 0).
                   Overrides model_path for player detection.
    ball_model   : COCO base model path for ball (class 32).
                   Overrides model_path for ball detection.
                   REQUIRED when fine-tuned model has zero ball training labels.

    If neither player_model nor ball_model is given, model_path handles everything.
    """
    seq_path   = Path(seq_path or SEQ_PATH)
    output_csv = Path(output_csv or OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    p_model_path = player_model or model_path
    b_model_path = ball_model   or model_path
    dual_mode    = bool(player_model or ball_model)

    frames = sorted(seq_path.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames in {seq_path}")

    print(f"\n  goalX — Joint Player + Ball Detector")
    print(f"  {'─' * 40}")
    print(f"  Frames       : {len(frames)}")
    if dual_mode:
        print(f"  Player model : {p_model_path}")
        print(f"  Ball model   : {b_model_path}  ← separate COCO model (class 32)")
        print(f"  Mode         : DUAL-MODEL")
    else:
        print(f"  Model        : {p_model_path}")
    print(f"  SAHI         : {'enabled' if _SAHI_AVAILABLE else 'NOT installed — vanilla fallback'}")
    print(f"  Device       : {DEVICE}")
    print(f"  Ball conf    : {BALL_CONF}")

    # ── Players ───────────────────────────────────────────────────
    print()
    p_model = YOLO(p_model_path)
    p_model.to(DEVICE)
    p_model.fuse()
    player_rows = detect_players_batched(frames, p_model)

    # ── Ball ──────────────────────────────────────────────────────
    # ALWAYS use a COCO-trained model (yolov8s.pt) for ball — it has class 32.
    # A fine-tuned model with ball_class=1 but zero ball training data = 0 detections.
    if ball_model:
        # Explicit separate ball model specified
        if _SAHI_AVAILABLE:
            ball_rows = detect_ball_sahi(frames, b_model_path)
        else:
            b_model = YOLO(b_model_path)
            b_model.to(DEVICE)
            b_model.fuse()
            ball_rows = detect_ball_vanilla(frames, b_model)
    elif player_model:
        # Fine-tuned player model given but no ball model specified.
        # Fall back to the base model_path for ball.
        print(f"\n  ⚠  --player-model set but --ball-model not set.")
        print(f"     Using --model ({model_path}) for ball detection (class 32).")
        print(f"     If that model was also fine-tuned without ball labels,")
        print(f"     pass --ball-model yolov8s.pt explicitly.\n")
        if _SAHI_AVAILABLE:
            ball_rows = detect_ball_sahi(frames, model_path)
        else:
            b_model = YOLO(model_path)
            b_model.to(DEVICE)
            b_model.fuse()
            ball_rows = detect_ball_vanilla(frames, b_model)
    else:
        # Single model mode
        if _SAHI_AVAILABLE:
            ball_rows = detect_ball_sahi(frames, p_model_path)
        else:
            ball_rows = detect_ball_vanilla(frames, p_model)

    ball_rows = _keep_best_ball(ball_rows)

    # ── Merge ─────────────────────────────────────────────────────
    all_rows = player_rows + ball_rows
    df = pd.DataFrame(all_rows, columns=[
        "frame_id", "object_type", "track_id",
        "x1", "y1", "x2", "y2", "conf"
    ])
    df.sort_values(["frame_id", "object_type"], inplace=True, ignore_index=True)
    df.to_csv(output_csv, index=False)

    n_players          = (df["object_type"] == "player").sum()
    n_frames_with_ball = df[df["object_type"] == "ball"]["frame_id"].nunique()
    ball_recall_pct    = 100.0 * n_frames_with_ball / max(len(frames), 1)

    print(f"\n  Detection summary:")
    print(f"     Players     : {n_players:,}")
    print(f"     Ball frames : {n_frames_with_ball}/{len(frames)}"
          f"  ({ball_recall_pct:.1f}% recall)")

    if n_frames_with_ball == 0:
        print(f"\n  ✖  BALL RECALL = 0%")
        print(f"     Your ball model is almost certainly a fine-tuned model")
        print(f"     that was trained without ball labels.")
        print(f"     Fix: add --ball-model yolov8s.pt to the command.")
    elif ball_recall_pct < 20:
        print(f"  ⚠  Low recall. Try --ball-conf 0.03")

    print(f"\n  Saved → {output_csv}\n")
    return df


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Joint player + ball detection with dual-model support."
    )
    p.add_argument("--seq",          default=SEQ_PATH)
    p.add_argument("--out-csv",      default=OUTPUT_CSV)
    p.add_argument("--model",        default=MODEL_PATH,
                   help="Single model (players + ball). Default: yolov8s.pt")
    p.add_argument("--player-model", default="",
                   help="Fine-tuned model for players (class 0).")
    p.add_argument("--ball-model",   default="",
                   help="COCO base model for ball (class 32). "
                        "Use yolov8s.pt when fine-tuned model has no ball labels.")
    p.add_argument("--ball-conf",    type=float, default=BALL_CONF,
                   help=f"Ball confidence threshold (default: {BALL_CONF})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    BALL_CONF = args.ball_conf
    run(seq_path     = args.seq,
        output_csv   = args.out_csv,
        model_path   = args.model,
        player_model = args.player_model,
        ball_model   = args.ball_model)
