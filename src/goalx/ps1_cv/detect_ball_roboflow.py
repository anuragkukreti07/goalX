"""
detect_ball.py
──────────────
Joint player + ball detection with 4-class Roboflow model support.

MODEL OPTIONS
──────────────
Option A — Roboflow 4-class model (RECOMMENDED — matches notebook quality):
    A single model handles all 4 classes: ball, goalkeeper, player, referee.
    Download from: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
    Use: --player-model path/to/roboflow_football.pt
    (No --ball-model needed — the Roboflow model detects the ball natively)

Option B — Dual-model fallback (when you only have fine-tuned players model):
    --player-model models/fine_tuned/yolov8_football.pt
    --ball-model   yolov8s.pt
    (Fine-tuned model = class 0 players only, zero ball labels)

Option C — Single base COCO model:
    --model yolov8s.pt

OUTPUT CSV COLUMNS (unchanged for downstream compatibility)
────────────────────────────────────────────────────────────
  frame_id, object_type, role, track_id, x1, y1, x2, y2, conf

  object_type : "player" | "ball"   ← same as before, downstream unchanged
  role        : "player" | "goalkeeper" | "referee" | "ball"  ← NEW
                used by visualise_tracks.py for per-class colours
  track_id    : -1 for all rows at this stage

Usage — Roboflow 4-class model (best quality):
  python3 src/goalx/ps1_cv/detect_ball.py \\
      --seq          data/raw_videos/tracking/test3/SNMOT-193/img1/ \\
      --out-csv      outputs_193/detections_with_ball.csv \\
      --player-model models/roboflow_football.pt

Usage — dual-model fallback (fine-tuned without ball labels):
  python3 src/goalx/ps1_cv/detect_ball.py \\
      --seq          data/raw_videos/tracking/test3/SNMOT-193/img1/ \\
      --out-csv      outputs_193/detections_with_ball.csv \\
      --player-model models/fine_tuned/yolov8_football.pt \\
      --ball-model   yolov8s.pt \\
      --ball-conf    0.05
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
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"

PLAYER_BATCH  = 8
PLAYER_IMGSZ  = 1280
PLAYER_CONF   = 0.25

BALL_CONF     = 0.05
BALL_SLICE_H  = 400
BALL_SLICE_W  = 400
BALL_OVERLAP  = 0.2
BALL_MIN_AREA = 9

# Roboflow 4-class model class names → role mapping
# {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
_ROBOFLOW_ROLE_MAP = {
    "ball":       "ball",
    "goalkeeper": "goalkeeper",
    "player":     "player",
    "referee":    "referee",
}

# When using a single-class fine-tuned model (class 0 = player only)
_SINGLE_CLASS_ROLE_MAP = {
    0:  "player",
    32: "ball",
}


# ─────────────────────────────────────────────────────────────────
#  Detect from 4-class Roboflow-style model
# ─────────────────────────────────────────────────────────────────

def detect_4class_batched(frames, model, batch_size=PLAYER_BATCH):
    """
    Run a 4-class football model (ball/goalkeeper/player/referee) in batches.
    Detects all 4 classes in a single forward pass per frame — no SAHI needed
    because this model was trained specifically on football and reliably detects
    the ball at broadcast resolution.
    """
    rows = []
    use_half = (DEVICE != "cpu")

    # Determine model class names
    names = model.names  # dict: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
    name_to_id = {v: k for k, v in names.items()}

    # Build the classes list dynamically from what the model knows
    detect_classes = list(names.keys())  # detect all classes

    for i in tqdm(range(0, len(frames), batch_size),
                  desc="4-class detection", unit="batch"):
        batch   = [str(p) for p in frames[i:i + batch_size]]
        results = model.predict(batch, imgsz=PLAYER_IMGSZ,
                                conf=PLAYER_CONF,
                                device=DEVICE,
                                half=use_half,
                                classes=detect_classes,
                                verbose=False)
        for path, result in zip(batch, results):
            frame_id = int(Path(path).stem)
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, "player")
                role = _ROBOFLOW_ROLE_MAP.get(class_name, "player")
                # object_type: "ball" for ball, "player" for everyone else
                # (keeps downstream track_players.py unchanged)
                object_type = "ball" if role == "ball" else "player"

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append(dict(
                    frame_id    = frame_id,
                    object_type = object_type,
                    role        = role,
                    track_id    = -1,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    conf        = float(box.conf[0]),
                ))
    return rows


# ─────────────────────────────────────────────────────────────────
#  Player detection (single-class fine-tuned model)
# ─────────────────────────────────────────────────────────────────

def detect_players_batched(frames, model, player_class=0, batch_size=PLAYER_BATCH):
    rows = []
    use_half = (DEVICE != "cpu")
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
                rows.append(dict(
                    frame_id    = frame_id,
                    object_type = "player",
                    role        = "player",
                    track_id    = -1,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    conf        = float(box.conf[0]),
                ))
    return rows


# ─────────────────────────────────────────────────────────────────
#  Ball detection  (SAHI for models without native ball detection)
# ─────────────────────────────────────────────────────────────────

def detect_ball_sahi(frames, ball_model_path):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type           = "ultralytics",
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
            rows.append(dict(
                frame_id    = frame_id,
                object_type = "ball",
                role        = "ball",
                track_id    = -1,
                x1=b.minx, y1=b.miny, x2=b.maxx, y2=b.maxy,
                conf        = pred.score.value,
            ))
    return rows


def detect_ball_vanilla(frames, model, ball_class=32):
    rows = []
    use_half = (DEVICE != "cpu")
    for fp in tqdm(frames, desc="Ball detection (vanilla)", unit="frame"):
        frame_id = int(fp.stem)
        results  = model.predict(str(fp), imgsz=PLAYER_IMGSZ,
                                 conf=BALL_CONF, device=DEVICE,
                                 half=use_half, classes=[ball_class], verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append(dict(
                frame_id    = frame_id,
                object_type = "ball",
                role        = "ball",
                track_id    = -1,
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf        = float(box.conf[0]),
            ))
    return rows


# ─────────────────────────────────────────────────────────────────
#  Deduplication
# ─────────────────────────────────────────────────────────────────

def _keep_best_ball(ball_rows):
    if not ball_rows:
        return []
    df = pd.DataFrame(ball_rows)
    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
    df = df[df["area"] >= BALL_MIN_AREA].drop(columns="area")
    if df.empty:
        return []
    best = (df.sort_values("conf", ascending=False)
              .groupby("frame_id", sort=False)
              .first()
              .reset_index())
    return best.to_dict("records")


# ─────────────────────────────────────────────────────────────────
#  Model introspection: is this a 4-class football model?
# ─────────────────────────────────────────────────────────────────

def _is_4class_football_model(model: YOLO) -> bool:
    """
    Return True if the model has football-specific classes
    (ball, goalkeeper, player, referee).
    The Roboflow football-players-detection model outputs exactly these 4 names.
    """
    names = set(model.names.values())
    return bool(names & {"ball", "goalkeeper", "referee"})


# ─────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────

def run(seq_path="", output_csv="", model_path=MODEL_PATH,
        player_model="", ball_model=""):
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
    print(f"  Device       : {DEVICE}")
    print(f"  SAHI         : {'enabled' if _SAHI_AVAILABLE else 'NOT installed — vanilla fallback'}")
    print(f"  Ball conf    : {BALL_CONF}")

    # ── Load player model and auto-detect type ────────────────────
    p_model = YOLO(p_model_path)
    p_model.to(DEVICE)
    p_model.fuse()

    is_4class = _is_4class_football_model(p_model)

    if is_4class:
        # ── Roboflow 4-class model: one pass detects everything ───
        print(f"\n  Player model : {p_model_path}")
        print(f"  Mode         : 4-CLASS (ball+goalkeeper+player+referee in one pass)")
        print(f"  Classes      : {p_model.names}")
        print(f"  ← This is the Roboflow-quality model. No SAHI needed.\n")

        all_rows = detect_4class_batched(frames, p_model)
        ball_rows = [r for r in all_rows if r["role"] == "ball"]
        other_rows = [r for r in all_rows if r["role"] != "ball"]
        ball_rows = _keep_best_ball(ball_rows)
        all_rows = other_rows + ball_rows

    elif dual_mode and ball_model:
        # ── Dual-model: fine-tuned players + base COCO ball ───────
        print(f"  Player model : {p_model_path}")
        print(f"  Ball model   : {b_model_path}  ← COCO class 32")
        print(f"  Mode         : DUAL-MODEL\n")

        player_rows = detect_players_batched(frames, p_model)
        if _SAHI_AVAILABLE:
            ball_rows = detect_ball_sahi(frames, b_model_path)
        else:
            b_model = YOLO(b_model_path); b_model.to(DEVICE); b_model.fuse()
            ball_rows = detect_ball_vanilla(frames, b_model)
        ball_rows = _keep_best_ball(ball_rows)
        all_rows = player_rows + ball_rows

    elif dual_mode and player_model:
        # player model given, no ball model — use base model for ball
        print(f"  Player model : {p_model_path}")
        print(f"  ⚠  No --ball-model set. Using --model ({model_path}) for ball.\n")
        player_rows = detect_players_batched(frames, p_model)
        if _SAHI_AVAILABLE:
            ball_rows = detect_ball_sahi(frames, model_path)
        else:
            b_model = YOLO(model_path); b_model.to(DEVICE); b_model.fuse()
            ball_rows = detect_ball_vanilla(frames, b_model)
        ball_rows = _keep_best_ball(ball_rows)
        all_rows = player_rows + ball_rows

    else:
        # ── Single model mode ─────────────────────────────────────
        print(f"  Model        : {p_model_path}\n")
        if is_4class:
            all_rows = detect_4class_batched(frames, p_model)
            ball_rows = [r for r in all_rows if r["role"] == "ball"]
            all_rows  = [r for r in all_rows if r["role"] != "ball"] + _keep_best_ball(ball_rows)
        else:
            player_rows = detect_players_batched(frames, p_model)
            if _SAHI_AVAILABLE:
                ball_rows = detect_ball_sahi(frames, p_model_path)
            else:
                ball_rows = detect_ball_vanilla(frames, p_model)
            all_rows = player_rows + _keep_best_ball(ball_rows)

    # ── Build CSV ─────────────────────────────────────────────────
    df = pd.DataFrame(all_rows, columns=[
        "frame_id", "object_type", "role", "track_id",
        "x1", "y1", "x2", "y2", "conf"
    ])
    df.sort_values(["frame_id", "object_type"], inplace=True, ignore_index=True)
    df.to_csv(output_csv, index=False)

    n_players          = (df["object_type"] == "player").sum()
    n_frames_with_ball = df[df["object_type"] == "ball"]["frame_id"].nunique()
    ball_recall_pct    = 100.0 * n_frames_with_ball / max(len(frames), 1)

    # Role breakdown
    role_counts = df["role"].value_counts().to_dict()

    print(f"\n  Detection summary:")
    print(f"     Players     : {role_counts.get('player', 0):,}")
    print(f"     Goalkeepers : {role_counts.get('goalkeeper', 0):,}")
    print(f"     Referees    : {role_counts.get('referee', 0):,}")
    print(f"     Ball frames : {n_frames_with_ball}/{len(frames)}"
          f"  ({ball_recall_pct:.1f}% recall)")

    if n_frames_with_ball == 0:
        print(f"\n  ✖  BALL RECALL = 0%")
        if not is_4class:
            print(f"     Your model has no ball labels.")
            print(f"     Fix: use the Roboflow 4-class model OR add --ball-model yolov8s.pt")
    elif ball_recall_pct < 20:
        print(f"  ⚠  Low ball recall. If using dual-model, try --ball-conf 0.03")

    print(f"\n  Saved → {output_csv}\n")
    return df


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Joint detection: 4-class football model or dual-model fallback."
    )
    p.add_argument("--seq",          default=SEQ_PATH)
    p.add_argument("--out-csv",      default=OUTPUT_CSV)
    p.add_argument("--model",        default=MODEL_PATH,
                   help="Single model. If it has ball/goalkeeper/referee classes, "
                        "all 4 are detected in one pass. Default: yolov8s.pt")
    p.add_argument("--player-model", default="",
                   help="Model for players. If it is a 4-class Roboflow model, "
                        "--ball-model is not needed.")
    p.add_argument("--ball-model",   default="",
                   help="COCO base model for ball (class 32). "
                        "Only needed when --player-model is a players-only fine-tuned model.")
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