"""
team_classifier.py
──────────────────
Three-Stage Team Classification for goalX.

ARCHITECTURE (based on literature review + spatial advantage):
  ─────────────────────────────────────────────────────────────
  Stage 1 — SPATIAL GOALKEEPER DETECTION (unique to goalX)
    Uses projected pitch coordinates from smoothed_tracks.csv.
    Goalkeepers are permanently near their goal (x < GK_ZONE_PX
    or x > PITCH_W - GK_ZONE_PX). Identified geometrically before
    any colour analysis. This is more reliable than colour because
    GK jerseys vary wildly (yellow, green, orange) while position
    does not.

  Stage 2 — K=2 HSV CLUSTERING (home vs away only)
    With GKs removed, K-means has exactly two valid clusters.
    Anchor-only training (tracks >= MIN_ANCHOR_FRAMES) ensures
    cluster centroids reflect stable player appearances.

  Stage 3 — DARKNESS REFEREE DETECTION (sat + val two-channel gate)
    Checked UNCONDITIONALLY BEFORE trusting K-Means confidence.
    A referee jersey must satisfy BOTH:
      median HSV saturation < REFEREE_MAX_SAT  (30)
      median HSV value      < REFEREE_MAX_VAL  (90)
    Two-channel gate prevents false positives:
      Faded jersey:   low sat, HIGH val  → NOT referee
      Dark shadow:    HIGH sat, low val  → NOT referee
      Referee black:  low sat, low val   → referee ✓

RESULT: home / away / goalkeeper / referee / uncertain / unknown
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

SAMPLES_PER_TRACK  = 12
MIN_CROP_PIXELS    = 300
GRASS_HUE_LO       = 35
GRASS_HUE_HI       = 85
MIN_CONF_THRESHOLD = 0.50
TRACK_ID_BALL      = -1
MIN_ANCHOR_FRAMES  = 30

# Spatial GK detection
GK_ZONE_PX         = 120   # px from either goal line on 1050px canvas

# Referee colour detection — BOTH conditions must be true
REFEREE_MAX_SAT    = 30    # 0-255; pure black = 0, dark jerseys ~20-40
REFEREE_MAX_VAL    = 90    # 0-255; pure black = 0, dark jerseys ~50-90

# Pitch canvas dimensions
PITCH_W            = 1050
PITCH_H            = 680

DEFAULT_PROJECTED  = "outputs_193/smoothed_tracks.csv"


# ─────────────────────────────────────────────────────────────────
#  Colour extraction helpers
# ─────────────────────────────────────────────────────────────────

def _extract_hsv_histogram(bgr_crop: np.ndarray):
    """
    Returns (feature_vector, median_saturation, median_value)
    or (None, None, None) on failure.
    """
    if bgr_crop.size == 0 or bgr_crop.shape[0] * bgr_crop.shape[1] < MIN_CROP_PIXELS:
        return None, None, None

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

    green_mask = (hsv[:, :, 0] > GRASS_HUE_LO) & (hsv[:, :, 0] < GRASS_HUE_HI)
    hsv_filtered = hsv[~green_mask]

    if len(hsv_filtered) < 40:
        return None, None, None

    median_sat = float(np.median(hsv_filtered[:, 1]))
    median_val = float(np.median(hsv_filtered[:, 2]))

    hist_h = np.histogram(hsv_filtered[:, 0], bins=18, range=(0, 180))[0]
    hist_s = np.histogram(hsv_filtered[:, 1], bins=8,  range=(0, 256))[0]
    feature = np.concatenate([hist_h, hist_s]).astype(np.float32)

    total = feature.sum()
    if total > 0:
        feature /= total

    return feature, median_sat, median_val


def _sample_colours_and_crop(rows: pd.DataFrame, frames_dir: Path, n: int):
    """Returns (feature_list, sat_val_tuples, best_crop)."""
    if len(rows) > n:
        rows = rows.sample(n=n, random_state=42)

    features       = []
    sat_val_tuples = []
    best_crop      = None

    for _, row in rows.iterrows():
        fpath = frames_dir / f"{int(row['frame_id']):06d}.jpg"
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        x1 = max(0, int(row["x1"]));  x2 = min(w_img, int(row["x2"]))
        y1 = max(0, int(row["y1"]));  y2 = min(h_img, int(row["y2"]))
        box_h = y2 - y1

        torso_y1 = y1 + int(box_h * 0.15)
        torso_y2 = y1 + int(box_h * 0.70)
        torso_y1 = min(max(torso_y1, 0), h_img)
        torso_y2 = min(max(torso_y2, torso_y1 + 1), h_img)

        crop = img[torso_y1:torso_y2, x1:x2]
        feat, sat, val = _extract_hsv_histogram(crop)
        if feat is not None:
            features.append(feat)
            sat_val_tuples.append((sat, val))
            if best_crop is None or crop.size > best_crop.size:
                best_crop = crop

    return features, sat_val_tuples, best_crop


def create_mosaic(crops, out_path):
    if not crops:
        return
    size = 60
    cols = min(10, len(crops))
    rows = math.ceil(len(crops) / cols)
    mosaic = np.zeros((rows * size, cols * size, 3), dtype=np.uint8)
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        resized = cv2.resize(crop, (size, size))
        mosaic[r * size:(r + 1) * size, c * size:(c + 1) * size] = resized
    cv2.imwrite(str(out_path), mosaic)


# ─────────────────────────────────────────────────────────────────
#  Stage 1: Spatial Goalkeeper Detection
# ─────────────────────────────────────────────────────────────────

def _detect_goalkeepers_spatial(player_ids: list, projected_csv: str) -> set:
    try:
        proj = pd.read_csv(projected_csv)
    except FileNotFoundError:
        print(f"  ⚠  Projected tracks not found at {projected_csv}. Skipping GK detection.")
        return set()

    if "in_canvas" in proj.columns:
        proj = proj[proj["in_canvas"] == True]

    gk_ids = set()
    for tid in player_ids:
        track = proj[proj["track_id"] == tid]
        if len(track) < 5:
            continue
        median_x = track["pitch_x"].median()
        if median_x < GK_ZONE_PX or median_x > (PITCH_W - GK_ZONE_PX):
            gk_ids.add(int(tid))

    return gk_ids


# ─────────────────────────────────────────────────────────────────
#  Stage 3: Referee Detection — sat + val two-channel gate
# ─────────────────────────────────────────────────────────────────

def _is_referee(sat_val_tuples: list) -> bool:
    """
    True only if jersey is BOTH low-saturation AND low-brightness.
    Prevents faded jerseys (low sat, high val) and shadows (high sat,
    low val) from triggering false positives.
    """
    if not sat_val_tuples:
        return False
    med_sat = float(np.median([s for s, v in sat_val_tuples]))
    med_val = float(np.median([v for s, v in sat_val_tuples]))
    return med_sat < REFEREE_MAX_SAT and med_val < REFEREE_MAX_VAL


# ─────────────────────────────────────────────────────────────────
#  Main entry-point
# ─────────────────────────────────────────────────────────────────

def classify_teams(tracks_csv: str, frames_dir: str,
                   out_file_path: str,
                   projected_csv: str = DEFAULT_PROJECTED,
                   n_samples: int     = SAMPLES_PER_TRACK,
                   min_anchor: int    = MIN_ANCHOR_FRAMES) -> pd.DataFrame:

    tracks_csv = Path(tracks_csv)
    frames_dir = Path(frames_dir)
    out_path   = Path(out_file_path)
    out_dir    = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX — Three-Stage Team Classifier")
    print(f"  {'─' * 40}")
    print(f"  Referee gate : sat < {REFEREE_MAX_SAT} AND val < {REFEREE_MAX_VAL}")
    print(f"  GK zone      : pitch_x < {GK_ZONE_PX} OR > {PITCH_W - GK_ZONE_PX}")

    df = pd.read_csv(tracks_csv)
    missing = {"frame_id", "track_id", "x1", "y1", "x2", "y2"} - set(df.columns)
    if missing:
        raise ValueError(f"Tracks CSV missing columns: {missing}")

    all_track_ids = sorted(df["track_id"].unique())
    ball_ids   = [tid for tid in all_track_ids if tid < 0]
    player_ids = [tid for tid in all_track_ids if tid >= 0]

    if ball_ids:
        print(f"  ✔  Excluding {len(ball_ids)} ball track(s).")

    # ── STAGE 1 ───────────────────────────────────────────────────
    print(f"\n  [Stage 1] Spatial goalkeeper detection …")
    gk_ids = _detect_goalkeepers_spatial(player_ids, projected_csv)
    print(f"  ✔  {len(gk_ids)} track(s) flagged as goalkeeper: {sorted(gk_ids)}")

    field_ids = [tid for tid in player_ids if tid not in gk_ids]
    print(f"  ✔  {len(field_ids)} tracks remain for colour clustering")

    track_counts = df[df["track_id"].isin(field_ids)].groupby("track_id").size()
    anchor_ids   = track_counts[track_counts >= min_anchor].index.tolist()
    if len(anchor_ids) < 2:
        print(f"  ⚠  Not enough anchors ({len(anchor_ids)}), using all field tracks.")
        anchor_ids = field_ids
    print(f"  ✔  {len(anchor_ids)} anchor field tracks (>= {min_anchor} frames)")

    # ── Colour extraction ─────────────────────────────────────────
    print(f"\n  [Stage 2a] Extracting jersey colours …")
    colour_map  = {}
    sat_val_map = {}
    crop_map    = {}

    for tid in tqdm(player_ids, desc="  Jersey extraction"):
        rows = df[df["track_id"] == tid]
        feats, sv_tuples, best_crop = _sample_colours_and_crop(rows, frames_dir, n_samples)
        colour_map[int(tid)]  = feats
        sat_val_map[int(tid)] = sv_tuples
        if best_crop is not None:
            crop_map[int(tid)] = best_crop

    n_no_crops = sum(1 for tid in player_ids if not colour_map[int(tid)])
    if n_no_crops:
        print(f"  ⚠  {n_no_crops} tracks had zero valid crops.")

    # ── Diagnostic: show sat/val per track so threshold tuning is easy ──
    print(f"\n  Colour diagnostics (helps tune referee thresholds):")
    print(f"  {'Track':>6} | {'med_sat':>7} | {'med_val':>7} | Flag")
    print(f"  {'─' * 42}")
    for tid in sorted(field_ids):
        sv = sat_val_map.get(int(tid), [])
        if sv:
            ms = float(np.median([s for s, v in sv]))
            mv = float(np.median([v for s, v in sv]))
            flag = "  ← referee" if (ms < REFEREE_MAX_SAT and mv < REFEREE_MAX_VAL) else ""
            print(f"  {tid:>6} | {ms:>7.1f} | {mv:>7.1f} |{flag}")

    # ── STAGE 2: K=2 clustering ───────────────────────────────────
    print(f"\n  [Stage 2b] K=2 clustering on {len(anchor_ids)} anchor field tracks …")
    anchor_colours = []
    for tid in anchor_ids:
        anchor_colours.extend(colour_map[int(tid)])

    if len(anchor_colours) < 2:
        raise RuntimeError("Not enough colour samples for clustering.")

    X = np.stack(anchor_colours)
    km = KMeans(n_clusters=2, n_init=20, max_iter=500, random_state=42)
    km.fit(X)

    cluster_sizes = {0: 0, 1: 0}
    field_records = []

    for tid in field_ids:
        feats = colour_map[int(tid)]
        if not feats:
            field_records.append({
                "track_id": tid, "cluster_id": -1,
                "confidence": 0.0, "team": "unknown"
            })
            continue

        preds    = km.predict(np.stack(feats))
        counter  = Counter(preds.tolist())
        best_cl, best_cnt = counter.most_common(1)[0]
        conf = best_cnt / len(preds)

        field_records.append({
            "track_id":   int(tid),
            "cluster_id": int(best_cl),
            "confidence": round(conf, 3),
        })
        if conf >= MIN_CONF_THRESHOLD:
            cluster_sizes[int(best_cl)] += 1

    sorted_cl = sorted(cluster_sizes.keys(), key=lambda c: cluster_sizes[c], reverse=True)
    label_map = {sorted_cl[0]: "home", sorted_cl[1]: "away"}
    print(f"  ✔  Cluster {sorted_cl[0]} ({cluster_sizes[sorted_cl[0]]} tracks) → home")
    print(f"  ✔  Cluster {sorted_cl[1]} ({cluster_sizes[sorted_cl[1]]} tracks) → away")

    # ── STAGE 3: Referee check — runs FIRST, overrides K-Means ───
    print(f"\n  [Stage 3] Referee detection …")
    n_referees = 0
    for r in field_records:
        tid = r["track_id"]
        if r.get("cluster_id", -1) == -1:
            r["team"] = "unknown"
            continue

        colour_label = label_map.get(r["cluster_id"], "other")

        # Referee check is unconditional — overrides K-Means confidence
        if _is_referee(sat_val_map.get(int(tid), [])):
            r["team"] = "referee"
            n_referees += 1
        elif r["confidence"] >= MIN_CONF_THRESHOLD:
            r["team"] = colour_label
        else:
            r["team"] = "uncertain"

    print(f"  ✔  {n_referees} track(s) identified as referee")

    # ── Assemble all records ──────────────────────────────────────
    all_records = list(field_records)

    for tid in gk_ids:
        all_records.append({
            "track_id": int(tid), "team": "goalkeeper",
            "cluster_id": -1, "confidence": 1.0,
        })
    for tid in ball_ids:
        all_records.append({
            "track_id": int(tid), "team": "ball",
            "cluster_id": -1, "confidence": 1.0,
        })

    result_df = pd.DataFrame(all_records).sort_values("track_id", ignore_index=True)

    # ── Mosaics ───────────────────────────────────────────────────
    print("\n  Generating debug mosaics …")
    for label in ["home", "away", "goalkeeper", "referee", "uncertain", "unknown"]:
        tids  = result_df[result_df["team"] == label]["track_id"].tolist()
        crops = [crop_map[t] for t in tids if t in crop_map]
        if crops:
            create_mosaic(crops, out_dir / f"debug_team_{label}.jpg")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n  📊 Assignment summary:")
    player_df = result_df[result_df["team"] != "ball"]
    for team, cnt in player_df["team"].value_counts().items():
        print(f"     {team:<12} : {cnt} track IDs")

    result_df.to_csv(out_path, index=False)
    print(f"\n  💾 Saved → {out_path}")
    print(f"  🖼️  Mosaics → {out_dir}/debug_team_*.jpg")
    print(f"\n  ⚠️  CHECK mosaics. If home/away are swapped, edit the CSV manually.")
    print(f"     To tune referee detection: --referee-max-sat / --referee-max-val\n")
    return result_df


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Three-stage team classifier: spatial GK + K=2 + referee darkness."
    )
    parser.add_argument("--tracks",           required=True)
    parser.add_argument("--frames-dir",       required=True)
    parser.add_argument("--out",              required=True)
    parser.add_argument("--projected-tracks", default=DEFAULT_PROJECTED)
    parser.add_argument("--samples",          type=int, default=SAMPLES_PER_TRACK)
    parser.add_argument("--min-anchor",       type=int, default=MIN_ANCHOR_FRAMES)
    parser.add_argument("--gk-zone",          type=int, default=GK_ZONE_PX)
    parser.add_argument("--referee-max-sat",  type=int, default=REFEREE_MAX_SAT,
                        help="Max HSV saturation for referee (default 30)")
    parser.add_argument("--referee-max-val",  type=int, default=REFEREE_MAX_VAL,
                        help="Max HSV brightness for referee (default 90)")
    args = parser.parse_args()

    GK_ZONE_PX      = args.gk_zone
    REFEREE_MAX_SAT = args.referee_max_sat
    REFEREE_MAX_VAL = args.referee_max_val

    classify_teams(
        tracks_csv    = args.tracks,
        frames_dir    = args.frames_dir,
        out_file_path = args.out,
        projected_csv = args.projected_tracks,
        n_samples     = args.samples,
        min_anchor    = args.min_anchor,
    )
