"""
team_classifier.py
──────────────────
Advanced Anchor-Based Team Classification.

MERGED FEATURES:
  1. Ball excluded from K-Means (outputs team="ball")
  2. Low-confidence tracks marked "uncertain" (< 0.50)
  3. Minimum crop sizes and grass filtering
  4. Anchor-only K-Means (only uses tracks with >= 30 frames to define clusters)
  5. Auto-remap (Largest 2 clusters = home/away, remaining = other/refs)
  6. Debug Mosaics (outputs jpg collages of the shirts for easy verification)
  7. NEW: Torso-only cropping (15% to 70% of height) to ignore grass and socks
  8. NEW: HSV Histogram feature extraction for robust pigment clustering
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
DEFAULT_CLUSTERS   = 4      # K=4 to split Home, Away, GK, Ref
GRASS_HUE_LO       = 35
GRASS_HUE_HI       = 85
MIN_CONF_THRESHOLD = 0.50   # Tracks below this → "uncertain"
TRACK_ID_BALL      = -1     # Exclude from clustering
MIN_ANCHOR_FRAMES  = 30     # Only tracks with this many frames vote for centroids


# ─────────────────────────────────────────────────────────────────
#  Colour extraction & Mosaic helpers
# ─────────────────────────────────────────────────────────────────

def _extract_hsv_histogram(bgr_crop: np.ndarray) -> np.ndarray | None:
    """Extracts a normalized H/S histogram from non-grass pixels."""
    if bgr_crop.size == 0 or bgr_crop.shape[0] * bgr_crop.shape[1] < MIN_CROP_PIXELS:
        return None
    
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    
    # Mask out green grass pixels before computing histogram
    green_mask = (hsv[:,:,0] > GRASS_HUE_LO) & (hsv[:,:,0] < GRASS_HUE_HI)
    hsv_filtered = hsv[~green_mask]
    
    if len(hsv_filtered) < 40:
        return None
        
    # Use H and S channels only, flatten to feature vector
    hist_h = np.histogram(hsv_filtered[:,0], bins=18, range=(0,180))[0]
    hist_s = np.histogram(hsv_filtered[:,1], bins=8, range=(0,256))[0]
    feature = np.concatenate([hist_h, hist_s]).astype(np.float32)
    
    # Normalize the histogram so bounding box size doesn't skew clustering distance
    feature_sum = feature.sum()
    if feature_sum > 0:
        feature /= feature_sum
        
    return feature

def _sample_colours_and_crop(rows: pd.DataFrame, frames_dir: Path, n: int):
    """Returns (list_of_hsv_histograms, best_bgr_crop_for_mosaic)"""
    if len(rows) > n:
        rows = rows.sample(n=n, random_state=42)
    colours = []
    best_crop = None

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
        
        # Use only torso (skip head, stop before legs)
        torso_y1 = y1 + int(box_h * 0.15)
        torso_y2 = y1 + int(box_h * 0.70)
        
        # Bounds check
        torso_y1 = min(max(torso_y1, 0), h_img)
        torso_y2 = min(max(torso_y2, torso_y1 + 1), h_img)
        
        crop = img[torso_y1:torso_y2, x1:x2]
        
        c = _extract_hsv_histogram(crop)
        if c is not None:
            colours.append(c)
            if best_crop is None or crop.size > best_crop.size:
                best_crop = crop
                
    return colours, best_crop

def create_mosaic(crops, out_path):
    """Stitch player crops into a grid for debugging."""
    if not crops:
        return
    size = 60
    cols = min(10, len(crops))
    rows = math.ceil(len(crops) / cols)
    
    mosaic = np.zeros((rows * size, cols * size, 3), dtype=np.uint8)
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        resized = cv2.resize(crop, (size, size))
        mosaic[r*size:(r+1)*size, c*size:(c+1)*size] = resized
        
    cv2.imwrite(str(out_path), mosaic)


# ─────────────────────────────────────────────────────────────────
#  Main entry-point
# ─────────────────────────────────────────────────────────────────

def classify_teams(tracks_csv: str, frames_dir: str,
                   out_file_path: str,
                   n_samples: int = SAMPLES_PER_TRACK,
                   min_anchor: int = MIN_ANCHOR_FRAMES,
                   n_clusters: int = DEFAULT_CLUSTERS) -> pd.DataFrame:

    tracks_csv = Path(tracks_csv)
    frames_dir = Path(frames_dir)
    out_path   = Path(out_file_path)
    out_dir    = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX — Advanced Team Classifier")
    print(f"  {'─' * 40}")

    df = pd.read_csv(tracks_csv)
    missing = {"frame_id", "track_id", "x1", "y1", "x2", "y2"} - set(df.columns)
    if missing:
        raise ValueError(f"Tracks CSV missing columns: {missing}")

    # ── FIX 1: separate ball rows before clustering ────────────────
    all_track_ids = sorted(df["track_id"].unique())
    ball_ids      = [tid for tid in all_track_ids if tid < 0]
    player_ids    = [tid for tid in all_track_ids if tid >= 0]

    if ball_ids:
        print(f"  ✔  Excluding {len(ball_ids)} ball track(s) from colour clustering.")

    # Find Anchor Tracks
    track_counts = df[df["track_id"] >= 0].groupby("track_id").size()
    anchor_ids = track_counts[track_counts >= min_anchor].index.tolist()
    
    # Fallback if video is too short
    if len(anchor_ids) < n_clusters:
        print(f"  ⚠  Not enough tracks with >={min_anchor} frames. Falling back to all tracks.")
        anchor_ids = player_ids

    print(f"  ✔  {len(player_ids)} total player tracks")
    print(f"  ✔  {len(anchor_ids)} Anchor tracks (>= {min_anchor} frames) will define clusters")

    # ── Step 1: colour extraction (players only) ──────────────────
    colour_map:  dict[int, list] = {}
    crop_map:    dict[int, np.ndarray] = {}
    anchor_colours = []
    n_no_crops = 0

    for tid in tqdm(player_ids, desc="  Jersey extraction"):
        rows = df[df["track_id"] == tid]
        colours, best_crop = _sample_colours_and_crop(rows, frames_dir, n_samples)
        
        colour_map[int(tid)] = colours
        if best_crop is not None:
            crop_map[int(tid)] = best_crop
            
        if not colours:
            n_no_crops += 1
        elif tid in anchor_ids:
            anchor_colours.extend(colours)

    if n_no_crops > 0:
        print(f"  ⚠  {n_no_crops} player tracks had zero valid colour crops (too small).")

    if len(anchor_colours) < n_clusters:
        raise RuntimeError(f"Only {len(anchor_colours)} valid anchor samples — need at least {n_clusters}.")

    # ── Step 2: K-Means strictly on Anchor Tracks ─────────────────
    X_anchors = np.stack(anchor_colours)
    print(f"\n  Clustering {len(X_anchors)} anchor samples into {n_clusters} jersey groups …")
    km = KMeans(n_clusters=n_clusters, n_init=15, max_iter=300, random_state=42)
    km.fit(X_anchors)

    # ── Step 3: Majority-vote & Auto-Remap ────────────────────────
    records = []
    cluster_sizes = {i: 0 for i in range(n_clusters)}

    for tid in player_ids:
        colours = colour_map[int(tid)]
        if not colours:
            records.append({"track_id": tid, "team": "unknown", "cluster_id": -1, "confidence": 0.0})
            continue

        preds   = km.predict(np.stack(colours))
        counter = Counter(preds.tolist())
        best_cluster, best_count = counter.most_common(1)[0]
        conf = best_count / len(preds)

        records.append({
            "track_id":   int(tid),
            "cluster_id": int(best_cluster),
            "confidence": round(conf, 3),
        })
        
        # Count sizes for auto-remap (only confident tracks)
        if conf >= MIN_CONF_THRESHOLD:
            cluster_sizes[int(best_cluster)] += 1

    # Auto-remap: 2 largest are "home"/"away", remaining are "other" (refs/GKs)
    sorted_clusters = sorted(cluster_sizes.keys(), key=lambda c: cluster_sizes[c], reverse=True)
    label_map = {}
    if len(sorted_clusters) >= 2:
        label_map[sorted_clusters[0]] = "home"
        label_map[sorted_clusters[1]] = "away"
        for c in sorted_clusters[2:]:
            label_map[c] = "other"
    else:
        for c in sorted_clusters:
            label_map[c] = "other"

    # Apply labels and FIX 2 (uncertainties)
    for r in records:
        if r["cluster_id"] == -1:
            continue
        if r["confidence"] < MIN_CONF_THRESHOLD:
            r["team"] = "uncertain"
        else:
            r["team"] = label_map[r["cluster_id"]]

    # ── Add ball rows with team="ball" ────────────────────────────
    for tid in ball_ids:
        records.append({
            "track_id":   int(tid),
            "team":       "ball",
            "cluster_id": -1,
            "confidence": 1.0,
        })

    result_df = pd.DataFrame(records).sort_values("track_id", ignore_index=True)

    # ── Generate Mosaics ──────────────────────────────────────────
    print("\n  Generating debug mosaics...")
    for cluster_id in range(n_clusters):
        team_name = label_map.get(cluster_id, "unknown")
        tids = result_df[result_df["cluster_id"] == cluster_id]["track_id"].tolist()
        crops = [crop_map[t] for t in tids if t in crop_map]
        mosaic_path = out_dir / f"debug_team_{team_name}_cluster{cluster_id}.jpg"
        create_mosaic(crops, mosaic_path)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n  📊 Assignment summary (players only):")
    player_df = result_df[result_df["team"] != "ball"]
    for team, cnt in player_df["team"].value_counts().items():
        print(f"     {team:<12} : {cnt} track IDs")

    if (player_df["team"] == "uncertain").any():
        n_unc = (player_df["team"] == "uncertain").sum()
        print(f"\n  ⚠  {n_unc} tracks labelled 'uncertain' (conf < {MIN_CONF_THRESHOLD}).")

    result_df.to_csv(out_path, index=False)
    print(f"\n  💾 Saved CSV -> {out_path}")
    print(f"  🖼️  Saved Mosaics -> {out_dir}/debug_team_*.jpg")
    print(f"  ⚠️  CHECK THE MOSAICS! If home/away are swapped, edit {out_path} manually.\n")
    return result_df

# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign player tracks to Home / Away / Other via jersey colour."
    )
    parser.add_argument("--tracks",     required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--out",        required=True)
    parser.add_argument("--samples",    type=int, default=SAMPLES_PER_TRACK)
    parser.add_argument("--min-anchor", type=int, default=MIN_ANCHOR_FRAMES)
    parser.add_argument("--clusters",   type=int, default=DEFAULT_CLUSTERS, help="Number of KMeans clusters to find")
    args = parser.parse_args()
    
    classify_teams(
        tracks_csv    = args.tracks,
        frames_dir    = args.frames_dir,
        out_file_path = args.out,
        n_samples     = args.samples,
        min_anchor    = args.min_anchor,
        n_clusters    = args.clusters
    )
