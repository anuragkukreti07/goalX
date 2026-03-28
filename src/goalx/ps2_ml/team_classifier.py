# """
# team_classifier.py
# ──────────────────
# Assigns every track_id to a team (home, away, or other/referee) by clustering
# the dominant jersey colour extracted from bounding-box crops.

# Method
# ──────
# 1. For each track_id, randomly sample up to N frames.
# 2. Crop the top 55 % of the box (shirt region — excludes legs and boots).
# 3. Mask out grass pixels in HSV, then run K-Means(2) inside the crop to
#    find the dominant non-grass colour.
# 4. Collect one HSV colour vector per sample.
# 5. Run K-Means(3) on all vectors to produce three global clusters.
# 6. Majority-vote per track_id across its samples to lock in a label.

# ⚠️  The labels "home / away / other" are mapped to cluster IDs 0 / 1 / 2 in
#     the order they appear.  Inspect the CSV and rename cluster_id values to
#     match your actual teams before running formation_detector.py.

# Input
# ─────
#   --tracks    CSV with frame_id, track_id, x1, y1, x2, y2  (image pixels)
#               This is the raw output of track_players.py — NOT the projected CSV.
#   --frames    Directory of original .jpg frames
#   --out-dir   Output directory

# Output
# ──────
#   <out-dir>/team_assignments.csv
#       Columns: track_id, team, cluster_id, confidence

# Usage
# ─────
#   python3 src/goalx/ps1_cv/team_classifier.py \\
#       --tracks  outputs/tracks.csv \\
#       --frames  data/raw_videos/tracking/test/SNMOT-116/img1/ \\
#       --out-dir outputs/teams
# """

# import argparse
# import random
# from collections import Counter
# from pathlib import Path

# import cv2
# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from tqdm import tqdm

# # ─────────────────────────────────────────────────────────────────
# #  CONFIG
# # ─────────────────────────────────────────────────────────────────

# SAMPLES_PER_TRACK = 12      # frames to sample per track_id
# SHIRT_CROP_FRAC   = 0.55    # top 55 % of box = shirt, not legs
# MIN_CROP_PIXELS   = 300     # skip crops smaller than this (tiny detections)
# N_TEAMS           = 3       # home, away, other
# GRASS_HUE_LO      = 35      # HSV hue band for grass (to discard)
# GRASS_HUE_HI      = 85
# GRASS_SAT_MIN     = 40


# # ─────────────────────────────────────────────────────────────────
# #  Colour extraction helpers
# # ─────────────────────────────────────────────────────────────────

# def _non_grass_mask(hsv: np.ndarray) -> np.ndarray:
#     """Boolean mask — True where pixel is NOT grass."""
#     h, s = hsv[:, :, 0], hsv[:, :, 1]
#     is_grass = (h >= GRASS_HUE_LO) & (h <= GRASS_HUE_HI) & (s > GRASS_SAT_MIN)
#     return ~is_grass


# def _dominant_hsv(bgr_crop: np.ndarray) -> np.ndarray | None:
#     """
#     Return the dominant non-grass HSV colour from a BGR crop.
#     Uses K-Means(2) on valid pixels and returns the larger cluster centroid.
#     Returns None when the crop has too few usable pixels.
#     """
#     if bgr_crop.size == 0 or bgr_crop.shape[0] * bgr_crop.shape[1] < MIN_CROP_PIXELS:
#         return None

#     hsv    = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
#     pixels = hsv[_non_grass_mask(hsv)]   # (N, 3)

#     if len(pixels) < 40:
#         return None

#     k  = min(2, len(pixels))
#     km = KMeans(n_clusters=k, n_init=3, random_state=0)
#     km.fit(pixels)

#     _, counts     = np.unique(km.labels_, return_counts=True)
#     dominant_idx  = int(np.argmax(counts))
#     return km.cluster_centers_[dominant_idx].astype(np.float32)


# # ─────────────────────────────────────────────────────────────────
# #  Per-track colour sampling
# # ─────────────────────────────────────────────────────────────────

# def _sample_colours(
#     rows:       pd.DataFrame,
#     frames_dir: Path,
#     n:          int,
# ) -> list[np.ndarray]:
#     """Sample up to n frames for one track and extract dominant shirt colours."""
#     if len(rows) > n:
#         rows = rows.sample(n=n, random_state=42)

#     colours = []
#     for _, row in rows.iterrows():
#         fpath = frames_dir / f"{int(row['frame_id']):06d}.jpg"
#         if not fpath.exists():
#             continue

#         img = cv2.imread(str(fpath))
#         if img is None:
#             continue

#         h_img, w_img = img.shape[:2]
#         x1 = max(0, int(row["x1"]))
#         y1 = max(0, int(row["y1"]))
#         x2 = min(w_img, int(row["x2"]))
#         y2 = min(h_img, int(row["y2"]))

#         box_h = y2 - y1
#         # Crop shirt region only (top 55 % of box height)
#         shirt_y2 = y1 + max(1, int(box_h * SHIRT_CROP_FRAC))
#         crop = img[y1:shirt_y2, x1:x2]

#         c = _dominant_hsv(crop)
#         if c is not None:
#             colours.append(c)

#     return colours


# # ─────────────────────────────────────────────────────────────────
# #  Main entry-point
# # ─────────────────────────────────────────────────────────────────

# def classify_teams(
#     tracks_csv: str,
#     frames_dir: str,
#     out_dir:    str,
#     n_samples:  int = SAMPLES_PER_TRACK,
# ) -> pd.DataFrame:

#     tracks_csv = Path(tracks_csv)
#     frames_dir = Path(frames_dir)
#     out_dir    = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     print(f"\n  goalX — Team Classifier")
#     print(f"  {'─' * 40}")

#     df = pd.read_csv(tracks_csv)
#     missing = {"frame_id", "track_id", "x1", "y1", "x2", "y2"} - set(df.columns)
#     if missing:
#         raise ValueError(f"Tracks CSV missing columns: {missing}")

#     track_ids = sorted(df["track_id"].unique())
#     print(f"  ✔  {len(track_ids)} unique tracks  |  sampling ≤{n_samples} frames each")

#     # ── Step 1: extract per-track colour vectors ─────────────────
#     all_colours:     list[np.ndarray]       = []
#     colour_map:      dict[int, list]        = {}

#     for tid in tqdm(track_ids, desc="Jersey extraction"):
#         rows    = df[df["track_id"] == tid]
#         colours = _sample_colours(rows, frames_dir, n_samples)
#         colour_map[int(tid)] = colours
#         all_colours.extend(colours)

#     if len(all_colours) < N_TEAMS:
#         raise RuntimeError(
#             "Too few valid colour samples — verify --frames directory is correct "
#             "and contains the original .jpg frames."
#         )

#     # ── Step 2: global K-Means → 3 jersey clusters ───────────────
#     X  = np.stack(all_colours)
#     print(f"\n  Clustering {len(X)} samples into {N_TEAMS} jersey groups …")
#     km = KMeans(n_clusters=N_TEAMS, n_init=15, max_iter=300, random_state=42)
#     km.fit(X)

#     # ── Step 3: majority-vote per track ──────────────────────────
#     records = []
#     for tid in track_ids:
#         colours = colour_map[int(tid)]
#         if not colours:
#             records.append({"track_id": tid, "team": "unknown",
#                             "cluster_id": -1, "confidence": 0.0})
#             continue

#         preds   = km.predict(np.stack(colours))
#         counter = Counter(preds.tolist())
#         best_cluster, best_count = counter.most_common(1)[0]
#         conf = best_count / len(preds)

#         # cluster_id is stable; label mapping is a placeholder —
#         # the user must rename home/away/other after inspecting the CSV.
#         label_map = {0: "home", 1: "away", 2: "other"}
#         records.append({
#             "track_id":   int(tid),
#             "team":       label_map.get(int(best_cluster), "other"),
#             "cluster_id": int(best_cluster),
#             "confidence": round(conf, 3),
#         })

#     result_df = pd.DataFrame(records).sort_values("track_id", ignore_index=True)

#     # ── Summary ───────────────────────────────────────────────────
#     print(f"\n  📊 Assignment summary:")
#     for team, cnt in result_df["team"].value_counts().items():
#         print(f"     {team:<8} : {cnt} track IDs")

#     print(f"\n  ⚠️  Cluster labels (home/away/other) are arbitrary.")
#     print(f"     Open team_assignments.csv, check which cluster_id corresponds")
#     print(f"     to which jersey colour, and update the label column manually.")

#     out_csv = out_dir / "team_assignments.csv"
#     result_df.to_csv(out_csv, index=False)
#     print(f"\n  💾 Saved → {out_csv}")
#     print(f"  ✅  Classification complete.")
#     print(f"      Next step: formation_detector.py --teams {out_csv}\n")
#     return result_df


# # ─────────────────────────────────────────────────────────────────
# #  CLI
# # ─────────────────────────────────────────────────────────────────

# def _parse_args():
#     p = argparse.ArgumentParser(
#         description="Assign player tracks to Home / Away / Other via jersey colour."
#     )
#     p.add_argument("--tracks",  required=True,
#                    help="tracks.csv from track_players.py  (image-space bbox coords)")
#     p.add_argument("--frames",  required=True,
#                    help="Directory of original .jpg frames")
#     p.add_argument("--out-dir", default="outputs/teams")
#     p.add_argument("--samples", type=int, default=SAMPLES_PER_TRACK,
#                    help=f"Frames sampled per track  (default: {SAMPLES_PER_TRACK})")
#     return p.parse_args()


# # if __name__ == "__main__":
# #     args = _parse_args()
# #     classify_teams(args.tracks, args.frames, args.out_dir, args.samples)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Assign player tracks to Home / Away / Other via jersey colour."
#     )
#     # These names now match exactly what run_goalx.py sends
#     parser.add_argument("--tracks", required=True)
#     parser.add_argument("--frames-dir", required=True)
#     parser.add_argument("--out", required=True)
    
#     # We add this as an optional one in case you ever want to change it from the CLI
#     parser.add_argument("--samples", type=int, default=12)
    
#     args = parser.parse_args()
    
#     # Passing the correct variables to the function
#     classify_teams(
#         tracks_csv=args.tracks, 
#         frames_dir=args.frames_dir, 
#         out_dir=args.out, 
#         n_samples=args.samples
#     )


"""
team_classifier.py
──────────────────
Assigns every track_id to a team (home, away, or other/referee) by clustering
the dominant jersey colour extracted from bounding-box crops.
"""

import argparse
import random
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

SAMPLES_PER_TRACK = 12      # frames to sample per track_id
SHIRT_CROP_FRAC   = 0.55    # top 55 % of box = shirt, not legs
MIN_CROP_PIXELS   = 300     # skip crops smaller than this (tiny detections)
N_TEAMS           = 3       # home, away, other
GRASS_HUE_LO      = 35      # HSV hue band for grass (to discard)
GRASS_HUE_HI      = 85
GRASS_SAT_MIN     = 40


# ─────────────────────────────────────────────────────────────────
#  Colour extraction helpers
# ─────────────────────────────────────────────────────────────────

def _non_grass_mask(hsv: np.ndarray) -> np.ndarray:
    """Boolean mask — True where pixel is NOT grass."""
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    is_grass = (h >= GRASS_HUE_LO) & (h <= GRASS_HUE_HI) & (s > GRASS_SAT_MIN)
    return ~is_grass


def _dominant_hsv(bgr_crop: np.ndarray) -> np.ndarray | None:
    """
    Return the dominant non-grass HSV colour from a BGR crop.
    """
    if bgr_crop.size == 0 or bgr_crop.shape[0] * bgr_crop.shape[1] < MIN_CROP_PIXELS:
        return None

    hsv    = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    pixels = hsv[_non_grass_mask(hsv)]

    if len(pixels) < 40:
        return None

    k  = min(2, len(pixels))
    km = KMeans(n_clusters=k, n_init=3, random_state=0)
    km.fit(pixels)

    _, counts     = np.unique(km.labels_, return_counts=True)
    dominant_idx  = int(np.argmax(counts))
    return km.cluster_centers_[dominant_idx].astype(np.float32)


# ─────────────────────────────────────────────────────────────────
#  Per-track colour sampling
# ─────────────────────────────────────────────────────────────────

def _sample_colours(
    rows:       pd.DataFrame,
    frames_dir: Path,
    n:          int,
) -> list[np.ndarray]:
    if len(rows) > n:
        rows = rows.sample(n=n, random_state=42)

    colours = []
    for _, row in rows.iterrows():
        fpath = frames_dir / f"{int(row['frame_id']):06d}.jpg"
        if not fpath.exists():
            continue

        img = cv2.imread(str(fpath))
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        x1 = max(0, int(row["x1"]))
        y1 = max(0, int(row["y1"]))
        x2 = min(w_img, int(row["x2"]))
        y2 = min(h_img, int(row["y2"]))

        box_h = y2 - y1
        shirt_y2 = y1 + max(1, int(box_h * SHIRT_CROP_FRAC))
        crop = img[y1:shirt_y2, x1:x2]

        c = _dominant_hsv(crop)
        if c is not None:
            colours.append(c)

    return colours


# ─────────────────────────────────────────────────────────────────
#  Main entry-point
# ─────────────────────────────────────────────────────────────────

def classify_teams(
    tracks_csv: str,
    frames_dir: str,
    out_file_path: str,  # This is now the target CSV path
    n_samples:  int = SAMPLES_PER_TRACK,
) -> pd.DataFrame:

    tracks_csv = Path(tracks_csv)
    frames_dir = Path(frames_dir)
    out_path   = Path(out_file_path)
    
    # ── PATH FIX ──────────────────────────────────────────────────
    # Creates 'outputs/', NOT 'outputs/team_labels.csv/'
    out_path.parent.mkdir(parents=True, exist_ok=True) 

    print(f"\n  goalX — Team Classifier")
    print(f"  {'─' * 40}")

    df = pd.read_csv(tracks_csv)
    missing = {"frame_id", "track_id", "x1", "y1", "x2", "y2"} - set(df.columns)
    if missing:
        raise ValueError(f"Tracks CSV missing columns: {missing}")

    track_ids = sorted(df["track_id"].unique())
    print(f"  ✔  {len(track_ids)} unique tracks  |  sampling ≤{n_samples} frames each")

    # ── Step 1: extract per-track colour vectors ─────────────────
    all_colours:     list[np.ndarray]       = []
    colour_map:      dict[int, list]        = {}

    for tid in tqdm(track_ids, desc="Jersey extraction"):
        rows    = df[df["track_id"] == tid]
        colours = _sample_colours(rows, frames_dir, n_samples)
        colour_map[int(tid)] = colours
        all_colours.extend(colours)

    if len(all_colours) < N_TEAMS:
        raise RuntimeError("Too few valid colour samples.")

    # ── Step 2: global K-Means → 3 jersey clusters ───────────────
    X  = np.stack(all_colours)
    print(f"\n  Clustering {len(X)} samples into {N_TEAMS} jersey groups …")
    km = KMeans(n_clusters=N_TEAMS, n_init=15, max_iter=300, random_state=42)
    km.fit(X)

    # ── Step 3: majority-vote per track ──────────────────────────
    records = []
    for tid in track_ids:
        colours = colour_map[int(tid)]
        if not colours:
            records.append({"track_id": tid, "team": "unknown",
                            "cluster_id": -1, "confidence": 0.0})
            continue

        preds   = km.predict(np.stack(colours))
        counter = Counter(preds.tolist())
        best_cluster, best_count = counter.most_common(1)[0]
        conf = best_count / len(preds)

        label_map = {0: "home", 1: "away", 2: "other"}
        records.append({
            "track_id":   int(tid),
            "team":       label_map.get(int(best_cluster), "other"),
            "cluster_id": int(best_cluster),
            "confidence": round(conf, 3),
        })

    result_df = pd.DataFrame(records).sort_values("track_id", ignore_index=True)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n  📊 Assignment summary:")
    for team, cnt in result_df["team"].value_counts().items():
        print(f"     {team:<8} : {cnt} track IDs")

    print(f"\n  ⚠️  Cluster labels (home/away/other) are arbitrary.")
    
    # Save directly to the file path provided by orchestrator
    result_df.to_csv(out_path, index=False)
    print(f"\n  💾 Saved → {out_path}")
    print(f"  ✅  Classification complete.")
    print(f"      Next step: formation_detector.py --teams {out_path}\n")
    return result_df


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign player tracks to Home / Away / Other via jersey colour."
    )
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=12)
    
    args = parser.parse_args()
    
    classify_teams(
        tracks_csv=args.tracks, 
        frames_dir=args.frames_dir, 
        out_file_path=args.out, 
        n_samples=args.samples
    )