# sharma_2018

Implementation of **"Automated Top View Registration of Broadcast Football Videos"**  
Sharma, Bhat, Gandhi, Jawahar — WACV 2018

This module is a **drop-in replacement** for `homography_picker.py`.  
It produces the **same `homography_data.npz` output format** — every downstream goalX script (`project_tracks.py`, `smooth_tracks.py`, `visualise_tracks.py`) works with zero changes.

---

## Thesis Structure This Implements

```
PHASE 1 — Replicating the 2018 Baseline
  → Milestone 1: ~88% mean IOU on SNMOT-193 (top-down broadcast)
  → Milestone 2: ~45% mean IOU on SNMOT-116 (corner camera) ← prove it fails

PHASE 2 — The Novelty (player masking)
  → Milestone 3: >75% mean IOU on SNMOT-116 with player masking

PHASE 3 — Grand Unification (already in goalX)
  → auto H from Phase 2 feeds directly into project_tracks.py onward
```

---

## Files

| File | Role | Paper Section |
|------|------|---------------|
| `dictionary_generator.py` | PTZ simulation → synthetic (edge_map, H) dictionary | §3.1 |
| `edge_extractor.py` | Broadcast frame → binary pitch-line edge map (pix2pix / classical) | §4.1.1 |
| `hog_matcher.py` | HOG features + FAISS nearest-neighbour search | §3.2.2 |
| `mrf_smoother.py` | MRF temporal smoothing + convex camera stabilization | §3.3 |
| `evaluate_iou.py` | Mean IOU metric computation + histogram | §4.1.2 |
| `auto_homography.py` | Full pipeline orchestrator — replaces `homography_picker.py` | All |

---

## Quick Start

### Prerequisites

```bash
pip install opencv-python numpy pandas tqdm faiss-cpu scipy matplotlib
# For Phase 2 pix2pix mode (optional):
pip install torch
```

### Step 0 — You already have (from goalX Phase 1)
```
data/pitch_map.png               ← from draw_pitch.py
data/homography_data.npz         ← from homography_picker.py (1-2 seed clicks needed)
data/SNMOT-193/img1/*.jpg        ← your frame sequence
data/tracking.csv                ← from track_players.py (needed for Phase 2 only)
```

---

### Step 1 — Build the Synthetic Dictionary

Takes your seed H (from a few manual clicks) and generates thousands of synthetic views via Pan/Tilt/Zoom simulation.

```bash
python -m goalx.sharma_2018.dictionary_generator \
    --pitch   data/pitch_map.png \
    --seeds   data/homography_data.npz \
    --frame-w 1280  --frame-h 720 \
    --n-pan   10  --n-tilt 10  --n-zoom 10 \
    --out-dir data/sharma_dict/
```

Output: `data/sharma_dict/dictionary.npz` (~1000 entries per seed, ~10K for 10 seeds)

> **Tip for thesis**: Use 5–10 seed H files from different time points in the same sequence. More seeds = better coverage = higher IOU.

---

### Step 2 — Extract Edge Maps

```bash
# Classical mode (no GPU, no model — good enough for Phase 1 baseline):
python -m goalx.sharma_2018.edge_extractor \
    --seq     data/SNMOT-193/img1/ \
    --out-dir outputs/edge_maps/

# Pix2Pix mode (if you have the model weights):
python -m goalx.sharma_2018.edge_extractor \
    --seq     data/SNMOT-193/img1/ \
    --model   models/pix2pix_football.pth \
    --out-dir outputs/edge_maps/

# Phase 2 — WITH player masking (the thesis novelty):
python -m goalx.sharma_2018.edge_extractor \
    --seq      data/SNMOT-116/img1/ \
    --tracking data/tracking_116.csv \
    --out-dir  outputs/edge_maps_masked/
```

Output: `outputs/edge_maps/*.png` — one binary edge map per frame.

---

### Step 3 — HOG Matching

```bash
python -m goalx.sharma_2018.hog_matcher \
    --edge-maps  outputs/edge_maps/ \
    --dictionary data/sharma_dict/dictionary.npz \
    --out-dir    outputs/sharma_H/ \
    --k          5
```

Output:
- `outputs/sharma_H/homographies.csv` — per-frame H matrices
- `outputs/sharma_H/homography_data.npz` — **drop-in for project_tracks.py**
- `outputs/sharma_H/faiss_index.bin` — cached index (reuse to skip HOG recomputation)

> **Note on speed**: First run computes HOG for all dictionary entries. Subsequent runs load the cached FAISS index instantly.

---

### Step 4 — MRF Smoothing

```bash
python -m goalx.sharma_2018.mrf_smoother \
    --homographies  outputs/sharma_H/homographies.csv \
    --out-dir       outputs/sharma_H_smooth/
```

Output: `outputs/sharma_H_smooth/homography_data.npz` — temporally smoothed, drop-in ready.

---

### Step 5 — Evaluate IOU

First generate ground truth from manually-labelled frames:

```bash
# Label 20–30 frames with homography_picker.py, save each as:
# data/gt_labels/000001.npz, data/gt_labels/000250.npz, etc.

python -m goalx.sharma_2018.evaluate_iou --make-gt \
    --gt-npz-dir data/gt_labels/ \
    --out        data/gt_homographies.csv

# Then evaluate:
python -m goalx.sharma_2018.evaluate_iou \
    --predicted    outputs/sharma_H_smooth/homographies_smooth.csv \
    --ground-truth data/gt_homographies.csv \
    --pitch        data/pitch_map.png \
    --out-dir      outputs/eval/ \
    --label        "Phase 1 — SNMOT-193"
```

---

### One-Command Pipeline (recommended)

```bash
# Phase 1 — Milestone 1 (SNMOT-193, should get ~88% IOU):
python -m goalx.sharma_2018.auto_homography \
    --mode  phase1 \
    --seq   data/SNMOT-193/img1/ \
    --pitch data/pitch_map.png \
    --seeds data/homography_data_193.npz \
    --out-dir outputs/auto_H_193/

# Phase 1 — Milestone 2 (SNMOT-116, should get ~45% IOU to PROVE FAILURE):
python -m goalx.sharma_2018.auto_homography \
    --mode  phase1 \
    --seq   data/SNMOT-116/img1/ \
    --pitch data/pitch_map.png \
    --seeds data/homography_data_116.npz \
    --out-dir outputs/auto_H_116/

# Phase 2 — Milestone 3 (SNMOT-116 + masking, target >75% IOU):
python -m goalx.sharma_2018.auto_homography \
    --mode     phase2 \
    --seq      data/SNMOT-116/img1/ \
    --pitch    data/pitch_map.png \
    --seeds    data/homography_data_116.npz \
    --tracking data/tracking_116.csv \
    --out-dir  outputs/auto_H_116_masked/

# COMPARE both (generates the thesis Table):
python -m goalx.sharma_2018.auto_homography \
    --mode         compare \
    --seq          data/SNMOT-116/img1/ \
    --pitch        data/pitch_map.png \
    --seeds        data/homography_data_116.npz \
    --tracking     data/tracking_116.csv \
    --ground-truth data/gt_116.csv \
    --out-dir      outputs/comparison_116/
```

---

### Step 6 — Feed Into goalX (unchanged)

```bash
# The auto-generated H replaces the manual one:
python -m goalx.ps1_cv.project_tracks \
    --tracks     data/tracking.csv \
    --homography outputs/auto_H_193/homography_data.npz \   # ← Sharma output
    --pitch      data/pitch_map.png \
    --out-dir    outputs/projected/

# Everything else (smooth_tracks, extract_events, PS2...) runs UNCHANGED.
```

---

## Thesis Results Table

Run all three configurations and fill in this table for your thesis:

| Method | Dataset | Mean IOU | Median IOU |
|--------|---------|----------|------------|
| Manual clicks (`homography_picker.py`) | SNMOT-193 | — | — |
| Sharma HOG (Phase 1) | SNMOT-193 | **~88%** | — |
| Sharma HOG (Phase 1) | SNMOT-116 | **~45%** | — |
| + Player masking (Phase 2) | SNMOT-116 | **>75%?** | — |

The delta between Phase 1 and Phase 2 on SNMOT-116 **is your thesis contribution**.

---

## Key Design Decisions

### Why HOG not Chamfer matching
The paper tests both. HOG beats chamfer by ~6% mean IOU and is 2× faster at test time (shorter descriptor → faster FAISS search). Chamfer requires a distance transform per query frame (O(H×W) per frame); HOG is O(D) where D = descriptor dimension.

### Why FAISS
HOG descriptors are ~125K-dimensional. Standard KD-trees become slower than brute-force above ~20 dimensions. FAISS uses BLAS-optimised matrix multiplication for exact L2 search — 10–100× faster than numpy for large dictionaries.

### Why PTZ simulation, not more manual labels
Manual labelling 100K frames is infeasible. PTZ simulation generates 1000× more views from each manually-labelled seed. This is the paper's core insight: a small manually-labelled set (10–80 frames) + simulation = exhaustive camera coverage.

### Why the ball needs SAHI but edge maps don't
Ball detection (detect_ball.py) uses SAHI because a 15px ball disappears when YOLO downscales to 640px. Edge maps don't need SAHI because pitch *lines* are large-scale structures visible at any resolution — the HOG descriptor is computed on the full edge map at once.

### Why player masking is novel
The paper's pix2pix preprocessing removes players but imperfectly — players standing on landmark intersections corrupt exactly the edge pixel you need. By masking detected player regions to black BEFORE pix2pix/classical extraction, pix2pix sees only unoccluded pitch and cannot hallucinate false lines. No prior work uses player detection bounding boxes as a preprocessing mask for homography estimation.

---

## Dependencies

```
opencv-python    >= 4.8    # HOGDescriptor, perspectiveTransform
numpy            >= 1.24
pandas           >= 2.0
tqdm
faiss-cpu        >= 1.7    # optional but strongly recommended
scipy            >= 1.11   # for Stage 2 convex stabilization
matplotlib       >= 3.7    # for IOU histogram plots
torch            >= 2.0    # optional, only for pix2pix Mode A
```

---

## Data Requirements

| Sequence | Camera | Purpose | Expected IOU |
|----------|--------|---------|-------------|
| SNMOT-193 | Top-down broadcast | Phase 1 baseline | ~88% |
| SNMOT-116 | Corner / end-zone | Failure case + Phase 2 | 45% → >75% |

Both are from the SportsMOT benchmark. Download from: https://sportsmot.github.io
