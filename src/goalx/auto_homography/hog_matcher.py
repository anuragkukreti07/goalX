"""
hog_matcher.py
──────────────
HOG feature extraction + FAISS nearest-neighbour search over the synthetic
dictionary.  This is the core matching stage of Sharma et al. §3.2.2.

WHY HOG OVER CHAMFER MATCHING
──────────────────────────────
The paper evaluates both chamfer matching (§3.2.1) and HOG matching (§3.2.2).
HOG outperforms chamfer by ~6% mean IOU and is significantly faster at test
time — chamfer requires computing a distance transform per query frame, while
HOG reduces search to a single nearest-neighbour lookup in a pre-built FAISS
index.  We implement HOG as the primary method.

HOG PARAMETERS (following paper)
──────────────────────────────────
  Cell size    : 8 × 8 pixels
  Block size   : 2 × 2 cells (16 × 16 px, normalised)
  Orientations : 9 bins
  Input size   : DICT_W × DICT_H = 640 × 360 px

At 640×360 with 8px cells → 80×45 cells → 79×44 blocks (stride 1 cell)
→ 79×44×4×9 = 125,136 dimensional descriptor.

FAISS INDEX SELECTION
──────────────────────
  N < 50K    : IndexFlatL2 (exact, fast enough)
  N ≥ 50K    : IndexIVFFlat (approximate, trained, much faster for 100K+)
  Fallback   : Pure numpy L2 if FAISS not installed (slower but always works)

WHY FAISS NOT SKLEARN KDTree
─────────────────────────────
HOG vectors are ~125K-dimensional.  KDTree becomes slower than brute-force
above ~20 dimensions.  FAISS uses BLAS-optimised matrix multiplication for
exact L2 search and inverted-file structures for approximate search — both
are orders of magnitude faster for high-dimensional vectors.

DUAL MATCHING (Phase 2 domain-gap fix)
───────────────────────────────────────
Phase 2 query frames have player bounding boxes blacked out before edge
extraction.  This creates HOG descriptors that differ from the clean
dictionary entries in the cells that cover blacked-out regions.

To handle this, match_frame() runs two FAISS queries per frame:
  1. Original edge map  (matches clean dictionary entries)
  2. Slightly eroded edge map near zero regions  (matches augmented entries)

The result with the lower L2 distance wins.  This costs essentially nothing
since the FAISS index is already in RAM (two matrix-multiply calls).

OUTPUT FORMAT
──────────────
Per-frame homography CSV:
    frame_id, h00..h22 (9 values), match_distance, match_rank

ALSO writes homography_data.npz for the FIRST frame — this is the
drop-in format for project_tracks.py so the full goalX pipeline can
consume Sharma's output with ZERO changes downstream.

CLI
───
  python -m goalx.sharma_2018.hog_matcher \\
      --edge-maps  outputs/edge_maps/ \\
      --dictionary data/sharma_dict/dictionary.npz \\
      --out-dir    outputs/sharma_H/ \\
      --k          5

  # Outputs:
  #   outputs/sharma_H/homographies.csv
  #   outputs/sharma_H/homography_data.npz  ← drop-in for project_tracks.py
  #   outputs/sharma_H/faiss_index.bin      ← cached index for re-use
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
#  Optional FAISS import
# ─────────────────────────────────────────────────────────────────

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
#  Constants (must match dictionary_generator.py)
# ─────────────────────────────────────────────────────────────────

DICT_W: int = 320
DICT_H: int = 176

# HOG descriptor parameters — exactly as reported in the paper
HOG_CELL_SIZE   = (8, 8)
HOG_BLOCK_SIZE  = (2, 2)   # in cells
HOG_BLOCK_STRIDE= (1, 1)   # in cells
HOG_N_BINS      = 9


# ─────────────────────────────────────────────────────────────────
#  HOG descriptor computation
# ─────────────────────────────────────────────────────────────────

def _build_hog_descriptor() -> cv2.HOGDescriptor:
    """
    Build an OpenCV HOGDescriptor with paper-specified parameters.

    WHY cv2.HOGDescriptor OVER skimage.hog:
    OpenCV's HOGDescriptor is ~10× faster on CPU and produces an identical
    descriptor — we are computing this for 10K–100K dictionary entries plus
    every query frame, so speed matters.
    """
    cell_px  = HOG_CELL_SIZE[0]      # 8
    block_px = cell_px * HOG_BLOCK_SIZE[0]   # 16

    descriptor = cv2.HOGDescriptor(
        _winSize    = (DICT_W, DICT_H),
        _blockSize  = (block_px, block_px),
        _blockStride= (cell_px * HOG_BLOCK_STRIDE[0],
                       cell_px * HOG_BLOCK_STRIDE[1]),
        _cellSize   = (cell_px, cell_px),
        _nbins      = HOG_N_BINS,
    )
    return descriptor


_HOG = _build_hog_descriptor()


def compute_hog(edge_map: np.ndarray) -> np.ndarray:
    """
    Compute HOG descriptor for a single binary edge map.

    Input  : (H, W) uint8 image at DICT_H × DICT_W resolution.
    Returns: (D,) float32 HOG feature vector.

    The edge map is converted to uint8 if needed and must already be at
    DICT_H × DICT_W — caller is responsible for resizing.
    """
    if edge_map.shape != (DICT_H, DICT_W):
        edge_map = cv2.resize(edge_map, (DICT_W, DICT_H),
                              interpolation=cv2.INTER_NEAREST)
    if edge_map.dtype != np.uint8:
        edge_map = edge_map.astype(np.uint8)

    descriptor = _HOG.compute(edge_map)
    return descriptor.flatten().astype(np.float32)


# (dual-match helper removed — augmented dictionary handles domain gap directly)


# ─────────────────────────────────────────────────────────────────
#  FAISS index management
# ─────────────────────────────────────────────────────────────────

class FAISSIndex:
    """
    Wraps a FAISS (or numpy fallback) L2 index over HOG vectors.

    Build once, reuse across all query frames.
    """

    def __init__(self, dim: int, n_entries: int, use_ivf: bool = False,
                 n_lists: int = 256):
        self.dim   = dim
        self._idx  = None
        self._vecs = None   # numpy fallback storage

        if _FAISS_AVAILABLE:
            if use_ivf and n_entries >= 50_000:
                quantizer  = faiss.IndexFlatL2(dim)
                self._idx  = faiss.IndexIVFFlat(quantizer, dim, n_lists,
                                                faiss.METRIC_L2)
                self._needs_train = True
            else:
                self._idx         = faiss.IndexFlatL2(dim)
                self._needs_train = False
        else:
            self._vecs = np.empty((0, dim), dtype=np.float32)

    def build(self, vectors: np.ndarray) -> None:
        """Add all vectors to the index."""
        if _FAISS_AVAILABLE:
            if self._needs_train:
                self._idx.train(vectors)
            self._idx.add(vectors)
        else:
            self._vecs = vectors.copy()

    def search(self, query: np.ndarray, k: int = 1):
        """
        Find k nearest neighbours to query vector.
        Returns (distances, indices) each shape (k,).
        """
        q = query.reshape(1, -1).astype(np.float32)
        if _FAISS_AVAILABLE:
            if hasattr(self._idx, 'nprobe'):
                self._idx.nprobe = min(64, self._idx.ntotal)
            dists, idxs = self._idx.search(q, k)
            return dists[0], idxs[0]
        else:
            # Numpy brute-force L2
            diffs  = self._vecs - q
            dists  = (diffs ** 2).sum(axis=1)
            idxs   = np.argsort(dists)[:k]
            return dists[idxs], idxs

    def save(self, path: Path) -> None:
        if _FAISS_AVAILABLE:
            faiss.write_index(self._idx, str(path))

    @classmethod
    def load(cls, path: Path, dim: int) -> "FAISSIndex":
        obj = cls.__new__(cls)
        obj.dim   = dim
        obj._vecs = None
        if _FAISS_AVAILABLE:
            obj._idx = faiss.read_index(str(path))
        else:
            raise FileNotFoundError("FAISS not installed — cannot load index file.")
        return obj


# ─────────────────────────────────────────────────────────────────
#  Dictionary loading and index building
# ─────────────────────────────────────────────────────────────────

def build_index(dictionary_path: Path,
                index_cache_path: Path | None = None) -> tuple:
    """
    Load the dictionary and build (or load cached) FAISS index.

    Returns (faiss_index, homographies_array, frame_w, frame_h)
    """
    print(f"  Loading dictionary: {dictionary_path}")
    data          = np.load(str(dictionary_path))
    edge_maps     = data["edge_maps"]       # (N, DICT_H, DICT_W)
    homographies  = data["homographies"]    # (N, 3, 3)
    frame_w       = int(data["frame_w"])
    frame_h       = int(data["frame_h"])
    N             = len(edge_maps)

    print(f"  Dictionary: {N:,} entries  |  frame: {frame_w}×{frame_h}")

    # Attempt to load cached FAISS index
    if index_cache_path is not None and index_cache_path.exists() and _FAISS_AVAILABLE:
        print(f"  Loading cached FAISS index: {index_cache_path}")
        dim   = compute_hog(edge_maps[0]).shape[0]
        idx   = FAISSIndex.load(index_cache_path, dim)
        print(f"  ✔  FAISS index loaded  ({idx._idx.ntotal:,} vectors, dim={dim:,})")
        return idx, homographies, frame_w, frame_h

    # Compute HOG for all dictionary entries
    print(f"  Computing HOG descriptors for {N:,} dictionary entries …")
    hog_vecs = []
    for i in tqdm(range(N), desc="  HOG", unit="entry"):
        hog_vecs.append(compute_hog(edge_maps[i]))
    hog_matrix = np.stack(hog_vecs, axis=0).astype(np.float32)

    dim = hog_matrix.shape[1]
    print(f"  HOG descriptor dimension: {dim:,}")
    if not _FAISS_AVAILABLE:
        print(f"  ⚠  FAISS not installed — using numpy brute-force (install with "
              f"'pip install faiss-cpu' for 10-100× speed improvement)")

    idx = FAISSIndex(dim, N, use_ivf=(N >= 50_000))
    idx.build(hog_matrix)

    if index_cache_path is not None and _FAISS_AVAILABLE:
        index_cache_path.parent.mkdir(parents=True, exist_ok=True)
        idx.save(index_cache_path)
        print(f"  FAISS index cached → {index_cache_path}")

    return idx, homographies, frame_w, frame_h


# ─────────────────────────────────────────────────────────────────
#  Per-frame matching
# ─────────────────────────────────────────────────────────────────

def match_frame(edge_map_path: Path,
                faiss_index: FAISSIndex,
                homographies: np.ndarray,
                k: int = 5) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Match a single query edge map against the dictionary.

    The dictionary now contains both clean and augmented (player-occluded)
    versions of every entry (see dictionary_generator._augment_occlusions).
    A single FAISS query on the original masked edge map is sufficient —
    FAISS will naturally find the augmented entry whose occlusion pattern
    is closest to the real player masks, returning the correct H.

    WHY NO ERODING
    ──────────────
    An earlier version eroded the query around zero regions to make it look
    more like the augmented dictionary entries.  This caused a bug: the
    top-35% crop (already all-black) got dilated across the whole image,
    turning every frame's HOG descriptor into a near-zero vector that matched
    the same dictionary entry regardless of camera angle.
    Eroding is unnecessary — the augmented dictionary already bridges the gap.

    Returns
    ───────
    H_best       : (3, 3) float32 homography of best match
    best_dist    : float, L2 distance of best match (lower = better)
    top_k_H      : (k, 3, 3) top-k homographies for MRF smoothing
    """
    edge_map = cv2.imread(str(edge_map_path), cv2.IMREAD_GRAYSCALE)
    if edge_map is None:
        return np.eye(3, dtype=np.float32), float("inf"), np.eye(3).reshape(1, 3, 3)

    hog_vec = compute_hog(edge_map)
    dists, idxs = faiss_index.search(hog_vec, k=k)

    valid = idxs[idxs >= 0]
    if len(valid) == 0:
        return np.eye(3, dtype=np.float32), float("inf"), np.eye(3).reshape(1, 3, 3)

    best_idx  = valid[0]
    best_dist = float(dists[0])
    H_best    = homographies[best_idx]
    top_k_H   = homographies[valid]

    return H_best, best_dist, top_k_H


# ─────────────────────────────────────────────────────────────────
#  Main matching pipeline
# ─────────────────────────────────────────────────────────────────

def run_matching(
    edge_maps_dir:   Path,
    dictionary_path: Path,
    out_dir:         Path,
    k:               int  = 5,
    index_cache:     Path | None = None,
) -> Path:
    """
    Match all edge maps in edge_maps_dir against the dictionary.

    Outputs
    ───────
    homographies.csv        — per-frame H matrix (flat 9 values + distance)
    homography_data.npz     — drop-in format for project_tracks.py
    match_distances.npy     — raw match distances (for confidence analysis)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX / Sharma 2018 — HOG Matcher")
    print(f"  {'─' * 40}")
    if not _FAISS_AVAILABLE:
        print(f"  ⚠  Install faiss-cpu for 10-100× faster search")

    faiss_index, homographies, frame_w, frame_h = build_index(
        dictionary_path,
        index_cache_path=index_cache or (out_dir / "faiss_index.bin"),
    )

    edge_maps = sorted(edge_maps_dir.glob("*.png"))
    if not edge_maps:
        raise FileNotFoundError(f"No .png edge maps in {edge_maps_dir}")

    print(f"\n  Matching {len(edge_maps)} query frames …")

    rows:      list[dict]            = []
    distances: list[float]           = []
    all_top_k: dict[int, np.ndarray] = {}   # for MRF smoother

    for em_path in tqdm(edge_maps, desc="  Matching", unit="frame"):
        frame_id = int(em_path.stem)

        H_best, dist, top_k_H = match_frame(
            em_path, faiss_index, homographies, k=k
        )

        rows.append({
            "frame_id": frame_id,
            "h00": H_best[0, 0], "h01": H_best[0, 1], "h02": H_best[0, 2],
            "h10": H_best[1, 0], "h11": H_best[1, 1], "h12": H_best[1, 2],
            "h20": H_best[2, 0], "h21": H_best[2, 1], "h22": H_best[2, 2],
            "match_distance": dist,
        })
        distances.append(dist)
        all_top_k[frame_id] = top_k_H

    df = pd.DataFrame(rows)
    df.sort_values("frame_id", inplace=True, ignore_index=True)

    csv_out = out_dir / "homographies.csv"
    df.to_csv(str(csv_out), index=False)

    dists_arr = np.array(distances, dtype=np.float32)
    np.save(str(out_dir / "match_distances.npy"), dists_arr)

    # ── Write drop-in homography_data.npz for the FIRST frame ────
    # project_tracks.py expects a single H + status + frame_pts + pitch_pts.
    # We write the median-ranked frame's H as the representative H so
    # project_tracks.py can be run for a quick visual sanity check.
    median_frame_row = df.iloc[len(df) // 2]
    H_rep = np.array([
        [median_frame_row["h00"], median_frame_row["h01"], median_frame_row["h02"]],
        [median_frame_row["h10"], median_frame_row["h11"], median_frame_row["h12"]],
        [median_frame_row["h20"], median_frame_row["h21"], median_frame_row["h22"]],
    ], dtype=np.float32)

    np.savez(
        str(out_dir / "homography_data.npz"),
        H          = H_rep,
        frame_pts  = np.zeros((4, 2), dtype=np.float32),   # placeholder
        pitch_pts  = np.zeros((4, 2), dtype=np.float32),
        status     = np.ones((4, 1),  dtype=np.uint8),
        method     = np.array(["sharma_hog"]),
    )

    # Summary stats
    n        = len(dists_arr)
    pct_high = float((dists_arr > np.percentile(dists_arr, 90)).mean() * 100)
    print(f"\n  ✅  Matching complete")
    print(f"      Frames matched     : {n:,}")
    print(f"      Dist  mean±std     : {dists_arr.mean():.1f} ± {dists_arr.std():.1f}")
    print(f"      Dist  median       : {np.median(dists_arr):.1f}")
    print(f"      High-dist frames   : {pct_high:.1f}% (potential bad matches)")
    print(f"\n  Saved:")
    print(f"      {csv_out}")
    print(f"      {out_dir / 'homography_data.npz'}  ← drop-in for project_tracks.py")
    print(f"      {out_dir / 'match_distances.npy'}")
    print(f"\n  Next steps:")
    print(f"    mrf_smoother.py  --homographies {csv_out}  (temporal smoothing)")
    print(f"    evaluate_iou.py  --predicted   {csv_out}   (measure accuracy)\n")

    return out_dir


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="HOG nearest-neighbour homography matching (Sharma §3.2.2)."
    )
    p.add_argument("--edge-maps",   required=True,
                   help="Directory of binary edge map .png files "
                        "(from edge_extractor.py)")
    p.add_argument("--dictionary",  required=True,
                   help="dictionary.npz from dictionary_generator.py")
    p.add_argument("--out-dir",     default="outputs/sharma_H")
    p.add_argument("--k",           type=int, default=5,
                   help="Number of top-k matches to retain for MRF smoothing")
    p.add_argument("--index-cache", default=None,
                   help="Path to save/load FAISS index. Avoids recomputing "
                        "HOG descriptors on dictionary re-use.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_matching(
        edge_maps_dir   = Path(args.edge_maps),
        dictionary_path = Path(args.dictionary),
        out_dir         = Path(args.out_dir),
        k               = args.k,
        index_cache     = Path(args.index_cache) if args.index_cache else None,
    )