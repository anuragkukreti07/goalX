"""
dictionary_generator.py
───────────────────────
Semi-supervised synthetic dictionary generation following Sharma et al. §3.1.

WHY THIS MODULE EXISTS
──────────────────────
Manually labelling H for 100K frames is infeasible.  Instead, the paper
seeds from a small set of manually-annotated frames (we reuse the outputs of
homography_picker.py) and applies Pan / Tilt / Zoom (PTZ) simulation to each
seed to generate thousands of synthetic (edge_map, H) pairs.

The dictionary entry for a given H is:
    edge_map = warpPerspective(pitch_line_binary, inv(H), frame_size)

That is: apply H^{-1} to the static pitch model (white lines on black) to
synthesise what the pitch lines would look like from that camera viewpoint.
This synthetic edge map + the known H form one dictionary entry.

PTZ SIMULATION (following Figure 3 of the paper)
─────────────────────────────────────────────────
Given a seed H, the 4 image-space corners p0..p3 of the frame map to 4
pitch-space points q0..q3.

  Pan  : rotate q0..q3 around the vanishing point (intersection of lines
         q0q3 and q1q2).  Simulates camera rotating left/right.

  Tilt : translate q0q3 and q1q2 along their respective perpendicular
         directions.  Simulates camera tilting up/down.

  Zoom : scale the quadrilateral q0..q3 about its centroid.
         Zoom-out expands, zoom-in shrinks.

Each new set of q' corners + original p corners gives a new H' via
cv2.findHomography, which is added to the dictionary.

CLI
───
  python -m goalx.sharma_2018.dictionary_generator \\
      --pitch   data/pitch_map.png \\
      --seeds   data/homography_data.npz \\
      --frame-w 1280  --frame-h 720 \\
      --n-pan   10  --n-tilt 10  --n-zoom 10 \\
      --pan-range  20  --tilt-range 15  --zoom-range 0.35 \\
      --out-dir data/sharma_dict/

Output
──────
  data/sharma_dict/dictionary.npz
      edge_maps   : (N, DICT_H, DICT_W) uint8   — resized binary edge maps
      homographies: (N, 3, 3)            float32 — corresponding H matrices
      source_ids  : (N,)                 int32   — which seed produced this entry
"""

from __future__ import annotations

import argparse
import itertools
import random as _random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

# HOG is computed on edge maps resized to this fixed resolution.
# Must match hog_matcher.py.  Chosen to balance feature resolution
# against descriptor size (640×360 → 14-cell × 8-cell × 9-bin × 4-block ≈ 28K dims).
DICT_W: int = 320
DICT_H: int = 176

# Pitch canvas dimensions — must match draw_pitch.py
PITCH_W: int = 1050
PITCH_H: int = 680


# ─────────────────────────────────────────────────────────────────
#  Synthetic occlusion augmentation  (Phase 2 domain-gap fix)
# ─────────────────────────────────────────────────────────────────

def _augment_occlusions(
    edge_map: np.ndarray,
    n_min: int = 2,
    n_max: int = 5,
    rng: _random.Random | None = None,
) -> np.ndarray:
    """
    Black out random player-shaped rectangles on a synthetic edge map.

    WHY THIS EXISTS
    ───────────────
    Phase 2 query frames have player bounding boxes blacked out by the YOLO
    tracker before edge extraction.  This creates *gaps* in pitch lines that
    are invisible to the unmodified dictionary (which contains only clean,
    complete line renders).

    When FAISS compares a "holey" query HOG descriptor against clean
    dictionary descriptors the L2 distance is systematically inflated even
    for geometrically correct matches — because the cells that cover blacked-
    out regions contribute zero gradient in the query but non-zero gradient
    in the dictionary.  The net effect is that FAISS returns a confident
    but geometrically wrong match from a different camera angle whose HOG
    pattern happens to be numerically close to the holey query.

    FIX: augment every dictionary entry with 2-5 random player-sized black
    rectangles.  Now the dictionary contains both fully-visible and
    partially-occluded line patterns.  HOG matching in Phase 2 sees similar
    occlusion patterns in the dictionary and picks the geometrically correct
    entry rather than an accidentally similar one.

    At 320×180 a typical player bbox projects to ~20-40 px wide, 25-55 px tall.
    """
    if rng is None:
        rng = _random.Random()
    out = edge_map.copy()
    h, w = out.shape[:2]
    n = rng.randint(n_min, n_max)
    for _ in range(n):
        bw = rng.randint(16, 40)
        bh = rng.randint(25, 55)
        x  = rng.randint(0, max(0, w - bw))
        y  = rng.randint(0, max(0, h - bh))
        out[y:y + bh, x:x + bw] = 0
    return out


# ─────────────────────────────────────────────────────────────────
#  Pitch line binary image
# ─────────────────────────────────────────────────────────────────

def load_pitch_lines(pitch_path: Path) -> np.ndarray:
    """
    Convert the colour pitch map from draw_pitch.py into a binary edge-map
    containing ONLY the white line markings (no green fill).

    WHY: The dictionary edge maps must contain only pitch lines — no grass
    texture, no colour — so they match the format of Pix2Pix-extracted query
    edge maps.  We threshold the pitch PNG to keep only near-white pixels.
    """
    pitch = cv2.imread(str(pitch_path))
    if pitch is None:
        raise FileNotFoundError(f"Cannot read pitch map: {pitch_path}")

    # White lines have high value in all channels.  Threshold at 200/255.
    gray = cv2.cvtColor(pitch, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Dilate slightly to make lines easier to match (same dilation applied
    # to query edge maps in edge_extractor.py)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)

    return binary  # shape (PITCH_H, PITCH_W), uint8, 0 or 255


# ─────────────────────────────────────────────────────────────────
#  Corner / intersection geometry helpers
# ─────────────────────────────────────────────────────────────────

def _line_intersection(p1: np.ndarray, p2: np.ndarray,
                       p3: np.ndarray, p4: np.ndarray) -> np.ndarray | None:
    """
    Compute intersection of line (p1,p2) and line (p3,p4).
    Returns None if lines are parallel.
    Used to find the vanishing point for pan simulation.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-8:
        return None  # parallel
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def _rotate_quad_around_point(quad: np.ndarray,
                               pivot: np.ndarray,
                               angle_deg: float) -> np.ndarray:
    """
    Rotate a 4-point quadrilateral around a pivot point by angle_deg degrees.
    Used to simulate camera pan.
    """
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    centered = quad - pivot
    rotated  = (R @ centered.T).T
    return rotated + pivot


def _frame_corners(frame_w: int, frame_h: int) -> np.ndarray:
    """
    Return the 4 corners of the broadcast frame in image space.
    Order: top-left, top-right, bottom-right, bottom-left.
    This matches the convention used in homography_picker.py.
    """
    return np.array([
        [0,       0],
        [frame_w, 0],
        [frame_w, frame_h],
        [0,       frame_h],
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
#  PTZ simulation
# ─────────────────────────────────────────────────────────────────

def simulate_ptz(
    H_seed: np.ndarray,
    frame_w: int,
    frame_h: int,
    n_pan:   int   = 10,
    n_tilt:  int   = 10,
    n_zoom:  int   = 10,
    pan_range:  float = 20.0,   # degrees
    tilt_range: float = 15.0,   # degrees (applied as fractional shift)
    zoom_range: float = 0.35,   # fraction (0.35 = ±35% size change)
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """
    Generate synthetic H matrices from a single seed H via PTZ simulation.

    Returns a list of (3,3) float32 H matrices — one per (pan, tilt, zoom)
    combination that produces a valid homography with RANSAC inlier count >= 4.

    WHY ALL COMBINATIONS: The paper generates permutations of pan × tilt × zoom
    to exhaustively cover the camera parameter space around each seed viewpoint.
    Pure random sampling misses systematic coverage; a grid covers all extremes.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    p_corners = _frame_corners(frame_w, frame_h)

    # Project frame corners to pitch space using seed H
    q_corners = cv2.perspectiveTransform(
        p_corners.reshape(-1, 1, 2), H_seed
    ).reshape(-1, 2)

    # Compute centroid of pitch-space quad (used for zoom)
    centroid = q_corners.mean(axis=0)

    # Vanishing point for pan: intersection of q0q3 and q1q2
    vp = _line_intersection(q_corners[0], q_corners[3],
                             q_corners[1], q_corners[2])
    if vp is None:
        vp = centroid  # fallback: rotate around centroid

    # Generate parameter grids
    pan_angles  = np.linspace(-pan_range,  pan_range,  n_pan)
    tilt_shifts = np.linspace(-tilt_range, tilt_range, n_tilt)
    zoom_scales = np.linspace(1.0 - zoom_range, 1.0 + zoom_range, n_zoom)

    results: list[np.ndarray] = []

    for pan, tilt, zoom in itertools.product(pan_angles, tilt_shifts, zoom_scales):
        q = q_corners.copy()

        # ── Pan: rotate quad around vanishing point ──────────────
        if abs(pan) > 0.5:
            q = _rotate_quad_around_point(q, vp, pan)

        # ── Tilt: shift top edge and bottom edge independently ───
        # Direction perpendicular to the baseline (q0→q1 direction)
        top_dir = q[1] - q[0]
        top_len = np.linalg.norm(top_dir)
        if top_len > 1e-6:
            top_perp = np.array([-top_dir[1], top_dir[0]]) / top_len
            shift_px = tilt * (PITCH_H * 0.1)   # scale tilt to pitch pixels
            q[0] += top_perp * shift_px * 0.5
            q[1] += top_perp * shift_px * 0.5
            q[2] -= top_perp * shift_px * 0.5
            q[3] -= top_perp * shift_px * 0.5

        # ── Zoom: scale quad about centroid ──────────────────────
        if abs(zoom - 1.0) > 0.01:
            q = centroid + (q - centroid) * zoom

        # ── Compute new H from modified quad ─────────────────────
        H_new, status = cv2.findHomography(p_corners, q, cv2.RANSAC, 5.0)
        if H_new is None:
            continue
        if status is not None and int(status.sum()) < 4:
            continue

        # Sanity: condition number guard (must be < 1e6)
        cond = float(np.linalg.cond(H_new))
        if cond > 1e6:
            continue

        results.append(H_new.astype(np.float32))

    return results


# ─────────────────────────────────────────────────────────────────
#  Edge map synthesis
# ─────────────────────────────────────────────────────────────────

def synthesise_edge_map(
    H: np.ndarray,
    pitch_lines: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """
    Generate the synthetic edge map for a given H by applying H^{-1} to the
    pitch line binary image.

    Formally:  edge_map = warpPerspective(pitch_lines, H^{-1}, (frame_w, frame_h))

    This gives the appearance of the pitch lines as seen from the camera
    viewpoint described by H.  The result is then resized to DICT_H × DICT_W
    for storage and HOG computation.

    WHY H^{-1}: H maps image → pitch.  We want pitch → image, which is H^{-1}.
    """
    H_inv = np.linalg.inv(H)
    synth = cv2.warpPerspective(
        pitch_lines, H_inv, (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    # Re-binarise (interpolation may create grey fringing)
    _, synth = cv2.threshold(synth, 50, 255, cv2.THRESH_BINARY)
    synth[0:int(frame_h * 0.35), :] = 0
    # Resize to  fixed dictionary resolution
    synth_resized = cv2.resize(synth, (DICT_W, DICT_H), interpolation=cv2.INTER_NEAREST)
    return synth_resized


# ─────────────────────────────────────────────────────────────────
#  Main build function
# ─────────────────────────────────────────────────────────────────

def build_dictionary(
    pitch_path:  Path,
    seeds_paths: list[Path],
    frame_w:     int,
    frame_h:     int,
    n_pan:       int   = 10,
    n_tilt:      int   = 10,
    n_zoom:      int   = 10,
    pan_range:   float = 20.0,
    tilt_range:  float = 15.0,
    zoom_range:  float = 0.35,
    out_dir:     Path  = Path("data/sharma_dict"),
    seed_also:   bool  = True,
) -> Path:
    """
    Build the synthetic dictionary from one or more seed .npz files and
    the pitch_map.png.

    Parameters
    ──────────
    seeds_paths : List of .npz files from homography_picker.py or
                  calibrate_full_pitch.py.  Each must contain key 'H'.
    seed_also   : If True, include the seed H itself (not just its PTZ
                  children) as a dictionary entry.  Recommended.

    Domain-gap fix
    ──────────────
    Each synthesised edge map is stored TWICE:
      1. Clean (no occlusions) — matches Phase 1 unmasked query frames.
      2. Augmented (random player-shaped black rectangles) — matches Phase 2
         masked query frames where player bounding boxes are blacked out.

    This ensures Phase 2 HOG matching finds the geometrically correct
    dictionary entry instead of an accidentally similar one from a different
    camera angle.

    Returns
    ───────
    Path to the saved dictionary.npz.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX / Sharma 2018 — Dictionary Generator")
    print(f"  {'─' * 44}")
    print(f"  Pitch       : {pitch_path}")
    print(f"  Seeds       : {[str(s) for s in seeds_paths]}")
    print(f"  Frame size  : {frame_w}×{frame_h}")
    print(f"  PTZ grid    : pan={n_pan} × tilt={n_tilt} × zoom={n_zoom} "
          f"= {n_pan * n_tilt * n_zoom} candidates per seed")
    print(f"  Augmentation: each entry stored clean + occluded (2× entries)")

    pitch_lines = load_pitch_lines(pitch_path)
    rng_np  = np.random.default_rng(2024)
    rng_aug = _random.Random(2024)   # separate RNG for augmentation

    all_edge_maps:    list[np.ndarray] = []
    all_homographies: list[np.ndarray] = []
    all_source_ids:   list[int]        = []

    for seed_idx, seed_path in enumerate(seeds_paths):
        data   = np.load(str(seed_path))
        H_seed = data["H"].astype(np.float32)

        print(f"\n  Seed {seed_idx + 1}/{len(seeds_paths)}: {seed_path.name}")

        # Optionally include the seed itself
        if seed_also:
            em = synthesise_edge_map(H_seed, pitch_lines, frame_w, frame_h)
            # Clean version
            all_edge_maps.append(em)
            all_homographies.append(H_seed)
            all_source_ids.append(seed_idx)
            # Augmented version (for Phase 2 domain matching)
            all_edge_maps.append(_augment_occlusions(em, rng=rng_aug))
            all_homographies.append(H_seed)
            all_source_ids.append(seed_idx)

        # PTZ children
        h_list = simulate_ptz(
            H_seed, frame_w, frame_h,
            n_pan=n_pan, n_tilt=n_tilt, n_zoom=n_zoom,
            pan_range=pan_range, tilt_range=tilt_range,
            zoom_range=zoom_range, rng=rng_np,
        )

        print(f"     PTZ generated {len(h_list)} valid H matrices "
              f"(of {n_pan * n_tilt * n_zoom} candidates)")

        for H_new in tqdm(h_list, desc="  Synthesising edge maps", unit="entry"):
            em = synthesise_edge_map(H_new, pitch_lines, frame_w, frame_h)
            # Discard near-empty edge maps (too few line pixels = bad viewpoint)
            if em.sum() / 255 < 100:
                continue
            # Clean version
            all_edge_maps.append(em)
            all_homographies.append(H_new)
            all_source_ids.append(seed_idx)
            # Augmented version (closes Phase 2 domain gap)
            all_edge_maps.append(_augment_occlusions(em, rng=rng_aug))
            all_homographies.append(H_new)
            all_source_ids.append(seed_idx)

    N = len(all_edge_maps)
    if N == 0:
        raise RuntimeError("Dictionary is empty — check seed H quality and PTZ ranges.")

    edge_maps_arr    = np.stack(all_edge_maps,    axis=0).astype(np.uint8)
    homographies_arr = np.stack(all_homographies, axis=0).astype(np.float32)
    source_ids_arr   = np.array(all_source_ids,   dtype=np.int32)

    out_path = out_dir / "dictionary.npz"
    np.savez_compressed(
        str(out_path),
        edge_maps    = edge_maps_arr,
        homographies = homographies_arr,
        source_ids   = source_ids_arr,
        frame_w      = np.int32(frame_w),
        frame_h      = np.int32(frame_h),
        dict_w       = np.int32(DICT_W),
        dict_h       = np.int32(DICT_H),
    )

    print(f"\n  ✅  Dictionary built: {N:,} entries  "
          f"({N//2:,} unique viewpoints × 2 augmentation variants)")
    print(f"      edge_maps shape    : {edge_maps_arr.shape}")
    print(f"      homographies shape : {homographies_arr.shape}")
    print(f"      Saved → {out_path}")
    print(f"\n  Next step: run hog_matcher.py --dictionary {out_path}\n")
    return out_path


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Build synthetic (edge_map, H) dictionary via PTZ simulation."
    )
    p.add_argument("--pitch",       required=True,
                   help="pitch_map.png from draw_pitch.py")
    p.add_argument("--seeds",       nargs="+", required=True,
                   help="One or more homography_data.npz files from "
                        "homography_picker.py (one per manually-labelled seed view)")
    p.add_argument("--frame-w",     type=int, default=1280)
    p.add_argument("--frame-h",     type=int, default=720)
    p.add_argument("--n-pan",       type=int, default=10,
                   help="Number of pan steps (default 10)")
    p.add_argument("--n-tilt",      type=int, default=10)
    p.add_argument("--n-zoom",      type=int, default=10)
    p.add_argument("--pan-range",   type=float, default=20.0,
                   help="Max pan angle in degrees (default ±20°)")
    p.add_argument("--tilt-range",  type=float, default=15.0)
    p.add_argument("--zoom-range",  type=float, default=0.35,
                   help="Fractional zoom range (default ±35%%)")
    p.add_argument("--out-dir",     default="data/sharma_dict")
    p.add_argument("--no-seed",     action="store_true",
                   help="Do not include seed H itself in the dictionary")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dictionary(
        pitch_path   = Path(args.pitch),
        seeds_paths  = [Path(s) for s in args.seeds],
        frame_w      = args.frame_w,
        frame_h      = args.frame_h,
        n_pan        = args.n_pan,
        n_tilt       = args.n_tilt,
        n_zoom       = args.n_zoom,
        pan_range    = args.pan_range,
        tilt_range   = args.tilt_range,
        zoom_range   = args.zoom_range,
        out_dir      = Path(args.out_dir),
        seed_also    = not args.no_seed,
    )