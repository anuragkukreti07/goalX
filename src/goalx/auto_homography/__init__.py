"""
sharma_2018/
────────────
Implementation of:
  "Automated Top View Registration of Broadcast Football Videos"
  Sharma, Bhat, Gandhi, Jawahar — WACV 2018

This module is a drop-in replacement for goalX's homography_picker.py.
It produces the same homography_data.npz output format so project_tracks.py
and all downstream scripts work unchanged.

Pipeline
────────
  edge_extractor.py       → binary pitch-line edge maps from broadcast frames
  dictionary_generator.py → synthetic (edge_map, H) dictionary via PTZ simulation
  hog_matcher.py          → HOG feature extraction + FAISS nearest-neighbour search
  mrf_smoother.py         → MRF temporal smoothing + convex camera stabilization
  player_masker.py        → Phase 2 novelty: mask player regions before extraction
  evaluate_iou.py         → mean/median IOU evaluation against ground truth
  auto_homography.py      → full pipeline orchestrator (CLI entry-point)
"""
