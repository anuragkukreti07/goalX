# goalX

**goalX** is an end-to-end football analytics system that converts raw broadcast match video into structured tactical data, spatial intelligence, and contextual goal metrics.

It combines computer vision, multi-object tracking, geometric reasoning, kinematic analytics, and machine learning to transform raw footage into research-grade football insights — built as a modular M.Tech thesis platform.

---

## System Architecture

```
Raw Match Video
      ↓ PS1 — Computer Vision ──────────────────────────────────────
      ↓
Frame Extraction          extract_frames.py
Object Detection          detect_ball.py          ← YOLOv8s + SAHI slicing
Multi-Object Tracking     track_players.py        ← ByteTrack
Pitch Calibration         homography_picker.py    ← RANSAC homography
                          calibrate_full_pitch.py ← Full-pitch FIFA landmarks
Coordinate Projection     project_tracks.py       ← H matrix, foot-point mapping
Trajectory Smoothing      smooth_tracks.py        ← Rolling mean, gap-aware
Kinematic Analytics       spatial_analytics.py    ← Distance, speed, acceleration
Event Extraction          extract_events.py       ← Shot / possession / pressure
Visualisation             visualise_tracks.py, showcase_radar.py
      ↓ PS2 — Tactical Intelligence ───────────────────────────────
Team Classification       team_classifier.py      ← K-Means jersey colour
Formation Detection       formation_detector.py   ← K-Means depth clustering
Pitch Control             pitch_control.py        ← Voronoi tessellation
Clutch Score              clutch_score.py         ← xG × pressure × control × time
Tactical Radar            tactical_radar.py       ← Per-player + team radar charts
Broadcast Overlay         broadcast_overlay.py    ← Voronoi + trails + HUD
      ↓ PS3 — Model Training ───────────────────────────────────────
xG Model                  ps3_ml/train_xg.py      ← Logistic regression + calibration
Ball Trajectory           ps3_ml/ball_trajectory.py ← Parabolic interpolation
Pass Network              ps3_ml/pass_network.py  ← NetworkX + PageRank
Action Classifier         ps3_ml/action_classifier.py ← Kinematics + optional RF
Match Report              ps3_ml/export_report.py ← Self-contained HTML report
      ↓ PS4 — Research Extensions ───────────────────────────────────
Offside Detection         ps4_research/offside_detector.py ← Geometric Law 11
Momentum Score            ps4_research/momentum_score.py   ← Multi-signal timeline
Player Rating             ps4_research/player_rating.py    ← Composite 0-10 score
Temporal xG               ps4_research/temporal_xg.py      ← LSTM context model
YOLO Fine-tune            ps4_research/fine_tune_yolo.py   ← Domain adaptation
```

---

## Repository Structure

```
goalX/
├── src/goalx/
│   ├── ps1_cv/
│   │   ├── extract_frames.py
│   │   ├── detect_players_full.py    # SAHI + batched detection
│   │   ├── detect_ball.py
│   │   ├── track_players.py          # ByteTrack
│   │   ├── draw_pitch.py             # 1050×680 FIFA canvas
│   │   ├── homography_picker.py      # Interactive RANSAC
│   │   ├── calibrate_full_pitch.py   # 21 FIFA landmark calibrator
│   │   ├── project_tracks.py
│   │   ├── smooth_tracks.py
│   │   ├── spatial_analytics.py
│   │   ├── extract_events.py
│   │   ├── visualise_tracks.py
│   │   ├── visualize_detections.py
│   │   ├── showcase_radar.py
│   │   ├── test_vision.py
│   │   └── bytetrack/
│   │
│   ├── ps2_ml/
│   │   ├── team_classifier.py
│   │   ├── formation_detector.py
│   │   ├── pitch_control.py
│   │   ├── clutch_score.py
│   │   ├── tactical_radar.py
│   │   └── broadcast_overlay.py
│   │
│   ├── ps3_ml/
│   │   ├── train_xg.py
│   │   ├── ball_trajectory.py
│   │   ├── pass_network.py
│   │   ├── action_classifier.py
│   │   └── export_report.py
│   │
│   └── ps4_research/
│       ├── offside_detector.py
│       ├── momentum_score.py
│       ├── player_rating.py
│       ├── temporal_xg.py
│       └── fine_tune_yolo.py
│
├── data/
├── outputs/
├── models/
├── scripts/
├── requirements.txt
├── pyproject.toml
├── README.md
└── GUIDE.md
```

---

## Phase Status

### PS1 ✅ Complete

| Component               | File                      | Status                                 |
| ----------------------- | ------------------------- | -------------------------------------- |
| Frame extraction        | `extract_frames.py`       | ✅                                     |
| Player detection        | `detect_players_full.py`  | ✅ SAHI full-pitch + batched broadcast |
| Ball detection          | `detect_ball.py`          | ✅ SAHI slicing                        |
| Multi-object tracking   | `track_players.py`        | ✅ ByteTrack                           |
| Pitch canvas            | `draw_pitch.py`           | ✅ FIFA markings, 10 px/m              |
| Homography (broadcast)  | `homography_picker.py`    | ✅ Interactive RANSAC                  |
| Homography (full-pitch) | `calibrate_full_pitch.py` | ✅ 21 FIFA landmarks                   |
| Coordinate projection   | `project_tracks.py`       | ✅ Vectorised                          |
| Trajectory smoothing    | `smooth_tracks.py`        | ✅ Gap-aware                           |
| Kinematic analytics     | `spatial_analytics.py`    | ✅                                     |
| Event extraction        | `extract_events.py`       | ✅ Shot/possession/pressure            |

### PS2 ✅ Complete

| Component           | File                    | Status                            |
| ------------------- | ----------------------- | --------------------------------- |
| Team classification | `team_classifier.py`    | ✅ K-Means jersey                 |
| Formation detection | `formation_detector.py` | ✅ Depth-axis clustering          |
| Pitch control       | `pitch_control.py`      | ✅ Voronoi + Shapely              |
| Clutch Score        | `clutch_score.py`       | ✅ xG × pressure × control × time |
| Tactical radar      | `tactical_radar.py`     | ✅ 6-stat per-player              |
| Broadcast overlay   | `broadcast_overlay.py`  | ✅ Voronoi + trails + HUD         |

### PS3 ✅ Complete

| Component         | File                   | Status                          |
| ----------------- | ---------------------- | ------------------------------- |
| xG model          | `train_xg.py`          | ✅ Logistic + Platt calibration |
| Ball trajectory   | `ball_trajectory.py`   | ✅ Quadratic interpolation      |
| Pass network      | `pass_network.py`      | ✅ NetworkX + PageRank          |
| Action classifier | `action_classifier.py` | ✅ Rule-based + optional RF     |
| Match report      | `export_report.py`     | ✅ Self-contained HTML          |

### PS4 ✅ Complete

| Component         | File                  | Status                           |
| ----------------- | --------------------- | -------------------------------- |
| Offside detection | `offside_detector.py` | ✅ Geometric Law 11              |
| Momentum score    | `momentum_score.py`   | ✅ Multi-signal rolling timeline |
| Player rating     | `player_rating.py`    | ✅ 6-dimension composite 0–10    |
| Temporal xG       | `temporal_xg.py`      | ✅ Bidirectional LSTM            |
| YOLO fine-tuning  | `fine_tune_yolo.py`   | ✅ Supervised + pseudo-label     |

---

## Installation

```bash
git clone https://github.com/anuragkukreti07/goalX.git
cd goalX && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# PS2 extras
pip install scikit-learn scipy shapely matplotlib --break-system-packages

# PS3 extras
pip install networkx joblib sahi --break-system-packages

# PS4 extras
pip install torch --break-system-packages   # temporal_xg.py
```

---

## Quick Run Order

```bash
# Generate pitch and calibrate
python -m goalx.ps1_cv.draw_pitch --out data/pitch_map.png
python -m goalx.ps1_cv.homography_picker --frame data/.../000001.jpg --pitch data/pitch_map.png --out data/homography_data.npz

# PS1 pipeline
python -m goalx.ps1_cv.detect_ball --seq data/.../img1/ --out-csv outputs/detections_with_ball.csv
python -m goalx.ps1_cv.track_players --input outputs/detections_with_ball.csv --output data/tracking.csv
python -m goalx.ps1_cv.project_tracks --tracks data/tracking.csv --homography data/homography_data.npz --pitch data/pitch_map.png --out-dir outputs/projected
python -m goalx.ps1_cv.smooth_tracks --projected outputs/projected/projected_tracks.csv --out-dir outputs/smoothed
python -m goalx.ps1_cv.extract_events --tracks outputs/smoothed/smoothed_tracks.csv --ball outputs/detections_with_ball.csv --out-dir outputs/events

# PS2 pipeline
python -m goalx.ps2_ml.team_classifier --tracks data/tracking.csv --frames data/.../img1/ --out-dir outputs/teams
python -m goalx.ps2_ml.pitch_control --tracks outputs/smoothed/smoothed_tracks.csv --teams outputs/teams/team_assignments.csv --out-dir outputs/pitch_control --heatmap
python -m goalx.ps2_ml.clutch_score --events outputs/events/events.csv --tracks outputs/smoothed/smoothed_tracks.csv --teams outputs/teams/team_assignments.csv --control outputs/pitch_control/pitch_control.csv --out-dir outputs/clutch
python -m goalx.ps2_ml.tactical_radar --tracks outputs/smoothed/smoothed_tracks.csv --teams outputs/teams/team_assignments.csv --control outputs/pitch_control/pitch_control.csv --clutch outputs/clutch/clutch_scores.csv --ball outputs/detections_with_ball.csv --out-dir outputs/radar
python -m goalx.ps2_ml.broadcast_overlay --frames-dir data/.../img1 --tracks outputs/smoothed/smoothed_tracks.csv --teams outputs/teams/team_assignments.csv --events outputs/events/events.csv --pitch-control outputs/pitch_control/pitch_control.csv --homography data/homography_data.npz --pitch data/pitch_map.png --out outputs/broadcast_overlay.mp4

# PS3 pipeline
python -m goalx.ps3_ml.train_xg --synthetic --events outputs/events/events.csv --tracks outputs/smoothed/smoothed_tracks.csv --control outputs/pitch_control/pitch_control.csv --out-dir outputs/xg_model
python -m goalx.ps3_ml.ball_trajectory --ball outputs/detections_with_ball.csv --out-dir outputs/ball_trajectory
python -m goalx.ps3_ml.pass_network --events outputs/events/events.csv --tracks outputs/smoothed/smoothed_tracks.csv --teams outputs/teams/team_assignments.csv --ball outputs/ball_trajectory/interpolated_ball.csv --pitch data/pitch_map.png --out-dir outputs/pass_network
python -m goalx.ps3_ml.action_classifier --tracks outputs/smoothed/smoothed_tracks.csv --teams outputs/teams/team_assignments.csv --ball outputs/ball_trajectory/interpolated_ball.csv --events outputs/events/events.csv --out-dir outputs/actions
python -m goalx.ps3_ml.export_report --tracks outputs/smoothed/smoothed_tracks.csv --events outputs/events/events.csv --clutch outputs/clutch/clutch_scores.csv --xg-pred outputs/xg_model/xg_predictions.csv --pass-net-dir outputs/pass_network --actions outputs/actions/actions.csv --radar-dir outputs/radar --heatmaps-dir outputs/heatmaps --formation outputs/teams/formations.csv --out outputs/match_report.html --title "Match Analysis"

# PS4 pipeline
python -m goalx.ps4_research.offside_detector --tracks outputs/smoothed/smoothed_tracks.csv --teams outputs/teams/team_assignments.csv --ball outputs/ball_trajectory/interpolated_ball.csv --passes outputs/pass_network/pass_network.csv --pitch data/pitch_map.png --out-dir outputs/offside
python -m goalx.ps4_research.momentum_score --tracks outputs/smoothed/smoothed_tracks.csv --events outputs/events/events.csv --teams outputs/teams/team_assignments.csv --out-dir outputs/momentum
python -m goalx.ps4_research.player_rating --tracks outputs/smoothed/smoothed_tracks.csv --teams outputs/teams/team_assignments.csv --clutch outputs/clutch/clutch_scores.csv --centrality outputs/pass_network/centrality_all.csv --events outputs/events/events.csv --analytics outputs/spatial_analytics.csv --control outputs/pitch_control/pitch_control.csv --out-dir outputs/player_ratings
python -m goalx.ps4_research.temporal_xg --events outputs/events/events.csv --tracks outputs/smoothed/smoothed_tracks.csv --ball outputs/ball_trajectory/interpolated_ball.csv --xg-csv outputs/xg_model/xg_predictions.csv --out-dir outputs/temporal_xg --synthetic
python -m goalx.ps4_research.fine_tune_yolo --pseudo-labels --detections outputs/detections_with_ball.csv --frames-dir data/.../img1 --model yolov8s.pt --out-dir models/fine_tuned --epochs 50
```

---

## Scale Reference

| Quantity                | Value                        |
| ----------------------- | ---------------------------- |
| Pitch canvas            | 1050 × 680 px                |
| Scale                   | 10 px / metre                |
| Pitch dimensions        | 105 m × 68 m (FIFA standard) |
| Default FPS             | 25                           |
| Max player speed cap    | 12 m/s                       |
| Max ball speed (interp) | 40 m/s                       |
| ByteTrack buffer        | 90 frames (3 s)              |
| Momentum window         | 375 frames (15 s)            |
| Temporal xG context     | 50 frames (2 s)              |

---

## Status Checklist

- [x] Frame extraction
- [x] Player + ball detection (YOLOv8 + SAHI)
- [x] Multi-object tracking (ByteTrack)
- [x] Homography calibration (broadcast + full-pitch)
- [x] Coordinate projection + smoothing
- [x] Kinematic analytics
- [x] Event extraction (shot, possession, pressure)
- [x] Team classification + formation detection
- [x] Pitch control (Voronoi)
- [x] Clutch Score
- [x] Tactical radar charts + broadcast overlay
- [x] xG model (logistic, 3 data modes)
- [x] Ball trajectory interpolation
- [x] Pass network + centrality metrics
- [x] Action classifier
- [x] HTML match report
- [x] **Offside detector** (Law 11 geometry)
- [x] **Momentum score** (multi-signal rolling timeline)
- [x] **Player rating** (6-dimension composite)
- [x] **Temporal xG** (LSTM sequence model)
- [x] **YOLO fine-tuning** (supervised + pseudo-label)
