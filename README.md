# goalX

**goalX** is an end-to-end football analytics system that converts raw match video into structured event data and contextual goal intelligence.

It combines computer vision, multi-object tracking, geometric reasoning, and contextual modeling to transform broadcast footage into actionable football insights.

---

## Project Vision

Modern football analytics relies heavily on event data, but most amateur and grassroots matches lack structured data pipelines.

goalX aims to:

- Detect players and ball from broadcast video
- Track player trajectories across time
- Map camera view to pitch coordinates
- Infer meaningful match events
- Compute contextual “Clutch Score” for goals

---

## System Architecture

Raw Match Video  
 ↓  
Frame Extraction  
 ↓  
Object Detection (YOLO)  
 ↓  
Multi-Object Tracking (ByteTrack)  
 ↓  
Homography (Image → Pitch Coordinates)  
 ↓  
Event Logic Engine  
 ↓  
Clutch Score (Contextual Goal Rating)

---

## Repository Structure

goalX/

├── src/goalx/  
│ ├── ps1_cv/ # Computer Vision pipeline  
│ │ ├── detect_players_full.py  
│ │ ├── track_players.py  
│ │ ├── extract_frames.py  
│ │ ├── visualise_tracks.py  
│ │ └── bytetrack/  
│ │  
│ └── ps2_ml/ # Contextual goal intelligence  
│  
├── scripts/  
├── models/ # (ignored)  
├── data/ # (ignored)  
├── outputs/ # (ignored)  
├── requirements.txt  
├── pyproject.toml  
└── README.md

---

## Phase Breakdown

### 🔹 PS1 – Computer Vision Pipeline

Goal: Convert raw video into structured spatio-temporal player data.

Components:

- [ ] Frame Extraction
- [ ] Player Detection (YOLOv8)
- [ ] Multi-Object Tracking (ByteTrack)
- [ ] Homography (camera → pitch mapping)
- [ ] Event Logic (goal, shot, possession)

Output:

- Player trajectories
- Structured event logs
- Visual overlays

---

### PS2 – Contextual Goal Rating

Goal: Move beyond “a goal is a goal.”

Introduce a **Clutch Score** that considers:

- Match minute
- Scoreline state
- Player pressure
- Match importance
- Momentum

Output:

- Goal importance ranking
- Context-aware scoring metric

---

## Installation

\`\`\`bash
git clone https://github.com/anuragkukreti07/goalX.git
cd goalX
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

---

## ▶ Example Usage

Extract frames:

\`\`\`bash
python -m goalx.ps1_cv.extract_frames --video path/to/video.mp4
\`\`\`

Run tracking:

\`\`\`bash
python -m goalx.ps1_cv.track_players --video path/to/video.mp4
\`\`\`

---

## Future Work

- Ball detection integration
- Possession modeling
- Expected Goals (xG) extension
- Tactical formation inference
- Real-time inference optimization

---

## Current Status

- [x] Repository initialized
- [ ] Frame extraction
- [ ] Detection
- [ ] Tracking
- [ ] Homography
- [ ] Event logic
- [ ] Clutch score

---

## Research Direction

goalX is designed as a modular research platform for:

- Sports analytics
- Spatio-temporal modeling
- Multi-object tracking
- Event-based video understanding

---
