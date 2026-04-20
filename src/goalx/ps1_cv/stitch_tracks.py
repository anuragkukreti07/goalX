"""
stitch_tracks.py
────────────────
Performs 'Spatial-Temporal Stitching' on 2D projected tracks.
Fixes ID fragmentation caused by occlusions in the camera view by connecting
tracks that end and respawn near each other on the 2D pitch canvas.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def run_stitching(input_csv: str, output_csv: str, max_frame_gap: int, max_distance: float):
    print("\n  goalX — Spatial Track Stitcher")
    print("  " + "─" * 40)

    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Projected tracks not found: {input_path}")

    print(f"  Loading {input_path.name}...")
    df = pd.read_csv(input_path)

    # Separate ball (track_id -1) from players
    ball_df = df[df['track_id'] == -1].copy()
    players_df = df[df['track_id'] != -1].copy()

    initial_ids = players_df['track_id'].nunique()
    print(f"  ✔  Starting with {initial_ids} unique player IDs")

    # Get start and end states for every track
    track_stats = []
    for track_id, group in players_df.groupby('track_id'):
        start_row = group.loc[group['frame_id'].idxmin()]
        end_row = group.loc[group['frame_id'].idxmax()]
        track_stats.append({
            'track_id': track_id,
            'start_frame': start_row['frame_id'],
            'end_frame': end_row['frame_id'],
            'start_x': start_row['pitch_x'],
            'start_y': start_row['pitch_y'],
            'end_x': end_row['pitch_x'],
            'end_y': end_row['pitch_y']
        })

    # Sort tracks by when they first appear
    stats_df = pd.DataFrame(track_stats).sort_values('start_frame')
    active_tracks = stats_df.to_dict('records')

    # Greedy matching logic
    mapping = {}
    merges = 0

    for i, track_a in enumerate(active_tracks):
        if track_a['track_id'] in mapping:
            continue  # Already merged into a previous track
            
        for j in range(i + 1, len(active_tracks)):
            track_b = active_tracks[j]
            if track_b['track_id'] in mapping:
                continue
                
            frame_gap = track_b['start_frame'] - track_a['end_frame']
            
            # If B starts after A ends, and within our time threshold (e.g. 3 seconds)
            if 0 < frame_gap <= max_frame_gap:
                dist = calculate_distance(track_a['end_x'], track_a['end_y'], track_b['start_x'], track_b['start_y'])
                
                # If they are physically close on the pitch, they are the same person
                if dist <= max_distance:
                    mapping[track_b['track_id']] = track_a['track_id']
                    merges += 1
                    
                    # Update A's end state to B's end state so it can chain to a third track!
                    track_a['end_frame'] = track_b['end_frame']
                    track_a['end_x'] = track_b['end_x']
                    track_a['end_y'] = track_b['end_y']

    # Apply the mapping to the dataframe
    players_df['track_id'] = players_df['track_id'].replace(mapping)
    final_ids = players_df['track_id'].nunique()

    print(f"  ✔  Performed {merges} spatial merges")
    print(f"  ✔  Reduced fragmentation: {initial_ids} IDs → {final_ids} IDs")

    # Recombine with ball, sort, and save
    final_df = pd.concat([ball_df, players_df]).sort_values(['frame_id', 'track_id'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"\n  ✅  Saved stitched tracks to {output_path}\n")


def _parse_args():
    p = argparse.ArgumentParser(description="Merge broken tracks using 2D pitch proximity.")
    p.add_argument("--projected", required=True, help="Input CSV from project_tracks.py")
    p.add_argument("--out", required=True, help="Output stitched CSV")
    p.add_argument("--max-gap", type=int, default=90, help="Max frames between death and respawn (default: 90)")
    p.add_argument("--max-dist", type=float, default=50.0, help="Max pitch distance between death and respawn. (default: 50.0 px = ~5 meters)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_stitching(
        input_csv=args.projected,
        output_csv=args.out,
        max_frame_gap=args.max_gap,
        max_distance=args.max_dist
    )
