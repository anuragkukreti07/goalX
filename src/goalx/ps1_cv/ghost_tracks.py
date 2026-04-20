"""
ghost_tracks.py
───────────────
Extends visible player tracks with predicted positions for frames
where the player is outside the camera's field of view (off-screen).

For each track that exits the canvas (in_canvas=False for >3 consecutive
frames), propagate position using last known velocity until the track
re-enters or the sequence ends.

Optionally blends toward formation centroid to prevent runaway extrapolation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_velocity(group: pd.DataFrame, window: int = 5) -> tuple[float, float]:
    """Average velocity over last `window` visible frames."""
    visible = group[group['in_canvas'] == True].tail(window)
    if len(visible) < 2:
        return 0.0, 0.0
    dx = visible['pitch_x'].diff().dropna().mean()
    dy = visible['pitch_y'].diff().dropna().mean()
    return float(dx), float(dy)


def run_ghost_propagation(
    tracks_csv: str,
    out_csv: str,
    pitch_w: int = 1050,
    pitch_h: int = 680,
    max_speed_px: float = 30.0,   # max px/frame ~= 12 m/s at 10px/m, 25fps
    formation_pull: float = 0.05, # how strongly to pull toward median position
):
    print("\n  goalX — Ghost Track Propagator")
    print("  " + "─" * 40)

    df = pd.read_csv(tracks_csv).sort_values(['track_id', 'frame_id'])
    all_frames = sorted(df['frame_id'].unique())
    result_rows = []

    for tid, group in df.groupby('track_id'):
        if tid == -1:  # ball handled separately
            result_rows.append(group)
            continue

        group = group.sort_values('frame_id').copy()
        group['is_ghost'] = False

        # Compute median position as formation anchor
        visible = group[group['in_canvas'] == True]
        if visible.empty:
            result_rows.append(group)
            continue

        median_x = visible['pitch_x'].median()
        median_y = visible['pitch_y'].median()

        # Find gaps where player is off-screen
        frames_present = set(group['frame_id'].values)
        first_frame = group['frame_id'].min()
        last_frame = group['frame_id'].max()

        ghost_rows = []
        last_visible = None

        for fid in range(int(first_frame), int(last_frame) + 1):
            if fid in frames_present:
                row = group[group['frame_id'] == fid].iloc[0]
                if row['in_canvas']:
                    last_visible = row
                continue

            # Frame is missing — player is off-screen, generate ghost
            if last_visible is None:
                continue

            vx, vy = compute_velocity(
                group[group['frame_id'] <= last_visible['frame_id']]
            )

            # Clamp velocity to max speed
            speed = np.sqrt(vx**2 + vy**2)
            if speed > max_speed_px:
                vx = vx * max_speed_px / speed
                vy = vy * max_speed_px / speed

            steps = fid - int(last_visible['frame_id'])
            pred_x = last_visible['pitch_x'] + vx * steps
            pred_y = last_visible['pitch_y'] + vy * steps

            # Formation pull — blend toward median position over time
            pull = min(formation_pull * steps, 0.5)
            pred_x = pred_x * (1 - pull) + median_x * pull
            pred_y = pred_y * (1 - pull) + median_y * pull

            # Keep within pitch bounds
            pred_x = float(np.clip(pred_x, 0, pitch_w))
            pred_y = float(np.clip(pred_y, 0, pitch_h))

            ghost_rows.append({
                'frame_id': fid,
                'track_id': int(tid),
                'pitch_x': pred_x,
                'pitch_y': pred_y,
                'in_canvas': False,
                'is_ghost': True,
                'img_x': np.nan,
                'img_y': np.nan,
            })

        if ghost_rows:
            ghost_df = pd.DataFrame(ghost_rows)
            group = pd.concat([group, ghost_df]).sort_values('frame_id')

        result_rows.append(group)

    final_df = pd.concat(result_rows).sort_values(['frame_id', 'track_id'])
    final_df.to_csv(out_csv, index=False)

    n_ghost = final_df['is_ghost'].sum() if 'is_ghost' in final_df.columns else 0
    print(f"  ✔  Original rows : {len(df):,}")
    print(f"  ✔  Ghost rows added : {n_ghost:,}")
    print(f"  ✅  Saved → {out_csv}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run_ghost_propagation(args.tracks, args.out)
