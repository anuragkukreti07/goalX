"""
fix_ball.py
Applies a physics velocity gate to the ball track. If the ball teleports
more than 120 pixels (physically impossible), it deletes the spike and interpolates.
"""
import pandas as pd
import numpy as np
import argparse

def fix_ball(csv_path):
    print("\n  goalX — Ball Velocity Gate")
    df = pd.read_csv(csv_path)
    
    ball_mask = df['track_id'] == -1
    if ball_mask.sum() == 0:
        print("  ⚠ No ball found.")
        return

    ball_df = df[ball_mask].copy().sort_values('frame_id')
    
    for col in ['pitch_x', 'pitch_y']:
        vals = ball_df[col].copy()
        delta = vals.diff().abs()
        
        # 120 pixels in 1 frame = Mach 1. Impossible. 
        impossible = delta > 120
        n_flagged = impossible.sum()
        
        if n_flagged > 0:
            print(f"  ⚽ Flagged {n_flagged} impossible teleports in {col}.")
            vals[impossible] = np.nan
            vals = vals.interpolate(method='linear', limit_direction='both')
            ball_df[col] = vals
            
    df.loc[ball_mask, 'pitch_x'] = ball_df['pitch_x'].values
    df.loc[ball_mask, 'pitch_y'] = ball_df['pitch_y'].values
    
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Ball physics corrected. Saved -> {csv_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    args = parser.parse_args()
    fix_ball(args.tracks)
