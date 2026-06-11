"""
fix_ball.py
Advanced Trajectory Outlier Rejector.
Uses a rolling median to identify and destroy clusters of false positives.
"""
import pandas as pd
import numpy as np
import argparse

def fix_ball(csv_path):
    print("\n  goalX — Advanced Ball Trajectory Filter")
    df = pd.read_csv(csv_path)
    
    ball_mask = df['track_id'] == -1
    if ball_mask.sum() == 0:
        print("  ⚠ No ball found.")
        return

    ball_df = df[ball_mask].copy().sort_values('frame_id')
    window_size = 15 # 15-frame context window
    
    for col in ['pitch_x', 'pitch_y']:
        vals = ball_df[col].copy()
        
        # Calculate true path ignoring sudden spikes
        rolling_med = vals.rolling(window=window_size, center=True, min_periods=1).median()
        
        # Flag detections > 60 pixels off the true path
        outliers = (vals - rolling_med).abs() > 60
        n_flagged = outliers.sum()
        
        if n_flagged > 0:
            print(f"  ⚽ Killed {n_flagged} hallucinated outliers in {col}.")
            vals[outliers] = np.nan
            vals = vals.interpolate(method='linear', limit_direction='both')
            ball_df[col] = vals
            
    df.loc[ball_mask, 'pitch_x'] = ball_df['pitch_x'].values
    df.loc[ball_mask, 'pitch_y'] = ball_df['pitch_y'].values
    
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Ball trajectory locked and smoothed. Saved -> {csv_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    args = parser.parse_args()
    fix_ball(args.tracks)
