import pandas as pd
import numpy as np

# --- CONFIG ---
CSV_PATH = "outputs/projected/projected_tracks.csv"
FPS = 25
# Update this based on your pitch_map.png dimensions vs real pitch (105m x 68m)
# Example: If your map is 1050px wide, PIXELS_PER_METER = 10.0
PIXELS_PER_METER = 10.0  

def run_analytics():
    print("\n  goalX — Spatial Analytics")
    print("  " + "─" * 40)
    
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(['track_id', 'frame_id'])
    
    # 1. Calculate Vector Displacement (Δx, Δy)
    df['dx'] = df.groupby('track_id')['pitch_x'].diff()
    df['dy'] = df.groupby('track_id')['pitch_y'].diff()
    
    # 2. Calculate Euclidean Distance
    df['dist_m'] = np.sqrt(df['dx']**2 + df['dy']**2) / PIXELS_PER_METER
    
    # 3. Filter "Teleportation" Noise (Speed > 12m/s is biologically impossible)
    # 12 m/s is roughly 43 km/h (Usain Bolt speed)
    df.loc[df['dist_m'] > (12.0 / FPS), 'dist_m'] = 0
    
    # 4. Calculate Speed (km/h)
    # Distance / Time * 3.6 conversion
    df['speed_kmh'] = (df['dist_m'] * FPS) * 3.6
    
    # Smooth jitter with a rolling mean (5-frame window)
    df['speed_kmh'] = df.groupby('track_id')['speed_kmh'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # 5. Build Leaderboard
    stats = df.groupby('track_id').agg(
        total_dist_m=('dist_m', 'sum'),
        top_speed_kmh=('speed_kmh', 'max'),
        frames_active=('frame_id', 'count')
    ).reset_index()
    
    # Filter out bench players/short tracks (less than 2 seconds of footage)
    leaderboard = stats[stats['frames_active'] > 50].sort_values('total_dist_m', ascending=False)
    
    print(f"  📊 Analyzed {len(stats)} unique tracks.")
    print("  🏆 Top 10 High-Activity Players:")
    print("  " + "─" * 40)
    print(leaderboard[['track_id', 'total_dist_m', 'top_speed_kmh']].head(10).to_string(index=False))
    print("\n  ✅ Analytics Complete.")

if __name__ == "__main__":
    run_analytics()