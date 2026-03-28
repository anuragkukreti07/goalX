"""
velocity_heatmap.py
───────────────────
Generates spatial velocity and activity heatmaps per team and player:

  • Speed heatmap     : where on the pitch do players run fastest?
  • Activity heatmap  : where do players spend most time?
  • Pressure heatmap  : where does defensive pressure concentrate?

Uses scipy.ndimage Gaussian smoothing for publication-quality output.
Overlaid on the 2D pitch canvas from draw_pitch.py.

Usage
─────
  python3 src/goalx/velocity_heatmap.py \\
      --tracks   outputs/smoothed_tracks.csv \\
      --teams    outputs/team_labels.csv \\
      --pitch    data/pitch_map.png \\
      --out-dir  outputs/heatmaps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────────
#  Style
# ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#0e1117",
    "text.color":       "#c8c8d0",
    "font.size":        10,
})


# ─────────────────────────────────────────────────────────────────
#  Heatmap builder
# ─────────────────────────────────────────────────────────────────

# def _compute_velocity(tracks: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute per-frame speed (pitch pixels per frame) from position deltas.
#     Adds a 'speed' column to the dataframe.
#     """
#     tracks = tracks.sort_values(["track_id", "frame_id"]).copy()
#     tracks["dx"] = tracks.groupby("track_id")["pitch_x"].diff().fillna(0)
#     tracks["dy"] = tracks.groupby("track_id")["pitch_y"].diff().fillna(0)
#     tracks["speed"] = np.sqrt(tracks["dx"]**2 + tracks["dy"]**2)
#     return tracks

def _compute_velocity(tracks: pd.DataFrame) -> pd.DataFrame:
    tracks = tracks.sort_values(["track_id", "frame_id"]).copy()
    tracks["dx"] = tracks.groupby("track_id")["smooth_x"].diff().fillna(0)
    tracks["dy"] = tracks.groupby("track_id")["smooth_y"].diff().fillna(0)
    tracks["speed"] = np.sqrt(tracks["dx"]**2 + tracks["dy"]**2)
    return tracks


def _make_grid(tracks: pd.DataFrame,
               pitch_w: int, pitch_h: int,
               value_col: str = "speed",
               grid_scale: float = 0.5) -> np.ndarray:
    """
    Rasterise pitch_x / pitch_y coordinates onto a grid, accumulating
    the chosen value_col (speed, count=1, etc.).

    grid_scale : fraction of pitch resolution for the grid
                 (0.5 = half resolution for speed; 1.0 for full)
    """
    gw = int(pitch_w * grid_scale)
    gh = int(pitch_h * grid_scale)
    grid = np.zeros((gh, gw), dtype=np.float32)

    # for _, row in tracks.iterrows():
    #     px = int(np.clip(row["pitch_x"] * grid_scale, 0, gw - 1))
    #     py = int(np.clip(row["pitch_y"] * grid_scale, 0, gh - 1))
    #     val = float(row.get(value_col, 1))
    #     grid[py, px] += val

    # return grid
    for _, row in tracks.iterrows():
        px = int(np.clip(row["smooth_x"] * grid_scale, 0, gw - 1))
        py = int(np.clip(row["smooth_y"] * grid_scale, 0, gh - 1))
        val = float(row.get(value_col, 1))
        grid[py, px] += val
    return grid

def _overlay_heatmap(pitch_img: np.ndarray,
                      grid: np.ndarray,
                      sigma: float = 12.0,
                      alpha: float = 0.60,
                      cmap: str = "plasma") -> np.ndarray:
    """
    Gaussian-smooth the grid, colour-map it, and alpha-blend it onto
    a copy of the pitch canvas.
    """
    # Resize grid to match pitch image
    ph, pw = pitch_img.shape[:2]
    grid_r = cv2.resize(grid, (pw, ph), interpolation=cv2.INTER_LINEAR)

    # Gaussian smoothing for publication quality
    grid_s = gaussian_filter(grid_r, sigma=sigma)

    if grid_s.max() > 0:
        grid_n = grid_s / grid_s.max()
    else:
        return pitch_img.copy()

    # Colormap
    cmap_fn = plt.get_cmap(cmap)
    coloured = (cmap_fn(grid_n)[..., :3] * 255).astype(np.uint8)
    coloured_bgr = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)

    # Blend: only colour cells with non-zero signal
    mask = (grid_n > 0.02).astype(np.float32)
    mask_3 = np.stack([mask]*3, axis=-1)

    result = pitch_img.copy().astype(np.float32)
    coloured_f = coloured_bgr.astype(np.float32)

    result = result * (1 - mask_3 * alpha) + coloured_f * (mask_3 * alpha)
    return result.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class VelocityHeatmapper:
    def __init__(self, tracks_csv: Path, teams_csv: Path,
                 pitch_path: Path, out_dir: Path):
        self.tracks_csv = tracks_csv
        self.teams_csv  = teams_csv
        self.pitch_path = pitch_path
        self.out_dir    = out_dir

    def run(self) -> None:
        print(f"\n  goalX — Velocity & Activity Heatmaps")
        print(f"  {'─'*42}\n")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        pitch_img = cv2.imread(str(self.pitch_path))
        if pitch_img is None:
            raise FileNotFoundError(f"Cannot read pitch: {self.pitch_path}")
        ph, pw = pitch_img.shape[:2]

        tracks = pd.read_csv(self.tracks_csv)
        if "pitch_x" not in tracks.columns:
            raise ValueError("Tracks CSV must have 'pitch_x' and 'pitch_y' columns.")

        tracks = tracks.dropna(subset=["smooth_x", "smooth_y"])
        tracks = tracks[tracks["track_id"] != -1].copy()
        # Merge team labels
        teams_df = pd.read_csv(self.teams_csv)
        team_map = dict(zip(teams_df["track_id"].astype(int),
                            teams_df["team"].astype(str)))
        tracks["team"] = tracks["track_id"].map(team_map).fillna("unknown")

        # Compute speed
        tracks = _compute_velocity(tracks)
        # Add synthetic activity column (count = 1 per row)
        tracks["activity"] = 1.0
        configs = [
            ("all",  tracks,                          "plasma",  "speed",    "All players — speed heatmap"),
            ("all",  tracks,                          "YlOrRd",  "activity", "All players — activity heatmap"),
            ("home", tracks[tracks["team"]=="home"],  "Blues",   "speed",    "Home team — speed heatmap"),
            ("away", tracks[tracks["team"]=="away"],  "Reds",    "speed",    "Away team — speed heatmap"),
            ("home", tracks[tracks["team"]=="home"],  "Blues",   "activity", "Home team — activity"),
            ("away", tracks[tracks["team"]=="away"],  "Reds",    "activity", "Away team — activity"),
        ]

        

        for subset_name, df, cmap, value, title in configs:
            if df.empty:
                continue

            vcol = value if value in df.columns else "speed"
            grid = _make_grid(df, pw, ph, value_col=vcol, grid_scale=0.5)
            vis  = _overlay_heatmap(pitch_img, grid, sigma=14.0,
                                    alpha=0.65, cmap=cmap)

            # Matplotlib figure with colorbar
            fig, ax = plt.subplots(figsize=(12, 7.5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")

            # Show pitch as base, then heatmap overlay
            pitch_rgb = cv2.cvtColor(pitch_img, cv2.COLOR_BGR2RGB)
            vis_rgb   = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            ax.imshow(pitch_rgb, alpha=0.35)
            im = ax.imshow(cv2.cvtColor(
                    _overlay_heatmap(
                        np.zeros_like(pitch_img),
                        _make_grid(df, pw, ph, value_col=vcol, grid_scale=0.5),
                        sigma=14.0, alpha=1.0, cmap=cmap,
                    ), cv2.COLOR_BGR2RGB),
                    alpha=0.65,
                    cmap=cmap,
                    vmin=0, vmax=1,
            )

            cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
            cb.set_label("Normalised intensity", color="#c8c8d0", fontsize=10)
            cb.ax.yaxis.set_tick_params(color="#9090a0")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="#9090a0")

            ax.set_title(title, fontsize=14, color="#e8e8f0", weight="bold", pad=10)
            ax.axis("off")
            plt.tight_layout(pad=0.5)

            fname = f"heatmap_{subset_name}_{value}.png"
            fpath = self.out_dir / fname
            fig.savefig(fpath, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  ✔  {fname}")

        print(f"\n  ✅  Heatmaps saved → {self.out_dir}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Velocity and activity heatmaps for goalX."
    )
    p.add_argument("--tracks",  required=True)
    p.add_argument("--teams",   required=True)
    p.add_argument("--pitch",   required=True)
    p.add_argument("--out-dir", default="outputs/heatmaps")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    VelocityHeatmapper(
        tracks_csv = Path(args.tracks),
        teams_csv  = Path(args.teams),
        pitch_path = Path(args.pitch),
        out_dir    = Path(args.out_dir),
    ).run()