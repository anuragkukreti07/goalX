"""
momentum_score.py  —  PS4 Step 2
──────────────────────────────────
Computes and visualises rolling match momentum for each team.

Why this file exists
─────────────────────
The existing pipeline produces many independent metrics (possession %, shot
events, pressure counts, speed). Momentum_score synthesises all of them into
a single time-varying signal that answers: "who is winning the game right now?"

Momentum formula
─────────────────
  M(t) = w_poss  × Poss(t)         # fraction of last W frames in possession
       + w_shots × ShotRate(t)     # shots per minute in rolling window
       + w_press × PressRate(t)    # press events per minute
       + w_speed × NormSpeed(t)    # normalised mean team speed

All signals are normalised to [0, 1] before weighting.
Final momentum is expressed as HOME advantage over AWAY:
  MomentumDelta(t) = M_home(t) - M_away(t)  ∈ [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

PITCH_SCALE = 10.0
FPS         = 25.0

# Signal weights — tunable
W_POSS  = 0.35
W_SHOTS = 0.25
W_PRESS = 0.20
W_SPEED = 0.20

_DARK  = "#0e1117"; _SURF = "#1a1d23"; _BORD = "#2e3140"; _TEXT = "#c8c8d0"
_HOME  = "#3b82f6"; _AWAY = "#ef4444"; _NEUT = "#6b7280"

plt.rcParams.update({
    "figure.facecolor": _DARK,  "axes.facecolor":  _SURF,
    "axes.edgecolor":   _BORD,  "axes.labelcolor": _TEXT,
    "xtick.color":      "#9090a0", "ytick.color":  "#9090a0",
    "text.color":       _TEXT,  "grid.color":      _BORD,
    "grid.linewidth":   0.5,    "font.size":       10,
    "legend.facecolor": _SURF,  "legend.edgecolor": _BORD,
})


# ─────────────────────────────────────────────────────────────────
#  Signal computers
# ─────────────────────────────────────────────────────────────────

def _norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns 0.5 if all values identical."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def compute_possession_signal(events: pd.DataFrame,
                               all_frames: np.ndarray,
                               team: str,
                               window: int) -> np.ndarray:
    """
    For each frame, compute the fraction of the last `window` frames where
    `team` had possession. Returns array aligned to all_frames.
    """
    poss = events[events["event_type"] == "possession"].copy()
    if poss.empty or "team" not in poss.columns:
        return np.full(len(all_frames), 0.5)

    poss_set = set(poss[poss["team"] == team]["frame_id"].astype(int).tolist())
    total_set = set(poss["frame_id"].astype(int).tolist())

    result = np.zeros(len(all_frames))
    for i, fid in enumerate(all_frames):
        window_frames = range(max(0, fid - window), fid + 1)
        in_poss = sum(1 for f in window_frames if f in poss_set)
        in_total = sum(1 for f in window_frames if f in total_set)
        result[i] = in_poss / max(in_total, 1)

    return result


def compute_event_rate(events: pd.DataFrame,
                       all_frames: np.ndarray,
                       event_type: str,
                       team: str,
                       window: int,
                       team_map: dict[int, str]) -> np.ndarray:
    """
    Compute events-per-minute rate in a rolling window for one team.
    """
    team_events = events[events["event_type"] == event_type].copy()

    # Filter to team if track_id is available
    if "track_id" in team_events.columns:
        team_ids = {tid for tid, t in team_map.items() if t == team}
        team_events = team_events[team_events["track_id"].isin(team_ids)]

    event_frames = set(team_events["frame_id"].astype(int).tolist())
    result = np.zeros(len(all_frames))

    for i, fid in enumerate(all_frames):
        window_frames = range(max(0, fid - window), fid + 1)
        count = sum(1 for f in window_frames if f in event_frames)
        minutes = len(list(window_frames)) / FPS / 60
        result[i] = count / max(minutes, 1/60)

    return result


def compute_speed_signal(tracks: pd.DataFrame,
                          all_frames: np.ndarray,
                          team: str,
                          team_map: dict[int, str],
                          window: int) -> np.ndarray:
    """
    Mean player speed (m/s) for the team in a rolling window.
    """
    team_ids = {tid for tid, t in team_map.items() if t == team}
    team_tracks = tracks[tracks["track_id"].isin(team_ids)].copy()

    if team_tracks.empty or "pitch_x" not in team_tracks.columns:
        return np.zeros(len(all_frames))

    # Compute speed per frame
    team_tracks = team_tracks.sort_values(["track_id", "frame_id"])
    team_tracks["dx"] = team_tracks.groupby("track_id")["pitch_x"].diff().fillna(0)
    team_tracks["dy"] = team_tracks.groupby("track_id")["pitch_y"].diff().fillna(0)
    team_tracks["speed_ms"] = np.sqrt(
        (team_tracks["dx"] / PITCH_SCALE * FPS) ** 2 +
        (team_tracks["dy"] / PITCH_SCALE * FPS) ** 2
    )

    frame_mean_speed = (team_tracks.groupby("frame_id")["speed_ms"]
                        .mean()
                        .reindex(all_frames, fill_value=0.0)
                        .values)

    # Rolling mean
    return uniform_filter1d(frame_mean_speed, size=window, mode="nearest")


# ─────────────────────────────────────────────────────────────────
#  Momentum aggregator
# ─────────────────────────────────────────────────────────────────

def compute_momentum(tracks: pd.DataFrame,
                     events: pd.DataFrame,
                     team_map: dict[int, str],
                     window: int,
                     step: int = 25) -> pd.DataFrame:
    """
    Compute home and away momentum at every `step` frames.
    Returns DataFrame with frame_id, home_momentum, away_momentum, delta.
    """
    all_frame_ids = sorted(tracks["frame_id"].unique())
    sampled = np.array(all_frame_ids[::step])

    print(f"  Computing momentum over {len(sampled)} sample points "
          f"(window={window} frames = {window/FPS:.1f}s)…")

    signals: dict[str, dict[str, np.ndarray]] = {}

    for team in ("home", "away"):
        poss  = compute_possession_signal(events, sampled, team, window)
        shots = compute_event_rate(events, sampled, "shot",
                                   team, window, team_map)
        press = compute_event_rate(events, sampled, "pressure",
                                   team, window, team_map)
        speed = compute_speed_signal(tracks, sampled, team, team_map, window)
        signals[team] = {
            "poss": poss, "shots": shots, "press": press, "speed": speed
        }

    # Normalise each signal jointly (so home and away are on same scale)
    for sig in ("poss", "shots", "press", "speed"):
        combined = np.concatenate([signals["home"][sig],
                                    signals["away"][sig]])
        normed   = _norm(combined)
        n        = len(sampled)
        signals["home"][sig] = normed[:n]
        signals["away"][sig] = normed[n:]

    # Weighted sum
    for team in ("home", "away"):
        s = signals[team]
        signals[team]["momentum"] = (
            W_POSS  * s["poss"] +
            W_SHOTS * s["shots"] +
            W_PRESS * s["press"] +
            W_SPEED * s["speed"]
        )

    df = pd.DataFrame({
        "frame_id":       sampled,
        "home_momentum":  signals["home"]["momentum"].round(4),
        "away_momentum":  signals["away"]["momentum"].round(4),
        "delta":          (signals["home"]["momentum"] -
                           signals["away"]["momentum"]).round(4),
    })

    return df


# ─────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────

def make_momentum_chart(momentum_df: pd.DataFrame,
                         events: pd.DataFrame,
                         out_path: Path) -> None:
    """
    Dual-panel momentum chart:
      Top panel   : home vs away momentum as area chart
      Bottom panel: delta bar chart (positive = home, negative = away)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(_DARK)
    fig.suptitle("Match momentum timeline", color="#e8e8f0",
                 fontsize=14, fontweight="bold")

    t = momentum_df["frame_id"].values / FPS / 60   # convert to minutes
    home_m = momentum_df["home_momentum"].values
    away_m = momentum_df["away_momentum"].values
    delta  = momentum_df["delta"].values

    # ── Top: area chart ────────────────────────────────────────
    ax1.fill_between(t, home_m, alpha=0.35, color=_HOME, label="Home")
    ax1.fill_between(t, away_m, alpha=0.35, color=_AWAY, label="Away")
    ax1.plot(t, home_m, color=_HOME, lw=1.8, alpha=0.9)
    ax1.plot(t, away_m, color=_AWAY, lw=1.8, alpha=0.9)
    ax1.set_ylabel("Momentum index")
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.2, zorder=0)
    ax1.spines[["top","right"]].set_visible(False)

    # Annotate shot events
    shot_events = events[events["event_type"] == "shot"]
    for _, row in shot_events.iterrows():
        t_shot = int(row["frame_id"]) / FPS / 60
        ax1.axvline(t_shot, color=_AWAY, lw=0.6, alpha=0.5, linestyle=":")

    ax1.legend(loc="upper right", facecolor=_SURF, edgecolor=_TEXT,
               labelcolor=_TEXT, fontsize=9)

    # ── Bottom: delta bar ──────────────────────────────────────
    colors = np.where(delta > 0, _HOME, _AWAY)
    ax2.bar(t, delta, width=(t[1]-t[0]) * 0.9 if len(t) > 1 else 0.5,
            color=colors, alpha=0.75, zorder=3)
    ax2.axhline(0, color=_BORD, lw=0.8)
    ax2.set_xlabel("Match time (minutes)")
    ax2.set_ylabel("Home – Away")
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_ylim(-0.6, 0.6)

    # Shade regions
    ax2.fill_between(t, delta, 0, where=(delta > 0),
                     alpha=0.25, color=_HOME)
    ax2.fill_between(t, delta, 0, where=(delta < 0),
                     alpha=0.25, color=_AWAY)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Momentum chart → {out_path}")


def make_momentum_summary(momentum_df: pd.DataFrame) -> str:
    """Generate a text summary for inclusion in thesis tables."""
    delta = momentum_df["delta"].values
    home_dom = (delta > 0.05).mean()
    away_dom = (delta < -0.05).mean()
    balanced  = 1 - home_dom - away_dom

    peak_home_idx = momentum_df["home_momentum"].idxmax()
    peak_away_idx = momentum_df["away_momentum"].idxmax()

    peak_home_min = momentum_df.loc[peak_home_idx, "frame_id"] / FPS / 60
    peak_away_min = momentum_df.loc[peak_away_idx, "frame_id"] / FPS / 60

    lines = [
        "Momentum Score Summary",
        "=" * 40,
        f"Home dominant periods  : {home_dom:.1%}",
        f"Away dominant periods  : {away_dom:.1%}",
        f"Balanced periods       : {balanced:.1%}",
        f"Peak home momentum     : {momentum_df['home_momentum'].max():.3f}  "
        f"(minute {peak_home_min:.1f})",
        f"Peak away momentum     : {momentum_df['away_momentum'].max():.3f}  "
        f"(minute {peak_away_min:.1f})",
        f"Mean delta (home–away) : {delta.mean():.3f}",
        f"Std delta              : {delta.std():.3f}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class MomentumAnalyser:
    def __init__(self, tracks_csv: Path, events_csv: Path, teams_csv: Path,
                 out_dir: Path, window_frames: int, step: int):
        self.tracks_csv    = tracks_csv
        self.events_csv    = events_csv
        self.teams_csv     = teams_csv
        self.out_dir       = out_dir
        self.window_frames = window_frames
        self.step          = step

    def run(self):
        print(f"\n  goalX PS4 — Momentum Analyser")
        print(f"  {'─'*44}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        tracks = pd.read_csv(self.tracks_csv)
        events = pd.read_csv(self.events_csv)
        teams  = pd.read_csv(self.teams_csv)

        for df in [tracks, events]:
            if "frame" in df.columns and "frame_id" not in df.columns:
                df.rename(columns={"frame": "frame_id"}, inplace=True)

        team_map: dict[int, str] = dict(
            zip(teams["track_id"].astype(int), teams["team"].astype(str))
        )

        momentum_df = compute_momentum(
            tracks, events, team_map,
            window=self.window_frames, step=self.step,
        )

        # Save CSV
        csv_path = self.out_dir / "momentum_timeline.csv"
        momentum_df.to_csv(csv_path, index=False)
        print(f"  ✔  Timeline CSV → {csv_path}")

        # Chart
        make_momentum_chart(momentum_df, events,
                             self.out_dir / "momentum_chart.png")

        # Summary
        summary = make_momentum_summary(momentum_df)
        summary_path = self.out_dir / "momentum_summary.txt"
        summary_path.write_text(summary)
        print(f"  ✔  Summary → {summary_path}")
        print(f"\n{summary}\n")

        print(f"  ✅  Momentum analysis complete → {self.out_dir}\n")


def _parse_args():
    p = argparse.ArgumentParser(description="Match momentum scorer (goalX PS4).")
    # Updated defaults to point straight to pipeline outputs
    p.add_argument("--tracks",        default="outputs/smoothed_tracks.csv")
    p.add_argument("--events",        default="outputs/events.csv")
    p.add_argument("--teams",         default="outputs/team_labels.csv")
    p.add_argument("--out-dir",       default="outputs/momentum")
    p.add_argument("--window-frames", type=int, default=375,
                   help="Rolling window size in frames (default: 375 = 15s at 25fps)")
    p.add_argument("--step",          type=int, default=25,
                   help="Sampling step in frames (default: 25 = 1s)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    MomentumAnalyser(
        tracks_csv    = Path(args.tracks),
        events_csv    = Path(args.events),
        teams_csv     = Path(args.teams),
        out_dir       = Path(args.out_dir),
        window_frames = args.window_frames,
        step          = args.step,
    ).run()