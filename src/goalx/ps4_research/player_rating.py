"""
player_rating.py  —  PS4 Step 3
─────────────────────────────────
Composite per-player rating — the final synthesis of the entire goalX pipeline.

Why this file exists
─────────────────────
After running all pipeline stages, we have rich data about each player:
clutch scores, PageRank, pressure/min, distance covered, action breakdown.
But these exist as separate CSVs. A thesis committee needs a single number
that answers: "Who performed best in this clip?"

player_rating.py aggregates everything into one interpretable rating on a
0–10 scale, with separate dimensional scores so the committee can see why
a player received their rating.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
#  Weights — must sum to 1.0
# ─────────────────────────────────────────────────────────────────

W_CLUTCH      = 0.25
W_INFLUENCE   = 0.20
W_PRESSING    = 0.18
W_WORK_RATE   = 0.15
W_VERSATILITY = 0.12
W_POSITIONING = 0.10

assert abs(W_CLUTCH + W_INFLUENCE + W_PRESSING +
           W_WORK_RATE + W_VERSATILITY + W_POSITIONING - 1.0) < 0.001

_DARK  = "#0e1117"; _SURF  = "#1a1d23"; _BORD = "#2e3140"; _TEXT = "#c8c8d0"
_GOLD  = "#f59e0b"; _BLUE  = "#3b82f6"; _GREEN = "#3dbf7a"
_RED   = "#ef4444"; _PURP  = "#8b5cf6"; _TEAL  = "#14b8a6"

DIM_COLORS = {
    "clutch":       _GOLD,
    "influence":    _BLUE,
    "pressing":     _RED,
    "work_rate":    _GREEN,
    "versatility":  _PURP,
    "positioning":  _TEAL,
}

plt.rcParams.update({
    "figure.facecolor": _DARK,  "axes.facecolor":  _SURF,
    "axes.edgecolor":   _BORD,  "axes.labelcolor": _TEXT,
    "xtick.color":      "#9090a0","ytick.color":   "#9090a0",
    "text.color":       _TEXT,  "grid.color":      _BORD,
    "font.size":        10,     "legend.facecolor": _SURF,
    "legend.edgecolor": _BORD,
})

FPS = 25.0


# ─────────────────────────────────────────────────────────────────
#  Normalisation
# ─────────────────────────────────────────────────────────────────

def _norm_col(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]. Returns 0.5 if all values identical."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────
#  Dimension builders
# ─────────────────────────────────────────────────────────────────

def _clutch_dim(clutch_df: pd.DataFrame) -> pd.Series:
    """Max clutch score per player (highest-stakes moment)."""
    if clutch_df.empty or "clutch_score" not in clutch_df.columns:
        return pd.Series(dtype=float)
    col = "track_id" if "track_id" in clutch_df.columns else clutch_df.columns[0]
    return clutch_df.groupby(col)["clutch_score"].max().rename("clutch")


def _influence_dim(centrality_df: pd.DataFrame) -> pd.Series:
    """PageRank from pass network."""
    if centrality_df.empty or "pagerank" not in centrality_df.columns:
        return pd.Series(dtype=float)
    
    # FIX: Filter out duplicates from centrality_all.csv
    df = centrality_df.copy()
    if "team" in df.columns and "both" in df["team"].values:
        df = df[df["team"] == "both"]
        
    df = df.drop_duplicates(subset=["track_id"])
    return df.set_index("track_id")["pagerank"].rename("influence")


def _pressing_dim(events_df: pd.DataFrame,
                   tracks_df: pd.DataFrame) -> pd.Series:
    """
    Pressing events per minute per player.
    Uses pressure events where track_id is available.
    """
    press = events_df[events_df["event_type"].isin(["pressure", "press"])].copy()
    if press.empty or "track_id" not in press.columns:
        return pd.Series(dtype=float)

    total_minutes = tracks_df["frame_id"].nunique() / FPS / 60
    counts = press.groupby("track_id").size()
    return (counts / max(total_minutes, 1/60)).rename("pressing")


def _work_rate_dim(analytics_df: pd.DataFrame,
                    tracks_df: pd.DataFrame) -> pd.Series:
    """
    Total distance covered in km.
    Uses spatial_analytics.csv if available; falls back to computing
    from smoothed_tracks directly.
    """
    if (not analytics_df.empty and
            "total_distance_m" in analytics_df.columns and
            "track_id" in analytics_df.columns):
        dist = analytics_df.set_index("track_id")["total_distance_m"] / 1000
        # FIX: Drop duplicates just in case spatial_analytics has multiple rows per ID
        dist = dist[~dist.index.duplicated(keep="first")]
        return dist.rename("work_rate")

    # Fallback: compute from tracks
    if tracks_df.empty or "pitch_x" not in tracks_df.columns:
        return pd.Series(dtype=float)

    tracks_s = tracks_df.sort_values(["track_id", "frame_id"])
    tracks_s["dx"] = tracks_s.groupby("track_id")["pitch_x"].diff().fillna(0)
    tracks_s["dy"] = tracks_s.groupby("track_id")["pitch_y"].diff().fillna(0)
    tracks_s["dist_px"] = np.sqrt(tracks_s["dx"]**2 + tracks_s["dy"]**2)
    dist_km = (tracks_s.groupby("track_id")["dist_px"].sum() / 10.0 / 1000)
    return dist_km.rename("work_rate")


def _versatility_dim(actions_df: pd.DataFrame) -> pd.Series:
    """
    Shannon entropy of action distribution per player.
    A player with balanced IDLE/CARRY/DRIBBLE/PRESS/etc. scores higher
    than one who only ever IDLEs.
    """
    if actions_df.empty or "action" not in actions_df.columns:
        return pd.Series(dtype=float)

    def _entropy(counts):
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    dist = (actions_df.groupby(["track_id", "action"])
            .size().unstack(fill_value=0))
    entropy = dist.apply(_entropy, axis=1)
    return entropy.rename("versatility")


def _positioning_dim(control_df: pd.DataFrame,
                      tracks_df: pd.DataFrame,
                      team_map: dict[int, str]) -> pd.Series:
    """
    Mean pitch control area (normalised Voronoi fraction) that a player's
    team controls when they are on the pitch.
    """
    if (control_df.empty or
            "home_poss" not in control_df.columns or
            "frame_id" not in control_df.columns):
        return pd.Series(dtype=float)

    poss_map: dict[int, float] = dict(
        zip(control_df["frame_id"].astype(int),
            control_df["home_poss"].astype(float))
    )

    result: dict[int, list[float]] = {}
    for _, row in tracks_df.iterrows():
        tid  = int(row["track_id"])
        if tid < 0:
            continue
        fid  = int(row["frame_id"])
        team = team_map.get(tid, "")
        poss = poss_map.get(fid, 0.5)
        val  = poss if team == "home" else (1 - poss)
        result.setdefault(tid, []).append(val)

    means = {tid: float(np.mean(vals)) for tid, vals in result.items()}
    return pd.Series(means).rename("positioning")


# ─────────────────────────────────────────────────────────────────
#  Rating builder
# ─────────────────────────────────────────────────────────────────

def build_ratings(clutch_df, centrality_df, events_df, actions_df, analytics_df,
                   control_df, tracks_df, team_map: dict[int, str]
                   ) -> pd.DataFrame:
    """
    Assemble all dimensions and compute composite rating.
    Returns one row per track_id with all dimensional scores + composite.
    """
    dims: list[pd.Series] = [
        _clutch_dim(clutch_df),
        _influence_dim(centrality_df),
        _pressing_dim(events_df, tracks_df),
        _work_rate_dim(analytics_df, tracks_df),
        _versatility_dim(actions_df),
        _positioning_dim(control_df, tracks_df, team_map),
    ]
    dim_names = ["clutch", "influence", "pressing",
                 "work_rate", "versatility", "positioning"]
    weights   = [W_CLUTCH, W_INFLUENCE, W_PRESSING,
                 W_WORK_RATE, W_VERSATILITY, W_POSITIONING]

    # Align all series on track_id
    all_ids = sorted(set(tracks_df[tracks_df["track_id"] >= 0]["track_id"]
                         .astype(int).unique()))
    df = pd.DataFrame(index=all_ids)
    df.index.name = "track_id"

    for name, series in zip(dim_names, dims):
        if not series.empty:
            # Drop duplicates gracefully if they sneak in
            series = series[~series.index.duplicated(keep='first')]
            df[name] = series.reindex(all_ids, fill_value=0.0)
        else:
            df[name] = 0.0

    # Normalise each dimension to [0, 10]
    for name in dim_names:
        df[f"{name}_n"] = _norm_col(df[name]) * 10

    # Composite
    df["composite"] = sum(
        w * df[f"{n}_n"]
        for w, n in zip(weights, dim_names)
    )

    # Add team label
    df["team"] = df.index.map(lambda t: team_map.get(t, "unknown"))
    df["rank"] = df["composite"].rank(ascending=False, method="min").astype(int)

    return df.reset_index().sort_values("rank")


# ─────────────────────────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────────────────────────

def make_leaderboard_figure(ratings: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart ranked by composite score."""
    top = ratings.head(min(20, len(ratings)))
    fig, ax = plt.subplots(figsize=(11, max(4, len(top) * 0.45)))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_SURF)

    labels  = [f"ID {int(r['track_id'])}  [{r['team']}]"
               for _, r in top.iterrows()]
    values  = top["composite"].values
    colors  = [_BLUE if t == "home" else _RED
               for t in top["team"].values]

    bars = ax.barh(labels[::-1], values[::-1],
                   color=colors[::-1], height=0.65, zorder=3)
    ax.set_xlabel("Composite rating (0–10)")
    ax.set_title("Player rating leaderboard", color="#e8e8f0",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 11)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8.5, color=_TEXT)

    legend_patches = [
        mpatches.Patch(color=_BLUE, label="Home"),
        mpatches.Patch(color=_RED,  label="Away"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor=_SURF, edgecolor=_TEXT, labelcolor=_TEXT)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Leaderboard → {out_path}")


def make_radar_comparison(ratings: pd.DataFrame, out_path: Path,
                           top_n: int = 5) -> None:
    """
    Spider/radar chart overlaying the top-N players across all 6 dimensions.
    """
    dims = ["clutch_n", "influence_n", "pressing_n",
            "work_rate_n", "versatility_n", "positioning_n"]
    dim_labels = ["Clutch", "Influence", "Pressing",
                  "Work rate", "Versatility", "Positioning"]

    top = ratings.head(min(top_n, len(ratings)))
    N   = len(dims)

    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8),
                            subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_SURF)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, color=_TEXT, fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"],
                        color="#6a6d80", fontsize=7)
    ax.grid(color=_BORD, linewidth=0.6)

    palette = [_GOLD, _BLUE, _GREEN, _RED, _PURP, _TEAL]

    for i, (_, row) in enumerate(top.iterrows()):
        values = [row[d] for d in dims]
        values += values[:1]
        color = palette[i % len(palette)]
        ax.plot(angles, values, color=color, lw=2, alpha=0.85)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_title("Top player radar comparison", color="#e8e8f0",
                 fontsize=12, fontweight="bold", pad=20)

    legend_handles = [
        mpatches.Patch(color=palette[i % len(palette)],
                       label=f"ID {int(row['track_id'])} [{row['team']}] "
                             f"({row['composite']:.2f})")
        for i, (_, row) in enumerate(top.iterrows())
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.35, 1.1),
              facecolor=_SURF, edgecolor=_TEXT, labelcolor=_TEXT, fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Radar comparison → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class PlayerRatingEngine:
    def __init__(self, clutch_csv: Path, centrality_csv: Path,
                 events_csv: Path, actions_csv: Path, analytics_csv: Path,
                 control_csv: Path, tracks_csv: Path,
                 teams_csv: Path, out_dir: Path):
        self.clutch_csv     = clutch_csv
        self.centrality_csv = centrality_csv
        self.events_csv     = events_csv
        self.actions_csv    = actions_csv
        self.analytics_csv  = analytics_csv
        self.control_csv    = control_csv
        self.tracks_csv     = tracks_csv
        self.teams_csv      = teams_csv
        self.out_dir        = out_dir

    def _read(self, p: Path) -> pd.DataFrame:
        if p and p.exists():
            df = pd.read_csv(p)
            if "frame" in df.columns and "frame_id" not in df.columns:
                df.rename(columns={"frame": "frame_id"}, inplace=True)
            return df
        return pd.DataFrame()

    def run(self):
        print(f"\n  goalX PS4 — Player Rating Engine")
        print(f"  {'─'*44}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        clutch     = self._read(self.clutch_csv)
        centrality = self._read(self.centrality_csv)
        events     = self._read(self.events_csv)
        actions    = self._read(self.actions_csv)
        analytics  = self._read(self.analytics_csv)
        control    = self._read(self.control_csv)
        tracks     = self._read(self.tracks_csv)
        teams      = self._read(self.teams_csv)

        team_map: dict[int, str] = (
            dict(zip(teams["track_id"].astype(int), teams["team"].astype(str)))
            if not teams.empty else {}
        )

        print("  Building dimensional scores…")
        ratings = build_ratings(
            clutch, centrality, events, actions, analytics,
            control, tracks, team_map,
        )

        # Save
        csv_path = self.out_dir / "player_ratings.csv"
        ratings.to_csv(csv_path, index=False)
        print(f"  ✔  Ratings CSV → {csv_path}")

        # Print leaderboard
        print(f"\n  {'Rank':<5} {'ID':<6} {'Team':<8} {'Rating':<8} "
              f"{'Clutch':>7} {'Influence':>10} {'Pressing':>9} "
              f"{'Work':>7} {'Vers':>6} {'Pos':>5}")
        print(f"  {'─'*74}")
        for _, row in ratings.head(15).iterrows():
            print(f"  {int(row['rank']):<5} {int(row['track_id']):<6} "
                  f"{str(row['team']):<8} {row['composite']:>7.2f}  "
                  f"{row['clutch_n']:>6.2f}  {row['influence_n']:>9.2f}  "
                  f"{row['pressing_n']:>8.2f}  {row['work_rate_n']:>6.2f}  "
                  f"{row['versatility_n']:>5.2f}  {row['positioning_n']:>4.2f}")

        # Figures
        make_leaderboard_figure(ratings, self.out_dir / "rating_leaderboard.png")
        make_radar_comparison(ratings, self.out_dir / "rating_radar_top5.png")

        # Text report
        report_lines = ["goalX Player Ratings Report", "=" * 50, ""]
        for _, row in ratings.iterrows():
            report_lines.append(
                f"  #{int(row['rank']):2d}  ID {int(row['track_id']):4d}  "
                f"[{row['team']}]  "
                f"Rating: {row['composite']:.2f}/10"
            )
        rpt_path = self.out_dir / "ratings_report.txt"
        rpt_path.write_text("\n".join(report_lines))
        print(f"  ✔  Report → {rpt_path}")

        print(f"\n  ✅  Player rating complete → {self.out_dir}\n")


def _parse_args():
    p = argparse.ArgumentParser(description="Composite player rater (goalX PS4).")
    # Aligned defaults to standard pipeline output paths
    p.add_argument("--clutch",     default="outputs/clutch_scores.csv")
    p.add_argument("--centrality", default="outputs/pass_network/centrality_all.csv")
    p.add_argument("--events",     default="outputs/events.csv")
    p.add_argument("--actions",    default="outputs/actions/actions.csv")
    p.add_argument("--analytics",  default="outputs/spatial_analytics.csv")
    p.add_argument("--control",    default="outputs/pitch_control/pitch_control.csv")
    p.add_argument("--tracks",     default="outputs/smoothed_tracks.csv")
    p.add_argument("--teams",      default="outputs/team_labels.csv")
    p.add_argument("--out-dir",    default="outputs/player_ratings")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    PlayerRatingEngine(
        clutch_csv     = Path(args.clutch),
        centrality_csv = Path(args.centrality),
        events_csv     = Path(args.events),
        actions_csv    = Path(args.actions),
        analytics_csv  = Path(args.analytics),
        control_csv    = Path(args.control),
        tracks_csv     = Path(args.tracks),
        teams_csv      = Path(args.teams),
        out_dir        = Path(args.out_dir),
    ).run()