"""
tactical_radar.py
─────────────────
Aggregates all pipeline outputs into per-player and per-team statistics,
then renders them as radar (spider) charts for thesis figures.

Stats computed per player
──────────────────────────
  distance_m        Total distance covered  (pixels → metres via --px-per-m)
  top_speed         Peak frame-to-frame displacement  (smoothed)
  possession_inv    Fraction of frames within proximity of the ball  [0–1]
  pressure_applied  Frames where player was within PRESSURE_RADIUS of ball-carrier
  avg_control       Mean pitch control % for their team across all frames
  clutch_score      Max Clutch Score from any shot event (0 if no shots)

Output
──────
  <out-dir>/player_stats.csv          — raw stat table
  <out-dir>/radar_<track_id>.png      — individual player radar
  <out-dir>/radar_team_comparison.png — home vs away team averages

Usage
─────
  python3 src/goalx/ps1_cv/tactical_radar.py \
      --tracks   outputs/smoothed/smoothed_tracks.csv \
      --teams    outputs/teams/team_assignments.csv \
      --control  outputs/pitch_control/pitch_control.csv \
      --clutch   outputs/clutch/clutch_scores.csv \
      --out-dir  outputs/radar
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless rendering — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

PX_PER_METRE   = 9.45    # tune to your draw_pitch.py scale
FPS            = 25
BALL_PROX_PX   = 150     # within this distance = "in possession"
PRESSURE_PX    = 200     # within this = applying pressure on carrier
RADAR_STATS    = [
    "distance_m",
    "top_speed",
    "possession_inv",
    "pressure_applied",
    "avg_control",
    "clutch_score",
]
STAT_LABELS    = [
    "Distance (m)",
    "Top Speed",
    "Possession\nInvolvement",
    "Pressure\nApplied",
    "Pitch\nControl",
    "Clutch\nScore",
]

# Team colours for charts (hex, readable on white background)
TEAM_COLOURS = {"home": "#1A73E8", "away": "#E84435", "other": "#888888"}


# ─────────────────────────────────────────────────────────────────
#  Stat computation
# ─────────────────────────────────────────────────────────────────

def _compute_player_stats(
    tracks:  pd.DataFrame,
    teams:   pd.DataFrame,
    control: pd.DataFrame,
    clutch:  pd.DataFrame,
) -> pd.DataFrame:
    """Compute all six radar stats for every track_id."""

    merged = tracks.merge(teams[["track_id", "team"]], on="track_id", how="left")
    merged["team"] = merged["team"].fillna("other")
    merged = merged.sort_values(["track_id", "frame_id"])

    # ── Extract ball directly from tracks (Coordinate-Aligned!) ──
    ball_by_frame: dict[int, tuple[float, float]] = {}
    ball_rows = tracks[tracks["track_id"] == -1]
    for _, row in ball_rows.iterrows():
        ball_by_frame[int(row["frame_id"])] = (float(row["smooth_x"]), float(row["smooth_y"]))

    # Per-team avg control (averaged over all frames)
    home_ctrl_mean = float(control["home_pct"].mean()) if not control.empty else 50.0
    away_ctrl_mean = float(control["away_pct"].mean()) if not control.empty else 50.0
    team_ctrl = {"home": home_ctrl_mean, "away": away_ctrl_mean, "other": 50.0}

    records = []
    
    # Exclude the ball from player stats
    players_only = merged[merged["track_id"] != -1]
    
    for tid, grp in tqdm(players_only.groupby("track_id"), desc="Computing player stats"):
        grp = grp.sort_values("frame_id")
        team = str(grp["team"].mode()[0])

        # ── Distance covered ──────────────────────────────────────
        dx = grp["smooth_x"].diff().fillna(0)
        dy = grp["smooth_y"].diff().fillna(0)
        step_px     = np.hypot(dx, dy)
        distance_m  = float(step_px.sum()) / PX_PER_METRE

        # ── Top speed (px/frame → km/h equivalent) ───────────────
        top_speed = float(step_px.max())   # px/frame; normalised in radar

        # ── Possession involvement ────────────────────────────────
        poss_frames = 0
        press_frames = 0
        for _, row in grp.iterrows():
            fid = int(row["frame_id"])
            if fid not in ball_by_frame:
                continue
            bx, by = ball_by_frame[fid]
            d = math.hypot(row["smooth_x"] - bx, row["smooth_y"] - by)
            if d < BALL_PROX_PX:
                poss_frames += 1
            if d < PRESSURE_PX:
                press_frames += 1

        total_frames   = len(grp)
        possession_inv = poss_frames  / total_frames if total_frames > 0 else 0.0
        pressure_app   = press_frames / total_frames if total_frames > 0 else 0.0

        # ── Clutch score ──────────────────────────────────────────
        if not clutch.empty and tid in clutch["track_id"].values:
            cs = float(clutch.loc[clutch["track_id"] == tid, "clutch_score"].max())
        else:
            cs = 0.0

        records.append({
            "track_id":        int(tid),
            "team":            team,
            "distance_m":      round(distance_m, 1),
            "top_speed":       round(top_speed, 3),
            "possession_inv":  round(possession_inv, 4),
            "pressure_applied":round(pressure_app, 4),
            "avg_control":     round(team_ctrl.get(team, 50.0) / 100.0, 4),
            "clutch_score":    round(cs, 4),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────
#  Normalisation  (min-max per stat across all players)
# ─────────────────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    norm = df.copy()
    for col in RADAR_STATS:
        lo = df[col].min()
        hi = df[col].max()
        if hi > lo:
            norm[col] = (df[col] - lo) / (hi - lo)
        else:
            norm[col] = 0.0
    return norm


# ─────────────────────────────────────────────────────────────────
#  Radar chart renderer
# ─────────────────────────────────────────────────────────────────

def _radar_chart(
    values:  list[float],
    labels:  list[str],
    title:   str,
    colour:  str,
    out_path: Path,
) -> None:
    """Render a single radar chart and save as PNG."""
    n   = len(values)
    angles = [i * 2 * math.pi / n for i in range(n)] + [0]
    vals   = list(values) + [values[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color="#444")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="#aaa")
    ax.set_ylim(0, 1.0)

    ax.plot(angles, vals, color=colour, linewidth=2)
    ax.fill(angles, vals, color=colour, alpha=0.25)

    ax.spines["polar"].set_color("#ddd")
    ax.yaxis.grid(color="#eee", linestyle="--", linewidth=0.5)
    ax.xaxis.grid(color="#ddd", linewidth=0.5)

    ax.set_title(title, pad=18, fontsize=11, fontweight="medium", color="#222")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def _team_comparison_chart(
    norm_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Overlay radar for home vs away team averages."""
    n       = len(RADAR_STATS)
    angles  = [i * 2 * math.pi / n for i in range(n)] + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(STAT_LABELS, fontsize=9, color="#444")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([], fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.spines["polar"].set_color("#ddd")
    ax.yaxis.grid(color="#eee", linestyle="--", linewidth=0.5)
    ax.xaxis.grid(color="#ddd", linewidth=0.5)

    for team_label in ("home", "away"):
        subset = norm_df[norm_df["team"] == team_label]
        if subset.empty:
            continue
        means  = subset[RADAR_STATS].mean().tolist()
        vals   = means + [means[0]]
        colour = TEAM_COLOURS[team_label]
        ax.plot(angles, vals, color=colour, linewidth=2, label=team_label.title())
        ax.fill(angles, vals, color=colour, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Home vs Away — Average Tactical Profile",
                 pad=20, fontsize=11, fontweight="medium", color="#222")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
#  Main entry-point
# ─────────────────────────────────────────────────────────────────

def generate_radars(
    tracks_csv:  str,
    teams_csv:   str,
    control_csv: str,
    clutch_csv:  str,
    out_dir:     str,
    top_n:       int = 5,
) -> pd.DataFrame:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  goalX — Tactical Radar")
    print(f"  {'─' * 40}")

    tracks  = pd.read_csv(tracks_csv)
    teams   = pd.read_csv(teams_csv)
    control = pd.read_csv(control_csv)
    clutch  = pd.read_csv(clutch_csv) if clutch_csv else pd.DataFrame()

    print(f"  ✔  Computing per-player stats …")
    stats_df = _compute_player_stats(tracks, teams, control, clutch)

    stats_csv = out_dir / "player_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"  💾 Player stats → {stats_csv}")

    # ── Normalise for radar ───────────────────────────────────────
    norm_df = _normalise(stats_df)

    # ── Team comparison chart ─────────────────────────────────────
    comp_path = out_dir / "radar_team_comparison.png"
    _team_comparison_chart(norm_df, comp_path)
    print(f"  🖼️  Team radar → {comp_path}")

    # ── Individual player radars (top_n by clutch_score + distance) ──
    top_ids = (
        stats_df
        .assign(_rank=stats_df["clutch_score"] + stats_df["distance_m"] / stats_df["distance_m"].max())
        .nlargest(top_n, "_rank")["track_id"]
        .tolist()
    )

    print(f"  Rendering individual radars for top {top_n} players …")
    for tid in tqdm(top_ids, desc="Radars"):
        row    = norm_df[norm_df["track_id"] == tid]
        if row.empty:
            continue
        row    = row.iloc[0]
        team   = str(row["team"])
        colour = TEAM_COLOURS.get(team, "#555")
        vals   = [float(row[s]) for s in RADAR_STATS]

        raw = stats_df[stats_df["track_id"] == tid].iloc[0]
        subtitle = (
            f"track {tid}  ·  {team}  ·  "
            f"{raw['distance_m']:.0f}m  ·  "
            f"xCS={raw['clutch_score']:.3f}"
        )

        _radar_chart(
            values   = vals,
            labels   = STAT_LABELS,
            title    = subtitle,
            colour   = colour,
            out_path = out_dir / f"radar_{int(tid):04d}.png",
        )

    print(f"\n  📊 Stats summary:")
    for team in ("home", "away"):
        subset = stats_df[stats_df["team"] == team]
        if subset.empty:
            continue
        print(f"     {team:<6}: {len(subset)} players  |  "
              f"avg dist={subset['distance_m'].mean():.0f}m  |  "
              f"avg clutch={subset['clutch_score'].mean():.3f}")

    print(f"\n  ✅  Tactical radars complete.")
    print(f"      All PNGs saved to: {out_dir}\n")
    return stats_df


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate per-player and team radar charts from goalX pipeline outputs."
    )
    p.add_argument("--tracks",  required=True)
    p.add_argument("--teams",   required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--clutch",  default=None,
                   help="clutch_scores.csv  (optional — zeros used if absent)")
    p.add_argument("--out-dir", default="outputs/radar")
    p.add_argument("--top-n",   type=int, default=5,
                   help="Number of individual player radars to generate  (default: 5)")
    return p.parse_args()


# if __name__ == "__main__":
#     args = _parse_args()
#     generate_radars(
#         tracks_csv  = args.tracks,
#         teams_csv   = args.teams,
#         control_csv = args.control,
#         clutch_csv  = args.clutch,
#         out_dir     = args.out_dir,
#         top_n       = args.top_n,
#     )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="goalX Tactical Radar Generation")
    
    # Arguments sent by run_goalx.py
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--clutch", required=True)
    parser.add_argument("--teams", required=True)
    parser.add_argument("--out-dir", required=True)
    
    # Making --control optional to match the orchestrator's current handshake
    parser.add_argument("--control", required=False)
    
    args = parser.parse_args()
    
    generate_radars(
        tracks_csv = args.tracks,
        clutch_csv = args.clutch,
        teams_csv = args.teams,
        out_dir = args.out_dir,
        control_csv = args.control
    )