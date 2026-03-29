"""
offside_detector.py  —  PS4 Step 1
────────────────────────────────────
Geometric offside detection from calibrated player tracking data.

Why this file exists
─────────────────────
Every existing piece of infrastructure is already in place: we have
pitch-space coordinates for every player (project_tracks.py), team labels
(team_classification.py), pass events (pass_network.py), and a calibrated
homography. Offside detection is a purely geometric consequence of those
inputs.

Algorithm  (Laws of the Game, Law 11)
──────────────────────────────────────
A player is in an offside POSITION if they are:
  - In the opponent's half, AND
  - Nearer to the opponent's goal line than BOTH the ball AND the
    second-last defender (the last defender being the goalkeeper)

A player is guilty of an OFFSIDE OFFENCE only when they:
  - Are in an offside position at the moment the ball is last played
    by a teammate (a "pass moment")
  - Are active in play (receiving, seeking to gain advantage)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

PITCH_SCALE  = 10.0        # px / metre — must match draw_pitch.py
PITCH_W_PX   = 1050        # canvas width
PITCH_H_PX   = 680         # canvas height
PITCH_W_M    = 105.0
PITCH_H_M    = 68.0
FPS          = 25.0
OFFSIDE_MIN_DEPTH_GAP_M = 0.0   # must be strictly ahead (≥ 0 m)
ACTIVE_PLAY_RADIUS_M    = 10.0  # player must be within this of ball to be "active"

_DARK  = "#0e1117"; _SURF = "#1a1d23"; _BORD = "#2e3140"; _TEXT = "#c8c8d0"
_BLUE  = "#3b82f6"; _RED  = "#ef4444"; _AMBER = "#f5a623"; _GREEN = "#3dbf7a"

plt.rcParams.update({
    "figure.facecolor": _DARK, "axes.facecolor": "#1a3a1a",
    "text.color": _TEXT, "axes.labelcolor": _TEXT,
    "xtick.color": "#9090a0", "ytick.color": "#9090a0",
    "grid.color": _BORD, "font.size": 9,
})


# ─────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────

def _px_to_m(v: float) -> float:
    return v / PITCH_SCALE


def _get_positions(tracks: pd.DataFrame, frame_id: int) -> pd.DataFrame:
    return tracks[tracks["frame_id"] == frame_id].copy()


def _get_ball(ball: pd.DataFrame, frame_id: int) -> tuple[float, float] | None:
    row = ball[ball["frame_id"] == frame_id]
    # FIX: Safely check for empty rows and NaN ball coordinates
    if row.empty or pd.isna(row.iloc[0].get("pitch_x")):
        return None
    return float(row.iloc[0]["pitch_x"]), float(row.iloc[0]["pitch_y"])


# ─────────────────────────────────────────────────────────────────
#  Offside geometry
# ─────────────────────────────────────────────────────────────────

@dataclass
class OffsideCheck:
    """Result of an offside check at one pass moment."""
    frame_id:           int
    attacking_team:     str
    ball_x_m:           float
    ball_y_m:           float
    offside_line_x_m:   float          # depth of second-last defender
    flagged_players:    list[int] = field(default_factory=list)   # track_ids
    flagged_depths:     list[float] = field(default_factory=list) # gap in metres
    is_attacking_right: bool = True  # direction of attack


def _attacking_direction(team: str, tracks: pd.DataFrame,
                          team_map: dict[int, str]) -> bool:
    """
    Infer whether `team` is attacking toward the right goal (x → PITCH_W_PX).
    Uses the mean x-position of team players across all frames.
    """
    team_ids = [tid for tid, t in team_map.items() if t == team]
    team_tracks = tracks[tracks["track_id"].isin(team_ids)]
    if team_tracks.empty:
        return True
    mean_x = team_tracks["pitch_x"].mean()
    return bool(mean_x < PITCH_W_PX / 2)   # if they're on the left, they attack right


def check_offside_at_frame(
    frame_id: int,
    attacking_team: str,
    defending_team: str,
    tracks: pd.DataFrame,
    ball: pd.DataFrame,
    team_map: dict[int, str],
    attacking_right: bool,
) -> OffsideCheck | None:
    """
    Run the geometric offside check for one pass moment.

    Returns an OffsideCheck (possibly with empty flagged_players) or None
    if there's insufficient data.
    """
    ball_pos = _get_ball(ball, frame_id)
    if ball_pos is None:
        return None

    ball_x_m = _px_to_m(ball_pos[0])
    ball_y_m = _px_to_m(ball_pos[1])

    frame_players = _get_positions(tracks, frame_id)
    if frame_players.empty:
        return None

    def _team_depth_x(player_row) -> float:
        """Depth in metres measured from the defending goal line."""
        x_m = _px_to_m(float(player_row["pitch_x"]))
        if attacking_right:
            return x_m                                  # 0 = left goal line
        else:
            return PITCH_W_M - x_m                      # 0 = right goal line

    # Split into attacking and defending players
    att_ids = [tid for tid, t in team_map.items() if t == attacking_team]
    def_ids = [tid for tid, t in team_map.items() if t == defending_team]

    attackers  = frame_players[frame_players["track_id"].isin(att_ids)]
    defenders  = frame_players[frame_players["track_id"].isin(def_ids)]

    if len(defenders) < 2:
        return None   # need at least 2 defenders (GK + 1) for the rule

    # Compute depth of each defender from their goal line
    def_depths = defenders.apply(_team_depth_x, axis=1).values
    def_depths_sorted = np.sort(def_depths)         # ascending
    # Second-last defender is at index -2 (last = GK, nearest to goal)
    second_last_depth = def_depths_sorted[-2]

    # Ball depth from defending goal line
    ball_depth_m = _px_to_m(ball_pos[0]) if attacking_right else PITCH_W_M - _px_to_m(ball_pos[0])
    offside_line_depth = min(second_last_depth, ball_depth_m)

    # Convert offside_line depth back to pitch x coordinate
    if attacking_right:
        offside_line_x_m = offside_line_depth
    else:
        offside_line_x_m = PITCH_W_M - offside_line_depth

    flagged_players = []
    flagged_depths  = []

    for _, att_row in attackers.iterrows():
        att_depth = _team_depth_x(att_row)
        att_x_m   = _px_to_m(float(att_row["pitch_x"]))
        att_y_m   = _px_to_m(float(att_row["pitch_y"]))

        # Must be in the opponent's half
        if attacking_right and att_x_m < PITCH_W_M / 2:
            continue
        if not attacking_right and att_x_m > PITCH_W_M / 2:
            continue

        # Must be ahead of offside line
        depth_gap = att_depth - offside_line_depth
        if depth_gap <= OFFSIDE_MIN_DEPTH_GAP_M:
            continue

        # Must be "active" — within ACTIVE_PLAY_RADIUS_M of ball
        dist_to_ball = np.sqrt((att_x_m - ball_x_m) ** 2 +
                                (att_y_m - ball_y_m) ** 2)
        if dist_to_ball > ACTIVE_PLAY_RADIUS_M:
            continue

        flagged_players.append(int(att_row["track_id"]))
        flagged_depths.append(round(depth_gap, 3))

    return OffsideCheck(
        frame_id         = frame_id,
        attacking_team   = attacking_team,
        ball_x_m         = ball_x_m,
        ball_y_m         = ball_y_m,
        offside_line_x_m = offside_line_x_m,
        flagged_players  = flagged_players,
        flagged_depths   = flagged_depths,
        is_attacking_right = attacking_right,
    )


# ─────────────────────────────────────────────────────────────────
#  Visualisation — one frame
# ─────────────────────────────────────────────────────────────────

def draw_offside_frame(check: OffsideCheck,
                        frame_id: int,
                        tracks: pd.DataFrame,
                        team_map: dict[int, str],
                        pitch_img: np.ndarray,
                        out_path: Path) -> None:
    """
    Draw a static offside diagram:
      - Pitch canvas background
      - All players as colored dots
      - Offside line as a vertical red dashed line
      - Flagged players highlighted in red with gap annotation
      - Ball position
    """
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_DARK)

    # Pitch background
    pitch_rgb = cv2.cvtColor(pitch_img, cv2.COLOR_BGR2RGB)
    ax.imshow(pitch_rgb, extent=[0, PITCH_W_M, PITCH_H_M, 0],
              alpha=0.55, zorder=0)
    ax.set_xlim(0, PITCH_W_M);  ax.set_ylim(PITCH_H_M, 0)

    frame_players = _get_positions(tracks, frame_id)

    for _, row in frame_players.iterrows():
        tid  = int(row["track_id"])
        if tid < 0:
            continue
        x_m = _px_to_m(float(row["pitch_x"]))
        y_m = _px_to_m(float(row["pitch_y"]))

        is_flagged = tid in check.flagged_players
        team       = team_map.get(tid, "ref")
        base_color = _BLUE if team == "home" else _RED if team == "away" else "#94a3b8"
        color      = "#ff2222" if is_flagged else base_color
        size       = 120 if is_flagged else 60
        zorder     = 5 if is_flagged else 3

        ax.scatter(x_m, y_m, s=size, c=color, zorder=zorder,
                   edgecolors="white", linewidths=0.8)
        ax.text(x_m, y_m - 1.2, str(tid),
                ha="center", fontsize=6.5, color="white", zorder=6)

        if is_flagged:
            idx = check.flagged_players.index(tid)
            gap = check.flagged_depths[idx]
            ax.annotate(f"OFFSIDE +{gap:.1f}m",
                        xy=(x_m, y_m),
                        xytext=(x_m + 2.5, y_m - 2.5),
                        fontsize=7.5, color="#ff4444", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#ff4444",
                                        lw=0.8))

    # Ball
    bx = check.ball_x_m;  by = check.ball_y_m
    ax.scatter(bx, by, s=80, c=_AMBER, zorder=7,
               edgecolors="white", linewidths=1.0, marker="*")

    # Offside line (vertical)
    ax.axvline(x=check.offside_line_x_m, color="#ff3333", lw=1.5,
               linestyle="--", alpha=0.9, zorder=4,
               label=f"Offside line ({check.offside_line_x_m:.1f} m)")

    # Labels
    attack_dir = "→" if check.is_attacking_right else "←"
    n_flagged  = len(check.flagged_players)
    verdict    = f"{n_flagged} OFFSIDE" if n_flagged else "ONSIDE"
    color_v    = "#ff3333" if n_flagged else _GREEN

    ax.set_title(f"Frame {frame_id}  |  {check.attacking_team.upper()} attacking {attack_dir}  "
                 f"|  {verdict}",
                 color=color_v, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Pitch length (m)");  ax.set_ylabel("Pitch width (m)")
    ax.legend(loc="lower right", facecolor=_SURF, edgecolor=_TEXT,
              labelcolor=_TEXT, fontsize=8)
    ax.grid(True, alpha=0.15, zorder=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class OffsideDetector:
    def __init__(self, tracks_csv: Path, teams_csv: Path, ball_csv: Path,
                 passes_csv: Path, pitch_path: Path, out_dir: Path):
        self.tracks_csv = tracks_csv
        self.teams_csv  = teams_csv
        self.ball_csv   = ball_csv
        self.passes_csv = passes_csv
        self.pitch_path = pitch_path
        self.out_dir    = out_dir

    def _load(self):
        self.tracks = pd.read_csv(self.tracks_csv)
        self.teams  = pd.read_csv(self.teams_csv)
        self.ball   = (pd.read_csv(self.ball_csv)
                       if self.ball_csv.exists() else pd.DataFrame())
        self.passes = (pd.read_csv(self.passes_csv)
                       if self.passes_csv.exists() else pd.DataFrame())
        self.pitch  = cv2.imread(str(self.pitch_path))

        for df in [self.tracks, self.ball, self.passes]:
            if not df.empty and "frame" in df.columns and "frame_id" not in df.columns:
                df.rename(columns={"frame": "frame_id"}, inplace=True)

        self.team_map: dict[int, str] = dict(
            zip(self.teams["track_id"].astype(int),
                self.teams["team"].astype(str))
        )

    def run(self):
        print(f"\n  goalX PS4 — Offside Detector")
        print(f"  {'─'*44}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._load()

        # Identify pass frames — when possession transitions between teams
        if not self.passes.empty and "frame_start" in self.passes.columns:
            pass_frames = self.passes["frame_start"].astype(int).tolist()
            pass_teams  = self.passes["passer_team"].tolist()
        else:
            # Fallback: sample every 50 frames
            all_frames = sorted(self.tracks["frame_id"].unique())
            pass_frames = all_frames[::50]
            pass_teams  = ["home"] * len(pass_frames)
            print("  ⚠  No pass CSV found — sampling every 50 frames.")

        # Infer attack directions
        att_right_home = _attacking_direction("home", self.tracks, self.team_map)
        att_right_away = not att_right_home

        results: list[dict] = []
        n_checked = 0; n_flagged_total = 0
        frames_dir = self.out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        for frame_id, passer_team in zip(pass_frames, pass_teams):
            if passer_team not in ("home", "away"):
                continue

            # The RECEIVING team (just received ball) might be offside
            attacking_team = passer_team
            defending_team = "away" if passer_team == "home" else "home"
            attacking_right = att_right_home if attacking_team == "home" else att_right_away

            check = check_offside_at_frame(
                frame_id, attacking_team, defending_team,
                self.tracks, self.ball, self.team_map, attacking_right,
            )
            if check is None:
                continue

            n_checked += 1
            n_offside  = len(check.flagged_players)

            results.append({
                "frame_id":          frame_id,
                "attacking_team":    attacking_team,
                "offside_line_x_m":  round(check.offside_line_x_m, 3),
                "ball_x_m":          round(check.ball_x_m, 3),
                "flagged_players":   str(check.flagged_players),
                "flagged_depths_m":  str(check.flagged_depths),
                "n_offside":         n_offside,
            })

            if n_offside > 0:
                n_flagged_total += 1
                if self.pitch is not None:
                    draw_offside_frame(
                        check, frame_id, self.tracks, self.team_map,
                        self.pitch,
                        frames_dir / f"offside_{frame_id:06d}.png",
                    )

        # Save results
        df_out = pd.DataFrame(results)
        df_out.to_csv(self.out_dir / "offside_events.csv", index=False)
        print(f"  Pass moments checked : {n_checked}")
        print(f"  Offside incidents    : {n_flagged_total}")
        print(f"  CSV → {self.out_dir}/offside_events.csv")
        if n_flagged_total:
            print(f"  Frames → {frames_dir}/  ({n_flagged_total} diagrams)")

        # Summary figure
        if not df_out.empty:
            self._make_summary_figure(df_out)

        print(f"\n  ✅  Offside detection complete → {self.out_dir}\n")

    def _make_summary_figure(self, df: pd.DataFrame):
        flagged = df[df["n_offside"] > 0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Offside detection summary", color="#e8e8f0",
                     fontsize=13, fontweight="bold")
        fig.patch.set_facecolor(_DARK)

        # Timeline
        ax = axes[0]
        ax.scatter(df["frame_id"], df["n_offside"],
                   s=20, c=_BLUE, alpha=0.5, zorder=3)
        ax.scatter(flagged["frame_id"], flagged["n_offside"],
                   s=50, c=_RED, zorder=4, label="Flagged")
        ax.set_xlabel("Frame"); ax.set_ylabel("Players offside")
        ax.set_title("Offside incidents over time")
        ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        if not flagged.empty:
            ax.legend(facecolor=_SURF, edgecolor=_TEXT, labelcolor=_TEXT)

        # Offside line x distribution
        ax = axes[1]
        ax.hist(df["offside_line_x_m"], bins=20, color=_AMBER,
                edgecolor=_DARK, linewidth=0.5, zorder=3)
        ax.axvline(PITCH_W_M / 2, color=_RED, lw=1, linestyle="--",
                   label="Halfway line")
        ax.set_xlabel("Offside line position (m)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of offside line positions")
        ax.legend(facecolor=_SURF, edgecolor=_TEXT, labelcolor=_TEXT)
        ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        fig.savefig(self.out_dir / "offside_summary.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✔  Summary figure → {self.out_dir}/offside_summary.png")


def _parse_args():
    p = argparse.ArgumentParser(description="Geometric offside detector (goalX PS4).")
    # FIX: Use smart defaults pointing to the existing output pipeline
    p.add_argument("--tracks",   default="outputs/smoothed_tracks.csv")
    p.add_argument("--teams",    default="outputs/team_labels.csv")
    p.add_argument("--ball",     default="outputs/ball_trajectory/interpolated_ball.csv")
    p.add_argument("--passes",   default="outputs/pass_network/pass_network.csv")
    p.add_argument("--pitch",    default="data/pitch_map.png")
    p.add_argument("--out-dir",  default="outputs/offside")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    OffsideDetector(
        tracks_csv = Path(args.tracks),
        teams_csv  = Path(args.teams),
        ball_csv   = Path(args.ball)  if args.ball  else Path(""),
        passes_csv = Path(args.passes) if args.passes else Path(""),
        pitch_path = Path(args.pitch),
        out_dir    = Path(args.out_dir),
    ).run()