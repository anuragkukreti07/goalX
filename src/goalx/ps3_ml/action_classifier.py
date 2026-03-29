"""
action_classifier.py  —  PS3 Step 4
──────────────────────────────────────
Per-frame action classification for each tracked player.

Action taxonomy
───────────────
  SHOT       : player accelerates toward goal, ball in close proximity
  DRIBBLE    : player in possession, sustained forward movement
  PRESS      : opponent has ball, player closing at high speed
  TACKLE     : high-speed approach + possession change nearby
  PASS       : player releases ball (possession transfer event)
  CARRY      : player has ball, moving slowly / changing direction
  IDLE       : low speed, no ball interaction
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    import joblib
    _SK_AVAILABLE = True
except ImportError:
    _SK_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

PITCH_SCALE  = 10.0       # px/m
FPS          = 25.0       # frames/second
GOAL_R_X     = 1050.0     # right goal x (px)
GOAL_L_X     = 0.0        # left goal x (px)
GOAL_Y       = 340.0      # goal centre y (px)

# Physics thresholds
SPEED_IDLE   = 0.5        # m/s   — below this = idle
SPEED_WALK   = 2.0        # m/s
SPEED_RUN    = 4.5        # m/s
SPEED_SPRINT = 7.0        # m/s
BALL_POSS_M  = 2.5        # m     — within this = possession candidate
PRESS_RANGE  = 5.0        # m     — pressing radius
ACCEL_SHOT   = 2.0        # m/s²  — minimum acceleration for a shot action

ACTION_LABELS = ["IDLE", "CARRY", "DRIBBLE", "PASS",
                 "SHOT", "PRESS", "TACKLE"]

_DARK_BG = "#0e1117"
_SURFACE = "#1a1d23"
_TEXT    = "#c8c8d0"

ACTION_COLORS = {
    "IDLE":    "#6b7280",
    "CARRY":   "#3b82f6",
    "DRIBBLE": "#8b5cf6",
    "PASS":    "#10b981",
    "SHOT":    "#f59e0b",
    "PRESS":   "#ef4444",
    "TACKLE":  "#ec4899",
}


# ─────────────────────────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────────────────────────

def _px_to_m(v: float) -> float:
    return v / PITCH_SCALE


def engineer_features(tracks: pd.DataFrame,
                       ball: pd.DataFrame,
                       team_map: dict[int, str]
                       ) -> pd.DataFrame:
    """
    Compute per-frame kinematic features for every player.
    Returns a DataFrame with feature columns + frame_id, track_id.
    """
    tracks = tracks.sort_values(["track_id", "frame_id"]).copy()
    tracks["px_m"] = tracks["pitch_x"].apply(_px_to_m)
    tracks["py_m"] = tracks["pitch_y"].apply(_px_to_m)

    # Per-player velocity + acceleration (finite difference)
    for col in ["px_m", "py_m"]:
        tracks[f"d{col}"] = tracks.groupby("track_id")[col].diff().fillna(0)

    tracks["speed_ms"] = np.sqrt(
        (tracks["dpx_m"] * FPS) ** 2 + (tracks["dpy_m"] * FPS) ** 2
    )
    tracks["accel_ms2"] = (
        tracks.groupby("track_id")["speed_ms"].diff().fillna(0).abs() * FPS
    )

    # Direction (angle of velocity vector)
    tracks["dir_rad"] = np.arctan2(tracks["dpy_m"], tracks["dpx_m"])
    tracks["dir_change_rad"] = (
        tracks.groupby("track_id")["dir_rad"]
        .diff().fillna(0).abs()
    )

    # Ball proximity
    ball_pos = (ball[["frame_id", "pitch_x", "pitch_y"]]
                .rename(columns={"pitch_x": "bx", "pitch_y": "by"})
                .drop_duplicates("frame_id"))

    merged = tracks.merge(ball_pos, on="frame_id", how="left")
    merged["ball_dist_m"] = np.sqrt(
        ((merged["pitch_x"] - merged["bx"].fillna(9999)) / PITCH_SCALE) ** 2 +
        ((merged["pitch_y"] - merged["by"].fillna(9999)) / PITCH_SCALE) ** 2
    )
    merged["ball_dist_delta"] = (
        merged.groupby("track_id")["ball_dist_m"].diff().fillna(0)
    )

    # Nearest player to ball (has possession)
    poss_map = {}
    for fid, grp in merged.groupby("frame_id"):
        min_idx = grp["ball_dist_m"].idxmin()
        if grp.loc[min_idx, "ball_dist_m"] < BALL_POSS_M:
            poss_map[fid] = int(grp.loc[min_idx, "track_id"])
    merged["has_possession"] = merged.apply(
        lambda r: int(poss_map.get(int(r["frame_id"]), -1) == r["track_id"]),
        axis=1,
    )

    # Opponents within PRESS_RANGE
    def _count_opponents(row):
        fid  = int(row["frame_id"])
        tid  = int(row["track_id"])
        team = team_map.get(tid, "unknown")
        ft   = merged[merged["frame_id"] == fid]
        opp  = ft[ft["track_id"].apply(lambda t: team_map.get(int(t), "x")) != team]
        if opp.empty:
            return 0
        dx = (opp["pitch_x"] - row["pitch_x"]) / PITCH_SCALE
        dy = (opp["pitch_y"] - row["pitch_y"]) / PITCH_SCALE
        return int((np.sqrt(dx**2 + dy**2) < PRESS_RANGE).sum())

    # Vectorise opponent count per frame (faster than apply)
    opp_counts: dict[tuple, int] = {}
    for fid, grp in merged.groupby("frame_id"):
        for _, row in grp.iterrows():
            tid  = int(row["track_id"])
            team = team_map.get(tid, "unknown")
            opp  = grp[grp["track_id"].apply(
                lambda t: team_map.get(int(t), "x")) != team]
            if not opp.empty:
                dx = (opp["pitch_x"] - row["pitch_x"]) / PITCH_SCALE
                dy = (opp["pitch_y"] - row["pitch_y"]) / PITCH_SCALE
                opp_counts[(fid, tid)] = int((np.sqrt(dx**2+dy**2)<PRESS_RANGE).sum())
            else:
                opp_counts[(fid, tid)] = 0

    merged["n_opponents_5m"] = merged.apply(
        lambda r: opp_counts.get((int(r["frame_id"]), int(r["track_id"])), 0),
        axis=1,
    )

    # Goal angle cosine: angle player→ball→nearest goal
    goal_x = merged.apply(
        lambda r: GOAL_R_X if r.get("px_m", 0) * PITCH_SCALE < 525 else GOAL_L_X,
        axis=1,
    )
    pb_x = (merged["bx"].fillna(merged["pitch_x"]) - merged["pitch_x"]) / PITCH_SCALE
    pb_y = (merged["by"].fillna(merged["pitch_y"]) - merged["pitch_y"]) / PITCH_SCALE
    pg_x = (goal_x - merged["pitch_x"]) / PITCH_SCALE
    pg_y = (GOAL_Y  - merged["pitch_y"]) / PITCH_SCALE
    dot  = pb_x * pg_x + pb_y * pg_y
    mag  = (np.sqrt(pb_x**2 + pb_y**2) * np.sqrt(pg_x**2 + pg_y**2)).replace(0, 1)
    merged["goal_angle_cos"] = (dot / mag).clip(-1, 1)

    feature_cols = [
        "frame_id", "track_id",
        "speed_ms", "accel_ms2",
        "ball_dist_m", "ball_dist_delta",
        "goal_angle_cos", "n_opponents_5m",
        "dir_change_rad", "has_possession",
    ]

    return merged[feature_cols].fillna(0)


# ─────────────────────────────────────────────────────────────────
#  Rule-based classifier
# ─────────────────────────────────────────────────────────────────

def classify_rule_based(features: pd.DataFrame,
                         team_map: dict[int, str],
                         pass_frames: set[int]
                         ) -> pd.DataFrame:
    """
    Deterministic rule-based action classifier.
    Returns features DataFrame with added 'action' and 'confidence' columns.
    """
    actions     = []
    confidences = []

    for _, row in features.iterrows():
        spd   = float(row["speed_ms"])
        acc   = float(row["accel_ms2"])
        bdist = float(row["ball_dist_m"])
        bdelt = float(row["ball_dist_delta"])
        gocos = float(row["goal_angle_cos"])
        nopp  = int(row["n_opponents_5m"])
        poss  = int(row["has_possession"])
        fid   = int(row["frame_id"])

        # Priority order: highest certainty rules first

        if fid in pass_frames and poss:
            actions.append("PASS");  confidences.append(0.80)

        elif (poss and spd > SPEED_RUN and
              acc > ACCEL_SHOT and gocos > 0.7):
            actions.append("SHOT");  confidences.append(0.75)

        elif (not poss and spd > SPEED_RUN and
              bdelt < -0.5 and nopp >= 1):
            actions.append("PRESS");  confidences.append(0.72)

        elif (not poss and spd > SPEED_SPRINT and
              bdelt < -1.0 and bdist < PRESS_RANGE):
            actions.append("TACKLE");  confidences.append(0.68)

        elif poss and spd > SPEED_RUN:
            actions.append("DRIBBLE");  confidences.append(0.65)

        elif poss and SPEED_IDLE < spd <= SPEED_RUN:
            actions.append("CARRY");  confidences.append(0.60)

        else:
            actions.append("IDLE");  confidences.append(0.55)

    result = features.copy()
    result["action"]     = actions
    result["confidence"] = confidences
    return result


# ─────────────────────────────────────────────────────────────────
#  ML classifier (optional)
# ─────────────────────────────────────────────────────────────────

_FEAT_COLS = [
    "speed_ms", "accel_ms2", "ball_dist_m", "ball_dist_delta",
    "goal_angle_cos", "n_opponents_5m", "dir_change_rad", "has_possession",
]


def train_and_apply_rf(features: pd.DataFrame,
                        labeled_csv: Path,
                        out_dir: Path) -> pd.DataFrame:
    """
    Train a Random Forest classifier on labeled data and apply to features.
    """
    if not _SK_AVAILABLE:
        print("  ⚠  scikit-learn not available. Falling back to rule-based.")
        return pd.DataFrame()

    labeled = pd.read_csv(labeled_csv)
    missing = set(_FEAT_COLS + ["action"]) - set(labeled.columns)
    if missing:
        print(f"  ⚠  Labeled CSV missing columns: {missing}")
        return pd.DataFrame()

    X_train = labeled[_FEAT_COLS].values
    le      = LabelEncoder()
    y_train = le.fit_transform(labeled["action"].values)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"  RF cross-val F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X_train, y_train)
    joblib.dump({"model": clf, "label_encoder": le},
                out_dir / "action_rf_model.pkl")
    print(f"  ✔  RF model saved → {out_dir}/action_rf_model.pkl")

    # Apply to unlabeled features
    X_infer = features[_FEAT_COLS].fillna(0).values
    y_pred  = clf.predict(X_infer)
    y_prob  = clf.predict_proba(X_infer).max(axis=1)

    result = features.copy()
    result["action"]     = le.inverse_transform(y_pred)
    result["confidence"] = y_prob
    return result


# ─────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────

def make_action_timeline(classified: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar chart: action distribution across time (binned by minute)."""
    classified = classified.copy()
    classified["minute"] = (classified["frame_id"] / FPS / 60).astype(int)

    pivot = (classified.groupby(["minute", "action"])
             .size().unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_SURFACE)

    bottom = np.zeros(len(pivot))
    for action in ACTION_LABELS:
        if action not in pivot.columns:
            continue
        vals = pivot[action].values
        ax.bar(pivot.index, vals, bottom=bottom, width=0.8,
                color=ACTION_COLORS.get(action, "#999"),
                label=action, zorder=3)
        bottom += vals

    ax.set_xlabel("Match minute", color=_TEXT)
    ax.set_ylabel("Action count", color=_TEXT)
    ax.set_title("Action distribution over time", color="#e8e8f0",
                 fontsize=13, weight="bold")
    ax.tick_params(colors=_TEXT)
    ax.grid(axis="y", color=_SURFACE, alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper right", facecolor=_SURFACE,
              edgecolor=_TEXT, labelcolor=_TEXT, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Action timeline → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class ActionClassifier:
    def __init__(self, tracks_csv: Path, teams_csv: Path,
                 ball_csv: Path, events_csv: Path,
                 out_dir: Path, train_csv: Path | None = None):
        self.tracks_csv = tracks_csv
        self.teams_csv  = teams_csv
        self.ball_csv   = ball_csv
        self.events_csv = events_csv
        self.out_dir    = out_dir
        self.train_csv  = train_csv

    def run(self) -> None:
        print(f"\n  goalX PS3 — Action Classifier")
        print(f"  {'─'*42}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        tracks = pd.read_csv(self.tracks_csv)
        teams  = pd.read_csv(self.teams_csv)
        ball   = (pd.read_csv(self.ball_csv)
                  if self.ball_csv.exists() else pd.DataFrame())
        events = pd.read_csv(self.events_csv)

        for df in [tracks, events]:
            if "frame_id" not in df.columns and "frame" in df.columns:
                df.rename(columns={"frame": "frame_id"}, inplace=True)

        team_map: dict[int, str] = dict(
            zip(teams["track_id"].astype(int), teams["team"].astype(str))
        )

        # ── FIX: Safely extract pass frames ──
        if "event_type" in events.columns:
            pass_frames = set(
                events[events["event_type"] == "possession"]["frame_id"]
                .dropna().astype(int).tolist()
            )
        else:
            pass_frames = set()

        print("  Engineering kinematic features…")
        features = engineer_features(tracks, ball, team_map)
        print(f"  Features: {len(features):,} rows across "
              f"{features['frame_id'].nunique():,} frames")

        # ── Classify ───────────────────────────────────────────
        if self.train_csv and Path(self.train_csv).exists():
            print(f"\n  ML mode — training on {self.train_csv}")
            classified = train_and_apply_rf(features, Path(self.train_csv),
                                             self.out_dir)
            if classified.empty:
                print("  Falling back to rule-based.")
                classified = classify_rule_based(features, team_map, pass_frames)
        else:
            print("  Rule-based classification (no training data provided)")
            classified = classify_rule_based(features, team_map, pass_frames)

        # ── Save ──────────────────────────────────────────────
        out_csv = self.out_dir / "actions.csv"
        classified.to_csv(out_csv, index=False)
        print(f"\n  ✔  Actions CSV → {out_csv}")

        # ── Per-player summary ─────────────────────────────────
        summary = (classified.groupby(["track_id", "action"])
                   .size().unstack(fill_value=0))
        summary["dominant_action"] = summary.idxmax(axis=1)
        summary.to_csv(self.out_dir / "action_summary.csv")
        print(f"  ✔  Summary → {self.out_dir}/action_summary.csv")

        # ── Print distribution ────────────────────────────────
        print(f"\n  Action distribution:")
        dist = classified["action"].value_counts()
        for act, cnt in dist.items():
            bar = "█" * int(cnt / max(dist) * 30)
            print(f"  {act:<10} {cnt:>6}  {bar}")

        # ── Figure ─────────────────────────────────────────────
        make_action_timeline(classified, self.out_dir / "action_timeline.png")

        print(f"\n  ✅  Classification complete → {self.out_dir}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Action classifier for goalX PS3."
    )
    p.add_argument("--tracks",  default="outputs/smoothed_tracks.csv")
    p.add_argument("--teams",   default="outputs/team_labels.csv")
    p.add_argument("--ball",    default="outputs/ball_trajectory/interpolated_ball.csv")
    p.add_argument("--events",  default="outputs/events.csv")
    p.add_argument("--out-dir", default="outputs/actions")
    p.add_argument("--train",   default=None,
                   help="Labeled CSV for RF training (optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ActionClassifier(
        tracks_csv = Path(args.tracks),
        teams_csv  = Path(args.teams),
        ball_csv   = Path(args.ball) if args.ball else Path(""),
        events_csv = Path(args.events),
        out_dir    = Path(args.out_dir),
        train_csv  = args.train,
    ).run()