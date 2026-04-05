"""
action_classifier.py  —  PS3 Step 4
──────────────────────────────────────
Per-frame action classification from kinematic features.

FIXES
─────────────────────────────────
FIX 1 — Speed thresholds lowered for broadcast/corner view (CRITICAL)
  Introduced a VIEW_MODE parameter. 'broadcast' uses standard speeds,
  while 'corner' uses halved thresholds matching the pixel-speed
  distribution of end-zone/corner cameras (like SNMOT-116).

FIX 2 — SHOT detection threshold lowered
  Original required speed > 4.5 m/s. Corner view requires > 1.5 m/s,
  preventing missed shot events.

FIX 3 — ball track (track_id < 0) excluded from classification
  Ball rows were getting classified as IDLE, inflating the count.

FIX 4 — IDLE label uses all-else-idle logic
  Priority order is strictly enforced to catch high-intensity actions first.
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
#  Threshold sets per view mode
# ─────────────────────────────────────────────────────────────────

_THRESHOLDS = {
    "broadcast": {   # full-pitch camera — original values
        "SPEED_IDLE":   0.50,
        "SPEED_WALK":   2.00,
        "SPEED_RUN":    4.50,
        "SPEED_SPRINT": 7.00,
        "ACCEL_SHOT":   2.00,
        "GOAL_COS_SHOT":0.70,
        "BALL_POSS_M":  2.50,
        "PRESS_RANGE":  5.00,
    },
    "corner": {      # corner/end-zone camera — halved thresholds
        "SPEED_IDLE":   0.20,
        "SPEED_WALK":   0.80,
        "SPEED_RUN":    1.80,   # Matches observed p90 speed
        "SPEED_SPRINT": 3.00,
        "ACCEL_SHOT":   0.80,   # Lowered: shot still causes detectable accel
        "GOAL_COS_SHOT":0.50,   # Wider angle acceptance
        "BALL_POSS_M":  3.00,   # Slightly wider for pixel scale
        "PRESS_RANGE":  5.00,
    },
}

PITCH_SCALE = 10.0
FPS         = 25.0
GOAL_R_X    = 1050.0
GOAL_L_X    = 0.0
GOAL_Y      = 340.0

ACTION_LABELS = ["IDLE", "CARRY", "DRIBBLE", "PASS", "SHOT", "PRESS", "TACKLE"]

ACTION_COLORS = {
    "IDLE":    "#6b7280", "CARRY":   "#3b82f6", "DRIBBLE": "#8b5cf6",
    "PASS":    "#10b981", "SHOT":    "#f59e0b", "PRESS":   "#ef4444",
    "TACKLE":  "#ec4899",
}

_DARK_BG = "#0e1117"
_SURFACE = "#1a1d23"
_TEXT    = "#c8c8d0"
plt.rcParams.update({"figure.facecolor": _DARK_BG, "axes.facecolor": _SURFACE,
                     "text.color": _TEXT, "font.size": 10})


# ─────────────────────────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────────────────────────

def engineer_features(tracks: pd.DataFrame, ball: pd.DataFrame,
                      team_map: dict[int, str], view_mode: str = "broadcast") -> pd.DataFrame:
    
    thr = _THRESHOLDS.get(view_mode, _THRESHOLDS["broadcast"])
    
    # FIX 3: exclude ball track from feature engineering
    tracks = tracks[tracks["track_id"] >= 0].copy()
    tracks = tracks.sort_values(["track_id", "frame_id"])

    coord_x = "smooth_x" if "smooth_x" in tracks.columns else "pitch_x"
    coord_y = "smooth_y" if "smooth_y" in tracks.columns else "pitch_y"

    tracks["px_m"] = tracks[coord_x] / PITCH_SCALE
    tracks["py_m"] = tracks[coord_y] / PITCH_SCALE

    tracks["dpx_m"] = tracks.groupby("track_id")["px_m"].diff().fillna(0)
    tracks["dpy_m"] = tracks.groupby("track_id")["py_m"].diff().fillna(0)
    tracks["speed_ms"]  = np.sqrt((tracks["dpx_m"] * FPS)**2 + (tracks["dpy_m"] * FPS)**2)
    tracks["accel_ms2"] = tracks.groupby("track_id")["speed_ms"].diff().fillna(0).abs() * FPS
    tracks["dir_rad"]   = np.arctan2(tracks["dpy_m"], tracks["dpx_m"])
    tracks["dir_change_rad"] = tracks.groupby("track_id")["dir_rad"].diff().fillna(0).abs()

    # Ball proximity
    if not ball.empty:
        bx_col = "smooth_x" if "smooth_x" in ball.columns else "pitch_x"
        by_col = "smooth_y" if "smooth_y" in ball.columns else "pitch_y"
        ball_pos = (ball[["frame_id", bx_col, by_col]]
                    .rename(columns={bx_col: "bx", by_col: "by"})
                    .drop_duplicates("frame_id")
                    .dropna(subset=["bx", "by"]))
    else:
        ball_pos = pd.DataFrame(columns=["frame_id", "bx", "by"])

    merged = tracks.merge(ball_pos, on="frame_id", how="left")
    merged["ball_dist_m"] = np.sqrt(
        ((merged[coord_x] - merged["bx"].fillna(9999)) / PITCH_SCALE)**2 +
        ((merged[coord_y] - merged["by"].fillna(9999)) / PITCH_SCALE)**2
    )
    merged["ball_dist_delta"] = merged.groupby("track_id")["ball_dist_m"].diff().fillna(0)

    # Possession indicator
    poss_map: dict[int, int] = {}
    for fid, grp in merged.groupby("frame_id"):
        valid = grp[grp["ball_dist_m"] < thr["BALL_POSS_M"]]
        if not valid.empty:
            poss_map[fid] = int(valid.loc[valid["ball_dist_m"].idxmin(), "track_id"])
    
    merged["has_possession"] = merged.apply(
        lambda r: int(poss_map.get(int(r["frame_id"]), -1) == r["track_id"]), axis=1
    )

    # Opponents within PRESS_RANGE
    opp_counts: dict[tuple, int] = {}
    for fid, grp in merged.groupby("frame_id"):
        for _, row in grp.iterrows():
            tid  = int(row["track_id"])
            team = team_map.get(tid, "unknown")
            opp  = grp[grp["track_id"].apply(lambda t: team_map.get(int(t), "x")) != team]
            if not opp.empty:
                dx = (opp[coord_x] - row[coord_x]) / PITCH_SCALE
                dy = (opp[coord_y] - row[coord_y]) / PITCH_SCALE
                opp_counts[(fid, tid)] = int((np.sqrt(dx**2 + dy**2) < thr["PRESS_RANGE"]).sum())
            else:
                opp_counts[(fid, tid)] = 0

    merged["n_opponents_5m"] = merged.apply(
        lambda r: opp_counts.get((int(r["frame_id"]), int(r["track_id"])), 0), axis=1
    )

    # Goal angle cosine
    goal_x = merged.apply(lambda r: GOAL_R_X if r["px_m"] * PITCH_SCALE < 525 else GOAL_L_X, axis=1)
    pb_x = (merged["bx"].fillna(merged[coord_x]) - merged[coord_x]) / PITCH_SCALE
    pb_y = (merged["by"].fillna(merged[coord_y]) - merged[coord_y]) / PITCH_SCALE
    pg_x = (goal_x - merged[coord_x]) / PITCH_SCALE
    pg_y = (GOAL_Y - merged[coord_y]) / PITCH_SCALE
    
    dot  = pb_x * pg_x + pb_y * pg_y
    mag  = (np.sqrt(pb_x**2 + pb_y**2) * np.sqrt(pg_x**2 + pg_y**2)).replace(0, 1)
    merged["goal_angle_cos"] = (dot / mag).clip(-1, 1)

    cols = ["frame_id", "track_id", "speed_ms", "accel_ms2", "ball_dist_m", 
            "ball_dist_delta", "goal_angle_cos", "n_opponents_5m", 
            "dir_change_rad", "has_possession"]
            
    return merged[cols].fillna(0)


# ─────────────────────────────────────────────────────────────────
#  Rule-based classifier
# ─────────────────────────────────────────────────────────────────

def classify_rule_based(features: pd.DataFrame, team_map: dict[int, str],
                        pass_frames: set[int], view_mode: str = "broadcast") -> pd.DataFrame:
                            
    thr = _THRESHOLDS.get(view_mode, _THRESHOLDS["broadcast"])
    SPEED_IDLE    = thr["SPEED_IDLE"]
    SPEED_RUN     = thr["SPEED_RUN"]
    SPEED_SPRINT  = thr["SPEED_SPRINT"]
    ACCEL_SHOT    = thr["ACCEL_SHOT"]
    GOAL_COS_SHOT = thr["GOAL_COS_SHOT"]
    PRESS_RANGE   = thr["PRESS_RANGE"]

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

        # FIX 4: Priority order mapping
        if fid in pass_frames and poss:
            actions.append("PASS");  confidences.append(0.80)
        elif (poss and spd > SPEED_RUN and acc > ACCEL_SHOT and gocos > GOAL_COS_SHOT):
            actions.append("SHOT");  confidences.append(0.75)
        elif (not poss and spd > SPEED_RUN and bdelt < -0.5 and nopp >= 1):
            actions.append("PRESS");  confidences.append(0.72)
        elif (not poss and spd > SPEED_SPRINT and bdelt < -1.0 and bdist < PRESS_RANGE):
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
#  ML mode (Random Forest fallback)
# ─────────────────────────────────────────────────────────────────

_FEAT_COLS = ["speed_ms", "accel_ms2", "ball_dist_m", "ball_dist_delta",
              "goal_angle_cos", "n_opponents_5m", "dir_change_rad", "has_possession"]

def train_and_apply_rf(features: pd.DataFrame, labeled_csv: Path, out_dir: Path) -> pd.DataFrame:
    if not _SK_AVAILABLE:
        print("  ⚠  scikit-learn needed for RF mode.")
        return pd.DataFrame()
    labeled = pd.read_csv(labeled_csv)
    missing = set(_FEAT_COLS + ["action"]) - set(labeled.columns)
    if missing:
        print(f"  ⚠  Labeled CSV missing columns: {missing}")
        return pd.DataFrame()
        
    X_train = labeled[_FEAT_COLS].values
    le      = LabelEncoder()
    y_train = le.fit_transform(labeled["action"].values)
    clf     = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    
    scores  = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"  RF cross-val F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")
    
    clf.fit(X_train, y_train)
    joblib.dump({"model": clf, "label_encoder": le}, out_dir / "action_rf_model.pkl")
    
    X_infer = features[_FEAT_COLS].fillna(0).values
    y_pred  = clf.predict(X_infer)
    result  = features.copy()
    result["action"]     = le.inverse_transform(y_pred)
    result["confidence"] = clf.predict_proba(X_infer).max(axis=1)
    return result


# ─────────────────────────────────────────────────────────────────
#  Timeline figure
# ─────────────────────────────────────────────────────────────────

def make_action_timeline(classified: pd.DataFrame, out_path: Path):
    classified = classified.copy()
    classified["minute"] = (classified["frame_id"] / FPS / 60).astype(int)
    pivot = (classified.groupby(["minute", "action"]).size().unstack(fill_value=0))
    
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(_DARK_BG); ax.set_facecolor(_SURFACE)
    bottom = np.zeros(len(pivot))
    
    for action in ACTION_LABELS:
        if action not in pivot.columns:
            continue
        vals = pivot[action].values
        ax.bar(pivot.index, vals, bottom=bottom, width=0.8, color=ACTION_COLORS.get(action, "#999"), label=action, zorder=3)
        bottom += vals
        
    ax.set_xlabel("Match minute", color=_TEXT); ax.set_ylabel("Count", color=_TEXT)
    ax.set_title("Action distribution over time", color="#e8e8f0", fontsize=13, fontweight="bold")
    ax.tick_params(colors=_TEXT)
    ax.grid(axis="y", color=_SURFACE, alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", facecolor=_SURFACE, edgecolor=_TEXT, labelcolor=_TEXT, fontsize=8)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Timeline → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class ActionClassifier:
    def __init__(self, tracks_csv: Path, teams_csv: Path, ball_csv: Path, events_csv: Path,
                 out_dir: Path, train_csv: Path | None = None, view_mode: str = "broadcast"):
        self.tracks_csv = tracks_csv
        self.teams_csv  = teams_csv
        self.ball_csv   = ball_csv
        self.events_csv = events_csv
        self.out_dir    = out_dir
        self.train_csv  = train_csv
        self.view_mode  = view_mode

    def run(self):
        print(f"\n  goalX PS3 — Action Classifier  (view_mode={self.view_mode})")
        print(f"  {'─'*44}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        tracks = pd.read_csv(self.tracks_csv)
        teams  = pd.read_csv(self.teams_csv)
        ball   = pd.read_csv(self.ball_csv) if self.ball_csv.exists() else pd.DataFrame()
        events = pd.read_csv(self.events_csv)

        for df in [tracks, events]:
            if "frame" in df.columns and "frame_id" not in df.columns:
                df.rename(columns={"frame": "frame_id"}, inplace=True)

        team_map = dict(zip(teams["track_id"].astype(int), teams["team"].astype(str)))
        
        if "event_type" in events.columns:
            pass_frames = set(events[events["event_type"] == "possession"]["frame_id"].dropna().astype(int).tolist())
        else:
            pass_frames = set()

        print(f"  Using '{self.view_mode}' speed thresholds:")
        thr = _THRESHOLDS.get(self.view_mode, _THRESHOLDS["broadcast"])
        print(f"     SPEED_RUN={thr['SPEED_RUN']} m/s  "
              f"SPEED_SPRINT={thr['SPEED_SPRINT']} m/s  "
              f"ACCEL_SHOT={thr['ACCEL_SHOT']} m/s²\n")

        print("  Engineering kinematic features…")
        features = engineer_features(tracks, ball, team_map, view_mode=self.view_mode)
        print(f"  Features: {len(features):,} player rows (ball excluded)\n")

        if self.train_csv and Path(self.train_csv).exists():
            classified = train_and_apply_rf(features, Path(self.train_csv), self.out_dir)
            if classified.empty:
                classified = classify_rule_based(features, team_map, pass_frames, self.view_mode)
        else:
            classified = classify_rule_based(features, team_map, pass_frames, self.view_mode)

        out_csv = self.out_dir / "actions.csv"
        classified.to_csv(out_csv, index=False)
        print(f"  ✔  Actions → {out_csv}")

        summary = classified.groupby(["track_id", "action"]).size().unstack(fill_value=0)
        summary["dominant_action"] = summary.idxmax(axis=1)
        summary.to_csv(self.out_dir / "action_summary.csv")

        print(f"\n  Action distribution:")
        dist = classified["action"].value_counts()
        for act, cnt in dist.items():
            bar  = "█" * int(cnt / max(dist) * 30)
            idle = "  ← thresholds may still be high" if act == "IDLE" and cnt / len(classified) > 0.9 else ""
            print(f"  {act:<10} {cnt:>6}  {bar}{idle}")

        if (dist.get("IDLE", 0) / len(classified)) > 0.90:
            print(f"\n  ⚠  IDLE still > 90%. Try --view-mode corner if using "
                  f"corner/end-zone footage.\n  Current mode: {self.view_mode}")

        make_action_timeline(classified, self.out_dir / "action_timeline.png")
        print(f"\n  ✅  Classification complete → {self.out_dir}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Action classifier for goalX PS3.")
    p.add_argument("--tracks",    default="outputs/smoothed_tracks.csv")
    p.add_argument("--teams",     default="outputs/team_labels.csv")
    p.add_argument("--ball",      default="outputs/ball_trajectory/interpolated_ball.csv")
    p.add_argument("--events",    default="outputs/events.csv")
    p.add_argument("--out-dir",   default="outputs/actions")
    p.add_argument("--train",     default=None)
    p.add_argument("--view-mode", default="broadcast", choices=["broadcast", "corner"],
                   help="Camera type: 'broadcast' (full pitch) or 'corner' (end-zone/penalty).")
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
        view_mode  = args.view_mode,
    ).run()