"""
train_xg.py  —  PS3 Step 1
───────────────────────────
Trains a calibrated logistic regression Expected Goals (xG) model and
replaces the position-only heuristic in clutch_score.py with a
data-driven probability estimate.

Architecture
────────────
  Feature engineering  →  Logistic Regression (L2)  →  Platt calibration
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────
#  Pitch / goal geometry constants
# ─────────────────────────────────────────────────────────────────

PITCH_W_M  = 105.0
PITCH_H_M  = 68.0
GOAL_Y_M   = PITCH_H_M / 2          # centre of goal (y)
GOAL_WIDTH = 7.32                   # FIFA goal width
GOAL_LEFT  = GOAL_Y_M - GOAL_WIDTH / 2
GOAL_RIGHT = GOAL_Y_M + GOAL_WIDTH / 2
GOAL_X_R   = PITCH_W_M              # right goal line x
GOAL_X_L   = 0.0                    # left goal line x
PITCH_SCALE = 10.0                  # px/m (must match draw_pitch.py)


def _distance_to_goal(x_m: float, y_m: float, shooting_right: bool) -> float:
    """Euclidean distance from shot position to nearest goal centre."""
    goal_x = GOAL_X_R if shooting_right else GOAL_X_L
    return float(np.sqrt((x_m - goal_x) ** 2 + (y_m - GOAL_Y_M) ** 2))


def _angle_to_goal(x_m: float, y_m: float, shooting_right: bool) -> float:
    """
    Angle subtended by the goal mouth at the shot position (radians).
    Derived from the law of cosines on the triangle
    [shot_pos, left_post, right_post].
    """
    goal_x = GOAL_X_R if shooting_right else GOAL_X_L
    a = np.sqrt((x_m - goal_x) ** 2 + (y_m - GOAL_LEFT) ** 2)
    b = np.sqrt((x_m - goal_x) ** 2 + (y_m - GOAL_RIGHT) ** 2)
    c = GOAL_WIDTH
    cos_theta = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b + 1e-9)
    cos_theta = np.clip(cos_theta, -1, 1)
    return float(np.arccos(cos_theta))


def _px_to_m(px_coord: float) -> float:
    return px_coord / PITCH_SCALE


# ─────────────────────────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "distance_m", "angle_rad", "pressure_n",
    "pitch_control", "shot_speed", "head_proxy",
]


def _build_features_from_events(events_df: pd.DataFrame,
                                 tracks_df: pd.DataFrame,
                                 control_df: pd.DataFrame
                                 ) -> pd.DataFrame:
    """
    Engineer xG features from goalX pipeline outputs.
    Returns a DataFrame with FEATURE_COLS + 'goal' column.
    """
    shots = events_df[events_df["event_type"] == "shot"].copy()
    if shots.empty:
        # Return an empty dataframe with correct columns if no shots found
        return pd.DataFrame(columns=FEATURE_COLS + ["goal", "frame_id", "xg_heuristic"])

    if not control_df.empty and "home_poss" in control_df.columns:
        control_map = dict(zip(
            control_df["frame_id"].astype(int),
            control_df["home_poss"].astype(float)
        ))
    else:
        control_map = {}

    rows = []
    for _, shot in shots.iterrows():
        fid = int(shot["frame_id"])

        x_px = float(shot.get("pitch_x", shot.get("x1", 0)))
        y_px = float(shot.get("pitch_y", shot.get("y1", 0)))
        x_m  = _px_to_m(x_px)
        y_m  = _px_to_m(y_px)

        shooting_right = x_m > PITCH_W_M / 2
        dist  = _distance_to_goal(x_m, y_m, shooting_right)
        angle = _angle_to_goal(x_m, y_m, shooting_right)

        frame_players = tracks_df[tracks_df["frame_id"] == fid].copy()
        shooter_tid   = shot.get("track_id", -999)
        opponents     = frame_players[frame_players["track_id"] != shooter_tid]
        if not opponents.empty and "pitch_x" in opponents.columns:
            opp_x = _px_to_m(opponents["pitch_x"].values)
            opp_y = _px_to_m(opponents["pitch_y"].values)
            dists = np.sqrt((opp_x - x_m) ** 2 + (opp_y - y_m) ** 2)
            pressure_n = int((dists < 5.0).sum())
        else:
            pressure_n = 0

        pc = float(control_map.get(fid, 0.5))

        ball_before = tracks_df[
            (tracks_df["track_id"] == -1) &
            (tracks_df["frame_id"] >= fid - 5) &
            (tracks_df["frame_id"] < fid)
        ].sort_values("frame_id")

        if len(ball_before) >= 2:
            dx = ball_before["pitch_x"].diff().dropna()
            dy = ball_before["pitch_y"].diff().dropna()
            # FIX: Prevent taking the mean of an empty array which results in NaN
            if len(dx) > 0:
                shot_speed = float(np.sqrt(dx**2 + dy**2).mean())
            else:
                shot_speed = 0.0
        else:
            shot_speed = 0.0

        head_proxy = 1 if (dist < 8.0 and abs(y_m - GOAL_Y_M) > 3.0) else 0

        xg_heuristic = float(
            np.clip(angle / np.pi * np.exp(-dist / 20.0), 0, 1)
        )
        goal_synthetic = int(np.random.random() < (xg_heuristic if not np.isnan(xg_heuristic) else 0))

        rows.append({
            "distance_m":   dist,
            "angle_rad":    angle,
            "pressure_n":   pressure_n,
            "pitch_control": pc,
            "shot_speed":   shot_speed,
            "head_proxy":   head_proxy,
            "goal":         goal_synthetic,
            "frame_id":     fid,
            "xg_heuristic": xg_heuristic,
        })

    return pd.DataFrame(rows)


def _build_features_from_statsbomb(events_dir: Path) -> pd.DataFrame:
    """Parse StatsBomb open-data event JSON files."""
    rows = []
    json_files = list(events_dir.glob("*.json"))
    print(f"  Found {len(json_files)} StatsBomb event files.")

    for jf in json_files:
        with open(jf) as f:
            events = json.load(f)

        for evt in events:
            if evt.get("type", {}).get("name") != "Shot":
                continue

            loc  = evt.get("location", [None, None])
            if loc[0] is None:
                continue

            x_sb, y_sb = float(loc[0]), float(loc[1])
            x_m = x_sb / 120.0 * PITCH_W_M
            y_m = y_sb / 80.0  * PITCH_H_M

            shooting_right = x_m > PITCH_W_M / 2
            dist  = _distance_to_goal(x_m, y_m, shooting_right)
            angle = _angle_to_goal(x_m, y_m, shooting_right)

            shot_detail  = evt.get("shot", {})
            outcome      = shot_detail.get("outcome", {}).get("name", "")
            goal         = 1 if outcome == "Goal" else 0
            technique    = shot_detail.get("technique", {}).get("name", "")
            head_proxy   = 1 if "Head" in technique else 0

            freeze = shot_detail.get("freeze_frame", [])
            defenders_close = sum(
                1 for p in freeze
                if not p.get("teammate", False)
                and np.sqrt((p["location"][0] - x_sb) ** 2 +
                            (p["location"][1] - y_sb) ** 2) < 5.0
            ) if freeze else 0

            rows.append({
                "distance_m":    dist,
                "angle_rad":     angle,
                "pressure_n":    defenders_close,
                "pitch_control": 0.5,
                "shot_speed":    0.0,
                "head_proxy":    head_proxy,
                "goal":          goal,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
#  Model training
# ─────────────────────────────────────────────────────────────────

def train_xg_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Train and calibrate a logistic regression xG model."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["goal"].values.astype(int)

    print(f"\n  Training on {len(df):,} shots  "
          f"(goals={y.sum():,}  "
          f"non-goals={len(y)-y.sum():,}  "
          f"rate={y.mean():.3f})")

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=1.0, max_iter=1000,
                                       class_weight="balanced",
                                       random_state=42)),
    ])

    model = CalibratedClassifierCV(base, cv=5, method="sigmoid")

    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob   = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    auc      = roc_auc_score(y, y_prob)
    brier    = brier_score_loss(y, y_prob)
    ap       = average_precision_score(y, y_prob)

    print(f"  Cross-validated metrics (5-fold):")
    print(f"    ROC-AUC   : {auc:.4f}  (ideal: 1.0, random: 0.5)")
    print(f"    Brier     : {brier:.4f}  (ideal: 0.0)")
    print(f"    Avg Prec  : {ap:.4f}")

    model.fit(X, y)

    coefficients = {}
    try:
        for estimator in model.calibrated_classifiers_:
            lr = estimator.estimator.named_steps["lr"]
            sc = estimator.estimator.named_steps["scaler"]
            coef = lr.coef_[0] * sc.scale_ ** -1 
            for feat, c in zip(FEATURE_COLS, coef):
                coefficients[feat] = float(c)
            break 
    except Exception:
        pass

    metrics = {
        "roc_auc":       round(auc,  4),
        "brier_score":   round(brier, 4),
        "avg_precision": round(ap,    4),
        "n_shots":       int(len(df)),
        "n_goals":       int(y.sum()),
        "goal_rate":     round(float(y.mean()), 4),
        "coefficients":  coefficients,
        "y_true":        y.tolist(),
        "y_prob":        y_prob.tolist(),
    }

    return model, metrics


# ─────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────

_DARK_BG    = "#0e1117"
_SURFACE    = "#1a1d23"
_BORDER     = "#2e3140"
_TEXT       = "#c8c8d0"
_ACCENT     = "#5b8dee"
_GREEN      = "#3dbf7a"
_AMBER      = "#f5a623"
_RED        = "#e05c5c"

plt.rcParams.update({
    "figure.facecolor": _DARK_BG,
    "axes.facecolor":   _SURFACE,
    "axes.edgecolor":   _BORDER,
    "axes.labelcolor":  _TEXT,
    "axes.titlecolor":  "#e8e8f0",
    "xtick.color":      "#9090a0",
    "ytick.color":      "#9090a0",
    "text.color":       _TEXT,
    "grid.color":       _BORDER,
    "grid.linewidth":   0.5,
    "font.size":        10,
    "legend.facecolor": _SURFACE,
    "legend.edgecolor": _BORDER,
})


def make_evaluation_figure(metrics: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("xG model evaluation", fontsize=14,
                 color="#e8e8f0", weight="bold")

    y_true = np.array(metrics["y_true"])
    y_prob = np.array(metrics["y_prob"])

    # ── ROC curve ─────────────────────────────────────────────
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, color=_ACCENT, lw=2,
            label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=_BORDER, lw=1, linestyle="--",
            label="Random")
    ax.set_xlabel("False positive rate");  ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(fontsize=9)
    ax.grid(True, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Calibration curve ─────────────────────────────────────
    ax = axes[1]
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, color=_GREEN, lw=2, marker="o", ms=4,
            label="Model")
    ax.plot([0, 1], [0, 1], color=_BORDER, lw=1, linestyle="--",
            label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration  (Brier={metrics['brier_score']:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Feature importance ─────────────────────────────────────
    ax   = axes[2]
    coef = metrics.get("coefficients", {})
    if coef:
        feats  = list(coef.keys())
        vals   = [coef[f] for f in feats]
        colors = [_GREEN if v > 0 else _RED for v in vals]
        bars   = ax.barh(feats, vals, color=colors, height=0.55, zorder=3)
        ax.axvline(0, color=_BORDER, lw=1)
        ax.set_title("Feature coefficients")
        ax.set_xlabel("Log-odds contribution")
        ax.grid(axis="x", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(val + (0.02 if val >= 0 else -0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8, color=_TEXT)
    else:
        ax.text(0.5, 0.5, "Coefficients\nnot available",
                ha="center", va="center", transform=ax.transAxes,
                color="#6060a0")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Evaluation figure → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

class XGTrainer:
    def __init__(self, out_dir: Path, **kwargs):
        self.out_dir = out_dir
        self.kwargs  = kwargs

    def run(self) -> None:
        print(f"\n  goalX PS3 — xG Model Trainer")
        print(f"  {'─'*42}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.kwargs.get("statsbomb_dir"):
            print("  Mode A: StatsBomb open data")
            df = _build_features_from_statsbomb(Path(self.kwargs["statsbomb_dir"]))
        elif self.kwargs.get("from_csv"):
            print("  Mode B: labeled shot CSV")
            df = pd.read_csv(self.kwargs["from_csv"])
            missing = set(FEATURE_COLS + ["goal"]) - set(df.columns)
            if missing:
                raise ValueError(f"CSV missing columns: {missing}")
        else:
            print("  Mode C: synthetic from pipeline events")
            print("  ⚠  Synthetic labels reduce accuracy — use Mode A/B for thesis.")
            events  = pd.read_csv(self.kwargs["events"]) if Path(self.kwargs["events"]).exists() else pd.DataFrame(columns=["event_type"])
            tracks  = pd.read_csv(self.kwargs["tracks"]) if Path(self.kwargs["tracks"]).exists() else pd.DataFrame()
            control = pd.read_csv(self.kwargs["control"]) if Path(self.kwargs["control"]).exists() else pd.DataFrame()
            df = _build_features_from_events(events, tracks, control)

        # --- FIX: Clean Data before processing ---
        initial_len = len(df)
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        if len(df) < initial_len:
            print(f"  🧹 Dropped {initial_len - len(df)} shots due to missing coordinates (NaNs).")

        # --- Inject synthetic background data if the user's video clip is too short ---
        if len(df) < 20 or df["goal"].nunique() < 2:
            print("  ⚠️  Dataset too small or missing classes for cross-validation.")
            print("  🛠️  Injecting synthetic background data so the pipeline can compile...")
            
            dummy_rows = []
            for i in range(20):  # Guarantee 10 goals and 10 misses
                dummy_rows.append({
                    "distance_m": 5.0 + i, 
                    "angle_rad": 0.8,
                    "pressure_n": i % 3,
                    "pitch_control": 0.5,
                    "shot_speed": 5.0,
                    "head_proxy": 0,
                    "goal": i % 2, 
                    "frame_id": 99999,
                    "xg_heuristic": 0.5
                })
            df = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)

        print(f"  Dataset: {len(df)} shots")

        # ── Train ──────────────────────────────────────────────
        model, metrics = train_xg_model(df)

        # ── Save model ─────────────────────────────────────────
        model_path = self.out_dir / "xg_model.pkl"
        joblib.dump(model, model_path)
        print(f"  ✔  Model saved → {model_path}")

        # ── Save coefficients table ────────────────────────────
        coef = metrics.get("coefficients", {})
        if coef:
            coef_df = pd.DataFrame([
                {"feature": f, "coefficient": c,
                 "odds_ratio": float(np.exp(c))}
                for f, c in coef.items()
            ])
            coef_path = self.out_dir / "xg_coefficients.csv"
            coef_df.to_csv(coef_path, index=False)
            print(f"  ✔  Coefficients → {coef_path}")

        # ── Predictions CSV ────────────────────────────────────
        X = df[FEATURE_COLS].values.astype(np.float32)
        df["xg_predicted"] = model.predict_proba(X)[:, 1]
        df.to_csv(self.out_dir / "xg_predictions.csv", index=False)

        # ── Figure ─────────────────────────────────────────────
        make_evaluation_figure(metrics, self.out_dir / "xg_evaluation.png")

        # ── Summary ────────────────────────────────────────────
        print(f"\n  ✅  Training complete")
        print(f"  ROC-AUC   : {metrics['roc_auc']}")
        print(f"  Brier     : {metrics['brier_score']}")
        print(f"  Avg Prec  : {metrics['avg_precision']}")
        print(f"\n  To use in clutch_score.py, pass:")
        print(f"    --xg-model {self.out_dir}/xg_model.pkl\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train xG model for goalX PS3.")
    p.add_argument("--out-dir",       default="outputs/xg_model")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--statsbomb-dir", help="StatsBomb open-data events directory")
    g.add_argument("--from-csv",      help="Custom labeled shot CSV")
    g.add_argument("--synthetic",     action="store_true",
                   help="Generate synthetic training data from pipeline outputs")
    p.add_argument("--events",  default="outputs/events.csv")
    p.add_argument("--tracks",  default="outputs/smoothed_tracks.csv")
    p.add_argument("--control", default="outputs/pitch_control/pitch_control.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    XGTrainer(
        out_dir       = Path(args.out_dir),
        statsbomb_dir = args.statsbomb_dir,
        from_csv      = args.from_csv,
        synthetic     = args.synthetic,
        events        = args.events,
        tracks        = args.tracks,
        control       = args.control,
    ).run()