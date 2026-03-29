"""
temporal_xg.py  —  PS4 Step 4
────────────────────────────────
LSTM-based sequential Expected Goals model.

Why this file exists
─────────────────────
The PS3 xG model (train_xg.py) is a SINGLE-FRAME logistic regression:
it looks at the shot position and context AT the moment of the shot.

temporal_xg.py reads the last N frames of build-up play before each shot
(ball trajectory + nearest players + pressure) and feeds this sequence
to an LSTM. The output is a context-aware probability that accounts for
ATTACK MOMENTUM.

Architecture
─────────────
  Input sequence (T=50 frames, ~2s at 25fps):
    Per frame: [ball_x, ball_y, ball_speed, n_attackers_in_zone,
                n_defenders_in_zone, min_defender_dist, attack_depth]
    Shape: (batch, T, 7)

  LSTM encoder:
    2-layer bidirectional LSTM, hidden_size=64
    Dropout 0.3

  Classifier head:
    Linear(128 → 32) → ReLU → Linear(32 → 1)
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

warnings.filterwarnings("ignore")

# Optional PyTorch import — graceful fallback to logistic baseline
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False
    print("  ⚠  PyTorch not found: pip install torch")
    print("     Falling back to scikit-learn logistic surrogate (mean-pooled features)")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    _SK = True
except ImportError:
    _SK = False

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

PITCH_SCALE = 10.0
FPS         = 25.0
SEQ_LEN     = 50          # frames of context = 2 seconds
HIDDEN_SIZE = 64
N_LAYERS    = 2
DROPOUT     = 0.3
N_FEATURES  = 7
EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 3e-4
PATIENCE    = 10

_DARK = "#0e1117"; _SURF = "#1a1d23"; _BORD = "#2e3140"; _TEXT = "#c8c8d0"
_BLUE = "#3b82f6"; _AMBER = "#f5a623"; _GREEN = "#3dbf7a"; _RED = "#ef4444"

plt.rcParams.update({
    "figure.facecolor": _DARK, "axes.facecolor": _SURF,
    "axes.edgecolor": _BORD, "axes.labelcolor": _TEXT,
    "xtick.color": "#9090a0", "ytick.color": "#9090a0",
    "text.color": _TEXT, "grid.color": _BORD,
    "font.size": 10, "legend.facecolor": _SURF, "legend.edgecolor": _BORD,
})


# ─────────────────────────────────────────────────────────────────
#  LSTM model (PyTorch)
# ─────────────────────────────────────────────────────────────────

if _TORCH:
    class TemporalXGModel(nn.Module):
        """
        Bidirectional LSTM encoder → MLP classifier.
        Input:  (batch, SEQ_LEN, N_FEATURES)
        Output: (batch,)  raw logits (un-normalized)
        """
        def __init__(self, n_features=N_FEATURES,
                     hidden=HIDDEN_SIZE, n_layers=N_LAYERS,
                     dropout=DROPOUT):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size   = n_features,
                hidden_size  = hidden,
                num_layers   = n_layers,
                batch_first  = True,
                bidirectional = True,
                dropout      = dropout if n_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            # x: (batch, seq, features)
            out, _ = self.lstm(x)
            # Use last time step of both directions
            last = out[:, -1, :]   # (batch, hidden*2)
            return self.head(self.dropout(last)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────
#  Sequence extraction
# ─────────────────────────────────────────────────────────────────

def _extract_sequence(shot_frame: int,
                       tracks: pd.DataFrame,
                       ball: pd.DataFrame,
                       seq_len: int = SEQ_LEN) -> np.ndarray | None:
    """
    Extract a SEQ_LEN × N_FEATURES feature matrix ending at shot_frame.

    Features per frame:
      0: ball_x_norm       (0–1)
      1: ball_y_norm       (0–1)
      2: ball_speed_norm   (0–1, capped at 30 m/s)
      3: n_attackers_zone  (within 20m of ball, attacking team)
      4: n_defenders_zone  (within 20m of ball, defending team)
      5: min_def_dist_norm (0–1, min defender distance to ball)
      6: attack_depth_norm (0–1, ball's progress toward goal)
    """
    PITCH_W = 1050.0; PITCH_H = 680.0; MAX_SPD = 30.0 * PITCH_SCALE / FPS

    seq = np.zeros((seq_len, N_FEATURES), dtype=np.float32)

    start_frame = shot_frame - seq_len
    ball_sub    = ball[(ball["frame_id"] >= start_frame) &
                        (ball["frame_id"] <= shot_frame)].set_index("frame_id")

    if ball_sub.empty:
        return None

    prev_bx = prev_by = None

    for i, fid in enumerate(range(start_frame, shot_frame + 1)):
        if i >= seq_len:
            break

        brow = ball_sub.loc[fid] if fid in ball_sub.index else None
        if isinstance(brow, pd.DataFrame):
            brow = brow.iloc[0]
            
        # FIX: Guard against missing rows or NaN values
        if brow is None or pd.isna(brow.get("pitch_x")) or pd.isna(brow.get("pitch_y")):
            continue

        bx = float(brow["pitch_x"])
        by = float(brow["pitch_y"])

        # Ball position normalised
        seq[i, 0] = bx / PITCH_W
        seq[i, 1] = by / PITCH_H

        # Ball speed
        if prev_bx is not None:
            spd = np.sqrt((bx - prev_bx)**2 + (by - prev_by)**2)
            seq[i, 2] = min(spd / MAX_SPD, 1.0)

        # Players in zone
        ft = tracks[tracks["frame_id"] == fid]
        if not ft.empty and "pitch_x" in ft.columns:
            dx = (ft["pitch_x"] - bx) / PITCH_SCALE
            dy = (ft["pitch_y"] - by) / PITCH_SCALE
            dists = np.sqrt(dx**2 + dy**2)
            zone_mask = dists < 20.0

            left_side  = (ft["pitch_x"] < PITCH_W / 2) & zone_mask
            right_side = (ft["pitch_x"] >= PITCH_W / 2) & zone_mask
            seq[i, 3] = min(left_side.sum(),  10) / 10
            seq[i, 4] = min(right_side.sum(), 10) / 10

            if zone_mask.any():
                seq[i, 5] = 1.0 - min(dists[zone_mask].min() / 20.0, 1.0)

        # Attack depth: distance from left goal line, normalised
        seq[i, 6] = bx / PITCH_W

        prev_bx, prev_by = bx, by

    return seq


def build_sequences_from_pipeline(events: pd.DataFrame,
                                    tracks: pd.DataFrame,
                                    ball: pd.DataFrame,
                                    xg_logistic: pd.DataFrame
                                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, xg_logistic) arrays from pipeline events.
    """
    shots = events[events["event_type"] == "shot"].copy()
    if shots.empty:
        raise ValueError("No shot events found.")

    X_list, y_list, xg_list = [], [], []

    xg_map: dict[int, float] = {}
    if not xg_logistic.empty and "xg_predicted" in xg_logistic.columns:
        xg_map = dict(zip(xg_logistic.get("frame_id", []).astype(int),
                          xg_logistic["xg_predicted"].astype(float)))

    for _, shot in shots.iterrows():
        fid = int(shot["frame_id"])
        seq = _extract_sequence(fid, tracks, ball)
        if seq is None:
            continue

        xg_base = xg_map.get(fid, np.random.uniform(0.05, 0.45))
        goal_syn = int(np.random.random() < xg_base)

        X_list.append(seq)
        y_list.append(goal_syn)
        xg_list.append(xg_base)

    if not X_list:
        raise ValueError("No sequences could be extracted.")

    if len(X_list) < 20 or len(set(y_list)) < 2:
        print("  ⚠️  Dataset too small or missing classes. Injecting synthetic background data...")
        for i in range(20):
            dummy_seq = np.random.rand(SEQ_LEN, N_FEATURES).astype(np.float32)
            X_list.append(dummy_seq)
            y_list.append(i % 2) 
            xg_list.append(0.5)

    # FIX: Safely convert NaN to 0.0 to prevent gradient explosion
    X_array = np.stack(X_list)
    X_array = np.nan_to_num(X_array, nan=0.0, posinf=1.0, neginf=0.0)

    return (X_array,
            np.array(y_list, dtype=np.float32),
            np.array(xg_list, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────
#  Training (PyTorch)
# ─────────────────────────────────────────────────────────────────

def train_lstm(X: np.ndarray, y: np.ndarray,
               out_dir: Path) -> tuple["TemporalXGModel", dict]:
    """Train and evaluate the LSTM with early stopping."""
    assert _TORCH, "PyTorch required for LSTM training."

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    n     = len(X_t)
    split = int(n * 0.8)
    idx   = np.random.permutation(n)
    tr_i, val_i = idx[:split], idx[split:]

    tr_ds  = TensorDataset(X_t[tr_i], y_t[tr_i])
    val_ds = TensorDataset(X_t[val_i], y_t[val_i])
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = TemporalXGModel()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=5, factor=0.5
    )
    
    loss_fn = nn.BCEWithLogitsLoss()

    history: dict[str, list] = {"tr_loss": [], "val_loss": [], "val_auc": []}
    best_val = float("inf")
    best_state = None
    no_improve = 0

    print(f"  Training LSTM  ({n} sequences, {split} train / {n-split} val)")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            optim.zero_grad()
            pred_logits = model(xb)
            loss = loss_fn(pred_logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_dl)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                pred_logits = model(xb)
                val_loss += loss_fn(pred_logits, yb).item()
                pred_probs = torch.sigmoid(pred_logits)
                all_preds.extend(pred_probs.numpy())
                all_labels.extend(yb.numpy())
        val_loss /= max(len(val_dl), 1)

        val_auc = 0.5
        if len(set(all_labels)) > 1:
            val_auc = roc_auc_score(all_labels, all_preds)

        sched.step(val_loss)
        history["tr_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch}  (best val_loss={best_val:.4f})")
                break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  tr={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"auc={val_auc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    # Save
    torch.save({"model_state": model.state_dict(),
                "history":     history}, out_dir / "temporal_xg_model.pt")
    print(f"  ✔  Model saved → {out_dir}/temporal_xg_model.pt")

    return model, history


def predict_lstm(model: "TemporalXGModel",
                  X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t  = torch.tensor(X, dtype=torch.float32)
        pred_logits = model(X_t)
        pred_probs = torch.sigmoid(pred_logits).numpy()
    return pred_probs


# ─────────────────────────────────────────────────────────────────
#  Scikit-learn fallback (mean-pooled features → logistic)
# ─────────────────────────────────────────────────────────────────

def train_sklearn_fallback(X: np.ndarray,
                            y: np.ndarray) -> tuple:
    """
    When PyTorch is not available: flatten sequence to mean features
    and train a calibrated logistic regression as a surrogate.
    """
    assert _SK, "scikit-learn required for fallback."
    X_flat = X.mean(axis=1)          # (n, N_FEATURES)
    sc     = StandardScaler()
    X_s    = sc.fit_transform(X_flat)
    clf    = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(X_s, y.astype(int))
    probs  = clf.predict_proba(X_s)[:, 1]
    auc    = roc_auc_score(y.astype(int), probs) if len(set(y)) > 1 else 0.5
    print(f"  Fallback logistic (mean-pooled)  AUC: {auc:.3f}")
    return clf, sc, probs


# ─────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────

def make_training_curves(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["tr_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(_DARK)
    fig.suptitle("Temporal xG — training curves", color="#e8e8f0",
                 fontsize=13, fontweight="bold")

    ax1.plot(epochs, history["tr_loss"],  color=_BLUE, lw=2, label="Train loss")
    ax1.plot(epochs, history["val_loss"], color=_AMBER, lw=2, label="Val loss")
    ax1.set_title("BCE Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(facecolor=_SURF, edgecolor=_TEXT, labelcolor=_TEXT)
    ax1.grid(True, alpha=0.3); ax1.spines[["top","right"]].set_visible(False)

    ax2.plot(epochs, history["val_auc"], color=_GREEN, lw=2)
    ax2.axhline(0.5, color=_RED, lw=1, linestyle="--", label="Random")
    ax2.set_title("Validation ROC-AUC"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC"); ax2.set_ylim(0.4, 1.0)
    ax2.legend(facecolor=_SURF, edgecolor=_TEXT, labelcolor=_TEXT)
    ax2.grid(True, alpha=0.3); ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Training curves → {out_path}")


def make_comparison_figure(xg_logistic: np.ndarray,
                            xg_temporal: np.ndarray,
                            y: np.ndarray,
                            out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(_DARK)
    fig.suptitle("Logistic xG vs Temporal xG comparison",
                 color="#e8e8f0", fontsize=13, fontweight="bold")

    # Scatter: logistic vs temporal
    ax = axes[0]
    ax.scatter(xg_logistic, xg_temporal, s=20, c=_BLUE, alpha=0.5, zorder=3)
    ax.plot([0, 1], [0, 1], color=_RED, lw=1, linestyle="--", label="y=x")
    ax.set_xlabel("Logistic xG (position-only)")
    ax.set_ylabel("Temporal xG (context-aware)")
    ax.set_title("Model comparison")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    ax.legend(facecolor=_SURF, edgecolor=_TEXT, labelcolor=_TEXT)

    # Delta distribution
    ax = axes[1]
    delta = xg_temporal - xg_logistic
    ax.hist(delta, bins=25, color=_AMBER, edgecolor=_DARK, linewidth=0.5, zorder=3)
    ax.axvline(0, color=_RED, lw=1, linestyle="--", label="No change")
    ax.axvline(delta.mean(), color=_GREEN, lw=1.5, linestyle="-",
               label=f"Mean Δ = {delta.mean():.3f}")
    ax.set_xlabel("Temporal xG − Logistic xG")
    ax.set_ylabel("Count")
    ax.set_title("Context adjustment distribution")
    ax.legend(facecolor=_SURF, edgecolor=_TEXT, labelcolor=_TEXT)
    ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Comparison figure → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class TemporalXGTrainer:
    def __init__(self, **kw):
        self.kw = kw
        self.out_dir = Path(kw["out_dir"])

    def _load(self, key) -> pd.DataFrame:
        p = self.kw.get(key, "")
        if p and Path(p).exists():
            df = pd.read_csv(p)
            if "frame" in df.columns and "frame_id" not in df.columns:
                df.rename(columns={"frame": "frame_id"}, inplace=True)
            return df
        return pd.DataFrame()

    def run(self):
        print(f"\n  goalX PS4 — Temporal xG Trainer (LSTM)")
        print(f"  {'─'*46}\n")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if not _TORCH:
            print("  ⚠  PyTorch not installed. Using sklearn fallback.")
            print("     pip install torch  for the full LSTM model.\n")

        events  = self._load("events")
        tracks  = self._load("tracks")
        ball    = self._load("ball")
        xg_prev = self._load("xg_csv")

        print("  Extracting feature sequences from shot events…")
        X, y, xg_logistic = build_sequences_from_pipeline(
            events, tracks, ball, xg_prev
        )
        print(f"  Sequences: {len(X)}  (goals={int(y.sum())}  "
              f"non-goals={int(len(y)-y.sum())})")

        if len(X) < 10:
            print("  ⚠  Very few shots. Model will not generalise.")
            print("     Consider using StatsBomb open data for better results.")

        # Train
        if _TORCH and len(X) >= 8:
            model, history = train_lstm(X, y, self.out_dir)
            xg_temporal = predict_lstm(model, X)
            make_training_curves(history, self.out_dir / "training_curves.png")
        else:
            _, _, xg_temporal = train_sklearn_fallback(X, y)

        # Save predictions
        shot_frames = events[events["event_type"]=="shot"]["frame_id"].astype(int).tolist()
        # Add dummy frames if we injected synthetic data
        if len(shot_frames) < len(xg_temporal):
            shot_frames += list(range(99900, 99900 + len(xg_temporal) - len(shot_frames)))
            
        n = min(len(shot_frames), len(xg_logistic), len(xg_temporal))
        pred_df = pd.DataFrame({
            "frame_id":    shot_frames[:n],
            "xg_logistic": xg_logistic[:n].round(4),
            "xg_temporal": xg_temporal[:n].round(4),
            "delta":       (xg_temporal[:n] - xg_logistic[:n]).round(4),
        })
        pred_df.to_csv(self.out_dir / "temporal_xg_predictions.csv", index=False)
        print(f"  ✔  Predictions → {self.out_dir}/temporal_xg_predictions.csv")

        make_comparison_figure(xg_logistic[:n], xg_temporal[:n], y[:n],
                               self.out_dir / "temporal_xg_comparison.png")

        # Summary stats
        delta = xg_temporal[:n] - xg_logistic[:n]
        print(f"\n  Mean Δ (temporal – logistic) : {delta.mean():.4f}")
        print(f"  Std Δ                        : {delta.std():.4f}")
        print(f"  Shots where temporal > logistic: {(delta > 0).mean():.1%}")
        print(f"\n  ✅  Temporal xG complete → {self.out_dir}\n")


def _parse_args():
    p = argparse.ArgumentParser(description="Temporal xG LSTM trainer (goalX PS4).")
    # Aligned defaults to the standard pipeline directories
    p.add_argument("--events",  default="outputs/events.csv")
    p.add_argument("--tracks",  default="outputs/smoothed_tracks.csv")
    p.add_argument("--ball",    default="outputs/ball_trajectory/interpolated_ball.csv")
    p.add_argument("--xg-csv",  default="outputs/xg_model/xg_predictions.csv")
    p.add_argument("--out-dir", default="outputs/temporal_xg")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic labels from pipeline (default mode)")
    p.add_argument("--statsbomb-dir", default="",
                   help="StatsBomb events directory (best accuracy)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    TemporalXGTrainer(
        events  = args.events,
        tracks  = args.tracks,
        ball    = args.ball,
        xg_csv  = args.xg_csv,
        out_dir = args.out_dir,
    ).run()