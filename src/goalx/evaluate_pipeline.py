"""
evaluate_pipeline.py
─────────────────────
Automated quantitative evaluation of the goalX pipeline against
ground-truth annotations. Produces publication-ready metric tables
and Matplotlib figures for the M.Tech thesis defense.

Metrics computed
────────────────
  Tracking   : MOTA, MOTP, IDF1, Precision, Recall (MOTChallenge standard)
  Events     : Precision, Recall, F1 per event type; temporal IoU for shots
  Formation  : Frame-level accuracy vs ground-truth formation labels
  Clutch     : Score distribution statistics + Pearson correlation with xG

Ground-truth format  (CSV files in --gt-dir)
────────────────────
  tracking_gt.csv    : frame_id, track_id, x1, y1, x2, y2
  events_gt.csv      : frame_id, event_type  (shot/possession/pressure)
  formation_gt.csv   : frame_id, team, formation  (e.g. 4-3-3)

Usage
─────
  python3 src/goalx/evaluate_pipeline.py \\
      --tracks    outputs/smoothed_tracks.csv \\
      --events    outputs/events.csv \\
      --formation outputs/formations.csv \\
      --clutch    outputs/clutch_scores.csv \\
      --gt-dir    data/ground_truth \\
      --out-dir   outputs/evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────────
#  Plot style
# ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#1a1d23",
    "axes.edgecolor":    "#3a3d45",
    "axes.labelcolor":   "#c8c8d0",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   "#e8e8f0",
    "xtick.color":       "#9090a0",
    "ytick.color":       "#9090a0",
    "text.color":        "#c8c8d0",
    "grid.color":        "#2a2d35",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "legend.facecolor":  "#1a1d23",
    "legend.edgecolor":  "#3a3d45",
})

_ACCENT   = "#5b8dee"
_GREEN    = "#3dbf7a"
_AMBER    = "#f5a623"
_RED      = "#e05c5c"
_PURPLE   = "#9b72cf"


# ─────────────────────────────────────────────────────────────────
#  Tracking metrics  (MOTChallenge standard)
# ─────────────────────────────────────────────────────────────────

def _iou_box(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def compute_tracking_metrics(pred_df: pd.DataFrame,
                              gt_df:   pd.DataFrame,
                              iou_thresh: float = 0.5
                              ) -> dict[str, float]:
    """
    Compute MOTA, MOTP, Precision, Recall, IDF1 per the MOTChallenge
    definition.

    MOTA = 1 - (FP + FN + IDSW) / GT
    MOTP = sum(IoU for matched pairs) / total matches
    IDF1 = 2 * IDTP / (2*IDTP + IDFP + IDFN)
    """
    if gt_df.empty:
        return {"note": "No ground truth provided — tracking metrics skipped."}

    TP = FP = FN = IDSW = 0
    iou_sum = 0.0
    n_matches = 0

    # Track last assignment: gt_track_id → pred_track_id
    last_assignment: dict[int, int] = {}

    for fid in sorted(gt_df["frame_id"].unique()):
        gt_f   = gt_df[gt_df["frame_id"] == fid]
        pred_f = pred_df[pred_df["frame_id"] == fid]

        gt_boxes   = gt_f[["x1","y1","x2","y2"]].values
        pred_boxes = pred_f[["x1","y1","x2","y2"]].values

        # Greedy IoU matching
        matched_gt   = set()
        matched_pred = set()

        for gi, gb in enumerate(gt_boxes):
            best_iou = iou_thresh
            best_pi  = -1
            for pi, pb in enumerate(pred_boxes):
                if pi in matched_pred:
                    continue
                iou = _iou_box(gb, pb)
                if iou >= best_iou:
                    best_iou = iou
                    best_pi  = pi

            if best_pi >= 0:
                TP       += 1
                iou_sum  += best_iou
                n_matches += 1
                matched_gt.add(gi)
                matched_pred.add(best_pi)

                gt_id   = int(gt_f.iloc[gi]["track_id"])
                pred_id = int(pred_f.iloc[best_pi]["track_id"])

                if gt_id in last_assignment and last_assignment[gt_id] != pred_id:
                    IDSW += 1
                last_assignment[gt_id] = pred_id

        FP += len(pred_boxes) - len(matched_pred)
        FN += len(gt_boxes)   - len(matched_gt)

    GT   = len(gt_df)
    MOTA = 1.0 - (FP + FN + IDSW) / max(GT, 1)
    MOTP = iou_sum / max(n_matches, 1)
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)

    # IDF1 approximation (treating TP as IDTP)
    IDF1 = 2 * TP / max(2 * TP + FP + FN, 1)

    return {
        "MOTA":      round(MOTA,  4),
        "MOTP":      round(MOTP,  4),
        "Precision": round(prec,  4),
        "Recall":    round(rec,   4),
        "IDF1":      round(IDF1,  4),
        "TP":  TP, "FP": FP, "FN": FN, "IDSW": IDSW,
        "GT_count": GT,
    }


# ─────────────────────────────────────────────────────────────────
#  Event detection metrics
# ─────────────────────────────────────────────────────────────────

def _temporal_iou(pred_frame: int, gt_frame: int,
                  window: int = 30) -> float:
    """Treat a frame-level event as a window; compute window overlap."""
    p_start, p_end = pred_frame - window, pred_frame + window
    g_start, g_end = gt_frame  - window, gt_frame  + window
    inter = max(0, min(p_end, g_end) - max(p_start, g_start))
    union = (p_end - p_start) + (g_end - g_start) - inter
    return inter / max(union, 1)


def compute_event_metrics(pred_df: pd.DataFrame,
                           gt_df:   pd.DataFrame,
                           window: int = 30
                           ) -> dict[str, dict]:
    """
    Per-event-type Precision / Recall / F1 using temporal-IoU matching.
    """
    if gt_df.empty:
        return {"note": "No event ground truth provided."}

    event_types = gt_df["event_type"].unique()
    results = {}

    for etype in event_types:
        gt_frames   = gt_df[gt_df["event_type"] == etype]["frame_id"].tolist()
        pred_frames = (pred_df[pred_df["event_type"] == etype]["frame_id"].tolist()
                       if "event_type" in pred_df.columns else [])

        matched_gt   = set()
        matched_pred = set()

        for pi, pf in enumerate(pred_frames):
            best_iou = 0.5
            best_gi  = -1
            for gi, gf in enumerate(gt_frames):
                if gi in matched_gt:
                    continue
                iou = _temporal_iou(pf, gf, window)
                if iou > best_iou:
                    best_iou = iou
                    best_gi  = gi
            if best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)

        TP   = len(matched_pred)
        FP   = len(pred_frames) - TP
        FN   = len(gt_frames)   - TP
        prec = TP / max(TP + FP, 1)
        rec  = TP / max(TP + FN, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)

        results[etype] = {
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1":        round(f1,   4),
            "TP": TP, "FP": FP, "FN": FN,
            "GT_count": len(gt_frames),
            "Pred_count": len(pred_frames),
        }

    return results


# ─────────────────────────────────────────────────────────────────
#  Formation accuracy
# ─────────────────────────────────────────────────────────────────

def compute_formation_accuracy(pred_df: pd.DataFrame,
                                gt_df:   pd.DataFrame) -> dict:
    if gt_df.empty or pred_df.empty:
        return {"note": "No formation ground truth provided."}

    merged = pd.merge(pred_df, gt_df,
                      on=["frame_id", "team"],
                      suffixes=("_pred", "_gt"))

    if merged.empty:
        return {"note": "No overlapping frame/team pairs found."}

    total   = len(merged)
    correct = (merged["formation_pred"] == merged["formation_gt"]).sum()
    accuracy = correct / total

    # Per-team breakdown
    by_team = {}
    for team in merged["team"].unique():
        tm = merged[merged["team"] == team]
        by_team[team] = round((tm["formation_pred"] == tm["formation_gt"]).mean(), 4)

    # Confusion distribution: what did we predict vs truth?
    confusion = (merged.groupby(["formation_gt", "formation_pred"])
                       .size()
                       .reset_index(name="count")
                       .to_dict("records"))

    return {
        "accuracy":      round(accuracy, 4),
        "correct_frames": int(correct),
        "total_frames":   int(total),
        "by_team":        by_team,
        "confusion":      confusion,
    }


# ─────────────────────────────────────────────────────────────────
#  Clutch score distribution
# ─────────────────────────────────────────────────────────────────

def compute_clutch_stats(clutch_df: pd.DataFrame) -> dict:
    if clutch_df.empty or "clutch_score" not in clutch_df.columns:
        return {"note": "No clutch scores found."}

    scores = clutch_df["clutch_score"].dropna()
    result = {
        "count":  int(len(scores)),
        "mean":   round(float(scores.mean()), 4),
        "median": round(float(scores.median()), 4),
        "std":    round(float(scores.std()),  4),
        "min":    round(float(scores.min()),  4),
        "max":    round(float(scores.max()),  4),
        "p90":    round(float(np.percentile(scores, 90)), 4),
    }

    if "xg" in clutch_df.columns:
        corr, pval = pearsonr(scores, clutch_df["xg"].dropna())
        result["xg_pearson_r"]  = round(float(corr), 4)
        result["xg_pearson_p"]  = round(float(pval), 6)

    return result


# ─────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────

def _bar_chart(ax, labels, values, colors, title, ylabel):
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                color="#e8e8f0")


def make_tracking_figure(metrics: dict, out_path: Path) -> None:
    if "note" in metrics:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Tracking evaluation (MOTChallenge)", fontsize=14,
                 color="#e8e8f0", weight="bold")

    # Left: main metrics
    keys   = ["MOTA", "MOTP", "Precision", "Recall", "IDF1"]
    vals   = [metrics.get(k, 0) for k in keys]
    colors = [_ACCENT, _GREEN, _PURPLE, _AMBER, _RED]
    _bar_chart(axes[0], keys, vals, colors, "Core tracking metrics", "Score")

    # Right: count breakdown
    count_keys  = ["TP", "FP", "FN", "IDSW"]
    count_vals  = [metrics.get(k, 0) for k in count_keys]
    count_colors = [_GREEN, _RED, _AMBER, _PURPLE]
    axes[1].bar(count_keys, count_vals, color=count_colors, width=0.5, zorder=3)
    axes[1].set_title("Detection counts")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", zorder=0)
    axes[1].spines[["top","right"]].set_visible(False)
    for bar, val in zip(axes[1].patches, count_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     str(val), ha="center", fontsize=9, color="#e8e8f0")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Tracking figure → {out_path}")


def make_event_figure(event_metrics: dict, out_path: Path) -> None:
    if "note" in event_metrics or not event_metrics:
        return

    event_types = [k for k in event_metrics if k != "note"]
    n  = len(event_types)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    fig.suptitle("Event detection evaluation", fontsize=14,
                 color="#e8e8f0", weight="bold")

    for ax, etype in zip(axes, event_types):
        m      = event_metrics[etype]
        labels = ["Precision", "Recall", "F1"]
        vals   = [m["Precision"], m["Recall"], m["F1"]]
        _bar_chart(ax, labels, vals, [_ACCENT, _GREEN, _AMBER],
                   etype.capitalize(), "Score")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Event figure → {out_path}")


def make_clutch_figure(clutch_df: pd.DataFrame,
                        clutch_stats: dict, out_path: Path) -> None:
    if "note" in clutch_stats or clutch_df.empty:
        return

    scores = clutch_df["clutch_score"].dropna()

    fig = plt.figure(figsize=(12, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Histogram
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(scores, bins=30, color=_ACCENT, edgecolor="#0e1117", linewidth=0.5)
    ax1.axvline(clutch_stats["mean"],   color=_AMBER, lw=1.5,
                linestyle="--", label=f"mean={clutch_stats['mean']:.3f}")
    ax1.axvline(clutch_stats["median"], color=_GREEN, lw=1.5,
                linestyle="--", label=f"median={clutch_stats['median']:.3f}")
    ax1.set_title("Clutch score distribution")
    ax1.set_xlabel("Score");  ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=8)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="y", zorder=0)

    # Stat summary as text
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    summary = (
        f"  N         {clutch_stats['count']}\n"
        f"  Mean      {clutch_stats['mean']:.4f}\n"
        f"  Median    {clutch_stats['median']:.4f}\n"
        f"  Std       {clutch_stats['std']:.4f}\n"
        f"  Min       {clutch_stats['min']:.4f}\n"
        f"  Max       {clutch_stats['max']:.4f}\n"
        f"  P90       {clutch_stats['p90']:.4f}\n"
    )
    if "xg_pearson_r" in clutch_stats:
        summary += (f"\n  Corr(xG)  r={clutch_stats['xg_pearson_r']:.3f}"
                    f"  p={clutch_stats['xg_pearson_p']:.4f}")
    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes,
             va="top", fontsize=10, fontfamily="monospace",
             color="#c8c8d0")
    ax2.set_title("Summary statistics")

    # xG scatter (if available)
    ax3 = fig.add_subplot(gs[2])
    if "xg" in clutch_df.columns:
        ax3.scatter(clutch_df["xg"], scores,
                    alpha=0.5, s=20, color=_ACCENT)
        ax3.set_xlabel("xG baseline");  ax3.set_ylabel("Clutch score")
        ax3.set_title("Clutch vs xG")
        ax3.spines[["top","right"]].set_visible(False)
        ax3.grid(zorder=0)
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "xG column not\nfound in CSV",
                 ha="center", va="center", transform=ax3.transAxes,
                 color="#6060a0")

    fig.suptitle("Clutch Score Analysis", fontsize=14,
                 color="#e8e8f0", weight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Clutch figure → {out_path}")


def make_summary_table(all_metrics: dict, out_path: Path) -> None:
    """Write a clean plain-text metric table for copy-pasting into the thesis."""
    lines = ["=" * 56,
             "  goalX — Evaluation Summary",
             "=" * 56, ""]

    def _section(title, d):
        lines.append(f"  {title}")
        lines.append("  " + "─" * 40)
        if isinstance(d, dict):
            if "note" in d:
                lines.append(f"    {d['note']}")
            else:
                for k, v in d.items():
                    if not isinstance(v, (dict, list)):
                        lines.append(f"    {k:<20} {v}")
        lines.append("")

    _section("TRACKING (MOTChallenge)",  all_metrics.get("tracking", {}))
    for etype, m in all_metrics.get("events", {}).items():
        if etype != "note":
            _section(f"EVENTS — {etype.upper()}", m)
    _section("FORMATION ACCURACY",   all_metrics.get("formation", {}))
    _section("CLUTCH SCORE STATS",   all_metrics.get("clutch",    {}))

    lines += ["=" * 56]
    out_path.write_text("\n".join(lines))
    print(f"  ✔  Summary table → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Evaluator
# ─────────────────────────────────────────────────────────────────

class PipelineEvaluator:
    def __init__(self, tracks_csv, events_csv, formation_csv,
                 clutch_csv, gt_dir: Path, out_dir: Path):
        self.tracks_csv    = tracks_csv
        self.events_csv    = events_csv
        self.formation_csv = formation_csv
        self.clutch_csv    = clutch_csv
        self.gt_dir        = gt_dir
        self.out_dir       = out_dir

    def _read(self, path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def run(self) -> None:
        print(f"\n  goalX — Pipeline Evaluator")
        print(f"  {'─'*40}\n")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        tracks   = self._read(Path(self.tracks_csv))
        events   = self._read(Path(self.events_csv))
        formation = self._read(Path(self.formation_csv))
        clutch   = self._read(Path(self.clutch_csv))

        gt_tracks    = self._read(self.gt_dir / "tracking_gt.csv")
        gt_events    = self._read(self.gt_dir / "events_gt.csv")
        gt_formation = self._read(self.gt_dir / "formation_gt.csv")

        # ── Compute ─────────────────────────────────────────────
        print("  Computing tracking metrics…")
        track_metrics = compute_tracking_metrics(tracks, gt_tracks)

        print("  Computing event metrics…")
        event_metrics = compute_event_metrics(events, gt_events)

        print("  Computing formation accuracy…")
        form_metrics  = compute_formation_accuracy(formation, gt_formation)

        print("  Computing clutch score stats…")
        clutch_stats  = compute_clutch_stats(clutch)

        all_metrics = {
            "tracking":  track_metrics,
            "events":    event_metrics,
            "formation": form_metrics,
            "clutch":    clutch_stats,
        }

        # ── Figures ─────────────────────────────────────────────
        print("\n  Generating figures…")
        make_tracking_figure(track_metrics,
                             self.out_dir / "eval_tracking.png")
        make_event_figure(event_metrics,
                          self.out_dir / "eval_events.png")
        make_clutch_figure(clutch, clutch_stats,
                           self.out_dir / "eval_clutch.png")
        make_summary_table(all_metrics,
                           self.out_dir / "eval_summary.txt")

        # ── JSON dump ───────────────────────────────────────────
        json_path = self.out_dir / "eval_metrics.json"
        with open(json_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)

        print(f"\n  ✅  Evaluation complete → {self.out_dir}")
        print(f"      eval_metrics.json  — machine-readable all metrics")
        print(f"      eval_summary.txt   — human-readable for thesis\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the goalX pipeline against ground-truth labels."
    )
    p.add_argument("--tracks",    required=True)
    p.add_argument("--events",    required=True)
    p.add_argument("--formation", required=True)
    p.add_argument("--clutch",    required=True)
    p.add_argument("--gt-dir",    default="data/ground_truth",
                   help="Directory containing *_gt.csv files")
    p.add_argument("--out-dir",   default="outputs/evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    PipelineEvaluator(
        tracks_csv    = args.tracks,
        events_csv    = args.events,
        formation_csv = args.formation,
        clutch_csv    = args.clutch,
        gt_dir        = Path(args.gt_dir),
        out_dir       = Path(args.out_dir),
    ).run()