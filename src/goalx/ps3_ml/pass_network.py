"""
pass_network.py  —  PS3 Step 3
────────────────────────────────
Constructs a directed pass network from possession event sequences,
computes graph-theoretic centrality metrics, and renders a publication-
quality network visualisation on the 2D pitch canvas.

FIXES
─────────────────────────────────
FIX 1 — Ball excluded from graph (CRITICAL)
  Filters out all rows where passer_id < 0 OR receiver_id < 0 before
  building the NetworkX graph. Excludes the ball from being the #1 
  playmaker by PageRank.

FIX 2 — Same-player self-passes filtered
  Prevents A→A passes when possession flickers.

FIX 3 — "ball" / "uncertain" teams excluded
  Pass network only builds per-team graphs for "home" and "away".
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    print("  ⚠  networkx not installed: pip install networkx")

# ─────────────────────────────────────────────────────────────────
#  Style & Config
# ─────────────────────────────────────────────────────────────────

PITCH_SCALE = 10.0
_DARK_BG    = "#0e1117"
_SURFACE    = "#1a1d23"
_TEXT       = "#c8c8d0"

TEAM_PALETTE = {
    "home": ("#3b82f6", "#1d4ed8"),   # (node_color, edge_color)
    "away": ("#ef4444", "#b91c1c"),
    "ref":  ("#94a3b8", "#64748b"),
}

plt.rcParams.update({
    "figure.facecolor": _DARK_BG,
    "axes.facecolor":   "#1a3a1a",
    "text.color":       _TEXT,
    "font.size":        9,
})

VALID_PASS_TEAMS = {"home", "away"}   # FIX 3: only real teams


# ─────────────────────────────────────────────────────────────────
#  Pass extraction
# ─────────────────────────────────────────────────────────────────

def _nearest_player(frame_id: int,
                    tracks: pd.DataFrame,
                    ball_x: float,
                    ball_y: float) -> int | None:
    """Return track_id of the closest player to the ball at this frame."""
    
    # FIX: If the ball position is NaN, we can't find a nearest player!
    if pd.isna(ball_x) or pd.isna(ball_y):
        return None
        
    # FIX: Only look at valid players (track_id >= 0) who have valid coordinates
    ft = tracks[(tracks["frame_id"] == frame_id) & 
                (tracks["track_id"] >= 0)].dropna(subset=["pitch_x", "pitch_y"])
                
    if ft.empty:
        return None
        
    dx = ft["pitch_x"] - ball_x
    dy = ft["pitch_y"] - ball_y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    
    # Fallback guard just in case
    if dist.isna().all():
        return None
        
    idx  = dist.idxmin()
    return int(ft.loc[idx, "track_id"])


def extract_passes(events: pd.DataFrame,
                   tracks: pd.DataFrame,
                   ball: pd.DataFrame,
                   team_map: dict[int, str]
                   ) -> pd.DataFrame:
    """Infer pass events from possession transitions."""
    poss = events[events["event_type"] == "possession"].copy()
    
    if poss.empty:
        print("  ⚠  No possession events found. Using ball proximity to infer.")
        if ball.empty:
            print("  ❌  No ball data either. Cannot extract passes.")
            return pd.DataFrame()

        rows = []
        for _, brow in ball.iterrows():
            fid = int(brow["frame_id"])
            pid = _nearest_player(fid, tracks, float(brow["pitch_x"]), float(brow["pitch_y"]))
            if pid is not None:
                rows.append({"frame_id": fid, "track_id": pid, "event_type": "possession"})
        poss = pd.DataFrame(rows)

    poss = poss.sort_values("frame_id").reset_index(drop=True)

    passes: list[dict] = []
    prev_id    = None
    prev_frame = None
    prev_team  = None

    for _, row in poss.iterrows():
        tid_val = row.get("track_id", -1)
        curr_id = int(tid_val) if pd.notna(tid_val) else -1
        curr_frame = int(row["frame_id"])
        curr_team  = team_map.get(curr_id, "unknown")

        if (prev_id is not None and
                curr_id != prev_id and
                curr_team == prev_team and
                curr_team in VALID_PASS_TEAMS):   # FIX 3: Only valid teams

            # ── FIX 1: skip passes involving the ball (track_id < 0) ──
            if prev_id < 0 or curr_id < 0:
                prev_id = curr_id; prev_frame = curr_frame; prev_team = curr_team
                continue

            # ── FIX 2: skip self-passes ──────────────────────────────
            if prev_id == curr_id:
                prev_id = curr_id; prev_frame = curr_frame; prev_team = curr_team
                continue

            def _pos(fid, tid):
                ft = tracks[(tracks["frame_id"] == fid) & (tracks["track_id"] == tid)]
                if ft.empty:
                    return (np.nan, np.nan)
                row_ = ft.iloc[0]
                x = row_.get("smooth_x", row_.get("pitch_x", np.nan))
                y = row_.get("smooth_y", row_.get("pitch_y", np.nan))
                return (float(x), float(y))

            px, py = _pos(prev_frame, prev_id)
            rx, ry = _pos(curr_frame, curr_id)

            passes.append({
                "passer_id":    prev_id,
                "receiver_id":  curr_id,
                "passer_team":  prev_team,
                "receiver_team": curr_team,
                "frame_start":  prev_frame,
                "frame_end":    curr_frame,
                "passer_x":    px,
                "passer_y":    py,
                "receiver_x":  rx,
                "receiver_y":  ry,
            })

        prev_id    = curr_id
        prev_frame = curr_frame
        prev_team  = curr_team

    df = pd.DataFrame(passes)
    if not df.empty:
        n_total = len(df)
        n_valid = len(df[df["passer_id"] >= 0])
        print(f"  Passes extracted: {n_total:,} (after filtering ball rows: {n_valid:,})")
    return df


# ─────────────────────────────────────────────────────────────────
#  Graph construction + centrality
# ─────────────────────────────────────────────────────────────────

def build_network(passes: pd.DataFrame, team_filter: str | None = None) -> "nx.DiGraph":
    """Build a weighted directed graph from the pass DataFrame."""
    G = nx.DiGraph()
    if passes.empty:
        return G

    filtered = passes[passes["passer_team"] == team_filter] if team_filter else passes

    # FIX 1: Ensure no ball node ever enters the graph
    filtered = filtered[(filtered["passer_id"] >= 0) & (filtered["receiver_id"] >= 0)]

    for _, row in filtered.iterrows():
        u, v = int(row["passer_id"]), int(row["receiver_id"])
        if u == v:
            continue   # FIX 2: Skip self-loops
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)

    return G


def compute_centrality(G: "nx.DiGraph") -> pd.DataFrame:
    """Compute all centrality metrics and return as a tidy DataFrame."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    in_deg      = dict(G.in_degree(weight="weight"))
    out_deg     = dict(G.out_degree(weight="weight"))
    pagerank    = nx.pagerank(G, weight="weight", alpha=0.85, max_iter=500)
    clust       = nx.clustering(G.to_undirected(), weight="weight")

    nodes  = sorted(G.nodes())
    return pd.DataFrame({
        "track_id":    nodes,
        "betweenness": [round(betweenness.get(n, 0), 5) for n in nodes],
        "in_degree":   [in_deg.get(n, 0)  for n in nodes],
        "out_degree":  [out_deg.get(n, 0) for n in nodes],
        "pagerank":    [round(pagerank.get(n, 0), 5) for n in nodes],
        "clustering":  [round(clust.get(n, 0), 5)  for n in nodes],
    }).sort_values("pagerank", ascending=False)


# ─────────────────────────────────────────────────────────────────
#  Visualisation — network on pitch
# ─────────────────────────────────────────────────────────────────

def draw_network_on_pitch(G: "nx.DiGraph",
                           centrality: pd.DataFrame,
                           passes: pd.DataFrame,
                           tracks: pd.DataFrame,
                           team_map: dict[int, str],
                           pitch_img: np.ndarray,
                           title: str,
                           out_path: Path) -> None:
    """Draw the pass network overlaid on the 2D pitch canvas."""
    ph, pw = pitch_img.shape[:2]
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)

    pitch_rgb = cv2.cvtColor(pitch_img, cv2.COLOR_BGR2RGB)
    ax.imshow(pitch_rgb, extent=[0, pw, ph, 0], alpha=0.55, zorder=0)
    ax.set_xlim(0, pw);  ax.set_ylim(ph, 0)
    ax.axis("off")

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No pass network data", ha="center", va="center", transform=ax.transAxes, color=_TEXT, fontsize=14)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        return

    coord_x = "smooth_x" if "smooth_x" in tracks.columns else "pitch_x"
    coord_y = "smooth_y" if "smooth_y" in tracks.columns else "pitch_y"
    
    avg_pos = tracks[tracks["track_id"].isin(G.nodes())].dropna(subset=[coord_x, coord_y]).groupby("track_id")[[coord_x, coord_y]].mean()
    pos = {int(tid): (row[coord_x], row[coord_y]) for tid, row in avg_pos.iterrows() if int(tid) in G.nodes()}

    pr_map = dict(zip(centrality["track_id"], centrality["pagerank"]))
    max_pr = max(pr_map.values()) if pr_map else 1
    max_w  = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)

    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]; x1, y1 = pos[v]
        w = data["weight"]
        team = team_map.get(u, "ref")
        _, ecol = TEAM_PALETTE.get(team, ("#94a3b8", "#64748b"))

        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=ecol, lw=0.5 + (w / max_w) * 3.5, alpha=0.55 + (w / max_w) * 0.35, connectionstyle="arc3,rad=0.12"), zorder=2)

    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        team = team_map.get(node, "ref")
        ncol, _ = TEAM_PALETTE.get(team, ("#94a3b8", "#64748b"))
        size = 120 + (pr_map.get(node, 0) / max_pr) * 800

        ax.scatter(x, y, s=size, c=ncol, zorder=4, edgecolors="white", linewidths=1.0)
        ax.text(x, y - 18, str(node), ha="center", va="bottom", fontsize=7, color="white", fontweight="bold", zorder=5)

    patches = [mpatches.Patch(color="#3b82f6", label="Home"), mpatches.Patch(color="#ef4444", label="Away")]
    ax.legend(handles=patches, loc="lower right", facecolor=_SURFACE, edgecolor=_TEXT, labelcolor=_TEXT, fontsize=9)
    ax.set_title(title, color="#e8e8f0", fontsize=13, pad=10, fontweight="bold")

    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Network figure → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────

class PassNetworkAnalyser:
    def __init__(self, events_csv: Path, tracks_csv: Path, teams_csv: Path, ball_csv: Path, pitch_path: Path, out_dir: Path):
        self.events_csv = events_csv
        self.tracks_csv = tracks_csv
        self.teams_csv  = teams_csv
        self.ball_csv   = ball_csv
        self.pitch_path = pitch_path
        self.out_dir    = out_dir

    def run(self) -> None:
        print(f"\n  goalX PS3 — Pass Network Analyser")
        print(f"  {'─'*42}\n")

        if not _NX_AVAILABLE:
            print("  ❌  networkx required: pip install networkx"); return

        self.out_dir.mkdir(parents=True, exist_ok=True)

        events = pd.read_csv(self.events_csv)
        tracks = pd.read_csv(self.tracks_csv)
        teams  = pd.read_csv(self.teams_csv)
        ball   = pd.read_csv(self.ball_csv) if self.ball_csv.exists() else pd.DataFrame()
        pitch  = cv2.imread(str(self.pitch_path))

        for df in [events, tracks]:
            if "frame" in df.columns and "frame_id" not in df.columns:
                df.rename(columns={"frame": "frame_id"}, inplace=True)

        team_map: dict[int, str] = dict(zip(teams["track_id"].astype(int), teams["team"].astype(str)))

        passes = extract_passes(events, tracks, ball, team_map)
        if passes.empty:
            print("  ❌  No passes extracted. Check events.csv."); return

        passes.to_csv(self.out_dir / "pass_network.csv", index=False)
        print(f"  ✔  Pass CSV → {self.out_dir}/pass_network.csv")

        all_metrics: list[pd.DataFrame] = []

        for team in ("home", "away", None):
            label = team if team else "both"
            G = build_network(passes, team_filter=team)

            if G.number_of_nodes() < 2:
                print(f"  ⚠  {label}: fewer than 2 nodes — skipping.")
                continue

            centrality = compute_centrality(G)
            centrality["team"] = label
            all_metrics.append(centrality)
            centrality.to_csv(self.out_dir / f"centrality_{label}.csv", index=False)

            print(f"\n  [{label.upper()} team] Top 5 by PageRank:")
            print(f"  {'ID':<6} {'PageRank':>9} {'Betweenness':>12} {'In-deg':>8} {'Out-deg':>9}")
            print(f"  {'─'*48}")
            for _, row in centrality.head(5).iterrows():
                print(f"  {int(row['track_id']):<6} {row['pagerank']:>9.4f} {row['betweenness']:>12.4f} {int(row['in_degree']):>8} {int(row['out_degree']):>9}")

            if pitch is not None:
                draw_network_on_pitch(G, centrality, passes, tracks, team_map, pitch, title=f"Pass network — {label} team", out_path=self.out_dir / f"pass_network_{label}.png")

        if all_metrics:
            pd.concat(all_metrics, ignore_index=True).to_csv(self.out_dir / "centrality_all.csv", index=False)
            print(f"\n  ✔  All centrality → {self.out_dir}/centrality_all.csv")

        print(f"\n  ✅  Pass network complete → {self.out_dir}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Pass network analysis for goalX PS3.")
    p.add_argument("--events",  default="outputs/events.csv")
    p.add_argument("--tracks",  default="outputs/smoothed_tracks.csv")
    p.add_argument("--teams",   default="outputs/team_labels.csv")
    p.add_argument("--ball",    default="outputs/ball_trajectory/interpolated_ball.csv")
    p.add_argument("--pitch",   default="data/pitch_map.png")
    p.add_argument("--out-dir", default="outputs/pass_network")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    PassNetworkAnalyser(
        events_csv = Path(args.events),
        tracks_csv = Path(args.tracks),
        teams_csv  = Path(args.teams),
        ball_csv   = Path(args.ball) if args.ball else Path(""),
        pitch_path = Path(args.pitch),
        out_dir    = Path(args.out_dir),
    ).run()