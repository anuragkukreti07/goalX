"""
export_report.py  —  PS3 Step 5
─────────────────────────────────
Self-contained HTML match report. All images embedded as base64 data URIs.
Open on any machine at your defense with no internet required.
"""
from __future__ import annotations
import argparse, base64, io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── style ────────────────────────────────────────────────────────
_DARK = "#0e1117"; _SURF = "#1a1d23"; _BORD = "#2e3140"; _TEXT = "#c8c8d0"
_ACCENT = "#5b8dee"; _GREEN = "#3dbf7a"; _AMBER = "#f5a623"; _RED = "#e05c5c"

plt.rcParams.update({
    "figure.facecolor":_DARK,"axes.facecolor":_SURF,"axes.edgecolor":_BORD,
    "axes.labelcolor":_TEXT,"xtick.color":"#9090a0","ytick.color":"#9090a0",
    "text.color":_TEXT,"grid.color":_BORD,"grid.linewidth":0.5,"font.size":10,
    "legend.facecolor":_SURF,"legend.edgecolor":_BORD,
})

# ── asset helpers ────────────────────────────────────────────────
def _b64(path: Path) -> str:
    if not path or not path.exists(): return ""
    ext = path.suffix.lower().lstrip(".")
    mime = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg"}.get(ext,"image/png")
    return f"data:{mime};base64,"+base64.b64encode(path.read_bytes()).decode()

def _img(path: Path, caption: str = "", width: str = "100%") -> str:
    src = _b64(path)
    if not src: return f'<p class="missing">Not found: <code>{path.name}</code></p>'
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" width="{width}" alt="{caption}" loading="lazy">{cap}</figure>'

def _pngs(directory: Path) -> list:
    if not directory or not directory.exists(): return []
    return sorted(directory.glob("*.png"))

def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return "data:image/png;base64,"+base64.b64encode(buf.read()).decode()

# ── inline chart builders ────────────────────────────────────────
def _xg_chart(xg_df: pd.DataFrame) -> str:
    if xg_df.empty or "xg_predicted" not in xg_df.columns: return ""
    fig, ax = plt.subplots(figsize=(13, 3.2))
    fig.patch.set_facecolor(_DARK)
    f = xg_df["frame_id"].values; x = xg_df["xg_predicted"].values
    ax.fill_between(f, x, alpha=0.3, color=_ACCENT)
    ax.plot(f, x, color=_ACCENT, lw=1.5)
    ax.set_xlabel("Frame"); ax.set_ylabel("xG")
    ax.set_title("Expected goals timeline", color="#e8e8f0", fontweight="bold")
    ax.set_ylim(0, 1.05); ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    return _fig_b64(fig)

def _formation_chart(form_df: pd.DataFrame) -> str:
    if form_df.empty or "formation" not in form_df.columns: return ""
    teams = form_df["team"].unique() if "team" in form_df.columns else ["all"]
    fig, axes = plt.subplots(len(teams), 1,
                              figsize=(13, 2.5*len(teams)), squeeze=False)
    fig.patch.set_facecolor(_DARK)
    for i, team in enumerate(teams):
        ax = axes[i, 0]; ax.set_facecolor(_SURF)
        sub = form_df[form_df["team"]==team].sort_values("frame_id") \
              if "team" in form_df.columns else form_df.sort_values("frame_id")
        fmts = sub["formation"].unique()
        cmap = {f: plt.cm.tab10(j/max(len(fmts),1)) for j,f in enumerate(fmts)}
        prev_fmt = prev_fid = None
        for _, row in sub.iterrows():
            fmt = row["formation"]; fid = int(row["frame_id"])
            if prev_fmt is not None:
                ax.barh(0, fid-prev_fid, left=prev_fid, height=0.8,
                        color=cmap[prev_fmt], alpha=0.85)
            prev_fmt=fmt; prev_fid=fid
        ax.set_yticks([])
        ax.set_title(f"Formation — {team}", color="#e8e8f0", fontsize=10)
        ax.spines[["top","right","left"]].set_visible(False)
        handles=[mpatches.Patch(color=cmap[f],label=f) for f in fmts]
        ax.legend(handles=handles, loc="lower right", fontsize=8,
                  facecolor=_SURF, edgecolor=_TEXT, labelcolor=_TEXT)
    plt.tight_layout(pad=0.8)
    return _fig_b64(fig)

def _action_chart(actions_df: pd.DataFrame) -> str:
    if actions_df.empty or "action" not in actions_df.columns: return ""
    dist = actions_df.groupby(["track_id","action"]).size().unstack(fill_value=0)
    if dist.empty: return ""
    ACOLS = {"IDLE":"#6b7280","CARRY":"#3b82f6","DRIBBLE":"#8b5cf6",
             "PASS":"#10b981","SHOT":"#f59e0b","PRESS":"#ef4444","TACKLE":"#ec4899"}
    fig, ax = plt.subplots(figsize=(13, max(3, len(dist)*0.45)))
    fig.patch.set_facecolor(_DARK); ax.set_facecolor(_SURF)
    bottom = np.zeros(len(dist))
    for action in dist.columns:
        v = dist[action].values
        ax.bar(range(len(dist)), v, bottom=bottom, width=0.75,
               color=ACOLS.get(action,"#999"), label=action, zorder=3)
        bottom += v
    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels([f"ID {t}" for t in dist.index], rotation=45, ha="right")
    ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_title("Per-player action breakdown", color="#e8e8f0", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper right", facecolor=_SURF, edgecolor=_TEXT,
              labelcolor=_TEXT, fontsize=8)
    plt.tight_layout(); return _fig_b64(fig)

# ── stat cards ───────────────────────────────────────────────────
def _card(val, lbl, col="#5b8dee"):
    return f'<div class="stat-card"><div class="stat-val" style="color:{col}">{val}</div><div class="stat-lbl">{lbl}</div></div>'

def _stat_row(tracks, events, clutch):
    cards = []
    if not tracks.empty and "frame_id" in tracks.columns:
        mins = round(tracks["frame_id"].nunique()/25/60, 1)
        cards.append(_card(f"{mins} min","Tracking duration",_GREEN))
    if not tracks.empty and "track_id" in tracks.columns:
        cards.append(_card(tracks[tracks["track_id"]>=0]["track_id"].nunique(),"Player IDs",_ACCENT))
    if not events.empty and "event_type" in events.columns:
        cards.append(_card(len(events[events["event_type"]=="shot"]),"Shots",_AMBER))
    if not clutch.empty and "clutch_score" in clutch.columns:
        cards.append(_card(f"{clutch['clutch_score'].max():.3f}","Peak clutch",_RED))
    return '<div class="stat-row">'+"".join(cards)+"</div>"

# ── clutch table ─────────────────────────────────────────────────
def _clutch_table(clutch):
    if clutch.empty or "clutch_score" not in clutch.columns:
        return '<p class="note">No clutch data.</p>'
    top = clutch.nlargest(10,"clutch_score")
    cols = [c for c in top.columns if c!="xg_heuristic"]
    hdr = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for _, row in top[cols].iterrows():
        body += "<tr>"+"".join(f"<td>{f'{v:.4f}' if isinstance(v,float) else v}</td>"
                               for v in row)+"</tr>\n"
    return f"<table><thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table>"

# ── CSS ──────────────────────────────────────────────────────────
_CSS = """
:root{--bg:#0e1117;--surf:#1a1d23;--bord:#2e3140;--text:#c8c8d0;--r:10px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.7}
header{background:var(--surf);border-bottom:1px solid var(--bord);padding:32px 48px}
header h1{font-size:28px;color:#e8e8f0;font-weight:600}
header p{color:#6a6d80;margin-top:6px}
.badge{display:inline-block;background:#5b8dee;color:#fff;font-size:11px;font-weight:600;padding:2px 10px;border-radius:20px;margin-left:10px;vertical-align:middle}
nav{background:var(--surf);border-bottom:1px solid var(--bord);padding:0 48px;display:flex;flex-wrap:wrap}
nav a{color:#6a6d80;text-decoration:none;padding:12px 16px;font-size:13px;border-bottom:2px solid transparent;transition:color .15s,border-color .15s}
nav a:hover{color:#5b8dee;border-color:#5b8dee}
.content{max-width:1280px;margin:0 auto;padding:40px 48px}
section{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);padding:28px 32px;margin-bottom:28px}
section h2{font-size:18px;color:#e8e8f0;font-weight:600;margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid var(--bord)}
section h3{font-size:15px;color:#d0d0e0;margin:18px 0 8px}
figure{margin:14px 0;text-align:center}
figure img{border-radius:8px;border:1px solid var(--bord);max-width:100%}
figcaption{margin-top:6px;font-size:12px;color:#6a6d80;font-style:italic}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:14px 0}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px}
th{background:#252830;color:#d0d0e0;text-align:left;padding:8px 12px;font-weight:600;border-bottom:1px solid var(--bord)}
td{padding:7px 12px;border-bottom:1px solid #202328}
tr:hover td{background:#1f222a}
.stat-row{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}
.stat-card{background:#141720;border:1px solid var(--bord);border-radius:8px;padding:16px 20px;text-align:center;flex:1;min-width:110px}
.stat-val{font-size:26px;font-weight:700}
.stat-lbl{font-size:11px;color:#6a6d80;margin-top:4px}
.note{color:#6a6d80;font-size:12px;font-style:italic;margin:8px 0}
.missing{color:#e05c5c;font-size:12px}
code{background:#252830;padding:2px 6px;border-radius:4px;font-family:monospace;font-size:12px}
ol{margin:12px 0 0 20px;line-height:2}
footer{text-align:center;padding:36px;color:#6a6d80;font-size:12px;border-top:1px solid var(--bord);margin-top:32px}
"""

# ── HTML builder ─────────────────────────────────────────────────
def build_html(tracks, events, clutch, xg_df, actions, form_df,
               pass_net_dir, radar_dir, heatmaps_dir, title, ts):

    stat_row_html  = _stat_row(tracks, events, clutch)
    xg_src         = _xg_chart(xg_df)
    form_src       = _formation_chart(form_df)
    action_src     = _action_chart(actions)
    clutch_tbl_html = _clutch_table(clutch)

    def _inline(src, alt):
        return (f'<figure><img src="{src}" width="100%" alt="{alt}" loading="lazy"></figure>'
                if src else f'<p class="note">No data for {alt}.</p>')

    pass_imgs = ("".join(_img(p, p.stem.replace("_"," ")) for p in _pngs(pass_net_dir))
                 or '<p class="note">No pass network images.</p>')

    radar_imgs = (('<div class="grid-2">' +
                   "".join(_img(p,p.stem) for p in _pngs(radar_dir)) + "</div>")
                  if _pngs(radar_dir) else '<p class="note">No radar charts.</p>')

    hmap_imgs = (('<div class="grid-2">' +
                  "".join(_img(p,p.stem.replace("_"," ")) for p in _pngs(heatmaps_dir)) + "</div>")
                 if _pngs(heatmaps_dir) else '<p class="note">No heatmaps.</p>')

    nav = "".join(f'<a href="#{s}">{l}</a>' for s,l in [
        ("overview","Overview"),("xg","xG"),("clutch","Clutch"),
        ("passes","Passes"),("formation","Formation"),("actions","Actions"),
        ("radars","Radar"),("heatmaps","Heatmaps"),("method","Method"),
    ])

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title><style>{_CSS}</style></head><body>
<header>
  <h1>{title} <span class="badge">goalX PS3</span></h1>
  <p>End-to-end football analytics — M.Tech thesis output</p>
  <p style="font-size:12px;margin-top:8px;color:#555">Generated {ts}</p>
</header>
<nav>{nav}</nav>
<div class="content">

<section id="overview"><h2>Match overview</h2>{stat_row_html}</section>

<section id="xg">
  <h2>Expected goals (xG) timeline</h2>
  <p>Shot probability per event from the logistic regression xG model
  (<code>train_xg.py</code>). Peaks = high-probability shot moments.</p>
  {_inline(xg_src,"xG timeline")}
</section>

<section id="clutch">
  <h2>Clutch score leaderboard</h2>
  <p>Clutch Score = xG × (1 + 0.5 × pressure) × control_weight ×
  temporal_weight. Higher = goal under greater contextual difficulty.</p>
  {clutch_tbl_html}
</section>

<section id="passes">
  <h2>Pass network</h2>
  <p>Node size ∝ PageRank influence. Edge thickness ∝ pass frequency.
  Computed via NetworkX (<code>pass_network.py</code>).</p>
  {pass_imgs}
</section>

<section id="formation">
  <h2>Formation timeline</h2>
  <p>Rolling majority-vote formation per team via depth-axis K-Means.</p>
  {_inline(form_src,"Formation timeline")}
</section>

<section id="actions">
  <h2>Player action breakdown</h2>
  <p>Per-frame labels: IDLE, CARRY, DRIBBLE, PASS, SHOT, PRESS, TACKLE.
  Rule-based kinematic classifier (<code>action_classifier.py</code>).</p>
  {_inline(action_src,"Action breakdown")}
</section>

<section id="radars">
  <h2>Tactical radar charts</h2>
  <p>Six-dimensional player profile: distance, speed, possession, pressure,
  pitch control, clutch score.</p>
  {radar_imgs}
</section>

<section id="heatmaps">
  <h2>Velocity &amp; activity heatmaps</h2>
  <p>Gaussian-smoothed spatial speed and time-on-pitch per team (σ = 14 px).</p>
  {hmap_imgs}
</section>

<section id="method">
  <h2>Methodology note</h2>
  <p>15-stage pipeline from raw broadcast video to structured tactical data:</p>
  <ol>
    <li>Frame extraction (<code>extract_frames.py</code>)</li>
    <li>Detection — SAHI + YOLOv8 (<code>detect_ball.py</code>)</li>
    <li>Tracking — ByteTrack (<code>track_players.py</code>)</li>
    <li>Homography calibration (<code>homography_picker.py</code>)</li>
    <li>Coordinate projection (<code>project_tracks.py</code>)</li>
    <li>Trajectory smoothing (<code>smooth_tracks.py</code>)</li>
    <li>Event extraction (<code>extract_events.py</code>)</li>
    <li>Team classification — K-Means jersey colour</li>
    <li>Formation detection — depth-axis clustering</li>
    <li>Pitch control — Voronoi tessellation</li>
    <li>Clutch score — xG × pressure × control × time</li>
    <li>xG model — logistic regression + Platt calibration (<code>train_xg.py</code>)</li>
    <li>Ball trajectory interpolation (<code>ball_trajectory.py</code>)</li>
    <li>Pass network — NetworkX + PageRank (<code>pass_network.py</code>)</li>
    <li>Action classification — kinematic rules + optional RF (<code>action_classifier.py</code>)</li>
  </ol>
  <p style="margin-top:12px">Coordinate system: 1050×680 px at 10 px/m (105×68 m FIFA). FPS: 25.</p>
</section>

</div>
<footer>goalX &middot; M.Tech Thesis &middot; Self-contained &middot; Generated {ts}</footer>
</body></html>"""

# ── main class ───────────────────────────────────────────────────
class ReportExporter:
    def __init__(self, **kw):
        self.k = kw

    def _load(self, key):
        p = self.k.get(key,"")
        if p and Path(p).exists(): return pd.read_csv(p)
        return pd.DataFrame()

    def run(self):
        print(f"\n  goalX PS3 — Report Exporter\n  {'─'*40}\n")

        tracks  = self._load("tracks")
        events  = self._load("events")
        clutch  = self._load("clutch")
        xg_df   = self._load("xg_pred")
        actions = self._load("actions")
        form_df = self._load("formation")

        for df in [tracks, events, clutch, xg_df, actions, form_df]:
            if not df.empty and "frame" in df.columns and "frame_id" not in df.columns:
                df.rename(columns={"frame":"frame_id"}, inplace=True)

        ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = self.k.get("title","goalX Match Report")

        html = build_html(
            tracks       = tracks,   events     = events,
            clutch       = clutch,   xg_df      = xg_df,
            actions      = actions,  form_df    = form_df,
            pass_net_dir = Path(self.k.get("pass_net_dir","")),
            radar_dir    = Path(self.k.get("radar_dir","")),
            heatmaps_dir = Path(self.k.get("heatmaps_dir","")),
            title        = title,    ts         = ts,
        )

        out = Path(self.k["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        size_kb = out.stat().st_size // 1024
        print(f"  ✅  Report → {out}  ({size_kb} KB, self-contained)")
        print(f"      Open: firefox {out}\n")

# ── CLI ──────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Export HTML match report (goalX PS3).")
    # Aligned default paths to match flat outputs structure used by run_goalx.py
    p.add_argument("--tracks",       default="outputs/smoothed_tracks.csv")
    p.add_argument("--events",       default="outputs/events.csv")
    p.add_argument("--clutch",       default="outputs/clutch_scores.csv")
    p.add_argument("--xg-pred",      default="outputs/xg_model/xg_predictions.csv")
    p.add_argument("--pass-net-dir", default="outputs/pass_network")
    p.add_argument("--actions",      default="outputs/actions/actions.csv")
    p.add_argument("--radar-dir",    default="outputs/radar_charts")
    p.add_argument("--heatmaps-dir", default="outputs/heatmaps")
    p.add_argument("--formation",    default="outputs/formations.csv")
    p.add_argument("--out",          default="outputs/match_report.html")
    p.add_argument("--title",        default="goalX Match Report")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    ReportExporter(
        tracks       = args.tracks,    events     = args.events,
        clutch       = args.clutch,    xg_pred    = args.xg_pred,
        pass_net_dir = args.pass_net_dir, actions = args.actions,
        radar_dir    = args.radar_dir, heatmaps_dir = args.heatmaps_dir,
        formation    = args.formation, out        = args.out,
        title        = args.title,
    ).run()