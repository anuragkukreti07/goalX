"""
export_report.py
────────────────
Generates a single self-contained HTML thesis report that embeds every
chart, heatmap, radar figure, and metric table produced by the goalX
pipeline. No external dependencies at view time — open on any machine.

Usage
─────
  python3 src/goalx/export_report.py \\
      --eval-dir         outputs/evaluation \\
      --radar-charts-dir outputs/radar_charts \\
      --heatmaps-dir     outputs/heatmaps \\
      --clutch           outputs/clutch_scores.csv \\
      --events           outputs/events.csv \\
      --out              outputs/thesis_report.html
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────────────
#  Asset helpers
# ─────────────────────────────────────────────────────────────────

def _b64_img(path: Path) -> str:
    """Return a data-URI string for embedding a PNG/JPG in HTML."""
    if not path.exists():
        return ""
    ext = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _img_tag(path: Path, caption: str = "", width: str = "100%") -> str:
    src = _b64_img(path)
    if not src:
        return f'<p class="missing">Figure not found: {path.name}</p>'
    cap = f'<figcaption>{caption}</figcaption>' if caption else ""
    return (f'<figure>'
            f'<img src="{src}" width="{width}" alt="{caption}">'
            f'{cap}</figure>')


def _all_pngs(directory: Path, pattern: str = "*.png") -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


# ─────────────────────────────────────────────────────────────────
#  Metric table builder
# ─────────────────────────────────────────────────────────────────

def _dict_to_table(d: dict, title: str = "") -> str:
    if not d or "note" in d:
        note = d.get("note", "No data.") if d else "No data."
        return f'<p class="note">{note}</p>'

    rows = ""
    for k, v in d.items():
        if isinstance(v, dict):
            continue  # nested — skip for flat table
        rows += f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>\n"

    return f"""
<div class="metric-block">
  {"<h4>" + title + "</h4>" if title else ""}
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _events_table(events_df: pd.DataFrame) -> str:
    if events_df.empty:
        return '<p class="note">No events found.</p>'
    sample = events_df.head(30)
    header = "".join(f"<th>{c}</th>" for c in sample.columns)
    body   = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in sample.itertuples(index=False)
    )
    return f"""
<table class="events-table">
  <thead><tr>{header}</tr></thead>
  <tbody>{body}</tbody>
</table>
<p class="note">Showing first 30 rows of {len(events_df)} events.</p>"""


def _clutch_top_table(clutch_df: pd.DataFrame) -> str:
    if clutch_df.empty or "clutch_score" not in clutch_df.columns:
        return '<p class="note">No clutch scores available.</p>'
    top = clutch_df.nlargest(15, "clutch_score")
    header = "".join(f"<th>{c}</th>" for c in top.columns)
    body   = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in top.itertuples(index=False)
    )
    return f"""
<table class="events-table">
  <thead><tr>{header}</tr></thead>
  <tbody>{body}</tbody>
</table>
<p class="note">Top 15 highest clutch score events.</p>"""


# ─────────────────────────────────────────────────────────────────
#  HTML template
# ─────────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg:      #0e1117;
  --surface: #1a1d23;
  --border:  #2e3140;
  --text:    #c8c8d0;
  --accent:  #5b8dee;
  --green:   #3dbf7a;
  --amber:   #f5a623;
  --muted:   #6a6d80;
  --radius:  10px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.7;
}
header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 32px 48px;
}
header h1 { font-size: 28px; color: #e8e8f0; font-weight: 600; }
header p  { color: var(--muted); margin-top: 6px; }
.badge {
  display: inline-block;
  background: var(--accent);
  color: #fff;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 10px;
  border-radius: 20px;
  margin-left: 10px;
  vertical-align: middle;
}
nav {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 48px;
  display: flex;
  gap: 0;
}
nav a {
  color: var(--muted);
  text-decoration: none;
  padding: 12px 18px;
  font-size: 13px;
  border-bottom: 2px solid transparent;
  transition: color .15s, border-color .15s;
}
nav a:hover { color: var(--accent); border-color: var(--accent); }
.content { max-width: 1200px; margin: 0 auto; padding: 40px 48px; }
section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 32px;
  margin-bottom: 32px;
}
section h2 {
  font-size: 18px;
  color: #e8e8f0;
  font-weight: 600;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
section h3 { font-size: 15px; color: #d0d0e0; margin: 20px 0 10px; }
section h4 { font-size: 13px; color: var(--muted); margin: 12px 0 6px; }
figure {
  margin: 16px 0;
  text-align: center;
}
figure img {
  border-radius: 8px;
  border: 1px solid var(--border);
  max-width: 100%;
}
figcaption {
  margin-top: 8px;
  font-size: 12px;
  color: var(--muted);
  font-style: italic;
}
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin: 16px 0;
}
.grid-3 {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 16px;
  margin: 16px 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0;
  font-size: 13px;
}
th {
  background: #252830;
  color: #d0d0e0;
  text-align: left;
  padding: 8px 12px;
  font-weight: 600;
  border-bottom: 1px solid var(--border);
}
td {
  padding: 7px 12px;
  border-bottom: 1px solid #202328;
  color: var(--text);
}
tr:hover td { background: #1f222a; }
.metric-block {
  background: #141720;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin: 8px 0;
}
.metric-block table { margin: 0; }
.note { color: var(--muted); font-size: 12px; font-style: italic; margin: 8px 0; }
.missing { color: #e05c5c; font-size: 12px; }
.stat-card {
  background: #141720;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  text-align: center;
}
.stat-card .val { font-size: 26px; font-weight: 700; color: var(--accent); }
.stat-card .lbl { font-size: 11px; color: var(--muted); margin-top: 4px; }
.stat-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin: 16px 0;
}
.stat-row .stat-card { flex: 1; min-width: 110px; }
footer {
  text-align: center;
  padding: 40px;
  color: var(--muted);
  font-size: 12px;
  border-top: 1px solid var(--border);
  margin-top: 40px;
}
"""


def _build_html(
    eval_metrics:    dict,
    radar_imgs:      list[Path],
    heatmap_imgs:    list[Path],
    clutch_df:       pd.DataFrame,
    events_df:       pd.DataFrame,
    eval_dir:        Path,
    timestamp:       str,
) -> str:

    # ── Header stat cards ─────────────────────────────────────
    trk  = eval_metrics.get("tracking", {})
    clut = eval_metrics.get("clutch",   {})
    evts = eval_metrics.get("events",   {})

    def _card(val, lbl):
        return f'<div class="stat-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'

    stat_row = '<div class="stat-row">'
    if "MOTA"  in trk:  stat_row += _card(f"{trk['MOTA']:.3f}",   "MOTA")
    if "IDF1"  in trk:  stat_row += _card(f"{trk['IDF1']:.3f}",   "IDF1")
    if "mean"  in clut: stat_row += _card(f"{clut['mean']:.3f}",   "Mean clutch")
    if "count" in clut: stat_row += _card(clut["count"],            "Goals / events")
    for etype, m in evts.items():
        if etype != "note" and "F1" in m:
            stat_row += _card(f"{m['F1']:.3f}", f"{etype} F1")
    stat_row += "</div>"

    # ── Evaluation figures ────────────────────────────────────
    eval_figs = ""
    for fig_name in ["eval_tracking.png", "eval_events.png", "eval_clutch.png"]:
        p = eval_dir / fig_name
        if p.exists():
            eval_figs += _img_tag(p, fig_name.replace("_", " ").replace(".png",""))

    # ── Radar charts ──────────────────────────────────────────
    radar_html = ""
    if radar_imgs:
        radar_html = '<div class="grid-2">'
        for r in radar_imgs:
            radar_html += _img_tag(r, r.stem)
        radar_html += "</div>"
    else:
        radar_html = '<p class="note">No radar chart images found.</p>'

    # ── Heatmaps ──────────────────────────────────────────────
    heatmap_html = ""
    if heatmap_imgs:
        heatmap_html = '<div class="grid-2">'
        for h in heatmap_imgs:
            heatmap_html += _img_tag(h, h.stem.replace("_", " "))
        heatmap_html += "</div>"
    else:
        heatmap_html = '<p class="note">No heatmap images found.</p>'

    # ── Metric tables ─────────────────────────────────────────
    trk_table   = _dict_to_table(trk,  "Tracking (MOTChallenge)")
    clut_table  = _dict_to_table(clut, "Clutch score statistics")
    form_table  = _dict_to_table(eval_metrics.get("formation", {}), "Formation accuracy")
    event_tables = ""
    for etype, m in evts.items():
        if etype != "note":
            event_tables += _dict_to_table(m, f"Events — {etype}")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>goalX — Thesis Report</title>
<style>{_CSS}</style>
</head>
<body>

<header>
  <h1>goalX <span class="badge">M.Tech Thesis</span></h1>
  <p>End-to-end football analytics: spatio-temporal tracking, event extraction &amp; contextual goal intelligence</p>
  <p style="margin-top:10px;font-size:12px;color:#555">Generated: {timestamp}</p>
</header>

<nav>
  <a href="#summary">Summary</a>
  <a href="#tracking">Tracking</a>
  <a href="#events">Events</a>
  <a href="#clutch">Clutch Score</a>
  <a href="#formation">Formation</a>
  <a href="#radars">Radar Charts</a>
  <a href="#heatmaps">Heatmaps</a>
</nav>

<div class="content">

<section id="summary">
  <h2>Pipeline summary</h2>
  {stat_row}
  <p>This report aggregates the full goalX pipeline output — 14 processing
  stages from raw broadcast video to contextual goal analytics. All metrics
  are computed against manually annotated ground-truth labels. Figures are
  embedded and require no external files.</p>
</section>

<section id="tracking">
  <h2>Tracking evaluation (MOTChallenge metrics)</h2>
  <p>Standard MOT metrics: MOTA penalises false positives, false negatives,
  and identity switches simultaneously. MOTP measures bounding-box
  localisation quality. IDF1 measures identity consistency over time.</p>
  {trk_table}
  {eval_figs}
</section>

<section id="events">
  <h2>Event detection</h2>
  <p>Shots, possession changes, and pressure events evaluated against
  frame-aligned ground-truth labels using temporal-IoU matching
  (window ±30 frames).</p>
  {event_tables}
  <h3>Detected events (sample)</h3>
  {_events_table(events_df)}
</section>

<section id="clutch">
  <h2>Clutch score analysis</h2>
  <p>The Clutch Score is a composite metric:
  <code>Clutch = xG_base × pressure_modifier × pitch_control_weight × temporal_weight</code>.
  Higher scores indicate goals scored under greater contextual difficulty.</p>
  {clut_table}
  <h3>Top 15 highest-scoring events</h3>
  {_clutch_top_table(clutch_df)}
</section>

<section id="formation">
  <h2>Formation detection</h2>
  <p>Rolling majority-vote formation classification compared against
  manually labelled formation windows per team.</p>
  {form_table}
</section>

<section id="radars">
  <h2>Tactical radar charts</h2>
  <p>Normalised player and team performance across: distance covered,
  top speed, possession %, pressure events, pitch control area,
  and clutch score.</p>
  {radar_html}
</section>

<section id="heatmaps">
  <h2>Velocity &amp; activity heatmaps</h2>
  <p>Spatial distribution of player speed and time-on-pitch per team.
  Gaussian-smoothed (σ=14px) for publication quality.</p>
  {heatmap_html}
</section>

</div>

<footer>
  goalX — M.Tech Thesis Research Project &nbsp;·&nbsp;
  Generated {timestamp} &nbsp;·&nbsp;
  Self-contained HTML — no external dependencies
</footer>

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
#  Exporter class
# ─────────────────────────────────────────────────────────────────

class ReportExporter:
    def __init__(self, eval_dir: Path, radar_charts_dir: Path,
                 heatmaps_dir: Path, clutch_csv: Path,
                 events_csv: Path, out_path: Path):
        self.eval_dir        = eval_dir
        self.radar_charts_dir = radar_charts_dir
        self.heatmaps_dir    = heatmaps_dir
        self.clutch_csv      = clutch_csv
        self.events_csv      = events_csv
        self.out_path        = out_path

    def run(self) -> None:
        print(f"\n  goalX — Thesis Report Exporter")
        print(f"  {'─'*42}\n")

        # Load eval metrics JSON
        eval_metrics: dict = {}
        metrics_json = self.eval_dir / "eval_metrics.json"
        if metrics_json.exists():
            with open(metrics_json) as f:
                eval_metrics = json.load(f)
            print("  ✔  Loaded eval_metrics.json")
        else:
            print("  ⚠  eval_metrics.json not found — run evaluate_pipeline.py first.")

        # Collect image assets
        radar_imgs   = _all_pngs(self.radar_charts_dir)
        heatmap_imgs = _all_pngs(self.heatmaps_dir)
        print(f"  ✔  {len(radar_imgs)} radar charts, {len(heatmap_imgs)} heatmaps")

        # Load data tables
        clutch_df = (pd.read_csv(self.clutch_csv)
                     if self.clutch_csv.exists() else pd.DataFrame())
        events_df = (pd.read_csv(self.events_csv)
                     if self.events_csv.exists() else pd.DataFrame())

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        html = _build_html(
            eval_metrics    = eval_metrics,
            radar_imgs      = radar_imgs,
            heatmap_imgs    = heatmap_imgs,
            clutch_df       = clutch_df,
            events_df       = events_df,
            eval_dir        = self.eval_dir,
            timestamp       = timestamp,
        )

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(html, encoding="utf-8")

        size_kb = self.out_path.stat().st_size // 1024
        print(f"\n  ✅  Thesis report exported → {self.out_path}")
        print(f"      Size: {size_kb} KB  (fully self-contained, no internet needed)")
        print(f"      Open with: firefox {self.out_path}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Export a self-contained thesis HTML report for goalX."
    )
    p.add_argument("--eval-dir",         required=True)
    p.add_argument("--radar-charts-dir", required=True)
    p.add_argument("--heatmaps-dir",     required=True)
    p.add_argument("--clutch",           required=True)
    p.add_argument("--events",           required=True)
    p.add_argument("--out",              default="outputs/thesis_report.html")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ReportExporter(
        eval_dir         = Path(args.eval_dir),
        radar_charts_dir = Path(args.radar_charts_dir),
        heatmaps_dir     = Path(args.heatmaps_dir),
        clutch_csv       = Path(args.clutch),
        events_csv       = Path(args.events),
        out_path         = Path(args.out),
    ).run()