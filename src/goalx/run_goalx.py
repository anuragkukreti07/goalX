"""
run_goalx.py
────────────
Single-command orchestrator for the complete goalX pipeline.

Runs all 10 stages sequentially. Each stage is defined as a dataclass
with its CLI command, inputs, and outputs. The orchestrator:
  • Validates that all inputs exist before launching each stage.
  • Times each stage and the total run.
  • Skips completed stages if --resume is passed (checks output existence).
  • Writes a machine-readable run log to outputs/run_log.json.
  • Exits early on any stage failure with a clear error message.

Usage
─────
  # Full pipeline (fresh run):
  python3 src/goalx/run_goalx.py --config config/goalx_config.yaml

  # Resume from where it last succeeded:
  python3 src/goalx/run_goalx.py --config config/goalx_config.yaml --resume

  # Dry-run: print commands without executing:
  python3 src/goalx/run_goalx.py --config config/goalx_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # pip install pyyaml


# ─────────────────────────────────────────────────────────────────
#  Console helpers
# ─────────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"

def _hdr(msg: str) -> None:
    print(f"\n{_BOLD}{_CYAN}{'─'*60}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {msg}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─'*60}{_RESET}")

def _ok(msg: str) -> None:  print(f"  {_GREEN}✔{_RESET}  {msg}")
def _warn(msg: str) -> None: print(f"  {_YELLOW}⚠{_RESET}  {msg}")
def _err(msg: str) -> None:  print(f"  {_RED}✖{_RESET}  {msg}")
def _info(msg: str) -> None: print(f"  {_DIM}{msg}{_RESET}")

def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ─────────────────────────────────────────────────────────────────
#  Stage definition
# ─────────────────────────────────────────────────────────────────

@dataclass
class Stage:
    """
    One pipeline stage.

    name        : human-readable label
    script      : path to the Python script (relative to project root)
    args        : list of CLI tokens, built by the orchestrator from config
    inputs      : paths that must exist before this stage runs
    outputs     : paths that must exist after this stage succeeds
    skip_check  : if True, presence of outputs[0] means "already done"
    """
    name:       str
    script:     str
    args:       list[str]         = field(default_factory=list)
    inputs:     list[str]         = field(default_factory=list)
    outputs:    list[str]         = field(default_factory=list)
    skip_check: bool              = True


# ─────────────────────────────────────────────────────────────────
#  Pipeline builder
# ─────────────────────────────────────────────────────────────────

def _build_pipeline(cfg: dict[str, Any]) -> list[Stage]:
    """
    Translate the YAML config dict into an ordered list of Stage objects.
    This is the single source of truth for the pipeline topology.
    """
    d  = cfg["data"]
    o  = cfg["outputs"]
    m  = cfg["model"]
    src = "src/goalx"

    return [
        # ── PS1: Computer Vision ──────────────────────────────────
        Stage(
            name   = "PS1-1  detect players & ball",
            script = f"{src}/ps1_cv/detect_players_full.py",
            args   = [
                "--seq",   d["frames_dir"],
                "--out",   o["detections_csv"],
                "--model", m["yolo_weights"],
                "--imgsz", str(m["imgsz"]),
                "--conf",  str(m["conf"]),
                "--batch", str(m["batch_size"]),
            ],
            inputs  = [d["frames_dir"]],
            outputs = [o["detections_csv"]],
        ),
        Stage(
            name   = "PS1-2  ByteTrack player IDs",
            script = f"{src}/ps1_cv/track_players.py",
            args   = [
                "--detections",   o["detections_csv"],
                "--sample-img",   d["sample_frame"],
                "--out",          o["tracking_csv"],
                "--track-thresh", str(m["track_thresh"]),
                "--match-thresh", str(m["match_thresh"]),
                "--track-buffer", str(m["track_buffer"]),
            ],
            inputs  = [o["detections_csv"]],
            outputs = [o["tracking_csv"]],
        ),
        Stage(
            name   = "PS1-3  project tracks → pitch coords",
            script = f"{src}/ps1_cv/project_tracks.py",
            args   = [
                "--tracks",      o["tracking_csv"],
                "--homography",  d["homography_npz"],
                "--pitch",       d["pitch_map"],
                "--out-dir",     o["projected_dir"],
            ],
            inputs  = [o["tracking_csv"], d["homography_npz"], d["pitch_map"]],
            outputs = [f"{o['projected_dir']}/projected_tracks.csv"],
        ),
        Stage(
            name   = "PS1-4  smooth trajectories",
            script = f"{src}/ps1_cv/smooth_tracks.py",
            args   = [
                "--tracks",   f"{o['projected_dir']}/projected_tracks.csv",
                "--out",      o["smoothed_csv"],
                "--window",   str(cfg["smoothing"]["window"]),
                "--clamp",    str(cfg["smoothing"]["teleport_clamp_px"]),
            ],
            inputs  = [f"{o['projected_dir']}/projected_tracks.csv"],
            outputs = [o["smoothed_csv"]],
        ),
        Stage(
            name   = "PS1-5  extract semantic events",
            script = f"{src}/ps1_cv/extract_events.py",
            args   = [
                "--tracks",  o["smoothed_csv"],
                "--out",     o["events_csv"],
            ],
            inputs  = [o["smoothed_csv"]],
            outputs = [o["events_csv"]],
        ),
        Stage(
            name   = "PS1-6  2D tactical radar video",
            script = f"{src}/ps1_cv/showcase_radar.py",
            args   = [
                "--tracks", o["smoothed_csv"],
                "--events", o["events_csv"],
                "--pitch",  d["pitch_map"],
                "--out",    o["radar_mp4"],
            ],
            inputs  = [o["smoothed_csv"], o["events_csv"]],
            outputs = [o["radar_mp4"]],
        ),

        # ── PS2: Tactical Intelligence ────────────────────────────
        Stage(
            name   = "PS2-1  team classification",
            script = f"{src}/ps2_ml/team_classifier.py",
            args   = [
                "--tracks",     o["tracking_csv"],
                "--frames-dir", d["frames_dir"],
                "--out",        o["team_csv"],
            ],
            inputs  = [o["tracking_csv"]],
            outputs = [o["team_csv"]],
        ),
        Stage(
            name   = "PS2-2  formation detection",
            script = f"{src}/ps2_ml/formation_detector.py",
            args   = [
                "--tracks", o["smoothed_csv"],
                "--teams",  o["team_csv"],
                "--out",    o["formation_csv"],
            ],
            inputs  = [o["smoothed_csv"], o["team_csv"]],
            outputs = [o["formation_csv"]],
        ),
        Stage(
            name   = "PS2-3  pitch control (Voronoi)",
            script = f"{src}/ps2_ml/pitch_control.py",
            args   = [
                "--tracks", o["smoothed_csv"],
                "--teams",  o["team_csv"],
                "--pitch",  d["pitch_map"],
                "--out-dir", o["pitch_control_dir"],
            ],
            inputs  = [o["smoothed_csv"], o["team_csv"]],
            outputs = [f"{o['pitch_control_dir']}/pitch_control.csv"],
        ),
        Stage(
            name   = "PS2-4  clutch score",
            script = f"{src}/ps2_ml/clutch_score.py",
            args   = [
                "--events",        o["events_csv"],
                "--tracks",        o["smoothed_csv"],
                "--pitch-control", f"{o['pitch_control_dir']}/pitch_control.csv",
                "--out",           o["clutch_csv"],
            ],
            inputs  = [o["events_csv"], o["smoothed_csv"],
                       f"{o['pitch_control_dir']}/pitch_control.csv"],
            outputs = [o["clutch_csv"]],
        ),
        Stage(
            name   = "PS2-5  tactical radar charts",
            script = f"{src}/ps2_ml/tactical_radar.py",
            args   = [
                "--tracks",  o["smoothed_csv"],
                "--clutch",  o["clutch_csv"],
                "--teams",   o["team_csv"],
                "--out-dir", o["radar_charts_dir"],
            ],
            inputs  = [o["smoothed_csv"], o["clutch_csv"], o["team_csv"]],
            outputs = [o["radar_charts_dir"]],
        ),
        Stage(
            name   = "PS2-6  broadcast overlay (god mode)",
            script = f"{src}/ps2_ml/broadcast_overlay.py",
            args   = [
                "--frames-dir",    d["frames_dir"],
                "--tracks",        o["smoothed_csv"],
                "--teams",         o["team_csv"],
                "--events",        o["events_csv"],
                "--pitch-control", f"{o['pitch_control_dir']}/pitch_control.csv",
                "--homography",    d["homography_npz"],
                "--out",           o["overlay_mp4"],
            ],
            inputs  = [o["smoothed_csv"], o["team_csv"], d["homography_npz"]],
            outputs = [o["overlay_mp4"]],
        ),
        Stage(
            name   = "SYS-1  velocity heatmaps",
            script = f"{src}/velocity_heatmap.py",
            args   = [
                "--tracks",    o["smoothed_csv"],
                "--teams",     o["team_csv"],
                "--pitch",     d["pitch_map"],
                "--out-dir",   o["heatmaps_dir"],
            ],
            inputs  = [o["smoothed_csv"], o["team_csv"]],
            outputs = [o["heatmaps_dir"]],
        ),
        Stage(
            name   = "SYS-2  thesis evaluation report",
            script = f"{src}/evaluate_pipeline.py",
            args   = [
                "--tracks",     o["smoothed_csv"],
                "--events",     o["events_csv"],
                "--formation",  o["formation_csv"],
                "--clutch",     o["clutch_csv"],
                "--gt-dir",     d.get("ground_truth_dir", "data/ground_truth"),
                "--out-dir",    o["eval_dir"],
            ],
            inputs  = [o["smoothed_csv"]],
            outputs = [o["eval_dir"]],
            skip_check = False,  # always re-evaluate
        ),
        # Stage(
        #     name   = "SYS-3  export thesis report",
        #     script = f"{src}/ps3_ml/export_report.py",
        #     args   = [
        #         "--eval-dir",         o["eval_dir"],
        #         "--radar-charts-dir", o["radar_charts_dir"],
        #         "--heatmaps-dir",     o["heatmaps_dir"],
        #         "--clutch",           o["clutch_csv"],
        #         "--events",           o["events_csv"],
        #         "--out",              o["thesis_report_html"],
        #     ],
        #     inputs  = [o["eval_dir"]],
        #     outputs = [o["thesis_report_html"]],
        #     skip_check = False,
        # ),
        Stage(
            name   = "SYS-3  export thesis report",
            script = f"{src}/ps3_ml/export_report.py",
            args   = [
                "--radar-dir",    o["radar_charts_dir"],  # Updated to --radar-dir
                "--heatmaps-dir", o["heatmaps_dir"],
                "--clutch",       o["clutch_csv"],
                "--events",       o["events_csv"],
                "--out",          o["thesis_report_html"],
            ],
            inputs  = [o["clutch_csv"]],  # Removed eval_dir dependency here
            outputs = [o["thesis_report_html"]],
            skip_check = False,
        ),
    ]


# ─────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    def __init__(self, cfg: dict, resume: bool, dry_run: bool,
                 only: list[str] | None):
        self.cfg     = cfg
        self.resume  = resume
        self.dry_run = dry_run
        self.only    = only
        self.log: list[dict] = []

    def _inputs_ok(self, stage: Stage) -> bool:
        missing = [p for p in stage.inputs if not Path(p).exists()]
        if missing:
            for m in missing:
                _err(f"Missing input: {m}")
        return len(missing) == 0

    def _already_done(self, stage: Stage) -> bool:
        if not stage.skip_check or not self.resume:
            return False
        return all(Path(p).exists() for p in stage.outputs)

    def _run_stage(self, stage: Stage) -> tuple[bool, float]:
        cmd = [sys.executable, stage.script] + stage.args
        _info("CMD: " + " ".join(cmd))

        if self.dry_run:
            return True, 0.0

        t0  = time.perf_counter()
        ret = subprocess.run(cmd)
        dt  = time.perf_counter() - t0
        return ret.returncode == 0, dt

    def run(self) -> int:
        stages   = _build_pipeline(self.cfg)
        t_total  = time.perf_counter()
        n_ok     = 0
        n_skip   = 0
        n_fail   = 0

        _hdr(f"goalX pipeline  —  {len(stages)} stages")
        if self.dry_run:
            _warn("DRY-RUN mode: commands will be printed but not executed.")
        if self.resume:
            _warn("RESUME mode: stages with existing outputs will be skipped.")

        for i, stage in enumerate(stages, 1):
            # Filter if --only was passed
            if self.only and not any(k in stage.name for k in self.only):
                continue

            print(f"\n  [{i:02d}/{len(stages)}] {_BOLD}{stage.name}{_RESET}")

            if self._already_done(stage):
                _ok("Skipped (outputs already exist).")
                n_skip += 1
                self.log.append({"stage": stage.name, "status": "skipped", "time_s": 0})
                continue

            if not self._inputs_ok(stage):
                _err("Stage aborted — missing inputs.")
                n_fail += 1
                self.log.append({"stage": stage.name, "status": "failed_inputs", "time_s": 0})
                break

            ok, dt = self._run_stage(stage)

            if ok:
                _ok(f"Done in {_fmt_time(dt)}.")
                n_ok += 1
                self.log.append({"stage": stage.name, "status": "ok", "time_s": round(dt, 2)})
            else:
                _err(f"Stage FAILED after {_fmt_time(dt)}. Aborting pipeline.")
                n_fail += 1
                self.log.append({"stage": stage.name, "status": "failed", "time_s": round(dt, 2)})
                break

        elapsed = time.perf_counter() - t_total
        _hdr(f"Pipeline finished in {_fmt_time(elapsed)}")
        print(f"  {_GREEN}Succeeded{_RESET} : {n_ok}")
        print(f"  {_YELLOW}Skipped  {_RESET} : {n_skip}")
        print(f"  {_RED}Failed   {_RESET} : {n_fail}")

        # Write run log
        log_path = Path(self.cfg["outputs"].get("run_log", "outputs/run_log.json"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({
                "stages":       self.log,
                "total_time_s": round(elapsed, 2),
                "n_ok":         n_ok,
                "n_skip":       n_skip,
                "n_fail":       n_fail,
            }, f, indent=2)
        _info(f"Run log → {log_path}")

        return 1 if n_fail else 0


# ─────────────────────────────────────────────────────────────────
#  Default config template (written if no config file exists)
# ─────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = """\
# goalx_config.yaml — edit paths before running

data:
  frames_dir:       data/raw_videos/tracking/test/SNMOT-116/img1
  sample_frame:     data/raw_videos/tracking/test/SNMOT-116/img1/000001.jpg
  pitch_map:        data/pitch_map.png
  homography_npz:   data/homography_data.npz
  ground_truth_dir: data/ground_truth     # optional, for evaluation

model:
  yolo_weights: yolov8s.pt
  imgsz:        1280
  conf:         0.25
  batch_size:   8
  track_thresh: 0.20
  match_thresh: 0.90
  track_buffer: 90

smoothing:
  window:            7
  teleport_clamp_px: 50

outputs:
  detections_csv:     data/detections_raw.csv
  tracking_csv:       data/tracking.csv
  projected_dir:      outputs/projected
  smoothed_csv:       outputs/smoothed_tracks.csv
  events_csv:         outputs/events.csv
  radar_mp4:          outputs/tactical_radar.mp4
  team_csv:           outputs/team_labels.csv
  formation_csv:      outputs/formations.csv
  pitch_control_dir:  outputs/pitch_control
  clutch_csv:         outputs/clutch_scores.csv
  radar_charts_dir:   outputs/radar_charts
  overlay_mp4:        outputs/broadcast_overlay.mp4
  heatmaps_dir:       outputs/heatmaps
  eval_dir:           outputs/evaluation
  thesis_report_html: outputs/thesis_report.html
  run_log:            outputs/run_log.json
"""


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="goalX master pipeline orchestrator")
    p.add_argument("--config",   default="config/goalx_config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--resume",   action="store_true",
                   help="Skip stages whose outputs already exist")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print commands without executing them")
    p.add_argument("--only",     nargs="+", metavar="KEYWORD",
                   help="Run only stages whose names contain these keywords")
    p.add_argument("--init-config", action="store_true",
                   help="Write a default config file and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.init_config:
        cfg_path = Path(args.config)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(_DEFAULT_CONFIG)
        print(f"  Config written → {cfg_path}")
        print("  Edit the paths, then run:  python3 src/goalx/run_goalx.py --config config/goalx_config.yaml")
        sys.exit(0)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"  Config not found: {cfg_path}")
        print("  Run with --init-config to generate a template.")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    orch = PipelineOrchestrator(
        cfg     = cfg,
        resume  = args.resume,
        dry_run = args.dry_run,
        only    = args.only,
    )
    sys.exit(orch.run())