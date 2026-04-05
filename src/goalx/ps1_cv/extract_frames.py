"""
extract_frames.py
─────────────────
Converts a raw video file into a numbered JPEG frame sequence.

CHANGES FROM ORIGINAL
─────────────────────
  BUG FIX   : Original had hardcoded VIDEO_PATH / OUTPUT_DIR / FPS at top.
               Running it without editing the file would write to the wrong
               place or crash if the path didn't exist.  Now all three are
               CLI arguments so it works for any video.
  IMPROVED  : Added JPEG quality=95 so detections are sharper on compressed frames.
  IMPROVED  : 1-indexed output (000001.jpg) matching SNMOT dataset convention.
  IMPROVED  : Reports estimated clip duration on completion.
  IMPROVED  : Validates that the video file exists before opening.

Usage
─────
  python3 src/goalx/ps1_cv/extract_frames.py \
      --video  data/raw_videos/match_01.mp4 \
      --out    data/frames/match_01 \
      --fps    25
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str, fps: int = 25) -> int:
    """
    Extract frames from a video at the target fps.

    Returns
    ───────
    Number of frames written to disk.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open: {video_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # How many source frames to skip between each saved frame
    frame_interval = max(1, round(video_fps / fps))

    print(f"\n  goalX — Frame Extractor")
    print(f"  {'─' * 40}")
    print(f"  Source   : {video_path.name}  ({video_fps:.1f} fps,  {total_frames} frames)")
    print(f"  Target   : {fps} fps  (save every {frame_interval} source frames)")
    print(f"  Output   : {output_dir}\n")

    frame_id = 0   # source-frame counter
    saved_id = 1   # 1-indexed output counter (matches SNMOT convention)

    with tqdm(total=total_frames, desc="Extracting", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % frame_interval == 0:
                out_path = output_dir / f"{saved_id:06d}.jpg"
                cv2.imwrite(str(out_path), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_id += 1
            frame_id += 1
            pbar.update(1)

    cap.release()
    n_saved = saved_id - 1
    est_min = n_saved / fps / 60

    print(f"\n  ✅  {n_saved} frames → {output_dir}  (~{est_min:.1f} min clip)")
    print(f"      Next: detect_ball.py --seq {output_dir}\n")
    return n_saved


def _parse_args():
    p = argparse.ArgumentParser(
        description="Extract JPEG frames from a video for the goalX pipeline."
    )
    p.add_argument("--video", required=True,
                   help="Input video file (.mp4, .mov, .avi)")
    p.add_argument("--out",   required=True,
                   help="Output directory for numbered .jpg frames")
    p.add_argument("--fps",   type=int, default=25,
                   help="Frames per second to extract (default: 25)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_frames(args.video, args.out, args.fps)