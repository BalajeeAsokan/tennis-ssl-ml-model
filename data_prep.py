"""
data_prep.py — Extract frames from tennis videos for SSL training.

Usage:
    python src/data_prep.py --input data/raw --output data/processed --fps 5

This walks data/raw for .avi/.mp4 files and saves frames at `fps` frames/sec
to data/processed. For SimCLR we don't need labels, just lots of images.

If you downloaded THETIS, your data/raw will look like:
    data/raw/Video_RGB/forehand/p1_forehand_s1.avi
    data/raw/Video_RGB/backhand/p1_backhand_s1.avi
    ...

We flatten everything into one big unlabeled pool in data/processed/.

166512 samples cannot be trained on Nvidia T600
New processed data comprisiing 10000-15000 images are available in data/processed_subset

"""
import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, target_fps: int = 5):
    """Extract frames at target_fps from a video, saving as JPGs."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}")
        return 0

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(original_fps / target_fps)))

    stem = video_path.stem
    frame_idx = 0
    saved = 0
    success, frame = cap.read()
    while success:
        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"{stem}_f{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1
        success, frame = cap.read()
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw",
                        help="Folder containing videos (searched recursively)")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output folder for extracted frames")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second to extract")
    parser.add_argument("--extensions", type=str, default=".avi,.mp4,.mov,.mkv")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip().lower() for e in args.extensions.split(","))
    videos = [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]

    if not videos:
        print(f"No videos found in {in_dir}. If you have images already, "
              f"just copy them to {out_dir} directly.")
        return

    print(f"Found {len(videos)} videos. Extracting at {args.fps} fps...")
    total = 0
    for vid in tqdm(videos):
        total += extract_frames(vid, out_dir, args.fps)
    print(f"Done. Saved {total} frames to {out_dir}")


if __name__ == "__main__":
    main()
