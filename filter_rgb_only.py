"""
filter_rgb_only.py — Remove non-RGB THETIS frames from the dataset.

THETIS provides 5 data modalities for each shot:
  - RGB (the real video)        ← we want this
  - mask (segmentation)         ← we DON'T want
  - depth (depth map)           ← we DON'T want
  - skelet2D (2D skeleton)      ← we DON'T want
  - skelet3D (3D skeleton)      ← we DON'T want

This script creates a new folder with ONLY the RGB frames.
"""
import shutil
from pathlib import Path

SRC = Path("data/processed_subset")
DST = Path("data/processed_rgb_only")

# These tokens identify non-RGB data
NON_RGB_TOKENS = ["mask", "depth", "skelet2d", "skelet3d", "skelet"]


def is_rgb_frame(filename: str) -> bool:
    name = filename.lower()
    for token in NON_RGB_TOKENS:
        if token in name:
            return False
    return True


def main():
    if not SRC.exists():
        print(f"[ERROR] {SRC} not found")
        return

    DST.mkdir(parents=True, exist_ok=True)

    files = list(SRC.rglob("*.jpg"))
    print(f"[Scan] {len(files)} total files in {SRC}")

    rgb_files = [f for f in files if is_rgb_frame(f.name)]
    non_rgb_files = [f for f in files if not is_rgb_frame(f.name)]

    print(f"[Filter] RGB frames:     {len(rgb_files)}")
    print(f"[Filter] Non-RGB frames: {len(non_rgb_files)} (will be excluded)")

    print(f"\n[Copy] Copying RGB frames to {DST}...")
    for i, f in enumerate(rgb_files):
        shutil.copy(f, DST / f.name)
        if (i + 1) % 1000 == 0:
            print(f"  Copied {i+1}/{len(rgb_files)}")

    print(f"\n[Done] {len(rgb_files)} RGB frames in {DST}")
    print(f"\n[NEXT] You'll need to retrain SimCLR on the clean data:")
    print(f"  1. Edit configs/simclr_config.yaml: change image_dir to 'data/processed_rgb_only'")
    print(f"  2. Re-run training: python src/train_simclr.py")
    print(f"  3. Re-extract embeddings and re-cluster")


if __name__ == "__main__":
    main()
