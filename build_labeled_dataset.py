"""
build_labeled_dataset.py — Create a labeled subset from THETIS data
for training/testing a downstream classifier.

THETIS organizes videos by shot type in folder names like:
    data/raw/Video_RGB/forehand/p1_forehand_s1.avi
    data/raw/Video_RGB/backhand/p1_backhand_s1.avi
    ...

This script uses those folder labels (which we never used during SSL training!)
to build labeled_data/ with ~30 images per shot class for the classifier.

We also reserve a separate test set for honest evaluation.

Usage:
    python build_labeled_dataset.py
"""
import random
import shutil
from pathlib import Path

# Adjust these paths if needed
PROCESSED_DIR = Path("data/processed_rgb_only")  # Your training pool
RAW_DIR = Path("data/raw")  # Original videos with labeled folders
TRAIN_OUT = Path("labeled_data")
TEST_OUT = Path("test_data_labeled")

# How many images per class
N_TRAIN_PER_CLASS = 30
N_TEST_PER_CLASS = 10

# THETIS shot categories — adjust to whatever folder names your THETIS download has
# Check inside data/raw/Video_RGB/ to see actual folder names
SHOT_CATEGORIES = {
    "forehand": ["forehand", "forehand_flat", "forehand_open_stands", "forehand_slice"],
    "backhand": ["backhand", "backhand2hands", "backhand_slice"],
    "serve": ["service_flat", "service_kick", "service_slice", "serve"],
    "volley": ["forehand_volley", "backhand_volley", "smash", "volley"],
}


def find_class_for_image(img_path: Path) -> str:
    """
    Determine the class of an image by its filename.
    THETIS frame filenames look like: p1_forehand_s1_f00010.jpg
    so we can match keywords from the filename.
    """
    name = img_path.stem.lower()
    for class_label, keywords in SHOT_CATEGORIES.items():
        for kw in keywords:
            if kw.lower() in name:
                return class_label
    return None


def main():
    random.seed(42)

    if not PROCESSED_DIR.exists():
        print(f"[ERROR] {PROCESSED_DIR} not found. Update PROCESSED_DIR in script.")
        return

    # Group images by inferred class
    print("[Scan] Grouping images by class from filenames...")
    by_class = {c: [] for c in SHOT_CATEGORIES}

    all_images = list(PROCESSED_DIR.rglob("*.jpg"))
    print(f"[Scan] Found {len(all_images)} total images")

    for img in all_images:
        cls = find_class_for_image(img)
        if cls:
            by_class[cls].append(img)

    print("\n[Counts] Images per class found:")
    for c, imgs in by_class.items():
        print(f"  {c}: {len(imgs)}")

    # Sample for train and test sets
    for out_dir, n_per_class in [(TRAIN_OUT, N_TRAIN_PER_CLASS),
                                  (TEST_OUT, N_TEST_PER_CLASS)]:
        out_dir.mkdir(exist_ok=True)
        print(f"\n[Build] Creating {out_dir}/ with {n_per_class} per class")
        for cls, imgs in by_class.items():
            cls_dir = out_dir / cls
            cls_dir.mkdir(exist_ok=True)
            if len(imgs) < n_per_class:
                print(f"  [WARN] Only {len(imgs)} images for {cls}, using all")
                sampled = imgs
            else:
                sampled = random.sample(imgs, n_per_class)
            for img in sampled:
                shutil.copy(img, cls_dir / img.name)
            # Remove sampled images so train/test don't overlap
            for s in sampled:
                if s in imgs:
                    imgs.remove(s)
            print(f"  {cls}: copied {len(sampled)} images")

    print(f"\n[Done] Train: {TRAIN_OUT}/  Test: {TEST_OUT}/")
    print(f"[Next] python src/finetune_classifier.py --labeled_dir {TRAIN_OUT} --checkpoint models/simclr_best.pt")


if __name__ == "__main__":
    main()