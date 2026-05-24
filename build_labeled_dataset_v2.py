"""
build_labeled_dataset.py — Build labeled folders from THETIS filenames.

Correctly handles the actual THETIS naming convention:
  foreflat, foreopen, fslice  -> forehand
  backhand2h, backhand, bslice -> backhand
  serflat, serkick, serslice  -> serve
  fvolley, bvolley, smash     -> volley

ALSO filters out non-RGB frames (mask, depth, skeleton) which contaminate the data.
"""
import random
import shutil
from pathlib import Path

# Use the CLEAN RGB-only folder if it exists, else fall back to subset
SRC_RGB = Path("data/processed_rgb_only")
SRC_FALLBACK = Path("data/processed_subset")
TRAIN_OUT = Path("labeled_data")
TEST_OUT = Path("test_data_labeled")

N_TRAIN_PER_CLASS = 30
N_TEST_PER_CLASS = 10

# Non-RGB tokens we want to EXCLUDE
NON_RGB_TOKENS = ["mask", "depth", "skelet2d", "skelet3d", "skelet"]

# Keywords that map THETIS filename tokens to broad shot classes
# Order matters: check more specific patterns first
SHOT_KEYWORDS = {
    # Check volleys + smash BEFORE forehand/backhand
    # (because "fvolley" contains "f" which could match "fore")
    "volley":   ["fvolley", "bvolley", "smash", "volley"],
    "serve":    ["serflat", "serkick", "serslice", "service", "serve"],
    "forehand": ["foreflat", "foreopen", "fslice", "forehand", "fore"],
    "backhand": ["backhand2h", "bslice", "backhand"],
}


def is_rgb_frame(filename: str) -> bool:
    name = filename.lower()
    for token in NON_RGB_TOKENS:
        if token in name:
            return False
    return True


def find_class(img_name: str) -> str:
    """Return shot class for an image based on its filename keywords."""
    name = img_name.lower()
    # Order matters — check volley/smash before forehand (because
    # 'fvolley' starts with 'f' which could mistakenly match 'fore')
    for cls in ["volley", "serve", "forehand", "backhand"]:
        for kw in SHOT_KEYWORDS[cls]:
            if kw in name:
                return cls
    return None


def main():
    random.seed(42)

    # Pick the source folder
    if SRC_RGB.exists():
        src = SRC_RGB
        print(f"[Source] Using clean RGB folder: {src}")
    else:
        src = SRC_FALLBACK
        print(f"[Source] Using fallback (may contain non-RGB!): {src}")
        print(f"[Source] RECOMMENDED: run filter_rgb_only.py first")

    if not src.exists():
        print(f"[ERROR] {src} not found")
        return

    # Group images by shot class, skipping non-RGB
    by_class = {c: [] for c in SHOT_KEYWORDS}
    skipped_non_rgb = 0
    no_match = 0

    all_images = list(src.rglob("*.jpg"))
    print(f"[Scan] Found {len(all_images)} total images")

    for img in all_images:
        # Safety filter even if non-RGB folder is used
        if not is_rgb_frame(img.name):
            skipped_non_rgb += 1
            continue

        cls = find_class(img.name)
        if cls:
            by_class[cls].append(img)
        else:
            no_match += 1

    print(f"\n[Filter] Non-RGB frames skipped: {skipped_non_rgb}")
    print(f"[Filter] Frames with unknown shot type: {no_match}")

    print("\n[Counts] RGB images per class:")
    for c, imgs in by_class.items():
        print(f"  {c}: {len(imgs)}")

    # Sanity check
    min_count = min(len(v) for v in by_class.values())
    if min_count < N_TRAIN_PER_CLASS + N_TEST_PER_CLASS:
        print(f"\n[WARN] Some classes have fewer than "
              f"{N_TRAIN_PER_CLASS + N_TEST_PER_CLASS} images.")
        print(f"[WARN] Will use whatever is available.")

    # Build train and test sets
    for out_dir, n_per_class in [(TRAIN_OUT, N_TRAIN_PER_CLASS),
                                  (TEST_OUT, N_TEST_PER_CLASS)]:
        out_dir.mkdir(exist_ok=True)
        # Clear existing files to avoid stale data
        for class_dir in out_dir.iterdir():
            if class_dir.is_dir():
                for f in class_dir.glob("*.jpg"):
                    f.unlink()

        print(f"\n[Build] Creating {out_dir}/ with up to {n_per_class} per class")
        for cls, imgs in by_class.items():
            cls_dir = out_dir / cls
            cls_dir.mkdir(exist_ok=True)
            if len(imgs) < n_per_class:
                print(f"  [WARN] Only {len(imgs)} images for '{cls}', using all")
                sampled = imgs
            else:
                sampled = random.sample(imgs, n_per_class)
            for img in sampled:
                shutil.copy(img, cls_dir / img.name)
            # Remove sampled so train/test don't overlap
            for s in sampled:
                if s in imgs:
                    imgs.remove(s)
            print(f"  {cls}: copied {len(sampled)} images")

    print(f"\n[Done] Train labeled: {TRAIN_OUT}/")
    print(f"[Done] Test labeled:  {TEST_OUT}/")
    print(f"\n[Next] python src\\finetune_classifier.py "
          f"--labeled_dir labeled_data --checkpoint models\\simclr_best.pt --epochs 30")


if __name__ == "__main__":
    main()
