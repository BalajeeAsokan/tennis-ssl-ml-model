"""
dataset.py — PyTorch Dataset for tennis images with SimCLR two-view augmentation.

The core idea of SimCLR: take each image, apply augmentations TWICE (independently)
to get two "views". The model is trained so these two views produce similar embeddings,
while being different from all other images' embeddings.

The `lightly` library handles this elegantly.
"""
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Standard ImageNet stats — used because ResNet was trained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_simclr_transform(image_size: int = 224, cfg: dict = None) -> transforms.Compose:
    """
    Build the SimCLR augmentation pipeline.

    Each augmentation teaches the model to IGNORE some variation:
      - RandomResizedCrop => ignore zoom/position
      - HorizontalFlip    => ignore left/right symmetry (fine for tennis)
      - ColorJitter       => ignore jersey color, lighting
      - Grayscale         => force reliance on shape/pose, not color
      - GaussianBlur      => ignore high-frequency details (fine texture)

    After SSL, the model preserves what SURVIVES augmentation: body pose,
    stance, racquet position — exactly what defines a tennis shot.
    """
    cfg = cfg or {}
    crop_scale = tuple(cfg.get("random_resized_crop", [0.2, 1.0]))
    flip_p = cfg.get("horizontal_flip", 0.5)
    jitter_s = cfg.get("color_jitter_strength", 0.5)
    jitter_p = cfg.get("color_jitter_prob", 0.8)
    gray_p = cfg.get("grayscale_prob", 0.2)
    blur_p = cfg.get("gaussian_blur_prob", 0.5)

    color_jitter = transforms.ColorJitter(
        brightness=0.8 * jitter_s,
        contrast=0.8 * jitter_s,
        saturation=0.8 * jitter_s,
        hue=0.2 * jitter_s,
    )

    kernel_size = max(3, int(0.1 * image_size) | 1)  # odd number

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=flip_p),
        transforms.RandomApply([color_jitter], p=jitter_p),
        transforms.RandomGrayscale(p=gray_p),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))],
            p=blur_p,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Clean transform used for embedding extraction / evaluation (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TennisSSLDataset(Dataset):
    """
    Dataset that returns TWO augmented views of each image for SimCLR.

    __getitem__ returns (view1, view2, filename) — we don't return a label
    because this is unsupervised. Labels are never seen during SSL training.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, image_dir: str, image_size: int = 224, aug_cfg: dict = None):
        self.image_dir = Path(image_dir)
        self.paths: List[Path] = sorted([
            p for p in self.image_dir.rglob("*")
            if p.suffix.lower() in self.IMG_EXTS
        ])
        if len(self.paths) == 0:
            raise RuntimeError(
                f"No images found under {self.image_dir}. "
                f"Did you run data_prep.py?"
            )
        self.transform = get_simclr_transform(image_size, aug_cfg)
        print(f"[Dataset] Found {len(self.paths)} images under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img = Image.open(self.paths[idx]).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, str(self.paths[idx])


class TennisEvalDataset(Dataset):
    """Single-view dataset for extracting embeddings after SSL training."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, image_dir: str, image_size: int = 224):
        self.image_dir = Path(image_dir)
        self.paths = sorted([
            p for p in self.image_dir.rglob("*")
            if p.suffix.lower() in self.IMG_EXTS
        ])
        self.transform = get_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])
