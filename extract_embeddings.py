"""
extract_embeddings.py — Run the trained SimCLR backbone on all images
to get their learned feature vectors. These embeddings are what we
cluster, visualize, and use for downstream tasks.

Usage:
    python src/extract_embeddings.py --checkpoint models/simclr_final.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import TennisEvalDataset
from src.model import SimCLRModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained SimCLR checkpoint (.pt)")
    parser.add_argument("--image_dir", type=str, default="data/processed_subset",
                        help="Directory of images to embed")
    parser.add_argument("--output_dir", type=str, default="data/embeddings")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint ---
    print(f"[Load] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = SimCLRModel(
        backbone=cfg["model"]["backbone"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Data ---
    dataset = TennisEvalDataset(
        image_dir=args.image_dir,
        image_size=cfg["data"]["image_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[Extract] Computing embeddings for {len(dataset)} images...")

    all_features = []
    all_paths = []

    with torch.no_grad():
        for images, paths in tqdm(loader):
            images = images.to(device, non_blocking=True)
            feats = model.encode(images)         # (B, feat_dim)
            feats = torch.nn.functional.normalize(feats, dim=1)
            all_features.append(feats.cpu().numpy())
            all_paths.extend(paths)

    features = np.concatenate(all_features, axis=0)
    paths = np.array(all_paths)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", features)
    np.save(out_dir / "paths.npy", paths)

    print(f"[Done] Features: {features.shape} saved to {out_dir / 'features.npy'}")
    print(f"[Done] Paths: {paths.shape} saved to {out_dir / 'paths.npy'}")


if __name__ == "__main__":
    main()
