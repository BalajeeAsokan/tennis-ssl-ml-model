"""
batch_similarity_test.py — Run nearest-neighbor test on every image in a folder.

Usage:
    python src/batch_similarity_test.py --query_folder test_images --top_k 5

For each image in --query_folder, runs the similarity test and saves a
visualization. Then generates a summary HTML page showing all results.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_eval_transform
from src.model import SimCLRModel
from src.test_similarity import load_model, embed_image, find_neighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_folder", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="models/simclr_best.pt")
    parser.add_argument("--features", type=str, default="data/embeddings/features.npy")
    parser.add_argument("--paths", type=str, default="data/embeddings/paths.npy")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/similarity_tests")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    model, cfg = load_model(args.checkpoint, device)
    transform = get_eval_transform(cfg["data"]["image_size"])

    print("[Load] Loading precomputed embeddings...")
    all_features = np.load(args.features)
    all_paths = np.load(args.paths, allow_pickle=True)

    query_folder = Path(args.query_folder)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # queries = sorted(list(query_folder.glob("*.jpg")) + list(query_folder.glob("*.png")))
    # print(f"[Run] Processing {len(queries)} query images...")

    # Look for many image extensions, both lower and uppercase, recursively
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                  "*.bmp", "*.webp"]
    queries = []
    for ext in extensions:
        queries.extend(query_folder.rglob(ext))
    queries = sorted(set(queries))  # dedupe

    if len(queries) == 0:
        print(f"\n[ERROR] No image files found in '{query_folder}'")
        print(f"[ERROR] Absolute path searched: {query_folder.resolve()}")
        print(f"[ERROR] Folder exists: {query_folder.exists()}")
        if query_folder.exists():
            all_files = list(query_folder.iterdir())
            print(f"[ERROR] Items in folder ({len(all_files)}):")
            for f in all_files[:20]:
                print(f"          {f.name}")
            if not all_files:
                print(f"          (folder is empty)")
        print(f"\n[FIX] Make sure your images are in: {query_folder.resolve()}")
        print(f"[FIX] Supported extensions: {extensions}")
        return

    print(f"[Run] Processing {len(queries)} query images...")

    # Build one big composite figure
    fig, axes = plt.subplots(len(queries), args.top_k + 1,
                              figsize=(2.5 * (args.top_k + 1), 2.5 * len(queries)))
    if len(queries) == 1:
        axes = axes.reshape(1, -1)

    for row, query_path in enumerate(queries):
        print(f"  [{row+1}/{len(queries)}] {query_path.name}")
        query_feat = embed_image(query_path, model, transform, device)
        neighbor_paths, scores = find_neighbors(
            query_feat, all_features, all_paths, top_k=args.top_k
        )

        # Query in first column
        query_img = Image.open(query_path).convert("RGB")
        axes[row, 0].imshow(query_img)
        axes[row, 0].set_title(f"QUERY\n{query_path.stem[:20]}",
                                fontsize=9, color="red", fontweight="bold")
        axes[row, 0].axis("off")

        # Neighbors in following columns
        for i, (p, s) in enumerate(zip(neighbor_paths, scores)):
            img = Image.open(p).convert("RGB")
            axes[row, i + 1].imshow(img)
            axes[row, i + 1].set_title(f"sim={s:.3f}", fontsize=9)
            axes[row, i + 1].axis("off")

    plt.suptitle("Batch Nearest-Neighbor Test Results", fontsize=14, y=1.001)
    plt.tight_layout()
    output_path = out_dir / "batch_results.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[Done] Combined results saved to {output_path}")


if __name__ == "__main__":
    main()
