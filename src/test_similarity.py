"""
test_similarity.py — Find images in your dataset most similar to a query image.

This is the simplest "test" of your SSL model. Give it any tennis image
(real-world, downloaded from web, screenshot, etc.) and it returns the top-K
most visually similar images from your training set.

Usage:
    python src/test_similarity.py --query path/to/your/test_image.jpg --top_k 5

Example:
    # Test with a YouTube screenshot of Federer's forehand
    python src/test_similarity.py --query test_images/federer_fh.jpg --top_k 8

What this proves:
    If neighbors are visually similar to the query, your model learned
    a meaningful representation of tennis images.
"""
import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_eval_transform
from src.model import SimCLRModel


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained SSL model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SimCLRModel(
        backbone=cfg["model"]["backbone"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def embed_image(image_path: str, model, transform, device):
    """Convert one image to its 512-dim feature vector."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode(tensor)
        feat = torch.nn.functional.normalize(feat, dim=1)
    return feat.cpu().numpy()[0]  # shape (512,)


def find_neighbors(query_feat, all_features, all_paths, top_k=5):
    """Find top-K most similar images using cosine similarity."""
    # Cosine similarity = dot product since both are L2-normalized
    similarities = all_features @ query_feat
    # Get indices of top_k highest similarities
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]
    top_paths = all_paths[top_indices]
    return top_paths, top_scores


def visualize_results(query_path, neighbor_paths, scores, output_path):
    """Show the query image and its nearest neighbors side-by-side."""
    n = len(neighbor_paths) + 1  # +1 for the query
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))

    # Query image (highlighted)
    query_img = Image.open(query_path).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title("QUERY", fontsize=12, color="red", fontweight="bold")
    axes[0].axis("off")
    # Add red border
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("red")
        spine.set_linewidth(3)

    # Neighbors
    for i, (path, score) in enumerate(zip(neighbor_paths, scores)):
        img = Image.open(path).convert("RGB")
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"#{i+1}\nsim={score:.3f}", fontsize=10)
        axes[i + 1].axis("off")

    plt.suptitle("Nearest neighbors in SSL embedding space", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True,
                        help="Path to the query image (any tennis image)")
    parser.add_argument("--checkpoint", type=str, default="models/simclr_best.pt")
    parser.add_argument("--features", type=str, default="data/embeddings/features.npy")
    parser.add_argument("--paths", type=str, default="data/embeddings/paths.npy")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/similarity_tests")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    # Load model
    model, cfg = load_model(args.checkpoint, device)
    transform = get_eval_transform(cfg["data"]["image_size"])

    # Load precomputed embeddings of training set
    print("[Load] Loading precomputed embeddings...")
    all_features = np.load(args.features)
    all_paths = np.load(args.paths, allow_pickle=True)
    print(f"[Load] Database: {len(all_paths)} images")

    # Embed the query
    print(f"[Query] Embedding: {args.query}")
    query_feat = embed_image(args.query, model, transform, device)

    # Find neighbors
    neighbor_paths, scores = find_neighbors(
        query_feat, all_features, all_paths, top_k=args.top_k
    )

    # Print results
    print(f"\n[Results] Top-{args.top_k} most similar images:")
    for i, (path, score) in enumerate(zip(neighbor_paths, scores)):
        print(f"  #{i+1}  sim={score:.3f}  {Path(path).name}")

    # Save visualization
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_name = Path(args.query).stem
    output_path = out_dir / f"neighbors_{query_name}.png"
    visualize_results(args.query, neighbor_paths, scores, output_path)

    # Also copy the neighbors to a folder for inspection
    neighbors_folder = out_dir / f"neighbors_{query_name}"
    neighbors_folder.mkdir(exist_ok=True)
    shutil.copy(args.query, neighbors_folder / f"00_QUERY_{Path(args.query).name}")
    for i, p in enumerate(neighbor_paths):
        shutil.copy(p, neighbors_folder / f"{i+1:02d}_neighbor_{Path(p).name}")
    print(f"[Saved] Neighbor images copied to {neighbors_folder}")


if __name__ == "__main__":
    main()
