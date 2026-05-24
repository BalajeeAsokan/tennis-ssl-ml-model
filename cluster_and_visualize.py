"""
cluster_and_visualize.py — The payoff! Cluster the learned embeddings and see
if the model discovered tennis shot types on its own.

Usage:
    python src/cluster_and_visualize.py --n_clusters 4

Produces:
    outputs/plots/umap_clusters.png      — scatter plot, colored by cluster
    outputs/plots/umap_with_thumbs.png   — scatter with image thumbnails
    outputs/plots/cluster_samples/       — sample images from each cluster
"""
import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def reduce_to_2d(features: np.ndarray, method: str = "umap") -> np.ndarray:
    """Project high-dim features to 2D for plotting."""
    if method == "umap":
        try:
            import umap
        except ImportError:
            print("[WARN] umap-learn not installed, falling back to t-SNE")
            method = "tsne"
        else:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            return reducer.fit_transform(features)

    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2, perplexity=30, metric="cosine", random_state=42
        ).fit_transform(features)

    raise ValueError(f"Unknown reduction method: {method}")


def plot_scatter(embedding_2d, labels, out_path, title="Learned shot clusters"):
    plt.figure(figsize=(10, 8))
    n_clusters = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    for k in range(n_clusters):
        mask = labels == k
        plt.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            s=15, alpha=0.7, label=f"Cluster {k}", color=cmap(k),
        )
    plt.legend(loc="best", fontsize=10)
    plt.title(title)
    plt.xlabel("UMAP dim 1")
    plt.ylabel("UMAP dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved {out_path}")


def plot_with_thumbnails(embedding_2d, paths, out_path,
                          n_thumbs: int = 80, thumb_size: int = 40):
    """Show a scatter with a subset of actual tennis images drawn at their positions."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, alpha=0.3, color="gray")

    # Randomly pick images to display
    rng = np.random.default_rng(42)
    idx = rng.choice(len(paths), size=min(n_thumbs, len(paths)), replace=False)
    for i in idx:
        try:
            img = Image.open(paths[i]).convert("RGB")
            img.thumbnail((thumb_size, thumb_size))
            ax.imshow(
                img,
                extent=(
                    embedding_2d[i, 0] - 0.3, embedding_2d[i, 0] + 0.3,
                    embedding_2d[i, 1] - 0.3, embedding_2d[i, 1] + 0.3,
                ),
                aspect="auto", zorder=2,
            )
        except Exception as e:
            continue
    ax.set_title("Learned tennis embeddings (thumbnails)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved {out_path}")


def save_cluster_samples(paths, labels, out_dir: Path, n_samples: int = 16):
    """Copy a few images from each cluster so you can inspect what the model grouped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in np.unique(labels):
        cluster_dir = out_dir / f"cluster_{k:02d}"
        cluster_dir.mkdir(exist_ok=True)
        cluster_paths = paths[labels == k]
        picked = cluster_paths[:n_samples]
        for p in picked:
            shutil.copy(p, cluster_dir / Path(p).name)
    print(f"[Samples] Saved cluster samples under {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/embeddings/features.npy")
    parser.add_argument("--paths", type=str, default="data/embeddings/paths.npy")
    parser.add_argument("--output_dir", type=str, default="outputs/plots")
    parser.add_argument("--n_clusters", type=int, default=4,
                        help="Try 4 (FH/BH/serve/volley), 6 (add smash, etc.)")
    parser.add_argument("--reduction", type=str, default="umap",
                        choices=["umap", "tsne"])
    args = parser.parse_args()

    features = np.load(args.features)
    paths = np.load(args.paths, allow_pickle=True)
    print(f"[Load] Features {features.shape}, {len(paths)} paths")

    # --- Cluster ---
    print(f"[Cluster] Running k-means with k={args.n_clusters}...")
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)
    sil = silhouette_score(features, labels, metric="cosine", sample_size=2000)
    print(f"[Cluster] Silhouette score (cosine): {sil:.3f} "
          f"(higher is better; > 0.2 means reasonably distinct groups)")

    # --- Reduce to 2D ---
    print(f"[Reduce] Projecting to 2D with {args.reduction}...")
    emb2d = reduce_to_2d(features, method=args.reduction)

    # --- Save plots & samples ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_scatter(emb2d, labels, out_dir / "umap_clusters.png")
    plot_with_thumbnails(emb2d, paths, out_dir / "umap_with_thumbs.png")
    save_cluster_samples(paths, labels, out_dir / "cluster_samples")

    # Save cluster assignments for later
    np.save(out_dir / "cluster_labels.npy", labels)
    np.save(out_dir / "embedding_2d.npy", emb2d)
    print("\n[Next step] Open cluster_samples/ and manually inspect each cluster.")
    print("[Next step] Do they correspond to FH / BH / serve / volley?")


if __name__ == "__main__":
    main()
