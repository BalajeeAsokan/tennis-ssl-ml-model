"""
diagnose_ssl_features.py — Check whether SSL embeddings cluster by shot type.

This is the CRITICAL diagnostic: do your SSL features actually capture
shot-type information, or did they learn something else (player, angle, etc)?

We use a simple but powerful test:
  For each image in labeled_data/, compute its SSL embedding.
  Then check: do images of the same shot type have SIMILAR embeddings
  compared to images of different shot types?

If yes -> SSL worked, classifier failure is fixable
If no  -> SSL needs to be retrained with different settings
"""
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_eval_transform
from src.model import SimCLRModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SSL model
    ckpt = torch.load("models/simclr_best.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SimCLRModel(
        backbone=cfg["model"]["backbone"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = get_eval_transform(cfg["data"]["image_size"])

    # Use labeled_data — these have known shot labels from folder names
    labeled_dir = Path("labeled_data")
    if not labeled_dir.exists():
        print(f"[ERROR] {labeled_dir} not found. Run build_labeled_dataset.py first.")
        return

    # Compute embeddings + remember labels
    embeddings = []
    labels = []
    paths = []
    classes = sorted([d.name for d in labeled_dir.iterdir() if d.is_dir()])
    print(f"[Setup] Classes: {classes}")

    for cls_idx, cls in enumerate(classes):
        for img_path in (labeled_dir / cls).glob("*.jpg"):
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode(tensor)
                feat = torch.nn.functional.normalize(feat, dim=1)
            embeddings.append(feat.cpu().numpy()[0])
            labels.append(cls_idx)
            paths.append(img_path)

    X = np.array(embeddings)
    y = np.array(labels)
    print(f"[Data] Embeddings: {X.shape}, Labels: {y.shape}")

    # Test 1: Within-class vs between-class similarity
    print("\n[Test 1] Computing within-class vs between-class similarity...")
    # Cosine similarity matrix (X already normalized, so it's just X @ X.T)
    sim_matrix = X @ X.T
    n = len(y)
    within_sims = []
    between_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])

    within_mean = np.mean(within_sims)
    between_mean = np.mean(between_sims)
    diff = within_mean - between_mean

    print(f"  Within-class average similarity:  {within_mean:.4f}")
    print(f"  Between-class average similarity: {between_mean:.4f}")
    print(f"  Difference (higher is better):    {diff:.4f}")

    if diff > 0.05:
        print("  [GOOD] SSL features somewhat capture shot type")
    elif diff > 0.01:
        print("  [WEAK] SSL captures very little shot-type info")
    else:
        print("  [BAD]  SSL features don't capture shot type at all")

    # Test 2: 1-NN accuracy
    print("\n[Test 2] 1-Nearest-Neighbor accuracy (leave-one-out)...")
    correct = 0
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        nearest = np.argmax(sims)
        if y[nearest] == y[i]:
            correct += 1
    knn_acc = correct / n
    print(f"  1-NN accuracy: {knn_acc:.3f}")
    print(f"  (random would be {1/len(classes):.3f})")

    if knn_acc > 0.6:
        print("  [GOOD] SSL features ARE discriminative for shot type")
        print("         Classifier failure is fixable (more data, longer training)")
    elif knn_acc > 0.35:
        print("  [WEAK] SSL features partially discriminate shot type")
    else:
        print("  [BAD]  SSL features cannot discriminate shot type")
        print("         You need to retrain SSL with better augmentations / more data")

    # Per-class kNN accuracy
    print("\n[Test 3] Per-class 1-NN accuracy:")
    per_class_correct = {c: 0 for c in range(len(classes))}
    per_class_total = {c: 0 for c in range(len(classes))}
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        nearest = np.argmax(sims)
        per_class_total[y[i]] += 1
        if y[nearest] == y[i]:
            per_class_correct[y[i]] += 1
    for cls_idx, cls_name in enumerate(classes):
        acc = per_class_correct[cls_idx] / per_class_total[cls_idx]
        print(f"  {cls_name:10s}: {acc:.3f}")

    # Save histogram of similarities
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(within_sims, bins=40, alpha=0.6, label="Same shot type", color="green")
    ax.hist(between_sims, bins=40, alpha=0.6, label="Different shot types", color="red")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("SSL embedding similarity: same vs different shot types")
    ax.legend()
    out_path = Path("outputs/plots/ssl_diagnosis.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n[Saved] Similarity histogram: {out_path}")
    print("\n[Action] Share these numbers — they tell us exactly what to fix.")


if __name__ == "__main__":
    main()
