"""
test_imagenet_baseline.py — Use raw ImageNet ResNet18 features (no SSL) on
your labeled_data, and see how well they discriminate shot types.

This is your "baseline" — if SSL is worse than this, SSL is hurting performance.
If SSL is better, SSL is adding value.

Run:
    python src/test_imagenet_baseline.py
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_eval_transform


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    # Load ImageNet-pretrained ResNet18, strip classification layer
    net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    net.fc = nn.Identity()
    net = net.to(device).eval()

    transform = get_eval_transform(image_size=224)  # ImageNet was trained at 224

    labeled_dir = Path("labeled_data")
    if not labeled_dir.exists():
        print(f"[ERROR] {labeled_dir} not found")
        return

    classes = sorted([d.name for d in labeled_dir.iterdir() if d.is_dir()])
    print(f"[Classes] {classes}")

    # Extract features for each image
    embeddings, labels = [], []
    for cls_idx, cls in enumerate(classes):
        for img_path in (labeled_dir / cls).glob("*.jpg"):
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = net(x)
                f = torch.nn.functional.normalize(f, dim=1)
            embeddings.append(f.cpu().numpy()[0])
            labels.append(cls_idx)

    X = np.array(embeddings)
    y = np.array(labels)
    print(f"[Data] Embeddings: {X.shape}")

    # 1-NN leave-one-out
    sim = X @ X.T
    correct = 0
    for i in range(len(y)):
        s = sim[i].copy()
        s[i] = -1
        if y[np.argmax(s)] == y[i]:
            correct += 1
    knn_acc = correct / len(y)

    # Within vs between
    within, between = [], []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j]:
                within.append(sim[i, j])
            else:
                between.append(sim[i, j])

    print(f"\n[Baseline: ImageNet ResNet18, NO SSL]")
    print(f"  1-NN accuracy:       {knn_acc:.3f}")
    print(f"  Within-class sim:    {np.mean(within):.4f}")
    print(f"  Between-class sim:   {np.mean(between):.4f}")
    print(f"  Difference:          {np.mean(within) - np.mean(between):.4f}")

    print(f"\n[Compare to your SSL]")
    print(f"  Your SSL 1-NN: 0.417")
    print(f"  ImageNet 1-NN: {knn_acc:.3f}")
    if knn_acc > 0.5:
        print(f"\n[Verdict] ImageNet features alone are MUCH better than your SSL.")
        print(f"          Recommendation: use ImageNet weights + skip or redo SSL.")
    elif knn_acc > 0.417:
        print(f"\n[Verdict] ImageNet features are better than your SSL.")
        print(f"          Use ImageNet pretrained weights as SSL initialization.")
    else:
        print(f"\n[Verdict] Your SSL is doing slightly better than ImageNet alone.")
        print(f"          Combine: ImageNet weights + SSL fine-tuning.")


if __name__ == "__main__":
    main()
