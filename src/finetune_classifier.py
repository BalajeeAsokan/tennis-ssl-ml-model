"""
finetune_classifier.py — Optional bonus: train a small linear classifier
on top of the frozen SimCLR backbone. Shows the practical power of SSL.

The workflow:
    1. Manually label a small dataset (e.g. 20 images each of FH/BH/serve/volley)
    2. Place them in labeled_data/<class_name>/*.jpg
    3. Run this script — it freezes the backbone and only trains a linear layer
    4. See how well SSL pretraining transfers with minimal labels

Usage:
    python src/finetune_classifier.py --labeled_dir labeled_data --checkpoint models/simclr_best.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_eval_transform
from src.model import SimCLRModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_dir", type=str, required=True,
                        help="ImageFolder-style: labeled_dir/<class>/*.jpg")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    transform = get_eval_transform(cfg["data"]["image_size"])
    full_dataset = ImageFolder(args.labeled_dir, transform=transform)
    n_classes = len(full_dataset.classes)
    print(f"[Data] Classes: {full_dataset.classes} ({n_classes} total)")
    print(f"[Data] Total labeled: {len(full_dataset)}")

    n_val = max(1, int(len(full_dataset) * args.val_frac))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load the SSL-pretrained backbone, freeze it
    model = SimCLRModel(
        backbone=cfg["model"]["backbone"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    # A simple linear classifier on the 512-d (or 2048-d) backbone features
    classifier = nn.Linear(model.feat_dim, n_classes).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        # train
        classifier.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = model.encode(x)
            logits = classifier(feats)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        # val
        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                feats = model.encode(x)
                logits = classifier(feats)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        tr_acc = train_correct / train_total
        va_acc = val_correct / val_total
        print(f"Epoch {epoch:3d} | train_loss={train_loss/train_total:.4f} "
              f"| train_acc={tr_acc:.3f} | val_acc={va_acc:.3f}")

    torch.save({
        "classifier": classifier.state_dict(),
        "classes": full_dataset.classes,
        "backbone_ckpt": args.checkpoint,
    }, "models/linear_classifier.pt")
    print("\n[Done] Linear classifier saved to models/linear_classifier.pt")


if __name__ == "__main__":
    main()
