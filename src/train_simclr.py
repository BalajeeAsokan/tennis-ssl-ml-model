"""
train_simclr.py — Train SimCLR on tennis images (unlabeled!).

Usage:
    python src/train_simclr.py --config configs/simclr_config.yaml

This is the CORE of your project. It runs the self-supervised training loop.
At no point do we use labels — the model learns purely from augmentation consistency.
"""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make src imports work whether run as module or script
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import TennisSSLDataset
from src.model import SimCLRModel, NTXentLoss


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for view1, view2, _ in pbar:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        _, z1 = model(view1)
        _, z2 = model(view2)

        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/simclr_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["output"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")
    if device.type == "cuda":
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")

    # --- Data ---
    dataset = TennisSSLDataset(
        image_dir=cfg["data"]["image_dir"],
        image_size=cfg["data"]["image_size"],
        aug_cfg=cfg["augmentations"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,  # important for contrastive loss
    )

    # --- Model ---
    model = SimCLRModel(
        backbone=cfg["model"]["backbone"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)

    criterion = NTXentLoss(temperature=cfg["training"]["temperature"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # --- Output directories ---
    model_dir = Path(cfg["output"]["model_dir"])
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    for d in (model_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "training_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,loss,lr,time_sec\n")

    # --- Training loop ---
    total_epochs = cfg["training"]["epochs"]
    save_every = cfg["training"]["save_every"]
    best_loss = float("inf")

    print(f"\n[Train] Starting SimCLR training for {total_epochs} epochs")
    print(f"[Train] Batch size: {cfg['data']['batch_size']}, "
          f"Dataset size: {len(dataset)}")

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, loader, criterion, optimizer, device, epoch, total_epochs
        )
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{total_epochs} | "
              f"loss={avg_loss:.4f} | lr={current_lr:.6f} | {elapsed:.1f}s")

        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{current_lr:.6f},{elapsed:.1f}\n")

        # Save intermediate checkpoints
        if epoch % save_every == 0:
            ckpt_path = ckpt_dir / f"simclr_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "loss": avg_loss,
            }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "loss": avg_loss,
            }, model_dir / "simclr_best.pt")

    # Save final model
    final_path = model_dir / "simclr_final.pt"
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "loss": avg_loss,
    }, final_path)
    print(f"\n[Done] Final model saved to {final_path}")
    print(f"[Done] Best model saved to {model_dir / 'simclr_best.pt'}")


if __name__ == "__main__":
    main()
