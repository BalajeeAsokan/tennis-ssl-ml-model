"""
predict_with_overlay.py — Predict shot type for any image(s) and draw the
prediction + confidence directly on the image.

Usage:
    # Single image
    python src/predict_with_overlay.py --image test_images/federer.jpg

    # Whole folder
    python src/predict_with_overlay.py --folder test_images

Output:
    Labeled images saved to outputs/predictions/
    A summary grid saved to outputs/predictions/summary_grid.png

The label drawn on each image shows:
    "FOREHAND" (87.3%)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_eval_transform
from src.model import SimCLRModel


# Colors per class for visual distinction (RGB)
CLASS_COLORS = {
    "forehand": (220, 50, 50),     # red
    "backhand": (50, 130, 220),    # blue
    "serve":    (50, 180, 80),     # green
    "volley":   (220, 150, 30),    # orange
}
DEFAULT_COLOR = (150, 150, 150)


def load_full_model(ssl_ckpt_path, cls_ckpt_path, device):
    """Load the SSL backbone + classifier head."""
    ssl_ckpt = torch.load(ssl_ckpt_path, map_location=device, weights_only=False)
    cls_ckpt = torch.load(cls_ckpt_path, map_location=device, weights_only=False)
    cfg = ssl_ckpt["config"]
    classes = cls_ckpt["classes"]

    model = SimCLRModel(
        backbone=cfg["model"]["backbone"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(ssl_ckpt["model_state_dict"])
    model.eval()

    classifier = nn.Linear(model.feat_dim, len(classes)).to(device)
    classifier.load_state_dict(cls_ckpt["classifier"])
    classifier.eval()

    return model, classifier, classes, cfg


def predict_one(image_path, model, classifier, transform, device, classes):
    """Returns (predicted_class, confidence, all_probabilities_dict)."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode(tensor)
        logits = classifier(feat)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}
    return pred_class, confidence, prob_dict


def draw_label_on_image(image_path, pred_class, confidence, output_path):
    """Open image, draw prediction label, save."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    W, H = img.size
    # Scale fonts/banner based on image size
    banner_h = max(40, H // 10)

    # Try to load a nice font; fall back to default
    font_size = max(18, banner_h // 2)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        small_font = ImageFont.truetype("arial.ttf", max(12, font_size // 2))
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            small_font = ImageFont.truetype("DejaVuSans.ttf", max(12, font_size // 2))
        except (IOError, OSError):
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    color = CLASS_COLORS.get(pred_class.lower(), DEFAULT_COLOR)
    # Semi-transparent banner at the top
    banner_color = color + (200,)  # add alpha
    draw.rectangle([(0, 0), (W, banner_h)], fill=banner_color)

    # Main label text
    label_text = pred_class.upper()
    conf_text = f"{confidence*100:.1f}%"
    full_text = f"{label_text}  ({conf_text})"

    # Center the text
    try:
        bbox = draw.textbbox((0, 0), full_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(full_text, font=font)

    text_x = (W - text_w) // 2
    text_y = (banner_h - text_h) // 2 - 4

    # Drop shadow for readability
    shadow_offset = 2
    draw.text((text_x + shadow_offset, text_y + shadow_offset),
              full_text, fill=(0, 0, 0, 180), font=font)
    draw.text((text_x, text_y), full_text, fill=(255, 255, 255, 255), font=font)

    # Confidence bar at the bottom
    bar_h = 8
    bar_y = H - bar_h - 4
    # Background bar (semi-transparent dark)
    draw.rectangle([(0, bar_y), (W, bar_y + bar_h)], fill=(0, 0, 0, 150))
    # Filled portion proportional to confidence
    fill_w = int(W * confidence)
    draw.rectangle([(0, bar_y), (fill_w, bar_y + bar_h)], fill=color + (255,))

    img.save(output_path)


def make_summary_grid(predictions, output_path):
    """Build a single grid image showing all predictions side-by-side."""
    n = len(predictions)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Handle different axes shapes
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (path, pred, conf, probs) in enumerate(predictions):
        r, c = idx // cols, idx % cols
        img = Image.open(path).convert("RGB")
        axes[r, c].imshow(img)
        color = tuple(v / 255 for v in CLASS_COLORS.get(pred.lower(), DEFAULT_COLOR))
        axes[r, c].set_title(f"{Path(path).name}\n{pred.upper()} ({conf*100:.1f}%)",
                              fontsize=10, color=color, fontweight="bold")
        axes[r, c].axis("off")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Single image to predict")
    parser.add_argument("--folder", type=str, help="Folder of images to predict")
    parser.add_argument("--ssl_checkpoint", type=str, default="models/simclr_best.pt")
    parser.add_argument("--classifier_checkpoint", type=str,
                        default="models/linear_classifier.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    if not args.image and not args.folder:
        print("[ERROR] Provide --image or --folder")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    model, classifier, classes, cfg = load_full_model(
        args.ssl_checkpoint, args.classifier_checkpoint, device
    )
    transform = get_eval_transform(cfg["data"]["image_size"])
    print(f"[Setup] Classes: {classes}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    if args.image:
        image_paths = [Path(args.image)]
    else:
        folder = Path(args.folder)
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                "*.bmp", "*.webp"]
        image_paths = []
        for ext in exts:
            image_paths.extend(folder.rglob(ext))
        image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"[ERROR] No images found in {args.folder or args.image}")
        return

    print(f"[Predict] Processing {len(image_paths)} images...")
    predictions = []

    for img_path in image_paths:
        pred, conf, probs = predict_one(
            img_path, model, classifier, transform, device, classes
        )
        out_path = out_dir / f"labeled_{img_path.stem}.jpg"
        draw_label_on_image(img_path, pred, conf, out_path)
        predictions.append((img_path, pred, conf, probs))

        # Print to console
        prob_str = " | ".join(f"{c}={probs[c]*100:5.1f}%" for c in classes)
        print(f"  {img_path.name:35s} -> {pred.upper():10s} ({conf*100:5.1f}%)  [{prob_str}]")

    # Build a summary grid
    grid_path = out_dir / "summary_grid.png"
    make_summary_grid(predictions, grid_path)
    print(f"\n[Saved] Labeled images: {out_dir}")
    print(f"[Saved] Summary grid:    {grid_path}")


if __name__ == "__main__":
    main()