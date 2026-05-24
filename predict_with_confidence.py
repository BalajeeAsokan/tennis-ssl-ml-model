"""
predict_with_confidence.py — Final predictor for the project.

Given the modest model accuracy, this version:
  - Predicts shot type with confidence
  - Flags low-confidence predictions as "Uncertain"
  - Shows all class probabilities, not just the top one
  - Draws clear visualization on each image
  - Provides honest accuracy metrics on test set

Usage:
    python src/predict_with_confidence.py --folder test_images
    python src/predict_with_confidence.py --folder test_data_labeled --eval
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_eval_transform
from src.model import SimCLRModel


# Confidence threshold — below this, output "Uncertain"
CONFIDENCE_THRESHOLD = 0.45

CLASS_COLORS = {
    "forehand": (220, 50, 50),
    "backhand": (50, 130, 220),
    "serve":    (50, 180, 80),
    "volley":   (220, 150, 30),
    "uncertain":(120, 120, 120),
}


def load_models(ssl_ckpt_path, cls_ckpt_path, device):
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


def predict(image_path, model, classifier, transform, device, classes):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode(x)
        logits = classifier(feat)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    if confidence < CONFIDENCE_THRESHOLD:
        return "uncertain", confidence, {c: float(p) for c, p in zip(classes, probs)}, pred_class
    return pred_class, confidence, {c: float(p) for c, p in zip(classes, probs)}, pred_class


def draw_label(image_path, displayed_class, confidence, all_probs,
                model_top_pick, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    banner_h = max(50, H // 8)
    font_size = max(20, banner_h // 2 - 4)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        small_font = ImageFont.truetype("arial.ttf", font_size // 2)
    except (IOError, OSError):
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    color = CLASS_COLORS.get(displayed_class.lower(), (120, 120, 120))
    banner_color = color + (210,)
    draw.rectangle([(0, 0), (W, banner_h)], fill=banner_color)

    if displayed_class == "uncertain":
        # Show: "UNCERTAIN (best guess: forehand 43%)"
        main_text = "UNCERTAIN"
        sub_text = f"best guess: {model_top_pick} ({confidence*100:.0f}%)"
    else:
        main_text = displayed_class.upper()
        sub_text = f"{confidence*100:.1f}% confidence"

    try:
        bbox = draw.textbbox((0, 0), main_text, font=font)
        main_w = bbox[2] - bbox[0]
    except AttributeError:
        main_w, _ = draw.textsize(main_text, font=font)

    main_x = (W - main_w) // 2
    main_y = 5
    draw.text((main_x + 2, main_y + 2), main_text, fill=(0, 0, 0, 200), font=font)
    draw.text((main_x, main_y), main_text, fill=(255, 255, 255), font=font)

    try:
        bbox = draw.textbbox((0, 0), sub_text, font=small_font)
        sub_w = bbox[2] - bbox[0]
    except AttributeError:
        sub_w, _ = draw.textsize(sub_text, font=small_font)
    sub_x = (W - sub_w) // 2
    sub_y = main_y + font_size + 2
    draw.text((sub_x, sub_y), sub_text, fill=(255, 255, 255), font=small_font)

    # Probability bars at bottom
    bar_area_h = max(60, H // 6)
    bar_y_start = H - bar_area_h
    draw.rectangle([(0, bar_y_start), (W, H)], fill=(0, 0, 0, 160))

    n_bars = len(all_probs)
    bar_h = (bar_area_h - 10) // n_bars
    margin = 5
    label_w = max(80, W // 5)

    for i, (cname, p) in enumerate(all_probs.items()):
        by = bar_y_start + margin + i * bar_h
        # Class label
        draw.text((margin, by), f"{cname[:8]}:", fill=(255, 255, 255), font=small_font)
        # Bar background
        bar_start_x = label_w
        bar_end_x = W - 60
        draw.rectangle([(bar_start_x, by + 2), (bar_end_x, by + bar_h - 4)],
                       fill=(60, 60, 60, 200))
        # Bar fill
        fill_w = int((bar_end_x - bar_start_x) * p)
        bcolor = CLASS_COLORS.get(cname.lower(), (150, 150, 150))
        draw.rectangle([(bar_start_x, by + 2), (bar_start_x + fill_w, by + bar_h - 4)],
                       fill=bcolor + (255,))
        # Percentage
        draw.text((bar_end_x + 5, by), f"{p*100:.0f}%",
                  fill=(255, 255, 255), font=small_font)

    img.save(output_path)


def make_grid(predictions, output_path):
    n = len(predictions)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (path, displayed, conf, _, top_pick) in enumerate(predictions):
        r, c = idx // cols, idx % cols
        img = Image.open(path).convert("RGB")
        axes[r, c].imshow(img)
        color = tuple(v / 255 for v in CLASS_COLORS.get(displayed.lower(), (120, 120, 120)))
        if displayed == "uncertain":
            title = f"{Path(path).name}\nUNCERTAIN (best: {top_pick} {conf*100:.0f}%)"
        else:
            title = f"{Path(path).name}\n{displayed.upper()} ({conf*100:.0f}%)"
        axes[r, c].set_title(title, fontsize=10, color=color, fontweight="bold")
        axes[r, c].axis("off")

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    global CONFIDENCE_THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--ssl_checkpoint", type=str, default="models/simclr_best.pt")
    parser.add_argument("--classifier_checkpoint", type=str,
                        default="models/linear_classifier.pt")
    parser.add_argument("--eval", action="store_true",
                        help="If folder has class subdirs, compute accuracy")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions_v2")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classifier, classes, cfg = load_models(
        args.ssl_checkpoint, args.classifier_checkpoint, device)
    transform = get_eval_transform(cfg["data"]["image_size"])

    # global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.threshold

    print(f"[Setup] Classes: {classes}")
    print(f"[Setup] Confidence threshold: {CONFIDENCE_THRESHOLD}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_paths = [Path(args.image)]
    elif args.folder:
        folder = Path(args.folder)
        if args.eval:
            # Class subdirs
            image_paths = []
            for cls_dir in folder.iterdir():
                if cls_dir.is_dir():
                    image_paths.extend(cls_dir.glob("*.jpg"))
        else:
            exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
            image_paths = []
            for e in exts:
                image_paths.extend(folder.rglob(e))
        image_paths = sorted(set(image_paths))
    else:
        print("[ERROR] Need --image or --folder")
        return

    if not image_paths:
        print(f"[ERROR] No images found")
        return

    print(f"[Run] {len(image_paths)} images\n")
    predictions = []
    eval_stats = {"total": 0, "correct": 0, "uncertain": 0, "high_conf_total": 0,
                  "high_conf_correct": 0}

    for img_path in image_paths:
        displayed, conf, all_probs, top_pick = predict(
            img_path, model, classifier, transform, device, classes)
        out = out_dir / f"labeled_{img_path.stem}.jpg"
        draw_label(img_path, displayed, conf, all_probs, top_pick, out)
        predictions.append((img_path, displayed, conf, all_probs, top_pick))

        prob_str = " ".join(f"{c[:3]}={all_probs[c]*100:4.0f}%" for c in classes)
        if displayed == "uncertain":
            print(f"  {img_path.name:35s} -> UNCERTAIN  ({prob_str})  best: {top_pick}")
        else:
            print(f"  {img_path.name:35s} -> {displayed.upper():10s} ({prob_str})")

        if args.eval:
            true_class = img_path.parent.name
            eval_stats["total"] += 1
            if displayed == "uncertain":
                eval_stats["uncertain"] += 1
            else:
                eval_stats["high_conf_total"] += 1
                if displayed == true_class:
                    eval_stats["high_conf_correct"] += 1
                    eval_stats["correct"] += 1

    grid_path = out_dir / "summary_grid.png"
    make_grid(predictions, grid_path)
    print(f"\n[Saved] Labeled images: {out_dir}")
    print(f"[Saved] Summary grid: {grid_path}")

    if args.eval:
        print(f"\n[Evaluation]")
        print(f"  Total images:           {eval_stats['total']}")
        print(f"  Marked uncertain:       {eval_stats['uncertain']} "
              f"({eval_stats['uncertain']/eval_stats['total']*100:.0f}%)")
        if eval_stats['high_conf_total'] > 0:
            high_conf_acc = eval_stats['high_conf_correct'] / eval_stats['high_conf_total']
            print(f"  Confident predictions:  {eval_stats['high_conf_total']}")
            print(f"  Correct when confident: {eval_stats['high_conf_correct']} "
                  f"({high_conf_acc*100:.1f}%)")
        overall = eval_stats['correct'] / eval_stats['total']
        print(f"  Overall accuracy:       {overall*100:.1f}%")


if __name__ == "__main__":
    main()
