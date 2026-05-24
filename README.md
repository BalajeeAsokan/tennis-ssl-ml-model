# Tennis SSL — Self-Supervised Learning on Tennis Images

A beginner-friendly project that learns tennis shot representations **without any labels** using SimCLR (a contrastive self-supervised learning method), then uses those representations for:

1. **Unsupervised shot clustering** — let the model discover forehand/backhand/serve/volley on its own
2. **Low-label fine-tuning** — train a classifier with only a handful of labels per class
3. **Nearest-neighbor shot retrieval** — "find me shots that look like this one"

## Setup

```bash
# 1. Create environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU works
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Workflow

### Step 1 — Get data
- Download THETIS dataset: https://github.com/THETIS-dataset/dataset
- OR scrape images yourself (see `src/data_prep.py`)
- Put videos/images in `data/raw/`

### Step 2 - Extracting RGQ images alone to create a new data set instead of using processed image data set which contained 15000 images. And those 15000 images were mostly raw images as well. Took the script filter_rgb_only.py to filter out 3590 RGB images & out them inside a folder data/processed_rgb_only
```bash
python src\filter_rgb_only.py
Next modify the simclr_config.yaml file as image_dir: "data/processed_rgb_only"
```

### Step 3 — Train SimCLR (no labels needed!)
```bash
python src/train_simclr.py --config configs/simclr_config.yaml
```

### Step 4 - Extract embeddings on best trained simclr model for processed rgb images
```bash
python src\extract_embeddings.py --checkpoint models\simclr_best.pt --image_dir data\processed_rgb_only
```

### Step 5 — Cluster and visualize
```bash
python src/cluster_and_visualize.py
```

Open `outputs/plots/umap_clusters.png` to see your model's discovered shot groupings.

### Step 6 - Build labeled dataset
```bash
python src\build_labeled_dataset.py
```

### Step 7 -  Train classifier
```bash
python src\finetune_classifier.py --labeled_dir labeled_data --checkpoint models\simclr_best.pt --epochs 30
```

### Step 8 - Testing on real world images
```bash
python src\predict_with_overlay.py --folder test_images_google
```

### Step9 - Testing the SSL Diagnostic Feature
```bash
python src\diagnose_ssl_features.py
```

### Step10 - On THETIS test set (sanity check with eval)
```bash
python src\predict_with_confidence.py --folder test_data_labeled --eval
```
