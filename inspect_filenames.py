"""
inspect_filenames.py — Diagnose what THETIS filenames look like in your data.

Run this to figure out why build_labeled_dataset.py isn't finding files.
"""
import re
from collections import Counter
from pathlib import Path

PROCESSED = Path("data/processed_subset")

if not PROCESSED.exists():
    print(f"[ERROR] {PROCESSED} not found")
    raise SystemExit(1)

files = list(PROCESSED.rglob("*.jpg"))
print(f"[Total] {len(files)} images in {PROCESSED}")
print(f"\n[Sample] First 20 filenames:")
for f in files[:20]:
    print(f"  {f.name}")

# Try to find any recognizable shot keyword in filenames (case-insensitive)
keywords_to_check = [
    "forehand", "backhand", "serve", "service", "volley", "smash",
    "fh", "bh", "fore", "back", "serv", "vol", "drop", "lob", "slice",
    "Forehand", "Backhand", "Service", "Volley", "Smash",
    "FOREHAND", "BACKHAND", "SERVICE", "VOLLEY", "SMASH",
]

print(f"\n[Search] Looking for shot-type keywords (case-insensitive)...")
counts = {}
for kw in keywords_to_check:
    matching = [f for f in files if kw.lower() in f.name.lower()]
    if matching:
        counts[kw] = len(matching)

if not counts:
    print("  [WARN] No common shot keywords found in filenames!")
    print("  Your filenames don't seem to encode the shot type.")
else:
    # Deduplicate by lowercasing
    seen = set()
    unique_counts = {}
    for kw, n in counts.items():
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique_counts[kw.lower()] = n
    for kw, n in sorted(unique_counts.items(), key=lambda x: -x[1]):
        print(f"  '{kw}': {n} files match")

# Show distribution by parent folder (in case shot type is in folder, not filename)
print(f"\n[Folder structure] Files grouped by parent folder:")
parent_counts = Counter(f.parent.name for f in files)
for parent, n in parent_counts.most_common(20):
    print(f"  {parent}: {n} files")

# Show what the filename "tokens" look like (split by _ and .)
print(f"\n[Tokens] Most common tokens in filenames (excluding numbers):")
all_tokens = []
for f in files:
    tokens = re.split(r'[_.]', f.stem)
    for t in tokens:
        if t and not t.isdigit() and not re.match(r'^f\d+$', t.lower()) and not re.match(r'^s\d+$', t.lower()) and not re.match(r'^p\d+$', t.lower()):
            all_tokens.append(t.lower())

token_counts = Counter(all_tokens)
for token, n in token_counts.most_common(30):
    print(f"  '{token}': {n}")
