"""
Dataset Integration Pipeline
Ingests and merges user-provided real-world datasets:
1. Real disaster imagery from archive.zip (fire, flood, landslide, normal, smoke, earthquake: >9,000 images)
2. Real emergency tweets from nlp-getting-started.zip (>7,600 tweets)
3. Merges with existing curated splits to form high-diversity combined datasets
"""

import os
import re
import csv
import random
from pathlib import Path
import pandas as pd
from PIL import Image

from training.common import set_seed

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
SPLITS_DIR = ROOT_DIR / "data" / "splits"

DISASTER_CLASSES = [
    "cyclone_storm",
    "earthquake",
    "flood",
    "landslide",
    "not_disaster",
    "wildfire",
]
LABEL_MAP = {lbl: i for i, lbl in enumerate(DISASTER_CLASSES)}


def integrate_real_images(max_per_class: int = 500):
    """Integrate real images from archive.zip into image_classifier splits."""
    print("\n[Data Ingestion] Ingesting real disaster imagery...")
    img_raw_base = RAW_DIR / "disaster_images"
    
    category_sources = {
        "wildfire": [
            img_raw_base / "disaster_dataset" / "disaster_dataset" / "fire",
            img_raw_base / "disaster_dataset" / "disaster_dataset" / "smoke",
        ],
        "flood": [
            img_raw_base / "disaster_dataset" / "disaster_dataset" / "flood",
        ],
        "landslide": [
            img_raw_base / "disaster_dataset" / "disaster_dataset" / "landslide",
        ],
        "not_disaster": [
            img_raw_base / "disaster_dataset" / "disaster_dataset" / "normal",
        ],
        "earthquake": [
            img_raw_base / "earthquake" / "earthquake",
        ]
    }

    real_records = []
    idx = 0
    for category, dirs in category_sources.items():
        cat_files = []
        for d in dirs:
            if d.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    cat_files.extend(list(d.glob(ext)))
        
        random.shuffle(cat_files)
        selected = cat_files[:max_per_class]
        print(f"  • {category:<14}: Found {len(cat_files)} real images -> Sampled {len(selected)}")

        for p in selected:
            # Relative path from project root
            rel_path = str(p.relative_to(ROOT_DIR))
            real_records.append({
                "image_id": f"real_img_{idx:05d}",
                "image_path": rel_path,
                "label": category,
                "label_id": LABEL_MAP[category]
            })
            idx += 1

    # Load existing image split
    img_split_dir = SPLITS_DIR / "image_classifier"
    existing_train = pd.read_csv(img_split_dir / "train.csv").to_dict(orient="records")
    existing_val = pd.read_csv(img_split_dir / "val.csv").to_dict(orient="records")
    existing_test = pd.read_csv(img_split_dir / "test.csv").to_dict(orient="records")

    # Split real records 70/15/15
    random.shuffle(real_records)
    n_total = len(real_records)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    combined_train = existing_train + real_records[:n_train]
    combined_val = existing_val + real_records[n_train:n_train + n_val]
    combined_test = existing_test + real_records[n_train + n_val:]

    random.shuffle(combined_train)
    random.shuffle(combined_val)
    random.shuffle(combined_test)

    # Save merged splits
    fieldnames = ["image_id", "image_path", "label", "label_id"]
    for split_name, recs in [("train", combined_train), ("val", combined_val), ("test", combined_test)]:
        out_csv = img_split_dir / f"{split_name}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(recs)

    print(f"✓ Merged Image Dataset: {len(combined_train)} train, {len(combined_val)} val, {len(combined_test)} test")
    return real_records


def _categorize_tweet(text: str, keyword: str, target: int) -> str:
    """Categorize tweet into disaster classes based on text & keywords."""
    if target == 0:
        return "not_disaster"
    
    t_lower = (str(text) + " " + str(keyword)).lower()
    if any(k in t_lower for k in ["flood", "deluge", "inundat", "overflow", "drown", "monsoon"]):
        return "flood"
    if any(k in t_lower for k in ["earthquake", "quake", "tremor", "aftershock", "seismic"]):
        return "earthquake"
    if any(k in t_lower for k in ["fire", "wildfire", "bushfire", "blaze", "burning", "forest fire", "smoke"]):
        return "wildfire"
    if any(k in t_lower for k in ["landslide", "mudslide", "avalanche", "rockslide"]):
        return "landslide"
    if any(k in t_lower for k in ["storm", "cyclone", "hurricane", "typhoon", "tornado", "wind"]):
        return "cyclone_storm"
    
    return "wildfire"  # default disaster fallback


def integrate_real_tweets(max_samples: int = 1500):
    """Integrate real Kaggle disaster tweets into text_classifier splits."""
    print("\n[Data Ingestion] Ingesting real disaster tweets...")
    tweet_csv = RAW_DIR / "disaster_tweets" / "train.csv"
    if not tweet_csv.exists():
        print(f"Warning: {tweet_csv} not found")
        return []

    df = pd.read_csv(tweet_csv)
    real_text_records = []
    
    for idx, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        keyword = str(row.get("keyword", ""))
        target = int(row.get("target", 1))
        
        category = _categorize_tweet(text, keyword, target)
        real_text_records.append({
            "text_id": f"real_tw_{idx:05d}",
            "text": text,
            "label": category,
            "label_id": LABEL_MAP[category]
        })

    random.shuffle(real_text_records)
    selected_tweets = real_text_records[:max_samples]
    print(f"  • Sampled {len(selected_tweets)} real tweets across disaster categories")

    # Load existing text split
    text_split_dir = SPLITS_DIR / "text_classifier"
    existing_train = pd.read_csv(text_split_dir / "train.csv").to_dict(orient="records")
    existing_val = pd.read_csv(text_split_dir / "val.csv").to_dict(orient="records")
    existing_test = pd.read_csv(text_split_dir / "test.csv").to_dict(orient="records")

    n_total = len(selected_tweets)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    combined_train = existing_train + selected_tweets[:n_train]
    combined_val = existing_val + selected_tweets[n_train:n_train + n_val]
    combined_test = existing_test + selected_tweets[n_train + n_val:]

    random.shuffle(combined_train)
    random.shuffle(combined_val)
    random.shuffle(combined_test)

    fields = ["text_id", "text", "label", "label_id"]
    for split_name, recs in [("train", combined_train), ("val", combined_val), ("test", combined_test)]:
        out_csv = text_split_dir / f"{split_name}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(recs)

    print(f"✓ Merged Text Dataset: {len(combined_train)} train, {len(combined_val)} val, {len(combined_test)} test")
    return selected_tweets


def integrate_multimodal_pairs(real_images, real_tweets):
    """Create paired multimodal (real image + real tweet) records."""
    print("\n[Data Ingestion] Pairing real image & text records for Multimodal Dataset...")
    mm_split_dir = SPLITS_DIR / "multimodal"
    
    # Group tweets by category
    tweets_by_cat = {}
    for tw in real_tweets:
        c = tw["label"]
        tweets_by_cat.setdefault(c, []).append(tw["text"])

    real_mm_records = []
    idx = 0
    for img_rec in real_images:
        cat = img_rec["label"]
        candidate_tweets = tweets_by_cat.get(cat, [])
        if candidate_tweets:
            text = random.choice(candidate_tweets)
            real_mm_records.append({
                "incident_id": f"real_mm_{idx:05d}",
                "image_path": img_rec["image_path"],
                "text": text,
                "label": cat,
                "label_id": img_rec["label_id"]
            })
            idx += 1

    # Merge with existing
    existing_train = pd.read_csv(mm_split_dir / "train.csv").to_dict(orient="records")
    existing_val = pd.read_csv(mm_split_dir / "val.csv").to_dict(orient="records")
    existing_test = pd.read_csv(mm_split_dir / "test.csv").to_dict(orient="records")

    random.shuffle(real_mm_records)
    n_total = len(real_mm_records)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    combined_train = existing_train + real_mm_records[:n_train]
    combined_val = existing_val + real_mm_records[n_train:n_train + n_val]
    combined_test = existing_test + real_mm_records[n_train + n_val:]

    random.shuffle(combined_train)
    random.shuffle(combined_val)
    random.shuffle(combined_test)

    fields = ["incident_id", "image_path", "text", "label", "label_id"]
    for split_name, recs in [("train", combined_train), ("val", combined_val), ("test", combined_test)]:
        out_csv = mm_split_dir / f"{split_name}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(recs)

    print(f"✓ Merged Multimodal Dataset: {len(combined_train)} train, {len(combined_val)} val, {len(combined_test)} test")


def run_integration():
    set_seed(42)
    print("=" * 65)
    print("🚀 MERGING USER REAL-WORLD DATASETS INTO PROJECT SPLITS")
    print("=" * 65)
    real_images = integrate_real_images(max_per_class=400)
    real_tweets = integrate_real_tweets(max_samples=1500)
    integrate_multimodal_pairs(real_images, real_tweets)
    print("\n" + "=" * 65)
    print("✅ DATASET INTEGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_integration()
