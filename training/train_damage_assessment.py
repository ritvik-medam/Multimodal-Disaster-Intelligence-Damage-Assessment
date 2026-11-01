"""
Building Damage Assessment Model Training Pipeline
Trains a satellite imagery damage assessment backbone (ResNet-50)
to categorize buildings into:
- no_damage
- minor_damage
- major_damage
- destroyed
Includes weighted cross-entropy loss for class imbalance handling and full test evaluation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np
import timm

from training.common import (
    set_seed, get_device, compute_classification_metrics,
    save_checkpoint, save_label_map, save_config,
    save_evaluation_results, save_training_history,
    print_metrics, load_yaml_config
)

ROOT_DIR = Path(__file__).resolve().parent.parent


class SatelliteDamageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]
        img_full_path = self.root_dir / img_rel_path

        try:
            image = Image.open(img_full_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        label_id = int(row["label_id"])
        return image, label_id


def get_damage_transforms(image_size: int = 224):
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=90),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_damage_assessment(config_path: str, max_epochs_override: int = None):
    cfg_dict = load_yaml_config(config_path)
    set_seed(cfg_dict.get("seed", 42))
    device = get_device(cfg_dict.get("device", "auto"))

    splits_dir = ROOT_DIR / cfg_dict.get("splits_dir", "data/splits/damage_assessment")
    checkpoint_dir = ROOT_DIR / cfg_dict.get("checkpoint_dir", "models/damage_assessment")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    damage_order = ["no_damage", "minor_damage", "major_damage", "destroyed"]
    label_map = {lbl: i for i, lbl in enumerate(damage_order)}
    train_df["label_id"] = train_df["label"].map(label_map)
    val_df["label_id"] = val_df["label"].map(label_map)
    test_df["label_id"] = test_df["label"].map(label_map)

    num_classes = len(damage_order)
    cfg_dict["num_classes"] = num_classes
    cfg_dict["label_map"] = label_map

    print(f"\n[Damage Assessment Training] Classes ({num_classes}): {damage_order}")
    print(f"[Damage Assessment Training] Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Class weights for imbalance
    class_counts = train_df["label_id"].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(float))
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"[Damage Assessment Training] Class weights: {class_weights}")

    img_size = cfg_dict.get("image_size", 224)
    train_tf, val_tf = get_damage_transforms(img_size)

    train_ds = SatelliteDamageDataset(train_df, ROOT_DIR, train_tf)
    val_ds = SatelliteDamageDataset(val_df, ROOT_DIR, val_tf)
    test_ds = SatelliteDamageDataset(test_df, ROOT_DIR, val_tf)

    batch_size = cfg_dict.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    arch = cfg_dict.get("architecture", "resnet50")
    print(f"[Damage Assessment Training] Building model: {arch} (pretrained=True, num_classes={num_classes})")
    model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
    lr = float(cfg_dict.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    epochs = max_epochs_override or cfg_dict.get("epochs", 10)
    best_f1 = 0.0
    history = []

    print(f"[Damage Assessment Training] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        train_metrics = compute_classification_metrics(all_targets, all_preds, damage_order)

        # Validation
        model.eval()
        v_loss = 0.0
        v_preds, v_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                v_preds.extend(preds)
                v_targets.extend(targets.cpu().numpy())

        avg_val_loss = v_loss / len(val_loader.dataset)
        val_metrics = compute_classification_metrics(v_targets, v_preds, damage_order)
        val_f1 = val_metrics["f1_macro"]

        history_entry = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1_macro"],
            "val_loss": round(avg_val_loss, 4),
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_f1
        }
        history.append(history_entry)

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {avg_loss:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1_macro']:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=cfg_dict,
                checkpoint_dir=checkpoint_dir,
                label_map=label_map,
                filename="best_model.pth"
            )
            print(f"  ★ New best damage model saved! (Val F1: {best_f1:.4f})")

    # Evaluate on held-out test set
    print("\n[Damage Evaluation] Evaluating best model on test set...")
    best_state = torch.load(str(checkpoint_dir / "best_model.pth"), map_location=device)
    model.load_state_dict(best_state["model_state_dict"])

    model.eval()
    t_loss = 0.0
    t_preds, t_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            t_loss += criterion(outputs, targets).item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            t_preds.extend(preds)
            t_targets.extend(targets.cpu().numpy())

    test_metrics = compute_classification_metrics(t_targets, t_preds, damage_order)
    test_metrics["loss"] = round(t_loss / len(test_loader.dataset), 4)

    print_metrics(test_metrics, prefix="[Damage Test Set] ")

    # Persist all artifacts
    save_label_map(label_map, checkpoint_dir)
    save_config(cfg_dict, checkpoint_dir)
    save_training_history(history, checkpoint_dir)
    save_evaluation_results(test_metrics, checkpoint_dir, split="test")

    print(f"✓ Damage assessment training complete! Saved to {checkpoint_dir}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Satellite Damage Assessment Model")
    parser.add_argument("--config", type=str, default="configs/damage_model.yaml", help="Path to config")
    parser.add_argument("--epochs", type=int, default=6, help="Epoch count override")
    args = parser.parse_args()

    train_damage_assessment(args.config, max_epochs_override=args.epochs)
