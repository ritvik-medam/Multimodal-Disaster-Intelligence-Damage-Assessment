"""
Disaster Image Classification Training Pipeline
Fine-tunes a CNN/Vision Backbone (EfficientNet-B2 / ResNet) on disaster imagery.
Saves model checkpoint, label map, configs, training history, and test evaluation metrics.
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
import timm

from training.common import (
    set_seed, get_device, compute_classification_metrics,
    save_checkpoint, save_label_map, save_config,
    save_evaluation_results, save_training_history,
    print_metrics, load_yaml_config
)

ROOT_DIR = Path(__file__).resolve().parent.parent


class DisasterImageDataset(Dataset):
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
            # Fallback if image load fails
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        label_id = int(row["label_id"])
        return image, label_id


def get_transforms(image_size: int = 224):
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in dataloader:
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

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, all_targets, all_preds


def evaluate(model, dataloader, criterion, device, label_names):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds, label_names)
    metrics["loss"] = round(avg_loss, 4)
    return metrics


def train_disaster_classifier(config_path: str, max_epochs_override: int = None):
    cfg_dict = load_yaml_config(config_path)
    set_seed(cfg_dict.get("seed", 42))
    device = get_device(cfg_dict.get("device", "auto"))

    splits_dir = ROOT_DIR / cfg_dict.get("splits_dir", "data/splits/image_classifier")
    checkpoint_dir = ROOT_DIR / cfg_dict.get("checkpoint_dir", "models/disaster_classifier")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    # Map labels
    unique_labels = sorted(train_df["label"].unique().tolist())
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    train_df["label_id"] = train_df["label"].map(label_map)
    val_df["label_id"] = val_df["label"].map(label_map)
    test_df["label_id"] = test_df["label"].map(label_map)

    num_classes = len(unique_labels)
    cfg_dict["num_classes"] = num_classes
    cfg_dict["label_map"] = label_map

    print(f"\n[Training] Classes ({num_classes}): {unique_labels}")
    print(f"[Training] Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Transforms and Dataloaders
    img_size = cfg_dict.get("image_size", 224)
    train_tf, val_tf = get_transforms(img_size)

    train_ds = DisasterImageDataset(train_df, ROOT_DIR, train_tf)
    val_ds = DisasterImageDataset(val_df, ROOT_DIR, val_tf)
    test_ds = DisasterImageDataset(test_df, ROOT_DIR, val_tf)

    batch_size = cfg_dict.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model architecture
    arch = cfg_dict.get("architecture", "efficientnet_b2")
    print(f"[Training] Building model: {arch} (pretrained=True, num_classes={num_classes})")
    model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    model.to(device)

    # Loss & Optimizer
    label_smoothing = cfg_dict.get("label_smoothing", 0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    lr = float(cfg_dict.get("learning_rate", 1e-4))
    weight_decay = float(cfg_dict.get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = max_epochs_override or cfg_dict.get("epochs", 15)
    patience = cfg_dict.get("patience", 5)

    best_f1 = 0.0
    patience_counter = 0
    history = []

    print(f"[Training] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_targets, train_preds = train_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = compute_classification_metrics(train_targets, train_preds, unique_labels)

        val_metrics = evaluate(model, val_loader, criterion, device, unique_labels)
        val_f1 = val_metrics["f1_macro"]

        history_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_f1
        }
        history.append(history_entry)

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1_macro']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
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
            print(f"  ★ New best model saved! (Val F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] Patience reached at epoch {epoch}")
                break

    # Final Test Set Evaluation
    print("\n[Evaluation] Evaluating best checkpoint on held-out test set...")
    best_state = torch.load(str(checkpoint_dir / "best_model.pth"), map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, unique_labels)

    print_metrics(test_metrics, prefix="[Test Set] ")

    # Persist all artifacts
    save_label_map(label_map, checkpoint_dir)
    save_config(cfg_dict, checkpoint_dir)
    save_training_history(history, checkpoint_dir)
    save_evaluation_results(test_metrics, checkpoint_dir, split="test")

    print(f"✓ Training complete! All artifacts saved to {checkpoint_dir}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Disaster Image Classifier")
    parser.add_argument("--config", type=str, default="configs/image_model.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=8, help="Epoch count override")
    args = parser.parse_args()

    train_disaster_classifier(args.config, max_epochs_override=args.epochs)
