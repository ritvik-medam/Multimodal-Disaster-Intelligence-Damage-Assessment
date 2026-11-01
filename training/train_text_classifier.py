"""
Disaster Text Classification Training Pipeline
Fine-tunes DistilBERT (or RoBERTa) on disaster text reports.
Saves model checkpoint in Hugging Face format + metadata, label map, and evaluation metrics.
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
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from training.common import (
    set_seed, get_device, compute_classification_metrics,
    save_label_map, save_config,
    save_evaluation_results, save_training_history,
    print_metrics, load_yaml_config
)

ROOT_DIR = Path(__file__).resolve().parent.parent


class DisasterTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_text_classifier(config_path: str, max_epochs_override: int = None):
    cfg_dict = load_yaml_config(config_path)
    set_seed(cfg_dict.get("seed", 42))
    device = get_device(cfg_dict.get("device", "auto"))

    splits_dir = ROOT_DIR / cfg_dict.get("splits_dir", "data/splits/text_classifier")
    checkpoint_dir = ROOT_DIR / cfg_dict.get("checkpoint_dir", "models/text_classifier")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    unique_labels = sorted(train_df["label"].unique().tolist())
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for lbl, i in label_map.items()}
    train_df["label_id"] = train_df["label"].map(label_map)
    val_df["label_id"] = val_df["label"].map(label_map)
    test_df["label_id"] = test_df["label"].map(label_map)

    num_classes = len(unique_labels)
    cfg_dict["num_classes"] = num_classes
    cfg_dict["label_map"] = label_map

    print(f"\n[Text Training] Classes ({num_classes}): {unique_labels}")
    print(f"[Text Training] Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    model_name = cfg_dict.get("model_name", "distilbert-base-uncased")
    max_length = cfg_dict.get("max_length", 128)

    print(f"[Text Training] Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label_map
    )
    model.to(device)

    # Datasets and Loaders
    train_ds = DisasterTextDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_length)
    val_ds = DisasterTextDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, max_length)
    test_ds = DisasterTextDataset(test_df["text"].tolist(), test_df["label_id"].tolist(), tokenizer, max_length)

    batch_size = cfg_dict.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    epochs = max_epochs_override or cfg_dict.get("epochs", 5)
    lr = float(cfg_dict.get("learning_rate", 2e-5))
    weight_decay = float(cfg_dict.get("weight_decay", 0.01))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * cfg_dict.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_f1 = 0.0
    history = []

    print(f"[Text Training] Fine-tuning for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_metrics = compute_classification_metrics(train_targets, train_preds, unique_labels)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item() * len(labels)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_metrics = compute_classification_metrics(val_targets, val_preds, unique_labels)
        val_f1 = val_metrics["f1_macro"]

        history_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1_macro"],
            "val_loss": round(avg_val_loss, 4),
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_f1
        }
        history.append(history_entry)

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1_macro']:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"  ★ Saving best model to {checkpoint_dir} (Val F1: {best_f1:.4f})")
            model.save_pretrained(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))

    # Evaluate on held-out test set
    print("\n[Text Evaluation] Evaluating best model on test set...")
    best_model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    best_model.to(device)
    best_model.eval()

    test_loss = 0.0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            test_loss += outputs.loss.item() * len(labels)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.cpu().numpy())

    test_metrics = compute_classification_metrics(test_targets, test_preds, unique_labels)
    test_metrics["loss"] = round(test_loss / len(test_loader.dataset), 4)

    print_metrics(test_metrics, prefix="[Text Test Set] ")

    # Persist all artifacts
    save_label_map(label_map, checkpoint_dir)
    with open(checkpoint_dir / "training_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    save_training_history(history, checkpoint_dir)
    save_evaluation_results(test_metrics, checkpoint_dir, split="test")

    print(f"✓ Text classifier training complete! Saved to {checkpoint_dir}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Disaster Text Classifier")
    parser.add_argument("--config", type=str, default="configs/text_model.yaml", help="Path to config")
    parser.add_argument("--epochs", type=int, default=4, help="Epoch count override")
    args = parser.parse_args()

    train_text_classifier(args.config, max_epochs_override=args.epochs)
