"""
Common utilities shared across all training scripts.
Device selection, metric computation, checkpoint saving, seed setting.
"""
import os, json, random, csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


def set_seed(seed: int = 42):
    """Set reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(device_setting: str = "auto") -> "torch.device":
    import torch
    if device_setting == "auto":
        if torch.cuda.is_available():
            d = torch.device("cuda")
        elif torch.backends.mps.is_available():
            d = torch.device("mps")
        else:
            d = torch.device("cpu")
    else:
        d = torch.device(device_setting)
    print(f"[Device] Using: {d}")
    return d


def compute_classification_metrics(
    y_true: List[int], y_pred: List[int], labels: List[str]
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1 (macro), per-class metrics, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report, confusion_matrix
    )
    metrics = {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "precision_macro":   round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":      round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":          round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted":       round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix":  confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=labels,
                                                        zero_division=0, output_dict=True),
        "num_samples": len(y_true),
        "num_classes": len(labels),
        "labels": labels,
    }
    return metrics


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: Dict,
    config: Dict,
    checkpoint_dir: Path,
    label_map: Dict,
    filename: str = "best_model.pth",
):
    """Save model checkpoint with full metadata."""
    import torch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "label_map": label_map,
        "saved_at": datetime.utcnow().isoformat(),
    }
    path = checkpoint_dir / filename
    torch.save(checkpoint, str(path))
    print(f"[Checkpoint] Saved to {path}")
    return path


def save_label_map(label_map: Dict, checkpoint_dir: Path):
    with open(checkpoint_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)


def save_config(config: Dict, checkpoint_dir: Path):
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_evaluation_results(metrics: Dict, checkpoint_dir: Path, split: str = "test"):
    path = checkpoint_dir / "evaluation_results.json"
    results = {}
    if path.exists():
        with open(path) as f:
            results = json.load(f)
    results[split] = metrics
    results["evaluated_at"] = datetime.utcnow().isoformat()
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Evaluation] Results saved to {path}")


def save_training_history(history: List[Dict], checkpoint_dir: Path):
    path = checkpoint_dir / "training_history.json"
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def print_metrics(metrics: Dict, prefix: str = ""):
    print(f"\n{prefix}{'='*50}")
    print(f"{prefix}Accuracy:  {metrics.get('accuracy',0):.4f}")
    print(f"{prefix}F1 Macro:  {metrics.get('f1_macro',0):.4f}")
    print(f"{prefix}Precision: {metrics.get('precision_macro',0):.4f}")
    print(f"{prefix}Recall:    {metrics.get('recall_macro',0):.4f}")
    print(f"{prefix}{'='*50}\n")


def load_yaml_config(config_path: str) -> Dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)
