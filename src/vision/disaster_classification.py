"""
Disaster Image Classification Module
Loads a fine-tuned EfficientNet/ResNet model trained on MEDIC/CrisisMMD.
If checkpoint is unavailable, reports NOT_CONFIGURED — never fakes predictions.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image
from src.utils.logging_config import get_logger
from src.utils import configuration as cfg

logger = get_logger("vision.disaster_classification")

_classifier_model   = None
_classifier_labels  = None
_classifier_transform = None
_load_error         = None


def _load_classifier():
    """Load the disaster image classifier from checkpoint."""
    global _classifier_model, _classifier_labels, _classifier_transform, _load_error

    if _classifier_model is not None:
        return True, None
    if _load_error is not None:
        return False, _load_error

    checkpoint_path = cfg.DISASTER_CLASSIFIER_PATH
    label_path      = checkpoint_path.parent / "label_map.json"

    if not checkpoint_path.exists():
        _load_error = (
            f"Checkpoint not found: {checkpoint_path}. "
            "Run: python -m training.train_disaster_classifier --config configs/image_model.yaml"
        )
        logger.warning(_load_error)
        return False, _load_error

    try:
        import torch
        import torchvision.transforms as T

        # Load label map
        if label_path.exists():
            with open(label_path) as f:
                _classifier_labels = json.load(f)
        else:
            _load_error = f"Label map not found: {label_path}"
            return False, _load_error

        # Load model config
        config_path = checkpoint_path.parent / "config.json"
        architecture = "efficientnet_b2"
        num_classes  = len(_classifier_labels)
        if config_path.exists():
            with open(config_path) as f:
                model_cfg = json.load(f)
            architecture = model_cfg.get("architecture", architecture)
            num_classes  = model_cfg.get("num_classes", num_classes)

        device = cfg.get_device()

        # Build model architecture
        import timm
        _classifier_model = timm.create_model(
            architecture, pretrained=False, num_classes=num_classes
        )
        state = torch.load(str(checkpoint_path), map_location=device)
        if "model_state_dict" in state:
            _classifier_model.load_state_dict(state["model_state_dict"])
        else:
            _classifier_model.load_state_dict(state)
        _classifier_model.to(device)
        _classifier_model.eval()

        # Preprocessing (must match training)
        _classifier_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"Disaster classifier loaded: {architecture}, {num_classes} classes")
        return True, None

    except Exception as e:
        _load_error = str(e)
        logger.error(f"Classifier load error: {e}", exc_info=True)
        return False, _load_error


def classify_disaster(image: Image.Image) -> Dict[str, Any]:
    """
    Run disaster image classification on a PIL image.

    Returns prediction only when model is loaded and evaluated.
    Returns NOT_CONFIGURED status when checkpoint is missing.
    """
    result = {
        "model_name": "DisasterNet",
        "model_type": "disaster_image_classification",
        "status": "not_configured",
        "predicted_category": None,
        "confidence": None,
        "top_predictions": [],
        "error_message": None,
        "configuration_required": (
            "Train the disaster classifier first: "
            "python -m training.train_disaster_classifier --config configs/image_model.yaml"
        ),
    }

    ok, err = _load_classifier()
    if not ok:
        result["error_message"] = err
        return result

    try:
        import torch
        device = cfg.get_device()
        img    = image.convert("RGB")
        tensor = _classifier_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = _classifier_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # id_to_label mapping
        id_to_label = {str(v): k for k, v in _classifier_labels.items()}

        top_k = min(3, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_preds   = [
            {"label": id_to_label.get(str(i), f"class_{i}"),
             "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        result.update({
            "status": "success",
            "predicted_category": top_preds[0]["label"],
            "confidence": top_preds[0]["confidence"],
            "top_predictions": top_preds,
            "error_message": None,
            "configuration_required": None,
        })
        logger.info(
            f"Disaster classification: {top_preds[0]['label']} "
            f"({top_preds[0]['confidence']:.3f})"
        )

    except Exception as e:
        logger.error(f"Classification inference error: {e}", exc_info=True)
        result["status"] = "error"
        result["error_message"] = str(e)

    return result


def get_model_info() -> Dict[str, Any]:
    checkpoint_path = cfg.DISASTER_CLASSIFIER_PATH
    label_path      = checkpoint_path.parent / "label_map.json"
    metrics_path    = checkpoint_path.parent / "evaluation_results.json"

    labels = []
    if label_path.exists():
        with open(label_path) as f:
            labels = list(json.load(f).keys())

    metrics = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {
        "name": "DisasterNet (Disaster Image Classifier)",
        "type": "Image Classification",
        "dataset": "MEDIC + CrisisMMD",
        "architecture": "EfficientNet-B2",
        "is_configured": checkpoint_path.exists(),
        "is_evaluated": metrics is not None,
        "supported_labels": labels,
        "evaluation_metrics": metrics,
        "limitations": [
            "Trained on MEDIC / CrisisMMD images — may not generalize to all image types",
            "Does not assess structural damage",
            "Performance on non-English context images is unverified",
        ],
    }
