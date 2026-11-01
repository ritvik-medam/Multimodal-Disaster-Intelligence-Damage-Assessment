"""
Building Damage Assessment Module
Loads a model trained on xBD/xView2 satellite imagery.
Supports: no_damage, minor_damage, major_damage, destroyed.
IMPORTANT: Only valid for imagery type matching the training data (satellite/aerial).
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
from src.utils.logging_config import get_logger
from src.utils import configuration as cfg

logger = get_logger("vision.damage_assessment")

_damage_model    = None
_damage_labels   = None
_damage_transform = None
_load_error      = None

SUPPORTED_DAMAGE_LEVELS = ["no_damage", "minor_damage", "major_damage", "destroyed"]


def _load_damage_model():
    """Load damage assessment model from checkpoint."""
    global _damage_model, _damage_labels, _damage_transform, _load_error

    if _damage_model is not None:
        return True, None
    if _load_error is not None:
        return False, _load_error

    checkpoint_path = cfg.DAMAGE_ASSESSMENT_PATH
    if not checkpoint_path.exists():
        _load_error = (
            f"Checkpoint not found: {checkpoint_path}. "
            "Run: python -m training.train_damage_assessment --config configs/damage_model.yaml"
        )
        logger.warning(_load_error)
        return False, _load_error

    try:
        import torch
        import torchvision.transforms as T

        label_path = checkpoint_path.parent / "label_map.json"
        if label_path.exists():
            with open(label_path) as f:
                _damage_labels = json.load(f)
        else:
            _damage_labels = {k: i for i, k in enumerate(SUPPORTED_DAMAGE_LEVELS)}

        config_path = checkpoint_path.parent / "config.json"
        architecture = "resnet50"
        num_classes  = len(_damage_labels)
        if config_path.exists():
            with open(config_path) as f:
                model_cfg = json.load(f)
            architecture = model_cfg.get("architecture", architecture)
            num_classes  = model_cfg.get("num_classes", num_classes)

        device = cfg.get_device()

        import timm
        _damage_model = timm.create_model(
            architecture, pretrained=False, num_classes=num_classes
        )
        state = torch.load(str(checkpoint_path), map_location=device)
        if "model_state_dict" in state:
            _damage_model.load_state_dict(state["model_state_dict"])
        else:
            _damage_model.load_state_dict(state)
        _damage_model.to(device)
        _damage_model.eval()

        _damage_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"Damage assessment model loaded: {architecture}")
        return True, None

    except Exception as e:
        _load_error = str(e)
        logger.error(f"Damage model load error: {e}", exc_info=True)
        return False, _load_error


def assess_damage(image: Image.Image) -> Dict[str, Any]:
    """
    Assess building damage from an image.

    CRITICAL: This model was trained on xBD satellite/aerial imagery.
    Predictions on ground-level photos may be unreliable.
    Always display the imagery-type warning.
    """
    result = {
        "model_name": "xBD Damage Assessor",
        "model_type": "building_damage_assessment",
        "status": "not_configured",
        "predicted_damage_level": None,
        "confidence": None,
        "top_predictions": [],
        "imagery_type_warning": (
            "⚠️ This model was trained on satellite/aerial imagery (xBD dataset). "
            "Predictions on ground-level smartphone photos may be unreliable. "
            "Use only with appropriate imagery types."
        ),
        "error_message": None,
        "configuration_required": (
            "Train the damage assessment model: "
            "python -m training.train_damage_assessment --config configs/damage_model.yaml"
        ),
    }

    ok, err = _load_damage_model()
    if not ok:
        result["error_message"] = err
        return result

    try:
        import torch
        device = cfg.get_device()
        img    = image.convert("RGB")
        tensor = _damage_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = _damage_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        id_to_label = {str(v): k for k, v in _damage_labels.items()}
        top_k       = min(4, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_preds   = [
            {"label": id_to_label.get(str(i), f"class_{i}"),
             "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        result.update({
            "status": "success",
            "predicted_damage_level": top_preds[0]["label"],
            "confidence": top_preds[0]["confidence"],
            "top_predictions": top_preds,
            "error_message": None,
            "configuration_required": None,
        })
        logger.info(
            f"Damage assessment: {top_preds[0]['label']} "
            f"({top_preds[0]['confidence']:.3f})"
        )

    except Exception as e:
        logger.error(f"Damage assessment error: {e}", exc_info=True)
        result["status"] = "error"
        result["error_message"] = str(e)

    return result


def get_model_info() -> Dict[str, Any]:
    checkpoint_path = cfg.DAMAGE_ASSESSMENT_PATH
    metrics_path    = checkpoint_path.parent / "evaluation_results.json"
    metrics = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    return {
        "name": "xBD Building Damage Assessor",
        "type": "Building Damage Classification",
        "dataset": "xBD / xView2",
        "architecture": "ResNet-50",
        "imagery_type": "Satellite / Aerial",
        "is_configured": checkpoint_path.exists(),
        "is_evaluated": metrics is not None,
        "supported_labels": SUPPORTED_DAMAGE_LEVELS,
        "evaluation_metrics": metrics,
        "limitations": [
            "Trained only on xBD satellite imagery",
            "Ground-level images are out-of-distribution",
            "Requires pre/post disaster image pairs in some configurations",
            "Not suitable for general disaster image classification",
        ],
    }
