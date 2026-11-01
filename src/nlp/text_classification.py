"""
Disaster Text Classification Module
Fine-tuned DistilBERT/RoBERTa on CrisisMMD text annotations.
Returns NOT_CONFIGURED when checkpoint missing — never fakes predictions.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logging_config import get_logger
from src.utils import configuration as cfg

logger = get_logger("nlp.text_classification")

_text_model     = None
_text_tokenizer = None
_text_labels    = None
_load_error     = None


def _load_text_classifier():
    """Load fine-tuned text classification model from checkpoint."""
    global _text_model, _text_tokenizer, _text_labels, _load_error

    if _text_model is not None:
        return True, None
    if _load_error is not None:
        return False, _load_error

    checkpoint_path = cfg.TEXT_CLASSIFIER_PATH
    model_dir       = checkpoint_path.parent

    if not model_dir.exists() or not (model_dir / "config.json").exists():
        _load_error = (
            f"Text classifier not found at {model_dir}. "
            "Run: python -m training.train_text_classifier --config configs/text_model.yaml"
        )
        logger.warning(_load_error)
        return False, _load_error

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        device = cfg.get_device()

        _text_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        _text_model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        _text_model.to(device)
        _text_model.eval()

        label_path = model_dir / "label_map.json"
        if label_path.exists():
            with open(label_path) as f:
                _text_labels = json.load(f)
        else:
            _load_error = f"label_map.json not found in {model_dir}"
            return False, _load_error

        logger.info(f"Text classifier loaded from {model_dir}")
        return True, None

    except Exception as e:
        _load_error = str(e)
        logger.error(f"Text classifier load error: {e}", exc_info=True)
        return False, _load_error


def classify_text(report: str) -> Dict[str, Any]:
    """
    Classify a disaster-related text report.

    Returns actual model probabilities when checkpoint is available.
    Returns NOT_CONFIGURED when model is not trained yet.
    """
    result = {
        "model_name": "DisasterTextClassifier",
        "model_type": "disaster_text_classification",
        "status": "not_configured",
        "predicted_category": None,
        "confidence": None,
        "top_predictions": [],
        "error_message": None,
        "configuration_required": (
            "Train the text classifier: "
            "python -m training.train_text_classifier --config configs/text_model.yaml"
        ),
    }

    if not report or not report.strip():
        result["status"] = "error"
        result["error_message"] = "Empty report provided."
        return result

    ok, err = _load_text_classifier()
    if not ok:
        result["error_message"] = err
        return result

    try:
        import torch
        device = cfg.get_device()

        inputs = _text_tokenizer(
            report,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _text_model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        import numpy as np
        id_to_label = {str(v): k for k, v in _text_labels.items()}
        top_k       = min(3, len(probs))
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
        logger.info(f"Text classification: {top_preds[0]['label']} ({top_preds[0]['confidence']:.3f})")

    except Exception as e:
        logger.error(f"Text classification error: {e}", exc_info=True)
        result["status"] = "error"
        result["error_message"] = str(e)

    return result


def get_model_info() -> Dict[str, Any]:
    model_dir    = cfg.TEXT_CLASSIFIER_PATH.parent
    metrics_path = model_dir / "evaluation_results.json"
    metrics = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    labels = []
    label_path = model_dir / "label_map.json"
    if label_path.exists():
        with open(label_path) as f:
            labels = list(json.load(f).keys())
    return {
        "name": "DisasterTextClassifier",
        "type": "Text Classification",
        "dataset": "CrisisMMD Text Annotations",
        "architecture": "DistilBERT / RoBERTa",
        "is_configured": (model_dir / "config.json").exists(),
        "is_evaluated": metrics is not None,
        "supported_labels": labels,
        "evaluation_metrics": metrics,
        "limitations": [
            "Trained on English disaster-related tweets and reports",
            "May not generalize to other languages or formal documents",
            "Task: informative/humanitarian classification — see training docs",
        ],
    }
