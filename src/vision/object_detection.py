"""
YOLO Object Detection Module
Uses Ultralytics YOLOv8 pretrained on COCO dataset.
Detects: person, car, truck, bus, motorcycle, bicycle, boat, etc.
NOTE: Standard YOLO does NOT classify disaster types or damage levels.
      That is handled by separate purpose-trained models.
"""
import io
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
from src.utils.logging_config import get_logger
from src.utils import configuration as cfg

logger = get_logger("vision.object_detection")

# COCO objects relevant to disaster scenarios
DISASTER_RELEVANT_OBJECTS = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "boat", "airplane", "helicopter", "fire truck", "ambulance",
    "backpack", "suitcase", "umbrella",
}

_yolo_model = None


def _load_yolo():
    """Load YOLO model once and cache it."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model, None

    try:
        from ultralytics import YOLO
        model_path = cfg.YOLO_MODEL_PATH
        if model_path.exists():
            logger.info(f"Loading YOLO from {model_path}")
            _yolo_model = YOLO(str(model_path))
        else:
            # Download YOLOv8n (smallest, ~6MB) automatically
            logger.info("Downloading YOLOv8n (first run only)...")
            _yolo_model = YOLO("yolov8n.pt")
            # Save to models dir for next time
            cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return _yolo_model, None
    except ImportError:
        return None, "Ultralytics YOLO is not installed. Run: pip install ultralytics"
    except Exception as e:
        logger.error(f"YOLO load error: {e}")
        return None, str(e)


def run_object_detection(
    image: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Run YOLOv8 object detection on a PIL image.

    Returns structured result dict with detections, counts, annotated image.
    Does NOT claim to detect disaster types — only COCO objects.
    """
    result = {
        "model_name": "YOLOv8n",
        "model_version": "8.x",
        "model_type": "object_detection",
        "dataset": "COCO",
        "detected_objects": [],
        "object_counts": {},
        "confidence_info": {},
        "annotated_image": None,
        "disaster_relevant_objects": [],
        "processing_status": "success",
        "error_message": None,
        "warning": (
            "This model detects COCO objects (people, vehicles, etc.). "
            "It does NOT classify disaster types or structural damage."
        ),
    }

    model, err = _load_yolo()
    if err:
        result["processing_status"] = "model_unavailable"
        result["error_message"] = err
        return result

    try:
        import cv2
        img_array = np.array(image.convert("RGB"))

        # Run inference
        predictions = model(
            img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detections = []
        counts: Dict[str, int] = {}

        for pred in predictions:
            boxes = pred.boxes
            names = pred.names

            for box in boxes:
                cls_id = int(box.cls[0])
                label  = names[cls_id]
                conf   = float(box.conf[0])
                xyxy   = box.xyxy[0].tolist()

                detection = {
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 1) for v in xyxy],  # [x1,y1,x2,y2]
                }
                detections.append(detection)
                counts[label] = counts.get(label, 0) + 1

        result["detected_objects"] = detections
        result["object_counts"] = counts

        # Confidence summary per class
        conf_summary: Dict[str, List[float]] = {}
        for d in detections:
            conf_summary.setdefault(d["label"], []).append(d["confidence"])
        result["confidence_info"] = {
            k: {"min": round(min(v), 4), "max": round(max(v), 4),
                "mean": round(sum(v)/len(v), 4), "count": len(v)}
            for k, v in conf_summary.items()
        }

        # Flag disaster-relevant objects
        result["disaster_relevant_objects"] = [
            label for label in counts
            if label.lower() in DISASTER_RELEVANT_OBJECTS
        ]

        # Generate annotated image
        annotated_array = predictions[0].plot()  # Returns BGR numpy array
        annotated_rgb   = cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB)
        result["annotated_image"] = Image.fromarray(annotated_rgb)

        logger.info(
            f"YOLO detection: {len(detections)} objects, "
            f"classes={list(counts.keys())}"
        )

    except Exception as e:
        logger.error(f"YOLO inference error: {e}", exc_info=True)
        result["processing_status"] = "error"
        result["error_message"] = str(e)

    return result


def get_model_info() -> Dict[str, Any]:
    """Return YOLO model metadata for the evaluation page."""
    model, err = _load_yolo()
    return {
        "name": "YOLOv8n",
        "type": "Object Detection",
        "dataset": "COCO (80 classes)",
        "architecture": "CSPDarknet + PANet + YOLOv8 head",
        "is_configured": model is not None,
        "is_evaluated": False,
        "evaluation_notes": "Pretrained on COCO. Not evaluated on disaster imagery.",
        "limitations": [
            "Detects only COCO categories (person, car, truck, etc.)",
            "Does not classify disaster types",
            "Does not assess structural damage",
            "May underperform on aerial / satellite images",
        ],
        "error": err,
    }
