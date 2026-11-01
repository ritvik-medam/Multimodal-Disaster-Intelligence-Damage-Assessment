"""
Configuration Manager
Loads environment variables and provides typed configuration access.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ── Project Root ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── Directories ─────────────────────────────────────────────
DATA_DIR         = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
MODELS_DIR       = PROJECT_ROOT / os.getenv("MODELS_DIR", "models")
UPLOADS_DIR      = PROJECT_ROOT / os.getenv("UPLOADS_DIR", "data/uploads")
PROCESSED_DIR    = PROJECT_ROOT / os.getenv("PROCESSED_DIR", "data/processed")
SAMPLE_DIR       = PROJECT_ROOT / "data" / "sample_reports"
LOGS_DIR         = PROJECT_ROOT / "logs"

# ── Database ─────────────────────────────────────────────────
DATABASE_PATH = PROJECT_ROOT / os.getenv("DATABASE_PATH", "data/disaster_intelligence.db")

# ── Model Paths ──────────────────────────────────────────────
YOLO_MODEL_PATH             = PROJECT_ROOT / os.getenv("YOLO_MODEL_PATH", "models/yolov8n.pt")
DISASTER_CLASSIFIER_PATH    = PROJECT_ROOT / os.getenv("DISASTER_CLASSIFIER_PATH", "models/disaster_classifier/best_model.pth")
TEXT_CLASSIFIER_PATH        = PROJECT_ROOT / os.getenv("TEXT_CLASSIFIER_PATH", "models/text_classifier/best_model.pth")
DAMAGE_ASSESSMENT_PATH      = PROJECT_ROOT / os.getenv("DAMAGE_ASSESSMENT_PATH", "models/damage_assessment/best_model.pth")
MULTIMODAL_MODEL_PATH       = PROJECT_ROOT / os.getenv("MULTIMODAL_MODEL_PATH", "models/multimodal_model/best_model.pth")

# ── App Settings ─────────────────────────────────────────────
APP_ENV     = os.getenv("APP_ENV", "development")
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")
DEVICE      = os.getenv("DEVICE", "auto")

# ── File Limits ──────────────────────────────────────────────
MAX_IMAGE_SIZE_MB   = int(os.getenv("MAX_IMAGE_SIZE_MB", "50"))
MAX_VIDEO_SIZE_MB   = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
ALLOWED_IMAGE_TYPES = set(os.getenv("ALLOWED_IMAGE_TYPES", "jpg,jpeg,png").split(","))
ALLOWED_VIDEO_TYPES = set(os.getenv("ALLOWED_VIDEO_TYPES", "mp4,avi,mov").split(","))

# ── Disaster Categories ───────────────────────────────────────
DISASTER_CATEGORIES = [
    "flood",
    "earthquake",
    "landslide",
    "wildfire",
    "cyclone_storm",
    "other_disaster",
    "not_disaster",
    "unknown",
]

# ── Priority Thresholds (configurable) ───────────────────────
PRIORITY_THRESHOLDS = {
    "P1_CRITICAL": 80,
    "P2_HIGH":     60,
    "P3_MODERATE": 35,
    "P4_LOW":       0,
}

# ── Damage Categories ─────────────────────────────────────────
DAMAGE_CATEGORIES = [
    "no_damage",
    "minor_damage",
    "major_damage",
    "destroyed",
    "uncertain",
]

# ── Severity Scoring Weights ──────────────────────────────────
SEVERITY_WEIGHTS = {
    "rescue_request":          25,
    "trapped_people":          25,
    "injuries_reported":       20,
    "casualties_reported":     30,
    "building_damage_severe":  20,
    "infrastructure_blocked":  15,
    "evacuation_needed":       20,
    "disaster_confirmed_both": 15,
    "disaster_confirmed_one":   8,
    "evidence_completeness":   10,
}

# ── Training Defaults ─────────────────────────────────────────
TRAINING_BATCH_SIZE = int(os.getenv("TRAINING_BATCH_SIZE", "32"))
TRAINING_EPOCHS     = int(os.getenv("TRAINING_EPOCHS", "50"))
TRAINING_LR         = float(os.getenv("TRAINING_LR", "0.0001"))
TRAINING_WORKERS    = int(os.getenv("TRAINING_WORKERS", "4"))


def get_device():
    """Auto-detect best available device."""
    import torch
    setting = DEVICE.lower()
    if setting == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return setting


def ensure_dirs():
    """Create all required directories."""
    dirs = [DATA_DIR, MODELS_DIR, UPLOADS_DIR, PROCESSED_DIR,
            SAMPLE_DIR, LOGS_DIR,
            MODELS_DIR / "disaster_classifier",
            MODELS_DIR / "text_classifier",
            MODELS_DIR / "damage_assessment",
            MODELS_DIR / "multimodal_model"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    # keep-files so git tracks empty dirs
    for d in [UPLOADS_DIR, PROCESSED_DIR, MODELS_DIR]:
        keep = d / ".gitkeep"
        if not keep.exists():
            keep.touch()
