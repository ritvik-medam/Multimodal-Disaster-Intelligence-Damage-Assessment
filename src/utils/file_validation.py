"""
File Validation Utilities
Validates uploaded images and videos before processing.
"""
import os
import io
import re
from pathlib import Path
from typing import Tuple, Optional, Any
from src.utils.logging_config import get_logger
from src.utils import configuration as cfg

logger = get_logger("file_validation")

# Magic bytes for format identification
IMAGE_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
}

VIDEO_MAGIC = {
    b"\x00\x00\x00": "mp4",   # ftyp boxes
    b"RIFF": "avi",
}


def _read_magic(data: bytes, n: int = 8) -> bytes:
    return data[:n]


def validate_image(file_data: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate an uploaded image file.

    Returns:
        (is_valid, error_message)
    """
    filename = filename.lower()
    ext = Path(filename).suffix.lstrip(".")

    # Extension check
    if ext not in cfg.ALLOWED_IMAGE_TYPES:
        return False, f"Unsupported file type '.{ext}'. Allowed: {cfg.ALLOWED_IMAGE_TYPES}"

    # Size check
    size_mb = len(file_data) / (1024 * 1024)
    if size_mb > cfg.MAX_IMAGE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Max: {cfg.MAX_IMAGE_SIZE_MB} MB"

    # Empty file check
    if len(file_data) == 0:
        return False, "File is empty."

    # Magic bytes / corruption check
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(file_data))
        img.verify()
        return True, ""
    except Exception as e:
        logger.warning(f"Image validation failed for {filename}: {e}")
        return False, f"File appears corrupted or is not a valid image: {e}"


def validate_video(file_data: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate an uploaded video file.

    Returns:
        (is_valid, error_message)
    """
    filename_lower = filename.lower()
    ext = Path(filename_lower).suffix.lstrip(".")

    if ext not in cfg.ALLOWED_VIDEO_TYPES:
        return False, f"Unsupported video type '.{ext}'. Allowed: {cfg.ALLOWED_VIDEO_TYPES}"

    size_mb = len(file_data) / (1024 * 1024)
    if size_mb > cfg.MAX_VIDEO_SIZE_MB:
        return False, f"Video too large ({size_mb:.1f} MB). Max: {cfg.MAX_VIDEO_SIZE_MB} MB"

    if len(file_data) == 0:
        return False, "Video file is empty."

    # We can't fully decode video in memory without writing to disk.
    # Basic sanity: file must be > 1 KB
    if len(file_data) < 1024:
        return False, "Video file is too small to be valid."

    return True, ""


def validate_text_report(text: str) -> Tuple[bool, str]:
    """
    Validate a textual emergency report.

    Returns:
        (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Report text is empty."

    if len(text.strip()) < 10:
        return False, "Report is too short. Please provide a meaningful description."

    if len(text) > 50_000:
        return False, "Report is too long. Max 50,000 characters."

    return True, ""


def _parse_single_coord_token(token: Any) -> Optional[float]:
    """Parses a single coordinate string, recognizing cardinal directions (N/S/E/W)."""
    if token is None:
        return None
    s = str(token).strip().upper()
    if not s:
        return None
    multiplier = 1.0
    if "S" in s or "W" in s:
        multiplier = -1.0
    cleaned = re.sub(r"[^0-9.-]", "", s)
    if not cleaned or cleaned in ("-", ".", "-."):
        return None
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def parse_and_normalize_coordinates(
    lat_raw: Any, 
    lon_raw: Any = None
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Intelligently parses, extracts, normalizes, and validates coordinate inputs.
    
    Handles:
    1. Single input containing comma/space separated pairs: "13.0827, 80.2707"
    2. Cardinal directional inputs: "3.4653 S", "62.2159 W" -> (-3.4653, -62.2159)
    3. Swapped latitude & longitude: if |lat| > 90 and |lon| <= 90, auto-swaps them.
    
    Returns:
        (latitude_float, longitude_float, message)
    """
    lat_s = str(lat_raw or "").strip()
    lon_s = str(lon_raw or "").strip()
    
    if not lat_s and not lon_s:
        return None, None, ""

    # Check if a combined coordinate string was pasted into lat_raw (e.g., "13.0827, 80.2707")
    if ("," in lat_s or " " in lat_s) and not lon_s:
        parts = [p.strip() for p in re.split(r"[,;\s]+", lat_s) if p.strip()]
        if len(parts) >= 2:
            lat_s, lon_s = parts[0], parts[1]

    lat = _parse_single_coord_token(lat_s)
    lon = _parse_single_coord_token(lon_s)

    if lat is None and lon is None:
        return None, None, "Coordinates could not be parsed as numeric values."
    if lat is None or lon is None:
        return None, None, "Both latitude and longitude must be provided."

    msg = ""
    # Auto-detect swapped coordinates where user placed longitude in the latitude field
    # (e.g. lat = 120.5 and lon = 14.2)
    if (abs(lat) > 90 and abs(lon) <= 90):
        lat, lon = lon, lat
        msg = f"Inverted coordinates detected and corrected: Latitude {lat:.4f}, Longitude {lon:.4f}."

    return lat, lon, msg


def validate_coordinates(lat: Optional[float], lon: Optional[float]) -> Tuple[bool, str]:
    """
    Validate geographic coordinates.

    Returns:
        (is_valid, error_message)
    """
    if lat is None and lon is None:
        return True, ""  # Both absent is fine

    if lat is None or lon is None:
        return False, "Both latitude and longitude must be provided together."

    if not (-90 <= lat <= 90):
        return False, f"Latitude {lat} is out of range [-90, 90]."

    if not (-180 <= lon <= 180):
        return False, f"Longitude {lon} is out of range [-180, 180]."

    return True, ""


def save_upload(file_data: bytes, filename: str, subdir: str = "") -> Path:
    """
    Save an uploaded file to the uploads directory.

    Returns:
        Path to saved file.
    """
    import uuid
    upload_dir = cfg.UPLOADS_DIR
    if subdir:
        upload_dir = upload_dir / subdir
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Unique name to avoid collisions
    stem = Path(filename).stem
    ext  = Path(filename).suffix
    unique_name = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = upload_dir / unique_name

    with open(save_path, "wb") as f:
        f.write(file_data)

    logger.info(f"Saved upload: {save_path} ({len(file_data)/1024:.1f} KB)")
    return save_path
