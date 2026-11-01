"""
Video Processing Module
Frame extraction, sampling, and per-frame CV analysis using OpenCV.
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
from src.utils.logging_config import get_logger
from src.utils import configuration as cfg
from src.vision import object_detection, disaster_classification

logger = get_logger("vision.video_processing")


def _save_temp_video(file_data: bytes, suffix: str = ".mp4") -> Path:
    """Write video bytes to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_data)
    tmp.close()
    return Path(tmp.name)


def extract_metadata(video_path: Path) -> Dict[str, Any]:
    """Extract basic video metadata using OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path}"}

        fps       = cap.get(cv2.CAP_PROP_FPS)
        width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration  = total_frames / fps if fps > 0 else 0
        cap.release()

        return {
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            "resolution": f"{width}x{height}",
        }
    except Exception as e:
        logger.error(f"Metadata extraction error: {e}")
        return {"error": str(e)}


def extract_frames(
    video_path: Path,
    sample_every_n_seconds: float = 2.0,
    max_frames: int = 30,
) -> Tuple[List[Image.Image], List[float]]:
    """
    Sample frames from video at configurable intervals.
    Returns (frames_as_PIL_images, timestamps_in_seconds).

    NOTE: Only sampled frames are analyzed. We do NOT claim to examine
    every frame of the video.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return [], []

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    sample_every = max(1, int(fps * sample_every_n_seconds))

    frames     = []
    timestamps = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            frames.append(pil)
            timestamps.append(round(frame_idx / fps, 2))

            if len(frames) >= max_frames:
                logger.info(f"Reached max_frames={max_frames}, stopping early.")
                break

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path.name}")
    return frames, timestamps


def _are_similar(img1: Image.Image, img2: Image.Image, threshold: float = 0.95) -> bool:
    """Simple duplicate detection using mean absolute difference."""
    try:
        a1 = np.array(img1.resize((64, 64)).convert("L"), dtype=float)
        a2 = np.array(img2.resize((64, 64)).convert("L"), dtype=float)
        diff = np.mean(np.abs(a1 - a2)) / 255.0
        return diff < (1.0 - threshold)
    except Exception:
        return False


def deduplicate_frames(
    frames: List[Image.Image], timestamps: List[float]
) -> Tuple[List[Image.Image], List[float]]:
    """Remove near-duplicate consecutive frames."""
    if not frames:
        return frames, timestamps
    unique_frames = [frames[0]]
    unique_ts     = [timestamps[0]]
    for f, t in zip(frames[1:], timestamps[1:]):
        if not _are_similar(unique_frames[-1], f):
            unique_frames.append(f)
            unique_ts.append(t)
    removed = len(frames) - len(unique_frames)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate frames")
    return unique_frames, unique_ts


def analyze_video(
    file_data: bytes,
    filename: str,
    sample_every_n_seconds: float = 2.0,
    max_frames: int = 20,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """
    Full video analysis pipeline:
      1. Save to temp file
      2. Extract metadata
      3. Sample frames
      4. Deduplicate
      5. Run object detection on each frame
      6. Run disaster classification if model available
      7. Build consolidated summary

    Returns structured result. Does NOT count the same object across frames
    as unique instances.
    """
    result = {
        "source_filename": filename,
        "input_type": "video",
        "metadata": {},
        "total_frames_extracted": 0,
        "frames_analyzed": 0,
        "sample_interval_seconds": sample_every_n_seconds,
        "frame_results": [],
        "consolidated_objects": {},
        "consolidated_disaster_prediction": None,
        "processing_status": "success",
        "error_message": None,
        "important_note": (
            f"Only sampled frames were analyzed "
            f"(1 frame every ~{sample_every_n_seconds}s, max {max_frames} frames). "
            "The entire video was not exhaustively examined."
        ),
    }

    ext    = Path(filename).suffix.lower()
    suffix = ext if ext in [".mp4", ".avi", ".mov"] else ".mp4"
    tmp_path = _save_temp_video(file_data, suffix)

    try:
        # Metadata
        result["metadata"] = extract_metadata(tmp_path)
        if "error" in result["metadata"]:
            result["processing_status"] = "error"
            result["error_message"] = result["metadata"]["error"]
            return result

        # Frame extraction
        frames, timestamps = extract_frames(tmp_path, sample_every_n_seconds, max_frames)
        result["total_frames_extracted"] = len(frames)

        if not frames:
            result["processing_status"] = "warning"
            result["error_message"] = "No frames could be extracted from the video."
            return result

        # Deduplication
        if deduplicate:
            frames, timestamps = deduplicate_frames(frames, timestamps)
        result["frames_analyzed"] = len(frames)

        # Per-frame analysis
        frame_results    = []
        all_object_counts: Dict[str, int] = {}
        disaster_votes:    Dict[str, int] = {}

        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            frame_result = {
                "frame_index": i,
                "timestamp_seconds": ts,
                "object_detection": None,
                "disaster_classification": None,
            }

            # Object detection
            det_result = object_detection.run_object_detection(frame)
            frame_result["object_detection"] = {
                "detected_objects": det_result["detected_objects"],
                "object_counts": det_result["object_counts"],
                "status": det_result["processing_status"],
            }
            for obj, count in det_result["object_counts"].items():
                all_object_counts[obj] = all_object_counts.get(obj, 0) + count

            # Disaster classification (if available)
            cls_result = disaster_classification.classify_disaster(frame)
            if cls_result["status"] == "success":
                frame_result["disaster_classification"] = {
                    "predicted_category": cls_result["predicted_category"],
                    "confidence": cls_result["confidence"],
                }
                cat = cls_result["predicted_category"]
                disaster_votes[cat] = disaster_votes.get(cat, 0) + 1
            else:
                frame_result["disaster_classification"] = {"status": cls_result["status"]}

            frame_results.append(frame_result)

        result["frame_results"] = frame_results
        result["consolidated_objects"] = all_object_counts

        # Majority vote for disaster category
        if disaster_votes:
            top_cat = max(disaster_votes, key=disaster_votes.get)
            result["consolidated_disaster_prediction"] = {
                "predicted_category": top_cat,
                "vote_count": disaster_votes[top_cat],
                "total_frames": len(frames),
                "vote_distribution": disaster_votes,
            }

        logger.info(
            f"Video analysis complete: {len(frames)} frames, "
            f"objects={all_object_counts}"
        )

    except Exception as e:
        logger.error(f"Video processing error: {e}", exc_info=True)
        result["processing_status"] = "error"
        result["error_message"] = str(e)
    finally:
        try:
            os.unlink(str(tmp_path))
        except Exception:
            pass

    return result
