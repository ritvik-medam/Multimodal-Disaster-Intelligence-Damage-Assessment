"""
Logging Configuration
Structured, leveled logging for the entire application.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """
    Configure root logger with console + optional file handler.
    Returns the root application logger.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger
    logger = logging.getLogger("disaster_intel")
    logger.setLevel(level)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_to_file:
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger."""
    return logging.getLogger(f"disaster_intel.{name}")
