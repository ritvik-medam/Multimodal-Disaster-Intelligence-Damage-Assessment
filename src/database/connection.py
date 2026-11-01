"""
Database Connection Manager
Thread-safe SQLite connection with WAL mode for concurrent reads.
"""
import sqlite3
import threading
from pathlib import Path
from contextlib import contextmanager
from src.utils.logging_config import get_logger
from src.utils import configuration as cfg

logger = get_logger("database.connection")

# Thread-local storage for connections
_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Return a thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = cfg.DATABASE_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
        logger.debug(f"Opened SQLite connection: {db_path}")
    return _local.conn


@contextmanager
def get_db():
    """Context manager yielding a connection with auto-commit/rollback."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error, rolled back: {e}")
        raise


def close_connection():
    """Close the thread-local connection."""
    if hasattr(_local, "conn") and _local.conn:
        _local.conn.close()
        _local.conn = None
        logger.debug("SQLite connection closed.")
