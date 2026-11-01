"""
Database Schema — Table creation and migration.
Run initialize_database() on application startup.
"""
from src.database.connection import get_db
from src.utils.logging_config import get_logger

logger = get_logger("database.models")

# ─────────────────────────────────────────────────────────────
# TABLE DEFINITIONS
# ─────────────────────────────────────────────────────────────

CREATE_INCIDENTS = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id      TEXT PRIMARY KEY,
    title            TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT,
    input_type       TEXT NOT NULL,          -- image|video|text|image_text|video_text
    original_report  TEXT,
    reported_location TEXT,
    latitude         REAL,
    longitude        REAL,
    disaster_type    TEXT,
    severity_score   REAL,
    priority         TEXT,                   -- P1|P2|P3|P4|UNASSESSED
    consistency_status TEXT,
    review_status    TEXT DEFAULT 'pending', -- pending|reviewed|escalated|closed
    processing_status TEXT DEFAULT 'processing',
    notes            TEXT
);
"""

CREATE_VISUAL_ANALYSIS = """
CREATE TABLE IF NOT EXISTS visual_analysis (
    analysis_id         TEXT PRIMARY KEY,
    incident_id         TEXT NOT NULL,
    source_filename     TEXT,
    input_type          TEXT,                -- image|video
    model_name          TEXT,
    model_version       TEXT,
    detected_objects    TEXT,                -- JSON list
    confidence_info     TEXT,                -- JSON dict
    disaster_prediction TEXT,                -- JSON dict or NULL
    damage_prediction   TEXT,                -- JSON dict or NULL
    annotated_output_path TEXT,
    frame_count         INTEGER,
    frames_analyzed     INTEGER,
    processing_status   TEXT DEFAULT 'success',
    error_message       TEXT,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id) ON DELETE CASCADE
);
"""

CREATE_TEXT_ANALYSIS = """
CREATE TABLE IF NOT EXISTS text_analysis (
    analysis_id          TEXT PRIMARY KEY,
    incident_id          TEXT NOT NULL,
    model_name           TEXT,
    model_version        TEXT,
    original_report      TEXT,
    predicted_category   TEXT,
    classification_conf  REAL,
    extracted_entities   TEXT,               -- JSON list
    extracted_location   TEXT,
    reported_damage      TEXT,               -- JSON list
    assistance_indicators TEXT,              -- JSON dict
    evidence_phrases     TEXT,               -- JSON list
    confidence_info      TEXT,               -- JSON dict
    uncertainty_notes    TEXT,
    processing_status    TEXT DEFAULT 'success',
    error_message        TEXT,
    created_at           TEXT NOT NULL,
    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id) ON DELETE CASCADE
);
"""

CREATE_FUSION_RESULTS = """
CREATE TABLE IF NOT EXISTS fusion_results (
    fusion_id            TEXT PRIMARY KEY,
    incident_id          TEXT NOT NULL,
    visual_evidence      TEXT,               -- JSON
    textual_evidence     TEXT,               -- JSON
    combined_summary     TEXT,
    consistency_findings TEXT,               -- JSON
    severity_score       REAL,
    priority             TEXT,
    scoring_explanation  TEXT,               -- JSON list of factors
    uncertainty_notes    TEXT,
    review_recommendation TEXT,
    missing_information  TEXT,               -- JSON list
    created_at           TEXT NOT NULL,
    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id) ON DELETE CASCADE
);
"""

CREATE_MODEL_REGISTRY = """
CREATE TABLE IF NOT EXISTS model_registry (
    model_id          TEXT PRIMARY KEY,
    model_name        TEXT NOT NULL,
    model_type        TEXT NOT NULL,         -- vision|text|damage|multimodal|detection
    version           TEXT,
    architecture      TEXT,
    dataset_used      TEXT,
    supported_labels  TEXT,                  -- JSON list
    checkpoint_path   TEXT,
    is_configured     INTEGER DEFAULT 0,     -- 0=no, 1=yes
    is_evaluated      INTEGER DEFAULT 0,
    evaluation_metrics TEXT,                 -- JSON dict
    known_limitations TEXT,
    registered_at     TEXT NOT NULL,
    last_updated      TEXT
);
"""

ALL_TABLES = [
    CREATE_INCIDENTS,
    CREATE_VISUAL_ANALYSIS,
    CREATE_TEXT_ANALYSIS,
    CREATE_FUSION_RESULTS,
    CREATE_MODEL_REGISTRY,
]


def initialize_database():
    """Create all tables if they do not exist."""
    with get_db() as conn:
        for ddl in ALL_TABLES:
            conn.execute(ddl)
    logger.info("Database initialized — all tables ready.")


def drop_all_tables():
    """Drop all tables (for testing / reset only)."""
    tables = ["fusion_results", "text_analysis", "visual_analysis",
              "incidents", "model_registry"]
    with get_db() as conn:
        for t in tables:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
    logger.warning("All tables dropped.")
