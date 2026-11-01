"""
Database Repository
All CRUD operations for incidents and analysis results.
Completely separated from UI and AI inference code.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from src.database.connection import get_db
from src.utils.logging_config import get_logger

logger = get_logger("database.repository")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def _dumps(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj, default=str)


def _loads(s: Optional[str]) -> Any:
    if s is None:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


# ── INCIDENTS ─────────────────────────────────────────────────

def create_incident(
    input_type: str,
    title: str = "",
    original_report: str = "",
    reported_location: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> str:
    """Insert a new incident record. Returns incident_id."""
    incident_id = _new_id("INC_")
    now = _now()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO incidents
            (incident_id, title, created_at, updated_at, input_type,
             original_report, reported_location, latitude, longitude,
             processing_status, review_status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (incident_id, title, now, now, input_type,
              original_report, reported_location, latitude, longitude,
              "processing", "pending"))
    logger.info(f"Created incident {incident_id}")
    return incident_id


def update_incident_assessment(
    incident_id: str,
    disaster_type: str = None,
    severity_score: float = None,
    priority: str = None,
    consistency_status: str = None,
    processing_status: str = "completed",
):
    """Update the assessment fields on an existing incident."""
    with get_db() as conn:
        conn.execute("""
            UPDATE incidents SET
                disaster_type=COALESCE(?,disaster_type),
                severity_score=COALESCE(?,severity_score),
                priority=COALESCE(?,priority),
                consistency_status=COALESCE(?,consistency_status),
                processing_status=?,
                updated_at=?
            WHERE incident_id=?
        """, (disaster_type, severity_score, priority, consistency_status,
              processing_status, _now(), incident_id))


def get_incident(incident_id: str) -> Optional[Dict]:
    """Fetch a single incident by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM incidents WHERE incident_id=?", (incident_id,)
        ).fetchone()
    return dict(row) if row else None


def list_incidents(
    limit: int = 100,
    offset: int = 0,
    disaster_type: str = None,
    priority: str = None,
    review_status: str = None,
    search: str = None,
) -> List[Dict]:
    """List incidents with optional filters."""
    query = "SELECT * FROM incidents WHERE 1=1"
    params: List[Any] = []
    if disaster_type:
        query += " AND disaster_type=?"
        params.append(disaster_type)
    if priority:
        query += " AND priority=?"
        params.append(priority)
    if review_status:
        query += " AND review_status=?"
        params.append(review_status)
    if search:
        query += " AND (title LIKE ? OR original_report LIKE ? OR reported_location LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_incident_counts() -> Dict[str, int]:
    """Dashboard summary counts from actual database records."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        with_visual = conn.execute(
            "SELECT COUNT(DISTINCT incident_id) FROM visual_analysis WHERE processing_status='success'"
        ).fetchone()[0]
        with_text = conn.execute(
            "SELECT COUNT(DISTINCT incident_id) FROM text_analysis WHERE processing_status='success'"
        ).fetchone()[0]
        with_fusion = conn.execute(
            "SELECT COUNT(DISTINCT incident_id) FROM fusion_results"
        ).fetchone()[0]
        pending_review = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE review_status='pending'"
        ).fetchone()[0]
        high_priority = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE priority IN ('P1','P2')"
        ).fetchone()[0]
    return {
        "total": total,
        "with_visual": with_visual,
        "with_text": with_text,
        "with_fusion": with_fusion,
        "pending_review": pending_review,
        "high_priority": high_priority,
    }


def get_category_distribution() -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT disaster_type, COUNT(*) as count
            FROM incidents
            WHERE disaster_type IS NOT NULL
            GROUP BY disaster_type ORDER BY count DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_priority_distribution() -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT priority, COUNT(*) as count
            FROM incidents
            WHERE priority IS NOT NULL
            GROUP BY priority ORDER BY count DESC
        """).fetchall()
    return [dict(r) for r in rows]


def update_review_status(incident_id: str, status: str):
    with get_db() as conn:
        conn.execute(
            "UPDATE incidents SET review_status=?, updated_at=? WHERE incident_id=?",
            (status, _now(), incident_id)
        )


# ── VISUAL ANALYSIS ──────────────────────────────────────────

def save_visual_analysis(incident_id: str, data: Dict) -> str:
    analysis_id = _new_id("VA_")
    with get_db() as conn:
        conn.execute("""
            INSERT INTO visual_analysis
            (analysis_id, incident_id, source_filename, input_type, model_name,
             model_version, detected_objects, confidence_info, disaster_prediction,
             damage_prediction, annotated_output_path, frame_count, frames_analyzed,
             processing_status, error_message, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            analysis_id, incident_id,
            data.get("source_filename"), data.get("input_type"),
            data.get("model_name"), data.get("model_version"),
            _dumps(data.get("detected_objects")),
            _dumps(data.get("confidence_info")),
            _dumps(data.get("disaster_prediction")),
            _dumps(data.get("damage_prediction")),
            data.get("annotated_output_path"),
            data.get("frame_count"), data.get("frames_analyzed"),
            data.get("processing_status", "success"),
            data.get("error_message"),
            _now()
        ))
    logger.info(f"Saved visual analysis {analysis_id} for {incident_id}")
    return analysis_id


def get_visual_analysis(incident_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM visual_analysis WHERE incident_id=? ORDER BY created_at DESC LIMIT 1",
            (incident_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    for key in ["detected_objects", "confidence_info", "disaster_prediction", "damage_prediction"]:
        d[key] = _loads(d.get(key))
    return d


# ── TEXT ANALYSIS ────────────────────────────────────────────

def save_text_analysis(incident_id: str, data: Dict) -> str:
    analysis_id = _new_id("TA_")
    with get_db() as conn:
        conn.execute("""
            INSERT INTO text_analysis
            (analysis_id, incident_id, model_name, model_version, original_report,
             predicted_category, classification_conf, extracted_entities,
             extracted_location, reported_damage, assistance_indicators,
             evidence_phrases, confidence_info, uncertainty_notes,
             processing_status, error_message, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            analysis_id, incident_id,
            data.get("model_name"), data.get("model_version"),
            data.get("original_report"),
            data.get("predicted_category"),
            data.get("classification_conf"),
            _dumps(data.get("extracted_entities")),
            data.get("extracted_location"),
            _dumps(data.get("reported_damage")),
            _dumps(data.get("assistance_indicators")),
            _dumps(data.get("evidence_phrases")),
            _dumps(data.get("confidence_info")),
            data.get("uncertainty_notes"),
            data.get("processing_status", "success"),
            data.get("error_message"),
            _now()
        ))
    logger.info(f"Saved text analysis {analysis_id} for {incident_id}")
    return analysis_id


def get_text_analysis(incident_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM text_analysis WHERE incident_id=? ORDER BY created_at DESC LIMIT 1",
            (incident_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    for key in ["extracted_entities", "reported_damage",
                "assistance_indicators", "evidence_phrases", "confidence_info"]:
        d[key] = _loads(d.get(key))
    return d


# ── FUSION RESULTS ───────────────────────────────────────────

def save_fusion_result(incident_id: str, data: Dict) -> str:
    fusion_id = _new_id("FR_")
    with get_db() as conn:
        conn.execute("""
            INSERT INTO fusion_results
            (fusion_id, incident_id, visual_evidence, textual_evidence,
             combined_summary, consistency_findings, severity_score,
             priority, scoring_explanation, uncertainty_notes,
             review_recommendation, missing_information, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            fusion_id, incident_id,
            _dumps(data.get("visual_evidence")),
            _dumps(data.get("textual_evidence")),
            data.get("combined_summary"),
            _dumps(data.get("consistency_findings")),
            data.get("severity_score"),
            data.get("priority"),
            _dumps(data.get("scoring_explanation")),
            data.get("uncertainty_notes"),
            data.get("review_recommendation"),
            _dumps(data.get("missing_information")),
            _now()
        ))
    logger.info(f"Saved fusion result {fusion_id} for {incident_id}")
    return fusion_id


def get_fusion_result(incident_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM fusion_results WHERE incident_id=? ORDER BY created_at DESC LIMIT 1",
            (incident_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    for key in ["visual_evidence", "textual_evidence", "consistency_findings",
                "scoring_explanation", "missing_information"]:
        d[key] = _loads(d.get(key))
    return d


def get_incidents_with_coords() -> List[Dict]:
    """Return only incidents that have valid lat/lon for mapping."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT incident_id, title, disaster_type, priority,
                   latitude, longitude, reported_location, created_at
            FROM incidents
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """).fetchall()
    return [dict(r) for r in rows]
