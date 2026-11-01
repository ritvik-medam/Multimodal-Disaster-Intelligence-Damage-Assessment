"""
Multimodal Evidence Fusion Module
Combines structured CV and NLP outputs into a unified incident assessment.
Preserves provenance of every finding.
Does NOT simply concatenate outputs — performs structured evidence merging.
"""
from typing import Dict, Any, List, Optional
from src.utils.logging_config import get_logger

logger = get_logger("fusion.evidence_fusion")

# Provenance labels
VISUAL_ONLY    = "visually_detected"
TEXT_ONLY      = "reported_in_text"
BOTH           = "supported_by_both_modalities"
INFERRED       = "inferred"
UNVERIFIED     = "unverified"
UNKNOWN        = "unknown"


def _normalize_disaster_category(category: Optional[str]) -> Optional[str]:
    """Normalize disaster category strings to a common form."""
    if not category:
        return None
    mapping = {
        "flood": "flood", "flooding": "flood", "inundation": "flood",
        "earthquake": "earthquake", "seismic": "earthquake",
        "landslide": "landslide", "mudslide": "landslide",
        "wildfire": "wildfire", "fire": "wildfire", "forest fire": "wildfire",
        "cyclone_storm": "cyclone_storm", "cyclone": "cyclone_storm",
        "hurricane": "cyclone_storm", "storm": "cyclone_storm", "tornado": "cyclone_storm",
        "other_disaster": "other_disaster", "other": "other_disaster",
        "not_disaster": "not_disaster",
    }
    return mapping.get(category.lower().strip(), category.lower().strip())


def fuse_evidence(
    cv_result:  Optional[Dict[str, Any]],
    nlp_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fuse Computer Vision and NLP results into a unified evidence record.

    Each finding is tagged with its provenance.
    Missing modalities are explicitly flagged — not silently ignored.

    Args:
        cv_result:  Structured output from vision pipeline (or None)
        nlp_result: Structured output from NLP pipeline (or None)

    Returns:
        Unified evidence dict with provenance annotations.
    """
    has_visual = cv_result is not None
    has_text   = nlp_result is not None

    result = {
        # Modality availability
        "has_visual_input": has_visual,
        "has_text_input":   has_text,
        "input_modalities": _get_modality_label(has_visual, has_text),

        # Visual evidence block
        "visual_evidence": {},
        # Text evidence block
        "textual_evidence": {},
        # Combined evidence
        "combined_findings": [],
        # Missing information
        "missing_information": [],
        # Uncertainty
        "uncertainty_notes": [],
    }

    # ── Build Visual Evidence ──────────────────────────────────
    if has_visual:
        ve = {}
        det = cv_result.get("detection", {}) or {}
        ve["detected_objects"]   = det.get("detected_objects", [])
        ve["object_counts"]      = det.get("object_counts", {})

        cls = cv_result.get("classification", {}) or {}
        if cls.get("status") == "success":
            ve["disaster_category"] = {
                "value": _normalize_disaster_category(cls.get("predicted_category")),
                "confidence": cls.get("confidence"),
                "provenance": VISUAL_ONLY,
            }
        else:
            ve["disaster_category"] = {
                "value": None,
                "status": cls.get("status", "not_configured"),
                "provenance": UNKNOWN,
            }

        dmg = cv_result.get("damage", {}) or {}
        if dmg.get("status") == "success":
            ve["damage_level"] = {
                "value": dmg.get("predicted_damage_level"),
                "confidence": dmg.get("confidence"),
                "provenance": VISUAL_ONLY,
                "imagery_warning": dmg.get("imagery_type_warning"),
            }
        else:
            ve["damage_level"] = {
                "value": None,
                "status": dmg.get("status", "not_configured"),
                "provenance": UNKNOWN,
            }

        result["visual_evidence"] = ve

    # ── Build Textual Evidence ────────────────────────────────
    if has_text:
        te = {}
        cls = nlp_result.get("classification", {}) or {}
        if cls.get("status") == "success":
            te["disaster_category"] = {
                "value": _normalize_disaster_category(cls.get("predicted_category")),
                "confidence": cls.get("confidence"),
                "provenance": TEXT_ONLY,
            }
        else:
            te["disaster_category"] = {
                "value": None,
                "status": cls.get("status", "not_configured"),
                "provenance": UNKNOWN,
            }

        em = nlp_result.get("emergency_info", {}) or {}
        te["rescue_request"]    = {"present": em.get("rescue_request", {}).get("present", False), "provenance": TEXT_ONLY}
        te["trapped_people"]    = {"present": em.get("trapped_people", {}).get("present", False), "provenance": TEXT_ONLY}
        te["injuries_reported"] = {"present": em.get("injuries_reported", {}).get("present", False), "provenance": TEXT_ONLY}
        te["casualties_reported"]={"present": em.get("casualties_reported", {}).get("present", False), "provenance": TEXT_ONLY}
        te["building_damage"]   = {"present": em.get("building_damage", {}).get("present", False), "provenance": TEXT_ONLY}
        te["flooding_reported"] = {"present": em.get("flooding_reported", {}).get("present", False), "provenance": TEXT_ONLY}
        te["road_blocked"]      = {"present": em.get("road_blocked", {}).get("present", False), "provenance": TEXT_ONLY}
        te["evacuation"]        = {"present": em.get("evacuation_mentioned", {}).get("present", False), "provenance": TEXT_ONLY}
        te["reported_affected_count"] = em.get("reported_affected_count")
        te["urgency_score"]     = em.get("urgency_score", 0)

        ner = nlp_result.get("entities", {}) or {}
        te["locations"]      = ner.get("locations", [])
        te["dates"]          = ner.get("dates", [])
        te["organizations"]  = ner.get("organizations", [])
        te["evidence_phrases"] = em.get("all_evidence_phrases", [])

        result["textual_evidence"] = te

    # ── Build Combined Findings ────────────────────────────────
    findings = []

    # Disaster category agreement
    if has_visual and has_text:
        vis_cat = result["visual_evidence"].get("disaster_category", {}).get("value")
        txt_cat = result["textual_evidence"].get("disaster_category", {}).get("value")
        if vis_cat and txt_cat:
            if vis_cat == txt_cat:
                findings.append({
                    "finding": f"Disaster category '{vis_cat}' supported by both modalities",
                    "provenance": BOTH,
                    "confidence": "high",
                })
            else:
                findings.append({
                    "finding": f"Potential category discrepancy: visual='{vis_cat}', text='{txt_cat}'",
                    "provenance": UNVERIFIED,
                    "confidence": "low",
                })

    # Rescue / emergency need
    if has_text:
        te = result["textual_evidence"]
        for field, label in [
            ("rescue_request", "Rescue assistance requested"),
            ("trapped_people", "People reported as trapped"),
            ("casualties_reported", "Casualties reported"),
            ("injuries_reported", "Injuries reported"),
            ("evacuation", "Evacuation mentioned"),
        ]:
            if te.get(field, {}).get("present"):
                findings.append({
                    "finding": label,
                    "provenance": TEXT_ONLY,
                    "confidence": "medium",
                })

    # People/vehicles detected visually
    if has_visual:
        obj_counts = result["visual_evidence"].get("object_counts", {})
        if "person" in obj_counts:
            findings.append({
                "finding": f"{obj_counts['person']} person(s) detected visually",
                "provenance": VISUAL_ONLY,
                "confidence": "medium",
            })
        for vehicle in ["car", "truck", "bus", "motorcycle", "boat"]:
            if vehicle in obj_counts:
                findings.append({
                    "finding": f"{obj_counts[vehicle]} {vehicle}(s) detected visually",
                    "provenance": VISUAL_ONLY,
                    "confidence": "medium",
                })

    result["combined_findings"] = findings

    # Primary disaster category resolution
    vis_cat = (result["visual_evidence"].get("disaster_category") or {}).get("value")
    txt_cat = (result["textual_evidence"].get("disaster_category") or {}).get("value")
    
    if vis_cat and txt_cat:
        if vis_cat == txt_cat:
            result["primary_disaster_category"] = vis_cat
            result["primary_category_provenance"] = BOTH
        else:
            # Prefer visual or highest confidence
            result["primary_disaster_category"] = vis_cat
            result["primary_category_provenance"] = VISUAL_ONLY
    elif vis_cat:
        result["primary_disaster_category"] = vis_cat
        result["primary_category_provenance"] = VISUAL_ONLY
    elif txt_cat:
        result["primary_disaster_category"] = txt_cat
        result["primary_category_provenance"] = TEXT_ONLY
    else:
        result["primary_disaster_category"] = "unknown"
        result["primary_category_provenance"] = UNKNOWN

    # ── Missing Information ────────────────────────────────────
    missing = []
    if not has_visual:
        missing.append("No visual input provided — image/video analysis unavailable.")
    if not has_text:
        missing.append("No textual report provided — NLP analysis unavailable.")
    if has_text and not result["textual_evidence"].get("locations"):
        missing.append("No specific geographic location found in text.")
    if has_visual and result["visual_evidence"].get("disaster_category", {}).get("status") == "not_configured":
        missing.append("Disaster classifier not configured — visual disaster type unknown.")

    result["missing_information"] = missing

    # ── Uncertainty Notes ─────────────────────────────────────
    uncertainty = []
    if has_visual:
        ve = result["visual_evidence"]
        if ve.get("damage_level", {}).get("status") in ("not_configured", "error"):
            uncertainty.append("Damage assessment model not configured.")
        if ve.get("disaster_category", {}).get("status") in ("not_configured", "error"):
            uncertainty.append("Disaster image classifier not configured.")
    if has_text:
        te = result["textual_evidence"]
        if te.get("disaster_category", {}).get("status") in ("not_configured", "error"):
            uncertainty.append("Text classifier not configured.")

    result["uncertainty_notes"] = uncertainty

    logger.info(
        f"Fusion complete: modalities={result['input_modalities']}, "
        f"findings={len(findings)}, missing={len(missing)}"
    )
    return result


def _get_modality_label(has_visual: bool, has_text: bool) -> str:
    if has_visual and has_text:
        return "image_text" if True else "video_text"
    if has_visual:
        return "image_only"
    if has_text:
        return "text_only"
    return "none"


def build_incident_summary(fusion_result: Dict, severity_result: Dict) -> str:
    """
    Generate a human-readable consolidated incident summary.
    Based only on actual evidence — no invented information.
    """
    lines = []
    modalities = fusion_result.get("input_modalities", "unknown")
    lines.append(f"**Incident Analysis — Modalities: {modalities.replace('_',' ').title()}**\n")

    # Disaster category
    vis_cat = fusion_result.get("visual_evidence", {}).get("disaster_category", {}).get("value")
    txt_cat = fusion_result.get("textual_evidence", {}).get("disaster_category", {}).get("value")
    if vis_cat and txt_cat and vis_cat == txt_cat:
        lines.append(f"- **Disaster Type:** {vis_cat.replace('_', ' ').title()} *(confirmed by both modalities)*")
    elif vis_cat or txt_cat:
        cats = []
        if vis_cat: cats.append(f"Visual: {vis_cat}")
        if txt_cat: cats.append(f"Text: {txt_cat}")
        lines.append(f"- **Disaster Type:** {', '.join(cats)}")
    else:
        lines.append("- **Disaster Type:** Not determined from available evidence.")

    # Emergency indicators
    te = fusion_result.get("textual_evidence", {})
    emergency_items = []
    if te.get("rescue_request", {}).get("present"):    emergency_items.append("rescue assistance")
    if te.get("trapped_people", {}).get("present"):    emergency_items.append("trapped people")
    if te.get("casualties_reported", {}).get("present"):emergency_items.append("reported casualties")
    if te.get("injuries_reported", {}).get("present"): emergency_items.append("injuries reported")
    if emergency_items:
        lines.append(f"- **Emergency Indicators:** {', '.join(emergency_items).capitalize()}")

    # Affected count
    count = te.get("reported_affected_count")
    if count:
        lines.append(f"- **Reported Affected:** {count} *(exact phrase from report)*")

    # Priority
    priority = severity_result.get("priority", "UNASSESSED")
    score    = severity_result.get("severity_score", 0)
    lines.append(f"- **Preliminary Priority:** {priority} (Score: {score}/100)")

    # Missing info
    missing = fusion_result.get("missing_information", [])
    if missing:
        lines.append(f"\n**Missing Information:**")
        for m in missing:
            lines.append(f"  - {m}")

    lines.append("\n*This is a preliminary AI-assisted assessment. Human review is required.*")
    return "\n".join(lines)
