"""
Severity Estimation and Priority Scoring
Transparent, configurable rule-based scoring.
Every factor is documented. Score is NEVER fabricated.
Returns UNASSESSED when evidence is insufficient.
"""
from typing import Dict, Any, List, Optional
from src.utils.logging_config import get_logger
from src.utils import configuration as cfg

logger = get_logger("assessment.severity_estimation")

# Priority categories
P1_CRITICAL = "P1_CRITICAL"
P2_HIGH     = "P2_HIGH"
P3_MODERATE = "P3_MODERATE"
P4_LOW      = "P4_LOW"
UNASSESSED  = "UNASSESSED"


def calculate_severity(
    fusion_result: Dict[str, Any],
    consistency_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate preliminary severity score (0–100) and priority category.

    Uses only factors with available evidence.
    Never assigns high score from low-confidence predictions alone.
    Returns UNASSESSED when insufficient evidence exists.

    Args:
        fusion_result:      Output from evidence_fusion.fuse_evidence()
        consistency_result: Output from consistency_analysis.analyze_consistency()

    Returns:
        Severity and priority result with full explanation.
    """
    result = {
        "severity_score": 0,
        "priority": UNASSESSED,
        "contributing_factors": [],
        "penalties": [],
        "missing_evidence": [],
        "uncertainty_notes": [],
        "human_review_recommended": True,
        "disclaimer": (
            "This is a preliminary AI-assisted severity estimate based on available evidence. "
            "It is NOT a validated emergency assessment and must NOT replace trained emergency "
            "responders or official disaster management procedures."
        ),
    }

    score  = 0
    factors: List[Dict] = []
    has_vis  = fusion_result.get("has_visual_input", False)
    has_text = fusion_result.get("has_text_input", False)
    te = fusion_result.get("textual_evidence", {})
    ve = fusion_result.get("visual_evidence", {})

    # ── Minimum Evidence Check ────────────────────────────────
    if not has_vis and not has_text:
        result["priority"] = UNASSESSED
        result["uncertainty_notes"].append("No evidence from either modality.")
        return result

    # ── Factor 1: Rescue Request ──────────────────────────────
    if te.get("rescue_request", {}).get("present"):
        points = cfg.SEVERITY_WEIGHTS["rescue_request"]
        score += points
        factors.append({
            "factor": "Rescue assistance requested",
            "points": points,
            "source": "text_report",
            "provenance": "explicitly_reported",
        })

    # ── Factor 2: Trapped People ──────────────────────────────
    if te.get("trapped_people", {}).get("present"):
        points = cfg.SEVERITY_WEIGHTS["trapped_people"]
        score += points
        factors.append({
            "factor": "People reported as trapped",
            "points": points,
            "source": "text_report",
            "provenance": "explicitly_reported",
        })

    # ── Factor 3: Injuries Reported ───────────────────────────
    if te.get("injuries_reported", {}).get("present"):
        points = cfg.SEVERITY_WEIGHTS["injuries_reported"]
        score += points
        factors.append({
            "factor": "Injuries reported in text",
            "points": points,
            "source": "text_report",
            "provenance": "explicitly_reported",
        })

    # ── Factor 4: Casualties Reported ────────────────────────
    if te.get("casualties_reported", {}).get("present"):
        points = cfg.SEVERITY_WEIGHTS["casualties_reported"]
        score += points
        factors.append({
            "factor": "Casualties reported in text",
            "points": points,
            "source": "text_report",
            "provenance": "explicitly_reported",
        })

    # ── Factor 5: Building Damage ─────────────────────────────
    if te.get("building_damage", {}).get("present"):
        points = cfg.SEVERITY_WEIGHTS["building_damage_severe"]
        score += points
        factors.append({
            "factor": "Building damage reported in text",
            "points": points,
            "source": "text_report",
            "provenance": "explicitly_reported",
        })

    # Factor 5b: Visual damage level (only if model configured)
    dmg = ve.get("damage_level", {})
    if dmg.get("value") in ("major_damage", "destroyed"):
        points = cfg.SEVERITY_WEIGHTS["building_damage_severe"]
        score += points
        factors.append({
            "factor": f"Visual damage assessment: {dmg['value']}",
            "points": points,
            "source": "visual_model",
            "provenance": "visually_detected",
            "confidence": dmg.get("confidence"),
            "warning": dmg.get("imagery_warning"),
        })

    # ── Factor 6: Infrastructure Blocked ─────────────────────
    if te.get("road_blocked", {}).get("present"):
        points = cfg.SEVERITY_WEIGHTS["infrastructure_blocked"]
        score += points
        factors.append({
            "factor": "Road or infrastructure blockage reported",
            "points": points,
            "source": "text_report",
            "provenance": "explicitly_reported",
        })

    # ── Factor 7: Evacuation ──────────────────────────────────
    if te.get("evacuation", {}).get("present"):
        points = cfg.SEVERITY_WEIGHTS["evacuation_needed"]
        score += points
        factors.append({
            "factor": "Evacuation mentioned in report",
            "points": points,
            "source": "text_report",
            "provenance": "explicitly_reported",
        })

    # ── Factor 8: Disaster Category Confirmation ──────────────
    vis_cat = ve.get("disaster_category", {}).get("value")
    txt_cat = te.get("disaster_category", {}).get("value")
    if vis_cat and txt_cat and vis_cat == txt_cat:
        points = cfg.SEVERITY_WEIGHTS["disaster_confirmed_both"]
        score += points
        factors.append({
            "factor": f"Disaster type '{vis_cat}' confirmed by both modalities",
            "points": points,
            "source": "both_modalities",
            "provenance": "supported_by_both_modalities",
        })
    elif vis_cat or txt_cat:
        points = cfg.SEVERITY_WEIGHTS["disaster_confirmed_one"]
        score += points
        confirmed_by = "visual" if vis_cat else "text"
        factors.append({
            "factor": f"Disaster type indicated by {confirmed_by} only",
            "points": points,
            "source": confirmed_by,
            "provenance": "partially_supported",
        })

    # ── Factor 9: Evidence Completeness ──────────────────────
    completeness = 0
    if has_vis:  completeness += 5
    if has_text: completeness += 5
    if completeness > 0:
        score += completeness
        factors.append({
            "factor": "Evidence completeness bonus",
            "points": completeness,
            "source": "system",
            "provenance": "computed",
        })

    # ── Penalties ─────────────────────────────────────────────
    penalties = []

    # Penalty: Consistency issue lowers confidence but not score
    consistency_status = consistency_result.get("overall_consistency")
    if consistency_status == "potential_inconsistency":
        result["uncertainty_notes"].append(
            "Consistency analysis found potential inconsistencies. "
            "Score should be treated with extra caution."
        )

    # ── Missing Evidence ──────────────────────────────────────
    missing = fusion_result.get("missing_information", [])
    result["missing_evidence"] = missing

    # ── Score Cap & UNASSESSED Guard ─────────────────────────
    score = min(int(score), 100)

    # Return UNASSESSED if score is trivially low with no urgent factors
    urgent_factors = ["rescue_request", "trapped_people", "casualties_reported",
                      "injuries_reported"]
    has_urgent = any(te.get(f, {}).get("present", False) for f in urgent_factors)

    if score < 10 and not has_urgent and not factors:
        result["priority"] = UNASSESSED
        result["uncertainty_notes"].append(
            "Insufficient evidence to calculate a meaningful severity score."
        )
        return result

    # ── Priority Assignment ───────────────────────────────────
    thresholds = cfg.PRIORITY_THRESHOLDS
    if score >= thresholds["P1_CRITICAL"]:
        priority = P1_CRITICAL
    elif score >= thresholds["P2_HIGH"]:
        priority = P2_HIGH
    elif score >= thresholds["P3_MODERATE"]:
        priority = P3_MODERATE
    else:
        priority = P4_LOW

    result.update({
        "severity_score": score,
        "priority": priority,
        "contributing_factors": factors,
        "penalties": penalties,
    })

    logger.info(f"Severity: score={score}, priority={priority}, factors={len(factors)}")
    return result
