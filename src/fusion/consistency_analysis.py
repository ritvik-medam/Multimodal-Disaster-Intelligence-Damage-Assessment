"""
Cross-Modal Consistency Analysis
Compares textual claims against available visual evidence.
Carefully distinguishes actual contradictions from missing/insufficient evidence.
"""
from typing import Dict, Any, List
from src.utils.logging_config import get_logger

logger = get_logger("fusion.consistency_analysis")

# Consistency outcome labels
CONSISTENT        = "consistent_both_modalities"
POTENTIAL_INCONS  = "potential_inconsistency"
INSUF_VISUAL      = "insufficient_visual_evidence"
INSUF_TEXT        = "insufficient_textual_evidence"
UNABLE_TO_DETERMINE = "unable_to_determine"

# Disaster category semantic groups (for soft matching)
WATER_EVENTS   = {"flood", "cyclone_storm"}
GROUND_EVENTS  = {"earthquake", "landslide"}
FIRE_EVENTS    = {"wildfire"}


def _soft_category_match(cat1: str, cat2: str) -> bool:
    """Check if two categories belong to the same semantic group."""
    if cat1 == cat2:
        return True
    for group in [WATER_EVENTS, GROUND_EVENTS, FIRE_EVENTS]:
        if cat1 in group and cat2 in group:
            return True
    return False


def analyze_consistency(fusion_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform cross-modal consistency analysis on a fusion result.

    Returns a list of comparison items, each with:
    - what was compared
    - outcome (consistent/inconsistency/insufficient/unable)
    - explanation
    - important note: we do NOT conclude text is false just because
      the image lacks evidence

    Args:
        fusion_result: Output from evidence_fusion.fuse_evidence()

    Returns:
        Consistency analysis result dict.
    """
    checks   = []
    has_vis  = fusion_result.get("has_visual_input", False)
    has_text = fusion_result.get("has_text_input", False)

    ve = fusion_result.get("visual_evidence", {})
    te = fusion_result.get("textual_evidence", {})

    result = {
        "overall_consistency": UNABLE_TO_DETERMINE,
        "checks": [],
        "summary": "",
        "important_caveat": (
            "A textual claim is not concluded to be false merely because "
            "an image does not show supporting evidence. The image may cover "
            "only part of the incident, or the text may describe conditions "
            "outside the captured scene."
        ),
    }

    # ── Check 1: Disaster Category Agreement ──────────────────
    vis_cat = ve.get("disaster_category", {}).get("value") if has_vis else None
    txt_cat = te.get("disaster_category", {}).get("value") if has_text else None
    vis_cat_status = ve.get("disaster_category", {}).get("status") if has_vis else "no_visual"
    txt_cat_status = te.get("disaster_category", {}).get("status") if has_text else "no_text"

    if vis_cat and txt_cat:
        if _soft_category_match(vis_cat, txt_cat):
            checks.append(_check(
                "Disaster category",
                CONSISTENT,
                f"Visual ({vis_cat}) and textual ({txt_cat}) categories are consistent.",
            ))
        else:
            checks.append(_check(
                "Disaster category",
                POTENTIAL_INCONS,
                f"Visual model predicted '{vis_cat}' but text indicates '{txt_cat}'. "
                "This may reflect a partial view, different event aspects, or model error. "
                "Human verification recommended.",
            ))
    elif vis_cat and not txt_cat:
        checks.append(_check(
            "Disaster category",
            INSUF_TEXT,
            f"Visual model indicates '{vis_cat}' but text classification is unavailable "
            f"(status: {txt_cat_status}). Cannot compare.",
        ))
    elif txt_cat and not vis_cat:
        checks.append(_check(
            "Disaster category",
            INSUF_VISUAL,
            f"Text indicates '{txt_cat}' but visual classification is unavailable "
            f"(status: {vis_cat_status}). Cannot compare.",
        ))
    else:
        checks.append(_check(
            "Disaster category",
            UNABLE_TO_DETERMINE,
            "Neither visual nor textual disaster classification is available.",
        ))

    # ── Check 2: Vehicles — text reports stranded vs visual detection ──
    if has_vis and has_text:
        vis_vehicles = sum(
            ve.get("object_counts", {}).get(v, 0)
            for v in ["car", "truck", "bus", "motorcycle"]
        )
        txt_stranded = te.get("rescue_request", {}).get("present", False)

        if txt_stranded and vis_vehicles > 0:
            checks.append(_check(
                "Stranded vehicles",
                CONSISTENT,
                f"Text mentions stranded/rescue situation; "
                f"{vis_vehicles} vehicle(s) detected visually.",
            ))
        elif txt_stranded and vis_vehicles == 0:
            checks.append(_check(
                "Stranded vehicles",
                INSUF_VISUAL,
                "Text mentions a rescue/stranded situation but no vehicles detected in image. "
                "Note: vehicles may be off-frame, submerged, or the image may not show that area.",
            ))

    # ── Check 3: Flooding — text vs visual ────────────────────
    if has_text:
        txt_flood = te.get("flooding_reported", {}).get("present", False)
        vis_flood_cat = vis_cat in WATER_EVENTS if vis_cat else False

        if txt_flood and vis_flood_cat:
            checks.append(_check(
                "Flooding",
                CONSISTENT,
                "Both text and visual classification indicate flooding / water-related event.",
            ))
        elif txt_flood and has_vis and not vis_flood_cat and vis_cat:
            checks.append(_check(
                "Flooding",
                POTENTIAL_INCONS,
                f"Text reports flooding but visual model predicted '{vis_cat}'. "
                "The image may not show the flooded area. Human review recommended.",
            ))
        elif txt_flood and has_vis and not vis_cat:
            checks.append(_check(
                "Flooding",
                INSUF_VISUAL,
                "Text reports flooding but visual disaster classifier is not configured.",
            ))

    # ── Check 4: People — text mentions trapped vs visual detection ──
    if has_vis and has_text:
        vis_persons  = ve.get("object_counts", {}).get("person", 0)
        txt_trapped  = te.get("trapped_people", {}).get("present", False)
        txt_affected = te.get("reported_affected_count")

        if txt_trapped and vis_persons > 0:
            checks.append(_check(
                "People presence",
                CONSISTENT,
                f"Text mentions trapped people; {vis_persons} person(s) detected visually.",
            ))
        elif txt_trapped and vis_persons == 0:
            checks.append(_check(
                "People presence",
                INSUF_VISUAL,
                "Text reports trapped people but no persons detected in image. "
                "Note: people may be off-frame, obscured, or inside structures.",
            ))

    # ── Overall Consistency Score ─────────────────────────────
    outcome_counts = {}
    for c in checks:
        outcome_counts[c["outcome"]] = outcome_counts.get(c["outcome"], 0) + 1

    if not checks:
        overall = UNABLE_TO_DETERMINE
        summary = "Insufficient evidence from either modality to perform consistency checks."
    elif outcome_counts.get(POTENTIAL_INCONS, 0) > 0:
        overall = POTENTIAL_INCONS
        summary = (
            f"{outcome_counts[POTENTIAL_INCONS]} potential inconsistency(ies) found. "
            "Human review strongly recommended."
        )
    elif outcome_counts.get(CONSISTENT, 0) >= len(checks):
        overall = CONSISTENT
        summary = "Available evidence from both modalities is consistent."
    elif outcome_counts.get(INSUF_VISUAL, 0) + outcome_counts.get(INSUF_TEXT, 0) >= len(checks):
        overall = INSUF_VISUAL
        summary = "Insufficient evidence from one or both modalities to determine consistency."
    else:
        overall = UNABLE_TO_DETERMINE
        summary = "Mixed consistency outcomes — see individual checks."

    result["checks"]               = checks
    result["overall_consistency"]  = overall
    result["summary"]              = summary
    result["outcome_distribution"] = outcome_counts

    logger.info(f"Consistency analysis: overall={overall}, checks={len(checks)}")
    return result


def _check(comparison: str, outcome: str, explanation: str) -> Dict:
    return {
        "comparison": comparison,
        "outcome":    outcome,
        "explanation": explanation,
    }
