"""
Emergency Information Extraction Module
Identifies rescue requests, medical needs, trapped people, damage descriptions,
infrastructure blockages, and evacuation requirements using transparent rules.
All extracted facts are labeled with their source (explicit/inferred).
"""
import re
from typing import Dict, Any, List, Optional
from src.utils.logging_config import get_logger

logger = get_logger("nlp.emergency_information")

# ── Keyword Patterns ─────────────────────────────────────────

_RESCUE_KEYWORDS = [
    r'\brescue\b', r'\bhelp\b', r'\bassistance\b', r'\baid\b', r'\bsave\b',
    r'\bstranded\b', r'\btrapped\b', r'\bsend help\b', r'\bSOS\b',
    r'\bemergency\b', r'\bneed help\b',
]

_MEDICAL_KEYWORDS = [
    r'\binjured\b', r'\binjuries\b', r'\bwounded\b', r'\bmedical\b',
    r'\bambulance\b', r'\bhospital\b', r'\bdoctor\b', r'\bblood\b',
    r'\bparamedic\b', r'\bfirst aid\b', r'\bcasualt(?:y|ies)\b',
]

_TRAPPED_KEYWORDS = [
    r'\btrapped\b', r'\bstuck\b', r'\bcaught\b', r'\bunable to escape\b',
    r'\bcannot leave\b', r'\bblocked\b.*(?:inside|building|home)',
    r'\bsurrounded by (?:water|fire|debris)\b',
]

_INJURED_KEYWORDS = [
    r'\binjur(?:ed|ies|y)\b', r'\bwound(?:ed|s)\b', r'\bhurt\b',
    r'\bcritical condition\b', r'\bseriously hurt\b',
]

_CASUALTY_KEYWORDS = [
    r'\bdead\b', r'\bdeaths?\b', r'\bkilled\b', r'\bfatalit(?:y|ies)\b',
    r'\bcasualt(?:y|ies)\b', r'\bmissing\b', r'\bmissing person\b',
]

_BUILDING_DAMAGE_KEYWORDS = [
    r'\bbuilding(?:s)? (?:collapsed?|damaged?|destroyed?|cracked?)\b',
    r'\bhouse(?:s)? (?:collapsed?|damaged?|destroyed?|flooded?)\b',
    r'\bstructure(?:s)? (?:damaged?|collapsed?)\b',
    r'\bcollapse[d]?\b', r'\bdemolished?\b', r'\bdestroyed?\b',
    r'\bwall(?:s)? cracked?\b',
]

_FLOOD_KEYWORDS = [
    r'\bflood(?:ed|ing|water|s)?\b', r'\binundat(?:ed|ion)\b',
    r'\bsubmerged?\b', r'\bwater level\b', r'\boverflow\b',
    r'\bstorm surge\b',
]

_ROAD_BLOCKED_KEYWORDS = [
    r'\broad(?:s)? blocked?\b', r'\broad(?:s)? closed?\b',
    r'\btraffic blocked?\b', r'\bhighway blocked?\b',
    r'\bdebris on road\b', r'\blandslide(?:s)? blocked?\b',
    r'\bbridge (?:collapsed?|damaged?|closed?)\b',
]

_EVACUATION_KEYWORDS = [
    r'\bevacuat(?:e|ed|ion|ing)\b', r'\bfled\b', r'\bfleeing\b',
    r'\bdisplaced?\b', r'\bshelt(?:er|ering)\b', r'\breadlocated?\b',
    r'\bleaving (?:their )?home\b',
]

_TRANSPORT_DISRUPTION_KEYWORDS = [
    r'\btransport (?:disrupted?|stopped?)\b',
    r'\btrain(?:s)? (?:delayed?|cancelled?|stopped?)\b',
    r'\bflight(?:s)? (?:cancelled?|diverted?)\b',
    r'\bbus(?:es)? (?:stopped?|cancelled?)\b',
    r'\bno (?:transport|bus|train)\b',
]

# Quantity patterns near affected-people context
_AFFECTED_COUNT_PATTERN = re.compile(
    r'(\d[\d,]*)\s+(?:people|persons|families|residents|villagers|victims|individuals)\b',
    re.IGNORECASE
)


def _check_keywords(text: str, patterns: List[str]) -> List[str]:
    """Return matched phrases for a list of regex patterns."""
    matches = []
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            matches.append(m.group(0).strip())
    return list(dict.fromkeys(matches))  # deduplicate, preserve order


def _extract_affected_count(text: str) -> Optional[str]:
    """
    Extract reported number of affected people/families.
    Returns the exact phrase as reported — does NOT convert families → people.
    Returns None if no number is mentioned.
    """
    m = _AFFECTED_COUNT_PATTERN.search(text)
    if m:
        return m.group(0).strip()
    return None


def extract_emergency_info(text: str) -> Dict[str, Any]:
    """
    Extract structured emergency information from a disaster report.

    Every field is labeled as:
    - explicitly_reported  : directly stated in text
    - inferred             : derived by rule from explicit statement
    - not_found            : no relevant evidence in text

    Does NOT invent information not present in the text.
    """
    result = {
        "original_report": text,
        "rescue_request": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "medical_assistance": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "trapped_people": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "injuries_reported": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "casualties_reported": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "building_damage": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "flooding_reported": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "road_blocked": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "evacuation_mentioned": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "transport_disruption": {
            "present": False, "evidence_phrases": [], "provenance": "not_found"
        },
        "reported_affected_count": None,  # Exact phrase from text, not invented
        "urgency_score": 0,               # Simple count of urgent indicators
        "all_evidence_phrases": [],
    }

    if not text or not text.strip():
        return result

    def _fill(key, patterns):
        matches = _check_keywords(text, patterns)
        if matches:
            result[key]["present"]          = True
            result[key]["evidence_phrases"] = matches
            result[key]["provenance"]       = "explicitly_reported"
        return matches

    rescue_m   = _fill("rescue_request",      _RESCUE_KEYWORDS)
    medical_m  = _fill("medical_assistance",  _MEDICAL_KEYWORDS)
    trapped_m  = _fill("trapped_people",      _TRAPPED_KEYWORDS)
    injured_m  = _fill("injuries_reported",   _INJURED_KEYWORDS)
    casualty_m = _fill("casualties_reported", _CASUALTY_KEYWORDS)
    building_m = _fill("building_damage",     _BUILDING_DAMAGE_KEYWORDS)
    flood_m    = _fill("flooding_reported",   _FLOOD_KEYWORDS)
    road_m     = _fill("road_blocked",        _ROAD_BLOCKED_KEYWORDS)
    evac_m     = _fill("evacuation_mentioned",_EVACUATION_KEYWORDS)
    transport_m= _fill("transport_disruption",_TRANSPORT_DISRUPTION_KEYWORDS)

    # Affected count — report exactly what text says
    result["reported_affected_count"] = _extract_affected_count(text)

    # Collect all evidence phrases
    all_phrases = (rescue_m + medical_m + trapped_m + injured_m + casualty_m
                   + building_m + flood_m + road_m + evac_m + transport_m)
    result["all_evidence_phrases"] = list(dict.fromkeys(all_phrases))

    # Simple urgency score (0–10)
    urgency = 0
    if result["rescue_request"]["present"]:    urgency += 3
    if result["trapped_people"]["present"]:    urgency += 3
    if result["casualties_reported"]["present"]:urgency += 3
    if result["injuries_reported"]["present"]: urgency += 2
    if result["medical_assistance"]["present"]:urgency += 2
    if result["building_damage"]["present"]:   urgency += 1
    if result["evacuation_mentioned"]["present"]:urgency += 1
    result["urgency_score"] = min(urgency, 10)

    logger.info(
        f"Emergency extraction: urgency={result['urgency_score']}, "
        f"rescue={result['rescue_request']['present']}, "
        f"trapped={result['trapped_people']['present']}"
    )
    return result
