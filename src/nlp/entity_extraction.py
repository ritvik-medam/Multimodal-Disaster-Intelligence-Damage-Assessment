"""
Named Entity Recognition Module
Uses spaCy en_core_web_sm (or trf) to extract locations, dates, orgs, quantities.
Falls back to basic rule extraction if spaCy model is unavailable.
"""
import re
from typing import Dict, Any, List, Optional
from src.utils.logging_config import get_logger

logger = get_logger("nlp.entity_extraction")

_nlp = None
_nlp_error = None


def _load_spacy():
    global _nlp, _nlp_error
    if _nlp is not None:
        return True, None
    if _nlp_error is not None:
        return False, _nlp_error
    try:
        import spacy
        # Try transformer model first, fallback to small
        for model_name in ["en_core_web_trf", "en_core_web_sm", "en_core_web_md"]:
            try:
                _nlp = spacy.load(model_name)
                logger.info(f"spaCy model loaded: {model_name}")
                return True, None
            except OSError:
                continue
        _nlp_error = (
            "No spaCy model found. Install with:\n"
            "  python -m spacy download en_core_web_sm"
        )
        return False, _nlp_error
    except ImportError:
        _nlp_error = "spaCy not installed. Run: pip install spacy"
        return False, _nlp_error


# ── Rule-based fallback patterns ────────────────────────────────

_LOCATION_PATTERNS = [
    r'\b(?:near|at|in|around|close to|towards?)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+(?:Road|Street|River|Bridge|Area|District|City|Town|Village|Zone|Highway|Avenue))',
    r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\s+(?:district|village|city|town|area|region)',
]

_QUANTITY_PATTERNS = [
    r'(\d+[\d,]*)\s+(?:people|persons|families|residents|victims|casualties|injured|dead|missing|buildings?|houses?|vehicles?)',
    r'(?:approximately|around|about|nearly)\s+(\d+[\d,]*)',
]

_DATE_PATTERNS = [
    r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:,\s+\d{4})?)\b',
    r'\b(yesterday|today|this morning|last night|this evening)\b',
]


def _rule_based_entities(text: str) -> List[Dict]:
    """Fallback rule-based entity extraction."""
    entities = []

    for pattern in _LOCATION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "text": m.group(1).strip(),
                "label": "LOC",
                "start": m.start(),
                "end": m.end(),
                "source": "rule_based",
            })

    for pattern in _QUANTITY_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "text": m.group(0).strip(),
                "label": "QUANTITY",
                "start": m.start(),
                "end": m.end(),
                "source": "rule_based",
            })

    for pattern in _DATE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "text": m.group(1).strip(),
                "label": "DATE",
                "start": m.start(),
                "end": m.end(),
                "source": "rule_based",
            })

    # Deduplicate by text
    seen  = set()
    dedup = []
    for e in entities:
        key = (e["text"].lower(), e["label"])
        if key not in seen:
            seen.add(key)
            dedup.append(e)
    return dedup


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extract named entities from a text report.

    Uses spaCy when available, falls back to rule-based extraction.
    Preserves original text so users can verify extracted entities.
    """
    result = {
        "original_text": text,
        "entities": [],
        "locations": [],
        "dates": [],
        "organizations": [],
        "quantities": [],
        "extraction_method": None,
        "error_message": None,
    }

    if not text or not text.strip():
        result["error_message"] = "Empty text provided."
        return result

    ok, err = _load_spacy()

    if ok:
        try:
            doc = _nlp(text)
            entities = []
            for ent in doc.ents:
                entity = {
                    "text": ent.text.strip(),
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy",
                }
                entities.append(entity)

            result["entities"] = entities
            result["locations"]      = [e["text"] for e in entities if e["label"] in ("GPE", "LOC", "FAC")]
            result["dates"]          = [e["text"] for e in entities if e["label"] in ("DATE", "TIME")]
            result["organizations"]  = [e["text"] for e in entities if e["label"] == "ORG"]
            result["quantities"]     = [e["text"] for e in entities if e["label"] in ("CARDINAL", "QUANTITY")]
            result["extraction_method"] = "spacy"
            logger.info(f"spaCy NER: {len(entities)} entities found")

        except Exception as e:
            logger.error(f"spaCy NER error: {e}")
            # Fallback to rules
            entities = _rule_based_entities(text)
            result["entities"]  = entities
            result["locations"] = [e["text"] for e in entities if e["label"] == "LOC"]
            result["dates"]     = [e["text"] for e in entities if e["label"] == "DATE"]
            result["quantities"]= [e["text"] for e in entities if e["label"] == "QUANTITY"]
            result["extraction_method"] = "rule_based_fallback"
            result["error_message"]     = f"spaCy failed, used rule-based fallback: {e}"

    else:
        # Rule-based fallback
        entities = _rule_based_entities(text)
        result["entities"]  = entities
        result["locations"] = [e["text"] for e in entities if e["label"] == "LOC"]
        result["dates"]     = [e["text"] for e in entities if e["label"] == "DATE"]
        result["quantities"]= [e["text"] for e in entities if e["label"] == "QUANTITY"]
        result["extraction_method"] = "rule_based"
        result["error_message"]     = f"spaCy unavailable ({err}), using rule-based extraction."

    return result
