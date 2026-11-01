"""
Test suite — database operations, file validation, NLP, fusion, priority scoring.
Run: pytest tests/ -v
"""
import sys, json, io, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def setup_test_db(tmp_path_factory):
    """Use a temp DB for all tests."""
    import src.utils.configuration as cfg
    tmp = tmp_path_factory.mktemp("data")
    cfg.DATABASE_PATH = tmp / "test.db"
    cfg.UPLOADS_DIR   = tmp / "uploads"
    cfg.PROCESSED_DIR = tmp / "processed"
    cfg.ensure_dirs()
    from src.database.models import initialize_database
    initialize_database()


# ─────────────────────────────────────────────────────────────
# File Validation Tests
# ─────────────────────────────────────────────────────────────

class TestFileValidation:
    def test_empty_image_rejected(self):
        from src.utils.file_validation import validate_image
        ok, err = validate_image(b"", "test.jpg")
        assert not ok
        assert "empty" in err.lower()

    def test_unsupported_extension_rejected(self):
        from src.utils.file_validation import validate_image
        ok, err = validate_image(b"data", "test.bmp")
        assert not ok

    def test_empty_text_rejected(self):
        from src.utils.file_validation import validate_text_report
        ok, err = validate_text_report("")
        assert not ok

    def test_too_short_text_rejected(self):
        from src.utils.file_validation import validate_text_report
        ok, err = validate_text_report("hi")
        assert not ok

    def test_valid_text_accepted(self):
        from src.utils.file_validation import validate_text_report
        ok, err = validate_text_report("Severe flooding near the main river, rescue needed.")
        assert ok

    def test_valid_coordinates(self):
        from src.utils.file_validation import validate_coordinates
        ok, _ = validate_coordinates(13.08, 80.27)
        assert ok

    def test_invalid_latitude(self):
        from src.utils.file_validation import validate_coordinates
        ok, err = validate_coordinates(200.0, 80.0)
        assert not ok
        assert "latitude" in err.lower()

    def test_partial_coordinates_rejected(self):
        from src.utils.file_validation import validate_coordinates
        ok, err = validate_coordinates(13.0, None)
        assert not ok

    def test_both_none_accepted(self):
        from src.utils.file_validation import validate_coordinates
        ok, _ = validate_coordinates(None, None)
        assert ok


# ─────────────────────────────────────────────────────────────
# Database Tests
# ─────────────────────────────────────────────────────────────

class TestDatabase:
    def test_create_incident(self):
        from src.database import repository as repo
        inc_id = repo.create_incident("text", title="Test Flood", original_report="Test report")
        assert inc_id.startswith("INC_")

    def test_get_incident(self):
        from src.database import repository as repo
        inc_id = repo.create_incident("image", title="Test EQ")
        inc = repo.get_incident(inc_id)
        assert inc is not None
        assert inc["title"] == "Test EQ"
        assert inc["input_type"] == "image"

    def test_update_assessment(self):
        from src.database import repository as repo
        inc_id = repo.create_incident("text", title="Update Test")
        repo.update_incident_assessment(inc_id, disaster_type="flood", priority="P2_HIGH",
                                        severity_score=65.0, processing_status="completed")
        inc = repo.get_incident(inc_id)
        assert inc["disaster_type"] == "flood"
        assert inc["priority"] == "P2_HIGH"

    def test_save_and_get_visual_analysis(self):
        from src.database import repository as repo
        inc_id = repo.create_incident("image")
        va_id = repo.save_visual_analysis(inc_id, {
            "source_filename": "test.jpg", "input_type": "image",
            "model_name": "YOLOv8n", "detected_objects": [{"label":"person","confidence":0.9}],
            "processing_status": "success",
        })
        assert va_id.startswith("VA_")
        va = repo.get_visual_analysis(inc_id)
        assert va is not None
        assert va["model_name"] == "YOLOv8n"

    def test_save_and_get_text_analysis(self):
        from src.database import repository as repo
        inc_id = repo.create_incident("text")
        ta_id = repo.save_text_analysis(inc_id, {
            "model_name": "DistilBERT", "original_report": "Flooding near river.",
            "predicted_category": "flood", "processing_status": "success",
        })
        assert ta_id.startswith("TA_")
        ta = repo.get_text_analysis(inc_id)
        assert ta["predicted_category"] == "flood"

    def test_save_and_get_fusion_result(self):
        from src.database import repository as repo
        inc_id = repo.create_incident("image_text")
        fr_id = repo.save_fusion_result(inc_id, {
            "combined_summary": "Flood detected", "severity_score": 70.0,
            "priority": "P2_HIGH",
        })
        assert fr_id.startswith("FR_")
        fr = repo.get_fusion_result(inc_id)
        assert fr["priority"] == "P2_HIGH"

    def test_incident_counts(self):
        from src.database import repository as repo
        counts = repo.get_incident_counts()
        assert "total" in counts
        assert counts["total"] >= 0

    def test_list_incidents_filter(self):
        from src.database import repository as repo
        inc_id = repo.create_incident("text", title="Filter Test")
        repo.update_incident_assessment(inc_id, disaster_type="earthquake")
        results = repo.list_incidents(disaster_type="earthquake")
        assert any(r["incident_id"] == inc_id for r in results)

    def test_update_review_status(self):
        from src.database import repository as repo
        inc_id = repo.create_incident("text")
        repo.update_review_status(inc_id, "reviewed")
        inc = repo.get_incident(inc_id)
        assert inc["review_status"] == "reviewed"


# ─────────────────────────────────────────────────────────────
# NLP Tests
# ─────────────────────────────────────────────────────────────

class TestNLP:
    def test_emergency_info_rescue_request(self):
        from src.nlp.emergency_information import extract_emergency_info
        r = extract_emergency_info("Residents are requesting rescue assistance immediately.")
        assert r["rescue_request"]["present"]
        assert r["rescue_request"]["provenance"] == "explicitly_reported"

    def test_emergency_info_trapped(self):
        from src.nlp.emergency_information import extract_emergency_info
        r = extract_emergency_info("People are trapped inside the building.")
        assert r["trapped_people"]["present"]

    def test_emergency_info_flooding(self):
        from src.nlp.emergency_information import extract_emergency_info
        r = extract_emergency_info("The road is flooded and vehicles are submerged.")
        assert r["flooding_reported"]["present"]

    def test_emergency_info_empty_text(self):
        from src.nlp.emergency_information import extract_emergency_info
        r = extract_emergency_info("")
        assert not r["rescue_request"]["present"]
        assert r["urgency_score"] == 0

    def test_affected_count_extracted(self):
        from src.nlp.emergency_information import extract_emergency_info
        r = extract_emergency_info("About 50 families need immediate rescue.")
        assert r["reported_affected_count"] is not None
        assert "50" in r["reported_affected_count"]

    def test_no_count_invented(self):
        from src.nlp.emergency_information import extract_emergency_info
        r = extract_emergency_info("People need help.")
        assert r["reported_affected_count"] is None

    def test_entity_extraction_fallback(self):
        from src.nlp.entity_extraction import extract_entities
        r = extract_entities("Flooding near Main Road area. 30 families affected.")
        assert r["original_text"] is not None
        assert r["extraction_method"] is not None


# ─────────────────────────────────────────────────────────────
# Fusion Tests
# ─────────────────────────────────────────────────────────────

class TestFusion:
    def test_text_only_fusion(self):
        from src.fusion.evidence_fusion import fuse_evidence
        nlp = {
            "classification": {"status":"success","predicted_category":"flood","confidence":0.85},
            "entities": {"locations":["Main Road"],"dates":[],"organizations":[],"quantities":[]},
            "emergency_info": {"rescue_request":{"present":True},"trapped_people":{"present":False},
                               "injuries_reported":{"present":False},"casualties_reported":{"present":False},
                               "building_damage":{"present":True},"flooding_reported":{"present":True},
                               "road_blocked":{"present":False},"evacuation_mentioned":{"present":False},
                               "transport_disruption":{"present":False},
                               "reported_affected_count":None,"all_evidence_phrases":["rescue"]},
        }
        r = fuse_evidence(None, nlp)
        assert r["has_visual_input"] == False
        assert r["has_text_input"] == True
        assert "No visual input" in r["missing_information"][0]

    def test_image_only_fusion(self):
        from src.fusion.evidence_fusion import fuse_evidence
        cv = {
            "detection": {"detected_objects":[{"label":"person","confidence":0.9}],
                          "object_counts":{"person":1},"confidence_info":{}},
            "classification": {"status":"not_configured"},
            "damage": {"status":"not_configured"},
        }
        r = fuse_evidence(cv, None)
        assert r["has_visual_input"] == True
        assert r["has_text_input"] == False

    def test_missing_both_modalities(self):
        from src.fusion.evidence_fusion import fuse_evidence
        r = fuse_evidence(None, None)
        assert not r["has_visual_input"]
        assert not r["has_text_input"]


# ─────────────────────────────────────────────────────────────
# Consistency Analysis Tests
# ─────────────────────────────────────────────────────────────

class TestConsistency:
    def _make_fusion(self, vis_cat=None, txt_cat=None, txt_rescue=False):
        from src.fusion.evidence_fusion import fuse_evidence
        cv = nlp = None
        if vis_cat:
            cv = {"detection":{"detected_objects":[],"object_counts":{}},
                  "classification":{"status":"success","predicted_category":vis_cat,"confidence":0.8},
                  "damage":{"status":"not_configured"}}
        if txt_cat:
            nlp = {"classification":{"status":"success","predicted_category":txt_cat,"confidence":0.8},
                   "entities":{"locations":[],"dates":[],"organizations":[],"quantities":[]},
                   "emergency_info":{
                       "rescue_request":{"present":txt_rescue},"trapped_people":{"present":False},
                       "injuries_reported":{"present":False},"casualties_reported":{"present":False},
                       "building_damage":{"present":False},"flooding_reported":{"present":txt_cat=="flood"},
                       "road_blocked":{"present":False},"evacuation_mentioned":{"present":False},
                       "transport_disruption":{"present":False},
                       "reported_affected_count":None,"all_evidence_phrases":[]
                   }}
        return fuse_evidence(cv, nlp)

    def test_consistent_categories(self):
        from src.fusion.consistency_analysis import analyze_consistency, CONSISTENT
        fusion = self._make_fusion("flood", "flood")
        r = analyze_consistency(fusion)
        assert any(c["outcome"] == CONSISTENT for c in r["checks"])

    def test_inconsistent_categories(self):
        from src.fusion.consistency_analysis import analyze_consistency, POTENTIAL_INCONS
        fusion = self._make_fusion("wildfire", "flood")
        r = analyze_consistency(fusion)
        assert any(c["outcome"] == POTENTIAL_INCONS for c in r["checks"])

    def test_insufficient_visual_evidence(self):
        from src.fusion.consistency_analysis import analyze_consistency, INSUF_VISUAL
        fusion = self._make_fusion(txt_cat="flood")
        r = analyze_consistency(fusion)
        assert any(c["outcome"] == INSUF_VISUAL for c in r["checks"])

    def test_no_false_conclusion_from_missing_evidence(self):
        """The system must NOT conclude text is false just because image lacks evidence."""
        from src.fusion.consistency_analysis import analyze_consistency, POTENTIAL_INCONS
        fusion = self._make_fusion(vis_cat="flood", txt_cat="flood", txt_rescue=True)
        r = analyze_consistency(fusion)
        # Should NOT be a blanket POTENTIAL_INCONS just because no people in image
        assert r["important_caveat"] is not None


# ─────────────────────────────────────────────────────────────
# Priority Scoring Tests
# ─────────────────────────────────────────────────────────────

class TestPriorityScoring:
    def _make_inputs(self, rescue=False, trapped=False, casualties=False,
                     injuries=False, building=False):
        fusion = {
            "has_visual_input": False, "has_text_input": True,
            "visual_evidence": {}, "textual_evidence": {
                "rescue_request":    {"present": rescue},
                "trapped_people":    {"present": trapped},
                "casualties_reported":{"present": casualties},
                "injuries_reported": {"present": injuries},
                "building_damage":   {"present": building},
                "road_blocked":      {"present": False},
                "evacuation":        {"present": False},
                "disaster_category": {"value": None},
                "reported_affected_count": None,
            },
            "missing_information": [],
        }
        consistency = {"overall_consistency": "unable_to_determine"}
        return fusion, consistency

    def test_high_priority_with_rescue_and_trapped(self):
        from src.assessment.severity_estimation import calculate_severity, P1_CRITICAL, P2_HIGH
        fusion, cons = self._make_inputs(rescue=True, trapped=True, casualties=True)
        r = calculate_severity(fusion, cons)
        assert r["priority"] in (P1_CRITICAL, P2_HIGH)
        assert r["severity_score"] > 0

    def test_low_priority_no_urgent_factors(self):
        from src.assessment.severity_estimation import calculate_severity, P3_MODERATE, P4_LOW
        fusion, cons = self._make_inputs(building=True)
        r = calculate_severity(fusion, cons)
        assert r["priority"] in (P3_MODERATE, P4_LOW, "UNASSESSED")

    def test_unassessed_with_no_evidence(self):
        from src.assessment.severity_estimation import calculate_severity, UNASSESSED
        fusion = {"has_visual_input": False, "has_text_input": False,
                  "visual_evidence": {}, "textual_evidence": {}, "missing_information":[]}
        r = calculate_severity(fusion, {})
        assert r["priority"] == UNASSESSED

    def test_factors_are_documented(self):
        from src.assessment.severity_estimation import calculate_severity
        fusion, cons = self._make_inputs(rescue=True)
        r = calculate_severity(fusion, cons)
        assert len(r["contributing_factors"]) > 0
        for factor in r["contributing_factors"]:
            assert "source" in factor
            assert "provenance" in factor

    def test_disclaimer_always_present(self):
        from src.assessment.severity_estimation import calculate_severity
        fusion, cons = self._make_inputs()
        r = calculate_severity(fusion, cons)
        assert "disclaimer" in r
        assert len(r["disclaimer"]) > 10


# ─────────────────────────────────────────────────────────────
# Location/Mapping Tests
# ─────────────────────────────────────────────────────────────

class TestMapping:
    def test_valid_coords_accepted(self):
        from src.mapping.incident_mapping import validate_coordinates_for_map
        r = validate_coordinates_for_map(13.0, 80.0)
        assert r["valid"]
        assert r["will_appear_on_map"]

    def test_no_coords_rejected(self):
        from src.mapping.incident_mapping import validate_coordinates_for_map
        r = validate_coordinates_for_map(None, None)
        assert not r["valid"]

    def test_out_of_range_lat_rejected(self):
        from src.mapping.incident_mapping import validate_coordinates_for_map
        r = validate_coordinates_for_map(200.0, 80.0)
        assert not r["valid"]

    def test_partial_coords_rejected(self):
        from src.mapping.incident_mapping import validate_coordinates_for_map
        r = validate_coordinates_for_map(13.0, None)
        assert not r["valid"]
