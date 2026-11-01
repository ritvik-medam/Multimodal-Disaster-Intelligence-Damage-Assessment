"""
Page 2: Analyze Incident
Full end-to-end pipeline: image/video/text → CV → NLP → Fusion → Priority → DB
"""
import sys, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image

from src.utils.file_validation import validate_image, validate_video, validate_text_report, validate_coordinates, save_upload
from src.utils.logging_config import get_logger
from src.database.models import initialize_database
from src.database import repository as repo
from src.utils.theme import apply_custom_theme

logger = get_logger("pages.analyze")
st.set_page_config(
    page_title="Analyze Incident | Disaster Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
initialize_database()
apply_custom_theme()

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; flex-wrap:wrap; gap:10px;">
    <div>
        <h1 style="margin:0;">🔍 Incident Ingestion & Multimodal Triage</h1>
        <p style="color:var(--text-secondary); margin:0.25rem 0 0 0;">Ingest aerial imagery, drone video feeds, or ground reports for automated cross-modal analysis.</p>
    </div>
    <div>
        <span class="cyber-badge badge-p1-glow">AI Triage Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick Demo Loader ──────────────────────────────────────────
sample_dir = Path(__file__).parent.parent / "data" / "sample_reports"
if sample_dir.exists():
    with st.expander("⚡ Quick Demo Scenarios (Pre-packaged test incidents)", expanded=False):
        d_col1, d_col2, d_col3, d_col4, d_col5, d_col6 = st.columns(6)
        if d_col1.button("🌊 Flash Flood", use_container_width=True):
            st.session_state["demo_title"] = "Flash Flood Emergency — Riverside"
            st.session_state["demo_loc"] = "Riverside Marina, Sector 4"
            st.session_state["demo_lat"] = "29.9511"
            st.session_state["demo_lon"] = "-90.0715"
            st.session_state["demo_text"] = "Flash flood breached coastal embankment at Riverside. Over 40 families trapped on upper floors. Severe water currents preventing small boat rescue. Urgent helicopter airlift and medical evacuation requested."
            st.session_state["demo_img_name"] = "case_1_critical_flood.jpg"
            st.session_state["load_demo_video"] = False
            st.rerun()
        if d_col2.button("🔥 Wildfire Outbreak", use_container_width=True):
            st.session_state["demo_title"] = "Highland Ridge Forest Fire"
            st.session_state["demo_loc"] = "Highland Ridge Forest Boundary"
            st.session_state["demo_lat"] = "34.0522"
            st.session_state["demo_lon"] = "-118.2437"
            st.session_state["demo_text"] = "Fast-moving brushfire driven by 55mph winds crossed highway 12 into Highland Ridge. Five homes completely destroyed, twelve others actively burning. Power substation ignited. Zero visibility due to dense smoke."
            st.session_state["demo_img_name"] = "case_2_wildfire_outbreak.jpg"
            st.session_state["load_demo_video"] = False
            st.rerun()
        if d_col3.button("🏚 Earthquake Rupture", use_container_width=True):
            st.session_state["demo_title"] = "Central Metro M6.8 Seismic Collapse"
            st.session_state["demo_loc"] = "Central Metro District"
            st.session_state["demo_lat"] = "37.7749"
            st.session_state["demo_lon"] = "-122.4194"
            st.session_state["demo_text"] = "Magnitude 6.8 seismic rupture collapsed 3 commercial buildings and fractured elevated highway overpass in Central Metro. Acoustic search detected taps under rubble. Heavy extrication gear needed immediately."
            st.session_state["demo_img_name"] = "case_3_earthquake_collapse.jpg"
            st.session_state["load_demo_video"] = False
            st.rerun()
        if d_col4.button("🌀 Cyclone Surge", use_container_width=True):
            st.session_state["demo_title"] = "Cyclone Storm Surge & Flooding"
            st.session_state["demo_loc"] = "Coastal Bay General Hospital"
            st.session_state["demo_lat"] = "25.7617"
            st.session_state["demo_lon"] = "-80.1918"
            st.session_state["demo_text"] = "Category 4 tropical cyclone landfall generated 14-foot storm surge. Hospital emergency wing flooded and lost grid power. Backup generator running low on diesel. 80 critical patients require priority triage."
            st.session_state["demo_img_name"] = "case_4_cyclone_surge.jpg"
            st.session_state["load_demo_video"] = False
            st.rerun()
        if d_col5.button("🟢 Normal Monitoring", use_container_width=True):
            st.session_state["demo_title"] = "Reservoir Routine Inspection"
            st.session_state["demo_loc"] = "West Valley Reservoir Dam"
            st.session_state["demo_lat"] = "36.1699"
            st.session_state["demo_lon"] = "-115.1398"
            st.session_state["demo_text"] = "Routine aerial monitoring of West Valley water reservoir. Water levels within standard seasonal thresholds. All spillways operating nominally with zero structural anomalies detected."
            st.session_state["demo_img_name"] = "case_5_normal_monitoring.jpg"
            st.session_state["load_demo_video"] = False
            st.rerun()
        if d_col6.button("🚁 Drone Video Feed", use_container_width=True):
            st.session_state["demo_title"] = "Aerial Drone River Patrol"
            st.session_state["demo_loc"] = "Riverside River Crossing"
            st.session_state["demo_lat"] = "29.9515"
            st.session_state["demo_lon"] = "-90.0720"
            st.session_state["demo_text"] = "Aerial drone surveillance over flooded sector. Rescue boat deployed with emergency personnel. Water currents high."
            st.session_state["demo_img_name"] = None
            st.session_state["load_demo_video"] = True
            st.rerun()

# ── Input Form ────────────────────────────────────────────────
with st.form("incident_form", clear_on_submit=False):
    st.subheader("1. Incident Details")
    title    = st.text_input("Incident Title (optional)", value=st.session_state.get("demo_title", ""), placeholder="e.g. Flood near River Road")
    location = st.text_input("Reported Location (optional)", value=st.session_state.get("demo_loc", ""), placeholder="e.g. Main Road, Chennai District")

    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat_str = st.text_input("Latitude (optional)", value=st.session_state.get("demo_lat", ""), placeholder="e.g. 13.0827")
    with col_lon:
        lon_str = st.text_input("Longitude (optional)", value=st.session_state.get("demo_lon", ""), placeholder="e.g. 80.2707")

    st.subheader("2. Visual Input (optional)")
    image_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg","jpeg","png"])
    video_file = st.file_uploader("Upload Video (MP4/AVI/MOV)", type=["mp4","avi","mov"])

    st.subheader("3. Text Report (optional)")
    report_text = st.text_area(
        "Emergency Report Text",
        value=st.session_state.get("demo_text", ""),
        height=150,
        placeholder="Example: Severe flooding has affected houses near the river. Residents are requesting rescue assistance."
    )

    with st.expander("⚙️ Advanced Video Sampling Settings (Optional)", expanded=False):
        sample_interval = st.slider("Frame sampling interval (seconds)", 1.0, 10.0, 2.0, 0.5)
        max_frames      = st.slider("Max frames to analyze", 5, 50, 20, 5)

    submitted = st.form_submit_button("🚀 Analyze Incident", use_container_width=True, type="primary")

# ── Processing ────────────────────────────────────────────────
if submitted:
    # Parse coordinates
    lat, lon = None, None
    try:
        if lat_str.strip(): lat = float(lat_str.strip())
        if lon_str.strip(): lon = float(lon_str.strip())
    except ValueError:
        st.error("Latitude and longitude must be numeric values.")
        st.stop()

    # Validate coordinates
    coord_valid, coord_err = validate_coordinates(lat, lon)
    if not coord_valid:
        st.error(f"Coordinate error: {coord_err}")
        st.stop()

    # Determine input modality
    has_image  = image_file is not None
    has_video  = video_file is not None
    
    # Auto-load demo video if requested
    if not has_video and st.session_state.get("load_demo_video"):
        demo_vid_path = Path(__file__).parent.parent / "data" / "sample_reports" / "drone_flood_surveillance.mp4"
        if demo_vid_path.exists():
            class DemoVideoFile:
                name = "drone_flood_surveillance.mp4"
                def read(self):
                    with open(demo_vid_path, "rb") as f:
                        return f.read()
            video_file = DemoVideoFile()
            has_video = True
            st.session_state["load_demo_video"] = False

    # Auto-load demo image if requested and no manual upload provided
    if not has_image and not has_video and st.session_state.get("demo_img_name"):
        demo_img_name = st.session_state["demo_img_name"]
        demo_img_path = Path(__file__).parent.parent / "data" / "sample_reports" / demo_img_name
        if demo_img_path.exists():
            class DemoImageFile:
                def __init__(self, fname, fpath):
                    self.name = fname
                    self._fpath = fpath
                def read(self):
                    with open(self._fpath, "rb") as f:
                        return f.read()
            image_file = DemoImageFile(demo_img_name, demo_img_path)
            has_image = True
            
    has_text   = bool(report_text and report_text.strip())

    if not has_image and not has_video and not has_text:
        st.error("Please provide at least one input: an image, video, or text report.")
        st.stop()

    if has_image and has_video:
        st.error("Please upload either an image OR a video, not both.")
        st.stop()

    # Determine input type label
    if has_image and has_text:   input_type = "image_text"
    elif has_video and has_text: input_type = "video_text"
    elif has_image:              input_type = "image"
    elif has_video:              input_type = "video"
    else:                        input_type = "text"

    # ── Create Incident Record ─────────────────────────────────
    incident_id = repo.create_incident(
        input_type=input_type, title=title,
        original_report=report_text if has_text else "",
        reported_location=location, latitude=lat, longitude=lon,
    )
    st.session_state["current_incident_id"] = incident_id
    st.success(f"Incident created: `{incident_id}`")

    progress = st.progress(0, text="Starting analysis...")
    status_box = st.container()

    cv_result_stored  = None
    nlp_result_stored = None
    errors = []

    # ── STEP 1: Computer Vision ────────────────────────────────
    if has_image or has_video:
        progress.progress(15, text="Running Computer Vision...")
        with status_box:
            st.info("👁 Processing visual input...")

        try:
            from src.vision import object_detection, disaster_classification, damage_assessment, video_processing

            if has_image:
                img_bytes = image_file.read()
                valid, err = validate_image(img_bytes, image_file.name)
                if not valid:
                    st.error(f"Image validation failed: {err}")
                    errors.append(f"Image validation: {err}")
                else:
                    saved_path = save_upload(img_bytes, image_file.name, "images")
                    pil_img    = Image.open(io.BytesIO(img_bytes))

                    det_res = object_detection.run_object_detection(pil_img)
                    cls_res = disaster_classification.classify_disaster(pil_img)
                    dmg_res = damage_assessment.assess_damage(pil_img)

                    cv_result_stored = {
                        "source_filename": image_file.name,
                        "input_type": "image",
                        "model_name": det_res.get("model_name"),
                        "model_version": det_res.get("model_version"),
                        "detected_objects": det_res.get("detected_objects"),
                        "confidence_info": det_res.get("confidence_info"),
                        "disaster_prediction": cls_res if cls_res["status"] == "success" else {"status": cls_res["status"]},
                        "damage_prediction": dmg_res if dmg_res["status"] == "success" else {"status": dmg_res["status"]},
                        "processing_status": "success",
                    }

                    # Save annotated image if available
                    ann_img = det_res.get("annotated_image")
                    if ann_img:
                        from src.utils import configuration as cfg
                        ann_path = cfg.PROCESSED_DIR / f"{incident_id}_annotated.jpg"
                        ann_img.save(str(ann_path))
                        cv_result_stored["annotated_output_path"] = str(ann_path)

                    st.session_state["cv_det_result"]  = det_res
                    st.session_state["cv_cls_result"]  = cls_res
                    st.session_state["cv_dmg_result"]  = dmg_res
                    st.session_state["original_image"] = pil_img

            elif has_video:
                vid_bytes = video_file.read()
                valid, err = validate_video(vid_bytes, video_file.name)
                if not valid:
                    st.error(f"Video validation failed: {err}")
                    errors.append(f"Video validation: {err}")
                else:
                    vid_result = video_processing.analyze_video(
                        vid_bytes, video_file.name,
                        sample_every_n_seconds=sample_interval,
                        max_frames=max_frames
                    )
                    cv_result_stored = {
                        "source_filename": video_file.name,
                        "input_type": "video",
                        "model_name": "YOLOv8n",
                        "detected_objects": list(vid_result.get("consolidated_objects", {}).keys()),
                        "confidence_info": {},
                        "frame_count": vid_result.get("metadata", {}).get("total_frames"),
                        "frames_analyzed": vid_result.get("frames_analyzed"),
                        "disaster_prediction": vid_result.get("consolidated_disaster_prediction"),
                        "processing_status": vid_result.get("processing_status", "success"),
                        "error_message": vid_result.get("error_message"),
                    }
                    st.session_state["video_result"] = vid_result

            if cv_result_stored:
                repo.save_visual_analysis(incident_id, cv_result_stored)
                with status_box:
                    st.success("✅ Computer Vision analysis complete")

        except Exception as e:
            err_msg = f"Computer Vision error: {e}"
            errors.append(err_msg)
            with status_box:
                st.warning(f"⚠️ {err_msg}")
            logger.error(err_msg, exc_info=True)

    # ── STEP 2: NLP ────────────────────────────────────────────
    if has_text:
        progress.progress(45, text="Running NLP analysis...")
        with status_box:
            st.info("📝 Analyzing text report...")

        try:
            from src.nlp import text_classification, entity_extraction, emergency_information

            valid_text, text_err = validate_text_report(report_text)
            if not valid_text:
                st.error(f"Text validation: {text_err}")
                errors.append(f"Text validation: {text_err}")
            else:
                cls_res  = text_classification.classify_text(report_text)
                ent_res  = entity_extraction.extract_entities(report_text)
                emer_res = emergency_information.extract_emergency_info(report_text)

                # Best location: from NER or user-provided
                best_location = (ent_res.get("locations") or [None])[0] or location or None

                nlp_result_stored = {
                    "model_name": cls_res.get("model_name"),
                    "original_report": report_text,
                    "predicted_category": cls_res.get("predicted_category"),
                    "classification_conf": cls_res.get("confidence"),
                    "extracted_entities": ent_res.get("entities"),
                    "extracted_location": best_location,
                    "reported_damage": [
                        k for k, v in {
                            "building_damage": emer_res.get("building_damage", {}).get("present"),
                            "flooding": emer_res.get("flooding_reported", {}).get("present"),
                            "road_blocked": emer_res.get("road_blocked", {}).get("present"),
                        }.items() if v
                    ],
                    "assistance_indicators": {
                        "rescue_request":    emer_res.get("rescue_request", {}).get("present", False),
                        "medical_assistance":emer_res.get("medical_assistance", {}).get("present", False),
                        "trapped_people":    emer_res.get("trapped_people", {}).get("present", False),
                        "evacuation":        emer_res.get("evacuation_mentioned", {}).get("present", False),
                    },
                    "evidence_phrases": emer_res.get("all_evidence_phrases"),
                    "processing_status": "success",
                }
                repo.save_text_analysis(incident_id, nlp_result_stored)

                st.session_state["nlp_cls_result"]  = cls_res
                st.session_state["nlp_ent_result"]  = ent_res
                st.session_state["nlp_emer_result"] = emer_res

                with status_box:
                    st.success("✅ NLP analysis complete")

        except Exception as e:
            err_msg = f"NLP error: {e}"
            errors.append(err_msg)
            with status_box:
                st.warning(f"⚠️ {err_msg}")
            logger.error(err_msg, exc_info=True)

    # ── STEP 3: Fusion ─────────────────────────────────────────
    progress.progress(70, text="Running multimodal fusion...")
    try:
        from src.fusion import evidence_fusion, consistency_analysis
        from src.assessment import severity_estimation

        # Build structured CV/NLP for fusion
        cv_for_fusion  = None
        nlp_for_fusion = None

        if cv_result_stored and cv_result_stored.get("processing_status") == "success":
            det = st.session_state.get("cv_det_result", {})
            cls = st.session_state.get("cv_cls_result", {})
            dmg = st.session_state.get("cv_dmg_result", {})
            cv_for_fusion = {"detection": det, "classification": cls, "damage": dmg}

        if nlp_result_stored and nlp_result_stored.get("processing_status") == "success":
            nlp_for_fusion = {
                "classification": st.session_state.get("nlp_cls_result", {}),
                "entities":       st.session_state.get("nlp_ent_result", {}),
                "emergency_info": st.session_state.get("nlp_emer_result", {}),
            }

        fusion_res      = evidence_fusion.fuse_evidence(cv_for_fusion, nlp_for_fusion)
        consistency_res = consistency_analysis.analyze_consistency(fusion_res)
        severity_res    = severity_estimation.calculate_severity(fusion_res, consistency_res)
        summary_text    = evidence_fusion.build_incident_summary(fusion_res, severity_res)

        fusion_db_data = {
            "visual_evidence":      fusion_res.get("visual_evidence"),
            "textual_evidence":     fusion_res.get("textual_evidence"),
            "combined_summary":     summary_text,
            "consistency_findings": consistency_res,
            "severity_score":       severity_res.get("severity_score"),
            "priority":             severity_res.get("priority"),
            "scoring_explanation":  severity_res.get("contributing_factors"),
            "uncertainty_notes":    "; ".join(severity_res.get("uncertainty_notes", [])),
            "review_recommendation":"Human review required",
            "missing_information":  fusion_res.get("missing_information"),
        }
        repo.save_fusion_result(incident_id, fusion_db_data)

        # Update main incident record
        disaster_type = (
            (fusion_res.get("visual_evidence") or {}).get("disaster_category", {}).get("value")
            or (fusion_res.get("textual_evidence") or {}).get("disaster_category", {}).get("value")
        )
        repo.update_incident_assessment(
            incident_id,
            disaster_type=disaster_type,
            severity_score=severity_res.get("severity_score"),
            priority=severity_res.get("priority"),
            consistency_status=consistency_res.get("overall_consistency"),
            processing_status="completed",
        )

        st.session_state["fusion_result"]      = fusion_res
        st.session_state["consistency_result"] = consistency_res
        st.session_state["severity_result"]    = severity_res

        progress.progress(100, text="Analysis complete!")
        with status_box:
            st.success("✅ Multimodal fusion and assessment complete")

    except Exception as e:
        err_msg = f"Fusion error: {e}"
        errors.append(err_msg)
        with status_box:
            st.error(f"❌ {err_msg}")
        repo.update_incident_assessment(incident_id, processing_status="partial_failure")
        logger.error(err_msg, exc_info=True)

    # ── Results Summary ────────────────────────────────────────
    st.divider()
    st.subheader("📊 Analysis Summary")

    sev = st.session_state.get("severity_result", {})
    priority = sev.get("priority", "UNASSESSED")
    score    = sev.get("severity_score", 0)

    priority_colors = {
        "P1_CRITICAL": "🔴", "P2_HIGH": "🟠",
        "P3_MODERATE": "🔵", "P4_LOW": "🟢", "UNASSESSED": "⚫"
    }
    icon = priority_colors.get(priority, "⚫")

    col1, col2, col3 = st.columns(3)
    col1.metric("Priority", f"{icon} {priority}")
    col2.metric("Severity Score", f"{score}/100")
    col3.metric("Input Type", input_type.replace("_"," ").title())

    if errors:
        with st.expander(f"⚠️ {len(errors)} warning(s) during analysis"):
            for e in errors:
                st.warning(e)

    st.success(f"✅ Incident `{incident_id}` saved. Navigate to the pages below to view detailed results.")

    # ── SITREP Download ────────────────────────────────────────
    sitrep_content = f"""# SITUATION REPORT (SITREP) — EMERGENCY INCIDENT {incident_id}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
Title: {title or 'Unspecified'}
Location: {location or 'Unknown'} (Lat: {lat or 'N/A'}, Lon: {lon or 'N/A'})
Input Modality: {input_type.replace('_', ' ').title()}

## 1. ASSESSMENT SUMMARY
- Priority Level: {priority}
- Severity Score: {score}/100
- Consistency Finding: {st.session_state.get('consistency_result', {}).get('overall_consistency', 'N/A')}
- Primary Identified Disaster: {disaster_type or 'Undetermined'}

## 2. CONTRIBUTING TRIAGE FACTORS
"""
    for fac in sev.get("contributing_factors", []):
        sitrep_content += f"- {fac.get('factor')}: +{fac.get('points')} pts ({fac.get('description', '')})\n"

    sitrep_content += f"""
## 3. REPORTED TEXT & EMERGENCY FINDINGS
{report_text or 'No textual report provided.'}

---
*Notice: This is an AI-assisted operational decision support report. Field deployment requires trained on-scene officer validation.*
"""
    st.download_button(
        label="📥 Download Official SITREP Report (.md)",
        data=sitrep_content,
        file_name=f"SITREP_{incident_id}.md",
        mime="text/markdown",
        use_container_width=True
    )

    # Navigation buttons
    c1, c2, c3, c4 = st.columns(4)
    c1.page_link("pages/3_Computer_Vision.py",     label="CV Results",         icon="👁")
    c2.page_link("pages/4_NLP_Analysis.py",        label="NLP Results",        icon="📝")
    c3.page_link("pages/5_Multimodal_Assessment.py",label="Multimodal Triage", icon="🔗")
    c4.page_link("pages/7_Incident_History.py",    label="Audit History",      icon="📋")
