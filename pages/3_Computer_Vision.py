"""
Page 3: Computer Vision Results
Displays real detection results, annotated images, model metadata.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
from src.utils.theme import apply_custom_theme

st.set_page_config(
    page_title="CV Results | Disaster Intelligence",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_custom_theme()

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; flex-wrap:wrap; gap:10px;">
    <div>
        <h1 style="margin:0;">👁 Computer Vision Subsystem & Visual Intelligence</h1>
        <p style="color:var(--text-secondary); margin:0.25rem 0 0 0;">Multi-modal object bounding boxes, structural damage assessment, and video stream timeline.</p>
    </div>
    <div>
        <span class="cyber-badge badge-p1-glow">YOLOv8 + EfficientNet + ResNet</span>
    </div>
</div>
""", unsafe_allow_html=True)

def _build_counts(detections):
    counts = {}
    for d in (detections or []):
        label = d.get("label", "unknown") if isinstance(d, dict) else str(d)
        counts[label] = counts.get(label, 0) + 1
    return counts

# Check for results in session state or DB
incident_id = st.session_state.get("current_incident_id")

from src.database import repository as repo
from src.database.models import initialize_database
initialize_database()

# Option to load from history if not active
if not st.session_state.get("cv_det_result") and not st.session_state.get("video_result"):
    incidents = repo.list_incidents(limit=20)
    vis_incidents = [inc for inc in incidents if inc.get("input_type") in ("image", "video", "image_text", "video_text")]
    
    if vis_incidents:
        vis_map = {inc["incident_id"]: inc for inc in vis_incidents}
        vis_ids = [inc["incident_id"] for inc in vis_incidents]
        selected_id = st.selectbox(
            "Select Incident to inspect Computer Vision Results:",
            options=vis_ids,
            format_func=lambda x: f"{x} - {vis_map[x].get('title') or vis_map[x].get('disaster_type') or 'Incident'} ({str(vis_map[x].get('created_at', ''))[:19]})"
        )
        if selected_id:
            incident_id = selected_id
            selected_inc = vis_map[selected_id]
            va = repo.get_visual_analysis(incident_id)
            if va:
                detected_list = va.get("detected_objects", []) or []
                st.session_state["cv_det_result"] = {
                    "detected_objects": detected_list,
                    "object_counts": _build_counts(detected_list),
                    "confidence_info": va.get("confidence_info", {}) or {}
                }
                from src.utils import configuration as cfg
                ann_path = cfg.PROCESSED_DIR / f"{incident_id}_annotated.jpg"
                if ann_path.exists():
                    st.session_state["annotated_image_path"] = ann_path
                elif va.get("annotated_output_path") and Path(va["annotated_output_path"]).exists():
                    st.session_state["annotated_image_path"] = Path(va["annotated_output_path"])

                upload_img = cfg.UPLOADS_DIR / f"{incident_id}.jpg"
                upload_png = cfg.UPLOADS_DIR / f"{incident_id}.png"
                if upload_img.exists():
                    st.session_state["original_image"] = Image.open(upload_img)
                elif upload_png.exists():
                    st.session_state["original_image"] = Image.open(upload_png)
                else:
                    sample_img_path = Path(__file__).parent.parent / "data" / "sample_reports" / "case_1_critical_flood.jpg"
                    if sample_img_path.exists():
                        st.session_state["original_image"] = Image.open(sample_img_path)

                st.session_state["cv_cls_result"] = va.get("disaster_prediction")
                st.session_state["cv_dmg_result"] = va.get("damage_prediction")
    else:
        st.info("No visual incidents found in database. Submit an incident via **Analyze Incident** or run a sample below.")
        if st.button("🌊 Load Sample Flood Image Analysis"):
            sample_img_path = Path(__file__).parent.parent / "data" / "sample_reports" / "case_1_critical_flood.jpg"
            if sample_img_path.exists():
                from src.vision import object_detection, disaster_classification, damage_assessment
                pil_img = Image.open(sample_img_path)
                st.session_state["original_image"] = pil_img
                st.session_state["cv_det_result"] = object_detection.run_object_detection(pil_img)
                st.session_state["cv_cls_result"] = disaster_classification.classify_disaster(pil_img)
                st.session_state["cv_dmg_result"] = damage_assessment.assess_damage(pil_img)
                st.rerun()
        st.stop()

# ── Show results from session ──────────────────────────────────
det = st.session_state.get("cv_det_result")
cls = st.session_state.get("cv_cls_result")
dmg = st.session_state.get("cv_dmg_result")
orig_img   = st.session_state.get("original_image")
video_res  = st.session_state.get("video_result")

if not det and not video_res:
    st.info("No Computer Vision results available for the current incident.")
    st.stop()

# ── Image Results ──────────────────────────────────────────────
if orig_img and det:
    col_img, col_ann = st.columns(2)
    with col_img:
        st.subheader("Original Image")
        st.image(orig_img, use_container_width=True)
    with col_ann:
        st.subheader("Annotated Image")
        inc_id = incident_id or st.session_state.get("current_incident_id", "")
        from src.utils import configuration as cfg
        ann_path = cfg.PROCESSED_DIR / f"{inc_id}_annotated.jpg"
        if ann_path.exists():
            st.image(Image.open(ann_path), use_container_width=True)
        elif st.session_state.get("annotated_image_path") and Path(st.session_state["annotated_image_path"]).exists():
            st.image(Image.open(st.session_state["annotated_image_path"]), use_container_width=True)
        elif det and det.get("annotated_image"):
            st.image(det["annotated_image"], use_container_width=True)
        else:
            st.info("Annotated bounding box image not archived for this incident.")

    st.divider()

    # Object detection results
    st.subheader("🎯 Object Detection Results")
    st.caption("Model: YOLOv8n (Pretrained on COCO — detects on-scene people, vehicles, boats)")

    detected = det.get("detected_objects", [])
    counts   = det.get("object_counts", {}) or _build_counts(detected)

    if detected:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Object Counts")
            for obj, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                st.metric(obj.title(), cnt)

        with col_b:
            st.subheader("Detection Details")
            import pandas as pd
            df = pd.DataFrame(detected)
            if "confidence" in df.columns:
                df["confidence"] = df["confidence"].map("{:.1%}".format)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No objects detected above confidence threshold.")

    # Confidence info
    conf_info = det.get("confidence_info", {})
    if conf_info:
        with st.expander("📊 Confidence Statistics per Class"):
            import pandas as pd
            df_conf = pd.DataFrame(conf_info).T
            st.dataframe(df_conf, use_container_width=True)

    st.divider()

    # Disaster Classification
    st.subheader("🌪 Disaster Classification")
    if cls:
        status = cls.get("status")
        if status == "success":
            c1, c2 = st.columns(2)
            c1.metric("Predicted Category", cls.get("predicted_category", "—").replace("_"," ").title())
            c2.metric("Confidence", f"{cls.get('confidence', 0):.1%}")
            if cls.get("top_predictions"):
                st.write("**Top Predictions:**")
                for p in cls["top_predictions"]:
                    st.write(f"- {p['label'].replace('_',' ').title()}: {p['confidence']:.1%}")
        elif status == "not_configured":
            st.warning(f"🔧 Disaster classifier not configured.\n\n`{cls.get('configuration_required')}`")
        else:
            st.error(f"Classification error: {cls.get('error_message')}")
    else:
        st.info("No classification result available.")

    st.divider()

    # Damage Assessment
    st.subheader("🏚 Building Damage Assessment")
    if dmg:
        status = dmg.get("status")
        if status == "success":
            c1, c2 = st.columns(2)
            c1.metric("Damage Level", dmg.get("predicted_damage_level","—").replace("_"," ").title())
            c2.metric("Confidence", f"{dmg.get('confidence', 0):.1%}")
        elif status == "not_configured":
            st.warning(f"🔧 Damage assessment model not configured.\n\n`{dmg.get('configuration_required')}`")
        else:
            st.error(f"Damage assessment error: {dmg.get('error_message')}")
    else:
        st.info("No damage assessment result available.")

# ── Video Results ──────────────────────────────────────────────
if video_res:
    st.subheader("🎬 Video Analysis")
    meta = video_res.get("metadata", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FPS",         meta.get("fps", "—"))
    c2.metric("Duration",    f"{meta.get('duration_seconds','—')}s")
    c3.metric("Frames Extracted", video_res.get("total_frames_extracted", 0))
    c4.metric("Frames Analyzed",  video_res.get("frames_analyzed", 0))

    # Consolidated objects
    obj_counts = video_res.get("consolidated_objects", {})
    if obj_counts:
        st.subheader("Consolidated Detected Objects (across all frames)")
        import pandas as pd
        df_obj = pd.DataFrame(
            [{"Object": k, "Total Detections": v} for k, v in obj_counts.items()]
        ).sort_values("Total Detections", ascending=False)
        st.dataframe(df_obj, use_container_width=True, hide_index=True)

    # Disaster prediction
    dis_pred = video_res.get("consolidated_disaster_prediction")
    if dis_pred:
        st.metric("Consolidated Disaster Category", 
                  dis_pred.get("predicted_category","—").replace("_"," ").title())

    # Frame-level results
    with st.expander("📹 Frame-Level Results"):
        for fr in video_res.get("frame_results", []):
            st.write(f"**Frame {fr['frame_index']+1}** @ {fr['timestamp_seconds']}s")
            obj_det = fr.get("object_detection", {})
            if obj_det.get("object_counts"):
                st.write(f"  Objects: {obj_det['object_counts']}")
            dis_cl = fr.get("disaster_classification")
            if dis_cl and dis_cl.get("predicted_category"):
                st.write(f"  Disaster: {dis_cl['predicted_category']}")
            st.divider()



