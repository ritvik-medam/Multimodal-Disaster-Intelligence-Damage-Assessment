"""
Page 4: NLP Analysis Results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.utils.theme import apply_custom_theme

st.set_page_config(
    page_title="NLP Analysis | Disaster Intelligence",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_custom_theme()

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; flex-wrap:wrap; gap:10px;">
    <div>
        <h1 style="margin:0;">📝 NLP Emergency Semantics & Entity Extraction</h1>
        <p style="color:var(--text-secondary); margin:0.25rem 0 0 0;">DistilBERT intent classification, regex rule parsing, and geolocational named-entity extraction.</p>
    </div>
    <div>
        <span class="cyber-badge badge-p3-glow">DistilBERT Language Model</span>
    </div>
</div>
""", unsafe_allow_html=True)

from src.database import repository as repo
from src.database.models import initialize_database

initialize_database()

# Check for results in session state or DB
incident_id = st.session_state.get("current_incident_id")

cls  = st.session_state.get("nlp_cls_result")
ent  = st.session_state.get("nlp_ent_result")
emer = st.session_state.get("nlp_emer_result")

# If session state is empty, auto-fallback to last incident in database
if not cls and not emer:
    incidents = repo.list_incidents(limit=10)
    text_incidents = [inc for inc in incidents if inc.get("original_report")]
    
    if text_incidents:
        txt_map = {inc["incident_id"]: inc for inc in text_incidents}
        txt_ids = [inc["incident_id"] for inc in text_incidents]
        selected_id = st.selectbox(
            "Select Incident to view NLP Analysis:",
            options=txt_ids,
            format_func=lambda x: f"{x} - {txt_map[x].get('title') or txt_map[x].get('disaster_type') or 'Report'} ({str(txt_map[x].get('created_at', ''))[:19]})"
        )
        if selected_id:
            incident_id = selected_id
            selected_inc = txt_map[selected_id]
            ta = repo.get_text_analysis(incident_id)
            if ta:
                cls = {
                    "status": "success",
                    "predicted_category": ta.get("predicted_category"),
                    "confidence": ta.get("classification_conf") or 0.85,
                    "top_predictions": [{"label": ta.get("predicted_category") or "flood", "confidence": ta.get("classification_conf") or 0.85}]
                }
                ent = {"entities": ta.get("extracted_entities", []), "locations": [ta.get("extracted_location")] if ta.get("extracted_location") else []}
                emer = {
                    "original_report": ta.get("original_report", selected_inc.get("original_report", "")),
                    "rescue_request": {"present": ta.get("assistance_indicators", {}).get("rescue_request", False)},
                    "medical_assistance": {"present": ta.get("assistance_indicators", {}).get("medical_assistance", False)},
                    "trapped_people": {"present": ta.get("assistance_indicators", {}).get("trapped_people", False)},
                    "evacuation": {"present": ta.get("assistance_indicators", {}).get("evacuation", False)},
                    "building_damage": {"present": "building_damage" in ta.get("reported_damage", [])},
                    "flooding_reported": {"present": "flooding" in ta.get("reported_damage", [])},
                    "road_blocked": {"present": "road_blocked" in ta.get("reported_damage", [])},
                    "evidence_phrases": ta.get("evidence_phrases", [])
                }
            elif selected_inc.get("original_report"):
                from src.nlp import text_classification, entity_extraction, emergency_information
                txt = selected_inc["original_report"]
                cls = text_classification.classify_text(txt)
                ent = entity_extraction.extract_entities(txt)
                emer = emergency_information.extract_emergency_info(txt)
    else:
        st.info("No text incidents found in database. Submit an incident via **Analyze Incident** or test a sample below.")
        if st.button("📝 Load Sample Flash Flood Text Analysis"):
            sample_txt = "Flash flood breached coastal embankment at Riverside Marina. Over 40 families trapped on upper floors. Severe water currents preventing small boat rescue. Urgent helicopter airlift and medical evacuation requested."
            from src.nlp import text_classification, entity_extraction, emergency_information
            st.session_state["nlp_cls_result"] = text_classification.classify_text(sample_txt)
            st.session_state["nlp_ent_result"] = entity_extraction.extract_entities(sample_txt)
            st.session_state["nlp_emer_result"] = emergency_information.extract_emergency_info(sample_txt)
            st.rerun()
        st.stop()

# ── Original Report ────────────────────────────────────────────
if emer:
    with st.expander("📄 Original Report Text", expanded=True):
        st.write(emer.get("original_report",""))

st.divider()

# ── Text Classification ────────────────────────────────────────
st.subheader("🏷 Disaster Text Classification")
if cls:
    status = cls.get("status")
    if status == "success":
        c1, c2 = st.columns(2)
        c1.metric("Predicted Category", cls.get("predicted_category","—").replace("_"," ").title())
        c2.metric("Confidence", f"{cls.get('confidence',0):.1%}")
        if cls.get("top_predictions"):
            st.write("**Top Predictions:**")
            for p in cls["top_predictions"]:
                st.progress(p["confidence"], text=f"{p['label'].replace('_',' ').title()} — {p['confidence']:.1%}")
    elif status == "not_configured":
        st.warning(f"🔧 Text classifier not configured.\n\n`{cls.get('configuration_required')}`")
    else:
        st.error(f"Classification error: {cls.get('error_message')}")

st.divider()

# ── Named Entity Recognition ────────────────────────────────────
st.subheader("🔍 Named Entity Recognition")
if ent:
    col_l, col_d, col_q, col_o = st.columns(4)
    with col_l:
        st.write("**📍 Locations**")
        locs = ent.get("locations", [])
        st.write(", ".join(locs) if locs else "*None found*")
    with col_d:
        st.write("**📅 Dates/Times**")
        dates = ent.get("dates", [])
        st.write(", ".join(dates) if dates else "*None found*")
    with col_q:
        st.write("**🔢 Quantities**")
        quantities = ent.get("quantities", [])
        st.write(", ".join(quantities) if quantities else "*None found*")
    with col_o:
        st.write("**🏛 Organizations**")
        orgs = ent.get("organizations", [])
        st.write(", ".join(orgs) if orgs else "*None found*")

    if ent.get("entities"):
        with st.expander("All Extracted Entities"):
            import pandas as pd
            df = pd.DataFrame(ent["entities"])
            st.dataframe(df[["text","label","source"]] if "source" in df.columns else df[["text","label"]],
                         use_container_width=True, hide_index=True)

st.divider()

# ── Emergency Information ──────────────────────────────────────
st.subheader("🚨 Emergency Information Extraction")
if emer:
    urgency = emer.get("urgency_score", 0)
    st.metric("Urgency Score", f"{urgency}/10")

    FIELDS = {
        "rescue_request":     "🆘 Rescue Request",
        "medical_assistance": "🏥 Medical Assistance",
        "trapped_people":     "⚠️ Trapped People",
        "injuries_reported":  "🩹 Injuries Reported",
        "casualties_reported":"💀 Casualties Reported",
        "building_damage":    "🏚 Building Damage",
        "flooding_reported":  "🌊 Flooding Reported",
        "road_blocked":       "🚧 Road Blocked",
        "evacuation_mentioned":"🚶 Evacuation Mentioned",
        "transport_disruption":"🚌 Transport Disruption",
    }

    triggered_flags = []
    for key, label in FIELDS.items():
        field_data = emer.get(key, {})
        if field_data.get("present", False):
            phrases = field_data.get("evidence_phrases", [])
            triggered_flags.append((label, phrases))

    if triggered_flags:
        st.write("**Triggered Emergency Indicators:**")
        cols = st.columns(min(len(triggered_flags), 3))
        for idx, (label, phrases) in enumerate(triggered_flags):
            target_col = cols[idx % len(cols)]
            with target_col:
                st.success(f"✅ **{label}**")
                if phrases:
                    st.caption(f"Evidence: *{'; '.join(phrases[:2])}*")
    else:
        st.info("No critical emergency indicators flagged in this text.")

    # Affected count
    affected = emer.get("reported_affected_count")
    if affected:
        st.info(f"**Reported Affected Count:** {affected}")

    # Evidence phrases
    all_phrases = emer.get("all_evidence_phrases", [])
    if all_phrases:
        with st.expander("📎 All Evidence Phrases from Report"):
            for phrase in all_phrases:
                st.write(f"- *\"{phrase}\"*")


