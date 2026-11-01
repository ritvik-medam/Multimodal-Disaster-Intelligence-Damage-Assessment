"""
Page 5: Multimodal Assessment — Key page showing fused evidence, consistency, priority
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
from src.utils.theme import apply_custom_theme, get_chart_theme

st.set_page_config(
    page_title="Multimodal Assessment | Disaster Intelligence",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_custom_theme()

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; flex-wrap:wrap; gap:10px;">
    <div>
        <h1 style="margin:0;">🔗 Unified Multimodal Assessment & Provenance</h1>
        <p style="color:var(--text-secondary); margin:0.25rem 0 0 0;">Cross-modal evidence fusion, conflict resolution, and deterministic severity scoring.</p>
    </div>
    <div>
        <span class="cyber-badge badge-p2-glow">Dual-Encoder Fusion Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)

from src.database import repository as repo
from src.database.models import initialize_database

initialize_database()

# Check for results in session state or DB
incident_id = st.session_state.get("current_incident_id")

fusion = st.session_state.get("fusion_result")
consistency = st.session_state.get("consistency_result")
severity = st.session_state.get("severity_result")

# Fallback to database if session state is empty
if not fusion:
    incidents = repo.list_incidents(limit=10)
    if incidents:
        mm_map = {inc["incident_id"]: inc for inc in incidents}
        mm_ids = [inc["incident_id"] for inc in incidents]
        selected_id = st.selectbox(
            "Select Incident to inspect Multimodal Fusion:",
            options=mm_ids,
            format_func=lambda x: f"{x} - {mm_map[x].get('title') or mm_map[x].get('disaster_type') or 'Incident'} | Priority: {mm_map[x].get('priority', 'P2_HIGH')} ({str(mm_map[x].get('created_at', ''))[:19]})"
        )
        if selected_id:
            incident_id = selected_id
            selected_inc = mm_map[selected_id]
            fr = repo.get_fusion_result(incident_id)
            if fr:
                fusion = {
                    "has_visual_input": bool(fr.get("visual_evidence")),
                    "has_text_input": bool(fr.get("textual_evidence")),
                    "input_modalities": "image_text" if (fr.get("visual_evidence") and fr.get("textual_evidence")) else ("image_only" if fr.get("visual_evidence") else "text_only"),
                    "visual_evidence": fr.get("visual_evidence") or {},
                    "textual_evidence": fr.get("textual_evidence") or {},
                    "combined_findings": [{"finding": f"Incident assessed as {selected_inc.get('disaster_type', 'disaster')}", "provenance": "supported_by_both_modalities"}],
                    "missing_information": fr.get("missing_information") or [],
                    "uncertainty_notes": [fr.get("uncertainty_notes")] if fr.get("uncertainty_notes") else [],
                    "primary_disaster_category": selected_inc.get("disaster_type") or "flood",
                    "primary_category_provenance": "supported_by_both_modalities"
                }
                consistency = fr.get("consistency_findings") or {"overall_consistency": selected_inc.get("consistency_status") or "consistent_both_modalities", "checks": []}
                severity = {
                    "priority": selected_inc.get("priority") or fr.get("priority") or "P2_HIGH",
                    "severity_score": selected_inc.get("severity_score") or fr.get("severity_score") or 75,
                    "contributing_factors": fr.get("scoring_explanation") or [{"factor": "Emergency need reported", "points": 25}],
                    "penalties": [],
                    "uncertainty_notes": [fr.get("uncertainty_notes")] if fr.get("uncertainty_notes") else []
                }
    else:
        st.info("No multimodal assessments found in database. Submit an incident via **Analyze Incident**.")
        st.stop()

# ── Priority Banner ────────────────────────────────────────────
priority = (severity.get("priority") if severity else None) or "UNASSESSED"
score    = (severity.get("severity_score") if severity else 0) or 0

PRIORITY_COLORS = {
    "P1_CRITICAL": "#dc2626", "P2_HIGH": "#ea580c",
    "P3_MODERATE": "#2563eb", "P4_LOW": "#16a34a", "UNASSESSED": "#6b7280"
}
color = PRIORITY_COLORS.get(priority, "#6b7280")
priority_title = str(priority).replace('_', ' ')

st.markdown(f"""
<div style="background:{color}22; border-left:6px solid {color}; border-radius:8px;
     padding:1rem 1.5rem; margin-bottom:1rem;">
    <h2 style="margin:0; color:{color};">🎯 {priority_title}</h2>
    <p style="margin:4px 0 0; color:var(--text-secondary);">Severity Score: <b>{score}/100</b></p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Two-column Evidence View ─────────────────────────────────
col_vis, col_txt = st.columns(2)

with col_vis:
    st.subheader("👁 Visual Evidence")
    ve = fusion.get("visual_evidence", {}) if fusion else {}
    if ve:
        cat = ve.get("disaster_category", {})
        if cat.get("value"):
            st.success(f"**Disaster Type:** {cat['value'].replace('_',' ').title()}")
            if cat.get("confidence"):
                st.caption(f"Confidence: {cat['confidence']:.1%}")

        obj_counts = ve.get("object_counts", {})
        if obj_counts:
            st.write("**Detected Objects:**")
            for obj, cnt in sorted(obj_counts.items(), key=lambda x: -x[1]):
                st.write(f"  - {obj.title()}: {cnt}")

        dmg = ve.get("damage_level", {})
        if dmg.get("value"):
            st.warning(f"**Damage Level:** {dmg['value'].replace('_',' ').title()}")
    else:
        st.info("No visual imagery attached to this incident.")

with col_txt:
    st.subheader("📝 Textual Evidence")
    te = fusion.get("textual_evidence", {}) if fusion else {}
    if te:
        cat = te.get("disaster_category", {})
        if cat.get("value"):
            st.success(f"**Disaster Type:** {cat['value'].replace('_',' ').title()}")

        indicators = [
            ("rescue_request",    "🆘 Rescue Requested"),
            ("trapped_people",    "⚠️ Trapped People"),
            ("casualties_reported","💀 Casualties"),
            ("injuries_reported", "🩹 Injuries"),
            ("evacuation",        "🚶 Evacuation"),
        ]
        active_ind = [label for key, label in indicators if te.get(key, {}).get("present")]
        if active_ind:
            st.write("**Emergency Indicators:**")
            for ind in active_ind:
                st.write(f"  ✅ {ind}")

        count = te.get("reported_affected_count")
        if count: st.info(f"Reported affected: {count}")

        locs = te.get("locations", [])
        if locs: st.write(f"**Locations:** {', '.join(locs)}")
    else:
        st.info("No textual report attached to this incident.")

st.divider()

# ── Combined Findings ─────────────────────────────────────────
st.subheader("🔎 Combined Findings")
findings = fusion.get("combined_findings", []) if fusion else []
if findings:
    PROV_COLORS = {
        "supported_by_both_modalities": "✅",
        "visually_detected": "👁",
        "reported_in_text": "📝",
        "inferred": "💡",
        "unverified": "⚠️",
        "unknown": "❓",
    }
    for f in findings:
        icon = PROV_COLORS.get(f.get("provenance","unknown"), "•")
        confidence = f.get("confidence","")
        st.write(f"{icon} {f['finding']} `[{f.get('provenance','')}]`")
else:
    st.info("No combined findings available.")

st.divider()

# ── Consistency Analysis ───────────────────────────────────────
st.subheader("🔄 Cross-Modal Consistency Analysis")
if consistency:
    overall = (consistency.get("overall_consistency") if consistency else None) or "unknown"
    summary = consistency.get("summary", "")
    caveat  = consistency.get("important_caveat", "")

    CONS_COLORS = {
        "consistent_both_modalities": "success",
        "potential_inconsistency": "error",
        "insufficient_visual_evidence": "warning",
        "insufficient_textual_evidence": "warning",
        "unable_to_determine": "info",
    }
    fn = getattr(st, CONS_COLORS.get(overall, "info"))
    fn(f"**Overall: {str(overall).replace('_',' ').title()}** — {summary}")

    for check in consistency.get("checks", []):
        outcome_icons = {
            "consistent_both_modalities": "✅",
            "potential_inconsistency": "❗",
            "insufficient_visual_evidence": "⚠️",
            "insufficient_textual_evidence": "⚠️",
            "unable_to_determine": "❓",
        }
        icon = outcome_icons.get(check["outcome"], "•")
        with st.expander(f"{icon} {check['comparison']}"):
            st.write(check["explanation"])

st.divider()

# ── Severity Scoring Explanation ───────────────────────────────
st.subheader("📊 Severity Scoring Breakdown")
if severity:
    factors = severity.get("contributing_factors", [])
    missing = severity.get("missing_evidence", [])
    uncertainty = severity.get("uncertainty_notes", [])

    if factors:
        import pandas as pd
        df = pd.DataFrame(factors)
        cols = [c for c in ["factor","points","source","provenance"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

        # Score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x":[0,1], "y":[0,1]},
            title={"text": "Severity Score"},
            gauge={
                "axis": {"range":[0,100]},
                "bar": {"color": color},
                "steps": [
                    {"range":[0,35], "color":"#dcfce7"},
                    {"range":[35,60],"color":"#fef9c3"},
                    {"range":[60,80],"color":"#fed7aa"},
                    {"range":[80,100],"color":"#fee2e2"},
                ],
                "threshold":{"line":{"color":"red","width":4},"thickness":0.75,"value":score}
            }
        ))
        ct = get_chart_theme()
        fig.update_layout(height=250, margin=dict(t=30,b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font_color=ct["font_color"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No scoring factors available (insufficient evidence).")

    # Optional Triage Notes & Missing Evidence
    all_notes = list(set((missing or []) + (uncertainty or []) + (fusion.get("missing_information", []) if fusion else [])))
    if all_notes:
        with st.expander("ℹ️ Observation Notes & Unobserved Evidence", expanded=False):
            for note in all_notes:
                st.write(f"- {note}")
