"""
Page 7: Incident History — Operational Audit Repository & Historical Logs
Provides executive KPI metrics, structured search console, interactive data registry,
and Human-in-the-Loop triage verification.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from src.database import repository as repo
from src.database.models import initialize_database
from src.utils import configuration as cfg
from src.utils.theme import apply_custom_theme

st.set_page_config(
    page_title="Incident History | Disaster Intelligence",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)
initialize_database()
apply_custom_theme()

# ── Header Command Bar ─────────────────────────────────────────
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; flex-wrap:wrap; gap:12px;">
    <div>
        <h1 style="margin:0; display:flex; align-items:center; gap:12px;">
            <span>📋</span> <span>Incident Audit Repository & <span class="gradient-text">Historical Logs</span></span>
        </h1>
        <p style="color:var(--text-secondary); margin:0.35rem 0 0 0; font-size:0.95rem;">
            Search, inspect, and export historical emergency records with full Human-in-the-Loop audit verification.
        </p>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
        <span class="cyber-badge badge-p4-glow">
            <span class="pulse-dot-emerald"></span> ACID Storage Synchronized
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Live Telemetry KPI Metrics ─────────────────────────────────
counts = repo.get_incident_counts()
total_cnt = counts.get("total", 0)
high_cnt  = counts.get("high_priority", 0)
fused_cnt = counts.get("with_fusion", 0)
pend_cnt  = counts.get("pending_review", 0)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card kpi-card-cyan">
        <div class="kpi-label"><span>🛰️</span> Total Ingested</div>
        <div class="kpi-value">{total_cnt}</div>
        <div class="kpi-subtext">Historical incident records</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card kpi-card-red">
        <div class="kpi-label"><span>🚨</span> Critical (P1/P2)</div>
        <div class="kpi-value">{high_cnt}</div>
        <div class="kpi-subtext">High priority dispatch queue</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card kpi-card-purple">
        <div class="kpi-label"><span>🔗</span> Multimodal Fused</div>
        <div class="kpi-value">{fused_cnt}</div>
        <div class="kpi-subtext">Cross-modal verified records</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card kpi-card-amber">
        <div class="kpi-label"><span>⏳</span> Pending Review</div>
        <div class="kpi-value">{pend_cnt}</div>
        <div class="kpi-subtext">Awaiting human sign-off</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ── Search & Filters Console ───────────────────────────────────
with st.container(border=True):
    st.markdown("""
    <div class="command-frame-header" style="margin-bottom:0.75rem;">
        <div class="command-frame-title">
            <span>🔍</span> Operational Search & Query Console
        </div>
        <span class="cyber-badge badge-p3-glow">Multi-Parameter Filtering</span>
    </div>
    """, unsafe_allow_html=True)

    col_s, col_c, col_p, col_r = st.columns([3, 2, 2, 2])
    with col_s:
        search = st.text_input("🔍 Keyword Search", placeholder="Filter by title, location, or report text...", label_visibility="visible")
    with col_c:
        cat_filter = st.selectbox(
            "Disaster Category", 
            ["All"] + cfg.DISASTER_CATEGORIES,
            format_func=lambda x: x.replace("_", " ").title() if x != "All" else "All Categories"
        )
    with col_p:
        pri_filter = st.selectbox(
            "Priority Tier", 
            ["All", "P1_CRITICAL", "P2_HIGH", "P3_MODERATE", "P4_LOW", "UNASSESSED"],
            format_func=lambda x: x if x == "All" else x.replace("_", " — ")
        )
    with col_r:
        rev_filter = st.selectbox(
            "Human Review Status", 
            ["All", "pending", "reviewed", "escalated", "closed"],
            format_func=lambda x: x.title() if x != "All" else "All Statuses"
        )

# ── Fetch Filtered Incidents ───────────────────────────────────
incidents = repo.list_incidents(
    limit=250,
    disaster_type=cat_filter if cat_filter != "All" else None,
    priority=pri_filter if pri_filter != "All" else None,
    review_status=rev_filter if rev_filter != "All" else None,
    search=search if search else None,
)

# ── Table Command Container ────────────────────────────────────
with st.container(border=True):
    st.markdown(f"""
    <div class="command-frame-header" style="margin-bottom:0.75rem;">
        <div class="command-frame-title">
            <span>📋</span> Incident Ingestion & Audit Registry
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span class="cyber-badge badge-p4-glow">{len(incidents)} Incidents Matching</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if not incidents:
    st.info("No incident records match the active filter criteria. Try adjusting the search keywords or filters above.")
else:
    df = pd.DataFrame(incidents)
    display_cols = [
        "incident_id", "title", "created_at", "input_type",
        "disaster_type", "priority", "reported_location", "review_status", "processing_status"
    ]
    df = df[[c for c in display_cols if c in df.columns]]
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    df["disaster_type"] = df["disaster_type"].fillna("Pending").astype(str).str.replace("_", " ").str.title()
    df["input_type"] = df["input_type"].astype(str).str.replace("_", " ").str.title()
    df["review_status"] = df["review_status"].fillna("Pending").astype(str).str.title()
    df["processing_status"] = df["processing_status"].fillna("Complete").astype(str).str.title()

    c_export, _ = st.columns([2, 6])
    with c_export:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Audit Log",
            data=csv_data,
            file_name="disaster_incidents_audit_log.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "incident_id": st.column_config.TextColumn("Incident ID", width=140),
            "title": st.column_config.TextColumn("Incident Title", width=220),
            "created_at": st.column_config.TextColumn("Logged At", width=140),
            "input_type": st.column_config.TextColumn("Modality", width=110),
            "disaster_type": st.column_config.TextColumn("Category", width=130),
            "priority": st.column_config.TextColumn("Priority Triage", width=120),
            "reported_location": st.column_config.TextColumn("Location", width=160),
            "review_status": st.column_config.TextColumn("Review State", width=110),
            "processing_status": st.column_config.TextColumn("Pipeline", width=100),
        }
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── Incident Detail Inspection & Review Dossier ────────────────
if incidents:
    with st.container(border=True):
        st.markdown("""
        <div class="command-frame-header" style="margin-bottom:0.75rem;">
            <div class="command-frame-title">
                <span>🛡️</span> Human-in-the-Loop Incident Dossier & Sign-Off
            </div>
            <span class="cyber-badge badge-p1-glow">Operational Audit Action</span>
        </div>
        """, unsafe_allow_html=True)

    inc_ids = [inc["incident_id"] for inc in incidents]
    inc_map = {inc["incident_id"]: inc for inc in incidents}

    selected_id = st.selectbox(
        "Select Incident Record to Inspect & Review:",
        options=inc_ids,
        format_func=lambda x: f"{x} — {inc_map[x].get('title') or inc_map[x].get('disaster_type') or 'Incident'} | Priority: {inc_map[x].get('priority', 'UNASSESSED')} ({str(inc_map[x].get('created_at',''))[:16]})"
    )

    if selected_id:
        inc = repo.get_incident(selected_id) or inc_map.get(selected_id) or {}

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        priority_val = inc.get("priority") or "UNASSESSED"
        score_val    = inc.get("severity_score") or 0
        type_val     = str(inc.get("disaster_type") or "Unknown").replace('_', ' ').title()
        status_val   = inc.get("review_status") or "pending"

        col_m1.metric("Priority Level", priority_val)
        col_m2.metric("Severity Score", f"{score_val}/100")
        col_m3.metric("Disaster Category", type_val)
        col_m4.metric("Review Status", status_val.title())

        st.markdown(f"""
        <div class="incident-alert-card" style="border-radius:12px; padding:1rem 1.25rem; margin:1rem 0;">
            <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.35rem; display:flex; align-items:center; gap:8px;">
                <span>📍</span> Location & Telemetry Verification
            </div>
            <div style="color:var(--text-secondary); font-size:0.9rem; line-height:1.6;">
                <b>Reported Location:</b> {inc.get('reported_location') or 'Not specified'} &nbsp;|&nbsp; 
                <b>Coordinates:</b> Lat <code>{inc.get('latitude') or 'N/A'}</code>, Lon <code>{inc.get('longitude') or 'N/A'}</code> &nbsp;|&nbsp;
                <b>Logged Timestamp:</b> {str(inc.get('created_at',''))[:19]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if inc.get("original_report"):
            with st.expander("📄 Original Incident Ground Report Text", expanded=True):
                st.write(inc["original_report"])

        # Operational status updater
        st.markdown("""
        <div style="font-weight:700; font-size:0.95rem; color:var(--text-primary); margin-top:1rem; margin-bottom:0.4rem;">
            ⚡ Operational Triage Status Action
        </div>
        """, unsafe_allow_html=True)

        c_status, c_save = st.columns([3, 1])
        with c_status:
            status_options = ["pending", "reviewed", "escalated", "closed"]
            current_rev = inc.get("review_status") or "pending"
            default_idx = status_options.index(current_rev) if current_rev in status_options else 0
            new_status = st.selectbox(
                "Update Triage Review Status",
                status_options,
                index=default_idx,
                label_visibility="collapsed",
                format_func=lambda s: f"Status: {s.title()}"
            )
        with c_save:
            if st.button("💾 Save Status Update", use_container_width=True, type="primary"):
                repo.update_review_status(selected_id, new_status)
                st.success(f"Incident {selected_id} status updated to '{new_status.title()}'")
                st.rerun()

        # Collapsible audit payload
        with st.expander("🛠 Full Raw Relational Audit Payload (JSON)", expanded=False):
            st.json(inc)
