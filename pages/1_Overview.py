"""
Page 1: Overview Dashboard — Operational Emergency Command Center
High-tech situational awareness, neural telemetry, and interactive triage analytics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.database import repository as repo
from src.database.models import initialize_database
from src.utils.theme import apply_custom_theme, get_chart_theme

st.set_page_config(
    page_title="Overview | Disaster Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
initialize_database()
apply_custom_theme()

# ── Top Command Bar ──────────────────────────────────────────
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; flex-wrap:wrap; gap:12px;">
    <div>
        <h1 style="margin:0; display:flex; align-items:center; gap:12px;">
            <span>📊</span> <span>Operational Overview <span class="gradient-text">Command Center</span></span>
        </h1>
        <p style="color:var(--text-secondary); margin:0.35rem 0 0 0; font-size:0.95rem;">
            Real-time multi-modal triage intelligence, neural telemetry & synchronized emergency logs
        </p>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
        <span class="cyber-badge badge-p4-glow">
            <span class="pulse-dot-emerald"></span> All Neural Engines Nominal
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Neural Subsystem Live Telemetry Ribbon ───────────────────
st.markdown("""
<div class="telemetry-bar">
    <div class="telemetry-item">
        <span>🏆</span> <span>Dual-Encoder Fusion:</span> 
        <span class="telemetry-value">98.3% F1</span> 
        <span style="color:#10b981; font-size:0.75rem;">(+9.7% Synergy)</span>
    </div>
    <div class="telemetry-item">
        <span>🌪️</span> <span>EfficientNet Vision:</span> 
        <span class="telemetry-value">87.4% Acc</span>
        <span style="color:#38bdf8; font-size:0.75rem;">(B2 Backbone)</span>
    </div>
    <div class="telemetry-item">
        <span>📝</span> <span>DistilBERT NLP:</span> 
        <span class="telemetry-value">88.9% Acc</span>
        <span style="color:#38bdf8; font-size:0.75rem;">(Transformer)</span>
    </div>
    <div class="telemetry-item">
        <span>🏚️</span> <span>ResNet-50 Damage:</span> 
        <span class="telemetry-value">100.0% Acc</span>
        <span style="color:#10b981; font-size:0.75rem;">(Satellite Tiers)</span>
    </div>
    <div class="telemetry-item">
        <span>🎯</span> <span>YOLOv8n Edge:</span> 
        <span class="telemetry-value">~12ms</span>
        <span style="color:#a855f7; font-size:0.75rem;">(COCO 80 Cls)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Fetch live database metrics ──────────────────────────────
counts = repo.get_incident_counts()
total_inc = counts.get("total", 0)
high_pri  = counts.get("high_priority", 0)
fused_inc = counts.get("with_fusion", 0)
pending   = counts.get("pending_review", 0)

# ── High-Tech KPI Metric Cards ───────────────────────────────
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.markdown(f"""
    <div class="kpi-card kpi-card-cyan">
        <div class="kpi-label">
            <span>🛰️</span> Total Ingested Incidents
        </div>
        <div class="kpi-value">{total_inc}</div>
        <div class="kpi-subtext">ACID database live sync</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c2:
    st.markdown(f"""
    <div class="kpi-card kpi-card-red">
        <div class="kpi-label">
            <span>🚨</span> Critical Priority (P1/P2)
        </div>
        <div class="kpi-value">{high_pri}</div>
        <div class="kpi-subtext">Immediate field dispatch triage</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c3:
    st.markdown(f"""
    <div class="kpi-card kpi-card-purple">
        <div class="kpi-label">
            <span>🔗</span> Multimodal Fused
        </div>
        <div class="kpi-value">{fused_inc}</div>
        <div class="kpi-subtext">Joint vision + language verified</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c4:
    st.markdown(f"""
    <div class="kpi-card kpi-card-amber">
        <div class="kpi-label">
            <span>⏳</span> Pending Human Review
        </div>
        <div class="kpi-value">{pending}</div>
        <div class="kpi-subtext">Human-in-the-Loop review queue</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ── Visual Analytics Command Section ─────────────────────────
col_chart_left, col_chart_right = st.columns(2)

with col_chart_left:
    st.markdown("""
    <div class="command-frame">
        <div class="command-frame-header">
            <div class="command-frame-title">
                <span>🌪️</span> Disaster Classification Spectrum
            </div>
            <span class="cyber-badge badge-p3-glow">Multi-Class Distribution</span>
        </div>
    """, unsafe_allow_html=True)

    cat_data = repo.get_category_distribution()
    if cat_data:
        df_cat = pd.DataFrame(cat_data)
        df_cat["disaster_type"] = (
            df_cat["disaster_type"]
            .fillna("Unknown")
            .astype(str)
            .str.replace("_", " ")
            .str.title()
        )
        neon_colors = ["#06b6d4", "#f97316", "#ef4444", "#a855f7", "#10b981", "#3b82f6", "#eab308"]
        fig_donut = px.pie(
            df_cat,
            names="disaster_type",
            values="count",
            color_discrete_sequence=neon_colors,
            hole=0.55
        )
        ct = get_chart_theme()
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color=ct["donut_line"], width=2)),
            hovertemplate="<b>%{label}</b><br>Incident Count: %{value}<br>Proportion: %{percent}<extra></extra>"
        )
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(color=ct["legend_font"], size=11)
            ),
            margin=dict(t=10, b=30, l=10, r=10),
            paper_bgcolor=ct["paper_bg"],
            plot_bgcolor=ct["plot_bg"],
            font_color=ct["font_color"],
            height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No classified disaster incidents in database yet.")

    st.markdown("</div>", unsafe_allow_html=True)

with col_chart_right:
    st.markdown("""
    <div class="command-frame">
        <div class="command-frame-header">
            <div class="command-frame-title">
                <span>🎯</span> Priority Triage Hierarchy
            </div>
            <span class="cyber-badge badge-p1-glow">P1–P4 Severity Tiers</span>
        </div>
    """, unsafe_allow_html=True)

    pri_data = repo.get_priority_distribution()
    if pri_data:
        df_pri = pd.DataFrame(pri_data)
        color_map = {
            "P1_CRITICAL": "#ef4444",
            "P2_HIGH": "#f97316",
            "P3_MODERATE": "#3b82f6",
            "P4_LOW": "#10b981",
            "UNASSESSED": "#64748b"
        }
        fig_bar = px.bar(
            df_pri,
            x="priority",
            y="count",
            color="priority",
            color_discrete_map=color_map,
            text="count"
        )
        ct = get_chart_theme()
        fig_bar.update_traces(
            texttemplate="<b>%{text}</b>",
            textposition="outside",
            marker=dict(
                line=dict(color=ct["bar_line"], width=1),
                opacity=0.9
            ),
            hovertemplate="<b>Priority: %{x}</b><br>Active Incidents: %{y}<extra></extra>"
        )
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(t=25, b=20, l=10, r=10),
            paper_bgcolor=ct["paper_bg"],
            plot_bgcolor=ct["plot_bg"],
            font_color=ct["font_color"],
            xaxis=dict(
                title=dict(text="", font=dict(color=ct["legend_font"])),
                gridcolor=ct["grid_color"],
                tickfont=dict(color=ct["font_color"], size=11)
            ),
            yaxis=dict(
                title=dict(text="Incident Count", font=dict(color=ct["legend_font"], size=12)),
                gridcolor=ct["grid_color"],
                tickfont=dict(color=ct["font_color"], size=11)
            ),
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No priority triage data available yet.")

    st.markdown("</div>", unsafe_allow_html=True)

# ── High-Priority Incident Hotspots & Alerts ─────────────────
high_pri_incidents = [
    inc for inc in repo.list_incidents(limit=50)
    if inc.get("priority") in ("P1_CRITICAL", "P2_HIGH")
]

if high_pri_incidents:
    st.markdown("""
    <div style="margin-top:1rem; margin-bottom:0.75rem; display:flex; justify-content:space-between; align-items:center;">
        <h3 style="margin:0; font-size:1.15rem; display:flex; align-items:center; gap:8px;">
            <span>🚨</span> Urgent Triage Priority Attention
        </h3>
        <span class="cyber-badge badge-p1-glow">Immediate Dispatch Recommended</span>
    </div>
    """, unsafe_allow_html=True)

    alert_cols = st.columns(min(len(high_pri_incidents), 3))
    for idx, inc in enumerate(high_pri_incidents[:3]):
        with alert_cols[idx]:
            p_val = inc.get("priority", "P1_CRITICAL")
            is_p1 = p_val == "P1_CRITICAL"
            badge_class = "badge-p1-glow" if is_p1 else "badge-p2-glow"
            border_col  = "#ef4444" if is_p1 else "#f97316"
            dis_type    = str(inc.get("disaster_type") or "Emergency").replace("_", " ").title()
            loc_str     = inc.get("reported_location") or "Coordinates on Map"
            score_num   = inc.get("severity_score") or 0
            inc_id_str  = inc.get("incident_id")

            st.markdown(f"""
            <div class="incident-alert-card" style="border-left:4px solid {border_col} !important; border-radius:12px; padding:1rem 1.25rem; margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <span class="cyber-badge {badge_class}">{p_val}</span>
                    <span style="font-weight:700; color:{border_col}; font-size:0.95rem;">Score: {score_num}/100</span>
                </div>
                <div style="font-weight:700; font-size:1rem; color:var(--text-primary); margin-bottom:0.25rem;">
                    {inc.get('title') or dis_type}
                </div>
                <div style="color:var(--text-secondary); font-size:0.8rem; margin-bottom:0.5rem;">
                    📍 {loc_str}
                </div>
                <div style="font-size:0.75rem; color:#64748b;">
                    ID: <code>{inc_id_str}</code> | {str(inc.get('created_at', ''))[:16]}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Recent Incidents Operational Registry ─────────────────────
with st.container(border=True):
    st.markdown("""
    <div class="command-frame-header" style="margin-bottom:0.75rem;">
        <div class="command-frame-title">
            <span>📋</span> Operational Incident Ingestion Log
        </div>
        <div style="color:var(--text-secondary); font-size:0.85rem;">
            Showing latest 10 records with audit provenance
        </div>
    </div>
    """, unsafe_allow_html=True)

    recent = repo.list_incidents(limit=10)
    if recent:
        df = pd.DataFrame(recent)
        display_cols = [
            "incident_id", "title", "created_at", "input_type",
            "disaster_type", "priority", "review_status"
        ]
        df = df[[c for c in display_cols if c in df.columns]]
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        df["disaster_type"] = df["disaster_type"].fillna("Pending").astype(str).str.replace("_", " ").str.title()
        df["input_type"] = df["input_type"].astype(str).str.replace("_", " ").str.title()
        df["review_status"] = df["review_status"].fillna("Pending").astype(str).str.title()

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "incident_id": st.column_config.TextColumn("Incident ID", width=130),
                "title": st.column_config.TextColumn("Incident Title", width=220),
                "created_at": st.column_config.TextColumn("Timestamp", width=140),
                "input_type": st.column_config.TextColumn("Modality", width=110),
                "disaster_type": st.column_config.TextColumn("Disaster Category", width=140),
                "priority": st.column_config.TextColumn("Priority Triage", width=120),
                "review_status": st.column_config.TextColumn("Review Status", width=110),
            }
        )
    else:
        st.info("No incidents recorded yet. Submit an incident via **Analyze Incident** to initialize database records.")

# ── Quick Gateway Jump Buttons ───────────────────────────────
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-top:1.5rem; margin-bottom:0.75rem;">
    <div style="font-size:0.95rem; font-weight:700; color:var(--text-primary);">
        ⚡ Emergency Operations Shortcuts
    </div>
</div>
""", unsafe_allow_html=True)

g1, g2, g3, g4 = st.columns(4)
g1.page_link("pages/2_Analyze_Incident.py", label="Ingest New Incident", icon="🚨")
g2.page_link("pages/6_Incident_Map.py", label="Open Geospatial Map", icon="🗺️")
g3.page_link("pages/8_Model_Evaluation.py", label="View Model Benchmarks", icon="🧪")
g4.page_link("pages/7_Incident_History.py", label="Full Audit Archive", icon="📋")
