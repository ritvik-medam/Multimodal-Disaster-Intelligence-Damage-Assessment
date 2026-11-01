"""
Page 6: Incident Map — Only incidents with valid coordinates are plotted.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.database import repository as repo
from src.database.models import initialize_database
from src.mapping.incident_mapping import build_incident_map
from src.utils import configuration as cfg
from src.utils.theme import apply_custom_theme

st.set_page_config(
    page_title="Incident Map | Disaster Intelligence",
    page_icon="🗺",
    layout="wide",
    initial_sidebar_state="expanded"
)
initialize_database()
apply_custom_theme()

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; flex-wrap:wrap; gap:12px;">
    <div>
        <h1 style="margin:0; display:flex; align-items:center; gap:12px;">
            <span>🗺️</span> <span>Geospatial Incident GIS & <span class="gradient-text">Situation Mapping</span></span>
        </h1>
        <p style="color:var(--text-secondary); margin:0.35rem 0 0 0; font-size:0.95rem;">Interactive spatial visualization of geocoded disaster reports and priority clusters.</p>
    </div>
    <div>
        <span class="cyber-badge badge-p3-glow"><span class="pulse-dot-emerald"></span> Interactive GIS Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Filters ────────────────────────────────────────────────────
col_f1, col_f2 = st.columns(2)
with col_f1:
    selected_categories = st.multiselect(
        "Filter by Disaster Type",
        options=cfg.DISASTER_CATEGORIES,
        default=[],
        format_func=lambda x: str(x).replace("_"," ").title() if x else ""
    )
with col_f2:
    selected_priorities = st.multiselect(
        "Filter by Priority",
        options=["P1_CRITICAL","P2_HIGH","P3_MODERATE","P4_LOW","UNASSESSED"],
        default=[],
    )

# ── Fetch incidents with coordinates ──────────────────────────
all_with_coords = repo.get_incidents_with_coords()
all_incidents   = repo.list_incidents(limit=500)

mapped_count = len(all_with_coords)
p1_count = sum(1 for inc in all_with_coords if inc.get("priority") == "P1_CRITICAL")
p2_count = sum(1 for inc in all_with_coords if inc.get("priority") == "P2_HIGH")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
    <div class="kpi-card kpi-card-cyan">
        <div class="kpi-label"><span>📍</span> Active Mapped Hotspots</div>
        <div class="kpi-value">{mapped_count}</div>
        <div class="kpi-subtext">Geocoded coordinates verified</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="kpi-card kpi-card-red">
        <div class="kpi-label"><span>🔴</span> Critical Priority (P1)</div>
        <div class="kpi-value">{p1_count}</div>
        <div class="kpi-subtext">Urgent on-scene deployment</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="kpi-card kpi-card-amber">
        <div class="kpi-label"><span>🟠</span> High Priority (P2)</div>
        <div class="kpi-value">{p2_count}</div>
        <div class="kpi-subtext">Monitored escalation zones</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ── Build and render map ───────────────────────────────────────
folium_map = build_incident_map(
    incidents=all_with_coords,
    selected_categories=selected_categories if selected_categories else None,
    selected_priorities=selected_priorities if selected_priorities else None,
)

if folium_map:
    try:
        from streamlit_folium import st_folium
        result = st_folium(folium_map, width=None, height=550, returned_objects=[])
    except ImportError:
        st.error("streamlit-folium not installed. Run: pip install streamlit-folium")
else:
    st.error("Folium not available. Run: pip install folium streamlit-folium")

# ── Table of mapped incidents ──────────────────────────────────
if all_with_coords:
    st.divider()
    st.subheader("📍 Mapped Incidents")
    import pandas as pd
    df = pd.DataFrame(all_with_coords)
    display_cols = ["incident_id","title","disaster_type","priority","reported_location","latitude","longitude","created_at"]
    df = df[[c for c in display_cols if c in df.columns]]
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info(
        "No incidents with valid coordinates found. "
        "Provide latitude and longitude when submitting incidents to enable mapping."
    )
