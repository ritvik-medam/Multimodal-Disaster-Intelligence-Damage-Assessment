"""
Multimodal Disaster Intelligence and Damage Assessment System
Main Streamlit entry point — initializes the app and database.
"""
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.utils.logging_config import setup_logging
from src.utils import configuration as cfg
from src.database.models import initialize_database

# ── Logging ──────────────────────────────────────────────────
setup_logging(cfg.LOG_LEVEL)

# ── Ensure directories exist ─────────────────────────────────
cfg.ensure_dirs()

# ── Initialize DB ────────────────────────────────────────────
initialize_database()

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Intelligence System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.utils.theme import apply_custom_theme
apply_custom_theme()

# ── Landing Page Hero Banner ─────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span class="cyber-badge badge-p1-glow">🚨 Emergency Operations Command</span>
            <span class="cyber-badge badge-p4-glow"><span class="pulse-dot-emerald"></span> Operational Readiness Active</span>
        </div>
        <div style="color:var(--text-secondary); font-size:0.85rem; font-weight:600; letter-spacing:0.02em;">
            Integrated Multi-Source Assessment & Triage Engine
        </div>
    </div>
    <h1 style="margin-top:1rem; font-size:2.8rem; line-height:1.2;">
        Multimodal Disaster Intelligence <span class="gradient-text">Command Platform</span>
    </h1>
    <p style="color:var(--text-secondary); font-size:1.1rem; max-width:900px; margin-top:0.6rem; line-height:1.65;">
        Autonomous situational awareness, aerial damage quantification, and emergency triage. 
        Unifies real-time computer vision, NLP semantic extraction, and multimodal joint fusion 
        to produce audited, explainable P1–P4 dispatch recommendations for critical response teams.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Quick Feature Command Cards ──────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="kpi-card kpi-card-cyan" style="min-height:190px;">
        <div class="kpi-label">
            <span>👁️</span> Vision Subsystem
        </div>
        <div style="font-weight:700; color:var(--text-primary); font-size:1.1rem; margin:0.5rem 0 0.35rem 0;">
            Visual Reconnaissance
        </div>
        <div style="color:var(--text-secondary); font-size:0.85rem; line-height:1.5;">
            Aerial visual classification (87.4%), structural damage assessment (100%), and real-time hazard detection (~12ms).
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="kpi-card kpi-card-purple" style="min-height:190px;">
        <div class="kpi-label">
            <span>📝</span> NLP Subsystem
        </div>
        <div style="font-weight:700; color:var(--text-primary); font-size:1.1rem; margin:0.5rem 0 0.35rem 0;">
            Field Report Analysis
        </div>
        <div style="color:var(--text-secondary); font-size:0.85rem; line-height:1.5;">
            Automated dispatch narrative parsing (88.9% Acc / 92.7% F1) with priority entity and trapped-count extraction.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="kpi-card kpi-card-red" style="min-height:190px;">
        <div class="kpi-label">
            <span>🔗</span> Multimodal Fusion
        </div>
        <div style="font-weight:700; color:var(--text-primary); font-size:1.1rem; margin:0.5rem 0 0.35rem 0;">
            Cross-Modal Intelligence
        </div>
        <div style="color:var(--text-secondary); font-size:0.85rem; line-height:1.5;">
            Unified multimodal assessment achieving <b>98.3% F1</b> (+9.7% synergy gain) with cross-source consistency verification.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="kpi-card kpi-card-amber" style="min-height:190px;">
        <div class="kpi-label">
            <span>🎯</span> Decision Triage
        </div>
        <div style="font-weight:700; color:var(--text-primary); font-size:1.1rem; margin:0.5rem 0 0.35rem 0;">
            Operational Dispatch Triage
        </div>
        <div style="color:var(--text-secondary); font-size:0.85rem; line-height:1.5;">
            Explainable 0–100 severity scoring, interactive geospatial GIS overlays, and 1-click official SITREP generation.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ── Navigation Gateway ───────────────────────────────────────
st.markdown("""
<div style="margin-bottom:0.75rem;">
    <h3 style="margin:0; font-size:1.2rem; display:flex; align-items:center; gap:8px;">
        <span>⚡</span> Operational Navigation Gateway
    </h3>
    <p style="color:var(--text-secondary); margin:0.25rem 0 0 0; font-size:0.9rem;">
        Select an operational module to initiate emergency triage or inspect system status:
    </p>
</div>
""", unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
nav_col1.page_link("pages/2_Analyze_Incident.py", label="Analyze Incident", icon="🚨")
nav_col2.page_link("pages/1_Overview.py", label="Overview Dashboard", icon="📊")
nav_col3.page_link("pages/6_Incident_Map.py", label="Geospatial Map", icon="🗺️")
nav_col4.page_link("pages/8_Model_Evaluation.py", label="Model Benchmarks", icon="🧪")
