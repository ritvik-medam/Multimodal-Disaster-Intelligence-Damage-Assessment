"""
UI Theme & Custom Styling Engine for Multimodal Disaster Intelligence
Provides full dual-theme support (Dark Mode and Light Mode) with instant sidebar toggle,
bespoke cyber-emergency aesthetics for dark mode, crisp high-contrast glassmorphism for light mode,
fluid typography, and adaptive Plotly chart color schemes.
"""

import streamlit as st


def get_theme_mode() -> str:
    """Returns the currently active theme mode ('dark' or 'light'). Defaults to 'dark'."""
    return st.session_state.get("theme_mode", "dark")


def get_chart_theme() -> dict:
    """
    Returns theme-aware colors for Plotly figures so that visualizations
    remain crisp, high-contrast, and beautiful in both dark and light modes.
    """
    mode = get_theme_mode()
    if mode == "light":
        return {
            "mode": "light",
            "font_color": "#0f172a",
            "grid_color": "rgba(0, 0, 0, 0.08)",
            "legend_font": "#475569",
            "donut_line": "#ffffff",
            "bar_line": "rgba(0, 0, 0, 0.12)",
            "paper_bg": "rgba(0, 0, 0, 0)",
            "plot_bg": "rgba(0, 0, 0, 0)",
            "gauge_bg": "#f8fafc",
        }
    return {
        "mode": "dark",
        "font_color": "#e2e8f0",
        "grid_color": "rgba(255, 255, 255, 0.06)",
        "legend_font": "#94a3b8",
        "donut_line": "#090d16",
        "bar_line": "rgba(255, 255, 255, 0.15)",
        "paper_bg": "rgba(0, 0, 0, 0)",
        "plot_bg": "rgba(0, 0, 0, 0)",
        "gauge_bg": "#090d16",
    }


def render_theme_toggle():
    """
    Renders an appearance toggle in the sidebar allowing instant switching
    between Dark and Light modes.
    """
    current_mode = get_theme_mode()

    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.25rem 0.25rem 0.85rem 0.25rem; border-bottom: 1px solid var(--border-glass); margin-bottom: 0.85rem;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.6rem; filter: drop-shadow(0 2px 8px rgba(56,189,248,0.4));">🛰️</span>
                <div>
                    <div style="font-weight:800; font-size:0.95rem; letter-spacing:0.02em; color:var(--text-primary); line-height:1.2;">DISASTER INTEL</div>
                    <div style="font-size:0.7rem; font-weight:600; color:var(--text-secondary); letter-spacing:0.05em; display:flex; align-items:center; gap:5px;">
                        <span class="pulse-dot-emerald" style="width:6px; height:6px;"></span> OSINT COMMAND v2.4
                    </div>
                </div>
            </div>
        </div>
        <div style="margin-top:0.15rem; margin-bottom:0.5rem; padding:0 2px;">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-secondary); margin-bottom:0.4rem; display:flex; align-items:center; gap:6px;">
                <span>🎨</span> Appearance Theme
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_dark, col_light = st.columns(2)
        with col_dark:
            is_dark = current_mode == "dark"
            if st.button("🌙 Dark", key="theme_btn_dark", use_container_width=True, type="primary" if is_dark else "secondary"):
                if current_mode != "dark":
                    st.session_state["theme_mode"] = "dark"
                    st.rerun()

        with col_light:
            is_light = current_mode == "light"
            if st.button("☀️ Light", key="theme_btn_light", use_container_width=True, type="primary" if is_light else "secondary"):
                if current_mode != "light":
                    st.session_state["theme_mode"] = "light"
                    st.rerun()

        st.markdown("<div style='margin-bottom:0.85rem; border-bottom:1px solid var(--border-glass);'></div>", unsafe_allow_html=True)


def get_theme_css(mode: str) -> str:
    """Generates the full CSS ruleset tailored for either Dark or Light mode."""
    if mode == "light":
        return """
        :root {
            --bg-main: #f8fafc;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --border-glass: #e2e8f0;
            --border-accent: rgba(2, 132, 199, 0.35);
            --primary: #0284c7;
            --primary-glow: rgba(2, 132, 199, 0.2);
            --accent-orange: #ea580c;
            --accent-red: #dc2626;
            --accent-emerald: #059669;
            --text-primary: #0f172a;
            --text-secondary: #475569;
        }

        /* Overall font application */
        html, body, p, label, .stMarkdown, .stMarkdown p {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: var(--text-primary);
        }

        /* Background glow effect for Light Mode */
        .stApp {
            background-color: var(--bg-main) !important;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(2, 132, 199, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(239, 68, 68, 0.04) 0%, transparent 40%) !important;
            background-attachment: fixed !important;
        }

        /* Sidebar Styling for Light Mode */
        [data-testid="stSidebar"] {
            background: #ffffff !important;
            backdrop-filter: blur(20px) !important;
            border-right: 1px solid #e2e8f0 !important;
            box-shadow: 2px 0 16px rgba(0, 0, 0, 0.03) !important;
        }
        [data-testid="stSidebar"] * {
            color: #0f172a;
        }
        [data-testid="stSidebarNav"] span {
            color: #334155 !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #e0f2fe !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] span {
            color: #0284c7 !important;
            font-weight: 700 !important;
        }

        /* Modern White Cards for Metrics */
        [data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 1.25rem !important;
            backdrop-filter: blur(16px) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px) !important;
            border-color: #0284c7 !important;
            box-shadow: 0 8px 24px rgba(2, 132, 199, 0.15) !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            color: #0f172a !important;
            background: linear-gradient(135deg, #0f172a 0%, #0284c7 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }

        /* Action Buttons for Light Mode */
        .stButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            transition: all 0.25s ease !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            color: #1e293b !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            border-color: #0284c7 !important;
            background: #f0f9ff !important;
            color: #0284c7 !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.18) !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            border: none !important;
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45) !important;
            color: #ffffff !important;
        }

        /* Expanders for Light Mode */
        [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 14px !important;
            margin-bottom: 0.75rem !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
        }
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] [data-testid="stExpanderSummary"],
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: transparent !important;
            color: #0f172a !important;
        }
        [data-testid="stExpander"] summary {
            border-radius: 12px !important;
            border: none !important;
            padding: 0.75rem 1rem !important;
        }
        [data-testid="stExpander"] summary:hover {
            background-color: #f8fafc !important;
            color: #0284c7 !important;
        }

        /* Tabs for Light Mode */
        .stTabs [data-baseweb="tab-list"] {
            background: #f1f5f9 !important;
            border-radius: 12px !important;
            padding: 6px !important;
            border: 1px solid #e2e8f0 !important;
            gap: 8px !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important;
            color: #64748b !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            border: none !important;
            background: transparent !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #0f172a !important;
            background: #e2e8f0 !important;
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff !important;
            color: #0284c7 !important;
            border-bottom: 2px solid #0284c7 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
        }

        /* Form Inputs for Light Mode */
        .stTextInput > div > div > input, 
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div,
        .stNumberInput > div > div > input {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
            color: #0f172a !important;
        }
        .stTextInput > div > div > input:focus, 
        .stTextArea > div > div > textarea:focus {
            border-color: #0284c7 !important;
            box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.2) !important;
        }

        /* Dataframes & Tables for Light Mode */
        [data-testid="stDataFrame"] {
            filter: invert(0.92) hue-rotate(180deg) !important;
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
            background-color: #ffffff !important;
        }
        [data-testid="stTable"] {
            background-color: #ffffff !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
        }
        [data-testid="stTable"] table {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        [data-testid="stTable"] th {
            background-color: #f1f5f9 !important;
            color: #0f172a !important;
            font-weight: 700 !important;
            border-bottom: 2px solid #e2e8f0 !important;
        }
        [data-testid="stTable"] td {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border-bottom: 1px solid #f1f5f9 !important;
        }

        /* Typography & Headings */
        h1 {
            color: #0f172a !important;
            font-weight: 800 !important;
        }
        .gradient-text {
            background: linear-gradient(135deg, #0f172a 20%, #0284c7 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            display: inline;
        }
        h2, h3, h4, h5, h6 {
            color: #1e293b !important;
            font-weight: 700 !important;
        }

        /* Command Frames & Border Containers for Light Mode */
        .command-frame,
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 1.25rem 1.5rem !important;
            backdrop-filter: blur(20px) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
            margin-bottom: 1.25rem !important;
        }
        .command-frame-header {
            border-bottom: 1px solid #f1f5f9 !important;
        }
        .command-frame-title {
            color: #0f172a !important;
        }

        /* KPI Cards for Light Mode */
        .kpi-card {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 1.25rem 1.5rem !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative;
            overflow: hidden;
        }
        .kpi-card:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08) !important;
        }
        .kpi-card-cyan {
            border-top: 3px solid #0284c7 !important;
        }
        .kpi-card-red {
            border-top: 3px solid #ef4444 !important;
        }
        .kpi-card-purple {
            border-top: 3px solid #a855f7 !important;
        }
        .kpi-card-amber {
            border-top: 3px solid #f59e0b !important;
        }
        .kpi-label {
            color: #64748b !important;
        }
        .kpi-value {
            color: #0f172a !important;
        }
        .kpi-subtext {
            color: #64748b !important;
        }
        .kpi-card > div {
            color: #0f172a;
        }

        /* Telemetry Ribbon for Light Mode */
        .telemetry-bar {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
        }
        .telemetry-item {
            color: #475569 !important;
        }
        .telemetry-value {
            color: #0f172a !important;
        }

        /* Hero Banner for Light Mode */
        .hero-banner {
            background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%) !important;
            border: 1px solid #bae6fd !important;
            border-radius: 20px !important;
            padding: 2rem !important;
            margin-bottom: 2rem !important;
            box-shadow: 0 10px 30px rgba(2, 132, 199, 0.08) !important;
        }
        .hero-banner h1, .hero-banner p {
            color: #0f172a !important;
        }

        /* Badges for Light Mode */
        .badge-p1-glow {
            background: #fee2e2 !important;
            color: #b91c1c !important;
            border: 1px solid #fca5a5 !important;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.15) !important;
        }
        .badge-p2-glow {
            background: #ffedd5 !important;
            color: #c2410c !important;
            border: 1px solid #fdba74 !important;
            box-shadow: 0 0 10px rgba(249, 115, 22, 0.15) !important;
        }
        .badge-p3-glow {
            background: #dbeafe !important;
            color: #1d4ed8 !important;
            border: 1px solid #93c5fd !important;
        }
        .badge-p4-glow {
            background: #dcfce7 !important;
            color: #15803d !important;
            border: 1px solid #86efac !important;
        }

        /* Alert and Information Containers */
        .incident-alert-card {
            background: #ffffff !important;
            border-top: 1px solid #e2e8f0 !important;
            border-right: 1px solid #e2e8f0 !important;
            border-bottom: 1px solid #e2e8f0 !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important;
        }

        /* Sidebar Close Button for Light Mode */
        [data-testid="stSidebarCollapseButton"] button,
        button[aria-label="Collapse sidebar"],
        button[aria-label="Close sidebar"] {
            color: #0284c7 !important;
            background: #f1f5f9 !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover,
        button[aria-label="Collapse sidebar"]:hover,
        button[aria-label="Close sidebar"]:hover {
            background: #e0f2fe !important;
            border-color: #0284c7 !important;
            color: #0284c7 !important;
            box-shadow: 0 0 12px rgba(2, 132, 199, 0.3) !important;
        }
        [data-testid="stSidebarCollapseButton"] button::after,
        button[aria-label="Collapse sidebar"]::after,
        button[aria-label="Close sidebar"]::after {
            color: #64748b !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover::after,
        button[aria-label="Collapse sidebar"]:hover::after,
        button[aria-label="Close sidebar"]:hover::after {
            color: #0284c7 !important;
        }
        [data-testid="stSidebarCollapseButton"] svg,
        button[aria-label="Collapse sidebar"] svg,
        button[aria-label="Close sidebar"] svg {
            fill: #0284c7 !important;
            stroke: #0284c7 !important;
            color: #0284c7 !important;
        }
        """

    # Default to Dark Mode CSS
    return """
    :root {
        --bg-main: #090d16;
        --bg-card: rgba(16, 24, 40, 0.75);
        --bg-card-hover: rgba(26, 38, 62, 0.9);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(56, 189, 248, 0.3);
        --primary: #38bdf8;
        --primary-glow: rgba(56, 189, 248, 0.25);
        --accent-orange: #fb923c;
        --accent-red: #f87171;
        --accent-emerald: #34d399;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    /* Overall font application */
    html, body, p, label, .stMarkdown, .stMarkdown p {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-primary);
    }

    /* Background glow effect for Dark Mode */
    .stApp {
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(248, 113, 113, 0.06) 0%, transparent 40%) !important;
        background-attachment: fixed !important;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 26, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--border-glass) !important;
    }

    /* Modern Glass Cards for Metrics */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        border-color: var(--primary) !important;
        box-shadow: 0 12px 40px 0 var(--primary-glow) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Action Buttons for Dark Mode */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s ease !important;
        border: 1px solid var(--border-glass) !important;
        background: rgba(30, 41, 59, 0.8) !important;
        color: #f1f5f9 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        border-color: var(--primary) !important;
        background: rgba(56, 189, 248, 0.15) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px var(--primary-glow) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
        box-shadow: 0 6px 25px rgba(56, 189, 248, 0.5) !important;
    }

    /* Expanders for Dark Mode */
    [data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(12px) !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stExpander"],
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] [data-testid="stExpanderSummary"],
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: transparent !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stExpander"] summary {
        border-radius: 12px !important;
        border: none !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stExpander"] summary:hover {
        background-color: rgba(30, 41, 59, 0.5) !important;
        color: var(--primary) !important;
    }

    /* Tabs for Dark Mode */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.7) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        border: 1px solid var(--border-glass) !important;
        gap: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.18) !important;
        color: #ffffff !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    /* Form Inputs for Dark Mode */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }
    .stTextInput > div > div > input:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px var(--primary-glow) !important;
    }

    /* Typography & Headings */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        color: #ffffff !important;
    }
    .gradient-text {
        background: linear-gradient(135deg, #ffffff 20%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
    }
    h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
        color: #f1f5f9 !important;
        letter-spacing: -0.02em !important;
    }

    /* Command Frames & Border Containers for Dark Mode */
    .command-frame,
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.5rem !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35) !important;
        margin-bottom: 1.25rem !important;
    }
    .command-frame-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .command-frame-title {
        color: #f8fafc;
    }

    /* High-Tech KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(16, 24, 40, 0.85) 0%, rgba(10, 15, 28, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        backdrop-filter: blur(16px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }
    .kpi-card-cyan {
        border-top: 3px solid #38bdf8;
    }
    .kpi-card-red {
        border-top: 3px solid #ef4444;
    }
    .kpi-card-purple {
        border-top: 3px solid #a855f7;
    }
    .kpi-card-amber {
        border-top: 3px solid #f59e0b;
    }
    .kpi-label {
        color: #94a3b8;
    }
    .kpi-value {
        color: #ffffff;
    }
    .kpi-subtext {
        color: #64748b;
    }
    .kpi-card > div {
        color: #ffffff;
    }

    /* Telemetry Ribbon */
    .telemetry-bar {
        background: rgba(10, 15, 26, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
    }
    .telemetry-item {
        color: #94a3b8;
    }
    .telemetry-value {
        color: #f1f5f9;
    }

    /* Hero Banner for Dark Mode */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    .hero-banner h1, .hero-banner p {
        color: #f8fafc !important;
    }

    /* Badges for Dark Mode */
    .badge-p1-glow {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.3);
    }
    .badge-p2-glow {
        background: rgba(249, 115, 22, 0.2);
        color: #fdba74;
        border: 1px solid rgba(249, 115, 22, 0.4);
        box-shadow: 0 0 12px rgba(249, 115, 22, 0.3);
    }
    .badge-p3-glow {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-p4-glow {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* Alert and Information Containers */
    .incident-alert-card {
        background: rgba(20, 26, 44, 0.85);
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Sidebar Close Button for Dark Mode */
    [data-testid="stSidebarCollapseButton"] button,
    button[aria-label="Collapse sidebar"],
    button[aria-label="Close sidebar"] {
        color: #38bdf8 !important;
        background: rgba(30, 41, 59, 0.95) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover,
    button[aria-label="Collapse sidebar"]:hover,
    button[aria-label="Close sidebar"]:hover {
        background: rgba(56, 189, 248, 0.25) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.5) !important;
    }
    [data-testid="stSidebarCollapseButton"] button::after,
    button[aria-label="Collapse sidebar"]::after,
    button[aria-label="Close sidebar"]::after {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover::after,
    button[aria-label="Collapse sidebar"]:hover::after,
    button[aria-label="Close sidebar"]:hover::after {
        color: #ffffff !important;
    }
    [data-testid="stSidebarCollapseButton"] svg,
    button[aria-label="Collapse sidebar"] svg,
    button[aria-label="Close sidebar"] svg {
        fill: #38bdf8 !important;
        stroke: #38bdf8 !important;
        color: #38bdf8 !important;
    }
    """


def apply_custom_theme():
    """
    Injects high-end, responsive custom styling supporting dynamic Dark and Light themes.
    Also renders the theme toggle control inside the sidebar.
    """
    mode = get_theme_mode()
    theme_css = get_theme_css(mode)

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

    /* Shared Typography and Base Layout */
    summary, summary p, summary div, summary span {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-feature-settings: normal !important;
    }}

    code, pre {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        padding-top: 0.5rem;
    }}

    /* Enhanced Sidebar Navigation Links */
    [data-testid="stSidebarNav"] {{
        border-top: 1px solid var(--border-glass) !important;
        padding-top: 0.6rem !important;
    }}
    [data-testid="stSidebarNav"] li {{
        margin-bottom: 3px !important;
    }}
    [data-testid="stSidebarNav"] li a {{
        border-radius: 10px !important;
        padding: 8px 14px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border-left: 3px solid transparent !important;
    }}
    [data-testid="stSidebarNav"] li a:hover {{
        border-left: 3px solid var(--primary) !important;
        background: rgba(56, 189, 248, 0.1) !important;
        transform: translateX(4px) !important;
    }}
    [data-testid="stSidebarNav"] a[aria-current="page"] {{
        border-left: 3px solid var(--primary) !important;
        background: rgba(56, 189, 248, 0.16) !important;
    }}
    /* Cleanly replace plain 'app' link title with '🚨 Command Gateway' */
    [data-testid="stSidebarNav"] li:first-child a span {{
        display: none !important;
    }}
    [data-testid="stSidebarNav"] li:first-child a::after {{
        content: "🚨 Command Gateway" !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: var(--text-primary) !important;
    }}

    /* Prevent icon ligature strings from rendering as raw text */
    details summary span span span,
    summary span span span,
    details summary > span > span,
    summary > span > span {{
        display: none !important;
        font-size: 0 !important;
        line-height: 0 !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    details summary, summary {{
        list-style: none !important;
    }}
    details summary::-webkit-details-marker {{
        display: none !important;
    }}
    details summary > span, summary > span {{
        display: flex !important;
        align-items: center !important;
    }}
    details summary > span::before, summary > span::before {{
        content: "▸";
        font-size: 1.15rem;
        color: var(--primary);
        margin-right: 8px;
        display: inline-block;
        transition: transform 0.2s ease;
        line-height: 1;
    }}
    details[open] summary > span::before {{
        transform: rotate(90deg) !important;
    }}

    /* Header settings: pointer-events none ensures sidebar toggle clicks work */
    header[data-testid="stHeader"], .stAppHeader {{
        background: transparent !important;
        pointer-events: none !important;
    }}

    #MainMenu,
    footer,
    .stDeployButton,
    [data-testid="stStatusWidget"] {{
        display: none !important;
    }}

    header[data-testid="stHeader"] > * {{
        pointer-events: none !important;
    }}

    /* Floating 'Menu' button when sidebar is collapsed */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {{
        display: block !important;
        visibility: visible !important;
        position: fixed !important;
        top: 14px !important;
        left: 16px !important;
        z-index: 9999999 !important;
        pointer-events: auto !important;
    }}

    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    button[aria-label="Expand sidebar"],
    button[aria-label="Open sidebar"] {{
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1.5px solid #38bdf8 !important;
        border-radius: 8px !important;
        padding: 6px 14px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.5) !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        transition: all 0.2s ease !important;
    }}

    [data-testid="collapsedControl"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    button[aria-label="Expand sidebar"]:hover,
    button[aria-label="Open sidebar"]:hover {{
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6) !important;
        transform: translateY(-1px) !important;
        color: #ffffff !important;
    }}

    [data-testid="collapsedControl"] button::after,
    [data-testid="stSidebarCollapsedControl"] button::after,
    button[aria-label="Expand sidebar"]::after,
    button[aria-label="Open sidebar"]::after {{
        content: "Menu" !important;
        display: inline-block !important;
        color: #ffffff !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        margin-left: 4px !important;
    }}

    [data-testid="collapsedControl"] svg,
    button[aria-label="Expand sidebar"] svg,
    button[aria-label="Open sidebar"] svg {{
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }}

    /* Sidebar collapse button setup */
    [data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
        visibility: visible !important;
        z-index: 1000005 !important;
        pointer-events: auto !important;
    }}

    [data-testid="stSidebarCollapseButton"] button,
    button[aria-label="Collapse sidebar"],
    button[aria-label="Close sidebar"] {{
        display: flex !important;
        align-items: center !important;
        gap: 4px !important;
        visibility: visible !important;
        opacity: 1 !important;
        border-radius: 8px !important;
        padding: 4px 10px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
    }}

    [data-testid="stSidebarCollapseButton"] button::after,
    button[aria-label="Collapse sidebar"]::after,
    button[aria-label="Close sidebar"]::after {{
        content: "Close" !important;
        display: inline-block !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        margin-left: 2px !important;
    }}

    /* Common Card and Component Layouts */
    .command-frame-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
    }}
    .command-frame-title {{
        font-size: 1.05rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .kpi-label {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .kpi-value {{
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0.35rem 0 0.15rem 0;
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1.1;
    }}
    .kpi-subtext {{
        font-size: 0.8rem;
        font-weight: 500;
    }}

    .telemetry-bar {{
        padding: 0.75rem 1.25rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }}
    .telemetry-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
    }}
    .telemetry-value {{
        font-weight: 600;
    }}

    /* Pulse animation */
    @keyframes pulse-emerald {{
        0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
        70% {{ box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}
    .pulse-dot-emerald {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        display: inline-block;
        animation: pulse-emerald 2s infinite;
    }}

    /* Badges */
    .cyber-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .stAlert {{
        border-radius: 12px !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-glass) !important;
    }}

    /* Download button */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35) !important;
    }}
    .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
        box-shadow: 0 6px 20px rgba(52, 211, 153, 0.5) !important;
        transform: translateY(-2px) !important;
    }}

    /* Dynamic Theme-Specific Rules */
    {theme_css}
    </style>
    """, unsafe_allow_html=True)

    # Render theme mode toggle in sidebar
    render_theme_toggle()
