from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Project Path & Module Imports ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ml_model.predict import predict_with_probability
    from ml_model.train_model import run_training
    from rule_based_system.rules import assess_patient
    from utils.data_processing import (
        FEATURE_COLS,
        TARGET_COL,
        process_heart_disease_data,
        load_dataset,
        handle_missing_values,
    )
except ImportError:
    st.error("Model modules not found. Ensure the directory structure matches your original project.")

ML_METRICS_PATH = ROOT_DIR / "reports" / "ml_metrics.json"
EXPERT_METRICS_PATH = ROOT_DIR / "reports" / "expert_metrics.json"

# ── Dashboard Palette ──────────────────────────────────────────────────────────
SIDEBAR_BG   = "#343d4c"
SIDEBAR_TEXT = "#adb5bd"
D_GREEN      = "#28d094"
D_PURPLE     = "#666ee8"
D_BLUE       = "#1e9ff2"
D_ORANGE     = "#ff9149"
BG_LIGHT     = "#eef3f8"
BORDER       = "#d7e0ea"
CARD_BG      = "#f7fafd"
CARD_SHADOW  = "0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)"
GRAY         = "#4b5563"
MAIN_TEXT    = "#1f2937"

GENERAL_PAGES = ["Dashboard", "Risk Prediction", "Data Analysis"]
MODEL_PAGES = ["Model Comparison", "Expert System Rules", "Machine Learning Model"]
PAGE_ICONS = {
    "Dashboard": "📊",
    "Risk Prediction": "🩺",
    "Data Analysis": "📈",
    "Model Comparison": "⚖️",
    "Expert System Rules": "🧠",
    "Machine Learning Model": "💻",
}

EXPERT_RULES_CATALOG = [
    {"Level": "High", "Rule": "High cholesterol (>240) with age >50"},
    {"Level": "High", "Rule": "High BP (>140) with exercise-induced angina"},
    {"Level": "Low", "Rule": "Good exercise capacity + no angina + young age"},
    {"Level": "High", "Rule": "Multiple risks: high cholesterol + BP + fasting blood sugar"},
    {"Level": "High", "Rule": "Significant chest pain with ST depression >1.5"},
    {"Level": "High", "Rule": "2+ major vessels colored by fluoroscopy"},
    {"Level": "Moderate", "Rule": "Thalassemia defect detected"},
    {"Level": "Low", "Rule": "Young age with normal indicators"},
    {"Level": "Low", "Rule": "Normal ECG with no exercise angina"},
    {"Level": "High", "Rule": "Flat slope with high ST depression (>2.0)"},
    {"Level": "Moderate", "Rule": "Female with chest pain symptoms"},
    {"Level": "High", "Rule": "Male >55 years with exercise-induced angina"},
]

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartDx - Admin Panel",
    page_icon=":heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

def resolve_network_ip() -> str:
    """Return a reachable LAN IP address for this machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


@st.cache_resource(show_spinner=False)
def print_runtime_urls_once() -> None:
    port = int(st.get_option("server.port") or os.environ.get("STREAMLIT_SERVER_PORT", "8501"))
    local_url = f"http://localhost:{port}"
    network_url = f"http://{resolve_network_ip()}:{port}"
    print(f"Local URL: {local_url}")
    print(f"Network URL: {network_url}")


print_runtime_urls_once()
# ── Complete Global CSS ────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif !important;
    }}
    [data-testid="stAppViewContainer"] {{
        background: {BG_LIGHT};
    }}
    [data-testid="stMain"] {{
        color: {MAIN_TEXT};
    }}
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] li,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6 {{
        color: {MAIN_TEXT};
    }}
    [data-testid="stMain"] .stCaption,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] small {{
        color: {GRAY} !important;
    }}
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Sidebar Navigation Styles */
    [data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG} !important; border-right: 1px solid {BORDER} !important; }}
    [data-testid="stSidebar"] > div:first-child {{ background-color: {SIDEBAR_BG} !important; }}
    [data-testid="stSidebarContent"] {{ background-color: {SIDEBAR_BG} !important; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: {SIDEBAR_TEXT} !important; }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
        background: transparent !important;
        color: {SIDEBAR_TEXT} !important;
        padding: 10px 15px !important;
        border-radius: 4px !important;
        margin-bottom: 4px;
        transition: 0.2s;
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
        background: rgba(255,255,255,0.05) !important;
        color: #fff !important;
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
        background: rgba(255,255,255,0.1) !important;
        color: #fff !important;
        border-left: 3px solid {D_BLUE};
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        color: {SIDEBAR_TEXT} !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 9px 10px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        font-size: 14px !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255,255,255,0.06) !important;
        color: #fff !important;
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: rgba(255,255,255,0.12) !important;
        color: #fff !important;
        border-left: 3px solid {D_BLUE} !important;
    }}

    /* Card & Metric Component Styles */
    .metric-card {{
        border-radius: 8px;
        padding: 22px;
        color: white;
        box-shadow: {CARD_SHADOW};
    }}
    .content-card {{
        background: linear-gradient(180deg, #fbfdff 0%, {CARD_BG} 100%);
        padding: 24px;
        border-radius: 8px;
        border: 1px solid {BORDER};
        box-shadow: {CARD_SHADOW};
        color: {MAIN_TEXT};
        margin-bottom: 24px;
    }}
    .section-title {{ font-size: 16px; font-weight: 700; color: {MAIN_TEXT}; margin-bottom: 4px; }}
    .section-sub {{ font-size: 13px; color: {GRAY}; margin-bottom: 16px; }}

    /* Risk Pills */
    .pill-high  {{ background:#FCEBEB; color:#A32D2D; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }}
    .pill-med   {{ background:#FAEEDA; color:#BA7517; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }}
    .pill-low   {{ background:#EAF3DE; color:#639922; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }}

    /* Forms & Buttons */
    .stButton > button {{
        background: {D_BLUE} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
    }}
    [data-testid="stFormSubmitButton"] button {{
        background: #ffffff !important;
        color: {MAIN_TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }}
    [data-testid="stFormSubmitButton"] button:hover {{
        background: #f8fbff !important;
        color: {D_BLUE} !important;
        border-color: {D_BLUE} !important;
    }}
    [data-testid="stForm"] button,
    [data-testid="stForm"] button[kind="primary"] {{
        background: #ffffff !important;
        color: {MAIN_TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stForm"] button:hover {{
        background: #f8fbff !important;
        color: {D_BLUE} !important;
        border-color: {D_BLUE} !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        color: {MAIN_TEXT} !important;
    }}
    .stSelectbox div[data-baseweb="select"] span {{
        color: {MAIN_TEXT} !important;
    }}
    .stNumberInput input, .stTextInput input, .stTextArea textarea, .stSelectbox input {{
        color: {MAIN_TEXT} !important;
    }}
    div[data-baseweb="popover"] > div {{
        background: #ffffff !important;
    }}
    div[data-baseweb="popover"] [role="listbox"],
    div[data-baseweb="popover"] [role="listbox"] ul,
    div[data-baseweb="popover"] [role="listbox"] li,
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {{
        background: #ffffff !important;
        color: {MAIN_TEXT} !important;
    }}
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] li:hover {{
        background: #f3f7fc !important;
        color: {MAIN_TEXT} !important;
    }}
    div[data-baseweb="popover"] [aria-selected="true"],
    div[data-baseweb="popover"] [role="option"][aria-selected="true"] {{
        background: #eaf4ff !important;
        color: {MAIN_TEXT} !important;
    }}
    div[data-baseweb="popover"] * {{
        color: {MAIN_TEXT} !important;
    }}
    .stDataFrame {{
        background: #ffffff !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    .light-table-wrap {{
        background: #ffffff;
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: auto;
        margin-bottom: 16px;
    }}
    .light-table {{
        width: 100%;
        border-collapse: collapse;
        color: {MAIN_TEXT};
        background: #ffffff;
        font-size: 13px;
    }}
    .light-table th {{
        background: #f3f7fc;
        color: {MAIN_TEXT};
        border: 1px solid {BORDER};
        padding: 7px 10px;
        text-align: left;
        position: sticky;
        top: 0;
        z-index: 1;
    }}
    .light-table td {{
        background: #ffffff;
        color: {MAIN_TEXT};
        border: 1px solid {BORDER};
        padding: 7px 10px;
    }}
    .light-table tr:nth-child(even) td {{
        background: #fbfdff;
    }}
    /* Plotly text visibility fix */
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text,
    .js-plotly-plot .plotly .legendtext,
    .js-plotly-plot .plotly .gtitle,
    .js-plotly-plot .plotly .annotation-text,
    .js-plotly-plot .plotly text {{
        fill: {MAIN_TEXT} !important;
        color: {MAIN_TEXT} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helper Functions ───────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    cleaned_path = ROOT_DIR / "data" / "cleaned_data.csv"
    if cleaned_path.exists():
        return pd.read_csv(cleaned_path)
    return process_heart_disease_data()

@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    raw_path = str(ROOT_DIR / "data" / "raw_data.csv")
    return handle_missing_values(load_dataset(raw_path))

def load_json(path: Path) -> dict | None:
    if not path.exists(): return None
    with path.open("r", encoding="utf-8") as f: return json.load(f)

def ensure_model_ready() -> None:
    if not (ROOT_DIR / "ml_model" / "decision_tree_model.pkl").exists():
        run_training()

def load_trained_model():
    model_path = ROOT_DIR / "ml_model" / "decision_tree_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)

def styled_metric(label, value, delta, color):
    # Determine arrow & clean string if a numeric delta is passed
    delta_str = str(delta)
    if delta_str.startswith("-"):
        arrow = "&darr;"
        delta_str = delta_str.replace("-", "")
    elif delta_str == "-" or delta_str == "":
        arrow = ""
        delta_str = ""
    else:
        arrow = "&uarr;"
        delta_str = delta_str.replace("+", "")
        
    st.markdown(
        f"""
        <div class="metric-card" style="background-color: {color};">
            <div style="font-size: 11px; text-transform: uppercase; opacity: 0.8; font-weight: 500;">{label}</div>
            <div style="font-size: 28px; font-weight: 600; margin: 6px 0;">{value}</div>
            <div style="font-size: 12px; opacity: 0.9;">{arrow} {delta_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def chart_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=12, color=MAIN_TEXT),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=MAIN_TEXT),
        ),
        polar=dict(
            radialaxis=dict(tickfont=dict(color=MAIN_TEXT)),
            angularaxis=dict(tickfont=dict(color=MAIN_TEXT)),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f1f1", zeroline=False, tickfont=dict(color=MAIN_TEXT), title_font=dict(color=MAIN_TEXT))
    fig.update_yaxes(showgrid=True, gridcolor="#f1f1f1", tickfont=dict(color=MAIN_TEXT), title_font=dict(color=MAIN_TEXT))
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=MAIN_TEXT),
            title=dict(font=dict(color=MAIN_TEXT)),
        )
    )
    return fig

def render_light_table(df: pd.DataFrame, height: int) -> None:
    table_html = df.to_html(index=True, border=0, classes="light-table")
    st.markdown(
        f'<div class="light-table-wrap" style="max-height:{height}px;">{table_html}</div>',
        unsafe_allow_html=True,
    )

def page_label(page_name: str) -> str:
    icon = PAGE_ICONS.get(page_name, "•")
    return f"{icon}  {page_name}"

def arrange_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    base_order = [col for col in FEATURE_COLS if col in df.columns]
    ordered = base_order.copy()
    if TARGET_COL in df.columns:
        ordered.append(TARGET_COL)
    extra_cols = sorted([col for col in df.columns if col not in ordered])
    return df[ordered + extra_cols]

# ── Sidebar Layout ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 0 30px;">
            <div style="width:40px;height:40px;background:{D_BLUE};border-radius:10px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:18px;box-shadow:0 6px 12px rgba(30,159,242,.25);">&#10084;</div>
            <span style="font-size:24px;font-weight:700;color:white;line-height:1;">HeartDx</span>
        </div>
        """, unsafe_allow_html=True
    )

    if "page" not in st.session_state:
        st.session_state.page = GENERAL_PAGES[0]

    st.markdown(
        f'<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:{SIDEBAR_TEXT};margin:0 0 6px;">General</div>',
        unsafe_allow_html=True,
    )
    for p in GENERAL_PAGES:
        if st.button(page_label(p), key=f"nav_general_{p}", use_container_width=True, type="primary" if st.session_state.page == p else "secondary"):
            st.session_state.page = p

    st.markdown(
        f"""
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:14px 0;"/>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:{SIDEBAR_TEXT};margin:0 0 6px;">Models</div>',
        unsafe_allow_html=True,
    )
    for p in MODEL_PAGES:
        if st.button(page_label(p), key=f"nav_models_{p}", use_container_width=True, type="primary" if st.session_state.page == p else "secondary"):
            st.session_state.page = p

page = st.session_state.page

# ── Page Header Helper ─────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:28px;">
            <h1 style="font-size:24px;font-weight:600;color:#1a1a18;margin:0 0 4px;">{title}</h1>
            <p style="font-size:13px;color:{GRAY};margin:0;">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Dashboard":
    page_header("Analytics Dashboard", "System Overview & Key Performance Indicators")
    df = load_dashboard_data()
    total = len(df)
    positives = int(df[TARGET_COL].sum()) if TARGET_COL in df.columns else 0
    ratio = positives / total if total else 0

    # -- Top Row: Vivid Metric Cards --
    c1, c2, c3, c4 = st.columns(4)
    with c1: styled_metric("Total Records", f"{total:,}", "12.5%", D_GREEN)
    with c2: styled_metric("Positive Cases", f"{positives:,}", "8.2%", D_PURPLE)
    with c3: styled_metric("Positive Rate", f"{ratio:.1%}", "5.7%", D_BLUE)
    with c4: styled_metric("Features Used", str(len(FEATURE_COLS)), "2.1%", D_ORANGE)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Middle Row: Core Charts --
    col_left, col_right = st.columns([1.7, 1])

    with col_left:
        st.markdown('<p class="section-title">Feature Correlation with Disease</p><p class="section-sub">Top 10 features by Pearson correlation</p>', unsafe_allow_html=True)
        correlation = df.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=True)
        corr_df = correlation.tail(10).reset_index()
        corr_df.columns = ["feature", "abs_corr"]
        fig_corr = px.bar(corr_df, x="abs_corr", y="feature", orientation="h", color_discrete_sequence=[D_BLUE])
        st.plotly_chart(chart_layout(fig_corr, height=300), width="stretch")

    with col_right:
        st.markdown('<p class="section-title">Disease Distribution</p><p class="section-sub">Ratio of healthy vs symptomatic</p>', unsafe_allow_html=True)
        dist = df[TARGET_COL].value_counts().reset_index()
        dist.columns = ['Status', 'count']
        dist['Status'] = dist['Status'].map({0: "Healthy", 1: "Disease"})
        fig_pie = go.Figure(go.Pie(labels=dist['Status'], values=dist['count'], hole=0.6, marker=dict(colors=[D_BLUE, D_PURPLE])))
        st.plotly_chart(chart_layout(fig_pie, height=300), width="stretch")

    # -- Correlation Heatmap --
    st.markdown('<p class="section-title">Correlation Heatmap</p><p class="section-sub">Relationship matrix for numeric clinical variables</p>', unsafe_allow_html=True)
    heatmap_cols = [c for c in df.select_dtypes(include="number").columns if c in df.columns]
    if heatmap_cols:
        heatmap_df = df[heatmap_cols].corr(numeric_only=True).round(2)
        fig_heat = px.imshow(
            heatmap_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(chart_layout(fig_heat, height=380), width="stretch")

    # -- Additional Dashboard Graphs --
    low_row, right_row = st.columns(2)
    with low_row:
        st.markdown('<p class="section-title">Age Distribution by Disease Status</p>', unsafe_allow_html=True)
        df_plot = df.copy()
        df_plot["Status"] = df_plot[TARGET_COL].map({0: "No Disease", 1: "Disease"})
        fig_age = px.histogram(
            df_plot,
            x="age",
            color="Status",
            barmode="overlay",
            color_discrete_map={"Disease": D_PURPLE, "No Disease": D_BLUE},
        )
        st.plotly_chart(chart_layout(fig_age, height=260), width="stretch")

    with right_row:
        st.markdown('<p class="section-title">Cholesterol vs Blood Pressure</p><p class="section-sub">Cluster view by diagnosis label</p>', unsafe_allow_html=True)
        scatter_df = df.copy()
        scatter_df["Status"] = scatter_df[TARGET_COL].map({0: "No Disease", 1: "Disease"})
        fig_scatter = px.scatter(
            scatter_df,
            x="trestbps",
            y="chol",
            color="Status",
            color_discrete_map={"Disease": D_PURPLE, "No Disease": D_BLUE},
            opacity=0.7,
        )
        st.plotly_chart(chart_layout(fig_scatter, height=260), width="stretch")

    st.markdown('<p class="section-title">Disease Rate by Age Group</p><p class="section-sub">Grouped prevalence overview</p>', unsafe_allow_html=True)
    age_group_df = df.copy()
    age_group_df["Age Group"] = pd.cut(
        age_group_df["age"],
        bins=[25, 35, 45, 55, 65, 100],
        labels=["25-35", "36-45", "46-55", "56-65", "66+"],
        include_lowest=True,
    )
    age_group_rate = (
        age_group_df.groupby("Age Group", observed=True)[TARGET_COL]
        .mean()
        .mul(100)
        .reset_index(name="Disease Rate (%)")
    )
    age_palette = {
        "25-35": D_BLUE,
        "36-45": D_PURPLE,
        "46-55": D_GREEN,
        "56-65": D_ORANGE,
        "66+": "#00b8d9",
    }
    fig_age_rate = px.bar(
        age_group_rate,
        x="Age Group",
        y="Disease Rate (%)",
        color="Age Group",
        category_orders={"Age Group": ["25-35", "36-45", "46-55", "56-65", "66+"]},
        color_discrete_map=age_palette,
    )
    fig_age_rate.update_layout(showlegend=False)
    st.plotly_chart(chart_layout(fig_age_rate, height=260), width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. RISK PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Risk Prediction":
    page_header("Risk Prediction", "Clinical Assessment via Hybrid Expert-ML System")
    ensure_model_ready()

    with st.form("risk_form"):
        st.markdown('<p class="section-title">Patient Clinical Profile</p>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Core Bio**")
            age = st.slider("Age", 20, 90, 45)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=0)
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x], index=2)
            trestbps = st.slider("Resting BP (mmHg)", 80, 220, 120)
            chol = st.slider("Cholesterol (mg/dl)", 100, 450, 200)
        with c2:
            st.markdown("**Tests**")
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG", [0, 1, 2], index=1)
            thalach = st.slider("Max Heart Rate", 60, 220, 160)
            exang = st.selectbox("Exercise-induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with c3:
            st.markdown("**Imaging**")
            oldpeak = st.slider("ST Depression", 0.0, 6.5, 0.0, 0.1)
            slope = st.selectbox("ST Slope", [0, 1, 2], index=2)
            ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4], index=0)
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], index=2)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Assessment")

    if submitted:
        patient = { "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs, 
                    "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal }
        
        expert_result = assess_patient(patient)
        ml_result = predict_with_probability(patient)
        
        r1, r2 = st.columns(2)
        with r1:
            risk = expert_result["risk_level"].lower()
            pill = "pill-high" if risk == "high" else ("pill-med" if risk == "medium" else "pill-low")
            st.markdown(f"""
                <div class="content-card">
                    <p class="section-title">Expert Rule Engine</p>
                    <div style="display:flex;align-items:center;gap:12px;margin:15px 0;">
                        <span style="font-size:24px;font-weight:600;">{expert_result['risk_level']}</span>
                        <span class="{pill}">Risk Level</span>
                    </div>
                    <p style="font-size:12px;color:{GRAY};">{expert_result['num_rules_matched']} rules triggered</p>
                </div>
            """, unsafe_allow_html=True)

        with r2:
            display_label = ml_result['label']
            ml_pill = "pill-high" if "High" in display_label else "pill-medium" if "Moderate" in display_label else "pill-low"
            
            st.markdown(f"""
                <div class="content-card">
                    <p class="section-title">Decision Tree Model</p>
                    <div style="display:flex;align-items:center;gap:12px;margin:15px 0;">
                        <span style="font-size:24px;font-weight:600;">{display_label}</span>
                        <span class="{ml_pill}">Risk Level</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Data Analysis":
    page_header("Data Exploration", "UCI Heart Disease Dataset Deep-Dive")
    df_before_scaling = load_dashboard_data()
    df = load_data()
    df_before_display = arrange_analysis_columns(df_before_scaling)
    df_after_display = arrange_analysis_columns(df)

    st.markdown('<p class="section-title">Data Before Scaling (First 20 Rows)</p>', unsafe_allow_html=True)
    render_light_table(df_before_display.head(20), height=260)

    st.markdown('<p class="section-title">Data After Scaling/Encoding (First 20 Rows)</p>', unsafe_allow_html=True)
    render_light_table(df_after_display.head(20), height=260)

    st.markdown('<p class="section-title">Data Describe (Before Scaling)</p>', unsafe_allow_html=True)
    render_light_table(df_before_scaling.describe(include="all").transpose(), height=360)

    st.markdown('<p class="section-title">Full Correlation Heatmap (22 Features)</p>', unsafe_allow_html=True)
    corr_matrix = df.corr()
    fig_corr = px.imshow(corr_matrix, color_continuous_scale="RdBu_r", text_auto=".1f", aspect="auto")
    fig_corr.update_layout(xaxis_nticks=len(corr_matrix.columns), yaxis_nticks=len(corr_matrix.columns))
    st.plotly_chart(chart_layout(fig_corr, height=600), width="stretch")

    available = [c for c in FEATURE_COLS if c in df_before_scaling.columns]
    feature = st.selectbox("Select metric to visualize", available)
    df_plot = df_before_scaling.copy()
    df_plot["Status"] = df_plot[TARGET_COL].map({0: "No Disease", 1: "Disease"})

    col_a, col_b = st.columns(2)
    with col_a:
        fig_h = px.histogram(df_plot, x=feature, color="Status", barmode="overlay", color_discrete_map={"Disease": D_PURPLE, "No Disease": D_BLUE})
        st.plotly_chart(chart_layout(fig_h), width="stretch")
    with col_b:
        fig_b = px.box(df_plot, y=feature, x="Status", color="Status", color_discrete_map={"Disease": D_PURPLE, "No Disease": D_BLUE})
        st.plotly_chart(chart_layout(fig_b), width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXPERT SYSTEM RULES
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Expert System Rules":
    page_header("Expert System Rules", "Knowledge base used by the hybrid clinical rule engine")
    rules_df = pd.DataFrame(EXPERT_RULES_CATALOG)

    c1, c2, c3 = st.columns(3)
    with c1: styled_metric("Total Rules", str(len(rules_df)), "-", D_BLUE)
    with c2: styled_metric("High Risk Rules", str(int((rules_df["Level"] == "High").sum())), "-", D_ORANGE)
    with c3: styled_metric("Low/Moderate Rules", str(int((rules_df["Level"] != "High").sum())), "-", D_GREEN)

    left, right = st.columns([1.7, 1])
    with left:
        st.markdown('<p class="section-title">Rule Catalog</p><p class="section-sub">Conditions used by the expert system</p>', unsafe_allow_html=True)
        render_light_table(rules_df, height=420)

    with right:
        st.markdown('<p class="section-title">Rule Distribution</p><p class="section-sub">By assessed risk level</p>', unsafe_allow_html=True)
        level_counts = rules_df["Level"].value_counts().reset_index()
        level_counts.columns = ["Level", "Count"]
        fig_levels = px.bar(
            level_counts,
            x="Level",
            y="Count",
            color="Level",
            color_discrete_map={"High": D_ORANGE, "Moderate": D_PURPLE, "Low": D_GREEN},
        )
        st.plotly_chart(chart_layout(fig_levels, height=320), width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MACHINE LEARNING MODEL
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Machine Learning Model":
    page_header("Machine Learning Model", "Decision Tree internals, parameters, and feature importance")
    ensure_model_ready()

    payload = load_json(ML_METRICS_PATH)
    model = load_trained_model()

    if not payload:
        st.warning("ML metrics not found. Run training first to populate this page.")
    else:
        ml_m = payload.get("metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1: styled_metric("Test Accuracy", f"{ml_m.get('test_accuracy', 0):.3f}", "-", D_BLUE)
        with c2: styled_metric("Test Precision", f"{ml_m.get('test_precision', 0):.3f}", "-", D_PURPLE)
        with c3: styled_metric("Test Recall", f"{ml_m.get('test_recall', 0):.3f}", "-", D_GREEN)
        with c4: styled_metric("Test F1", f"{ml_m.get('test_f1', 0):.3f}", "-", D_ORANGE)

        st.markdown('<p class="section-title">Best Hyperparameters</p>', unsafe_allow_html=True)
        st.json(payload.get("best_params", {}))

    if model is None:
        st.info("Trained model file is not available yet.")
    elif hasattr(model, "feature_importances_"):
        feature_names = list(getattr(model, "feature_names_in_", []))
        if not feature_names:
            processed_df = load_data()
            feature_names = [c for c in processed_df.columns if c != TARGET_COL]

        importances = list(model.feature_importances_)
        if feature_names and len(feature_names) == len(importances):
            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=False).head(15)
            st.markdown('<p class="section-title">Top Feature Importance</p><p class="section-sub">Most influential features in the Decision Tree model</p>', unsafe_allow_html=True)
            fig_imp = px.bar(
                imp_df.sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=[D_BLUE],
            )
            st.plotly_chart(chart_layout(fig_imp, height=420), width="stretch")

        cm = ml_m.get("confusion_matrix")
        if cm:
            st.markdown('<p class="section-title">Confusion Matrix Heatmap</p>', unsafe_allow_html=True)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", 
                               labels=dict(x="Predicted Label", y="True Label", color="Count"), 
                               x=["High Risk", "Normal"], y=["High Risk", "Normal"])
            st.plotly_chart(chart_layout(fig_cm, height=400), width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Model Comparison":
    page_header("Model Benchmarking", "Decision Tree vs Expert Rule Performance")
    ml_payload = load_json(ML_METRICS_PATH)
    expert_payload = load_json(EXPERT_METRICS_PATH)

    if not ml_payload or not expert_payload:
        st.warning("Comparison data not found. Run your training scripts first.")
    else:
        # Use .get() defensively to avoid KeyError issues and adapt if metrics are nested
        ml_m = ml_payload.get("metrics", ml_payload)
        ex_m = expert_payload.get("metrics", expert_payload)
        
        # Extract metrics safely with fallbacks just in case the JSON keys differ slightly
        ml_acc = ml_m.get('test_accuracy', ml_m.get('accuracy', 0.0))
        ml_prec = ml_m.get('test_precision', ml_m.get('precision', 0.0))
        ml_rec = ml_m.get('test_recall', ml_m.get('recall', 0.0))
        ml_f1 = ml_m.get('test_f1', ml_m.get('f1', 0.0))

        ex_acc = ex_m.get('accuracy', ex_m.get('test_accuracy', 0.0))
        ex_prec = ex_m.get('precision', ex_m.get('test_precision', 0.0))
        ex_rec = ex_m.get('recall', ex_m.get('test_recall', 0.0))
        ex_f1 = ex_m.get('f1', ex_m.get('test_f1', 0.0))
        
        # Metric score cards with Delta (Comparison vs Expert System)
        c1, c2, c3, c4 = st.columns(4)
        with c1: styled_metric("Accuracy", f"{ml_acc:.3f}", f"{ml_acc - ex_acc:+.3f}", D_BLUE)
        with c2: styled_metric("Precision", f"{ml_prec:.3f}", f"{ml_prec - ex_prec:+.3f}", D_PURPLE)
        with c3: styled_metric("Recall", f"{ml_rec:.3f}", f"{ml_rec - ex_rec:+.3f}", D_GREEN)
        with c4: styled_metric("F1-Score", f"{ml_f1:.3f}", f"{ml_f1 - ex_f1:+.3f}", D_ORANGE)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Radar Profile</p>', unsafe_allow_html=True)
        
        categories = ["F1-score", "Accuracy", "Precision", "Recall"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[ml_f1, ml_acc, ml_prec, ml_rec], 
            theta=categories, fill='toself', name='Decision Tree', line_color=D_BLUE
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[ex_f1, ex_acc, ex_prec, ex_rec], 
            theta=categories, fill='toself', name='Expert System', line_color=D_PURPLE
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], angle=90)))
        st.plotly_chart(chart_layout(fig_radar, height=400), width="stretch")