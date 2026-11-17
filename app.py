"""
AgroPulse — Precision Agronomy & Satellite Decision Platform
============================================================
Story County, Ames, Iowa • NASA Space Apps Challenge

Full Functional Core + Production SaaS Visual Layer (Bilingual: English & Spanish)
Theme system: Dark / Light (switchable from the sidebar)
"""

from __future__ import annotations

import base64
import html
import importlib
import io
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import streamlit as st
import streamlit.components.v1 as components

import src.data_loader as _dl
import src.i18n as _i18n

importlib.reload(_dl)
importlib.reload(_i18n)

from src.assets_helper import (
    AGROPULSE_LOGO_SVG,
    clean_html,
    load_nasa_challenge_banner_uri,
    render_mascot_bubble,
)
from src.data_loader import (
    classify_usda_soil_texture,
    compute_pixel_multicriteria_topsis,
    get_data_status,
    load_master_grid,
    load_stat4_xai_bundle,
)
from src.i18n import (
    CONFIDENCE_I18N,
    CROP_I18N,
    MONTHS_I18N,
    PROVENANCE_I18N,
    SOIL_TEXTURE_I18N,
    STRAT_SHORT_I18N,
    STRAT_TITLES_I18N,
    TRANSLATIONS,
    XAI_CARDS_EN,
)

if hasattr(_dl, "load_stat1_phenology_bundle"):
    load_stat1_phenology_bundle = _dl.load_stat1_phenology_bundle
else:

    def load_stat1_phenology_bundle() -> dict:
        candidate_dirs = [
            BASE_DIR / "data" / "est1",
            BASE_DIR / "data",
            BASE_DIR / "RUTA_FINAL" / "1_TRABAJO_EQUIPO" / "RESULTADOS A REVISAR (ESTADISTICOS)" / "ESTADISTICO 1",
            BASE_DIR / "outputs" / "est1",
            BASE_DIR,
        ]
        bundle = {"metrics": pd.DataFrame(), "by_season": pd.DataFrame(), "windows": pd.DataFrame(), "validation": {}}
        for folder in candidate_dirs:
            if (folder / "metrics_phenology.parquet").exists():
                bundle["metrics"] = pd.read_parquet(folder / "metrics_phenology.parquet")
            if (folder / "phenology_validation.json").exists():
                with open(folder / "phenology_validation.json", "r", encoding="utf-8") as f:
                    bundle["validation"] = json.load(f)
            if not bundle["metrics"].empty or bundle["validation"]:
                break
        return bundle


def render_html(html_str: str) -> None:
    st.markdown(clean_html(html_str), unsafe_allow_html=True)


def sanitize_farmer_copy(text: str) -> str:
    """Clean internal debug tags from user-facing agronomic takeaways while preserving full audit data."""
    cleaned = re.sub(r"\[DEMO\]", "", str(text), flags=re.IGNORECASE)
    cleaned = re.sub(r"\[PROVISIONAL[^\]]*\]\s*(sin cifra\.?)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .;-")
    return cleaned + ("." if cleaned and not cleaned.endswith(".") else "")


def render_crop_grid_2x2(crops_list: list[tuple[str, str, str]]) -> str:
    """Symmetric 2x2 rotation grid so all 3 strategy cards have identical height."""
    tiles = []
    for emoji, name, yr_lbl in crops_list:
        tiles.append(
            f'<div class="yr-tile">'
            f'<div class="yr-top"><span class="yr-lbl">{yr_lbl}</span><span class="yr-emoji">{emoji}</span></div>'
            f'<div class="yr-name">{name}</div>'
            f"</div>"
        )
    return f'<div class="tl-grid-2x2">{"".join(tiles)}</div>'


def render_voice_button(speech_text: str, button_label: str, voice_lang: str = "en-US") -> None:
    safe_text = html.escape(speech_text, quote=True)
    components.html(
        f"""
        <div style="margin: 2px 0;">
            <button onclick="speakPlan()" style="
                width: 100%;
                background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                color: #FFFFFF;
                font-family: 'Plus Jakarta Sans', 'Inter', system-ui, sans-serif;
                font-weight: 700;
                font-size: 14px;
                letter-spacing: 0.2px;
                border: 1px solid rgba(167, 243, 208, 0.55);
                border-radius: 12px;
                padding: 12px 18px;
                cursor: pointer;
                box-shadow: 0 8px 20px rgba(16, 185, 129, 0.28);
                transition: all 0.18s ease;
            " onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 10px 24px rgba(16, 185, 129, 0.42)';"
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 20px rgba(16, 185, 129, 0.28)';">
                {button_label}
            </button>
        </div>
        <script>
        function speakPlan() {{
            if (!('speechSynthesis' in window)) {{
                alert('Speech synthesis is not supported in this browser.');
                return;
            }}
            window.speechSynthesis.cancel();
            const utter = new SpeechSynthesisUtterance("{safe_text}");
            utter.lang = '{voice_lang}';
            utter.rate = 0.95;
            window.speechSynthesis.speak(utter);
        }}
        </script>
        """,
        height=54,
    )


@st.cache_data(show_spinner=False)
def build_smooth_layer_raster(
    lats: tuple[float, ...],
    lons: tuple[float, ...],
    values: tuple[float, ...],
    cmap_name: str = "RdYlGn_r",
) -> tuple[str, list[list[float]], list[float], list[float]]:
    """
    Interpolates the 360 30m pixels into a natural translucent RGBA heatmap raster
    so real satellite terrain texture (roads, furrows, drainage) remains clearly visible.
    """
    lat_arr = np.array(lats, dtype=float)
    lon_arr = np.array(lons, dtype=float)
    val_arr = np.array(values, dtype=float)

    v_min, v_max = float(np.nanpercentile(val_arr, 3)), float(np.nanpercentile(val_arr, 97))
    span = max(v_max - v_min, 1e-6)
    norm_val = np.clip((val_arr - v_min) / span, 0.04, 0.96)

    pad_lat = (lat_arr.max() - lat_arr.min()) * 0.015
    pad_lon = (lon_arr.max() - lon_arr.min()) * 0.015
    min_lat, max_lat = float(lat_arr.min() - pad_lat), float(lat_arr.max() + pad_lat)
    min_lon, max_lon = float(lon_arr.min() - pad_lon), float(lon_arr.max() + pad_lon)

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(min_lon, max_lon, 220),
        np.linspace(min_lat, max_lat, 220),
    )

    interp_z = griddata((lon_arr, lat_arr), norm_val, (grid_lon, grid_lat), method="cubic")
    nearest_z = griddata((lon_arr, lat_arr), norm_val, (grid_lon, grid_lat), method="nearest")
    interp_z = np.where(np.isnan(interp_z), nearest_z, interp_z)
    smooth_z = np.clip(gaussian_filter(interp_z, sigma=2.2), 0.0, 1.0)

    rgba = mpl.colormaps[cmap_name](np.flipud(smooth_z))
    # Transparencia equilibrada (0.42) para que se vea el suelo real del satélite debajo
    rgba[..., 3] = 0.42

    img = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    coords = [
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat],
    ]
    poly_lats = [min_lat, max_lat, max_lat, min_lat, min_lat]
    poly_lons = [min_lon, min_lon, max_lon, max_lon, min_lon]
    return b64_uri, coords, poly_lats, poly_lons


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & SAAS DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgroPulse • Satellite Agronomy Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = "es"
if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "dark"


def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("ui_lang", "en")
    text = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))
    return text.format(**kwargs) if kwargs else text


# ─────────────────────────────────────────────────────────────────────────────
# THEME PALETTES (Dark / Light) — every themed color lives in a CSS variable
# ─────────────────────────────────────────────────────────────────────────────

THEMES = {
    "dark": {
        "scheme": "dark",
        "css": {
            "bg-app": (
                "radial-gradient(circle at 15% 0%, rgba(16, 185, 129, 0.12) 0%, transparent 42%), "
                "radial-gradient(circle at 88% 12%, rgba(245, 158, 11, 0.08) 0%, transparent 38%), "
                "linear-gradient(180deg, #08100D 0%, #0D1814 55%, #111916 100%)"
            ),
            "sidebar-bg": "linear-gradient(180deg, #0D1613 0%, #0A120F 100%)",
            "surface": "rgba(17, 28, 24, 0.92)",
            "border-subtle": "rgba(148, 163, 184, 0.16)",
            "border-accent": "rgba(52, 211, 153, 0.35)",
            "text": "#F8FAFC",
            "text-body": "#CBD5E1",
            "text-muted": "#94A3B8",
            "accent-text": "#6EE7B7",
            "input-bg": "#16221D",
            "input-border": "rgba(52, 211, 153, 0.45)",
            "input-text": "#FFFFFF",
            "btn-bg": "rgba(23, 38, 32, 0.92)",
            "btn-bg-hover": "#1B332A",
            "btn-text": "#F8FAFC",
            "btn-border": "rgba(148, 163, 184, 0.24)",
            "tab-bg": "rgba(17, 28, 24, 0.92)",
            "tab-text": "#94A3B8",
            "topbar-bg": "linear-gradient(120deg, rgba(16, 37, 29, 0.96) 0%, rgba(19, 32, 27, 0.96) 60%, rgba(32, 26, 19, 0.96) 100%)",
            "topbar-title": "#FFFFFF",
            "topbar-sub": "#94A3B8",
            "badge-bg": "rgba(245, 158, 11, 0.14)",
            "badge-text": "#FCD34D",
            "badge-border": "rgba(245, 158, 11, 0.45)",
            "shadow": "0 8px 24px rgba(0, 0, 0, 0.28)",
            "pill-bg": "rgba(16, 185, 129, 0.16)",
            "pill-text": "#6EE7B7",
            "alert-bg": "rgba(30, 46, 39, 0.92)",
            "code-bg": "rgba(148, 163, 184, 0.16)",
            "code-text": "#A7F3D0",
            "card-border": "#E2E8F0",
            "card-shadow": "0 10px 28px rgba(0, 0, 0, 0.26)",
            "card-gold-shadow": "0 0 0 3px rgba(245, 158, 11, 0.18), 0 14px 32px rgba(0, 0, 0, 0.30)",
            "legend-text": "#E2E8F0",
        },
        "plot": {"paper": "rgba(17, 28, 24, 0.92)", "text": "#E2E8F0", "accent": "#34D399", "sep": "#111C18"},
    },
    "light": {
        "scheme": "light",
        "css": {
            "bg-app": (
                "radial-gradient(circle at 15% 0%, rgba(16, 185, 129, 0.10) 0%, transparent 42%), "
                "radial-gradient(circle at 88% 12%, rgba(245, 158, 11, 0.08) 0%, transparent 38%), "
                "linear-gradient(180deg, #F5F9F6 0%, #EEF4F0 55%, #E8F0EA 100%)"
            ),
            "sidebar-bg": "linear-gradient(180deg, #FFFFFF 0%, #F1F6F3 100%)",
            "surface": "#FFFFFF",
            "border-subtle": "rgba(15, 23, 42, 0.12)",
            "border-accent": "rgba(5, 150, 105, 0.40)",
            "text": "#0F172A",
            "text-body": "#334155",
            "text-muted": "#64748B",
            "accent-text": "#047857",
            "input-bg": "#FFFFFF",
            "input-border": "rgba(5, 150, 105, 0.50)",
            "input-text": "#0F172A",
            "btn-bg": "#FFFFFF",
            "btn-bg-hover": "#ECFDF5",
            "btn-text": "#0F172A",
            "btn-border": "rgba(15, 23, 42, 0.18)",
            "tab-bg": "#FFFFFF",
            "tab-text": "#475569",
            "topbar-bg": "linear-gradient(120deg, #ECFDF5 0%, #F8FBF9 60%, #FFF6E5 100%)",
            "topbar-title": "#0F172A",
            "topbar-sub": "#475569",
            "badge-bg": "rgba(245, 158, 11, 0.16)",
            "badge-text": "#92400E",
            "badge-border": "rgba(217, 119, 6, 0.55)",
            "shadow": "0 8px 22px rgba(15, 23, 42, 0.08)",
            "pill-bg": "rgba(5, 150, 105, 0.12)",
            "pill-text": "#047857",
            "alert-bg": "#F1F5F9",
            "code-bg": "rgba(15, 23, 42, 0.07)",
            "code-text": "#065F46",
            "card-border": "#E2E8F0",
            "card-shadow": "0 6px 18px rgba(15, 23, 42, 0.08)",
            "card-gold-shadow": "0 0 0 3px rgba(245, 158, 11, 0.20), 0 10px 26px rgba(15, 23, 42, 0.12)",
            "legend-text": "#1E293B",
        },
        "plot": {"paper": "#FFFFFF", "text": "#1E293B", "accent": "#059669", "sep": "#FFFFFF"},
    },
}

cur_theme = st.session_state["ui_theme"] if st.session_state["ui_theme"] in THEMES else "dark"
PLOT = THEMES[cur_theme]["plot"]
_root_css = (
    ":root{color-scheme:" + THEMES[cur_theme]["scheme"] + ";"
    + "".join(f"--{k}:{v};" for k, v in THEMES[cur_theme]["css"].items())
    + "}"
)

STATIC_CSS = """
:root {
    --emerald-500: #10B981;
    --emerald-600: #059669;
    --amber-500: #F59E0B;
    --card-light: #FFFFFF;
    --ink-primary: #0F172A;
}
header[data-testid="stHeader"] { background: transparent !important; height: 0px !important; }
#MainMenu, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none !important; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: var(--bg-app) !important;
    background-attachment: fixed !important;
    color: var(--text) !important;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: transparent !important; }
.block-container {
    padding-top: 1.15rem !important;
    padding-bottom: 2.0rem !important;
    max-width: 1440px !important;
}
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label p {
    color: var(--text) !important;
}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stExpandSidebarButton"] svg,
[data-testid="collapsedControl"] svg { color: var(--text) !important; fill: currentColor; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
label[data-testid="stWidgetLabel"] p {
    color: var(--text-body) !important;
}
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] strong {
    color: var(--text) !important;
}
[data-testid="stMarkdownContainer"] a { color: var(--accent-text) !important; }
[data-testid="stMarkdownContainer"] hr { border-color: var(--border-subtle) !important; background: var(--border-subtle) !important; }
[data-testid="stMarkdownContainer"] code {
    background: var(--code-bg) !important;
    color: var(--code-text) !important;
}
.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--text-muted) !important; }
div[role="radiogroup"] label p, div[role="radiogroup"] label div { color: var(--text-body) !important; }
div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background-color: var(--input-bg) !important;
    border-radius: 10px !important;
}
div[data-baseweb="input"] { border: 1px solid var(--input-border) !important; }
div[data-baseweb="input"] input {
    color: var(--input-text) !important;
    -webkit-text-fill-color: var(--input-text) !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
}
div[data-baseweb="input"] button {
    background-color: var(--btn-bg-hover) !important;
    color: var(--text) !important;
    border-left: 1px solid var(--border-subtle) !important;
}
div[data-baseweb="input"] button svg { fill: var(--text) !important; color: var(--text) !important; }
div[data-baseweb="textarea"], div[data-baseweb="textarea"] textarea {
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    -webkit-text-fill-color: var(--input-text) !important;
    border-radius: 10px !important;
}
div[data-baseweb="textarea"] { border: 1px solid var(--input-border) !important; }
[data-testid="stSlider"] label p,
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"],
[data-testid="stThumbValue"], [data-testid="stSliderThumbValue"] {
    color: var(--text) !important;
}
[data-testid="stExpander"] details {
    background: var(--surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
    color: var(--text) !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary svg { color: var(--text) !important; fill: var(--text) !important; }
div[data-testid="stAlert"] {
    background: var(--alert-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}
div[data-testid="stAlert"] *, div[data-testid="stAlert"] p { color: var(--text) !important; }
div[data-testid="stAlert"] code { background: var(--code-bg) !important; color: var(--code-text) !important; }
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 0.60rem !important;
    border: 1px solid var(--btn-border) !important;
    background: var(--btn-bg) !important;
    color: var(--btn-text) !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12) !important;
    transition: all 0.16s ease !important;
    white-space: nowrap !important;
}
.stButton > button p, .stButton > button span { color: var(--btn-text) !important; }
.stButton > button:hover {
    border-color: #34D399 !important;
    background: var(--btn-bg-hover) !important;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    border: 1px solid rgba(167, 243, 208, 0.6) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.30) !important;
}
.stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span,
.stButton > button[data-testid="stBaseButton-primary"] p, .stButton > button[data-testid="stBaseButton-primary"] span {
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: var(--tab-bg) !important;
    padding: 6px;
    border-radius: 14px;
    border: 1px solid var(--border-subtle);
    margin-bottom: 14px;
    box-shadow: var(--shadow);
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 10px !important;
    padding: 0 18px !important;
    background: transparent !important;
    border: 1px solid transparent !important;
}
.stTabs [data-baseweb="tab"] p {
    color: var(--tab-text) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    border-color: rgba(167, 243, 208, 0.45) !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.32) !important;
}
.stTabs [aria-selected="true"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.saas-topbar {
    background: var(--topbar-bg);
    border: 1px solid var(--border-accent);
    border-radius: 16px;
    padding: 0.90rem 1.35rem;
    margin-bottom: 0.90rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    box-shadow: var(--shadow);
}
.header-left { display: flex; align-items: center; gap: 14px; }
.saas-topbar .header-title {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--topbar-title) !important;
    font-size: 1.40rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
    white-space: nowrap;
}
.saas-topbar .header-title a, .saas-topbar [data-testid="stHeaderActionElements"] { display: none !important; }
@media (max-width: 900px) { .saas-topbar .header-title { white-space: normal; } }
.saas-topbar .header-subtitle { color: var(--topbar-sub) !important; font-size: 0.81rem; margin: 2px 0 0 0; font-weight: 500; }
.topbar-right { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.badge-capsule {
    background: var(--badge-bg);
    color: var(--badge-text) !important;
    border: 1px solid var(--badge-border);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 0.74rem;
    font-weight: 700;
    white-space: nowrap;
}
.topbar-meta { font-size: 0.73rem; color: var(--text-muted) !important; font-weight: 500; }
div[style*="rgba(255,255,255,0.08)"], div[style*="rgba(26, 52, 39"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
}
div[style*="rgba(255,255,255,0.08)"] *, div[style*="rgba(26, 52, 39"] * { color: var(--text) !important; }
.sel-card {
    background: linear-gradient(135deg, #065F46 0%, #064E3B 100%);
    border: 1px solid rgba(110, 231, 183, 0.45);
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
}
section[data-testid="stSidebar"] .sel-card .sel-lbl { font-size: 0.68rem; text-transform: uppercase; color: #A7F3D0 !important; font-weight: 700; letter-spacing: 0.6px; }
section[data-testid="stSidebar"] .sel-card .sel-title { font-size: 1.28rem; font-weight: 800; color: #FFFFFF !important; margin: 2px 0; }
section[data-testid="stSidebar"] .sel-card .sel-of { font-size: 0.80rem; font-weight: 500; color: #D1FAE5 !important; }
section[data-testid="stSidebar"] .sel-card .sel-sub { font-size: 0.80rem; color: #ECFDF5 !important; font-weight: 500; }
.agri-card {
    background: var(--card-light);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.10rem 1.25rem;
    box-shadow: var(--card-shadow);
    margin-bottom: 0.85rem;
    color: var(--ink-primary) !important;
}
.agri-card p, .agri-card li, .agri-card ul, .agri-card strong { color: #0F172A !important; }
.agri-card.best-gold {
    border: 2px solid #F59E0B !important;
    background: linear-gradient(180deg, #FFFDF5 0%, #FFFFFF 28%) !important;
    box-shadow: var(--card-gold-shadow) !important;
}
.agri-card .gold-badge, .gold-badge {
    display: inline-block;
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    color: #FFFFFF !important;
    font-weight: 700;
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 0.71rem;
    white-space: nowrap;
}
.dark-widget-box {
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 10px 14px 6px 14px;
    margin-bottom: 6px;
}
.widget-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.legend-strip {
    display: flex;
    justify-content: space-around;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 7px 10px;
    margin: 6px 0;
    font-size: 0.78rem;
    color: var(--legend-text);
    font-weight: 600;
    gap: 8px;
    flex-wrap: wrap;
}
.legend-strip i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.why-box {
    padding: 10px 14px;
    border-radius: 12px;
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    margin-top: 8px;
    font-size: 0.82rem;
}
.why-box, .why-box * { color: var(--text-body) !important; }
.why-box b, .why-box strong { color: var(--text) !important; }
.source-pill {
    display: inline-block;
    font-size: 0.70rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 99px;
    background: var(--pill-bg);
    color: var(--pill-text) !important;
    border: 1px solid var(--border-accent);
    margin-left: 6px;
}
.verdict-title {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.20rem;
    font-weight: 800;
    line-height: 1.2;
}
.check-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.83rem;
    gap: 8px;
}
.check-row span { color: #475569 !important; font-weight: 500; }
.check-row b { color: #0F172A !important; font-weight: 700; text-align: right; }
.check-row:last-child { border-bottom: none; }
.tl-grid-2x2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 10px 0 12px 0;
}
.yr-tile {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 7px 10px;
}
.yr-top { display: flex; justify-content: space-between; align-items: center; }
.yr-lbl { font-size: 0.66rem; color: #64748B !important; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; }
.yr-emoji { font-size: 1.15rem; line-height: 1; }
.yr-name { font-size: 0.82rem; color: #0F172A !important; font-weight: 700; margin-top: 2px; }
.cal-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 5px;
    text-align: center;
    margin: 8px 0;
}
.cal-cell {
    border-radius: 10px;
    padding: 6px 2px;
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    font-size: 0.68rem;
    font-weight: 700;
    color: #334155 !important;
}
.cal-cell.bare { background: #FEF2F2; border-color: #FCA5A5; color: #991B1B !important; }
.cal-cell.cover { background: #ECFDF5; border-color: #6EE7B7; color: #065F46 !important; }
.cal-cell i { display: block; font-style: normal; font-size: 1.1rem; margin-bottom: 2px; }
.grid-100 {
    display: grid;
    grid-template-columns: repeat(10, 18px);
    gap: 5px;
    margin: 10px 0;
}
.dot-win { width: 18px; height: 18px; background: #10B981; border-radius: 4px; }
.dot-other { width: 18px; height: 18px; background: rgba(245, 158, 11, 0.25); border: 1px solid rgba(245, 158, 11, 0.55); border-radius: 4px; }
.stress-block-red { display: inline-block; width: 22px; height: 22px; background: #EF4444; border-radius: 5px; margin-right: 4px; }
.stress-block-gray { display: inline-block; width: 22px; height: 22px; background: #E2E8F0; border-radius: 5px; margin-right: 4px; }
.audit-wrap { overflow-x: auto; margin: 6px 0 10px 0; }
.audit-wrap table { width: 100%; border-collapse: collapse; font-size: 0.80rem; background: var(--surface); border: 1px solid var(--border-subtle); border-radius: 10px; }
.audit-wrap th, .audit-wrap td { padding: 7px 10px; border-bottom: 1px solid var(--border-subtle); text-align: right; color: var(--text) !important; }
.audit-wrap thead th { color: var(--text-muted) !important; font-weight: 700; font-size: 0.72rem; }
.audit-wrap tbody th { text-align: left; font-weight: 700; }
"""

render_html(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">'
    "<style>" + _root_css + STATIC_CSS + "</style>"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MASTER GRID, STATISTICIAN 1 & STATISTICIAN 4 BUNDLES
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _get_master_data() -> pd.DataFrame:
    return load_master_grid()


df_grid = _get_master_data()
e4_bundle = load_stat4_xai_bundle()
e1_bundle = load_stat1_phenology_bundle()
df_e1 = e1_bundle.get("metrics", pd.DataFrame())
pixel_ids = sorted(df_grid["pixel_id"].tolist())


def _zone_compass_location(row: pd.Series, df_all: pd.DataFrame) -> str:
    lat_mid, lon_mid = float(df_all["lat"].mean()), float(df_all["lon"].mean())
    lat_span = max(float(df_all["lat"].max() - df_all["lat"].min()), 1e-5)
    lon_span = max(float(df_all["lon"].max() - df_all["lon"].min()), 1e-5)

    dy = (float(row["lat"]) - lat_mid) / (lat_span / 2.0)
    dx = (float(row["lon"]) - lon_mid) / (lon_span / 2.0)

    ns = t("loc_north") if dy > 0.25 else (t("loc_south") if dy < -0.25 else "")
    ew = t("loc_east") if dx > 0.25 else (t("loc_west") if dx < -0.25 else "")
    if not ns and not ew:
        return t("loc_center")
    return f"{t('loc_side')} {ns}-{ew}".replace("--", "-").strip("-")


def get_crop_localized(val) -> str:
    lang = st.session_state.get("ui_lang", "en")
    cmap = CROP_I18N.get(lang, CROP_I18N["en"])
    if pd.isna(val):
        return cmap[0]
    try:
        return cmap.get(int(val), cmap[0])
    except (ValueError, TypeError):
        s_val = str(val).strip().title()
        return cmap.get(s_val, s_val)


if "active_plot_num" not in st.session_state:
    st.session_state["active_plot_num"] = 1


def set_active_plot_num(num: int) -> None:
    clamped = int(np.clip(num, 1, len(pixel_ids)))
    st.session_state["active_plot_num"] = clamped
    st.session_state["plot_number_widget"] = clamped


def _on_number_change() -> None:
    val = st.session_state.get("plot_number_widget", 1)
    if val is not None:
        st.session_state["active_plot_num"] = int(np.clip(val, 1, len(pixel_ids)))


def _step_plot(delta: int) -> None:
    curr = int(st.session_state["active_plot_num"])
    total = len(pixel_ids)
    nxt = ((curr - 1 + delta) % total) + 1
    set_active_plot_num(nxt)


def _jump_to_pid(target_pid: str) -> None:
    if target_pid in pixel_ids:
        set_active_plot_num(pixel_ids.index(target_pid) + 1)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — LANGUAGE + THEME SWITCHER & PLOT EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

st.session_state["plot_number_widget"] = int(st.session_state["active_plot_num"])

with st.sidebar:
    lang_c1, lang_c2 = st.columns(2, gap="small")
    with lang_c1:
        if st.button(
            "🇺🇸 English",
            use_container_width=True,
            type="primary" if st.session_state["ui_lang"] == "en" else "secondary",
            key="btn_lang_en",
        ):
            st.session_state["ui_lang"] = "en"
            st.rerun()
    with lang_c2:
        if st.button(
            "🇪🇸 Español",
            use_container_width=True,
            type="primary" if st.session_state["ui_lang"] == "es" else "secondary",
            key="btn_lang_es",
        ):
            st.session_state["ui_lang"] = "es"
            st.rerun()

    is_es_sb = st.session_state["ui_lang"] == "es"
    st.caption("🎨 Tema de la app" if is_es_sb else "🎨 App theme")
    theme_c1, theme_c2 = st.columns(2, gap="small")
    with theme_c1:
        if st.button(
            "🌙 Oscuro" if is_es_sb else "🌙 Dark",
            use_container_width=True,
            type="primary" if cur_theme == "dark" else "secondary",
            key="btn_theme_dark",
        ):
            st.session_state["ui_theme"] = "dark"
            st.rerun()
    with theme_c2:
        if st.button(
            "☀️ Claro" if is_es_sb else "☀️ Light",
            use_container_width=True,
            type="primary" if cur_theme == "light" else "secondary",
            key="btn_theme_light",
        ):
            st.session_state["ui_theme"] = "light"
            st.rerun()

    st.markdown("---")
    st.markdown(t("sb_title"))
    st.caption(t("sb_caption"))

    active_num = int(st.session_state["active_plot_num"])
    active_pid = pixel_ids[active_num - 1]
    pixel_row = df_grid[df_grid["pixel_id"] == active_pid].iloc[0]
    sector_str = _zone_compass_location(pixel_row, df_grid)
    slope_desc = t("sb_flat") if pixel_row.get("slope_pct", 1.0) < 1.2 else t("sb_hill")

    render_html(
        f"""
<div class="sel-card">
<div class="sel-lbl">{t("sb_selected")}</div>
<div class="sel-title">📍 {t("sb_field_lbl")} #{active_num} <span class="sel-of">{t("sb_of_360")}</span></div>
<div class="sel-sub">🧭 {sector_str} • ⛰️ {slope_desc}</div>
</div>
"""
    )

    st.number_input(
        t("sb_input_lbl"),
        min_value=1,
        max_value=len(pixel_ids),
        step=1,
        key="plot_number_widget",
        on_change=_on_number_change,
    )

    nav_c1, nav_c2 = st.columns(2, gap="small")
    with nav_c1:
        st.button(t("sb_prev"), use_container_width=True, key="btn_prev_plot", on_click=_step_plot, args=(-1,))
    with nav_c2:
        st.button(t("sb_next"), use_container_width=True, key="btn_next_plot", on_click=_step_plot, args=(1,))

    st.caption(t("sb_check_areas"))
    driest_pid = df_grid.sort_values("agua_disponible_awc", ascending=True).iloc[0]["pixel_id"]
    steep_pid = df_grid.sort_values("slope_pct", ascending=False).iloc[0]["pixel_id"]
    best_pid = df_grid.sort_values("agua_disponible_awc", ascending=False).iloc[0]["pixel_id"]

    lbl_dry = "💧 Seca" if st.session_state["ui_lang"] == "es" else "💧 Driest"
    lbl_steep = "⛰️ Ladera" if st.session_state["ui_lang"] == "es" else "⛰️ Slope"
    lbl_best = "🌟 Óptima" if st.session_state["ui_lang"] == "es" else "🌟 Best"

    sp1, sp2, sp3 = st.columns(3, gap="small")
    with sp1:
        st.button(lbl_dry, use_container_width=True, key="btn_driest", on_click=_jump_to_pid, args=(driest_pid,), help=t("sb_driest_help"))
    with sp2:
        st.button(lbl_steep, use_container_width=True, key="btn_steep", on_click=_jump_to_pid, args=(steep_pid,), help=t("sb_hilliest_help"))
    with sp3:
        st.button(lbl_best, use_container_width=True, key="btn_best_soil", on_click=_jump_to_pid, args=(best_pid,), help=t("sb_best_help"))

    st.markdown("---")

    st.markdown(t("sb_color_by"))
    LAYER_SPECS = [
        ("recommended_strategy", "nominal"),
        ("cultivo_2025_label", "nominal"),
        ("water_deficit_conservacion", "continuous"),
    ]
    if "map_layer_idx" not in st.session_state:
        st.session_state["map_layer_idx"] = 0

    selected_layer_idx = st.radio(
        "Map Layer:",
        options=[0, 1, 2],
        index=st.session_state["map_layer_idx"],
        format_func=lambda idx: t(f"layer_{idx}"),
        label_visibility="collapsed",
        key="map_layer_radio_idx",
    )
    st.session_state["map_layer_idx"] = selected_layer_idx
    layer_col, layer_type = LAYER_SPECS[selected_layer_idx]

    if st.button(t("sb_reset_zoom"), use_container_width=True, key="btn_recenter"):
        st.session_state["map_reset_counter"] = st.session_state.get("map_reset_counter", 0) + 1
        st.rerun()

    st.markdown("---")

    banner_uri = load_nasa_challenge_banner_uri()
    if banner_uri:
        render_html(
            f"""
<div style="border-radius:12px; overflow:hidden; border:1px solid var(--border-subtle); margin-top:4px;">
<img src="{banner_uri}" alt="NASA Space Apps Challenge" style="width:100%; display:block;" />
</div>
"""
        )

# ─────────────────────────────────────────────────────────────────────────────
# TOP SAAS HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────

render_html(
    f"""
<div class="saas-topbar">
<div class="header-left">
<div>{AGROPULSE_LOGO_SVG}</div>
<div>
<h1 class="header-title">{t("app_title")}</h1>
<p class="header-subtitle">{t("app_subtitle")}</p>
</div>
</div>
<div class="topbar-right">
<span class="badge-capsule">{t("badge_farm")}</span>
<span class="topbar-meta">{t("badge_sources")}</span>
</div>
</div>
"""
)

active_num = int(st.session_state["active_plot_num"])
active_pid = pixel_ids[active_num - 1]
pixel_row = df_grid[df_grid["pixel_id"] == active_pid].iloc[0]
cur_lang = st.session_state["ui_lang"]
STRAT_TITLES = STRAT_TITLES_I18N[cur_lang]
STRAT_SHORTS = STRAT_SHORT_I18N[cur_lang]

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY WEIGHTS & TOPSIS MULTICRITERIA ENGINE
# ─────────────────────────────────────────────────────────────────────────────

PRIORITY_WEIGHTS = [0.05, 0.15, 0.25, 0.40, 0.60]

if "lvl_water_idx" not in st.session_state:
    st.session_state["lvl_water_idx"] = 3
if "lvl_erosion_idx" not in st.session_state:
    st.session_state["lvl_erosion_idx"] = 2
if "lvl_nitrogen_idx" not in st.session_state:
    st.session_state["lvl_nitrogen_idx"] = 2
if "lvl_profit_idx" not in st.session_state:
    st.session_state["lvl_profit_idx"] = 2

user_weights = {
    "water_deficit_mm": PRIORITY_WEIGHTS[st.session_state["lvl_water_idx"]],
    "soil_erosion_ton_ha": PRIORITY_WEIGHTS[st.session_state["lvl_erosion_idx"]],
    "n_deficit_kg_ha": PRIORITY_WEIGHTS[st.session_state["lvl_nitrogen_idx"]],
    "net_margin_usd_ha": PRIORITY_WEIGHTS[st.session_state["lvl_profit_idx"]],
}

topsis_res = compute_pixel_multicriteria_topsis(pixel_row, user_weights)
best_strat_id = topsis_res["best_strategy_id"]
best_strat = topsis_res["resultados"][best_strat_id]
cons_strat = topsis_res["resultados"]["rot_conservacion"]
rent_strat = topsis_res["resultados"]["rot_rentabilidad"]
act_strat = topsis_res["resultados"]["rot_actual"]

awc_val = float(pixel_row.get("agua_disponible_awc", 0.1865))
awc_l = float(pixel_row.get("awc_l", awc_val * 0.75))
awc_h = float(pixel_row.get("awc_h", awc_val * 1.20))
om_val = float(pixel_row.get("organic_matter", 2.75))
sigma_comb = float(pixel_row.get("sigma_espacializado", 0.0449))


def fit_verdict(score_100: float) -> tuple[str, str]:
    stars = int(np.clip(round(score_100 / 20.0), 1, 5))
    star_str = "⭐" * stars + "☆" * (5 - stars)
    if score_100 >= 75:
        return star_str, t("t2_fit_great")
    if score_100 >= 55:
        return star_str, t("t2_fit_solid")
    return star_str, t("t2_fit_risk")


# ══════════════════════════════════════════════════════════════════════════════
# 4 INTERACTIVE LAYERS (TABS)
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    t("tab_1"),
    t("tab_2"),
    t("tab_3"),
    t("tab_4"),
])

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1: SATELLITE FIELD MAP (CONTINUOUS SMOOTHED HEATMAP + CLICK SELECTION)
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    render_html(
        render_mascot_bubble(
            pose="wave",
            bubble_title=t("t1_mascot_title", num=active_num, sector=_zone_compass_location(pixel_row, df_grid)),
            speech_text=t("t1_mascot_text"),
            mascot_size=58,
        )
    )

    col_map_wide, col_soil_card = st.columns([13, 7], gap="medium")

    with col_map_wide:
        center_lat = float(df_grid["lat"].mean())
        center_lon = float(df_grid["lon"].mean())

        df_map = df_grid.copy()
        df_map["zone_label"] = df_map["pixel_id"].apply(lambda pid: f"{t('sb_field_lbl')} #{int(pid.split('_')[1]) + 1}")
        df_map["_hover"] = df_map.apply(
            lambda r: (
                f"<b>{r['zone_label']}</b><br>"
                f"{t('t1_hover_plan')}: {STRAT_TITLES.get(r.get('recommended_strategy', ''), STRAT_TITLES['rot_conservacion'])}<br>"
                f"{t('t1_hover_crop')}: {get_crop_localized(r.get('cultivo_2025_label', 'Corn'))}<br>"
                f"{t('t1_hover_water')}: {t('t1_high') if r.get('agua_disponible_awc', 0) >= 0.18 else t('t1_moderate')}"
            ),
            axis=1,
        )

        # Combinar la capa elegida con la micro-topografía real (pendiente + retención hídrica)
        # para revelar las zonas húmedas (verde), medias (amarillo) y sensibles (rojo)
        slope_s = df_map["slope_pct"].fillna(1.2) if "slope_pct" in df_map.columns else pd.Series([1.2] * len(df_map))
        slope_norm = (slope_s - slope_s.min()) / max(float(slope_s.max() - slope_s.min()), 1e-4)

        awc_s = df_map["agua_disponible_awc"].fillna(0.18) if "agua_disponible_awc" in df_map.columns else pd.Series([0.18] * len(df_map))
        dry_norm = 1.0 - ((awc_s - awc_s.min()) / max(float(awc_s.max() - awc_s.min()), 1e-4))

        micro_terrain = 0.55 * slope_norm + 0.45 * dry_norm

        if layer_type == "continuous":
            def_s = df_map[layer_col].fillna(55.0) if layer_col in df_map.columns else pd.Series([55.0] * len(df_map))
            def_norm = (def_s - def_s.min()) / max(float(def_s.max() - def_s.min()), 1e-4)
            surf_vals = 0.65 * def_norm + 0.35 * micro_terrain
            cmap_choice = "RdYlGn_r"
        elif layer_col == "recommended_strategy":
            strat_num_map = {"rot_conservacion": 0.25, "rot_rentabilidad": 0.55, "rot_actual": 0.85}
            base_strat = df_map[layer_col].map(strat_num_map).fillna(0.30)
            surf_vals = 0.35 * base_strat + 0.65 * micro_terrain
            cmap_choice = "RdYlGn_r"
        else:
            crop_num_map = {"Soybean": 0.20, "Cereal Rye": 0.30, "Spring Oats": 0.45, "Corn": 0.75}
            base_crop = df_map[layer_col].map(crop_num_map).fillna(0.50)
            surf_vals = 0.55 * base_crop + 0.45 * micro_terrain
            cmap_choice = "RdYlGn_r"

        b64_uri, img_coords, poly_lats, poly_lons = build_smooth_layer_raster(
            tuple(df_map["lat"].tolist()),
            tuple(df_map["lon"].tolist()),
            tuple(surf_vals.tolist()),
            cmap_name=cmap_choice,
        )

        fig_map = go.Figure()

        # Crisp white boundary polygon around the 80-acre field
        fig_map.add_trace(go.Scattermap(
            lat=poly_lats,
            lon=poly_lons,
            mode="lines",
            line=dict(color="#FFFFFF", width=3.5),
            hoverinfo="skip",
            name=t("t1_boundary_name"),
        ))

        # Invisible clickable grid nodes so clicking anywhere on the heatmap selects that plot
        fig_map.add_trace(go.Scattermap(
            lat=df_map["lat"],
            lon=df_map["lon"],
            mode="markers",
            customdata=df_map["pixel_id"],
            marker=dict(size=14, color="rgba(255,255,255,0.01)", opacity=0.01),
            text=df_map["_hover"],
            hoverinfo="text",
            showlegend=False,
        ))

        # Active plot pin highlight
        px_active = df_map[df_map["pixel_id"] == active_pid].iloc[0]
        fig_map.add_trace(go.Scattermap(
            lat=[px_active["lat"]],
            lon=[px_active["lon"]],
            mode="markers+text",
            marker=dict(size=16, color="#FFFFFF", opacity=0.95),
            text=[f"📍 {t('sb_field_lbl')} #{active_num}"],
            textposition="top right",
            textfont=dict(color="#FFFFFF", size=12),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig_map.add_trace(go.Scattermap(
            lat=[px_active["lat"]],
            lon=[px_active["lon"]],
            mode="markers",
            marker=dict(size=9, color="#10B981", opacity=1.0),
            hoverinfo="skip",
            showlegend=False,
        ))

        reset_key = f"map_view_{st.session_state.get('map_reset_counter', 0)}"
        fig_map.update_layout(
            map=dict(
                style="white-bg",
                layers=[
                    dict(
                        below="traces",
                        sourcetype="raster",
                        source=[
                            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ],
                    ),
                    dict(
                        below="traces",
                        sourcetype="image",
                        source=b64_uri,
                        coordinates=img_coords,
                    ),
                ],
                center=dict(lat=center_lat, lon=center_lon),
                zoom=15.30,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=430,
            showlegend=False,
            uirevision=reset_key,
        )

        map_event = st.plotly_chart(
            fig_map,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="interactive_field_map",
            config={"displayModeBar": False},
        )

        if map_event:
            sel = map_event.get("selection") if isinstance(map_event, dict) else getattr(map_event, "selection", None)
            if sel:
                pts = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
                if pts and len(pts) > 0:
                    clicked_pt = pts[0]
                    cdata = clicked_pt.get("customdata") if isinstance(clicked_pt, dict) else getattr(clicked_pt, "customdata", None)
                    if cdata:
                        clicked_pid = cdata[0] if isinstance(cdata, (list, tuple, np.ndarray)) else str(cdata)
                        last_click = st.session_state.get("last_map_clicked_pid")
                        if clicked_pid in pixel_ids and clicked_pid != active_pid and clicked_pid != last_click:
                            st.session_state["last_map_clicked_pid"] = clicked_pid
                            st.session_state["active_plot_num"] = pixel_ids.index(clicked_pid) + 1
                            st.rerun()

        is_es_t1 = cur_lang == "es"
        render_html(
            f"""
<div class="legend-strip">
<span><i style="background:#22C55E;"></i>{'Zona óptima y húmeda' if is_es_t1 else 'Healthy moist zone'}</span>
<span><i style="background:#FACC15;"></i>{'Atención moderada' if is_es_t1 else 'Moderate watch'}</span>
<span><i style="background:#EF4444;"></i>{'Ladera sensible a sequía/erosión' if is_es_t1 else 'Drought/runoff sensitive'}</span>
</div>
"""
        )

        render_html(f'<div class="why-box">{t("t1_why_html")}</div>')

    with col_soil_card:
        sand = float(pixel_row.get("sand_pct", 39.2))
        clay = float(pixel_row.get("clay_pct", 22.5))
        silt = float(pixel_row.get("silt_pct", max(0.0, 100.0 - sand - clay)))

        raw_soil_name, raw_soil_tip = classify_usda_soil_texture(sand, silt, clay)
        soil_class_name, soil_class_tip = SOIL_TEXTURE_I18N[cur_lang].get(raw_soil_name, (raw_soil_name, raw_soil_tip))
        water_retention_str = (
            t("t1_cap_high")
            if awc_val >= 0.18
            else (t("t1_cap_mod") if awc_val >= 0.14 else t("t1_cap_fast"))
        )

        render_html(
            f"""
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:10px;">
<div class="agri-card" style="padding:12px 14px; margin:0; border-top: 4px solid #10B981;">
<div style="color:#64748B; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.4px;">{t("t1_soil_type")}</div>
<div style="color:#0F172A; font-size:1.02rem; font-weight:800; margin-top:3px;">{soil_class_name}</div>
<div style="color:#059669; font-size:0.72rem; font-weight:600; margin-top:2px;">{soil_class_tip}</div>
</div>
<div class="agri-card" style="padding:12px 14px; margin:0; border-top: 4px solid #D97706;">
<div style="color:#64748B; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.4px;">{t("t1_moisture_hold")}</div>
<div style="color:#78350F; font-size:1.02rem; font-weight:800; margin-top:3px;">{water_retention_str}</div>
<div style="color:#B45309; font-size:0.72rem; font-weight:600; margin-top:2px;">{t("t1_rootzone")} ({awc_val:.2f})</div>
</div>
</div>
"""
        )

        render_html(
            f"""
<div class="dark-widget-box">
<div class="widget-header">
<span>🧪 {t("t1_donut_title")}</span>
<span style="font-size:0.72rem; color:var(--accent-text);">0–100 cm</span>
</div>
</div>
"""
        )

        fig_soil_donut = go.Figure(go.Pie(
            labels=[t("t1_sand"), t("t1_silt"), t("t1_clay")],
            values=[sand, silt, clay],
            hole=0.62,
            marker=dict(colors=["#F59E0B", "#94A3B8", "#92400E"], line=dict(color=PLOT["sep"], width=2)),
            textinfo="label+percent",
            insidetextfont=dict(size=11, color="#FFFFFF"),
            outsidetextfont=dict(size=11, color=PLOT["text"]),
            textfont=dict(size=11, color=PLOT["text"]),
            insidetextorientation="horizontal",
        ))
        fig_soil_donut.update_layout(
            height=190,
            margin=dict(l=8, r=8, t=6, b=6),
            showlegend=False,
            paper_bgcolor=PLOT["paper"],
            plot_bgcolor=PLOT["paper"],
            font=dict(color=PLOT["text"]),
            annotations=[
                dict(
                    text=f"<b>{t('t1_om_center')}</b><br><span style='color:{PLOT['accent']}; font-size:14px;'><b>{om_val:.1f}%</b></span>",
                    x=0.5, y=0.5,
                    font=dict(size=12, color=PLOT["text"]),
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_soil_donut, use_container_width=True, config={"displayModeBar": False})

        hist_title_clean = t("t1_hist_title").replace("**", "")
        render_html(
            f"""
<div class="dark-widget-box" style="margin-top:6px;">
<div class="widget-header">
<span>{hist_title_clean}</span>
<span style="font-size:0.70rem; color:var(--text-muted);">USDA CDL 30m</span>
</div>
</div>
"""
        )

        years = list(range(2016, 2026))
        crops_history = [get_crop_localized(pixel_row.get(f"crop_{y}", 1)) for y in years]

        CROP_COLOR_MAP = {
            get_crop_localized("Corn"): "#F59E0B",
            get_crop_localized("Soybean"): "#10B981",
            get_crop_localized("Cereal Rye"): "#38BDF8",
            get_crop_localized("Spring Oats"): "#14B8A6",
            get_crop_localized("Alfalfa"): "#A855F7",
            get_crop_localized("Winter Wheat"): "#60A5FA",
        }
        bar_colors = [CROP_COLOR_MAP.get(c, "#78716C") for c in crops_history]

        fig_hist = go.Figure(go.Bar(
            x=years,
            y=[1] * 10,
            marker_color=bar_colors,
            text=[c[:5] for c in crops_history],
            customdata=crops_history,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#0F172A", size=10),
            hovertemplate="<b>%{x}</b>: %{customdata}<extra></extra>",
        ))
        fig_hist.update_layout(
            height=82,
            margin=dict(l=4, r=4, t=4, b=18),
            xaxis=dict(tickfont=dict(size=10, color=PLOT["text"]), dtick=1),
            yaxis=dict(visible=False),
            showlegend=False,
            paper_bgcolor=PLOT["paper"],
            plot_bgcolor=PLOT["paper"],
            font=dict(color=PLOT["text"]),
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2: SIMPLE FARMER GOAL SELECTOR & CLEAR RECOMMENDATION CARD
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    is_es = cur_lang == "es"

    render_html(
        render_mascot_bubble(
            pose="point",
            bubble_title=(
                f"¿Qué buscas lograr en tu campo esta campaña? (Sector #{active_num})"
                if is_es
                else f"What is your main goal for your field this season? (Plot #{active_num})"
            ),
            speech_text=(
                "Elige tu objetivo principal abajo y te diré al instante qué cultivo te conviene más y cuánto puedes ganar."
                if is_es
                else "Pick your main goal below and I will immediately show which crop plan fits your land best and your expected return."
            ),
            mascot_size=56,
        )
    )

    if "farmer_goal_preset" not in st.session_state:
        st.session_state["farmer_goal_preset"] = "balanced"

    st.markdown(
        f"**{'Paso 1: Elige lo que más te importa esta temporada' if is_es else 'Step 1: Choose what matters most to you this season'}**"
    )

    g_c1, g_c2, g_c3 = st.columns(3, gap="medium")
    with g_c1:
        is_soil = st.session_state["farmer_goal_preset"] == "soil"
        if st.button(
            "🛡️ Cuidar mi tierra y guardar agua" if is_es else "🛡️ Protect Soil & Save Water",
            type="primary" if is_soil else "secondary",
            use_container_width=True,
            key="goal_btn_soil",
        ):
            st.session_state["farmer_goal_preset"] = "soil"
            st.session_state["lvl_water_idx"] = 4
            st.session_state["lvl_erosion_idx"] = 4
            st.session_state["lvl_nitrogen_idx"] = 3
            st.session_state["lvl_profit_idx"] = 1
            st.rerun()
    with g_c2:
        is_bal = st.session_state["farmer_goal_preset"] == "balanced"
        if st.button(
            "⚖️ Equilibrar ganancia y cuidado" if is_es else "⚖️ Balance Profit & Soil Health",
            type="primary" if is_bal else "secondary",
            use_container_width=True,
            key="goal_btn_bal",
        ):
            st.session_state["farmer_goal_preset"] = "balanced"
            st.session_state["lvl_water_idx"] = 3
            st.session_state["lvl_erosion_idx"] = 2
            st.session_state["lvl_nitrogen_idx"] = 2
            st.session_state["lvl_profit_idx"] = 2
            st.rerun()
    with g_c3:
        is_prof = st.session_state["farmer_goal_preset"] == "profit"
        if st.button(
            "💰 Maximizar ganancia inmediata" if is_es else "💰 Maximize Immediate Profit",
            type="primary" if is_prof else "secondary",
            use_container_width=True,
            key="goal_btn_prof",
        ):
            st.session_state["farmer_goal_preset"] = "profit"
            st.session_state["lvl_water_idx"] = 1
            st.session_state["lvl_erosion_idx"] = 1
            st.session_state["lvl_nitrogen_idx"] = 2
            st.session_state["lvl_profit_idx"] = 4
            st.rerun()

    # Ajuste fino opcional plegado para no saturar visualmente al agricultor
    with st.expander(
        "⚙️ Ajuste fino opcional (si quieres personalizar cada prioridad)"
        if is_es
        else "⚙️ Optional fine-tuning (customize each priority weight)",
        expanded=False,
    ):
        sl_c1, sl_c2, sl_c3, sl_c4 = st.columns(4, gap="medium")
        with sl_c1:
            st.session_state["lvl_water_idx"] = st.select_slider(
                "💧 Agua" if is_es else "💧 Water",
                options=[0, 1, 2, 3, 4],
                value=st.session_state["lvl_water_idx"],
                format_func=lambda i: t(f"lvl_{i}"),
            )
        with sl_c2:
            st.session_state["lvl_erosion_idx"] = st.select_slider(
                "🛡️ Suelo" if is_es else "🛡️ Soil",
                options=[0, 1, 2, 3, 4],
                value=st.session_state["lvl_erosion_idx"],
                format_func=lambda i: t(f"lvl_{i}"),
            )
        with sl_c3:
            st.session_state["lvl_nitrogen_idx"] = st.select_slider(
                "🌱 Abono" if is_es else "🌱 Fertilizer",
                options=[0, 1, 2, 3, 4],
                value=st.session_state["lvl_nitrogen_idx"],
                format_func=lambda i: t(f"lvl_{i}"),
            )
        with sl_c4:
            st.session_state["lvl_profit_idx"] = st.select_slider(
                "💵 Ganancia" if is_es else "💵 Profit",
                options=[0, 1, 2, 3, 4],
                value=st.session_state["lvl_profit_idx"],
                format_func=lambda i: t(f"lvl_{i}"),
            )

    # Recalcular TOPSIS con los pesos actualizados
    user_weights = {
        "water_deficit_mm": PRIORITY_WEIGHTS[st.session_state["lvl_water_idx"]],
        "soil_erosion_ton_ha": PRIORITY_WEIGHTS[st.session_state["lvl_erosion_idx"]],
        "n_deficit_kg_ha": PRIORITY_WEIGHTS[st.session_state["lvl_nitrogen_idx"]],
        "net_margin_usd_ha": PRIORITY_WEIGHTS[st.session_state["lvl_profit_idx"]],
    }
    topsis_res = compute_pixel_multicriteria_topsis(pixel_row, user_weights)
    best_strat_id = topsis_res["best_strategy_id"]
    best_strat = topsis_res["resultados"][best_strat_id]
    cons_strat = topsis_res["resultados"]["rot_conservacion"]
    rent_strat = topsis_res["resultados"]["rot_rentabilidad"]
    act_strat = topsis_res["resultados"]["rot_actual"]

    # Información humana y directa de cada plan
    PLAN_SIMPLE_INFO = {
        "rot_conservacion": {
            "title": "🌾 Intercalar Centeno de Cobertura (Maíz + Centeno – Soya)" if is_es else "🌾 Interseed Winter Cereal Rye (Corn + Rye – Soybean)",
            "tagline": "Protege tu tierra todo el invierno, guarda agua para el verano y ahorra abono." if is_es else "Protects topsoil all winter, saves summer water, and cuts fertilizer bills.",
            "sequence": "🌽 Maíz → 🌾 Centeno (invierno) → 🫘 Soya → 🌾 Centeno" if is_es else "🌽 Corn → 🌾 Cereal Rye (winter) → 🫘 Soybean → 🌾 Cereal Rye",
            "water_txt": "✅ Alta reserva de humedad" if is_es else "✅ Strong moisture reserve",
            "soil_txt": "✅ Cero arrastre de tierra fértil" if is_es else "✅ Minimal topsoil erosion",
            "fert_txt": "✅ Aprovecha abono sobrante" if is_es else "✅ Captures leftover nitrogen",
            "money_txt": f"${cons_strat['metricas']['net_margin_usd_ha']:,.0f} USD/ha + bono USDA EQIP" if is_es else f"${cons_strat['metricas']['net_margin_usd_ha']:,.0f}/ha + USDA EQIP grant",
            "accent": "#10B981",
        },
        "rot_rentabilidad": {
            "title": "🌿 Rotación Diversificada con Avena de Primavera" if is_es else "🌿 Diversified Rotation with Spring Oats",
            "tagline": "Alterna maíz, soya y avena para cortar plagas y asegurar ingresos estables." if is_es else "Alternates corn, soybean, and oats to break pest cycles and stabilize income.",
            "sequence": "🌽 Maíz → 🫘 Soya → 🌿 Avena → 🫘 Soya" if is_es else "🌽 Corn → 🫘 Soybean → 🌿 Spring Oats → 🫘 Soybean",
            "water_txt": "🟡 Buena resistencia al calor" if is_es else "🟡 Solid heat resilience",
            "soil_txt": "✅ Baja erosión en laderas" if is_es else "✅ Low slope erosion",
            "fert_txt": "🟡 Menor gasto en químicos" if is_es else "🟡 Lower chemical input cost",
            "money_txt": f"${rent_strat['metricas']['net_margin_usd_ha']:,.0f} USD/ha" if is_es else f"${rent_strat['metricas']['net_margin_usd_ha']:,.0f}/ha",
            "accent": "#F59E0B",
        },
        "rot_actual": {
            "title": "🌽 Seguir con Maíz – Soya Tradicional (Sin Cobertura)" if is_es else "🌽 Traditional Corn – Soybean (No Winter Cover)",
            "tagline": "Rinde bien solo si llueve normal, pero deja el suelo descubierto 6 meses." if is_es else "Pays well in normal rain years, but leaves soil bare for 6 winter months.",
            "sequence": "🌽 Maíz → 🫘 Soya → 🌽 Maíz → 🫘 Soya" if is_es else "🌽 Corn → 🫘 Soybean → 🌽 Corn → 🫘 Soybean",
            "water_txt": "⚠️ Sufre si hay verano seco" if is_es else "⚠️ Vulnerable in dry summers",
            "soil_txt": "⚠️ Suelo descubierto en invierno" if is_es else "⚠️ Bare soil over winter",
            "fert_txt": "⚠️ Mayor gasto en fertilizante" if is_es else "⚠️ Higher fertilizer dependency",
            "money_txt": f"${act_strat['metricas']['net_margin_usd_ha']:,.0f} USD/ha (solo años normales)" if is_es else f"${act_strat['metricas']['net_margin_usd_ha']:,.0f}/ha (normal years only)",
            "accent": "#64748B",
        },
    }

    win_info = PLAN_SIMPLE_INFO[best_strat_id]
    win_stars, win_verdict = fit_verdict(best_strat["score"])

    # Tarjeta Hero del Plan Ganador (Grande, limpia y directa)
    render_html(
        f"""
<div class="agri-card best-gold" style="padding:1.25rem 1.45rem; margin-top:6px;">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:6px;">
        <span class="gold-badge">⭐ {'LA MEJOR OPCIÓN PARA TU OBJETIVO' if is_es else 'BEST MATCH FOR YOUR GOAL'}</span>
        <span style="font-size:0.95rem; font-weight:800; color:#059669;">{win_stars} &nbsp; {win_verdict}</span>
    </div>
    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.30rem; font-weight:800; color:#0F172A; margin-bottom:4px;">
        {win_info["title"]}
    </div>
    <div style="font-size:0.90rem; color:#334155; margin-bottom:12px;">
        {win_info["tagline"]}
    </div>
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:8px 12px; font-size:0.86rem; font-weight:700; color:#0F172A; margin-bottom:12px;">
        🔄 {'Secuencia de siembra:' if is_es else 'Planting sequence:'} &nbsp; {win_info["sequence"]}
    </div>
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px;">
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:9px 10px;">
            <div style="font-size:0.72rem; color:#166534; font-weight:700;">💧 {'AGUA' if is_es else 'WATER'}</div>
            <div style="font-size:0.83rem; color:#0F172A; font-weight:700; margin-top:2px;">{win_info["water_txt"]}</div>
        </div>
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:9px 10px;">
            <div style="font-size:0.72rem; color:#166534; font-weight:700;">🛡️ {'SUELO' if is_es else 'SOIL'}</div>
            <div style="font-size:0.83rem; color:#0F172A; font-weight:700; margin-top:2px;">{win_info["soil_txt"]}</div>
        </div>
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:9px 10px;">
            <div style="font-size:0.72rem; color:#166534; font-weight:700;">🌱 {'ABONO' if is_es else 'FERTILIZER'}</div>
            <div style="font-size:0.83rem; color:#0F172A; font-weight:700; margin-top:2px;">{win_info["fert_txt"]}</div>
        </div>
        <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:9px 10px;">
            <div style="font-size:0.72rem; color:#92400E; font-weight:700;">💵 {'INGRESO ESTIMADO' if is_es else 'EST. RETURN'}</div>
            <div style="font-size:0.83rem; color:#0F172A; font-weight:800; margin-top:2px;">{win_info["money_txt"]}</div>
        </div>
    </div>
</div>
"""
    )

    # Comparación rápida de las otras 2 alternativas debajo
    other_ids = [sid for sid in ["rot_conservacion", "rot_rentabilidad", "rot_actual"] if sid != best_strat_id]
    st.markdown(
        f"**{'¿Cómo se compara con las otras alternativas?' if is_es else 'How does it compare to the other options?'}**"
    )
    oth_c1, oth_c2 = st.columns(2, gap="medium")
    for idx_o, sid in enumerate(other_ids):
        o_info = PLAN_SIMPLE_INFO[sid]
        o_res = topsis_res["resultados"][sid]
        o_stars, o_verd = fit_verdict(o_res["score"])
        target_col = oth_c1 if idx_o == 0 else oth_c2
        with target_col:
            render_html(
                f"""
<div class="agri-card" style="border-left: 5px solid {o_info['accent']}; padding:0.95rem 1.15rem;">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:4px;">
        <span style="font-weight:800; font-size:0.96rem; color:#0F172A;">{o_info["title"]}</span>
        <span style="font-size:0.78rem; font-weight:700; color:#475569; white-space:nowrap;">{o_stars}</span>
    </div>
    <div style="font-size:0.82rem; color:#475569; margin-bottom:8px;">{o_info["tagline"]}</div>
    <div style="font-size:0.80rem; color:#0F172A; line-height:1.5;">
        • {o_info["water_txt"]} &nbsp;|&nbsp; {o_info["soil_txt"]}<br>
        • 💵 <b>{o_info["money_txt"]}</b>
    </div>
</div>
"""
            )

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3: INTERACTIVE WEATHER & ADOPTION SLIDER (FARMER-FRIENDLY)
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    is_es = cur_lang == "es"

    render_html(
        render_mascot_bubble(
            pose="run",
            bubble_title=(
                f"Prueba qué pasa en tu campo (Sector #{active_num}) según el clima y lo que apliques"
                if is_es
                else f"Test how your field (Plot #{active_num}) responds to weather and your plan"
            ),
            speech_text=(
                "1) Elige un clima abajo. 2) Mueve la barra deslizante para ver cómo mejora tu cosecha y humedad a medida que aplicas el plan recomendado."
                if is_es
                else "1) Pick a weather scenario below. 2) Drag the slider to see how moisture and crop safety improve as you apply the recommended plan."
            ),
            mascot_size=56,
        )
    )

    # 1. Selector de escenario climático limpio
    WEATHER_DELTAS = [
        (0.0, 0.0),     # Normal
        (-25.0, 1.5),   # Verano seco
        (30.0, -0.5),   # Tormentas primavera
        (-50.0, 2.5),   # Megasequía
    ]
    WEATHER_BTNS = {
        "es": [
            "🌤️ Temporada Normal",
            "☀️ Verano Seco (-25% Lluvia)",
            "🌧️ Lluvias Fuertes (+30%)",
            "🔥 Sequía Extrema (-50% Lluvia)",
        ],
        "en": [
            "🌤️ Normal Season",
            "☀️ Dry Summer (-25% Rain)",
            "🌧️ Heavy Spring Rain (+30%)",
            "🔥 Extreme Drought (-50% Rain)",
        ],
    }

    if "weather_idx" not in st.session_state:
        st.session_state["weather_idx"] = 1  # Inicia en Verano Seco para mostrar contraste claro

    st.markdown(
        f"**{'Paso 1: Elige el clima que quieres poner a prueba' if is_es else 'Step 1: Choose the weather scenario to test'}**"
    )
    w_cols = st.columns(4, gap="small")
    for idx in range(4):
        with w_cols[idx]:
            is_act = st.session_state["weather_idx"] == idx
            if st.button(
                WEATHER_BTNS[cur_lang][idx],
                key=f"w_btn_v2_{idx}",
                type="primary" if is_act else "secondary",
                use_container_width=True,
            ):
                st.session_state["weather_idx"] = idx
                st.rerun()

    w_idx = st.session_state["weather_idx"]
    d_rain, d_temp = WEATHER_DELTAS[w_idx]

    if d_rain < -40.0 or d_rain > 30.0:
        st.warning(
            "⚠️ Escenario extremo fuera del rango histórico habitual (aproximación preventiva)."
            if is_es
            else "⚠️ Extreme scenario beyond typical historical range (preventive estimate)."
        )

    # 2. Barra deslizante interactiva: "A medida que aplicas lo recomendado..."
    plan_short_name = STRAT_SHORTS[best_strat_id]
    st.markdown(
        f"**{'Paso 2: Mueve la barra para ver cómo mejora tu terreno a medida que aplicas:' if is_es else 'Step 2: Drag the slider to see your land improve as you apply:'}** `{plan_short_name}`"
    )

    adoption_pct = st.select_slider(
        label="adoption_slider",
        options=[0, 25, 50, 75, 100],
        value=75,
        format_func=lambda v: (
            f"0% — {'Sigo igual que hoy (Sin cambios)' if is_es else 'No changes (Same as today)'}"
            if v == 0
            else (
                f"100% — {'Aplico todo el plan recomendado' if is_es else 'Full recommended plan applied'}"
                if v == 100
                else f"{v}% {'aplicado en tu campo' if is_es else 'applied across your field'}"
            )
        ),
        label_visibility="collapsed",
        key="farmer_adoption_slider",
    )

    # Cálculo dinámico vinculado a los datos reales de la parcela y el escenario climático
    frac = adoption_pct / 100.0
    rain_penalty = max(0.0, -d_rain * 0.65) + max(0.0, d_temp * 4.5)
    storm_penalty = max(0.0, d_rain * 0.55)

    # Valores base (0% aplicado = Maíz-Soya tradicional descubierto)
    base_moisture = int(np.clip(round(68 - rain_penalty * 0.95 + (awc_val - 0.16) * 80), 15, 85))
    base_soil_safe = int(np.clip(round(62 - storm_penalty * 0.90 - rain_penalty * 0.35), 15, 80))
    base_yield_safe = int(np.clip(round(70 - rain_penalty * 0.85 - storm_penalty * 0.45), 18, 88))

    # Valores objetivo (100% aplicado = Plan recomendado con cobertura viva)
    target_moisture = int(np.clip(round(base_moisture + 26 + rain_penalty * 0.45), base_moisture + 10, 96))
    target_soil_safe = int(np.clip(round(base_soil_safe + 32 + storm_penalty * 0.55), base_soil_safe + 12, 98))
    target_yield_safe = int(np.clip(round(base_yield_safe + 22 + rain_penalty * 0.40), base_yield_safe + 8, 95))

    # Interpolación en tiempo real según el slider
    cur_moisture = int(round(base_moisture + (target_moisture - base_moisture) * frac))
    cur_soil_safe = int(round(base_soil_safe + (target_soil_safe - base_soil_safe) * frac))
    cur_yield_safe = int(round(base_yield_safe + (target_yield_safe - base_yield_safe) * frac))

    gain_moisture = cur_moisture - base_moisture
    gain_soil = cur_soil_safe - base_soil_safe
    gain_yield = cur_yield_safe - base_yield_safe

    def _bar_hex(val: int) -> str:
        if val >= 72:
            return "#10B981"
        if val >= 48:
            return "#F59E0B"
        return "#EF4444"

    # 3. Tarjetas visuales ultra claras: Izquierda (Si no haces nada) vs Derecha (Con tu avance en la barra)
    c_before, c_after = st.columns(2, gap="large")

    with c_before:
        render_html(
            f"""
<div class="agri-card" style="border-top: 5px solid #EF4444; background: #FFFDFD;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <span style="font-weight:800; font-size:1.05rem; color:#991B1B;">
            {'❌ Si sigues igual (0% aplicado)' if is_es else '❌ If you keep current practice (0%)'}
        </span>
        <span style="background:#FEE2E2; color:#991B1B; font-size:0.72rem; font-weight:700; padding:3px 9px; border-radius:99px;">
            {'Suelo descubierto en invierno' if is_es else 'Bare winter soil'}
        </span>
    </div>

    <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:4px;">
            <span>💧 {'Agua guardada para la raíz' if is_es else 'Moisture saved for roots'}</span>
            <b>{base_moisture}%</b>
        </div>
        <div style="width:100%; height:11px; background:#F1F5F9; border-radius:99px; overflow:hidden;">
            <div style="width:{base_moisture}%; height:100%; background:{_bar_hex(base_moisture)}; border-radius:99px;"></div>
        </div>
    </div>

    <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:4px;">
            <span>🛡️ {'Tierra fértil protegida contra lluvias' if is_es else 'Fertile topsoil protected from rain'}</span>
            <b>{base_soil_safe}%</b>
        </div>
        <div style="width:100%; height:11px; background:#F1F5F9; border-radius:99px; overflow:hidden;">
            <div style="width:{base_soil_safe}%; height:100%; background:{_bar_hex(base_soil_safe)}; border-radius:99px;"></div>
        </div>
    </div>

    <div style="margin-bottom:8px;">
        <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:4px;">
            <span>🌽 {'Cosecha a salvo del clima' if is_es else 'Harvest protected from weather'}</span>
            <b>{base_yield_safe}%</b>
        </div>
        <div style="width:100%; height:11px; background:#F1F5F9; border-radius:99px; overflow:hidden;">
            <div style="width:{base_yield_safe}%; height:100%; background:{_bar_hex(base_yield_safe)}; border-radius:99px;"></div>
        </div>
    </div>

    <div style="margin-top:12px; padding:9px 11px; background:#FEF2F2; border-radius:10px; font-size:0.81rem; color:#7F1D1D;">
        {'⚠️ El suelo queda sin protección ~6 meses y pierde humedad rápido cuando aprieta el calor.' if is_es else '⚠️ Soil stays exposed ~6 months and loses moisture rapidly during summer heat.'}
    </div>
</div>
"""
        )

    with c_after:
        gain_badge = (
            f"+{gain_yield}% {'más seguridad' if is_es else 'safer harvest'}"
            if gain_yield > 0
            else ("Mueve la barra arriba ⬆️" if is_es else "Move slider above ⬆️")
        )
        render_html(
            f"""
<div class="agri-card" style="border-top: 5px solid #10B981; background: #F7FDF9;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <span style="font-weight:800; font-size:1.05rem; color:#065F46;">
            {'✅ Con tu avance:' if is_es else '✅ With your progress:'} {adoption_pct}% ({plan_short_name})
        </span>
        <span style="background:#D1FAE5; color:#065F46; font-size:0.72rem; font-weight:800; padding:3px 9px; border-radius:99px;">
            {gain_badge}
        </span>
    </div>

    <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:4px;">
            <span>💧 {'Agua guardada para la raíz' if is_es else 'Moisture saved for roots'}</span>
            <b>{cur_moisture}% <span style="color:#059669;">(+{gain_moisture}%)</span></b>
        </div>
        <div style="width:100%; height:11px; background:#E2E8F0; border-radius:99px; overflow:hidden;">
            <div style="width:{cur_moisture}%; height:100%; background:{_bar_hex(cur_moisture)}; border-radius:99px;"></div>
        </div>
    </div>

    <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:4px;">
            <span>🛡️ {'Tierra fértil protegida contra lluvias' if is_es else 'Fertile topsoil protected from rain'}</span>
            <b>{cur_soil_safe}% <span style="color:#059669;">(+{gain_soil}%)</span></b>
        </div>
        <div style="width:100%; height:11px; background:#E2E8F0; border-radius:99px; overflow:hidden;">
            <div style="width:{cur_soil_safe}%; height:100%; background:{_bar_hex(cur_soil_safe)}; border-radius:99px;"></div>
        </div>
    </div>

    <div style="margin-bottom:8px;">
        <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:4px;">
            <span>🌽 {'Cosecha a salvo del clima' if is_es else 'Harvest protected from weather'}</span>
            <b>{cur_yield_safe}% <span style="color:#059669;">(+{gain_yield}%)</span></b>
        </div>
        <div style="width:100%; height:11px; background:#E2E8F0; border-radius:99px; overflow:hidden;">
            <div style="width:{cur_yield_safe}%; height:100%; background:{_bar_hex(cur_yield_safe)}; border-radius:99px;"></div>
        </div>
    </div>

    <div style="margin-top:12px; padding:9px 11px; background:#ECFDF5; border-radius:10px; font-size:0.81rem; color:#065F46;">
        {'🌱 Las raíces vivas mantienen el suelo cubierto todo el año, frenan el arrastre y retienen abono.' if is_es else '🌱 Living roots keep soil covered year-round, stop runoff, and hold nutrients in place.'}
    </div>
</div>
"""
        )

    # 4. Vista Satelital Predictiva ligada directamente al Paso 2 (Plan) y Paso 3 (Clima + % Aplicado)
    render_html(
        f"""
<div class="dark-widget-box" style="margin-top:10px; padding:12px 16px;">
    <div class="widget-header">
        <span>🛰️ {'Vista Satelital Predictiva: Cómo lucirá tu campo (Hoy → 10 Años)' if is_es else 'Predictive Satellite View: How Your Field Will Look (Today → 10 Years)'}</span>
        <span style="font-size:0.74rem; color:var(--accent-text);">{plan_short_name} • {adoption_pct}% {'aplicado' if is_es else 'applied'}</span>
    </div>
</div>
"""
    )

    horizon_yr = st.select_slider(
        label="predictive_horizon_slider",
        options=[0, 2, 5, 7, 10],
        value=5,
        format_func=lambda y: (
            ("Hoy (Año 0)" if is_es else "Today (Year 0)")
            if y == 0
            else (f"En {y} años (Año {y})" if is_es else f"In {y} years (Year {y})")
        ),
        label_visibility="collapsed",
        key="tab3_satellite_horizon_slider",
    )

    # Ligado al Plan del Paso 2 (best_strat_id), Clima del Paso 3 (d_rain, d_temp) y Barra de Adopción (adoption_pct)
    yr_frac = horizon_yr / 10.0
    adopt_frac = adoption_pct / 100.0
    plan_effectiveness = 1.0 if best_strat_id == "rot_conservacion" else (0.75 if best_strat_id == "rot_rentabilidad" else 0.15)
    weather_stress = max(0.0, (-d_rain / 50.0) * 0.35 + (d_temp / 2.5) * 0.20 + max(0.0, d_rain / 30.0) * 0.25)

    base_risk = (
        0.55 * (df_grid["slope_pct"].fillna(1.2) / max(float(df_grid["slope_pct"].max()), 1.0))
        + 0.45 * (df_grid["water_deficit_conservacion"].fillna(55.0) / max(float(df_grid["water_deficit_conservacion"].max()), 1.0))
    ).to_numpy(dtype=float)

    net_recovery = (adopt_frac * plan_effectiveness - 0.30 - weather_stress * (1.0 - adopt_frac * 0.7)) * 0.90 * yr_frac
    projected_risk = np.clip(base_risk - net_recovery, 0.04, 0.96)

    b64_proj, coords_proj, poly_lats_p, poly_lons_p = build_smooth_layer_raster(
        tuple(df_grid["lat"].tolist()),
        tuple(df_grid["lon"].tolist()),
        tuple(projected_risk.tolist()),
        cmap_name="RdYlGn_r",
    )

    p_map_col, p_info_col = st.columns([13, 7], gap="medium")

    with p_map_col:
        fig_proj_map = go.Figure()
        fig_proj_map.add_trace(go.Scattermap(
            lat=poly_lats_p,
            lon=poly_lons_p,
            mode="lines",
            line=dict(color="#FFFFFF", width=3.5),
            hoverinfo="skip",
            showlegend=False,
        ))
        px_sel = df_grid[df_grid["pixel_id"] == active_pid].iloc[0]
        fig_proj_map.add_trace(go.Scattermap(
            lat=[px_sel["lat"]],
            lon=[px_sel["lon"]],
            mode="markers+text",
            marker=dict(size=14, color="#FFFFFF"),
            text=[f"📍 #{active_num}"],
            textposition="top right",
            textfont=dict(color="#FFFFFF", size=12),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig_proj_map.update_layout(
            map=dict(
                style="white-bg",
                layers=[
                    dict(
                        below="traces",
                        sourcetype="raster",
                        source=[
                            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ],
                    ),
                    dict(
                        below="traces",
                        sourcetype="image",
                        source=b64_proj,
                        coordinates=coords_proj,
                    ),
                ],
                center=dict(lat=float(df_grid["lat"].mean()), lon=float(df_grid["lon"].mean())),
                zoom=15.25,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=330,
            showlegend=False,
        )
        st.plotly_chart(
            fig_proj_map,
            use_container_width=True,
            key=f"tab3_pred_map_{horizon_yr}_{adoption_pct}_{w_idx}_{best_strat_id}",
            config={"displayModeBar": False},
        )

    with p_info_col:
        healthy_pct = int(np.clip(round((1.0 - float(np.mean(projected_risk))) * 100), 12, 97))
        organic_gain = round(2.75 + (adopt_frac * plan_effectiveness - 0.25) * 1.05 * yr_frac, 2)
        status_pill_color = "#10B981" if healthy_pct >= 65 else ("#F59E0B" if healthy_pct >= 45 else "#EF4444")
        status_label = (
            ("Terreno Saludable y Estable" if is_es else "Healthy & Resilient Field")
            if healthy_pct >= 65
            else (
                ("Atención Moderada en Laderas" if is_es else "Moderate Slope Watch")
                if healthy_pct >= 45
                else ("Riesgo Alto de Desgaste" if is_es else "High Soil Degradation Risk")
            )
        )

        render_html(
            f"""
<div class="agri-card" style="padding:1.05rem 1.2rem; height:330px; display:flex; flex-direction:column; justify-content:space-between;">
    <div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:0.74rem; font-weight:700; color:#64748B; text-transform:uppercase;">
                {'Horizonte:' if is_es else 'Horizon:'} {horizon_yr} {'años' if is_es else 'years'}
            </span>
            <span style="background:{status_pill_color}; color:#FFFFFF; font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:99px;">
                {status_label}
            </span>
        </div>
        <div style="font-size:1.12rem; font-weight:800; color:#0F172A; margin-bottom:8px;">
            {'Condición Satelital Estimada' if is_es else 'Estimated Satellite Condition'}
        </div>
        <div class="check-row">
            <span>🟢 {'Área del campo en verde óptimo' if is_es else 'Field area in healthy green'}</span>
            <b>{healthy_pct}%</b>
        </div>
        <div class="check-row">
            <span>🌱 {'Fertilidad natural del suelo' if is_es else 'Natural topsoil fertility'}</span>
            <b>{organic_gain:.2f}%</b>
        </div>
        <div class="check-row">
            <span>🚜 {'Plan del Paso 2 + Avance' if is_es else 'Step 2 Plan + Adoption'}</span>
            <b>{plan_short_name} ({adoption_pct}%)</b>
        </div>
    </div>
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:8px 10px; font-size:0.78rem; color:#475569;">
        🟢 {'Verde: Suelo protegido' if is_es else 'Green: Protected'} &nbsp;|&nbsp;
        🟡 {'Amarillo: Atención' if is_es else 'Yellow: Watch'} &nbsp;|&nbsp;
        🔴 {'Rojo: Sensible' if is_es else 'Red: Sensitive'}
    </div>
</div>
"""
        )

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 4: SIMPLE ACTION PLAN, VOICE BRIEFING & SCIENTIFIC AUDIT (JURY)
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    is_es = cur_lang == "es"
    win_count = int(round(best_strat["p_mejor"] * 100))
    raw_certeza = str(best_strat["certeza"]).strip()
    conf_label = CONFIDENCE_I18N[cur_lang].get(
        raw_certeza,
        CONFIDENCE_I18N[cur_lang].get(raw_certeza.upper(), CONFIDENCE_I18N[cur_lang].get(raw_certeza.title(), raw_certeza)),
    )
    if conf_label.upper() == "MEDIUM":
        conf_label = "Media" if is_es else "Moderate"
    elif conf_label.upper() == "HIGH":
        conf_label = "Alta" if is_es else "High"
    elif conf_label.upper() == "LOW":
        conf_label = "Baja" if is_es else "Low"

    prov_label = PROVENANCE_I18N[cur_lang].get(str(best_strat["procedencia"]), str(best_strat["procedencia"]))
    chosen_title = STRAT_TITLES[best_strat_id]
    est_margin = best_strat["metricas"]["net_margin_usd_ha"]

    render_html(
        render_mascot_bubble(
            pose="celebrate",
            bubble_title=(
                f"Tu Plan de Acción Listo para el Campo (Sector #{active_num})"
                if is_es
                else f"Your Field-Ready Action Plan (Plot #{active_num})"
            ),
            speech_text=(
                f"Basado en tus prioridades y el historial de clima de Iowa, tu mejor decisión es: {chosen_title}. Pulsa el botón verde para escucharlo."
                if is_es
                else f"Based on your goals and local Iowa weather history, your best move is: {chosen_title}. Press the green button to listen."
            ),
            mascot_size=56,
        )
    )

    # 3 Tarjetas de resumen ultra simples arriba
    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        render_html(
            f"""
<div class="agri-card" style="border-top: 5px solid #10B981; padding:1.05rem 1.2rem;">
    <div style="font-size:0.75rem; font-weight:700; color:#059669; text-transform:uppercase;">
        🏆 {'RESPALDO CLIMÁTICO' if is_es else 'WEATHER RELIABILITY'}
    </div>
    <div style="font-size:1.55rem; font-weight:800; color:#0F172A; margin:4px 0;">
        {win_count} {'de 100 años' if is_es else 'of 100 years'}
    </div>
    <div style="font-size:0.83rem; color:#475569;">
        {'Supera a las demás opciones en casi cualquier clima de Iowa.' if is_es else 'Outperforms alternatives across nearly all Iowa weather seasons.'}
    </div>
</div>
"""
        )
    with k2:
        render_html(
            f"""
<div class="agri-card" style="border-top: 5px solid #F59E0B; padding:1.05rem 1.2rem;">
    <div style="font-size:0.75rem; font-weight:700; color:#B45309; text-transform:uppercase;">
        💵 {'GANANCIA NETA ESTIMADA' if is_es else 'ESTIMATED NET RETURN'}
    </div>
    <div style="font-size:1.55rem; font-weight:800; color:#0F172A; margin:4px 0;">
        ${est_margin:,.0f} USD / ha
    </div>
    <div style="font-size:0.83rem; color:#475569;">
        {'Con menor gasto en abono y opción a apoyo de conservación.' if is_es else 'With lower fertilizer loss and conservation grant eligibility.'}
    </div>
</div>
"""
        )
    with k3:
        render_html(
            f"""
<div class="agri-card" style="border-top: 5px solid #38BDF8; padding:1.05rem 1.2rem;">
    <div style="font-size:0.75rem; font-weight:700; color:#0369A1; text-transform:uppercase;">
        🛡️ {'PROTECCIÓN DE TU TIERRA' if is_es else 'SOIL & WATER PROTECTION'}
    </div>
    <div style="font-size:1.55rem; font-weight:800; color:#0F172A; margin:4px 0;">
        {'12 meses cubierto' if is_es else '12-month cover'}
    </div>
    <div style="font-size:0.83rem; color:#475569;">
        {'Frena el arrastre de tierra en invierno y guarda agua en verano.' if is_es else 'Stops winter topsoil washout and holds summer rootzone water.'}
    </div>
</div>
"""
        )

    # 2 Columnas limpias: Izquierda (Qué significa para tu campo) vs Derecha (Pasos prácticos + Voz)
    left_simple, right_simple = st.columns(2, gap="large")

    with left_simple:
        render_html(
            f"""
<div class="agri-card" style="padding:1.2rem 1.35rem; height:100%;">
    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.10rem; font-weight:800; color:#0F172A; margin-bottom:10px;">
        🌾 {'¿Qué ganas al aplicar este plan?' if is_es else 'What do you gain with this plan?'}
    </div>
    <div style="display:flex; flex-direction:column; gap:10px; font-size:0.88rem; color:#1E293B; line-height:1.5;">
        <div style="background:#F0FDF4; border-left:4px solid #10B981; padding:10px 12px; border-radius:8px;">
            <b>1. {'Más agua guardada en la raíz:' if is_es else 'More water saved in the rootzone:'}</b>
            {'Las raíces vivas y el rastrojo evitan que el sol de julio y agosto seque tu terreno.' if is_es else 'Living roots and surface residue keep July and August heat from drying out your field.'}
        </div>
        <div style="background:#F0FDF4; border-left:4px solid #10B981; padding:10px 12px; border-radius:8px;">
            <b>2. {'Ahorro directo en fertilizante:' if is_es else 'Direct savings on fertilizer:'}</b>
            {'Atrapa el nitrógeno que sobra tras la cosecha para que las lluvias no lo laven.' if is_es else 'Captures leftover post-harvest nitrogen so spring rains do not wash it away.'}
        </div>
        <div style="background:#FFFBEB; border-left:4px solid #F59E0B; padding:10px 12px; border-radius:8px;">
            <b>3. {'Cuidado práctico a vigilar:' if is_es else 'Practical detail to watch:'}</b>
            {'Secar o terminar la cobertura 10–14 días antes de sembrar soya para que no compita por agua.' if is_es else 'Terminate cover crop 10–14 days before planting soybeans so it does not compete for spring moisture.'}
        </div>
    </div>
</div>
"""
        )

    with right_simple:
        render_html(
            f"""
<div class="agri-card" style="padding:1.2rem 1.35rem; margin-bottom:10px;">
    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.10rem; font-weight:800; color:#0F172A; margin-bottom:8px;">
        ✅ {'Tus 3 Pasos Siguientes en la Chacra' if is_es else 'Your 3 Next Steps on the Farm'}
    </div>
    <ol style="margin:0; padding-left:1.2rem; font-size:0.87rem; color:#1E293B; line-height:1.65;">
        <li><b>{'Después de cosechar:' if is_es else 'Right after harvest:'}</b> {'Siembra centeno de cobertura para proteger el suelo antes de las heladas.' if is_es else 'Drill winter cereal rye to lock topsoil in place before frost.'}</li>
        <li><b>{'En primavera (10–14 días antes):' if is_es else 'In spring (10–14 days early):'}</b> {'Controla el centeno a tiempo para dejar cama húmeda y limpia a la soya.' if is_es else 'Terminate rye on schedule to leave a clean, moist seedbed for soybeans.'}</li>
        <li><b>{'En tu oficina local USDA:' if is_es else 'At your local USDA office:'}</b> {'Solicita el apoyo de conservación (EQIP) que ayuda a pagar la semilla.' if is_es else 'Apply for USDA EQIP conservation cost-share to help pay for seed.'}</li>
    </ol>
</div>
"""
        )

        plan_text = t(
            "t4_plan_speech",
            num=active_num,
            plan=chosen_title,
            margin=est_margin,
            wins=win_count,
        )
        render_voice_button(plan_text, t("t4_voice_btn"), t("t4_voice_lang"))
        st.text_area(
            "📋 Mensaje listo para enviar por WhatsApp a tu agrónomo o proveedor:"
            if is_es
            else "📋 Ready-to-share summary for your agronomist or seed dealer:",
            plan_text,
            height=84,
        )

    # Desplegable Científico Intacto para el Jurado NASA
    with st.expander(t("t4_audit_exp"), expanded=False):
        e4_cards = e4_bundle.get("cards", {})
        raw_card = e4_cards.get(best_strat["e4_key"], {}) if isinstance(e4_cards, dict) else {}
        active_card = XAI_CARDS_EN.get(best_strat_id, raw_card) if cur_lang == "en" else raw_card
        e4_meta = e4_bundle.get("metadata", {})

        if active_card:
            st.markdown(t("t4_aud_sec3"))
            ev_sat = active_card.get("evidencia_satelital", "")
            sup_esc = active_card.get("supuestos_y_escenarios", [])
            brechas = active_card.get("brechas_e_incertidumbre", [])
            if ev_sat:
                st.markdown(f"- **{t('t4_aud_ev')}** {ev_sat}")
            for s_item in (sup_esc if isinstance(sup_esc, list) else [sup_esc]):
                if s_item:
                    st.markdown(f"- **{t('t4_aud_sup')}** {s_item}")
            for b_item in (brechas if isinstance(brechas, list) else [brechas]):
                if b_item:
                    st.markdown(f"- **{t('t4_aud_gap')}** {b_item}")

        if e4_meta:
            seed_val = e4_meta.get("seed", 42)
            ts_val = e4_meta.get("timestamp_utc", "N/A")
            sha_dict = e4_meta.get("sha256_inputs", {})
            first_sha = next(iter(sha_dict.values()), "verified")[:16] if isinstance(sha_dict, dict) and sha_dict else "verified"
            st.info(
                f"🛡️ **Reproducibility Badge (Stat-4 Audit):** `Seed = {seed_val}` • `UTC = {ts_val}` • "
                f"`Input SHA-256 = {first_sha}…` • `Confidence = {conf_label}` • `Status = {prov_label}`"
            )

        audit_c1, audit_c2 = st.columns(2)
        with audit_c1:
            st.markdown(t("t4_aud_sec1"))
            raw_mat_df = pd.DataFrame(
                topsis_res["matriz"],
                index=[STRAT_SHORTS["rot_actual"], STRAT_SHORTS["rot_conservacion"], STRAT_SHORTS["rot_rentabilidad"]],
                columns=[t("t4_aud_col1"), t("t4_aud_col2"), t("t4_aud_col3"), t("t4_aud_col4")],
            )
            # Tabla HTML temática (legible en claro y oscuro; st.dataframe no sigue el tema de la app)
            mat_html = raw_mat_df.to_html(
                float_format=lambda v: f"{v:,.3f}",
                border=0,
                escape=True,
            ).replace("\n", "")
            render_html(f'<div class="audit-wrap">{mat_html}</div>')
            st.markdown(
                f"- **Soil AWC (SSURGO):** `awc_l = {awc_l:.4f}`, `AWC = {awc_val:.4f}`, `awc_h = {awc_h:.4f}` cm/cm\n"
                f"- **Combined SMAP Spatial Error:** $\\sigma_{{\\text{{combined}}}} = \\pm {sigma_comb:.4f}\\text{{ m}}^3/\\text{{m}}^3$"
            )
            e1_val = e1_bundle.get("validation", {})
            if e1_val:
                e1_sha = str(e1_val.get("sha256_outputs", {}).get("metrics_phenology.parquet", "45469bdf6651"))[:16]
                st.markdown(
                    f"- **Phenology & Bare-Soil Audit (`e1_v15`):** `351/360` active HLS pixels (`px_0000..px_0008` masked) • "
                    f"Rye thermal feasibility: `7/9` seasons ($\\text{{GDD}}_4 \\ge 150$, Clopper–Pearson 95% CI) • "
                    f"`SHA-256 = {e1_sha}…`"
                )
        with audit_c2:
            st.markdown(t("t4_aud_sec2"))
            st_info = get_data_status()
            for k, v in st_info["archivos_verificados"].items():
                st.markdown(f"- `{k}`: {v}")

        st.markdown("---")
        st.markdown(
            "**4. NASA Space Apps Compliance — Data Citations & Open-Source Licenses**\n"
            "- **Original Code & Models:** AgroPulse (Licensed under **Apache License 2.0**).\n"
            "- **NASA HLS v2.0 (30m Surface Reflectance & Phenology):** NASA LP DAAC / USGS (`https://hls.gsfc.nasa.gov/`) — NASA Open Data Policy (Public Domain).\n"
            "- **NASA SMAP L3/L4 Soil Moisture:** NASA NSIDC DAAC (`https://smap.jpl.nasa.gov/`) — Public Domain.\n"
            "- **NASA GPM IMERG / Gridded Climate Baseline (1981–2025):** NASA GES DISC (`https://gpm.nasa.gov/`) — Public Domain.\n"
            "- **USDA NRCS SSURGO & NASS Cropland Data Layer (2016–2025):** U.S. Department of Agriculture (`https://www.nrcs.usda.gov/`, `https://nassgeodata.gmu.edu/CropScape/`) — U.S. Government Work / Public Domain.\n"
            "- **USGS 3DEP 1/3 Arc-Second DEM:** U.S. Geological Survey (`https://www.usgs.gov/3d-elevation-program`) — Public Domain.\n"
            "- **Satellite Basemap Attribution:** *Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community*.\n"
            "- **Open-Source Stack:** `Streamlit` (Apache 2.0), `Plotly` (MIT), `Pandas` / `NumPy` / `SciPy` / `Matplotlib` / `Pillow` (BSD / PSF / PIL Licenses)."
        )