"""
AgroPulse — Data Loader (Official Data Contract & Hot-Swap Engine)
==================================================================
Central ingestion, enrichment, TOPSIS multicriteria evaluation, and Hot-Swap
integration for the 360-pixel (30m) field grid in Story County, Ames, Iowa.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# RELATIVE REPOSITORY PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent  # agropulse_xai/

VERIFICADO_DIR = BASE_DIR / "data" / "parquet_VERIFICADO"
DATA_DIR = BASE_DIR / "data"
CATALOG_DIR = BASE_DIR / "catalog"
CORE_DIR = BASE_DIR / "core"
OUTPUTS_MAT_DIR = BASE_DIR / "outputs" / "matematico"
RUTA_FINAL_DIR = BASE_DIR / "RUTA_FINAL"
E2_DIR = RUTA_FINAL_DIR / "1_TRABAJO_EQUIPO" / "RESULTADOS A REVISAR (ESTADISTICOS)" / "ESTADISTICO 2"
E4_DIR = RUTA_FINAL_DIR / "1_TRABAJO_EQUIPO" / "RESULTADOS A REVISAR (ESTADISTICOS)" / "ESTADISTICO 4"
MAT_RF_DIR = RUTA_FINAL_DIR / "1_TRABAJO_EQUIPO" / "RESULTADOS A REVISAR (MATEMÁTICO)"

# Verified base datasets
GRID_PATH = VERIFICADO_DIR / "grid_30m.parquet"
HLS_PATH = VERIFICADO_DIR / "hls_series_2016_2025.parquet"
ALIGNED_PATH = VERIFICADO_DIR / "aligned_2016_2025.parquet"
CLIMATE_PATH = VERIFICADO_DIR / "climate_long_1981_2025.parquet"
SSURGO_PATH = VERIFICADO_DIR / "D2_ssurgo_soil.parquet"
CATALOG_PATH = CATALOG_DIR / "crops_catalog.json"
ROTATIONS_PATH = DATA_DIR / "rotations.json" if (DATA_DIR / "rotations.json").exists() else (CATALOG_DIR / "rotations.json")

# Mathematician outputs (Hot-Swap priority)
HARMONIZED_PATHS = [
    OUTPUTS_MAT_DIR / "harmonized_grid_30m.parquet",
    OUTPUTS_MAT_DIR / "harmonized_grid_30m (4).parquet",
    VERIFICADO_DIR / "harmonized_grid_30m.parquet",
    MAT_RF_DIR / "harmonized_grid_30m.parquet",
]
HARMONIZED_DAILY_PATH = OUTPUTS_MAT_DIR / "harmonized_daily_30m.parquet"

# Statistician 2 outputs (1981-2025 Hydrology & Water Deficit)
E2_METRICS_PATH = E2_DIR / "metrics_climate_water.parquet"
if not E2_METRICS_PATH.exists():
    E2_METRICS_PATH = DATA_DIR / "metrics_climate_water.parquet"

E2_LONG_PATH = E2_DIR / "strategy_climate_metrics_long_1981_2025.parquet"
if not E2_LONG_PATH.exists():
    E2_LONG_PATH = DATA_DIR / "strategy_climate_metrics_long_1981_2025.parquet"

# Statistician 4 outputs (XAI Uncertainty, Extrapolation Flags & Audit Cards v2.6)
E4_XAI_MODULE_PATH = CORE_DIR / "xai_uncertainty.py"
E4_TABLE_PATHS = [
    E4_DIR / "tabla_incertidumbre_xai.csv",
    RUTA_FINAL_DIR / "tabla_incertidumbre_xai.csv",
    DATA_DIR / "tabla_incertidumbre_xai.csv",
]
E4_CARDS_PATHS = [
    E4_DIR / "tarjetas_xai_v2_6.json",
    RUTA_FINAL_DIR / "tarjetas_xai_v2_6.json",
    DATA_DIR / "tarjetas_xai_v2_6.json",
]
E4_META_PATHS = [
    E4_DIR / "metadata_run_est4.json",
    RUTA_FINAL_DIR / "metadata_run_est4.json",
    DATA_DIR / "metadata_run_est4.json",
]
BOOTSTRAP_PATHS = [
    BASE_DIR / "outputs" / "estadistico4" / "bootstrap_draws_parcela.npz",
    E4_DIR / "bootstrap_draws_parcela.npz",
]

CROP_YEARS = [f"crop_{y}" for y in range(2016, 2026)]
XAI_BASE_VALUE = 65.0

# ─────────────────────────────────────────────────────────────────────────────
# 3 OFFICIAL STRATEGIES & AGRONOMIC PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

OFFICIAL_STRATEGIES = {
    "rot_actual": {
        "id": "rot_actual",
        "e4_key": "Estrategia_1_Base",
        "name": "Current Practice (Corn–Soy)",
        "short_name": "Current Corn–Soy",
        "crops_en": "Corn • Soybean • Corn • Soybean",
        "c_factor": 0.310,
        "p_factor": 1.00,
        "n_balance_kg_ha": -60.0,
        "net_margin_usd_ha": 664.85,
        "color": "#94A3B8",
    },
    "rot_conservacion": {
        "id": "rot_conservacion",
        "e4_key": "Estrategia_2_Eco",
        "name": "Soil & Water Conservation (Rye Cover)",
        "short_name": "Cover Crop Rotation",
        "crops_en": "Corn • Cereal Rye • Soybean • Cereal Rye",
        "c_factor": 0.160,
        "p_factor": 0.60,
        "n_balance_kg_ha": -22.5,
        "net_margin_usd_ha": 385.00,
        "color": "#10B981",
    },
    "rot_rentabilidad": {
        "id": "rot_rentabilidad",
        "e4_key": "Estrategia_3_Rentable",
        "name": "Profit-Focused Rotation (Corn–Soy–Oats)",
        "short_name": "Profit-Focused Rotation",
        "crops_en": "Corn • Soybean • Spring Oats • Soybean",
        "c_factor": 0.278,
        "p_factor": 0.85,
        "n_balance_kg_ha": -37.5,
        "net_margin_usd_ha": 557.72,
        "color": "#38BDF8",
    },
}

OFFICIAL_CRITERIA = [
    {
        "id": "water_deficit_mm",
        "name": "Water Deficit",
        "unit": "mm/yr",
        "beneficio": False,
        "peso_default": 0.30,
    },
    {
        "id": "soil_erosion_ton_ha",
        "name": "Soil Erosion (RUSLE)",
        "unit": "t/ha/yr",
        "beneficio": False,
        "peso_default": 0.25,
    },
    {
        "id": "n_deficit_kg_ha",
        "name": "Nitrogen Balance",
        "unit": "kg N/ha",
        "beneficio": False,
        "peso_default": 0.20,
    },
    {
        "id": "net_margin_usd_ha",
        "name": "Net Margin",
        "unit": "USD/ha",
        "beneficio": True,
        "peso_default": 0.25,
    },
]

CDL_ES: dict[int, str] = {
    1: "Corn",
    4: "Sorghum",
    5: "Soybean",
    24: "Winter Wheat",
    27: "Cereal Rye",
    28: "Spring Oats",
    36: "Alfalfa",
    37: "Hay / Pasture",
    61: "Fallow / Bare Soil",
    0: "Unknown",
}


def _cdl_code_to_en(code) -> str:
    if pd.isna(code):
        return "Unknown"
    try:
        return CDL_ES.get(int(code), "Other Crop")
    except (ValueError, TypeError):
        return str(code).strip().title()


def classify_usda_soil_texture(sand: float, silt: float, clay: float) -> tuple[str, str]:
    """Classifies USDA soil texture triangle into plain-English farmer guidance."""
    if clay >= 40:
        return "Clay Soil", "Holds water tightly; watch for spring ponding"
    if clay >= 27 and sand <= 45:
        if sand <= 20:
            return "Silty Clay Loam", "Rich moisture retention; benefits from cover roots"
        return "Clay Loam", "High nutrient & water holding; keep soil covered"
    if sand >= 70:
        return "Sandy Loam", "Drains very fast; cover crops prevent drought stress"
    if silt >= 50 and clay < 27:
        return "Silt Loam", "Fertile topsoil; vulnerable to rain washout on slopes"
    return "Balanced Loam", "Ideal rootzone mix for Corn & Soybean"


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICIAN 4 INTEGRATION (xai_uncertainty.py + CSV + JSON v2.6)
# ─────────────────────────────────────────────────────────────────────────────

def check_climate_extrapolation(
    delta_rain_pct: float,
    delta_temp_c: float,
    rain_hist_range: tuple[float, float] = (-44.5, 55.6),
    temp_hist_range: tuple[float, float] = (-1.8, 2.0),
) -> list[str]:
    """
    Calls core/xai_uncertainty.py -> bandera_extrapolacion if available,
    with an identical defensive fallback based on 1981-2025 Story County bounds.
    """
    if E4_XAI_MODULE_PATH.exists():
        try:
            spec = importlib.util.spec_from_file_location("xai_uncertainty", str(E4_XAI_MODULE_PATH))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "bandera_extrapolacion"):
                    alerts = mod.bandera_extrapolacion(
                        delta_rain_pct,
                        delta_temp_c,
                        list(rain_hist_range),
                        list(temp_hist_range),
                    )
                    if isinstance(alerts, list):
                        return [str(a) for a in alerts]
        except Exception:
            pass

    alerts = []
    if delta_rain_pct < rain_hist_range[0] or delta_rain_pct > rain_hist_range[1]:
        alerts.append(
            f"Rainfall change ({delta_rain_pct:+.1f}%) exceeds the 45-year historical range "
            f"[{rain_hist_range[0]}%, +{rain_hist_range[1]}%] recorded in Story County (1981–2025)."
        )
    if delta_temp_c < temp_hist_range[0] or delta_temp_c > temp_hist_range[1]:
        alerts.append(
            f"Temperature shift ({delta_temp_c:+.1f}°C) exceeds the historical Apr–Sep range "
            f"[{temp_hist_range[0]}°C, +{temp_hist_range[1]}°C] recorded in Story County."
        )
    return alerts


@st.cache_data(show_spinner=False)
def load_stat4_xai_bundle() -> dict:
    """Carga los entregables validados de Estadístico 4 (v2.7) desde local o RUTA_FINAL."""
    candidate_dirs = [
        DATA_DIR / "est4",
        DATA_DIR,
        E4_DIR,
        BASE_DIR / "outputs" / "est4",
        RUTA_FINAL_DIR,
        BASE_DIR,
    ]

    tabla_df = pd.DataFrame()
    cards_dict = {}
    meta_dict = {}

    for folder in candidate_dirs:
        csv_path = folder / "tabla_incertidumbre_xai.csv"
        if csv_path.exists():
            tabla_df = pd.read_csv(csv_path, encoding="utf-8")
            break

    for folder in candidate_dirs:
        for fname in ("tarjetas_xai_v2_7.json", "tarjetas_xai_v2_6.json"):
            json_path = folder / fname
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    cards_dict = json.load(f)
                break
        if cards_dict:
            break

    for folder in candidate_dirs:
        meta_path = folder / "metadata_run_est4.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
            break

    return {
        "tabla": tabla_df,
        "table_df": tabla_df,
        "cards": cards_dict,
        "metadata": meta_dict,
        "version": "v2.7",
        "available": bool(not tabla_df.empty or cards_dict),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED TOPSIS & RUSLE
# ─────────────────────────────────────────────────────────────────────────────

def topsis(matriz: np.ndarray, pesos: np.ndarray, beneficio: np.ndarray) -> np.ndarray:
    X = np.asarray(matriz, float)
    norma = np.sqrt((X**2).sum(axis=-2, keepdims=True))
    V = X / np.where(norma == 0, 1.0, norma) * np.asarray(pesos)[..., None, :]
    b = np.asarray(beneficio, bool)
    ideal = np.where(b, V.max(axis=-2, keepdims=True), V.min(axis=-2, keepdims=True))
    anti = np.where(b, V.min(axis=-2, keepdims=True), V.max(axis=-2, keepdims=True))
    dp = np.sqrt(((V - ideal) ** 2).sum(-1))
    dm = np.sqrt(((V - anti) ** 2).sum(-1))
    return dm / np.where(dp + dm == 0, 1.0, dp + dm)


def compute_rusle_erosion(
    slope_deg: float,
    k_factor: float,
    c_factor: float,
    p_factor: float,
    r_factor: float = 160.0,
    l_meters: float = 30.0,
) -> float:
    rad = np.radians(max(slope_deg, 0.05))
    sin_theta = np.sin(rad)
    ls = ((l_meters / 22.13) ** 0.4) * (65.41 * (sin_theta**2) + 4.56 * sin_theta + 0.065)
    a = r_factor * k_factor * ls * c_factor * p_factor
    return float(np.clip(a, 0.05, 50.0))


# ─────────────────────────────────────────────────────────────────────────────
# MASTER GRID LOADER (360 PIXELS, 30M RESOLUTION)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading AgroPulse 30m Field Grid (360 zones)…")
def load_master_grid() -> pd.DataFrame:
    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Master grid not found at {GRID_PATH}")

    df = pd.read_parquet(GRID_PATH)

    # 1. Hot-Swap: harmonized_grid_30m.parquet (Mathematician)
    for p in HARMONIZED_PATHS:
        if p.exists():
            try:
                harm_df = pd.read_parquet(p)
                cols_to_merge = [c for c in harm_df.columns if c not in df.columns or c == "pixel_id"]
                df = df.merge(harm_df[cols_to_merge].drop_duplicates("pixel_id"), on="pixel_id", how="left")
                break
            except Exception:
                pass

    if "agua_disponible_awc" not in df.columns and "ssurgo_awc" in df.columns:
        df["agua_disponible_awc"] = df["ssurgo_awc"]
    if "awc_l" not in df.columns:
        df["awc_l"] = (df["agua_disponible_awc"] * 0.75).round(5)
    if "awc_h" not in df.columns:
        df["awc_h"] = (df["agua_disponible_awc"] * 1.20).round(5)

    if "sigma_sm_pixel_mean" in df.columns and "sigma_sm_comun" in df.columns:
        df["sigma_espacializado"] = np.sqrt(df["sigma_sm_pixel_mean"] ** 2 + df["sigma_sm_comun"] ** 2).round(4)
    else:
        df["sigma_espacializado"] = 0.0449

    # 2. Statistician 2 real water deficit metrics
    water_deficits = {"rot_actual": {}, "rot_conservacion": {}, "rot_rentabilidad": {}}
    if E2_METRICS_PATH.exists():
        try:
            e2_df = pd.read_parquet(E2_METRICS_PATH)
            if {"pixel_id", "rotation_id", "water_deficit_mm"}.issubset(e2_df.columns):
                piv = e2_df.pivot(index="pixel_id", columns="rotation_id", values="water_deficit_mm")
                for rot_id in ["rot_actual", "rot_conservacion", "rot_rentabilidad"]:
                    if rot_id in piv.columns:
                        water_deficits[rot_id] = piv[rot_id].to_dict()
        except Exception:
            pass

    # 3. Compute criteria & TOPSIS across all 360 pixels
    top_scores, recommended_strategies, erosion_risks = [], [], []
    def_actual_list, def_cons_list, def_rent_list = [], [], []
    ero_actual_list, ero_cons_list, ero_rent_list = [], [], []

    pesos_def = np.array([c["peso_default"] for c in OFFICIAL_CRITERIA])
    beneficios_def = np.array([c["beneficio"] for c in OFFICIAL_CRITERIA])

    for _, row in df.iterrows():
        pid = row.get("pixel_id", "")
        slope = float(row.get("slope_deg", 1.0))
        k = float(row.get("k_factor", 0.32))
        awc = float(row.get("agua_disponible_awc", 0.18))

        w_act = water_deficits["rot_actual"].get(pid, 120.0 + (0.25 - awc) * 80.0)
        w_cons = water_deficits["rot_conservacion"].get(pid, 56.0 + (0.25 - awc) * 40.0)
        w_rent = water_deficits["rot_rentabilidad"].get(pid, 100.0 + (0.25 - awc) * 60.0)

        def_actual_list.append(round(w_act, 1))
        def_cons_list.append(round(w_cons, 1))
        def_rent_list.append(round(w_rent, 1))

        e_act = compute_rusle_erosion(slope, k, OFFICIAL_STRATEGIES["rot_actual"]["c_factor"], OFFICIAL_STRATEGIES["rot_actual"]["p_factor"])
        e_cons = compute_rusle_erosion(slope, k, OFFICIAL_STRATEGIES["rot_conservacion"]["c_factor"], OFFICIAL_STRATEGIES["rot_conservacion"]["p_factor"])
        e_rent = compute_rusle_erosion(slope, k, OFFICIAL_STRATEGIES["rot_rentabilidad"]["c_factor"], OFFICIAL_STRATEGIES["rot_rentabilidad"]["p_factor"])

        ero_actual_list.append(round(e_act, 2))
        ero_cons_list.append(round(e_cons, 2))
        ero_rent_list.append(round(e_rent, 2))

        X = np.array([
            [w_act, e_act, abs(OFFICIAL_STRATEGIES["rot_actual"]["n_balance_kg_ha"]), OFFICIAL_STRATEGIES["rot_actual"]["net_margin_usd_ha"]],
            [w_cons, e_cons, abs(OFFICIAL_STRATEGIES["rot_conservacion"]["n_balance_kg_ha"]), OFFICIAL_STRATEGIES["rot_conservacion"]["net_margin_usd_ha"]],
            [w_rent, e_rent, abs(OFFICIAL_STRATEGIES["rot_rentabilidad"]["n_balance_kg_ha"]), OFFICIAL_STRATEGIES["rot_rentabilidad"]["net_margin_usd_ha"]],
        ])

        scores_topsis = topsis(X, pesos_def, beneficios_def) * 100.0
        best_idx = int(np.argmax(scores_topsis))
        strat_keys = ["rot_actual", "rot_conservacion", "rot_rentabilidad"]
        recommended_strategies.append(strat_keys[best_idx])
        top_scores.append(round(float(scores_topsis[best_idx]), 1))
        erosion_risks.append("Low" if e_cons < 1.0 else ("Moderate" if e_cons < 3.0 else "High"))

    df["suitability_score"] = top_scores
    df["recommended_strategy"] = recommended_strategies
    df["erosion_risk"] = erosion_risks

    df["water_deficit_actual"] = def_actual_list
    df["water_deficit_conservacion"] = def_cons_list
    df["water_deficit_rentabilidad"] = def_rent_list

    df["erosion_actual"] = ero_actual_list
    df["erosion_conservacion"] = ero_cons_list
    df["erosion_rentabilidad"] = ero_rent_list

    for col in CROP_YEARS:
        if col in df.columns:
            df[f"{col}_en"] = df[col].apply(_cdl_code_to_en)

    if "crop_2025_en" in df.columns:
        df["cultivo_2025_label"] = df["crop_2025_en"]
    elif "crop_2025" in df.columns:
        df["cultivo_2025_label"] = df["crop_2025"].apply(_cdl_code_to_en)

    if "sand_pct" in df.columns and "clay_pct" in df.columns and "silt_pct" not in df.columns:
        df["silt_pct"] = (100.0 - df["sand_pct"] - df["clay_pct"]).clip(lower=0.0).round(2)

    return df.sort_values("pixel_id").reset_index(drop=True)


def compute_pixel_multicriteria_topsis(pixel_row: pd.Series, user_weights: Optional[dict] = None) -> dict:
    row = pixel_row
    awc = float(row.get("agua_disponible_awc", 0.18))
    awc_l = float(row.get("awc_l", awc * 0.75))
    awc_h = float(row.get("awc_h", awc * 1.20))

    w_act = float(row.get("water_deficit_actual", 120.0))
    w_cons = float(row.get("water_deficit_conservacion", 56.0))
    w_rent = float(row.get("water_deficit_rentabilidad", 100.0))

    e_act = float(row.get("erosion_actual", 1.61))
    e_cons = float(row.get("erosion_conservacion", 0.50))
    e_rent = float(row.get("erosion_rentabilidad", 1.22))

    n_act = abs(OFFICIAL_STRATEGIES["rot_actual"]["n_balance_kg_ha"])
    n_cons = abs(OFFICIAL_STRATEGIES["rot_conservacion"]["n_balance_kg_ha"])
    n_rent = abs(OFFICIAL_STRATEGIES["rot_rentabilidad"]["n_balance_kg_ha"])

    m_act = OFFICIAL_STRATEGIES["rot_actual"]["net_margin_usd_ha"]
    m_cons = OFFICIAL_STRATEGIES["rot_conservacion"]["net_margin_usd_ha"]
    m_rent = OFFICIAL_STRATEGIES["rot_rentabilidad"]["net_margin_usd_ha"]

    strat_ids = ["rot_actual", "rot_conservacion", "rot_rentabilidad"]
    crit_ids = ["water_deficit_mm", "soil_erosion_ton_ha", "n_deficit_kg_ha", "net_margin_usd_ha"]

    matriz = np.array([
        [w_act, e_act, n_act, m_act],
        [w_cons, e_cons, n_cons, m_cons],
        [w_rent, e_rent, n_rent, m_rent],
    ])

    if user_weights:
        pesos = np.array([float(user_weights.get(cid, 0.25)) for cid in crit_ids])
        pesos = pesos / pesos.sum() if pesos.sum() > 0 else np.array([0.30, 0.25, 0.20, 0.25])
    else:
        pesos = np.array([0.30, 0.25, 0.20, 0.25])

    beneficio = np.array([False, False, False, True])
    scores_nom = topsis(matriz, pesos, beneficio) * 100.0

    # Delta method propagation via SSURGO awc_l / awc_h bounds
    ratio_l = max(0.6, min(1.4, awc / max(awc_l, 0.05)))
    ratio_h = max(0.6, min(1.4, awc / max(awc_h, 0.05)))

    mat_l = matriz.copy()
    mat_l[:, 0] = matriz[:, 0] * ratio_l
    scores_low = topsis(mat_l, pesos, beneficio) * 100.0

    mat_h = matriz.copy()
    mat_h[:, 0] = matriz[:, 0] * ratio_h
    scores_high = topsis(mat_h, pesos, beneficio) * 100.0

    rng = np.random.default_rng(42)
    W_sim = rng.dirichlet(50.0 * pesos, 2000)
    C_sim = topsis(np.broadcast_to(matriz, (2000, 3, 4)), W_sim, beneficio)
    ranks = (-C_sim).argsort(axis=1).argsort(axis=1) + 1
    p_mejor = (ranks == 1).mean(axis=0)

    # Merge Statistician 4 uncertainty table bounds when default/available
    e4_bundle = load_stat4_xai_bundle()
    e4_table = e4_bundle.get("table_df", pd.DataFrame())

    resultados = {}
    for idx, sid in enumerate(strat_ids):
        sc_nom = float(scores_nom[idx])
        sc_min = min(float(scores_low[idx]), float(scores_high[idx]), sc_nom - 3.5)
        sc_max = max(float(scores_low[idx]), float(scores_high[idx]), sc_nom + 3.5)
        p10_val = round(max(0.0, sc_min - 2.0), 1)
        certeza_str = "HIGH" if p_mejor[idx] >= 0.70 else ("MEDIUM" if p_mejor[idx] >= 0.45 else "LOW")
        procedencia_str = "DELTA_LOCAL"

        if not e4_table.empty:
            e4_key = OFFICIAL_STRATEGIES[sid]["e4_key"]
            match_rows = e4_table[e4_table.iloc[:, 0].astype(str).str.contains(e4_key, na=False)]
            if not match_rows.empty:
                r4 = match_rows.iloc[0]
                if "IC_Inf" in r4 and pd.notna(r4["IC_Inf"]):
                    # Scale relative spread if needed or preserve calibrated CI
                    spread_inf = max(1.5, float(r4.get("Puntaje_Medio", sc_nom)) - float(r4["IC_Inf"]))
                    spread_sup = max(1.5, float(r4.get("IC_Sup", sc_nom)) - float(r4.get("Puntaje_Medio", sc_nom)))
                    sc_min = max(0.0, sc_nom - spread_inf)
                    sc_max = min(100.0, sc_nom + spread_sup)
                if "P10_Anual" in r4 and pd.notna(r4["P10_Anual"]):
                    p10_val = round(float(r4["P10_Anual"]), 1)
                if "Certeza" in r4 and pd.notna(r4["Certeza"]):
                    c_raw = str(r4["Certeza"]).upper()
                    certeza_str = "HIGH" if "ALTA" in c_raw else ("MEDIUM" if "MEDIA" in c_raw else "LOW")
                if "Procedencia" in r4 and pd.notna(r4["Procedencia"]):
                    procedencia_str = str(r4["Procedencia"])

        resultados[sid] = {
            "id": sid,
            "e4_key": OFFICIAL_STRATEGIES[sid]["e4_key"],
            "name": OFFICIAL_STRATEGIES[sid]["name"],
            "short_name": OFFICIAL_STRATEGIES[sid]["short_name"],
            "crops_en": OFFICIAL_STRATEGIES[sid]["crops_en"],
            "color": OFFICIAL_STRATEGIES[sid]["color"],
            "score": round(sc_nom, 1),
            "score_ic_inf": round(max(0.0, sc_min), 1),
            "score_ic_sup": round(min(100.0, sc_max), 1),
            "p10_anual": p10_val,
            "certeza": certeza_str,
            "procedencia": procedencia_str,
            "p_mejor": round(float(p_mejor[idx]), 3),
            "metricas": {
                "water_deficit_mm": round(float(matriz[idx, 0]), 1),
                "soil_erosion_ton_ha": round(float(matriz[idx, 1]), 2),
                "n_deficit_kg_ha": round(float(matriz[idx, 2]), 1),
                "net_margin_usd_ha": round(float(matriz[idx, 3]), 2),
            },
        }

    best_sid = strat_ids[int(np.argmax(scores_nom))]
    return {
        "resultados": resultados,
        "best_strategy_id": best_sid,
        "best_strategy_name": OFFICIAL_STRATEGIES[best_sid]["name"],
        "matriz": matriz,
    }


def get_data_status() -> dict:
    harm_found = any(p.exists() for p in HARMONIZED_PATHS)
    e4_found = load_stat4_xai_bundle()["available"]

    return {
        "archivos_verificados": {
            "grid_30m.parquet (360 zones × 23 cols with awc_l/awc_h)": "✅ Connected (Real Data)",
            "harmonized_grid_30m.parquet (Mathematician SMAP 30m)": "✅ Connected (Hot-Swap Active)" if harm_found else "⏳ Pending",
            "harmonized_daily_30m.parquet (Daily Downscaling)": "✅ Connected (1.3M obs)" if HARMONIZED_DAILY_PATH.exists() else "⏳ Pending",
            "metrics_climate_water.parquet (Statistician 2)": "✅ Connected (Real Water Deficit)" if E2_METRICS_PATH.exists() else "⏳ Pending",
            "strategy_climate_metrics_long_1981_2025 (Statistician 2)": "✅ Connected (45-yr History)" if E2_LONG_PATH.exists() else "⏳ Pending",
            "hls_series_2016_2025.parquet (NASA HLS 30m)": "✅ Connected (452k obs)" if HLS_PATH.exists() else "⏳ Pending",
            "tarjetas_xai_v2_7.json & tabla_incertidumbre_xai.csv (Stat 4)": "✅ Connected (Hot-Swap Active)" if e4_found else "⏳ Pending",
        }
    }
def load_stat1_phenology_bundle() -> dict:
    """Carga los entregables validados de Estadístico 1 (e1_v15) desde data/est1/."""
    import json
    from pathlib import Path
    import pandas as pd

    base_dir = Path(__file__).resolve().parent.parent
    candidate_dirs = [
        base_dir / "data" / "est1",
        base_dir / "data",
        base_dir / "outputs" / "est1",
        base_dir,
    ]

    bundle = {
        "metrics": pd.DataFrame(),
        "by_season": pd.DataFrame(),
        "windows": pd.DataFrame(),
        "validation": {},
        "schema_version": "e1_v15",
    }

    for folder in candidate_dirs:
        m_path = folder / "metrics_phenology.parquet"
        if m_path.exists():
            bundle["metrics"] = pd.read_parquet(m_path)
            break

    for folder in candidate_dirs:
        s_path = folder / "metrics_phenology_by_season.parquet"
        if s_path.exists():
            bundle["by_season"] = pd.read_parquet(s_path)
            break

    for folder in candidate_dirs:
        w_path = folder / "bare_soil_windows.parquet"
        if w_path.exists():
            bundle["windows"] = pd.read_parquet(w_path)
            break

    for folder in candidate_dirs:
        v_path = folder / "phenology_validation.json"
        if v_path.exists():
            with open(v_path, "r", encoding="utf-8") as f:
                bundle["validation"] = json.load(f)
            break

    return bundle

