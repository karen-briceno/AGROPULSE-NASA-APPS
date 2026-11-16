# REGLAS ESTRICTAS DE ARQUITECTURA — AGROPULSE XAI
1. PROHIBIDO conectarse a internet, GEE o APIs externas. Todos los datos están en `data/` y `catalog/`.
2. USA RUTAS PORTABLES:
   from pathlib import Path
   DRIVE_PATH = Path("/content/drive/MyDrive/agropulse_xai")
   BASE_DIR = DRIVE_PATH if DRIVE_PATH.exists() else Path(__file__).resolve().parent.parent
3. ESQUEMAS EXACTOS EN `data/`:
   - `D1_D3_grid30m_cdl_srtm.parquet`: `pixel_id`, `lon`, `lat`, `elevation_m`, `slope_deg`, `slope_pct`, `crop_2018`..`crop_2024`.
   - `D2_ssurgo_soil.parquet`: `mukey`, `muname`, `compname`, `comppct_r`, `hzdept_r`, `hzdepb_r`, `materia_organica`, `agua_disponible_awc`, `erodabilidad_k`, `arena`, `limo`, `arcilla`.
   - `D4_hls_ndvi_bsi_series.parquet`: `date`, `ndvi_mean`, `bsi_mean`, `bare_soil_pct`.
   - `D5_D6_smap_gpm_daily.parquet`: `date`, `sm_rootzone`, `precipitation`.
   - `D7_nasa_power_daily.parquet`: `date`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `ALLSKY_SFC_SW_DWN`, `RH2M`.
   - `catalog/crops_catalog.json`: llaves `cdl_to_crop` y `crops`.
4. CONTRATO DE FUNCIONES EN `core/`:
   - `harmonizer.py`: `build_harmonized_grid() -> pd.DataFrame` y `rank_rotations(weights: dict, metrics_df: pd.DataFrame) -> pd.DataFrame`.
   - `phenology.py`: `analyze_phenology() -> dict`.
   - `climate_water.py`: `compute_hydro_thermal() -> pd.DataFrame` y `simulate_what_if(delta_rain_pct: float, delta_temp_c: float) -> dict`.
   - `soil_agronomy.py`: `evaluate_strategies(grid_df: pd.DataFrame) -> pd.DataFrame`.
   - `xai_uncertainty.py`: `generate_xai_report(strategies_df: pd.DataFrame, pheno: dict, hydro: dict) -> dict`.
5. Código vectorizado (`numpy`/`pandas`/`scipy`), explicable y sin cajas negras.
