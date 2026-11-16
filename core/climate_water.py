# -*- coding: utf-8 -*-
"""core/climate_water.py — [Estadístico 2] AgroPulse XAI (Versión Unificada Final)
Índices hídrico-térmicos (CWHI, z-scores, SPI-30), balance hídrico por cultivo (bucket FAO-56 + CN Williams 1995),
ponderación por cultivos comerciales con efecto físico de cobertura, serie larga 1981-2025 y simulador What-If.

Entradas  (data/parquet_VERIFICADO): aligned_2016_2025, climate_long_1981_2025, grid_30m
          (catalog/crops_catalog.json, data/rotations.json, data/calib_estadistico2.json)
Salidas   (data/): metrics_climate_water, strategy_climate_metrics, crop_climate_metrics,
          climate_indices_weekly, whatif_grid, strategy_climate_metrics_long_1981_2025 (.parquet) y climate_params.json
Uso: from core import climate_water as cw; cw.run_all(); cw.simulate_what_if(-20, 2); cw.whatif_fast(-15, 1.5)
"""
import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def _resolve_base():
    """Carpeta del proyecto: la primera candidata que tenga data/parquet_VERIFICADO/grid_30m.parquet."""
    cands = [Path('/content/drive/MyDrive/agropulse_xai')] + sorted(Path('/content/drive/MyDrive').glob('*agropulse*')) \
        + sorted(Path('/content/drive').glob('Shareddrives/*/agropulse_xai'))
    for c in cands:
        if (c / 'data' / 'parquet_VERIFICADO' / 'grid_30m.parquet').exists():
            return c
    return cands[0]


BASE = _resolve_base()
DATA, PQ = BASE / 'data', BASE / 'data' / 'parquet_VERIFICADO'
FINAL_REL = ('RUTA_FINAL', '1_TRABAJO_EQUIPO', 'RESULTADOS A REVISAR (ESTADISTICOS)', 'ESTADISTICO 2')
OUT_FILES = ['metrics_climate_water.parquet', 'strategy_climate_metrics.parquet', 'crop_climate_metrics.parquet',
             'climate_indices_weekly.parquet', 'whatif_grid.parquet', 'strategy_climate_metrics_long_1981_2025.parquet', 'climate_params.json']
R_RANGE, T_RANGE = (-40.0, 30.0), (0.0, 4.0)

PRM = dict(k_et0=0.937,                      # ET0 = k * Hargreaves (calibrado vs Priestley-Taylor)
           zr_ref=1.0, p_ref=0.70, cn_ref=78, # bucket de referencia (calibrado vs SMAP)
           p_scale=1.0,                      # factor IMERG (1 = sin ajuste)
           cwhi_w=(0.4, 0.3, 0.3),           # peso Sw, ST, Sw*ST
           cri_w=(0.4, 0.2, 0.2, 0.2),       # pérdida rend., calor, helada, CWHI
           heat_c=32.0, hardfrost_c=-2.2, erosive_mm=12.7,
           slope_edges=[1, 2, 3, 4, 6, 8],   # bordes de pendiente (°) -> 7 clases hidro-topográficas
           cover_cn_bonus=-4.0,              # reducción de CN (menor escorrentía) en cultivo comercial tras cobertura (NRCS/SWAT)
           cover_awc_bonus=1.08, sigma_smap_m3m3=0.04,             # +8% retención efectiva por residuo de cultivo de cobertura invernal
           r_grid=list(range(-40, 31, 10)), t_grid=[0, 1, 2, 3, 4])

if (DATA / 'calib_estadistico2.json').exists():
    try:
        _c = json.load(open(DATA / 'calib_estadistico2.json'))
        PRM.update(k_et0=_c.get('k_et0', PRM['k_et0']), zr_ref=_c.get('zr_ref_m', PRM['zr_ref']), p_ref=_c.get('p_dep_ref', PRM['p_ref']))
    except Exception:
        pass

P90 = ['water_stress_index', 'yield_loss_water_pct', 'climate_risk_idx']
M5 = ['water_stress_index', 'yield_loss_water_pct', 'climate_risk_idx', 'water_demand_mm', 'water_deficit_mm']
MS = M5 + ['water_use_ratio', 'frost_prob']
ALIAS = {'soybeans': 'soybean', 'soybean': 'soybean', 'soy': 'soybean', 'soja': 'soybean', 'soya': 'soybean',
         'cover_crop_rye': 'cereal_rye', 'cover_crop': 'cereal_rye', 'cereal_rye': 'cereal_rye', 'centeno': 'cereal_rye', 'rye': 'cereal_rye',
         'cdl_corn': 'corn', 'corn': 'corn', 'maize': 'corn', 'maiz': 'corn', 'maíz': 'corn',
         'trigo': 'winter_wheat', 'wheat': 'winter_wheat', 'winter_wheat': 'winter_wheat',
         'alfalfa': 'alfalfa', 'avena': 'oats', 'oats': 'oats', 'sorgo': 'sorghum', 'sorghum': 'sorghum', 'barbecho': 'fallow', 'fallow': 'fallow'}


def _cd(plant, st, kc, zr, p, ky, tc, tf, fsd, cn):
    return dict(plant_doy=plant, stage_days=dict(zip(('ini', 'dev', 'mid', 'late'), st)), kc=dict(zip(('ini', 'mid', 'end'), kc)),
                root_depth_m=zr, p_depletion=p, ky_total=ky, t_crit_c=tc, t_frost_c=tf, frost_sensitive_days=fsd, cn=cn)


CROP_DEF = {
    'corn': _cd(120, (25, 40, 50, 30), (.30, 1.20, .60), 1.0, .55, 1.25, 32, -2.2, 20, 78),
    'soybean': _cd(135, (20, 30, 60, 25), (.40, 1.15, .50), 0.9, .50, 0.85, 35, -2.2, 20, 78),
    'alfalfa': _cd(100, (10, 30, 120, 30), (.40, .95, .90), 1.0, .55, 1.10, 35, -4.0, 10, 58),
    'winter_wheat': _cd(275, (120, 60, 70, 35), (.40, 1.15, .25), 1.0, .55, 1.00, 30, -2.2, 0, 75),
    'cereal_rye': _cd(270, (60, 60, 75, 30), (.30, .90, .40), 0.8, .60, 0.0, 35, -10.0, 0, 75),
    'sorghum': _cd(140, (20, 35, 40, 30), (.30, 1.00, .55), 1.0, .55, 0.90, 38, -2.2, 15, 78),
    'oats': _cd(95, (20, 30, 40, 25), (.30, 1.15, .25), 0.9, .55, 1.00, 30, -2.2, 10, 75),
    'other_hay': _cd(100, (10, 30, 100, 30), (.40, .90, .85), 0.9, .55, 0.90, 35, -4.0, 10, 58),
    'fallow': _cd(1, (365, 0, 0, 0), (.30, .30, .30), 0.3, .50, 0.0, 99, -99, 0, 86),
}
ROT_FALLBACK = {'R1': ['corn', 'soybean', 'corn'], 'R2': ['corn', 'soybean', 'winter_wheat'], 'R3': ['corn', 'soybean', 'alfalfa']}
_CTX, _CTX_L = {}, {}


# ----------------------------------------------------------------------------- utilidades
def _doy_clim(s, win=7, floor=0.0):
    """Media y desviación por día del año, suavizadas con ventana circular (2*win+1)."""
    g = s.groupby(s.index.dayofyear)
    out = []
    for x in (g.mean().reindex(range(1, 367)), g.std().reindex(range(1, 367))):
        x = x.interpolate(limit_direction='both').values
        v = np.r_[x[-win:], x, x[:win]]
        out.append(pd.Series(v).rolling(2 * win + 1, center=True).mean().values[win:-win])
    return out[0], np.maximum(out[1], floor)


def cn_slope(cn2, slope_deg):
    """CN (AMC-II) ajustado por pendiente, Williams (1995)/SWAT."""
    cn2 = float(cn2); s = float(np.tan(np.radians(slope_deg)))
    cn3 = cn2 * np.exp(0.00673 * (100.0 - cn2))
    return float(np.clip((cn3 - cn2) / 3.0 * (1.0 - 2.0 * np.exp(-13.86 * s)) + cn2, 30.0, 98.0))


def _cwhi(sw, st):
    a, b, c = PRM['cwhi_w']
    return 100.0 * (a * sw + b * st + c * sw * st)


def ref_bucket(p, e, taw, pdep, cn):
    """Bucket de referencia (pasto): devuelve contenido relativo de agua RWC = 1 - Dr/TAW."""
    s_ = 25400.0 / cn - 254.0; ia = 0.2 * s_; raw = pdep * taw; dr = 0.0; out = np.empty(len(p))
    for i in range(len(p)):
        ro = (p[i] - ia) ** 2 / (p[i] - ia + s_) if p[i] > ia else 0.0
        ks = 1.0 if dr <= raw else max(0.0, (taw - dr) / (taw - raw))
        dr = min(max(dr - (p[i] - ro) + ks * e[i], 0.0), taw)
        out[i] = 1.0 - dr / taw
    return out


def _kc_curve(kc, st):
    a, b, c, d = (int(st[k]) for k in ('ini', 'dev', 'mid', 'late'))
    dev = np.linspace(kc['ini'], kc['mid'], b + 2)[1:-1] if b > 0 else np.array([])
    late = np.linspace(kc['mid'], kc['end'], d + 2)[1:-1] if d > 0 else np.array([])
    return np.concatenate([np.full(a, kc['ini']), dev, np.full(c, kc['mid']), late])


# ----------------------------------------------------------------------------- carga de insumos
def load_crop_params(cat):
    cp = json.loads(json.dumps(CROP_DEF)); src = {k: 'defecto' for k in cp}
    crops_dict = cat.get('crops') if isinstance(cat, dict) and 'crops' in cat else (cat if isinstance(cat, dict) else {})
    for k, v in (crops_dict or {}).items():
        if not isinstance(v, dict):
            continue
        k_norm = ALIAS.get(str(k).lower(), str(k).lower())
        cl = v.get('climate')
        if k_norm not in cp and not cl:
            continue
        cur = cp.setdefault(k_norm, {}); src.setdefault(k_norm, 'defecto')
        if ('kc_mid' in v or 'kc' in v) and not (cl and 'kc' in cl) and 'kc' in cur:
            cur['kc']['mid'] = float(v.get('kc_mid', v.get('kc')))
        for kk, vv in (cl or {}).items():
            if kk == 'cn_by_hsg':
                cur['cn'] = vv.get('B', cur.get('cn', 78))
            elif isinstance(vv, dict) and isinstance(cur.get(kk), dict):
                cur[kk].update(vv)
            else:
                cur[kk] = vv
            src[k_norm] = 'catalogo'
    need = ['plant_doy', 'stage_days', 'kc', 'root_depth_m', 'p_depletion', 'ky_total', 't_crit_c', 't_frost_c', 'frost_sensitive_days', 'cn']
    for k, v in cp.items():
        miss = [n for n in need if n not in v]
        if miss:
            raise KeyError(f"Cultivo '{k}' sin parámetros: {miss}")
    return cp, src


def _find_catalog():
    """Busca TODAS las copias de crops_catalog.json bajo BASE y elige la que tenga más cultivos con bloque 'climate'."""
    roots = [BASE] + [b for b in [Path('/content/drive/MyDrive/agropulse_xai')] + sorted(Path('/content/drive/MyDrive').glob('*agropulse*'))
                      if b.exists() and b != BASE]
    found = {}
    for b in roots:
        for root, dirs, files in os.walk(b, followlinks=True):
            dirs[:] = [d for d in dirs if d != 'RUTA_FINAL' and 'ANTIGUO' not in d]
            for fn in ('crops_catalog.json', 'crops_catalog_crudo.json'):
                if fn in files:
                    p = Path(root) / fn
                    try:
                        raw = p.read_bytes()
                    except OSError:
                        continue
                    found.setdefault(hashlib.md5(raw).hexdigest(), (p, raw))
    if not found:
        if (DATA / 'rotations.json').exists():
            return DATA / 'rotations.json'
        raise FileNotFoundError(f"crops_catalog.json no encontrado bajo {BASE}.")
    def n_climate(raw):
        try:
            d = json.loads(raw)
            cd = d.get('crops') if isinstance(d, dict) and 'crops' in d else (d if isinstance(d, dict) else {})
            return sum(1 for v in (cd or {}).values() if isinstance(v, dict) and v.get('climate'))
        except Exception:
            return -1
    info = [(n_climate(raw), 1 if p.name == 'crops_catalog.json' else 0, p.stat().st_mtime, p) for p, raw in found.values()]
    best = max(info, key=lambda t: (t[0], t[1], t[2]))[3]
    if len(info) > 1:
        det = '; '.join(f"{p} (climate={n}, mod={pd.Timestamp(m, unit='s').strftime('%Y-%m-%d %H:%M')})" for n, _, m, p in info)
        warnings.warn(f"Hay {len(info)} versiones distintas de catálogo: {det}. Uso: {best}")
    return best


def _find_rotations():
    """rotations*.json bajo data/ (sigue accesos directos de Drive, ignora RUTA_FINAL)."""
    fixed = DATA / 'rotations.json'
    if fixed.exists():
        return [fixed]
    found = []
    for root, _, files in os.walk(DATA, followlinks=True):
        if 'RUTA_FINAL' in Path(root).parts:
            continue
        found += [Path(root) / f for f in files if f.lower().startswith('rotations') and f.lower().endswith('.json')]
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(found) > 1:
        warnings.warn(f"Varias copias de rotations*.json; uso la más reciente: {found[0]}")
    return found


def load_rotations(cat, cp):
    paths = _find_rotations()
    if not paths:
        warnings.warn(f"rotations.json NO encontrado en {DATA}: usando rotaciones de respaldo R1-R3 (provisorias)")
        return dict(ROT_FALLBACK), 'respaldo'
    raw = json.load(open(paths[0])); s = raw.get('strategies') or raw.get('rotations') or raw
    items = [(it.get('rotation_id') or it.get('strategy_id') or it.get('id'), it) for it in s] if isinstance(s, list) else list(s.items())
    c2c = cat.get('cdl_to_crop', {}) if isinstance(cat, dict) else {}; out = {}
    for rid, v in items:
        if str(rid).lower() in ('crops', 'metadata', 'description', 'version', 'notes'):
            continue
        seq = v if isinstance(v, list) else (v.get('seq') or v.get('sequence') or v.get('crops') or []) if isinstance(v, dict) else []
        if seq and all(isinstance(x, dict) for x in seq):
            seq = sorted(seq, key=lambda x: x.get('year_idx', x.get('year', 0))); seq = [x.get('crop_id') or x.get('crop') for x in seq]
        seq = [c2c.get(str(x), x) if not isinstance(x, str) or str(x).isdigit() else x for x in seq]
        seq = [ALIAS.get(str(x).lower().strip(), str(x).lower().strip()) for x in seq]
        bad = [x for x in seq if x not in cp]
        if bad or not seq:
            warnings.warn(f"Rotación {rid} descartada (cultivos sin parámetros: {bad})"); continue
        out[str(rid)] = seq
    if not out:
        raise ValueError("rotations.json sin rotaciones válidas")
    return out, str(paths[0])


def build_context(rebuild=False, source='aligned'):
    """source='aligned': clima 2016-2025 (IMERG+SMAP+POWER). source='long': POWER 1981-2025 con P reescalada a IMERG (sin SMAP)."""
    store = _CTX if source == 'aligned' else _CTX_L
    if store and not rebuild:
        return store
    lg = pd.read_parquet(PQ / 'climate_long_1981_2025.parquet').set_index('date').sort_index()
    al = pd.read_parquet(PQ / 'aligned_2016_2025.parquet').set_index('date').sort_index()
    al.index = pd.to_datetime(al.index); lg.index = pd.to_datetime(lg.index)
    ov = pd.concat([al.gpm_rain_mm, lg.precip_power_mm], axis=1, sort=True).dropna()
    kp = float(ov.iloc[:, 0].sum() / max(ov.iloc[:, 1].sum(), 1e-9))
    oe = pd.concat([al.et0_mm, lg.et0_mm], axis=1, sort=True, keys=['a', 'l']).dropna()
    ket = float(oe.a.sum() / max(oe.l.sum(), 1e-9))
    if abs(ket - 1) > 0.05 and source == 'aligned':
        warnings.warn(f"ET0 de aligned y climate_long difieren x{ket:.3f} en 2016-2025: la serie larga se reescala por ese factor")

    # Estadísticos históricos 1981-2025 para bandera_extrapolacion y propagación de error de entrada
    p_ann = (lg.precip_power_mm * kp).groupby(lg.index.year).sum()
    t_ann = lg.tmax_c.groupby(lg.index.year).mean()
    r_obs_bounds = (float(((p_ann.quantile(0.05) / p_ann.mean()) - 1) * 100), float(((p_ann.quantile(0.95) / p_ann.mean()) - 1) * 100))
    t_obs_bounds = (float(t_ann.quantile(0.05) - t_ann.mean()), float(t_ann.quantile(0.95) - t_ann.mean()))
    ann_diff = ov.groupby(ov.index.year).apply(lambda x: (x.iloc[:, 0].sum() - x.iloc[:, 1].sum() * kp) / max(x.iloc[:, 1].sum() * kp, 1) * 100)
    sigma_in = dict(sigma_rain_pct=float(ann_diff.std()), sigma_temp_c=1.2)

    g = pd.read_parquet(PQ / 'grid_30m.parquet').copy()
    if source == 'long':
        al = pd.DataFrame(dict(tmax_c=lg.tmax_c, tmin_c=lg.tmin_c, et0_mm=lg.et0_mm * ket, gpm_rain_mm=lg.precip_power_mm * kp, smap_rootzone=np.nan))
    al = al.reindex(pd.date_range(al.index.min(), al.index.max(), freq='D'))
    for c in ('tmax_c', 'tmin_c', 'et0_mm', 'smap_rootzone'):
        al[c] = al[c].interpolate(limit=3, limit_direction='both')
    nan_p = float(al.gpm_rain_mm.isna().mean()); al['gpm_rain_mm'] = al.gpm_rain_mm.fillna(0.0).clip(lower=0)
    dates = al.index; doy = dates.dayofyear.values
    P = al.gpm_rain_mm.values * PRM['p_scale']; Tx, Tn = al.tmax_c.values, al.tmin_c.values; Tm = (Tx + Tn) / 2
    E = (al.et0_mm * PRM['k_et0']).ffill().bfill().values
    mu_t, sd_t = _doy_clim(lg.tmax_c, floor=1.0); zT = (Tx - mu_t[doy - 1]) / sd_t[doy - 1]
    if source == 'long':
        zS, cw_obs = np.zeros(len(P)), None
    else:
        mu_s, sd_s = _doy_clim(al.smap_rootzone.dropna(), floor=0.01)
        zS = (al.smap_rootzone.ffill().bfill().values - mu_s[doy - 1]) / sd_s[doy - 1]
        cw_obs = _cwhi(np.clip(-zS / 2, 0, 1), np.clip(zT / 2, 0, 1))
    C = dict(dates=dates, doy=doy, P=P, E=E, Tx=Tx, Tn=Tn, Tm=Tm, zT=zT, zS=zS, sdT=sd_t, cw_obs=cw_obs, years=sorted(set(dates.year)), N=len(dates),
             p_long=lg.precip_power_mm * kp, kp=kp, ket=ket, nan_p=nan_p, smap=al.smap_rootzone.values, source=source,
             r_obs_bounds=r_obs_bounds, t_obs_bounds=t_obs_bounds, sigma_in=sigma_in)

    # Clases hidro-topográficas: suelo (awc) x pendiente (Williams 1995)
    g['hydro_class'] = np.digitize(g.slope_deg.fillna(g.slope_deg.median()).values, PRM['slope_edges']).astype(int)
    g['awc_key'] = g.awc.round(3).astype(str) + '|' + g.hydro_class.astype(str)
    grp = {}
    for k, gg in g.groupby('awc_key'):
        a, sl = float(gg.awc.mean()), float(gg.slope_deg.mean()); cn_r = cn_slope(PRM['cn_ref'], sl)
        rwc = ref_bucket(P, E, a * 1000 * PRM['zr_ref'], PRM['p_ref'], cn_r)
        mu, sd = _doy_clim(pd.Series(rwc, index=dates), floor=0.02)
        sw = np.clip(-((rwc - mu[doy - 1]) / sd[doy - 1]) / 2, 0, 1)
        grp[k] = dict(awc=a, slope=sl, cn_ref=cn_r, n_px=len(gg), rwc=rwc, mu=mu, sd=sd, cw_b=_cwhi(sw, np.clip(zT / 2, 0, 1)))
    if source == 'long':
        b = build_context()
        cat, cp, src, rot, rsrc, cat_path = b['cat'], b['cp'], b['src'], b['rot'], b['rsrc'], b['cat_path']
    else:
        cat_path = _find_catalog()
        with open(cat_path, encoding='utf-8') as f:
            cat = json.load(f)
        cp, src = load_crop_params(cat); rot, rsrc = load_rotations(cat, cp)
    store.clear(); store.update(C=C, grp=grp, grid=g, cp=cp, src=src, rot=rot, rsrc=rsrc, cat=cat, cat_path=cat_path, lg=lg, al=al)
    return store


# ----------------------------------------------------------------------------- balance hídrico por cultivo
def _simulate_crop(crop, cp, awc, P, E, Tx, Tn, rwc, cw, C):
    st = cp['stage_days']; L = int(sum(st.values())); kc = _kc_curve(cp['kc'], st); N = C['N']
    starts = [pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=int(cp['plant_doy']) - 1) for y in C['years']]
    s0 = C['dates'].get_indexer(starts); s0 = np.where(s0 < 0, N, s0)
    vlen = np.clip(N - s0, 0, L)
    raw_idx = s0[:, None] + np.arange(L)[None, :]
    idx = np.where(raw_idx < N, raw_idx, np.clip(raw_idx - 365, 0, N - 1))
    taw = awc * 1000.0 * cp['root_depth_m']; raw = cp['p_depletion'] * taw
    s_ = 25400.0 / cp['cn'] - 254.0; ia = 0.2 * s_
    Pw, Ew = P[idx], E[idx]; Dr = np.clip((1 - rwc[np.clip(s0, 0, N - 1)]) * taw, 0, taw)
    ny = len(C['years']); z = lambda: np.zeros(ny)
    etc_s, eta_s, ro_s, pe_s, dp_s, nst, ksm = z(), z(), z(), z(), z(), z(), np.ones(ny)
    for t in range(L):
        p = Pw[:, t]; ro = np.where(p > ia, (p - ia) ** 2 / (p - ia + s_), 0.0); pe = p - ro
        etc = kc[t] * Ew[:, t]
        ks = np.where(Dr <= raw, 1.0, np.maximum(0.0, (taw - Dr) / (taw - raw)))
        eta = ks * etc; dn = Dr - pe + eta
        dp_s += np.maximum(-dn, 0.0); Dr = np.clip(dn, 0.0, taw)
        etc_s += etc; eta_s += eta; ro_s += ro; pe_s += pe; nst += (ks < 1.0); ksm = np.minimum(ksm, ks)
    wsi = np.where(etc_s > 0, 1 - eta_s / np.maximum(etc_s, 1e-9), 0.0)
    yl = np.clip(100 * cp['ky_total'] * wsi, 0, 100)
    heat = (Tx[idx] > cp['t_crit_c']).sum(1); fsd = int(cp['frost_sensitive_days'])
    frost = (Tn[idx][:, :fsd] <= cp['t_frost_c']).any(1).astype(float) if fsd > 0 else np.zeros(ny)
    cwm = cw[idx].mean(1); a, b, c_, d = PRM['cri_w']
    cri = a * np.minimum(yl / 50, 1) + b * heat / L + c_ * frost + d * cwm / 100
    df = pd.DataFrame(dict(crop_id=crop, weather_year=C['years'], etc_mm=etc_s, eta_mm=eta_s, water_deficit_mm=etc_s - eta_s, p_eff_mm=pe_s,
                           runoff_mm=ro_s, drainage_mm=dp_s, balance_p_etc_mm=Pw.sum(1) - etc_s, ks_min=ksm, stress_days=nst,
                           water_stress_index=wsi, yield_loss_water_pct=yl, heat_days_crit=heat, frost_event=frost, cwhi_mean=cwm,
                           cwhi_max=cw[idx].max(1), climate_risk_idx=cri, valid_frac=vlen / L))
    bad = vlen < min(L, 60)
    df.loc[bad, df.columns.difference(['crop_id', 'weather_year', 'valid_frac'])] = np.nan
    return df


def run_group(key, ctx, dR=0.0, dT=0.0):
    C, G = ctx['C'], ctx['grp'][key]; awc = G['awc']
    P = C['P'] * (1 + dR / 100.0); Tx, Tn = C['Tx'] + dT, C['Tn'] + dT
    E = C['E'] * np.maximum(C['Tm'] + dT + 17.8, 1.0) / np.maximum(C['Tm'] + 17.8, 1.0)
    if dR == 0 and dT == 0:
        rwc, cw = G['rwc'], (G['cw_b'] if C['cw_obs'] is None else C['cw_obs'])
    else:
        rwc = ref_bucket(P, E, awc * 1000 * PRM['zr_ref'], PRM['p_ref'], G['cn_ref'])
        sw = np.clip(-((rwc - G['mu'][C['doy'] - 1]) / G['sd'][C['doy'] - 1]) / 2, 0, 1)
        st = np.clip((C['zT'] + dT / C['sdT'][C['doy'] - 1]) / 2, 0, 1)
        cw = _cwhi(sw, st) if C['cw_obs'] is None else np.clip(C['cw_obs'] + _cwhi(sw, st) - G['cw_b'], 0, 100)
    sims = []
    for k, cpar in ctx['cp'].items():
        cn_k = cn_slope(cpar['cn'], G['slope'])
        sims.append(_simulate_crop(k, dict(cpar, cn=cn_k), awc, P, E, Tx, Tn, rwc, cw, C))
        if cpar['ky_total'] > 0:
            # Variante física con residuo de cobertura precedente (menor CN + mayor retención hídrica)
            cn_cov = cn_slope(max(35.0, cpar['cn'] + PRM['cover_cn_bonus']), G['slope'])
            sims.append(_simulate_crop(k + '__cov', dict(cpar, cn=cn_cov), awc * PRM['cover_awc_bonus'], P, E, Tx, Tn, rwc, cw, C))
    out = pd.concat(sims, ignore_index=True)
    out.insert(0, 'awc_key', key)
    return out


def aggregate_rotations(cm, rot, cp=None):
    """Una fila por (rotación, año climático de inicio). Métricas de ingreso (estrés, pérdida, riesgo, helada) promedian solo los cultivos
    comerciales (ky_total>0): la cobertura (centeno) no diluye la pérdida, pero transmite su efecto físico (residuo/infiltración) al cultivo comercial."""
    years = np.array(sorted(cm.weather_year.unique()))
    cols = ['etc_mm', 'water_deficit_mm', 'p_eff_mm', 'water_stress_index', 'yield_loss_water_pct', 'climate_risk_idx', 'frost_event']
    cash_m = ('water_stress_index', 'yield_loss_water_pct', 'climate_risk_idx', 'frost_event')
    piv = {m: cm.pivot(index='weather_year', columns='crop_id', values=m) for m in cols}
    rows = []
    for rid, seq in rot.items():
        n = len(seq); wy = years[years + n - 1 <= years.max()]
        has_cover = any((cp is not None and cp[c]['ky_total'] == 0) for c in seq)
        w = np.array([1.0 if (cp is None or cp[c]['ky_total'] > 0) else 0.0 for c in seq])
        seq_eff = [c + '__cov' if (has_cover and cp is not None and cp[c]['ky_total'] > 0 and (c + '__cov') in piv['etc_mm'].columns) else c for c in seq]
        v = {}
        for m in cols:
            arr = np.array([piv[m][c].reindex(wy + j).values for j, c in enumerate(seq_eff)])
            v[m] = np.where(w[:, None] > 0, arr, 0.0).sum(0) / w.sum() if (m in cash_m and w.sum() > 0) else arr.mean(0)
        rows.append(pd.DataFrame(dict(rotation_id=rid, clim_year=wy, water_demand_mm=v['etc_mm'], water_deficit_mm=v['water_deficit_mm'],
                                      water_stress_index=v['water_stress_index'], yield_loss_water_pct=v['yield_loss_water_pct'],
                                      climate_risk_idx=v['climate_risk_idx'], water_use_ratio=v['p_eff_mm'] / v['etc_mm'], frost_prob=v['frost_event'])).dropna())
    df = pd.concat(rows, ignore_index=True)
    common = set.intersection(*[set(g.clim_year) for _, g in df.groupby('rotation_id')])
    return df[df.clim_year.isin(common)].reset_index(drop=True)


def summarise(st):
    g = st.groupby('rotation_id')[MS]
    p10 = st.groupby('rotation_id')[M5].quantile(0.10).add_suffix('_p10')
    p90 = st.groupby('rotation_id')[M5].quantile(0.90).add_suffix('_p90')
    return g.mean().join(g.std().add_suffix('_sd')).join(p10).join(p90).reset_index()


def _to_pixels(df, grid):
    return grid[['pixel_id', 'awc_key', 'hydro_class']].merge(df, on='awc_key', how='left').drop(columns='awc_key')


def _all_groups(ctx, dR, dT):
    cms, sts, evs = [], [], []
    for a in ctx['grp']:
        cm = run_group(a, ctx, dR, dT)
        cm_clean = cm[~cm.crop_id.str.endswith('__cov')].reset_index(drop=True)
        st = aggregate_rotations(cm, ctx['rot'], ctx['cp']); ev = summarise(st)
        for d in (st, ev):
            d.insert(0, 'awc_key', a)
        cms.append(cm_clean); sts.append(st); evs.append(ev)
    return pd.concat(cms, ignore_index=True), pd.concat(sts, ignore_index=True), pd.concat(evs, ignore_index=True)


# ----------------------------------------------------------------------------- índices semanales (baseline)
def _spi30(p_obs, p_long):
    p30l, p30o = p_long.rolling(30, min_periods=30).sum(), p_obs.rolling(30, min_periods=30).sum()
    spi, pan = pd.Series(np.nan, index=p_obs.index), pd.Series(np.nan, index=p_obs.index)
    for m in range(1, 13):
        xl = p30l[p30l.index.month == m].dropna(); mk = (p30o.index.month == m); xo = p30o[mk]
        q0 = float((xl <= 0.1).mean()); pos = xl[xl > 0.1]
        a, _, sc = stats.gamma.fit(pos, floc=0)
        cdf = q0 + (1 - q0) * stats.gamma.cdf(xo.clip(lower=0).values, a, loc=0, scale=sc)
        spi[mk] = stats.norm.ppf(np.clip(cdf, 1e-4, 1 - 1e-4)); pan[mk] = ((xo / xl.mean()) - 1).values * 100
    return spi, pan


def weekly_indices(ctx):
    C = ctx['C']; d = C['dates']
    spi, pan = _spi30(pd.Series(C['P'], index=d), C['p_long'])
    df = pd.DataFrame(dict(z_sm=C['zS'], z_tmax=C['zT'], spi30=spi.values, p_anom_pct=pan.values, cwhi=C['cw_obs'],
                           compound=((C['zS'] < -1) & (C['zT'] > 1)).astype(int), heat=(C['Tx'] >= PRM['heat_c']).astype(int),
                           frost=(C['Tn'] <= 0).astype(int), hardfrost=(C['Tn'] <= PRM['hardfrost_c']).astype(int),
                           erosive=(C['P'] >= PRM['erosive_mm']).astype(int), p=C['P'], et0=C['E'], valid=1.0), index=d)
    w = df.resample('W-MON', closed='left', label='left').agg(dict(
        z_sm='mean', z_tmax='mean', spi30='mean', p_anom_pct='mean', cwhi=['mean', 'max'], compound='sum', heat='sum', frost='sum',
        hardfrost='sum', erosive='sum', p=['sum', 'max'], et0='sum', valid='sum'))
    w.columns = ['z_sm', 'z_tmax', 'spi30', 'p_anom_pct', 'cwhi_mean', 'cwhi_max', 'compound_days', 'heat_days', 'frost_days',
                 'hardfrost_days', 'n_erosive_days', 'p_week_mm', 'p_max_day_mm', 'et0_week_mm', 'valid_frac']
    w['valid_frac'] = (w.valid_frac / 7).clip(upper=1); w['balance_p_et0_mm'] = w.p_week_mm - w.et0_week_mm
    w['cwhi_class'] = pd.cut(w.cwhi_mean, [-1, 15, 35, 60, 101], labels=['sin estrés', 'leve', 'moderado', 'severo']).astype(str)
    w = w.reset_index(); w = w.rename(columns={w.columns[0]: 'date'})
    return ctx['grid'][['pixel_id']].merge(w, how='cross')


# ----------------------------------------------------------------------------- What-If e Incertidumbre
def is_extrapolation(delta_rain_pct, delta_temp_c, ctx=None):
    """True si el escenario supera el soporte de los sliders o el dominio histórico observado en 1981-2025 (Fase 1 Mejorada, Sec. 4)."""
    ctx = ctx or build_context()
    out_slider = not (R_RANGE[0] <= delta_rain_pct <= R_RANGE[1] and T_RANGE[0] <= delta_temp_c <= T_RANGE[1])
    rmin, rmax = ctx['C']['r_obs_bounds']; tmin, tmax = ctx['C']['t_obs_bounds']
    out_hist = (delta_rain_pct < rmin or delta_rain_pct > rmax or delta_temp_c < tmin or delta_temp_c > tmax)
    joint_z = ((delta_rain_pct / 20.0) ** 2 + (delta_temp_c / 1.8) ** 2) ** 0.5
    return bool(out_slider or out_hist or joint_z > 1.5)


def simulate_what_if(delta_rain_pct, delta_temp_c, rotation_ids=None, ctx=None):
    """Recalcula exacto (sin interpolar) para un escenario. Devuelve pixel_id × rotation_id con métricas, sd y deltas vs línea base."""
    ctx = ctx or build_context()
    flag = is_extrapolation(delta_rain_pct, delta_temp_c, ctx)
    dR, dT = float(np.clip(delta_rain_pct, *R_RANGE)), float(np.clip(delta_temp_c, *T_RANGE))
    ev = _all_groups(ctx, dR, dT)[2]; b = _all_groups(ctx, 0.0, 0.0)[2]
    key = ['awc_key', 'rotation_id']; ev = ev.merge(b[key + M5], on=key, suffixes=('', '_base'))
    for m in M5:
        ev[m + '_delta'] = ev[m] - ev[m + '_base']
    ev = ev.drop(columns=[m + '_base' for m in M5]); ev.insert(1, 'delta_rain_pct', dR); ev.insert(2, 'delta_temp_c', dT); ev['bandera_extrapolacion'] = flag
    out = _to_pixels(ev, ctx['grid'])
    return out if rotation_ids is None else out[out.rotation_id.isin(rotation_ids)].reset_index(drop=True)


def _bracket(arr, x):
    arr = np.sort(np.asarray(arr, float)); x = float(np.clip(x, arr[0], arr[-1]))
    i = min(int(np.searchsorted(arr, x, 'right')) - 1, len(arr) - 2)
    return arr[i], arr[i + 1], (x - arr[i]) / (arr[i + 1] - arr[i])


def whatif_fast(delta_rain_pct, delta_temp_c, grid=None, ctx=None):
    """Interpolación bilineal sobre whatif_grid.parquet (respuesta instantánea para los sliders)."""
    ctx = ctx or build_context()
    if grid is None:
        grid = ctx['grid_whatif'] if 'grid_whatif' in ctx else pd.read_parquet(DATA / 'whatif_grid.parquet')
    key = ['pixel_id', 'rotation_id']; num = [c for c in grid.columns if c not in key + ['delta_rain_pct', 'delta_temp_c', 'bandera_extrapolacion']]
    r0, r1, wr = _bracket(grid.delta_rain_pct.unique(), delta_rain_pct); t0, t1, wt = _bracket(grid.delta_temp_c.unique(), delta_temp_c)
    cr = lambda a, b: grid[(grid.delta_rain_pct == a) & (grid.delta_temp_c == b)].set_index(key)[num]
    out = cr(r0, t0) * (1 - wr) * (1 - wt) + cr(r1, t0) * wr * (1 - wt) + cr(r0, t1) * (1 - wr) * wt + cr(r1, t1) * wr * wt
    base = cr(0.0, 0.0)
    for m in M5:
        out[m + '_delta'] = out[m] - base[m]
    flag = is_extrapolation(delta_rain_pct, delta_temp_c, ctx)
    out.insert(0, 'delta_rain_pct', float(np.clip(delta_rain_pct, *R_RANGE))); out.insert(1, 'delta_temp_c', float(np.clip(delta_temp_c, *T_RANGE)))
    out['bandera_extrapolacion'] = flag
    return out.reset_index()


def propagate_input_uncertainty(n_rep=100, seed=42, ctx=None):
    """Reutiliza el motor What-If como propagador de errores de entrada (IMERG vs POWER y Tmax)."""
    ctx = ctx or build_context()
    rng = np.random.default_rng(seed)
    sr = ctx['C']['sigma_in']['sigma_rain_pct']; st = ctx['C']['sigma_in']['sigma_temp_c']
    drs = np.clip(rng.normal(0.0, sr, n_rep), *R_RANGE)
    dts = np.clip(np.abs(rng.normal(0.0, st, n_rep)), *T_RANGE)
    sims = [whatif_fast(float(r), float(t), ctx=ctx).groupby('rotation_id')[M5].mean() for r, t in zip(drs, dts)]
    arr = np.stack([s.values for s in sims], axis=0)
    return pd.DataFrame(arr.std(axis=0), index=sims[0].index, columns=[m + '_sigma_in' for m in M5]).reset_index()


# ----------------------------------------------------------------------------- ejecución completa
def run_all(write=True):
    ctx = build_context(rebuild=True); C, grid = ctx['C'], ctx['grid']
    cm, st, ev = _all_groups(ctx, 0.0, 0.0)
    crop_px, strat_px = _to_pixels(cm, grid), _to_pixels(st, grid)
    ev_px = _to_pixels(ev, grid)
    cols = ['pixel_id', 'rotation_id', 'hydro_class'] + [c for m in M5 for c in (m, m + '_sd', m + '_p10', m + '_p90')] + ['water_use_ratio', 'frost_prob']
    metrics = ev_px[cols]
    wk = weekly_indices(ctx)

    cl = build_context(rebuild=True, source='long')
    st_l, ev_l = _all_groups(cl, 0.0, 0.0)[1:]
    long_px = _to_pixels(st_l, grid)

    gl = []
    for dR in PRM['r_grid']:
        for dT in PRM['t_grid']:
            e = _all_groups(ctx, float(dR), float(dT))[2]
            e.insert(1, 'delta_rain_pct', dR); e.insert(2, 'delta_temp_c', dT)
            e['bandera_extrapolacion'] = is_extrapolation(dR, dT, ctx)
            gl.append(e)
    grid_wi = _to_pixels(pd.concat(gl, ignore_index=True), grid); ctx['grid_whatif'] = grid_wi
    sig_in = propagate_input_uncertainty(ctx=ctx)
    metrics = metrics.merge(sig_in, on='rotation_id', how='left')

    G0 = next(iter(ctx['grp'].values()))
    sp_smap = float(stats.spearmanr(G0['rwc'], C['smap'], nan_policy='omit')[0])

    f32 = lambda d: d.astype({c: 'float32' for c in d.select_dtypes('float64').columns})
    if write:
        f32(metrics).to_parquet(DATA / 'metrics_climate_water.parquet', index=False)
        f32(strat_px).to_parquet(DATA / 'strategy_climate_metrics.parquet', index=False)
        f32(crop_px).to_parquet(DATA / 'crop_climate_metrics.parquet', index=False)
        f32(wk).to_parquet(DATA / 'climate_indices_weekly.parquet', index=False)
        f32(grid_wi).to_parquet(DATA / 'whatif_grid.parquet', index=False)
        f32(long_px).to_parquet(DATA / 'strategy_climate_metrics_long_1981_2025.parquet', index=False)
        with open(DATA / 'climate_params.json', 'w', encoding='utf-8') as fj:
            json.dump(dict(
                corrida_utc=pd.Timestamp.now('UTC').isoformat(),
                entradas_mtime={f: pd.Timestamp(os.path.getmtime(PQ / f), unit='s').isoformat() for f in os.listdir(PQ) if f.endswith('.parquet')},
                clases_hidro={k: dict(awc=v['awc'], pendiente_media_deg=round(v['slope'], 2), cn_ref=round(v['cn_ref'], 1), n_px=v['n_px']) for k, v in ctx['grp'].items()},
                params={k: v for k, v in PRM.items() if k not in ('r_grid', 't_grid')},
                imerg_sobre_power_baseline=round(C['kp'], 3),
                et0_aligned_sobre_long=round(C['ket'], 3),
                validacion_externa=dict(
                    imerg_sobre_power_ratio=round(C['kp'], 3),
                    et0_aligned_sobre_long=round(C['ket'], 3),
                    spearman_bucket_vs_smap=round(sp_smap, 3),
                    sigma_input_rain_pct=round(C['sigma_in']['sigma_rain_pct'], 2),
                    sigma_input_temp_c=C['sigma_in']['sigma_temp_c'],
                    sigma_smap_m3m3=PRM.get('sigma_smap_m3m3', 0.04),
                    rango_historico_lluvia_pct=[round(x, 1) for x in C['r_obs_bounds']],
                    rango_historico_temp_c=[round(x, 2) for x in C['t_obs_bounds']]
                ),
                anios_clima=[C['years'][0], C['years'][-1]],
                anios_clima_largo=[cl['C']['years'][0], cl['C']['years'][-1]],
                rotaciones=ctx['rot'], rotaciones_fuente=ctx['rsrc'],
                parametros_cultivo_fuente=ctx['src'], n_pixeles=int(len(grid)), n_grupos_awc=len(ctx['grp']),
                supuestos=[
                    'clima puntual (una celda IMERG/POWER) difundido a todos los pixeles: la variacion espacial viene de pendiente (CN Williams 1995) y suelo',
                    'CN ajustado por pendiente (Williams 1995), HSG=B por defecto',
                    'sin riego ni drenaje subsuperficial ni ascenso capilar',
                    'metricas de ingreso (estres, perdida, riesgo) promedian solo cultivos comerciales (ky>0) e incorporan el efecto fisico de infiltracion/retencion del residuo de cobertura',
                    'ventanas de rotacion solapadas (n_eff << n) con interseccion comun para comparacion pareada',
                    'serie larga 1981-2025: P POWER reescalada por IMERG/POWER, ET0 x k_et0'
                ]
            ), fj, indent=1, ensure_ascii=False)
        fd = BASE.joinpath(*FINAL_REL)
        fd.mkdir(parents=True, exist_ok=True)
        for f in OUT_FILES:
            try:
                shutil.copy2(DATA / f, fd / f)
            except OSError as e:
                warnings.warn(f"No se pudo copiar {f} a RUTA_FINAL: {e}")
        shutil.copy2(BASE / 'core' / 'climate_water.py', fd / 'climate_water.py')
        print(f"Copiado a RUTA_FINAL: {fd}")

    print(f"BASE={BASE} | catálogo={ctx['cat_path']}")
    print(f"Píxeles={len(grid)} | grupos AWC={len(ctx['grp'])} | años clima={C['years'][0]}–{C['years'][-1]} | rotaciones ({ctx['rsrc']}): {ctx['rot']}")
    print("Cultivos (fuente parámetros):", ctx['src'])
    print(f"P anual medio={C['P'].sum() / len(C['years']):.0f} mm | ET0 anual calibrado={C['E'].sum() / len(C['years']):.0f} mm | POWER→IMERG x{C['kp']:.3f} | ET0 aligned/long x{C['ket']:.3f}")
    print(f"Clases hidro-topográficas={len(ctx['grp'])} (pendiente media por clase: {[round(v['slope'], 1) for v in ctx['grp'].values()]}) | Spearman bucket-SMAP={sp_smap:.3f}")
    r = ev_px.groupby('rotation_id').water_stress_index.agg(wsi_min_px='min', wsi_max_px='max')
    print('--- 2016-2025 (media, p90 año malo, rango entre píxeles) ---')
    print(ev.groupby('rotation_id')[M5 + [m + '_p90' for m in P90]].mean().join(r).round(3).to_string())
    print(f"--- 1981-2025 ({st_l.clim_year.nunique()} ventanas vs {st.clim_year.nunique()}) ---")
    print(ev_l.groupby('rotation_id')[M5 + [m + '_p90' for m in P90]].mean().round(3).to_string())
    return ctx


if __name__ == '__main__':
    run_all()
