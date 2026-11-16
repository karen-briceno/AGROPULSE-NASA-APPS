"""
core/xai_uncertainty.py  --  AgroPulse XAI  --  v2.6
Estadístico 4: incertidumbre + explicabilidad.

Cambios vs v1 (ver reporte):
  1. Errores sistemáticos (downscaling, clasificación CDL): UNA realización por réplica
     bootstrap (v1 los promediaba sobre los n años -> IC subestimado en sqrt(n)).
  2. Bootstrap estacionario (Politis-Romano) + factor kappa = sqrt(n/n_eff)*t/z (el percentil
     iid con n=10 cubre ~88% y cae a ~59% con rho=0.6); diagnóstico de autocorrelación y
     tendencia (Mann-Kendall).
  3. Error de clasificación CDL como mezcla de Bernoulli (prob. de rotación mal inferida),
     no ruido gaussiano; solo aplica a series históricas inferidas (no a contrafactuales).
  4. Métricas acotadas [0,100] propagadas en espacio logit (sin salir de rango).
  5. Comparación PAREADA entre estrategias con números aleatorios comunes + P(mejor).
  6. Estabilidad de ranking TOPSIS ante pesos del agricultor (Dirichlet) + atribución.
  7. Certeza con topes por calidad de calibración (heurístico / n pequeño / modo global).
  8. Tarjetas XAI por estrategia: evidencia, beneficios, RIESGOS, brechas calculadas,
     criterio humano, y bandera de extrapolación para el simulador What-If.

v2.1: métricas diferenciales (agua/erosión, pueden ser <=0) en espacio aditivo (acotado=None);
      estrategia de referencia sin ruido; p_err usa n real de años; sigma_modelo (incertidumbre
      estructural de contrafactuales); etiquetas honestas (intercomparación != validación en terreno).

v2.2: diagnostico['procedencia'] = {clave: 'medido'|'derivado'|'placeholder'}: la evidencia con
      procedencia placeholder NO se presenta como evidencia satelital y se lista en las brechas;
      validar_contrato() para ensamblar entregables de otros módulos.

v2.3: FIX sigma_modelo ahora entra en el IC del puntaje (v2.1/v2.2 solo lo aplicaban a agua/erosión
      y a la comparación pareada); bandera_extrapolacion devuelve SIEMPRE el aviso de delta-change;
      es_demo por estrategia; notas_brechas en diagnóstico; textos de tarjeta corregidos.

v2.4: evidencia de suelo desnudo como INTERVALO (diag['meses_suelo_desnudo_rango'] = (con_residuo_excluido,
      NDVI/BSI)); texto de frac_hls_valida sin "ventana" (+ diag['frac_hls_invierno']); validar_contrato
      exige coherencia valor/procedencia (None sin placeholder = ERROR).

v2.5: sigma_downscaling_score <= 0/NaN ya NO se acepta como "calibrado" (σ=0 sin variación != error nulo):
      se ignora (fallback heurístico, certeza <= MEDIA) y validar_contrato lo marca ERROR; tarjeta con
      'resumen_agricultor' (3 líneas en lenguaje llano).

v2.6: FIX p_err usaba la longitud de la serie climática (42) como años de CDL (10): ahora n_anios_cdl
      (default 10) -> con 42 años p_err pasaba de 0.40 a 0.88 y ensanchaba artificialmente la línea base;
      demo_metricas por estrategia (cifras provisionales no se muestran como estimaciones);
      evidencia de déficit hídrico con fuente declarada (diag['fuente_deficit']), sin atribuir a SMAP en 1981-2015;
      tensor_bootstrap_criterios() y p_rango_tensor() públicos: errores de entrada COMUNES entre estrategias
      (rho_error), kappa de autocorrelación y criterios no negativos.

Constantes marcadas PLACEHOLDER deben calibrarse con el benchmark de Iowa
(validación cruzada espacial por bloques; matriz de confusión CDL; CoCoRaHS/PRISM).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constantes (PLACEHOLDER = calibrar; no presentar como resultado empírico)
# ----------------------------------------------------------------------------
EFICACIA_CONFIRMACION = 0.90      # PLACEHOLDER: reducción de p_err tras confirmar el agricultor
DELTA_ERROR_CLASIF = 8.0          # PLACEHOLDER: salto del puntaje si la rotación está mal inferida
CORTES_CERTEZA = (0.065, 0.12)    # fracción del rango de la métrica (equivale a 6.5 / 12 en 0-100)


# ----------------------------------------------------------------------------
# Utilidades estadísticas
# ----------------------------------------------------------------------------
def _idx_estacionario(rng: np.random.Generator, n: int, n_boot: int, l: float) -> np.ndarray:
    """Índices del bootstrap estacionario circular (bloque medio l)."""
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    nuevo = rng.random((n_boot, n)) < 1.0 / l
    ini = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(nuevo[:, t], ini[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _bloque_medio(n: int, override: float | None = None) -> float:
    return float(override) if override else float(max(2, round(n ** (1 / 3))))


def diagnostico_serie(x: np.ndarray) -> dict:
    """rho1 (corregido por sesgo), n_eff y tendencia Mann-Kendall (aprox. normal, sin empates)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        return {"n": n, "rho1": np.nan, "n_eff": float(n), "mk_z": np.nan, "tendencia": False}
    xc = x - x.mean()
    den = float(np.sum(xc**2))
    rho = float(np.sum(xc[1:] * xc[:-1]) / den) if den > 0 else 0.0
    rho = float(np.clip(rho, 0.0, 0.9))   # solo autocorrelación positiva; crudo (la corrección de
    n_eff = float(np.clip(n * (1 - rho) / (1 + rho), 2, n))   # sesgo sobre-penaliza: ver simulación)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    return {"n": n, "rho1": round(rho, 3), "n_eff": round(n_eff, 1),
            "mk_z": round(float(z), 2), "tendencia": bool(abs(z) > 1.96)}


def _t975(df: float) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:   # Cornish-Fisher (df >= 3)
        z = 1.959964
        return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def _kappa(serie: np.ndarray) -> float:
    """Factor de ensanchamiento del bootstrap de la media climática: sqrt(n/n_eff) * t_{n_eff-1}/z.
    Corrige (i) autocorrelación (n_eff) y (ii) subcobertura del percentil con n pequeño."""
    n = len(serie)
    ne = diagnostico_serie(serie)["n_eff"]
    return float(np.sqrt(n / ne) * _t975(max(ne - 1.0, 2.0)) / 1.959964)


def _logit(p):
    return np.log(p / (1 - p))


def _a_unidad(x, lo, hi):
    return np.clip((np.asarray(x, float) - lo) / (hi - lo), 1e-3, 1 - 1e-3)


# ----------------------------------------------------------------------------
# Núcleo: una réplica = (remuestreo climático) + (UNA realización de errores sistemáticos)
# ----------------------------------------------------------------------------
def _sigma_espacial(error_downscaling_math, ratio_grueso, sigma_downscaling_score):
    """Preferir sigma_downscaling_score (RMSE de CV espacial propagado al score, en unidades
    del score, calculado por Est. 2 / Matemático). Fallback heurístico v1 si no existe."""
    if sigma_downscaling_score is not None and np.isfinite(sigma_downscaling_score) and sigma_downscaling_score > 0:
        return float(sigma_downscaling_score), "calibrado"
    return float(error_downscaling_math) * float(ratio_grueso), "heuristico"


def _p_error_rotacion(acc_cdl, n_anios, confirmado):
    """P(al menos un año mal clasificado) = 1 - acc^n; el agricultor reduce pero no anula."""
    p = 1.0 - float(acc_cdl) ** int(n_anios)
    return p * (1.0 - EFICACIA_CONFIRMACION) if confirmado else p


def _replicas(serie, idx, rng, sigma_esp, p_err, delta, acotado, kappa=1.0):
    """Devuelve (media_boot, anual_boot) con errores sistemáticos comunes a todos los años."""
    n_boot = idx.shape[0]
    signo = rng.choice([-1.0, 1.0], n_boot)                   # el error de CDL puede ir en ambos sentidos
    clasif = delta * signo * (rng.random(n_boot) < p_err)     # Bernoulli: rotación mal inferida
    ybar = serie.mean()
    med_clim = ybar + kappa * (serie[idx].mean(axis=1) - ybar)
    anual = serie[idx[:, 0]]                                  # un año "futuro" remuestreado
    if acotado is None:                                       # métricas diferenciales: aditivo, sin transformar
        ruido = rng.normal(0.0, sigma_esp, n_boot)            # UNA realización por réplica
        return med_clim + ruido + clasif, anual + ruido + clasif
    lo, hi = acotado
    p_m = _a_unidad(med_clim, lo, hi)
    p_a = _a_unidad(anual, lo, hi)
    pm = float(np.mean(_a_unidad(serie, lo, hi)))
    escala = (hi - lo) * pm * (1 - pm)                        # dx/dz en la media (delta method)
    ruido_z = rng.normal(0.0, sigma_esp / escala, n_boot)     # UNA realización por réplica
    cl_z = clasif / escala
    inv = lambda z: lo + (hi - lo) / (1 + np.exp(-z))
    return inv(_logit(p_m) + ruido_z + cl_z), inv(_logit(p_a) + ruido_z + cl_z)


def _nivel_certeza(amplitud, rango, n, fuente_sigma, modo):
    rel = amplitud / rango
    nivel = "ALTA" if rel < CORTES_CERTEZA[0] else ("MEDIA" if rel < CORTES_CERTEZA[1] else "BAJA")
    orden = ["BAJA", "MEDIA", "ALTA"]
    tope = 2
    if fuente_sigma == "heuristico":
        tope = min(tope, 1)       # sin calibración empírica nunca "ALTA"
    if n < 8:
        tope = min(tope, 1)
    if n < 5:
        tope = 0
    if modo == "fallback_global":
        tope = min(tope, 1)
    return orden[min(orden.index(nivel), tope)]


def calcular_incertidumbre_bootstrap(
    serie_valores: np.ndarray,
    ratio_grueso: float = 0.45,
    error_downscaling_math: float = 3.5,
    es_cultivo_menor_cdl: bool = False,
    confirmado_por_agricultor: bool = False,
    n_boot: int = 5000,
    seed: int = 42,
    *,
    sigma_downscaling_score: float | None = None,   # RMSE CV espacial en unidades del score
    acc_cdl: float | None = None,                   # exactitud CDL por año del cultivo (matriz confusión)
    es_historico_inferido: bool = True,             # False en estrategias contrafactuales
    acotado: tuple[float, float] | None = (0.0, 100.0),   # None para métricas diferenciales
    sigma_modelo: float = 0.0,                      # incertidumbre estructural (Kc, C-RUSLE, rendimientos)
    n_anios_cdl: int = 10,                          # años de historia CDL usados para inferir la rotación
    p_error_rotacion: float | None = None,          # override: P(rotación mal inferida) estimada con datos
    bloque_medio: float | None = None,
    modo: str = "benchmark",                        # "benchmark" | "fallback_global"
) -> dict:
    serie = np.asarray(serie_valores, float)
    serie = serie[np.isfinite(serie)]
    n = len(serie)
    if n < 3:
        return {"media": float(np.nanmean(serie)) if n else np.nan, "ic_95_inf": np.nan,
                "ic_95_sup": np.nan, "amplitud_ic": np.nan, "nivel_certeza": "BAJA",
                "sigma_total_aplicada": np.nan, "n": n, "fuente_sigma": "insuficiente",
                "p10_anual": np.nan, "p90_anual": np.nan, "diag": diagnostico_serie(serie)}
    rng = np.random.default_rng(seed)
    sigma_esp, fuente = _sigma_espacial(error_downscaling_math, ratio_grueso, sigma_downscaling_score)
    sigma_esp = float(np.hypot(sigma_esp, sigma_modelo))
    if acc_cdl is None:   # compatibilidad v1: mapear bandera a exactitud supuesta (PLACEHOLDER)
        acc_cdl = 0.80 if es_cultivo_menor_cdl else 0.95
    if not es_historico_inferido:
        p_err = 0.0
    elif p_error_rotacion is not None:
        p_err = float(np.clip(p_error_rotacion, 0.0, 1.0))
    else:
        p_err = _p_error_rotacion(acc_cdl, n_anios_cdl, confirmado_por_agricultor)
    idx = _idx_estacionario(rng, n, n_boot, _bloque_medio(n, bloque_medio))
    med, anual = _replicas(serie, idx, rng, sigma_esp, p_err, DELTA_ERROR_CLASIF, acotado,
                           _kappa(serie))
    lo_, hi_ = np.percentile(med, [2.5, 97.5])
    amp = float(hi_ - lo_)
    sigma_total = float(np.sqrt(sigma_esp**2 + p_err * DELTA_ERROR_CLASIF**2))   # var(δ·signo·Bern(p)) = pδ²
    return {
        "media": round(float(med.mean()), 2),
        "ic_95_inf": round(float(lo_), 2),
        "ic_95_sup": round(float(hi_), 2),
        "amplitud_ic": round(amp, 2),
        "nivel_certeza": _nivel_certeza(amp, (acotado[1] - acotado[0]) if acotado else 100.0, n, fuente, modo),
        "sigma_total_aplicada": round(sigma_total, 2),
        "p10_anual": round(float(np.percentile(anual, 10)), 2),   # riesgo: año malo (p10)
        "p90_anual": round(float(np.percentile(anual, 90)), 2),
        "n": n, "fuente_sigma": fuente, "p_error_rotacion": round(p_err, 3),
        "diag": diagnostico_serie(serie),
    }


# ----------------------------------------------------------------------------
# Comparación pareada (números aleatorios comunes) y estabilidad de ranking
# ----------------------------------------------------------------------------
def comparar_estrategias(estrategias: dict, base: str | None = None, n_boot: int = 5000,
                         seed: int = 42, acotado=(0.0, 100.0)) -> dict:
    """
    estrategias[k] = {"puntajes_anuales": array (mismo orden de años en todas),
                      "es_historico_inferido": bool, "sigma_downscaling_score": float|None,
                      "ratio_datos_gruesos", "error_downscaling_math", "acc_cdl",
                      "confirmado_por_agricultor"}
    Devuelve Δ pareado vs. base (IC 95%) y P(estrategia k es la mejor). Mismos años y mismos
    índices bootstrap en todas -> la covarianza climática se cancela en Δ.
    """
    claves = list(estrategias)
    base = base or claves[0]
    S = {k: np.asarray(v["puntajes_anuales"], float) for k, v in estrategias.items()}
    L = {len(s) for s in S.values()}
    if len(L) != 1:
        raise ValueError("Las series deben estar alineadas por año (misma longitud).")
    ok = np.all([np.isfinite(s) for s in S.values()], axis=0)   # soporte común
    S = {k: s[ok] for k, s in S.items()}
    n = int(ok.sum())
    rng = np.random.default_rng(seed)
    idx = _idx_estacionario(rng, n, n_boot, _bloque_medio(n))
    M = {}
    for k, info in estrategias.items():
        sig, _ = _sigma_espacial(info.get("error_downscaling_math", 3.5),
                                 info.get("ratio_datos_gruesos", 0.45),
                                 info.get("sigma_downscaling_score"))
        sig = float(np.hypot(sig, info.get("sigma_modelo", 0.0)))
        hist = info.get("es_historico_inferido", k == base)
        if not hist:
            p_err = 0.0
        elif info.get("p_error_rotacion") is not None:
            p_err = float(np.clip(info["p_error_rotacion"], 0.0, 1.0))
        else:
            p_err = _p_error_rotacion(info.get("acc_cdl") or 0.95, info.get("n_anios_cdl", 10),
                                      info.get("confirmado_por_agricultor", False))
        M[k], _ = _replicas(S[k], idx, rng, sig, p_err, DELTA_ERROR_CLASIF, acotado, _kappa(S[k]))
    mat = np.vstack([M[k] for k in claves])
    mejor = np.argmax(mat, axis=0)
    delta = {k: {"mediana": round(float(np.median(M[k] - M[base])), 2),
                 "ic_95": [round(float(x), 2) for x in np.percentile(M[k] - M[base], [2.5, 97.5])],
                 "p_mayor_que_base": round(float(np.mean(M[k] > M[base])), 3)}
             for k in claves if k != base}
    p_mejor = {k: round(float(np.mean(mejor == i)), 3) for i, k in enumerate(claves)}
    return {"delta_vs_base": delta, "p_mejor": p_mejor, "n_anios_comunes": n}


def topsis(matriz: np.ndarray, pesos: np.ndarray, beneficio: np.ndarray) -> np.ndarray:
    """TOPSIS vectorizado. matriz (m,c) o (R,m,c); pesos (c,) o (R,c). Devuelve C* (…,m).
    Referencia para contrastar con harmonizer.py (Matemático)."""
    X = np.asarray(matriz, float)
    norma = np.sqrt((X**2).sum(axis=-2, keepdims=True))
    V = X / np.where(norma == 0, 1, norma) * np.asarray(pesos)[..., None, :]
    b = np.asarray(beneficio, bool)
    ideal = np.where(b, V.max(axis=-2, keepdims=True), V.min(axis=-2, keepdims=True))
    anti = np.where(b, V.min(axis=-2, keepdims=True), V.max(axis=-2, keepdims=True))
    dp = np.sqrt(((V - ideal) ** 2).sum(-1))
    dm = np.sqrt(((V - anti) ** 2).sum(-1))
    return dm / np.where(dp + dm == 0, 1, dp + dm)


def estabilidad_ranking_pesos(matriz, pesos, beneficio, nombres, concentracion=50.0,
                              n_sim=5000, seed=42) -> dict:
    """P(rango 1) y rango medio bajo pesos W ~ Dirichlet(concentracion * pesos).
    concentracion alta = el agricultor está seguro de sus prioridades."""
    rng = np.random.default_rng(seed)
    p = np.asarray(pesos, float)
    p = p / p.sum()
    W = rng.dirichlet(concentracion * p, n_sim)
    C = topsis(np.broadcast_to(np.asarray(matriz, float), (n_sim,) + np.shape(matriz)), W, beneficio)
    rangos = (-C).argsort(axis=1).argsort(axis=1) + 1
    return {nombres[i]: {"p_primero": round(float(np.mean(rangos[:, i] == 1)), 3),
                         "rango_medio": round(float(rangos[:, i].mean()), 2)}
            for i in range(len(nombres))}


def atribucion_topsis(matriz, pesos, beneficio, nombres_estrategias, nombres_criterios) -> pd.DataFrame:
    """Participación (%) de cada criterio en la distancia al ideal (D+^2) por estrategia:
    qué criterio es el que más aleja a la estrategia de lo ideal."""
    X = np.asarray(matriz, float)
    V = X / np.where((np.sqrt((X**2).sum(0)) == 0), 1, np.sqrt((X**2).sum(0))) * np.asarray(pesos)
    b = np.asarray(beneficio, bool)
    ideal = np.where(b, V.max(0), V.min(0))
    d2 = (V - ideal) ** 2
    share = 100 * d2 / np.where(d2.sum(1, keepdims=True) == 0, 1, d2.sum(1, keepdims=True))
    return pd.DataFrame(share.round(1), index=nombres_estrategias, columns=nombres_criterios)


# ----------------------------------------------------------------------------
# What-If: bandera de extrapolación (el IC histórico no es válido fuera del soporte)
# ----------------------------------------------------------------------------
def bandera_extrapolacion(delta_lluvia_pct: float, delta_temp_c: float,
                          rango_lluvia_pct_obs: tuple[float, float],
                          rango_temp_c_obs: tuple[float, float]) -> list[str]:
    """rango_*_obs: anomalías interanuales observadas (p. ej. de POWER/PRISM 1981-hoy)."""
    av = []
    if not (rango_lluvia_pct_obs[0] <= delta_lluvia_pct <= rango_lluvia_pct_obs[1]):
        av.append(f"Lluvia {delta_lluvia_pct:+.0f}% fuera del rango observado "
                  f"({rango_lluvia_pct_obs[0]:+.0f}% a {rango_lluvia_pct_obs[1]:+.0f}%): "
                  "extrapolación, el IC no es válido.")
    if not (rango_temp_c_obs[0] <= delta_temp_c <= rango_temp_c_obs[1]):
        av.append(f"Temperatura {delta_temp_c:+.1f} °C fuera del rango observado "
                  f"({rango_temp_c_obs[0]:+.1f} a {rango_temp_c_obs[1]:+.1f} °C): extrapolación.")
    av.append("Escenario = sensibilidad (delta-change), no pronóstico: no modela cambios en la "
              "varianza ni en la frecuencia de extremos. El rango observado corresponde a anomalías de "
              "UN año; un desplazamiento sostenido no tiene análogo histórico.")
    return av


# ----------------------------------------------------------------------------
# Tarjetas XAI
# ----------------------------------------------------------------------------
def _fmt_ic(r, unidad=""):
    return f"{r['media']}{unidad} (IC 95%: {r['ic_95_inf']}{unidad} a {r['ic_95_sup']}{unidad})"


def _construir_tarjeta_xai(clave, info, res_p, res_a, res_e, diag, extra=None) -> dict:
    extra = extra or {}
    diag = diag or {}
    dm = set(info.get("demo_metricas", []))
    prov = diag.get("procedencia", {})
    v = lambda k: (diag.get(k) if (diag.get(k) is not None and prov.get(k) != "placeholder") else 0)
    ev, ben, ries, brechas = [], [], [], []

    # 1) Evidencia (parcela + específica de la estrategia)
    rng_sd = diag.get("meses_suelo_desnudo_rango")
    if rng_sd and prov.get("meses_suelo_desnudo") != "placeholder":
        ev.append(f"HLS (30 m): entre {rng_sd[0]} (excluyendo residuo con SWIR2) y {rng_sd[1]} meses/año "
                  "(solo NDVI/BSI) con baja cobertura vegetal; sin verdad terreno no se separa suelo expuesto de rastrojo.")
    elif v("meses_suelo_desnudo") >= 3.0:
        ev.append(f"HLS (30 m) estima {diag['meses_suelo_desnudo']} meses/año con baja cobertura "
                  "vegetal (NDVI bajo y BSI alto); el detector no separa rastrojo de suelo expuesto (ver brechas).")
    if v("deficit_hidrico_verano_mm") > 50.0:
        fte = diag.get("fuente_deficit", "balance hídrico del Estadístico 2 (fuente a confirmar)")
        ev.append(f"Déficit hídrico estival de la línea base: {diag['deficit_hidrico_verano_mm']} mm ({fte}).")
    if v("repeticion_cultivo_anios") >= 4:   # = racha máxima del MISMO cultivo (calcular de crop_YYYY)
        ev.append(f"CDL indica una racha de {int(diag['repeticion_cultivo_anios'])} años consecutivos del "
                  "mismo cultivo; se asocia a mayor presión de plagas/enfermedades y mayor demanda de N.")
    for e in info.get("evidencia_especifica", []):   # p. ej. "cubierta invernal reduce C de RUSLE"
        ev.append(e)
    if not ev:
        ev.append("Los indicadores históricos no superan los umbrales de alerta definidos.")

    # 2) Beneficios con IC
    if "Base" in clave:
        ben.append(f"Sin costos de transición. Puntaje {_fmt_ic(res_p)}.")
    elif "Eco" in clave:
        ben.append("Reducción de erosión: [PROVISIONAL, pendiente del motor RUSLE] sin cifra." if "reduccion_erosion_pct" in dm
                   else f"Reducción de erosión estimada: {_fmt_ic(res_e, '%')}.")
        ben.append("Mejora de infiltración/retención; ahorro de agua [PROVISIONAL] sin cifra." if "ahorro_agua_pct" in dm
                   else f"Mejora de infiltración/retención; ahorro de agua {_fmt_ic(res_a, '%')}.")
    else:
        ben.append("Leguminosas fijadoras de N y ruptura de ciclos de plagas.")
        ben.append("Ahorro de agua [PROVISIONAL] sin cifra." if "ahorro_agua_pct" in dm
                   else f"Ahorro de agua {_fmt_ic(res_a, '%')}.")
        if "margen" in extra:
            ben.append(f"Margen económico: {_fmt_ic(extra['margen'], ' USD/ha')}.")

    # 3) Riesgos y trade-offs (el XAI no puede ser solo positivo)
    if "Eco" in clave:
        ries += ["La cobertura consume agua antes de la siembra: riesgo en primaveras secas.",
                 "Inmovilización de N con gramíneas (p. ej. centeno): ajustar fertilización.",
                 "Fecha de terminación y calentamiento del suelo pueden afectar la emergencia."]
    elif "Base" not in clave:
        ries += ["Riesgo de mercado/logística de leguminosas; verificar compradores locales.",
                 "Ventana de siembra y cosecha distinta: validar maquinaria disponible."]
    ries += info.get("riesgos_especificos", [])

    # 4) Brechas (calculadas, no fijas) e incertidumbre
    if v("meses_suelo_desnudo") > 0:
        brechas.append("Detector NDVI/BSI sin SWIR2 (no separa rastrojo de suelo desnudo): puede sobrestimar "
                       "el periodo sin cobertura en lotes con residuo; umbrales sin calibrar con verdad terreno.")
    brechas.extend([t for t in diag.get("notas_brechas", []) if isinstance(t, str) and t.strip()])
    if info.get("es_demo", False):
        brechas.insert(0, "DEMO: la serie de puntajes de esta estrategia es provisional (no proviene del motor de scoring).")
    prov_ph = [k for k, p in prov.items() if p == "placeholder"]
    if prov_ph:
        brechas.append("Datos provisionales (no medidos, excluidos de la evidencia): " + ", ".join(prov_ph) + ".")
    brechas.append(f"Certeza: {res_p['nivel_certeza']} (ancho IC95 = {res_p['amplitud_ic']} pts; "
                   f"n = {res_p['n']} años; σ aplicada = {res_p['sigma_total_aplicada']}; "
                   f"año malo p10 = {res_p['p10_anual']}).")
    if res_p.get("fuente_sigma") == "heuristico":
        brechas.append("Los errores de escala usan un valor heurístico, aún sin calibrar con "
                       "validación cruzada espacial: la certeza no puede superar MEDIA.")
    d = res_p.get("diag", {})
    if d.get("tendencia"):
        brechas.append("Se detecta tendencia temporal: el promedio histórico no representa el estado actual.")
    if d.get("rho1") is not None and d.get("rho1") == d.get("rho1") and d["rho1"] > 0.3:
        brechas.append(f"Autocorrelación temporal (ρ₁ = {d['rho1']}): n efectivo ≈ {d['n_eff']} años.")
    if info.get("ratio_datos_gruesos", 0.45) >= 0.40:
        brechas.append("Escala: SMAP (9 km), IMERG (≈11 km) y POWER (≈55 km) fueron desagregados a 30 m; "
                       "la variación de humedad dentro del lote no se resuelve.")
    for k, txt in (("frac_hls_valida", "los píxeles-fecha HLS libres de nubes/nieve"),
                   ("frac_hls_invierno", "los píxeles-fecha HLS invernales (nov-mar) libres de nubes/nieve"),
                   ("frac_smap_valida", "días SMAP con calidad aceptable")):
        if k in diag and diag[k] < 0.6:
            brechas.append(f"Solo {100*diag[k]:.0f}% de {txt} (promedio del período analizado).")
    if not info.get("confirmado_por_agricultor", False) and info.get("es_historico_inferido", "Base" in clave):
        brechas.append("La rotación histórica fue inferida por satélite (CDL; ~"
                       f"{100*res_p.get('p_error_rotacion', 0):.0f}% de prob. de al menos un año mal "
                       "clasificado): requiere confirmación del agricultor.")
    for k, txt in diag.get("validacion_terreno", {}).items():    # contra mediciones en tierra
        brechas.append(f"Validación contra terreno: {k}: {txt}.")
    for k, txt in {**diag.get("intercomparacion", {}), **diag.get("validacion_externa", {})}.items():
        brechas.append(f"Intercomparación entre productos (no es verdad terreno): {k}: {txt}.")
    if not info.get("es_historico_inferido", "Base" in clave) and not info.get("es_referencia", False):
        brechas.append("Puntajes simulados (contrafactual): el IC refleja variabilidad climática y error "
                       "de escala" + ("" if info.get("sigma_modelo", 0) else
                       "; NO incluye incertidumbre estructural del modelo (Kc, C-RUSLE, rendimientos)") + ".")
    brechas.append("Rendimientos y precios son promedios regionales (USDA NASS); no reflejan contratos "
                   "ni costos locales.")

    humano = ["Maquinaria y ventanas de labor para establecer la rotación.",
              "Drenaje subterráneo (tile drainage), compactación y manchas del lote que el satélite no ve.",
              "Comercialización: acopios y compradores cercanos para los cultivos propuestos.",
              "Sus prioridades: confirmar o ajustar los pesos usados en la comparación."]

    limit = []
    if info.get("es_demo", False):
        limit.append("las cifras de esta estrategia son provisionales")
    if res_p.get("fuente_sigma") == "heuristico":
        limit.append("el error de escala satelital aún no está calibrado")
    if res_p["n"] < 10:
        limit.append(f"solo se usaron {res_p['n']} años de datos")
    if d_ := res_p.get("diag", {}):
        if d_.get("tendencia"):
            limit.append("hay una tendencia en los años analizados")
    resumen = [f"{info['nombre']}: puntaje {res_p['media']} (rango probable {res_p['ic_95_inf']} a {res_p['ic_95_sup']}); "
               f"confianza {res_p['nivel_certeza']}.",
               "Lo que más limita la confianza: " + ("; ".join(limit[:3]) if limit else "sin limitaciones mayores detectadas") + ".",
               "Qué debe confirmar usted: " + " ".join(humano[:2])]
    return {"estrategia": info["nombre"], "resumen_agricultor": resumen, "puntaje_medio": res_p["media"],
            "ic_95_inf": res_p["ic_95_inf"], "ic_95_sup": res_p["ic_95_sup"],
            "certeza": res_p["nivel_certeza"], "evidencia_satelital": ev,
            "beneficios_agronomicos": ben, "riesgos_y_tradeoffs": ries,
            "brechas_e_incertidumbre": brechas, "validacion_humana_requerida": humano}


def generar_reporte_completo_xai(estrategias_dict: dict, diagnostico_parcela: dict,
                                 modo: str = "benchmark", base: str | None = None,
                                 criterios: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Compatible con app.py (v1). Novedades:
      - df incluye Delta_vs_Base (IC pareado), P_Mejor y P10_Anual.
      - criterios (opcional): {"matriz","pesos","beneficio","nombres_criterios","concentracion"}
        añade P_Primero (estabilidad de ranking ante pesos) y la atribución por criterio.
      - Devuelve df, tarjetas; las claves '_comparacion' y '_atribucion' de tarjetas llevan lo global.
    """
    claves = list(estrategias_dict)
    base = base or claves[0]
    comp = comparar_estrategias(estrategias_dict, base=base)
    filas, tarjetas = [], {}
    for k, info in estrategias_dict.items():
        kw = dict(ratio_grueso=info.get("ratio_datos_gruesos", 0.45),
                  error_downscaling_math=info.get("error_downscaling_math", 3.5),
                  es_cultivo_menor_cdl=info.get("es_cultivo_menor_cdl", False),
                  confirmado_por_agricultor=info.get("confirmado_por_agricultor", False),
                  sigma_downscaling_score=info.get("sigma_downscaling_score"),
                  acc_cdl=info.get("acc_cdl"),
                  es_historico_inferido=info.get("es_historico_inferido", k == base),
                  sigma_modelo=info.get("sigma_modelo", 0.0),
                  n_anios_cdl=info.get("n_anios_cdl", 10),
                  p_error_rotacion=info.get("p_error_rotacion"),
                  modo=modo)
        res_p = calcular_incertidumbre_bootstrap(info["puntajes_anuales"], **kw)
        ref = info.get("es_referencia", k == base)
        kw_a = dict(kw, es_historico_inferido=False, acotado=None)   # diferenciales: aditivo
        if ref:   # la referencia es 0 por definición (diferencia consigo misma): sin ruido
            res_a = res_e = {**res_p, "media": 0.0, "ic_95_inf": 0.0, "ic_95_sup": 0.0}
        else:
            res_a = calcular_incertidumbre_bootstrap(info["ahorro_agua_pct"], **kw_a)
            res_e = calcular_incertidumbre_bootstrap(info["reduccion_erosion_pct"], **kw_a)
        d = comp["delta_vs_base"].get(k)
        filas.append({"clave": k, "Estrategia": info["nombre"], "Puntaje_Medio": res_p["media"],
                      "IC_Inf": res_p["ic_95_inf"], "IC_Sup": res_p["ic_95_sup"],
                      "Error_Inf": round(res_p["media"] - res_p["ic_95_inf"], 2),
                      "Error_Sup": round(res_p["ic_95_sup"] - res_p["media"], 2),
                      "P10_Anual": res_p["p10_anual"],
                      "Delta_vs_Base": None if d is None else d["mediana"],
                      "Delta_IC_Inf": None if d is None else d["ic_95"][0],
                      "Delta_IC_Sup": None if d is None else d["ic_95"][1],
                      "P_Mayor_Base": None if d is None else d["p_mayor_que_base"],
                      "P_Mejor": comp["p_mejor"][k],
                      "Ahorro_Agua_Pct": res_a["media"], "Red_Erosion_Pct": res_e["media"],
                      "Certeza": res_p["nivel_certeza"],
                      "Procedencia": "DEMO" if info.get("es_demo", False) else "motor"})
        tarjetas[k] = _construir_tarjeta_xai(k, info, res_p, res_a, res_e, diagnostico_parcela)
    df = pd.DataFrame(filas)
    if criterios:
        nombres = [f["clave"] for f in filas]
        est = estabilidad_ranking_pesos(criterios["matriz"], criterios["pesos"],
                                        criterios["beneficio"], nombres,
                                        criterios.get("concentracion", 50.0))
        df["P_Primero_TOPSIS"] = [est[n]["p_primero"] for n in nombres]
        tarjetas["_atribucion"] = atribucion_topsis(criterios["matriz"], criterios["pesos"],
                                                    criterios["beneficio"], nombres,
                                                    criterios["nombres_criterios"])
    tarjetas["_comparacion"] = comp
    return df, tarjetas


# ----------------------------------------------------------------------------
# Validación de entregables de otros módulos (ejecutar ANTES de generar_reporte_completo_xai)
# ----------------------------------------------------------------------------
_REQ_ESTR = ("nombre", "puntajes_anuales", "ahorro_agua_pct", "reduccion_erosion_pct")
_REQ_DIAG = ("meses_suelo_desnudo", "deficit_hidrico_verano_mm", "repeticion_cultivo_anios", "frac_hls_valida")


def validar_contrato(estrategias: dict, diagnostico: dict, estricto: bool = True) -> list[tuple[str, str]]:
    """Devuelve [(nivel, mensaje)] con nivel ERROR|AVISO. Con estricto=True lanza ValueError si hay ERROR."""
    out = []
    E = lambda m: out.append(("ERROR", m))
    A = lambda m: out.append(("AVISO", m))
    S = {}
    for k, info in estrategias.items():
        for c in _REQ_ESTR:
            if c not in info:
                E(f"{k}: falta '{c}'")
        if all(c in info for c in _REQ_ESTR):
            L = {c: len(info[c]) for c in _REQ_ESTR[1:]}
            if len(set(L.values())) != 1:
                E(f"{k}: series de distinta longitud {L}")
            p = np.asarray(info["puntajes_anuales"], float)
            S[k] = p
            if not np.all(np.isfinite(p)):
                A(f"{k}: NaN/inf en puntajes (se usará soporte común)")
            if np.nanstd(p) == 0:
                A(f"{k}: serie constante (sin variabilidad climática)")
            if np.nanmin(p) < 0 or np.nanmax(p) > 100:
                E(f"{k}: puntajes fuera de [0,100]")
            if np.nanmax(np.asarray(info["reduccion_erosion_pct"], float)) > 100:
                E(f"{k}: reduccion_erosion_pct > 100")
    if len({len(p) for p in S.values()}) > 1:
        E("estrategias con distinto número de años: alinear por año calendario")
    ks = list(S)
    if len(ks) < 3:
        E("se requieren >= 3 estrategias (incluida la línea base)")
    if len({len(p) for p in S.values()}) == 1:
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                m = np.isfinite(S[ks[i]]) & np.isfinite(S[ks[j]])
                if m.sum() >= 4 and np.nanstd(S[ks[i]][m]) > 0 and np.nanstd(S[ks[j]][m]) > 0:
                    r = np.corrcoef(S[ks[i]][m], S[ks[j]][m])[0, 1]
                    if r < 0:
                        A(f"corr({ks[i]},{ks[j]}) = {r:.2f} < 0: con el mismo clima se esperan "
                          "series positivamente correlacionadas; revisar origen de los puntajes")
    for k, info in estrategias.items():
        if "es_demo" not in info:
            A(f"{k}: declara es_demo (True si la serie es provisional)")
        elif info["es_demo"]:
            A(f"{k}: serie DEMO (provisional)")
    for k, info in estrategias.items():
        sg = info.get("sigma_downscaling_score")
        if sg is not None and not (np.isfinite(sg) and sg > 0):
            E(f"{k}: sigma_downscaling_score = {sg} no válido (σ=0 no es error nulo; usar None o un valor > 0 en unidades del score)")
    refs = [k for k, i in estrategias.items() if i.get("es_referencia")]
    if len(refs) > 1:
        E(f"más de una estrategia es_referencia: {refs}")
    elif not refs:
        A("ninguna estrategia marcada es_referencia (se asume la primera)")
    prov = diagnostico.get("procedencia", {})
    for c in _REQ_DIAG:
        if diagnostico.get(c) is not None and prov.get(c) == "placeholder":
            A(f"diagnostico['{c}'] tiene valor pero procedencia placeholder: se excluye de la evidencia")
        if diagnostico.get(c) is None and prov.get(c) in ("medido", "derivado"):
            E(f"diagnostico['{c}'] es None pero procedencia '{prov.get(c)}' (¿default inventado?)")
        if diagnostico.get(c) is None:
            A(f"diagnostico['{c}'] ausente: la tarjeta omitirá esa evidencia")
        elif c not in prov:
            A(f"diagnostico['{c}'] sin procedencia declarada (medido|derivado|placeholder)")
    if estricto and any(n == "ERROR" for n, _ in out):
        raise ValueError("; ".join(m for n, m in out if n == "ERROR"))
    return out


# ----------------------------------------------------------------------------
# Tensor de criterios para TOPSIS del Matemático: (S, estrategias, criterios)
# ----------------------------------------------------------------------------
def tensor_bootstrap_criterios(X_anual, sigmas, S: int = 5000, seed: int = 42, rho_error: float = 1.0,
                               no_negativos=None, usar_kappa: bool = True) -> np.ndarray:
    """X_anual: (n_anios, E, C) criterios por año climático, alineados por año (sin NaN).
    sigmas: (E, C) error sistemático de ENTRADA en unidades del criterio.
    rho_error: correlación del error sistemático entre estrategias (1 = comunes: SMAP/lluvia/ET0 son
               los mismos para todas; SUPUESTO, pedir a Est. 2/3 las covarianzas o draws pareados).
    Devuelve (S, E, C): media climática (bootstrap estacionario, ensanchada por kappa) + error común."""
    X = np.asarray(X_anual, float)
    if X.ndim != 3:
        raise ValueError("X_anual debe tener forma (n_anios, E, C)")
    n, E, C = X.shape
    if not np.isfinite(X).all():
        raise ValueError("NaN/inf en X_anual: alinear por año calendario antes de remuestrear")
    sg = np.asarray(sigmas, float)
    if sg.shape != (E, C):
        raise ValueError(f"sigmas debe ser ({E},{C}), recibido {sg.shape}")
    rng = np.random.default_rng(seed)
    idx = _idx_estacionario(rng, n, S, _bloque_medio(n))
    ybar = X.mean(axis=0)
    med = X[idx].mean(axis=1)
    if usar_kappa:
        kap = np.array([[_kappa(X[:, e, c]) for c in range(C)] for e in range(E)])
        med = ybar + kap * (med - ybar)
    z = (np.sqrt(rho_error) * rng.standard_normal((S, 1, C))
         + np.sqrt(1.0 - rho_error) * rng.standard_normal((S, E, C)))
    draws = med + sg[None] * z
    if no_negativos is not None:
        for c, flag in enumerate(no_negativos):
            if flag:
                draws[:, :, c] = np.maximum(draws[:, :, c], 0.0)
    return draws


def p_rango_tensor(draws, pesos, beneficio, nombres, concentracion: float = 50.0, seed: int = 42) -> dict:
    """P(rango 1) y rango medio con incertidumbre de datos (draws) Y de pesos (Dirichlet).
    Sirve de contraste independiente de la implementación TOPSIS del Matemático."""
    rng = np.random.default_rng(seed)
    D = np.asarray(draws, float)
    p = np.asarray(pesos, float); p = p / p.sum()
    W = rng.dirichlet(concentracion * p, D.shape[0])
    C = topsis(D, W, beneficio)
    rangos = (-C).argsort(axis=1).argsort(axis=1) + 1
    return {nombres[i]: {"p_primero": round(float(np.mean(rangos[:, i] == 1)), 3),
                         "rango_medio": round(float(rangos[:, i].mean()), 2)} for i in range(D.shape[1])}
