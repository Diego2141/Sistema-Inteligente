# -*- coding: utf-8 -*-
"""
step005_validar_sadf.py
=======================
Valida dos detectores de régimen antes de integrarlos en la matriz de features:

  1. SADF sobre garch_vol (volatilidad del flujo neto)
     — detecta explosividad en la volatilidad (stress auto-reforzado)

  2. HMM de 3 estados sobre [flujo_neto, garch_vol]
     — clasifica el régimen oculto: calma / stress moderado / stress severo

Gráfico de 3 paneles:
  Panel 1: Flujo neto diario (azul/rojo)
  Panel 2: SADF sobre volatilidad con umbrales P95/P99
  Panel 3: Régimen HMM coloreado (verde=calma, amarillo=moderado, rojo=severo)

No modifica ningún archivo — solo genera el gráfico de validación.

Uso:
    python step005_validar_sadf.py
    runfile('...step005_validar_sadf.py', wdir='...')
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

try:
    from hmmlearn import hmm as hmmlearn_hmm
    _HMM_OK = True
except ImportError:
    _HMM_OK = False
    print("AVISO: hmmlearn no instalado — panel HMM omitido.")

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
RUTA_BCRP    = BASE_SISTEMA / "1. Data" / "Raw" / "series_bcrp.xlsx"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output"
HOJA_BCRP_EMBI = "EMBIG"   # hoja con PD04709XD (Spread EMBIG Perú)

BANCO        = "SISTEMA"
H_REF        = 2          # h mínimo — usamos target y garch_vol de este h

# SADF params (aplicado sobre garch_vol)
TAU_MIN      = 30
VENTANAS     = [60, 120, 252]
LAGS         = 1

# HMM params
N_ESTADOS           = 3
HMM_INICIO          = "2010-01-01"
HMM_PRIMER_VENTANA  = 6    # años del primer bloque de entrenamiento (= primer fold train)
                            # → modelo 1: 2010-2015 (in-sample) + etiqueta 2016
                            # → modelo 2: 2010-2016 → etiqueta 2017  (expanding)
HMM_ROLLING_AÑOS    = 6    # tamaño ventana rolling fija (en años, para comparar)

# Episodios de stress conocidos
EPISODIOS = [
    ("2020-03-01", "2020-12-31", "COVID-19"),
    ("2021-03-01", "2021-08-31", "Elecciones 2021"),
]

# ── SADF (numpy vectorizado) ───────────────────────────────────────────────────

def calcular_sadf(values: np.ndarray, tau_min: int = 30,
                  ventana_max: int = 120, lags: int = 1) -> np.ndarray:
    """Supremum ADF sobre todas las subventanas [t0, t] con t0 en [t-ventana_max, t-tau_min]."""
    n    = len(values)
    sadf = np.full(n, np.nan)
    dy   = np.diff(values)

    for t in range(ventana_max, n):
        t_stats = []
        for t0 in range(max(lags + 1, t - ventana_max), t - tau_min + 1):
            y_win  = values[t0 : t + 1]
            dy_win = dy[t0 : t]
            T      = len(dy_win)
            if T < lags + 3:
                continue
            Y      = dy_win[lags:]
            X_cols = [np.ones(T - lags), y_win[lags:-1]]
            for l in range(1, lags + 1):
                X_cols.append(dy_win[lags - l : T - l])
            X_mat  = np.column_stack(X_cols)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
                resid  = Y - X_mat @ coeffs
                s2     = np.sum(resid ** 2) / max(len(Y) - X_mat.shape[1], 1)
                XtXi   = np.linalg.inv(X_mat.T @ X_mat)
                se     = np.sqrt(max(s2 * XtXi[1, 1], 1e-12))
                t_stats.append(coeffs[1] / se)
            except Exception:
                continue
        if t_stats:
            sadf[t] = max(t_stats)

    return sadf


# ── Carga de datos ─────────────────────────────────────────────────────────────

def cargar_datos(ruta: Path, banco: str, h_ref: int) -> pd.DataFrame:
    """
    Carga flujo neto (target) y garch_vol para un banco y h_ref dados.
    Indexa por fecha_flujo = fecha_t + h_ref días hábiles.
    """
    print(f"Cargando matriz: {ruta}")
    cols = ["fecha_t", "banco", "h", "target", "garch_vol"]
    df   = pd.read_parquet(
        ruta,
        columns=cols,
        filters=[("banco", "==", banco), ("h", "==", h_ref)]
    )
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values("fecha_t").reset_index(drop=True)

    offset = pd.tseries.offsets.BusinessDay(h_ref)
    df["fecha"] = df["fecha_t"] + offset
    df = df.set_index("fecha").sort_index()

    print(f"  {banco} | {len(df):,} obs | "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


def cargar_embig(ruta: Path = RUTA_BCRP, hoja: str = HOJA_BCRP_EMBI) -> pd.Series:
    """
    Lee el EMBIG Perú directo del Excel crudo del Add-In BCRPData
    (no está en matriz_features.parquet — fue excluido en FEATURES_EXCLUIR).
    Mismo parser que step001._leer_bcrp_excel().
    """
    if not ruta.exists():
        print(f"  AVISO: no se encontró {ruta} — panel EMBIG omitido.")
        return pd.Series(dtype=float, name="EMBI_PERU")

    try:
        raw = pd.read_excel(ruta, sheet_name=hoja, header=None)
        header_row = None
        for i, row in raw.iterrows():
            vals = [str(v).strip().lower() for v in row if pd.notna(v)]
            if any(v in ("date", "fecha") for v in vals):
                header_row = i
                break
        if header_row is None:
            header_row = 5

        df = pd.read_excel(ruta, sheet_name=hoja, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        col_fecha = next((c for c in df.columns if c.lower() in ("date", "fecha")), df.columns[0])
        col_valor = next((c for c in df.columns if c.lower() in ("valores", "value", "values")), df.columns[1])
        df = df[[col_fecha, col_valor]].copy()
        df.columns = ["fecha", "EMBI_PERU"]

        _meses_es = {
            "Ene": "Jan", "Feb": "Feb", "Mar": "Mar", "Abr": "Apr",
            "May": "May", "Jun": "Jun", "Jul": "Jul", "Ago": "Aug",
            "Set": "Sep", "Oct": "Oct", "Nov": "Nov", "Dic": "Dec",
        }
        def _parse_fecha_bcrp(s):
            s = str(s).strip()
            for es, en in _meses_es.items():
                s = s.replace(es, en)
            return pd.to_datetime(s, errors="coerce", dayfirst=True)

        df["fecha"] = df["fecha"].apply(_parse_fecha_bcrp)
        df["EMBI_PERU"] = pd.to_numeric(
            df["EMBI_PERU"].astype(str).str.replace(",", "."), errors="coerce"
        )
        df = df.dropna(subset=["fecha"]).set_index("fecha").sort_index()
        print(f"  EMBIG leído: {len(df):,} obs, {df.index.min().date()} → {df.index.max().date()}")
        return df["EMBI_PERU"]
    except Exception as e:
        print(f"  AVISO: error leyendo EMBIG ({e}) — panel omitido.")
        return pd.Series(dtype=float, name="EMBI_PERU")


# ── HMM ───────────────────────────────────────────────────────────────────────
#
# Modelo: Gaussian HMM de K=3 estados ocultos S_t ∈ {0,1,2}
#
# Componentes:
#   π_i   = P(S_1 = i)                         distribución inicial
#   a_ij  = P(S_t = j | S_{t-1} = i)           matriz de transición A (3×3)
#   b_i(x)= N(x | μ_i, Σ_i)                   emisión Gaussiana por estado
#
# Observación: X_t = [flujo_neto_norm, garch_vol_norm]  (vector 2D)
#
# Entrenamiento (Baum-Welch / EM):
#   Maximiza  P(X_1,...,X_T | π, A, {μ_i,Σ_i})  iterando:
#     E-step: calcular γ_t(i) = P(S_t=i | X_{1:T})  con forward-backward
#     M-step: actualizar π, A, μ_i, Σ_i usando γ_t(i)
#
# Predicción de estados (Viterbi):
#   S*_{1:T} = argmax P(S_{1:T} | X_{1:T}, π, A, {μ_i,Σ_i})
#   Algoritmo de programación dinámica O(K²·T)
#
# Ordenamiento post-entrenamiento:
#   Estados reordenados por det(Σ_i) ascendente → 0=calma, 2=severo


def _fit_hmm_single(X_train: np.ndarray) -> tuple:
    """Ajusta un HMM y devuelve (modelo, scaler, sorted_states)."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    modelo = hmmlearn_hmm.GaussianHMM(
        n_components=N_ESTADOS, covariance_type="full",
        n_iter=1000, random_state=42,
    )
    modelo.fit(Xs)
    varianzas     = [np.linalg.det(modelo.covars_[s]) for s in range(N_ESTADOS)]
    sorted_states = np.argsort(varianzas)   # 0=calma, 2=severo
    return modelo, scaler, sorted_states


def _predecir_estados(modelo, scaler, sorted_states, X: np.ndarray) -> np.ndarray:
    """Predice estados reordenados (0=calma, 2=severo) para X."""
    Xs          = scaler.transform(X)
    estados_raw = modelo.predict(Xs)
    mapa        = {sorted_states[i]: i for i in range(N_ESTADOS)}
    return np.array([mapa[e] for e in estados_raw])


# ── GARCH(1,1) por ventana (sin leakage de parámetros) ───────────────────────
#
# Para que el HMM expanding/rolling sea limpio de principio a fin, el GARCH
# también se re-estima en cada ventana: los parámetros ω, α, β solo ven datos
# hasta Y-1 antes de etiquetar el año Y.
#
# Tres funciones de bajo nivel:
#   _garch_fit          → estima (ω, α, β, escala, var_unc) sobre un array
#   _garch_vol_insample → σ_t causal con los parámetros estimados (in-sample)
#   _garch_vol_oos      → σ_t causal para datos OOS, continuando la recursión
#                         desde el último punto del entrenamiento

def _garch_fit(flujo_arr: np.ndarray):
    """
    Estima GARCH(1,1) en flujo_arr por MLE (L-BFGS-B).
    Retorna (omega, alpha, beta, escala, var_unc) o None si falla / serie corta.
    """
    from scipy.optimize import minimize
    escala = float(np.std(flujo_arr))
    if escala < 1e-9 or len(flujo_arr) < 60:
        return None
    x       = (flujo_arr / escala).astype(float)
    n       = len(x)
    var_unc = float(np.var(x))

    def _s2(params):
        w, a, b = params
        s2 = np.empty(n)
        s2[0] = var_unc
        for t in range(1, n):
            s2[t] = w + a * x[t-1]**2 + b * s2[t-1]
        return s2

    def _nll(params):
        w, a, b = params
        if w <= 0 or a <= 0 or b <= 0 or a + b >= 0.9999:
            return 1e10
        s2 = _s2(params)
        if np.any(s2 <= 0):
            return 1e10
        return 0.5 * float(np.sum(np.log(s2) + x**2 / s2))

    try:
        res = minimize(_nll, [0.01, 0.08, 0.88], method="L-BFGS-B",
                       bounds=[(1e-7, 0.5), (1e-7, 0.5), (1e-7, 0.9999)],
                       options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7})
        if res.fun > 1e9:
            return None
        return (*res.x, escala, var_unc)
    except Exception:
        return None


def _garch_vol_insample(params: tuple, flujo_arr: np.ndarray) -> np.ndarray:
    """σ_t in-sample: recursión causal con parámetros estimados en flujo_arr."""
    w, a, b, escala, var_unc = params
    x  = (flujo_arr / escala).astype(float)
    n  = len(x)
    s2 = np.empty(n)
    s2[0] = var_unc
    for t in range(1, n):
        s2[t] = w + a * x[t-1]**2 + b * s2[t-1]
    return np.sqrt(np.maximum(s2, 0)) * escala


def _garch_vol_oos(params: tuple,
                   flujo_train: np.ndarray,
                   flujo_oos:   np.ndarray) -> np.ndarray:
    """
    σ_t OOS para flujo_oos.
    Continúa la recursión GARCH desde el último estado del entrenamiento
    usando los parámetros estimados en train — sin ver datos futuros.
    """
    w, a, b, escala, _ = params
    x_tr    = (flujo_train / escala).astype(float)
    n_tr    = len(x_tr)
    var_tr  = float(np.var(x_tr))

    # Propagar recursión hasta el final de train para obtener σ²_{n_train}
    s2 = var_tr
    for t in range(1, n_tr):
        s2 = w + a * x_tr[t-1]**2 + b * s2
    # Primer punto OOS: σ²_{n_train+1}
    s2_init = w + a * x_tr[-1]**2 + b * s2

    x_oos   = (flujo_oos / escala).astype(float)
    n_oos   = len(x_oos)
    s2_arr  = np.empty(n_oos)
    s2_arr[0] = s2_init
    for t in range(1, n_oos):
        s2_arr[t] = w + a * x_oos[t-1]**2 + b * s2_arr[t-1]
    return np.sqrt(np.maximum(s2_arr, 0)) * escala


def hmm_muestra_completa(df: pd.DataFrame, inicio: str) -> tuple:
    """
    VERSION CON LEAKAGE — entrena en 2016→hoy y etiqueta toda la serie.
    Solo para comparación visual.
    """
    df_base = df[df.index >= inicio][["target", "garch_vol"]].dropna()
    modelo, scaler, sorted_states = _fit_hmm_single(df_base.values)

    df_full = df[["target", "garch_vol"]].dropna()
    estados = _predecir_estados(modelo, scaler, sorted_states, df_full.values)

    trans = modelo.transmat_
    print(f"\n  [MUESTRA COMPLETA] Matriz de transición:")
    for i in range(N_ESTADOS):
        fila   = "  ".join([f"{trans[sorted_states[i], sorted_states[j]]:.2f}"
                             for j in range(N_ESTADOS)])
        nombre = ["Calma    ", "Moderado ", "Severo   "][i]
        print(f"    {nombre}: {fila}")

    return df_full.index, estados


def hmm_rolling(df: pd.DataFrame, inicio: str,
                ventana_años: int = 6) -> tuple:
    """
    VERSION SIN LEAKAGE — rolling window de longitud fija.
    GARCH(1,1) re-estimado en cada ventana: ω, α, β solo ven datos de esa ventana.

    Para etiquetar el año Y:
      - Entrena GARCH + HMM en [Y - ventana_años, Y-1]  (ventana fija)
      - σ_t OOS de año Y continúa la recursión desde el último punto de train
      - Requiere al menos ventana_años de historia desde `inicio`

    Con ventana_años=6 e inicio=2010:
      - Primer entrenamiento: 2010-2015 (6 años) → etiqueta 2016
      - 2do entrenamiento:    2011-2016 (6 años) → etiqueta 2017
      - ...y así sucesivamente
    """
    flujo_base = df[df.index >= inicio]["target"].dropna()
    año_inicio = flujo_base.index.year.min()
    años       = sorted(flujo_base.index.year.unique())
    todos_idx     = []
    todos_estados = []

    print(f"\n  [ROLLING {ventana_años}a] HMM+GARCH/ventana | inicio={inicio}")
    print(f"  Primeras etiquetas desde {año_inicio + ventana_años} "
          f"(ventana fija {ventana_años} años, GARCH re-estimado por ventana)")

    for año in años:
        años_historia = año - año_inicio
        if años_historia < ventana_años:
            print(f"    {año}: historia insuficiente ({años_historia} años < {ventana_años}) — omitiendo")
            continue

        año_ini_ventana = año - ventana_años
        flujo_train = flujo_base[
            (flujo_base.index.year >= año_ini_ventana) &
            (flujo_base.index.year < año)
        ]
        flujo_año = flujo_base[flujo_base.index.year == año]
        if flujo_año.empty or len(flujo_train) < 50:
            continue

        params = _garch_fit(flujo_train.values)
        if params is None:
            print(f"    {año}: GARCH falló — omitiendo")
            continue

        sigma_train = _garch_vol_insample(params, flujo_train.values)
        sigma_oos   = _garch_vol_oos(params, flujo_train.values, flujo_año.values)
        X_train = np.column_stack([flujo_train.values, sigma_train])
        X_año   = np.column_stack([flujo_año.values,  sigma_oos])

        try:
            modelo, scaler, sorted_states = _fit_hmm_single(X_train)
        except Exception as e:
            print(f"    {año}: HMM falló — {e}")
            continue

        estados_año = _predecir_estados(modelo, scaler, sorted_states, X_año)
        todos_idx.extend(flujo_año.index.tolist())
        todos_estados.extend(estados_año.tolist())
        pct_severo = (np.array(estados_año) == 2).mean() * 100
        print(f"    {año}: ventana [{año_ini_ventana}-{año-1}] ({len(flujo_train):,} obs) → "
              f"severo={pct_severo:.0f}%")

    return pd.DatetimeIndex(todos_idx), np.array(todos_estados)


def hmm_expanding(df: pd.DataFrame, inicio: str,
                  primer_ventana: int = 6) -> tuple:
    """
    VERSION SIN LEAKAGE — GARCH + HMM expanding.
    GARCH(1,1) re-estimado en cada ventana: ω, α, β solo ven datos hasta Y-1.

      Paso 1 — Primer bloque (in-sample):
        GARCH estimado en [inicio, inicio+primer_ventana-1] (in-sample)
        HMM entrenado con [flujo, σ_t_insample]
        Etiqueta ESE mismo periodo con Viterbi (sin hueco al inicio)

      Paso 2 — Expanding año a año:
        Para etiquetar año Y (Y > inicio + primer_ventana - 1):
          GARCH estimado en [inicio, Y-1] → σ_t OOS para año Y (continúa recursión)
          HMM entrenado con [flujo_train, σ_train] → etiqueta [flujo_Y, σ_Y_oos]

    Con primer_ventana=6 e inicio=2010:
      - Modelo 1: GARCH+HMM en 2010-2015 → etiqueta 2010-2015 (in-sample) + 2016
      - Modelo 2: GARCH+HMM en 2010-2016 → etiqueta 2017
      - ...
      → SIN huecos, SIN leakage de parámetros: toda la historia desde 2010
    """
    flujo_base   = df[df.index >= inicio]["target"].dropna()
    año_inicio   = flujo_base.index.year.min()
    año_fin_prim = año_inicio + primer_ventana - 1
    años         = sorted(flujo_base.index.year.unique())
    todos_idx     = []
    todos_estados = []

    print(f"\n  [EXPANDING] HMM+GARCH/ventana | inicio={inicio} | primer_ventana={primer_ventana}a")
    print(f"  Modelo 1: GARCH+HMM {año_inicio}-{año_fin_prim} → etiqueta {año_inicio}-{año_fin_prim} "
          f"(in-sample) + {año_fin_prim + 1}")
    print(f"  Modelo 2+: expanding año a año desde {año_fin_prim + 1}, GARCH re-estimado cada año")

    # ── Paso 1: primer bloque in-sample ───────────────────────────────────────
    flujo_prim = flujo_base[flujo_base.index.year <= año_fin_prim]
    if not flujo_prim.empty:
        params = _garch_fit(flujo_prim.values)
        if params is None:
            print(f"    Paso 1: GARCH falló — bloque inicial omitido")
        else:
            try:
                sigma_prim = _garch_vol_insample(params, flujo_prim.values)
                X_prim = np.column_stack([flujo_prim.values, sigma_prim])
                modelo, scaler, sorted_states = _fit_hmm_single(X_prim)
                estados_prim = _predecir_estados(modelo, scaler, sorted_states, X_prim)
                todos_idx.extend(flujo_prim.index.tolist())
                todos_estados.extend(estados_prim.tolist())
                pct_sev = (estados_prim == 2).mean() * 100
                print(f"    {año_inicio}-{año_fin_prim}: in-sample ({len(flujo_prim):,} obs) → "
                      f"severo={pct_sev:.0f}%")
            except Exception as e:
                print(f"    Paso 1 falló — {e}")

    # ── Paso 2: expanding desde año_fin_prim + 1 ──────────────────────────────
    for año in años:
        if año <= año_fin_prim:
            continue
        flujo_train = flujo_base[flujo_base.index.year < año]
        flujo_año   = flujo_base[flujo_base.index.year == año]
        if flujo_año.empty or len(flujo_train) < 50:
            continue

        params = _garch_fit(flujo_train.values)
        if params is None:
            print(f"    {año}: GARCH falló — omitiendo")
            continue

        sigma_train = _garch_vol_insample(params, flujo_train.values)
        sigma_oos   = _garch_vol_oos(params, flujo_train.values, flujo_año.values)
        X_train = np.column_stack([flujo_train.values, sigma_train])
        X_año   = np.column_stack([flujo_año.values,  sigma_oos])

        try:
            modelo, scaler, sorted_states = _fit_hmm_single(X_train)
            estados_año = _predecir_estados(modelo, scaler, sorted_states, X_año)
            todos_idx.extend(flujo_año.index.tolist())
            todos_estados.extend(estados_año.tolist())
            pct_sev = (estados_año == 2).mean() * 100
            años_train = año - año_inicio
            print(f"    {año}: entrenado en {len(flujo_train):,} obs ({años_train}a) → "
                  f"severo={pct_sev:.0f}%")
        except Exception as e:
            print(f"    {año}: HMM falló — {e}")

    return pd.DatetimeIndex(todos_idx), np.array(todos_estados)


# ── Helper ────────────────────────────────────────────────────────────────────

def df_base_year(inicio: str) -> int:
    return pd.Timestamp(inicio).year


# ── Gráfico 5 paneles ─────────────────────────────────────────────────────────

def _pintar_regimen(ax, idx, estados, vol_ref_idx, vol_ref_vals):
    """Pinta fondo por régimen HMM y superpone garch_vol normalizada."""
    colores_hmm = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}
    for estado, color in colores_hmm.items():
        fechas_e = idx[estados == estado]
        for f in fechas_e:
            ax.axvspan(f, f + pd.offsets.BusinessDay(1),
                       alpha=0.55, color=color, linewidth=0)
    vol_n = (vol_ref_vals - np.nanmean(vol_ref_vals)) / (np.nanstd(vol_ref_vals) + 1e-12)
    ax.plot(vol_ref_idx, vol_n, color="k", lw=0.6, alpha=0.55)
    parches = [
        mpatches.Patch(color="#4CAF50", label="Calma"),
        mpatches.Patch(color="#FFC107", label="Moderado"),
        mpatches.Patch(color="#F44336", label="Severo"),
    ]
    ax.legend(handles=parches, fontsize=8, loc="upper left")
    ax.set_ylabel("Régimen HMM", fontsize=9)


def graficar(df: pd.DataFrame, sadf_dict: dict,
             hmm_completo=None, hmm_expanding_=None,
             hmm_rolling_=None) -> None:

    n_panels = 2 + (1 if hmm_completo   is not None else 0) \
                 + (1 if hmm_expanding_  is not None else 0) \
                 + (1 if hmm_rolling_    is not None else 0)
    h_ratios = [2.0] + [1.0] * (n_panels - 1)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(19, 4 * n_panels),
        sharex=True, gridspec_kw={"height_ratios": h_ratios}
    )
    axes = list(axes)

    ventanas_str = ", ".join(f"{v}d" for v in VENTANAS)
    año_ini      = df_base_year(HMM_INICIO)
    año_fin_prim = año_ini + HMM_PRIMER_VENTANA - 1
    fig.suptitle(
        f"Validación SADF y HMM — {BANCO}\n"
        f"SADF: tau_min={TAU_MIN} dh | ventanas {ventanas_str}  |  "
        f"HMM: {N_ESTADOS} estados | expanding (como GARCH) vs rolling ({HMM_ROLLING_AÑOS}a)",
        fontsize=12, fontweight="bold"
    )

    ax_idx = 0
    fechas = df.index
    flujo  = df["target"].values

    # ── Panel 1: flujo neto ────────────────────────────────────────────────────
    ax1 = axes[ax_idx]; ax_idx += 1
    colores = np.where(flujo >= 0, "#2196F3", "#F44336")
    ax1.bar(fechas, flujo, color=colores, alpha=0.65, width=1.2)
    ax1.axhline(0, color="k", lw=0.7, ls="--")
    ax1.set_ylabel("Flujo neto", fontsize=9)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B"))
    ax1.set_title("Flujo neto diario (azul=entradas, rojo=salidas netas)", fontsize=9)

    # ── Panel 2: SADF ─────────────────────────────────────────────────────────
    ax2 = axes[ax_idx]; ax_idx += 1
    colores_s = ["#E91E63", "#FF6F00", "#7B1FA2"]
    for (nombre, svals), color in zip(sadf_dict.items(), colores_s):
        ax2.plot(fechas, svals, lw=1.0, color=color, label=nombre, alpha=0.85)
        p95 = np.nanpercentile(svals, 95)
        p99 = np.nanpercentile(svals, 99)
        ax2.axhline(p95, color=color, lw=0.9, ls=":", label=f"P95={p95:.2f}")
        ax2.axhline(p99, color=color, lw=0.9, ls="--", alpha=0.6,
                    label=f"P99={p99:.2f}")
    ax2.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax2.set_ylabel("SADF (t-stat sup.)", fontsize=9)
    ax2.set_title("SADF sobre garch_vol — spikes positivos = volatilidad explosiva",
                  fontsize=9)
    ax2.legend(fontsize=8, loc="upper left", ncol=2)

    # ── Panel 3: HMM muestra completa (con leakage) ───────────────────────────
    if hmm_completo is not None:
        ax3 = axes[ax_idx]; ax_idx += 1
        idx_c, est_c = hmm_completo
        vol_c = df.loc[df.index.isin(idx_c), "garch_vol"]
        _pintar_regimen(ax3, idx_c, est_c, vol_c.index, vol_c.values)
        ax3.set_title(
            "CON LEAKAGE — HMM entrenado en TODA la muestra: "
            "el estado de 2013 'sabe' que en 2020 vino COVID",
            fontsize=9, color="#B71C1C"
        )

    # ── Panel 4: HMM expanding — OPCIÓN A (sin leakage) ──────────────────────
    if hmm_expanding_ is not None:
        ax4 = axes[ax_idx]; ax_idx += 1
        idx_e, est_e = hmm_expanding_
        vol_e = df.loc[df.index.isin(idx_e), "garch_vol"]
        _pintar_regimen(ax4, idx_e, est_e, vol_e.index, vol_e.values)
        ax4.set_title(
            f"OPCIÓN A — como GARCH (SIN leakage):  "
            f"{año_ini}–{año_fin_prim} in-sample  |  "
            f"{año_fin_prim+1}+ expanding: hmm_estado[Y] = Viterbi( HMM[{año_ini}, Y-1] )",
            fontsize=9, color="#1B5E20"
        )

    # ── Panel 5: HMM rolling (sin leakage) ────────────────────────────────────
    if hmm_rolling_ is not None:
        ax5 = axes[ax_idx]; ax_idx += 1
        idx_r, est_r = hmm_rolling_
        vol_r = df.loc[df.index.isin(idx_r), "garch_vol"]
        _pintar_regimen(ax5, idx_r, est_r, vol_r.index, vol_r.values)
        ax5.set_title(
            f"ROLLING {HMM_ROLLING_AÑOS}a (SIN leakage) — ventana fija: "
            f"hmm_estado[Y] = Viterbi( HMM entrenado en [Y-{HMM_ROLLING_AÑOS}, Y-1] )",
            fontsize=9, color="#0D47A1"
        )

    # ── Bandas de episodios ────────────────────────────────────────────────────
    for ini, fin, etiqueta in EPISODIOS:
        for ax in axes:
            ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin),
                       alpha=0.10, color="navy", zorder=0)
        mid  = pd.Timestamp(ini) + (pd.Timestamp(fin) - pd.Timestamp(ini)) / 2
        ymax = ax1.get_ylim()[1]
        ax1.text(mid, ymax * 0.88, etiqueta, ha="center", fontsize=8,
                 color="#1A237E",
                 bbox=dict(boxstyle="round,pad=0.2", fc="lavender", alpha=0.8))

    plt.tight_layout()
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    ruta = DIR_OUTPUT / f"validacion_sadf_hmm_{BANCO}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"\n  Gráfico guardado: {ruta}")
    plt.show()


# ── Estadísticas de validación ────────────────────────────────────────────────

def reportar_estadisticas(df, sadf_dict, hmm_completo=None,
                          hmm_expanding_=None, hmm_rolling_=None):
    print("\n── SADF: % días sobre P95 en episodios de stress ───────────────────")
    for nombre, svals in sadf_dict.items():
        p95 = np.nanpercentile(svals, 95)
        s   = pd.Series(svals, index=df.index)
        for ini, fin, etiqueta in EPISODIOS:
            sub = s[(s.index >= ini) & (s.index <= fin)]
            if len(sub) > 0:
                pct = (sub > p95).mean() * 100
                print(f"  {nombre:<18} | {etiqueta:<15}: {pct:5.1f}% sobre P95")

    variantes = [
        ("MUESTRA COMPLETA (leakage)    ", hmm_completo),
        (f"EXPANDING  mín {HMM_PRIMER_VENTANA}a (sin leak)", hmm_expanding_),
        (f"ROLLING    {HMM_ROLLING_AÑOS}a  (sin leak) ", hmm_rolling_),
    ]
    for tag, hmm_res in variantes:
        if hmm_res is None:
            continue
        idx_h, estados = hmm_res
        total = len(estados)
        print(f"\n── HMM {tag}: distribución de estados ──────")
        for e, nom in enumerate(["Calma   ", "Moderado", "Severo  "]):
            n = (estados == e).sum()
            print(f"  Estado {e} {nom}: {n:,} días ({n/total*100:.1f}%)")
        print(f"  % días en stress severo (E2) por episodio:")
        for ini, fin, etiqueta in EPISODIOS:
            mask = (idx_h >= ini) & (idx_h <= fin)
            sub  = estados[mask]
            if len(sub) > 0:
                pct = (sub == 2).mean() * 100
                print(f"    {etiqueta:<20}: {pct:5.1f}% en estado severo")



# ── Pre-cómputo (simula lo que haría step001) ────────────────────────────────

def precomputar_hmm_features(df: pd.DataFrame) -> pd.Series:
    """
    Simula lo que haría step001_build_feature_matrix.py:
    calcula hmm_estado[t] una sola vez con expanding window y lo devuelve
    como Serie indexada por fecha — lista para añadir al parquet.

    Paso 1 — Bloque inicial in-sample (como GARCH):
        Entrena en [HMM_INICIO, HMM_INICIO+HMM_PRIMER_VENTANA-1]
        → etiqueta ESE mismo periodo con Viterbi (sin hueco al inicio)

    Paso 2 — Expanding año a año:
        Para año Y > HMM_INICIO+HMM_PRIMER_VENTANA-1:
        hmm_estado[Y] = Viterbi( HMM entrenado en [HMM_INICIO, Y-1] )
    """
    idx, estados = hmm_expanding(df, HMM_INICIO, primer_ventana=HMM_PRIMER_VENTANA)
    return pd.Series(estados, index=idx, name="hmm_estado", dtype="Int8")


def hmm_evolucion(df: pd.DataFrame, primer_ventana: int = HMM_PRIMER_VENTANA) -> dict:
    """
    Para cada año Y desde (HMM_INICIO + primer_ventana) hasta el último año:
      - Re-estima GARCH(1,1) en [HMM_INICIO, Y]  (parámetros sin leakage del futuro)
      - Entrena HMM con [flujo, σ_t_insample] en [HMM_INICIO, Y]
      - Re-etiqueta TODOS los días de [HMM_INICIO, Y] con ese modelo

    GARCH re-estimado en cada bloque: ω, α, β solo ven datos hasta Y.
    Devuelve dict {año_fin: (DatetimeIndex, np.array estados)}
    → permite ver cómo evoluciona/estabiliza la clasificación al añadir cada año.
    """
    flujo_base   = df[df.index >= HMM_INICIO]["target"].dropna()
    año_inicio   = flujo_base.index.year.min()
    año_fin_prim = año_inicio + primer_ventana - 1
    años_oos     = sorted(y for y in flujo_base.index.year.unique() if y > año_fin_prim)

    resultados = {}
    print(f"\n  [EVOLUCIÓN] Re-etiquetando cada año con modelo acumulado (GARCH por bloque)...")

    # Año base: GARCH + HMM entrenado en primer bloque
    flujo_prim = flujo_base[flujo_base.index.year <= año_fin_prim]
    params = _garch_fit(flujo_prim.values)
    if params is None:
        print(f"    Modelo base: GARCH falló")
    else:
        try:
            sigma_prim = _garch_vol_insample(params, flujo_prim.values)
            X_prim = np.column_stack([flujo_prim.values, sigma_prim])
            modelo, scaler, sorted_states = _fit_hmm_single(X_prim)
            estados = _predecir_estados(modelo, scaler, sorted_states, X_prim)
            resultados[año_fin_prim] = (flujo_prim.index, estados)
            pct_sev = (estados == 2).mean() * 100
            print(f"    hasta {año_fin_prim}: {len(flujo_prim):,} obs → severo={pct_sev:.0f}%")
        except Exception as e:
            print(f"    Modelo base falló: {e}")

    # Cada año siguiente: re-estima GARCH y HMM con toda la historia hasta ese año
    for año in años_oos:
        flujo_hasta = flujo_base[flujo_base.index.year <= año]
        params = _garch_fit(flujo_hasta.values)
        if params is None:
            print(f"    {año}: GARCH falló — omitiendo")
            continue
        try:
            sigma_hasta = _garch_vol_insample(params, flujo_hasta.values)
            X_hasta = np.column_stack([flujo_hasta.values, sigma_hasta])
            modelo, scaler, sorted_states = _fit_hmm_single(X_hasta)
            estados = _predecir_estados(modelo, scaler, sorted_states, X_hasta)
            resultados[año] = (flujo_hasta.index, estados)
            pct_sev = (estados == 2).mean() * 100
            print(f"    hasta {año}: {len(flujo_hasta):,} obs → severo={pct_sev:.0f}%")
        except Exception as e:
            print(f"    {año}: falló — {e}")

    return resultados


def graficar_evolucion(df: pd.DataFrame, evol: dict,
                       embig: pd.Series = None) -> None:
    """
    Fila 0: EMBIG Perú (riesgo país) — referencia visual, justo bajo el título.
    Fila 1: flujo neto diario (referencia visual).
    Filas 2..N+1: un panel por año — re-etiqueta toda la historia con el
    modelo entrenado hasta ese año.
    """
    años = sorted(evol.keys())
    n    = len(años)
    if n == 0:
        return

    tiene_embig = embig is not None and not embig.empty
    n_extra  = 2 if tiene_embig else 1
    n_rows   = n_extra + n
    h_ratios = ([1.2] if tiene_embig else []) + [2.0] + [1.0] * n

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(19, 2.0 + 2.5 * n),
        sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.05}
    )

    fig.suptitle(
        f"Evolución del HMM — {BANCO}  |  {N_ESTADOS} estados\n"
        f"Cada panel re-etiqueta TODA la historia con el modelo entrenado hasta ese año",
        fontsize=12, fontweight="bold", y=0.995
    )

    colores_hmm = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}

    siguiente = 0

    # ── Panel EMBIG (justo debajo del título) ─────────────────────────────────
    if tiene_embig:
        ax_embig = axes[siguiente]
        embig_v  = embig.reindex(df.index.union(embig.index)).sort_index()
        embig_v  = embig_v.loc[embig_v.index >= df.index.min()]
        ax_embig.plot(embig_v.index, embig_v.values, color="#6A1B9A", lw=0.8)
        ax_embig.fill_between(embig_v.index, embig_v.values, alpha=0.12, color="#6A1B9A")
        ax_embig.set_ylabel("EMBIG\n(pbs)", fontsize=8)
        ax_embig.set_title("EMBIG Perú — riesgo país (pbs)", fontsize=9)
        for ini, fin, _ in EPISODIOS:
            ax_embig.axvspan(pd.Timestamp(ini), pd.Timestamp(fin),
                             alpha=0.10, color="navy", zorder=0)
        siguiente += 1

    # ── Panel: flujo neto diario ───────────────────────────────────────────────
    ax_flujo  = axes[siguiente]
    siguiente += 1
    flujo     = df["target"].values
    fechas    = df.index
    colores_f = np.where(flujo >= 0, "#2196F3", "#F44336")
    ax_flujo.bar(fechas, flujo, color=colores_f, alpha=0.65, width=1.2)
    ax_flujo.axhline(0, color="k", lw=0.7, ls="--")
    ax_flujo.set_ylabel("Flujo neto", fontsize=9)
    ax_flujo.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B"))
    ax_flujo.set_title("Flujo neto diario (azul=entradas, rojo=salidas netas)", fontsize=9)
    for ini, fin, etiqueta in EPISODIOS:
        ax_flujo.axvspan(pd.Timestamp(ini), pd.Timestamp(fin),
                         alpha=0.10, color="navy", zorder=0)
        mid  = pd.Timestamp(ini) + (pd.Timestamp(fin) - pd.Timestamp(ini)) / 2
        ymax = ax_flujo.get_ylim()[1]
        ax_flujo.text(mid, ymax * 0.88, etiqueta, ha="center", fontsize=8,
                      color="#1A237E",
                      bbox=dict(boxstyle="round,pad=0.2", fc="lavender", alpha=0.8))

    # ── Paneles HMM por año ───────────────────────────────────────────────────
    for i, año in enumerate(años):
        ax           = axes[siguiente + i]
        idx_a, est_a = evol[año]
        vol_a        = df.loc[df.index.isin(idx_a), "garch_vol"]

        for estado, color in colores_hmm.items():
            fechas_e = idx_a[est_a == estado]
            for f in fechas_e:
                ax.axvspan(f, f + pd.offsets.BusinessDay(1),
                           alpha=0.55, color=color, linewidth=0)

        vol_n = (vol_a.values - np.nanmean(vol_a.values)) / (np.nanstd(vol_a.values) + 1e-12)
        ax.plot(vol_a.index, vol_n, color="k", lw=0.5, alpha=0.5)

        for ini, fin, _ in EPISODIOS:
            if pd.Timestamp(ini) <= idx_a.max():
                ax.axvspan(pd.Timestamp(ini),
                           min(pd.Timestamp(fin), idx_a.max()),
                           alpha=0.12, color="navy", zorder=0)

        pct_sev = (est_a == 2).mean() * 100
        ax.set_ylabel(str(año), fontsize=8, rotation=0, labelpad=35, va="center")
        ax.set_title(f"Modelo entrenado hasta {año}  ({len(idx_a):,} obs)  —  "
                     f"severo={pct_sev:.0f}%", fontsize=8, loc="left", pad=2)
        ax.set_yticks([])

        if i == 0:
            parches = [mpatches.Patch(color=c, label=l)
                       for c, l in zip(["#4CAF50", "#FFC107", "#F44336"],
                                       ["Calma", "Moderado", "Severo"])]
            ax.legend(handles=parches, fontsize=7, loc="upper left", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(top=0.93, hspace=0.05)
    ruta = DIR_OUTPUT / f"evolucion_hmm_{BANCO}.png"
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    print(f"\n  Gráfico evolución guardado: {ruta}")
    plt.show()


def mostrar_como_feature(df: pd.DataFrame, hmm_serie: pd.Series,
                         hmm_leakage: tuple) -> None:
    """
    Imprime cómo quedaría la columna hmm_estado en la matriz de features
    y la compara contra la versión con leakage en fechas clave.
    """
    idx_leak, est_leak = hmm_leakage
    leak_serie = pd.Series(est_leak, index=idx_leak, name="hmm_leakage")

    # Unir en un DataFrame de comparación
    comp = df[["target", "garch_vol"]].copy()
    comp["hmm_opcion_A"] = hmm_serie           # pre-computado sin leakage
    comp["hmm_leakage"]  = leak_serie          # con leakage (referencia)

    etiquetas = {0: "calma", 1: "mod.", 2: "severo"}

    print("\n" + "═" * 75)
    print("PASO 1 — Lo que step001 grabaría en matriz_features.parquet")
    print("         (columna hmm_estado calculada UNA SOLA VEZ, expanding)")
    print("═" * 75)
    print(f"\n  {'Fecha':<13} {'garch_vol':>10} {'hmm_OpcionA':>13} {'hmm_Leakage':>13}")
    print("  " + "-" * 55)

    # Fechas clave para comparar
    fechas_muestra = []
    for ini, fin, _ in EPISODIOS:
        rng = comp.loc[ini:fin].index
        if len(rng) >= 3:
            fechas_muestra.extend(rng[:2].tolist())
            fechas_muestra.append(rng[-1])

    # Añadir algunas fechas de calma para contraste
    año_calma = str(df_base_year(HMM_INICIO) + HMM_PRIMER_VENTANA + 1)
    rng_calma = comp.loc[año_calma:año_calma].index
    if len(rng_calma) >= 2:
        fechas_muestra = rng_calma[:2].tolist() + fechas_muestra

    for f in sorted(set(fechas_muestra)):
        if f not in comp.index:
            continue
        row    = comp.loc[f]
        a_val  = row["hmm_opcion_A"]
        l_val  = row["hmm_leakage"]
        a_str  = f"{int(a_val)}={etiquetas[int(a_val)]}" if pd.notna(a_val) else "NaN"
        l_str  = f"{int(l_val)}={etiquetas[int(l_val)]}" if pd.notna(l_val) else "NaN"
        marca  = " ←DIFIEREN" if (pd.notna(a_val) and pd.notna(l_val)
                                   and int(a_val) != int(l_val)) else ""
        print(f"  {str(f.date()):<13} {row['garch_vol']:>10.4f} "
              f"{a_str:>13} {l_str:>13}{marca}")

    print("\n" + "═" * 75)
    print("PASO 2 — Cómo lo vería un fold de step005")
    print("═" * 75)
    print("""
  Fold k:  train = [2010-01-01, 2019-12-31]
           val   = [2020-01-01, 2020-06-30]
           test  = [2020-07-01, 2020-12-31]

  X_train incluye columna hmm_estado (pre-computada, sin leakage)
  X_test  incluye columna hmm_estado (pre-computada, sin leakage)

  XGBoost NO sabe que es un HMM — lo trata como feature entero {0, 1, 2}.
  No hay re-entrenamiento de HMM dentro del fold.
""")

    # Resumen de discrepancias en episodios
    print("  Discrepancias Opción A vs Leakage en episodios de stress:")
    for ini, fin, etiqueta in EPISODIOS:
        sub = comp.loc[ini:fin].dropna(subset=["hmm_opcion_A", "hmm_leakage"])
        if sub.empty:
            continue
        pct_sev_A = (sub["hmm_opcion_A"] == 2).mean() * 100
        pct_sev_L = (sub["hmm_leakage"]  == 2).mean() * 100
        acuerdo   = (sub["hmm_opcion_A"] == sub["hmm_leakage"]).mean() * 100
        print(f"  {etiqueta:<20}: severo OpA={pct_sev_A:.0f}%  "
              f"Leak={pct_sev_L:.0f}%  |  acuerdo={acuerdo:.0f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Cargar datos
    df = cargar_datos(RUTA_MATRIZ, BANCO, H_REF)

    # 2. SADF sobre garch_vol
    vol = df["garch_vol"].fillna(method="ffill").values
    sadf_dict = {}
    for ventana in VENTANAS:
        nombre = f"sadf_vol_{ventana}d"
        print(f"\nCalculando {nombre}  ({len(vol):,} puntos, ~2-5 min)…")
        sadf_dict[nombre] = calcular_sadf(vol, tau_min=TAU_MIN,
                                          ventana_max=ventana, lags=LAGS)
        p95 = np.nanpercentile(sadf_dict[nombre], 95)
        p99 = np.nanpercentile(sadf_dict[nombre], 99)
        print(f"  P95={p95:.2f} | P99={p99:.2f}")

    # 3. HMM — tres versiones
    hmm_c  = None   # muestra completa (con leakage, referencia)
    hmm_ex = None   # expanding = OPCIÓN A (pre-computado, sin leakage)
    hmm_ro = None   # rolling ventana fija (sin leakage)

    if _HMM_OK:
        print(f"\n[LEAKAGE] HMM muestra completa — solo referencia visual…")
        try:
            hmm_c = hmm_muestra_completa(df, HMM_INICIO)
            print("  OK")
        except Exception as e:
            print(f"  Falló: {e}")

        print(f"\n[OPCIÓN A] Pre-computar HMM expanding (simula step001)…")
        try:
            hmm_ex_serie = precomputar_hmm_features(df)
            # Convertir a tupla (idx, array) para graficar
            hmm_ex = (hmm_ex_serie.index, hmm_ex_serie.fillna(-1).astype(int).values)
            # Filtrar NaN
            mask   = hmm_ex_serie.notna()
            hmm_ex = (hmm_ex_serie.index[mask], hmm_ex_serie[mask].astype(int).values)
            print("  OK")
        except Exception as e:
            print(f"  Falló: {e}")
            hmm_ex_serie = None

        print(f"\n[ROLLING {HMM_ROLLING_AÑOS}a] HMM rolling (sin leakage)…")
        try:
            hmm_ro = hmm_rolling(df, HMM_INICIO, ventana_años=HMM_ROLLING_AÑOS)
            print("  OK")
        except Exception as e:
            print(f"  Falló: {e}")

        # Mostrar tabla comparativa Opción A vs Leakage
        if hmm_ex is not None and hmm_c is not None:
            hmm_ex_serie_full = pd.Series(
                hmm_ex[1], index=hmm_ex[0], name="hmm_estado"
            )
            mostrar_como_feature(df, hmm_ex_serie_full, hmm_c)

    # 4. Estadísticas
    reportar_estadisticas(df, sadf_dict, hmm_c, hmm_ex, hmm_ro)

    # 5. Gráfico comparativo (hasta 5 paneles)
    graficar(df, sadf_dict, hmm_c, hmm_ex, hmm_ro)

    # 6. Gráfico de evolución: cómo se re-etiqueta la historia al añadir cada año
    if _HMM_OK:
        print(f"\nGenerando gráfico de evolución HMM...")
        evol  = hmm_evolucion(df, primer_ventana=HMM_PRIMER_VENTANA)
        embig = cargar_embig()
        graficar_evolucion(df, evol, embig=embig)


if __name__ == "__main__":
    main()
