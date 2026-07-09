# -*- coding: utf-8 -*-
"""
step005_walk_forward_cv_3.py
Walk-forward CV con ventana EXPANDABLE o RODANTE y evaluación TEST out-of-sample.

===============================================================================
NOVEDADES RESPECTO A v2
===============================================================================

1. Toggle EXPANDING (True/False)
   -----------------------------
   EXPANDING = True  -> ventana CRECIENTE: train_start fijo en el origen,
                        train_end crece PASO_AÑOS cada fold.
   EXPANDING = False -> ventana RODANTE   (idéntico a v2): train_start y
                        train_end avanzan juntos; tamaño fijo = VENTANA_TRAIN_AÑOS.

   En modo EXPANDING, VENTANA_TRAIN_AÑOS es el tamaño MÍNIMO inicial.
   Cada fold siguiente tiene un año (o PASO_AÑOS) más de historia.

   Comparación de enfoques:
     Expanding -> más datos en folds tardíos; mejor si el proceso es estacionario.
     Rolling   -> se adapta a regímenes recientes; mejor si hay quiebres estructurales.

2. Fan charts TEST out-of-sample (nuevo)
   --------------------------------------
   Por cada fold se generan hasta 4 snapshots del fan chart en el período TEST,
   con origen separado ~3 meses entre sí. Cada snapshot muestra:
     . Bandas Q01-Q99 (muy suave) y Q05-Q95 (media) -> incertidumbre
     . Línea Q50 -> mediana predicha
     . Puntos realizados: verde si caen dentro de Q05-Q95, rojo si fuera
     . Coverage empírico anotado en el título de cada panel

3. Separación VAL / TEST
   ----------------------
   Idéntica a v2: VAL solo para Optuna, TEST solo para métricas OOS.

===============================================================================
Anti-leakage -- idéntico a v1 y v2
===============================================================================
  #1 EMBARGO 90dh post-TRAIN
  #2 GARCH por fold: omega/alpha/beta estimados solo en TRAIN, propagados a VAL+TEST
  #3 Medianas fold: imputación calculada en TRAIN, aplicada a VAL+TEST
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize

try:
    import lightgbm as lgb
    _LGBM_OK = True
except ImportError:
    lgb = None
    _LGBM_OK = False

_SHAP_OK = True  # XGBoost nativo pred_contribs=True siempre disponible

# Overlay: la logica vive aqui; step007 solo provee el Excel con peor_total diario.

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


###############################################################################
# PARTE 0 -- Configuración
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v3"

# -- Tipo de ventana -----------------------------------------------------------
# True  -> EXPANDING: train_start fijo, train_end crece cada fold
# False -> ROLLING  : ventana fija que desliza (idéntico a v2)
EXPANDING = True

# -- Recorte de inicio del train -----------------------------------------------
# True  -> el train nunca empieza antes de TRAIN_INICIO_CUTOFF (nueva estrategia 2020)
# False -> el train empieza desde el primer dato disponible
RECORTAR_INICIO_TRAIN = True
TRAIN_INICIO_CUTOFF   = "2020-01-01"

# -- Tamaños de ventana --------------------------------------------------------
# EXPANDING=True : VENTANA_TRAIN_AÑOS es el mínimo inicial; crece PASO_AÑOS/fold
# EXPANDING=False: VENTANA_TRAIN_AÑOS es el tamaño fijo (igual a v2)
VENTANA_TRAIN_AÑOS  = 3      # años de TRAIN iniciales / fijos (mínimo desde 2020)
VENTANA_VAL_AÑOS    = 0.5    # años de VAL (solo Optuna) -- 6 meses, igual que step004
VENTANA_TEST_AÑOS   = 1      # años de TEST (solo métricas OOS)
PASO_AÑOS           = 1      # desplazamiento / crecimiento entre folds

# -- Anti-leakage: purga + burn-in --------------------------------------------
# H_MAX_DIAS_HAB   : horizonte máximo de predicción (h_max en step001)
# PURGE_DIAS_HAB   : días hábiles excluidos entre TRAIN-end y VAL-start.
#                    Cubre: (a) solapamiento de etiquetas Y (h_max dh) y
#                           (b) feature lookback (MA22 = 22 dh) -- 75 >= 22 -> redundante.
# PURGE_VAL_TEST   : análogo entre VAL-end y TEST-start (labels Optuna no cruzan TEST)
# BURN_IN_DIAS_HAB : excluye los primeros días de TRAIN donde MA22 aún no maduró
H_MAX_DIAS_HAB   = 75   # igual que h_max en step001
PURGE_DIAS_HAB   = H_MAX_DIAS_HAB   # purga TRAIN -> VAL
PURGE_VAL_TEST   = H_MAX_DIAS_HAB   # purga VAL   -> TEST
BURN_IN_DIAS_HAB = 22               # warm-up MA22 al inicio de TRAIN

# -- Diagnóstico de features (PARTE 7-bis) ------------------------------------
# True  -> mide gain(train) + block-perm(val) + SHAP(val) por fold y consolida
# False -> sin diagnóstico extra (comportamiento original)
DIAGNOSTICO_FEATURES  = True
# h va de 2 a H_MAX_DIAS_HAB=75 -> n_h = 74 filas por fecha_t.
# block_size debe ser múltiplo de n_h para que cada bloque = exactamente 1 fecha_t
# y la permutación solo intercambie fechas completas (unidad temporal correcta).
# Con block_size < n_h se mezclan h-values de distintos horizontes dentro del
# mismo día, creando combinaciones (fecha_t, h_features, target_h') imposibles.
DIAG_BLOCK_SIZE       = H_MAX_DIAS_HAB - 1   # = 74 = n_h (h: 2..75)
DIAG_N_REPEATS        = 3     # repeticiones de la permutación
DIAG_PERM_MAX_SAMPLES = None  # ver _diag_block_perm_promedio; None = todo VAL
DIAG_SHAP_MAX_SAMPLES = 800   # muestras para SHAP por cuantil

# Pares cíclicos sin/cos que deben permutarse SIMULTÁNEAMENTE (mismo índice de shuffle).
# Distribuimos el delta combinado D como D/sqrt2 en cada componente para que la norma
# euclidiana en aux_comparar_features.py reconstruya exactamente D.
_CICL_BASES_PERM = [
    "mes", "dias_sem",
    "dias_al_cierre_mes", "dias_al_cierre_trim", "dias_al_cierre_anio",
    "elec",
]
_CICL_PARES_PERM = {
    f"{b}_{s}": f"{b}_cyc"
    for b in _CICL_BASES_PERM
    for s in ("sin", "cos")
}

# -- Modelo --------------------------------------------------------------------
QUANTILES        = [0.01, 0.05, 0.50, 0.95, 0.99]
S_MIN_FACTOR     = 0.01
S_MAX_FACTOR     = 1.00   # targets en escala raw (MM USD), no estandarizados -> rango amplio

# -- Trials Optuna -------------------------------------------------------------
# True  -> número de trials varía por cuantil (TRIALS_POR_TAU)
# False -> número fijo para todos los cuantiles (TRIALS_FLAT)
ADAPTIVE_TRIALS  = True

TRIALS_FLAT      = 60        # usado cuando ADAPTIVE_TRIALS = False

TRIALS_POR_TAU   = {         # usado cuando ADAPTIVE_TRIALS = True
    # Llaves = round(tau, 1) para QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]
    0.0: 110,   # tau=0.01 -> colas extremas, más difíciles de calibrar
    0.1: 110,   # tau=0.05
    0.5:  90,   # tau=0.50 -> cuantil central, converge más rápido
    0.9: 110,   # tau=0.95
    1.0: 110,   # tau=0.99 -> colas extremas
}

# -- Opciones de salida --------------------------------------------------------
BANCOS_A_EVALUAR          = ["SISTEMA"]
GUARDAR_MODELO_FINAL      = True
# True  -> guarda modelos de TODOS los folds (permite fan chart histórico sin lookahead)
# False -> solo guarda el último fold (comportamiento anterior)
GUARDAR_MODELOS_TODOS_FOLDS = True

# True  -> omite Optuna/entrenamiento, carga modelos del disco y solo regenera los plots
# False -> entrenamiento completo (comportamiento normal)
SOLO_REGENERAR_PLOTS = False
COLS_EXCLUIR              = {
    "fecha_t", "banco", "target",
    # T10Y y EMBI_PERU son series I(0): T10Y es nivel de tasa; EMBI es diferencial de tasas.
    # Para I(0) el FFD tiene d_opt~=0 -> T10Y_frac~=T10Y y EMBI_PERU_frac~=EMBI_PERU (redundantes).
    # Todas muestran gain~=1.0/perm~=0 en val -> overfitting por correlación espuria en train.
    # Se mantienen: delta_EMBI (shocks de corto plazo), garch_vol_embi (volatilidad del spread).
    "T10Y", "T10Y_frac", "EMBI_PERU", "EMBI_PERU_frac",
}

# -- Límite de folds -----------------------------------------------------------
# None -> usa todos los folds generados
# N    -> usa solo los primeros N folds (los más antiguos); deja el resto como OOS
# Expanding=True: 9 folds (fold 9 incluye test 2023+)
# Expanding=False (rolling): 8 folds
N_MAX_FOLDS = 9 if EXPANDING else 8

# -- Selector de modelo --------------------------------------------------------
MODELO_CV = "xgb"
# Opciones: "xgb" | "lgbm" | "xgb_qt"
assert MODELO_CV in ("xgb", "lgbm", "xgb_qt"), \
    f"MODELO_CV debe ser 'xgb', 'lgbm' o 'xgb_qt', recibido: {MODELO_CV!r}"

# -- Parámetro s (suavizado Pinball-Arctan) ------------------------------------
# True  -> s fijo en S_FACTOR_FIJO × std_y (recomendado por el paper 2406.02293)
#          Optuna no busca s; libera trials para otros hiperparámetros
# False -> Optuna busca s en [S_MIN_FACTOR, S_MAX_FACTOR] × std_y
S_FIJO         = False
S_FACTOR_FIJO  = 0.05   # equivale a s=0.05 en datos estandarizados (centro del rango paper)

# -- Calibración post-hoc (shift aditivo estimado en VAL) ---------------------
# True  -> después de predecir, calcula el sesgo sistemático en VAL y lo corrige
#         en TEST: todas las predicciones se desplazan por el percentil P del
#         residuo (actual - Q50_pred) en VAL.
#         P=50 -> elimina sesgo (mediana de errores)
#         P<50 -> introduce sesgo negativo (conservador para riesgo de liquidez)
#         P=25 -> desplaza hacia abajo con sesgo negativo moderado
# False -> sin calibración (comportamiento original)
CALIBRACION_POSTHOC   = False
CALIBRACION_PERCENTIL = 25    # percentil del residuo VAL usado como shift
# Límite del shift como fracción de std_y: previene correcciones exageradas
# que distorsionan el eje Y cuando el modelo tiene sesgo grande.
# None -> sin límite (el valor original antes de esta corrección)
CALIBRACION_MAX_SHIFT_FACTOR = 0.5   # e.g. 0.5 -> shift <= ±0.5×std_y

# -- Mondrian CQR (Conformalized Quantile Regression por horizonte h) ----------
# True  -> calibra el intervalo [CQR_TAU_LO, CQR_TAU_HI] usando scores de
#          conformidad en VAL (modelo solo-train, OOS honesto) y aplica
#          la corrección simétrica q_hat_h a TEST (modelo retrenado train+val).
#          Sin leakage: scores vienen de modelos (solo-train), preds_test de
#          modelos_final (retrain). Mondrian = q_hat separado por cada h.
# False -> sin calibración CQR (comportamiento original)
CALIBRACION_CQR  = False
CQR_TAU_LO       = 0.05   # cuantil inferior del intervalo a calibrar
CQR_TAU_HI       = 0.95   # cuantil superior del intervalo a calibrar
CQR_ALPHA        = 0.10   # miscoverage objetivo: 1 - cobertura deseada (90% -> 0.10)

# -- Overlay sobreencaje (step007) --------------------------------------------
# Requiere haber ejecutado step007 para generar saldos_retiros_bancos.xlsx.
# El ajuste diario (peor_total) se lee desde la tab "Ajuste_diario" de ese archivo.
OVERLAY_SOBREENCAJE             = False
RUTA_AJUSTE_OVERLAY             = BASE_SISTEMA / "2. Output" / "analisis_cc" / "saldos_retiros_bancos.xlsx"
OVERLAY_VENTANA_DH              = 7   # dias habiles de la ventana de retiro (mismo valor que step007)
OVERLAY_TAU_REFERENCIA          = 0.05  # quantil usado como denominador del factor
OVERLAY_CONOCIMIENTO_ANTICIPADO = 2   # T+N: flujos conocidos con N dias habiles de antelacion


# -- Fan chart TEST: número de snapshots por fold ------------------------------
FANCHART_N_SNAPSHOTS = 4   # 1 cada ~3 meses para TEST de 1 año

# -- Diagnóstico de features (PARTE 7-bis) -------------------------------------
# True  -> corre gain / block-perm / SHAP por fold y genera los gráficos
# False -> omite el diagnóstico (más rápido)
DIAGNOSTICO_FEATURES  = True
DIAG_BLOCK_SIZE       = H_MAX_DIAS_HAB - 1   # = 74 = n_h (h: 2..75); 1 bloque = 1 fecha_t
DIAG_N_REPEATS        = 3     # repeticiones por feature para estabilizar la estimación
DIAG_SHAP_MAX_SAMPLES = 800   # máximo de filas VAL para SHAP (None = todas)

# -- Comparación con Step004 en fan charts -------------------------------------
# True  -> superpone predicciones del modelo step004 (línea naranja discontinua)
#          para comparar visualmente con step005 fold-by-fold
# False -> solo muestra predicciones step005 (comportamiento normal)
COMPARAR_CON_STEP004 = False
# Directorio donde están los modelos step004 (eval = entrenado solo hasta TRAIN)
_STEP004_SUFIJO = {
    "xgb"   : ("modelos_xgb",    "eval", "xgb"),
    "xgb_qt": ("modelos_xgb_qt", "eval", "xgb_qt"),
    "lgbm"  : ("modelos_lgbm",   "eval", "lgbm"),
}
_s4_carpeta, _s4_subcarpeta, _s4_prefijo = _STEP004_SUFIJO.get(
    MODELO_CV, ("modelos_xgb", "eval", "xgb")
)
DIR_MODELOS_STEP004 = BASE_SISTEMA / "2. Output" / _s4_carpeta / _s4_subcarpeta

# -- Folds manuales -- comparación directa con step004 -------------------------
# Lista de folds con fechas exactas, añadidos a (o reemplazando) los folds auto.
# Permite replicar el split de step004 sin embargo entre TRAIN y VAL.
# Descomentar el ejemplo para activar el fold equivalente a step004:
FOLDS_MANUALES: list[dict] = []
# FOLDS_MANUALES = [
#     {
#         "train_start": "2015-01-02",   # inicio real de los datos
#         "train_end"  : "2022-06-30",   # mismo corte que step004 (CORTE_VAL - 1d)
#         "val_start"  : "2022-07-01",   # sin embargo, igual que step004
#         "val_end"    : "2023-01-02",   # CORTE_TEST - 1d
#         "test_start" : "2023-01-03",   # mismo CORTE_TEST que step004
#         "test_end"   : "2024-06-30",   # hasta donde haya datos
#     }
# ]

# True  -> corre SOLO los folds manuales (omite los generados automáticamente)
# False -> añade los folds manuales al final de los generados
SOLO_FOLDS_MANUALES = False

# -- Rutas de salida -----------------------------------------------------------
_modo           = "expanding" if EXPANDING else "rolling"
_ventanas       = f"{VENTANA_TRAIN_AÑOS}{VENTANA_VAL_AÑOS}{VENTANA_TEST_AÑOS}"
DIR_MODO        = DIR_OUTPUT / f"{MODELO_CV}_{_modo}_{_ventanas}"
DIR_MODELOS            = DIR_MODO / "modelos"
DIR_PLOTS              = DIR_MODO / "plots"
DIR_FANCHARTS          = DIR_MODO / "fancharts_test"
DIR_FANCHARTS_MANUALES = DIR_MODO / "fancharts_manuales"   # plots de FOLDS_MANUALES

for _d in (DIR_OUTPUT, DIR_MODO, DIR_MODELOS, DIR_PLOTS,
           DIR_FANCHARTS, DIR_FANCHARTS_MANUALES):
    _d.mkdir(parents=True, exist_ok=True)


def get_n_trials(tau: float) -> int:
    """Devuelve el número de trials Optuna para el cuantil dado."""
    if ADAPTIVE_TRIALS:
        return TRIALS_POR_TAU.get(round(tau, 1), 90)
    return TRIALS_FLAT


###############################################################################
# PARTE 1 -- Métricas
###############################################################################

def pinball_loss(y_true, y_pred, tau):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def winkler_score(y_true, q_lo, q_hi, alpha=0.10):
    width = q_hi - q_lo
    below = np.maximum(q_lo - y_true, 0)
    above = np.maximum(y_true - q_hi, 0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def coverage(y_true, q_lo, q_hi):
    return float(np.mean((y_true >= q_lo) & (y_true <= q_hi)))


def crps_approx(y_true, preds):
    taus  = sorted(preds.keys())
    pbs   = [pinball_loss(y_true, preds[tau], tau) for tau in taus]
    n     = len(taus)
    pesos = []
    for i in range(n):
        lo = taus[i - 1] if i > 0     else taus[0]
        hi = taus[i + 1] if i < n - 1 else taus[-1]
        pesos.append((hi - lo) / 2.0)
    return float(2.0 * sum(w * pb for w, pb in zip(pesos, pbs)))


###############################################################################
# PARTE 2 -- GARCH por fold
###############################################################################

# Número de workers paralelos (uno por cuantil) y threads XGBoost por worker.
# Windows/macOS usan spawn en ProcessPoolExecutor -> re-importa el módulo completo
# (~500 MB/proceso) y serializa los arrays de datos por pickle, lo que provoca
# MemoryError con datasets grandes.  Solución: ThreadPoolExecutor en Windows/macOS.
# XGBoost libera el GIL durante el entrenamiento (C++ interno), por lo que el
# paralelismo real se preserva con threads sin copiar objetos entre procesos.
# Linux usa ProcessPoolExecutor con fork (copy-on-write) -> overhead mínimo.
_N_QUANTILES   = len([0.01, 0.05, 0.50, 0.95, 0.99])  # = 5
_spawn_ctx     = sys.platform in ("win32", "darwin")
_N_QUANTILES_PARALLEL = _N_QUANTILES          # threads no tienen límite de spawn
_XGB_NTHREAD = max(2, (os.cpu_count() or 10) // _N_QUANTILES_PARALLEL)
# Clase de executor según plataforma
_ExecutorCls = ThreadPoolExecutor if _spawn_ctx else ProcessPoolExecutor

# Cache de parámetros GARCH por fecha de corte de TRAIN -- evita re-estimación en el
# mismo fold y para el guardado de metadata (antes se estimaba 2-3 veces por fold).
_garch_params_cache: dict[str, dict] = {}

def _ajustar_garch_params(x_train):
    n       = len(x_train)
    var_unc = max(float(np.var(x_train)), 1e-12)

    def _s2(omega, alpha, beta):
        s2    = np.empty(n)
        s2[0] = var_unc
        for t in range(1, n):
            s2[t] = omega + alpha * x_train[t - 1] ** 2 + beta * s2[t - 1]
        return s2

    def _neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.9999:
            return 1e10
        s2 = _s2(omega, alpha, beta)
        if np.any(s2 <= 0):
            return 1e10
        return 0.5 * float(np.sum(np.log(s2) + x_train ** 2 / s2))

    try:
        res = minimize(
            _neg_loglik, [0.01, 0.08, 0.88], method="L-BFGS-B",
            bounds=[(1e-7, 0.5), (1e-7, 0.5), (1e-7, 0.9999)],
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        if res.fun < 1e9:
            return float(res.x[0]), float(res.x[1]), float(res.x[2])
    except Exception:
        pass
    return 0.01, 0.08, 0.88


def _garch_vol_fold(serie, train_end):
    serie_full  = serie.ffill().fillna(0.0)
    serie_train = serie_full[serie_full.index <= train_end]
    if len(serie_train) < 60 or serie_train.std() < 1e-9:
        return serie_full.rolling(20).std().fillna(serie_train.std())
    escala  = float(serie_train.std())
    x_train = (serie_train / escala).values.astype(float)
    var_unc = max(float(np.var(x_train)), 1e-12)
    omega, alpha, beta = _ajustar_garch_params(x_train)
    x_full = (serie_full / escala).values.astype(float)
    n_full = len(x_full)
    s2     = np.empty(n_full)
    s2[0]  = var_unc
    for t in range(1, n_full):
        s2[t] = omega + alpha * x_full[t - 1] ** 2 + beta * s2[t - 1]
    return pd.Series(np.sqrt(np.maximum(s2, 0)) * escala, index=serie_full.index)


def reemplazar_ffd_fold(df_fold, train_end):
    """
    Re-calibra FFD para cada columna _frac usando solo datos hasta train_end.
    Preserva la misma d_opt que la calibrada en build_feature_matrix pero
    garantiza que no se filtra información posterior a train_end.
    """
    frac_cols = [c for c in df_fold.columns if c.endswith("_frac")]
    if not frac_cols:
        return df_fold
    df_fold = df_fold.copy()

    try:
        from statsmodels.tsa.stattools import adfuller
        _statsmodels_ok = True
    except ImportError:
        _statsmodels_ok = False

    def _ffd_weights_local(d, thresh=1e-5):
        w, k = [1.0], 1
        while True:
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < thresh:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1])

    def _fracdiff_local(series, d, thresh=1e-5):
        w = _ffd_weights_local(d, thresh)
        width = len(w)
        vals = series.values.astype(float)
        out  = np.full(len(vals), np.nan)
        for i in range(width - 1, len(vals)):
            chunk = vals[i - width + 1: i + 1]
            if not np.any(np.isnan(chunk)):
                out[i] = float(np.dot(w, chunk))
        return pd.Series(out, index=series.index)

    def _find_d_local(series, n_steps=20, target_pval=0.05):
        if not _statsmodels_ok or len(series.dropna()) < 30:
            return 0.4
        for d in np.linspace(0.05, 1.0, n_steps):
            fd = _fracdiff_local(series.dropna(), round(float(d), 4))
            fd_clean = fd.dropna()
            if len(fd_clean) < 20:
                continue
            try:
                pval = adfuller(fd_clean, maxlag=1, regression="c", autolag=None)[1]
                if pval <= target_pval:
                    return round(float(d), 4)
            except Exception:
                continue
        return 1.0

    idx_col = "fecha_t"
    raw = (df_fold[[idx_col] + [c.replace("_frac", "") for c in frac_cols
                                if c.replace("_frac", "") in df_fold.columns]]
           .drop_duplicates(idx_col).set_index(idx_col).sort_index())

    for col_frac in frac_cols:
        col_raw = col_frac.replace("_frac", "")
        if col_raw not in raw.columns:
            continue
        serie_train = raw[col_raw][raw.index <= train_end].dropna()
        if len(serie_train) < 60:
            continue
        d_opt = _find_d_local(serie_train)
        frac_full = _fracdiff_local(raw[col_raw], d_opt)
        df_fold[col_frac] = df_fold[idx_col].map(frac_full)

    return df_fold


def reemplazar_garch_fold(df_fold, train_end):
    df_fold = df_fold.copy()
    idx_cols = ["fecha_t", "R_t0", "D_t0", "TC_PEN_USD", "EMBI_PERU"]
    avail    = [c for c in idx_cols if c in df_fold.columns]
    raw      = (df_fold[avail].drop_duplicates("fecha_t")
                .set_index("fecha_t").sort_index())
    if {"R_t0", "D_t0"}.issubset(raw.columns):
        sigma = _garch_vol_fold(raw["D_t0"] - raw["R_t0"], train_end)
        df_fold["garch_vol"] = df_fold["fecha_t"].map(sigma)
    if "TC_PEN_USD" in raw.columns:
        tc  = raw["TC_PEN_USD"].replace(0, np.nan).ffill()
        tci = tc.reindex(pd.bdate_range(tc.index.min(), tc.index.max())).ffill()
        ret = np.log(tci / tci.shift(1)).reindex(tc.index)
        df_fold["garch_vol_tc"] = df_fold["fecha_t"].map(_garch_vol_fold(ret, train_end))
    if "EMBI_PERU" in raw.columns:
        df_fold["garch_vol_embi"] = df_fold["fecha_t"].map(
            _garch_vol_fold(raw["EMBI_PERU"].diff(1), train_end))
    return df_fold


def _extraer_garch_params_fold(df, train_end):
    """
    Estima parámetros GARCH(1,1) usando solo datos de TRAIN (hasta train_end).
    Resultado cacheado por fecha de corte para no re-estimar en el mismo fold.
    """
    cache_key = str(train_end.date()) if hasattr(train_end, "date") else str(train_end)
    if cache_key in _garch_params_cache:
        return _garch_params_cache[cache_key]

    params   = {}
    idx_cols = ["fecha_t", "R_t0", "D_t0", "TC_PEN_USD", "EMBI_PERU"]
    avail    = [c for c in idx_cols if c in df.columns]
    raw      = (df[avail].drop_duplicates("fecha_t")
                .set_index("fecha_t").sort_index())

    def _params_serie(serie):
        serie_full  = serie.ffill().fillna(0.0)
        serie_train = serie_full[serie_full.index <= train_end]
        if len(serie_train) < 60 or serie_train.std() < 1e-9:
            return None
        escala  = float(serie_train.std())
        x_train = (serie_train / escala).values.astype(float)
        var_unc = max(float(np.var(x_train)), 1e-12)
        omega, alpha, beta = _ajustar_garch_params(x_train)
        n  = len(x_train)
        s2 = np.empty(n)
        s2[0] = var_unc
        for t in range(1, n):
            s2[t] = omega + alpha * x_train[t - 1] ** 2 + beta * s2[t - 1]
        return {
            "omega"         : omega,
            "alpha"         : alpha,
            "beta"          : beta,
            "escala"        : escala,
            "var_unc"       : var_unc,
            "last_sigma2"   : float(s2[-1]),
            "last_x_scaled" : float(x_train[-1]),
        }

    if {"R_t0", "D_t0"}.issubset(raw.columns):
        p = _params_serie(raw["D_t0"] - raw["R_t0"])
        if p:
            params["garch_vol"] = p

    if "TC_PEN_USD" in raw.columns:
        tc  = raw["TC_PEN_USD"].replace(0, np.nan).ffill()
        tci = tc.reindex(pd.bdate_range(tc.index.min(), tc.index.max())).ffill()
        ret = np.log(tci / tci.shift(1)).reindex(tc.index)
        p = _params_serie(ret)
        if p:
            params["garch_vol_tc"] = p

    if "EMBI_PERU" in raw.columns:
        p = _params_serie(raw["EMBI_PERU"].diff(1))
        if p:
            params["garch_vol_embi"] = p

    _garch_params_cache[cache_key] = params
    return params


###############################################################################
# PARTE 3 -- Generación de folds  (EXPANDING o ROLLING según toggle)
###############################################################################

def generar_folds(
    fechas_disponibles,
    ventana_train_años,
    ventana_val_años,
    ventana_test_años,
    paso_años,
    purge_dias_hab,
    purge_val_test,
    expanding,
):
    """
    EXPANDING=True  -> train_start fijo en f_min; train_end crece paso_años/fold.
    EXPANDING=False -> ventana rodante fija.

    Estructura por fold (López de Prado §12):
      TRAIN -> [purge_dias_hab] -> VAL (Optuna) -> [purge_val_test] -> TEST (métricas OOS)

    purge_dias_hab cubre tanto solapamiento de etiquetas Y (h_max dh) como
    el warm-up de features de lookback (MA22 <= 22 dh <= h_max).

    Genera folds mientras test_end <= última fecha disponible.
    """
    folds   = []
    f_min   = fechas_disponibles.min()
    f_max   = fechas_disponibles.max()
    # Aplica recorte de inicio si está activado
    _origin = (max(f_min, pd.Timestamp(TRAIN_INICIO_CUTOFF))
               if RECORTAR_INICIO_TRAIN else f_min)
    fold_idx = 0

    while True:
        if expanding:
            train_start = _origin
            train_end   = _origin + pd.DateOffset(
                months=int(round((ventana_train_años + fold_idx * paso_años) * 12)))
        else:
            train_start = _origin + pd.DateOffset(
                months=int(round(fold_idx * paso_años * 12)))
            train_end   = train_start + pd.DateOffset(
                months=int(round(ventana_train_años * 12)))

        val_start  = train_end  + pd.offsets.BusinessDay(purge_dias_hab)
        val_end    = val_start  + pd.DateOffset(months=int(round(ventana_val_años * 12)))
        test_start = val_end    + pd.offsets.BusinessDay(purge_val_test)
        test_end   = test_start + pd.DateOffset(months=int(round(ventana_test_años * 12)))

        if test_end > f_max or train_end >= f_max or val_start >= f_max:
            break

        burn_cutoff  = train_start + pd.offsets.BusinessDay(BURN_IN_DIAS_HAB)
        n_train_all  = int(((fechas_disponibles >= train_start) &
                            (fechas_disponibles <= train_end)).sum())
        n_train_eff  = int(((fechas_disponibles >= burn_cutoff) &
                            (fechas_disponibles <= train_end)).sum())
        n_val        = int(((fechas_disponibles >= val_start) &
                            (fechas_disponibles <  test_start)).sum())
        n_test       = int(((fechas_disponibles >= test_start) &
                            (fechas_disponibles <= test_end)).sum())

        if n_train_eff < 60 or n_val < 10 or n_test < 10:
            fold_idx += 1
            continue

        folds.append({
            "fold"              : fold_idx + 1,
            "train_start"       : train_start,
            "train_end"         : train_end,
            "val_start"         : val_start,
            "val_end"           : val_end,
            "test_start"        : test_start,
            "test_end"          : test_end,
            "burn_cutoff"       : burn_cutoff,
            "n_train_fechas"    : n_train_all,
            "n_train_efectivo"  : n_train_eff,
            "n_val_fechas"      : n_val,
            "n_test_fechas"     : n_test,
        })
        fold_idx += 1

    return folds


def resolver_folds_manuales(
    folds_cfg: list[dict],
    fechas_disponibles,
    n_folds_previos: int,
) -> list[dict]:
    """
    Convierte la lista FOLDS_MANUALES en dicts compatibles con el pipeline.

    Cada entrada de folds_cfg debe tener:
      train_start, train_end, val_start, val_end, test_start, test_end  (str o Timestamp)

    Los fold numbers se asignan como n_folds_previos+1, +2, ...
    El campo "_manual": True permite identificarlos en los logs.
    """
    folds = []
    for i, cfg in enumerate(folds_cfg):
        ts = pd.Timestamp(cfg["train_start"])
        te = pd.Timestamp(cfg["train_end"])
        vs = pd.Timestamp(cfg["val_start"])
        ve = pd.Timestamp(cfg["val_end"])
        xs = pd.Timestamp(cfg["test_start"])
        xe = pd.Timestamp(cfg["test_end"])

        n_train = int(((fechas_disponibles >= ts) & (fechas_disponibles <= te)).sum())
        n_val   = int(((fechas_disponibles >= vs) & (fechas_disponibles <  xs)).sum())
        n_test  = int(((fechas_disponibles >= xs) & (fechas_disponibles <= xe)).sum())

        folds.append({
            "fold"           : n_folds_previos + i + 1,
            "train_start"    : ts,
            "train_end"      : te,
            "val_start"      : vs,
            "val_end"        : ve,
            "test_start"     : xs,
            "test_end"       : xe,
            "n_train_fechas" : n_train,
            "n_val_fechas"   : n_val,
            "n_test_fechas"  : n_test,
            "_manual"        : True,
        })
    return folds


###############################################################################
# PARTE 4 -- Modelos: objetivos, optimización, entrenamiento
###############################################################################

def make_quantile_objective(tau, s, std_y):
    # _scale ajusta el lambda efectivo de XGBoost según la escala del target
    # (target no estandarizado: std_y~80,000 -> _scale~1e9).
    # Se aplica a grad; hess ya lo incorpora algebraicamente en su forma simplificada.
    _scale = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)
    def objective(y_pred, dtrain):
        # Clamp predicciones divergentes: en trials Optuna con malos
        # hiperparámetros las predicciones XGBoost (float32) pueden llegar
        # a ±inf, produciendo u=±inf -> u²=inf -> overflow en el hessiano.
        y_pred = np.clip(y_pred, -1e15, 1e15)
        u      = dtrain.get_label() - y_pred
        s2u2   = s ** 2 + u ** 2          # siempre > 0, sin overflow en float64
        grad   = -((tau - 0.5 + np.arctan(u / s) / np.pi)
                   + u * s / (np.pi * s2u2)) * _scale
        # Forma simplificada: hess = ((s²+std_y²)/(s²+u²))²
        # Algebraicamente idéntica a la original pero evita los intermedios
        # ~1e33 de _scale × ~1e-13 del cociente que provocan pérdida de precisión.
        hess   = ((s ** 2 + std_y ** 2) / s2u2) ** 2
        return grad, hess
    return objective


def make_pinball_metric(tau):
    def metric(y_pred, dtrain):
        y_true = dtrain.get_label()
        u      = y_true - y_pred
        return "pinball", float(np.mean(np.where(u >= 0, tau * u, (tau - 1) * u)))
    return metric


class _PinballEarlyStopping(xgb.callback.TrainingCallback):
    """
    Early stopping sobre la métrica 'pinball' del eval set de validación.
    Reemplaza xgb.callback.EarlyStopping porque ese callback busca el nombre
    en un dict plano ("val-pinball") mientras que XGBoost con custom_metric
    almacena en estructura anidada: evals_log["val"]["pinball"].
    """
    def __init__(self, rounds: int = 50):
        super().__init__()
        self.rounds  = rounds
        self._best   = float("inf")
        self._since  = 0

    def after_iteration(self, model, epoch, evals_log):
        score = None
        for metrics in evals_log.values():
            if "pinball" in metrics:
                score = metrics["pinball"][-1]
                break
        if score is None:
            return False
        if score < self._best - 1e-9:
            self._best  = score
            self._since = 0
        else:
            self._since += 1
        return self._since >= self.rounds


def _objective_optuna(trial, X_tr, y_tr, X_va, y_va, std_y):
    s = (std_y * S_FACTOR_FIJO if S_FIJO
         else trial.suggest_float("s", std_y * S_MIN_FACTOR, std_y * S_MAX_FACTOR, log=True))
    params = {
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth"       : trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 200),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method"     : "hist",
        "nthread"         : _XGB_NTHREAD,
        "seed"            : 42,
    }
    n_est  = trial.suggest_int("n_estimators", 100, 1500)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    model  = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(0.50, s, std_y),
        custom_metric=make_pinball_metric(0.50),
        evals=[(dval, "val")],
        callbacks=[_PinballEarlyStopping(rounds=50)],
        verbose_eval=False,
    )
    return pinball_loss(y_va.values, model.predict(dval), 0.50)


def optimizar_hiperparametros(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
    logger.info(f"    Optuna fold {fold_num} ({n_trials} trials, tau=0.50 en VAL)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold_num),
    )
    study.optimize(
        lambda t: _objective_optuna(t, X_tr, y_tr, X_va, y_va, std_y),
        n_trials=n_trials, show_progress_bar=False,
    )
    bp = study.best_params
    if S_FIJO:
        bp["s"] = std_y * S_FACTOR_FIJO
    logger.info(f"    Fold {fold_num} best pinball(Q50/VAL)={study.best_value:.4f} "
                f"n_est={bp['n_estimators']} lr={bp['learning_rate']:.4f} s={bp['s']:.2f}")
    return bp


def entrenar_quantiles(X_tr, y_tr, best_params, quantiles, std_y):
    s_best = best_params["s"]
    n_est  = best_params["n_estimators"]
    params = {k: v for k, v in best_params.items() if k not in ("s", "n_estimators")}
    params.update({"tree_method": "hist", "seed": 42, "nthread": _XGB_NTHREAD})

    # DMatrix creado una vez y compartido entre threads para evitar
    # fragmentación de heap en Windows con múltiples allocaciones simultáneas.
    gc.collect()
    dtrain = xgb.DMatrix(X_tr, label=y_tr)

    def _train_tau(tau):
        return tau, xgb.train(
            params, dtrain, num_boost_round=n_est,
            obj=make_quantile_objective(tau, s_best, std_y),
            verbose_eval=False,
        )

    modelos = {}
    with ThreadPoolExecutor(max_workers=len(quantiles)) as ex:
        for tau, model in ex.map(lambda t: _train_tau(t), quantiles):
            modelos[tau] = model
    return modelos


def predecir_y_corregir(modelos, X):
    dmat      = xgb.DMatrix(X)
    preds_raw = {tau: m.predict(dmat) for tau, m in modelos.items()}
    # "mean" key is not a quantile -- exclude from crossing correction
    mean_pred = preds_raw.pop("mean", None)
    taus      = sorted(preds_raw)
    matrix    = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    result    = {t: matrix[:, i] for i, t in enumerate(taus)}
    if mean_pred is not None:
        result["mean"] = mean_pred
    return result


# -- LightGBM ------------------------------------------------------------------

def _objetivo_optuna_lgbm(trial, X_tr, y_tr, X_va, y_va):
    if not _LGBM_OK:
        raise ImportError("lightgbm no está instalado")
    tau = 0.50
    params = {
        "objective": "quantile", "alpha": tau, "verbosity": -1, "seed": 42,
        "learning_rate"    : trial.suggest_float("learning_rate",    0.01,  0.3,  log=True),
        "num_leaves"       : trial.suggest_int(  "num_leaves",        15,   255),
        "max_depth"        : trial.suggest_int(  "max_depth",          3,    10),
        "min_child_samples": trial.suggest_int(  "min_child_samples", 10,   200),
        "subsample"        : trial.suggest_float("subsample",         0.5,   1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree",  0.4,  1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "subsample_freq"   : 1,
    }
    n_est  = trial.suggest_int("n_estimators", 100, 1500)
    dtrain = lgb.Dataset(X_tr.values, label=y_tr.values)
    dval   = lgb.Dataset(X_va.values, label=y_va.values, reference=dtrain)
    cbs    = [lgb.log_evaluation(-1)]
    try:
        cbs.append(lgb.early_stopping(50, verbose=False))
    except Exception:
        pass
    model = lgb.train(params, dtrain, num_boost_round=n_est,
                      valid_sets=[dval], valid_names=["val"], callbacks=cbs)
    return pinball_loss(y_va.values, model.predict(X_va.values), tau)


def optimizar_hiperparametros_lgbm(X_tr, y_tr, X_va, y_va, n_trials, fold_num):
    logger.info(f"    Optuna[LGBM] fold {fold_num} ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42 + fold_num))
    study.optimize(lambda t: _objetivo_optuna_lgbm(t, X_tr, y_tr, X_va, y_va),
                   n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def entrenar_quantiles_lgbm(X_tr, y_tr, best_params, quantiles):
    if not _LGBM_OK:
        raise ImportError("lightgbm no está instalado")
    n_est = int(best_params.get("n_estimators", 300))
    hp    = {k: v for k, v in best_params.items() if k != "n_estimators"}
    modelos = {}
    for tau in quantiles:
        params = {"objective": "quantile", "alpha": tau,
                  "verbosity": -1, "seed": 42, "subsample_freq": 1, **hp}
        dtrain = lgb.Dataset(X_tr.values, label=y_tr.values)
        modelos[tau] = lgb.train(params, dtrain, num_boost_round=n_est,
                                 callbacks=[lgb.log_evaluation(-1)])
    return modelos


def predecir_lgbm(modelos, X):
    preds_raw = {tau: m.predict(X.values) for tau, m in modelos.items()}
    taus   = sorted(preds_raw)
    matrix = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    return {t: matrix[:, i] for i, t in enumerate(taus)}


# -- XGBoost QT ----------------------------------------------------------------

def _objetivo_optuna_xgb_qt_tau(trial, tau, X_tr, y_tr, X_va, y_va, std_y):
    s = (std_y * S_FACTOR_FIJO if S_FIJO
         else trial.suggest_float("s", std_y * S_MIN_FACTOR, std_y * S_MAX_FACTOR, log=True))
    params = {
        "learning_rate"   : trial.suggest_float("learning_rate",   0.01,  0.3,  log=True),
        "max_depth"       : trial.suggest_int(  "max_depth",         3,    10),
        "min_child_weight": trial.suggest_int(  "min_child_weight", 10,   200),
        "colsample_bytree": trial.suggest_float("colsample_bytree",  0.4,  1.0),
        "subsample"       : trial.suggest_float("subsample",         0.5,  1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method"     : "hist",
        "nthread"         : _XGB_NTHREAD,   # limita threads por trial en paralelo
        "seed"            : 42,
    }
    n_est  = trial.suggest_int("n_estimators", 100, 1500)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    model  = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(tau, s, std_y),
        custom_metric=make_pinball_metric(tau),
        evals=[(dval, "val")],
        callbacks=[_PinballEarlyStopping(rounds=50)],
        verbose_eval=False,
    )
    return pinball_loss(y_va.values, model.predict(dval), tau)


def _worker_optuna_tau(args):
    """
    Función de módulo (picklable) para ProcessPoolExecutor.
    Recibe numpy arrays (no DataFrames) para minimizar el tamaño del pickle.
    Crea DMatrix UNA VEZ por proceso y la reutiliza en todos los trials de Optuna.
    """
    tau, X_tr_np, y_tr_np, X_va_np, y_va_np, col_names, std_y, n_trials, fold_num = args

    # DMatrix creado una vez -- se reutiliza en cada trial (no se recrea 110×)
    dtrain = xgb.DMatrix(X_tr_np, label=y_tr_np, feature_names=col_names)
    dval   = xgb.DMatrix(X_va_np, label=y_va_np, feature_names=col_names)

    def _obj(trial):
        s = (std_y * S_FACTOR_FIJO if S_FIJO
             else trial.suggest_float("s", std_y * S_MIN_FACTOR, std_y * S_MAX_FACTOR, log=True))
        params = {
            "learning_rate"   : trial.suggest_float("learning_rate",   0.01,  0.3,  log=True),
            "max_depth"       : trial.suggest_int(  "max_depth",         3,    10),
            "min_child_weight": trial.suggest_int(  "min_child_weight", 10,   200),
            "colsample_bytree": trial.suggest_float("colsample_bytree",  0.4,  1.0),
            "subsample"       : trial.suggest_float("subsample",         0.5,  1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tree_method"     : "hist",
            "nthread"         : _XGB_NTHREAD,
            "seed"            : 42,
        }
        n_est = trial.suggest_int("n_estimators", 100, 1500)
        model = xgb.train(
            params, dtrain, num_boost_round=n_est,
            obj=make_quantile_objective(tau, s, std_y),
            custom_metric=make_pinball_metric(tau),
            evals=[(dval, "val")],
            callbacks=[_PinballEarlyStopping(rounds=50)],
            verbose_eval=False,
        )
        return pinball_loss(y_va_np, model.predict(dval), tau)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold_num + int(tau * 100)),
    )
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    if S_FIJO:
        bp["s"] = std_y * S_FACTOR_FIJO
    s     = bp["s"]
    n_est = bp["n_estimators"]
    params = {k: v for k, v in bp.items() if k not in ("s", "n_estimators")}
    params.update({"tree_method": "hist", "seed": 42, "nthread": _XGB_NTHREAD})
    model = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(tau, s, std_y),
        verbose_eval=False,
    )
    return tau, model, bp, study.best_value


def _entrenar_fold_xgb_qt(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
    """
    Entrena un modelo por cuantil con Optuna independiente para cada uno.
    Los estudios corren en paralelo con ProcessPoolExecutor (un proceso por
    cuantil, GIL independiente) -> paralelismo real en múltiples núcleos.
    Cada XGBoost usa _XGB_NTHREAD threads -> sin over-subscription de CPU.
    Se pasan numpy arrays al worker para reducir overhead de pickle.
    """
    col_names = list(X_tr.columns)
    X_tr_np   = X_tr.values
    y_tr_np   = y_tr.values
    X_va_np   = X_va.values
    y_va_np   = y_va.values

    worker_args = [
        (tau, X_tr_np, y_tr_np, X_va_np, y_va_np, col_names,
         std_y, get_n_trials(tau), fold_num)
        for tau in QUANTILES
    ]

    modelos     = {}
    best_by_tau = {}
    with _ExecutorCls(max_workers=_N_QUANTILES_PARALLEL) as ex:
        futures = {ex.submit(_worker_optuna_tau, args): args[0] for args in worker_args}
        for fut in as_completed(futures):
            tau, model, bp, best_val = fut.result()
            modelos[tau]     = model
            best_by_tau[tau] = bp
            logger.info(f"    [xgb_qt] tau={tau:.2f} fold {fold_num}: "
                        f"pinball/VAL={best_val:.4f}  s={bp.get('s', std_y*S_FACTOR_FIJO):.3e}  "
                        f"n_est={bp['n_estimators']}")

    # Mean model -- reg:squarederror con los mejores HP de Q50
    bp_mean = best_by_tau.get(0.50, list(best_by_tau.values())[0])
    params_mean = {k: v for k, v in bp_mean.items() if k not in ("s", "n_estimators")}
    params_mean.update({"objective": "reg:squarederror",
                        "tree_method": "hist",
                        "nthread": _XGB_NTHREAD,
                        "seed": 42})
    dtrain_mean = xgb.DMatrix(X_tr_np, label=y_tr_np, feature_names=col_names)
    modelos["mean"] = xgb.train(params_mean, dtrain_mean,
                                num_boost_round=bp_mean["n_estimators"],
                                verbose_eval=False)
    logger.info(f"    [xgb_qt] mean fold {fold_num}: reg:squarederror "
                f"(n_est={bp_mean['n_estimators']})")

    return modelos, best_by_tau          # dict completo {tau: bp}


def _retrain_train_val_qt(X_tr, y_tr, X_va, y_va, best_by_tau, std_y, fold_num):
    """
    Reentrenamiento final sobre train+val con HPs fijos (ya optimizados por Optuna).

    Diseño deliberado sin early stopping: añadir early stopping requeriría un set
    de evaluación independiente; la única opción sería test, lo que introduciría
    leakage directo. Se usa el n_estimators encontrado por Optuna, que fue
    seleccionado con early stopping activo sobre val durante la búsqueda.

    s viene de best_by_tau[tau] -- calibrado por Optuna, no derivado de std_y.
    std_y se mantiene de y_train para consistencia con el objetivo original.
    preds_val NO usa este modelo; se evalúa con el modelo solo-train para
    mantener val como estimación out-of-sample honesta.
    """
    X_tv      = pd.concat([X_tr, X_va], ignore_index=True)
    y_tv      = pd.concat([y_tr, y_va], ignore_index=True)
    col_names = list(X_tv.columns)
    dtv       = xgb.DMatrix(X_tv.values, label=y_tv.values, feature_names=col_names)

    modelos_tv = {}
    for tau, bp in best_by_tau.items():
        s      = bp["s"]
        n_est  = bp["n_estimators"]
        params = {k: v for k, v in bp.items() if k not in ("s", "n_estimators")}
        params.update({"tree_method": "hist", "seed": 42, "nthread": _XGB_NTHREAD})
        modelos_tv[tau] = xgb.train(
            params, dtv, num_boost_round=n_est,
            obj=make_quantile_objective(tau, s, std_y),
            verbose_eval=False,
        )

    bp_mean = best_by_tau.get(0.50, list(best_by_tau.values())[0])
    params_mean = {k: v for k, v in bp_mean.items() if k not in ("s", "n_estimators")}
    params_mean.update({"objective": "reg:squarederror",
                        "tree_method": "hist", "nthread": _XGB_NTHREAD, "seed": 42})
    modelos_tv["mean"] = xgb.train(
        params_mean, dtv,
        num_boost_round=bp_mean["n_estimators"],
        verbose_eval=False,
    )
    logger.info(f"    [retrain_tv] fold {fold_num}: train+val "
                f"({len(y_tv)} obs, s de Optuna, std_y de train, sin early stopping)")
    return modelos_tv


# -- Dispatchers ---------------------------------------------------------------

def entrenar_fold(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
    if MODELO_CV == "xgb":
        bp = optimizar_hiperparametros(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num)
        return entrenar_quantiles(X_tr, y_tr, bp, QUANTILES, std_y), bp
    elif MODELO_CV == "lgbm":
        if not _LGBM_OK:
            raise ImportError("MODELO_CV='lgbm' requiere lightgbm instalado")
        bp = optimizar_hiperparametros_lgbm(X_tr, y_tr, X_va, y_va, n_trials, fold_num)
        return entrenar_quantiles_lgbm(X_tr, y_tr, bp.copy(), QUANTILES), bp
    elif MODELO_CV == "xgb_qt":
        return _entrenar_fold_xgb_qt(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num)
    raise ValueError(f"MODELO_CV desconocido: {MODELO_CV!r}")


def predecir_fold(modelos, X):
    if MODELO_CV == "lgbm":
        return predecir_lgbm(modelos, X)
    return predecir_y_corregir(modelos, X)


###############################################################################
# PARTE 5 -- Preparación de datos  (devuelve también fechas_t para TEST)
###############################################################################

def get_feature_cols(df):
    excluir = COLS_EXCLUIR | {"fecha_th"}
    return [c for c in df.columns
            if c not in excluir and df[c].dtype.kind in ("f", "i", "u", "b")]


def preparar_fold_data(df, fold, cols_feat):
    """
    Retorna:
      X_train, y_train, X_val, y_val, X_test, y_test,
      h_train, h_val, h_test, fechas_t_test

    fechas_t_test: array de pd.Timestamp con la fecha de origen de cada fila
    de TEST -- necesario para construir los fan charts por snapshot.
    """
    train_start = fold["train_start"]
    train_end   = fold["train_end"]
    val_start   = fold["val_start"]
    val_end     = fold["val_end"]      # límite real de VAL -- evita incluir el purge en X_val/y_val
    test_start  = fold["test_start"]
    test_end    = fold["test_end"]

    mask_train = (df["fecha_t"] >= train_start) & (df["fecha_t"] <= train_end)
    # df_fold_all incluye el purge period (val_end -> test_start) para que GARCH y FFD
    # operen sobre una serie continua sin saltos. El purge se excluye de df_val abajo.
    mask_fold_val = (df["fecha_t"] >= val_start) & (df["fecha_t"] < test_start)
    mask_test  = (df["fecha_t"] >= test_start)  & (df["fecha_t"] <= test_end)

    df_fold_all = df[mask_train | mask_fold_val | mask_test].copy()

    garch_cols = [c for c in ["garch_vol", "garch_vol_tc", "garch_vol_embi"]
                  if c in df_fold_all.columns]
    if garch_cols:
        df_fold_all = reemplazar_garch_fold(df_fold_all, train_end)

    frac_cols = [c for c in df_fold_all.columns if c.endswith("_frac")]
    if frac_cols:
        df_fold_all = reemplazar_ffd_fold(df_fold_all, train_end)

    df_train = df_fold_all[df_fold_all["fecha_t"] <= train_end]

    # Burn-in: excluir primeros BURN_IN_DIAS_HAB donde MA22 aún no maduró
    if BURN_IN_DIAS_HAB > 0:
        burn_cutoff = fold.get("burn_cutoff",
                               train_start + pd.offsets.BusinessDay(BURN_IN_DIAS_HAB))
        df_train = df_train[df_train["fecha_t"] >= burn_cutoff]
    # df_val usa val_end como límite superior -- excluye el purge period de X_val/y_val
    # para que Optuna no vea targets cuyo fecha_th cae dentro del período TEST.
    df_val   = df_fold_all[(df_fold_all["fecha_t"] >= val_start) &
                           (df_fold_all["fecha_t"] <= val_end)]
    df_test  = df_fold_all[df_fold_all["fecha_t"] >= test_start]

    medianas_fold = df_train[cols_feat].median()
    df_train = df_train.copy(); df_val = df_val.copy(); df_test = df_test.copy()
    df_train[cols_feat] = df_train[cols_feat].fillna(medianas_fold)
    df_val[cols_feat]   = df_val[cols_feat].fillna(medianas_fold)
    df_test[cols_feat]  = df_test[cols_feat].fillna(medianas_fold)

    mt  = df_train["target"].notna()
    mv  = df_val["target"].notna()
    mte = df_test["target"].notna()

    X_train        = df_train.loc[mt,  cols_feat].copy()
    y_train        = df_train.loc[mt,  "target"].copy()
    h_train        = df_train.loc[mt,  "h"].values

    X_val          = df_val.loc[mv,    cols_feat].copy()
    y_val          = df_val.loc[mv,    "target"].copy()
    h_val          = df_val.loc[mv,    "h"].values

    X_test         = df_test.loc[mte,  cols_feat].copy()
    y_test         = df_test.loc[mte,  "target"].copy()
    h_test         = df_test.loc[mte,  "h"].values
    fechas_t_test  = pd.to_datetime(df_test.loc[mte, "fecha_t"].values)

    # Convertir columnas pandas ExtensionArray (Int8, boolean, etc.) a float64
    # para compatibilidad con XGBoost DMatrix (no acepta dtype object/nullable)
    for _df in [X_train, X_val, X_test]:
        ext_cols = [c for c in _df.columns
                    if pd.api.types.is_extension_array_dtype(_df[c])]
        if ext_cols:
            _df[ext_cols] = _df[ext_cols].astype("float64")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            h_train, h_val, h_test, fechas_t_test)


###############################################################################
# PARTE 6 -- Métricas
###############################################################################

def calcular_metricas_fold(preds, y_true, fold, periodo="test"):
    row = {
        "fold"            : fold["fold"],
        "train_start"     : fold["train_start"].date(),
        "train_end"       : fold["train_end"].date(),
        "n_train_fechas"  : fold["n_train_fechas"],
        "val_start"       : fold["val_start"].date(),
        "val_end"         : fold["val_end"].date(),
        "n_val_fechas"    : fold["n_val_fechas"],
        "test_start"      : fold["test_start"].date(),
        "test_end"        : fold["test_end"].date(),
        "n_test_fechas"   : fold["n_test_fechas"],
        f"n_obs_{periodo}": int(len(y_true)),
        "periodo_metricas": periodo,
        "expanding"       : EXPANDING,
    }
    q_preds = {t: v for t, v in preds.items() if t != "mean"}
    for tau in sorted(q_preds.keys()):
        row[f"pinball_q{int(tau*100):02d}"] = round(
            pinball_loss(y_true, q_preds[tau], tau), 2)
    if {0.05, 0.95}.issubset(q_preds):
        row["coverage_90"] = round(coverage(y_true, q_preds[0.05], q_preds[0.95]), 4)
        row["winkler_90"]  = round(winkler_score(y_true, q_preds[0.05], q_preds[0.95]), 2)
    row["crps_approx"] = round(crps_approx(y_true, q_preds), 2)
    return row


def calcular_metricas_por_h(preds, y_true, h_arr, fold_num):
    rows = []
    for h in np.unique(h_arr):
        mask = h_arr == h
        if mask.sum() < 5:
            continue
        row = {"fold": fold_num, "h": int(h)}
        q_preds = {t: v for t, v in preds.items() if t != "mean"}
        for tau in sorted(q_preds.keys()):
            row[f"pinball_q{int(tau*100):02d}"] = round(
                pinball_loss(y_true[mask], q_preds[tau][mask], tau), 2)
        if {0.05, 0.95}.issubset(q_preds):
            row["coverage_90"] = round(
                coverage(y_true[mask], q_preds[0.05][mask], q_preds[0.95][mask]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


###############################################################################
# PARTE 7 -- Visualización
###############################################################################

def graficar_metricas_wfcv(df_test_m, banco):
    if df_test_m.empty:
        return
    folds = df_test_m["fold"].values
    modo  = "EXPANDING" if EXPANDING else "ROLLING"

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Walk-forward CV v3 [{modo}] -- {banco}  [métricas TEST out-of-sample]\n"
        f"TRAIN {VENTANA_TRAIN_AÑOS}yr{'(min)' if EXPANDING else ''} / "
        f"VAL {VENTANA_VAL_AÑOS}yr (Optuna) / TEST {VENTANA_TEST_AÑOS}yr / "
        f"paso {PASO_AÑOS}yr / purge {PURGE_DIAS_HAB}dh / burn-in {BURN_IN_DIAS_HAB}dh",
        fontweight="bold", fontsize=10,
    )
    metricas_config = [
        ("pinball_q50", "Pinball Q50 -- TEST OOS",   "steelblue",  None),
        ("coverage_90", "Coverage 90% -- TEST OOS",  "seagreen",   0.90),
        ("winkler_90",  "Winkler score -- TEST",     "darkorange", None),
        ("crps_approx", "CRPS aprox. -- TEST",       "crimson",    None),
    ]
    for ax, (col, titulo, color, hline) in zip(axes, metricas_config):
        if col not in df_test_m.columns:
            ax.set_visible(False); continue
        vals = df_test_m[col].values.astype(float)
        ax.plot(folds, vals, "o-", color=color, lw=2, ms=7, zorder=4)
        if len(folds) >= 3:
            z    = np.polyfit(folds, vals, 1)
            xfit = np.linspace(folds[0], folds[-1], 50)
            ax.plot(xfit, np.polyval(z, xfit), "--", color=color, alpha=0.5, lw=1.2,
                    label=f"tendencia: {z[0]:+.2e}/fold")
            ax.legend(fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="grey", lw=1.2, ls="--", alpha=0.6,
                       label=f"nominal={hline:.0%}")
            ax.legend(fontsize=8)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Fold", fontsize=9)
        ax.set_xticks(folds)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _, _c=col: f"{x:.1%}" if _c == "coverage_90" else f"{x:,.0f}"))
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v3_metricas_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico métricas TEST: {nombre}")


def graficar_comparacion_val_test(df_val_m, df_test_m, banco):
    if df_val_m.empty or df_test_m.empty:
        return
    if "coverage_90" not in df_val_m.columns:
        return
    folds = df_test_m["fold"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"VAL (Optuna, sesgo) vs TEST (OOS real) -- {banco} "
        f"[{'EXPANDING' if EXPANDING else 'ROLLING'}]",
        fontweight="bold", fontsize=11,
    )
    for ax, col, titulo in [
        (axes[0], "coverage_90", "Coverage 90%"),
        (axes[1], "pinball_q50", "Pinball Q50"),
    ]:
        if col not in df_val_m.columns or col not in df_test_m.columns:
            continue
        v_val  = df_val_m[col].values.astype(float)
        v_test = df_test_m[col].values.astype(float)
        ax.plot(folds, v_val,  "s--", color="steelblue", lw=1.5, ms=7, label="VAL (Optuna)")
        ax.plot(folds, v_test, "o-",  color="crimson",   lw=2,   ms=7, label="TEST (OOS)")
        if col == "coverage_90":
            ax.axhline(0.90, color="grey", lw=1.2, ls=":", alpha=0.7, label="nominal 90%")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
        diff = float(np.mean(v_val - v_test))
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Fold  (sesgo promedio VAL-TEST: {diff:+.4f})", fontsize=9)
        ax.set_xticks(folds)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v3_val_vs_test_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Comparación VAL vs TEST: {nombre}")


def graficar_fanchart_acum_test_fold(
    preds_test: dict,
    y_test: np.ndarray,
    h_test: np.ndarray,
    fechas_t_test,
    fold: dict,
    banco: str,
    dir_out: Path | None = None,
):
    """
    Fan chart de flujo neto ACUMULADO (cumsum sobre h) para un fold TEST.
    Mismas fechas de origen que graficar_fanchart_test_fold; cada cuantil
    se acumula con np.cumsum -> banda de incertidumbre del total acumulado.
    """
    fechas_unicas = pd.DatetimeIndex(sorted(set(fechas_t_test)))
    if len(fechas_unicas) == 0:
        return

    test_start = fold["test_start"]
    n_snap     = FANCHART_N_SNAPSHOTS
    meses_paso = 12.0 / n_snap

    origenes = []
    for i in range(n_snap):
        target  = test_start + pd.DateOffset(months=int(round(i * meses_paso)))
        diffs   = np.abs((fechas_unicas - target).total_seconds())
        nearest = fechas_unicas[np.argmin(diffs)]
        if nearest not in origenes:
            origenes.append(nearest)

    if not origenes:
        return

    ncols = 2
    nrows = int(np.ceil(len(origenes) / ncols))
    modo  = "EXPANDING" if EXPANDING else "ROLLING"

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5), sharey=False)
    axes_flat = np.array(axes).flatten()

    fig.suptitle(
        f"Fan chart ACUMULADO TEST OOS -- Fold {fold['fold']} -- {banco} [{modo}]\n"
        f"TEST: {fold['test_start'].date()} -> {fold['test_end'].date()}  |  "
        f"TRAIN hasta: {fold['train_end'].date()}",
        fontweight="bold", fontsize=11,
    )

    for ax, t0 in zip(axes_flat, origenes):
        mask = np.array([pd.Timestamp(f) == t0 for f in fechas_t_test])
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        h_s = h_test[mask]
        y_s = y_test[mask]
        p_s = {tau: arr[mask] for tau, arr in preds_test.items()}

        order = np.argsort(h_s)
        h_s   = h_s[order]
        y_s   = y_s[order]
        p_s   = {tau: arr[order] for tau, arr in p_s.items()}

        # Acumular con cumsum
        y_cum = np.cumsum(y_s)
        p_cum = {tau: np.cumsum(arr) for tau, arr in p_s.items()}

        if {0.01, 0.99}.issubset(p_cum):
            ax.fill_between(h_s, p_cum[0.01] / 1e6, p_cum[0.99] / 1e6,
                            alpha=0.12, color="steelblue", label="Q01-Q99 (CV)")
        if {0.05, 0.95}.issubset(p_cum):
            ax.fill_between(h_s, p_cum[0.05] / 1e6, p_cum[0.95] / 1e6,
                            alpha=0.28, color="steelblue", label="Q05-Q95 (CV)")
        if 0.50 in p_cum:
            lw_q50 = 1.4 if "mean" in p_cum else 1.8
            ls_q50 = "--" if "mean" in p_cum else "-"
            ax.plot(h_s, p_cum[0.50] / 1e6, color="steelblue", lw=lw_q50,
                    ls=ls_q50, zorder=3, label="Q50 (CV)")
        if "mean" in p_cum:
            ax.plot(h_s, p_cum["mean"] / 1e6, color="navy", lw=2.2,
                    zorder=4, label="Media (CV)")

        q_lo_cum   = p_cum.get(0.05, np.full_like(y_cum, -np.inf))
        q_hi_cum   = p_cum.get(0.95, np.full_like(y_cum,  np.inf))
        dentro_cum = (y_cum >= q_lo_cum) & (y_cum <= q_hi_cum)

        ax.plot(h_s, y_cum / 1e6, color="dimgray", lw=1.0, ls="--",
                zorder=4, alpha=0.75, label="Realizado acum.")
        ax.scatter(h_s[dentro_cum],  y_cum[dentro_cum]  / 1e6, color="seagreen", s=20,
                   zorder=5, label="Dentro Q05-Q95")
        ax.scatter(h_s[~dentro_cum], y_cum[~dentro_cum] / 1e6, color="crimson",  s=20,
                   zorder=5, label="Fuera Q05-Q95")

        cov_snap = float(dentro_cum.mean())
        ax.set_title(
            f"Origen: {t0.strftime('%Y-%m-%d')}\nCoverage 90% acum.: {cov_snap:.1%}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=8)
        ax.set_ylabel("Flujo acumulado D-R (MM USD)", fontsize=8)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3, ls="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, alpha=0.25)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(handles, labels, fontsize=7, loc="best")
    for ax in axes_flat[len(origenes):]:
        ax.set_visible(False)

    plt.tight_layout()
    _dir = dir_out if dir_out is not None else DIR_FANCHARTS
    nombre = _dir / f"fanchart_acum_test_fold{fold['fold']:02d}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Fan chart ACUMULADO TEST fold {fold['fold']}: {nombre.name}")


def graficar_fanchart_acum_punto_test_fold(
    preds_test: dict,
    y_test: np.ndarray,
    h_test: np.ndarray,
    fechas_t_test,
    fold: dict,
    banco: str,
    dir_out: Path | None = None,
):
    """
    Fan chart acumulado mostrando SOLO realizado, media y mediana (sin bandas).
    Permite evaluar la calidad del punto central sin ruido visual de intervalos.
    """
    fechas_unicas = pd.DatetimeIndex(sorted(set(fechas_t_test)))
    if len(fechas_unicas) == 0:
        return

    test_start = fold["test_start"]
    n_snap     = FANCHART_N_SNAPSHOTS
    meses_paso = 12.0 / n_snap

    origenes = []
    for i in range(n_snap):
        target  = test_start + pd.DateOffset(months=int(round(i * meses_paso)))
        diffs   = np.abs((fechas_unicas - target).total_seconds())
        nearest = fechas_unicas[np.argmin(diffs)]
        if nearest not in origenes:
            origenes.append(nearest)

    if not origenes:
        return

    ncols = 2
    nrows = int(np.ceil(len(origenes) / ncols))
    modo  = "EXPANDING" if EXPANDING else "ROLLING"

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5), sharey=False)
    axes_flat = np.array(axes).flatten()

    fig.suptitle(
        f"Acumulado TEST OOS -- Realizado vs Media vs Mediana -- Fold {fold['fold']} -- {banco} [{modo}]\n"
        f"TEST: {fold['test_start'].date()} -> {fold['test_end'].date()}  |  "
        f"TRAIN hasta: {fold['train_end'].date()}",
        fontweight="bold", fontsize=11,
    )

    for ax, t0 in zip(axes_flat, origenes):
        mask = np.array([pd.Timestamp(f) == t0 for f in fechas_t_test])
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        h_s = h_test[mask]
        y_s = y_test[mask]
        p_s = {tau: arr[mask] for tau, arr in preds_test.items()}

        order = np.argsort(h_s)
        h_s   = h_s[order]
        y_s   = y_s[order]
        p_s   = {tau: arr[order] for tau, arr in p_s.items()}

        y_cum = np.cumsum(y_s)
        p_cum = {tau: np.cumsum(arr) for tau, arr in p_s.items()}

        ax.plot(h_s, y_cum / 1e6, color="dimgray", lw=2.0, ls="--",
                zorder=4, label="Realizado acum.")
        if 0.50 in p_cum:
            ax.plot(h_s, p_cum[0.50] / 1e6, color="steelblue", lw=2.0,
                    ls="--", zorder=3, label="Mediana Q50 (CV)")
        if "mean" in p_cum:
            ax.plot(h_s, p_cum["mean"] / 1e6, color="crimson", lw=2.0,
                    zorder=4, label="Media (CV)")

        ax.axhline(0, color="black", lw=0.5, alpha=0.3, ls="--")
        ax.set_title(f"Origen: {t0.strftime('%Y-%m-%d')}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=8)
        ax.set_ylabel("Flujo acumulado D-R (MM USD)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(origenes):]:
        ax.set_visible(False)

    plt.tight_layout()
    _dir = dir_out if dir_out is not None else DIR_FANCHARTS
    nombre = _dir / f"fanchart_acum_punto_test_fold{fold['fold']:02d}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Fan chart ACUMULADO PUNTO TEST fold {fold['fold']}: {nombre.name}")


def graficar_fanchart_acum_punto_q05_test_fold(
    preds_test: dict,
    y_test: np.ndarray,
    h_test: np.ndarray,
    fechas_t_test,
    fold: dict,
    banco: str,
    dir_out: Path | None = None,
):
    """
    Fan chart acumulado con Realizado, Mediana, Media y Q05-acumulado.
    El Q05-acumulado (suma de Q05 diarios) representa el escenario desfavorable
    acumulado paso a paso, útil para comparar con la media/mediana.
    """
    fechas_unicas = pd.DatetimeIndex(sorted(set(fechas_t_test)))
    if len(fechas_unicas) == 0:
        return

    test_start = fold["test_start"]
    n_snap     = FANCHART_N_SNAPSHOTS
    meses_paso = 12.0 / n_snap

    origenes = []
    for i in range(n_snap):
        target  = test_start + pd.DateOffset(months=int(round(i * meses_paso)))
        diffs   = np.abs((fechas_unicas - target).total_seconds())
        nearest = fechas_unicas[np.argmin(diffs)]
        if nearest not in origenes:
            origenes.append(nearest)

    if not origenes:
        return

    ncols = 2
    nrows = int(np.ceil(len(origenes) / ncols))
    modo  = "EXPANDING" if EXPANDING else "ROLLING"

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5), sharey=False)
    axes_flat = np.array(axes).flatten()

    fig.suptitle(
        f"Acumulado TEST OOS -- Realizado / Media / Mediana / Q05-acum -- Fold {fold['fold']} -- {banco} [{modo}]\n"
        f"TEST: {fold['test_start'].date()} -> {fold['test_end'].date()}  |  "
        f"TRAIN hasta: {fold['train_end'].date()}",
        fontweight="bold", fontsize=11,
    )

    for ax, t0 in zip(axes_flat, origenes):
        mask = np.array([pd.Timestamp(f) == t0 for f in fechas_t_test])
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        h_s = h_test[mask]
        y_s = y_test[mask]
        p_s = {tau: arr[mask] for tau, arr in preds_test.items()}

        order = np.argsort(h_s)
        h_s   = h_s[order]
        y_s   = y_s[order]
        p_s   = {tau: arr[order] for tau, arr in p_s.items()}

        y_cum = np.cumsum(y_s)
        p_cum = {tau: np.cumsum(arr) for tau, arr in p_s.items()}

        # Escenario estresado: cumsum(Q50(h)) - (Q50(h) - Q05(h))
        # = cumsum(Q50(h-1)) + Q05(h)
        # Días 1..h-1 a su mediana; solo el día terminal h cae a su Q05.
        # La severidad no se acumula: interviene el P5 de un único día.
        med_stress_cum = None
        if 0.50 in p_s and 0.05 in p_s:
            med_stress_cum = np.cumsum(p_s[0.50]) - (p_s[0.50] - p_s[0.05])

        ax.plot(h_s, y_cum / 1e6, color="dimgray", lw=2.0, ls="--",
                zorder=5, label="Realizado acum.")
        if 0.50 in p_cum:
            ax.plot(h_s, p_cum[0.50] / 1e6, color="steelblue", lw=2.0,
                    ls="--", zorder=3, label="Mediana Q50 acum.")
        if "mean" in p_cum:
            ax.plot(h_s, p_cum["mean"] / 1e6, color="crimson", lw=2.0,
                    zorder=4, label="Media acum.")
        if med_stress_cum is not None:
            ax.plot(h_s, med_stress_cum / 1e6, color="darkorange", lw=1.8,
                    ls=":", zorder=2, label="Mediana acum. - riesgo P5 día h")

        ax.axhline(0, color="black", lw=0.5, alpha=0.3, ls="--")
        ax.set_title(f"Origen: {t0.strftime('%Y-%m-%d')}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=8)
        ax.set_ylabel("Flujo acumulado D-R (MM USD)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(origenes):]:
        ax.set_visible(False)

    plt.tight_layout()
    _dir = dir_out if dir_out is not None else DIR_FANCHARTS
    nombre = _dir / f"fanchart_acum_puntq05_test_fold{fold['fold']:02d}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Fan chart ACUMULADO PUNTO+Q05 TEST fold {fold['fold']}: {nombre.name}")


def graficar_cobertura_por_h(df_por_h, banco, sufijo="test"):
    if df_por_h.empty or "coverage_90" not in df_por_h.columns:
        return
    pivot = df_por_h.pivot(index="fold", columns="h", values="coverage_90")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.25), 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0.70, vmax=1.0, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Coverage 90%")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Fold {i}" for i in pivot.index], fontsize=8)
    ax.set_xlabel("Horizonte h (días hábiles)", fontsize=9)
    ax.set_title(
        f"Coverage 90% por fold y horizonte -- {banco} [{sufijo.upper()} OOS] "
        f"[{'EXPANDING' if EXPANDING else 'ROLLING'}]\n"
        "(verde=bien calibrado ~90%, rojo=sub/sobre-cobertura)",
        fontsize=10,
    )
    xticks_pos = range(0, len(pivot.columns), 10)
    ax.set_xticks([p for p in xticks_pos if p < len(pivot.columns)])
    ax.set_xticklabels([str(pivot.columns[p]) for p in xticks_pos
                        if p < len(pivot.columns)], fontsize=8)
    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v3_cobertura_{sufijo}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Heatmap cobertura ({sufijo.upper()}): {nombre}")


def graficar_hiperparametros_wfcv(df_test_m, banco):
    if df_test_m.empty:
        return
    HP_CONFIG = [
        ("s_optimo",         "s (smoothing arctan)",  "steelblue",   "log"),
        ("learning_rate",    "learning rate",          "darkorange",  "log"),
        ("max_depth",        "max depth",              "seagreen",    "linear"),
        ("n_estimators",     "n_estimators",           "crimson",     "linear"),
        ("min_child_weight", "min child weight",       "purple",      "linear"),
        ("subsample",        "subsample",              "saddlebrown", "linear"),
        ("colsample_bytree", "colsample bytree",       "teal",        "linear"),
        ("reg_alpha",        "reg_alpha (L1)",         "goldenrod",   "log"),
        ("reg_lambda",       "reg_lambda (L2)",        "slategrey",   "log"),
    ]
    hp_presentes = [(col, lbl, clr, sc) for col, lbl, clr, sc in HP_CONFIG
                    if col in df_test_m.columns]
    if not hp_presentes:
        return
    n     = len(hp_presentes)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    folds = df_test_m["fold"].values
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    axes_flat = np.array(axes).flatten()
    fig.suptitle(
        f"Estabilidad HP -- Walk-forward CV v3 [{' EXPANDING' if EXPANDING else 'ROLLING'}] -- {banco}\n"
        f"TRAIN {VENTANA_TRAIN_AÑOS}yr / VAL {VENTANA_VAL_AÑOS}yr / TEST {VENTANA_TEST_AÑOS}yr  "
        f"({len(folds)} folds)",
        fontweight="bold", fontsize=11,
    )
    for ax, (col, label, color, scale) in zip(axes_flat, hp_presentes):
        vals    = df_test_m[col].values.astype(float)
        mu, sig = vals.mean(), vals.std()
        ax.plot(folds, vals, "o-", color=color, lw=2, ms=7, zorder=4, label=label)
        ax.axhline(mu, color=color, lw=1.0, ls="--", alpha=0.5, label=f"media={mu:.4g}")
        if sig > 0:
            ax.axhspan(mu - sig, mu + sig, alpha=0.08, color=color,
                       label=f"±1sigma ({sig:.4g})")
        if len(folds) >= 3:
            z    = np.polyfit(folds, vals if scale == "linear"
                              else np.log(np.maximum(vals, 1e-12)), 1)
            xfit = np.linspace(folds[0], folds[-1], 50)
            yfit = np.polyval(z, xfit)
            if scale == "log":
                yfit = np.exp(yfit)
            ax.plot(xfit, yfit, "-", color="black", lw=1.0, alpha=0.4, label="tendencia")
        if scale == "log":
            ax.set_yscale("log")
        cv = sig / abs(mu) if abs(mu) > 1e-9 else 0.0
        ax.set_title(f"{label}  (CV={cv:.1%})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Fold", fontsize=8)
        ax.set_xticks(folds)
        ax.legend(fontsize=7, framealpha=0.85, loc="best")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    nombre = DIR_PLOTS / f"wfcv_v3_hiperparametros_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico hiperparámetros: {nombre}")


def _extraer_importancias(modelos, cols_feat):
    """
    Devuelve {feature: gain_promedio_entre_cuantiles} para XGBoost o LightGBM.
    Los features sin importancia en algún cuantil reciben gain=0.
    Excluye el modelo 'mean' (reg:squarederror) para ser consistente con
    _diag_gain_promedio: ambas funciones promedian solo sobre cuantiles.
    """
    acum = {f: 0.0 for f in cols_feat}
    n    = 0
    for tau, model in modelos.items():
        if tau == "mean":
            continue
        if MODELO_CV == "lgbm" and _LGBM_OK:
            names  = model.feature_name()
            gains  = model.feature_importance(importance_type="gain")
            imp    = dict(zip(names, gains.astype(float)))
        else:
            # xgb.Booster: solo devuelve features con importancia > 0
            imp = model.get_score(importance_type="gain")

        for feat in cols_feat:
            acum[feat] = acum[feat] + float(imp.get(feat, 0.0))
        n += 1

    if n > 0:
        acum = {f: v / n for f, v in acum.items()}
    return acum


def graficar_importancia_por_fold(
    importancias_folds: list,
    cols_feat: list,
    banco: str,
):
    """
    Genera dos gráficos a partir de la lista de importancias por fold:

    1. Heatmap (Features × Folds) -- ganancia normalizada por fold.
       Verde intenso = feature dominante en ese fold; blanco/amarillo = marginal.
       Útil para detectar features consistentes vs. régimen-dependientes.

    2. Rank-stability (Top-10) -- muestra cómo cambia el ranking de las features
       más importantes entre folds.  Línea plana = feature robusto;
       línea con saltos grandes = feature régimen-dependiente.

    También guarda un CSV con las ganancias brutas por fold.
    """
    if not importancias_folds:
        return

    # -- Pivot: filas = feature, columnas = fold -------------------------------
    registros = []
    for item in importancias_folds:
        fold_id = item["fold"]
        imp     = item["importancias"]
        total   = sum(imp.values()) or 1.0
        for feat, gain in imp.items():
            registros.append({
                "fold"     : fold_id,
                "feature"  : feat,
                "gain_norm": gain / total,
                "gain_raw" : gain,
            })

    df_imp = pd.DataFrame(registros)
    if df_imp.empty:
        return

    pivot_norm = df_imp.pivot_table(
        index="feature", columns="fold", values="gain_norm", aggfunc="mean"
    ).fillna(0.0)
    pivot_raw  = df_imp.pivot_table(
        index="feature", columns="fold", values="gain_raw",  aggfunc="mean"
    ).fillna(0.0)

    # Ordenar por importancia media descendente
    pivot_norm["_mean"] = pivot_norm.mean(axis=1)
    pivot_norm = pivot_norm.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    pivot_raw  = pivot_raw.loc[pivot_norm.index]  # mismo orden

    TOP_N  = min(25, len(pivot_norm))
    modo   = "EXPANDING" if EXPANDING else "ROLLING"
    folds  = sorted(pivot_norm.columns.tolist())

    # -- Gráfico 1: Heatmap ----------------------------------------------------
    top_feats_h = pivot_norm.iloc[:TOP_N]
    fig_h, ax_h = plt.subplots(
        figsize=(max(8, len(folds) * 1.2), max(6, TOP_N * 0.45))
    )

    im = ax_h.imshow(
        top_feats_h.values, aspect="auto", cmap="YlOrRd",
        vmin=0.0, vmax=max(top_feats_h.values.max() * 1.05, 1e-9),
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax_h, label="Ganancia normalizada por fold")

    ax_h.set_yticks(range(TOP_N))
    ax_h.set_yticklabels(top_feats_h.index.tolist(), fontsize=8)
    ax_h.set_xticks(range(len(folds)))
    ax_h.set_xticklabels([f"F{c}" for c in folds], fontsize=9)
    ax_h.set_xlabel("Fold", fontsize=10)
    ax_h.set_ylabel("Feature", fontsize=10)
    ax_h.set_title(
        f"Importancia de features por fold -- {banco} [{modo}]\n"
        f"Top {TOP_N} features . ganancia XGBoost normalizada por fold "
        f"(TRAIN {VENTANA_TRAIN_AÑOS}yr / VAL {VENTANA_VAL_AÑOS}yr / TEST {VENTANA_TEST_AÑOS}yr)",
        fontweight="bold", fontsize=11,
    )

    # Anotar valores en celdas cuando la tabla es pequeña
    if TOP_N <= 20 and len(folds) <= 12:
        for i in range(TOP_N):
            for j in range(len(folds)):
                val = top_feats_h.values[i, j]
                ax_h.text(
                    j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if val > top_feats_h.values.max() * 0.6 else "black",
                )

    plt.tight_layout()
    nombre_h = DIR_PLOTS / f"wfcv_v3_importancia_heatmap_{banco}.png"
    plt.savefig(nombre_h, dpi=150, bbox_inches="tight")
    plt.close(fig_h)
    logger.info(f"  Heatmap importancia: {nombre_h.name}")

    # -- Gráfico 2: Rank-stability ---------------------------------------------
    TOP_RANK   = min(10, len(pivot_norm))
    top_feats_r = pivot_norm.iloc[:TOP_RANK].index.tolist()

    # Rango en cada fold (1 = mayor ganancia) sobre TODOS los features
    rank_data = {feat: [] for feat in top_feats_r}
    for fold_id in folds:
        col_vals = pivot_norm[fold_id]
        ranked   = col_vals.rank(ascending=False, method="min")
        for feat in top_feats_r:
            rank_data[feat].append(int(ranked.get(feat, len(pivot_norm) + 1)))

    fig_r, ax_r = plt.subplots(figsize=(max(7, len(folds) * 1.2), 5))
    cmap_r = plt.cm.tab10
    for i, feat in enumerate(top_feats_r):
        ax_r.plot(
            folds, rank_data[feat], "o-",
            lw=2, ms=7, color=cmap_r(i / max(TOP_RANK, 1)),
            label=feat, alpha=0.85,
        )

    ax_r.invert_yaxis()   # rango 1 arriba
    ax_r.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_r.set_xticks(folds)
    ax_r.set_xticklabels([f"Fold {f}" for f in folds], fontsize=9)
    ax_r.set_xlabel("Fold", fontsize=10)
    ax_r.set_ylabel("Ranking (1 = mayor ganancia)", fontsize=10)
    ax_r.set_title(
        f"Estabilidad de ranking -- Top {TOP_RANK} features -- {banco} [{modo}]\n"
        f"Línea plana = feature robusto . saltos grandes = régimen-dependiente",
        fontweight="bold", fontsize=10,
    )
    ax_r.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.85)
    ax_r.grid(True, alpha=0.25)
    plt.tight_layout()
    nombre_r = DIR_PLOTS / f"wfcv_v3_importancia_ranking_{banco}.png"
    plt.savefig(nombre_r, dpi=150, bbox_inches="tight")
    plt.close(fig_r)
    logger.info(f"  Ranking importancia: {nombre_r.name}")

    # -- CSV importancias brutas ------------------------------------------------
    df_csv = pivot_raw.copy().reset_index()
    df_csv.columns.name = None
    ruta_csv = DIR_MODO / f"wfcv_v3_importancias_{banco}.csv"
    df_csv.to_csv(ruta_csv, index=False)
    logger.info(f"  CSV importancias: {ruta_csv.name}")


def graficar_fanchart_test_fold(
    preds_test: dict,
    y_test: np.ndarray,
    h_test: np.ndarray,
    fechas_t_test,
    fold: dict,
    banco: str,
    preds_overlay: dict | None = None,   # step004 predictions para comparación
    dir_out: Path | None = None,         # carpeta de salida; None -> DIR_FANCHARTS
):
    """
    Fan chart TEST out-of-sample para un fold.

    Selecciona hasta FANCHART_N_SNAPSHOTS fechas de origen separadas ~3 meses
    dentro del período TEST. Para cada fecha muestra:
      . Bandas Q01-Q99 (muy transparente) y Q05-Q95 (media)
      . Línea Q50 (mediana predicha)
      . Puntos realizados: verde = dentro de Q05-Q95 / rojo = fuera

    Ayuda a detectar si el modelo sobre/sub-estima en horizontes cortos o largos,
    y en qué trimestres del período TEST falla la cobertura.
    """
    fechas_unicas = pd.DatetimeIndex(sorted(set(fechas_t_test)))
    if len(fechas_unicas) == 0:
        return

    # Seleccionar fechas de origen: una cada ~(12/N_SNAPSHOTS) meses del TEST
    test_start = fold["test_start"]
    n_snap     = FANCHART_N_SNAPSHOTS
    meses_paso = 12.0 / n_snap

    origenes = []
    for i in range(n_snap):
        target  = test_start + pd.DateOffset(months=int(round(i * meses_paso)))
        diffs   = np.abs((fechas_unicas - target).total_seconds())
        nearest = fechas_unicas[np.argmin(diffs)]
        if nearest not in origenes:
            origenes.append(nearest)

    if not origenes:
        return

    # Layout 2×2
    ncols = 2
    nrows = int(np.ceil(len(origenes) / ncols))
    modo  = "EXPANDING" if EXPANDING else "ROLLING"

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5), sharey=False)
    axes_flat = np.array(axes).flatten()

    s4_tag = "  |  [naranja] naranja = Step004 (GARCH global, lookahead)" if preds_overlay is not None else ""
    fig.suptitle(
        f"Fan chart TEST OOS -- Fold {fold['fold']} -- {banco} [{modo}]\n"
        f"TEST: {fold['test_start'].date()} -> {fold['test_end'].date()}  |  "
        f"TRAIN hasta: {fold['train_end'].date()}{s4_tag}",
        fontweight="bold", fontsize=11,
    )

    for ax, t0 in zip(axes_flat, origenes):
        mask = np.array([pd.Timestamp(f) == t0 for f in fechas_t_test])
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        h_s = h_test[mask]
        y_s = y_test[mask]
        p_s = {tau: arr[mask] for tau, arr in preds_test.items()}

        order = np.argsort(h_s)
        h_s   = h_s[order]
        y_s   = y_s[order]
        p_s   = {tau: arr[order] for tau, arr in p_s.items()}

        # Predicciones step004 overlay (misma máscara y orden)
        p_s4 = None
        if preds_overlay is not None:
            p_s4 = {tau: arr[mask][order] for tau, arr in preds_overlay.items()
                    if tau != "mean"}

        # Bandas de incertidumbre -- step005 (azul)
        if {0.01, 0.99}.issubset(p_s):
            ax.fill_between(h_s, p_s[0.01] / 1e6, p_s[0.99] / 1e6,
                            alpha=0.12, color="steelblue", label="Q01-Q99 (CV)")
        if {0.05, 0.95}.issubset(p_s):
            ax.fill_between(h_s, p_s[0.05] / 1e6, p_s[0.95] / 1e6,
                            alpha=0.28, color="steelblue", label="Q05-Q95 (CV)")
        if 0.50 in p_s:
            lw_q50 = 1.4 if "mean" in p_s else 1.8
            ls_q50 = "--" if "mean" in p_s else "-"
            ax.plot(h_s, p_s[0.50] / 1e6, color="steelblue", lw=lw_q50,
                    ls=ls_q50, zorder=3, label="Q50 (CV)")
        if "mean" in p_s:
            ax.plot(h_s, p_s["mean"] / 1e6, color="navy", lw=2.2,
                    zorder=4, label="Media (CV)")

        # Overlay step004 -- naranja discontinuo
        if p_s4 is not None:
            if {0.05, 0.95}.issubset(p_s4):
                ax.fill_between(h_s, p_s4[0.05] / 1e6, p_s4[0.95] / 1e6,
                                alpha=0.10, color="darkorange")
            if 0.50 in p_s4:
                ax.plot(h_s, p_s4[0.50] / 1e6, color="darkorange", lw=1.6,
                        ls="--", zorder=4, alpha=0.85, label="Q50 Step004")

        # Realizados: línea punteada + scatter coloreado por cobertura
        q_lo   = p_s.get(0.05, np.full_like(y_s, -np.inf))
        q_hi   = p_s.get(0.95, np.full_like(y_s,  np.inf))
        dentro = (y_s >= q_lo) & (y_s <= q_hi)

        ax.plot(h_s, y_s / 1e6, color="dimgray", lw=1.0, ls="--",
                zorder=4, alpha=0.75, label="Realizado")
        ax.scatter(h_s[dentro],  y_s[dentro]  / 1e6, color="seagreen", s=20,
                   zorder=5, label="Dentro Q05-Q95")
        ax.scatter(h_s[~dentro], y_s[~dentro] / 1e6, color="crimson",  s=20,
                   zorder=5, label="Fuera Q05-Q95")

        cov_snap = float(dentro.mean())
        ax.set_title(
            f"Origen: {t0.strftime('%Y-%m-%d')}\nCoverage 90%: {cov_snap:.1%}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=8)
        ax.set_ylabel("Flujo D-R (MM USD)", fontsize=8)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3, ls="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, alpha=0.25)

    # Leyenda en el primer panel; ocultar paneles sobrantes
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(handles, labels, fontsize=7, loc="best")
    for ax in axes_flat[len(origenes):]:
        ax.set_visible(False)

    plt.tight_layout()
    _dir = dir_out if dir_out is not None else DIR_FANCHARTS
    nombre = _dir / f"fanchart_test_fold{fold['fold']:02d}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Fan chart TEST fold {fold['fold']}: {nombre.name}")


###############################################################################
# PARTE 7-bis -- Diagnóstico de features (gain / block-perm / SHAP)
###############################################################################

def _diag_predict_un_modelo(model, X):
    if MODELO_CV == "lgbm" and _LGBM_OK:
        return model.predict(X.values)
    return model.predict(xgb.DMatrix(X))


def _diag_gain_promedio(modelos, cols_feat):
    """Gain promedio entre cuantiles (TRAIN, in-sample). Excluye 'mean'."""
    acum = {f: 0.0 for f in cols_feat}
    n = 0
    for tau, model in modelos.items():
        if tau == "mean":
            continue
        if MODELO_CV == "lgbm" and _LGBM_OK:
            imp = dict(zip(model.feature_name(),
                           model.feature_importance(importance_type="gain").astype(float)))
        else:
            imp = model.get_score(importance_type="gain")
        for f in cols_feat:
            acum[f] += float(imp.get(f, 0.0))
        n += 1
    if n:
        acum = {f: v / n for f, v in acum.items()}
    return pd.Series(acum)


def _consolidar_cicl_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida columnas sin/cos -> _cyc con norma euclidiana sqrt(I_sin²+I_cos²).
    Opera sobre DataFrame folds × features (una fila por fold).
    Reutiliza _CICL_PARES_PERM definido junto a los constants de diagnóstico.
    """
    if df.empty:
        return df
    grupos = {}
    for feat in df.columns:
        if feat in _CICL_PARES_PERM:
            grupos.setdefault(_CICL_PARES_PERM[feat], []).append(feat)
    if not grupos:
        return df
    feat_en_par = {f for cols in grupos.values() for f in cols}
    result = df[[c for c in df.columns if c not in feat_en_par]].copy()
    for cyc_name, cols in grupos.items():
        present = [c for c in cols if c in df.columns]
        if present:
            result[cyc_name] = np.sqrt(
                (df[present].fillna(0.0) ** 2).sum(axis=1))
    return result


def _diag_block_perm_un_cuantil(model, X, y, tau, block_size, n_repeats, rng):
    n = len(X)
    base = pinball_loss(y, _diag_predict_un_modelo(model, X), tau)
    block_starts = np.arange(0, n, block_size)

    # Identificar pares sin/cos presentes en X -> permutación simultánea
    grupos_cicl = {}              # nombre_cyc -> [col_sin, col_cos]
    for c in X.columns:
        if c in _CICL_PARES_PERM:
            grupos_cicl.setdefault(_CICL_PARES_PERM[c], []).append(c)
    feat_en_par = {c for cols in grupos_cicl.values() for c in cols}

    imp = {}

    # -- Features sueltas: permutación individual (comportamiento original) --
    for c in X.columns:
        if c in feat_en_par:
            continue
        col = X[c].values
        deltas = []
        for _ in range(n_repeats):
            perm_starts = rng.permutation(block_starts)
            new_col = np.concatenate([col[s:s + block_size] for s in perm_starts])[:n]
            Xp = X.copy(); Xp[c] = new_col
            deltas.append(pinball_loss(y, _diag_predict_un_modelo(model, Xp), tau) - base)
        imp[c] = float(np.mean(deltas))

    # -- Pares sin/cos: UN solo shuffle por repetición -> delta combinado real --
    # Se distribuye como D/sqrtn_comp en cada componente para que la norma euclidiana
    # en aux_comparar_features.py reconstruya exactamente D = sqrt(I_sin²+I_cos²).
    for cyc_name, cols in grupos_cicl.items():
        orig_vals = {c: X[c].values.copy() for c in cols}
        deltas = []
        for _ in range(n_repeats):
            perm_starts = rng.permutation(block_starts)   # mismo shuffle para todo el par
            Xp = X.copy()
            for c in cols:
                Xp[c] = np.concatenate(
                    [orig_vals[c][s:s + block_size] for s in perm_starts])[:n]
            deltas.append(pinball_loss(y, _diag_predict_un_modelo(model, Xp), tau) - base)
        d_comp = float(np.mean(deltas)) / float(np.sqrt(len(cols)))
        for c in cols:
            imp[c] = d_comp

    return pd.Series(imp)


def _diag_block_perm_promedio(modelos, X_val, y_val, cols_feat, fold_num):
    X = X_val.reset_index(drop=True)[cols_feat]
    y = np.asarray(y_val)
    if DIAG_PERM_MAX_SAMPLES is not None and len(X) > DIAG_PERM_MAX_SAMPLES:
        # Muestreo estratificado por h: evita sesgo hacia fechas tempranas o
        # h pequeños que produce el slice contiguo iloc[:N].
        if "h" in X.columns:
            n_h_vals = X["h"].nunique()
            per_h    = max(1, DIAG_PERM_MAX_SAMPLES // n_h_vals)
            rng_sub  = np.random.default_rng(99 + fold_num)
            idx_list = []
            for _, grp in X.groupby("h"):
                take = min(len(grp), per_h)
                idx_list.append(rng_sub.choice(grp.index.values, take, replace=False))
            idx = np.concatenate(idx_list)
            idx.sort()   # mantener orden (fecha_t, h) para que los bloques sean contiguos
            X = X.loc[idx].reset_index(drop=True)
            y = y[idx]
        else:
            X = X.iloc[:DIAG_PERM_MAX_SAMPLES].copy()
            y = y[:DIAG_PERM_MAX_SAMPLES]
    bs  = max(2, min(DIAG_BLOCK_SIZE, len(X) // 3))
    rng = np.random.default_rng(42 + fold_num)
    acum = pd.Series(0.0, index=cols_feat)
    n = 0
    for tau in QUANTILES:
        model = modelos.get(tau)
        if model is None:
            continue
        s = _diag_block_perm_un_cuantil(model, X, y, tau, bs, DIAG_N_REPEATS, rng)
        acum = acum.add(s.reindex(cols_feat).fillna(0.0), fill_value=0.0)
        n += 1
    if n:
        acum /= n
    return acum


def _diag_shap_promedio(modelos, X_val, cols_feat, fold_num):
    X = X_val.reset_index(drop=True)[cols_feat]

    # Submuestreo estratificado por h -- mismo criterio que block-perm para
    # que ambas métricas evalúen la misma distribución de horizontes.
    if len(X) > DIAG_SHAP_MAX_SAMPLES:
        if "h" in X.columns:
            n_h_vals = X["h"].nunique()
            per_h    = max(1, DIAG_SHAP_MAX_SAMPLES // n_h_vals)
            rng_sub  = np.random.default_rng(99 + fold_num)
            idx_list = []
            for _, grp in X.groupby("h"):
                take = min(len(grp), per_h)
                idx_list.append(rng_sub.choice(grp.index.values, take, replace=False))
            idx = np.sort(np.concatenate(idx_list))
            X = X.loc[idx].reset_index(drop=True)
        else:
            X = X.sample(DIAG_SHAP_MAX_SAMPLES, random_state=42 + fold_num)

    # h es variable estructural del problema (siempre conocida) -- excluirla
    # del ranking SHAP evita que desplace features reales del top_n.
    cols_shap = [c for c in cols_feat if c != "h"]
    dmat = xgb.DMatrix(X.values, feature_names=cols_feat)
    acum = pd.Series(0.0, index=cols_shap)
    n = 0
    for tau in QUANTILES:
        model = modelos.get(tau)
        if model is None:
            continue
        try:
            # pred_contribs=True devuelve SHAP values nativos de XGBoost
            # shape: (n_samples, n_features + 1); última col = bias -> se descarta
            sv = model.predict(dmat, pred_contribs=True)[:, :-1]
            # sv tiene columnas en el mismo orden que cols_feat
            h_idx   = list(cols_feat).index("h") if "h" in cols_feat else None
            sv_cols = [c for c in cols_feat if c != "h"]
            sv_filt = np.delete(sv, h_idx, axis=1) if h_idx is not None else sv
            s = pd.Series(np.abs(sv_filt).mean(axis=0), index=sv_cols)
            acum = acum.add(s.fillna(0.0), fill_value=0.0)
            n += 1
        except Exception as e:
            logger.warning(f"      [diag] SHAP tau={tau} falló: {e}")
    if n == 0:
        return pd.Series(np.nan, index=cols_shap)
    return acum / n


def diagnosticar_fold(modelos, X_val, y_val, cols_feat, fold_num):
    """Tres señales (gain train, perm val, shap val) promediadas sobre cuantiles."""
    logger.info(f"    [diag] Fold {fold_num}: gain(train) + block-perm(val) + shap(val)")
    gain = _diag_gain_promedio(modelos, cols_feat)
    perm = _diag_block_perm_promedio(modelos, X_val, y_val, cols_feat, fold_num)
    shp  = _diag_shap_promedio(modelos, X_val, cols_feat, fold_num)
    return {"fold": fold_num, "gain_train": gain, "perm_val": perm, "shap_val": shp}


def _diag_matriz(diag_por_fold, senal, cols_feat):
    filas = {d["fold"]: d[senal].reindex(cols_feat) for d in diag_por_fold}
    m = pd.DataFrame(filas).T
    m.index.name = "fold"
    return m


def _plot_gain_perm_shap(matrices, orden_top, folds, banco):
    """Gráfico horizontal de 3 barras por feature: gain / perm / SHAP."""
    gain_m = matrices["gain_train"]
    perm_m = matrices["perm_val"]
    shap_m = matrices["shap_val"]
    if perm_m.dropna(how="all").empty:
        return

    gain_mean = gain_m.mean()
    perm_mean = perm_m.mean()
    shap_mean = shap_m.mean()

    g = (gain_mean / (gain_mean.max() + 1e-12)).reindex(orden_top)
    p = (perm_mean.clip(lower=0) /
         (perm_mean.clip(lower=0).max() + 1e-12)).reindex(orden_top)
    s = (shap_mean.clip(lower=0) /
         (shap_mean.clip(lower=0).max() + 1e-12)).reindex(orden_top).fillna(0.0)

    fig, ax = plt.subplots(figsize=(9, max(5, len(orden_top) * 0.45)))
    ypos = np.arange(len(orden_top))
    h = 0.26
    ax.barh(ypos + h,  g.values, height=h, color="#4878CF",
            label="gain (TRAIN, in-sample)")
    ax.barh(ypos,      p.values, height=h, color="#6ACC65",
            label="perm (VAL, OOS)")
    ax.barh(ypos - h,  s.values, height=h, color="#D65F5F",
            label="SHAP |mean| (VAL, OOS)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(orden_top, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia normalizada", fontsize=10)
    nota = "" if _SHAP_OK else "  [!] SHAP no disponible"
    ax.set_title(f"gain / perm / SHAP -- {banco}{nota}\n"
                 f"convergencia gain~=perm~=SHAP -> feature genuinamente útil",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    ruta = DIR_PLOTS / f"diag_gain_perm_shap_{banco}.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    [diag] gain+perm+SHAP: {ruta.name}")


def consolidar_diagnostico(diag_por_fold, cols_feat, banco, top_n=25):
    """CSVs + heatmaps + ranking + gain vs perm + gain/perm/SHAP. Solo reporta, no depura."""
    if not diag_por_fold:
        logger.warning("    [diag] Sin folds para consolidar")
        return
    senales = {
        "gain_train": "Gain (TRAIN, in-sample)",
        "perm_val":   "Block-Permutation (VAL, OOS)",
        "shap_val":   "SHAP (VAL, OOS)",
    }
    matrices = {s: _diag_matriz(diag_por_fold, s, cols_feat) for s in senales}

    # Consolidar pares sin/cos -> _cyc (norma euclidiana) en las 3 señales
    matrices = {s: _consolidar_cicl_df(m) for s, m in matrices.items()}
    cols_feat_cyc = matrices["gain_train"].columns.tolist()

    for s, m in matrices.items():
        ruta = DIR_MODO / f"diag_{s}_{banco}.csv"
        m.to_csv(ruta)
        logger.info(f"    [diag] CSV {s}: {ruta.name}")

    estab_rows = []
    for s, m in matrices.items():
        if m.dropna(how="all").empty:
            continue
        ranks = m.rank(axis=1, ascending=False, method="min")
        for feat in cols_feat_cyc:
            estab_rows.append({
                "senal"    : s,
                "feature"  : feat,
                "imp_mean" : float(m[feat].mean(skipna=True)),
                "rank_mean": float(ranks[feat].mean(skipna=True)),
                "rank_std" : float(ranks[feat].std(skipna=True)),
            })
    pd.DataFrame(estab_rows).to_csv(
        DIR_MODO / f"diag_estabilidad_{banco}.csv", index=False)
    logger.info(f"    [diag] CSV estabilidad: diag_estabilidad_{banco}.csv")

    perm_m = matrices["perm_val"]
    orden  = (perm_m.mean().sort_values(ascending=False).index.tolist()
              if not perm_m.dropna(how="all").empty
              else matrices["gain_train"].mean().sort_values(ascending=False).index.tolist())
    orden_top = orden[:top_n]
    folds     = sorted(matrices["gain_train"].index.tolist())

    # Heatmaps por señal
    for s, m in matrices.items():
        if m.dropna(how="all").empty:
            continue
        sub      = m[orden_top].T
        col_max  = sub.max(axis=0).replace(0, np.nan)
        sub_norm = (sub / col_max).fillna(0.0)
        fig, ax  = plt.subplots(figsize=(max(7, len(folds) * 1.1),
                                         max(6, len(orden_top) * 0.42)))
        im = ax.imshow(sub_norm.values, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Importancia normalizada por fold")
        ax.set_yticks(range(len(orden_top)))
        ax.set_yticklabels(orden_top, fontsize=8)
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels([f"F{f}" for f in folds], fontsize=9)
        ax.set_xlabel("Fold", fontsize=10)
        ax.set_title(f"Diagnóstico VAL -- {senales[s]} -- {banco}\n"
                     f"Top {len(orden_top)} features . normalizado por fold "
                     f"(solo diagnóstico, NO depuración)", fontweight="bold", fontsize=10)
        plt.tight_layout()
        ruta = DIR_PLOTS / f"diag_heatmap_{s}_{banco}.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"    [diag] Heatmap {s}: {ruta.name}")

    # Ranking de estabilidad (block-perm VAL)
    if not perm_m.dropna(how="all").empty:
        top_r = orden[:min(10, len(orden))]
        ranks = perm_m.rank(axis=1, ascending=False, method="min")
        fig, ax = plt.subplots(figsize=(max(7, len(folds) * 1.1), 5))
        cmap = plt.cm.tab10
        for i, feat in enumerate(top_r):
            ax.plot(folds, ranks[feat].values, "o-", lw=2, ms=7,
                    color=cmap(i / 10), label=feat, alpha=0.85)
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=9)
        ax.set_ylabel("Ranking perm(VAL)  (1 = más importante)", fontsize=10)
        ax.set_title(f"Estabilidad de ranking -- Block-Permutation VAL -- {banco}\n"
                     f"Línea plana = feature robusto . saltos = régimen-dependiente",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.85)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        ruta = DIR_PLOTS / f"diag_ranking_perm_val_{banco}.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"    [diag] Ranking perm(VAL): {ruta.name}")

    # Gráfico 2 barras: gain(train) vs perm(val)
    gain_m = matrices["gain_train"]
    if not perm_m.dropna(how="all").empty:
        gain_mean = gain_m.mean()
        perm_mean = perm_m.mean()
        g = (gain_mean / (gain_mean.max() + 1e-12)).reindex(orden_top)
        p = (perm_mean.clip(lower=0) /
             (perm_mean.clip(lower=0).max() + 1e-12)).reindex(orden_top)
        fig, ax = plt.subplots(figsize=(8, max(5, len(orden_top) * 0.4)))
        ypos = np.arange(len(orden_top))
        ax.barh(ypos - 0.2, g.values, height=0.4, color="slategrey",
                label="gain (TRAIN, in-sample)")
        ax.barh(ypos + 0.2, p.values, height=0.4, color="seagreen",
                label="perm (VAL, OOS)")
        ax.set_yticks(ypos)
        ax.set_yticklabels(orden_top, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importancia normalizada", fontsize=10)
        ax.set_title(f"gain(TRAIN) vs perm(VAL) -- {banco}\n"
                     f"gain alto + perm bajo -> sospecha de sobreajuste",
                     fontweight="bold", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        ruta = DIR_PLOTS / f"diag_gain_vs_perm_{banco}.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"    [diag] gain(train) vs perm(val): {ruta.name}")

    # Gráfico 3 barras: gain / perm / SHAP
    _plot_gain_perm_shap(matrices, orden_top, folds, banco)

    logger.info(f"    [diag] Consolidación completa -- {len(folds)} folds, "
                f"{len(cols_feat)} features")


###############################################################################
###############################################################################
# Overlay sobreencaje -- lee ajuste de step007 y aplica a las predicciones
###############################################################################

def _aplicar_overlay_sobreencaje(
    preds: dict,
    h_arr: np.ndarray,
    fechas_t: pd.DatetimeIndex,
    df_hist: "pd.DataFrame | None" = None,
) -> dict:
    """
    Lee peor_total diario desde la tab 'Ajuste_diario' del Excel de step007
    y aplica un factor multiplicativo uniforme sobre toda la ventana de proyeccion.

    Logica T+N (OVERLAY_CONOCIMIENTO_ANTICIPADO = N):
    - Los flujos de los proximos N dias habiles son conocidos con antelacion.
    - Si h_cierre <= N: el cierre ya es completamente conocido -> se usa el
      SIGUIENTE cierre trimestral (que aun tiene incertidumbre real).
    - Para el cierre elegido: los dias conocidos (realizados + N dias adelante)
      consumen el tope peor_total (netting). Solo el resto es incierto.
    - Q[TAU]_acum se calcula sobre la porcion incierta: h en [N+1 .. h_cierre].

    peor_restante = max(0, peor_total - retiros_conocidos_en_ventana)
    f = peor_restante / |sum(Q[TAU] para h en [max(N+1,h_inicio)..h_cierre])|
    """
    if not OVERLAY_SOBREENCAJE:
        return preds

    try:
        df_aj = pd.read_excel(
            RUTA_AJUSTE_OVERLAY, sheet_name="Ajuste_diario",
            index_col=0, parse_dates=True,
        )
        df_aj.index = pd.DatetimeIndex(df_aj.index)
    except Exception as e:
        logger.warning(f"[OVERLAY] No se pudo leer Ajuste_diario: {e} -- sin ajuste")
        return preds

    if "peor_total" not in df_aj.columns:
        logger.warning("[OVERLAY] Columna peor_total no encontrada en Ajuste_diario -- sin ajuste")
        return preds

    preds_adj = {tau: arr.copy() for tau, arr in preds.items()}
    N = OVERLAY_CONOCIMIENTO_ANTICIPADO

    for fecha_t in sorted(set(fechas_t)):
        disponibles = df_aj.index[df_aj.index <= fecha_t]
        if len(disponibles) == 0:
            continue
        peor_total = float(df_aj.loc[disponibles[-1], "peor_total"])
        if peor_total <= 0:
            continue

        bh = pd.bdate_range(
            start=fecha_t + pd.offsets.BDay(1),
            periods=N + OVERLAY_VENTANA_DH + 75,
        )
        bh_list = list(bh)
        df_bh = pd.DataFrame({"fecha": bh, "mes": bh.month, "anio": bh.year, "trim": bh.quarter})
        cierres_proy = (
            df_bh[df_bh["mes"].isin([3, 6, 9, 12])]
            .groupby(["anio", "trim"])["fecha"].max()
            .sort_values().tolist()
        )
        if not cierres_proy:
            continue

        # Primer cierre con horizonte incierto > N (si h_cierre <= N, ya es conocido)
        fecha_cierre = None
        h_cierre = None
        for _cand in cierres_proy:
            try:
                _h = bh_list.index(_cand) + 1
            except ValueError:
                _diffs = [abs((d - _cand).days) for d in bh_list]
                _h = _diffs.index(min(_diffs)) + 1
            if _h > N:
                fecha_cierre = _cand
                h_cierre = _h
                break

        if fecha_cierre is None:
            continue

        mask_origen = (fechas_t == fecha_t)
        h_inicio = max(1, h_cierre - OVERLAY_VENTANA_DH + 1)

        # -- Netting: dias conocidos = realizados hasta fecha_t + N dias hábiles
        fecha_inicio_ventana_real = fecha_cierre - pd.offsets.BDay(OVERLAY_VENTANA_DH - 1)
        fecha_conocida_hasta = min(
            fecha_t + pd.offsets.BDay(N),
            fecha_cierre,
        )
        retiro_conocido = 0.0
        if df_hist is not None and fecha_inicio_ventana_real <= fecha_conocida_hasta:
            known_days = pd.bdate_range(
                start=fecha_inicio_ventana_real, end=fecha_conocida_hasta
            )
            for _d in known_days:
                _d_prev = _d - pd.offsets.BDay(1)
                _rows = df_hist[(df_hist["fecha_t"] == _d_prev) & (df_hist["h"] == 1)]
                if not _rows.empty:
                    _flow = float(_rows["target"].values[0])
                    if _flow < 0:
                        retiro_conocido += abs(_flow)

        peor_restante = max(0.0, peor_total - retiro_conocido)
        if peor_restante < 1e-6:
            logger.info(
                f"[OVERLAY] {fecha_t.date()} | cap consumido "
                f"(conocidos={retiro_conocido:,.0f} >= peor={peor_total:,.0f}) -- sin ajuste"
            )
            continue

        # Q[TAU] sobre la porcion incierta: h > N (proximos N dias ya son conocidos)
        h_inicio_incierto = max(h_inicio, N + 1)
        mask_ventana = mask_origen & (h_arr >= h_inicio_incierto) & (h_arr <= h_cierre)
        if not mask_ventana.any():
            h_t_unc = h_arr[mask_origen & (h_arr > N)]
            if len(h_t_unc) == 0:
                continue
            h_fb = h_t_unc[np.abs(h_t_unc - h_cierre).argmin()]
            mask_ventana = mask_origen & (h_arr == h_fb)

        q01_acum = float(preds[OVERLAY_TAU_REFERENCIA][mask_ventana].sum())
        if q01_acum >= 0 or abs(q01_acum) < 1e-6:
            continue

        f = peor_restante / abs(q01_acum)
        if f <= 1.0:
            continue

        for tau in preds_adj:
            preds_adj[tau][mask_origen] *= f

        logger.info(
            f"[OVERLAY] {fecha_t.date()} | cierre: {fecha_cierre.date()} "
            f"h=[{h_inicio_incierto},{h_cierre}] | "
            f"Q{int(OVERLAY_TAU_REFERENCIA*100):02d}_acum={q01_acum:+.0f} | "
            f"peor={peor_total:,.0f} | conocidos={retiro_conocido:,.0f} | "
            f"restante={peor_restante:,.0f} | f={f:.3f}"
        )

    return preds_adj


# PARTE 8 -- Pipeline principal
###############################################################################

def _cargar_metadata_disco(banco: str) -> dict:
    """
    Encuentra y carga el metadata JSON más reciente para el banco y MODELO_CV
    en DIR_MODELOS. Lanza FileNotFoundError si no existe.
    """
    sfx = "lgbm_wfcv_v3" if MODELO_CV == "lgbm" else f"{MODELO_CV}_wfcv_v3"
    candidatos = sorted(DIR_MODELOS.glob(f"metadata_{sfx}_{banco}_*.json"), reverse=True)
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró metadata para banco={banco} modelo={MODELO_CV} en {DIR_MODELOS}"
        )
    meta = json.loads(candidatos[0].read_text(encoding="utf-8"))
    logger.info(f"  [REPLOT] Metadata cargado: {candidatos[0].name}")
    return meta


def _cargar_modelos_fold_disco(fold_info: dict, banco: str) -> dict:
    """
    Carga desde disco los modelos de un fold específico.
    Usa fecha_hoy guardada en fold_info (del manifest original).
    """
    sfx      = "lgbm_wfcv_v3" if MODELO_CV == "lgbm" else f"{MODELO_CV}_wfcv_v3"
    ext      = ".txt" if MODELO_CV == "lgbm" else ".json"
    fold_num = fold_info["fold"]
    fecha    = fold_info["fecha_hoy"]

    modelos = {}
    for tau in QUANTILES:
        ruta = DIR_MODELOS / f"{sfx}_{banco}_fold{fold_num:02d}_q{int(tau*100):02d}_{fecha}{ext}"
        if not ruta.exists():
            raise FileNotFoundError(f"Modelo fold {fold_num} no encontrado: {ruta}")
        if MODELO_CV == "lgbm":
            import lightgbm as lgb_load
            modelos[tau] = lgb_load.Booster(model_file=str(ruta))
        else:
            b = xgb.Booster(); b.load_model(str(ruta)); modelos[tau] = b

    # Mean model (opcional -- solo existe si fue entrenado con la nueva versión)
    ruta_mean = DIR_MODELOS / f"{sfx}_{banco}_fold{fold_num:02d}_mean_{fecha}{ext}"
    if ruta_mean.exists():
        b = xgb.Booster(); b.load_model(str(ruta_mean)); modelos["mean"] = b

    has_mean = "mean" in modelos
    logger.info(f"  [REPLOT] Fold {fold_num} cargado desde disco"
                f"{'  (+ media)' if has_mean else ''}")
    return modelos


def _cargar_modelos_step004(banco: str) -> dict | None:
    """
    Carga los modelos del step004 (eval) para el banco indicado.
    Retorna dict {tau: Booster} o None si no se encuentran archivos.
    Soporta naming xgb_qt y xgb.
    """
    if not DIR_MODELOS_STEP004.exists():
        logger.warning(f"  [S4] Directorio step004 no encontrado: {DIR_MODELOS_STEP004}")
        return None

    sfx = _s4_prefijo
    # Buscar metadata para obtener fecha del modelo
    metas = sorted(DIR_MODELOS_STEP004.glob(f"metadata_{sfx}_{banco}_*.json"), reverse=True)
    if not metas:
        # fallback: sin metadata, buscar modelos directamente
        metas = sorted(DIR_MODELOS_STEP004.glob(f"metadata_{sfx}_wfcv_v3_{banco}_*.json"), reverse=True)
    if not metas:
        logger.warning(f"  [S4] No se encontró metadata step004 en {DIR_MODELOS_STEP004}")
        return None

    meta  = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha = metas[0].stem.split("_")[-1]
    quantiles_s4 = meta.get("quantiles", QUANTILES)

    modelos_s4 = {}
    for tau in quantiles_s4:
        ruta = DIR_MODELOS_STEP004 / f"{sfx}_{banco}_q{int(tau*100):02d}_{fecha}.json"
        if not ruta.exists():
            logger.warning(f"  [S4] Modelo no encontrado: {ruta.name}")
            return None
        b = xgb.Booster(); b.load_model(str(ruta))
        modelos_s4[tau] = b

    logger.info(f"  [S4] Modelos step004 cargados: {metas[0].name}  "
                f"({len(modelos_s4)} cuantiles)")
    return modelos_s4


def aplicar_mondrian_cqr(preds_val, y_val, h_val, preds_test, h_test):
    """
    Mondrian Conformalized Quantile Regression (CQR) por horizonte h.

    Scores calculados con preds_val (modelo solo-train, OOS honesto).
    Corrección aplicada a preds_test (modelo retrenado train+val).
    Sin leakage: el modelo que genera scores nunca vio VAL como entrenamiento;
    el modelo que predice TEST nunca vio TEST.

    Score de conformidad (Romano et al. 2019):
        score_i = max(CQR_TAU_LO_pred - y_i,  y_i - CQR_TAU_HI_pred)
    Cuantil de corrección por horizonte h:
        q_hat_h = quantile(scores_h, (1 - CQR_ALPHA)(1 + 1/n_h))
    Intervalo calibrado en TEST:
        [Q_lo_pred - q_hat_h,  Q_hi_pred + q_hat_h]
    """
    if CQR_TAU_LO not in preds_val or CQR_TAU_HI not in preds_val:
        logger.warning("    [CQR] Cuantiles CQR_TAU_LO/HI no encontrados en preds_val -- omitiendo")
        return preds_test

    lo_val = preds_val[CQR_TAU_LO]
    hi_val = preds_val[CQR_TAU_HI]
    y_arr  = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)

    q_hat_h = {}
    for h in np.unique(h_val):
        mask   = h_val == h
        scores = np.maximum(lo_val[mask] - y_arr[mask],
                            y_arr[mask]  - hi_val[mask])
        n_h    = len(scores)
        if n_h < 2:
            continue
        nivel        = (1 - CQR_ALPHA) * (1 + 1 / n_h)
        nivel        = min(nivel, 1.0)
        q_hat_h[int(h)] = float(np.quantile(scores, nivel))

    if not q_hat_h:
        logger.warning("    [CQR] Sin horizontes con suficientes datos -- omitiendo")
        return preds_test

    adj = np.array([q_hat_h.get(int(h), 0.0) for h in h_test])

    preds_cqr = dict(preds_test)
    if CQR_TAU_LO in preds_cqr:
        preds_cqr[CQR_TAU_LO] = preds_test[CQR_TAU_LO] - adj
    if CQR_TAU_HI in preds_cqr:
        preds_cqr[CQR_TAU_HI] = preds_test[CQR_TAU_HI] + adj

    _sample_h = sorted(q_hat_h)
    _log_vals = "  ".join(
        f"h={h}->{q_hat_h[h]:,.0f}" for h in _sample_h[::max(1, len(_sample_h)//4)]
    )
    logger.info(
        f"    [CQR] Mondrian q_hat por h (alpha={CQR_ALPHA}, "
        f"tau=[{CQR_TAU_LO},{CQR_TAU_HI}]): {_log_vals}"
    )
    return preds_cqr


def evaluar_banco(banco: str):
    modo = "EXPANDING" if EXPANDING else "ROLLING"
    logger.info(f"\n{'='*65}")
    logger.info(f"BANCO: {banco}  -- Walk-Forward CV v3  [{modo}]  [TEST OOS]")
    logger.info(f"{'='*65}")
    logger.info(
        f"  TRAIN {VENTANA_TRAIN_AÑOS}yr{'(min)' if EXPANDING else ''} | "
        f"VAL {VENTANA_VAL_AÑOS}yr (Optuna) | TEST {VENTANA_TEST_AÑOS}yr (métricas) | "
        f"paso {PASO_AÑOS}yr | purge {PURGE_DIAS_HAB}dh | burn-in {BURN_IN_DIAS_HAB}dh | "
        f"trials={'adaptivo' if ADAPTIVE_TRIALS else f'flat={TRIALS_FLAT}'}"
    )
    if ADAPTIVE_TRIALS:
        _tau_trials = {tau: get_n_trials(tau) for tau in QUANTILES}
        logger.info(f"  Trials por cuantil: " +
                    " | ".join(f"tau={t:.2f}->{n}" for t, n in _tau_trials.items()))

    t_inicio = time.time()

    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    if df.empty or df["target"].notna().sum() < 500:
        logger.warning(f"  [{banco}] Datos insuficientes -- omitiendo")
        return None

    cols_feat = get_feature_cols(df)
    fechas    = pd.DatetimeIndex(df["fecha_t"].unique())
    logger.info(f"  [{banco}] {len(df):,} filas | {len(cols_feat)} features | "
                f"rango: {fechas.min().date()} -> {fechas.max().date()}")

    # Cargar modelos step004 para comparación (opcional)
    modelos_s4 = _cargar_modelos_step004(banco) if COMPARAR_CON_STEP004 else None
    if COMPARAR_CON_STEP004 and modelos_s4 is None:
        logger.warning("  [S4] No se pudo cargar step004 -- comparación desactivada")

    folds = generar_folds(
        fechas_disponibles=fechas,
        ventana_train_años=VENTANA_TRAIN_AÑOS,
        ventana_val_años=VENTANA_VAL_AÑOS,
        ventana_test_años=VENTANA_TEST_AÑOS,
        paso_años=PASO_AÑOS,
        purge_dias_hab=PURGE_DIAS_HAB,
        purge_val_test=PURGE_VAL_TEST,
        expanding=EXPANDING,
    )
    if not folds:
        logger.error(f"  [{banco}] No se generaron folds")
        return None

    if N_MAX_FOLDS is not None and len(folds) > N_MAX_FOLDS:
        logger.info(f"  [{banco}] Limitando a {N_MAX_FOLDS} folds "
                    f"(de {len(folds)} disponibles) -- "
                    f"datos desde {folds[N_MAX_FOLDS]['test_start'].date()} quedan OOS")
        folds = folds[:N_MAX_FOLDS]

    # Folds manuales (p.ej. réplica exacta del split de step004)
    if FOLDS_MANUALES:
        n_previos = 0 if SOLO_FOLDS_MANUALES else len(folds)
        folds_man = resolver_folds_manuales(FOLDS_MANUALES, fechas, n_previos)
        if SOLO_FOLDS_MANUALES:
            folds = folds_man
            logger.info(f"  [{banco}] Modo SOLO_FOLDS_MANUALES -- "
                        f"{len(folds_man)} fold(s) manual(es) en lugar de los automáticos")
        else:
            folds = folds + folds_man
            logger.info(f"  [{banco}] +{len(folds_man)} fold(s) manual(es) añadido(s)")

    logger.info(f"  [{banco}] {len(folds)} folds generados:")
    for f in folds:
        n_train_yr = round(f["n_train_fechas"] / 252, 1)
        tag = " [MANUAL]" if f.get("_manual") else ""
        logger.info(
            f"    Fold {f['fold']:2d}{tag} | TRAIN {f['train_start'].date()} -> "
            f"{f['train_end'].date()} ({n_train_yr}yr, {f['n_train_fechas']}dh) | "
            f"VAL  {f['val_start'].date()} -> {f['val_end'].date()} | "
            f"TEST {f['test_start'].date()} -> {f['test_end'].date()}"
        )

    resultados_test   = []
    resultados_val    = []
    por_h_test        = []
    por_h_val         = []
    importancias_folds = []
    diag_por_fold      = []
    modelos_ultimo    = None
    params_ultimo     = None
    folds_manifest    = []   # registro de todos los folds para fan chart histórico
    fecha_hoy         = pd.Timestamp.today().strftime("%Y%m%d")

    # -- Modo regenerar plots: carga metadata del disco para obtener fecha_hoy ---
    _meta_disco = None
    if SOLO_REGENERAR_PLOTS:
        try:
            _meta_disco   = _cargar_metadata_disco(banco)
            folds_manifest = _meta_disco.get("folds_manifest", [])
            # Indexar por fold_num para acceso rápido
            _fm_idx = {fi["fold"]: fi for fi in folds_manifest}
            logger.info(f"  [REPLOT] {len(folds_manifest)} folds en manifest -- "
                        f"solo se regenerarán los fan charts")
        except FileNotFoundError as _e_meta:
            logger.error(f"  [REPLOT] {_e_meta}")
            logger.error("  [REPLOT] Ejecuta primero con SOLO_REGENERAR_PLOTS=False")
            return None

    all_preds_overlay = []   # acumula predicciones finales (con overlay) para exportar

    for fold in folds:
        t_fold = time.time()
        logger.info(f"\n  -- Fold {fold['fold']}/{len(folds)} ----------------------")

        try:
            (X_train, y_train,
             X_val,   y_val,
             X_test,  y_test,
             h_train, h_val, h_test,
             fechas_t_test) = preparar_fold_data(df, fold, cols_feat)
        except Exception as e:
            logger.warning(f"  Fold {fold['fold']}: error preparando datos -- {e}")
            continue

        if len(X_train) < 200 or len(X_val) < 20 or len(X_test) < 20:
            logger.warning(f"  Fold {fold['fold']}: datos insuficientes -- omitiendo")
            continue

        std_y = float(y_train.std())
        if std_y < 1.0:
            logger.warning(f"    Fold {fold['fold']}: std_y={std_y:.4f} anormalmente bajo -- "
                           f"forzado a 1.0 para evitar división por cero en objetivo GARCH")
            std_y = 1.0
        logger.info(f"    X_train={len(X_train):,} | X_val={len(X_val):,} | "
                    f"X_test={len(X_test):,} | std_y={std_y:,.0f}")

        if SOLO_REGENERAR_PLOTS:
            # -- Modo replot: carga modelos del disco, salta Optuna -----------
            fold_num  = fold["fold"]
            fold_info = _fm_idx.get(fold_num)
            if fold_info is None:
                logger.warning(f"  [REPLOT] Fold {fold_num} no está en el manifest -- omitiendo")
                continue
            try:
                modelos = _cargar_modelos_fold_disco(fold_info, banco)
            except FileNotFoundError as _e_load:
                logger.warning(f"  [REPLOT] {_e_load} -- omitiendo fold {fold_num}")
                continue
            best_params = {}
            modelos_final = modelos   # en modo replot no hay retrain
        else:
            # -- Modo normal: Optuna + entrenamiento -------------------------
            modelos, best_params = entrenar_fold(
                X_train, y_train, X_val, y_val, std_y,
                get_n_trials(0.5), fold["fold"]
            )

            # -- Retrain final sobre train+val (HPs fijos de Optuna) ---------
            # modelos_final se usa para preds_test; modelos (solo-train) para
            # preds_val y métricas de diagnóstico (val out-of-sample honesto).
            if MODELO_CV == "xgb_qt":
                modelos_final = _retrain_train_val_qt(
                    X_train, y_train, X_val, y_val,
                    best_params, std_y, fold["fold"]
                )
            else:
                X_tv = pd.concat([X_train, X_val], ignore_index=True)
                y_tv = pd.concat([y_train, y_val], ignore_index=True)
                modelos_final = entrenar_quantiles(X_tv, y_tv, best_params, QUANTILES, std_y)

            # Gain: refleja el modelo final desplegado
            try:
                imp = _extraer_importancias(modelos_final, cols_feat)
                importancias_folds.append({"fold": fold["fold"], "importancias": imp})
            except Exception as _e_imp:
                logger.warning(f"    Importancia fold {fold['fold']}: {_e_imp}")

            # Perm + SHAP: val es out-of-sample solo para modelos (solo-train)
            if DIAGNOSTICO_FEATURES:
                try:
                    diag_por_fold.append(
                        diagnosticar_fold(modelos, X_val, y_val, cols_feat, fold["fold"])
                    )
                except Exception as _e_diag:
                    logger.warning(f"    [diag] Fold {fold['fold']} falló: {_e_diag}")

        preds_test = predecir_fold(modelos_final, X_test)  # modelo train+val
        preds_val  = predecir_fold(modelos,       X_val)   # modelo solo-train -> val honesto

        # -- Calibración post-hoc (por horizonte h) ---------------------------
        # Shift calculado por separado para cada h en VAL y aplicado solo a
        # filas del mismo h en TEST/VAL. Evita que residuos de horizontes
        # largos (mayor varianza) contaminen el percentil de horizontes cortos.
        # CALIBRACION_MAX_SHIFT_FACTOR limita el shift a ±factor×std_y para
        # evitar que un sesgo grande del modelo distorsione el eje Y.
        if CALIBRACION_POSTHOC and 0.50 in preds_val:
            _cap = (CALIBRACION_MAX_SHIFT_FACTOR * std_y
                    if CALIBRACION_MAX_SHIFT_FACTOR is not None else np.inf)
            _shifts_h = {}
            for _h in np.unique(h_val):
                _mask_h = h_val == _h
                _res_h  = y_val.values[_mask_h] - preds_val[0.50][_mask_h]
                if len(_res_h) >= 2:
                    _s = float(np.percentile(_res_h, CALIBRACION_PERCENTIL))
                    _shifts_h[int(_h)] = float(np.clip(_s, -_cap, _cap))
            _adj_test = np.array([_shifts_h.get(int(h), 0.0) for h in h_test])
            _adj_val  = np.array([_shifts_h.get(int(h), 0.0) for h in h_val])
            preds_test = {tau: arr + _adj_test for tau, arr in preds_test.items()}
            preds_val  = {tau: arr + _adj_val  for tau, arr in preds_val.items()}
            logger.info(
                f"    [CALIBRACION] P{CALIBRACION_PERCENTIL} por-h "
                f"(cap=±{_cap:,.0f} = ±{CALIBRACION_MAX_SHIFT_FACTOR}×std_y): "
                f"h=1->{_shifts_h.get(1,0):,.0f}  "
                f"h=38->{_shifts_h.get(38,0):,.0f}  "
                f"h=75->{_shifts_h.get(75,0):,.0f}"
            )

        # -- Mondrian CQR -----------------------------------------------------
        # Scores en preds_val (solo-train, OOS) -> q_hat_h -> ajuste en preds_test
        # (retrain train+val). Sin leakage: cada modelo predice datos que nunca vio.
        if CALIBRACION_CQR:
            preds_test = aplicar_mondrian_cqr(
                preds_val, y_val, h_val, preds_test, h_test
            )

        # -- Overlay sobreencaje -----------------------------------------------
        # Lee peor_total desde Excel (step007) y aplica factor multiplicativo
        # uniforme sobre todo el horizonte, por fecha_t unica.
        if OVERLAY_SOBREENCAJE:
            preds_test = _aplicar_overlay_sobreencaje(preds_test, h_test, fechas_t_test, df)

        # -- Guardar predicciones finales (con todos los ajustes aplicados) ----
        _preds_df = pd.DataFrame({
            "banco"  : banco,
            "fold"   : fold["fold"],
            "fecha_t": pd.DatetimeIndex(fechas_t_test),
            "h"      : h_test,
            "target" : y_test.values,
        })
        for _tau, _arr in preds_test.items():
            _col = "mean" if _tau == "mean" else f"q{int(_tau * 100):02d}"
            _preds_df[_col] = _arr
        all_preds_overlay.append(_preds_df)

        if not SOLO_REGENERAR_PLOTS:
            row_test = calcular_metricas_fold(preds_test, y_test.values, fold, "test")
            row_val  = calcular_metricas_fold(preds_val,  y_val.values,  fold, "val")
            row_test["tiempo_min"] = round((time.time() - t_fold) / 60, 2)
            row_val["modelo_cv"]   = MODELO_CV
            row_test["modelo_cv"]  = MODELO_CV

            def _hp(d, key, default=0):
                return d.get(key, default)

            # xgb_qt devuelve best_params como {tau: bp}; extraer Q50 para logging
            _bp_log = (best_params.get(0.50, list(best_params.values())[0])
                       if MODELO_CV == "xgb_qt" and best_params
                       else best_params)

            for row in (row_test, row_val):
                if MODELO_CV == "lgbm":
                    row.update({
                        "s_optimo"        : 0.0,
                        "learning_rate"   : round(_hp(_bp_log, "learning_rate"), 4),
                        "max_depth"       : int(_hp(_bp_log, "num_leaves")),
                        "n_estimators"    : int(_hp(_bp_log, "n_estimators")),
                        "min_child_weight": int(_hp(_bp_log, "min_child_samples")),
                        "subsample"       : round(_hp(_bp_log, "subsample"), 3),
                        "colsample_bytree": round(_hp(_bp_log, "colsample_bytree"), 3),
                        "reg_alpha"       : round(_hp(_bp_log, "reg_alpha"), 5),
                        "reg_lambda"      : round(_hp(_bp_log, "reg_lambda"), 5),
                    })
                else:
                    row.update({
                        "s_optimo"        : round(_hp(_bp_log, "s"), 4),
                        "learning_rate"   : round(_hp(_bp_log, "learning_rate"), 4),
                        "max_depth"       : int(_hp(_bp_log, "max_depth")),
                        "n_estimators"    : int(_hp(_bp_log, "n_estimators")),
                        "min_child_weight": int(_hp(_bp_log, "min_child_weight")),
                        "subsample"       : round(_hp(_bp_log, "subsample"), 3),
                        "colsample_bytree": round(_hp(_bp_log, "colsample_bytree"), 3),
                        "reg_alpha"       : round(_hp(_bp_log, "reg_alpha"), 5),
                        "reg_lambda"      : round(_hp(_bp_log, "reg_lambda"), 5),
                    })

            resultados_test.append(row_test)
            resultados_val.append(row_val)
            por_h_test.append(calcular_metricas_por_h(preds_test, y_test.values, h_test, fold["fold"]))
            por_h_val.append(calcular_metricas_por_h(preds_val,  y_val.values,  h_val,  fold["fold"]))

            cov_t = row_test.get("coverage_90", float("nan"))
            cov_v = row_val.get("coverage_90",  float("nan"))
            logger.info(
                f"    TEST: coverage={cov_t:.1%}  pinball_Q50={row_test.get('pinball_q50','?'):,.0f}  "
                f"winkler={row_test.get('winkler_90','?'):,.0f}  "
                f"| VAL coverage={cov_v:.1%} (sesgo={cov_v-cov_t:+.1%})  "
                f"({row_test['tiempo_min']} min)"
            )

        # Fan charts TEST -- folds manuales van a carpeta separada
        preds_s4  = predecir_y_corregir(modelos_s4, X_test) if modelos_s4 is not None else None
        _fanchart_dir = DIR_FANCHARTS_MANUALES if fold.get("_manual") else None
        graficar_fanchart_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test, fold, banco,
            preds_overlay=preds_s4,
            dir_out=_fanchart_dir,
        )
        graficar_fanchart_acum_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test, fold, banco,
            dir_out=_fanchart_dir,
        )
        graficar_fanchart_acum_punto_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test, fold, banco,
            dir_out=_fanchart_dir,
        )
        graficar_fanchart_acum_punto_q05_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test, fold, banco,
            dir_out=_fanchart_dir,
        )

        modelos_ultimo = modelos_final   # modelo retrenado train+val = el que se despliega
        params_ultimo  = best_params

        if not SOLO_REGENERAR_PLOTS:
            # -- Guardar modelo del fold + manifest ---------------------------
            sfx = "lgbm_wfcv_v3" if MODELO_CV == "lgbm" else f"{MODELO_CV}_wfcv_v3"
            ext = ".txt" if MODELO_CV == "lgbm" else ".json"
            fold_num = fold["fold"]

            garch_fold = {}
            try:
                garch_fold = _extraer_garch_params_fold(df, fold["train_end"])
            except Exception as _eg:
                logger.warning(f"  Fold {fold_num}: no se pudo extraer GARCH -- {_eg}")

            if GUARDAR_MODELOS_TODOS_FOLDS:
                for tau, model in modelos_final.items():
                    if tau == "mean":
                        ruta_f = (DIR_MODELOS /
                                  f"{sfx}_{banco}_fold{fold_num:02d}_mean_{fecha_hoy}{ext}")
                    else:
                        ruta_f = (DIR_MODELOS /
                                  f"{sfx}_{banco}_fold{fold_num:02d}_q{int(tau*100):02d}_{fecha_hoy}{ext}")
                    model.save_model(str(ruta_f))
                logger.info(f"    Modelos fold {fold_num} guardados en {DIR_MODELOS.name}/")

            folds_manifest.append({
                "fold"       : fold_num,
                "train_start": str(fold["train_start"].date()),
                "train_end"  : str(fold["train_end"].date()),
                "test_start" : str(fold["test_start"].date()),
                "test_end"   : str(fold["test_end"].date()),
                "fecha_hoy"  : fecha_hoy,
                "garch"      : garch_fold,
            })

        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()

    # -- Exportar predicciones finales a parquet (input para orquestador/video) --
    if all_preds_overlay:
        df_preds_all = pd.concat(all_preds_overlay, ignore_index=True)
        ruta_preds = DIR_MODO / f"preds_overlay_{banco}_{fecha_hoy}.parquet"
        df_preds_all.to_parquet(ruta_preds, index=False)
        logger.info(f"  [{banco}] Predicciones finales guardadas: {ruta_preds.name}")

    if SOLO_REGENERAR_PLOTS:
        logger.info(f"\n  [REPLOT] Fan charts regenerados para {banco}. "
                    f"Métricas y metadata no actualizados.")
        return None

    if not resultados_test:
        logger.error(f"  [{banco}] Ningún fold completado")
        return None

    df_test_m  = pd.DataFrame(resultados_test)
    df_val_m   = pd.DataFrame(resultados_val)
    df_por_h_t = pd.concat(por_h_test, ignore_index=True) if por_h_test else pd.DataFrame()
    df_por_h_v = pd.concat(por_h_val,  ignore_index=True) if por_h_val  else pd.DataFrame()
    # fecha_hoy ya fue definido antes del loop para consistencia en nombres de archivo

    def _save(df, nombre):
        ruta = DIR_MODO / nombre
        df.to_csv(ruta, index=False)
        logger.info(f"    {nombre}")

    logger.info(f"  [{banco}] Archivos en: {DIR_MODO}")
    _save(df_test_m,  f"wfcv_v3_metricas_TEST_{banco}_{fecha_hoy}.csv")
    _save(df_val_m,   f"wfcv_v3_metricas_VAL_{banco}_{fecha_hoy}.csv")
    if not df_por_h_t.empty:
        _save(df_por_h_t, f"wfcv_v3_por_h_TEST_{banco}_{fecha_hoy}.csv")
    if not df_por_h_v.empty:
        _save(df_por_h_v, f"wfcv_v3_por_h_VAL_{banco}_{fecha_hoy}.csv")

    cols_hp    = ["fold", "train_start", "train_end", "modelo_cv", "expanding",
                  "s_optimo", "learning_rate", "max_depth", "n_estimators",
                  "min_child_weight", "subsample", "colsample_bytree",
                  "reg_alpha", "reg_lambda"]
    cols_hp_ok = [c for c in cols_hp if c in df_test_m.columns]
    _save(df_test_m[cols_hp_ok], f"wfcv_v3_hiperparametros_{banco}_{fecha_hoy}.csv")

    tag = f"{banco}_{MODELO_CV}_{'exp' if EXPANDING else 'rol'}"
    graficar_metricas_wfcv(df_test_m, tag)
    graficar_comparacion_val_test(df_val_m, df_test_m, tag)
    graficar_cobertura_por_h(df_por_h_t, tag, "test")
    graficar_cobertura_por_h(df_por_h_v, tag, "val")
    graficar_hiperparametros_wfcv(df_test_m, tag)
    graficar_importancia_por_fold(importancias_folds, cols_feat, tag)

    if DIAGNOSTICO_FEATURES and diag_por_fold:
        try:
            consolidar_diagnostico(diag_por_fold, cols_feat, tag)
        except Exception as _e_cons:
            logger.warning(f"  [diag] consolidar_diagnostico falló: {_e_cons}")

    if GUARDAR_MODELO_FINAL and modelos_ultimo is not None:
        ultimo = folds[-1]

        # Reusar GARCH del último fold ya calculado en el manifest (evita 3ª estimación)
        garch_params_prod = folds_manifest[-1]["garch"] if folds_manifest else {}
        if garch_params_prod:
            logger.info(f"  GARCH producción (del manifest): {list(garch_params_prod.keys())}")
        else:
            try:
                garch_params_prod = _extraer_garch_params_fold(df, ultimo["train_end"])
            except Exception as _eg:
                logger.warning(f"  No se pudieron extraer GARCH params: {_eg}")

        for tau, model in modelos_ultimo.items():
            sfx = "lgbm_wfcv_v3" if MODELO_CV == "lgbm" else f"{MODELO_CV}_wfcv_v3"
            ext = ".txt" if MODELO_CV == "lgbm" else ".json"
            if tau == "mean":
                ruta_m = DIR_MODELOS / f"{sfx}_{banco}_mean_{fecha_hoy}{ext}"
            else:
                ruta_m = DIR_MODELOS / f"{sfx}_{banco}_q{int(tau*100):02d}_{fecha_hoy}{ext}"
            model.save_model(str(ruta_m))
        metadata = {
            "banco": banco, "modelo": f"{MODELO_CV}_wfcv_v3",
            "fecha_entrenamiento": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "version": "v3 -- expanding/rolling + TEST OOS",
            "config": {
                "expanding"          : EXPANDING,
                "ventana_train_años" : VENTANA_TRAIN_AÑOS,
                "ventana_val_años"   : VENTANA_VAL_AÑOS,
                "ventana_test_años"  : VENTANA_TEST_AÑOS,
                "paso_años"          : PASO_AÑOS,
                "purge_dias_hab"     : PURGE_DIAS_HAB,
                "purge_val_test"     : PURGE_VAL_TEST,
                "burn_in_dias_hab"   : BURN_IN_DIAS_HAB,
                "adaptive_trials"    : ADAPTIVE_TRIALS,
                "trials_flat"        : TRIALS_FLAT,
                "trials_por_tau"     : TRIALS_POR_TAU if ADAPTIVE_TRIALS else {},
            },
            "anti_leakage": {
                "purga_train_val"  : f"{PURGE_DIAS_HAB} dh post-TRAIN (cubre h_max={H_MAX_DIAS_HAB} + MA22)",
                "purga_val_test"   : f"{PURGE_VAL_TEST} dh post-VAL",
                "burn_in"          : f"{BURN_IN_DIAS_HAB} dh inicio TRAIN excluidos (MA22 warm-up)",
                "garch_por_fold"   : "omega/alpha/beta estimados en TRAIN, propagados a VAL+TEST",
                "medianas_por_fold": "calculadas en TRAIN, aplicadas a VAL+TEST",
                "val_test_sep"     : "VAL=Optuna only / TEST=métricas OOS only",
            },
            "ultimo_fold": {
                "train_start": str(ultimo["train_start"].date()),
                "train_end"  : str(ultimo["train_end"].date()),
                "test_start" : str(ultimo["test_start"].date()),
                "test_end"   : str(ultimo["test_end"].date()),
            },
            "garch_produccion": {
                "train_end" : str(ultimo["train_end"].date()),
                "series"    : garch_params_prod,
                "uso"       : (
                    "Usar omega/alpha/beta para propagar GARCH desde train_end "
                    "en producción -- garantiza consistencia entrenamiento-predicción"
                ),
            },
            "n_folds": len(folds), "quantiles": QUANTILES,
            "features": cols_feat, "best_params_ultimo_fold": params_ultimo,
            "folds_manifest": folds_manifest,
            "guardar_todos_folds": GUARDAR_MODELOS_TODOS_FOLDS,
        }
        ruta_meta = DIR_MODELOS / f"metadata_{MODELO_CV}_wfcv_v3_{banco}_{fecha_hoy}.json"
        with open(ruta_meta, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)

    t_total = time.time() - t_inicio
    logger.info(f"\n  {'-'*60}")
    logger.info(f"  RESUMEN -- {banco}  [{modo}]  [TEST OOS]")
    logger.info(f"  {'-'*60}")
    cols_d  = ["fold", "train_start", "train_end", "test_start", "test_end",
               "pinball_q50", "coverage_90", "winkler_90", "crps_approx"]
    cols_ok = [c for c in cols_d if c in df_test_m.columns]
    logger.info("\n" + df_test_m[cols_ok].to_string(index=False))

    prom_t = {c: df_test_m[c].mean() for c in ["pinball_q50", "coverage_90",
              "winkler_90", "crps_approx"] if c in df_test_m.columns}
    logger.info(f"\n  Promedios TEST:")
    for k, v in prom_t.items():
        logger.info(f"    {k:20s}: {f'{v:.1%}' if k=='coverage_90' else f'{v:,.2f}'}")

    if "coverage_90" in df_val_m.columns and "coverage_90" in df_test_m.columns:
        sesgo = df_val_m["coverage_90"].mean() - df_test_m["coverage_90"].mean()
        logger.info(f"\n  Sesgo VAL-TEST coverage: {sesgo:+.2%}")

    logger.info(f"\n  [OK] Completado en {t_total/60:.1f} min  ({len(folds)} folds)")
    return df_test_m


###############################################################################
# Main
###############################################################################

def main():
    modo = "EXPANDING" if EXPANDING else "ROLLING"
    logger.info("=" * 65)
    logger.info(f"STEP005 v3 -- Walk-Forward CV [{modo}] + TEST OOS  [{MODELO_CV.upper()}]")
    logger.info("=" * 65)
    logger.info(f"  EXPANDING={EXPANDING}  TRAIN_min={VENTANA_TRAIN_AÑOS}yr  "
                f"VAL={VENTANA_VAL_AÑOS}yr  TEST={VENTANA_TEST_AÑOS}yr  "
                f"paso={PASO_AÑOS}yr")

    if MODELO_CV == "lgbm" and not _LGBM_OK:
        logger.error("MODELO_CV='lgbm' pero lightgbm no está instalado.")
        return
    if not RUTA_MATRIZ.exists():
        logger.error(f"Matriz no encontrada: {RUTA_MATRIZ}")
        return

    t0 = time.time()
    todos = []
    for banco in BANCOS_A_EVALUAR:
        df_m = evaluar_banco(banco)
        if df_m is not None:
            todos.append((banco, df_m))

    if todos:
        logger.info("\n" + "=" * 65)
        logger.info(f"RESUMEN GLOBAL -- TEST OOS  [{modo}]")
        logger.info("=" * 65)
        for banco, df_m in todos:
            avg_cov = df_m["coverage_90"].mean() if "coverage_90" in df_m.columns else float("nan")
            avg_pb  = df_m["pinball_q50"].mean()  if "pinball_q50"  in df_m.columns else float("nan")
            logger.info(f"  {banco:15s}: {len(df_m)} folds | "
                        f"coverage_90_avg={avg_cov:.1%} | pinball_Q50_avg={avg_pb:,.0f}")

    logger.info(f"\n[OK] Total: {(time.time()-t0)/60:.1f} min  ->  {DIR_MODO}")


if __name__ == "__main__":
    main()
