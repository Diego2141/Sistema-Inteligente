# -*- coding: utf-8 -*-
"""
step005_walk_forward_cv.py
Walk-forward cross-validation con ventana rodante para XGBoost (arctan pinball).
Basado en step004 (tuning con τ=0.50, mismos hiperparámetros para todos los cuantiles).

═══════════════════════════════════════════════════════════════════════════════
MEDIDAS ANTI-LEAKAGE — qué se hizo y por qué
═══════════════════════════════════════════════════════════════════════════════

PROBLEMA 1 — Confirmed features (R_conf_t1, R_conf_t2, D_conf_t1)
  En la matriz estas features son shift(-1/-2) del target realizado:
  la fila fecha_t=T tiene R_conf_t1=R(T+1), R_conf_t2=R(T+2), D_conf_t1=D(T+1).
  Si el TRAIN termina en T y el VAL empieza en T+1, el modelo ve partes del
  target de VAL como features de la última fila de TRAIN.

  SOLUCIÓN ► EMBARGO de EMBARGO_DIAS_HAB días hábiles (= h_max = 90 por defecto).
             Las fechas_t entre fin-TRAIN y inicio-VAL se excluyen completamente
             del dataset.  El último fecha_t del TRAIN tiene confirmed values
             apuntando a T+1 y T+2; con 90 dh de colchón, VAL empieza en T+90,
             por lo que ninguna feature de TRAIN puede "ver" un target de VAL.

PROBLEMA 2 — GARCH estimado sobre la serie completa (step001)
  _garch_vol() en step001 ajusta ω, α, β por MLE sobre todos los datos
  2010→hoy. Los parámetros capturan regímenes futuros (COVID 2020,
  elecciones 2021, ciclo de tasas 2022). Afecta tres features:
    · garch_vol      → flujo neto D−R del banco/sistema
    · garch_vol_tc   → retornos log del tipo de cambio PEN/USD
    · garch_vol_embi → cambios diarios del EMBI Perú

  SOLUCIÓN ► Por cada fold, _garch_vol_fold() re-estima ω, α, β usando
             ÚNICAMENTE las observaciones de TRAIN del fold (fechas_t ≤ train_end).
             La varianza condicional σ²_t se propaga recursivamente al período VAL
             con esos parámetros; la recursión σ²_t = ω+α·ε²_{t-1}+β·σ²_{t-1}
             solo necesita la historia pasada, sin look-ahead.
             Los valores pre-computados de step001 (serie completa) se reemplazan
             in-situ dentro del loop de folds antes de entrenar o predecir.

PROBLEMA 3 — Medianas de imputación calculadas sobre CORTE_VAL global
  En step004, medianas_train = df_train[cols].median() donde df_train incluye
  hasta 2022-07.  En un fold cuyo TRAIN termina en 2015, imputar NaN con
  medianas de 2010-2022 introduce información de 7 años futuros.

  SOLUCIÓN ► Cada fold recalcula medianas de imputación usando solo las
             fechas_t pertenecientes a su propio TRAIN, antes de cualquier
             operación de entrenamiento o predicción.

FEATURES SIN LEAKAGE (no requieren corrección):
  · sigma_R_5d, sigma_D_5d, ma_R_5d, ma_D_5d, sigma_flujo_5d, ma_flujo_20d…
    → rolling backward-looking puras (ventana fija hacia el pasado).
  · VIX_ma22, tc_vol_5d, tc_vol_22d, delta_TC, delta_EMBI…
    → ídem, series macroeconómicas con rolling/diff.
  · Features de calendario (dias_al_cierre_mes, is_pre_eleccion…)
    → determinísticas, no dependen de datos observados.

═══════════════════════════════════════════════════════════════════════════════
Parámetros configurables
═══════════════════════════════════════════════════════════════════════════════
  VENTANA_TRAIN_AÑOS : años de historia de entrenamiento por fold (5 ó 7)
  VENTANA_VAL_AÑOS   : años de validación por fold (1 ó 0.5)
  PASO_AÑOS          : desplazamiento entre folds (1 ó 0.5)
  EMBARGO_DIAS_HAB   : días hábiles de buffer post-TRAIN (90 = h_max)
  N_TRIALS_OPTUNA    : trials Optuna por fold (30 por defecto)
  GUARDAR_MODELO_FINAL: True → re-entrena sobre último fold TRAIN y guarda
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import gc
import json
import logging
import time
import warnings
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

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


###############################################################################
# PARTE 0 — Configuración
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv"   # raíz compartida

# ── Ventana rodante ───────────────────────────────────────────────────────────
VENTANA_TRAIN_AÑOS  = 5      # años de TRAIN por fold  (probar: 5 ó 7)
VENTANA_VAL_AÑOS    = 1      # años de VAL por fold    (probar: 1 ó 0.5)
PASO_AÑOS           = 1      # desplazamiento entre folds (probar: 1 ó 0.5)
EMBARGO_DIAS_HAB    = 90     # días hábiles entre fin-TRAIN e inicio-VAL (= h_max)

# ── Modelo ────────────────────────────────────────────────────────────────────
QUANTILES        = [0.01, 0.05, 0.50, 0.95, 0.99]
N_TRIALS_OPTUNA  = 30        # trials por fold (30 × N_folds ≈ carga total)
S_MIN_FACTOR     = 0.01      # s_min = std(y_train) * factor
S_MAX_FACTOR     = 1.0       # s_max = std(y_train) * factor

# ── Límite del conjunto de evaluación ────────────────────────────────────────
# Los folds no invaden el TEST fijo (2023-01-03+); si CORTE_TEST está definido
# el último fold_val_end ≤ CORTE_TEST.
CORTE_TEST = pd.Timestamp("2023-01-03")

# ── Opciones de salida ────────────────────────────────────────────────────────
BANCOS_A_EVALUAR  = ["SISTEMA"]
GUARDAR_MODELO_FINAL = True    # True → guarda modelo del último fold TRAIN
COLS_EXCLUIR      = {"fecha_t", "banco", "target"}

# ── Selector de modelo ────────────────────────────────────────────────────────
MODELO_CV = "xgb"
# Opciones:
#   "xgb"    → XGBoost arctan pinball, Optuna τ=0.50 compartido (estilo step004)
#   "lgbm"   → LightGBM cuantil nativo,  Optuna τ=0.50 compartido (estilo step003)
#   "xgb_qt" → XGBoost arctan pinball,   Optuna per-cuantil       (estilo step004_qt)
assert MODELO_CV in ("xgb", "lgbm", "xgb_qt"), \
    f"MODELO_CV debe ser 'xgb', 'lgbm' o 'xgb_qt', recibido: {MODELO_CV!r}"

# ── Regularización Optuna ─────────────────────────────────────────────────────
# FIX_REG_ALPHA: fija reg_alpha=0.0 en lugar de buscarlo en [1e-4,10] log-scale.
#   True  → reg_alpha=0 fijo; libera trials para params con gradiente real.
#   False → búsqueda original (CV>200%, paisaje plano, trials desperdiciados).
#   Evidencia: CV>200% en xgb y xgb_qt; efecto sobre pérdida <0.04%.
#   RECOMENDADO: True — no afecta cobertura, solo elimina ruido de búsqueda.
FIX_REG_ALPHA = True

# FIX_REG_LAMBDA: acota reg_lambda a [0.1, 5.0] lineal en vez de [1e-4,10] log.
#   True  → rango estrecho centrado en el default de XGBoost (1.0).
#   False → búsqueda original (log-scale amplio que permite valores ~0.001).
#   ADVERTENCIA: True probado y empeoró cobertura folds 5-8 (modelo eligió
#   valores <0.1 que quedaron fuera del nuevo rango → underfitting en régimen
#   volátil). Mantener False hasta nueva evidencia.
FIX_REG_LAMBDA = False

# ── Rutas de salida — subcarpeta específica por modelo ───────────────────────
# Estructura: step005_wfcv/<modelo>/  (CSVs y PNGs)
#             step005_wfcv/<modelo>/modelos/  (archivos .json/.txt guardados)
#             step005_wfcv/<modelo>/plots/    (gráficos PNG)
DIR_MODELO_OUTPUT = DIR_OUTPUT / MODELO_CV
DIR_MODELOS       = DIR_MODELO_OUTPUT / "modelos"
DIR_PLOTS         = DIR_MODELO_OUTPUT / "plots"

for _d in (DIR_OUTPUT, DIR_MODELO_OUTPUT, DIR_MODELOS, DIR_PLOTS):
    _d.mkdir(parents=True, exist_ok=True)


###############################################################################
# PARTE 1 — Utilidades métricas
###############################################################################

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss promedio exacta."""
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def winkler_score(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray,
                  alpha: float = 0.10) -> float:
    """
    Winkler score para un intervalo de cobertura nominal (1-alpha).
    Penaliza tanto amplitud excesiva como salidas del intervalo.
    """
    width  = q_hi - q_lo
    below  = np.maximum(q_lo - y_true, 0)
    above  = np.maximum(y_true - q_hi, 0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def coverage(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    """Cobertura empírica: fracción de observaciones dentro del intervalo."""
    return float(np.mean((y_true >= q_lo) & (y_true <= q_hi)))


def crps_approx(y_true: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """
    Aproximación del CRPS usando la regla trapezoidal sobre los cuantiles
    disponibles: CRPS ≈ 2 · Σ PB(τᵢ) · Δτᵢ.
    """
    taus = sorted(preds.keys())
    pbs  = [pinball_loss(y_true, preds[tau], tau) for tau in taus]
    # Pesos trapezoide: (τ_{i+1} - τ_{i-1}) / 2 para nodos interiores
    n    = len(taus)
    pesos = []
    for i in range(n):
        lo = taus[i - 1] if i > 0     else taus[0]
        hi = taus[i + 1] if i < n - 1 else taus[-1]
        pesos.append((hi - lo) / 2.0)
    return float(2.0 * sum(w * pb for w, pb in zip(pesos, pbs)))


###############################################################################
# PARTE 2 — GARCH por fold (anti-leakage medida #2)
###############################################################################

def _ajustar_garch_params(x_train: np.ndarray) -> tuple[float, float, float]:
    """
    Estima ω, α, β de GARCH(1,1) por Máxima Verosimilitud Gaussiana
    sobre x_train (serie ya escalada por std de TRAIN).
    Restricciones: ω>0, α>0, β>0, α+β<1 (estacionariedad).
    Fallback a parámetros típicos si la optimización diverge.
    """
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
    return 0.01, 0.08, 0.88   # parámetros conservadores de fallback


def _garch_vol_fold(serie: pd.Series, train_end: pd.Timestamp) -> pd.Series:
    """
    Estimación GARCH(1,1) honesta para un fold de walk-forward CV.

    Anti-leakage (medida #2):
      · ω, α, β se estiman SOLO sobre fechas ≤ train_end (período TRAIN).
      · escala = std(serie_train); la serie escalada también usa solo TRAIN.
      · La varianza condicional σ²_t se propaga recursivamente hacia el futuro
        (período VAL) usando la fórmula σ²_t = ω+α·ε²_{t-1}+β·σ²_{t-1}.
        Esta recursión solo requiere ε_{t-1} y σ²_{t-1}, no datos futuros.
      · El parámetro `escala` también viene de TRAIN → sin información de VAL.

    Contraste con step001: allí _garch_vol() recibe la serie completa y ajusta
    los parámetros sobre 2010→hoy, "enseñándole" al modelo los regímenes que
    el fold no debería conocer.

    Fallback: si hay <60 observaciones en TRAIN o la serie es constante,
    devuelve rolling std(20d) calculada solo hasta train_end.
    """
    serie_full  = serie.ffill().fillna(0.0)
    serie_train = serie_full[serie_full.index <= train_end]

    if len(serie_train) < 60 or serie_train.std() < 1e-9:
        return serie_full.rolling(20).std().fillna(serie_train.std())

    escala   = float(serie_train.std())
    x_train  = (serie_train / escala).values.astype(float)
    var_unc  = max(float(np.var(x_train)), 1e-12)

    omega, alpha, beta = _ajustar_garch_params(x_train)

    # Propagar σ² sobre la serie completa (TRAIN + VAL) con parámetros de TRAIN
    x_full = (serie_full / escala).values.astype(float)
    n_full = len(x_full)
    s2     = np.empty(n_full)
    s2[0]  = var_unc
    for t in range(1, n_full):
        s2[t] = omega + alpha * x_full[t - 1] ** 2 + beta * s2[t - 1]

    return pd.Series(np.sqrt(np.maximum(s2, 0)) * escala, index=serie_full.index)


def reemplazar_garch_fold(df_fold: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """
    Re-estima las 3 features GARCH del fold usando solo datos hasta train_end
    y reemplaza los valores pre-computados en step001.

    Reconstrucción de las series crudas:
      · flujo neto D−R  →  D_t0 − R_t0  (una observación por fecha_t)
      · retornos log TC →  log(TC_t / TC_{t-1})
      · cambios EMBI    →  EMBI_t − EMBI_{t-1}

    Estas tres series son las mismas que usó step001 para estimar GARCH
    originalmente, pero ahora el ajuste se limita al período TRAIN del fold.
    """
    df_fold = df_fold.copy()

    # Serie única por fecha_t (el mismo valor se repite para todos los h)
    idx_cols = ["fecha_t", "R_t0", "D_t0", "TC_PEN_USD", "EMBI_PERU"]
    avail    = [c for c in idx_cols if c in df_fold.columns]
    raw      = (df_fold[avail]
                .drop_duplicates("fecha_t")
                .set_index("fecha_t")
                .sort_index())

    # 1. garch_vol — flujo neto del banco/sistema
    if {"R_t0", "D_t0"}.issubset(raw.columns):
        flujo        = raw["D_t0"] - raw["R_t0"]
        sigma_flujo  = _garch_vol_fold(flujo, train_end)
        df_fold["garch_vol"] = df_fold["fecha_t"].map(sigma_flujo)
        logger.debug(f"    garch_vol re-estimado hasta {train_end.date()}")

    # 2. garch_vol_tc — volatilidad condicional de retornos log del TC
    if "TC_PEN_USD" in raw.columns:
        tc = raw["TC_PEN_USD"].replace(0, np.nan).ffill()
        # Rellenar el gap del embargo con la última observación de TRAIN para
        # que shift(1) no genere un retorno de 90 días en el primer día de VAL.
        tc_reidx    = tc.reindex(pd.bdate_range(tc.index.min(), tc.index.max()))
        tc_reidx    = tc_reidx.ffill()
        retornos_tc = np.log(tc_reidx / tc_reidx.shift(1)).reindex(tc.index)
        sigma_tc      = _garch_vol_fold(retornos_tc, train_end)
        df_fold["garch_vol_tc"] = df_fold["fecha_t"].map(sigma_tc)
        logger.debug(f"    garch_vol_tc re-estimado hasta {train_end.date()}")

    # 3. garch_vol_embi — volatilidad condicional de cambios del EMBI
    if "EMBI_PERU" in raw.columns:
        delta_embi      = raw["EMBI_PERU"].diff(1)
        sigma_embi      = _garch_vol_fold(delta_embi, train_end)
        df_fold["garch_vol_embi"] = df_fold["fecha_t"].map(sigma_embi)
        logger.debug(f"    garch_vol_embi re-estimado hasta {train_end.date()}")

    return df_fold


###############################################################################
# PARTE 3 — Generación de folds
###############################################################################

def generar_folds(
    fechas_disponibles: pd.DatetimeIndex,
    ventana_train_años: float,
    ventana_val_años: float,
    paso_años: float,
    embargo_dias_hab: int,
    corte_test: pd.Timestamp,
) -> list[dict]:
    """
    Genera los folds de walk-forward CV con ventana rodante.

    Estructura de cada fold:
      ├── TRAIN  : [train_start, train_end]          → se entrena aquí
      ├── EMBARGO: (train_end, val_start)             → excluido del dataset
      └── VAL    : [val_start, val_end]               → se evalúa aquí

    La ventana de TRAIN tiene longitud fija (rolling, no expanding):
    al avanzar un fold, tanto train_start como train_end se desplazan PASO_AÑOS.

    Restricción: val_end ≤ corte_test para no invadir el holdout final.
    """
    folds  = []
    f_min  = fechas_disponibles.min()
    # Desplazar en años usando DateOffset (maneja años bisiestos correctamente)
    paso   = pd.DateOffset(months=int(round(paso_años * 12)))

    train_start = f_min
    fold_num    = 1

    while True:
        train_end = train_start + pd.DateOffset(months=int(round(ventana_train_años * 12)))
        # Embargo: avanzar N días hábiles desde train_end
        val_start = train_end + pd.offsets.BusinessDay(embargo_dias_hab)
        val_end   = val_start + pd.DateOffset(months=int(round(ventana_val_años * 12)))

        # Clip al límite de datos disponibles y al corte TEST
        if train_end >= corte_test:
            break
        if val_start >= corte_test:
            break
        val_end_clip = min(val_end, corte_test - pd.offsets.BusinessDay(1))
        if val_start >= val_end_clip:
            break

        # Verificar que hay fechas reales en TRAIN y VAL
        n_train = ((fechas_disponibles >= train_start) &
                   (fechas_disponibles <= train_end)).sum()
        n_val   = ((fechas_disponibles >= val_start) &
                   (fechas_disponibles <= val_end_clip)).sum()

        if n_train < 60 or n_val < 10:
            train_start += paso
            continue

        folds.append({
            "fold"        : fold_num,
            "train_start" : train_start,
            "train_end"   : train_end,
            "val_start"   : val_start,
            "val_end"     : val_end_clip,
            "n_train_fechas": int(n_train),
            "n_val_fechas"  : int(n_val),
        })
        fold_num    += 1
        train_start += paso

    return folds


###############################################################################
# PARTE 4 — Objetivo arctan pinball y entrenamiento XGBoost
# (adaptado directamente de step004 sin cambios en la lógica matemática)
###############################################################################

def make_quantile_objective(tau: float, s: float, std_y: float):
    """
    Gradiente y hessiana de la arctan pinball loss — ecuación (11) Zheng et al.
    Normalización por scale = π(s²+std_y²)²/(2s³) para dar semántica de
    conteo a min_child_weight (ver step004 para derivación completa).
    """
    _scale = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)

    def objective(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        u    = dtrain.get_label() - y_pred
        grad = -((tau - 0.5 + np.arctan(u / s) / np.pi)
                 + u * s / (np.pi * (s ** 2 + u ** 2))) * _scale
        hess = 2 * s ** 3 / (np.pi * (s ** 2 + u ** 2) ** 2) * _scale
        return grad, hess
    return objective


def make_pinball_metric(tau: float):
    """Pinball exacta para early stopping en XGBoost."""
    def metric(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label()
        u      = y_true - y_pred
        loss   = float(np.mean(np.where(u >= 0, tau * u, (tau - 1) * u)))
        return "pinball", loss
    return metric


def _objective_optuna(trial, X_tr, y_tr, X_va, y_va, std_y):
    """Función objetivo Optuna: pinball τ=0.50 en VAL."""
    s = trial.suggest_float("s", std_y * S_MIN_FACTOR, std_y * S_MAX_FACTOR, log=True)
    params = {
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth"       : trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 200),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha"       : 0.0 if FIX_REG_ALPHA else trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 0.1, 5.0) if FIX_REG_LAMBDA else trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method"     : "hist",
        "seed"            : 42,
    }
    n_est = trial.suggest_int("n_estimators", 100, 1000)

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    tau    = 0.50

    model  = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(tau, s, std_y),
        custom_metric=make_pinball_metric(tau),
        evals=[(dval, "val")],
        callbacks=[xgb.callback.EarlyStopping(rounds=50, metric_name="pinball",
                                               save_best=False, maximize=False)],
        verbose_eval=False,
    )
    return pinball_loss(y_va.values, model.predict(dval), tau)


def optimizar_hiperparametros(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
    """Optuna con TPESampler — seed fijo por fold para reproducibilidad."""
    logger.info(f"    Optuna fold {fold_num} ({n_trials} trials, τ=0.50)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold_num),
    )
    study.optimize(
        lambda t: _objective_optuna(t, X_tr, y_tr, X_va, y_va, std_y),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    bp = study.best_params
    logger.info(f"    Fold {fold_num} best pinball(Q50)={study.best_value:.4f} "
                f"n_est={bp['n_estimators']} lr={bp['learning_rate']:.4f} s={bp['s']:.2f}")
    return bp


def entrenar_quantiles(X_tr, y_tr, best_params, quantiles, std_y):
    """Entrena un booster por cuantil con los mejores hiperparámetros de Optuna."""
    s_best  = best_params["s"]
    n_est   = best_params["n_estimators"]
    params  = {k: v for k, v in best_params.items() if k not in ("s", "n_estimators")}
    params.update({"tree_method": "hist", "seed": 42})
    dtrain  = xgb.DMatrix(X_tr, label=y_tr)

    modelos = {}
    for tau in quantiles:
        modelos[tau] = xgb.train(
            params, dtrain, num_boost_round=n_est,
            obj=make_quantile_objective(tau, s_best, std_y),
            verbose_eval=False,
        )
    return modelos


def predecir_y_corregir(modelos, X):
    """Predice todos los cuantiles y corrige cruces de cuantiles."""
    dmat = xgb.DMatrix(X)
    preds_raw = {tau: m.predict(dmat) for tau, m in modelos.items()}
    taus   = sorted(preds_raw)
    matrix = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    return {t: matrix[:, i] for i, t in enumerate(taus)}


###############################################################################
# PARTE 4b — LightGBM (estilo step003) y XGBoost QT (estilo step004_qt)
###############################################################################

# ── LightGBM ──────────────────────────────────────────────────────────────────

def _objetivo_optuna_lgbm(trial, X_tr, y_tr, X_va, y_va):
    """Optuna para LightGBM cuantil nativo, optimiza τ=0.50 en VAL."""
    if not _LGBM_OK:
        raise ImportError("lightgbm no está instalado (pip install lightgbm)")
    tau = 0.50
    params = {
        "objective"        : "quantile",
        "alpha"            : tau,
        "verbosity"        : -1,
        "seed"             : 42,
        "learning_rate"    : trial.suggest_float("learning_rate",  0.01,  0.3,  log=True),
        "num_leaves"       : trial.suggest_int(  "num_leaves",      15,   255),
        "max_depth"        : trial.suggest_int(  "max_depth",        3,    10),
        "min_child_samples": trial.suggest_int(  "min_child_samples", 10,  200),
        "subsample"        : trial.suggest_float("subsample",       0.5,   1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4,  1.0),
        "reg_alpha"        : 0.0 if FIX_REG_ALPHA else trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 0.1, 5.0) if FIX_REG_LAMBDA else trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "subsample_freq"   : 1,
    }
    n_est = trial.suggest_int("n_estimators", 100, 1000)
    dtrain = lgb.Dataset(X_tr.values, label=y_tr.values)
    dval   = lgb.Dataset(X_va.values, label=y_va.values, reference=dtrain)
    callbacks = [lgb.log_evaluation(-1)]
    try:
        callbacks.append(lgb.early_stopping(50, verbose=False))
    except Exception:
        pass
    model = lgb.train(
        params, dtrain, num_boost_round=n_est,
        valid_sets=[dval], valid_names=["val"],
        callbacks=callbacks,
    )
    return pinball_loss(y_va.values, model.predict(X_va.values), tau)


def optimizar_hiperparametros_lgbm(X_tr, y_tr, X_va, y_va, n_trials, fold_num):
    logger.info(f"    Optuna[LGBM] fold {fold_num} ({n_trials} trials, τ=0.50)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold_num),
    )
    study.optimize(
        lambda t: _objetivo_optuna_lgbm(t, X_tr, y_tr, X_va, y_va),
        n_trials=n_trials, show_progress_bar=False,
    )
    bp = study.best_params
    logger.info(f"    Fold {fold_num} [LGBM] best pinball(Q50)={study.best_value:.4f} "
                f"n_est={bp['n_estimators']} lr={bp['learning_rate']:.4f}")
    return bp


def entrenar_quantiles_lgbm(X_tr, y_tr, best_params, quantiles):
    if not _LGBM_OK:
        raise ImportError("lightgbm no está instalado")
    n_est  = int(best_params.get("n_estimators", 300))
    hp     = {k: v for k, v in best_params.items() if k != "n_estimators"}
    modelos = {}
    for tau in quantiles:
        params = {
            "objective"     : "quantile",
            "alpha"         : tau,
            "verbosity"     : -1,
            "seed"          : 42,
            "subsample_freq": 1,
            **hp,
        }
        dtrain = lgb.Dataset(X_tr.values, label=y_tr.values)
        modelos[tau] = lgb.train(
            params, dtrain, num_boost_round=n_est,
            callbacks=[lgb.log_evaluation(-1)],
        )
    return modelos


def predecir_lgbm(modelos, X):
    """Predice y corrige cruces de cuantiles para LightGBM."""
    preds_raw = {tau: m.predict(X.values) for tau, m in modelos.items()}
    taus   = sorted(preds_raw)
    matrix = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    return {t: matrix[:, i] for i, t in enumerate(taus)}


# ── XGBoost QT (per-quantile Optuna) ─────────────────────────────────────────

def _objetivo_optuna_xgb_qt_tau(trial, tau, X_tr, y_tr, X_va, y_va, std_y):
    """Optuna para un cuantil τ específico (xgb_qt mode)."""
    s = trial.suggest_float("s", std_y * S_MIN_FACTOR, std_y * S_MAX_FACTOR, log=True)
    params = {
        "learning_rate"   : trial.suggest_float("learning_rate",   0.01,  0.3,  log=True),
        "max_depth"       : trial.suggest_int(  "max_depth",         3,    10),
        "min_child_weight": trial.suggest_int(  "min_child_weight", 10,   200),
        "colsample_bytree": trial.suggest_float("colsample_bytree",  0.4,  1.0),
        "subsample"       : trial.suggest_float("subsample",         0.5,  1.0),
        "reg_alpha"       : 0.0 if FIX_REG_ALPHA else trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 0.1, 5.0) if FIX_REG_LAMBDA else trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method"     : "hist",
        "seed"            : 42,
    }
    n_est  = trial.suggest_int("n_estimators", 100, 1000)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    model  = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(tau, s, std_y),
        custom_metric=make_pinball_metric(tau),
        evals=[(dval, "val")],
        callbacks=[xgb.callback.EarlyStopping(rounds=50, metric_name="pinball",
                                               save_best=False, maximize=False)],
        verbose_eval=False,
    )
    return pinball_loss(y_va.values, model.predict(dval), tau)


def _entrenar_fold_xgb_qt(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
    """
    Per-quantile Optuna (estilo step004_qt):
    cada τ tiene su propio estudio con sus propios hiperparámetros y s.
    Retorna: modelos dict, y best_params de τ=0.50 para HP tracking.
    """
    best_by_tau = {}
    modelos     = {}
    for tau in QUANTILES:
        seed_tau = 42 + fold_num + int(tau * 100)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed_tau),
        )
        study.optimize(
            lambda t, _tau=tau: _objetivo_optuna_xgb_qt_tau(
                t, _tau, X_tr, y_tr, X_va, y_va, std_y),
            n_trials=n_trials, show_progress_bar=False,
        )
        bp = study.best_params
        best_by_tau[tau] = bp
        logger.info(f"    [xgb_qt] τ={tau} fold {fold_num}: "
                    f"pinball={study.best_value:.4f} s={bp['s']:.3f}")
        # Entrenar el modelo final para este τ
        s      = bp["s"]
        n_est  = bp["n_estimators"]
        params = {k: v for k, v in bp.items() if k not in ("s", "n_estimators")}
        params.update({"tree_method": "hist", "seed": 42})
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        modelos[tau] = xgb.train(
            params, dtrain, num_boost_round=n_est,
            obj=make_quantile_objective(tau, s, std_y),
            verbose_eval=False,
        )
    # Devolver Q50 params para HP tracking (más informativo que cualquier otro τ)
    best_q50 = best_by_tau.get(0.50, list(best_by_tau.values())[0])
    return modelos, best_q50


# ── Dispatchers ───────────────────────────────────────────────────────────────

def entrenar_fold(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
    """
    Entrena un fold completo (Optuna + todos los cuantiles) según MODELO_CV.
    Retorna: (modelos: dict[float, booster], best_params: dict)
    best_params es siempre el diccionario del estudio τ=0.50 para HP tracking.
    """
    if MODELO_CV == "xgb":
        best_params = optimizar_hiperparametros(
            X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num)
        modelos = entrenar_quantiles(X_tr, y_tr, best_params, QUANTILES, std_y)
        return modelos, best_params

    elif MODELO_CV == "lgbm":
        if not _LGBM_OK:
            raise ImportError("MODELO_CV='lgbm' requiere lightgbm instalado")
        best_params = optimizar_hiperparametros_lgbm(
            X_tr, y_tr, X_va, y_va, n_trials, fold_num)
        modelos = entrenar_quantiles_lgbm(X_tr, y_tr, best_params.copy(), QUANTILES)
        return modelos, best_params

    elif MODELO_CV == "xgb_qt":
        return _entrenar_fold_xgb_qt(
            X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num)

    else:
        raise ValueError(f"MODELO_CV desconocido: {MODELO_CV!r}")


def predecir_fold(modelos, X):
    """Predice según MODELO_CV y corrige cruces de cuantiles."""
    if MODELO_CV == "lgbm":
        return predecir_lgbm(modelos, X)
    else:
        return predecir_y_corregir(modelos, X)


###############################################################################
# PARTE 5 — Preparación de datos por fold (anti-leakage medidas #2 y #3)
###############################################################################

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Infiere columnas numéricas de features (idéntico a step004)."""
    excluir = COLS_EXCLUIR | {"fecha_th"}
    return [c for c in df.columns
            if c not in excluir and df[c].dtype.kind in ("f", "i", "u", "b")]


def preparar_fold_data(
    df: pd.DataFrame,
    fold: dict,
    cols_feat: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """
    Extrae y prepara los datos de TRAIN y VAL para un fold, aplicando
    las tres medidas anti-leakage:

    1. EMBARGO (medida #1): las filas con fecha_t en el período de embargo
       (train_end < fecha_t < val_start) se descartan completamente.

    2. GARCH por fold (medida #2): llama a reemplazar_garch_fold() con
       train_end del fold antes de calcular medianas o extraer X/y.
       Esto garantiza que tanto TRAIN como VAL usan parámetros GARCH
       estimados únicamente sobre el TRAIN de este fold.

    3. Medianas de imputación por fold (medida #3): las medianas se calculan
       sobre las filas de TRAIN del fold (no sobre CORTE_VAL global).
       Luego se aplican a TRAIN y VAL por igual.

    Retorna: X_train, y_train, X_val, y_val, h_train_arr, h_val_arr
    """
    train_start = fold["train_start"]
    train_end   = fold["train_end"]
    val_start   = fold["val_start"]
    val_end     = fold["val_end"]

    # ── Medida #1: Embargo ────────────────────────────────────────────────────
    # Excluir el período (train_end, val_start) del dataset del fold
    mask_train = (df["fecha_t"] >= train_start) & (df["fecha_t"] <= train_end)
    mask_val   = (df["fecha_t"] >= val_start)   & (df["fecha_t"] <= val_end)

    df_fold_all = df[mask_train | mask_val].copy()

    # ── Medida #2: GARCH por fold ─────────────────────────────────────────────
    # Re-estima garch_vol, garch_vol_tc, garch_vol_embi usando solo TRAIN
    garch_cols = [c for c in ["garch_vol", "garch_vol_tc", "garch_vol_embi"]
                  if c in df_fold_all.columns]
    if garch_cols:
        df_fold_all = reemplazar_garch_fold(df_fold_all, train_end)

    df_train = df_fold_all[df_fold_all["fecha_t"] <= train_end]
    df_val   = df_fold_all[df_fold_all["fecha_t"] >= val_start]

    # ── Medida #3: Medianas de imputación por fold ────────────────────────────
    # Solo con las filas de TRAIN del fold (no con CORTE_VAL global)
    medianas_fold = df_train[cols_feat].median()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_train[cols_feat] = df_train[cols_feat].fillna(medianas_fold)
    df_val[cols_feat]   = df_val[cols_feat].fillna(medianas_fold)

    # Filtrar filas con target disponible
    mask_t = df_train["target"].notna()
    mask_v = df_val["target"].notna()

    X_train = df_train.loc[mask_t, cols_feat].copy()
    y_train = df_train.loc[mask_t, "target"].copy()
    h_train = df_train.loc[mask_t, "h"].values

    X_val   = df_val.loc[mask_v, cols_feat].copy()
    y_val   = df_val.loc[mask_v, "target"].copy()
    h_val   = df_val.loc[mask_v, "h"].values

    return X_train, y_train, X_val, y_val, h_train, h_val


###############################################################################
# PARTE 6 — Cálculo de métricas por fold
###############################################################################

def calcular_metricas_fold(
    preds: dict[float, np.ndarray],
    y_true: np.ndarray,
    fold: dict,
) -> dict:
    """
    Métricas globales (sobre todos los horizontes) para el fold.
    Retorna un dict plano listo para concatenar en el DataFrame de resultados.
    """
    row = {
        "fold"          : fold["fold"],
        "train_start"   : fold["train_start"].date(),
        "train_end"     : fold["train_end"].date(),
        "val_start"     : fold["val_start"].date(),
        "val_end"       : fold["val_end"].date(),
        "n_train_fechas": fold["n_train_fechas"],
        "n_val_fechas"  : fold["n_val_fechas"],
        "n_obs_val"     : int(len(y_true)),
    }

    for tau in sorted(preds.keys()):
        row[f"pinball_q{int(tau*100):02d}"] = round(
            pinball_loss(y_true, preds[tau], tau), 2
        )

    if {0.05, 0.95}.issubset(preds):
        row["coverage_90"] = round(coverage(y_true, preds[0.05], preds[0.95]), 4)
        row["winkler_90"]  = round(winkler_score(y_true, preds[0.05], preds[0.95]), 2)

    row["crps_approx"] = round(crps_approx(y_true, preds), 2)
    return row


def calcular_metricas_por_h(
    preds: dict[float, np.ndarray],
    y_true: np.ndarray,
    h_arr: np.ndarray,
    fold_num: int,
) -> pd.DataFrame:
    """
    Métricas de pinball Q50 y coverage 90% desagregadas por horizonte h.
    Útil para detectar si el modelo falla en horizontes cortos o largos.
    """
    hs_unicos = np.unique(h_arr)
    rows = []
    for h in hs_unicos:
        mask = h_arr == h
        if mask.sum() < 5:
            continue
        row = {"fold": fold_num, "h": int(h)}
        for tau in sorted(preds.keys()):
            row[f"pinball_q{int(tau*100):02d}"] = round(
                pinball_loss(y_true[mask], preds[tau][mask], tau), 2
            )
        if {0.05, 0.95}.issubset(preds):
            row["coverage_90"] = round(
                coverage(y_true[mask], preds[0.05][mask], preds[0.95][mask]), 4
            )
        rows.append(row)
    return pd.DataFrame(rows)


###############################################################################
# PARTE 7 — Visualización
###############################################################################

def graficar_metricas_wfcv(df_metricas: pd.DataFrame, banco: str):
    """
    4 subplots horizontales por banco:
      1. Pinball Q50 por fold — deriva de desempeño en la mediana
      2. Coverage 90% empírica — línea horizontal en 0.90 (nivel nominal)
      3. Winkler score 90% — calibración del intervalo de incertidumbre
      4. CRPS aproximado — medida resumen de toda la distribución predicha
    Cada panel muestra la tendencia lineal para detectar degradación temporal.
    """
    if df_metricas.empty:
        return

    folds = df_metricas["fold"].values

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Walk-forward CV — {banco}  |  "
        f"TRAIN {VENTANA_TRAIN_AÑOS}yr / VAL {VENTANA_VAL_AÑOS}yr / "
        f"paso {PASO_AÑOS}yr / embargo {EMBARGO_DIAS_HAB}dh",
        fontweight="bold", fontsize=11,
    )

    metricas_config = [
        ("pinball_q50",  "Pinball Q50 (mediana)",   "steelblue",   None),
        ("coverage_90",  "Coverage 90% empírica",   "seagreen",    0.90),
        ("winkler_90",   "Winkler score 90%",        "darkorange",  None),
        ("crps_approx",  "CRPS aproximado",          "crimson",     None),
    ]

    for ax, (col, titulo, color, hline) in zip(axes, metricas_config):
        if col not in df_metricas.columns:
            ax.set_visible(False)
            continue
        vals = df_metricas[col].values.astype(float)
        ax.plot(folds, vals, "o-", color=color, lw=2, ms=7, zorder=4)

        # Línea de tendencia lineal
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
            lambda x, _, _col=col: f"{x:,.0f}" if _col != "coverage_90" else f"{x:.1%}"
        ))
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_metricas_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico métricas guardado: {nombre}")


def graficar_cobertura_por_h(df_por_h: pd.DataFrame, banco: str):
    """
    Heatmap de coverage 90% por (fold, h):
    muestra en qué horizontes y en qué épocas el modelo falla en calibración.
    Ideal para detectar si horizontes cortos tienen under-coverage sistemático.
    """
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
        f"Coverage 90% por fold y horizonte — {banco}\n"
        "(verde=bien calibrado ~90%, rojo=sub/sobre-cobertura)",
        fontsize=10,
    )
    # Mostrar solo cada 10 etiquetas en eje h para no saturar
    xticks_pos = range(0, len(pivot.columns), 10)
    ax.set_xticks([p for p in xticks_pos if p < len(pivot.columns)])
    ax.set_xticklabels([str(pivot.columns[p]) for p in xticks_pos
                        if p < len(pivot.columns)], fontsize=8)
    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_cobertura_por_h_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Heatmap cobertura guardado: {nombre}")


def graficar_hiperparametros_wfcv(df_metricas: pd.DataFrame, banco: str):
    """
    Evolución temporal de los hiperparámetros óptimos por fold.

    Por qué es útil:
      · s (smoothing): si crece en folds recientes → el régimen se volvió más
        ruidoso y la arctan necesita más suavizado para producir hessianas
        estables. Caída de s → más confianza en datos más nítidos.
      · learning_rate: tasas altas suelen indicar un paisaje de pérdida más
        suave (menor complejidad efectiva del fold).
      · max_depth: árbol más profundo en ciertos períodos puede indicar
        interacciones no lineales más fuertes (ej. combinaciones de régimen).
      · n_estimators: si cae con el tiempo el modelo converge más rápido,
        posiblemente porque los patrones son más fáciles de aprender.
      · min_child_weight: controla cuántos datos mínimos necesita cada hoja;
        valores altos → más regularización por tamaño de muestra.
      · subsample / colsample_bytree: nivel de stochasticity preferido por
        Optuna en cada época; inestabilidad aquí puede reflejar overfitting.
      · reg_alpha / reg_lambda: regularización L1/L2; si suben en folds
        recientes puede indicar mayor riesgo de sobreajuste (series más cortas
        relativas al ruido).

    Un hiperparámetro estable entre folds indica que el modelo aprende
    estructuras consistentes. Saltos bruscos señalan cambios de régimen
    que el modelo gestiona ajustando su complejidad interna.
    """
    if df_metricas.empty:
        return

    HP_CONFIG = [
        ("s_optimo",         "s (smoothing arctan)",       "steelblue",   "log"),
        ("learning_rate",    "learning rate",               "darkorange",  "log"),
        ("max_depth",        "max depth",                   "seagreen",    "linear"),
        ("n_estimators",     "n_estimators",                "crimson",     "linear"),
        ("min_child_weight", "min child weight",            "purple",      "linear"),
        ("subsample",        "subsample",                   "saddlebrown", "linear"),
        ("colsample_bytree", "colsample bytree",            "teal",        "linear"),
        ("reg_alpha",        "reg_alpha (L1)",              "goldenrod",   "log"),
        ("reg_lambda",       "reg_lambda (L2)",             "slategrey",   "log"),
    ]

    # Solo graficar hiperparámetros presentes en el CSV
    hp_presentes = [(col, lbl, clr, sc) for col, lbl, clr, sc in HP_CONFIG
                    if col in df_metricas.columns]
    if not hp_presentes:
        return

    n     = len(hp_presentes)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    folds = df_metricas["fold"].values

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    axes_flat = axes.flatten() if n > 1 else [axes]

    fig.suptitle(
        f"Estabilidad de hiperparámetros — Walk-forward CV — {banco}\n"
        f"TRAIN {VENTANA_TRAIN_AÑOS}yr / VAL {VENTANA_VAL_AÑOS}yr / "
        f"paso {PASO_AÑOS}yr / embargo {EMBARGO_DIAS_HAB}dh  "
        f"({len(folds)} folds)",
        fontweight="bold", fontsize=11,
    )

    for ax, (col, label, color, scale) in zip(axes_flat, hp_presentes):
        vals = df_metricas[col].values.astype(float)

        ax.plot(folds, vals, "o-", color=color, lw=2, ms=7, zorder=4, label=label)

        # Banda ±1 std alrededor de la media para referencia visual
        mu, sigma = vals.mean(), vals.std()
        ax.axhline(mu, color=color, lw=1.0, ls="--", alpha=0.5,
                   label=f"media={mu:.4g}")
        if sigma > 0:
            ax.axhspan(mu - sigma, mu + sigma, alpha=0.08, color=color,
                       label=f"±1σ ({sigma:.4g})")

        # Tendencia lineal
        if len(folds) >= 3:
            z    = np.polyfit(folds, vals if scale == "linear"
                              else np.log(np.maximum(vals, 1e-12)), 1)
            xfit = np.linspace(folds[0], folds[-1], 50)
            yfit = np.polyval(z, xfit)
            if scale == "log":
                yfit = np.exp(yfit)
            ax.plot(xfit, yfit, "-", color="black", lw=1.0, alpha=0.4,
                    label=f"tendencia")

        if scale == "log":
            ax.set_yscale("log")

        # Coeficiente de variación como indicador de inestabilidad
        cv = sigma / abs(mu) if abs(mu) > 1e-9 else 0.0
        ax.set_title(f"{label}  (CV={cv:.1%})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Fold", fontsize=8)
        ax.set_xticks(folds)
        ax.legend(fontsize=7, framealpha=0.85, loc="best")
        ax.grid(True, alpha=0.25)

    # Ocultar paneles sobrantes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    nombre = DIR_PLOTS / f"wfcv_hiperparametros_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico hiperparámetros guardado: {nombre}")


###############################################################################
# PARTE 8 — Pipeline principal por banco
###############################################################################

def evaluar_banco(banco: str):
    logger.info(f"\n{'='*65}")
    logger.info(f"BANCO: {banco}  — Walk-Forward CV con ventana rodante")
    logger.info(f"{'='*65}")
    logger.info(f"  Configuración:")
    logger.info(f"    TRAIN    : {VENTANA_TRAIN_AÑOS} años")
    logger.info(f"    VAL      : {VENTANA_VAL_AÑOS} años")
    logger.info(f"    Paso     : {PASO_AÑOS} años")
    logger.info(f"    Embargo  : {EMBARGO_DIAS_HAB} días hábiles")
    logger.info(f"    Trials   : {N_TRIALS_OPTUNA} por fold")
    logger.info(f"  Anti-leakage activo: EMBARGO + GARCH por fold + medianas por fold")

    t_inicio = time.time()

    # ── 1. Lectura ────────────────────────────────────────────────────────────
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    if df.empty or df["target"].notna().sum() < 500:
        logger.warning(f"  [{banco}] Datos insuficientes — omitiendo")
        return None

    cols_feat = get_feature_cols(df)
    fechas    = pd.DatetimeIndex(df["fecha_t"].unique())
    logger.info(f"  [{banco}] {len(df):,} filas | {len(cols_feat)} features | "
                f"{len(fechas)} fechas únicas")
    logger.info(f"  [{banco}] Rango: {fechas.min().date()} → {fechas.max().date()}")

    # ── 2. Generar folds ──────────────────────────────────────────────────────
    folds = generar_folds(
        fechas_disponibles=fechas,
        ventana_train_años=VENTANA_TRAIN_AÑOS,
        ventana_val_años=VENTANA_VAL_AÑOS,
        paso_años=PASO_AÑOS,
        embargo_dias_hab=EMBARGO_DIAS_HAB,
        corte_test=CORTE_TEST,
    )
    if not folds:
        logger.error(f"  [{banco}] No se generaron folds — revisa VENTANA_TRAIN_AÑOS")
        return None

    logger.info(f"  [{banco}] {len(folds)} folds generados:")
    for f in folds:
        logger.info(
            f"    Fold {f['fold']:2d} | TRAIN {f['train_start'].date()} → "
            f"{f['train_end'].date()} ({f['n_train_fechas']}dh) | "
            f"embargo {EMBARGO_DIAS_HAB}dh | "
            f"VAL {f['val_start'].date()} → {f['val_end'].date()} "
            f"({f['n_val_fechas']}dh)"
        )

    # ── 3. Loop de folds ──────────────────────────────────────────────────────
    resultados_fold   = []
    resultados_por_h  = []
    modelos_ultimo    = None
    params_ultimo     = None

    for fold in folds:
        t_fold = time.time()
        logger.info(f"\n  ── Fold {fold['fold']}/{len(folds)} ──────────────────────")

        # 3a. Preparar datos con anti-leakage
        try:
            X_train, y_train, X_val, y_val, h_train, h_val = preparar_fold_data(
                df, fold, cols_feat
            )
        except Exception as e:
            logger.warning(f"  Fold {fold['fold']}: error preparando datos — {e}")
            continue

        if len(X_train) < 200 or len(X_val) < 20:
            logger.warning(f"  Fold {fold['fold']}: datos insuficientes "
                           f"(train={len(X_train)}, val={len(X_val)}) — omitiendo")
            continue

        std_y = float(y_train.std())
        logger.info(f"    X_train={len(X_train):,} | X_val={len(X_val):,} | "
                    f"std_y={std_y:,.0f}")

        # Registrar cuántas observaciones GARCH fueron re-estimadas
        garch_cols_presentes = [c for c in ["garch_vol", "garch_vol_tc", "garch_vol_embi"]
                                 if c in cols_feat]
        if garch_cols_presentes:
            logger.info(f"    GARCH re-estimado (medida #2): {garch_cols_presentes}")
        logger.info(f"    Medianas imputación (medida #3): calculadas sobre "
                    f"{fold['n_train_fechas']} fechas TRAIN")

        # 3b-3d. Entrenar y predecir — dispatcher según MODELO_CV
        modelos, best_params = entrenar_fold(
            X_train, y_train, X_val, y_val, std_y,
            N_TRIALS_OPTUNA, fold["fold"]
        )
        preds = predecir_fold(modelos, X_val)

        # 3e. Métricas
        row_fold = calcular_metricas_fold(preds, y_val.values, fold)
        row_fold["tiempo_min"]  = round((time.time() - t_fold) / 60, 2)
        row_fold["modelo_cv"]   = MODELO_CV
        # Hiperparámetros óptimos — columnas normalizadas para HP stability plot.
        # lgbm: num_leaves→max_depth col, min_child_samples→min_child_weight col
        # xgb_qt: best_params = Q50 best (re: entrenar_fold dispatcher)
        if MODELO_CV == "lgbm":
            row_fold["s_optimo"]         = 0.0   # N/A (no arctan pinball)
            row_fold["learning_rate"]    = round(best_params.get("learning_rate",     0), 4)
            row_fold["max_depth"]        = int(  best_params.get("num_leaves",        0))
            row_fold["n_estimators"]     = int(  best_params.get("n_estimators",      0))
            row_fold["min_child_weight"] = int(  best_params.get("min_child_samples", 0))
            row_fold["subsample"]        = round(best_params.get("subsample",         0), 3)
            row_fold["colsample_bytree"] = round(best_params.get("colsample_bytree",  0), 3)
            row_fold["reg_alpha"]        = round(best_params.get("reg_alpha",         0), 5)
            row_fold["reg_lambda"]       = round(best_params.get("reg_lambda",        0), 5)
        else:  # "xgb" y "xgb_qt" comparten la misma estructura de params
            row_fold["s_optimo"]         = round(best_params.get("s",                0), 4)
            row_fold["learning_rate"]    = round(best_params.get("learning_rate",    0), 4)
            row_fold["max_depth"]        = int(  best_params.get("max_depth",        0))
            row_fold["n_estimators"]     = int(  best_params.get("n_estimators",     0))
            row_fold["min_child_weight"] = int(  best_params.get("min_child_weight", 0))
            row_fold["subsample"]        = round(best_params.get("subsample",        0), 3)
            row_fold["colsample_bytree"] = round(best_params.get("colsample_bytree", 0), 3)
            row_fold["reg_alpha"]        = round(best_params.get("reg_alpha",        0), 5)
            row_fold["reg_lambda"]       = round(best_params.get("reg_lambda",       0), 5)
        resultados_fold.append(row_fold)

        df_h = calcular_metricas_por_h(preds, y_val.values, h_val, fold["fold"])
        resultados_por_h.append(df_h)

        logger.info(
            f"    Fold {fold['fold']} resultados: "
            f"pinball_Q50={row_fold.get('pinball_q50', '?'):,.0f}  "
            f"coverage_90={row_fold.get('coverage_90', 0):.1%}  "
            f"winkler={row_fold.get('winkler_90', '?'):,.0f}  "
            f"crps≈{row_fold.get('crps_approx', '?'):,.0f}  "
            f"({row_fold['tiempo_min']} min)"
        )

        # Guardar modelos del último fold para posible uso como modelo final
        modelos_ultimo = modelos
        params_ultimo  = best_params

        del X_train, y_train, X_val, y_val
        gc.collect()

    if not resultados_fold:
        logger.error(f"  [{banco}] Ningún fold completado")
        return None

    # ── 4. Consolidar resultados ──────────────────────────────────────────────
    df_metricas = pd.DataFrame(resultados_fold)
    df_por_h    = pd.concat(resultados_por_h, ignore_index=True) if resultados_por_h else pd.DataFrame()

    # ── 5. Guardar CSVs ───────────────────────────────────────────────────────
    fecha_hoy = pd.Timestamp.today().strftime("%Y%m%d")
    # Todos los CSVs y plots van a la subcarpeta del modelo
    ruta_metricas = DIR_MODELO_OUTPUT / f"wfcv_metricas_{banco}_{fecha_hoy}.csv"
    ruta_por_h    = DIR_MODELO_OUTPUT / f"wfcv_metricas_por_h_{banco}_{fecha_hoy}.csv"
    df_metricas.to_csv(ruta_metricas, index=False)
    if not df_por_h.empty:
        df_por_h.to_csv(ruta_por_h, index=False)

    # CSV de hiperparámetros separado para análisis de estabilidad
    cols_hp = ["fold", "train_start", "train_end", "modelo_cv",
               "s_optimo", "learning_rate", "max_depth", "n_estimators",
               "min_child_weight", "subsample", "colsample_bytree",
               "reg_alpha", "reg_lambda"]
    cols_hp_ok = [c for c in cols_hp if c in df_metricas.columns]
    ruta_hp = DIR_MODELO_OUTPUT / f"wfcv_hiperparametros_{banco}_{fecha_hoy}.csv"
    df_metricas[cols_hp_ok].to_csv(ruta_hp, index=False)

    logger.info(f"  [{banco}] Archivos guardados en: {DIR_MODELO_OUTPUT}")
    logger.info(f"    {ruta_metricas.name}")
    logger.info(f"    {ruta_por_h.name}")
    logger.info(f"    {ruta_hp.name}")

    # ── 6. Gráficos ───────────────────────────────────────────────────────────
    graficar_metricas_wfcv(df_metricas, f"{banco}_{MODELO_CV}")
    graficar_cobertura_por_h(df_por_h, f"{banco}_{MODELO_CV}")
    graficar_hiperparametros_wfcv(df_metricas, f"{banco}_{MODELO_CV}")

    # ── 7. Modelo final (opcional) ────────────────────────────────────────────
    if GUARDAR_MODELO_FINAL and modelos_ultimo is not None:
        ultimo_fold = folds[-1]
        logger.info(
            f"  [{banco}] Guardando modelo final (TRAIN del último fold: "
            f"{ultimo_fold['train_start'].date()} → {ultimo_fold['train_end'].date()})"
        )
        for tau, model in modelos_ultimo.items():
            if MODELO_CV == "lgbm":
                ruta_m = DIR_MODELOS / f"lgbm_wfcv_{banco}_q{int(tau*100):02d}_{fecha_hoy}.txt"
                model.save_model(str(ruta_m))
            else:
                ruta_m = DIR_MODELOS / f"{MODELO_CV}_wfcv_{banco}_q{int(tau*100):02d}_{fecha_hoy}.json"
                model.save_model(str(ruta_m))

        metadata = {
            "banco"                : banco,
            "modelo"               : f"{MODELO_CV}_wfcv",
            "fecha_entrenamiento"  : pd.Timestamp.today().strftime("%Y-%m-%d"),
            "config": {
                "ventana_train_años" : VENTANA_TRAIN_AÑOS,
                "ventana_val_años"   : VENTANA_VAL_AÑOS,
                "paso_años"          : PASO_AÑOS,
                "embargo_dias_hab"   : EMBARGO_DIAS_HAB,
                "n_trials_optuna"    : N_TRIALS_OPTUNA,
            },
            "anti_leakage": {
                "embargo"           : f"{EMBARGO_DIAS_HAB} días hábiles post-TRAIN",
                "garch_por_fold"    : "ω/α/β estimados solo sobre TRAIN del fold",
                "medianas_por_fold" : "imputación calculada sobre TRAIN del fold",
            },
            "ultimo_fold" : {
                "train_start": str(ultimo_fold["train_start"].date()),
                "train_end"  : str(ultimo_fold["train_end"].date()),
            },
            "n_folds"   : len(folds),
            "quantiles" : QUANTILES,
            "features"  : cols_feat,
            "best_params_ultimo_fold": params_ultimo,
        }
        ruta_meta = DIR_MODELOS / f"metadata_{MODELO_CV}_wfcv_{banco}_{fecha_hoy}.json"
        with open(ruta_meta, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)
        logger.info(f"    Modelo final guardado en: {DIR_MODELOS}")

    # ── 8. Resumen en consola ─────────────────────────────────────────────────
    t_total = time.time() - t_inicio
    logger.info(f"\n  {'─'*55}")
    logger.info(f"  RESUMEN WALK-FORWARD CV — {banco}")
    logger.info(f"  {'─'*55}")

    cols_display = ["fold", "train_start", "train_end", "val_start",
                    "pinball_q50", "coverage_90", "winkler_90", "crps_approx"]
    cols_ok = [c for c in cols_display if c in df_metricas.columns]
    logger.info("\n" + df_metricas[cols_ok].to_string(index=False))

    promedios = {c: df_metricas[c].mean()
                 for c in ["pinball_q50", "coverage_90", "winkler_90", "crps_approx"]
                 if c in df_metricas.columns}
    logger.info(f"\n  Promedios entre folds:")
    for k, v in promedios.items():
        fmt = f"{v:.1%}" if k == "coverage_90" else f"{v:,.2f}"
        logger.info(f"    {k:20s}: {fmt}")

    logger.info(f"\n  ✓ Completado en {t_total/60:.1f} min  ({len(folds)} folds)")
    return df_metricas


###############################################################################
# Main
###############################################################################

def main():
    logger.info("=" * 65)
    logger.info(f"STEP005 — Walk-Forward CV con ventana rodante  [MODELO: {MODELO_CV.upper()}]")
    logger.info("=" * 65)
    if MODELO_CV == "lgbm" and not _LGBM_OK:
        logger.error("MODELO_CV='lgbm' pero lightgbm no está instalado.  pip install lightgbm")
        return
    logger.info("Medidas anti-leakage activas:")
    logger.info(f"  #1 Embargo          : {EMBARGO_DIAS_HAB} días hábiles")
    logger.info(f"  #2 GARCH por fold   : garch_vol / garch_vol_tc / garch_vol_embi")
    logger.info(f"  #3 Medianas fold    : imputación sobre TRAIN del fold")
    logger.info(f"Configuración de folds:")
    logger.info(f"  TRAIN {VENTANA_TRAIN_AÑOS}yr | VAL {VENTANA_VAL_AÑOS}yr | "
                f"paso {PASO_AÑOS}yr | TEST fijo desde {CORTE_TEST.date()}")

    if not RUTA_MATRIZ.exists():
        logger.error(f"Matriz no encontrada: {RUTA_MATRIZ}")
        logger.error("Ejecutar step001_build_feature_matrix.py primero.")
        return

    t0 = time.time()
    todos_resultados = []
    for banco in BANCOS_A_EVALUAR:
        df_m = evaluar_banco(banco)
        if df_m is not None:
            todos_resultados.append((banco, df_m))

    if todos_resultados:
        logger.info("\n" + "=" * 65)
        logger.info("RESUMEN GLOBAL")
        logger.info("=" * 65)
        for banco, df_m in todos_resultados:
            avg_cov = df_m["coverage_90"].mean() if "coverage_90" in df_m.columns else float("nan")
            avg_pb  = df_m["pinball_q50"].mean() if "pinball_q50" in df_m.columns else float("nan")
            logger.info(f"  {banco:15s}: {len(df_m)} folds | "
                        f"pinball_Q50_avg={avg_pb:,.0f} | coverage_90_avg={avg_cov:.1%}")

    logger.info(f"\n✓ Total: {(time.time()-t0)/60:.1f} min")
    logger.info(f"  Resultados en: {DIR_MODELO_OUTPUT}")


if __name__ == "__main__":
    main()
