# -*- coding: utf-8 -*-
"""
step005_walk_forward_cv_2.py
Walk-forward CV con ventana rodante — evaluación VERDADERAMENTE out-of-sample.

═══════════════════════════════════════════════════════════════════════════════
DIFERENCIA CLAVE RESPECTO A step005_walk_forward_cv.py (v1)
═══════════════════════════════════════════════════════════════════════════════

v1 — VAL sirve para DOS cosas simultáneamente:
  1. Optuna selecciona hiperparámetros minimizando pinball en VAL
  2. Coverage / Winkler / CRPS se calculan sobre ese mismo VAL
  → Sesgo optimista: los HP fueron elegidos porque funcionaron bien en VAL.
    El coverage reportado es ~1-3 pp mayor que el real.

v2 — Separación estricta de roles:
  ├── VAL  (VENTANA_VAL_AÑOS)  → SOLO para Optuna. Nunca se reporta como métrica.
  └── TEST (VENTANA_TEST_AÑOS) → SOLO para métricas. Nunca influye en tuning.

  Estructura por fold:
  |──── TRAIN (5yr) ────|── embargo 90dh ──|── VAL (1yr) ──|── TEST (1yr) ──|
                                                 Optuna           métricas

  TEST comienza donde termina VAL → sin leakage (el modelo fue entrenado solo
  en TRAIN; los hiperparámetros se eligieron en VAL; TEST es futuro puro).
  No se necesita embargo adicional entre VAL y TEST porque las confirmed
  features solo pueden filtrar TRAIN→VAL, no VAL→TEST.

═══════════════════════════════════════════════════════════════════════════════
MEDIDAS ANTI-LEAKAGE — idénticas a v1 (preservadas sin cambio)
═══════════════════════════════════════════════════════════════════════════════

  #1 EMBARGO 90dh  entre TRAIN y VAL (= h_max)
  #2 GARCH por fold  ω/α/β estimados solo sobre TRAIN del fold
  #3 Medianas fold   imputación calculada sobre TRAIN del fold

  Estas tres medidas se aplican igual a VAL y TEST: los parámetros GARCH
  y las medianas de imputación se estiman en TRAIN y se propagan hacia
  adelante (VAL y TEST) sin ver datos futuros.

═══════════════════════════════════════════════════════════════════════════════
Parámetros configurables
═══════════════════════════════════════════════════════════════════════════════
  VENTANA_TRAIN_AÑOS : años de historia de entrenamiento por fold
  VENTANA_VAL_AÑOS   : años de VAL por fold (solo para Optuna)
  VENTANA_TEST_AÑOS  : años de TEST por fold (métricas out-of-sample)
  PASO_AÑOS          : desplazamiento entre folds
  EMBARGO_DIAS_HAB   : días hábiles de buffer post-TRAIN (= h_max)
  N_TRIALS_OPTUNA    : trials Optuna por fold
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
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v2"   # carpeta separada de v1

# ── Ventana rodante ───────────────────────────────────────────────────────────
VENTANA_TRAIN_AÑOS  = 5      # años de TRAIN por fold
VENTANA_VAL_AÑOS    = 1      # años de VAL por fold  (SOLO para Optuna)
VENTANA_TEST_AÑOS   = 1      # años de TEST por fold (SOLO para métricas OOS)
PASO_AÑOS           = 1      # desplazamiento entre folds
EMBARGO_DIAS_HAB    = 90     # días hábiles entre fin-TRAIN e inicio-VAL (= h_max)

# ── Modelo ────────────────────────────────────────────────────────────────────
QUANTILES        = [0.01, 0.05, 0.50, 0.95, 0.99]
N_TRIALS_OPTUNA  = 30
S_MIN_FACTOR     = 0.01
S_MAX_FACTOR     = 1.0

# ── Opciones de salida ────────────────────────────────────────────────────────
BANCOS_A_EVALUAR     = ["SISTEMA"]
GUARDAR_MODELO_FINAL = True
COLS_EXCLUIR         = {"fecha_t", "banco", "target"}

# ── Selector de modelo ────────────────────────────────────────────────────────
MODELO_CV = "xgb"
# Opciones: "xgb" | "lgbm" | "xgb_qt"
assert MODELO_CV in ("xgb", "lgbm", "xgb_qt"), \
    f"MODELO_CV debe ser 'xgb', 'lgbm' o 'xgb_qt', recibido: {MODELO_CV!r}"

# ── Regularización Optuna ─────────────────────────────────────────────────────
FIX_REG_ALPHA  = True   # reg_alpha=0 fijo (CV>200%, paisaje plano, sin efecto real)
FIX_REG_LAMBDA = False  # False=búsqueda log [1e-4,10]; True demostró empeorar folds 5-8

# ── Rutas de salida ───────────────────────────────────────────────────────────
DIR_MODELO_OUTPUT = DIR_OUTPUT / MODELO_CV
DIR_MODELOS       = DIR_MODELO_OUTPUT / "modelos"
DIR_PLOTS         = DIR_MODELO_OUTPUT / "plots"

for _d in (DIR_OUTPUT, DIR_MODELO_OUTPUT, DIR_MODELOS, DIR_PLOTS):
    _d.mkdir(parents=True, exist_ok=True)


###############################################################################
# PARTE 1 — Utilidades métricas  (sin cambios respecto a v1)
###############################################################################

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def winkler_score(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray,
                  alpha: float = 0.10) -> float:
    width = q_hi - q_lo
    below = np.maximum(q_lo - y_true, 0)
    above = np.maximum(y_true - q_hi, 0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def coverage(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    return float(np.mean((y_true >= q_lo) & (y_true <= q_hi)))


def crps_approx(y_true: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    taus = sorted(preds.keys())
    pbs  = [pinball_loss(y_true, preds[tau], tau) for tau in taus]
    n    = len(taus)
    pesos = []
    for i in range(n):
        lo = taus[i - 1] if i > 0     else taus[0]
        hi = taus[i + 1] if i < n - 1 else taus[-1]
        pesos.append((hi - lo) / 2.0)
    return float(2.0 * sum(w * pb for w, pb in zip(pesos, pbs)))


###############################################################################
# PARTE 2 — GARCH por fold  (sin cambios respecto a v1)
###############################################################################

def _ajustar_garch_params(x_train: np.ndarray) -> tuple[float, float, float]:
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


def _garch_vol_fold(serie: pd.Series, train_end: pd.Timestamp) -> pd.Series:
    serie_full  = serie.ffill().fillna(0.0)
    serie_train = serie_full[serie_full.index <= train_end]

    if len(serie_train) < 60 or serie_train.std() < 1e-9:
        return serie_full.rolling(20).std().fillna(serie_train.std())

    escala   = float(serie_train.std())
    x_train  = (serie_train / escala).values.astype(float)
    var_unc  = max(float(np.var(x_train)), 1e-12)

    omega, alpha, beta = _ajustar_garch_params(x_train)

    x_full = (serie_full / escala).values.astype(float)
    n_full = len(x_full)
    s2     = np.empty(n_full)
    s2[0]  = var_unc
    for t in range(1, n_full):
        s2[t] = omega + alpha * x_full[t - 1] ** 2 + beta * s2[t - 1]

    return pd.Series(np.sqrt(np.maximum(s2, 0)) * escala, index=serie_full.index)


def reemplazar_garch_fold(df_fold: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    df_fold = df_fold.copy()

    idx_cols = ["fecha_t", "R_t0", "D_t0", "TC_PEN_USD", "EMBI_PERU"]
    avail    = [c for c in idx_cols if c in df_fold.columns]
    raw      = (df_fold[avail]
                .drop_duplicates("fecha_t")
                .set_index("fecha_t")
                .sort_index())

    if {"R_t0", "D_t0"}.issubset(raw.columns):
        flujo       = raw["D_t0"] - raw["R_t0"]
        sigma_flujo = _garch_vol_fold(flujo, train_end)
        df_fold["garch_vol"] = df_fold["fecha_t"].map(sigma_flujo)

    if "TC_PEN_USD" in raw.columns:
        tc = raw["TC_PEN_USD"].replace(0, np.nan).ffill()
        tc_reidx    = tc.reindex(pd.bdate_range(tc.index.min(), tc.index.max()))
        tc_reidx    = tc_reidx.ffill()
        retornos_tc = np.log(tc_reidx / tc_reidx.shift(1)).reindex(tc.index)
        sigma_tc    = _garch_vol_fold(retornos_tc, train_end)
        df_fold["garch_vol_tc"] = df_fold["fecha_t"].map(sigma_tc)

    if "EMBI_PERU" in raw.columns:
        delta_embi  = raw["EMBI_PERU"].diff(1)
        sigma_embi  = _garch_vol_fold(delta_embi, train_end)
        df_fold["garch_vol_embi"] = df_fold["fecha_t"].map(sigma_embi)

    return df_fold


###############################################################################
# PARTE 3 — Generación de folds  (MODIFICADO: agrega TEST window)
###############################################################################

def generar_folds(
    fechas_disponibles: pd.DatetimeIndex,
    ventana_train_años: float,
    ventana_val_años: float,
    ventana_test_años: float,
    paso_años: float,
    embargo_dias_hab: int,
) -> list[dict]:
    """
    Genera folds de walk-forward CV con ventana rodante.

    Estructura por fold (v2):
      ├── TRAIN  : [train_start, train_end]   → entrenamiento
      ├── EMBARGO: (train_end, val_start)      → excluido
      ├── VAL    : [val_start, val_end]        → Optuna (tuning ÚNICAMENTE)
      └── TEST   : [test_start, test_end]      → métricas out-of-sample

    TEST comienza justo donde termina VAL, sin embargo adicional.
    El fold se genera solo si test_end ≤ última fecha disponible.
    """
    folds       = []
    f_min       = fechas_disponibles.min()
    f_max       = fechas_disponibles.max()
    paso        = pd.DateOffset(months=int(round(paso_años * 12)))
    train_start = f_min
    fold_num    = 1

    while True:
        train_end  = train_start + pd.DateOffset(months=int(round(ventana_train_años * 12)))
        val_start  = train_end   + pd.offsets.BusinessDay(embargo_dias_hab)
        val_end    = val_start   + pd.DateOffset(months=int(round(ventana_val_años * 12)))
        test_start = val_end
        test_end   = test_start  + pd.DateOffset(months=int(round(ventana_test_años * 12)))

        # Detener si TEST supera los datos disponibles
        if test_end > f_max:
            break
        if train_end >= f_max or val_start >= f_max:
            break

        n_train = ((fechas_disponibles >= train_start) &
                   (fechas_disponibles <= train_end)).sum()
        n_val   = ((fechas_disponibles >= val_start) &
                   (fechas_disponibles <  test_start)).sum()
        n_test  = ((fechas_disponibles >= test_start) &
                   (fechas_disponibles <= test_end)).sum()

        if n_train < 60 or n_val < 10 or n_test < 10:
            train_start += paso
            continue

        folds.append({
            "fold"           : fold_num,
            "train_start"    : train_start,
            "train_end"      : train_end,
            "val_start"      : val_start,
            "val_end"        : val_end,
            "test_start"     : test_start,
            "test_end"       : test_end,
            "n_train_fechas" : int(n_train),
            "n_val_fechas"   : int(n_val),
            "n_test_fechas"  : int(n_test),
        })
        fold_num    += 1
        train_start += paso

    return folds


###############################################################################
# PARTE 4 — Objetivo arctan pinball y entrenamiento  (sin cambios)
###############################################################################

def make_quantile_objective(tau: float, s: float, std_y: float):
    _scale = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)

    def objective(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        u    = dtrain.get_label() - y_pred
        grad = -((tau - 0.5 + np.arctan(u / s) / np.pi)
                 + u * s / (np.pi * (s ** 2 + u ** 2))) * _scale
        hess = 2 * s ** 3 / (np.pi * (s ** 2 + u ** 2) ** 2) * _scale
        return grad, hess
    return objective


def make_pinball_metric(tau: float):
    def metric(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label()
        u      = y_true - y_pred
        loss   = float(np.mean(np.where(u >= 0, tau * u, (tau - 1) * u)))
        return "pinball", loss
    return metric


def _objective_optuna(trial, X_tr, y_tr, X_va, y_va, std_y):
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
    n_est  = trial.suggest_int("n_estimators", 100, 1000)
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
    logger.info(f"    Optuna fold {fold_num} ({n_trials} trials, τ=0.50 en VAL)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold_num),
    )
    study.optimize(
        lambda t: _objective_optuna(t, X_tr, y_tr, X_va, y_va, std_y),
        n_trials=n_trials, show_progress_bar=False,
    )
    bp = study.best_params
    logger.info(f"    Fold {fold_num} best pinball(Q50/VAL)={study.best_value:.4f} "
                f"n_est={bp['n_estimators']} lr={bp['learning_rate']:.4f} s={bp['s']:.2f}")
    return bp


def entrenar_quantiles(X_tr, y_tr, best_params, quantiles, std_y):
    s_best = best_params["s"]
    n_est  = best_params["n_estimators"]
    params = {k: v for k, v in best_params.items() if k not in ("s", "n_estimators")}
    params.update({"tree_method": "hist", "seed": 42})
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    modelos = {}
    for tau in quantiles:
        modelos[tau] = xgb.train(
            params, dtrain, num_boost_round=n_est,
            obj=make_quantile_objective(tau, s_best, std_y),
            verbose_eval=False,
        )
    return modelos


def predecir_y_corregir(modelos, X):
    dmat      = xgb.DMatrix(X)
    preds_raw = {tau: m.predict(dmat) for tau, m in modelos.items()}
    taus      = sorted(preds_raw)
    matrix    = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    return {t: matrix[:, i] for i, t in enumerate(taus)}


###############################################################################
# PARTE 4b — LightGBM y XGBoost QT  (sin cambios)
###############################################################################

def _objetivo_optuna_lgbm(trial, X_tr, y_tr, X_va, y_va):
    if not _LGBM_OK:
        raise ImportError("lightgbm no está instalado")
    tau = 0.50
    params = {
        "objective"        : "quantile",
        "alpha"            : tau,
        "verbosity"        : -1,
        "seed"             : 42,
        "learning_rate"    : trial.suggest_float("learning_rate",    0.01,  0.3,  log=True),
        "num_leaves"       : trial.suggest_int(  "num_leaves",        15,   255),
        "max_depth"        : trial.suggest_int(  "max_depth",          3,    10),
        "min_child_samples": trial.suggest_int(  "min_child_samples", 10,   200),
        "subsample"        : trial.suggest_float("subsample",         0.5,   1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree",  0.4,  1.0),
        "reg_alpha"        : 0.0 if FIX_REG_ALPHA else trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 0.1, 5.0) if FIX_REG_LAMBDA else trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "subsample_freq"   : 1,
    }
    n_est  = trial.suggest_int("n_estimators", 100, 1000)
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
    logger.info(f"    Optuna[LGBM] fold {fold_num} ({n_trials} trials, τ=0.50 en VAL)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold_num),
    )
    study.optimize(
        lambda t: _objetivo_optuna_lgbm(t, X_tr, y_tr, X_va, y_va),
        n_trials=n_trials, show_progress_bar=False,
    )
    bp = study.best_params
    logger.info(f"    Fold {fold_num} [LGBM] best pinball(Q50/VAL)={study.best_value:.4f} "
                f"n_est={bp['n_estimators']} lr={bp['learning_rate']:.4f}")
    return bp


def entrenar_quantiles_lgbm(X_tr, y_tr, best_params, quantiles):
    if not _LGBM_OK:
        raise ImportError("lightgbm no está instalado")
    n_est  = int(best_params.get("n_estimators", 300))
    hp     = {k: v for k, v in best_params.items() if k != "n_estimators"}
    modelos = {}
    for tau in quantiles:
        params = {"objective": "quantile", "alpha": tau,
                  "verbosity": -1, "seed": 42, "subsample_freq": 1, **hp}
        dtrain = lgb.Dataset(X_tr.values, label=y_tr.values)
        modelos[tau] = lgb.train(
            params, dtrain, num_boost_round=n_est,
            callbacks=[lgb.log_evaluation(-1)],
        )
    return modelos


def predecir_lgbm(modelos, X):
    preds_raw = {tau: m.predict(X.values) for tau, m in modelos.items()}
    taus   = sorted(preds_raw)
    matrix = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    return {t: matrix[:, i] for i, t in enumerate(taus)}


def _objetivo_optuna_xgb_qt_tau(trial, tau, X_tr, y_tr, X_va, y_va, std_y):
    s = trial.suggest_float("s", std_y * S_MIN_FACTOR, std_y * S_MAX_FACTOR, log=True)
    params = {
        "learning_rate"   : trial.suggest_float("learning_rate",   0.01,  0.3,  log=True),
        "max_depth"       : trial.suggest_int(  "max_depth",         3,    10),
        "min_child_weight": trial.suggest_int(  "min_child_weight", 10,   200),
        "colsample_bytree": trial.suggest_float("colsample_bytree",  0.4,  1.0),
        "subsample"       : trial.suggest_float("subsample",         0.5,  1.0),
        "reg_alpha"       : 0.0 if FIX_REG_ALPHA else trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
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
                    f"pinball/VAL={study.best_value:.4f} s={bp['s']:.3f}")
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
    best_q50 = best_by_tau.get(0.50, list(best_by_tau.values())[0])
    return modelos, best_q50


def entrenar_fold(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
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
    if MODELO_CV == "lgbm":
        return predecir_lgbm(modelos, X)
    return predecir_y_corregir(modelos, X)


###############################################################################
# PARTE 5 — Preparación de datos  (MODIFICADO: extrae también TEST)
###############################################################################

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    excluir = COLS_EXCLUIR | {"fecha_th"}
    return [c for c in df.columns
            if c not in excluir and df[c].dtype.kind in ("f", "i", "u", "b")]


def preparar_fold_data(
    df: pd.DataFrame,
    fold: dict,
    cols_feat: list[str],
):
    """
    Extrae TRAIN, VAL y TEST aplicando las 3 medidas anti-leakage.

    Anti-leakage aplicado al período TEST:
      · Los parámetros GARCH (ω/α/β) se estiman solo sobre TRAIN y se propagan
        recursivamente hasta el final de TEST — sin ver datos futuros.
      · Las medianas de imputación se calculan solo sobre TRAIN y se aplican
        a VAL y TEST por igual.
      · TEST no tiene acceso a información de VAL durante ningún paso.

    Retorna:
      X_train, y_train, X_val, y_val, X_test, y_test,
      h_train_arr, h_val_arr, h_test_arr
    """
    train_start = fold["train_start"]
    train_end   = fold["train_end"]
    val_start   = fold["val_start"]
    val_end     = fold["val_end"]
    test_start  = fold["test_start"]
    test_end    = fold["test_end"]

    # Medida #1 — Embargo: excluir (train_end, val_start)
    mask_train = (df["fecha_t"] >= train_start) & (df["fecha_t"] <= train_end)
    mask_val   = (df["fecha_t"] >= val_start)   & (df["fecha_t"] <  test_start)
    mask_test  = (df["fecha_t"] >= test_start)  & (df["fecha_t"] <= test_end)

    df_fold_all = df[mask_train | mask_val | mask_test].copy()

    # Medida #2 — GARCH por fold (parámetros estimados solo en TRAIN,
    #             propagados recursivamente sobre VAL y TEST)
    garch_cols = [c for c in ["garch_vol", "garch_vol_tc", "garch_vol_embi"]
                  if c in df_fold_all.columns]
    if garch_cols:
        df_fold_all = reemplazar_garch_fold(df_fold_all, train_end)

    df_train = df_fold_all[df_fold_all["fecha_t"] <= train_end]
    df_val   = df_fold_all[(df_fold_all["fecha_t"] >= val_start) &
                           (df_fold_all["fecha_t"] <  test_start)]
    df_test  = df_fold_all[df_fold_all["fecha_t"] >= test_start]

    # Medida #3 — Medianas solo de TRAIN, aplicadas a VAL y TEST
    medianas_fold = df_train[cols_feat].median()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()
    df_train[cols_feat] = df_train[cols_feat].fillna(medianas_fold)
    df_val[cols_feat]   = df_val[cols_feat].fillna(medianas_fold)
    df_test[cols_feat]  = df_test[cols_feat].fillna(medianas_fold)

    mask_t  = df_train["target"].notna()
    mask_v  = df_val["target"].notna()
    mask_te = df_test["target"].notna()

    X_train = df_train.loc[mask_t,  cols_feat].copy()
    y_train = df_train.loc[mask_t,  "target"].copy()
    h_train = df_train.loc[mask_t,  "h"].values

    X_val   = df_val.loc[mask_v,    cols_feat].copy()
    y_val   = df_val.loc[mask_v,    "target"].copy()
    h_val   = df_val.loc[mask_v,    "h"].values

    X_test  = df_test.loc[mask_te,  cols_feat].copy()
    y_test  = df_test.loc[mask_te,  "target"].copy()
    h_test  = df_test.loc[mask_te,  "h"].values

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            h_train, h_val, h_test)


###############################################################################
# PARTE 6 — Métricas por fold  (sin cambios en la lógica, etiquetas actualizadas)
###############################################################################

def calcular_metricas_fold(
    preds: dict[float, np.ndarray],
    y_true: np.ndarray,
    fold: dict,
    periodo: str = "test",   # "val" o "test"
) -> dict:
    """Métricas globales (todos los horizontes) para un período dado."""
    start_key = f"{periodo}_start"
    end_key   = f"{periodo}_end"
    n_key     = f"n_{periodo}_fechas"

    row = {
        "fold"          : fold["fold"],
        "train_start"   : fold["train_start"].date(),
        "train_end"     : fold["train_end"].date(),
        "val_start"     : fold["val_start"].date(),
        "val_end"       : fold["val_end"].date(),
        "test_start"    : fold["test_start"].date(),
        "test_end"      : fold["test_end"].date(),
        "n_train_fechas": fold["n_train_fechas"],
        "n_val_fechas"  : fold["n_val_fechas"],
        "n_test_fechas" : fold["n_test_fechas"],
        f"n_obs_{periodo}": int(len(y_true)),
        "periodo_metricas": periodo,
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
# PARTE 7 — Visualización  (títulos actualizados para reflejar TEST OOS)
###############################################################################

def graficar_metricas_wfcv(df_metricas: pd.DataFrame, banco: str):
    """Métricas TEST (out-of-sample) por fold."""
    if df_metricas.empty:
        return

    folds = df_metricas["fold"].values
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Walk-forward CV v2 — {banco}  [métricas TEST out-of-sample]\n"
        f"TRAIN {VENTANA_TRAIN_AÑOS}yr / VAL {VENTANA_VAL_AÑOS}yr (Optuna) / "
        f"TEST {VENTANA_TEST_AÑOS}yr (OOS) / paso {PASO_AÑOS}yr / embargo {EMBARGO_DIAS_HAB}dh",
        fontweight="bold", fontsize=11,
    )

    metricas_config = [
        ("pinball_q50", "Pinball Q50 — TEST OOS",    "steelblue",  None),
        ("coverage_90", "Coverage 90% — TEST OOS",   "seagreen",   0.90),
        ("winkler_90",  "Winkler score 90% — TEST",  "darkorange", None),
        ("crps_approx", "CRPS aproximado — TEST",    "crimson",    None),
    ]

    for ax, (col, titulo, color, hline) in zip(axes, metricas_config):
        if col not in df_metricas.columns:
            ax.set_visible(False)
            continue
        vals = df_metricas[col].values.astype(float)
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
            lambda x, _, _col=col: f"{x:,.0f}" if _col != "coverage_90" else f"{x:.1%}"
        ))
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v2_metricas_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico métricas TEST guardado: {nombre}")


def graficar_comparacion_val_test(df_val: pd.DataFrame, df_test: pd.DataFrame, banco: str):
    """
    Compara coverage y pinball Q50 entre VAL (sesgo optimista) y TEST (OOS real).
    Permite cuantificar empíricamente el sesgo de evaluar en el mismo período
    que se usó para Optuna.
    """
    if df_val.empty or df_test.empty:
        return
    if "coverage_90" not in df_val.columns or "coverage_90" not in df_test.columns:
        return

    folds = df_test["fold"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"VAL (Optuna, sesgo optimista) vs TEST (out-of-sample real) — {banco}",
        fontweight="bold", fontsize=11,
    )

    for ax, col, titulo, fmt_fn in [
        (axes[0], "coverage_90", "Coverage 90%",    lambda x: f"{x:.1%}"),
        (axes[1], "pinball_q50", "Pinball Q50",     lambda x: f"{x:,.0f}"),
    ]:
        if col not in df_val.columns or col not in df_test.columns:
            continue
        v_val  = df_val[col].values.astype(float)
        v_test = df_test[col].values.astype(float)
        ax.plot(folds, v_val,  "s--", color="steelblue",  lw=1.5, ms=7, label="VAL (Optuna)")
        ax.plot(folds, v_test, "o-",  color="crimson",    lw=2,   ms=7, label="TEST (OOS)")

        if col == "coverage_90":
            ax.axhline(0.90, color="grey", lw=1.2, ls=":", alpha=0.7, label="nominal 90%")
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.set_xlabel("Fold", fontsize=9)
        ax.set_xticks(folds)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        if col == "coverage_90":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))

        # Anotar diferencia media
        diff = float(np.mean(v_val - v_test))
        ax.set_xlabel(f"Fold  (sesgo promedio VAL−TEST: {diff:+.4f})", fontsize=9)

    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v2_val_vs_test_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico comparación VAL vs TEST guardado: {nombre}")


def graficar_cobertura_por_h(df_por_h: pd.DataFrame, banco: str, sufijo: str = "test"):
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
        f"Coverage 90% por fold y horizonte — {banco} [{sufijo.upper()} out-of-sample]\n"
        "(verde=bien calibrado ~90%, rojo=sub/sobre-cobertura)",
        fontsize=10,
    )
    xticks_pos = range(0, len(pivot.columns), 10)
    ax.set_xticks([p for p in xticks_pos if p < len(pivot.columns)])
    ax.set_xticklabels([str(pivot.columns[p]) for p in xticks_pos
                        if p < len(pivot.columns)], fontsize=8)
    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v2_cobertura_por_h_{sufijo}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Heatmap cobertura ({sufijo.upper()}) guardado: {nombre}")


def graficar_hiperparametros_wfcv(df_metricas: pd.DataFrame, banco: str):
    if df_metricas.empty:
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
        f"Estabilidad de hiperparámetros — Walk-forward CV v2 — {banco}\n"
        f"TRAIN {VENTANA_TRAIN_AÑOS}yr / VAL {VENTANA_VAL_AÑOS}yr (Optuna) / "
        f"TEST {VENTANA_TEST_AÑOS}yr / paso {PASO_AÑOS}yr / embargo {EMBARGO_DIAS_HAB}dh  "
        f"({len(folds)} folds)",
        fontweight="bold", fontsize=11,
    )

    for ax, (col, label, color, scale) in zip(axes_flat, hp_presentes):
        vals    = df_metricas[col].values.astype(float)
        mu, sig = vals.mean(), vals.std()

        ax.plot(folds, vals, "o-", color=color, lw=2, ms=7, zorder=4, label=label)
        ax.axhline(mu, color=color, lw=1.0, ls="--", alpha=0.5, label=f"media={mu:.4g}")
        if sig > 0:
            ax.axhspan(mu - sig, mu + sig, alpha=0.08, color=color,
                       label=f"±1σ ({sig:.4g})")
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

    nombre = DIR_PLOTS / f"wfcv_v2_hiperparametros_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico hiperparámetros guardado: {nombre}")


###############################################################################
# PARTE 8 — Pipeline principal  (MODIFICADO: métricas en TEST, guarda ambos)
###############################################################################

def evaluar_banco(banco: str):
    logger.info(f"\n{'='*65}")
    logger.info(f"BANCO: {banco}  — Walk-Forward CV v2  [TEST out-of-sample]")
    logger.info(f"{'='*65}")
    logger.info(f"  TRAIN={VENTANA_TRAIN_AÑOS}yr | VAL={VENTANA_VAL_AÑOS}yr (Optuna) | "
                f"TEST={VENTANA_TEST_AÑOS}yr (métricas) | paso={PASO_AÑOS}yr | "
                f"embargo={EMBARGO_DIAS_HAB}dh | trials={N_TRIALS_OPTUNA}")

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
                f"{len(fechas)} fechas | rango: {fechas.min().date()} → {fechas.max().date()}")

    # ── 2. Generar folds ──────────────────────────────────────────────────────
    folds = generar_folds(
        fechas_disponibles=fechas,
        ventana_train_años=VENTANA_TRAIN_AÑOS,
        ventana_val_años=VENTANA_VAL_AÑOS,
        ventana_test_años=VENTANA_TEST_AÑOS,
        paso_años=PASO_AÑOS,
        embargo_dias_hab=EMBARGO_DIAS_HAB,
    )
    if not folds:
        logger.error(f"  [{banco}] No se generaron folds — revisa VENTANA_TRAIN_AÑOS / datos")
        return None

    logger.info(f"  [{banco}] {len(folds)} folds generados:")
    for f in folds:
        logger.info(
            f"    Fold {f['fold']:2d} | TRAIN {f['train_start'].date()} → "
            f"{f['train_end'].date()} ({f['n_train_fechas']}dh) | "
            f"embargo {EMBARGO_DIAS_HAB}dh | "
            f"VAL  {f['val_start'].date()} → {f['val_end'].date()} ({f['n_val_fechas']}dh) | "
            f"TEST {f['test_start'].date()} → {f['test_end'].date()} ({f['n_test_fechas']}dh)"
        )

    # ── 3. Loop de folds ──────────────────────────────────────────────────────
    resultados_test   = []   # métricas OOS sobre TEST
    resultados_val    = []   # métricas sobre VAL (para comparación / cuantificar sesgo)
    resultados_por_h_test = []
    resultados_por_h_val  = []
    modelos_ultimo    = None
    params_ultimo     = None

    for fold in folds:
        t_fold = time.time()
        logger.info(f"\n  ── Fold {fold['fold']}/{len(folds)} ──────────────────────")

        # 3a. Preparar datos (TRAIN + VAL + TEST)
        try:
            (X_train, y_train,
             X_val,   y_val,
             X_test,  y_test,
             h_train, h_val, h_test) = preparar_fold_data(df, fold, cols_feat)
        except Exception as e:
            logger.warning(f"  Fold {fold['fold']}: error preparando datos — {e}")
            continue

        if len(X_train) < 200 or len(X_val) < 20 or len(X_test) < 20:
            logger.warning(
                f"  Fold {fold['fold']}: datos insuficientes "
                f"(train={len(X_train)}, val={len(X_val)}, test={len(X_test)}) — omitiendo"
            )
            continue

        std_y = float(y_train.std())
        logger.info(f"    X_train={len(X_train):,} | X_val={len(X_val):,} | "
                    f"X_test={len(X_test):,} | std_y={std_y:,.0f}")

        # 3b. Entrenar: Optuna usa VAL, modelo final se entrena sobre TRAIN
        modelos, best_params = entrenar_fold(
            X_train, y_train, X_val, y_val, std_y,
            N_TRIALS_OPTUNA, fold["fold"]
        )

        # 3c. Predecir sobre TEST (out-of-sample) y VAL (para comparación)
        preds_test = predecir_fold(modelos, X_test)
        preds_val  = predecir_fold(modelos, X_val)

        # 3d. Métricas TEST (las que importan — sin sesgo de tuning)
        row_test = calcular_metricas_fold(preds_test, y_test.values, fold, "test")
        row_test["tiempo_min"] = round((time.time() - t_fold) / 60, 2)
        row_test["modelo_cv"]  = MODELO_CV

        # 3e. Métricas VAL (referencia — tienen sesgo optimista de Optuna)
        row_val  = calcular_metricas_fold(preds_val, y_val.values, fold, "val")
        row_val["modelo_cv"] = MODELO_CV

        # 3f. Hiperparámetros óptimos (del estudio VAL, para HP stability plot)
        def _hp(d, key, default=0):
            return d.get(key, default)

        if MODELO_CV == "lgbm":
            for row in (row_test, row_val):
                row["s_optimo"]         = 0.0
                row["learning_rate"]    = round(_hp(best_params, "learning_rate"), 4)
                row["max_depth"]        = int(_hp(best_params, "num_leaves"))
                row["n_estimators"]     = int(_hp(best_params, "n_estimators"))
                row["min_child_weight"] = int(_hp(best_params, "min_child_samples"))
                row["subsample"]        = round(_hp(best_params, "subsample"), 3)
                row["colsample_bytree"] = round(_hp(best_params, "colsample_bytree"), 3)
                row["reg_alpha"]        = round(_hp(best_params, "reg_alpha"), 5)
                row["reg_lambda"]       = round(_hp(best_params, "reg_lambda"), 5)
        else:
            for row in (row_test, row_val):
                row["s_optimo"]         = round(_hp(best_params, "s"), 4)
                row["learning_rate"]    = round(_hp(best_params, "learning_rate"), 4)
                row["max_depth"]        = int(_hp(best_params, "max_depth"))
                row["n_estimators"]     = int(_hp(best_params, "n_estimators"))
                row["min_child_weight"] = int(_hp(best_params, "min_child_weight"))
                row["subsample"]        = round(_hp(best_params, "subsample"), 3)
                row["colsample_bytree"] = round(_hp(best_params, "colsample_bytree"), 3)
                row["reg_alpha"]        = round(_hp(best_params, "reg_alpha"), 5)
                row["reg_lambda"]       = round(_hp(best_params, "reg_lambda"), 5)

        resultados_test.append(row_test)
        resultados_val.append(row_val)
        resultados_por_h_test.append(
            calcular_metricas_por_h(preds_test, y_test.values, h_test, fold["fold"]))
        resultados_por_h_val.append(
            calcular_metricas_por_h(preds_val,  y_val.values,  h_val,  fold["fold"]))

        cov_test = row_test.get("coverage_90", float("nan"))
        cov_val  = row_val.get("coverage_90",  float("nan"))
        logger.info(
            f"    Fold {fold['fold']}  |  "
            f"TEST: pinball_Q50={row_test.get('pinball_q50','?'):,.0f}  "
            f"coverage={cov_test:.1%}  winkler={row_test.get('winkler_90','?'):,.0f}  "
            f"crps≈{row_test.get('crps_approx','?'):,.0f}  "
            f"|  VAL coverage={cov_val:.1%} (sesgo={cov_val-cov_test:+.1%})  "
            f"({row_test['tiempo_min']} min)"
        )

        modelos_ultimo = modelos
        params_ultimo  = best_params

        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()

    if not resultados_test:
        logger.error(f"  [{banco}] Ningún fold completado")
        return None

    # ── 4. Consolidar ─────────────────────────────────────────────────────────
    df_test_m  = pd.DataFrame(resultados_test)
    df_val_m   = pd.DataFrame(resultados_val)
    df_por_h_test = (pd.concat(resultados_por_h_test, ignore_index=True)
                     if resultados_por_h_test else pd.DataFrame())
    df_por_h_val  = (pd.concat(resultados_por_h_val,  ignore_index=True)
                     if resultados_por_h_val  else pd.DataFrame())

    # ── 5. Guardar CSVs ───────────────────────────────────────────────────────
    fecha_hoy = pd.Timestamp.today().strftime("%Y%m%d")

    def _guardar(df, nombre):
        ruta = DIR_MODELO_OUTPUT / nombre
        df.to_csv(ruta, index=False)
        logger.info(f"    {nombre}")
        return ruta

    logger.info(f"  [{banco}] Archivos guardados en: {DIR_MODELO_OUTPUT}")
    _guardar(df_test_m, f"wfcv_v2_metricas_TEST_{banco}_{fecha_hoy}.csv")
    _guardar(df_val_m,  f"wfcv_v2_metricas_VAL_{banco}_{fecha_hoy}.csv")
    if not df_por_h_test.empty:
        _guardar(df_por_h_test, f"wfcv_v2_por_h_TEST_{banco}_{fecha_hoy}.csv")
    if not df_por_h_val.empty:
        _guardar(df_por_h_val,  f"wfcv_v2_por_h_VAL_{banco}_{fecha_hoy}.csv")

    cols_hp = ["fold", "train_start", "train_end", "modelo_cv",
               "s_optimo", "learning_rate", "max_depth", "n_estimators",
               "min_child_weight", "subsample", "colsample_bytree",
               "reg_alpha", "reg_lambda"]
    cols_hp_ok = [c for c in cols_hp if c in df_test_m.columns]
    _guardar(df_test_m[cols_hp_ok], f"wfcv_v2_hiperparametros_{banco}_{fecha_hoy}.csv")

    # ── 6. Gráficos ───────────────────────────────────────────────────────────
    tag = f"{banco}_{MODELO_CV}"
    graficar_metricas_wfcv(df_test_m, tag)
    graficar_comparacion_val_test(df_val_m, df_test_m, tag)
    graficar_cobertura_por_h(df_por_h_test, tag, sufijo="test")
    graficar_cobertura_por_h(df_por_h_val,  tag, sufijo="val")
    graficar_hiperparametros_wfcv(df_test_m, tag)

    # ── 7. Modelo final ───────────────────────────────────────────────────────
    if GUARDAR_MODELO_FINAL and modelos_ultimo is not None:
        ultimo_fold = folds[-1]
        for tau, model in modelos_ultimo.items():
            if MODELO_CV == "lgbm":
                ruta_m = DIR_MODELOS / f"lgbm_wfcv_v2_{banco}_q{int(tau*100):02d}_{fecha_hoy}.txt"
                model.save_model(str(ruta_m))
            else:
                ruta_m = DIR_MODELOS / f"{MODELO_CV}_wfcv_v2_{banco}_q{int(tau*100):02d}_{fecha_hoy}.json"
                model.save_model(str(ruta_m))

        metadata = {
            "banco"               : banco,
            "modelo"              : f"{MODELO_CV}_wfcv_v2",
            "fecha_entrenamiento" : pd.Timestamp.today().strftime("%Y-%m-%d"),
            "version"             : "v2 — evaluación TEST out-of-sample",
            "config": {
                "ventana_train_años" : VENTANA_TRAIN_AÑOS,
                "ventana_val_años"   : VENTANA_VAL_AÑOS,
                "ventana_test_años"  : VENTANA_TEST_AÑOS,
                "paso_años"          : PASO_AÑOS,
                "embargo_dias_hab"   : EMBARGO_DIAS_HAB,
                "n_trials_optuna"    : N_TRIALS_OPTUNA,
            },
            "anti_leakage": {
                "embargo"           : f"{EMBARGO_DIAS_HAB} días hábiles post-TRAIN",
                "garch_por_fold"    : "ω/α/β estimados solo sobre TRAIN, propagados a VAL y TEST",
                "medianas_por_fold" : "imputación calculada sobre TRAIN, aplicada a VAL y TEST",
                "val_test_separados": "VAL solo para Optuna; TEST solo para métricas OOS",
            },
            "ultimo_fold": {
                "train_start": str(ultimo_fold["train_start"].date()),
                "train_end"  : str(ultimo_fold["train_end"].date()),
            },
            "n_folds"   : len(folds),
            "quantiles" : QUANTILES,
            "features"  : cols_feat,
            "best_params_ultimo_fold": params_ultimo,
        }
        ruta_meta = DIR_MODELOS / f"metadata_{MODELO_CV}_wfcv_v2_{banco}_{fecha_hoy}.json"
        with open(ruta_meta, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)
        logger.info(f"    Modelo final guardado en: {DIR_MODELOS}")

    # ── 8. Resumen en consola ─────────────────────────────────────────────────
    t_total = time.time() - t_inicio
    logger.info(f"\n  {'─'*65}")
    logger.info(f"  RESUMEN WALK-FORWARD CV v2 — {banco}  [TEST out-of-sample]")
    logger.info(f"  {'─'*65}")

    cols_display = ["fold", "train_start", "train_end", "test_start", "test_end",
                    "pinball_q50", "coverage_90", "winkler_90", "crps_approx"]
    cols_ok = [c for c in cols_display if c in df_test_m.columns]
    logger.info("\n" + df_test_m[cols_ok].to_string(index=False))

    promedios_test = {c: df_test_m[c].mean()
                      for c in ["pinball_q50", "coverage_90", "winkler_90", "crps_approx"]
                      if c in df_test_m.columns}
    promedios_val  = {c: df_val_m[c].mean()
                      for c in ["coverage_90", "pinball_q50"]
                      if c in df_val_m.columns}

    logger.info(f"\n  Promedios TEST (out-of-sample):")
    for k, v in promedios_test.items():
        fmt = f"{v:.1%}" if k == "coverage_90" else f"{v:,.2f}"
        logger.info(f"    {k:20s}: {fmt}")

    if "coverage_90" in promedios_val and "coverage_90" in promedios_test:
        sesgo = promedios_val["coverage_90"] - promedios_test["coverage_90"]
        logger.info(f"\n  Sesgo VAL−TEST en coverage: {sesgo:+.2%}  "
                    f"(VAL={promedios_val['coverage_90']:.1%} → "
                    f"TEST={promedios_test['coverage_90']:.1%})")

    logger.info(f"\n  ✓ Completado en {t_total/60:.1f} min  ({len(folds)} folds)")
    return df_test_m


###############################################################################
# Main
###############################################################################

def main():
    logger.info("=" * 65)
    logger.info(f"STEP005 v2 — Walk-Forward CV con TEST out-of-sample  [MODELO: {MODELO_CV.upper()}]")
    logger.info("=" * 65)
    logger.info("Diferencia vs v1: VAL=Optuna únicamente / TEST=métricas sin sesgo")
    if MODELO_CV == "lgbm" and not _LGBM_OK:
        logger.error("MODELO_CV='lgbm' pero lightgbm no está instalado.")
        return

    logger.info("Medidas anti-leakage activas:")
    logger.info(f"  #1 Embargo          : {EMBARGO_DIAS_HAB} días hábiles post-TRAIN")
    logger.info(f"  #2 GARCH por fold   : ω/α/β de TRAIN propagados a VAL+TEST")
    logger.info(f"  #3 Medianas fold    : imputación de TRAIN aplicada a VAL+TEST")
    logger.info(f"Configuración de folds:")
    logger.info(f"  TRAIN {VENTANA_TRAIN_AÑOS}yr | VAL {VENTANA_VAL_AÑOS}yr (Optuna) | "
                f"TEST {VENTANA_TEST_AÑOS}yr (OOS) | paso {PASO_AÑOS}yr")

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
        logger.info("RESUMEN GLOBAL — métricas TEST out-of-sample")
        logger.info("=" * 65)
        for banco, df_m in todos_resultados:
            avg_cov = df_m["coverage_90"].mean() if "coverage_90" in df_m.columns else float("nan")
            avg_pb  = df_m["pinball_q50"].mean()  if "pinball_q50"  in df_m.columns else float("nan")
            logger.info(f"  {banco:15s}: {len(df_m)} folds | "
                        f"pinball_Q50_avg={avg_pb:,.0f} | coverage_90_avg={avg_cov:.1%}")

    logger.info(f"\n✓ Total: {(time.time()-t0)/60:.1f} min")
    logger.info(f"  Resultados en: {DIR_MODELO_OUTPUT}")


if __name__ == "__main__":
    main()
