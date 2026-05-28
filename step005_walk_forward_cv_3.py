# -*- coding: utf-8 -*-
"""
step005_walk_forward_cv_3.py
Walk-forward CV con ventana EXPANDABLE o RODANTE y evaluación TEST out-of-sample.

═══════════════════════════════════════════════════════════════════════════════
NOVEDADES RESPECTO A v2
═══════════════════════════════════════════════════════════════════════════════

1. Toggle EXPANDING (True/False)
   ─────────────────────────────
   EXPANDING = True  → ventana CRECIENTE: train_start fijo en el origen,
                        train_end crece PASO_AÑOS cada fold.
   EXPANDING = False → ventana RODANTE   (idéntico a v2): train_start y
                        train_end avanzan juntos; tamaño fijo = VENTANA_TRAIN_AÑOS.

   En modo EXPANDING, VENTANA_TRAIN_AÑOS es el tamaño MÍNIMO inicial.
   Cada fold siguiente tiene un año (o PASO_AÑOS) más de historia.

   Comparación de enfoques:
     Expanding → más datos en folds tardíos; mejor si el proceso es estacionario.
     Rolling   → se adapta a regímenes recientes; mejor si hay quiebres estructurales.

2. Fan charts TEST out-of-sample (nuevo)
   ──────────────────────────────────────
   Por cada fold se generan hasta 4 snapshots del fan chart en el período TEST,
   con origen separado ~3 meses entre sí. Cada snapshot muestra:
     · Bandas Q01-Q99 (muy suave) y Q05-Q95 (media) → incertidumbre
     · Línea Q50 → mediana predicha
     · Puntos realizados: verde si caen dentro de Q05-Q95, rojo si fuera
     · Coverage empírico anotado en el título de cada panel

3. Separación VAL / TEST
   ──────────────────────
   Idéntica a v2: VAL solo para Optuna, TEST solo para métricas OOS.

═══════════════════════════════════════════════════════════════════════════════
Anti-leakage — idéntico a v1 y v2
═══════════════════════════════════════════════════════════════════════════════
  #1 EMBARGO 90dh post-TRAIN
  #2 GARCH por fold: ω/α/β estimados solo en TRAIN, propagados a VAL+TEST
  #3 Medianas fold: imputación calculada en TRAIN, aplicada a VAL+TEST
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
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v3"

# ── Tipo de ventana ───────────────────────────────────────────────────────────
# True  → EXPANDING: train_start fijo, train_end crece cada fold
# False → ROLLING  : ventana fija que desliza (idéntico a v2)
EXPANDING = True

# ── Tamaños de ventana ────────────────────────────────────────────────────────
# EXPANDING=True : VENTANA_TRAIN_AÑOS es el mínimo inicial; crece PASO_AÑOS/fold
# EXPANDING=False: VENTANA_TRAIN_AÑOS es el tamaño fijo (igual a v2)
VENTANA_TRAIN_AÑOS  = 5      # años de TRAIN iniciales / fijos
VENTANA_VAL_AÑOS    = 1      # años de VAL (solo Optuna)
VENTANA_TEST_AÑOS   = 1      # años de TEST (solo métricas OOS)
PASO_AÑOS           = 1      # desplazamiento / crecimiento entre folds
EMBARGO_DIAS_HAB    = 90     # días hábiles entre fin-TRAIN e inicio-VAL

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
FIX_REG_ALPHA  = False
FIX_REG_LAMBDA = False

# ── Fan chart TEST: número de snapshots por fold ──────────────────────────────
FANCHART_N_SNAPSHOTS = 4   # 1 cada ~3 meses para TEST de 1 año

# ── Rutas de salida ───────────────────────────────────────────────────────────
_modo           = "expanding" if EXPANDING else "rolling"
DIR_MODO        = DIR_OUTPUT / f"{MODELO_CV}_{_modo}"
DIR_MODELOS     = DIR_MODO / "modelos"
DIR_PLOTS       = DIR_MODO / "plots"
DIR_FANCHARTS   = DIR_MODO / "fancharts_test"

for _d in (DIR_OUTPUT, DIR_MODO, DIR_MODELOS, DIR_PLOTS, DIR_FANCHARTS):
    _d.mkdir(parents=True, exist_ok=True)


###############################################################################
# PARTE 1 — Métricas
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
# PARTE 2 — GARCH por fold
###############################################################################

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


###############################################################################
# PARTE 3 — Generación de folds  (EXPANDING o ROLLING según toggle)
###############################################################################

def generar_folds(
    fechas_disponibles,
    ventana_train_años,
    ventana_val_años,
    ventana_test_años,
    paso_años,
    embargo_dias_hab,
    expanding,
):
    """
    EXPANDING=True  → train_start fijo en f_min; train_end crece paso_años/fold.
    EXPANDING=False → ventana rodante fija (mismo comportamiento que v2).

    Estructura por fold:
      TRAIN → embargo → VAL (Optuna) → TEST (métricas OOS)

    Genera folds mientras test_end ≤ última fecha disponible.
    """
    folds   = []
    f_min   = fechas_disponibles.min()
    f_max   = fechas_disponibles.max()
    paso    = pd.DateOffset(months=int(round(paso_años * 12)))
    fold_idx = 0

    while True:
        if expanding:
            train_start = f_min
            train_end   = f_min + pd.DateOffset(
                months=int(round((ventana_train_años + fold_idx * paso_años) * 12)))
        else:
            train_start = f_min + pd.DateOffset(
                months=int(round(fold_idx * paso_años * 12)))
            train_end   = train_start + pd.DateOffset(
                months=int(round(ventana_train_años * 12)))

        val_start  = train_end   + pd.offsets.BusinessDay(embargo_dias_hab)
        val_end    = val_start   + pd.DateOffset(months=int(round(ventana_val_años * 12)))
        test_start = val_end
        test_end   = test_start  + pd.DateOffset(months=int(round(ventana_test_años * 12)))

        if test_end > f_max or train_end >= f_max or val_start >= f_max:
            break

        n_train = ((fechas_disponibles >= train_start) &
                   (fechas_disponibles <= train_end)).sum()
        n_val   = ((fechas_disponibles >= val_start) &
                   (fechas_disponibles <  test_start)).sum()
        n_test  = ((fechas_disponibles >= test_start) &
                   (fechas_disponibles <= test_end)).sum()

        if n_train < 60 or n_val < 10 or n_test < 10:
            fold_idx += 1
            continue

        folds.append({
            "fold"           : fold_idx + 1,
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
        fold_idx += 1

    return folds


###############################################################################
# PARTE 4 — Modelos: objetivos, optimización, entrenamiento
###############################################################################

def make_quantile_objective(tau, s, std_y):
    _scale = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)
    def objective(y_pred, dtrain):
        u    = dtrain.get_label() - y_pred
        grad = -((tau - 0.5 + np.arctan(u / s) / np.pi)
                 + u * s / (np.pi * (s ** 2 + u ** 2))) * _scale
        hess = 2 * s ** 3 / (np.pi * (s ** 2 + u ** 2) ** 2) * _scale
        return grad, hess
    return objective


def make_pinball_metric(tau):
    def metric(y_pred, dtrain):
        y_true = dtrain.get_label()
        u      = y_true - y_pred
        return "pinball", float(np.mean(np.where(u >= 0, tau * u, (tau - 1) * u)))
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
    model  = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(0.50, s, std_y),
        custom_metric=make_pinball_metric(0.50),
        evals=[(dval, "val")],
        callbacks=[xgb.callback.EarlyStopping(rounds=50, metric_name="pinball",
                                               save_best=False, maximize=False)],
        verbose_eval=False,
    )
    return pinball_loss(y_va.values, model.predict(dval), 0.50)


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
    dmat      = xgb.DMatrix(X)
    preds_raw = {tau: m.predict(dmat) for tau, m in modelos.items()}
    taus      = sorted(preds_raw)
    matrix    = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    return {t: matrix[:, i] for i, t in enumerate(taus)}


# ── LightGBM ──────────────────────────────────────────────────────────────────

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
        "reg_alpha"        : 0.0 if FIX_REG_ALPHA else trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 0.1, 5.0) if FIX_REG_LAMBDA else trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "subsample_freq"   : 1,
    }
    n_est  = trial.suggest_int("n_estimators", 100, 1000)
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


# ── XGBoost QT ────────────────────────────────────────────────────────────────

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
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42 + fold_num + int(tau * 100)),
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
    return modelos, best_by_tau.get(0.50, list(best_by_tau.values())[0])


# ── Dispatchers ───────────────────────────────────────────────────────────────

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
# PARTE 5 — Preparación de datos  (devuelve también fechas_t para TEST)
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
    de TEST — necesario para construir los fan charts por snapshot.
    """
    train_start = fold["train_start"]
    train_end   = fold["train_end"]
    val_start   = fold["val_start"]
    test_start  = fold["test_start"]
    test_end    = fold["test_end"]

    mask_train = (df["fecha_t"] >= train_start) & (df["fecha_t"] <= train_end)
    mask_val   = (df["fecha_t"] >= val_start)   & (df["fecha_t"] <  test_start)
    mask_test  = (df["fecha_t"] >= test_start)  & (df["fecha_t"] <= test_end)

    df_fold_all = df[mask_train | mask_val | mask_test].copy()

    garch_cols = [c for c in ["garch_vol", "garch_vol_tc", "garch_vol_embi"]
                  if c in df_fold_all.columns]
    if garch_cols:
        df_fold_all = reemplazar_garch_fold(df_fold_all, train_end)

    df_train = df_fold_all[df_fold_all["fecha_t"] <= train_end]
    df_val   = df_fold_all[(df_fold_all["fecha_t"] >= val_start) &
                           (df_fold_all["fecha_t"] <  test_start)]
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

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            h_train, h_val, h_test, fechas_t_test)


###############################################################################
# PARTE 6 — Métricas
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
    for tau in sorted(preds.keys()):
        row[f"pinball_q{int(tau*100):02d}"] = round(
            pinball_loss(y_true, preds[tau], tau), 2)
    if {0.05, 0.95}.issubset(preds):
        row["coverage_90"] = round(coverage(y_true, preds[0.05], preds[0.95]), 4)
        row["winkler_90"]  = round(winkler_score(y_true, preds[0.05], preds[0.95]), 2)
    row["crps_approx"] = round(crps_approx(y_true, preds), 2)
    return row


def calcular_metricas_por_h(preds, y_true, h_arr, fold_num):
    rows = []
    for h in np.unique(h_arr):
        mask = h_arr == h
        if mask.sum() < 5:
            continue
        row = {"fold": fold_num, "h": int(h)}
        for tau in sorted(preds.keys()):
            row[f"pinball_q{int(tau*100):02d}"] = round(
                pinball_loss(y_true[mask], preds[tau][mask], tau), 2)
        if {0.05, 0.95}.issubset(preds):
            row["coverage_90"] = round(
                coverage(y_true[mask], preds[0.05][mask], preds[0.95][mask]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


###############################################################################
# PARTE 7 — Visualización
###############################################################################

def graficar_metricas_wfcv(df_test_m, banco):
    if df_test_m.empty:
        return
    folds = df_test_m["fold"].values
    modo  = "EXPANDING" if EXPANDING else "ROLLING"

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Walk-forward CV v3 [{modo}] — {banco}  [métricas TEST out-of-sample]\n"
        f"TRAIN {VENTANA_TRAIN_AÑOS}yr{'(min)' if EXPANDING else ''} / "
        f"VAL {VENTANA_VAL_AÑOS}yr (Optuna) / TEST {VENTANA_TEST_AÑOS}yr / "
        f"paso {PASO_AÑOS}yr / embargo {EMBARGO_DIAS_HAB}dh",
        fontweight="bold", fontsize=10,
    )
    metricas_config = [
        ("pinball_q50", "Pinball Q50 — TEST OOS",   "steelblue",  None),
        ("coverage_90", "Coverage 90% — TEST OOS",  "seagreen",   0.90),
        ("winkler_90",  "Winkler score — TEST",     "darkorange", None),
        ("crps_approx", "CRPS aprox. — TEST",       "crimson",    None),
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
        f"VAL (Optuna, sesgo) vs TEST (OOS real) — {banco} "
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
        ax.set_xlabel(f"Fold  (sesgo promedio VAL−TEST: {diff:+.4f})", fontsize=9)
        ax.set_xticks(folds)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v3_val_vs_test_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Comparación VAL vs TEST: {nombre}")


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
        f"Coverage 90% por fold y horizonte — {banco} [{sufijo.upper()} OOS] "
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
    axes_flat = axes.flatten() if n > 1 else [axes]
    fig.suptitle(
        f"Estabilidad HP — Walk-forward CV v3 [{' EXPANDING' if EXPANDING else 'ROLLING'}] — {banco}\n"
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
    nombre = DIR_PLOTS / f"wfcv_v3_hiperparametros_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico hiperparámetros: {nombre}")


def graficar_fanchart_test_fold(
    preds_test: dict,
    y_test: np.ndarray,
    h_test: np.ndarray,
    fechas_t_test,
    fold: dict,
    banco: str,
):
    """
    Fan chart TEST out-of-sample para un fold.

    Selecciona hasta FANCHART_N_SNAPSHOTS fechas de origen separadas ~3 meses
    dentro del período TEST. Para cada fecha muestra:
      · Bandas Q01-Q99 (muy transparente) y Q05-Q95 (media)
      · Línea Q50 (mediana predicha)
      · Puntos realizados: verde = dentro de Q05-Q95 / rojo = fuera

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
    axes_flat = axes.flatten() if len(origenes) > 1 else [axes]

    fig.suptitle(
        f"Fan chart TEST OOS — Fold {fold['fold']} — {banco} [{modo}]\n"
        f"TEST: {fold['test_start'].date()} → {fold['test_end'].date()}  |  "
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

        # Bandas de incertidumbre
        if {0.01, 0.99}.issubset(p_s):
            ax.fill_between(h_s, p_s[0.01] / 1e6, p_s[0.99] / 1e6,
                            alpha=0.12, color="steelblue", label="Q01-Q99")
        if {0.05, 0.95}.issubset(p_s):
            ax.fill_between(h_s, p_s[0.05] / 1e6, p_s[0.95] / 1e6,
                            alpha=0.28, color="steelblue", label="Q05-Q95")
        if 0.50 in p_s:
            ax.plot(h_s, p_s[0.50] / 1e6, color="steelblue", lw=1.8,
                    zorder=3, label="Q50")

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
    nombre = DIR_FANCHARTS / f"fanchart_test_fold{fold['fold']:02d}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Fan chart TEST fold {fold['fold']}: {nombre.name}")


###############################################################################
# PARTE 8 — Pipeline principal
###############################################################################

def evaluar_banco(banco: str):
    modo = "EXPANDING" if EXPANDING else "ROLLING"
    logger.info(f"\n{'='*65}")
    logger.info(f"BANCO: {banco}  — Walk-Forward CV v3  [{modo}]  [TEST OOS]")
    logger.info(f"{'='*65}")
    logger.info(
        f"  TRAIN {VENTANA_TRAIN_AÑOS}yr{'(min)' if EXPANDING else ''} | "
        f"VAL {VENTANA_VAL_AÑOS}yr (Optuna) | TEST {VENTANA_TEST_AÑOS}yr (métricas) | "
        f"paso {PASO_AÑOS}yr | embargo {EMBARGO_DIAS_HAB}dh | trials {N_TRIALS_OPTUNA}"
    )

    t_inicio = time.time()

    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    if df.empty or df["target"].notna().sum() < 500:
        logger.warning(f"  [{banco}] Datos insuficientes — omitiendo")
        return None

    cols_feat = get_feature_cols(df)
    fechas    = pd.DatetimeIndex(df["fecha_t"].unique())
    logger.info(f"  [{banco}] {len(df):,} filas | {len(cols_feat)} features | "
                f"rango: {fechas.min().date()} → {fechas.max().date()}")

    folds = generar_folds(
        fechas_disponibles=fechas,
        ventana_train_años=VENTANA_TRAIN_AÑOS,
        ventana_val_años=VENTANA_VAL_AÑOS,
        ventana_test_años=VENTANA_TEST_AÑOS,
        paso_años=PASO_AÑOS,
        embargo_dias_hab=EMBARGO_DIAS_HAB,
        expanding=EXPANDING,
    )
    if not folds:
        logger.error(f"  [{banco}] No se generaron folds")
        return None

    logger.info(f"  [{banco}] {len(folds)} folds generados:")
    for f in folds:
        n_train_yr = round(f["n_train_fechas"] / 252, 1)
        logger.info(
            f"    Fold {f['fold']:2d} | TRAIN {f['train_start'].date()} → "
            f"{f['train_end'].date()} ({n_train_yr}yr, {f['n_train_fechas']}dh) | "
            f"VAL  {f['val_start'].date()} → {f['val_end'].date()} | "
            f"TEST {f['test_start'].date()} → {f['test_end'].date()}"
        )

    resultados_test  = []
    resultados_val   = []
    por_h_test       = []
    por_h_val        = []
    modelos_ultimo   = None
    params_ultimo    = None

    for fold in folds:
        t_fold = time.time()
        logger.info(f"\n  ── Fold {fold['fold']}/{len(folds)} ──────────────────────")

        try:
            (X_train, y_train,
             X_val,   y_val,
             X_test,  y_test,
             h_train, h_val, h_test,
             fechas_t_test) = preparar_fold_data(df, fold, cols_feat)
        except Exception as e:
            logger.warning(f"  Fold {fold['fold']}: error preparando datos — {e}")
            continue

        if len(X_train) < 200 or len(X_val) < 20 or len(X_test) < 20:
            logger.warning(f"  Fold {fold['fold']}: datos insuficientes — omitiendo")
            continue

        std_y = float(y_train.std())
        logger.info(f"    X_train={len(X_train):,} | X_val={len(X_val):,} | "
                    f"X_test={len(X_test):,} | std_y={std_y:,.0f}")

        modelos, best_params = entrenar_fold(
            X_train, y_train, X_val, y_val, std_y,
            N_TRIALS_OPTUNA, fold["fold"]
        )

        preds_test = predecir_fold(modelos, X_test)
        preds_val  = predecir_fold(modelos, X_val)

        row_test = calcular_metricas_fold(preds_test, y_test.values, fold, "test")
        row_val  = calcular_metricas_fold(preds_val,  y_val.values,  fold, "val")
        row_test["tiempo_min"] = round((time.time() - t_fold) / 60, 2)
        row_val["modelo_cv"]   = MODELO_CV
        row_test["modelo_cv"]  = MODELO_CV

        def _hp(d, key, default=0):
            return d.get(key, default)

        for row in (row_test, row_val):
            if MODELO_CV == "lgbm":
                row.update({
                    "s_optimo"        : 0.0,
                    "learning_rate"   : round(_hp(best_params, "learning_rate"), 4),
                    "max_depth"       : int(_hp(best_params, "num_leaves")),
                    "n_estimators"    : int(_hp(best_params, "n_estimators")),
                    "min_child_weight": int(_hp(best_params, "min_child_samples")),
                    "subsample"       : round(_hp(best_params, "subsample"), 3),
                    "colsample_bytree": round(_hp(best_params, "colsample_bytree"), 3),
                    "reg_alpha"       : round(_hp(best_params, "reg_alpha"), 5),
                    "reg_lambda"      : round(_hp(best_params, "reg_lambda"), 5),
                })
            else:
                row.update({
                    "s_optimo"        : round(_hp(best_params, "s"), 4),
                    "learning_rate"   : round(_hp(best_params, "learning_rate"), 4),
                    "max_depth"       : int(_hp(best_params, "max_depth")),
                    "n_estimators"    : int(_hp(best_params, "n_estimators")),
                    "min_child_weight": int(_hp(best_params, "min_child_weight")),
                    "subsample"       : round(_hp(best_params, "subsample"), 3),
                    "colsample_bytree": round(_hp(best_params, "colsample_bytree"), 3),
                    "reg_alpha"       : round(_hp(best_params, "reg_alpha"), 5),
                    "reg_lambda"      : round(_hp(best_params, "reg_lambda"), 5),
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

        # Fan charts TEST (una figura por fold con N snapshots)
        graficar_fanchart_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test, fold, banco
        )

        modelos_ultimo = modelos
        params_ultimo  = best_params
        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()

    if not resultados_test:
        logger.error(f"  [{banco}] Ningún fold completado")
        return None

    df_test_m  = pd.DataFrame(resultados_test)
    df_val_m   = pd.DataFrame(resultados_val)
    df_por_h_t = pd.concat(por_h_test, ignore_index=True) if por_h_test else pd.DataFrame()
    df_por_h_v = pd.concat(por_h_val,  ignore_index=True) if por_h_val  else pd.DataFrame()

    fecha_hoy = pd.Timestamp.today().strftime("%Y%m%d")

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

    if GUARDAR_MODELO_FINAL and modelos_ultimo is not None:
        ultimo = folds[-1]
        for tau, model in modelos_ultimo.items():
            sfx = "lgbm_wfcv_v3" if MODELO_CV == "lgbm" else f"{MODELO_CV}_wfcv_v3"
            ext = ".txt" if MODELO_CV == "lgbm" else ".json"
            ruta_m = DIR_MODELOS / f"{sfx}_{banco}_q{int(tau*100):02d}_{fecha_hoy}{ext}"
            model.save_model(str(ruta_m))
        metadata = {
            "banco": banco, "modelo": f"{MODELO_CV}_wfcv_v3",
            "fecha_entrenamiento": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "version": "v3 — expanding/rolling + TEST OOS",
            "config": {
                "expanding"          : EXPANDING,
                "ventana_train_años" : VENTANA_TRAIN_AÑOS,
                "ventana_val_años"   : VENTANA_VAL_AÑOS,
                "ventana_test_años"  : VENTANA_TEST_AÑOS,
                "paso_años"          : PASO_AÑOS,
                "embargo_dias_hab"   : EMBARGO_DIAS_HAB,
                "n_trials_optuna"    : N_TRIALS_OPTUNA,
            },
            "anti_leakage": {
                "embargo"          : f"{EMBARGO_DIAS_HAB} dh post-TRAIN",
                "garch_por_fold"   : "ω/α/β estimados en TRAIN, propagados a VAL+TEST",
                "medianas_por_fold": "calculadas en TRAIN, aplicadas a VAL+TEST",
                "val_test_sep"     : "VAL=Optuna only / TEST=métricas OOS only",
            },
            "ultimo_fold": {
                "train_start": str(ultimo["train_start"].date()),
                "train_end"  : str(ultimo["train_end"].date()),
            },
            "n_folds": len(folds), "quantiles": QUANTILES,
            "features": cols_feat, "best_params_ultimo_fold": params_ultimo,
        }
        ruta_meta = DIR_MODELOS / f"metadata_{MODELO_CV}_wfcv_v3_{banco}_{fecha_hoy}.json"
        with open(ruta_meta, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)

    t_total = time.time() - t_inicio
    logger.info(f"\n  {'─'*60}")
    logger.info(f"  RESUMEN — {banco}  [{modo}]  [TEST OOS]")
    logger.info(f"  {'─'*60}")
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
        logger.info(f"\n  Sesgo VAL−TEST coverage: {sesgo:+.2%}")

    logger.info(f"\n  ✓ Completado en {t_total/60:.1f} min  ({len(folds)} folds)")
    return df_test_m


###############################################################################
# Main
###############################################################################

def main():
    modo = "EXPANDING" if EXPANDING else "ROLLING"
    logger.info("=" * 65)
    logger.info(f"STEP005 v3 — Walk-Forward CV [{modo}] + TEST OOS  [{MODELO_CV.upper()}]")
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
        logger.info(f"RESUMEN GLOBAL — TEST OOS  [{modo}]")
        logger.info("=" * 65)
        for banco, df_m in todos:
            avg_cov = df_m["coverage_90"].mean() if "coverage_90" in df_m.columns else float("nan")
            avg_pb  = df_m["pinball_q50"].mean()  if "pinball_q50"  in df_m.columns else float("nan")
            logger.info(f"  {banco:15s}: {len(df_m)} folds | "
                        f"coverage_90_avg={avg_cov:.1%} | pinball_Q50_avg={avg_pb:,.0f}")

    logger.info(f"\n✓ Total: {(time.time()-t0)/60:.1f} min  →  {DIR_MODO}")


if __name__ == "__main__":
    main()
