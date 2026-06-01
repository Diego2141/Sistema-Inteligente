# -*- coding: utf-8 -*-
"""
aux_fanchart_media_trapezoidal.py

Fan charts con media trapezoidal (E[X] integrada desde cuantiles).

Configuración fija:
  EXPANDING=False (ROLLING), VENTANA_TRAIN=5yr, VENTANA_VAL=1yr,
  VENTANA_TEST=1yr, N_TRIALS=60, MODELO_CV="xgb_qt",
  QUANTILES=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

La media trapezoidal aproxima:
  E[X] = ∫₀¹ Q(p) dp  ≈  Σ (Q_i + Q_{i+1})/2 × (p_{i+1} - p_i)

Tres curvas de tendencia central en el fan chart:
  · Q50          (azul discontinuo) — mediana, suavizada
  · Media XGB    (navy)             — reg:squarederror separado
  · Media Trap.  (firebrick)        — integral trapezoidal de 9 cuantiles
"""

from __future__ import annotations

import gc
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


###############################################################################
# CONFIGURACIÓN
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_media_trapezoidal"

BANCO = "SISTEMA"

# ── Ventanas (ROLLING) ────────────────────────────────────────────────────────
EXPANDING           = False
VENTANA_TRAIN_AÑOS  = 5
VENTANA_VAL_AÑOS    = 1
VENTANA_TEST_AÑOS   = 1
PASO_AÑOS           = 1
EMBARGO_DIAS_HAB    = 90   # anti-leakage rolling

# ── Modelo ────────────────────────────────────────────────────────────────────
N_TRIALS_OPTUNA = 60
S_MIN_FACTOR    = 0.01
S_MAX_FACTOR    = 1.0
QUANTILES       = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# ── N_MAX_FOLDS=None → usa todos los folds disponibles ───────────────────────
N_MAX_FOLDS          = None
FANCHART_N_SNAPSHOTS = 4
COLS_EXCLUIR         = {"fecha_t", "banco", "target"}

# Threads por modelo (N cuantiles en paralelo)
_N_Q_PAR    = len(QUANTILES)
_XGB_NTHRD  = max(1, (os.cpu_count() or 4) // _N_Q_PAR)

for _d in (DIR_OUTPUT,):
    _d.mkdir(parents=True, exist_ok=True)


###############################################################################
# MÉTRICAS
###############################################################################

def pinball_loss(y_true, y_pred, tau):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def coverage(y_true, q_lo, q_hi):
    return float(np.mean((y_true >= q_lo) & (y_true <= q_hi)))


###############################################################################
# MEDIA TRAPEZOIDAL
###############################################################################

def media_trapezoidal(preds: dict) -> np.ndarray:
    """
    Aproxima E[X] = ∫₀¹ Q(p) dp con la regla trapezoidal.

    preds: {tau: array_1d}  con tau en QUANTILES
    Retorna array_1d con la media para cada observación.
    """
    q_sorted = sorted(q for q in preds if isinstance(q, float))
    vals = np.column_stack([preds[q] for q in q_sorted])   # (n_obs, n_q)
    media = np.zeros(vals.shape[0])
    for i in range(len(q_sorted) - 1):
        dp = q_sorted[i + 1] - q_sorted[i]
        media += (vals[:, i] + vals[:, i + 1]) / 2.0 * dp
    return media


###############################################################################
# GARCH (anti-leakage: estimado solo en TRAIN, propagado a VAL+TEST)
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
    escala      = float(serie_train.std())
    x_train     = (serie_train / escala).values.astype(float)
    var_unc     = max(float(np.var(x_train)), 1e-12)
    omega, alpha, beta = _ajustar_garch_params(x_train)
    x_full = (serie_full / escala).values.astype(float)
    n_full = len(x_full)
    s2     = np.empty(n_full)
    s2[0]  = var_unc
    for t in range(1, n_full):
        s2[t] = omega + alpha * x_full[t - 1] ** 2 + beta * s2[t - 1]
    return pd.Series(np.sqrt(np.maximum(s2, 0)) * escala, index=serie_full.index)


def reemplazar_garch_fold(df_fold, train_end):
    df_fold  = df_fold.copy()
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
        df_fold["garch_vol_tc"] = df_fold["fecha_t"].map(
            _garch_vol_fold(ret, train_end))
    if "EMBI_PERU" in raw.columns:
        df_fold["garch_vol_embi"] = df_fold["fecha_t"].map(
            _garch_vol_fold(raw["EMBI_PERU"].diff(1), train_end))
    return df_fold


###############################################################################
# GENERACIÓN DE FOLDS (ROLLING)
###############################################################################

def generar_folds(fechas_disponibles):
    folds    = []
    f_min    = fechas_disponibles.min()
    f_max    = fechas_disponibles.max()
    fold_idx = 0

    while True:
        train_start = f_min + pd.DateOffset(
            months=int(round(fold_idx * PASO_AÑOS * 12)))
        train_end   = train_start + pd.DateOffset(
            months=int(round(VENTANA_TRAIN_AÑOS * 12)))
        val_start   = train_end   + pd.offsets.BusinessDay(EMBARGO_DIAS_HAB)
        val_end     = val_start   + pd.DateOffset(
            months=int(round(VENTANA_VAL_AÑOS * 12)))
        test_start  = val_end
        test_end    = test_start  + pd.DateOffset(
            months=int(round(VENTANA_TEST_AÑOS * 12)))

        if test_end > f_max or train_end >= f_max or val_start >= f_max:
            break

        n_train = int(((fechas_disponibles >= train_start) &
                       (fechas_disponibles <= train_end)).sum())
        n_val   = int(((fechas_disponibles >= val_start) &
                       (fechas_disponibles <  test_start)).sum())
        n_test  = int(((fechas_disponibles >= test_start) &
                       (fechas_disponibles <= test_end)).sum())

        if n_train >= 60 and n_val >= 10 and n_test >= 10:
            folds.append({
                "fold"           : fold_idx + 1,
                "train_start"    : train_start,
                "train_end"      : train_end,
                "val_start"      : val_start,
                "val_end"        : val_end,
                "test_start"     : test_start,
                "test_end"       : test_end,
                "n_train_fechas" : n_train,
                "n_val_fechas"   : n_val,
                "n_test_fechas"  : n_test,
            })
        fold_idx += 1

    return folds


###############################################################################
# MODELO xgb_qt — Optuna independiente por cuantil + media reg:squarederror
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


def _optuna_y_train_tau(tau, X_tr, y_tr, X_va, y_va, std_y, fold_num):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold_num + int(tau * 100)),
    )
    study.optimize(
        lambda t: _objetivo_tau(t, tau, X_tr, y_tr, X_va, y_va, std_y),
        n_trials=N_TRIALS_OPTUNA, show_progress_bar=False,
    )
    bp    = study.best_params
    s     = bp["s"]
    n_est = bp["n_estimators"]
    params = {k: v for k, v in bp.items() if k not in ("s", "n_estimators")}
    params.update({"tree_method": "hist", "seed": 42, "nthread": _XGB_NTHRD})
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    model  = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(tau, s, std_y),
        verbose_eval=False,
    )
    logger.info(f"    [xgb_qt] τ={tau:.2f} fold {fold_num}: "
                f"pinball/VAL={study.best_value:.4f}  s={s:.3f}  n_est={n_est}")
    return tau, model, bp


def _objetivo_tau(trial, tau, X_tr, y_tr, X_va, y_va, std_y):
    s = trial.suggest_float("s", std_y * S_MIN_FACTOR, std_y * S_MAX_FACTOR, log=True)
    params = {
        "learning_rate"   : trial.suggest_float("learning_rate",   0.01, 0.3,  log=True),
        "max_depth"       : trial.suggest_int(  "max_depth",         3,   10),
        "min_child_weight": trial.suggest_int(  "min_child_weight", 10,  200),
        "colsample_bytree": trial.suggest_float("colsample_bytree",  0.4,  1.0),
        "subsample"       : trial.suggest_float("subsample",         0.5,  1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method"     : "hist",
        "nthread"         : _XGB_NTHRD,
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


def entrenar_fold_xgb_qt(X_tr, y_tr, X_va, y_va, std_y, fold_num):
    modelos     = {}
    best_by_tau = {}
    with ThreadPoolExecutor(max_workers=_N_Q_PAR) as ex:
        futures = {
            ex.submit(_optuna_y_train_tau, tau, X_tr, y_tr, X_va, y_va, std_y, fold_num): tau
            for tau in QUANTILES
        }
        for fut in as_completed(futures):
            tau, model, bp = fut.result()
            modelos[tau]     = model
            best_by_tau[tau] = bp

    # Media: reg:squarederror con hiperparámetros del Q50
    bp_mean = best_by_tau.get(0.50, list(best_by_tau.values())[0])
    params_mean = {k: v for k, v in bp_mean.items() if k not in ("s", "n_estimators")}
    params_mean.update({"objective": "reg:squarederror",
                        "tree_method": "hist",
                        "nthread": _XGB_NTHRD,
                        "seed": 42})
    dtrain_m       = xgb.DMatrix(X_tr, label=y_tr)
    modelos["mean_xgb"] = xgb.train(params_mean, dtrain_m,
                                    num_boost_round=bp_mean["n_estimators"],
                                    verbose_eval=False)
    logger.info(f"    [xgb_qt] mean fold {fold_num}: reg:squarederror")
    return modelos


def predecir(modelos, X):
    dmat      = xgb.DMatrix(X)
    # Cuantiles
    q_keys    = [k for k in modelos if isinstance(k, float)]
    preds_raw = {tau: modelos[tau].predict(dmat) for tau in q_keys}
    # Corrección de cruzamientos
    taus   = sorted(preds_raw)
    matrix = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    result = {t: matrix[:, i] for i, t in enumerate(taus)}
    # Media XGB
    if "mean_xgb" in modelos:
        result["mean_xgb"] = modelos["mean_xgb"].predict(dmat)
    # Media trapezoidal (calculada post-hoc)
    result["mean_trap"] = media_trapezoidal({t: result[t] for t in taus})
    return result


###############################################################################
# PREPARACIÓN DE DATOS
###############################################################################

def get_feature_cols(df):
    excluir = COLS_EXCLUIR | {"fecha_th"}
    return [c for c in df.columns
            if c not in excluir and df[c].dtype.kind in ("f", "i", "u", "b")]


def preparar_fold_data(df, fold, cols_feat):
    train_end   = fold["train_end"]
    val_start   = fold["val_start"]
    test_start  = fold["test_start"]
    test_end    = fold["test_end"]
    train_start = fold["train_start"]

    mask_train = (df["fecha_t"] >= train_start) & (df["fecha_t"] <= train_end)
    mask_val   = (df["fecha_t"] >= val_start)   & (df["fecha_t"] <  test_start)
    mask_test  = (df["fecha_t"] >= test_start)  & (df["fecha_t"] <= test_end)

    df_all = df[mask_train | mask_val | mask_test].copy()
    garch_cols = [c for c in ["garch_vol", "garch_vol_tc", "garch_vol_embi"]
                  if c in df_all.columns]
    if garch_cols:
        df_all = reemplazar_garch_fold(df_all, train_end)

    df_train = df_all[df_all["fecha_t"] <= train_end]
    df_val   = df_all[(df_all["fecha_t"] >= val_start) & (df_all["fecha_t"] < test_start)]
    df_test  = df_all[df_all["fecha_t"] >= test_start]

    medianas = df_train[cols_feat].median()
    for _df in (df_train, df_val, df_test):
        _df[cols_feat] = _df[cols_feat].fillna(medianas)

    mt  = df_train["target"].notna()
    mv  = df_val["target"].notna()
    mte = df_test["target"].notna()

    X_train       = df_train.loc[mt,  cols_feat]
    y_train       = df_train.loc[mt,  "target"]
    X_val         = df_val.loc[mv,    cols_feat]
    y_val         = df_val.loc[mv,    "target"]
    X_test        = df_test.loc[mte,  cols_feat]
    y_test        = df_test.loc[mte,  "target"]
    h_test        = df_test.loc[mte,  "h"].values
    fechas_t_test = pd.to_datetime(df_test.loc[mte, "fecha_t"].values)

    return X_train, y_train, X_val, y_val, X_test, y_test, h_test, fechas_t_test


###############################################################################
# FAN CHART CON MEDIA TRAPEZOIDAL
###############################################################################

def graficar_fanchart(preds, y_test, h_test, fechas_t_test, fold):
    fechas_unicas = pd.DatetimeIndex(sorted(set(fechas_t_test)))
    if len(fechas_unicas) == 0:
        return

    test_start = fold["test_start"]
    meses_paso = 12.0 / FANCHART_N_SNAPSHOTS
    origenes   = []
    for i in range(FANCHART_N_SNAPSHOTS):
        target  = test_start + pd.DateOffset(months=int(round(i * meses_paso)))
        diffs   = np.abs((fechas_unicas - target).total_seconds())
        nearest = fechas_unicas[np.argmin(diffs)]
        if nearest not in origenes:
            origenes.append(nearest)
    if not origenes:
        return

    ncols = 2
    nrows = int(np.ceil(len(origenes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5), sharey=False)
    axes_flat  = axes.flatten() if len(origenes) > 1 else [axes]

    fig.suptitle(
        f"Fan chart TEST OOS — Fold {fold['fold']} — {BANCO} [ROLLING]\n"
        f"TEST: {fold['test_start'].date()} → {fold['test_end'].date()}  |  "
        f"TRAIN hasta: {fold['train_end'].date()}\n"
        f"Azul discontinuo=Q50  ·  Navy=Media XGB  ·  Firebrick=Media Trapezoidal",
        fontweight="bold", fontsize=10,
    )

    for ax, t0 in zip(axes_flat, origenes):
        mask = np.array([pd.Timestamp(f) == t0 for f in fechas_t_test])
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        h_s = h_test[mask]
        y_s = y_test[mask]
        p_s = {k: arr[mask] for k, arr in preds.items()}

        order = np.argsort(h_s)
        h_s   = h_s[order]
        y_s   = y_s[order]
        p_s   = {k: arr[order] for k, arr in p_s.items()}

        # Bandas de incertidumbre
        if {0.01, 0.99}.issubset(p_s):
            ax.fill_between(h_s, p_s[0.01] / 1e6, p_s[0.99] / 1e6,
                            alpha=0.10, color="steelblue", label="Q01-Q99")
        if {0.05, 0.95}.issubset(p_s):
            ax.fill_between(h_s, p_s[0.05] / 1e6, p_s[0.95] / 1e6,
                            alpha=0.25, color="steelblue", label="Q05-Q95")

        # Q50 — mediana
        if 0.50 in p_s:
            ax.plot(h_s, p_s[0.50] / 1e6, color="steelblue", lw=1.4,
                    ls="--", zorder=3, label="Q50 (mediana)")

        # Media XGB (reg:squarederror)
        if "mean_xgb" in p_s:
            ax.plot(h_s, p_s["mean_xgb"] / 1e6, color="navy", lw=2.0,
                    zorder=4, label="Media XGB")

        # Media Trapezoidal — nueva
        if "mean_trap" in p_s:
            ax.plot(h_s, p_s["mean_trap"] / 1e6, color="firebrick", lw=2.2,
                    zorder=5, label="Media Trap. (E[X] cuantiles)")

        # Realizados
        q_lo   = p_s.get(0.05, np.full_like(y_s, -np.inf))
        q_hi   = p_s.get(0.95, np.full_like(y_s,  np.inf))
        dentro = (y_s >= q_lo) & (y_s <= q_hi)
        ax.plot(h_s, y_s / 1e6, color="dimgray", lw=1.0, ls="--",
                alpha=0.75, zorder=4, label="Realizado")
        ax.scatter(h_s[dentro],  y_s[dentro]  / 1e6, color="seagreen",
                   s=18, zorder=6, label="Dentro Q05-Q95")
        ax.scatter(h_s[~dentro], y_s[~dentro] / 1e6, color="crimson",
                   s=18, zorder=6, label="Fuera Q05-Q95")

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

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(handles, labels, fontsize=7, loc="best")
    for ax in axes_flat[len(origenes):]:
        ax.set_visible(False)

    plt.tight_layout()
    nombre = DIR_OUTPUT / f"fanchart_trap_fold{fold['fold']:02d}_{BANCO}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Fan chart guardado: {nombre.name}")


###############################################################################
# PIPELINE PRINCIPAL
###############################################################################

def main():
    logger.info("=" * 65)
    logger.info(f"aux_fanchart_media_trapezoidal — {BANCO} [ROLLING]")
    logger.info(f"QUANTILES: {QUANTILES}")
    logger.info(f"TRAIN={VENTANA_TRAIN_AÑOS}yr | VAL={VENTANA_VAL_AÑOS}yr | "
                f"TEST={VENTANA_TEST_AÑOS}yr | embargo={EMBARGO_DIAS_HAB}dh | "
                f"trials={N_TRIALS_OPTUNA}")
    logger.info("=" * 65)

    if not RUTA_MATRIZ.exists():
        logger.error(f"Matriz no encontrada: {RUTA_MATRIZ}")
        return

    t0 = time.time()
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", BANCO)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    if df.empty or df["target"].notna().sum() < 500:
        logger.error(f"Datos insuficientes para {BANCO}")
        return

    cols_feat = get_feature_cols(df)
    fechas    = pd.DatetimeIndex(df["fecha_t"].unique())
    logger.info(f"  {len(df):,} filas | {len(cols_feat)} features | "
                f"rango: {fechas.min().date()} → {fechas.max().date()}")

    folds = generar_folds(fechas)
    if N_MAX_FOLDS is not None:
        folds = folds[:N_MAX_FOLDS]

    logger.info(f"  {len(folds)} folds generados:")
    for f in folds:
        logger.info(
            f"    Fold {f['fold']:2d} | "
            f"TRAIN {f['train_start'].date()} → {f['train_end'].date()} | "
            f"VAL  {f['val_start'].date()} → {f['val_end'].date()} | "
            f"TEST {f['test_start'].date()} → {f['test_end'].date()}"
        )

    for fold in folds:
        t_fold = time.time()
        logger.info(f"\n  ── Fold {fold['fold']}/{len(folds)} ──────────────────────")

        try:
            (X_tr, y_tr, X_va, y_va,
             X_te, y_te, h_te, fechas_te) = preparar_fold_data(df, fold, cols_feat)
        except Exception as e:
            logger.warning(f"  Fold {fold['fold']}: error preparando datos — {e}")
            continue

        if len(X_tr) < 200 or len(X_va) < 20 or len(X_te) < 20:
            logger.warning(f"  Fold {fold['fold']}: datos insuficientes — omitiendo")
            continue

        std_y = float(y_tr.std())
        logger.info(f"    X_train={len(X_tr):,} | X_val={len(X_va):,} | "
                    f"X_test={len(X_te):,} | std_y={std_y:,.0f}")

        modelos   = entrenar_fold_xgb_qt(X_tr, y_tr, X_va, y_va, std_y, fold["fold"])
        preds_te  = predecir(modelos, X_te)

        # Log cobertura y diferencias entre estimadores
        q_lo = preds_te.get(0.05, np.full(len(y_te), -np.inf))
        q_hi = preds_te.get(0.95, np.full(len(y_te),  np.inf))
        cov  = coverage(y_te.values, q_lo, q_hi)

        q50_mean  = float(preds_te[0.50].mean())        if 0.50 in preds_te         else float("nan")
        xgb_mean  = float(preds_te["mean_xgb"].mean()) if "mean_xgb" in preds_te    else float("nan")
        trap_mean = float(preds_te["mean_trap"].mean()) if "mean_trap" in preds_te   else float("nan")

        logger.info(
            f"    coverage_90={cov:.1%}  |  "
            f"media Q50={q50_mean/1e6:+.1f}MM  "
            f"media_xgb={xgb_mean/1e6:+.1f}MM  "
            f"media_trap={trap_mean/1e6:+.1f}MM  "
            f"({(time.time()-t_fold)/60:.1f} min)"
        )

        graficar_fanchart(preds_te, y_te.values, h_te, fechas_te, fold)

        del X_tr, y_tr, X_va, y_va, X_te, y_te, modelos
        gc.collect()

    logger.info(f"\n✓ Completado en {(time.time()-t0)/60:.1f} min  →  {DIR_OUTPUT}")


if __name__ == "__main__":
    main()
