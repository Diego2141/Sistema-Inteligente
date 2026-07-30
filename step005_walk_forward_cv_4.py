"""
step005_walk_forward_cv_4.py
============================
DIRECT multi-step forecasting strategy for XGBoost Quantile Regression.

Key difference vs cv3:
  - One model per (fold × h × tau) — each model specialises in exactly h bdays ahead
  - h and log_h are NOT features
  - Fixed hyperparameters (no Optuna; per-h dataset is ~1/74 of total)
  - Output dir: step005_wfcv_v4_direct
"""

from __future__ import annotations

import gc
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
xgb.set_config(verbosity=0)   # silencia warnings C++ (binary format, etc.)

try:
    import shap as _shap_lib
    _SHAP_OK = True
except ImportError:
    _shap_lib = None
    _SHAP_OK = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_OK = True
except ImportError:
    optuna = None
    _OPTUNA_OK = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Silenciar loggers internos de matplotlib (findfont, colorbar, pcolormesh)
# y PIL que heredan el nivel DEBUG del root logger
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"

BANCO   = "SISTEMA"
H_MIN   = 2
H_MAX   = 75
TAUS    = [0.01, 0.05, 0.40, 0.50, 0.60, 0.95, 0.99]

EXPANDING             = True
RECORTAR_INICIO_TRAIN = True
TRAIN_INICIO_CUTOFF   = "2020-01-01"

VENTANA_TRAIN_AÑOS  = 3
VENTANA_VAL_AÑOS    = 0.5
VENTANA_TEST_AÑOS   = 0.5
PASO_AÑOS           = 0.5

# Purge & embargo — derivados de los parámetros del modelo
LOOKBACK_MAX_DIAS   = 22   # ventana máxima de lags del target usados como features
PURGE_DIAS_HAB      = H_MAX + LOOKBACK_MAX_DIAS   # 75 + 22 = 97
EMBARGO_DIAS_HAB    = LOOKBACK_MAX_DIAS            # 22: autocorrelación serial en bordes
# Gap total aplicado simétricamente: TRAIN→VAL y VAL→TEST
GAP_DIAS_HAB        = PURGE_DIAS_HAB + EMBARGO_DIAS_HAB   # 97 + 22 = 119

MIN_TRAIN_ROWS      = 50

# ---------------------------------------------------------------------------
# Early stopping + Optuna
# ---------------------------------------------------------------------------
N_ESTIMATORS_MAX      = 300   # techo de árboles; early stopping lo reduce en práctica
EARLY_STOPPING_ROUNDS = 10    # parar si val loss no mejora en N rondas consecutivas

USE_OPTUNA          = True   # False → HP fijos para todos los h (más rápido, sin optuna)
OPTUNA_N_TRIALS     = 30     # trials por h representativo por fold
OPTUNA_WARM_START   = True   # True → inyecta el HP óptimo del fold anterior como trial 0
                              # False → cada fold parte desde cero (comportamiento original)

# Modo debug: corre solo el primer fold para verificar el pipeline rápidamente
# True  → ejecuta únicamente fold 1 (el más antiguo); ideal para probar cambios
# False → corre todos los folds (comportamiento normal)
DEBUG_SINGLE_FOLD   = False

# Objetivo suavizado Pinball-Arctan (paper 2406.02293)
# True  → reemplaza reg:quantileerror por gradiente/hessiana suavizados
#          grad ∝ arctan(u/s)/π + ... | hess = ((s²+σ²)/(s²+u²))² > 0 siempre
#          Ventaja: hessiana no-nula → pasos Newton reales; gradiente tiene magnitud
# False → reg:quantileerror estándar (gradiente binario {-τ, 1-τ}, hess≈1 surrogate)
AJUSTE_ARCTAN       = True
# Cuando AJUSTE_ARCTAN=True, Optuna siempre estima s_factor en [0.01, 1.0] (log)
# s = s_factor × std_y(y_train del h) — rango cubre desde muy suavizado a muy agudo
OPTUNA_N_TRIALS_ARCTAN = 40  # trials cuando AJUSTE_ARCTAN=True (vs 30 base)
                               # +10 para compensar el HP extra s_factor

# Grupos de h y sus representantes para Optuna (Opción C)
# Un solo h "típico" por grupo → HP se buscan ahí y se transfieren a todo el grupo
# muy_corto separado: banco pre-reporta retiros a t+2 y depósitos a t+1 →
#   señal cualitativamente diferente a horizontes mayores
H_GRUPOS: dict = {
    "muy_corto": (list(range(H_MIN, 6)),          3),   # h=2–5,   rep h=3  (~1 semana hábil)
    "corto":     (list(range(6, 21)),            13),   # h=6–20,  rep h=13 (~1–4 semanas)
    "medio":     (list(range(21, 51)),           35),   # h=21–50, rep h=35 (~1–2.5 meses)
    "largo":     (list(range(51, H_MAX + 1)),    62),   # h=51–75, rep h=62 (~2.5–3.5 meses)
}

# Fixed hyperparameters — valores por defecto (Optuna sobreescribe los buscados)
HP: dict = {
    "n_estimators"    : N_ESTIMATORS_MAX,  # early stopping determina el real
    "max_depth"       : 3,
    "learning_rate"   : 0.08,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 1.0,
    "tree_method"     : "hist",
    "max_bin"         : 64,
    "n_jobs"          : 1,
    "random_state"    : 42,
}

# Columns never used as features (KEY: h and log_h excluded in cv4)
COLS_EXCLUIR = {
    "fecha_t", "banco", "target", "fecha_th",
    "h", "log_h",
}

# ---------------------------------------------------------------------------
# Feature diagnostics (gain / block-permutation / SHAP) per horizon
# ---------------------------------------------------------------------------
DIAG_FEATURES         = True   # False → skip (más rápido)
DIAG_BLOCK_SIZE       = 5      # bloques pequeños: val solo tiene ~120 filas
DIAG_N_REPEATS        = 3      # repeticiones por permutación
DIAG_SHAP_MAX_SAMPLES = None   # None = todas las filas de val (~120)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns not in COLS_EXCLUIR."""
    return [
        c for c in df.columns
        if c not in COLS_EXCLUIR and df[c].dtype.kind in "fiub"
    ]


def build_folds(df: pd.DataFrame) -> list[dict]:
    """
    Build walk-forward CV folds (expanding or rolling) with symmetric purge+embargo gaps.

    Window type is controlled by the EXPANDING flag:
      EXPANDING=True  → train_start anchored to train_min (growing window)
      EXPANDING=False → train_start = val_start - VENTANA_TRAIN_AÑOS (fixed-size rolling window)
                        clamped to train_min so early folds never go below the data origin.

    Gap structure (applied identically on both sides):
        TRAIN..train_end | GAP_DIAS_HAB | val_start..val_end | GAP_DIAS_HAB | test_start..test_end

    GAP = PURGE (H_MAX + LOOKBACK_MAX_DIAS) + EMBARGO (LOOKBACK_MAX_DIAS)
        = 97 + 22 = 119 business days

    The gap prevents TARGET values from TRAIN/VAL from appearing as LAG FEATURES
    in VAL/TEST (direct leakage) and adds an embargo for serial correlation at
    period boundaries.

    Returns a list of dicts with keys:
        fold, train_start, train_end, val_start, val_end, test_start, test_end
    """
    all_bdays = np.array(sorted(df["fecha_t"].unique()), dtype="datetime64[ns]")

    if RECORTAR_INICIO_TRAIN:
        train_min = max(df["fecha_t"].min(), pd.Timestamp(TRAIN_INICIO_CUTOFF))
    else:
        train_min = df["fecha_t"].min()

    def _offset(years: float) -> pd.DateOffset:
        """Convert fractional years to a DateOffset using months."""
        months = round(years * 12)
        return pd.DateOffset(months=months)

    # First test window starts after train + val
    test_start = train_min + _offset(VENTANA_TRAIN_AÑOS + VENTANA_VAL_AÑOS)

    folds = []
    fold_num = 0

    while True:
        test_end = test_start + _offset(VENTANA_TEST_AÑOS) - pd.Timedelta(days=1)

        if test_end > df["fecha_t"].max():
            break

        # --- GAP2: purge + embargo entre VAL y TEST ---
        # val_end se calcula retrocediendo GAP_DIAS_HAB antes de test_start
        idx_test = int(np.searchsorted(all_bdays, np.datetime64(test_start, "ns")))
        val_end_idx = max(0, idx_test - GAP_DIAS_HAB)
        val_end = pd.Timestamp(all_bdays[val_end_idx])

        # VAL: ventana de VENTANA_VAL_AÑOS terminando en val_end
        val_start = val_end - _offset(VENTANA_VAL_AÑOS)

        if EXPANDING:
            train_start = train_min
        else:
            train_start = max(train_min, val_start - _offset(VENTANA_TRAIN_AÑOS))

        # --- GAP1: purge + embargo entre TRAIN y VAL (simétrico a GAP2) ---
        idx_val = int(np.searchsorted(all_bdays, np.datetime64(val_start, "ns")))
        train_end_idx = max(0, idx_val - GAP_DIAS_HAB)
        train_end = pd.Timestamp(all_bdays[train_end_idx])

        fold_num += 1
        folds.append({
            "fold"       : fold_num,
            "train_start": pd.Timestamp(train_start),
            "train_end"  : train_end,
            "val_start"  : pd.Timestamp(val_start),
            "val_end"    : val_end,
            "test_start" : pd.Timestamp(test_start),
            "test_end"   : pd.Timestamp(test_end),
        })

        log.debug(
            "Fold %d | train %s..%s | [gap %dd] | val %s..%s | [gap %dd] | test %s..%s",
            fold_num,
            train_start.date(), train_end.date(),
            GAP_DIAS_HAB,
            val_start.date(),   val_end.date(),
            GAP_DIAS_HAB,
            test_start.date(),  test_end.date(),
        )

        test_start += _offset(PASO_AÑOS)

    return folds


def _strip_tz(series: pd.Series) -> pd.Series:
    """Remove timezone information from a datetime Series if present."""
    if hasattr(series.dt, "tz") and series.dt.tz is not None:
        return series.dt.tz_convert(None)
    return series


def preparar_fold_data_h(
    df_h: pd.DataFrame,
    fold: dict,
    cols_feat: list[str],
) -> tuple:
    """
    Split the single-h DataFrame into train/val/test partitions for one fold.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, fechas_t_test, fecha_th_test

    Raises
    ------
    ValueError  if train rows < MIN_TRAIN_ROWS or test rows == 0
    """
    mt  = (
        (df_h["fecha_t"] >= fold["train_start"]) &
        (df_h["fecha_t"] <= fold["train_end"]) &
        df_h["target"].notna()
    )
    mv  = (
        (df_h["fecha_t"] >= fold["val_start"]) &
        (df_h["fecha_t"] <= fold["val_end"]) &
        df_h["target"].notna()
    )
    mte = (
        (df_h["fecha_t"] >= fold["test_start"]) &
        (df_h["fecha_t"] <= fold["test_end"]) &
        df_h["target"].notna()
    )

    n_train = mt.sum()
    n_test  = mte.sum()

    if n_train < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Insuficientes filas de entrenamiento: {n_train} < {MIN_TRAIN_ROWS}"
        )
    if n_test == 0:
        raise ValueError("Sin filas de test para este fold/h")

    X_train = df_h.loc[mt, cols_feat]
    y_train = df_h.loc[mt, "target"]

    X_val   = df_h.loc[mv, cols_feat]
    y_val   = df_h.loc[mv, "target"]

    X_test  = df_h.loc[mte, cols_feat]
    y_test  = df_h.loc[mte, "target"]

    fechas_t_test = _strip_tz(df_h.loc[mte, "fecha_t"])

    if "fecha_th" in df_h.columns:
        fecha_th_test = _strip_tz(pd.to_datetime(df_h.loc[mte, "fecha_th"]))
    else:
        fecha_th_test = pd.Series([pd.NaT] * n_test, index=df_h.index[mte])

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        fechas_t_test,
        fecha_th_test,
    )


# ---------------------------------------------------------------------------
# Objetivo Pinball-Arctan suavizado (activo solo si AJUSTE_ARCTAN = True)
# Referencia: paper 2406.02293
# ---------------------------------------------------------------------------

def _make_quantile_objective(tau: float, s: float, std_y: float):
    """
    Devuelve la función objetivo suavizada (grad, hess) para xgb.train().
    s      — parámetro de suavizado; s = s_factor × std_y  (s_factor estimado por Optuna)
    std_y  — desviación estándar de y_train; normaliza la hessiana
    _scale — factor que ajusta el lambda efectivo de XGBoost a la escala
             del target no estandarizado (std_y ~ 80,000 → _scale ~ 1e9)
    """
    _scale = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)

    def objective(y_pred: np.ndarray, dtrain: "xgb.DMatrix"):
        y_pred = np.clip(y_pred, -1e15, 1e15)   # evita overflow con HP malos en Optuna
        u      = dtrain.get_label() - y_pred
        s2u2   = s ** 2 + u ** 2
        grad   = -((tau - 0.5 + np.arctan(u / s) / np.pi)
                   + u * s / (np.pi * s2u2)) * _scale
        # Forma simplificada que evita overflow numérico: hess = ((s²+std_y²)/(s²+u²))²
        hess   = ((s ** 2 + std_y ** 2) / s2u2) ** 2
        return grad, hess

    return objective


def _make_pinball_metric(tau: float):
    """Métrica pinball para custom_metric en xgb.train(); monitorea el eval set."""
    def metric(y_pred: np.ndarray, dtrain: "xgb.DMatrix"):
        u = dtrain.get_label() - y_pred
        return "pinball", float(np.mean(np.where(u >= 0, tau * u, (tau - 1) * u)))
    return metric


class _PinballEarlyStopping(xgb.callback.TrainingCallback):
    """
    Early stopping sobre la métrica 'pinball' del eval set de validación.
    Reemplaza xgb.callback.EarlyStopping porque ese callback busca el nombre
    de la métrica en un dict plano ("val-pinball"), pero XGBoost con
    custom_metric lo almacena en estructura anidada: evals_log["val"]["pinball"].
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


class _ArcTanBooster:
    """
    Wrapper mínimo sobre xgb.Booster para exponer .predict(X_df) compatible
    con XGBRegressor — permite usar xgb.train() con objetivo custom sin
    modificar el resto del pipeline (diagnosticar_h, scaffolds, métricas).
    """
    def __init__(self, booster: "xgb.Booster"):
        self._b = booster

    def predict(self, X) -> np.ndarray:
        return self._b.predict(xgb.DMatrix(X))

    def get_booster(self) -> "xgb.Booster":
        """Compatibilidad con XGBRegressor.get_booster() — usado en gain y SHAP."""
        return self._b

    @property
    def best_iteration(self) -> int:
        return self._b.num_boosted_rounds()


# Claves de HP que xgb.train() acepta en params (excluye sklearn-only)
_XGB_TRAIN_SKIP = {"n_estimators", "random_state", "n_jobs",
                   "early_stopping_rounds", "eval_metric",
                   "s_factor"}   # s_factor → usado en _make_quantile_objective, no en params


def _hp_para_xgb_train(hp: dict) -> dict:
    """Filtra hp_h para que sea compatible con xgb.train() params."""
    p = {k: v for k, v in hp.items() if k not in _XGB_TRAIN_SKIP}
    p.setdefault("nthread", 1)
    p.setdefault("seed",    hp.get("random_state", 42))
    return p


def entrenar_modelos_h(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hp: dict,
    std_y: float = 0.0,   # requerido solo si AJUSTE_ARCTAN = True
) -> dict:
    """
    Train one model per quantile tau plus one for mean.
    Uses early stopping on val when val is non-empty.

    AJUSTE_ARCTAN = False (default):
        XGBRegressor con reg:quantileerror — comportamiento original.
    AJUSTE_ARCTAN = True:
        xgb.train() con objetivo Pinball-Arctan suavizado (paper 2406.02293).
        El booster se envuelve en _ArcTanBooster para compatibilidad con predict(X_df).
        'mean' siempre usa reg:squarederror (arctan no aplica a la media).

    Returns
    -------
    dict {tau (float | 'mean'): XGBRegressor | _ArcTanBooster}
    """
    modelos: dict = {}
    use_es = len(X_val) > 0

    if AJUSTE_ARCTAN:
        # ── Rama arctan: xgb.train() + _PinballEarlyStopping ────────────────
        # s_factor estimado por Optuna; viaja en hp como cualquier otro HP
        s      = hp.get("s_factor", 0.05) * std_y
        params = _hp_para_xgb_train(hp)   # _XGB_TRAIN_SKIP excluye s_factor
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val) if use_es else None
        evals  = [(dval, "val")] if use_es else []

        for tau in TAUS:
            # IMPORTANTE: nueva instancia por tau — _PinballEarlyStopping tiene
            # estado interno (_best, _since). Reutilizar el mismo objeto entre taus
            # causa early stopping prematuro: el _best de Q01 (~30 MM) es mucho
            # menor que el pinball inicial de Q05 (~160 MM), disparando stop en
            # EARLY_STOPPING_ROUNDS iteraciones aunque Q05 aún no haya convergido.
            cbs = [_PinballEarlyStopping(rounds=EARLY_STOPPING_ROUNDS)] if use_es else []
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=N_ESTIMATORS_MAX,
                obj=_make_quantile_objective(tau, s, std_y),
                custom_metric=_make_pinball_metric(tau) if use_es else None,
                evals=evals,
                callbacks=cbs,
                verbose_eval=False,
            )
            modelos[tau] = _ArcTanBooster(booster)

    else:
        # ── Rama estándar: XGBRegressor con reg:quantileerror ────────────────
        for tau in TAUS:
            m = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=tau,
                eval_metric="quantile",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS if use_es else None,
                **hp,
            )
            if use_es:
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                m.fit(X_train, y_train, verbose=False)
            modelos[tau] = m

    # ── Media: siempre reg:squarederror (sin arctan) ─────────────────────────
    m_mean = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if use_es else None,
        **hp,
    )
    if use_es:
        m_mean.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        m_mean.fit(X_train, y_train, verbose=False)
    modelos["mean"] = m_mean

    return modelos


def calcular_metricas(
    preds_dict: dict,
    y_test: np.ndarray,
    h_val: int,
    fold_num: int,
    preds_val_dict: dict | None = None,
    y_val: np.ndarray | None = None,
) -> dict:
    """
    Compute:
      - Pinball loss por cuantil (TEST)
      - RMSE (TEST)
      - Coverage empírica TEST 90% y 98%
      - Coverage empírica VAL 90% y 98%   (para detectar overfitting de calibración)
      - Winkler Score 90% y 98%           (penaliza anchura + violaciones)
      - Interval Sharpness 90%            (anchura media sin penalización)
      - CRPS aproximado (trapz sobre taus discretos)
      - Calibración hit-rate por cuantil  (empírico debe ≈ tau)
      - Pinball relativo (/ std y_test)   (comparable entre horizontes)

    Returns a flat dict suitable for appending to a list before pd.DataFrame().
    """
    row: dict = {"fold": fold_num, "h": h_val}
    tau_pinballs: dict[float, float] = {}

    # ── Pinball + RMSE ────────────────────────────────────────────────────────
    for tau, preds in preds_dict.items():
        if tau == "mean":
            row["rmse"] = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        else:
            err    = y_test - preds
            pb     = np.where(err >= 0, tau * err, (tau - 1) * err)
            pb_val = float(np.mean(pb))
            row[f"pinball_q{int(tau * 100):02d}"] = pb_val
            tau_pinballs[tau] = pb_val

    # ── Coverage TEST ─────────────────────────────────────────────────────────
    if 0.05 in preds_dict and 0.95 in preds_dict:
        row["coverage_90"] = float(
            ((y_test >= preds_dict[0.05]) & (y_test <= preds_dict[0.95])).mean()
        )
    if 0.01 in preds_dict and 0.99 in preds_dict:
        row["coverage_98"] = float(
            ((y_test >= preds_dict[0.01]) & (y_test <= preds_dict[0.99])).mean()
        )

    # ── Coverage VAL + CRPS VAL ───────────────────────────────────────────────
    if preds_val_dict is not None and y_val is not None and len(y_val) > 0:
        if 0.05 in preds_val_dict and 0.95 in preds_val_dict:
            row["val_coverage_90"] = float(
                ((y_val >= preds_val_dict[0.05]) & (y_val <= preds_val_dict[0.95])).mean()
            )
        if 0.01 in preds_val_dict and 0.99 in preds_val_dict:
            row["val_coverage_98"] = float(
                ((y_val >= preds_val_dict[0.01]) & (y_val <= preds_val_dict[0.99])).mean()
            )
        # CRPS VAL ≈ 2 * ∫₀¹ pinball_τ dτ — permite comparar VAL vs TEST
        # para detectar overfitting distribucional (val_crps << crps → overfit)
        tau_pinballs_val: dict[float, float] = {}
        for tau, preds in preds_val_dict.items():
            if tau != "mean":
                err = y_val - preds
                tau_pinballs_val[tau] = float(
                    np.where(err >= 0, tau * err, (tau - 1) * err).mean()
                )
        if len(tau_pinballs_val) >= 2:
            sorted_taus_v = sorted(tau_pinballs_val.keys())
            pb_arr_v = np.array([tau_pinballs_val[t] for t in sorted_taus_v])
            row["val_crps"] = float(2.0 * np.trapz(pb_arr_v, sorted_taus_v))

    # ── Winkler Score: W = (U-L) + (2/α)*[max(0,L-y) + max(0,y-U)] ──────────
    if 0.05 in preds_dict and 0.95 in preds_dict:
        L, U  = preds_dict[0.05], preds_dict[0.95]
        alpha = 0.10
        width = U - L
        penalty = (2 / alpha) * (np.maximum(0.0, L - y_test) + np.maximum(0.0, y_test - U))
        row["winkler_90"]   = float((width + penalty).mean())
        row["sharpness_90"] = float(width.mean())
    if 0.01 in preds_dict and 0.99 in preds_dict:
        L, U  = preds_dict[0.01], preds_dict[0.99]
        alpha = 0.02
        width = U - L
        penalty = (2 / alpha) * (np.maximum(0.0, L - y_test) + np.maximum(0.0, y_test - U))
        row["winkler_98"] = float((width + penalty).mean())

    # ── CRPS ≈ 2 * ∫₀¹ pinball_τ dτ  (integración trapezoidal, 7 cuantiles) ──
    if len(tau_pinballs) >= 2:
        sorted_taus = sorted(tau_pinballs.keys())
        pb_arr = np.array([tau_pinballs[t] for t in sorted_taus])
        row["crps"] = float(2.0 * np.trapz(pb_arr, sorted_taus))

    # ── Calibración hit-rate por cuantil (empírico debe ≈ tau nominal) ────────
    for tau, preds in preds_dict.items():
        if tau != "mean":
            row[f"calib_q{int(tau * 100):02d}"] = float((y_test <= preds).mean())

    # ── Pinball relativo (/ std target TEST) — comparable entre horizontes ────
    y_std = float(np.std(y_test))
    if y_std > 0:
        for tau, pb in tau_pinballs.items():
            row[f"pinball_rel_q{int(tau * 100):02d}"] = pb / y_std

    return row


# ---------------------------------------------------------------------------
# Feature diagnostics helpers
# ---------------------------------------------------------------------------

def _pinball(y: np.ndarray, yhat: np.ndarray, tau: float) -> float:
    err = y - yhat
    return float(np.where(err >= 0, tau * err, (tau - 1) * err).mean())


def _sincos_pairs(cols: list[str]) -> dict[str, tuple[str, str]]:
    """
    Detecta features cíclicos codificados como par _sin / _cos.
    Devuelve {base: (sin_col, cos_col)} solo cuando AMBOS componentes están en cols.
    Ejemplo: 'dias_al_cierre_mes' → ('dias_al_cierre_mes_sin', 'dias_al_cierre_mes_cos')
    """
    sin_map = {c[:-4]: c for c in cols if c.endswith("_sin")}
    cos_map = {c[:-4]: c for c in cols if c.endswith("_cos")}
    return {base: (sin_map[base], cos_map[base])
            for base in sin_map if base in cos_map}


def _consolidar_sincos_pivot(
    pivot: pd.DataFrame,
    pairs: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """
    En un pivot (features × h), suma filas sin+cos en una única fila 'base'
    y elimina las filas individuales.
    Para perm: las filas ya llevan el Δ conjunto (sin=joint, cos=0) → la suma = joint.
    Para gain/shap: la suma aproxima la importancia total del feature circular.
    """
    rows_new: dict[str, pd.Series] = {}
    rows_drop: list[str] = []
    for base, (sin_c, cos_c) in pairs.items():
        if sin_c in pivot.index and cos_c in pivot.index:
            rows_new[base] = pivot.loc[sin_c] + pivot.loc[cos_c]
            rows_drop += [sin_c, cos_c]
    if rows_new:
        pivot = pd.concat(
            [pivot.drop(index=rows_drop), pd.DataFrame(rows_new).T]
        )
    return pivot


def _diag_gain_h(modelos: dict, cols_feat: list[str]) -> pd.Series:
    """Gain promedio entre cuantiles (in-sample, solo informativo)."""
    acum = {f: 0.0 for f in cols_feat}
    n = 0
    for tau, model in modelos.items():
        if tau == "mean":
            continue
        imp = model.get_booster().get_score(importance_type="gain")
        for f in cols_feat:
            acum[f] += float(imp.get(f, 0.0))
        n += 1
    if n:
        acum = {f: v / n for f, v in acum.items()}
    return pd.Series(acum, dtype=float)


def _diag_perm_h(
    modelos: dict,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cols_feat: list[str],
) -> pd.Series:
    """Block-permutation importance en VAL (OOS). Δpinball promedio entre cuantiles."""
    X = X_val[cols_feat].reset_index(drop=True).copy()
    y = np.asarray(y_val)
    n = len(X)
    bs = max(2, min(DIAG_BLOCK_SIZE, n // 3))
    block_starts = np.arange(0, n, bs)
    rng = np.random.default_rng(42)

    acum = pd.Series(0.0, index=cols_feat)
    n_tau = 0

    # Pares sin/cos → se permutar conjuntamente con el mismo shuffle
    pairs     = _sincos_pairs(cols_feat)
    paired_cs = {c for (sc, cc) in pairs.values() for c in (sc, cc)}

    for tau, model in modelos.items():
        if tau == "mean":
            continue
        base_preds = model.predict(X)
        base_loss  = _pinball(y, base_preds, tau)

        feat_deltas: dict[str, float] = {}

        # ── Features individuales (sin/cos no emparejados) ──────────────────
        for c in cols_feat:
            if c in paired_cs:
                continue  # se manejan abajo
            orig = X[c].values.copy()
            deltas = []
            for _ in range(DIAG_N_REPEATS):
                perm = rng.permutation(block_starts)
                new_col = np.concatenate([orig[s:s + bs] for s in perm])[:n]
                Xp = X.copy()
                Xp[c] = new_col
                deltas.append(_pinball(y, model.predict(Xp), tau) - base_loss)
            feat_deltas[c] = float(np.mean(deltas))

        # ── Pares sin/cos — permutación conjunta (mismo shuffle) ────────────
        # Strobl et al. (2008): permutar solo un componente deja al modelo
        # recuperar señal del otro → subestima importancia real del feature cíclico.
        # Solución: usar el mismo vector de permutación para ambos.
        # Almacenamos el Δ conjunto en sin_col y 0 en cos_col;
        # _consolidar_sincos_pivot los suma → importancia joint correcta.
        for base, (sin_c, cos_c) in pairs.items():
            orig_sin = X[sin_c].values.copy()
            orig_cos = X[cos_c].values.copy()
            deltas = []
            for _ in range(DIAG_N_REPEATS):
                perm   = rng.permutation(block_starts)
                Xp     = X.copy()
                Xp[sin_c] = np.concatenate([orig_sin[s:s + bs] for s in perm])[:n]
                Xp[cos_c] = np.concatenate([orig_cos[s:s + bs] for s in perm])[:n]
                deltas.append(_pinball(y, model.predict(Xp), tau) - base_loss)
            feat_deltas[sin_c] = float(np.mean(deltas))  # Δ joint completo
            feat_deltas[cos_c] = 0.0                     # placeholder; suma → Δ joint

        acum = acum.add(pd.Series(feat_deltas), fill_value=0.0)
        n_tau += 1

    if n_tau:
        acum /= n_tau
    return acum


def _shap_compat_booster(model):
    """
    Patch xgb.Booster.predict at CLASS level (once per session) to translate
    the deprecated ntree_limit kwarg to iteration_range (removed in XGBoost 2.0).

    Why class-level, not instance-level:
      SHAP < 0.43 has two distinct call sites for predict:
        1. TreeExplainer.__init__ (expected_value)  — accesses via instance __dict__
        2. _compute_tree_shap / assert_additivity   — calls xgb.Booster.predict(self, ...)
           directly as a class method, bypassing instance attribute lookup.
      An instance-level patch fixes site 1 but not site 2.  Patching the class
      method once fixes both sites for all Booster instances without touching the
      original XGBoost objects or requiring a library upgrade.
    """
    if not getattr(xgb.Booster, "_shap_ntree_patched", False):
        _orig_cls = xgb.Booster.predict

        def _compat(self, data, ntree_limit=None, **kw):
            if ntree_limit is not None and "iteration_range" not in kw:
                kw["iteration_range"] = (0, int(ntree_limit))
            return _orig_cls(self, data, **kw)

        xgb.Booster.predict = _compat
        xgb.Booster._shap_ntree_patched = True
        log.debug("xgb.Booster.predict parcheado para SHAP < 0.43 (una sola vez)")

    return model.get_booster()


def _diag_shap_h(
    modelos: dict,
    X_val: pd.DataFrame,
    cols_feat: list[str],
) -> pd.Series:
    """SHAP |mean| en VAL (OOS), promedio entre cuantiles."""
    if not _SHAP_OK:
        return pd.Series(np.nan, index=cols_feat)

    X = X_val[cols_feat].reset_index(drop=True)
    if DIAG_SHAP_MAX_SAMPLES and len(X) > DIAG_SHAP_MAX_SAMPLES:
        X = X.sample(DIAG_SHAP_MAX_SAMPLES, random_state=42)

    acum = pd.Series(0.0, index=cols_feat)
    n_tau = 0

    for tau, model in modelos.items():
        if tau == "mean":
            continue
        try:
            explainer = _shap_lib.TreeExplainer(_shap_compat_booster(model))
        except Exception as e:
            log.debug("SHAP τ=%.2f falló [init]: %s", tau, e)
            continue
        try:
            # check_additivity=False evita la llamada interna a predict(ntree_limit=N)
            # que SHAP < 0.43 usa para validar que sum(SHAP) == predicción del modelo
            sv = explainer.shap_values(X, check_additivity=False)
            s = pd.Series(np.abs(sv).mean(axis=0), index=cols_feat)
            acum = acum.add(s.fillna(0.0), fill_value=0.0)
            n_tau += 1
        except Exception as e:
            log.debug("SHAP τ=%.2f falló [values]: %s", tau, e)

    if n_tau == 0:
        return pd.Series(np.nan, index=cols_feat)
    return acum / n_tau


def diagnosticar_h(
    modelos: dict,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cols_feat: list[str],
    fold_num: int,
    h_val: int,
) -> dict:
    """Devuelve dict {fold, h, gain_<feat>, perm_<feat>, shap_<feat>}."""
    gain = _diag_gain_h(modelos, cols_feat)
    perm = _diag_perm_h(modelos, X_val, y_val, cols_feat)
    shp  = _diag_shap_h(modelos, X_val, cols_feat)

    row: dict = {"fold": fold_num, "h": h_val}
    for f in cols_feat:
        row[f"gain_{f}"] = float(gain.get(f, 0.0))
        row[f"perm_{f}"] = float(perm.get(f, 0.0))
        row[f"shap_{f}"] = float(shp.get(f, np.nan))
    return row


def guardar_hp_report(
    hp_rows: list[dict],
    dir_modo: "Path",
    banco: str,
    fecha_hoy: str,
) -> None:
    """
    Genera CSV + 2 gráficos con el reporte de hiperparámetros Optuna:

    1. Convergencia Optuna — curva 'mejor encontrado hasta el trial t' por
       fold × grupo → diagnostica si N_ESTIMATORS_MAX/OPTUNA_N_TRIALS es suficiente.
    2. Estabilidad HP — un subpanel por HP, mostrando su valor en cada fold
       para los 4 grupos → detecta inestabilidad o comportamiento anómalo.
    """
    if not hp_rows:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        log.warning("matplotlib no disponible — omitiendo reporte HP")
        return

    # ── CSV (sin la columna de listas) ───────────────────────────────────────
    df_hp = pd.DataFrame([{k: v for k, v in r.items() if k != "trial_values"}
                           for r in hp_rows])
    ruta_csv = dir_modo / f"hp_report_{banco}_{fecha_hoy}.csv"
    df_hp.to_csv(ruta_csv, index=False)
    log.info("HP report CSV: %s", ruta_csv.name)

    # Orden visual de grupos
    grupo_order = ["muy_corto", "corto", "medio", "largo"]
    grupo_colors = {"muy_corto": "#6D28D9", "corto": "#0369A1",
                    "medio": "#047857", "largo": "#B45309"}
    folds_sorted = sorted(df_hp["fold"].unique())

    # ── Gráfico 1: Convergencia Optuna ────────────────────────────────────────
    n_grupos = len(grupo_order)
    n_folds  = len(folds_sorted)
    fig1, axes1 = plt.subplots(
        1, n_grupos,
        figsize=(5 * n_grupos, 4.5),
        sharey=False,
        gridspec_kw={"wspace": 0.35},
    )
    if n_grupos == 1:
        axes1 = [axes1]

    fig1.suptitle(
        f"Convergencia Optuna por grupo — {banco}\n"
        f"Curva 'mejor hasta trial t' · {OPTUNA_N_TRIALS} trials · "
        f"¿converge antes del último trial?",
        fontsize=11, fontweight="bold", y=1.01,
    )

    cmap_folds = plt.cm.Set1
    for ax, grupo in zip(axes1, grupo_order):
        rows_g = [r for r in hp_rows if r["grupo"] == grupo]
        for i, row in enumerate(rows_g):
            tv    = row["trial_values"]
            if not tv:
                continue
            best_so_far = list(__import__("itertools").accumulate(tv, min))
            color = cmap_folds(i / max(n_folds, 1))
            ax.plot(range(1, len(best_so_far) + 1), best_so_far,
                    lw=1.6, color=color, alpha=0.85, label=f"Fold {row['fold']}")
            # marcar dónde deja de mejorar
            last_improvement = max(
                (j for j, (a, b) in enumerate(zip(best_so_far, best_so_far[1:]))
                 if b < a - 1e-8),
                default=len(best_so_far) - 1,
            )
            ax.axvline(last_improvement + 1, color=color, lw=0.7, ls=":", alpha=0.5)

        ax.set_title(
            f"{grupo}\n(h_rep={[r['h_rep'] for r in rows_g[:1]][0] if rows_g else '?'})",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Trial #", fontsize=9)
        ax.set_ylabel("Pinball val (mejor acumulado)", fontsize=8)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.grid(True, alpha=0.25)
        # línea vertical en N_TRIALS para referencia
        ax.axvline(OPTUNA_N_TRIALS, color="#DC2626", lw=1.2, ls="--", alpha=0.6,
                   label=f"N_TRIALS={OPTUNA_N_TRIALS}")
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    ruta_conv = dir_modo / f"hp_convergencia_{banco}_{fecha_hoy}.png"
    fig1.savefig(ruta_conv, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    log.info("HP convergencia: %s", ruta_conv.name)

    # ── Gráfico 2: Estabilidad HP por fold ───────────────────────────────────
    hp_names = ["max_depth", "min_child_weight", "learning_rate",
                 "reg_alpha", "reg_lambda", "subsample", "colsample_bytree"]
    hp_labels = {
        "max_depth"       : "max_depth",
        "min_child_weight": "min_child_weight",
        "learning_rate"   : "learning_rate",
        "reg_alpha"       : "reg_alpha (L1)",
        "reg_lambda"      : "reg_lambda (L2)",
        "subsample"       : "subsample",
        "colsample_bytree": "colsample_bytree",
    }

    n_hp  = len(hp_names)
    ncols = 4
    nrows = (n_hp + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols,
                                figsize=(ncols * 4.2, nrows * 3.2),
                                gridspec_kw={"hspace": 0.55, "wspace": 0.38})
    axes2_flat = axes2.flat if nrows > 1 else list(axes2)

    fig2.suptitle(
        f"Estabilidad de Hiperparámetros por fold — {banco}\n"
        f"4 grupos · línea por grupo · variación entre folds indica inestabilidad",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for ax, hp in zip(axes2_flat, hp_names):
        for grupo in grupo_order:
            vals = [r[hp] for r in hp_rows if r["grupo"] == grupo
                    and r.get(hp) is not None]
            fs   = [r["fold"] for r in hp_rows if r["grupo"] == grupo
                    and r.get(hp) is not None]
            if not vals:
                continue
            color = grupo_colors[grupo]
            ax.plot(fs, vals, "o-", lw=1.6, ms=5, color=color,
                    alpha=0.85, label=grupo)
            # rango observado como banda
            ax.fill_between(fs,
                             [min(vals)] * len(fs),
                             [max(vals)] * len(fs),
                             alpha=0.06, color=color)

        ax.set_title(hp_labels[hp], fontsize=9, fontweight="bold")
        ax.set_xlabel("Fold", fontsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=7, loc="best")

    # ocultar subplots sobrantes
    for ax in list(axes2_flat)[n_hp:]:
        ax.set_visible(False)

    plt.tight_layout()
    ruta_stab = dir_modo / f"hp_estabilidad_{banco}_{fecha_hoy}.png"
    fig2.savefig(ruta_stab, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info("HP estabilidad: %s", ruta_stab.name)

    # ── Tabla resumen en consola ──────────────────────────────────────────────
    print("\nHiperparámetros óptimos por fold y grupo:")
    col_print = ["fold", "grupo", "h_rep",
                 "max_depth", "min_child_weight", "learning_rate",
                 "reg_alpha", "reg_lambda", "subsample", "colsample_bytree",
                 "best_pinball_val"]
    df_show = df_hp[[c for c in col_print if c in df_hp.columns]].copy()
    df_show["learning_rate"]   = df_show["learning_rate"].map("{:.4f}".format)
    df_show["reg_alpha"]       = df_show["reg_alpha"].map("{:.3f}".format)
    df_show["reg_lambda"]      = df_show["reg_lambda"].map("{:.3f}".format)
    df_show["subsample"]       = df_show["subsample"].map("{:.3f}".format)
    df_show["colsample_bytree"]= df_show["colsample_bytree"].map("{:.3f}".format)
    df_show["best_pinball_val"]= df_show["best_pinball_val"].map("{:.4f}".format)
    print(df_show.to_string(index=False))

    # Diagnóstico N_TRIALS
    last_improvement_stats = []
    for r in hp_rows:
        tv = r["trial_values"]
        if not tv:
            continue
        best_so_far = list(__import__("itertools").accumulate(tv, min))
        last_imp = max(
            (j for j, (a, b) in enumerate(zip(best_so_far, best_so_far[1:]))
             if b < a - 1e-8),
            default=0,
        ) + 1
        last_improvement_stats.append(last_imp)
    if last_improvement_stats:
        p50 = int(np.percentile(last_improvement_stats, 50))
        p90 = int(np.percentile(last_improvement_stats, 90))
        print(
            f"\nDiagnóstico OPTUNA_N_TRIALS={OPTUNA_N_TRIALS}:"
            f"  última mejora mediana en trial #{p50},"
            f"  p90 en trial #{p90}."
        )
        if p90 >= OPTUNA_N_TRIALS - 2:
            print(
              "  ⚠  La convergencia llega hasta el final — considera aumentar OPTUNA_N_TRIALS.")
        else:
            print(
              f"  ✓  Converge bien antes del límite ({p90} < {OPTUNA_N_TRIALS}).")

    print(f"[OK] HP report guardado en: {dir_modo}")


def guardar_diag_y_plots(
    diag_rows: list[dict],
    cols_feat: list[str],
    dir_modo: Path,
    banco: str,
    fecha_hoy: str,
) -> None:
    """Guarda CSVs y heatmaps (feature × h) para gain / perm / SHAP."""
    if not diag_rows:
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        log.warning("matplotlib no disponible — omitiendo heatmaps")
        return

    df_d = pd.DataFrame(diag_rows)
    ruta_csv = dir_modo / f"diag_features_por_h_{banco}_{fecha_hoy}.csv"
    df_d.to_csv(ruta_csv, index=False)
    log.info("CSV diagnóstico: %s", ruta_csv.name)

    # Para cada señal, construir pivot feature × h (promedio de folds)
    for senal, label in [("gain", "Gain (TRAIN)"),
                          ("perm", "Block-Perm (VAL, OOS)"),
                          ("shap", "SHAP |mean| (VAL, OOS)")]:
        feat_cols = [c for c in df_d.columns if c.startswith(f"{senal}_")]
        if not feat_cols:
            continue
        rename = {c: c[len(senal) + 1:] for c in feat_cols}
        pivot = (
            df_d[["h"] + feat_cols]
            .rename(columns=rename)
            .groupby("h")
            .mean()
            .T   # features como filas, h como columnas
        )
        # Consolidar pares sin/cos → suma en fila 'base' (ej. 'dias_al_cierre_mes')
        pairs = _sincos_pairs(list(pivot.index))
        pivot = _consolidar_sincos_pivot(pivot, pairs)

        # Ordenar features por importancia media descendente
        pivot["_mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_mean", ascending=False).drop(columns=["_mean"])
        top_n = min(25, len(pivot))
        pivot = pivot.iloc[:top_n]

        # Normalizar por fila (por feature) para que el color sea comparativo
        row_max = pivot.max(axis=1).replace(0, np.nan)
        pivot_norm = pivot.div(row_max, axis=0).fillna(0.0)

        hs = pivot_norm.columns.tolist()
        fig, ax = plt.subplots(figsize=(max(10, len(hs) * 0.18), max(6, top_n * 0.42)))
        im = ax.imshow(pivot_norm.values, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Importancia norm. (0–1 por feature)")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(pivot_norm.index.tolist(), fontsize=7)
        # Mostrar etiquetas solo cada 5 horizontes para no saturar
        xtick_pos  = [i for i, h in enumerate(hs) if h % 5 == 0]
        xtick_labs = [str(hs[i]) for i in xtick_pos]
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labs, fontsize=8)
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=10)
        ax.set_title(
            f"{label} — {banco}\n"
            f"Top {top_n} features · cada fila normalizada a su propio máximo entre horizontes\n"
            f"Leer por fila: ¿en qué h importa más este feature?",
            fontweight="bold", fontsize=10,
        )
        plt.tight_layout()
        ruta_fig = dir_modo / f"diag_heatmap_{senal}_por_h_{banco}_{fecha_hoy}.png"
        plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Heatmap %s: %s", senal, ruta_fig.name)

    # Gráfico de líneas: top-10 features (por perm promedio) vs h
    perm_cols = [c for c in df_d.columns if c.startswith("perm_")]
    if perm_cols:
        rename_p = {c: c[5:] for c in perm_cols}
        pivot_perm = (
            df_d[["h"] + perm_cols].rename(columns=rename_p).groupby("h").mean().T
        )
        # Consolidar pares sin/cos antes del ranking
        pairs_p = _sincos_pairs(list(pivot_perm.index))
        pivot_perm = _consolidar_sincos_pivot(pivot_perm, pairs_p)
        pivot_perm["_mean"] = pivot_perm.mean(axis=1)
        top10 = pivot_perm.sort_values("_mean", ascending=False).head(10).drop(columns=["_mean"])
        hs = top10.columns.tolist()

        fig, ax = plt.subplots(figsize=(14, 5))
        cmap = plt.cm.tab10
        for i, feat in enumerate(top10.index):
            ax.plot(hs, top10.loc[feat].values, lw=1.8, label=feat,
                    color=cmap(i / 10), alpha=0.85)
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=10)
        ax.set_ylabel("Δ Pinball loss (perm VAL)", fontsize=10)
        ax.set_title(
            f"Block-Permutation importance por horizonte — Top 10 features — {banco}\n"
            f"Un feature útil a h corto puede ser irrelevante a h largo (y vice versa)",
            fontweight="bold", fontsize=10,
        )
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        ruta_line = dir_modo / f"diag_perm_top10_por_h_{banco}_{fecha_hoy}.png"
        plt.savefig(ruta_line, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Líneas perm top-10: %s", ruta_line.name)

    print(f"[OK] Diagnóstico features guardado en: {dir_modo}")


# ---------------------------------------------------------------------------
# Optuna HP search (Option C: representative h per group)
# ---------------------------------------------------------------------------

_HP_META_VACIO = {"best_pinball_val": float("nan"), "n_trials_ok": 0, "trial_values": []}


def optuna_tune_h(
    h_rep: int,
    df: pd.DataFrame,
    fold: dict,
    cols_feat: list[str],
    prev_best_params: dict | None = None,
) -> tuple:
    """
    Busca HP óptimos en el h representativo del grupo usando Optuna + early stopping.
    Espacio de búsqueda: 7 HP de regularización/árbol; n_estimators lo decide early stopping.

    Parameters
    ----------
    prev_best_params : dict | None
        HP óptimos del fold anterior para este mismo grupo.
        Si OPTUNA_WARM_START=True y prev_best_params no es None, se inyecta como
        trial 0 (study.enqueue_trial) para que el TPE parta de una región prometedora.
        Si es None o OPTUNA_WARM_START=False, comportamiento original (exploración libre).

    Returns
    -------
    (hp_dict, hp_meta) — siempre una tupla de dos elementos.
    """
    if not _OPTUNA_OK:
        log.warning("optuna no instalado — usando HP fijos para h_rep=%d", h_rep)
        return dict(HP), _HP_META_VACIO

    df_h = df[df["h"] == h_rep]
    try:
        X_tr, y_tr, X_vl, y_vl, _, _, _, _ = preparar_fold_data_h(df_h, fold, cols_feat)
    except ValueError as e:
        log.warning("Optuna h_rep=%d fold=%d omitido: %s — usando HP fijos", h_rep, fold["fold"], e)
        return dict(HP), _HP_META_VACIO

    y_tr_arr = np.asarray(y_tr)
    y_vl_arr = np.asarray(y_vl)

    # std_y del h representativo — requerido por el objetivo arctan
    _std_y_opt = max(float(y_tr_arr.std()), 1.0)

    def objective(trial: "optuna.Trial") -> float:
        hp_trial = {
            # ── HP buscados ────────────────────────────────────────────────
            "max_depth"       : trial.suggest_int(  "max_depth",         2,    5),
            "min_child_weight": trial.suggest_int(  "min_child_weight",   3,   20),
            "reg_alpha"       : trial.suggest_float("reg_alpha",         0.0,  2.0),
            "reg_lambda"      : trial.suggest_float("reg_lambda",        0.5,  5.0),
            "learning_rate"   : trial.suggest_float("learning_rate",    0.03, 0.15, log=True),
            "subsample"       : trial.suggest_float("subsample",         0.6,  1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",  0.5,  1.0),
            # ── HP fijos ───────────────────────────────────────────────────
            "n_estimators"    : N_ESTIMATORS_MAX,
            "tree_method"     : "hist",
            "max_bin"         : 64,
            "n_jobs"          : 1,
            "random_state"    : 42,
        }
        total_loss = 0.0

        if AJUSTE_ARCTAN:
            # Arctan: xgb.train() + _PinballEarlyStopping por tau
            # s_factor siempre buscado por Optuna en [0.01, 1.0] log
            _sf     = trial.suggest_float("s_factor", 0.01, 1.0, log=True)
            _s      = _sf * _std_y_opt
            _params = _hp_para_xgb_train(hp_trial)
            _dtrain = xgb.DMatrix(X_tr, label=y_tr_arr)
            _dval   = xgb.DMatrix(X_vl, label=y_vl_arr)
            for tau in TAUS:
                booster = xgb.train(
                    _params,
                    _dtrain,
                    num_boost_round=N_ESTIMATORS_MAX,
                    obj=_make_quantile_objective(tau, _s, _std_y_opt),
                    custom_metric=_make_pinball_metric(tau),
                    evals=[(_dval, "val")],
                    callbacks=[_PinballEarlyStopping(rounds=EARLY_STOPPING_ROUNDS)],
                    verbose_eval=False,
                )
                preds_vl = booster.predict(_dval)
                total_loss += _pinball(y_vl_arr, preds_vl, tau)
        else:
            # Estándar: XGBRegressor con reg:quantileerror
            for tau in TAUS:
                m = xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=tau,
                    eval_metric="quantile",
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    **hp_trial,
                )
                m.fit(X_tr, y_tr_arr, eval_set=[(X_vl, y_vl_arr)], verbose=False)
                total_loss += _pinball(y_vl_arr, m.predict(X_vl), tau)

        return total_loss / len(TAUS)   # media de pinball entre cuantiles

    # Seed distinta por fold para que la exploración inicial (fase aleatoria del TPE)
    # no sea idéntica entre folds — antes todos arrancaban con seed=42.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold["fold"]),
    )

    # Warm start: inyectar los HP óptimos del fold anterior como trial 0.
    # El TPE los evalúa primero y construye su modelo probabilístico desde
    # esa región, convergiendo más rápido con los mismos N trials.
    if OPTUNA_WARM_START and prev_best_params is not None:
        _search_keys = {"max_depth", "min_child_weight", "reg_alpha", "reg_lambda",
                        "learning_rate", "subsample", "colsample_bytree"}
        if AJUSTE_ARCTAN:
            _search_keys = _search_keys | {"s_factor"}   # s_factor siempre buscado
        warm_params = {k: v for k, v in prev_best_params.items() if k in _search_keys}
        if warm_params:
            study.enqueue_trial(warm_params)
            log.debug(
                "Optuna warm start fold=%d h_rep=%d — inyectando HP de fold anterior: %s",
                fold["fold"], h_rep,
                " ".join(f"{k}={v:.3g}" for k, v in warm_params.items()),
            )

    # Arctan añade s_factor al espacio → más trials para cubrir el HP extra
    n_trials_run = OPTUNA_N_TRIALS_ARCTAN if AJUSTE_ARCTAN else OPTUNA_N_TRIALS
    study.optimize(objective, n_trials=n_trials_run, show_progress_bar=False)

    best = study.best_params
    log.info(
        "Optuna fold=%d h_rep=%d → pinball_val=%.4f  %s",
        fold["fold"], h_rep, study.best_value,
        " ".join(f"{k}={v:.3g}" for k, v in best.items()),
    )

    # Meta para el reporte de HP (no se pasa al modelo)
    trial_vals = [t.value for t in study.trials if t.value is not None]
    hp_meta = {
        "best_pinball_val": study.best_value,
        "n_trials_ok"     : len(trial_vals),
        "trial_values"    : trial_vals,   # convergencia trial a trial
    }

    hp_dict = {
        **best,                        # HP encontrados por Optuna
        "n_estimators": N_ESTIMATORS_MAX,
        "tree_method" : "hist",
        "max_bin"     : 64,
        "n_jobs"      : 1,
        "random_state": 42,
    }
    return hp_dict, hp_meta


def get_hp_for_h(h_val: int, hp_grupos: dict) -> dict:
    """Devuelve el HP dict del grupo correspondiente al horizonte h_val."""
    if h_val <= 5:
        return hp_grupos["muy_corto"]
    elif h_val <= 20:
        return hp_grupos["corto"]
    elif h_val <= 50:
        return hp_grupos["medio"]
    else:
        return hp_grupos["largo"]


# ---------------------------------------------------------------------------
# Metrics plots
# ---------------------------------------------------------------------------

def graficar_metricas(
    df_res: pd.DataFrame,
    dir_modo: Path,
    banco: str,
    fecha_hoy: str,
) -> None:
    """6 paneles 2×3: RMSE | Pinball Q50+CRPS | Pinball cuantiles |
                      Coverage TEST | Coverage VAL vs TEST | Winkler + Sharpness."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        log.warning("matplotlib no disponible — omitiendo gráficos de métricas")
        return

    folds      = sorted(df_res["fold"].unique())
    cmap_folds = plt.cm.Set1
    mean_h     = df_res.groupby("h").mean(numeric_only=True)

    tau_colors = {
        "pinball_q01": "#6B21A8",
        "pinball_q05": "#9333EA",
        "pinball_q40": "#0EA5E9",
        "pinball_q50": "#2563EB",
        "pinball_q60": "#10B981",
        "pinball_q95": "#EA580C",
        "pinball_q99": "#991B1B",
    }

    fig, axes = plt.subplots(2, 3, figsize=(22, 12),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.34})
    fig.suptitle(
        f"Métricas de desempeño por horizonte — {banco}\n"
        f"cv4 DIRECT · 6 paneles: error | calibración | cobertura",
        fontsize=13, fontweight="bold", y=0.99,
    )
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    def _xgrid(ax):
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    # ── Panel 1: RMSE ────────────────────────────────────────────────────────
    for i, fn in enumerate(folds):
        sub = df_res[df_res["fold"] == fn].sort_values("h")
        ax1.plot(sub["h"], sub["rmse"] / 1e6, lw=1.5,
                 color=cmap_folds(i / max(len(folds), 1)), alpha=0.7, label=f"Fold {fn}")
    ax1.plot(mean_h.index, mean_h["rmse"] / 1e6, color="black", lw=2.5, label="Promedio")
    ax1.set_title("RMSE por horizonte h", fontweight="bold")
    ax1.set_xlabel("Horizonte h (días hábiles)")
    ax1.set_ylabel("RMSE (MM USD)")
    ax1.legend(fontsize=9)
    _xgrid(ax1)

    # ── Panel 2: Pinball Q50 + CRPS ──────────────────────────────────────────
    if "pinball_q50" in df_res.columns:
        for i, fn in enumerate(folds):
            sub = df_res[df_res["fold"] == fn].sort_values("h")
            ax2.plot(sub["h"], sub["pinball_q50"] / 1e6, lw=1.2,
                     color=cmap_folds(i / max(len(folds), 1)), alpha=0.55, label=f"Fold {fn}")
        ax2.plot(mean_h.index, mean_h["pinball_q50"] / 1e6,
                 color="#2563EB", lw=2.5, label="Pinball Q50 (prom.)")
    if "crps" in mean_h.columns:
        ax2b = ax2.twinx()
        ax2b.plot(mean_h.index, mean_h["crps"] / 1e6,
                  color="#DC2626", lw=2.0, ls="--", label="CRPS (prom.)")
        ax2b.set_ylabel("CRPS (MM USD)", color="#DC2626")
        ax2b.tick_params(axis="y", labelcolor="#DC2626")
        ax2b.legend(loc="upper right", fontsize=9)
    ax2.set_title("Pinball Q50 y CRPS por horizonte h", fontweight="bold")
    ax2.set_xlabel("Horizonte h (días hábiles)")
    ax2.set_ylabel("Pinball Q50 (MM USD)")
    ax2.legend(fontsize=9, loc="upper left")
    _xgrid(ax2)

    # ── Panel 3: Pinball todos cuantiles ─────────────────────────────────────
    pb_cols_present = [c for c in tau_colors if c in df_res.columns]
    for col in pb_cols_present:
        label = col.replace("pinball_q", "Q")
        ax3.plot(mean_h.index, mean_h[col] / 1e6,
                 color=tau_colors[col], lw=2.0, label=label)
    ax3.set_title("Pinball loss por cuantil y horizonte (promedio folds)", fontweight="bold")
    ax3.set_xlabel("Horizonte h (días hábiles)")
    ax3.set_ylabel("Pinball loss (MM USD)")
    ax3.legend(fontsize=9)
    _xgrid(ax3)

    # ── Panel 4: Coverage TEST por fold + promedio ───────────────────────────
    has_cov = False
    if "coverage_90" in df_res.columns:
        # Línea delgada por fold (misma paleta que RMSE)
        for i, fn in enumerate(folds):
            sub = df_res[df_res["fold"] == fn].sort_values("h")
            ax4.plot(sub["h"], sub["coverage_90"], lw=1.1,
                     color=cmap_folds(i / max(len(folds), 1)), alpha=0.55,
                     label=f"Fold {fn}")
        # Línea gruesa = promedio de folds
        m90 = df_res.groupby("h")["coverage_90"].mean()
        ax4.plot(m90.index, m90.values, color="#059669", lw=2.5,
                 label="Promedio folds 90%", zorder=5)
        ax4.fill_between(m90.index, m90.values, 0.90,
                         where=m90.values < 0.90, alpha=0.15, color="#DC2626")
        ax4.axhline(0.90, color="#059669", lw=1.2, ls="--", alpha=0.7,
                    label="Objetivo 90%")
        has_cov = True
    if "coverage_98" in df_res.columns:
        m98 = df_res.groupby("h")["coverage_98"].mean()
        ax4.plot(m98.index, m98.values, color="#7C3AED", lw=1.8, ls=":",
                 label="Promedio folds 98%", zorder=4)
        ax4.axhline(0.98, color="#7C3AED", lw=1.0, ls="--", alpha=0.5,
                    label="Objetivo 98%")
    if not has_cov:
        ax4.text(0.5, 0.5, "Sin datos de cobertura", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=11)
    ax4.set_title("Cobertura 90% TEST por fold y horizonte h\n(línea gruesa = promedio)",
                  fontweight="bold")
    ax4.set_xlabel("Horizonte h (días hábiles)")
    ax4.set_ylabel("Cobertura empírica")
    ax4.set_ylim(0.40, 1.05)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax4.legend(fontsize=8, ncol=2)
    _xgrid(ax4)

    # ── Panel 5: Coverage VAL vs TEST ─────────────────────────────────────────
    cov_specs = [
        ("coverage_90",     "TEST 90%", "#059669", "-"),
        ("val_coverage_90", "VAL  90%", "#059669", "--"),
        ("coverage_98",     "TEST 98%", "#7C3AED", "-"),
        ("val_coverage_98", "VAL  98%", "#7C3AED", "--"),
    ]
    plotted_5 = False
    for metric, label, color, ls in cov_specs:
        if metric in df_res.columns:
            m = df_res.groupby("h")[metric].mean()
            ax5.plot(m.index, m.values, color=color, lw=2.0, ls=ls, label=label)
            plotted_5 = True
    for nivel, color in [(0.90, "#059669"), (0.98, "#7C3AED")]:
        ax5.axhline(nivel, color=color, lw=0.8, ls=":", alpha=0.5)
    if not plotted_5:
        ax5.text(0.5, 0.5, "Sin datos VAL coverage", ha="center", va="center",
                 transform=ax5.transAxes, fontsize=11)
    ax5.set_title("Coverage VAL vs TEST\n(sólida = TEST, discontinua = VAL)",
                  fontweight="bold")
    ax5.set_xlabel("Horizonte h (días hábiles)")
    ax5.set_ylabel("Cobertura empírica")
    ax5.set_ylim(0.40, 1.05)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax5.legend(fontsize=9)
    _xgrid(ax5)

    # ── Panel 6: Winkler Score + Sharpness ───────────────────────────────────
    plotted_6 = False
    if "winkler_90" in mean_h.columns:
        ax6.plot(mean_h.index, mean_h["winkler_90"] / 1e6,
                 color="#DC2626", lw=2.0, label="Winkler 90% (cov+anchura)")
        plotted_6 = True
    if "sharpness_90" in mean_h.columns:
        ax6.plot(mean_h.index, mean_h["sharpness_90"] / 1e6,
                 color="#F59E0B", lw=2.0, ls="--", label="Sharpness 90% (solo anchura)")
        plotted_6 = True
    if not plotted_6:
        ax6.text(0.5, 0.5, "Sin datos Winkler", ha="center", va="center",
                 transform=ax6.transAxes, fontsize=11)
    ax6.set_title("Winkler Score y Sharpness del intervalo [Q05–Q95]", fontweight="bold")
    ax6.set_xlabel("Horizonte h (días hábiles)")
    ax6.set_ylabel("Puntuación (MM USD)")
    ax6.legend(fontsize=9)
    _xgrid(ax6)

    ruta_fig = dir_modo / f"metricas_por_h_{banco}_{fecha_hoy}.png"
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfico de métricas: %s", ruta_fig.name)
    print(f"[OK] Gráfico métricas guardado: {ruta_fig.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(banco: str = BANCO) -> None:
    t0_total = time.time()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nCargando datos: {RUTA_MATRIZ}")
    try:
        import pyarrow.parquet as pq
        df = pq.read_table(
            RUTA_MATRIZ,
            filters=[("banco", "==", banco)],
            memory_map=True,   # lee desde disco, reduce presión en RAM
            pre_buffer=False,
        ).to_pandas()
    except MemoryError:
        # Fallback: leer solo columnas esenciales + filtrar con pandas
        print("  [AVISO] MemoryError — leyendo por columnas y filtrando con pandas")
        import pyarrow.parquet as pq
        schema = pq.read_schema(RUTA_MATRIZ)
        all_cols = [f.name for f in schema]
        df = pq.read_table(
            RUTA_MATRIZ,
            columns=all_cols,
            memory_map=True,
        ).to_pandas()
        df = df[df["banco"] == banco].copy()

    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    if "fecha_th" in df.columns:
        df["fecha_th"] = pd.to_datetime(df["fecha_th"])
        if df["fecha_th"].dt.tz is not None:
            df["fecha_th"] = df["fecha_th"].dt.tz_convert(None)

    print(f"  Filas cargadas : {len(df):,}")
    print(f"  Rango fecha_t  : {df['fecha_t'].min().date()} → {df['fecha_t'].max().date()}")
    print(f"  Horizontes (h) : {df['h'].min()} – {df['h'].max()}")

    # ------------------------------------------------------------------
    # 2. Feature columns (h and log_h excluded)
    # ------------------------------------------------------------------
    cols_feat = get_feature_cols(df)
    print(f"\nFeatures: {len(cols_feat)} columnas (h y log_h excluidos)")
    log.debug("cols_feat = %s", cols_feat)

    # ------------------------------------------------------------------
    # 3. Build folds
    # ------------------------------------------------------------------
    folds = build_folds(df)
    print(f"Folds: {len(folds)}")
    for f in folds:
        print(
            f"  Fold {f['fold']:2d} | "
            f"train {f['train_start'].date()}..{f['train_end'].date()} | "
            f"val {f['val_start'].date()}..{f['val_end'].date()} | "
            f"test {f['test_start'].date()}..{f['test_end'].date()}"
        )

    # ------------------------------------------------------------------
    # 4. Walk-forward CV loop
    # ------------------------------------------------------------------
    resultados:    list[dict] = []
    hp_report_rows: list[dict] = []   # acumula HPs por fold × grupo para el reporte
    diag_rows:  list[dict] = []
    n_horizontes = H_MAX - H_MIN + 1

    # Warm start: guarda los HP óptimos del fold anterior por grupo.
    # Se pasan a optuna_tune_h() y se actualizan al final de cada fold.
    prev_hp_por_grupo: dict[str, dict | None] = {g: None for g in H_GRUPOS}

    # Output dir created early so per-fold parquets can be written immediately
    # El sufijo _arctan separa los resultados del objetivo suavizado para comparación
    _arctan_sfx = "_arctan" if AJUSTE_ARCTAN else ""
    _modo_base  = "exp" if EXPANDING else "roll"
    DIR_MODO = DIR_OUTPUT / f"fold_{_modo_base}{_arctan_sfx}"
    DIR_MODO.mkdir(parents=True, exist_ok=True)
    fecha_hoy = date.today().strftime("%Y%m%d")

    fold_parquet_paths: list[Path] = []

    folds_run = folds[:1] if DEBUG_SINGLE_FOLD else folds
    if DEBUG_SINGLE_FOLD:
        print(f"\n[DEBUG_SINGLE_FOLD=True] Corriendo solo fold {folds_run[0]['fold']} de {len(folds)}")

    for fold in folds_run:
        t0_fold = time.time()
        print(f"\n{'='*60}")
        print(
            f"FOLD {fold['fold']}  "
            f"train: {fold['train_start'].date()}..{fold['train_end'].date()}  "
            f"test:  {fold['test_start'].date()}..{fold['test_end'].date()}"
        )

        # ── Optuna: buscar HP por grupo de h (Opción C) ──────────────────────
        if USE_OPTUNA and _OPTUNA_OK:
            ws_tag = " · warm_start=ON" if OPTUNA_WARM_START else " · warm_start=OFF"
            print(f"  Buscando HP con Optuna ({OPTUNA_N_TRIALS} trials × 4 grupos{ws_tag})…")
            hp_grupos: dict = {}
            for grupo, (_, h_rep) in H_GRUPOS.items():
                t_opt = time.time()
                hp_grupos[grupo], hp_meta = optuna_tune_h(
                    h_rep, df, fold, cols_feat,
                    prev_best_params=prev_hp_por_grupo[grupo],
                )
                # Guardar para el próximo fold (solo si warm start activo)
                if OPTUNA_WARM_START:
                    prev_hp_por_grupo[grupo] = hp_grupos[grupo]
                elapsed_opt = time.time() - t_opt
                hp_g = hp_grupos[grupo]
                print(
                    f"    [{grupo:9s}] h_rep={h_rep:2d}  "
                    f"depth={hp_g.get('max_depth')}  "
                    f"min_cw={hp_g.get('min_child_weight')}  "
                    f"lr={hp_g.get('learning_rate', 0):.3f}  "
                    f"α={hp_g.get('reg_alpha', 0):.2f}  "
                    f"λ={hp_g.get('reg_lambda', 1):.2f}  "
                    f"sub={hp_g.get('subsample', 1):.2f}  "
                    f"col={hp_g.get('colsample_bytree', 1):.2f}  "
                    f"pinball={hp_meta['best_pinball_val']:.4f}  "
                    f"({elapsed_opt:.0f}s)"
                )
                hp_report_rows.append({
                    "fold"            : fold["fold"],
                    "grupo"           : grupo,
                    "h_rep"           : h_rep,
                    "best_pinball_val": hp_meta["best_pinball_val"],
                    "n_trials_ok"     : hp_meta["n_trials_ok"],
                    "trial_values"    : hp_meta["trial_values"],
                    "max_depth"       : hp_g.get("max_depth"),
                    "min_child_weight": hp_g.get("min_child_weight"),
                    "learning_rate"   : hp_g.get("learning_rate"),
                    "reg_alpha"       : hp_g.get("reg_alpha"),
                    "reg_lambda"      : hp_g.get("reg_lambda"),
                    "subsample"       : hp_g.get("subsample"),
                    "colsample_bytree": hp_g.get("colsample_bytree"),
                })
        else:
            if USE_OPTUNA and not _OPTUNA_OK:
                log.warning("USE_OPTUNA=True pero optuna no está instalado — usando HP fijos")
            hp_grupos = {g: dict(HP) for g in H_GRUPOS}

        # Write each h directly to a list; concat and flush to disk per fold
        fold_scaffolds:     list[pd.DataFrame] = []
        fold_val_scaffolds: list[pd.DataFrame] = []   # para CQR calibración
        n_h_ok = 0

        for h_val in range(H_MIN, H_MAX + 1):
            df_h = df[df["h"] == h_val]

            if h_val % 10 == 0:
                elapsed_fold = (time.time() - t0_fold) / 60
                print(
                    f"  h={h_val:3d} | "
                    f"ok={n_h_ok}/{h_val - H_MIN} | "
                    f"{elapsed_fold:.1f} min transcurridos"
                )

            try:
                (X_train, y_train, X_val, y_val,
                 X_test,  y_test,
                 fechas_t_test, fecha_th_test) = preparar_fold_data_h(
                    df_h, fold, cols_feat
                )
            except ValueError as e:
                log.debug("Fold %d, h=%d omitido: %s", fold["fold"], h_val, e)
                continue

            log.debug(
                "Fold %d, h=%d | train=%d val=%d test=%d",
                fold["fold"], h_val,
                len(X_train), len(X_val), len(X_test),
            )

            hp_h  = get_hp_for_h(h_val, hp_grupos)
            # std_y por h: escala natural del target en la ventana de entrenamiento
            # Requerido por _make_quantile_objective cuando AJUSTE_ARCTAN = True
            std_y = max(float(y_train.std()), 1.0) if AJUSTE_ARCTAN else 0.0
            modelos = entrenar_modelos_h(X_train, y_train, X_val, y_val, hp_h,
                                         std_y=std_y)

            _scaffold = pd.DataFrame({
                "banco"   : banco,
                "fold"    : fold["fold"],
                "fecha_t" : pd.DatetimeIndex(fechas_t_test),
                "fecha_th": pd.DatetimeIndex(fecha_th_test),
                "h"       : h_val,
                "target"  : y_test.values,
            })

            for tau, model in modelos.items():
                col = "mean" if tau == "mean" else f"q{int(tau * 100):02d}"
                _scaffold[col] = model.predict(X_test)

            fold_scaffolds.append(_scaffold)

            # ── VAL scaffold (para CQR en step006) ───────────────────────────
            if len(X_val) > 0:
                mv_mask = (
                    (df_h["fecha_t"] >= fold["val_start"]) &
                    (df_h["fecha_t"] <= fold["val_end"])   &
                    df_h["target"].notna()
                )
                fechas_t_val = _strip_tz(df_h.loc[mv_mask, "fecha_t"])
                _val_scaffold = pd.DataFrame({
                    "banco"  : banco,
                    "fold"   : fold["fold"],
                    "fecha_t": pd.DatetimeIndex(fechas_t_val),
                    "h"      : h_val,
                    "target" : y_val.values,
                })
                for tau, model in modelos.items():
                    col = "mean" if tau == "mean" else f"q{int(tau * 100):02d}"
                    _val_scaffold[col] = model.predict(X_val)
                fold_val_scaffolds.append(_val_scaffold)

            n_h_ok += 1

            preds_for_metrics = {tau: m.predict(X_test) for tau, m in modelos.items()}
            preds_val_for_metrics = (
                {tau: m.predict(X_val) for tau, m in modelos.items()}
                if len(X_val) > 0 else None
            )
            resultados.append(calcular_metricas(
                preds_for_metrics, y_test.values, h_val, fold["fold"],
                preds_val_for_metrics, y_val.values if len(X_val) > 0 else None,
            ))

            if DIAG_FEATURES:
                diag_rows.append(
                    diagnosticar_h(modelos, X_val, y_val, cols_feat,
                                   fold["fold"], h_val)
                )

            del modelos, X_train, y_train, X_val, y_val, X_test, y_test
            gc.collect()

        # Flush fold to disk immediately — don't accumulate all folds in RAM
        if fold_scaffolds:
            df_fold = pd.concat(fold_scaffolds, ignore_index=True)
            ruta_fold = DIR_MODO / f"preds_test_fold{fold['fold']:02d}_{banco}_{fecha_hoy}.parquet"
            df_fold.to_parquet(ruta_fold, index=False)
            fold_parquet_paths.append(ruta_fold)
            print(f"  → Guardado TEST: {ruta_fold.name}")
            del df_fold, fold_scaffolds
            gc.collect()

        if fold_val_scaffolds:
            df_val_fold = pd.concat(fold_val_scaffolds, ignore_index=True)
            ruta_val = DIR_MODO / f"preds_val_fold{fold['fold']:02d}_{banco}_{fecha_hoy}.parquet"
            df_val_fold.to_parquet(ruta_val, index=False)
            print(f"  → Guardado VAL:  {ruta_val.name}")
            del df_val_fold, fold_val_scaffolds
            gc.collect()

        elapsed_fold = (time.time() - t0_fold) / 60
        print(
            f"  Fold {fold['fold']}: "
            f"{n_h_ok}/{n_horizontes} horizontes completados | "
            f"{elapsed_fold:.1f} min"
        )

    # ------------------------------------------------------------------
    # 5. Consolidate per-fold parquets into preds_base (read one by one)
    # ------------------------------------------------------------------
    col_order = ["banco", "fold", "fecha_t", "fecha_th", "h", "target",
                 "q01", "q05", "q40", "q50", "q60", "q95", "q99", "mean"]

    if fold_parquet_paths:
        chunks = []
        for p in fold_parquet_paths:
            chunk = pd.read_parquet(p)
            extra = [c for c in chunk.columns if c not in col_order]
            ordered = [c for c in col_order if c in chunk.columns] + extra
            chunks.append(chunk[ordered])
        df_all = pd.concat(chunks, ignore_index=True)

        ruta_preds = DIR_MODO / f"preds_base_{banco}_{fecha_hoy}.parquet"
        df_all.to_parquet(ruta_preds, index=False)
        print(f"\n✓ Guardado: {ruta_preds}  ({len(df_all):,} filas)")
        print(f"  Columnas: {list(df_all.columns)}")
        del df_all, chunks
        gc.collect()
    else:
        print("\n⚠  Sin predicciones para guardar.")

    # ------------------------------------------------------------------
    # 6. Save metrics
    # ------------------------------------------------------------------
    if resultados:
        df_res = pd.DataFrame(resultados)
        ruta_met = DIR_MODO / f"metricas_{banco}_{fecha_hoy}.parquet"
        df_res.to_parquet(ruta_met, index=False)
        print(f"✓ Métricas: {ruta_met}  ({len(df_res):,} filas)")

        # Summary by horizon group
        bins   = [1, 5, 15, 30, 50, 75]
        labels = ["h02-05", "h06-15", "h16-30", "h31-50", "h51-75"]
        df_res["h_grupo"] = pd.cut(df_res["h"], bins=bins, labels=labels)

        print("\nRMSE medio por grupo de horizonte:")
        print(df_res.groupby("h_grupo", observed=True)["rmse"].mean().round(0))

        # Also print pinball for q50
        if "pinball_q50" in df_res.columns:
            print("\nPinball q50 medio por grupo de horizonte:")
            print(df_res.groupby("h_grupo", observed=True)["pinball_q50"].mean().round(4))

        if "coverage_90" in df_res.columns:
            print("\nCobertura empírica 90% [Q05-Q95] TEST media por grupo de horizonte:")
            print(df_res.groupby("h_grupo", observed=True)["coverage_90"].mean().map("{:.1%}".format))

            # Tabla por fold × grupo (coverage_90)
            print("\nCobertura 90% TEST por fold y grupo de horizonte:")
            tbl_90 = (
                df_res.groupby(["fold", "h_grupo"], observed=True)["coverage_90"]
                .mean()
                .unstack("h_grupo")
                .applymap("{:.1%}".format)
            )
            tbl_90.index = ["Fold {}".format(f) for f in tbl_90.index]
            print(tbl_90.to_string())

        if "coverage_98" in df_res.columns:
            print("\nCobertura empírica 98% [Q01-Q99] TEST media por grupo de horizonte:")
            print(df_res.groupby("h_grupo", observed=True)["coverage_98"].mean().map("{:.1%}".format))

            # Tabla por fold × grupo (coverage_98)
            print("\nCobertura 98% TEST por fold y grupo de horizonte:")
            tbl_98 = (
                df_res.groupby(["fold", "h_grupo"], observed=True)["coverage_98"]
                .mean()
                .unstack("h_grupo")
                .applymap("{:.1%}".format)
            )
            tbl_98.index = ["Fold {}".format(f) for f in tbl_98.index]
            print(tbl_98.to_string())

        if "val_coverage_90" in df_res.columns:
            print("\nCobertura empírica 90% [Q05-Q95] VAL media por grupo de horizonte:")
            print(df_res.groupby("h_grupo", observed=True)["val_coverage_90"].mean().map("{:.1%}".format))

        if "val_coverage_98" in df_res.columns:
            print("\nCobertura empírica 98% [Q01-Q99] VAL media por grupo de horizonte:")
            print(df_res.groupby("h_grupo", observed=True)["val_coverage_98"].mean().map("{:.1%}".format))

        if "winkler_90" in df_res.columns:
            print("\nWinkler Score 90% medio por grupo de horizonte (MM USD):")
            print((df_res.groupby("h_grupo", observed=True)["winkler_90"].mean() / 1e6).round(3))

        if "crps" in df_res.columns:
            print("\nCRPS medio por grupo de horizonte (MM USD):")
            print((df_res.groupby("h_grupo", observed=True)["crps"].mean() / 1e6).round(4))

        calib_cols = sorted([c for c in df_res.columns if c.startswith("calib_q")])
        if calib_cols:
            print("\nCalibración hit-rate por cuantil (promedio global; debe ≈ tau nominal):")
            calib_mean = df_res[calib_cols].mean().round(3)
            tau_labels = {f"calib_q{int(t*100):02d}": f"Q{int(t*100):02d} (τ={t:.2f})" for t in TAUS}
            calib_mean.index = [tau_labels.get(c, c) for c in calib_mean.index]
            print(calib_mean.to_string())

        graficar_metricas(df_res, DIR_MODO, banco, fecha_hoy)
    else:
        print("\n⚠  Sin métricas para guardar.")

    # ------------------------------------------------------------------
    # 7. HP report (convergencia Optuna + estabilidad por fold)
    # ------------------------------------------------------------------
    if hp_report_rows:
        guardar_hp_report(hp_report_rows, DIR_MODO, banco, fecha_hoy)

    # ------------------------------------------------------------------
    # 8. Feature diagnostics (gain / perm / SHAP per h)
    # ------------------------------------------------------------------
    if DIAG_FEATURES and diag_rows:
        guardar_diag_y_plots(diag_rows, cols_feat, DIR_MODO, banco, fecha_hoy)

    total_min = (time.time() - t0_total) / 60
    print(f"\n{'='*60}")
    print(f"Tiempo total: {total_min:.1f} min")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
