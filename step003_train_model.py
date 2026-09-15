# -*- coding: utf-8 -*-
from __future__ import annotations  # permite dict | None y list[str] en Python < 3.10
"""
step003_train_model.py
Entrenamiento de modelos LightGBM de quantile regression para predicción de
flujos netos D−R (liquidez en ME) del sistema bancario peruano.

Diseño:
  - Un modelo por banco, cubre todos los horizontes h en una sola pasada.
  - h y log_h son features explícitos → el modelo aprende la forma de la curva.
  - Quantiles producidos: [0.01, 0.05, 0.50, 0.95, 0.99].
  - Validación temporal walk-forward (expanding window) — NO hay split aleatorio.
  - Optimización Bayesiana (Optuna, TPE) sobre el cuantil mediano (τ=0.50).
  - Corrección de cruce de cuantiles: np.sort por fila sobre predicciones finales.
  - Los modelos entrenados se guardan en 2. Output/modelos/ como .txt + metadata .json.

Flujo:
  1. Leer matriz_features.parquet (banco por banco, column-subset).
  2. Walk-forward split: cortes por fecha fija (CORTE_VAL, CORTE_TEST).
  3. Optuna: optimizar pinball loss mediana (τ=0.50) en el split.
  4. Re-entrenar cada quantil con los mejores hiperparámetros sobre train completo.
  5. Evaluar en validación: pinball loss por quantil y RMSE mediana.
  6. Guardar modelos + metadata JSON + gráfico de feature importance.
"""

import gc
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


###############################################################################
# PARTE 0 — Parámetros globales
###############################################################################
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_MODELOS      = BASE_SISTEMA / "2. Output" / "modelos"
DIR_MODELOS_EVAL = DIR_MODELOS / "eval"   # modelos TRAIN-only → evaluación honesta OOS
DIR_PLOTS        = BASE_SISTEMA / "2. Output" / "plots_entrenamiento"

DIR_MODELOS.mkdir(parents=True, exist_ok=True)
DIR_MODELOS_EVAL.mkdir(parents=True, exist_ok=True)
DIR_PLOTS.mkdir(parents=True, exist_ok=True)

# Quantiles a producir
QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]

# Cortes temporales del split (walk-forward, orden cronológico estricto):
#   |─── TRAIN ──────────────|─── VAL ───|─── TEST ──────────────|
#   TRAIN : 2010 → jul 2022  (incluye ciclo electoral 2021)
#   VAL   : jul 2022 → ene 2023  (~6 meses, Optuna)
#   TEST  : 03 ene 2023 → hoy   (alineado con datos de tasas del allocation)
#
# Fechas fijas en lugar de semanas proporcionales porque el corte TEST tiene
# justificación externa: el modelo de allocation tiene tasas desde 03-ene-2023.
CORTE_VAL  = pd.Timestamp("2022-07-01")
CORTE_TEST = pd.Timestamp("2023-01-03")

# Trials Optuna por banco
N_TRIALS_OPTUNA = 60

# Filtro de bancos a entrenar.
# None  → entrena todos los bancos presentes en la matriz.
# Lista → entrena solo los bancos especificados.
# Ejemplo para validar solo el agregado: BANCOS_A_ENTRENAR = ["SISTEMA"]
BANCOS_A_ENTRENAR = ["SISTEMA"]   # cambiar a None para entrenar todos

# Features a excluir del entrenamiento (identificadores, no predictores)
COLS_EXCLUIR = {"fecha_t", "banco", "target"}

# True  → re-estima GARCH(1,1) solo con datos TRAIN (hasta CORTE_VAL) antes de entrenar
# False → usa GARCH del parquet tal cual (comportamiento original)
USAR_GARCH_SIN_LEAKAGE = False

# Columnas de texto (si las hubiera) — no deberían existir, pero por seguridad
COLS_TEXTO   = {"banco"}


###############################################################################
# PARTE 1 — Utilidades
###############################################################################

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss promedio."""
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def corregir_cruce_cuantiles(preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """
    Garantiza monotonicidad entre cuantiles: Q01 ≤ Q05 ≤ Q50 ≤ Q95 ≤ Q99.
    Ordena las predicciones por fila para los cuantiles solicitados.
    """
    taus    = sorted(preds.keys())
    matrix  = np.column_stack([preds[t] for t in taus])
    matrix  = np.sort(matrix, axis=1)
    return {t: matrix[:, i] for i, t in enumerate(taus)}


def leer_banco_parquet(ruta: Path, banco: str) -> pd.DataFrame:
    """
    Lee todas las filas de un banco desde el parquet usando filtro de pyarrow.
    Carga solo las columnas necesarias para entrenamiento (excluye fecha_t y banco
    después de usarlos como índice temporal).
    """
    df = pd.read_parquet(
        ruta,
        filters=[("banco", "==", banco)],
    )
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)
    return df


def preparar_Xy(df: pd.DataFrame, cols_feat: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Retorna (X, y) descartando filas sin target."""
    mask = df["target"].notna()
    X = df.loc[mask, cols_feat].copy()
    y = df.loc[mask, "target"].copy()
    return X, y


def split_walk_forward(
    df: pd.DataFrame,
    corte_val:  pd.Timestamp = CORTE_VAL,
    corte_test: pd.Timestamp = CORTE_TEST,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide temporalmente en tres particiones sin solapamiento:
      TRAIN | VAL | TEST  (orden cronológico estricto)

    Usa fechas fijas (CORTE_VAL, CORTE_TEST) para alinear TEST con el período
    donde el modelo de allocation tiene datos de tasas (desde 03-ene-2023).

    Garantías:
      · Ningún dato del futuro contamina el entrenamiento ni la optimización.
      · VAL y TEST son completamente disjuntos.
      · TRAIN incluye el ciclo electoral 2021 completo.
    """
    df_train = df[df["fecha_t"] <  corte_val].copy()
    df_val   = df[(df["fecha_t"] >= corte_val) & (df["fecha_t"] < corte_test)].copy()
    df_test  = df[df["fecha_t"] >= corte_test].copy()

    n_train = df_train["fecha_t"].nunique()
    n_val   = df_val["fecha_t"].nunique()
    n_test  = df_test["fecha_t"].nunique()
    n_total = n_train + n_val + n_test

    if n_train < n_total * 0.5:
        raise ValueError(
            f"TRAIN solo tiene {n_train} fechas ({100*n_train/n_total:.0f}% del total). "
            "Ajusta CORTE_VAL o CORTE_TEST."
        )
    return df_train, df_val, df_test


###############################################################################
# PARTE 2 — Construcción de features
###############################################################################

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Infiere las columnas de features a partir del DataFrame.
    Excluye identificadores y target; incluye h, log_h y todos los predictores.
    """
    excluir = COLS_EXCLUIR | {"fecha_th"}   # fecha_th no debería estar, por si acaso
    cols = [c for c in df.columns if c not in excluir]

    # Separar features numéricas (LightGBM nativo maneja categorías, pero las tenemos como int)
    cols_validas = []
    for c in cols:
        if df[c].dtype.kind in ("f", "i", "u", "b"):   # float, int, uint, bool
            cols_validas.append(c)
        else:
            logger.debug(f"  Columna ignorada (no numérica): {c}")
    return cols_validas


###############################################################################
# GARCH sin leakage  (estimación solo sobre TRAIN)
###############################################################################

def _ajustar_garch_s4(x_train):
    from scipy.optimize import minimize as _min
    n = len(x_train)
    var_unc = max(float(np.var(x_train)), 1e-12)
    def _s2(o, a, b):
        s2 = np.empty(n); s2[0] = var_unc
        for t in range(1, n): s2[t] = o + a * x_train[t-1]**2 + b * s2[t-1]
        return s2
    def _nll(p):
        o, a, b = p
        if o <= 0 or a <= 0 or b <= 0 or a + b >= 0.9999: return 1e10
        s2 = _s2(o, a, b)
        return 1e10 if np.any(s2 <= 0) else 0.5*float(np.sum(np.log(s2) + x_train**2/s2))
    try:
        r = _min(_nll, [0.01, 0.08, 0.88], method="L-BFGS-B",
                 bounds=[(1e-7,0.5),(1e-7,0.5),(1e-7,0.9999)],
                 options={"maxiter":500,"ftol":1e-10,"gtol":1e-7})
        if r.fun < 1e9: return float(r.x[0]), float(r.x[1]), float(r.x[2])
    except Exception: pass
    return 0.01, 0.08, 0.88

def _garch_vol_s4(serie, train_end):
    sf = serie.ffill().fillna(0.0)
    st = sf[sf.index <= train_end]
    if len(st) < 60 or st.std() < 1e-9:
        return sf.rolling(20).std().fillna(st.std())
    esc = float(st.std()); x_tr = (st / esc).values.astype(float)
    var_unc = max(float(np.var(x_tr)), 1e-12)
    o, a, b = _ajustar_garch_s4(x_tr)
    xf = (sf / esc).values.astype(float); nf = len(xf)
    s2 = np.empty(nf); s2[0] = var_unc
    for t in range(1, nf): s2[t] = o + a * xf[t-1]**2 + b * s2[t-1]
    return pd.Series(np.sqrt(np.maximum(s2, 0)) * esc, index=sf.index)

def _ffd_weights_s3(d, thresh=1e-5):
    w, k = [1.0], 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])

def _fracdiff_s3(series, d, thresh=1e-5):
    w = _ffd_weights_s3(d, thresh); width = len(w)
    vals = series.values.astype(float); out = np.full(len(vals), np.nan)
    for i in range(width - 1, len(vals)):
        chunk = vals[i - width + 1: i + 1]
        if not np.any(np.isnan(chunk)):
            out[i] = float(np.dot(w, chunk))
    return pd.Series(out, index=series.index)

def reemplazar_ffd_sin_leakage(df, train_end):
    """Re-calibra FFD solo con datos ≤ train_end y propaga hacia adelante."""
    frac_cols = [c for c in df.columns if c.endswith("_frac")]
    if not frac_cols:
        return df
    df = df.copy()
    try:
        from statsmodels.tsa.stattools import adfuller
        _sm_ok = True
    except ImportError:
        _sm_ok = False

    def _find_d(serie):
        if not _sm_ok or len(serie.dropna()) < 30:
            return 0.4
        for d in np.linspace(0.05, 1.0, 20):
            fd = _fracdiff_s3(serie.dropna(), round(float(d), 4)).dropna()
            if len(fd) < 20:
                continue
            try:
                if adfuller(fd, maxlag=1, regression="c", autolag=None)[1] <= 0.05:
                    return round(float(d), 4)
            except Exception:
                continue
        return 1.0

    raw = df[["fecha_t"]].drop_duplicates("fecha_t").set_index("fecha_t").sort_index()
    for col_frac in frac_cols:
        col_raw = col_frac.replace("_frac", "")
        if col_raw not in df.columns:
            continue
        raw_serie = (df[["fecha_t", col_raw]].drop_duplicates("fecha_t")
                     .set_index("fecha_t").sort_index()[col_raw])
        serie_train = raw_serie[raw_serie.index <= train_end].dropna()
        if len(serie_train) < 60:
            continue
        d_opt = _find_d(serie_train)
        frac_full = _fracdiff_s3(raw_serie, d_opt)
        df[col_frac] = df["fecha_t"].map(frac_full)
        logger.info(f"    {col_frac}  re-calibrado sin leakage  d_opt={d_opt:.4f}")
    return df


def reemplazar_garch_sin_leakage(df, train_end):
    """Re-estima GARCH(1,1) solo con datos ≤ train_end y propaga hacia adelante."""
    df = df.copy()
    avail = [c for c in ["fecha_t","R_t0","D_t0","TC_PEN_USD","EMBI_PERU"] if c in df.columns]
    raw   = df[avail].drop_duplicates("fecha_t").set_index("fecha_t").sort_index()
    if "garch_vol" in df.columns and {"R_t0","D_t0"}.issubset(raw.columns):
        df["garch_vol"] = df["fecha_t"].map(_garch_vol_s4(raw["D_t0"]-raw["R_t0"], train_end))
        logger.info("    garch_vol        re-estimado sin leakage")
    if "garch_vol_tc" in df.columns and "TC_PEN_USD" in raw.columns:
        tc = raw["TC_PEN_USD"].replace(0, np.nan).ffill()
        tci = tc.reindex(pd.bdate_range(tc.index.min(), tc.index.max())).ffill()
        ret = np.log(tci / tci.shift(1)).reindex(tc.index)
        df["garch_vol_tc"] = df["fecha_t"].map(_garch_vol_s4(ret, train_end))
        logger.info("    garch_vol_tc     re-estimado sin leakage")
    if "garch_vol_embi" in df.columns and "EMBI_PERU" in raw.columns:
        df["garch_vol_embi"] = df["fecha_t"].map(
            _garch_vol_s4(raw["EMBI_PERU"].diff(1), train_end))
        logger.info("    garch_vol_embi   re-estimado sin leakage")
    return df


###############################################################################
# PARTE 3 — Optimización Bayesiana con Optuna
###############################################################################

def objective_lgbm(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tau: float = 0.50,
) -> float:
    """
    Función objetivo para Optuna: pinball loss mediana en validación.
    Se optimiza sobre τ=0.50 para encontrar los mejores hiperparámetros
    de estructura del árbol; luego esos mismos hiperparámetros se usan
    para todos los cuantiles.
    """
    params = {
        "objective":        "quantile",
        "alpha":            tau,
        "metric":           "quantile",
        "verbosity":        -1,
        "n_jobs":           -1,
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 255),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    preds = model.predict(X_val)
    return pinball_loss(y_val.values, preds, tau)


def optimizar_hiperparametros(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int,
    banco: str,
) -> dict:
    """Ejecuta Optuna y retorna los mejores hiperparámetros."""
    logger.info(f"  [{banco}] Optimización Bayesiana ({n_trials} trials, τ=0.50)...")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective_lgbm(trial, X_train, y_train, X_val, y_val, tau=0.50),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info(
        f"  [{banco}] Mejor pinball(τ=0.50): {study.best_value:.4f} "
        f"| n_est={best['n_estimators']} lr={best['learning_rate']:.4f} "
        f"leaves={best['num_leaves']}"
    )
    return best


###############################################################################
# PARTE 4 — Entrenamiento por quantil
###############################################################################

def entrenar_quantiles(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    quantiles: list[float],
    banco: str,
) -> dict[float, lgb.LGBMRegressor]:
    """
    Re-entrena un modelo LightGBM por cada cuantil usando los mejores
    hiperparámetros encontrados por Optuna. Entrenamiento sobre conjunto
    completo (train + val) para maximizar datos en producción.
    """
    modelos = {}
    for tau in quantiles:
        logger.info(f"  [{banco}] Entrenando τ={tau:.2f}...")
        params = {
            "objective":         "quantile",
            "alpha":             tau,
            "metric":            "quantile",
            "verbosity":         -1,
            "n_jobs":            -1,
            **best_params,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])
        modelos[tau] = model
    return modelos


###############################################################################
# PARTE 5 — Evaluación
###############################################################################

def evaluar_modelos(
    modelos: dict[float, lgb.LGBMRegressor],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    banco: str,
    split_name: str = "val",
) -> tuple[dict, dict]:
    """
    Calcula pinball loss por quantil y RMSE de la mediana en el split indicado.
    Aplica corrección de cruce de cuantiles antes de evaluar.
    split_name: "val" o "test" — solo afecta los mensajes del log.
    """
    preds_raw = {tau: model.predict(X_eval) for tau, model in modelos.items()}
    preds     = corregir_cruce_cuantiles(preds_raw)

    metricas = {}
    for tau in sorted(modelos.keys()):
        pb = pinball_loss(y_eval.values, preds[tau], tau)
        metricas[f"pinball_{split_name}_q{int(tau*100):02d}"] = round(pb, 4)
        logger.info(f"  [{banco}] pinball(τ={tau:.2f}) [{split_name}] = {pb:,.2f}")

    if 0.50 in modelos:
        rmse = float(np.sqrt(np.mean((y_eval.values - preds[0.50]) ** 2)))
        metricas[f"rmse_{split_name}_mediana"] = round(rmse, 2)
        logger.info(f"  [{banco}] RMSE mediana [{split_name}] = {rmse:,.2f}")

    return metricas, preds


###############################################################################
# PARTE 6 — Persistencia
###############################################################################

def guardar_modelos(
    modelos: dict[float, lgb.LGBMRegressor],
    metricas: dict,
    best_params: dict,
    cols_feat: list[str],
    banco: str,
    dir_modelos: Path,
):
    """
    Guarda cada modelo como .txt (formato texto LightGBM) y un JSON con
    metadata: features, hiperparámetros, métricas de validación, fecha.
    """
    fecha_hoy = pd.Timestamp.today().strftime("%Y%m%d")

    for tau, model in modelos.items():
        nombre_base = f"lgbm_{banco}_q{int(tau*100):02d}_{fecha_hoy}"
        ruta_model  = dir_modelos / f"{nombre_base}.txt"
        model.booster_.save_model(str(ruta_model))

    metadata = {
        "banco":            banco,
        "fecha_entrenamiento": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "quantiles":        [float(t) for t in sorted(modelos.keys())],
        "n_features":       len(cols_feat),
        "features":         cols_feat,
        "best_params":      best_params,
        "metricas_val":     metricas,
    }
    ruta_meta = dir_modelos / f"metadata_{banco}_{fecha_hoy}.json"
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"  [{banco}] Modelos guardados en {dir_modelos}")


###############################################################################
# PARTE 7 — Gráficos de feature importance y fan chart de validación
###############################################################################

def graficar_importancia(
    modelos: dict[float, lgb.LGBMRegressor],
    cols_feat: list[str],
    banco: str,
    dir_plots: Path,
    top_n: int = 20,
):
    """
    Bar chart horizontal: importancia promedio entre quantiles (gain).
    Solo muestra top_n features.
    """
    importancias = np.zeros(len(cols_feat))
    for model in modelos.values():
        imp = model.booster_.feature_importance(importance_type="gain")
        importancias += imp / len(modelos)

    idx_top = np.argsort(importancias)[-top_n:]
    nombres = [cols_feat[i] for i in idx_top]
    valores = importancias[idx_top]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(nombres, valores, color="steelblue", alpha=0.85)
    ax.set_xlabel("Importancia promedio (gain)", fontsize=10)
    ax.set_title(
        f"Feature Importance — {banco}\n(promedio entre quantiles, top {top_n})",
        fontweight="bold", fontsize=11,
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    nombre = f"feature_importance_{banco}.png"
    plt.savefig(dir_plots / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [{banco}] Gráfico importancia guardado: {dir_plots / nombre}")


def graficar_fanchart_split(
    df_split: pd.DataFrame,
    preds: dict[float, np.ndarray],
    y_split: pd.Series,
    banco: str,
    dir_plots: Path,
    split_name: str = "test",
    h_ejemplo: int = 10,
    ventana_dias: int = 90,   # días hábiles por panel
):
    """
    Para un horizonte h fijo, grafica la banda de predicción (Q01–Q99, Q05–Q95)
    vs el valor realizado. El período se divide en subplots de ventana_dias días
    hábiles para facilitar la lectura.
    """
    mask_h = df_split["h"] == h_ejemplo
    if mask_h.sum() < 5:
        conteos = df_split.groupby("h").size()
        candidatos = conteos[conteos >= 5]
        if candidatos.empty:
            return
        h_ejemplo = int(candidatos.index[np.argmin(np.abs(candidatos.index - h_ejemplo))])
        mask_h = df_split["h"] == h_ejemplo

    idx_split = np.where(mask_h.values)[0]
    fechas    = pd.to_datetime(df_split.loc[mask_h, "fecha_t"].values)
    y_real    = y_split.values[idx_split] / 1e6

    label_titulo = "Test (holdout)" if split_name == "test" else "Validación"

    # ── Dividir en ventanas de ventana_dias días hábiles ────────────────────
    n_obs     = len(fechas)
    n_paneles = max(1, int(np.ceil(n_obs / ventana_dias)))

    fig, axes = plt.subplots(
        n_paneles, 1,
        figsize=(16, 4 * n_paneles),
        gridspec_kw={"hspace": 0.45},
    )
    if n_paneles == 1:
        axes = [axes]

    y_all = np.concatenate([
        preds[tau][idx_split] / 1e6
        for tau in preds
        if tau not in (0.50,)
    ] + [y_real])
    y_lim_min = np.nanpercentile(y_all, 1)
    y_lim_max = np.nanpercentile(y_all, 99)
    pad = (y_lim_max - y_lim_min) * 0.08
    ylim = (y_lim_min - pad, y_lim_max + pad)

    for p, ax in enumerate(axes):
        i0 = p * ventana_dias
        i1 = min(i0 + ventana_dias, n_obs)
        sl = slice(i0, i1)

        f_sl  = fechas[sl]
        yr_sl = y_real[sl]

        if 0.01 in preds and 0.99 in preds:
            ax.fill_between(f_sl,
                            preds[0.01][idx_split][sl] / 1e6,
                            preds[0.99][idx_split][sl] / 1e6,
                            alpha=0.10, color="steelblue", label="P01–P99")
        if 0.05 in preds and 0.95 in preds:
            ax.fill_between(f_sl,
                            preds[0.05][idx_split][sl] / 1e6,
                            preds[0.95][idx_split][sl] / 1e6,
                            alpha=0.22, color="steelblue", label="P05–P95")
        if 0.50 in preds:
            ax.plot(f_sl, preds[0.50][idx_split][sl] / 1e6,
                    color="steelblue", lw=1.8, label="Mediana pred.", zorder=4)
        ax.plot(f_sl, yr_sl, color="black", lw=1.2, alpha=0.85,
                label="Realizado", zorder=5)
        ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)

        ax.set_ylim(*ylim)
        ax.set_ylabel("MM USD", fontsize=9)
        ax.set_title(
            f"{f_sl[0].strftime('%d %b %Y')} → {f_sl[-1].strftime('%d %b %Y')}",
            fontsize=9, style="italic",
        )
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        if p == 0:
            ax.legend(fontsize=9, ncol=4, loc="upper left", framealpha=0.9)

    fig.suptitle(
        f"Fan Chart [{label_titulo}] — {banco}  |  h = {h_ejemplo} días hábiles\n"
        f"Bandas: Q01–Q99 y Q05–Q95  vs  realizado  |  ventana = {ventana_dias} días hábiles/panel",
        fontweight="bold", fontsize=11, y=1.01,
    )

    nombre = f"fanchart_{split_name}_{banco}_h{h_ejemplo:02d}.png"
    plt.savefig(dir_plots / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [{banco}] Fan chart [{split_name}] guardado: {dir_plots / nombre}")


###############################################################################
# PARTE 8 — Pipeline principal
###############################################################################

def entrenar_banco(banco: str) -> dict | None:
    """
    Pipeline completo para un banco con split TRAIN / VAL / TEST:

      TRAIN : ajuste de pesos del modelo (early stopping interno de LightGBM)
      VAL   : Optuna usa este período para elegir hiperparámetros
      TEST  : evaluación final completamente honesta — NUNCA vista durante
              el entrenamiento ni durante la optimización de hiperparámetros

    Fases:
      1. Leer Parquet (solo el banco).
      2. Imputar NaN de calentamiento con mediana de TRAIN (sin leak).
      3. Split walk-forward: TRAIN | VAL | TEST.
      4. Optuna sobre TRAIN→VAL para hiperparámetros óptimos.
      5. Entrenamiento sobre TRAIN con mejores params → evaluación en TEST.
      6. Re-entrenamiento final sobre TRAIN+VAL+TEST → modelo de producción.
      7. Guardar modelos + metadata + gráficos.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BANCO: {banco}")
    logger.info(f"{'='*60}")
    t_banco = time.time()

    # ── 1. Lectura ──────────────────────────────────────────────
    try:
        df = leer_banco_parquet(RUTA_MATRIZ, banco)
    except Exception as e:
        logger.error(f"  [{banco}] Error leyendo Parquet: {e}")
        return None

    if df.empty or df["target"].notna().sum() < 500:
        logger.warning(f"  [{banco}] Datos insuficientes — omitiendo")
        return None

    logger.info(f"  [{banco}] Filas totales: {len(df):,} | con target: {df['target'].notna().sum():,}")

    # ── 2. Features y split ─────────────────────────────────────
    cols_feat = get_feature_cols(df)
    logger.info(f"  [{banco}] Features: {len(cols_feat)}")

    # ── 3. Split walk-forward en tres particiones ────────────────
    if USAR_GARCH_SIN_LEAKAGE:
        logger.info(f"  [{banco}] Re-estimando GARCH sin leakage (TRAIN hasta {CORTE_VAL.date()})...")
        df = reemplazar_garch_sin_leakage(df, CORTE_VAL)

    frac_present = [c for c in df.columns if c.endswith("_frac")]
    if frac_present:
        logger.info(f"  [{banco}] Re-calibrando FFD sin leakage ({len(frac_present)} cols)...")
        df = reemplazar_ffd_sin_leakage(df, CORTE_VAL)

    df_train, df_val, df_test = split_walk_forward(df, CORTE_VAL, CORTE_TEST)

    logger.info(
        f"  [{banco}] TRAIN: {df_train['fecha_t'].min().date()} → "
        f"{df_train['fecha_t'].max().date()} ({df_train['fecha_t'].nunique()} fechas)"
    )
    logger.info(
        f"  [{banco}] VAL  : {df_val['fecha_t'].min().date()} → "
        f"{df_val['fecha_t'].max().date()} ({df_val['fecha_t'].nunique()} fechas)"
    )
    logger.info(
        f"  [{banco}] TEST : {df_test['fecha_t'].min().date()} → "
        f"{df_test['fecha_t'].max().date()} ({df_test['fecha_t'].nunique()} fechas)"
    )

    # Imputar NaN con mediana de TRAIN (sin filtración de información futura)
    medianas_train = df_train[cols_feat].median()
    for _df in (df_train, df_val, df_test):
        _df[cols_feat] = _df[cols_feat].fillna(medianas_train)

    X_train, y_train = preparar_Xy(df_train, cols_feat)
    X_val,   y_val   = preparar_Xy(df_val,   cols_feat)
    X_test,  y_test  = preparar_Xy(df_test,  cols_feat)

    if len(X_train) < 200 or len(X_val) < 50 or len(X_test) < 20:
        logger.warning(f"  [{banco}] Split demasiado pequeño — omitiendo")
        return None

    # ── 4. Optuna: TRAIN → VAL ───────────────────────────────────
    t_optuna = time.time()
    best_params = optimizar_hiperparametros(
        X_train, y_train, X_val, y_val, N_TRIALS_OPTUNA, banco
    )
    logger.info(f"  [{banco}] Optuna completado en {(time.time()-t_optuna)/60:.1f} min")

    # ── 5. Evaluación honesta en TEST ────────────────────────────
    t_eval = time.time()
    logger.info(f"  [{banco}] Entrenando sobre TRAIN para evaluación en TEST...")
    modelos_eval = entrenar_quantiles(X_train, y_train, best_params, QUANTILES, banco)
    metricas_test, preds_test = evaluar_modelos(
        modelos_eval, X_test, y_test, banco, split_name="test"
    )
    _, preds_val_diag = evaluar_modelos(
        modelos_eval, X_val, y_val, banco, split_name="val"
    )
    logger.info(f"  [{banco}] Evaluación TEST completada en {(time.time()-t_eval):.1f} s")

    metricas = {**metricas_test}

    # Guardar modelos_eval (TRAIN only) → usados por aux scripts para evaluación honesta OOS
    guardar_modelos(modelos_eval, metricas, best_params, cols_feat, banco, DIR_MODELOS_EVAL)

    # ── 6. Re-entrenamiento final sobre todos los datos ──────────
    t_prod = time.time()
    logger.info(f"  [{banco}] Re-entrenamiento final (TRAIN+VAL+TEST)...")
    X_full = pd.concat([X_train, X_val, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_val, y_test], ignore_index=True)
    modelos_prod = entrenar_quantiles(X_full, y_full, best_params, QUANTILES, banco)
    logger.info(f"  [{banco}] Re-entrenamiento completado en {(time.time()-t_prod):.1f} s")

    # ── 7. Guardar ───────────────────────────────────────────────
    guardar_modelos(modelos_prod, metricas, best_params, cols_feat, banco, DIR_MODELOS)

    # ── 8. Gráficos ─────────────────────────────────────────────
    graficar_importancia(modelos_prod, cols_feat, banco, DIR_PLOTS)
    df_test_plot = df_test[df_test["target"].notna()].reset_index(drop=True)
    df_val_plot  = df_val[df_val["target"].notna()].reset_index(drop=True)
    graficar_fanchart_split(df_test_plot, preds_test,     y_test, banco, DIR_PLOTS, split_name="test")
    graficar_fanchart_split(df_val_plot,  preds_val_diag, y_val,  banco, DIR_PLOTS, split_name="val")

    # ── Resumen de tiempos por banco ─────────────────────────────
    t_total_banco = time.time() - t_banco
    logger.info(
        f"  [{banco}] ✓ Completado en {t_total_banco/60:.1f} min  "
        f"(Optuna: {(time.time()-t_banco - (time.time()-t_optuna)):.0f}s estimado)"
    )

    # Liberar memoria
    del df, df_train, df_val, df_test
    del X_train, y_train, X_val, y_val, X_test, y_test
    del X_full, y_full, modelos_eval, modelos_prod
    gc.collect()

    return {"banco": banco, "tiempo_min": round(t_total_banco / 60, 1), **metricas}


def main():
    t_inicio = time.time()
    logger.info("=" * 70)
    logger.info("STEP003 — Entrenamiento LightGBM Quantile Regression")
    logger.info("=" * 70)
    logger.info(f"  Matriz de features : {RUTA_MATRIZ}")
    logger.info(f"  Directorio modelos : {DIR_MODELOS}")
    logger.info(f"  Quantiles          : {QUANTILES}")
    logger.info(f"  Corte VAL          : {CORTE_VAL.date()}  (inicio VAL / fin TRAIN)")
    logger.info(f"  Corte TEST         : {CORTE_TEST.date()} (inicio TEST, alineado con tasas allocation)")
    logger.info(f"  Trials Optuna      : {N_TRIALS_OPTUNA}")

    if not RUTA_MATRIZ.exists():
        logger.error(f"No se encontró la matriz: {RUTA_MATRIZ}")
        logger.error("Ejecutar step001_build_feature_matrix.py primero.")
        return

    # ── Leer lista de bancos desde el Parquet ────────────────────
    df_bancos = pd.read_parquet(RUTA_MATRIZ, columns=["banco"])
    bancos_disponibles = sorted(df_bancos["banco"].unique())
    del df_bancos
    gc.collect()

    # Aplicar filtro BANCOS_A_ENTRENAR
    if BANCOS_A_ENTRENAR is not None:
        no_encontrados = [b for b in BANCOS_A_ENTRENAR if b not in bancos_disponibles]
        if no_encontrados:
            logger.error(
                f"Bancos solicitados no encontrados en la matriz: {no_encontrados}\n"
                f"Disponibles: {bancos_disponibles}\n"
                "¿Se ejecutó step001 con la versión que incluye SISTEMA?"
            )
            return
        lista_bancos = [b for b in BANCOS_A_ENTRENAR if b in bancos_disponibles]
    else:
        lista_bancos = bancos_disponibles

    logger.info(f"  Bancos disponibles : {bancos_disponibles}")
    logger.info(f"  Bancos a entrenar  : {lista_bancos}")
    logger.info("")

    # ── Entrenar banco por banco ─────────────────────────────────
    resumen = []
    for banco in lista_bancos:
        resultado = entrenar_banco(banco)
        if resultado:
            resumen.append(resultado)

    # ── Tabla resumen de métricas ────────────────────────────────
    if resumen:
        df_resumen = pd.DataFrame(resumen).set_index("banco")
        logger.info("\n" + "=" * 70)
        logger.info("RESUMEN DE MÉTRICAS — TEST (holdout final, nunca visto en entrenamiento)")
        logger.info("=" * 70)
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", 20):
            logger.info("\n" + df_resumen.to_string())

        ruta_resumen = DIR_MODELOS / "resumen_metricas.csv"
        df_resumen.to_csv(ruta_resumen)
        logger.info(f"\nResumen guardado en: {ruta_resumen}")

    t_total = time.time() - t_inicio
    logger.info("\n" + "=" * 70)
    logger.info(f"✓ Entrenamiento completado en {t_total/60:.1f} min  ({t_total:.0f} s)")
    if resumen:
        logger.info("  Tiempo por banco:")
        for r in resumen:
            logger.info(f"    {r['banco']:20s}: {r.get('tiempo_min', '?')} min")
    logger.info(f"  Modelos en  : {DIR_MODELOS}")
    logger.info(f"  Gráficos en : {DIR_PLOTS}")
    logger.info("  → Siguiente paso: step004_predict.py (predicción en tiempo real)")


if __name__ == "__main__":
    main()
