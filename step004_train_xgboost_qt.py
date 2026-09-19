# -*- coding: utf-8 -*-
from __future__ import annotations
"""
step004_train_xgboost.py
Entrenamiento de modelos XGBoost de quantile regression para predicción de
flujos netos D−R (liquidez en ME) del sistema bancario peruano.

Diseño:
  - Misma estructura que step003 (LightGBM) para comparación directa.
  - Objetivo: arctan pinball loss — ecuación (11) de Zheng et al. (2024).
    Aproxima la hessiana cero de la pinball exacta con una campana suavizada,
    permitiendo que XGBoost construya árboles con splits estables.
  - Gradiente : g = (τ - 0.5 + arctan(u/s)/π) + u·s / (π·(s²+u²))
  - Hessiana  : h = 2s³ / (π·(s²+u²)²)    siempre positiva
  - Parámetro s (smoothing) tuneado por Optuna junto con los demás hiperparámetros.
  - Evaluación y early stopping con pinball loss EXACTA (no aproximada).4
  - Corrección de cruce de cuantiles: np.sort por fila sobre predicciones finales.
  - Modelos guardados como .json (formato nativo XGBoost) + metadata .json.

Flujo:
  1. Leer matriz_features.parquet (banco por banco).
  2. Walk-forward split con fechas fijas (mismo CORTE_VAL / CORTE_TEST que step003).
  3. Optuna: optimizar pinball exacta por cada cuantil (τ) en VAL → best_params_por_tau.
  4. Re-entrenar cada quantil con sus mejores hiperparámetros sobre TRAIN.
  5. Evaluar en TEST (holdout honesto).
  6. Re-entrenar final sobre TRAIN+VAL+TEST → modelo de producción.
  7. Guardar modelos + metadata JSON + gráficos.
"""

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
DIR_MODELOS      = BASE_SISTEMA / "2. Output" / "modelos_xgb_qt"
DIR_MODELOS_EVAL = DIR_MODELOS / "eval"   # modelos TRAIN-only → evaluación honesta OOS
DIR_PLOTS        = BASE_SISTEMA / "2. Output" / "plots_entrenamiento_xgb_qt"

DIR_MODELOS.mkdir(parents=True, exist_ok=True)
DIR_MODELOS_EVAL.mkdir(parents=True, exist_ok=True)
DIR_PLOTS.mkdir(parents=True, exist_ok=True)

# Quantiles a producir
QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]

# Mismo split que step003 — TEST alineado con tasas del allocation
CORTE_VAL  = pd.Timestamp("2022-07-01")
CORTE_TEST = pd.Timestamp("2023-01-03")

# Trials Optuna por banco × cuantil
N_TRIALS_OPTUNA = 60

# Rango del parámetro de suavizado s relativo a std(y_train)
# s pequeño → arctan más cercana a pinball exacta, hessiana más pequeña
# s grande  → hessiana más grande, pero pérdida más alejada de pinball exacta
S_MIN_FACTOR = 0.01   # s_min = std(y_train) * 0.01
S_MAX_FACTOR = 1.0    # s_max = std(y_train) * 1.0

BANCOS_A_ENTRENAR = ["SISTEMA"]

COLS_EXCLUIR = {"fecha_t", "banco", "target"}
COLS_TEXTO   = {"banco"}

# True  → re-estima GARCH(1,1) solo con datos TRAIN (hasta CORTE_VAL) antes de entrenar
# False → usa GARCH del parquet tal cual (comportamiento original)
USAR_GARCH_SIN_LEAKAGE = False


###############################################################################
# PARTE 1 — Utilidades
###############################################################################

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss promedio — exacta, sin aproximación."""
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def corregir_cruce_cuantiles(preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """
    Garantiza monotonicidad entre cuantiles: Q01 ≤ Q05 ≤ Q50 ≤ Q95 ≤ Q99.
    Ordena las predicciones por fila para los cuantiles solicitados.
    """
    taus   = sorted(preds.keys())
    matrix = np.column_stack([preds[t] for t in taus])
    matrix = np.sort(matrix, axis=1)
    return {t: matrix[:, i] for i, t in enumerate(taus)}


def leer_banco_parquet(ruta: Path, banco: str) -> pd.DataFrame:
    """Lee todas las filas de un banco desde el parquet."""
    df = pd.read_parquet(ruta, filters=[("banco", "==", banco)])
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
    Divide temporalmente en TRAIN | VAL | TEST sin solapamiento.
    Idéntico al step003 — mismos cortes para comparación directa.
    """
    df_train = df[df["fecha_t"] <  corte_val].copy()
    df_val   = df[(df["fecha_t"] >= corte_val) & (df["fecha_t"] < corte_test)].copy()
    df_test  = df[df["fecha_t"] >= corte_test].copy()

    n_train = df_train["fecha_t"].nunique()
    n_total = n_train + df_val["fecha_t"].nunique() + df_test["fecha_t"].nunique()

    if n_train < n_total * 0.5:
        raise ValueError(
            f"TRAIN solo tiene {n_train} fechas ({100*n_train/n_total:.0f}% del total). "
            "Ajusta CORTE_VAL o CORTE_TEST."
        )
    return df_train, df_val, df_test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Infiere columnas de features — idéntico al step003."""
    excluir = COLS_EXCLUIR | {"fecha_th"}
    cols_validas = []
    for c in df.columns:
        if c in excluir:
            continue
        if df[c].dtype.kind in ("f", "i", "u", "b"):
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
# PARTE 2 — Objetivo arctan y métrica de evaluación
###############################################################################

def make_quantile_objective(tau: float, s: float, std_y: float):
    """
    Gradiente y hessiana de la arctan pinball loss — ecuación (11):

      L(u) = (τ - 0.5 + arctan(u/s)/π) · u + s/π

    Gradiente (∂L/∂ŷ, regla de la cadena con u = y - ŷ, ∂u/∂ŷ = -1):
      g = -[(τ - 0.5 + arctan(u/s)/π) + u·s / (π·(s²+u²))]

    Hessiana (∂²L/∂ŷ², el (-1)² cancela):
      h = 2s³ / (π·(s²+u²)²)    — siempre positiva

    Normalización: cuando u >> s (típico en la práctica), h ≈ 2s³/(π·u⁴) → ~1e-13,
    lo que impide cualquier split porque Σh << min_child_weight.
    Se multiplica (g, h) por scale = π·(s²+std_y²)²/(2s³) para que
    h(u=std_y) = 1 por muestra, dando a min_child_weight semántica de conteo.
    Leaf = -Σg/Σh es invariante a la escala, por lo que las predicciones no cambian.
    """
    _scale = np.pi * (s**2 + std_y**2)**2 / (2.0 * s**3)

    def objective(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        u    = dtrain.get_label() - y_pred
        grad = -((tau - 0.5 + np.arctan(u / s) / np.pi) + u * s / (np.pi * (s**2 + u**2))) * _scale
        hess = 2 * s**3 / (np.pi * (s**2 + u**2)**2) * _scale
        return grad, hess
    return objective


def make_pinball_metric(tau: float):
    """
    Pinball loss EXACTA para early stopping.
    XGBoost usa esta métrica para decidir cuándo parar — no la arctan aproximada.
    Esto garantiza que el criterio de parada sea la métrica real que nos importa.
    """
    def metric(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label()
        u      = y_true - y_pred
        loss   = float(np.mean(np.where(u >= 0, tau * u, (tau - 1) * u)))
        return "pinball", loss
    return metric


###############################################################################
# PARTE 3 — Optimización Bayesiana con Optuna
###############################################################################

def objective_xgb(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    std_y: float,
    tau: float,
) -> float:
    """
    Función objetivo para Optuna: pinball loss exacta en validación para el τ dado.
    Cada cuantil tiene su propio estudio Optuna → best_params_por_tau.
    Incluye s (smoothing) como hiperparámetro adicional respecto a step003.
    """
    s = trial.suggest_float("s", std_y * S_MIN_FACTOR,
                                  std_y * S_MAX_FACTOR, log=True)
    params = {
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth"       : trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 200),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method"     : "hist",
        "seed"            : 42,
    }
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)

    dtrain_opt = xgb.DMatrix(X_train, label=y_train)
    dval_opt   = xgb.DMatrix(X_val,   label=y_val)

    model = xgb.train(
        params,
        dtrain_opt,
        num_boost_round=n_estimators,
        obj=make_quantile_objective(tau, s, std_y),
        custom_metric=make_pinball_metric(tau),
        evals=[(dval_opt, "val")],
        callbacks=[xgb.callback.EarlyStopping(
            rounds=50, metric_name="pinball", save_best=False, maximize=False
        )],
        verbose_eval=False,
    )
    preds_val = model.predict(dval_opt)
    return pinball_loss(y_val.values, preds_val, tau)


def optimizar_hiperparametros(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    std_y: float,
    n_trials: int,
    banco: str,
    tau: float,      # ← cuantil objetivo del estudio
) -> dict:
    """Ejecuta Optuna para un τ específico y retorna los mejores hiperparámetros."""
    logger.info(f"  [{banco}] Optimización Bayesiana τ={tau:.2f} ({n_trials} trials)...")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective_xgb(
            trial, X_train, y_train, X_val, y_val, std_y, tau
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info(
        f"  [{banco}] τ={tau:.2f} → mejor pinball: {study.best_value:.4f} "
        f"| n_est={best['n_estimators']} lr={best['learning_rate']:.4f} "
        f"s={best['s']:.2f}"
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
    std_y: float,
) -> dict[float, xgb.Booster]:
    """
    Re-entrena un modelo XGBoost por cada cuantil con los hiperparámetros dados.
    s y std_y se pasan al objetivo arctan para normalizar la hessiana.
    """
    s_best       = best_params["s"]
    n_estimators = best_params["n_estimators"]
    params_base  = {k: v for k, v in best_params.items()
                    if k not in ("s", "n_estimators")}
    params_base.update({"tree_method": "hist", "seed": 42})

    dtrain = xgb.DMatrix(X_train, label=y_train)

    modelos = {}
    for tau in quantiles:
        logger.info(f"  [{banco}] Entrenando τ={tau:.2f}  (s={s_best:.2f})...")
        model = xgb.train(
            params_base,
            dtrain,
            num_boost_round=n_estimators,
            obj=make_quantile_objective(tau, s_best, std_y),
            verbose_eval=False,
        )
        modelos[tau] = model
    return modelos


###############################################################################
# PARTE 5 — Evaluación
###############################################################################

def evaluar_modelos(
    modelos: dict[float, xgb.Booster],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    banco: str,
    split_name: str = "val",
) -> tuple[dict, dict]:
    """
    Calcula pinball loss exacta por quantil y RMSE de la mediana.
    Aplica corrección de cruce de cuantiles antes de evaluar.
    """
    deval     = xgb.DMatrix(X_eval)
    preds_raw = {tau: model.predict(deval) for tau, model in modelos.items()}
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
    modelos: dict[float, xgb.Booster],
    metricas: dict,
    best_params_por_tau: dict[float, dict],   # ← dict de dicts, uno por cuantil
    cols_feat: list[str],
    banco: str,
    dir_modelos: Path,
):
    """
    Guarda cada modelo como .json (formato nativo XGBoost) y un JSON con
    metadata: features, hiperparámetros por cuantil (incluye s óptimo), métricas, fecha.
    Las claves float del dict se convierten a str para ser serializables en JSON.
    """
    fecha_hoy = pd.Timestamp.today().strftime("%Y%m%d")

    for tau, model in modelos.items():
        ruta_model = dir_modelos / f"xgb_qt_{banco}_q{int(tau*100):02d}_{fecha_hoy}.json"
        model.save_model(str(ruta_model))

    metadata = {
        "banco"               : banco,
        "modelo"              : "xgboost_arctan_pinball",
        "fecha_entrenamiento" : pd.Timestamp.today().strftime("%Y-%m-%d"),
        "quantiles"           : [float(t) for t in sorted(modelos.keys())],
        "n_features"          : len(cols_feat),
        "features"            : cols_feat,
        # float keys → str para que JSON no proteste
        "best_params_por_tau" : {
            str(tau): bp for tau, bp in best_params_por_tau.items()
        },
        "metricas"            : metricas,
    }
    ruta_meta = dir_modelos / f"metadata_xgb_qt_{banco}_{fecha_hoy}.json"
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"  [{banco}] Modelos guardados en {dir_modelos}")


###############################################################################
# PARTE 7 — Gráficos
###############################################################################

def graficar_importancia(
    modelos: dict[float, xgb.Booster],
    cols_feat: list[str],
    banco: str,
    dir_plots: Path,
    top_n: int = 20,
):
    """
    Bar chart horizontal: importancia promedio entre quantiles (gain).
    XGBoost devuelve dict {feature: score} — se alinea con cols_feat
    rellenando con 0 las features no usadas en ningún split.
    """
    importancias = np.zeros(len(cols_feat))
    for model in modelos.values():
        scores = model.get_score(importance_type="gain")
        imp    = np.array([scores.get(c, 0.0) for c in cols_feat])
        importancias += imp / len(modelos)

    idx_top = np.argsort(importancias)[-top_n:]
    nombres = [cols_feat[i] for i in idx_top]
    valores = importancias[idx_top]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(nombres, valores, color="darkorange", alpha=0.85)
    ax.set_xlabel("Importancia promedio (gain)", fontsize=10)
    ax.set_title(
        f"Feature Importance XGBoost — {banco}\n(promedio entre quantiles, top {top_n})",
        fontweight="bold", fontsize=11,
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    nombre = f"feature_importance_xgb_{banco}.png"
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
    ventana_dias: int = 90,
):
    """Fan chart idéntico al step003 — paneles de ventana_dias días hábiles."""
    mask_h = df_split["h"] == h_ejemplo
    if mask_h.sum() < 5:
        conteos    = df_split.groupby("h").size()
        candidatos = conteos[conteos >= 5]
        if candidatos.empty:
            return
        h_ejemplo = int(candidatos.index[np.argmin(np.abs(candidatos.index - h_ejemplo))])
        mask_h    = df_split["h"] == h_ejemplo

    idx_split    = np.where(mask_h.values)[0]
    fechas       = pd.to_datetime(df_split.loc[mask_h, "fecha_t"].values)
    y_real       = y_split.values[idx_split] / 1e6
    label_titulo = "Test (holdout)" if split_name == "test" else "Validación"

    n_obs     = len(fechas)
    n_paneles = max(1, int(np.ceil(n_obs / ventana_dias)))

    fig, axes = plt.subplots(
        n_paneles, 1,
        figsize=(16, 4 * n_paneles),
        gridspec_kw={"hspace": 0.45},
    )
    if n_paneles == 1:
        axes = [axes]

    y_all     = np.concatenate([preds[tau][idx_split] / 1e6
                                for tau in preds if tau != 0.50] + [y_real])
    y_lim_min = np.nanpercentile(y_all, 1)
    y_lim_max = np.nanpercentile(y_all, 99)
    pad       = (y_lim_max - y_lim_min) * 0.08
    ylim      = (y_lim_min - pad, y_lim_max + pad)

    for p, ax in enumerate(axes):
        i0   = p * ventana_dias
        i1   = min(i0 + ventana_dias, n_obs)
        sl   = slice(i0, i1)
        f_sl = fechas[sl]

        if 0.01 in preds and 0.99 in preds:
            ax.fill_between(f_sl, preds[0.01][idx_split][sl] / 1e6,
                            preds[0.99][idx_split][sl] / 1e6,
                            alpha=0.10, color="darkorange", label="P01–P99")
        if 0.05 in preds and 0.95 in preds:
            ax.fill_between(f_sl, preds[0.05][idx_split][sl] / 1e6,
                            preds[0.95][idx_split][sl] / 1e6,
                            alpha=0.22, color="darkorange", label="P05–P95")
        if 0.50 in preds:
            ax.plot(f_sl, preds[0.50][idx_split][sl] / 1e6,
                    color="darkorange", lw=1.8, label="Mediana pred.", zorder=4)
        ax.plot(f_sl, y_real[sl], color="black", lw=1.2, alpha=0.85,
                label="Realizado", zorder=5)
        ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
        ax.set_ylim(*ylim)
        ax.set_ylabel("MM USD", fontsize=9)
        ax.set_title(f"{f_sl[0].strftime('%d %b %Y')} → {f_sl[-1].strftime('%d %b %Y')}",
                     fontsize=9, style="italic")
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        if p == 0:
            ax.legend(fontsize=9, ncol=4, loc="upper left", framealpha=0.9)

    fig.suptitle(
        f"Fan Chart XGBoost [{label_titulo}] — {banco}  |  h = {h_ejemplo} días hábiles\n"
        f"Bandas: Q01–Q99 y Q05–Q95  vs  realizado  |  ventana = {ventana_dias} días hábiles/panel",
        fontweight="bold", fontsize=11, y=1.01,
    )
    nombre = f"fanchart_xgb_{split_name}_{banco}_h{h_ejemplo:02d}.png"
    plt.savefig(dir_plots / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [{banco}] Fan chart [{split_name}] guardado: {dir_plots / nombre}")


###############################################################################
# PARTE 8 — Pipeline principal
###############################################################################

def entrenar_banco(banco: str) -> dict | None:
    """
    Pipeline completo idéntico al step003, reemplazando LightGBM por XGBoost
    con objetivo arctan pinball. Mismo split, mismas fases, mismas métricas
    → comparación directa entre ambos modelos.

    Cambio clave vs versión anterior: Optuna corre un estudio por cada cuantil,
    produciendo best_params_por_tau = {tau: best_params_dict}.
    Cada modelo se entrena y re-entrena con sus propios hiperparámetros óptimos.
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

    # ── 3. Split walk-forward ────────────────────────────────────
    if USAR_GARCH_SIN_LEAKAGE:
        logger.info(f"  [{banco}] Re-estimando GARCH sin leakage (TRAIN hasta {CORTE_VAL.date()})...")
        df = reemplazar_garch_sin_leakage(df, CORTE_VAL)

    df_train, df_val, df_test = split_walk_forward(df, CORTE_VAL, CORTE_TEST)

    logger.info(f"  [{banco}] TRAIN: {df_train['fecha_t'].min().date()} → "
                f"{df_train['fecha_t'].max().date()} ({df_train['fecha_t'].nunique()} fechas)")
    logger.info(f"  [{banco}] VAL  : {df_val['fecha_t'].min().date()} → "
                f"{df_val['fecha_t'].max().date()} ({df_val['fecha_t'].nunique()} fechas)")
    logger.info(f"  [{banco}] TEST : {df_test['fecha_t'].min().date()} → "
                f"{df_test['fecha_t'].max().date()} ({df_test['fecha_t'].nunique()} fechas)")

    # Imputar NaN con mediana de TRAIN (sin leak)
    medianas_train = df_train[cols_feat].median()
    for _df in (df_train, df_val, df_test):
        _df[cols_feat] = _df[cols_feat].fillna(medianas_train)

    X_train, y_train = preparar_Xy(df_train, cols_feat)
    X_val,   y_val   = preparar_Xy(df_val,   cols_feat)
    X_test,  y_test  = preparar_Xy(df_test,  cols_feat)

    if len(X_train) < 200 or len(X_val) < 50 or len(X_test) < 20:
        logger.warning(f"  [{banco}] Split demasiado pequeño — omitiendo")
        return None

    # std(y_train) para escalar el rango de s en Optuna
    std_y = float(y_train.std())
    logger.info(f"  [{banco}] std(y_train) = {std_y:,.0f}  "
                f"→ rango s: [{std_y*S_MIN_FACTOR:.1f}, {std_y*S_MAX_FACTOR:.0f}]")

    # ── 4. Optuna: un estudio por cuantil (TRAIN → VAL) ─────────
    # Resultado: best_params_por_tau = {tau: {hiperparámetros óptimos}}
    t_optuna = time.time()
    best_params_por_tau: dict[float, dict] = {}
    for tau in QUANTILES:
        best_params_por_tau[tau] = optimizar_hiperparametros(
            X_train, y_train, X_val, y_val, std_y, N_TRIALS_OPTUNA, banco, tau
        )
    logger.info(f"  [{banco}] Optuna completado en {(time.time()-t_optuna)/60:.1f} min "
                f"({len(QUANTILES)} cuantiles × {N_TRIALS_OPTUNA} trials)")

    # ── 5. Evaluación honesta en TEST ────────────────────────────
    # Cada cuantil se entrena sobre TRAIN con sus propios best_params
    t_eval = time.time()
    logger.info(f"  [{banco}] Entrenando sobre TRAIN para evaluación en TEST...")
    modelos_eval: dict[float, xgb.Booster] = {}
    for tau, bp in best_params_por_tau.items():
        modelos_eval[tau] = entrenar_quantiles(
            X_train, y_train, bp, [tau], banco, std_y
        )[tau]

    metricas_test, preds_test = evaluar_modelos(
        modelos_eval, X_test, y_test, banco, split_name="test"
    )
    _, preds_val_diag = evaluar_modelos(
        modelos_eval, X_val, y_val, banco, split_name="val"
    )
    logger.info(f"  [{banco}] Evaluación TEST completada en {(time.time()-t_eval):.1f} s")

    metricas = {**metricas_test}

    # Guardar modelos_eval (TRAIN only) → evaluación honesta OOS
    guardar_modelos(modelos_eval, metricas, best_params_por_tau, cols_feat, banco, DIR_MODELOS_EVAL)

    # ── 6. Re-entrenamiento final sobre todos los datos ──────────
    # Cada cuantil usa sus propios best_params del estudio Optuna
    t_prod = time.time()
    logger.info(f"  [{banco}] Re-entrenamiento final (TRAIN+VAL+TEST)...")
    X_full = pd.concat([X_train, X_val, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_val, y_test], ignore_index=True)

    modelos_prod: dict[float, xgb.Booster] = {}
    for tau, bp in best_params_por_tau.items():
        modelos_prod[tau] = entrenar_quantiles(
            X_full, y_full, bp, [tau], banco, std_y
        )[tau]
    logger.info(f"  [{banco}] Re-entrenamiento completado en {(time.time()-t_prod):.1f} s")

    # ── 7. Guardar modelos de producción ─────────────────────────
    guardar_modelos(modelos_prod, metricas, best_params_por_tau, cols_feat, banco, DIR_MODELOS)

    # ── 8. Gráficos ─────────────────────────────────────────────
    graficar_importancia(modelos_prod, cols_feat, banco, DIR_PLOTS)
    df_test_plot = df_test[df_test["target"].notna()].reset_index(drop=True)
    df_val_plot  = df_val[df_val["target"].notna()].reset_index(drop=True)
    graficar_fanchart_split(df_test_plot, preds_test,     y_test, banco, DIR_PLOTS, split_name="test")
    graficar_fanchart_split(df_val_plot,  preds_val_diag, y_val,  banco, DIR_PLOTS, split_name="val")

    t_total_banco = time.time() - t_banco
    logger.info(f"  [{banco}] ✓ Completado en {t_total_banco/60:.1f} min")

    del df, df_train, df_val, df_test
    del X_train, y_train, X_val, y_val, X_test, y_test
    del X_full, y_full, modelos_eval, modelos_prod
    gc.collect()

    return {"banco": banco, "tiempo_min": round(t_total_banco / 60, 1), **metricas}


def main():
    t_inicio = time.time()
    logger.info("=" * 70)
    logger.info("STEP004 — Entrenamiento XGBoost Arctan Pinball Quantile Regression")
    logger.info("=" * 70)
    logger.info(f"  Matriz de features : {RUTA_MATRIZ}")
    logger.info(f"  Directorio modelos : {DIR_MODELOS}")
    logger.info(f"  Quantiles          : {QUANTILES}")
    logger.info(f"  Corte VAL          : {CORTE_VAL.date()}")
    logger.info(f"  Corte TEST         : {CORTE_TEST.date()}")
    logger.info(f"  Trials Optuna      : {N_TRIALS_OPTUNA} por cuantil")
    logger.info(f"  Total trials       : {N_TRIALS_OPTUNA * len(QUANTILES)} ({N_TRIALS_OPTUNA}×{len(QUANTILES)} cuantiles)")
    logger.info(f"  Rango s            : [std*{S_MIN_FACTOR}, std*{S_MAX_FACTOR}]")

    if not RUTA_MATRIZ.exists():
        logger.error(f"No se encontró la matriz: {RUTA_MATRIZ}")
        logger.error("Ejecutar step001_build_feature_matrix.py primero.")
        return

    df_bancos          = pd.read_parquet(RUTA_MATRIZ, columns=["banco"])
    bancos_disponibles = sorted(df_bancos["banco"].unique())
    del df_bancos
    gc.collect()

    if BANCOS_A_ENTRENAR is not None:
        no_encontrados = [b for b in BANCOS_A_ENTRENAR if b not in bancos_disponibles]
        if no_encontrados:
            logger.error(f"Bancos no encontrados: {no_encontrados}")
            return
        lista_bancos = [b for b in BANCOS_A_ENTRENAR if b in bancos_disponibles]
    else:
        lista_bancos = bancos_disponibles

    logger.info(f"  Bancos disponibles : {bancos_disponibles}")
    logger.info(f"  Bancos a entrenar  : {lista_bancos}")
    logger.info("")

    resumen = []
    for banco in lista_bancos:
        resultado = entrenar_banco(banco)
        if resultado:
            resumen.append(resultado)

    if resumen:
        df_resumen = pd.DataFrame(resumen).set_index("banco")
        logger.info("\n" + "=" * 70)
        logger.info("RESUMEN DE MÉTRICAS — TEST (holdout final)")
        logger.info("=" * 70)
        with pd.option_context("display.float_format", "{:,.2f}".format,
                               "display.max_columns", 20):
            logger.info("\n" + df_resumen.to_string())

        ruta_resumen = DIR_MODELOS / "resumen_metricas_xgb.csv"
        df_resumen.to_csv(ruta_resumen)
        logger.info(f"\nResumen guardado en: {ruta_resumen}")

    t_total = time.time() - t_inicio
    logger.info("\n" + "=" * 70)
    logger.info(f"✓ Entrenamiento completado en {t_total/60:.1f} min  ({t_total:.0f} s)")
    if resumen:
        for r in resumen:
            logger.info(f"    {r['banco']:20s}: {r.get('tiempo_min', '?')} min")
    logger.info(f"  Modelos en  : {DIR_MODELOS}")
    logger.info(f"  Gráficos en : {DIR_PLOTS}")
    logger.info("  → Comparar métricas TEST vs step003 (LightGBM) con aux_calibracion.py")


if __name__ == "__main__":
    main()