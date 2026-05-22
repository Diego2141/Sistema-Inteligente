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
  2. Walk-forward split: últimas SEMANAS_VAL semanas hábiles como validación.
  3. Optuna: optimizar pinball loss mediana (τ=0.50) en el split.
  4. Re-entrenar cada quantil con los mejores hiperparámetros sobre train completo.
  5. Evaluar en validación: pinball loss por quantil y RMSE mediana.
  6. Guardar modelos + metadata JSON + gráfico de feature importance.
"""

import gc
import json
import logging
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
DIR_MODELOS  = BASE_SISTEMA / "2. Output" / "modelos"
DIR_PLOTS    = BASE_SISTEMA / "2. Output" / "plots_entrenamiento"

DIR_MODELOS.mkdir(parents=True, exist_ok=True)
DIR_PLOTS.mkdir(parents=True, exist_ok=True)

# Quantiles a producir
QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]

# Semanas hábiles reservadas para cada split (walk-forward, orden temporal):
#   |─── TRAIN ──────────────|─── VAL ───|─── TEST ───|
#   TRAIN: entrena modelos + usa early stopping interno
#   VAL:   Optuna ajusta hiperparámetros aquí — el modelo NUNCA ve TEST
#   TEST:  evaluación final honesta, nunca tocada durante entrenamiento/optimización
SEMANAS_VAL  = 26   # ~6 meses para ajuste de hiperparámetros (Optuna)
SEMANAS_TEST = 13   # ~3 meses como holdout final de evaluación

# Trials Optuna por banco
N_TRIALS_OPTUNA = 60

# Features a excluir del entrenamiento (identificadores, no predictores)
COLS_EXCLUIR = {"fecha_t", "banco", "target"}

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
    df: pd.DataFrame, semanas_val: int, semanas_test: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide temporalmente en tres particiones sin solapamiento:
      TRAIN | VAL | TEST  (orden cronológico estricto)

    - TEST  : últimas semanas_test semanas hábiles → evaluación final honesta.
    - VAL   : semanas_val semanas hábiles anteriores al TEST → Optuna.
    - TRAIN : todo lo anterior → ajuste del modelo.

    Garantías:
      · Ningún dato del futuro contamina el entrenamiento ni la optimización.
      · VAL y TEST son completamente disjuntos.
      · El mínimo de datos de TRAIN es 50% del total para evitar splits degenerados.
    """
    fechas_unicas = np.sort(df["fecha_t"].unique())
    n_fechas = len(fechas_unicas)

    # Convertir semanas a días hábiles (~5 por semana), con techo conservador
    n_test = min(semanas_test * 5, n_fechas // 6)
    n_val  = min(semanas_val  * 5, n_fechas // 4)

    # Garantizar que TRAIN tenga al menos 50% de las fechas
    if n_fechas - n_test - n_val < n_fechas // 2:
        n_val = max(10, n_fechas // 6)
        n_test = max(5, n_fechas // 8)

    corte_val  = fechas_unicas[n_fechas - n_test - n_val]
    corte_test = fechas_unicas[n_fechas - n_test]

    df_train = df[df["fecha_t"] <  corte_val ].copy()
    df_val   = df[(df["fecha_t"] >= corte_val) & (df["fecha_t"] < corte_test)].copy()
    df_test  = df[df["fecha_t"] >= corte_test].copy()
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
):
    """
    Para un horizonte h fijo, grafica la banda de predicción (Q01–Q99, Q05–Q95)
    vs el valor realizado a lo largo del período indicado (val o test).
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
    fechas    = df_split.loc[mask_h, "fecha_t"].values
    y_real    = y_split.values[idx_split] / 1e6

    label_titulo = "Test (holdout)" if split_name == "test" else "Validación"

    fig, ax = plt.subplots(figsize=(14, 6))

    if 0.01 in preds and 0.99 in preds:
        ax.fill_between(
            fechas,
            preds[0.01][idx_split] / 1e6,
            preds[0.99][idx_split] / 1e6,
            alpha=0.10, color="steelblue", label="P01–P99",
        )
    if 0.05 in preds and 0.95 in preds:
        ax.fill_between(
            fechas,
            preds[0.05][idx_split] / 1e6,
            preds[0.95][idx_split] / 1e6,
            alpha=0.20, color="steelblue", label="P05–P95",
        )
    if 0.50 in preds:
        ax.plot(fechas, preds[0.50][idx_split] / 1e6,
                color="steelblue", lw=1.8, label="Mediana pred.", zorder=4)

    ax.plot(fechas, y_real, color="black", lw=1.2, alpha=0.85,
            label="Realizado", zorder=5)
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)

    ax.set_xlabel("Fecha de predicción (t)", fontsize=10)
    ax.set_ylabel("Flujo neto D−R (MM USD)", fontsize=10)
    ax.set_title(
        f"Fan Chart [{label_titulo}] — {banco}  |  h = {h_ejemplo} días hábiles\n"
        f"Bandas: Q01–Q99 y Q05–Q95  vs  realizado",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=9, ncol=4, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.1f}"))
    plt.tight_layout()

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
    df_train, df_val, df_test = split_walk_forward(df, SEMANAS_VAL, SEMANAS_TEST)

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
    # TEST nunca se toca en este paso
    best_params = optimizar_hiperparametros(
        X_train, y_train, X_val, y_val, N_TRIALS_OPTUNA, banco
    )

    # ── 5. Evaluación honesta en TEST ────────────────────────────
    # Entrenamos sobre TRAIN (no sobre TRAIN+VAL) para que TEST sea limpio
    logger.info(f"  [{banco}] Entrenando sobre TRAIN para evaluación en TEST...")
    modelos_eval = entrenar_quantiles(X_train, y_train, best_params, QUANTILES, banco)
    metricas_test, preds_test = evaluar_modelos(
        modelos_eval, X_test, y_test, banco, split_name="test"
    )

    # También reportamos VAL para comparar con TEST (diagnosticar sobreajuste)
    _, preds_val_diag = evaluar_modelos(
        modelos_eval, X_val, y_val, banco, split_name="val"
    )

    metricas = {**metricas_test}

    # ── 6. Re-entrenamiento final sobre todos los datos ──────────
    # Este es el modelo que va a producción
    logger.info(f"  [{banco}] Re-entrenamiento final (TRAIN+VAL+TEST)...")
    X_full = pd.concat([X_train, X_val, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_val, y_test], ignore_index=True)
    modelos_prod = entrenar_quantiles(X_full, y_full, best_params, QUANTILES, banco)

    # ── 7. Guardar ───────────────────────────────────────────────
    guardar_modelos(modelos_prod, metricas, best_params, cols_feat, banco, DIR_MODELOS)

    # ── 8. Gráficos ─────────────────────────────────────────────
    graficar_importancia(modelos_prod, cols_feat, banco, DIR_PLOTS)
    # Alinear df con y y preds: filtrar solo filas con target (igual que preparar_Xy)
    df_test_plot = df_test[df_test["target"].notna()].reset_index(drop=True)
    df_val_plot  = df_val[df_val["target"].notna()].reset_index(drop=True)
    graficar_fanchart_split(df_test_plot, preds_test,     y_test, banco, DIR_PLOTS, split_name="test")
    graficar_fanchart_split(df_val_plot,  preds_val_diag, y_val,  banco, DIR_PLOTS, split_name="val")

    # Liberar memoria
    del df, df_train, df_val, df_test
    del X_train, y_train, X_val, y_val, X_test, y_test
    del X_full, y_full, modelos_eval, modelos_prod
    gc.collect()

    return {"banco": banco, **metricas}


def main():
    logger.info("=" * 70)
    logger.info("STEP003 — Entrenamiento LightGBM Quantile Regression")
    logger.info("=" * 70)
    logger.info(f"  Matriz de features : {RUTA_MATRIZ}")
    logger.info(f"  Directorio modelos : {DIR_MODELOS}")
    logger.info(f"  Quantiles          : {QUANTILES}")
    logger.info(f"  Semanas VAL        : {SEMANAS_VAL}  (~{SEMANAS_VAL//4} meses, Optuna)")
    logger.info(f"  Semanas TEST       : {SEMANAS_TEST} (~{SEMANAS_TEST//4} meses, holdout final)")
    logger.info(f"  Trials Optuna      : {N_TRIALS_OPTUNA}")

    if not RUTA_MATRIZ.exists():
        logger.error(f"No se encontró la matriz: {RUTA_MATRIZ}")
        logger.error("Ejecutar step001_build_feature_matrix.py primero.")
        return

    # ── Leer lista de bancos desde metadata del Parquet ──────────
    pf = pq.ParquetFile(RUTA_MATRIZ)
    # Lectura rápida solo de la columna banco para obtener la lista única
    df_bancos = pd.read_parquet(RUTA_MATRIZ, columns=["banco"])
    lista_bancos = sorted(df_bancos["banco"].unique())
    del df_bancos
    gc.collect()

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

    logger.info("\n✓ Entrenamiento completado.")
    logger.info(f"  Modelos en  : {DIR_MODELOS}")
    logger.info(f"  Gráficos en : {DIR_PLOTS}")
    logger.info("  → Siguiente paso: step004_predict.py (predicción en tiempo real)")


if __name__ == "__main__":
    main()
