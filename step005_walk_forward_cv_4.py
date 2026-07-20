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

import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"

BANCO   = "SISTEMA"
H_MIN   = 2
H_MAX   = 75
TAUS    = [0.01, 0.05, 0.50, 0.95, 0.99]

EXPANDING             = True
RECORTAR_INICIO_TRAIN = True
TRAIN_INICIO_CUTOFF   = "2020-01-01"

VENTANA_TRAIN_AÑOS  = 3
VENTANA_VAL_AÑOS    = 0.5
VENTANA_TEST_AÑOS   = 1
PASO_AÑOS           = 1
PURGE_DIAS_HAB      = 97   # h_max(75) + feature lookback(22)
MIN_TRAIN_ROWS      = 50

# Fixed hyperparameters — user-adjustable
HP: dict = {
    "n_estimators"    : 300,
    "max_depth"       : 4,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 1.0,
    "tree_method"     : "hist",
    "n_jobs"          : -1,
    "random_state"    : 42,
}

# Columns never used as features (KEY: h and log_h excluded in cv4)
COLS_EXCLUIR = {
    "fecha_t", "banco", "target", "fecha_th",
    "h", "log_h",
}

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
    Build walk-forward CV folds with an expanding training window.

    Returns a list of dicts with keys:
        fold, train_start, train_end, val_start, val_end, test_start, test_end
    """
    all_bdays = np.array(sorted(df["fecha_t"].unique()))

    if RECORTAR_INICIO_TRAIN:
        train_min = max(df["fecha_t"].min(), pd.Timestamp(TRAIN_INICIO_CUTOFF))
    else:
        train_min = df["fecha_t"].min()

    # First test window starts after train + val
    test_start = train_min + pd.DateOffset(
        years=VENTANA_TRAIN_AÑOS + VENTANA_VAL_AÑOS
    )

    folds = []
    fold_num = 0

    while True:
        test_end = test_start + pd.DateOffset(years=VENTANA_TEST_AÑOS) - pd.Timedelta(days=1)

        if test_end > df["fecha_t"].max():
            break

        val_start = test_start - pd.DateOffset(years=VENTANA_VAL_AÑOS)

        if EXPANDING:
            train_start = train_min
        else:
            train_start = val_start - pd.DateOffset(years=VENTANA_TRAIN_AÑOS)

        # --- Purge gap between train and val ---
        # Find the index of the first business day >= val_start
        idx_val = int(np.searchsorted(all_bdays, val_start))
        train_end_idx = max(0, idx_val - PURGE_DIAS_HAB - 1)
        train_end = pd.Timestamp(all_bdays[train_end_idx])

        # --- val_end: last business day before test_start (no purge val→test) ---
        idx_test = int(np.searchsorted(all_bdays, test_start))
        val_end_idx = max(0, idx_test - 1)
        val_end = pd.Timestamp(all_bdays[val_end_idx])

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
            "Fold %d  train %s–%s  val %s–%s  test %s–%s",
            fold_num,
            train_start.date(), train_end.date(),
            val_start.date(),   val_end.date(),
            test_start.date(),  test_end.date(),
        )

        test_start += pd.DateOffset(years=PASO_AÑOS)

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


def entrenar_modelos_h(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hp: dict,
) -> dict:
    """
    Train one XGBRegressor per quantile tau plus one for mean.

    Returns
    -------
    dict {tau (float | 'mean'): fitted XGBRegressor}
    """
    modelos: dict = {}

    for tau in TAUS:
        m = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=tau,
            **hp,
        )
        m.fit(X_train, y_train, verbose=False)
        modelos[tau] = m

    m_mean = xgb.XGBRegressor(objective="reg:squarederror", **hp)
    m_mean.fit(X_train, y_train, verbose=False)
    modelos["mean"] = m_mean

    return modelos


def calcular_metricas(
    preds_dict: dict,
    y_test: np.ndarray,
    h_val: int,
    fold_num: int,
) -> dict:
    """
    Compute pinball loss per quantile tau and RMSE for the mean model.

    Returns a flat dict suitable for appending to a list before pd.DataFrame().
    """
    row: dict = {"fold": fold_num, "h": h_val}

    for tau, preds in preds_dict.items():
        if tau == "mean":
            residuals = y_test - preds
            row["rmse"] = float(np.sqrt(np.mean(residuals ** 2)))
        else:
            errors    = y_test - preds
            pinball   = np.where(errors >= 0, tau * errors, (tau - 1) * errors)
            col_name  = f"pinball_q{int(tau * 100):02d}"
            row[col_name] = float(np.mean(pinball))

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(banco: str = BANCO) -> None:
    t0_total = time.time()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nCargando datos: {RUTA_MATRIZ}")
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])

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
    all_preds_base: list[pd.DataFrame] = []
    resultados:     list[dict]         = []

    n_horizontes = H_MAX - H_MIN + 1

    for fold in folds:
        t0_fold = time.time()
        print(f"\n{'='*60}")
        print(
            f"FOLD {fold['fold']}  "
            f"train: {fold['train_start'].date()}..{fold['train_end'].date()}  "
            f"test:  {fold['test_start'].date()}..{fold['test_end'].date()}"
        )

        fold_scaffolds: list[pd.DataFrame] = []
        n_h_ok = 0

        for h_val in range(H_MIN, H_MAX + 1):
            df_h = df[df["h"] == h_val]

            # Progress indicator every 10 horizons
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

            # Train models
            modelos = entrenar_modelos_h(X_train, y_train, HP)

            # Build predictions scaffold
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
            n_h_ok += 1

            # Metrics
            preds_for_metrics = {
                tau: m.predict(X_test) for tau, m in modelos.items()
            }
            metricas = calcular_metricas(
                preds_for_metrics, y_test.values, h_val, fold["fold"]
            )
            resultados.append(metricas)

        # Concatenate all h-scaffolds for this fold
        if fold_scaffolds:
            df_fold = pd.concat(fold_scaffolds, ignore_index=True)
            all_preds_base.append(df_fold)

        elapsed_fold = (time.time() - t0_fold) / 60
        print(
            f"  Fold {fold['fold']}: "
            f"{n_h_ok}/{n_horizontes} horizontes completados | "
            f"{elapsed_fold:.1f} min"
        )

    # ------------------------------------------------------------------
    # 5. Save predictions
    # ------------------------------------------------------------------
    DIR_MODO = DIR_OUTPUT / f"fold{'_exp' if EXPANDING else '_roll'}"
    DIR_MODO.mkdir(parents=True, exist_ok=True)
    fecha_hoy = date.today().strftime("%Y%m%d")

    if all_preds_base:
        df_all = pd.concat(all_preds_base, ignore_index=True)

        # Ensure column order matches cv3 output (fanchart compatibility)
        col_order = ["banco", "fold", "fecha_t", "fecha_th", "h", "target",
                     "q01", "q05", "q50", "q95", "q99", "mean"]
        # Add any extra columns at the end
        extra_cols = [c for c in df_all.columns if c not in col_order]
        col_order = [c for c in col_order if c in df_all.columns] + extra_cols
        df_all = df_all[col_order]

        ruta_preds = DIR_MODO / f"preds_base_{banco}_{fecha_hoy}.parquet"
        df_all.to_parquet(ruta_preds, index=False)
        print(f"\n✓ Guardado: {ruta_preds}  ({len(df_all):,} filas)")
        print(f"  Columnas: {list(df_all.columns)}")
    else:
        print("\n⚠  Sin predicciones para guardar.")
        df_all = pd.DataFrame()

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
    else:
        print("\n⚠  Sin métricas para guardar.")

    total_min = (time.time() - t0_total) / 60
    print(f"\n{'='*60}")
    print(f"Tiempo total: {total_min:.1f} min")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
