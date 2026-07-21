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

# ---------------------------------------------------------------------------
# Early stopping + Optuna
# ---------------------------------------------------------------------------
N_ESTIMATORS_MAX      = 300   # techo de árboles; early stopping lo reduce en práctica
EARLY_STOPPING_ROUNDS = 10    # parar si val loss no mejora en N rondas consecutivas

USE_OPTUNA      = True   # False → HP fijos para todos los h (más rápido, sin optuna)
OPTUNA_N_TRIALS = 25     # trials por h representativo por fold

# Grupos de h y sus representantes para Optuna (Opción C)
# Un solo h "típico" por grupo → HP se buscan ahí y se transfieren a todo el grupo
H_GRUPOS: dict = {
    "corto": (list(range(H_MIN, 21)),        10),   # h=2–20,  representante h=10
    "medio": (list(range(21, 51)),           35),   # h=21–50, representante h=35
    "largo": (list(range(51, H_MAX + 1)),    62),   # h=51–75, representante h=62
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
    Build walk-forward CV folds with an expanding training window.

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

        val_start = test_start - _offset(VENTANA_VAL_AÑOS)

        if EXPANDING:
            train_start = train_min
        else:
            train_start = val_start - _offset(VENTANA_TRAIN_AÑOS)

        # --- Purge gap between train and val ---
        # Find the index of the first business day >= val_start
        idx_val = int(np.searchsorted(all_bdays, np.datetime64(val_start, "ns")))
        train_end_idx = max(0, idx_val - PURGE_DIAS_HAB - 1)
        train_end = pd.Timestamp(all_bdays[train_end_idx])

        # --- val_end: last business day before test_start (no purge val→test) ---
        idx_test = int(np.searchsorted(all_bdays, np.datetime64(test_start, "ns")))
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


def entrenar_modelos_h(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hp: dict,
) -> dict:
    """
    Train one XGBRegressor per quantile tau plus one for mean.
    Uses early stopping on val when val is non-empty (Option A).

    Returns
    -------
    dict {tau (float | 'mean'): fitted XGBRegressor}
    """
    modelos: dict = {}
    use_es = len(X_val) > 0   # early stopping requires a non-empty val set

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
) -> dict:
    """
    Compute pinball loss per quantile tau, RMSE for mean, and empirical coverage.

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

    # Empirical coverage
    if 0.05 in preds_dict and 0.95 in preds_dict:
        row["coverage_90"] = float(
            ((y_test >= preds_dict[0.05]) & (y_test <= preds_dict[0.95])).mean()
        )
    if 0.01 in preds_dict and 0.99 in preds_dict:
        row["coverage_98"] = float(
            ((y_test >= preds_dict[0.01]) & (y_test <= preds_dict[0.99])).mean()
        )

    return row


# ---------------------------------------------------------------------------
# Feature diagnostics helpers
# ---------------------------------------------------------------------------

def _pinball(y: np.ndarray, yhat: np.ndarray, tau: float) -> float:
    err = y - yhat
    return float(np.where(err >= 0, tau * err, (tau - 1) * err).mean())


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

    for tau, model in modelos.items():
        if tau == "mean":
            continue
        base_preds = model.predict(X)
        base_loss  = _pinball(y, base_preds, tau)

        feat_deltas: dict[str, float] = {}
        for c in cols_feat:
            orig = X[c].values.copy()
            deltas = []
            for _ in range(DIAG_N_REPEATS):
                perm = rng.permutation(block_starts)
                new_col = np.concatenate([orig[s:s + bs] for s in perm])[:n]
                Xp = X.copy()
                Xp[c] = new_col
                deltas.append(_pinball(y, model.predict(Xp), tau) - base_loss)
            feat_deltas[c] = float(np.mean(deltas))

        acum = acum.add(pd.Series(feat_deltas), fill_value=0.0)
        n_tau += 1

    if n_tau:
        acum /= n_tau
    return acum


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
            explainer = _shap_lib.TreeExplainer(model.get_booster())
            sv = explainer.shap_values(X)
            s = pd.Series(np.abs(sv).mean(axis=0), index=cols_feat)
            acum = acum.add(s.fillna(0.0), fill_value=0.0)
            n_tau += 1
        except Exception as e:
            log.debug("SHAP τ=%.2f falló: %s", tau, e)

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

def optuna_tune_h(
    h_rep: int,
    df: pd.DataFrame,
    fold: dict,
    cols_feat: list[str],
) -> dict:
    """
    Busca HP óptimos en el h representativo del grupo usando Optuna + early stopping.
    Espacio de búsqueda: 7 HP de regularización/árbol; n_estimators lo decide early stopping.
    Devuelve un HP dict completo listo para entrenar.
    """
    if not _OPTUNA_OK:
        log.warning("optuna no instalado — usando HP fijos para h_rep=%d", h_rep)
        return dict(HP)

    df_h = df[df["h"] == h_rep]
    try:
        X_tr, y_tr, X_vl, y_vl, _, _, _, _ = preparar_fold_data_h(df_h, fold, cols_feat)
    except ValueError as e:
        log.warning("Optuna h_rep=%d fold=%d omitido: %s — usando HP fijos", h_rep, fold["fold"], e)
        return dict(HP)

    y_tr_arr = np.asarray(y_tr)
    y_vl_arr = np.asarray(y_vl)

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

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

    best = study.best_params
    log.info(
        "Optuna fold=%d h_rep=%d → pinball_val=%.4f  %s",
        fold["fold"], h_rep, study.best_value,
        " ".join(f"{k}={v:.3g}" for k, v in best.items()),
    )
    return {
        **best,                        # HP encontrados por Optuna
        "n_estimators": N_ESTIMATORS_MAX,
        "tree_method" : "hist",
        "max_bin"     : 64,
        "n_jobs"      : 1,
        "random_state": 42,
    }


def get_hp_for_h(h_val: int, hp_grupos: dict) -> dict:
    """Devuelve el HP dict del grupo correspondiente al horizonte h_val."""
    if h_val <= 20:
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
    """Genera figura con 4 paneles: RMSE, Pinball q50, todos los cuantiles, Coverage."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        log.warning("matplotlib no disponible — omitiendo gráficos de métricas")
        return

    folds      = sorted(df_res["fold"].unique())
    cmap_folds = plt.cm.Set1
    hs_all     = np.sort(df_res["h"].unique())

    # Paleta de colores para cuantiles
    tau_colors = {
        "pinball_q01": "#6B21A8",
        "pinball_q05": "#9333EA",
        "pinball_q50": "#2563EB",
        "pinball_q95": "#EA580C",
        "pinball_q99": "#991B1B",
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.32})
    fig.suptitle(
        f"Métricas de desempeño por horizonte — {banco}\n"
        f"cv4 DIRECT (1 modelo por h, h ∉ features)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    ax1, ax2, ax3, ax4 = axes.flat

    # ── Panel 1: RMSE por h ──────────────────────────────────────────────────
    for i, fold_num in enumerate(folds):
        sub = df_res[df_res["fold"] == fold_num].sort_values("h")
        ax1.plot(sub["h"], sub["rmse"] / 1e6, lw=1.5,
                 color=cmap_folds(i / max(len(folds), 1)),
                 alpha=0.7, label=f"Fold {fold_num}")
    mean_rmse = df_res.groupby("h")["rmse"].mean()
    ax1.plot(mean_rmse.index, mean_rmse.values / 1e6,
             color="black", lw=2.5, label="Promedio folds", zorder=5)
    ax1.set_title("RMSE por horizonte h", fontweight="bold")
    ax1.set_xlabel("Horizonte h (días hábiles)")
    ax1.set_ylabel("RMSE (MM USD)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(10))

    # ── Panel 2: Pinball q50 por h ───────────────────────────────────────────
    for i, fold_num in enumerate(folds):
        sub = df_res[df_res["fold"] == fold_num].sort_values("h")
        if "pinball_q50" in sub.columns:
            ax2.plot(sub["h"], sub["pinball_q50"] / 1e6, lw=1.5,
                     color=cmap_folds(i / max(len(folds), 1)),
                     alpha=0.7, label=f"Fold {fold_num}")
    if "pinball_q50" in df_res.columns:
        mean_pb50 = df_res.groupby("h")["pinball_q50"].mean()
        ax2.plot(mean_pb50.index, mean_pb50.values / 1e6,
                 color="black", lw=2.5, label="Promedio folds", zorder=5)
    ax2.set_title("Pinball loss Q50 (mediana) por horizonte h", fontweight="bold")
    ax2.set_xlabel("Horizonte h (días hábiles)")
    ax2.set_ylabel("Pinball loss (MM USD)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(10))

    # ── Panel 3: Pinball todos los cuantiles (promedio de folds) ─────────────
    pb_cols_present = [c for c in tau_colors if c in df_res.columns]
    for col in pb_cols_present:
        mean_pb = df_res.groupby("h")[col].mean()
        label   = col.replace("pinball_q", "Q").replace("01", "01 (1%)")\
                     .replace("05", "05 (5%)").replace("50", "50 (50%)")\
                     .replace("95", "95 (95%)").replace("99", "99 (99%)")
        ax3.plot(mean_pb.index, mean_pb.values / 1e6,
                 color=tau_colors[col], lw=2.0, label=label)
    ax3.set_title("Pinball loss por cuantil y horizonte (promedio folds)",
                  fontweight="bold")
    ax3.set_xlabel("Horizonte h (días hábiles)")
    ax3.set_ylabel("Pinball loss (MM USD)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_locator(mticker.MultipleLocator(10))

    # ── Panel 4: Cobertura empírica ──────────────────────────────────────────
    has_cov = False
    if "coverage_90" in df_res.columns:
        mean_cov90 = df_res.groupby("h")["coverage_90"].mean()
        ax4.plot(mean_cov90.index, mean_cov90.values,
                 color="#059669", lw=2.0, label="Cobertura 90% [Q05-Q95]")
        ax4.fill_between(mean_cov90.index, mean_cov90.values, 0.90,
                         where=mean_cov90.values < 0.90,
                         alpha=0.20, color="#DC2626", label="Bajo objetivo")
        ax4.fill_between(mean_cov90.index, mean_cov90.values, 0.90,
                         where=mean_cov90.values >= 0.90,
                         alpha=0.15, color="#059669", label="Sobre objetivo")
        ax4.axhline(0.90, color="#059669", lw=1.2, ls="--", alpha=0.6,
                    label="Objetivo 90%")
        has_cov = True
    if "coverage_98" in df_res.columns:
        mean_cov98 = df_res.groupby("h")["coverage_98"].mean()
        ax4.plot(mean_cov98.index, mean_cov98.values,
                 color="#7C3AED", lw=1.5, ls=":", label="Cobertura 98% [Q01-Q99]")
        ax4.axhline(0.98, color="#7C3AED", lw=1.0, ls="--", alpha=0.5,
                    label="Objetivo 98%")
        has_cov = True
    if not has_cov:
        ax4.text(0.5, 0.5, "Sin datos de cobertura",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=11)
    ax4.set_title("Cobertura empírica por horizonte h (promedio folds)",
                  fontweight="bold")
    ax4.set_xlabel("Horizonte h (días hábiles)")
    ax4.set_ylabel("Cobertura empírica")
    ax4.set_ylim(0.40, 1.05)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)
    ax4.xaxis.set_major_locator(mticker.MultipleLocator(10))

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
    resultados: list[dict] = []
    diag_rows:  list[dict] = []
    n_horizontes = H_MAX - H_MIN + 1

    # Output dir created early so per-fold parquets can be written immediately
    DIR_MODO = DIR_OUTPUT / f"fold{'_exp' if EXPANDING else '_roll'}"
    DIR_MODO.mkdir(parents=True, exist_ok=True)
    fecha_hoy = date.today().strftime("%Y%m%d")

    fold_parquet_paths: list[Path] = []

    for fold in folds:
        t0_fold = time.time()
        print(f"\n{'='*60}")
        print(
            f"FOLD {fold['fold']}  "
            f"train: {fold['train_start'].date()}..{fold['train_end'].date()}  "
            f"test:  {fold['test_start'].date()}..{fold['test_end'].date()}"
        )

        # ── Optuna: buscar HP por grupo de h (Opción C) ──────────────────────
        if USE_OPTUNA and _OPTUNA_OK:
            print(f"  Buscando HP con Optuna ({OPTUNA_N_TRIALS} trials × 3 grupos)…")
            hp_grupos: dict = {}
            for grupo, (_, h_rep) in H_GRUPOS.items():
                t_opt = time.time()
                hp_grupos[grupo] = optuna_tune_h(h_rep, df, fold, cols_feat)
                elapsed_opt = time.time() - t_opt
                hp_g = hp_grupos[grupo]
                print(
                    f"    [{grupo:5s}] h_rep={h_rep:2d}  "
                    f"depth={hp_g.get('max_depth')}  "
                    f"min_cw={hp_g.get('min_child_weight')}  "
                    f"lr={hp_g.get('learning_rate', 0):.3f}  "
                    f"α={hp_g.get('reg_alpha', 0):.2f}  "
                    f"λ={hp_g.get('reg_lambda', 1):.2f}  "
                    f"({elapsed_opt:.0f}s)"
                )
        else:
            if USE_OPTUNA and not _OPTUNA_OK:
                log.warning("USE_OPTUNA=True pero optuna no está instalado — usando HP fijos")
            hp_grupos = {g: dict(HP) for g in H_GRUPOS}

        # Write each h directly to a list; concat and flush to disk per fold
        fold_scaffolds: list[pd.DataFrame] = []
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

            hp_h = get_hp_for_h(h_val, hp_grupos)
            modelos = entrenar_modelos_h(X_train, y_train, X_val, y_val, hp_h)

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

            preds_for_metrics = {tau: m.predict(X_test) for tau, m in modelos.items()}
            resultados.append(calcular_metricas(preds_for_metrics, y_test.values, h_val, fold["fold"]))

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
            print(f"  → Guardado: {ruta_fold.name}")
            del df_fold, fold_scaffolds
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
                 "q01", "q05", "q50", "q95", "q99", "mean"]

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
            print("\nCobertura empírica 90% [Q05-Q95] media por grupo de horizonte:")
            print(df_res.groupby("h_grupo", observed=True)["coverage_90"].mean().map("{:.1%}".format))

        if "coverage_98" in df_res.columns:
            print("\nCobertura empírica 98% [Q01-Q99] media por grupo de horizonte:")
            print(df_res.groupby("h_grupo", observed=True)["coverage_98"].mean().map("{:.1%}".format))

        graficar_metricas(df_res, DIR_MODO, banco, fecha_hoy)
    else:
        print("\n⚠  Sin métricas para guardar.")

    # ------------------------------------------------------------------
    # 7. Feature diagnostics (gain / perm / SHAP per h)
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
