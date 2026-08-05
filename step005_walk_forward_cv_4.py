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
import sys
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
N_ESTIMATORS_MAX      = 500   # techo de árboles; early stopping lo reduce en práctica
                               # 500 (antes 300): con learning_rate=0.03 (mínimo del
                               # espacio Optuna) 300 árboles no alcanzan a converger
EARLY_STOPPING_ROUNDS = 30    # parar si val loss no mejora en N rondas consecutivas
                               # 30 (antes 10): la curva de pinball en VAL tiene alta
                               # varianza — 10 rondas detenía en mínimos locales

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
OPTUNA_N_TRIALS_ARCTAN = 50  # trials cuando AJUSTE_ARCTAN=True (vs 30 base)
                               # 50 (antes 40): con OPTUNA_N_ESTIMATORS=True el
                               # espacio tiene 9 HP (7 base + s_factor + n_estimators)
                               # y el diagnóstico daba p90 en el trial #36 de 40

# Cómo se decide el número de árboles en la rama arctan:
# True  → Optuna busca n_estimators. Una sola decisión compartida por los 7 taus,
#         acoplada con learning_rate en el mismo espacio (la capacidad efectiva
#         de un GBM va como η·T; con early stopping Optuna controla η pero sufre
#         T como ruido). Además promedia 7 curvas de pinball en vez de tomar 7
#         decisiones independientes sobre 121 filas de VAL → menos varianza.
# False → early stopping por modelo sobre el pinball de VAL.
# Solo aplica con AJUSTE_ARCTAN=True; la rama estándar usa el early stopping
# nativo de XGBRegressor, que sí restaura best_iteration correctamente.
OPTUNA_N_ESTIMATORS = True
N_ESTIMATORS_RANGE  = (50, 400)   # rango de búsqueda de n_estimators (escala log)

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

class _Tee:
    """
    Duplica sys.stdout a un archivo para conservar las tablas de consola.

    Solo intercepta stdout (los print). El logging va a stderr y queda fuera a
    propósito: log.debug("Fold %d, h=%d | …") se dispara una vez por (fold, h),
    unas 370 líneas por corrida, que ahogarían las ~100 líneas de tablas.

    __getattr__ delega el resto de atributos al stdout original (isatty,
    encoding, fileno…) para que cualquier librería que los consulte siga
    funcionando con normalidad.
    """

    def __init__(self, ruta: Path):
        self._orig = sys.stdout           # primero: __getattr__ depende de él
        self._f    = open(ruta, "w", encoding="utf-8")
        sys.stdout = self

    def write(self, s):
        self._orig.write(s)
        self._f.write(s)

    def flush(self):
        self._orig.flush()
        self._f.flush()

    def __getattr__(self, nombre):
        orig = self.__dict__.get("_orig")
        if orig is None:
            raise AttributeError(nombre)
        return getattr(orig, nombre)

    def cerrar(self):
        sys.stdout = self._orig
        self._f.close()


def _dir_modo() -> Path:
    """
    Carpeta de salida del modo actual, creada si no existe.

    Solo depende de constantes de configuración (DIR_OUTPUT, EXPANDING,
    AJUSTE_ARCTAN), así que puede resolverse antes de cargar datos — es lo que
    permite abrir el log de consola al inicio de run().
    """
    sfx  = "_arctan" if AJUSTE_ARCTAN else ""
    base = "exp" if EXPANDING else "roll"
    d = DIR_OUTPUT / f"fold_{base}{sfx}"
    d.mkdir(parents=True, exist_ok=True)
    return d


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

    # ── Verificación anti-leakage ────────────────────────────────────────────
    # Ningún par de particiones puede compartir filas. Se lanza AssertionError
    # (no ValueError) a propósito: los except ValueError del h-loop y de
    # optuna_tune_h saltan el horizonte silenciosamente, pero un leakage debe
    # detener el run completo — no tiene sentido gastar horas con datos filtrados.
    for (mask_a, name_a), (mask_b, name_b) in (
        ((mt, "TRAIN"), (mv,  "VAL")),
        ((mt, "TRAIN"), (mte, "TEST")),
        ((mv, "VAL"),   (mte, "TEST")),
    ):
        n_overlap = int((mask_a & mask_b).sum())
        if n_overlap:
            raise AssertionError(
                f"Leakage: {n_overlap} filas solapan {name_a} y {name_b} "
                f"(h={df_h['h'].iloc[0] if len(df_h) else '?'}) — "
                f"revisar GAP_DIAS_HAB={GAP_DIAS_HAB}"
            )

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

    NOTA: xgb.train() no revierte el modelo al mejor iteration cuando un
    callback corta — deja TODOS los árboles entrenados, incluidos los
    `rounds` posteriores al óptimo. Por eso se registra _best_iter y el
    _ArcTanBooster lo usa como iteration_range al predecir; si no, cada
    modelo arrastraría EARLY_STOPPING_ROUNDS árboles sobreajustados.
    """
    def __init__(self, rounds: int = 50):
        super().__init__()
        self.rounds     = rounds
        self._best      = float("inf")
        self._best_iter = 0
        self._since     = 0

    @property
    def best_iteration(self) -> int:
        """Epoch (0-based) donde el pinball de validación fue mínimo."""
        return self._best_iter

    def after_iteration(self, model, epoch, evals_log):
        score = None
        for metrics in evals_log.values():
            if "pinball" in metrics:
                score = metrics["pinball"][-1]
                break
        if score is None:
            return False
        if score < self._best - 1e-9:
            self._best      = score
            self._best_iter = epoch
            self._since     = 0
        else:
            self._since += 1
        return self._since >= self.rounds


class _ArcTanBooster:
    """
    Wrapper mínimo sobre xgb.Booster para exponer .predict(X_df) compatible
    con XGBRegressor — permite usar xgb.train() con objetivo custom sin
    modificar el resto del pipeline (diagnosticar_h, scaffolds, métricas).

    best_iter — epoch óptimo reportado por _PinballEarlyStopping. Se usa como
    iteration_range al predecir para descartar los árboles entrenados después
    del óptimo (xgb.train no revierte el modelo por sí solo). None cuando no
    hubo eval set: en ese caso se usan todos los árboles.
    """
    def __init__(self, booster: "xgb.Booster", best_iter: int | None = None):
        self._b = booster
        self._best_iter = best_iter

    def predict(self, X) -> np.ndarray:
        d = xgb.DMatrix(X)
        # No se pasa iteration_range=None: el default varía entre versiones de
        # XGBoost ((0,0) en 2.0, None en 2.1+). Se omite el kwarg si no aplica.
        if self._best_iter is None:
            return self._b.predict(d)
        return self._b.predict(d, iteration_range=(0, self._best_iter + 1))

    def get_booster(self) -> "xgb.Booster":
        """Compatibilidad con XGBRegressor.get_booster() — usado en gain y SHAP."""
        return self._b

    @property
    def best_iteration(self) -> int:
        if self._best_iter is not None:
            return self._best_iter
        return self._b.num_boosted_rounds() - 1


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
        # Con OPTUNA_N_ESTIMATORS el número de rondas ya viene de Optuna:
        # no hay early stopping por modelo, así que evals/custom_metric solo
        # serían coste (una predicción sobre VAL por ronda de boosting).
        n_rounds = int(hp.get("n_estimators", N_ESTIMATORS_MAX))
        usar_es  = use_es and not OPTUNA_N_ESTIMATORS

        for tau in TAUS:
            # IMPORTANTE: nueva instancia por tau — _PinballEarlyStopping tiene
            # estado interno (_best, _since). Reutilizar el mismo objeto entre taus
            # causa early stopping prematuro: el _best de Q01 (~30 MM) es mucho
            # menor que el pinball inicial de Q05 (~160 MM), disparando stop en
            # EARLY_STOPPING_ROUNDS iteraciones aunque Q05 aún no haya convergido.
            cb  = _PinballEarlyStopping(rounds=EARLY_STOPPING_ROUNDS) if usar_es else None
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=n_rounds,
                obj=_make_quantile_objective(tau, s, std_y),
                custom_metric=_make_pinball_metric(tau) if usar_es else None,
                evals=evals if usar_es else [],
                callbacks=[cb] if cb is not None else [],
                verbose_eval=False,
            )
            # best_iter recorta los árboles posteriores al óptimo de VAL
            modelos[tau] = _ArcTanBooster(
                booster, best_iter=cb.best_iteration if cb is not None else None
            )

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
    # s_factor solo alimenta _make_quantile_objective; XGBRegressor no lo
    # reconoce y lo reenviaría al backend C++ como parámetro desconocido.
    # Se filtra únicamente esa clave: n_estimators / random_state / n_jobs sí
    # son válidos para la API sklearn (a diferencia de _hp_para_xgb_train).
    _hp_mean = {k: v for k, v in hp.items() if k != "s_factor"}
    m_mean = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if use_es else None,
        **_hp_mean,
    )
    if use_es:
        m_mean.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        m_mean.fit(X_train, y_train, verbose=False)
    modelos["mean"] = m_mean

    return modelos


# ---------------------------------------------------------------------------
# Monotonía de los cuantiles
# ---------------------------------------------------------------------------

def _reordenar_cuantiles(preds: dict) -> tuple[dict, int]:
    """
    Rearrangement de Chernozhukov, Fernández-Val & Galichon (2010): ordena los
    cuantiles predichos fila a fila para garantizar Q01 ≤ Q05 ≤ … ≤ Q99.

    Se entrena un booster independiente por τ, con sus propios splits, así que
    nada impone monotonía entre modelos y aparecen cruces (el paper 2406.02293
    documenta esto como la desventaja de usar modelos separados frente a un
    único modelo composite con hojas multi-output). El operador de
    reordenamiento tiene la garantía de que la curva resultante está débilmente
    más cerca de la curva verdadera en cualquier norma Lp: nunca empeora.

    Se prefiere ordenar sobre isotonic regression porque conserva los valores
    predichos en vez de promediar los violadores, y es una operación vectorizada.

    Returns
    -------
    (dict reordenado, nº de filas que tenían al menos un cruce)
    """
    taus = sorted(t for t in preds if t != "mean")
    if len(taus) < 2:
        return preds, 0
    M = np.column_stack([np.asarray(preds[t], dtype=float) for t in taus])
    n_cruces = int((np.diff(M, axis=1) < 0).any(axis=1).sum())
    M_ord = np.sort(M, axis=1)
    out: dict = {t: M_ord[:, i] for i, t in enumerate(taus)}
    if "mean" in preds:
        out["mean"] = np.asarray(preds["mean"], dtype=float)
    return out, n_cruces


# ---------------------------------------------------------------------------
# CRPS por reconstrucción de la función cuantil
# ---------------------------------------------------------------------------
# np.trapz sobre los 7 pinballs subestimaba el CRPS ~9.5% de forma sistemática:
# el sesgo no venía de las colas (solo ~0.1%) sino de los huecos de la grilla
# de τ (0.05→0.40 y 0.60→0.95), donde la regla trapezoidal corta la curvatura
# de la curva de pinball. La solución reconstruye Q(τ) en una grilla densa
# antes de integrar; el error cae a <0.2% sin reentrenar nada.
# ---------------------------------------------------------------------------

CRPS_N_GRID = 201   # puntos de τ para integrar el CRPS (converge ya en ~101)

# np.trapz fue eliminado en numpy 2.0 (renombrado a np.trapezoid)
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _pchip_filas(xi: np.ndarray, Y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """
    PCHIP (Fritsch-Carlson) vectorizado por filas — interpolación cúbica de
    Hermite que preserva monotonía, así que no puede introducir cruces de
    cuantiles al interpolar.

    xi : (k,)   nodos compartidos por todas las filas (los TAUS)
    Y  : (n,k)  valores por fila (los cuantiles predichos)
    xq : (m,)   puntos de consulta
    Returns (n,m)
    """
    h = np.diff(xi)
    d = np.diff(Y, axis=1) / h
    m = np.zeros_like(Y)
    m[:, 0], m[:, -1] = d[:, 0], d[:, -1]
    for k in range(1, Y.shape[1] - 1):
        w1, w2  = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
        dk1, dk = d[:, k - 1], d[:, k]
        ok  = (dk1 * dk) > 0        # mismo signo → media armónica ponderada
        den = np.where(ok, w1 / np.where(ok, dk1, 1) + w2 / np.where(ok, dk, 1), 1.0)
        m[:, k] = np.where(ok, (w1 + w2) / den, 0.0)   # extremo local → pendiente 0

    j = np.clip(np.searchsorted(xi, xq) - 1, 0, len(h) - 1)
    H = h[j]
    t = (xq - xi[j]) / H
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return (h00 * Y[:, j] + h10 * H * m[:, j]
            + h01 * Y[:, j + 1] + h11 * H * m[:, j + 1])


def _crps_interp(preds: dict, y_true: np.ndarray,
                 n_grid: int = CRPS_N_GRID) -> float | None:
    """
    CRPS = 2·∫₀¹ pinball_τ dτ, evaluado sobre Q(τ) reconstruida con PCHIP.

    Fuera de [τ_min, τ_max] se extrapola plano: la masa es 1% por lado y la
    extrapolación lineal es inestable con colas pesadas (probado con t(gl=2)).

    Devuelve None si hay menos de 3 cuantiles (PCHIP necesita ≥3 nodos).
    """
    taus = sorted(t for t in preds if t != "mean")
    if len(taus) < 3:
        return None
    Q = np.column_stack([np.asarray(preds[t], dtype=float) for t in taus])
    Q = np.maximum.accumulate(Q, axis=1)   # anti-cruce por fila antes de interpolar
    td = np.linspace(1e-6, 1.0 - 1e-6, n_grid)
    Qd = _pchip_filas(np.asarray(taus, dtype=float), Q,
                      np.clip(td, taus[0], taus[-1]))
    e  = np.asarray(y_true, dtype=float)[:, None] - Qd
    pb = np.where(e >= 0, td * e, (td - 1) * e)
    return float(2.0 * _TRAPZ(pb.mean(axis=0), td))


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
        # CRPS VAL — permite comparar VAL vs TEST para detectar overfitting
        # distribucional (val_crps << crps → overfit)
        _crps_v = _crps_interp(preds_val_dict, y_val)
        if _crps_v is not None:
            row["val_crps"] = _crps_v

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

    # ── CRPS = 2 * ∫₀¹ pinball_τ dτ sobre Q(τ) reconstruida (ver _crps_interp) ──
    _crps = _crps_interp(preds_dict, y_test)
    if _crps is not None:
        row["crps"] = _crps

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


def _tau_label(tau) -> str:
    """Etiqueta canónica de un modelo: 'q01'…'q99' o 'mean'."""
    return "mean" if tau == "mean" else f"q{int(tau * 100):02d}"


def _orden_taus(modelos: dict) -> list:
    """Cuantiles en orden ascendente y 'mean' al final."""
    qs = sorted(t for t in modelos if t != "mean")
    return qs + (["mean"] if "mean" in modelos else [])


def _diag_gain_h(modelos: dict, cols_feat: list[str]) -> dict:
    """
    Participación de cada feature en la ganancia total del modelo, por cuantil.

    Dos decisiones importantes:

    1. Se usa `total_gain` y no `gain`. XGBoost define 'gain' como la ganancia
       PROMEDIO de los splits que usan el feature, así que un feature usado una
       sola vez en un split afortunado obtiene un valor enorme mientras que uno
       usado en treinta splits de calidad media obtiene un valor moderado.
       'total_gain' mide la contribución acumulada, que es lo que interesa.

    2. Se normaliza a participación (share del total). El gain crudo va como
       n·_scale², y _scale = σ·π(sf²+1)²/(2sf³) depende de std_y(h) y del
       s_factor del grupo: entre grupos de un mismo fold hay hasta 5x de
       diferencia sin que los datos cambien. Al dividir por la suma, ese factor
       —común a todos los features del mismo modelo— se cancela y los valores
       pasan a ser comparables entre horizontes, folds y cuantiles.

    Returns {tau: Series de participaciones (suman 1)}
    """
    out: dict = {}
    for tau, model in modelos.items():
        imp = model.get_booster().get_score(importance_type="total_gain")
        s = pd.Series({f: float(imp.get(f, 0.0)) for f in cols_feat}, dtype=float)
        total = s.sum()
        out[tau] = s / total if total > 0 else s
    return out


def _diag_perm_h(
    modelos: dict,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cols_feat: list[str],
) -> dict:
    """
    Block-permutation importance en VAL (OOS), por cuantil.

    Devuelve el Δloss RELATIVO a la pérdida base (Δloss / loss_base), que es
    adimensional y por tanto comparable entre horizontes, folds y cuantiles.
    No se normaliza a participación como gain y shap porque estos Δ pueden ser
    NEGATIVOS —permutar un feature a veces mejora el loss por azar— y con
    valores de signo mixto la suma puede acercarse a cero y las participaciones
    explotan.

    Para el modelo 'mean' la pérdida es el MSE, no el pinball: sus valores son
    interpretables pero NO comparables con los de las filas de cuantiles.

    Returns {tau: Series de Δloss relativo (con signo)}
    """
    X = X_val[cols_feat].reset_index(drop=True).copy()
    y = np.asarray(y_val)
    n = len(X)
    bs = max(2, min(DIAG_BLOCK_SIZE, n // 3))
    block_starts = np.arange(0, n, bs)
    rng = np.random.default_rng(42)

    out: dict = {}

    # Pares sin/cos → se permutar conjuntamente con el mismo shuffle
    pairs     = _sincos_pairs(cols_feat)
    paired_cs = {c for (sc, cc) in pairs.values() for c in (sc, cc)}

    for tau, model in modelos.items():
        # 'mean' se entrena con reg:squarederror → su pérdida natural es el MSE
        if tau == "mean":
            _loss = lambda yy, pp: float(np.mean((yy - pp) ** 2))
        else:
            _loss = lambda yy, pp, _t=tau: _pinball(yy, pp, _t)

        base_preds = model.predict(X)
        base_loss  = _loss(y, base_preds)

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
                deltas.append(_loss(y, model.predict(Xp)) - base_loss)
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
                deltas.append(_loss(y, model.predict(Xp)) - base_loss)
            feat_deltas[sin_c] = float(np.mean(deltas))  # Δ joint completo
            feat_deltas[cos_c] = 0.0                     # placeholder; suma → Δ joint

        # Δ relativo a la pérdida base → adimensional
        den = abs(base_loss) if base_loss != 0 else 1.0
        out[tau] = pd.Series(feat_deltas).reindex(cols_feat).fillna(0.0) / den

    return out


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
) -> dict:
    """
    SHAP |mean| en VAL (OOS), por cuantil, normalizado a participación.

    |shap| está en unidades del target; dividir por la suma lo vuelve
    adimensional y comparable entre horizontes, folds y cuantiles.

    Returns {tau: Series de participaciones (suman 1)}
    """
    if not _SHAP_OK:
        return {}

    X = X_val[cols_feat].reset_index(drop=True)
    if DIAG_SHAP_MAX_SAMPLES and len(X) > DIAG_SHAP_MAX_SAMPLES:
        X = X.sample(DIAG_SHAP_MAX_SAMPLES, random_state=42)

    out: dict = {}

    for tau, model in modelos.items():
        try:
            explainer = _shap_lib.TreeExplainer(_shap_compat_booster(model))
        except Exception as e:
            log.debug("SHAP %s falló [init]: %s", _tau_label(tau), e)
            continue
        try:
            # check_additivity=False evita la llamada interna a predict(ntree_limit=N)
            # que SHAP < 0.43 usa para validar que sum(SHAP) == predicción del modelo
            sv = explainer.shap_values(X, check_additivity=False)
            s = pd.Series(np.abs(sv).mean(axis=0), index=cols_feat).fillna(0.0)
            total = s.sum()
            out[tau] = s / total if total > 0 else s
        except Exception as e:
            log.debug("SHAP %s falló [values]: %s", _tau_label(tau), e)

    return out


def diagnosticar_h(
    modelos: dict,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cols_feat: list[str],
    fold_num: int,
    h_val: int,
) -> list[dict]:
    """
    Una fila por modelo (7 cuantiles + 'mean'):
        {fold, h, tau, gain_<feat>, perm_<feat>, shap_<feat>}

    Las tres señales ya se calculaban por cuantil y se promediaban al final;
    conservar el desglose no cuesta cómputo adicional. Es información necesaria
    para saber si las colas se apoyan en features distintos al centro.
    """
    gain = _diag_gain_h(modelos, cols_feat)
    perm = _diag_perm_h(modelos, X_val, y_val, cols_feat)
    shp  = _diag_shap_h(modelos, X_val, cols_feat)

    _vacio = pd.Series(dtype=float)
    filas: list[dict] = []
    for tau in _orden_taus(modelos):
        g = gain.get(tau, _vacio)
        p = perm.get(tau, _vacio)
        s = shp.get(tau,  _vacio)
        row: dict = {"fold": fold_num, "h": h_val, "tau": _tau_label(tau)}
        for f in cols_feat:
            row[f"gain_{f}"] = float(g.get(f, 0.0))
            row[f"perm_{f}"] = float(p.get(f, 0.0))
            row[f"shap_{f}"] = float(s.get(f, np.nan))
        filas.append(row)
    return filas


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
    _n_trials = OPTUNA_N_TRIALS_ARCTAN if AJUSTE_ARCTAN else OPTUNA_N_TRIALS
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
        f"Curva 'mejor hasta trial t' · {_n_trials} trials · "
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
        ax.axvline(_n_trials, color="#DC2626", lw=1.2, ls="--", alpha=0.6,
                   label=f"N_TRIALS={_n_trials}")
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
        "s_factor"        : "s_factor (Arctan)",
        "n_estimators"    : "n_estimators (árboles)",
    }
    if AJUSTE_ARCTAN:
        hp_names.append("s_factor")
        if OPTUNA_N_ESTIMATORS:
            hp_names.append("n_estimators")

    n_hp  = len(hp_names)
    ncols = 4
    nrows = (n_hp + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols,
                                figsize=(ncols * 4.2, nrows * 3.2),
                                gridspec_kw={"hspace": 0.55, "wspace": 0.38})
    # list(...) y no axes2.flat: flat es un iterador y el zip() de abajo lo
    # consumiría, dejando vacío el slice que oculta los paneles sobrantes
    axes2_flat = list(axes2.flat) if nrows > 1 else list(axes2)

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
        if hp == "s_factor":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
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
                 "s_factor", "n_estimators", "best_pinball_val"]
    df_show = df_hp[[c for c in col_print if c in df_hp.columns]].copy()
    df_show["learning_rate"]   = df_show["learning_rate"].map("{:.4f}".format)
    df_show["reg_alpha"]       = df_show["reg_alpha"].map("{:.3f}".format)
    df_show["reg_lambda"]      = df_show["reg_lambda"].map("{:.3f}".format)
    df_show["subsample"]       = df_show["subsample"].map("{:.3f}".format)
    df_show["colsample_bytree"]= df_show["colsample_bytree"].map("{:.3f}".format)
    if "s_factor" in df_show.columns:
        df_show["s_factor"] = df_show["s_factor"].map(lambda x: f"{x:.4f}" if x is not None else "—")
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
            f"\nDiagnóstico OPTUNA_N_TRIALS={_n_trials}:"
            f"  última mejora mediana en trial #{p50},"
            f"  p90 en trial #{p90}."
        )
        if p90 >= _n_trials - 2:
            print(
              "  ⚠  La convergencia llega hasta el final — considera aumentar OPTUNA_N_TRIALS.")
        else:
            print(
              f"  ✓  Converge bien antes del límite ({p90} < {_n_trials}).")

    print(f"[OK] HP report guardado en: {dir_modo}")


def guardar_diag_y_plots(
    diag_rows: list[dict],
    cols_feat: list[str],
    dir_modo: Path,
    banco: str,
    fecha_hoy: str,
) -> None:
    """
    Guarda el CSV de diagnóstico (fold × h × τ) y los heatmaps de importancia.

    Las tres señales están normalizadas de forma que sus celdas son comparables
    entre horizontes, folds y cuantiles (ver _diag_gain_h / _diag_perm_h /
    _diag_shap_h), así que NO se re-normaliza por fila: el color es el valor
    directo y el heatmap se lee tanto por fila ("¿en qué h importa este
    feature?") como por columna ("¿qué features pesan en este h?").

    Salidas, todas en la subcarpeta diag_por_fold_tau/:
      - <señal>_fold<NN>_<tau>.png   individuales, uno por (fold, τ, señal)
      - panel_<señal>_<tau>.png      resumen: los folds lado a lado
    """
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
    log.info("CSV diagnóstico: %s  (%d filas)", ruta_csv.name, len(df_d))

    if "tau" not in df_d.columns:
        log.warning("diag_rows sin columna 'tau' — omitiendo heatmaps")
        return

    dir_diag = dir_modo / "diag_por_fold_tau"
    dir_diag.mkdir(parents=True, exist_ok=True)

    folds = sorted(df_d["fold"].unique())
    taus  = [t for t in [f"q{int(x * 100):02d}" for x in TAUS] + ["mean"]
             if t in set(df_d["tau"])]

    SENALES = [
        # (prefijo, etiqueta, cmap, divergente)
        ("gain", "Gain (TRAIN) · participación del total",        "YlOrRd", False),
        ("perm", "Block-Perm (VAL, OOS) · Δloss relativo",        "RdBu_r", True),
        ("shap", "SHAP |mean| (VAL, OOS) · participación total",  "YlOrRd", False),
    ]

    def _pivot(sub: pd.DataFrame, senal: str, feat_cols: list[str],
               rename: dict) -> pd.DataFrame:
        """(features × h) para un subconjunto ya filtrado por fold y τ."""
        pv = (sub[["h"] + feat_cols].rename(columns=rename)
              .groupby("h").mean().T)
        return _consolidar_sincos_pivot(pv, _sincos_pairs(list(pv.index)))

    n_generados = 0

    for senal, etiqueta, cmap, divergente in SENALES:
        feat_cols = [c for c in df_d.columns if c.startswith(f"{senal}_")]
        if not feat_cols:
            continue
        rename = {c: c[len(senal) + 1:] for c in feat_cols}

        # ── Orden de filas y escala de color: FIJOS para las tres decenas de
        # figuras de esta señal. Sin esto, cada plot elegiría su propio top-25
        # y su propio máximo, y comparar folds o cuantiles sería imposible.
        pv_global = _pivot(df_d, senal, feat_cols, rename)
        orden = (pv_global.abs().mean(axis=1)
                 .sort_values(ascending=False).index.tolist())
        top_n = min(25, len(orden))
        orden = orden[:top_n]

        vals = pv_global.loc[orden].values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if divergente:
            lim  = float(np.percentile(np.abs(vals), 99)) or 1.0
            vmin, vmax = -lim, lim
        else:
            vmin = 0.0
            vmax = float(np.percentile(vals, 99)) or 1.0

        # Pre-calcula los pivots (fold, τ) una sola vez: se usan en la figura
        # individual y otra vez en el panel resumen.
        pivots: dict = {}
        for fo in folds:
            for ta in taus:
                sub = df_d[(df_d["fold"] == fo) & (df_d["tau"] == ta)]
                if sub.empty:
                    continue
                pivots[(fo, ta)] = _pivot(sub, senal, feat_cols, rename).reindex(orden)

        def _dibujar(ax, pv: pd.DataFrame, titulo: str, con_yticks: bool):
            hs = pv.columns.tolist()
            im = ax.imshow(pv.values, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_yticks(range(len(pv)))
            ax.set_yticklabels(pv.index.tolist() if con_yticks else [], fontsize=7)
            xt = [i for i, h in enumerate(hs) if h % 10 == 0]
            ax.set_xticks(xt)
            ax.set_xticklabels([str(hs[i]) for i in xt], fontsize=8)
            ax.set_xlabel("Horizonte h (días hábiles)", fontsize=9)
            ax.set_title(titulo, fontsize=10, fontweight="bold")
            return im

        # ── Figuras individuales: una por (fold, τ) ──────────────────────────
        for (fo, ta), pv in pivots.items():
            fig, ax = plt.subplots(figsize=(max(10, len(pv.columns) * 0.18),
                                            max(6, top_n * 0.42)))
            im = _dibujar(
                ax, pv,
                f"{etiqueta}\n{banco} · Fold {fo} · {ta} · top {top_n} features",
                con_yticks=True,
            )
            plt.colorbar(im, ax=ax, label=etiqueta.split(" · ")[-1])
            plt.tight_layout()
            ruta = dir_diag / f"{senal}_fold{fo:02d}_{ta}.png"
            fig.savefig(ruta, dpi=140, bbox_inches="tight")
            plt.close(fig)
            n_generados += 1

        # ── Panel resumen: los folds lado a lado, un archivo por τ ───────────
        for ta in taus:
            pvs = [(fo, pivots[(fo, ta)]) for fo in folds if (fo, ta) in pivots]
            if not pvs:
                continue
            fig, axes = plt.subplots(
                1, len(pvs),
                figsize=(max(6, len(pvs) * 5.2), max(6, top_n * 0.42)),
                gridspec_kw={"wspace": 0.08},
            )
            axes = np.atleast_1d(axes)
            for j, (fo, pv) in enumerate(pvs):
                im = _dibujar(axes[j], pv, f"Fold {fo}", con_yticks=(j == 0))
            fig.suptitle(
                f"{etiqueta} — {banco} · {ta}\n"
                f"Misma escala de color y mismo orden de filas en todos los paneles",
                fontsize=11, fontweight="bold", y=1.01,
            )
            fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.01,
                         label=etiqueta.split(" · ")[-1])
            ruta = dir_diag / f"panel_{senal}_{ta}.png"
            fig.savefig(ruta, dpi=140, bbox_inches="tight")
            plt.close(fig)
            n_generados += 1

    log.info("Heatmaps de diagnóstico: %d figuras en %s",
             n_generados, dir_diag.name)
    print(f"[OK] Diagnóstico features: {len(df_d):,} filas (fold × h × τ) | "
          f"{n_generados} figuras en {dir_diag}")



def _reg(tablas: dict, nombre: str, obj):
    """
    Registra una tabla para el Excel de resumen y la devuelve, de modo que el
    sitio de impresión pueda seguir siendo `print(_reg(...))`.

    Se guarda la versión NUMÉRICA, no la formateada: en consola conviene
    "91.5%" pero en Excel conviene 0.915, que se puede ordenar y graficar.
    """
    tablas[nombre] = (obj.to_frame(name=nombre)
                      if isinstance(obj, pd.Series) else obj.copy())
    return obj


def guardar_tablas_excel(
    tablas: dict,
    dir_modo: Path,
    banco: str,
    fecha_hoy: str,
) -> None:
    """
    Un Excel con una hoja por tabla de consola.

    Complementa a consola_*.txt: el .txt conserva el formato exacto de pantalla,
    este archivo conserva los valores numéricos para analizarlos.

    Si openpyxl no está disponible o el archivo está abierto en Excel, cae a
    CSVs sueltos en vez de romper la corrida — mismo criterio que
    guardar_hp_report y guardar_diag_y_plots.
    """
    if not tablas:
        return

    ruta = dir_modo / f"tablas_resumen_{banco}_{fecha_hoy}.xlsx"
    try:
        with pd.ExcelWriter(ruta, engine="openpyxl") as xl:
            for nombre, df in tablas.items():
                df.to_excel(xl, sheet_name=nombre[:31])   # Excel: 31 chars máx
    except Exception as e:
        log.warning("No se pudo escribir %s (%s) — se guardan CSVs sueltos",
                    ruta.name, e)
        for nombre, df in tablas.items():
            df.to_csv(dir_modo / f"tabla_{nombre}_{banco}_{fecha_hoy}.csv")
        print(f"[OK] Tablas de consola: {len(tablas)} CSVs en {dir_modo}")
        return

    log.info("Tablas de consola: %s (%d hojas)", ruta.name, len(tablas))
    print(f"[OK] Tablas de consola: {ruta.name} ({len(tablas)} hojas)")


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
    grupo_idx: int = 0,
) -> tuple:
    """
    Busca HP óptimos en el h representativo del grupo usando Optuna.
    Espacio de búsqueda: 7 HP de regularización/árbol, más s_factor cuando
    AJUSTE_ARCTAN=True y n_estimators cuando además OPTUNA_N_ESTIMATORS=True.

    Sobre n_estimators: con early stopping por modelo, Optuna elige learning_rate
    sin controlar el número de árboles, pese a que la capacidad efectiva de un GBM
    va como η·T — queda una dirección degenerada en el espacio de búsqueda. Además
    se toman 7 decisiones de parada independientes sobre las mismas ~121 filas de
    VAL, cada una en el mínimo de una curva ruidosa (sesgo del mínimo empírico).
    Buscando n_estimators se explora el plano (η, T) explícitamente y la decisión
    pasa a apoyarse en el promedio de las 7 curvas de pinball.

    Parameters
    ----------
    prev_best_params : dict | None
        HP óptimos del fold anterior para este mismo grupo.
        Si OPTUNA_WARM_START=True y prev_best_params no es None, se inyecta como
        trial 0 (study.enqueue_trial) para que el TPE parta de una región prometedora.
        Si es None o OPTUNA_WARM_START=False, comportamiento original (exploración libre).
    grupo_idx : int
        Índice del grupo de h (0..len(H_GRUPOS)-1). Entra en la semilla del TPE
        para que cada grupo explore una secuencia distinta: con una semilla común
        los ~10 trials de la fase aleatoria eran idénticos en los 4 grupos y dos
        de ellos podían converger al mismo punto exacto del espacio.

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
            # Rango desde 0: el paper 2406.02293 (§Modeling choices, punto 2)
            # recomienda min_child_weight=0 con la pérdida arctan, porque el
            # umbral actúa sobre la SUMA DE HESSIANOS y aquí el hessiano no es
            # constante. Con hess=1 en u=σ y 0.03 en u=3σ, un valor de 20 exige
            # ~20 observaciones centrales pero ~700 de cola: los splits que
            # aislarían las colas quedan bloqueados. Se amplía el rango en vez
            # de fijarlo en 0 para que Optuna decida con la opción disponible.
            "min_child_weight": trial.suggest_int(  "min_child_weight",   0,   20),
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
            # s_factor siempre buscado por Optuna en [0.01, 1.0] log
            _sf     = trial.suggest_float("s_factor", 0.01, 1.0, log=True)
            _s      = _sf * _std_y_opt
            if OPTUNA_N_ESTIMATORS:
                _n_est = trial.suggest_int("n_estimators", *N_ESTIMATORS_RANGE, log=True)
                hp_trial["n_estimators"] = _n_est   # viaja al modelo final
            else:
                _n_est = N_ESTIMATORS_MAX
            _params = _hp_para_xgb_train(hp_trial)
            _dtrain = xgb.DMatrix(X_tr, label=y_tr_arr)
            _dval   = xgb.DMatrix(X_vl, label=y_vl_arr)
            for tau in TAUS:
                if OPTUNA_N_ESTIMATORS:
                    # n fijo por trial → sin callbacks/evals/custom_metric
                    booster = xgb.train(
                        _params,
                        _dtrain,
                        num_boost_round=_n_est,
                        obj=_make_quantile_objective(tau, _s, _std_y_opt),
                        verbose_eval=False,
                    )
                    preds_vl = booster.predict(_dval)
                else:
                    # Early stopping por modelo: hay que recortar al óptimo, si no
                    # Optuna evaluaría modelos sobreajustados y elegiría HP erróneos
                    _cb = _PinballEarlyStopping(rounds=EARLY_STOPPING_ROUNDS)
                    booster = xgb.train(
                        _params,
                        _dtrain,
                        num_boost_round=N_ESTIMATORS_MAX,
                        obj=_make_quantile_objective(tau, _s, _std_y_opt),
                        custom_metric=_make_pinball_metric(tau),
                        evals=[(_dval, "val")],
                        callbacks=[_cb],
                        verbose_eval=False,
                    )
                    preds_vl = booster.predict(
                        _dval, iteration_range=(0, _cb.best_iteration + 1)
                    )
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
    # Semilla distinta por (fold, grupo): el *10 deja hueco para hasta 10 grupos
    # sin colisiones. Antes era 42+fold, igual para los 4 grupos → los ~10 trials
    # de la fase aleatoria del TPE eran idénticos entre grupos y la exploración
    # efectiva era menor que la nominal.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold["fold"] * 10 + grupo_idx),
    )

    # Warm start: inyectar los HP óptimos del fold anterior como trial 0.
    # El TPE los evalúa primero y construye su modelo probabilístico desde
    # esa región, convergiendo más rápido con los mismos N trials.
    if OPTUNA_WARM_START and prev_best_params is not None:
        _search_keys = {"max_depth", "min_child_weight", "reg_alpha", "reg_lambda",
                        "learning_rate", "subsample", "colsample_bytree"}
        if AJUSTE_ARCTAN:
            _search_keys = _search_keys | {"s_factor"}   # s_factor siempre buscado
            if OPTUNA_N_ESTIMATORS:
                _search_keys = _search_keys | {"n_estimators"}
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
        "tree_method" : "hist",
        "max_bin"     : 64,
        "n_jobs"      : 1,
        "random_state": 42,
    }
    # Si OPTUNA_N_ESTIMATORS=True, `best` ya trae n_estimators; si no, se usa el techo
    hp_dict.setdefault("n_estimators", N_ESTIMATORS_MAX)
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
    """
    Envoltura fina: abre el log de consola y delega en _run_interno.

    Se separa en dos funciones para no reindentar el cuerpo entero dentro de un
    try/finally. El finally garantiza que sys.stdout quede restaurado aunque la
    corrida falle a mitad.
    """
    try:
        _tee = _Tee(_dir_modo() /
                    f"consola_{banco}_{date.today().strftime('%Y%m%d')}.txt")
    except OSError as e:
        log.warning("No se pudo abrir el log de consola (%s) — se continúa", e)
        _run_interno(banco)
        return

    try:
        _run_interno(banco)
    finally:
        _tee.cerrar()


def _run_interno(banco: str = BANCO) -> None:
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
    cruces_rows: list[dict] = []      # cruces de cuantiles por fold
    _tablas: dict = {}                # tablas de consola → Excel de resumen
    n_horizontes = H_MAX - H_MIN + 1

    # Warm start: guarda los HP óptimos del fold anterior por grupo.
    # Se pasan a optuna_tune_h() y se actualizan al final de cada fold.
    prev_hp_por_grupo: dict[str, dict | None] = {g: None for g in H_GRUPOS}

    # Output dir created early so per-fold parquets can be written immediately.
    # El sufijo _arctan separa los resultados del objetivo suavizado; run() ya
    # resolvió esta misma ruta para abrir el log de consola.
    DIR_MODO  = _dir_modo()
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
            _n_trials_fold = OPTUNA_N_TRIALS_ARCTAN if AJUSTE_ARCTAN else OPTUNA_N_TRIALS
            print(f"  Buscando HP con Optuna ({_n_trials_fold} trials × 4 grupos{ws_tag})…")
            hp_grupos: dict = {}
            for _gi, (grupo, (_, h_rep)) in enumerate(H_GRUPOS.items()):
                t_opt = time.time()
                hp_grupos[grupo], hp_meta = optuna_tune_h(
                    h_rep, df, fold, cols_feat,
                    prev_best_params=prev_hp_por_grupo[grupo],
                    grupo_idx=_gi,
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
                    "s_factor"        : hp_g.get("s_factor"),
                    "n_estimators"    : hp_g.get("n_estimators"),
                })
        else:
            if USE_OPTUNA and not _OPTUNA_OK:
                log.warning("USE_OPTUNA=True pero optuna no está instalado — usando HP fijos")
            hp_grupos = {g: dict(HP) for g in H_GRUPOS}

        # Write each h directly to a list; concat and flush to disk per fold
        fold_scaffolds:     list[pd.DataFrame] = []
        fold_val_scaffolds: list[pd.DataFrame] = []   # para CQR calibración
        n_h_ok = 0
        n_cruces_test = n_filas_test = 0      # diagnóstico de cruces de cuantiles
        n_cruces_val  = n_filas_val  = 0

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

            # Predicciones una sola vez por conjunto. Se reordenan para eliminar
            # cruces de cuantiles y el mismo dict alimenta el scaffold y las
            # métricas, así que lo que se guarda y lo que se mide coinciden.
            _preds_test, _nx_t = _reordenar_cuantiles(
                {tau: m.predict(X_test) for tau, m in modelos.items()}
            )
            n_cruces_test += _nx_t
            n_filas_test  += len(y_test)

            _scaffold = pd.DataFrame({
                "banco"   : banco,
                "fold"    : fold["fold"],
                "fecha_t" : pd.DatetimeIndex(fechas_t_test),
                "fecha_th": pd.DatetimeIndex(fecha_th_test),
                "h"       : h_val,
                "target"  : y_test.values,
            })
            for tau, _p in _preds_test.items():
                col = "mean" if tau == "mean" else f"q{int(tau * 100):02d}"
                _scaffold[col] = _p

            fold_scaffolds.append(_scaffold)

            # ── VAL scaffold (para CQR en step006) ───────────────────────────
            _preds_val = None
            if len(X_val) > 0:
                _preds_val, _nx_v = _reordenar_cuantiles(
                    {tau: m.predict(X_val) for tau, m in modelos.items()}
                )
                n_cruces_val += _nx_v
                n_filas_val  += len(y_val)

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
                for tau, _p in _preds_val.items():
                    col = "mean" if tau == "mean" else f"q{int(tau * 100):02d}"
                    _val_scaffold[col] = _p
                fold_val_scaffolds.append(_val_scaffold)

            n_h_ok += 1

            _row_met = calcular_metricas(
                _preds_test, y_test.values, h_val, fold["fold"],
                _preds_val, y_val.values if len(X_val) > 0 else None,
            )
            # Árboles efectivos tras early stopping — diagnostica si
            # N_ESTIMATORS_MAX es holgado y si el recorte por best_iter opera
            for tau, model in modelos.items():
                _bi = getattr(model, "best_iteration", None)
                if _bi is not None:
                    _c = "mean" if tau == "mean" else f"q{int(tau * 100):02d}"
                    _row_met[f"n_trees_{_c}"] = int(_bi) + 1
            resultados.append(_row_met)

            if DIAG_FEATURES:
                # extend, no append: devuelve una fila por modelo (7 τ + mean)
                diag_rows.extend(
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
        # Cruces de cuantiles: mide cuánto cuesta entrenar un booster por τ en
        # vez del modelo composite multi-output que propone el paper 2406.02293
        if n_filas_test:
            _msg = (f"    Cruces de cuantiles corregidos: TEST "
                    f"{n_cruces_test:,}/{n_filas_test:,} "
                    f"({100 * n_cruces_test / n_filas_test:.1f}%)")
            if n_filas_val:
                _msg += (f" | VAL {n_cruces_val:,}/{n_filas_val:,} "
                         f"({100 * n_cruces_val / n_filas_val:.1f}%)")
            print(_msg)
            cruces_rows.append({
                "fold"          : fold["fold"],
                "cruces_test"   : n_cruces_test,
                "filas_test"    : n_filas_test,
                "pct_test"      : round(100 * n_cruces_test / n_filas_test, 2),
                "cruces_val"    : n_cruces_val,
                "filas_val"     : n_filas_val,
                "pct_val"       : (round(100 * n_cruces_val / n_filas_val, 2)
                                   if n_filas_val else float("nan")),
            })

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

        # _reg() registra la versión numérica para el Excel; la consola conserva
        # su formato. Donde el print usa porcentajes se separa en dos pasos.
        print("\nRMSE medio por grupo de horizonte:")
        print(_reg(_tablas, "rmse_por_grupo",
                   df_res.groupby("h_grupo", observed=True)["rmse"].mean().round(0)))

        # Also print pinball for q50
        if "pinball_q50" in df_res.columns:
            print("\nPinball q50 medio por grupo de horizonte:")
            print(_reg(_tablas, "pinball_q50_por_grupo",
                       df_res.groupby("h_grupo", observed=True)["pinball_q50"]
                       .mean().round(4)))

        if "coverage_90" in df_res.columns:
            print("\nCobertura empírica 90% [Q05-Q95] TEST media por grupo de horizonte:")
            _c90 = df_res.groupby("h_grupo", observed=True)["coverage_90"].mean()
            _reg(_tablas, "cov90_test_por_grupo", _c90)
            print(_c90.map("{:.1%}".format))

            # Tabla por fold × grupo (coverage_90)
            print("\nCobertura 90% TEST por fold y grupo de horizonte:")
            _t90 = (
                df_res.groupby(["fold", "h_grupo"], observed=True)["coverage_90"]
                .mean()
                .unstack("h_grupo")
            )
            _t90.index = ["Fold {}".format(f) for f in _t90.index]
            _reg(_tablas, "cov90_test_fold_grupo", _t90)
            print(_t90.applymap("{:.1%}".format).to_string())

        if "coverage_98" in df_res.columns:
            print("\nCobertura empírica 98% [Q01-Q99] TEST media por grupo de horizonte:")
            _c98 = df_res.groupby("h_grupo", observed=True)["coverage_98"].mean()
            _reg(_tablas, "cov98_test_por_grupo", _c98)
            print(_c98.map("{:.1%}".format))

            # Tabla por fold × grupo (coverage_98)
            print("\nCobertura 98% TEST por fold y grupo de horizonte:")
            _t98 = (
                df_res.groupby(["fold", "h_grupo"], observed=True)["coverage_98"]
                .mean()
                .unstack("h_grupo")
            )
            _t98.index = ["Fold {}".format(f) for f in _t98.index]
            _reg(_tablas, "cov98_test_fold_grupo", _t98)
            print(_t98.applymap("{:.1%}".format).to_string())

        if "val_coverage_90" in df_res.columns:
            print("\nCobertura empírica 90% [Q05-Q95] VAL media por grupo de horizonte:")
            _v90 = df_res.groupby("h_grupo", observed=True)["val_coverage_90"].mean()
            _reg(_tablas, "cov90_val_por_grupo", _v90)
            print(_v90.map("{:.1%}".format))

        if "val_coverage_98" in df_res.columns:
            print("\nCobertura empírica 98% [Q01-Q99] VAL media por grupo de horizonte:")
            _v98 = df_res.groupby("h_grupo", observed=True)["val_coverage_98"].mean()
            _reg(_tablas, "cov98_val_por_grupo", _v98)
            print(_v98.map("{:.1%}".format))

        if "winkler_90" in df_res.columns:
            print("\nWinkler Score 90% medio por grupo de horizonte (MM USD):")
            print(_reg(_tablas, "winkler90_por_grupo",
                       (df_res.groupby("h_grupo", observed=True)["winkler_90"]
                        .mean() / 1e6).round(3)))

        if "crps" in df_res.columns:
            print("\nCRPS medio por grupo de horizonte (MM USD):")
            print(_reg(_tablas, "crps_por_grupo",
                       (df_res.groupby("h_grupo", observed=True)["crps"]
                        .mean() / 1e6).round(4)))

        tree_cols = sorted([c for c in df_res.columns if c.startswith("n_trees_")])
        if tree_cols:
            print(f"\nÁrboles efectivos tras early stopping "
                  f"(techo N_ESTIMATORS_MAX={N_ESTIMATORS_MAX}, "
                  f"paciencia={EARLY_STOPPING_ROUNDS}):")
            _tr = df_res[tree_cols].mean().round(1)
            _tr.index = [c.replace("n_trees_", "") for c in _tr.index]
            _reg(_tablas, "arboles_efectivos", _tr)
            print(_tr.to_string())
            _mx = float(df_res[tree_cols].max().max())
            if _mx >= N_ESTIMATORS_MAX:
                print(f"  ⚠  Algún modelo agotó el techo ({int(_mx)}) — "
                      f"considera subir N_ESTIMATORS_MAX.")
            else:
                print(f"  ✓  Máximo observado {int(_mx)} < {N_ESTIMATORS_MAX} "
                      f"— el techo no limita.")

        calib_cols = sorted([c for c in df_res.columns if c.startswith("calib_q")])
        if calib_cols:
            print("\nCalibración hit-rate por cuantil (promedio global; debe ≈ tau nominal):")
            calib_mean = df_res[calib_cols].mean().round(3)
            tau_labels = {f"calib_q{int(t*100):02d}": f"Q{int(t*100):02d} (τ={t:.2f})" for t in TAUS}
            calib_mean.index = [tau_labels.get(c, c) for c in calib_mean.index]
            _reg(_tablas, "calibracion_hitrate", calib_mean)
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

    # ------------------------------------------------------------------
    # 9. Tablas de consola → Excel (una hoja por tabla)
    # ------------------------------------------------------------------
    if cruces_rows:
        _reg(_tablas, "cruces_por_fold",
             pd.DataFrame(cruces_rows).set_index("fold"))
    guardar_tablas_excel(_tablas, DIR_MODO, banco, fecha_hoy)

    total_min = (time.time() - t0_total) / 60
    print(f"\n{'='*60}")
    print(f"Tiempo total: {total_min:.1f} min")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
