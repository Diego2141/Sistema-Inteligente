"""
Regenera ÚNICAMENTE los fan charts a partir de los modelos ya entrenados.
No ejecuta Optuna ni reentrenamiento — requiere que step005_walk_forward_cv_3.py
haya corrido antes con GUARDAR_MODELOS_TODOS_FOLDS=True.

Orden de generación por fold:
  1. Fan chart flujos normales   (fanchart_test_fold*)
  2. Fan chart acumulado bandas  (fanchart_acum_test_fold*)
  3. Fan chart acumulado punto   (fanchart_acum_punto_test_fold*) — realizado / media / mediana
  4. Fan chart acumulado estresado (fanchart_acum_puntq05_test_fold*) — + escenario Q05 terminal

Uso:
    python step005_regenerar_fancharts.py
"""
import sys
import time
import logging
from pathlib import Path

# ── El script principal tiene guard __main__, así que importar es seguro ──────
sys.path.insert(0, str(Path(__file__).parent))

from step005_walk_forward_cv_3 import (
    # ── Constantes de configuración ──────────────────────────────────────────
    BANCOS_A_EVALUAR,
    EXPANDING,
    MODELO_CV,
    VENTANA_TRAIN_AÑOS,
    VENTANA_VAL_AÑOS,
    VENTANA_TEST_AÑOS,
    PASO_AÑOS,
    PURGE_DIAS_HAB,
    PURGE_VAL_TEST,
    N_MAX_FOLDS,
    FOLDS_MANUALES,
    SOLO_FOLDS_MANUALES,
    RUTA_MATRIZ,
    DIR_FANCHARTS,
    DIR_FANCHARTS_MANUALES,
    # ── Funciones de datos ───────────────────────────────────────────────────
    generar_folds,
    resolver_folds_manuales,
    preparar_fold_data,
    get_feature_cols,
    predecir_fold,
    # ── Carga de modelos desde disco ─────────────────────────────────────────
    _cargar_metadata_disco,
    _cargar_modelos_fold_disco,
    # ── Fan chart functions (en orden de generación) ─────────────────────────
    graficar_fanchart_test_fold,
    graficar_fanchart_acum_test_fold,
    graficar_fanchart_acum_punto_test_fold,
    graficar_fanchart_acum_punto_q05_test_fold,
)

import pandas as pd

# ── Logger propio para este script ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("replot")


# ─────────────────────────────────────────────────────────────────────────────
def regenerar_fancharts_banco(banco: str) -> None:
    modo = "EXPANDING" if EXPANDING else "ROLLING"
    log.info(f"\n{'='*65}")
    log.info(f"REPLOT — {banco}  [{modo}]  [{MODELO_CV.upper()}]")
    log.info(f"{'='*65}")

    # ── 1. Cargar feature matrix ──────────────────────────────────────────────
    if not RUTA_MATRIZ.exists():
        log.error(f"  Matriz no encontrada: {RUTA_MATRIZ}")
        return

    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    if df.empty or df["target"].notna().sum() < 500:
        log.warning(f"  [{banco}] Datos insuficientes — omitiendo")
        return

    cols_feat = get_feature_cols(df)
    fechas    = pd.DatetimeIndex(df["fecha_t"].unique())
    log.info(f"  [{banco}] {len(df):,} filas | {len(cols_feat)} features | "
             f"rango: {fechas.min().date()} → {fechas.max().date()}")

    # ── 2. Cargar metadata (manifest de folds) ────────────────────────────────
    try:
        meta        = _cargar_metadata_disco(banco)
        fm_list     = meta.get("folds_manifest", [])
        fm_idx      = {fi["fold"]: fi for fi in fm_list}
        log.info(f"  [{banco}] {len(fm_list)} folds en manifest")
    except FileNotFoundError as e:
        log.error(f"  [{banco}] {e}")
        log.error("  Ejecuta primero step005_walk_forward_cv_3.py con "
                  "GUARDAR_MODELOS_TODOS_FOLDS=True")
        return

    # ── 3. Generar mismos folds que el script principal ───────────────────────
    folds = generar_folds(
        fechas_disponibles=fechas,
        ventana_train_años=VENTANA_TRAIN_AÑOS,
        ventana_val_años=VENTANA_VAL_AÑOS,
        ventana_test_años=VENTANA_TEST_AÑOS,
        paso_años=PASO_AÑOS,
        purge_dias_hab=PURGE_DIAS_HAB,
        purge_val_test=PURGE_VAL_TEST,
        expanding=EXPANDING,
    )
    if not folds:
        log.error(f"  [{banco}] No se generaron folds")
        return

    if N_MAX_FOLDS is not None and len(folds) > N_MAX_FOLDS:
        folds = folds[:N_MAX_FOLDS]

    if FOLDS_MANUALES:
        n_previos  = 0 if SOLO_FOLDS_MANUALES else len(folds)
        folds_man  = resolver_folds_manuales(FOLDS_MANUALES, fechas, n_previos)
        folds      = folds_man if SOLO_FOLDS_MANUALES else folds + folds_man

    log.info(f"  [{banco}] {len(folds)} folds a procesar")

    # ── 4. Iterar folds ───────────────────────────────────────────────────────
    n_ok = 0
    for fold in folds:
        fold_num = fold["fold"]
        t_fold   = time.time()
        log.info(f"\n  ── Fold {fold_num}/{len(folds)} ──────────────────────")

        # 4a. Preparar datos del fold
        try:
            (_, _, _, _, X_test, y_test,
             _, _, h_test, fechas_t_test) = preparar_fold_data(df, fold, cols_feat)
        except Exception as e:
            log.warning(f"  Fold {fold_num}: error preparando datos — {e}")
            continue

        if len(X_test) < 20:
            log.warning(f"  Fold {fold_num}: datos de test insuficientes — omitiendo")
            continue

        # 4b. Cargar modelos del disco
        fold_info = fm_idx.get(fold_num)
        if fold_info is None:
            log.warning(f"  Fold {fold_num}: no está en el manifest — omitiendo")
            continue

        try:
            modelos = _cargar_modelos_fold_disco(fold_info, banco)
        except FileNotFoundError as e:
            log.warning(f"  Fold {fold_num}: {e} — omitiendo")
            continue

        # 4c. Predecir
        preds_test = predecir_fold(modelos, X_test)

        # 4d. Determinar carpeta de salida (manual vs automático)
        _dir_out = DIR_FANCHARTS_MANUALES if fold.get("_manual") else None

        # ── 1. Flujos normales ────────────────────────────────────────────────
        graficar_fanchart_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test,
            fold, banco,
            preds_overlay=None,
            dir_out=_dir_out,
        )

        # ── 2. Acumulado con bandas ───────────────────────────────────────────
        graficar_fanchart_acum_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test,
            fold, banco,
            dir_out=_dir_out,
        )

        # ── 3. Acumulado punto (realizado / media / mediana) ──────────────────
        graficar_fanchart_acum_punto_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test,
            fold, banco,
            dir_out=_dir_out,
        )

        # ── 4. Acumulado estresado (+ escenario Q05 terminal) ─────────────────
        graficar_fanchart_acum_punto_q05_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test,
            fold, banco,
            dir_out=_dir_out,
        )

        n_ok += 1
        log.info(f"  Fold {fold_num}: OK ({time.time()-t_fold:.1f}s)")

    log.info(f"\n  [{banco}] {n_ok}/{len(folds)} folds procesados → {DIR_FANCHARTS}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("STEP005 — Regenerar fan charts")
    log.info(f"  Bancos: {BANCOS_A_EVALUAR}")
    log.info(f"  Modo:   {'EXPANDING' if EXPANDING else 'ROLLING'}  [{MODELO_CV.upper()}]")

    for banco in BANCOS_A_EVALUAR:
        regenerar_fancharts_banco(banco)

    log.info(f"\n✓ Completado en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
