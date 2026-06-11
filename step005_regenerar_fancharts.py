# -*- coding: utf-8 -*-
"""
Regenera UNICAMENTE los fan charts a partir de los modelos ya entrenados
y exporta los pronosticos TEST (realizados + cuantiles) a CSV/parquet.

No ejecuta Optuna ni reentrenamiento -- requiere que step005_walk_forward_cv_3.py
haya corrido antes con GUARDAR_MODELOS_TODOS_FOLDS=True.

Uso:
    python step005_regenerar_fancharts.py
    runfile('...step005_regenerar_fancharts.py', wdir='...')
"""
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# El script principal tiene guard __main__, asi que importar es seguro
sys.path.insert(0, str(Path(__file__).parent))
import step005_walk_forward_cv_3 as _s5

###############################################################################
# CONFIGURACION — ajustar segun necesidad
###############################################################################

# Directorio raiz donde viven los outputs del step005 (subcarpetas por modo)
DIR_OUTPUT = _s5.DIR_OUTPUT          # H:/.../2. Output/step005_wfcv_v3

# Subcarpeta de modo activo (xgb_rolling_051, xgb_expanding_051, etc.)
DIR_MODO   = _s5.DIR_MODO

# Carpetas de fan charts (se puede cambiar a una ruta personalizada)
DIR_FANCHARTS          = _s5.DIR_FANCHARTS           # .../fancharts_test
DIR_FANCHARTS_MANUALES = _s5.DIR_FANCHARTS_MANUALES  # .../fancharts_manuales

# Carpeta donde se guardan los CSVs/parquets de pronosticos
DIR_PRONOSTICOS = DIR_MODO / "pronosticos_test"

# Formato de exportacion: "csv" o "parquet"
FORMATO_EXPORTACION = "csv"

# Guardar un archivo por fold (True) o un unico archivo por banco (False)
EXPORTAR_POR_FOLD = False

###############################################################################

# Constantes heredadas del script principal
BANCOS_A_EVALUAR    = _s5.BANCOS_A_EVALUAR
EXPANDING           = _s5.EXPANDING
MODELO_CV           = _s5.MODELO_CV
VENTANA_TRAIN       = _s5.__dict__["VENTANA_TRAIN_A\xd1OS"]
VENTANA_VAL         = _s5.__dict__["VENTANA_VAL_A\xd1OS"]
VENTANA_TEST        = _s5.__dict__["VENTANA_TEST_A\xd1OS"]
PASO                = _s5.__dict__["PASO_A\xd1OS"]
PURGE_DIAS_HAB      = _s5.PURGE_DIAS_HAB
PURGE_VAL_TEST      = _s5.PURGE_VAL_TEST
N_MAX_FOLDS         = _s5.N_MAX_FOLDS
FOLDS_MANUALES      = _s5.FOLDS_MANUALES
SOLO_FOLDS_MANUALES = _s5.SOLO_FOLDS_MANUALES
RUTA_MATRIZ         = _s5.RUTA_MATRIZ

# Funciones del script principal
generar_folds                              = _s5.generar_folds
resolver_folds_manuales                    = _s5.resolver_folds_manuales
preparar_fold_data                         = _s5.preparar_fold_data
get_feature_cols                           = _s5.get_feature_cols
predecir_fold                              = _s5.predecir_fold
_cargar_metadata_disco                     = _s5._cargar_metadata_disco
_cargar_modelos_fold_disco                 = _s5._cargar_modelos_fold_disco
graficar_fanchart_test_fold                = _s5.graficar_fanchart_test_fold
graficar_fanchart_acum_test_fold           = _s5.graficar_fanchart_acum_test_fold
graficar_fanchart_acum_punto_test_fold     = _s5.graficar_fanchart_acum_punto_test_fold
graficar_fanchart_acum_punto_q05_test_fold = _s5.graficar_fanchart_acum_punto_q05_test_fold

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("replot")


# =============================================================================
def _guardar_pronosticos(rows: list[dict], banco: str, fold_num: int) -> None:
    """Guarda la lista de filas como CSV o parquet segun FORMATO_EXPORTACION."""
    if not rows:
        return
    DIR_PRONOSTICOS.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)

    # Ordenar columnas: identificadores primero, luego cuantiles, luego media
    id_cols = ["banco", "fold", "fecha_t", "h"]
    q_cols  = sorted([c for c in df_out.columns if c.startswith("q")],
                     key=lambda c: float(c[1:]))
    extra   = [c for c in df_out.columns if c not in id_cols + q_cols]
    df_out  = df_out[id_cols + q_cols + extra]

    sfx = f"fold{fold_num:02d}_{banco}" if EXPORTAR_POR_FOLD else banco
    if FORMATO_EXPORTACION == "parquet":
        ruta = DIR_PRONOSTICOS / f"pronosticos_test_{sfx}.parquet"
        df_out.to_parquet(ruta, index=False)
    else:
        ruta = DIR_PRONOSTICOS / f"pronosticos_test_{sfx}.csv"
        df_out.to_csv(ruta, index=False)
    log.info(f"    Pronosticos guardados: {ruta.name}  ({len(df_out):,} filas)")


# =============================================================================
def regenerar_fancharts_banco(banco: str) -> None:
    modo = "EXPANDING" if EXPANDING else "ROLLING"
    log.info(f"\n{'='*65}")
    log.info(f"REPLOT — {banco}  [{modo}]  [{MODELO_CV.upper()}]")
    log.info(f"{'='*65}")

    # 1. Cargar feature matrix
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

    # 2. Cargar metadata (manifest de folds)
    try:
        meta   = _cargar_metadata_disco(banco)
        fm_idx = {fi["fold"]: fi for fi in meta.get("folds_manifest", [])}
        log.info(f"  [{banco}] {len(fm_idx)} folds en manifest")
    except FileNotFoundError as e:
        log.error(f"  [{banco}] {e}")
        log.error("  Ejecuta primero step005_walk_forward_cv_3.py con "
                  "GUARDAR_MODELOS_TODOS_FOLDS=True")
        return

    # 3. Generar mismos folds que el script principal
    folds = generar_folds(
        fechas_disponibles=fechas,
        ventana_train_años=VENTANA_TRAIN,
        ventana_val_años=VENTANA_VAL,
        ventana_test_años=VENTANA_TEST,
        paso_años=PASO,
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
        n_previos = 0 if SOLO_FOLDS_MANUALES else len(folds)
        folds_man = resolver_folds_manuales(FOLDS_MANUALES, fechas, n_previos)
        folds     = folds_man if SOLO_FOLDS_MANUALES else folds + folds_man

    log.info(f"  [{banco}] {len(folds)} folds a procesar")

    # Acumulador global de filas (cuando EXPORTAR_POR_FOLD=False)
    rows_global: list[dict] = []

    # 4. Iterar folds
    n_ok = 0
    for fold in folds:
        fold_num = fold["fold"]
        t_fold   = time.time()
        log.info(f"\n  -- Fold {fold_num}/{len(folds)} --")

        # 4a. Preparar datos del fold
        try:
            (_, _, _, _, X_test, y_test,
             _, _, h_test, fechas_t_test) = preparar_fold_data(df, fold, cols_feat)
        except Exception as e:
            log.warning(f"  Fold {fold_num}: error preparando datos — {e}")
            continue

        if len(X_test) < 20:
            log.warning(f"  Fold {fold_num}: datos insuficientes — omitiendo")
            continue

        # 4b. Cargar modelos del disco
        fold_info = fm_idx.get(fold_num)
        if fold_info is None:
            log.warning(f"  Fold {fold_num}: no en manifest — omitiendo")
            continue
        try:
            modelos = _cargar_modelos_fold_disco(fold_info, banco)
        except FileNotFoundError as e:
            log.warning(f"  Fold {fold_num}: {e} — omitiendo")
            continue

        # 4c. Predecir
        preds_test = predecir_fold(modelos, X_test)

        # 4d. Construir filas de pronosticos
        rows_fold: list[dict] = []
        n_obs = len(y_test)
        for i in range(n_obs):
            row = {
                "banco"       : banco,
                "fold"        : fold_num,
                "fecha_t"     : pd.Timestamp(fechas_t_test[i]).date(),
                "h"           : int(h_test[i]),
                "y_realizado" : float(y_test.values[i]),
            }
            for tau, arr in preds_test.items():
                if tau == "mean":
                    row["media"] = float(arr[i])
                else:
                    row[f"q{int(tau*100):02d}"] = float(arr[i])
            rows_fold.append(row)

        # 4e. Guardar pronosticos segun modo
        if EXPORTAR_POR_FOLD:
            _guardar_pronosticos(rows_fold, banco, fold_num)
        else:
            rows_global.extend(rows_fold)

        # 4f. Carpeta de fan charts
        _dir_out = DIR_FANCHARTS_MANUALES if fold.get("_manual") else None

        # ── 1. Flujos normales
        graficar_fanchart_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test,
            fold, banco, preds_overlay=None, dir_out=_dir_out,
        )
        # ── 2. Acumulado con bandas
        graficar_fanchart_acum_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test,
            fold, banco, dir_out=_dir_out,
        )
        # ── 3. Acumulado punto: realizado / media / mediana
        graficar_fanchart_acum_punto_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test,
            fold, banco, dir_out=_dir_out,
        )
        # ── 4. Acumulado estresado: + cumsum(Q50) - (Q50-Q05) terminal
        graficar_fanchart_acum_punto_q05_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test,
            fold, banco, dir_out=_dir_out,
        )

        n_ok += 1
        log.info(f"  Fold {fold_num}: OK ({time.time()-t_fold:.1f}s)")

    # Guardar archivo global (un unico archivo por banco)
    if not EXPORTAR_POR_FOLD and rows_global:
        _guardar_pronosticos(rows_global, banco, fold_num=-1)

    log.info(f"\n  [{banco}] {n_ok}/{len(folds)} folds procesados")
    log.info(f"  Fan charts → {DIR_FANCHARTS}")
    log.info(f"  Pronosticos → {DIR_PRONOSTICOS}")


# =============================================================================
def main():
    t0 = time.time()
    log.info("STEP005 — Regenerar fan charts + exportar pronosticos TEST")
    log.info(f"  Bancos       : {BANCOS_A_EVALUAR}")
    log.info(f"  Modo         : {'EXPANDING' if EXPANDING else 'ROLLING'}  [{MODELO_CV.upper()}]")
    log.info(f"  Pronosticos  : {DIR_PRONOSTICOS}  [{FORMATO_EXPORTACION}]")

    for banco in BANCOS_A_EVALUAR:
        regenerar_fancharts_banco(banco)

    log.info(f"\n✓ Completado en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
