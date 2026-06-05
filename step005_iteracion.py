# -*- coding: utf-8 -*-
"""
step005_iteracion.py
Corre step005_walk_forward_cv_3 para las 8 combinaciones de configuración:
  · MODELO_CV      : ["xgb", "xgb_qt"]
  · VENTANA_VAL    : [0.5, 1.0]  años
  · EXPANDING      : [True, False]

Cada combinación escribe en su propia subcarpeta dentro de DIR_OUTPUT.
N_TRIALS_OPTUNA se fuerza a 60 para todas las corridas.
"""

import gc
import logging
import time
from itertools import product

import pandas as pd

import step005_walk_forward_cv_3 as s5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

###############################################################################
# Configuración del grid
###############################################################################

MODELOS_CV   = ["xgb", "xgb_qt"]
VENTANAS_VAL = [0.5, 1.0]
EXPANDINGS   = [True, False]

N_TRIALS_FORZADO = 60   # sobreescribe el valor de step005 para todas las corridas
BANCOS           = ["SISTEMA"]


###############################################################################
# Helper: actualizar globals de step005
###############################################################################

def _actualizar_config(modelo_cv: str, ventana_val: float, expanding: bool):
    """
    Actualiza los globals relevantes del módulo step005 para la combinación
    indicada y crea las carpetas de salida correspondientes.
    """
    s5.MODELO_CV        = modelo_cv
    s5.VENTANA_VAL_AÑOS = ventana_val
    s5.EXPANDING        = expanding
    s5.EMBARGO_DIAS_HAB = 0 if expanding else 90
    s5.N_TRIALS_OPTUNA  = N_TRIALS_FORZADO

    _modo     = "expanding" if expanding else "rolling"
    _ventanas = f"{s5.VENTANA_TRAIN_AÑOS}{ventana_val}{s5.VENTANA_TEST_AÑOS}"

    s5.DIR_MODO               = s5.DIR_OUTPUT / f"{modelo_cv}_{_modo}_{_ventanas}"
    s5.DIR_MODELOS            = s5.DIR_MODO / "modelos"
    s5.DIR_PLOTS              = s5.DIR_MODO / "plots"
    s5.DIR_FANCHARTS          = s5.DIR_MODO / "fancharts_test"
    s5.DIR_FANCHARTS_MANUALES = s5.DIR_MODO / "fancharts_manuales"

    for _d in (s5.DIR_MODO, s5.DIR_MODELOS, s5.DIR_PLOTS,
               s5.DIR_FANCHARTS, s5.DIR_FANCHARTS_MANUALES):
        _d.mkdir(parents=True, exist_ok=True)

    # Limpiar cache GARCH para no contaminar entre combinaciones
    s5._garch_params_cache.clear()

    logger.info(f"  Output → {s5.DIR_MODO.name}")


###############################################################################
# Main
###############################################################################

def main():
    combos = list(product(MODELOS_CV, VENTANAS_VAL, EXPANDINGS))
    n      = len(combos)

    logger.info("=" * 70)
    logger.info(f"STEP005 — ITERACIÓN DE CONFIGURACIONES  ({n} combinaciones)")
    logger.info(f"  N_TRIALS_OPTUNA forzado : {N_TRIALS_FORZADO}")
    logger.info(f"  Bancos                  : {BANCOS}")
    logger.info("=" * 70)
    logger.info("  # | MODELO_CV | VAL (yr) | MODO      | Carpeta")
    for i, (m, v, e) in enumerate(combos, 1):
        modo = "expanding" if e else "rolling"
        logger.info(f"  {i} | {m:<9} | {v:<8} | {modo:<9} | "
                    f"{m}_{modo}_{s5.VENTANA_TRAIN_AÑOS}{v}{s5.VENTANA_TEST_AÑOS}")

    resumen = []
    t_total = time.time()

    for i, (modelo_cv, ventana_val, expanding) in enumerate(combos, 1):
        modo = "expanding" if expanding else "rolling"
        tag  = f"{modelo_cv} | VAL={ventana_val}yr | {modo.upper()}"

        logger.info(f"\n{'═'*70}")
        logger.info(f"  Combinación {i}/{n}: {tag}")
        logger.info(f"{'═'*70}")

        _actualizar_config(modelo_cv, ventana_val, expanding)

        t0    = time.time()
        ok    = True
        error = ""
        for banco in BANCOS:
            try:
                df_m = s5.evaluar_banco(banco)
                if df_m is not None and "coverage_90" in df_m.columns:
                    cov = df_m["coverage_90"].mean()
                    pb  = df_m["pinball_q50"].mean() if "pinball_q50" in df_m.columns else float("nan")
                    resumen.append({
                        "combo"       : i,
                        "modelo_cv"   : modelo_cv,
                        "ventana_val" : ventana_val,
                        "expanding"   : expanding,
                        "banco"       : banco,
                        "coverage_90" : round(cov, 4),
                        "pinball_q50" : round(pb, 2),
                        "tiempo_min"  : round((time.time() - t0) / 60, 1),
                    })
            except Exception as e:
                logger.error(f"  [{banco}] Error: {e}")
                ok    = False
                error = str(e)

        elapsed = (time.time() - t0) / 60
        estado  = "✓" if ok else f"✗ {error[:60]}"
        logger.info(f"  {estado}  ({elapsed:.1f} min)")
        gc.collect()

    # Resumen global
    logger.info(f"\n{'═'*70}")
    logger.info(f"RESUMEN GLOBAL — {n} combinaciones en {(time.time()-t_total)/60:.1f} min")
    logger.info(f"{'═'*70}")

    if resumen:
        df_res = pd.DataFrame(resumen)
        cols   = ["combo", "modelo_cv", "ventana_val", "expanding",
                  "banco", "coverage_90", "pinball_q50", "tiempo_min"]
        logger.info("\n" + df_res[cols].to_string(index=False))

        ruta_res = s5.DIR_OUTPUT / "iteracion_resumen.csv"
        df_res[cols].to_csv(ruta_res, index=False)
        logger.info(f"\n  Resumen guardado: {ruta_res}")

    logger.info(f"\n  Carpetas de output en: {s5.DIR_OUTPUT}")


if __name__ == "__main__":
    main()
