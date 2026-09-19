# -*- coding: utf-8 -*-
"""
step006_orquestador.py
=======================
Glue code que conecta los outputs REALES de:
  - step005_walk_forward_cv_4.py  → preds_test_fold*.parquet (con regimen_hmm,
    regimen_sigma, año_corte_regimen ya mergeados, sin leakage).
  - step005_validar_hmm_ewma.py (o la variante con winsor/EWMA que dejaste
    fija) → transmat_hmm_<banco>.parquet.

con el módulo step006_simulacion_paths.py (skew-t + cópula AR(1) + backtest
de 3 piezas). No reimplementa nada de la simulación/backtest — solo arma los
inputs correctos y llama a las funciones de ese módulo.

Por qué se agrupa por año_corte_regimen
----------------------------------------
Cada fold de XGBoost (step005_walk_forward_cv_4.py) puede haber usado un
bloque HMM distinto para su feature de régimen (el más reciente sin leakage
respecto a su propio train_end). Eso significa que distintos orígenes del
período de test pueden corresponder a matrices de transición DIFERENTES.
Se simula cada grupo (mismo año_corte_regimen) con SU propia matriz, y se
concatena el resultado al final — la métrica de backtest se calcula sobre
el conjunto completo, igual que se evaluaría en producción.

Por qué el estado inicial es POR ORIGEN, no uno global
-------------------------------------------------------
Para backtest histórico, cada fecha de origen YA tiene un estado de régimen
clasificado (columna regimen_hmm, calculado sin leakage por el HMM). Usar ESE
estado como punto de partida de la simulación de cada origen es lo correcto
— no hay que asumir un único "estado actual" para todo el período de test.

Uso:
    python step006_orquestador.py
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from step006_simulacion_paths import (
    pipeline_simulacion,
    backtest_completo,
    backtest_flujo_neto_completo,
    cargar_preds_test_reales,
    estimar_rho_por_regimen,
    fitear_distribuciones_por_horizonte,
    generar_fancharts_todos_origenes,
    generar_fancharts_neto_todos_origenes,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


###############################################################################
# Configuración — ajustar a tus rutas reales
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
BANCO        = "SISTEMA"

# Debe coincidir EXACTAMENTE con el DIR_MODO de step005_walk_forward_cv_4.py
# (se construye ahí como DIR_OUTPUT / f"{MODELO_CV}_{modo}_{ventanas}";
# pega aquí el valor resultante de esa corrida — ej. con MODELO_CV="xgb",
# EXPANDING=True, VENTANA_TRAIN_AÑOS=5, VENTANA_VAL_AÑOS=0.5, VENTANA_TEST_AÑOS=1):
DIR_MODO = (BASE_SISTEMA / "2. Output" / "step005_wfcv_v3" / "xgb_expanding_50.51")

# Carpeta donde step005_validar_hmm_*.py guardó transmat_hmm_<banco>.parquet
# (su DIR_OUTPUT — normalmente la misma "2. Output" del proyecto).
DIR_REGIMEN_HMM = BASE_SISTEMA / "2. Output"

# Salida de este orquestador
DIR_SALIDA = BASE_SISTEMA / "2. Output" / "step006_simulacion"

# Columna de Prophet en df_preds, si tu 'target' es un RESIDUO de Prophet que
# hay que sumar de vuelta. None si 'target'/'y_realizado' ya es el flujo
# directo (caso típico si XGBoost predice el flujo, no un residuo).
PROPHET_COL = None

VENTANAS = [1, 5, 22, 75]     # plazos en días hábiles (igual que step006)
N_PATHS  = 2000               # paths simulados por fecha de origen
N_JOBS   = -1                 # procesos paralelos para fitear skew-t/PIT
                              # (-1 = todos los cores; 1 = serial). El fit de
                              # cada skew-t es ~0.2s — sobre miles de filas el
                              # cómputo serial es impracticable, ver
                              # step006_simulacion_paths.py.
TAU_BACKTEST = 0.05           # cuantil usado para la pieza 1/2 del backtest
H_REFERENCIA_RHO = None      # horizonte para estimar ρ_s; None = autodetecta el
                              # mínimo h presente en los datos (revisa el log al
                              # correr — fija un entero aquí si quieres forzar otro)
SEED = 42

# ── Fan charts de flujo acumulado (uno por día de origen) ───────────────────
GENERAR_FANCHARTS    = True
DIR_FLUJOS_ACUMULADOS = BASE_SISTEMA / "2. Output" / "flujos_acumulados"
N_PATHS_FANCHART      = 1000   # paths por origen para el fan chart (no
                               # necesita tantos como el backtest; las bandas
                               # ya se ven suaves con 500-1000)
BANDAS_FANCHART       = None   # None = usa BANDAS_FANCHART_DEFAULT de
                               # step006_simulacion_paths.py (P40-60 ... P01-99)

# ── Fan charts de flujo neto diario (sin acumulación) ────────────────────────
# Los percentiles se extraen analíticamente de las AzzaliniT ya fiteadas —
# no requieren simulación adicional. Las bandas pueden no ser monótonas en h.
GENERAR_FANCHARTS_NETO = True
DIR_FLUJOS_NETOS       = BASE_SISTEMA / "2. Output" / "flujos_netos"

# ── Backtest extendido: taus evaluados ───────────────────────────────────────
# TAU_BACKTEST_ACUM : tau existente del backtest acumulado (no cambia).
# TAUS_BACKTEST_EXTRA: taus adicionales para el backtest acumulado (e.g. 0.01).
# TAUS_BACKTEST_NETO : taus para el backtest de flujos netos (pointwise).
TAU_BACKTEST_ACUM   = 0.05
TAUS_BACKTEST_EXTRA = [0.01]
TAUS_BACKTEST_NETO  = [0.01, 0.05]


###############################################################################
# Loader liviano de transmat_hmm_<banco>.parquet (sin importar el script HMM
# completo — evita depender de hmmlearn solo para leer un parquet pequeño)
###############################################################################

def cargar_transmat(banco: str, año_corte: int,
                    dir_regimen: Path = DIR_REGIMEN_HMM) -> np.ndarray:
    """
    Lee transmat_hmm_<banco>.parquet (guardado por step005_validar_hmm_*.py,
    columnas año_corte, p00..p_{n-1}{n-1}) y devuelve la matriz NxN del
    año_corte pedido, ya reordenada (calma/moderado/severo).
    """
    ruta = dir_regimen / f"transmat_hmm_{banco}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Corre primero step005_validar_hmm_*.py "
            f"con GUARDAR_OBJETOS_SIMULACION=True.")
    df_t = pd.read_parquet(ruta)
    fila = df_t[df_t["año_corte"] == año_corte]
    if fila.empty:
        disponibles = sorted(df_t["año_corte"].unique().tolist())
        raise ValueError(f"No hay transmat para año_corte={año_corte} en {ruta}. "
                        f"Disponibles: {disponibles}")
    n_estados = int(np.sqrt(len([c for c in df_t.columns if re.fullmatch(r"p\d+", c)])))
    cols = [f"p{i}{j}" for i in range(n_estados) for j in range(n_estados)]
    return fila.iloc[0][cols].values.astype(float).reshape(n_estados, n_estados)


###############################################################################
# Orquestación principal
###############################################################################

def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar predicciones TEST reales (todos los folds, sin duplicados) ──
    logger.info(f"Cargando preds_test de {BANCO} desde {DIR_MODO} ...")
    df_preds = cargar_preds_test_reales(DIR_MODO, BANCO)
    h_min, h_max = int(df_preds["h"].min()), int(df_preds["h"].max())
    logger.info(f"  {len(df_preds):,} filas | "
               f"{df_preds['fecha_t'].nunique()} orígenes | "
               f"{sorted(df_preds['fold'].unique())} folds | "
               f"h ∈ [{h_min}, {h_max}]")

    ventanas_validas = [v for v in VENTANAS if v >= h_min]
    if len(ventanas_validas) < len(VENTANAS):
        omitidas = [v for v in VENTANAS if v < h_min]
        logger.warning(f"  VENTANAS={omitidas} son menores al horizonte mínimo "
                       f"real (h_min={h_min}) — no hay datos que acumular ahí. "
                       f"Se omiten de esta corrida; ajusta VENTANAS en la config "
                       f"si quieres otro plazo corto.")
        globals()["VENTANAS"] = ventanas_validas

    if "regimen_hmm" not in df_preds.columns or df_preds["regimen_hmm"].isna().all():
        logger.error("No hay feature de régimen en los preds_test (regimen_hmm "
                     "ausente o todo NaN). Corre step005_walk_forward_cv_4.py "
                     "con USAR_FEATURE_REGIMEN=True antes de simular.")
        return

    # ── 2. ρ_s por régimen — leído de VAL (estimado sin leakage en step005) ─────
    #
    # ANTI-LEAKAGE: step005_walk_forward_cv_4.py (con ESTIMAR_RHO_EN_VAL=True)
    # clasifica el período de VALIDACION con los parámetros HMM ya fijos del
    # fold de TRAIN, calcula z_t=Phi^-1(PIT) con los cuantiles de XGBoost en
    # VAL y los realizados, y guarda rho_s_0/1/2 en los parquets de preds_test.
    # Aquí se leen esos valores por grupo (año_corte_regimen) en vez de
    # re-estimar sobre TEST donde los regímenes eran NaN/mediana imputados.
    # Si no existen las columnas (run previo sin el flag) se cae al método
    # antiguo con un aviso, para no romper compatibilidad hacia atrás.

    _tiene_rho_val = all(f"rho_s_{s}" in df_preds.columns for s in range(3))

    # ── 3. Agrupar por año_corte_regimen y simular cada grupo con su propia
    #       matriz de transición y el estado inicial real de cada origen ──────
    if "año_corte_regimen" not in df_preds.columns:
        logger.error("Falta columna 'año_corte_regimen' en preds_test — "
                     "no se puede saber qué matriz de transición usar.")
        return

    grupos = df_preds.dropna(subset=["año_corte_regimen"]).groupby("año_corte_regimen")
    sin_grupo = df_preds["año_corte_regimen"].isna().sum()
    if sin_grupo > 0:
        n_fechas_sin = df_preds.loc[df_preds["año_corte_regimen"].isna(),
                                    "fecha_t"].nunique()
        logger.warning(f"  {sin_grupo:,} filas ({n_fechas_sin} orígenes) sin "
                       f"año_corte_regimen — se EXCLUYEN de la simulación "
                       f"(no hay matriz de transición asociada).")

    resultados_sim = []
    for año_corte, df_grupo in grupos:
        año_corte = int(año_corte)
        try:
            A = cargar_transmat(BANCO, año_corte)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"  año_corte_regimen={año_corte}: {e} — grupo omitido "
                           f"({df_grupo['fecha_t'].nunique()} orígenes afectados).")
            continue

        # Estado inicial = el régimen YA clasificado de cada origen (sin leakage,
        # viene de step005_walk_forward_cv_4.py / step005_validar_hmm_*.py).
        estado_por_origen = (
            df_grupo.drop_duplicates("fecha_t")
            .set_index("fecha_t")["regimen_hmm"]
            .astype(int)
        )

        # ── rho_s para ESTE grupo: del parquet (VAL) o fallback antiguo ────
        if _tiene_rho_val:
            # Tomar la primera fila del grupo (rho_s_* es constante por fold)
            _primera = df_grupo.drop_duplicates("año_corte_regimen").iloc[0]
            rho_por_regimen = {s: float(_primera.get(f"rho_s_{s}", 0.3)) for s in range(3)}
            logger.info(f"  año_corte_regimen={año_corte}: rho_s leido de VAL: {rho_por_regimen}")
        else:
            # Fallback: estimar sobre test (comportamiento anterior)
            if año_corte == int(df_preds["año_corte_regimen"].min()):
                logger.warning("  [RHO] Columnas rho_s_* no encontradas en preds_test. "                               "Corre step005_walk_forward_cv_4.py con ESTIMAR_RHO_EN_VAL=True "                               "para estimar rho en VAL sin leakage. Usando estimacion sobre TEST.")
            h_ref = H_REFERENCIA_RHO if H_REFERENCIA_RHO is not None else int(df_preds["h"].min())
            rho_por_regimen = estimar_rho_por_regimen(
                df_grupo, h_referencia=h_ref, n_jobs=N_JOBS)

        logger.info(f"  año_corte_regimen={año_corte}: "
                   f"{df_grupo['fecha_t'].nunique()} orígenes | "
                   f"diag(A)={np.diag(A).round(3).tolist()} | "                   f"rho_s={rho_por_regimen}")

        # Se fitea UNA vez por grupo y se reusa tanto para el backtest como
        # para los fan charts — el ajuste de cada SplitT es el costo dominante
        # del pipeline (~0.2s c/u); fitearlo dos veces duplicaría ese costo
        # sin necesidad, ya que ambos consumen las mismas (fecha_t, h).
        distribuciones_grupo = fitear_distribuciones_por_horizonte(
            df_grupo, taus=None, n_jobs=N_JOBS)

        df_sim_grupo = pipeline_simulacion(
            df_grupo,
            estado_inicial    = estado_por_origen,
            matriz_transicion = A,
            rho_por_regimen   = rho_por_regimen,
            ventanas          = VENTANAS,
            n_paths           = N_PATHS,
            prophet_col       = PROPHET_COL,
            seed              = SEED + año_corte,
            n_jobs            = N_JOBS,
            distribuciones    = distribuciones_grupo,
        )
        resultados_sim.append(df_sim_grupo)

        if GENERAR_FANCHARTS:
            generar_fancharts_todos_origenes(
                df_grupo,
                distribuciones    = distribuciones_grupo,
                estado_inicial    = estado_por_origen,
                matriz_transicion = A,
                rho_por_regimen   = rho_por_regimen,
                dir_salida        = DIR_FLUJOS_ACUMULADOS,
                banco             = BANCO,
                n_paths           = N_PATHS_FANCHART,
                prophet_col       = PROPHET_COL,
                seed              = SEED + año_corte,
                bandas            = BANDAS_FANCHART,
            )

        if GENERAR_FANCHARTS_NETO:
            generar_fancharts_neto_todos_origenes(
                df_grupo,
                distribuciones = distribuciones_grupo,
                dir_salida     = DIR_FLUJOS_NETOS,
                banco          = BANCO,
                bandas         = BANDAS_FANCHART,
            )

    if not resultados_sim:
        logger.error("Ningún grupo pudo simularse — revisa que existan los "
                     "transmat_hmm_<banco>.parquet correspondientes.")
        return

    df_sim = pd.concat(resultados_sim, ignore_index=True)
    logger.info(f"Simulación completa: {len(df_sim):,} filas "
               f"({df_sim['fecha_t'].nunique()} orígenes simulados)")

    # ── 4. Backtest acumulado — tau principal + taus adicionales ──────────────
    logger.info(f"Backtest acumulado (3 piezas) tau={TAU_BACKTEST_ACUM}...")
    df_bt = backtest_completo(df_sim, tau_col=TAU_BACKTEST_ACUM, ventanas=VENTANAS)

    dfs_bt_extra = []
    for tau_extra in TAUS_BACKTEST_EXTRA:
        if tau_extra != TAU_BACKTEST_ACUM:
            logger.info(f"  Backtest acumulado tau={tau_extra}...")
            df_bt_e = backtest_completo(df_sim, tau_col=tau_extra, ventanas=VENTANAS)
            df_bt_e.insert(0, "tau", tau_extra)
            dfs_bt_extra.append(df_bt_e)

    df_bt_acum_all = pd.concat(
        [df_bt.assign(tau=TAU_BACKTEST_ACUM)] + dfs_bt_extra,
        ignore_index=True)

    # ── 4b. Backtest flujo neto — pointwise, todos los tau en TAUS_BACKTEST_NETO
    logger.info(f"Backtest flujo neto (3 piezas) taus={TAUS_BACKTEST_NETO}...")
    # Reusa distribuciones ya fiteadas por grupo, sin re-fit completo.
    distribuciones_test = {}
    for _gc, _grupo in df_preds.dropna(subset=["año_corte_regimen"]).groupby("año_corte_regimen"):
        distribuciones_test.update(fitear_distribuciones_por_horizonte(_grupo, n_jobs=N_JOBS))
    logger.info(f"  distribuciones_test: {len(distribuciones_test)} objetos")
    df_bt_neto = backtest_flujo_neto_completo(
        df_preds,
        distribuciones  = distribuciones_test,
        taus_eval       = TAUS_BACKTEST_NETO,
        horizontes_eval = VENTANAS,   # mismos horizontes que el backtest acumulado
    )

    # ── 5. Guardar y mostrar resultados ───────────────────────────────────────
    ruta_sim      = DIR_SALIDA / f"simulacion_paths_{BANCO}.parquet"
    ruta_bt       = DIR_SALIDA / f"backtest_acum_{BANCO}.parquet"
    ruta_bt_neto  = DIR_SALIDA / f"backtest_neto_{BANCO}.parquet"
    df_sim.to_parquet(ruta_sim, index=False)
    df_bt_acum_all.to_parquet(ruta_bt, index=False)
    df_bt_neto.to_parquet(ruta_bt_neto, index=False)
    logger.info(f"Guardado: {ruta_sim.name}")
    logger.info(f"Guardado: {ruta_bt.name}  ({len(df_bt_acum_all)} filas — taus={df_bt_acum_all['tau'].unique().tolist()})")
    logger.info(f"Guardado: {ruta_bt_neto.name}  ({len(df_bt_neto)} filas — backtest neto)")


    print("\n" + "=" * 78)
    print("=" * 78)
    print(df_bt_acum_all.to_string(index=False))
    print("\n" + "=" * 78)
    print(f"  BACKTEST FLUJO NETO — {BANCO}  (τ={TAUS_BACKTEST_NETO})")
    print("=" * 78)
    print(df_bt_neto.to_string(index=False))
    print("=" * 78)

    return df_sim, df_bt_acum_all, df_bt_neto


if __name__ == "__main__":
    main()
