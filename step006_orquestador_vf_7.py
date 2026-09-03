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

from step006_simulacion_paths_vf6 import (
    pipeline_simulacion,
    backtest_completo,
    backtest_flujo_neto_completo,
    cargar_preds_test_reales,
    estimar_rho_por_regimen,
    fitear_distribuciones_por_horizonte,
    generar_fancharts_todos_origenes,
    generar_fancharts_neto_todos_origenes,
    generar_fancharts_integrados_todos_origenes,
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
MODELO       = "expanding"

# Entidad de la que salen las ETIQUETAS de regimen. DEBE coincidir con
# BANCO_REGIMEN de step005_walk_forward_cv_3.7.py: la columna regimen_hmm de
# preds_test se usa aca como estado_inicial de la cadena de Markov, y la matriz
# de transicion que la propaga tiene que ser la del MISMO HMM. Un estado "2" de
# SISTEMA no significa lo mismo que un "2" de FOCO_BBVA — son ajustes distintos,
# con medias y covarianzas propias, aunque ambos esten ordenados por volatilidad.
# Mezclarlos daria paths de regimen sin sentido, sin ningun error visible.
# None → cada entidad usa su propia transmat (comportamiento anterior).
BANCO_REGIMEN = "SISTEMA"

# Debe coincidir EXACTAMENTE con el DIR_MODO de step005_walk_forward_cv_4.py
# (se construye ahí como DIR_OUTPUT / f"{MODELO_CV}_{modo}_{ventanas}";
# pega aquí el valor resultante de esa corrida — ej. con MODELO_CV="xgb",
# EXPANDING=True, VENTANA_TRAIN_AÑOS=5, VENTANA_VAL_AÑOS=0.5, VENTANA_TEST_AÑOS=1):
DIR_MODO = (BASE_SISTEMA / "2. Output" / "step005_wfcv_v3" / "xgb_qt_expanding_310.5")

# Carpeta donde step005_validar_hmm_*.py guardó transmat_hmm_<banco>.parquet
# (su DIR_OUTPUT — normalmente la misma "2. Output" del proyecto).
DIR_REGIMEN_HMM = BASE_SISTEMA / "2. Output"

# Salida de este orquestador
DIR_SALIDA = BASE_SISTEMA / "2. Output" / "step006_simulacion" / "xgb_qt_expanding_310.5"

# Columna de Prophet en df_preds, si tu 'target' es un RESIDUO de Prophet que
# hay que sumar de vuelta. None si 'target'/'y_realizado' ya es el flujo
# directo (caso típico si XGBoost predice el flujo, no un residuo).♣
PROPHET_COL = None

VENTANAS = list(np.linspace(2,75,74))
VENTANAS = [int(i) for i in VENTANAS]
#VENTANAS = [2, 5,10, 22,44,66,75]     # plazos en días hábiles (igual que step006)

N_PATHS  = 10000               # paths simulados por fecha de origen
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
DIR_FLUJOS_ACUMULADOS = BASE_SISTEMA / "2. Output" / "flujos_acumulados" / "xgb_qt_expanding_310.5"
N_PATHS_FANCHART      = 100  # paths por origen (default aumentado para aprovechar
                               # la paralelización — con 8 cores el tiempo de cómputo
                               # es comparable al anterior con 1000 paths serial)
BANDAS_FANCHART       = None   # None = usa BANDAS_FANCHART_DEFAULT de
                               # step006_simulacion_paths.py (P40-60 ... P01-99)

# ── Fan charts de flujo neto diario (sin acumulación) ────────────────────────
# Los percentiles se extraen analíticamente de las AzzaliniT ya fiteadas —
# no requieren simulación adicional. Las bandas pueden no ser monótonas en h.
GENERAR_FANCHARTS_NETO = True
DIR_FLUJOS_NETOS       = BASE_SISTEMA / "2. Output" / "flujos_netos" / "xgb_qt_expanding_310.5"

# ── Fanchart INTEGRADO (3 filas: neto XGBoost crudo / neto distribución / ────
#     acumulado simulado) — imagen adicional, NO reemplaza las anteriores.
GENERAR_FANCHARTS_INTEGRADO = True
DIR_FLUJOS_INTEGRADOS       = BASE_SISTEMA / "2. Output" / "flujos_integrados" / "xgb_qt_expanding_310.5"

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

def cargar_transmat(banco: str, año_corte,
                    dir_regimen: Path = DIR_REGIMEN_HMM) -> np.ndarray:
    """
    Lee transmat_hmm_<banco>.parquet y devuelve la matriz NxN del año_corte
    pedido. año_corte puede ser string ISO "YYYY-MM-DD" (nuevo) o int (legacy).
    La comparación se hace por string para ser insensible al tipo del parquet.
    """
    ruta = dir_regimen / f"transmat_hmm_{banco}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Corre primero step005_validar_hmm_*.py "
            f"con GUARDAR_OBJETOS_SIMULACION=True.")
    df_t = pd.read_parquet(ruta)
    # Normalizar a string para comparación tipo-insensible
    df_t["_año_corte_str"] = df_t["año_corte"].astype(str)
    año_corte_str = str(año_corte)
    fila = df_t[df_t["_año_corte_str"] == año_corte_str]
    if fila.empty:
        disponibles = sorted(df_t["año_corte"].astype(str).unique().tolist())
        raise ValueError(f"No hay transmat para año_corte={año_corte} en {ruta}. "
                        f"Disponibles: {disponibles}")
    n_estados = int(np.sqrt(len([c for c in df_t.columns if re.fullmatch(r"p\d+", c)])))
    cols = [f"p{i}{j}" for i in range(n_estados) for j in range(n_estados)]
    return fila.iloc[0][cols].values.astype(float).reshape(n_estados, n_estados)


###############################################################################
# Orquestación principal
###############################################################################

def _detectar_n_estados_rho(columnas) -> int | None:
    """
    Detecta cuántos estados hay a partir de las columnas rho_s_0, rho_s_1, ...
    realmente presentes en el parquet — nunca asume 2 o 3 fijo, porque
    step005_validar_hmm*.py puede haberse corrido con N_ESTADOS=2 o 3.

    Devuelve None si no hay al menos rho_s_0 y rho_s_1, o si la secuencia
    tiene huecos (p.ej. rho_s_0 y rho_s_2 sin rho_s_1 — eso es un dato
    corrupto, no un caso válido de N_ESTADOS, y se trata como ausente).
    """
    indices = sorted(
        int(m.group(1)) for c in columnas
        if (m := re.fullmatch(r"rho_s_(\d+)", str(c)))
    )
    if len(indices) < 2 or indices != list(range(len(indices))):
        return None
    return len(indices)


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
    # VAL y los realizados, y guarda rho_s_0/rho_s_1/... en los parquets de
    # preds_test — una columna por estado (2 o 3, según N_ESTADOS con el que
    # se corrió step005_validar_hmm*.py; nunca se asume un número fijo). Aquí
    # se leen esos valores por grupo (año_corte_regimen) en vez de re-estimar
    # sobre TEST donde los regímenes eran NaN/mediana imputados. Si no
    # existen las columnas (run previo sin el flag) se cae al método antiguo
    # con un aviso, para no romper compatibilidad hacia atrás.

    _n_estados_rho = _detectar_n_estados_rho(df_preds.columns)
    _tiene_rho_val = _n_estados_rho is not None

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

    distribuciones_todos = {}   # acumula distribuciones de todos los grupos
    resultados_sim = []
    rho_regimen_rows = []
    
    for año_corte, df_grupo in grupos:
        # año_corte puede ser string ISO "2022-07-01" (nuevo) o int (legacy).
        # NO forzar int() — rompe con fechas ISO. Se mantiene el tipo original.
        # Para la semilla (que debe ser int), derivar un hash reproducible.
        try:
            _seed_offset = int(str(año_corte).replace("-", "")) % 100_000
        except Exception:
            _seed_offset = 0
        try:
            A = cargar_transmat(BANCO if BANCO_REGIMEN is None else BANCO_REGIMEN,
                                año_corte)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"  año_corte_regimen={año_corte}: {e} — grupo omitido "
                           f"({df_grupo['fecha_t'].nunique()} orígenes afectados).")
            continue

        estado_por_origen = (
            df_grupo.drop_duplicates("fecha_t")
            .set_index("fecha_t")["regimen_hmm"]
            .astype(int)
        )

        # ── rho_s para ESTE grupo: del parquet (VAL) o fallback antiguo ────
        if _tiene_rho_val:
            _primera = df_grupo.drop_duplicates("año_corte_regimen").iloc[0]
            rho_por_regimen = {s: float(_primera[f"rho_s_{s}"]) for s in range(_n_estados_rho)}
            # Guard: la columna puede existir GLOBALMENTE (pasa _tiene_rho_val)
            # pero venir NaN para ESTE grupo específico si se mezcló un
            # preds_test_fold*.parquet viejo (guardado antes de activar
            # ESTIMAR_RHO_EN_VAL) con otros nuevos — pd.concat rellena con
            # NaN las columnas ausentes del archivo viejo. Un rho=NaN no
            # truena en la recursión AR(1) (NaN es un float válido), se
            # propaga en silencio si no se detecta aquí.
            if any(np.isnan(v) for v in rho_por_regimen.values()):
                logger.error(f"  año_corte_regimen={año_corte}: rho_s_0.."
                            f"rho_s_{_n_estados_rho - 1} contiene NaN para "
                            f"este grupo — grupo omitido "
                            f"({df_grupo['fecha_t'].nunique()} orígenes afectados). "
                            f"Probable mezcla de preds_test_fold*.parquet de una "
                            f"corrida sin ESTIMAR_RHO_EN_VAL con otros que sí lo "
                            f"tienen. Regenera todos los folds con el mismo flag.")
                continue
            logger.info(f"  año_corte_regimen={año_corte}: rho_s leido de VAL: {rho_por_regimen}")
        else:
            # Fallback: estimar sobre test (comportamiento anterior)
            _todos_str = df_preds["año_corte_regimen"].astype(str)
            if str(año_corte) == _todos_str.min():
                logger.warning("  [RHO] Columnas rho_s_* no encontradas en preds_test. "
                               "Corre step005_walk_forward_cv_4.py con ESTIMAR_RHO_EN_VAL=True.")
            h_ref = H_REFERENCIA_RHO if H_REFERENCIA_RHO is not None else int(df_preds["h"].min())
            rho_por_regimen = estimar_rho_por_regimen(
                df_grupo, h_referencia=h_ref, n_jobs=N_JOBS)

        logger.info(f"  año_corte_regimen={año_corte}: "
                   f"{df_grupo['fecha_t'].nunique()} orígenes | "
                   f"diag(A)={np.diag(A).round(3).tolist()} | "
                   f"rho_s={rho_por_regimen}")
 
        # Acumular para reporte
        diag_A = np.diag(A)
        rho_regimen_rows.append({
            "año_corte_regimen": str(año_corte),
            "n_origenes":        df_grupo["fecha_t"].nunique(),
            "rho_s_0":           rho_por_regimen.get(0, np.nan),
            "rho_s_1":           rho_por_regimen.get(1, np.nan),
            "rho_s_2":           rho_por_regimen.get(2, np.nan),
            "diag_A_0":          float(diag_A[0]) if len(diag_A) > 0 else np.nan,
            "diag_A_1":          float(diag_A[1]) if len(diag_A) > 1 else np.nan,
            "diag_A_2":          float(diag_A[2]) if len(diag_A) > 2 else np.nan,
        })
        
        distribuciones_grupo = fitear_distribuciones_por_horizonte(
            df_grupo, taus=None, n_jobs=N_JOBS)
        distribuciones_todos.update(distribuciones_grupo)   # acumular sin re-fitear

        df_sim_grupo = pipeline_simulacion(
            df_grupo,
            estado_inicial    = estado_por_origen,
            matriz_transicion = A,
            rho_por_regimen   = rho_por_regimen,
            ventanas          = VENTANAS,
            n_paths           = N_PATHS,
            prophet_col       = PROPHET_COL,
            seed              = SEED + _seed_offset,
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
                seed              = SEED + _seed_offset,
                bandas            = BANDAS_FANCHART,
                n_jobs            = N_JOBS,
            )

        if GENERAR_FANCHARTS_NETO:
            generar_fancharts_neto_todos_origenes(
                df_grupo,
                distribuciones = distribuciones_grupo,
                dir_salida     = DIR_FLUJOS_NETOS,
                banco          = BANCO,
                bandas         = BANDAS_FANCHART,
                n_jobs         = N_JOBS,
            )

        if GENERAR_FANCHARTS_INTEGRADO:
            generar_fancharts_integrados_todos_origenes(
                df_grupo,
                distribuciones    = distribuciones_grupo,
                estado_inicial    = estado_por_origen,
                matriz_transicion = A,
                rho_por_regimen   = rho_por_regimen,
                dir_salida        = DIR_FLUJOS_INTEGRADOS,
                banco             = BANCO,
                n_paths           = N_PATHS_FANCHART,
                prophet_col       = PROPHET_COL,
                seed              = SEED + _seed_offset,
                bandas            = BANDAS_FANCHART,
                n_jobs            = N_JOBS,
            )

    if not resultados_sim:
        logger.error("Ningún grupo pudo simularse — revisa que existan los "
                     "transmat_hmm_<banco>.parquet correspondientes.")
        return

    df_sim = pd.concat(resultados_sim, ignore_index=True)
    logger.info(f"Simulación completa: {len(df_sim):,} filas "
               f"({df_sim['fecha_t'].nunique()} orígenes simulados)")

    # ── 3b. Guardar parámetros de distribuciones ──────────────────────────────
    # Se guarda un parquet ligero con los parámetros de cada distribución ajustada.
    # El diagnóstico (diagnostico_fit_skewt.py) lo carga para reconstruir los
    # objetos sin re-fitear — el costo más caro del pipeline.
    # Formato: fecha_t | h | dist_type | p1 | p2 | p3 | p4
    #   AzzaliniT → p1=xi,  p2=omega,   p3=alpha,   p4=nu
    #   SplitT    → p1=loc, p2=scale_l, p3=scale_r, p4=df
    #   PchipGPD  → p1=xi_tail, p2=beta_l, p3=beta_r, p4=nu_origen
    #               (⚠ NO permite reconstruir el objeto: el cuerpo PCHIP
    #                necesita los cuantiles completos, no caben en 4 columnas.
    #                Se guardan como metadatos de cola para diagnóstico; para
    #                re-fitear hay que releer los cuantiles del parquet de
    #                preds_test.)
    logger.info(f"Guardando parámetros de {len(distribuciones_todos):,} distribuciones...")
    dist_rows = []
    for (fecha_t, h), d in distribuciones_todos.items():
        if hasattr(d, "xi_tail"):        # PchipGPD (híbrido)
            dist_rows.append({"fecha_t": fecha_t, "h": int(h),
                              "dist_type": "PchipGPD",
                              "p1": d.xi_tail, "p2": d._beta_l,
                              "p3": d._beta_r, "p4": d.nu_origen})
        elif hasattr(d, "xi"):           # AzzaliniT
            dist_rows.append({"fecha_t": fecha_t, "h": int(h),
                              "dist_type": "AzzaliniT",
                              "p1": d.xi, "p2": d.omega, "p3": d.alpha, "p4": d.nu})
        elif hasattr(d, "loc"):          # SplitT
            dist_rows.append({"fecha_t": fecha_t, "h": int(h),
                              "dist_type": "SplitT",
                              "p1": d.loc, "p2": d.scale_l, "p3": d.scale_r, "p4": d.df})
        else:
            # Tipo desconocido: se registra sin parámetros en vez de tronar
            # al final de una corrida larga por un AttributeError.
            logger.warning(f"  Distribución de tipo inesperado en "
                          f"({fecha_t}, h={h}): {type(d).__name__} — "
                          f"guardada sin parámetros.")
            dist_rows.append({"fecha_t": fecha_t, "h": int(h),
                              "dist_type": type(d).__name__,
                              "p1": np.nan, "p2": np.nan,
                              "p3": np.nan, "p4": np.nan})
    df_dists = pd.DataFrame(dist_rows)
    df_dists["fecha_t"] = pd.to_datetime(df_dists["fecha_t"])
    ruta_dists = DIR_SALIDA / f"distribuciones_{BANCO}.parquet"
    df_dists.to_parquet(ruta_dists, index=False)
    logger.info(f"  Guardado: {ruta_dists.name}  "
                f"({df_dists['dist_type'].value_counts().to_dict()})")

    
    # ── 4. Backtest acumulado — tau principal + taus adicionales ──────────────
    logger.info(f"Backtest acumulado (3 piezas) tau={TAU_BACKTEST_ACUM}...")
    df_bt, raw_acum_principal = backtest_completo(
        df_sim, tau_col=TAU_BACKTEST_ACUM, ventanas=VENTANAS,
        return_raw=True)
 
    dfs_bt_extra        = []
    raw_acum_por_tau    = {TAU_BACKTEST_ACUM: raw_acum_principal}
 
    for tau_extra in TAUS_BACKTEST_EXTRA:
        if tau_extra != TAU_BACKTEST_ACUM:
            logger.info(f"  Backtest acumulado tau={tau_extra}...")
            df_bt_e, raw_extra = backtest_completo(
                df_sim, tau_col=tau_extra, ventanas=VENTANAS,
                return_raw=True)
            df_bt_e.insert(0, "tau", tau_extra)
            dfs_bt_extra.append(df_bt_e)
            raw_acum_por_tau[tau_extra] = raw_extra
 
    df_bt_acum_all = pd.concat(
        [df_bt.assign(tau=TAU_BACKTEST_ACUM)] + dfs_bt_extra,
        ignore_index=True)
 
    # ── 4b. Backtest flujo neto ───────────────────────────────────────────────
    logger.info(f"Backtest flujo neto (3 piezas) taus={TAUS_BACKTEST_NETO}...")
    df_bt_neto, raw_neto_por_tau = backtest_flujo_neto_completo(
        df_preds,
        distribuciones  = distribuciones_todos,
        taus_eval       = TAUS_BACKTEST_NETO,
        horizontes_eval = VENTANAS,
        return_raw      = True,
    )
 
    # ── 5. Guardar resultados principales ─────────────────────────────────────
    ruta_sim     = DIR_SALIDA / f"simulacion_paths_{BANCO}.parquet"
    ruta_bt      = DIR_SALIDA / f"backtest_acum_{BANCO}.parquet"
    ruta_bt_neto = DIR_SALIDA / f"backtest_neto_{BANCO}.parquet"
    df_sim.to_parquet(ruta_sim, index=False)

    # Los flags de las 3 piezas ahora pueden valer None ("no concluyente"),
    # además de True/False. Se normalizan a object con None — NO a dtype
    # nullable "boolean": ese dtype introduce pd.NA, y _pasa_cell() en
    # reporte_backtest.py solo contempla None y float('nan'), así que un
    # pd.NA lo hace tronar con "boolean value of NA is ambiguous" al construir
    # la tabla. object+None es además lo que p2_independencia ya venía
    # produciendo, así que el esquema del parquet no cambia de forma.
    for _df in (df_bt_acum_all, df_bt_neto):
        for _col in ("p1_pasa", "p2_independencia", "p3_uniformidad",
                     "p3_test_valido"):
            if _col in _df.columns:
                _df[_col] = (_df[_col].astype(object)
                                      .where(_df[_col].notna(), None))

    # Resumen de las 3 piezas en el log: si un test vuelve a romperse en
    # silencio, se ve en la corrida en vez de descubrirse en el PDF tres
    # semanas después. Las tres piezas pueden devolver None ("no concluyente")
    # y ese conteo es lo primero que hay que mirar antes de leer PASA/FALLA:
    #   · Pieza 1 → None cuando no hubo NINGUNA violación observada. Con cero
    #     violaciones el intervalo de Wilson colapsa a [0, z²/(n_eff+z²)], un
    #     número que no depende del plazo. No es evidencia de calibración.
    #   · Pieza 2 → None cuando hay menos de 3 violaciones: sin violaciones no
    #     hay duraciones entre ellas y la Weibull no se puede ajustar. En el
    #     acumulado esto vacía la mayoría de las celdas, y la causa está en la
    #     Pieza 3: con el PIT medio subiendo a ~0.75 el realizado casi nunca
    #     rompe el piso a plazos largos.
    #   · Pieza 3 → None cuando el submuestreo no solapado deja menos de 20
    #     PITs, o si el Anderson-Darling falla.
    _PIEZAS = (("Pieza 1", "p1_pasa"), ("Pieza 2", "p2_independencia"),
               ("Pieza 3", "p3_uniformidad"))
    for _nombre, _df in (("acumulado", df_bt_acum_all), ("neto", df_bt_neto)):
        for _etq, _col in _PIEZAS:
            if _col not in _df.columns:
                continue
            _tot   = len(_df)
            _nd    = int(_df[_col].isna().sum())
            _pasa  = int((_df[_col] == True).sum())
            _falla = _tot - _nd - _pasa
            logger.info(f"  {_etq} ({_nombre}): {_tot - _nd}/{_tot} celdas "
                        f"concluyentes | PASA={_pasa} | FALLA={_falla} | "
                        f"N/D={_nd}")
            if _nd == _tot:
                logger.warning(f"  {_etq} ({_nombre}): NINGUNA celda "
                               f"concluyente — revisar antes de reportar.")
            elif _nd > _tot // 2:
                logger.warning(f"  {_etq} ({_nombre}): {_nd} de {_tot} celdas "
                               f"sin veredicto — el resultado agregado no es "
                               f"representativo.")

    df_bt_acum_all.to_parquet(ruta_bt, index=False)
    df_bt_neto.to_parquet(ruta_bt_neto, index=False)
 
    # ── 6. Guardar datos crudos para reporte completo ─────────────────────────
    logger.info("Guardando datos crudos para reporte...")
 
    # 6a. PITs acumulados: (tau, ventana, pit)
    pits_acum_rows = []
    for tau, raw in raw_acum_por_tau.items():
        for v, arr in raw["pits"].items():
            for val in arr:
                pits_acum_rows.append({"tau": tau, "ventana": v, "pit": float(val)})
    if pits_acum_rows:
        pd.DataFrame(pits_acum_rows).to_parquet(
            DIR_SALIDA / f"pits_acum_{BANCO}.parquet", index=False)
 
    # 6b. PITs netos: (tau, h, pit)
    pits_neto_rows = []
    for tau, raw in raw_neto_por_tau.items():
        for h, arr in raw["pits"].items():
            for val in arr:
                pits_neto_rows.append({"tau": tau, "h": h, "pit": float(val)})
    if pits_neto_rows:
        pd.DataFrame(pits_neto_rows).to_parquet(
            DIR_SALIDA / f"pits_neto_{BANCO}.parquet", index=False)
 
    # 6c. Indicadores de violación: (tau, ventana, fecha_t, I_t)
    indic_rows = []
    for tau, raw in raw_acum_por_tau.items():
        for v, d in raw["indicadores"].items():
            for ft, it in zip(d["fechas_t"], d["I"]):
                indic_rows.append({
                    "tau": tau, "ventana": v,
                    "fecha_t": pd.Timestamp(ft), "I_t": float(it)})
    if indic_rows:
        pd.DataFrame(indic_rows).to_parquet(
            DIR_SALIDA / f"indicadores_acum_{BANCO}.parquet", index=False)
 
    # 6d. Duraciones entre violaciones: (tau, ventana, duracion)
    durs_rows = []
    for tau, raw in raw_acum_por_tau.items():
        for v, arr in raw["duraciones"].items():
            for val in arr:
                durs_rows.append({"tau": tau, "ventana": v, "duracion": float(val)})
    if durs_rows:
        pd.DataFrame(durs_rows).to_parquet(
            DIR_SALIDA / f"duraciones_acum_{BANCO}.parquet", index=False)
 
    # 6e. rho_s + diag(A) por grupo HMM
    if rho_regimen_rows:
        pd.DataFrame(rho_regimen_rows).to_parquet(
            DIR_SALIDA / f"rho_regimen_{BANCO}.parquet", index=False)
 
    # 6f. Config del reporte (reproducibilidad)
    import json as _json
    config_reporte = {
        "banco":               BANCO,
        "ventanas":            VENTANAS,
        "n_paths":             N_PATHS,
        "tau_backtest_acum":   TAU_BACKTEST_ACUM,
        "taus_backtest_extra": TAUS_BACKTEST_EXTRA,
        "taus_backtest_neto":  TAUS_BACKTEST_NETO,
        "seed":                SEED,
        "dir_modo":            str(DIR_MODO),
        "fecha_corrida":       pd.Timestamp.today().strftime("%Y-%m-%d %H:%M"),
    }
    with open(DIR_SALIDA / f"config_reporte_{BANCO}.json", "w",
              encoding="utf-8") as _f:
        _json.dump(config_reporte, _f, indent=2, ensure_ascii=False)
 
    logger.info(f"  Datos reporte guardados en {DIR_SALIDA}:")
    for nombre in [
        f"backtest_acum_{BANCO}.parquet",
        f"backtest_neto_{BANCO}.parquet",
        f"pits_acum_{BANCO}.parquet",
        f"pits_neto_{BANCO}.parquet",
        f"indicadores_acum_{BANCO}.parquet",
        f"duraciones_acum_{BANCO}.parquet",
        f"rho_regimen_{BANCO}.parquet",
        f"config_reporte_{BANCO}.json",
    ]:
        ruta = DIR_SALIDA / nombre
        logger.info(f"    {'✓' if ruta.exists() else '✗'} {nombre}")
        
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
