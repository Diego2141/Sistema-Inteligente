# -*- coding: utf-8 -*-
"""
step005_walk_forward_cv_3.py
Walk-forward CV con ventana EXPANDABLE o RODANTE y evaluación TEST out-of-sample.

═══════════════════════════════════════════════════════════════════════════════
NOVEDADES RESPECTO A v2
═══════════════════════════════════════════════════════════════════════════════

1. Toggle EXPANDING (True/False)
   ─────────────────────────────
   EXPANDING = True  → ventana CRECIENTE: train_start fijo, train_end crece.
   EXPANDING = False → ventana RODANTE: train_start y train_end avanzan juntos.

   Comparación de enfoques:
     Expanding → más datos en folds tardíos; mejor si el proceso es estacionario.
     Rolling   → se adapta a regímenes recientes; mejor si hay quiebres estructurales.

2. Fan charts TEST out-of-sample (nuevo)
   ──────────────────────────────────────
   Por cada fold se generan hasta 4 snapshots del fan chart en el período TEST,
   con origen separado ~3 meses entre sí. Cada snapshot muestra:
     · Bandas Q01-Q99 (muy suave) y Q05-Q95 (media) → incertidumbre
     · Línea Q50 → mediana predicha
     · Puntos realizados: verde si caen dentro de Q05-Q95, rojo si fuera
     · Coverage empírico anotado en el título de cada panel

3. Separación VAL / TEST
   ──────────────────────
   Idéntica a v2: VAL solo para Optuna, TEST solo para métricas OOS.

═══════════════════════════════════════════════════════════════════════════════
Anti-leakage — idéntico a v1 y v2
═══════════════════════════════════════════════════════════════════════════════
  #1 EMBARGO 90dh post-TRAIN
  #2 GARCH por fold: ω/α/β estimados solo en TRAIN, propagados a VAL+TEST
  #3 Medianas fold: imputación calculada en TRAIN, aplicada a VAL+TEST
"""

from __future__ import annotations

import gc
import itertools
import json
import logging
import os
import sys
import threading
import time
import tracemalloc
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize

try:
    import lightgbm as lgb
    _LGBM_OK = True
except ImportError:
    lgb = None
    _LGBM_OK = False

try:
    import shap
    _SHAP_OK = True
except ImportError:
    shap = None
    _SHAP_OK = False

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


###############################################################################
# PARTE 0 — Configuración
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")

# ══ BOTÓN DE PARTICIÓN ═══════════════════════════════════════════════════════
# False → comportamiento IDÉNTICO al previo a este botón: matriz v1, solo
#         SISTEMA, y las carpetas de salida sin subnivel por entidad.
# True  → matriz de la partición activa y una carpeta por entidad.
#
# El subnivel de carpeta no es cosmético. Varias salidas de diagnóstico NO
# llevan el banco en el nombre del archivo (convergencia_{tau}.png,
# panel_{senal}_{tau}.png, {senal}_fold{NN}_{tau}.png), así que dos entidades
# escribiendo en la misma carpeta se pisan los diagnósticos sin ningún aviso.
# Con la lista de un solo elemento que había antes eso no podía pasar; con
# particiones sí, porque el caso natural es correr FOCO y RESTO.
PARTICIONES = False

# Solo se leen cuando PARTICIONES=True. Se declaran igual acá para que la
# configuración viva en un único lugar en vez de repartirse en dos ramas.
PARTICION = "bbva"      # "bbva" | "globales" — debe existir en step001 v2
ENTIDAD   = "FOCO"      # "SISTEMA" | "FOCO" | "RESTO"
# ═════════════════════════════════════════════════════════════════════════════

_STEM = (f"matriz_features_particiones_{PARTICION}" if PARTICIONES
         else "matriz_features")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / f"{_STEM}.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v3"

GUARDAR_PREDS_TEST = True   # guarda predicciones TEST por fold para step006

# ── Feature de régimen HMM (clasificación de step005_validar_hmm*.py) ────────
# True  → agrega "regimen_hmm" (0=calma...N-1=severo, según N_ESTADOS de ese
#         script — 2 o 3) y "regimen_sigma" (volatilidad EWMA/realizada usada
#         por el HMM) como features, leídas de los parquets guardados por
#         step005_validar_hmm*.py. Por fold se elige el bloque HMM (año_corte)
#         más reciente que NO vio ningún dato posterior a train_end — mismo
#         principio anti-leakage que GARCH/FFD.
# False → comportamiento original, sin esta feature.
USAR_FEATURE_REGIMEN = True

# True  → este script ajusta el HMM por su cuenta (importa step005_validar_hmm_v5)
#         para cada entidad de BANCOS_A_EVALUAR, usando como cortes los train_end
#         de sus propios folds. Es lo que permite que con PARTICIONES=True existan
#         los parquets de FOCO y RESTO sin correr el otro script tres veces
#         cambiando su constante BANCO.
# False → comportamiento anterior: se leen los parquets que ya haya en disco,
#         generados corriendo step005_validar_hmm_v5.py aparte.
#
# Solo regenera cuando hace falta: si el parquet ya tiene un bloque por cada
# train_end, no reajusta nada.
HMM_INTERNO = True
# Carpeta donde step005_validar_hmm*.py guarda estados_regimen_hmm_<banco>.parquet
# y transmat_hmm_<banco>.parquet (DIR_OUTPUT de ese script).
DIR_REGIMEN_HMM = BASE_SISTEMA / "2. Output"

# Estimacion de rho_s en VALIDACION (anti-leakage)
# True  -> lee estado/sigma/flujo ya calculados por step005_validar_hmm*.py
#          (sin reclasificar nada) y estima rho_s = correlación AR(1) de
#          z_t=flujo_t/sigma_t, ponderada exponencialmente por régimen
#          (ver _estimar_rho_val_fold). Guarda rho_s_0/rho_s_1/... en
#          preds_test por fold (una columna por estado — 2 o 3 según
#          N_ESTADOS) para que step006 los use.
# False -> comportamiento anterior (rho estimado en test con regimenes imputados)
ESTIMAR_RHO_EN_VAL  = True
H_MIN_RHO_VAL       = None   # None -> autodetectar como min(h) en val
MAX_GAP_DIAS_RHO    = 4      # max gap dias habiles entre origenes consecutivos
RHO_DEFAULT_VAL     = 0.3    # fallback si ningun regimen tiene pares suficientes
# Minimo de pares (z_{t-1}, z_t) por regimen para confiar en la estimacion.
# Si un regimen tiene menos pares que este umbral, se usa el fallback.
# Fallback: max(rhos de regimenes con datos suficientes), con piso de 0.3.
# La logica: regimen severo tiene mas persistencia que calma/moderado, asi
# que usar el maximo disponible es conservador y coherente con el proceso.
MIN_PARES_RHO_REGIMEN = 15   # bajar a 15 si la ventana de val es corta
TAU_EWMA_RHO_PARES  = 100.0  # vida-media (dias habiles) del peso exponencial
                             # sobre los PARES ya agrupados por regimen al
                             # estimar rho_s. Paleologo (Elements of
                             # Quantitative Investing): K=exp(-1/tau).
                             # tau=100 pondera muy suave — no penaliza fuerte
                             # a los pares lejanos. Esto NO es la sigma del
                             # HMM (esa sigue con su propio lambda_ewma=0.92).
# True -> si el pickle HMM del fold tiene diag_ok=False (state collapse),
# omitir estimacion de rho_s para ese fold — evita correlaciones basura.
# Requiere step005_validar_hmm_v6.py corrido con HMM_N_STARTS >= 1.
HMM_EXCLUIR_FOLDS_DEGENERADOS_WF = True

# ── Tipo de ventana ────────────────────────────────────────────────•───────────
# True  → EXPANDING: train_start fijo, train_end crece cada fold
# False → ROLLING  : ventana fija que desliza (idéntico a v2)
EXPANDING = True

# ── Fecha de inicio del PRIMER fold ───────────────────────────────────────────
# None    → usa la fecha minima disponible en la matriz de features.
# "YYYY-MM-DD" → el primer fold arranca en esa fecha aunque la matriz tenga
#   historia anterior (esa historia sigue disponible para features de lookback).
#   Util para excluir historia antigua con calidad de datos dudosa o con un
#   regimen estructural distinto, sin tocar la matriz.
FOLD_FECHA_INICIO = "2019-07-01"

# ── Tamaños de ventana ────────────────────────────────────────────────────────
# EXPANDING=True : VENTANA_TRAIN_AÑOS es el mínimo inicial; crece PASO_AÑOS/fold
# EXPANDING=False: VENTANA_TRAIN_AÑOS es el tamaño fijo (igual a v2)
VENTANA_TRAIN_AÑOS  = 3      # años de TRAIN iniciales / fijos
VENTANA_VAL_AÑOS    = 1    # años de VAL (solo Optuna) — 6 meses, igual que step004
VENTANA_TEST_AÑOS   = 0.5      # años de TEST (solo métricas OOS)
PASO_AÑOS           = 0.5      # desplazamiento / crecimiento entre folds

# ── Anti-leakage: purga + embargo + burn-in ──────────────────────────────────
# H_MAX_DIAS_HAB    : horizonte máximo de predicción (h_max en step001)
# LOOKBACK_MAX_DIAS : ventana máxima de lags del target usados como features
#                     (MA22 y equivalentes)
#
# PURGA = H_MAX + LOOKBACK, no max(H_MAX, LOOKBACK). Los dos efectos se
# ENCADENAN, no compiten:
#   Una fila en VAL con fecha t usa features que miran hasta LOOKBACK días
#   atrás (hasta t−22). Esa ventana puede tocar una fila de TRAIN con fecha
#   t−22, cuyo target se extiende H_MAX días hacia adelante (hasta t+53).
#   La contaminación alcanza t−22−75 en el peor caso ⟹ hay que purgar
#   H_MAX + LOOKBACK = 97 días, no 75.
#
# EMBARGO: la purga elimina el solapamiento MECÁNICO (el mismo dato en ambos
#   lados). El embargo cubre algo distinto: la autocorrelación serial en los
#   bordes. Dos observaciones separadas por pocos días siguen correlacionadas
#   —los flujos de liquidez tienen memoria fuerte— de modo que sin embargo la
#   frontera queda "tibia" y VAL/TEST se ven artificialmente fáciles.
#
# BURN_IN_DIAS_HAB : excluye los primeros días de TRAIN donde MA22 aún no
#   maduró. Es un problema distinto (features incompletas al inicio de la
#   serie), no de frontera entre bloques.
H_MAX_DIAS_HAB    = 75   # igual que h_max en step001
LOOKBACK_MAX_DIAS = 22   # ventana máxima de lags del target como feature
EMBARGO_DIAS_HAB  = LOOKBACK_MAX_DIAS   # autocorrelación serial en bordes

# Gap total, simétrico en ambas fronteras: TRAIN→VAL y VAL→TEST
PURGE_DIAS_HAB   = H_MAX_DIAS_HAB + LOOKBACK_MAX_DIAS + EMBARGO_DIAS_HAB   # 75+22+22 = 119
PURGE_VAL_TEST   = PURGE_DIAS_HAB                                          # simétrico
BURN_IN_DIAS_HAB = LOOKBACK_MAX_DIAS   # warm-up MA22 al inicio de TRAIN

# ── Grupos de horizonte (adoptado de step005_walk_forward_cv_4_final.py) ────
# Estrategia DIRECTA: un modelo por h exacto (h deja de ser feature), pero
# Optuna busca hiperparámetros UNA VEZ por grupo (sobre un h representativo)
# y esos mismos hiperparámetros se transfieren a todos los h del grupo —
# evita correr Optuna 74 veces (uno por h) manteniendo un modelo por h.
# "`" separado a propósito: el banco pre-reporta retiros a t+2 y
# depósitos a t+1 → señal cualitativamente distinta a horizontes mayores.
H_MIN    = 2
H_GRUPOS: dict = {
    "muy_corto": (list(range(H_MIN, 6)),                 3),   # h=2–5,   rep h=3
    "corto":     (list(range(6, 21)),                   13),   # h=6–20,  rep h=13
    "medio":     (list(range(21, 51)),                  35),   # h=21–50, rep h=35
    "largo":     (list(range(51, H_MAX_DIAS_HAB + 1)),  62),   # h=51–75, rep h=62
}

# ── Descomposición del target en h=2 (adoptado de step005_walk_forward_cv_4_2) ─
#
#   target(t, h=2)  =  D(t+2) − R(t+2)     y     R_conf_t2  =  R(t+2)
#
# R_conf_t2 NO es un predictor correlacionado: es EXACTAMENTE el componente de
# retiro del target a 2 días, confirmado por la banca (las excepciones se
# sancionan, así que son raras). Sin la descomposición el árbol gasta sus hojas
# aproximando una identidad con funciones escalón — con max_depth bajo tiene
# muy pocas hojas para representar una recta.
#
# Con la descomposición se entrena sobre D(t+2) = target + R_conf_t2 y se
# deshace después:   q_τ(target) = max(q_τ(D), 0) − R_conf_t2
#
# El recorte en 0 (D no puede ser negativo) hace que el piso físico
# target ≥ −R_conf_t2 se cumpla automáticamente, sin imponerlo aparte.
# Solo afecta a h=H_MIN; los demás horizontes quedan idénticos.
#
# ⚠ REQUISITO: la columna R_conf_t2 debe existir en la matriz de features y no
#   tener NaN en train/val/test del fold. Si falta, la descomposición se omite
#   con aviso y ese fold entrena h=2 como antes (degradación segura, no error).
DESCOMPONER_H2 = True
COL_R_CONF_H2  = "R_conf_t2"


def _destransformar_h2(preds: dict, r_conf: np.ndarray) -> dict:
    """
    Deshace la descomposición de h=2: pasa de cuantiles de D(t+2) a cuantiles
    del target.

        q_τ(target)  =  max( q_τ(D), 0 )  −  R_conf_t2

    Restar una constante desplaza todos los cuantiles por esa constante, así
    que la transformación es exacta y preserva el orden. El clip en 0 impone
    D ≥ 0 y, con ello, el piso físico target ≥ −R_conf_t2. Puede generar
    empates entre cuantiles bajos, lo que no viola la monotonía.
    """
    r = np.asarray(r_conf, dtype=float)
    return {tau: np.maximum(np.asarray(p, dtype=float), 0.0) - r
            for tau, p in preds.items()}


def _h2_descomponible(X_train, X_val, X_test) -> bool:
    """
    Verifica que la descomposición de h=2 sea aplicable en este fold: la
    columna R_conf_t2 debe existir y no tener NaN en ninguno de los tres
    conjuntos. Sin ella no se puede deshacer la transformación en predicción,
    de modo que aplicarla a medias produciría cuantiles en la escala
    equivocada — peor que no aplicarla.
    """
    if not DESCOMPONER_H2:
        return False
    for nombre, df in (("train", X_train), ("val", X_val), ("test", X_test)):
        if df is None or len(df) == 0:
            continue
        if COL_R_CONF_H2 not in df.columns:
            logger.warning(f"    h={H_MIN}: descomposición omitida — "
                          f"columna {COL_R_CONF_H2!r} ausente en {nombre}")
            return False
        if df[COL_R_CONF_H2].isna().any():
            logger.warning(f"    h={H_MIN}: descomposición omitida — "
                          f"{COL_R_CONF_H2!r} con NaN en {nombre}")
            return False
    return True


def get_hp_for_h(h_val: int, hp_grupos: dict) -> dict:
    """
    Devuelve los hiperparámetros del grupo al que pertenece h_val.
    hp_grupos: {nombre_grupo: dict_de_hiperparametros} — llenado una vez por
    fold tras correr Optuna sobre el h representativo de cada grupo.
    """
    for nombre, (hs, _rep) in H_GRUPOS.items():
        if h_val in hs:
            return hp_grupos[nombre]
    raise ValueError(f"h={h_val} no pertenece a ningún grupo de H_GRUPOS "
                     f"(rango esperado: {H_MIN}-{H_MAX_DIAS_HAB})")


def grupo_de_h(h_val: int) -> str:
    """Nombre del grupo (muy_corto/corto/medio/largo) al que pertenece h_val."""
    for nombre, (hs, _rep) in H_GRUPOS.items():
        if h_val in hs:
            return nombre
    raise ValueError(f"h={h_val} no pertenece a ningún grupo de H_GRUPOS")

# ── Diagnóstico de features (PARTE 7-bis) ────────────────────────────────────
# v3.7.0 — granularidad fold × h × τ (antes: promediado sobre h Y sobre τ).
#
# Las tres señales (gain TRAIN, block-perm VAL, SHAP VAL) YA se calculaban por
# cuantil y por horizonte; el código anterior las promediaba dos veces antes de
# guardarlas. Conservar el desglose NO cuesta cómputo adicional — solo memoria
# y figuras. Es la información necesaria para responder dos preguntas que la
# versión promediada no podía: ¿las colas se apoyan en features distintos al
# centro? ¿un feature que importa a h=3 sigue importando a h=60?
DIAGNOSTICO_FEATURES  = True
DIAG_BLOCK_SIZE       = 20    # tamaño de bloque para block-permutation (preserva autocorr.)
DIAG_N_REPEATS        = 3     # repeticiones de la permutación
DIAG_PERM_MAX_SAMPLES = None  # submuestreo contiguo de VAL (None = todo)
DIAG_SHAP_MAX_SAMPLES = 800   # muestras para SHAP por cuantil
DIAG_TOP_N            = 25    # features mostrados en heatmaps y barras

# Control de la huella de memoria del graficado. Con 8 folds × 8 τ × 3 señales
# son ~192 figuras individuales + 24 paneles + 8 de convergencia. Los paneles
# son los caros (folds lado a lado). Si aparece MemoryError, lo primero es
# DIAG_PLOTS_INDIVIDUALES=False: conserva los paneles comparativos, que son los
# que responden "¿cómo cambia entre folds?".
DIAG_PLOTS_INDIVIDUALES = True   # False → solo los paneles resumen
DIAG_PLOT_DPI           = 120    # el heatmap no gana nitidez por encima de esto
DIAG_PANEL_ANCHO_MAX    = 26.0   # pulgadas; techo del ancho de los paneles

# ── Pares cíclicos sin/cos ───────────────────────────────────────────────────
# v3.7.0 CAMBIO DE CONVENCIÓN (adoptado de step005_walk_forward_cv_4_3.py):
#
#   ANTES: el Δ conjunto D se repartía como D/√2 en cada componente y la
#          consolidación usaba la norma euclidiana √(I_sin² + I_cos²).
#   AHORA: el Δ conjunto completo va en la columna _sin, la columna _cos queda
#          en 0.0, y la consolidación SUMA sin+cos.
#
# Consecuencias que conviene tener presentes al comparar con corridas previas:
#   • perm: un lector externo que aplique la norma euclidiana al CSV nuevo
#     obtiene √(D² + 0²) = |D| — la MISMA magnitud que antes. Compatible.
#     Pero la columna perm_<base>_cos ahora vale exactamente 0.0 por diseño;
#     no es un bug.
#   • gain/shap: sus valores por componente NO se tocan (son reales), lo que
#     cambia es la consolidación en los gráficos: sum en vez de norma. Para un
#     par balanceado (a, a) la suma da 2a y la norma daba a√2 ≈ 1.41a, así que
#     los features cíclicos suben ~41% RELATIVO a los no cíclicos y el top-25
#     no coincidirá con el de corridas anteriores.
#
# Los pares se detectan automáticamente por sufijo _sin/_cos (ver
# _sincos_pairs), en vez de la lista fija de bases que usaba la versión previa:
# un feature cíclico nuevo en la matriz entra al diagnóstico sin tocar código.
#
# Se conservan las constantes de la convención antigua porque
# aux_comparar_features.py las importa; el pipeline de diagnóstico ya NO las usa.
_CICL_BASES_PERM = [
    "mes", "dias_sem",
    "dias_al_cierre_mes", "dias_al_cierre_trim", "dias_al_cierre_anio",
    "elec",
]
_CICL_PARES_PERM = {
    f"{b}_{s}": f"{b}_cyc"
    for b in _CICL_BASES_PERM
    for s in ("sin", "cos")
}

# ── Block-permutation POR FAMILIA ────────────────────────────────────────────
# _diag_perm_h baraja UNA columna a la vez, así que un grupo de features
# correlacionadas entre sí puede salir con perm individual ≈ 0 para TODAS
# aunque el grupo completo sí importe: el modelo compensa con las columnas
# correlacionadas que quedaron intactas. Es el único diagnóstico que distingue
# "no aporta" de "aporta pero es redundante consigo mismo".
#
# Acá se baraja TODA la familia con el MISMO vector de permutación de bloques.
# Los nombres que no existan en la matriz activa se ignoran en tiempo de
# ejecución (no truenan la corrida).
FAMILIAS_PERM: dict[str, list[str]] = {
    "calendario_cierre": ["dias_al_cierre_mes", "dias_desde_cierre_mes",
                          "dias_al_cierre_trim"],
    "pos_extremos":      ["esc_neto_min_pos", "esc_neto_max_pos", "esc_retiro_pos",
                          "esc_deposito_pos_ap", "esc_neto_max_pos_ap"],
    "pos_acumulado":     ["acum_neto_min_pos", "acum_neto_max_pos", "frec_flujo_pos"],
    "ccovn_bcr":         ["ccovn_sistema_lag1", "ccovn_bbva_lag1",
                          "var_ccovn_sistema_lag1", "var_ccovn_bbva_lag1"],
    "flujo_crudo":       ["sigma_flujo_ratio", "ma_flujo_5d", "ma_flujo_20d", "R_conf_t2"],
    "encaje_bbva":       ["avance_mes_lag1", "exceso_abs_lag1"],
}

# ── Modelo ────────────────────────────────────────────────────────────────────
QUANTILES        = [0.01, 0.05, 0.40, 0.50, 0.60,  0.95, 0.99]   # 7 cuantiles, igual que v4 (agrega 0.40/0.60 cerca de la mediana)
S_MIN_FACTOR     = 0.85
S_MAX_FACTOR     = 1.00   # paper recomienda s∈[0.05,0.10] sobre targets estandarizados

# ── Trials Optuna ─────────────────────────────────────────────────────────────
# True  → número de trials varía por cuantil (TRIALS_POR_TAU)
# False → número fijo para todos los cuantiles (TRIALS_FLAT)
ADAPTIVE_TRIALS  = False

TRIALS_FLAT      = 60        # usado cuando ADAPTIVE_TRIALS = False

# ── Paralelismo Optuna ────────────────────────────────────────────────────────
# Procesos Optuna en paralelo (uno por cuantil). Cada worker carga su copia de
# X_tr/X_va, construye su propio DMatrix de XGBoost (~3× los datos) y arranca un
# intérprete Python nuevo (spawn en Windows) → ~400 MB por proceso. Con 7 τ eso
# da ~2.8 GB de pico, suficiente para agotar un laptop de 16 GB en folds grandes
# (>70 k filas) y que el SO mate el proceso sin traceback.
#
#   None → auto-detecta según la RAM libre AL INICIO DE CADA FOLD (recomendado:
#          se adapta solo entre un laptop y un servidor, y entre folds)
#   int  → fuerza ese número de workers (reproducibilidad / debug)
MAX_WORKERS_OPTUNA = None

# RAM estimada por worker. Medido: ~0.22 GB en este dataset (954 obs × 23 feats);
# el margen cubre datasets mayores. Subirlo lanza menos workers, bajarlo más.
_MEM_POR_WORKER_GB = 0.5

# ── Diagnostico de memoria ────────────────────────────────────────────────────
# True → tracemalloc atribuye el crecimiento de RAM entre folds a la linea que
# lo asigna, y al cerrar cada fold sale un bloque [memtop] con los 10 sitios que
# mas crecieron. Cuesta ~2x en tiempo de asignacion, asi que se deja apagado
# salvo cuando se investiga una fuga.
#
# LIMITE IMPORTANTE: tracemalloc solo ve asignaciones de Python. Los boosters de
# XGBoost son objetos C++ y su memoria NO aparece aca — solo el wrapper Python,
# que son bytes. Por eso el resultado se lee en dos direcciones:
#   [memtop] muestra MB relevantes  → retencion en Python, se corrige soltando
#                                      la referencia que señala la linea
#   [memtop] no muestra casi nada   → el crecimiento esta en el heap de C++
#                                      (XGBoost); ahi la unica salida real es
#                                      un subproceso por fold
DIAGNOSTICO_MEMORIA = False

# Reserva que NUNCA se reparte entre workers. Sin ella la formula asigna toda la
# RAM libre y el proceso padre se queda sin aire a mitad del fold: sigue
# creciendo mientras los workers corren (acumula ~1,200 boosters por fold entre
# modelos_por_h y modelos_final_por_h), XGBoost pide bloques transitorios, y el
# SO necesita su parte. Medido: con ~3 GB libres la formula sin reserva autorizo
# 6 workers que solo usaron 1.31 GB, y aun asi XGBoost fallo al pedir 4.8 MB.
_RESERVA_PADRE_GB = 2.0

TRIALS_POR_TAU   = {         # usado cuando ADAPTIVE_TRIALS = True
    # Llaves = round(tau, 1) para QUANTILES = [0.01, 0.05, 0.40, 0.50, 0.60, 0.95, 0.99]
    0.0: 110,   # τ=0.01 → colas extremas, más difíciles de calibrar
    0.1: 110,   # τ=0.05
    0.4:  90,   # τ=0.40 → cerca de la mediana, converge rápido (igual que Q50)
    0.5:  90,   # τ=0.50 → cuantil central, converge más rápido
    0.6:  90,   # τ=0.60 → cerca de la mediana, converge rápido (igual que Q50)
    0.9: 110,   # τ=0.95
    1.0: 110,   # τ=0.99 → colas extremas
}

# ── Opciones de salida ────────────────────────────────────────────────────────
if not PARTICIONES:
    BANCOS_A_EVALUAR      = ["SISTEMA"]
else:
    # FOCO/RESTO solo existen si la matriz se generó con partición activa.
    # Pedirlos sobre la matriz v1 no falla al cargar: devuelve un DataFrame
    # vacío y el error recién aparece folds después, así que se resuelve el
    # nombre acá y se aborta temprano si la combinación no tiene sentido.
    if ENTIDAD == "SISTEMA":
        BANCO = "SISTEMA"
    elif ENTIDAD in ("FOCO", "RESTO"):
        BANCO = f"{ENTIDAD}_{PARTICION.upper()}"
    else:
        raise ValueError(
            f"ENTIDAD={ENTIDAD!r} no es válida. Opciones: 'SISTEMA', 'FOCO', "
            f"'RESTO'.")
    BANCOS_A_EVALUAR      = [BANCO]
    # v3.7 conserva la lista —a diferencia de v4_particiones, que corre una
    # entidad por corrida— así que las dos caras se pueden hacer de una sola
    # pasada. Con el subnivel de carpeta de dirs_de_banco() ya no se pisan:
    #   BANCOS_A_EVALUAR = [f"FOCO_{PARTICION.upper()}",
    #                       f"RESTO_{PARTICION.upper()}", "SISTEMA"]
GUARDAR_MODELO_FINAL      = True
# True  → guarda modelos de TODOS los folds (permite fan chart histórico sin lookahead)
# False → solo guarda el último fold (comportamiento anterior)
GUARDAR_MODELOS_TODOS_FOLDS = True

# True  → omite Optuna/entrenamiento, carga modelos del disco y solo regenera los plots
# False → entrenamiento completo (comportamiento normal)
SOLO_REGENERAR_PLOTS = False
COLS_EXCLUIR              = {"fecha_t", "banco", "target", "h", "log_h"}

# ── Límite de folds ───────────────────────────────────────────────────────────
# None → usa todos los folds generados
# N    → usa solo los primeros N folds (los más antiguos); deja el resto como OOS
# Expanding=True: 9 folds (fold 9 incluye test 2023+)
# Expanding=False (rolling): 8 folds
N_MAX_FOLDS = 9 if EXPANDING else 8

# ── Selector de modelo ────────────────────────────────────────────────────────
MODELO_CV = "xgb_qt"
# Opciones: "xgb" | "lgbm" | "xgb_qt"
assert MODELO_CV in ("xgb", "lgbm", "xgb_qt"), \
    f"MODELO_CV debe ser 'xgb', 'lgbm' o 'xgb_qt', recibido: {MODELO_CV!r}"

# ── Parámetro s (suavizado Pinball-Arctan) ────────────────────────────────────
# True  → s fijo en S_FACTOR_FIJO × std_y (recomendado por el paper 2406.02293)
#          Optuna no busca s; libera trials para otros hiperparámetros
# False → Optuna busca s en [S_MIN_FACTOR, S_MAX_FACTOR] × std_y
S_FIJO         = False
S_FACTOR_FIJO  = 0.05   # equivale a s=0.05 en datos estandarizados (centro del rango paper)

# ── Calibración post-hoc (shift aditivo estimado en VAL) ─────────────────────
# True  → después de predecir, calcula el sesgo sistemático en VAL y lo corrige
#         en TEST: todas las predicciones se desplazan por el percentil P del
#         residuo (actual − Q50_pred) en VAL.
#         P=50 → elimina sesgo (mediana de errores)
#         P<50 → introduce sesgo negativo (conservador para riesgo de liquidez)
#         P=25 → desplaza hacia abajo con sesgo negativo moderado
# False → sin calibración (comportamiento original)
CALIBRACION_POSTHOC   = False
CALIBRACION_PERCENTIL = 50    # percentil del residuo VAL usado como shift
# Límite del shift como fracción de std_y: previene correcciones exageradas
# que distorsionan el eje Y cuando el modelo tiene sesgo grande.
# None → sin límite (el valor original antes de esta corrección)
CALIBRACION_MAX_SHIFT_FACTOR = 0.5   # e.g. 0.5 → shift ≤ ±0.5×std_y


# -- Overlay sobreencaje (step007) --------------------------------------------
# Requiere haber ejecutado step007 para generar saldos_retiros_bancos.xlsx.
# El ajuste diario (peor_total) se lee desde la tab "Ajuste_diario" de ese archivo.
OVERLAY_SOBREENCAJE             = True
RUTA_AJUSTE_OVERLAY             = BASE_SISTEMA / "2. Output" / "analisis_cc" / "saldos_retiros_bancos.xlsx"
OVERLAY_VENTANA_DH              = 7   # dias habiles de la ventana de retiro (mismo valor que step007)
OVERLAY_TAU_REFERENCIA          = 0.05  # quantil usado como denominador del factor
OVERLAY_CONOCIMIENTO_ANTICIPADO = 0   # T+N: flujos conocidos con N dias habiles de antelacion

# Calendario de días hábiles PER + USA para el overlay.
# Se combinan ambos conjuntos de feriados porque los flujos D-R involucran
# operaciones en moneda extranjera (USD) que no liquidan en feriados de EE.UU.
#
# Preferencia: librería `holidays` (calcula feriados variables como Thanksgiving,
# Viernes Santo, MLK Day, etc.). Si no está instalada, se usa una lista curada
# con los feriados fijos más relevantes para la ventana de diciembre.
_ANOS_FERIADOS = range(2018, 2032)

try:
    import holidays as _hlib
    _dias_pe = _hlib.Peru(years=_ANOS_FERIADOS)
    _dias_us = _hlib.UnitedStates(years=_ANOS_FERIADOS)
    _FERIADOS_PEUSA = pd.to_datetime(
        sorted(set(list(_dias_pe.keys()) + list(_dias_us.keys())))
    )
    logger.info(f"[BDAY] Calendario PER+USA cargado via 'holidays' "
                f"({len(_FERIADOS_PEUSA)} días, {_ANOS_FERIADOS.start}-{_ANOS_FERIADOS.stop-1})")
except ImportError:
    # Fallback: feriados fijos PER + feriados fijos USA más relevantes (2018-2031)
    # PER: Año Nuevo, Jueves/Viernes Santo*, Día del Trabajo, Fiestas Patrias,
    #      Sta. Rosa de Lima, Combate de Angamos, Todos los Santos,
    #      Inmaculada Concepción, Nochebuena, Navidad
    # USA: Año Nuevo, MLK Day*, Pres. Day*, Memorial Day*, Juneteenth,
    #      Independence Day, Labor Day*, Columbus Day*, Veterans Day,
    #      Thanksgiving*, Christmas
    # (*) feriados variables no incluidos en el fallback — instalar `holidays`
    #     para cobertura completa: pip install holidays
    _f_pe_fijos = pd.to_datetime(
        [f"{y}-01-01" for y in _ANOS_FERIADOS]   # Año Nuevo PER
        + [f"{y}-05-01" for y in _ANOS_FERIADOS]  # Día del Trabajo PER
        + [f"{y}-06-29" for y in _ANOS_FERIADOS]  # San Pedro y San Pablo PER
        + [f"{y}-07-28" for y in _ANOS_FERIADOS]  # Fiestas Patrias PER (día 1)
        + [f"{y}-07-29" for y in _ANOS_FERIADOS]  # Fiestas Patrias PER (día 2)
        + [f"{y}-08-30" for y in _ANOS_FERIADOS]  # Sta. Rosa de Lima PER
        + [f"{y}-10-08" for y in _ANOS_FERIADOS]  # Combate de Angamos PER
        + [f"{y}-11-01" for y in _ANOS_FERIADOS]  # Todos los Santos PER
        + [f"{y}-12-08" for y in _ANOS_FERIADOS]  # Inmaculada Concepción PER
        + [f"{y}-12-24" for y in _ANOS_FERIADOS]  # Nochebuena PER
        + [f"{y}-12-25" for y in _ANOS_FERIADOS]  # Navidad PER/USA
    )
    _f_us_fijos = pd.to_datetime(
        [f"{y}-07-04" for y in _ANOS_FERIADOS]   # Independence Day USA
        + [f"{y}-06-19" for y in _ANOS_FERIADOS]  # Juneteenth USA (desde 2021)
        + [f"{y}-11-11" for y in _ANOS_FERIADOS]  # Veterans Day USA
    )
    _FERIADOS_PEUSA = pd.to_datetime(
        sorted(set(_f_pe_fijos.tolist() + _f_us_fijos.tolist()))
    )
    logger.warning(
        "[BDAY] Librería 'holidays' no encontrada — usando lista de feriados fijos PER+USA. "
        "Instalar con: pip install holidays"
    )

BDAY_PE = CustomBusinessDay(holidays=_FERIADOS_PEUSA)


# Limita el salto máximo de cada árbol para evitar overshooting con gradientes
# grandes. El paper usa 0.5 sobre targets estandarizados (std_y≈1); para datos
# sin estandarizar el equivalente es 0.5 × std_y por fold.
# None → desactivado (comportamiento anterior)
MAX_DELTA_STEP_FACTOR = None   # multiplica std_y de cada fold

# ── Fan chart TEST: número de snapshots por fold ──────────────────────────────
FANCHART_N_SNAPSHOTS = 4   # 1 cada ~3 meses para TEST de 1 año

# ── Diagnóstico de features (PARTE 7-bis) ─────────────────────────────────────
# True  → corre gain / block-perm / SHAP por fold y genera los gráficos
# False → omite el diagnóstico (más rápido)
DIAGNOSTICO_FEATURES  = True
DIAG_BLOCK_SIZE       = 20    # filas por bloque en la permutación (preserva autocorrelación)
DIAG_N_REPEATS        = 3     # repeticiones por feature para estabilizar la estimación
DIAG_SHAP_MAX_SAMPLES = 800   # máximo de filas VAL para SHAP (None = todas)

###############################################################################
# MODO DEBUG — corrida rápida para diagnóstico, NO para resultados
###############################################################################
# True → recorta el trabajo para que dos folds terminen en ~10 min en vez de
# ~60, y enciende la instrumentación de memoria. Pensado para localizar la
# retención de RAM del proceso padre (ver DIAGNOSTICO_MEMORIA), no para producir
# métricas: con estos valores los modelos están submuestreados y submuestreados
# y sus resultados NO son comparables con una corrida normal.
#
# Reparto del tiempo medido en un fold real (30.2 min, 74 horizontes):
#   [diag] gain+perm+shap  11.0 min (36%)  → DIAG_N_REPEATS, DIAG_SHAP_MAX_SAMPLES
#   retrain_tv             10.4 min (34%)  → H_GRUPOS
#   Optuna                  7.4 min (25%)  → TRIALS_FLAT
#   fanchart + guardado      1.4 min (5%)  → sin tocar
#
# Criterio de qué se recorta: se ESCALAN las fases, no se apagan. Poner
# DIAGNOSTICO_FEATURES = False ahorraría los 11 min de golpe, pero diag_rows
# vive ahí dentro y es uno de los sospechosos de la retención: apagarlo daría
# un falso negativo — "no hay fuga" cuando lo que se apagó fue la causa.
MODO_DEBUG = False

if MODO_DEBUG:
    # 74 horizontes → 8. Cada h_rep (3, 13, 35, 62) se conserva dentro de su
    # lista: si no estuviera, la búsqueda de HP del grupo aborta. h=2 se
    # mantiene para que la descomposición R_conf_t2 se siga ejercitando.
    H_GRUPOS = {
        "muy_corto": ([2, 3],    3),
        "corto":     ([6, 13],  13),
        "medio":     ([21, 35], 35),
        "largo":     ([51, 62], 62),
    }
    TRIALS_FLAT           = 15    # 60 → 15
    DIAG_N_REPEATS        = 1     # 3 → 1   (permutación)
    DIAG_SHAP_MAX_SAMPLES = 150   # 800 → 150
    DIAGNOSTICO_MEMORIA   = True  # el objetivo de la corrida
    # DIAGNOSTICO_FEATURES se deja en True a propósito (ver arriba).

# ── Comparación con Step004 en fan charts ─────────────────────────────────────
# True  → superpone predicciones del modelo step004 (línea naranja discontinua)
#          para comparar visualmente con step005 fold-by-fold
# False → solo muestra predicciones step005 (comportamiento normal)
COMPARAR_CON_STEP004 = False
# Directorio donde están los modelos step004 (eval = entrenado solo hasta TRAIN)
_STEP004_SUFIJO = {
    "xgb"   : ("modelos_xgb",    "eval", "xgb"),
    "xgb_qt": ("modelos_xgb_qt", "eval", "xgb_qt"),
    "lgbm"  : ("modelos_lgbm",   "eval", "lgbm"),
}
_s4_carpeta, _s4_subcarpeta, _s4_prefijo = _STEP004_SUFIJO.get(
    MODELO_CV, ("modelos_xgb", "eval", "xgb")
)
DIR_MODELOS_STEP004 = BASE_SISTEMA / "2. Output" / _s4_carpeta / _s4_subcarpeta

# ── Folds manuales — comparación directa con step004 ─────────────────────────
# Lista de folds con fechas exactas, añadidos a (o reemplazando) los folds auto.
# Permite replicar el split de step004 sin embargo entre TRAIN y VAL.
# Descomentar el ejemplo para activar el fold equivalente a step004:
FOLDS_MANUALES: list[dict] = []
# FOLDS_MANUALES = [
#     {
#         "train_start": "2015-01-02",   # inicio real de los datos
#         "train_end"  : "2022-06-30",   # mismo corte que step004 (CORTE_VAL − 1d)
#         "val_start"  : "2022-07-01",   # sin embargo, igual que step004
#         "val_end"    : "2023-01-02",   # CORTE_TEST − 1d
#         "test_start" : "2023-01-03",   # mismo CORTE_TEST que step004
#         "test_end"   : "2024-06-30",   # hasta donde haya datos
#     }
# ]

# True  → corre SOLO los folds manuales (omite los generados automáticamente)
# False → añade los folds manuales al final de los generados
SOLO_FOLDS_MANUALES = False

# ── Rutas de salida ───────────────────────────────────────────────────────────
_modo           = "expanding" if EXPANDING else "rolling"
_ventanas       = f"{VENTANA_TRAIN_AÑOS}{VENTANA_VAL_AÑOS}{VENTANA_TEST_AÑOS}"
_DIR_BASE       = DIR_OUTPUT / f"{MODELO_CV}_{_modo}_{_ventanas}"


def _fmt_anios(x: float) -> str:
    """0.5 -> '0.5', 1.0 -> '1'. Sin el .0 colgando, que ensucia la ruta."""
    return f"{x:g}"


def etiqueta_corrida(banco: str) -> str:
    """
    Identidad de la corrida: entidad + geometría del fold. Ej: FOCO_BBVA_1_0.5

    Mismo formato que etiqueta_corrida() de
    step005_walk_forward_cv_4_particiones.py, que es el que reconstruye
    aux_fanchart_cv4_direct_particiones.py para encontrar las predicciones. Las
    definiciones tienen que moverse juntas.

    Solo entra en la ruta con PARTICIONES=True (ver dirs_de_banco).
    """
    return f"{banco}_{_fmt_anios(VENTANA_VAL_AÑOS)}_{_fmt_anios(VENTANA_TEST_AÑOS)}"


def dirs_de_banco(banco: str) -> dict:
    """
    Carpetas de salida de una entidad, creadas si no existen.

    Con PARTICIONES=False devuelve EXACTAMENTE las rutas de siempre —sin
    subnivel— así que el archivo se comporta igual que antes de este botón.
    Con True agrega un nivel por entidad, porque los diagnósticos que no llevan
    el banco en el nombre del archivo se sobrescribirían entre corridas de
    entidades distintas.
    """
    dm = _DIR_BASE / etiqueta_corrida(banco) if PARTICIONES else _DIR_BASE
    d = {"modo"          : dm,
         "modelos"       : dm / "modelos",
         "plots"         : dm / "plots",
         "fancharts"     : dm / "fancharts_test",
         "fancharts_man" : dm / "fancharts_manuales"}   # plots de FOLDS_MANUALES
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


# Valores iniciales. Con PARTICIONES=False son los definitivos —nada los
# reasigna— y coinciden con los literales que había antes de este cambio.
_d0 = dirs_de_banco(BANCOS_A_EVALUAR[0])
DIR_MODO               = _d0["modo"]
DIR_MODELOS            = _d0["modelos"]
DIR_PLOTS              = _d0["plots"]
DIR_FANCHARTS          = _d0["fancharts"]
DIR_FANCHARTS_MANUALES = _d0["fancharts_man"]

# Guarda del bug latente: sin subnivel por entidad, correr más de un banco hace
# que el segundo pise los diagnósticos del primero. Antes no podía ocurrir
# porque la lista traía un solo elemento; esto lo impide si alguien la amplía
# sin activar el botón.
if not PARTICIONES and len(BANCOS_A_EVALUAR) > 1:
    raise ValueError(
        f"BANCOS_A_EVALUAR trae {len(BANCOS_A_EVALUAR)} entidades con "
        f"PARTICIONES=False. Los diagnósticos que no llevan el banco en el "
        f"nombre del archivo (convergencia_<tau>.png, panel_<senal>_<tau>.png, "
        f"<senal>_fold<NN>_<tau>.png) se pisarían entre entidades. Activar "
        f"PARTICIONES=True, que separa la salida por entidad.")


def get_n_trials(tau: float) -> int:
    """Devuelve el número de trials Optuna para el cuantil dado."""
    if ADAPTIVE_TRIALS:
        return TRIALS_POR_TAU.get(round(tau, 1), 90)
    return TRIALS_FLAT


def get_max_workers_optuna() -> int:
    """
    Número de procesos Optuna que caben en la RAM libre en este momento.

    Se llama al inicio de cada fold (no una sola vez al importar) porque la RAM
    disponible cae a medida que el proceso padre acumula datos: medir en el fold
    1 y reutilizar ese valor en el fold 3 sería optimista justo donde importa.

    Si MAX_WORKERS_OPTUNA trae un int, respeta ese valor y no mide nada.
    Detección en cascada: psutil → /proc/meminfo → fallback conservador (3).
    """
    if MAX_WORKERS_OPTUNA:
        return MAX_WORKERS_OPTUNA

    n_tau    = len(QUANTILES)
    avail_gb = None

    # Libera lo que dejó el fold anterior para no medir basura como si fuera
    # memoria ocupada — sin esto la estimación sale pesimista tras el fold 1.
    gc.collect()

    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
    except ImportError:
        try:                                   # Linux / WSL sin psutil
            with open("/proc/meminfo") as fmem:
                for line in fmem:
                    if line.startswith("MemAvailable:"):
                        avail_gb = int(line.split()[1]) / 1e6   # kB → GB
                        break
        except OSError:
            pass                               # Windows sin psutil

    if avail_gb is None:                       # no se pudo medir → asumir lo peor
        return min(n_tau, 3)

    # Solo se reparte lo que sobra tras la reserva del padre, no toda la RAM libre.
    usable_gb = avail_gb - _RESERVA_PADRE_GB
    if usable_gb <= 0:                         # RAM al limite → secuencial
        logger.warning(f"    [mem] solo {avail_gb:.1f} GB libres (< reserva de "
                       f"{_RESERVA_PADRE_GB} GB) → 1 worker. Reinicia el kernel "
                       f"de Spyder o corre desde terminal para liberar RAM.")
        return 1

    return max(1, min(n_tau, int(usable_gb / _MEM_POR_WORKER_GB)))


def _monitor_rss_hijos(stop_evt, out, intervalo=2.0):
    """
    Muestrea el RSS agregado de los procesos hijo hasta que stop_evt se active.

    Sirve para calibrar _MEM_POR_WORKER_GB con medición real en vez de la
    estimación analítica. Corre en un hilo daemon: no bloquea la salida del
    script ni altera el resultado del modelo.

    Guarda el pico y cuántos procesos había EN ESE MISMO INSTANTE. Rastrear el
    máximo del total y el máximo del conteo por separado daría un GB/worker
    falso, al dividir el pico de un momento entre el conteo de otro.
    """
    try:
        import psutil
    except ImportError:
        return                              # sin psutil no se mide, y no pasa nada

    padre = psutil.Process()
    while not stop_evt.wait(intervalo):     # wait() → True apenas se pide parar
        try:
            hijos = padre.children(recursive=True)
            if not hijos:
                continue
            total = 0
            for h in hijos:
                try:
                    total += h.memory_info().rss
                except psutil.Error:
                    pass                    # el worker terminó entre listar y medir
            if total > out.get("pico", 0):
                out["pico"]      = total
                out["n_en_pico"] = len(hijos)
        except psutil.Error:
            pass


def _guardar_preds_test(preds, y_real, h_arr, fechas_t,
                        fold_num, banco, fecha_hoy, dir_out,
                        regimen_hmm=None, regimen_sigma=None,
                        año_corte_regimen=None, rho_s_val=None, rho_ij=None):
    """
    Guarda las predicciones TEST por fold.
    - regimen_hmm / regimen_sigma: estado HMM y sigma de CADA fila de test.
    - año_corte_regimen: bloque HMM usado (constante por fold).
    - rho_s_val [NUEVO]: dict {estado: rho} estimado en VALIDACION con los
      parametros HMM del fold de TRAIN — z_t=flujo_t/sigma_t, correlación
      AR(1) ponderada exponencialmente por régimen (ver _estimar_rho_val_fold).
      Se guarda como columnas rho_s_0/rho_s_1/... (una por cada estado
      presente en el dict — 2 o 3 según N_ESTADOS de step005_validar_hmm*.py,
      constantes por fold). step006_orquestador.py las lee en vez de estimar
      rho sobre test donde los regimenes eran NaN/mediana imputados.
    """
    rows = {f"q{int(tau*100):02d}": arr
            for tau, arr in preds.items() if tau != "mean"}
    df = pd.DataFrame({
        "fecha_t":      pd.DatetimeIndex(fechas_t),
        "h":            h_arr.astype(int),
        "y_realizado":  y_real,
        **rows,
    })
    if regimen_hmm is not None:
        df["regimen_hmm"] = np.asarray(regimen_hmm)
    if regimen_sigma is not None:
        df["regimen_sigma"] = np.asarray(regimen_sigma)
    if año_corte_regimen is not None:
        df["año_corte_regimen"] = año_corte_regimen
    if rho_s_val is not None:
        for s in sorted(rho_s_val.keys()):
            df[f"rho_s_{s}"] = float(rho_s_val[s])
    if rho_ij:
        # Constantes por fold, igual que rho_s_*. Nombres: rho_ij (global),
        # rho_ij_s<estado> (condicional) y n_pares_rho_ij para poder juzgar
        # cuanta muestra hay detras al armar R en la simulacion conjunta.
        df["rho_ij"]         = float(rho_ij["global"])
        df["n_pares_rho_ij"] = int(rho_ij.get("n_pares", 0))
        df["contraparte_rho_ij"] = str(rho_ij.get("contraparte", ""))
        for k, v in rho_ij.items():
            if isinstance(k, int):
                df[f"rho_ij_s{k}"] = float(v)
    ruta = dir_out / f"preds_test_fold{fold_num:02d}_{banco}_{fecha_hoy}.parquet"
    df.to_parquet(ruta, index=False)
    logger.info(f"    Preds TEST fold {fold_num} guardadas: {ruta.name}")


###############################################################################
# PARTE 1 — Métricas
###############################################################################

def pinball_loss(y_true, y_pred, tau):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valido = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if not valido.any():
        return float("nan")
    e = y_true[valido] - y_pred[valido]
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def winkler_score(y_true, q_lo, q_hi, alpha=0.10):
    y_true = np.asarray(y_true, dtype=float)
    q_lo   = np.asarray(q_lo, dtype=float)
    q_hi   = np.asarray(q_hi, dtype=float)
    valido = ~np.isnan(q_lo) & ~np.isnan(q_hi) & ~np.isnan(y_true)
    if not valido.any():
        return float("nan")
    width = q_hi[valido] - q_lo[valido]
    below = np.maximum(q_lo[valido] - y_true[valido], 0)
    above = np.maximum(y_true[valido] - q_hi[valido], 0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def coverage(y_true, q_lo, q_hi):
    y_true = np.asarray(y_true, dtype=float)
    q_lo   = np.asarray(q_lo, dtype=float)
    q_hi   = np.asarray(q_hi, dtype=float)
    valido = ~np.isnan(q_lo) & ~np.isnan(q_hi) & ~np.isnan(y_true)
    if not valido.any():
        return float("nan")
    return float(np.mean((y_true[valido] >= q_lo[valido]) & (y_true[valido] <= q_hi[valido])))


CRPS_N_GRID = 201   # puntos de τ para integrar el CRPS (converge ya en ~101)

# np.trapz fue eliminado en numpy 2.0 (renombrado a np.trapezoid)
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def crps_approx(y_true, preds, n_grid: int = CRPS_N_GRID):
    """
    CRPS por reconstrucción de la función cuantil.

        CRPS = 2 ∫₀¹ pinball_τ(y, Q(τ)) dτ

    El método anterior ponderaba los pinballs de los τ disponibles con pesos
    (hi−lo)/2 —regla trapezoidal directa sobre los nodos— y subestimaba el
    CRPS ~9.4% de forma sistemática. El sesgo NO venía de las colas (0.1–0.4%)
    sino de los huecos de la grilla de τ: entre 0.05→0.40 y 0.60→0.95 hay
    saltos de 0.35, y ahí la recta del trapecio corta la curvatura real de la
    curva pinball(τ) — 12.4% y 12.5% de pérdida en esos dos tramos.

    Aquí se reconstruye Q(τ) en una grilla densa con PCHIP y se integra sobre
    ella. Se interpola Q(τ) —no la curva de pinball— porque Q es monótona por
    definición y PCHIP preserva monotonía (un spline cúbico normal podría
    producir cuantiles cruzados). Una vez estimado Q(τ), el pinball en ese
    punto se evalúa de forma exacta, no aproximada.

    Error resultante <0.2% frente a <-9.4% del método anterior, sin reentrenar
    nada: solo cambia cómo se integra a partir de los cuantiles ya predichos.

    ⚠ Los valores dejan de ser comparables con los CRPS calculados antes de
      este cambio (los nuevos son ~9% mayores porque el método viejo
      subestimaba, no porque el modelo empeore).
    """
    from scipy.interpolate import PchipInterpolator

    taus = sorted(preds.keys())
    if len(taus) < 2:
        return float("nan")

    y = np.asarray(y_true, dtype=float)
    Q = np.array([np.asarray(preds[t], dtype=float) for t in taus])   # (n_taus, n_obs)

    valido = ~np.isnan(y) & ~np.isnan(Q).any(axis=0)
    if not valido.any():
        return float("nan")
    y, Q = y[valido], Q[:, valido]

    # Monotonía en τ: PCHIP la exige, y un cruce de cuantiles la rompería.
    Q = np.sort(Q, axis=0)

    tau_grid = np.linspace(taus[0], taus[-1], n_grid)
    Q_grid   = PchipInterpolator(taus, Q, axis=0)(tau_grid)   # (n_grid, n_obs)

    e   = y[None, :] - Q_grid
    t   = tau_grid[:, None]
    pbs = np.mean(np.where(e >= 0, t * e, (t - 1.0) * e), axis=1)

    return float(2.0 * _TRAPZ(pbs, tau_grid))


###############################################################################
# PARTE 2 — GARCH por fold
###############################################################################

# Número de procesos paralelos (uno por cuantil) y threads XGBoost por proceso.
# Con ProcessPoolExecutor cada proceso tiene su propio GIL → paralelismo real.
# Reparto óptimo: N procesos × (cpu_count // N) threads = ~cpu_count cores activos.
# Derivado de QUANTILES (no hardcodeado) — con 7 taus da 7, no 5.
_N_QUANTILES_PARALLEL = len(QUANTILES)
_XGB_NTHREAD = max(2, (os.cpu_count() or 10) // _N_QUANTILES_PARALLEL)

# Cache de parámetros GARCH por fecha de corte de TRAIN — evita re-estimación en el
# mismo fold y para el guardado de metadata (antes se estimaba 2-3 veces por fold).
_garch_params_cache: dict[str, dict] = {}

def _ajustar_garch_params(x_train):
    n       = len(x_train)
    var_unc = max(float(np.var(x_train)), 1e-12)

    def _s2(omega, alpha, beta):
        s2    = np.empty(n)
        s2[0] = var_unc
        for t in range(1, n):
            s2[t] = omega + alpha * x_train[t - 1] ** 2 + beta * s2[t - 1]
        return s2

    def _neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.9999:
            return 1e10
        s2 = _s2(omega, alpha, beta)
        if np.any(s2 <= 0):
            return 1e10
        return 0.5 * float(np.sum(np.log(s2) + x_train ** 2 / s2))

    try:
        res = minimize(
            _neg_loglik, [0.01, 0.08, 0.88], method="L-BFGS-B",
            bounds=[(1e-7, 0.5), (1e-7, 0.5), (1e-7, 0.9999)],
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        if res.fun < 1e9:
            return float(res.x[0]), float(res.x[1]), float(res.x[2])
    except Exception as _e_gev:
        logger.debug(f"_fit_weibull_gevlike: optimize falló → "
                     f"{type(_e_gev).__name__}: {_e_gev} — usando parámetros por defecto")
    return 0.01, 0.08, 0.88


def _garch_vol_fold(serie, train_end):
    serie_full  = serie.ffill().fillna(0.0)
    serie_train = serie_full[serie_full.index <= train_end]
    if len(serie_train) < 60 or serie_train.std() < 1e-9:
        return serie_full.rolling(20).std().fillna(serie_train.std())
    escala  = float(serie_train.std())
    x_train = (serie_train / escala).values.astype(float)
    var_unc = max(float(np.var(x_train)), 1e-12)
    omega, alpha, beta = _ajustar_garch_params(x_train)
    x_full = (serie_full / escala).values.astype(float)
    n_full = len(x_full)
    s2     = np.empty(n_full)
    s2[0]  = var_unc
    for t in range(1, n_full):
        s2[t] = omega + alpha * x_full[t - 1] ** 2 + beta * s2[t - 1]
    return pd.Series(np.sqrt(np.maximum(s2, 0)) * escala, index=serie_full.index)


def reemplazar_ffd_fold(df_fold, train_end):
    """
    Re-calibra FFD para cada columna _frac usando solo datos hasta train_end.
    Preserva la misma d_opt que la calibrada en build_feature_matrix pero
    garantiza que no se filtra información posterior a train_end.
    """
    frac_cols = [c for c in df_fold.columns if c.endswith("_frac")]
    if not frac_cols:
        return df_fold
    df_fold = df_fold.copy()

    try:
        from statsmodels.tsa.stattools import adfuller
        _statsmodels_ok = True
    except ImportError:
        _statsmodels_ok = False

    def _ffd_weights_local(d, thresh=1e-5):
        w, k = [1.0], 1
        while True:
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < thresh:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1])

    def _fracdiff_local(series, d, thresh=1e-5):
        w = _ffd_weights_local(d, thresh)
        width = len(w)
        vals = series.values.astype(float)
        out  = np.full(len(vals), np.nan)
        for i in range(width - 1, len(vals)):
            chunk = vals[i - width + 1: i + 1]
            if not np.any(np.isnan(chunk)):
                out[i] = float(np.dot(w, chunk))
        return pd.Series(out, index=series.index)

    def _find_d_local(series, n_steps=20, target_pval=0.05):
        if not _statsmodels_ok or len(series.dropna()) < 30:
            return 0.4
        for d in np.linspace(0.05, 1.0, n_steps):
            fd = _fracdiff_local(series.dropna(), round(float(d), 4))
            fd_clean = fd.dropna()
            if len(fd_clean) < 20:
                continue
            try:
                pval = adfuller(fd_clean, maxlag=1, regression="c", autolag=None)[1]
                if pval <= target_pval:
                    return round(float(d), 4)
            except Exception as _e_adf:
                logger.debug(f"FFD ADF test falló en d={d:.2f} → "
                             f"{type(_e_adf).__name__}: {_e_adf}")
                continue
        return 1.0

    idx_col = "fecha_t"
    raw = (df_fold[[idx_col] + [c.replace("_frac", "") for c in frac_cols
                                if c.replace("_frac", "") in df_fold.columns]]
           .drop_duplicates(idx_col).set_index(idx_col).sort_index())

    for col_frac in frac_cols:
        col_raw = col_frac.replace("_frac", "")
        if col_raw not in raw.columns:
            continue
        serie_train = raw[col_raw][raw.index <= train_end].dropna()
        if len(serie_train) < 60:
            continue
        d_opt = _find_d_local(serie_train)
        frac_full = _fracdiff_local(raw[col_raw], d_opt)
        df_fold[col_frac] = df_fold[idx_col].map(frac_full)

    return df_fold


def reemplazar_garch_fold(df_fold, train_end):
    df_fold = df_fold.copy()
    idx_cols = ["fecha_t", "R_t0", "D_t0", "TC_PEN_USD", "EMBI_PERU"]
    avail    = [c for c in idx_cols if c in df_fold.columns]
    raw      = (df_fold[avail].drop_duplicates("fecha_t")
                .set_index("fecha_t").sort_index())
    if {"R_t0", "D_t0"}.issubset(raw.columns):
        sigma = _garch_vol_fold(raw["D_t0"] - raw["R_t0"], train_end)
        df_fold["garch_vol"] = df_fold["fecha_t"].map(sigma)
    if "TC_PEN_USD" in raw.columns:
        tc  = raw["TC_PEN_USD"].replace(0, np.nan).ffill()
        tci = tc.reindex(pd.bdate_range(tc.index.min(), tc.index.max())).ffill()
        ret = np.log(tci / tci.shift(1)).reindex(tc.index)
        df_fold["garch_vol_tc"] = df_fold["fecha_t"].map(_garch_vol_fold(ret, train_end))
    if "EMBI_PERU" in raw.columns:
        df_fold["garch_vol_embi"] = df_fold["fecha_t"].map(
            _garch_vol_fold(raw["EMBI_PERU"].diff(1), train_end))
    return df_fold


def _extraer_garch_params_fold(df, train_end):
    """
    Estima parámetros GARCH(1,1) usando solo datos de TRAIN (hasta train_end).
    Resultado cacheado por fecha de corte para no re-estimar en el mismo fold.
    """
    cache_key = str(train_end.date()) if hasattr(train_end, "date") else str(train_end)
    if cache_key in _garch_params_cache:
        return _garch_params_cache[cache_key]

    params   = {}
    idx_cols = ["fecha_t", "R_t0", "D_t0", "TC_PEN_USD", "EMBI_PERU"]
    avail    = [c for c in idx_cols if c in df.columns]
    raw      = (df[avail].drop_duplicates("fecha_t")
                .set_index("fecha_t").sort_index())

    def _params_serie(serie):
        serie_full  = serie.ffill().fillna(0.0)
        serie_train = serie_full[serie_full.index <= train_end]
        if len(serie_train) < 60 or serie_train.std() < 1e-9:
            return None
        escala  = float(serie_train.std())
        x_train = (serie_train / escala).values.astype(float)
        var_unc = max(float(np.var(x_train)), 1e-12)
        omega, alpha, beta = _ajustar_garch_params(x_train)
        n  = len(x_train)
        s2 = np.empty(n)
        s2[0] = var_unc
        for t in range(1, n):
            s2[t] = omega + alpha * x_train[t - 1] ** 2 + beta * s2[t - 1]
        return {
            "omega"         : omega,
            "alpha"         : alpha,
            "beta"          : beta,
            "escala"        : escala,
            "var_unc"       : var_unc,
            "last_sigma2"   : float(s2[-1]),
            "last_x_scaled" : float(x_train[-1]),
        }

    if {"R_t0", "D_t0"}.issubset(raw.columns):
        p = _params_serie(raw["D_t0"] - raw["R_t0"])
        if p:
            params["garch_vol"] = p

    if "TC_PEN_USD" in raw.columns:
        tc  = raw["TC_PEN_USD"].replace(0, np.nan).ffill()
        tci = tc.reindex(pd.bdate_range(tc.index.min(), tc.index.max())).ffill()
        ret = np.log(tci / tci.shift(1)).reindex(tc.index)
        p = _params_serie(ret)
        if p:
            params["garch_vol_tc"] = p

    if "EMBI_PERU" in raw.columns:
        p = _params_serie(raw["EMBI_PERU"].diff(1))
        if p:
            params["garch_vol_embi"] = p

    _garch_params_cache[cache_key] = params
    return params


###############################################################################
# PARTE 2-bis — Feature de régimen HMM (lee parquets de step005_validar_hmm)
###############################################################################

_regimen_cache: dict[str, pd.DataFrame] = {}   # cache por banco — evita re-leer el parquet en cada fold


def _ruta_estados_regimen(banco: str) -> Path:
    return DIR_REGIMEN_HMM / f"estados_regimen_hmm_{banco}.parquet"


# Entidades cuyos regimenes ya se generaron en ESTA corrida — evita repetir el
# ajuste cuando BANCOS_A_EVALUAR trae varias y cada evaluar_banco lo pide.
_hmm_generado: set[str] = set()


def _cortes_cubren_folds(banco: str, train_ends: list) -> bool:
    """
    True si el parquet de regimen de `banco` ya tiene un bloque con año_corte
    EXACTAMENTE igual a cada train_end de los folds.

    No basta con que el parquet exista. _elegir_año_corte_regimen toma el
    bloque mas reciente con fecha maxima <= train_end: si los cortes se
    generaron con otra grilla (p.ej. HMM_PASO_AVANCE anual mientras el
    walk-forward avanza 0.5 años), el bloque elegido termina ANTES de
    train_end y el fold pierde los meses intermedios de clasificacion, sin
    ningun aviso. Exigir coincidencia exacta convierte ese desalineamiento
    silencioso en una regeneracion.
    """
    ruta = _ruta_estados_regimen(banco)
    if not ruta.exists():
        return False
    try:
        cortes = set(pd.to_datetime(
            pd.read_parquet(ruta, columns=["año_corte"])["año_corte"].unique()))
    except Exception as e:
        logger.warning(f"  [HMM] No se pudo leer los cortes de {ruta.name} "
                       f"({type(e).__name__}: {e}) — se regenera.")
        return False
    return all(pd.Timestamp(te) in cortes for te in train_ends)


def asegurar_regimenes_hmm(bancos: list[str], folds: list[dict]) -> None:
    """
    Ajusta el HMM y guarda estados_regimen_hmm_<banco>.parquet (+ transmat y
    pickles por fold) para cada entidad de `bancos`, usando como fechas de
    corte los train_end de `folds`.

    Genera tambien los regimenes de la CONTRAPARTE y de SISTEMA aunque no esten
    en `bancos`. La razon es que rho_ij no consume nada de XGBoost: sus insumos
    son flujo, sigma y estado, los tres producidos por validar_hmm_v5 a partir
    de la serie cruda. Sin esto habria que correr el walk-forward completo de
    RESTO —Optuna, retrain, diagnosticos, ~25 min— solo para obtener su parquet
    de regimen, que sale de un ajuste de HMM de segundos. Son insumos baratos,
    no entidades a evaluar.

    (Eso vale para la base actual z=flujo/sigma. Si algun dia rho_ij se migra a
    base PIT necesitaria las distribuciones predictivas de ambas entidades, y
    ahi si harian falta las dos corridas completas.)

    Se corre para TODAS las entidades de una vez, no solo la que se esta
    evaluando: la correlacion transversal rho_ij del paper necesita el bloque
    de la contraparte, y con BANCOS_A_EVALUAR=[FOCO, RESTO, SISTEMA] el primer
    evaluar_banco corre antes de que exista el parquet del segundo.

    Alinear los cortes con los folds no es solo comodidad. hmm_evolucion
    documenta fechas_corte como "tipicamente = train_end de cada fold del
    walk-forward", y pasarlos explicitamente GARANTIZA que el bloque elegido
    por _elegir_año_corte_regimen termine exactamente en train_end. Corriendo
    el script aparte esa coincidencia depende de que HMM_INICIO y
    HMM_PASO_AVANCE casen por casualidad con la grilla de folds.

    No hace nada si HMM_INTERNO=False, si hmmlearn no esta disponible, o si el
    parquet ya cubre esos cortes. Cualquier fallo se degrada a warning: el
    feature de regimen ya es opcional y reemplazar_regimen_fold sabe seguir
    sin el.
    """
    if not (HMM_INTERNO and USAR_FEATURE_REGIMEN):
        return

    train_ends = [pd.Timestamp(f["train_end"]) for f in folds]

    # Entidades cuyo REGIMEN hace falta, que no son solo las que se evaluan:
    #   - la contraparte, porque rho_ij correlaciona los dos lados
    #   - SISTEMA, porque el rho_ij condicional se condiciona por SU regimen
    #     (no por el propio: ver _estimar_rho_transversal, el regimen propio
    #     volveria rho_ij asimetrico entre FOCO y RESTO)
    requeridos = set(bancos)
    for b in list(bancos):
        contra = _banco_contraparte(b)
        if contra is not None:
            requeridos.update({contra, "SISTEMA"})

    pendientes = [b for b in sorted(requeridos)
                  if b not in _hmm_generado and not _cortes_cubren_folds(b, train_ends)]
    if not pendientes:
        return

    try:
        import step005_validar_hmm_v5 as _hmm
    except ImportError as e:
        logger.warning(f"  [HMM] No se pudo importar step005_validar_hmm_v5 "
                       f"({e}) — se omite la generacion interna. El feature de "
                       f"regimen usara los parquets que ya existan.")
        return
    if not getattr(_hmm, "_HMM_OK", False):
        logger.warning("  [HMM] hmmlearn/sklearn no disponibles en el entorno "
                       "— se omite la generacion interna de regimenes.")
        return

    logger.info(f"  [HMM] Generando regimenes para {pendientes} "
                f"con {len(train_ends)} cortes = train_end de cada fold")
    for b in pendientes:
        try:
            flujo = _hmm.cargar_datos(banco=b, ruta_matriz=RUTA_MATRIZ)
            evol  = _hmm.hmm_evolucion(flujo, fechas_corte=train_ends)
            _hmm.guardar_objetos_simulacion(evol, flujo, banco=b,
                                            dir_output=DIR_REGIMEN_HMM)
            # El cache guarda el DataFrame leido del parquet ANTERIOR; sin
            # invalidarlo, reemplazar_regimen_fold seguiria usando los cortes
            # viejos durante el resto de la corrida.
            _regimen_cache.pop(b, None)
            _hmm_generado.add(b)
            logger.info(f"  [HMM] {b}: regimenes generados")
        except Exception as e:
            logger.warning(f"  [HMM] {b}: fallo la generacion "
                           f"({type(e).__name__}: {e}) — se sigue sin "
                           f"regenerar. Se usara el parquet existente si lo hay.")


def _ruta_transmat_regimen(banco: str) -> Path:
    return DIR_REGIMEN_HMM / f"transmat_hmm_{banco}.parquet"


def _cargar_estados_regimen_disco(banco: str) -> pd.DataFrame | None:
    """
    Carga estados_regimen_hmm_<banco>.parquet (formato largo: una fila por
    año_corte × fecha, ver step005_validar_hmm*.py). Cacheado por banco.
    Devuelve None si el archivo no existe (con un aviso) — el feature se omite
    sin romper el pipeline.
    """
    if banco in _regimen_cache:
        return _regimen_cache[banco]
    ruta = _ruta_estados_regimen(banco)
    if not ruta.exists():
        logger.warning(f"  [REGIMEN] No se encontró {ruta} — "
                       f"se omite el feature de régimen para {banco} "
                       f"(correr primero step005_validar_hmm*.py con "
                       f"GUARDAR_OBJETOS_SIMULACION=True).")
        _regimen_cache[banco] = None
        return None
    df_est = pd.read_parquet(ruta)
    df_est["fecha"] = pd.to_datetime(df_est["fecha"])
    _regimen_cache[banco] = df_est
    return df_est


def _elegir_año_corte_regimen(df_estados: pd.DataFrame, train_end: pd.Timestamp) -> int | None:
    """
    Elige, con precisión de fecha exacta (no de año calendario), el bloque HMM
    (año_corte) más reciente cuya fecha máxima clasificada no supera train_end.
    Mismo principio que GARCH/FFD: el feature de régimen usado en un fold nunca
    proviene de un modelo que haya visto datos posteriores a train_end de ESE
    fold, sin importar en qué mes del año caiga train_end.
    Devuelve None si ningún bloque califica (train_end anterior al primer bloque HMM).
    """
    max_fecha_por_corte = df_estados.groupby("año_corte")["fecha"].max()
    # año_corte puede ser string ISO "YYYY-MM-DD" (nuevo) o int (legacy).
    # La comparación max_fecha <= train_end funciona en ambos casos porque
    # 'fecha' ya es datetime. Pero idxmax() devolvería el tipo original de la
    # clave — devolvemos tal cual sin forzar int, para que el merge posterior
    # sea tipo-seguro.
    validos = max_fecha_por_corte[max_fecha_por_corte <= train_end]
    if validos.empty:
        return None
    return validos.idxmax()   # str ISO o int según el parquet que se cargó



# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES ANTI-LEAKAGE — Estimacion de rho_s
# ─────────────────────────────────────────────────────────────────────────────
# No se reclasifica nada aqui. estado, sigma y flujo por fecha ya vienen
# calculados por step005_validar_hmm_v3.py y guardados en
# estados_regimen_hmm_<banco>.parquet — este script SOLO los levanta, vía
# las mismas _cargar_estados_regimen_disco / _elegir_año_corte_regimen que
# ya usa reemplazar_regimen_fold() para el feature de XGBoost. Ver el bloque
# de llamada mas abajo (ESTIMAR_RHO_EN_VAL).

def _n_estados_de(df, col: str = "estado", default: int = 3) -> int:
    """
    Deriva cuántos estados hay a partir de los datos mismos (max(estado)+1)
    — nunca asume un valor fijo, porque estados_regimen_hmm_<banco>.parquet
    puede venir de una corrida con N_ESTADOS=2 o 3 (ver step005_validar_hmm*.py).
    Si no hay datos de dónde inferir (df vacío/None), usa `default`.
    """
    if df is not None and col in getattr(df, "columns", []) and len(df) > 0:
        return int(df[col].max()) + 1
    return default


def _nombres_estados(n_estados: int) -> list[str]:
    """
    Nombres de régimen según cuántos estados haya — misma convención que
    NOMBRES_ESTADOS en step005_validar_hmm_v3_1.py. Se replica la regla
    aquí (en vez de importarla) para no acoplar los dos scripts — cada uno
    debe poder correr solo.
    """
    if n_estados == 2:
        return ["calma", "severo"]
    elif n_estados == 3:
        return ["calma", "moderado", "severo"]
    return [f"estado_{i}" for i in range(n_estados)]


def _estimar_rho_val_fold(clasif: pd.DataFrame,
                           h_min: int | None = None,
                           max_gap_dias: int = 4,
                           rho_default: float = 0.3,
                           tau_ewma_pares: float = 100.0) -> dict:
    """
    Estima rho_s = Corr_ponderada(z_{t-1}, z_t | regimen_t = s) sobre TODO
    el historial clasificado disponible para el fold (tipicamente TRAIN+VAL,
    no solo VAL — la muestra mas grande es lo que permite que calma y
    moderado dejen de caer en el fallback por falta de pares).

    z_t = flujo_t / sigma_t. El sigma_t viene YA calculado por
    step005_validar_hmm_v3.py (columna "sigma" de
    estados_regimen_hmm_<banco>.parquet) — no se ajusta ninguna distribucion
    nueva aqui, ni se reclasifica nada. Esto NO es un modelo GARCH: no hay
    parametros que estimar, sigma_t ya viene resuelto de afuera. Tampoco hay
    PIT ni skew-t — se elimino la dependencia dinamica con
    step006_simulacion_paths.fitear_skewt que existia antes, precisamente
    para evitar el sesgo de usar un metodo de ajuste (aunque fuera de otro
    modulo) sobre la misma serie que se quiere correlacionar.

    Ponderacion exponencial de los PARES ya agrupados por regimen (no de
    sigma_t — esa es una serie aparte con su propio lambda_ewma=0.92):
    pares mas recientes pesan mas, pares lejanos pesan menos, siguiendo la
    parametrizacion de vida-media de Paleologo (Elements of Quantitative
    Investing):
        K = exp(-1/tau)      w_i = K ** t_i
    donde t_i = dias habiles entre la fecha del par i y la fecha mas
    reciente del historial clasificado para este fold. El prefactor de
    normalizacion (1-K) de la formula original se omite a propósito: es una
    constante multiplicativa que se cancela en la razon de la correlacion
    ponderada (media/covarianza/varianza todas escalan igual), asi que no
    cambia el resultado — solo importaria si se necesitaran los pesos
    normalizados para otra cosa fuera de esta funcion.

    Parametros
    ----------
    clasif : DataFrame indexado por fecha, con columnas:
             "flujo"  — flujo neto diario (mismo signo/escala que target)
             "sigma"  — sigma EWMA, tal cual la guarda
                        estados_regimen_hmm_<banco>.parquet
             "estado" — regimen HMM (0/1/2), tal cual lo guarda ese mismo parquet
    h_min          : solo para logging — clasif ya viene con una fila por
                      fecha, este parametro no filtra nada aqui.
    max_gap_dias   : gap maximo (dias habiles) entre origenes consecutivos
                      para que cuenten como "par" valido (evita correlacionar
                      observaciones separadas por feriados largos o huecos
                      de datos como si fueran consecutivas).
    rho_default    : piso conservador si ningun regimen tiene pares suficientes.
    tau_ewma_pares : vida-media (dias habiles) del peso exponencial sobre
                      los pares. tau=100 pondera muy suave.

    Devuelve {estado: rho_s} para todos los estados presentes en los datos
    (2 o 3, según N_ESTADOS con el que se corrió step005_validar_hmm*.py).
    """
    if clasif is None or len(clasif) < 6:
        n = 0 if clasif is None else len(clasif)
        n_estados = _n_estados_de(clasif)
        logger.warning(f"  [RHO_VAL] Muy pocas filas clasificadas ({n}) — usando rho_default")
        return {s: rho_default for s in range(n_estados)}

    clasif = clasif.sort_index()
    flujo_arr = clasif["flujo"].values.astype(float)
    sigma_arr = clasif["sigma"].values.astype(float)
    r_arr     = clasif["estado"].values.astype(int)
    f_arr     = np.array([pd.Timestamp(d) for d in clasif.index])

    # z_t = flujo_t / sigma_t — reutiliza el sigma tal cual viene guardado
    # en estados_regimen_hmm_<banco>.parquet. sigma<=0 no deberia ocurrir
    # en la practica, pero se guarda por seguridad numerica: esas filas se
    # excluyen, no se rompen.
    with np.errstate(divide="ignore", invalid="ignore"):
        z_arr = np.where(sigma_arr > 1e-9, flujo_arr / sigma_arr, np.nan)

    valido = ~np.isnan(z_arr)
    if valido.sum() < 4:
        n_estados = _n_estados_de(clasif)
        logger.warning(f"  [RHO_VAL] Muy pocos z_t validos ({int(valido.sum())}) — usando rho_default")
        return {s: rho_default for s in range(n_estados)}

    z_arr = z_arr[valido]
    f_arr = f_arr[valido]
    r_arr = r_arr[valido]

    fecha_mas_reciente = f_arr.max()
    K = float(np.exp(-1.0 / tau_ewma_pares))

    # n_estados derivado de r_arr (ya filtrado/validado) — nunca fijo.
    n_estados  = int(r_arr.max()) + 1
    nombres_s  = _nombres_estados(n_estados)

    # Primera pasada: estimar rho ponderado solo para regimenes con
    # suficientes pares
    min_pares = MIN_PARES_RHO_REGIMEN
    pares_por_s  = {}
    rho_estimado = {}   # solo los que tienen >= min_pares

    for s in range(n_estados):
        pz_prev, pz_cur, pesos = [], [], []
        for i in range(1, len(z_arr)):
            if r_arr[i] != s:
                continue
            gap = np.busday_count(f_arr[i-1].date(), f_arr[i].date())
            if gap < 1 or gap > max_gap_dias:
                continue
            pz_prev.append(z_arr[i-1])
            pz_cur.append(z_arr[i])
            t_dias = np.busday_count(f_arr[i].date(), fecha_mas_reciente.date())
            pesos.append(K ** t_dias)
        pares_por_s[s] = (pz_prev, pz_cur, pesos)

        if len(pz_prev) >= min_pares:
            w  = np.asarray(pesos, dtype=float)
            zp = np.asarray(pz_prev, dtype=float)
            zc = np.asarray(pz_cur,  dtype=float)
            w_sum = w.sum()
            # Correlacion de Pearson ponderada — formula estandar, el
            # prefactor (1-K) de los pesos se cancela aqui (aparece tanto
            # en numerador como denominador), por eso no se aplica arriba.
            mu_p  = np.sum(w * zp) / w_sum
            mu_c  = np.sum(w * zc) / w_sum
            cov   = np.sum(w * (zp - mu_p) * (zc - mu_c)) / w_sum
            var_p = np.sum(w * (zp - mu_p) ** 2) / w_sum
            var_c = np.sum(w * (zc - mu_c) ** 2) / w_sum
            denom = np.sqrt(var_p * var_c)
            rho = float(cov / denom) if denom > 1e-12 else 0.0
            rho = float(np.clip(rho, -0.98, 0.98))
            rho_estimado[s] = rho

    # Fallback: max de los rhos POSITIVOS estimados, con piso rho_default.
    # Razon: la persistencia genuina es positiva; una rho negativa estimada
    # es estadisticamente posible pero no es un prior valido para el fallback
    # de un regimen con pocos datos. Si todos los estimados son negativos
    # (inusual), el piso rho_default (0.3) prevalece como conservador.
    _rhos_positivos = [r for r in rho_estimado.values() if r > 0]
    fallback = max(_rhos_positivos + [rho_default])

    rho_por_s = {}
    for s in range(n_estados):
        n_pares = len(pares_por_s[s][0])
        if s in rho_estimado:
            rho_por_s[s] = rho_estimado[s]
            logger.info(f"    [RHO_VAL] {nombres_s[s]:8s} (s={s}): "
                        f"rho={rho_estimado[s]:+.3f}  n_pares={n_pares} "
                        f"(ponderado, tau={tau_ewma_pares:.0f}d)")
        else:
            rho_por_s[s] = fallback
            logger.info(f"    [RHO_VAL] {nombres_s[s]:8s} (s={s}): "
                        f"rho={fallback:+.3f} [FALLBACK — solo {n_pares} pares < minimo={min_pares}]")

    logger.info(f"    [RHO_VAL] resumen fold -> {rho_por_s}")
    return rho_por_s


def _banco_contraparte(banco: str) -> str | None:
    """
    Entidad complementaria dentro de la misma particion: FOCO_<P> <-> RESTO_<P>.

    Se deriva del nombre en vez de leerla de config porque BANCO ya se arma asi
    en la seccion de opciones (f"{ENTIDAD}_{PARTICION.upper()}"), y duplicar la
    regla en dos lugares es como se desincronizan.

    Devuelve None para SISTEMA o cualquier nombre sin ese prefijo: no tienen
    contraparte y rho_ij no aplica.
    """
    for pref, otro in (("FOCO_", "RESTO_"), ("RESTO_", "FOCO_")):
        if banco.startswith(pref):
            return otro + banco[len(pref):]
    return None


def _estimar_rho_transversal(clasif_i: pd.DataFrame,
                             clasif_j: pd.DataFrame,
                             estados_cond: pd.Series | None = None,
                             tau_ewma_pares: float = 100.0) -> dict:
    """
    Estima rho_ij = Corr_ponderada(z_i(t), z_j(t)) — la condicion (D2) del paper
    de agregacion por grupos: correlacion CONTEMPORANEA entre dos entidades,
    frente a la (D1) temporal que estima _estimar_rho_val_fold.

    MISMA BASE QUE phi, a proposito. La ecuacion (5) del paper construye la
    matriz de innovaciones como

        Sigma_e = R (x) (11' - phi phi')      (Sigma_e)_ij = rho_ij (1 - phi_i phi_j)

    o sea multiplica rho_ij por (1 - phi_i phi_j). Si phi se midiera sobre
    z=flujo/sigma y rho sobre otra cosa (p.ej. normal scores del PIT), el
    producto mezclaria dos objetos distintos y Sigma_e dejaria de significar lo
    que la derivacion dice. Por eso aca se replica exactamente el z, la
    ponderacion EWMA y la formula de Pearson ponderada de _estimar_rho_val_fold.
    Si algun dia se migra phi a base PIT, hay que migrar esta funcion en el
    mismo commit.

    SIN filtro de gap, y no es un olvido: el max_gap_dias de la version temporal
    existe para no correlacionar t-1 con t cuando los separa un feriado largo,
    tratandolos como consecutivos. Aca los dos valores son del MISMO dia, asi
    que la consecutividad no interviene — solo se exige que ambas entidades
    tengan dato ese dia (join interno por fecha).

    Parametros
    ----------
    clasif_i, clasif_j : DataFrame indexado por fecha con columnas "flujo" y
                         "sigma", tal como los guarda
                         estados_regimen_hmm_<banco>.parquet. Se cruzan por
                         interseccion de fechas.
    estados_cond       : serie {fecha: estado} para calcular ademas un rho por
                         regimen. OPCIONAL y deliberadamente externa: cada
                         entidad tiene su propio HMM y sus etiquetas pueden
                         diferir el mismo dia, asi que condicionar por el
                         regimen "propio" volveria rho_ij asimetrico — el valor
                         calculado desde FOCO no coincidiria con el calculado
                         desde RESTO. Pasar una serie comun (la de SISTEMA)
                         mantiene la simetria. Si es None, solo se devuelve el
                         global.
    tau_ewma_pares     : vida media (dias habiles) del peso exponencial, igual
                         que en la version temporal.

    Devuelve
    --------
    {"global": rho, "n_pares": n, <estado>: rho_s, ...} — las claves por estado
    solo aparecen si estados_cond se paso y ese regimen junto suficientes dias.
    """
    if clasif_i is None or clasif_j is None:
        return {}

    # Interseccion de fechas: rho_ij es contemporaneo, necesita ambas series el
    # mismo dia. El sort es necesario para que el peso por recencia sea correcto.
    idx = clasif_i.index.intersection(clasif_j.index).sort_values()
    if len(idx) < MIN_PARES_RHO_REGIMEN:
        logger.warning(f"    [RHO_IJ] Solo {len(idx)} fechas en comun entre las "
                       f"dos entidades — insuficiente, se omite rho_ij.")
        return {}

    def _z(clasif: pd.DataFrame) -> np.ndarray:
        f = clasif.loc[idx, "flujo"].values.astype(float)
        s = clasif.loc[idx, "sigma"].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 1e-9, f / s, np.nan)

    z_i, z_j = _z(clasif_i), _z(clasif_j)
    valido = ~(np.isnan(z_i) | np.isnan(z_j))
    if valido.sum() < MIN_PARES_RHO_REGIMEN:
        logger.warning(f"    [RHO_IJ] Solo {int(valido.sum())} pares con sigma>0 "
                       f"en ambas entidades — se omite rho_ij.")
        return {}

    z_i, z_j = z_i[valido], z_j[valido]
    fechas   = np.array([pd.Timestamp(d) for d in idx[valido]])

    K = float(np.exp(-1.0 / tau_ewma_pares))
    mas_reciente = fechas.max()
    pesos = np.array([K ** np.busday_count(d.date(), mas_reciente.date())
                      for d in fechas], dtype=float)

    def _pearson_ponderada(w, a, b) -> float:
        """Identica a la de _estimar_rho_val_fold; el prefactor (1-K) se cancela."""
        w_sum = w.sum()
        if w_sum <= 0:
            return 0.0
        mu_a  = np.sum(w * a) / w_sum
        mu_b  = np.sum(w * b) / w_sum
        cov   = np.sum(w * (a - mu_a) * (b - mu_b)) / w_sum
        var_a = np.sum(w * (a - mu_a) ** 2) / w_sum
        var_b = np.sum(w * (b - mu_b) ** 2) / w_sum
        den   = np.sqrt(var_a * var_b)
        # El clip a +-0.98 no es cosmetico: rho_ij=+-1 vuelve R singular y la
        # factorizacion de Cholesky del paso P1 del algoritmo falla.
        return float(np.clip(cov / den, -0.98, 0.98)) if den > 1e-12 else 0.0

    out = {"global": _pearson_ponderada(pesos, z_i, z_j),
           "n_pares": int(valido.sum())}
    logger.info(f"    [RHO_IJ] global: rho={out['global']:+.3f}  "
                f"n_pares={out['n_pares']} (ponderado, tau={tau_ewma_pares:.0f}d)")

    if estados_cond is not None:
        est = pd.Series(estados_cond).reindex(idx[valido]).values
        for s in sorted({int(v) for v in est if pd.notna(v)}):
            m = np.array([pd.notna(v) and int(v) == s for v in est])
            if m.sum() < MIN_PARES_RHO_REGIMEN:
                logger.info(f"    [RHO_IJ] estado {s}: solo {int(m.sum())} pares "
                            f"< minimo={MIN_PARES_RHO_REGIMEN} — se usa el global")
                continue
            out[s] = _pearson_ponderada(pesos[m], z_i[m], z_j[m])
            logger.info(f"    [RHO_IJ] estado {s}: rho={out[s]:+.3f}  "
                        f"n_pares={int(m.sum())}")
    return out


def reemplazar_regimen_fold(df_fold: pd.DataFrame, train_end: pd.Timestamp,
                            banco: str) -> tuple[pd.DataFrame, int | None]:
    """
    Asigna "regimen_hmm" (estado 0/1/2) y "regimen_sigma" a TODAS las filas del
    fold (train+val+test), tomados del bloque HMM (año_corte) elegido por
    _elegir_año_corte_regimen — análogo a reemplazar_garch_fold/reemplazar_ffd_fold,
    pero leyendo de un parquet pre-calculado en vez de reestimar en el momento.

    Merge por 'fecha_t': el eje 'fecha' del HMM (calendario real del flujo) y
    'fecha_t' de la matriz de features son el mismo eje de días hábiles.

    Devuelve (df_fold con las 2 columnas asignadas, año_corte usado o None).
    Si USAR_FEATURE_REGIMEN=False o no hay parquet disponible, devuelve
    df_fold sin cambios y año_corte=None (las columnas baseline ya puestas en
    NaN por evaluar_banco quedan así, y se imputan con la mediana de TRAIN
    igual que cualquier otro feature con huecos).
    """
    if not USAR_FEATURE_REGIMEN:
        return df_fold, None

    df_estados = _cargar_estados_regimen_disco(banco)
    if df_estados is None:
        return df_fold, None

    año_corte = _elegir_año_corte_regimen(df_estados, train_end)
    if año_corte is None:
        logger.warning(f"  [REGIMEN] Sin bloque HMM con corte ≤ {train_end.date()} "
                       f"— se omite el feature en este fold.")
        return df_fold, None

    bloque = df_estados[df_estados["año_corte"] == año_corte]
    serie_estado = bloque.drop_duplicates("fecha").set_index("fecha")["estado"]
    serie_sigma  = bloque.drop_duplicates("fecha").set_index("fecha")["sigma"]

    df_fold = df_fold.copy()
    df_fold["regimen_hmm"]   = df_fold["fecha_t"].map(serie_estado)
    df_fold["regimen_sigma"] = df_fold["fecha_t"].map(serie_sigma)
    return df_fold, año_corte


def cargar_transmat_regimen(banco: str, año_corte: int) -> np.ndarray:
    """
    Carga transmat_hmm_<banco>.parquet y devuelve la matriz N_ESTADOS x
    N_ESTADOS (reordenada por volatilidad ascendente — calma...severo) del
    año_corte pedido. N_ESTADOS se infiere de cuántas columnas p## existen
    en el parquet (2x2 o 3x3, según cómo se corrió step005_validar_hmm*.py)
    — no se asume fijo. Para usar en la simulación de paths (step006): el
    año_corte guardado junto a cada fila en preds_test_fold*.parquet
    (columna 'año_corte_regimen') indica exactamente qué matriz corresponde
    a esas predicciones.
    """
    ruta = _ruta_transmat_regimen(banco)
    df_t = pd.read_parquet(ruta)
    fila = df_t[df_t["año_corte"] == año_corte]
    if fila.empty:
        raise ValueError(f"No hay transmat guardada para banco={banco}, "
                        f"año_corte={año_corte} en {ruta}")
    n_estados = int(np.sqrt(len([c for c in df_t.columns if c.startswith("p")])))
    cols = [f"p{i}{j}" for i in range(n_estados) for j in range(n_estados)]
    return fila.iloc[0][cols].values.astype(float).reshape(n_estados, n_estados)


###############################################################################
# PARTE 3 — Generación de folds  (EXPANDING o ROLLING según toggle)
###############################################################################

def generar_folds(
    fechas_disponibles,
    ventana_train_años,
    ventana_val_años,
    ventana_test_años,
    paso_años,
    purge_dias_hab,
    purge_val_test,
    expanding,
    fold_fecha_inicio=None,
):
    """
    EXPANDING=True  → train_start fijo en f_min; train_end crece paso_años/fold.
    EXPANDING=False → ventana rodante fija.

    Estructura por fold (López de Prado §12):
      TRAIN → [purge_dias_hab] → VAL (Optuna) → [purge_val_test] → TEST (métricas OOS)

    purge_dias_hab cubre tanto solapamiento de etiquetas Y (h_max dh) como
    el warm-up de features de lookback (MA22 ≤ 22 dh ≤ h_max).

    Genera folds mientras test_end ≤ última fecha disponible.
    """
    folds   = []
    f_min   = fechas_disponibles.min()
    # fold_fecha_inicio: permite fijar la fecha de inicio del primer fold sin
    # filtrar la matriz (que puede tener historia anterior necesaria para
    # features de lookback). Si se especifica y es posterior al minimo, se
    # usa como train_start del primer fold; si no hay datos desde esa fecha,
    # se ignora con un aviso. Se recibe como PARAMETRO (no global) para que la
    # funcion sea autocontenida y testeable.
    if fold_fecha_inicio is not None:
        _fecha_inicio_forzada = pd.Timestamp(fold_fecha_inicio)
        if _fecha_inicio_forzada > f_min:
            if _fecha_inicio_forzada <= fechas_disponibles.max():
                import logging as _log
                _log.getLogger(__name__).info(
                    f"  [FOLDS] fold_fecha_inicio={fold_fecha_inicio}: "
                    f"primer fold arranca desde {_fecha_inicio_forzada.date()} "
                    f"(datos desde {f_min.date()} siguen usandose para features)")
                f_min = _fecha_inicio_forzada
            else:
                import logging as _log
                _log.getLogger(__name__).warning(
                    f"  [FOLDS] fold_fecha_inicio={fold_fecha_inicio} posterior "
                    f"a ultima fecha disponible — ignorado")
    f_max   = fechas_disponibles.max()
    fold_idx = 0

    while True:
        if expanding:
            train_start = f_min
            train_end   = f_min + pd.DateOffset(
                months=int(round((ventana_train_años + fold_idx * paso_años) * 12)))
        else:
            train_start = f_min + pd.DateOffset(
                months=int(round(fold_idx * paso_años * 12)))
            train_end   = train_start + pd.DateOffset(
                months=int(round(ventana_train_años * 12)))

        val_start  = train_end  + pd.offsets.BusinessDay(purge_dias_hab)
        val_end    = val_start  + pd.DateOffset(months=int(round(ventana_val_años * 12)))
        test_start = val_end    + pd.offsets.BusinessDay(purge_val_test)
        test_end   = test_start + pd.DateOffset(months=int(round(ventana_test_años * 12)))

        if test_end > f_max or train_end >= f_max or val_start >= f_max:
            break

        burn_cutoff  = train_start + pd.offsets.BusinessDay(BURN_IN_DIAS_HAB)
        n_train_all  = int(((fechas_disponibles >= train_start) &
                            (fechas_disponibles <= train_end)).sum())
        n_train_eff  = int(((fechas_disponibles >= burn_cutoff) &
                            (fechas_disponibles <= train_end)).sum())
        n_val        = int(((fechas_disponibles >= val_start) &
                            (fechas_disponibles <  test_start)).sum())
        n_test       = int(((fechas_disponibles >= test_start) &
                            (fechas_disponibles <= test_end)).sum())

        if n_train_eff < 60 or n_val < 10 or n_test < 10:
            fold_idx += 1
            continue

        folds.append({
            "fold"              : fold_idx + 1,
            "train_start"       : train_start,
            "train_end"         : train_end,
            "val_start"         : val_start,
            "val_end"           : val_end,
            "test_start"        : test_start,
            "test_end"          : test_end,
            "burn_cutoff"       : burn_cutoff,
            "n_train_fechas"    : n_train_all,
            "n_train_efectivo"  : n_train_eff,
            "n_val_fechas"      : n_val,
            "n_test_fechas"     : n_test,
        })
        fold_idx += 1

    return folds


def resolver_folds_manuales(
    folds_cfg: list[dict],
    fechas_disponibles,
    n_folds_previos: int,
) -> list[dict]:
    """
    Convierte la lista FOLDS_MANUALES en dicts compatibles con el pipeline.

    Cada entrada de folds_cfg debe tener:
      train_start, train_end, val_start, val_end, test_start, test_end  (str o Timestamp)

    Los fold numbers se asignan como n_folds_previos+1, +2, ...
    El campo "_manual": True permite identificarlos en los logs.
    """
    folds = []
    for i, cfg in enumerate(folds_cfg):
        ts = pd.Timestamp(cfg["train_start"])
        te = pd.Timestamp(cfg["train_end"])
        vs = pd.Timestamp(cfg["val_start"])
        ve = pd.Timestamp(cfg["val_end"])
        xs = pd.Timestamp(cfg["test_start"])
        xe = pd.Timestamp(cfg["test_end"])

        n_train = int(((fechas_disponibles >= ts) & (fechas_disponibles <= te)).sum())
        n_val   = int(((fechas_disponibles >= vs) & (fechas_disponibles <  xs)).sum())
        n_test  = int(((fechas_disponibles >= xs) & (fechas_disponibles <= xe)).sum())

        folds.append({
            "fold"           : n_folds_previos + i + 1,
            "train_start"    : ts,
            "train_end"      : te,
            "val_start"      : vs,
            "val_end"        : ve,
            "test_start"     : xs,
            "test_end"       : xe,
            "n_train_fechas" : n_train,
            "n_val_fechas"   : n_val,
            "n_test_fechas"  : n_test,
            "_manual"        : True,
        })
    return folds


###############################################################################
# PARTE 4 — Modelos: objetivos, optimización, entrenamiento
###############################################################################

# def make_quantile_objective(tau, s, std_y):
#     _scale = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)
#     def objective(y_pred, dtrain):
#         u    = dtrain.get_label() - y_pred
#         grad = -((tau - 0.5 + np.arctan(u / s) / np.pi)
#                  + u * s / (np.pi * (s ** 2 + u ** 2))) * _scale
#         hess = 2 * s ** 3 / (np.pi * (s ** 2 + u ** 2) ** 2) * _scale
#         return grad, hess
#     return objective


def make_quantile_objective(tau, s, std_y):
    # _scale ajusta el lambda efectivo de XGBoost según la escala del target
    # (target no estandarizado: std_y~80,000 -> _scale~1e9).
    # Se aplica a grad; hess ya lo incorpora algebraicamente en su forma simplificada.
    _scale = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)
    def objective(y_pred, dtrain):
        # Clamp predicciones divergentes: en trials Optuna con malos
        # hiperparámetros las predicciones XGBoost (float32) pueden llegar
        # a ±inf, produciendo u=±inf -> u²=inf -> overflow en el hessiano.
        y_pred = np.clip(y_pred, -1e15, 1e15)
        u      = dtrain.get_label() - y_pred
        s2u2   = s ** 2 + u ** 2          # siempre > 0, sin overflow en float64
        grad   = -((tau - 0.5 + np.arctan(u / s) / np.pi)
                   + u * s / (np.pi * s2u2)) * _scale
        # Forma simplificada: hess = ((s²+std_y²)/(s²+u²))²
        # Algebraicamente idéntica a la original pero evita los intermedios
        # ~1e33 de _scale × ~1e-13 del cociente que provocan pérdida de precisión.
        hess   = ((s ** 2 + std_y ** 2) / s2u2) ** 2
        return grad, hess
    return objective



def make_pinball_metric(tau):
    def metric(y_pred, dtrain):
        y_true = dtrain.get_label()
        u      = y_true - y_pred
        return "pinball", float(np.mean(np.where(u >= 0, tau * u, (tau - 1) * u)))
    return metric

class _PinballEarlyStopping(xgb.callback.TrainingCallback):
    """
    Early stopping sobre la métrica 'pinball' del eval set de validación.
    Reemplaza xgb.callback.EarlyStopping porque ese callback busca el nombre
    en un dict plano ("val-pinball") mientras que XGBoost con custom_metric
    almacena en estructura anidada: evals_log["val"]["pinball"].
    """
    def __init__(self, rounds: int = 50):
        super().__init__()
        self.rounds  = rounds
        self._best   = float("inf")
        self._since  = 0
 
    def after_iteration(self, model, epoch, evals_log):
        score = None
        for metrics in evals_log.values():
            if "pinball" in metrics:
                score = metrics["pinball"][-1]
                break
        if score is None:
            return False          # métrica no encontrada → no para
        if score < self._best - 1e-9:
            self._best  = score
            self._since = 0
        else:
            self._since += 1
        return self._since >= self.rounds  # True = parar

def _objective_optuna(trial, X_tr, y_tr, X_va, y_va, std_y):
    s = (std_y * S_FACTOR_FIJO if S_FIJO
         else trial.suggest_float("s", std_y * S_MIN_FACTOR, std_y * S_MAX_FACTOR, log=True))
    params = {
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth"       : trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 200),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method"     : "hist",
        "nthread"         : _XGB_NTHREAD,
        "seed"            : 42,
    }
    if MAX_DELTA_STEP_FACTOR is not None:
        params["max_delta_step"] = MAX_DELTA_STEP_FACTOR * std_y
    n_est  = trial.suggest_int("n_estimators", 100, 1000)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    model  = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(0.50, s, std_y),
        custom_metric=make_pinball_metric(0.50),
        evals=[(dval, "val")],
        callbacks=[_PinballEarlyStopping(rounds=50)],
        verbose_eval=False,
    )
    return pinball_loss(y_va.values, model.predict(dval), 0.50)


def optimizar_hiperparametros(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
    logger.info(f"    Optuna fold {fold_num} ({n_trials} trials, τ=0.50 en VAL)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold_num),
    )
    study.optimize(
        lambda t: _objective_optuna(t, X_tr, y_tr, X_va, y_va, std_y),
        n_trials=n_trials, show_progress_bar=False,
    )
    bp = study.best_params
    if S_FIJO:
        bp["s"] = std_y * S_FACTOR_FIJO
    logger.info(f"    Fold {fold_num} best pinball(Q50/VAL)={study.best_value:.4f} "
                f"n_est={bp['n_estimators']} lr={bp['learning_rate']:.4f} s={bp['s']:.2f}")
    return bp


def entrenar_quantiles(X_tr, y_tr, best_params, quantiles, std_y):
    s_best = best_params["s"]
    n_est  = best_params["n_estimators"]
    params = {k: v for k, v in best_params.items() if k not in ("s", "n_estimators")}
    params.update({"tree_method": "hist", "seed": 42, "nthread": _XGB_NTHREAD})
    if MAX_DELTA_STEP_FACTOR is not None:
        params["max_delta_step"] = MAX_DELTA_STEP_FACTOR * std_y

    def _train_tau(tau):
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        return tau, xgb.train(
            params, dtrain, num_boost_round=n_est,
            obj=make_quantile_objective(tau, s_best, std_y),
            verbose_eval=False,
        )

    modelos = {}
    with ThreadPoolExecutor(max_workers=len(quantiles)) as ex:
        for tau, model in ex.map(lambda t: _train_tau(t), quantiles):
            modelos[tau] = model
    return modelos


def predecir_y_corregir(modelos, X):
    dmat      = xgb.DMatrix(X)
    preds_raw = {tau: m.predict(dmat) for tau, m in modelos.items()}
    # "mean" key is not a quantile — exclude from crossing correction
    mean_pred = preds_raw.pop("mean", None)
    taus      = sorted(preds_raw)
    matrix    = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    result    = {t: matrix[:, i] for i, t in enumerate(taus)}
    if mean_pred is not None:
        result["mean"] = mean_pred
    return result


# ── LightGBM ──────────────────────────────────────────────────────────────────

def _objetivo_optuna_lgbm(trial, X_tr, y_tr, X_va, y_va):
    if not _LGBM_OK:
        raise ImportError("lightgbm no está instalado")
    tau = 0.50
    params = {
        "objective": "quantile", "alpha": tau, "verbosity": -1, "seed": 42,
        "learning_rate"    : trial.suggest_float("learning_rate",    0.01,  0.3,  log=True),
        "num_leaves"       : trial.suggest_int(  "num_leaves",        15,   255),
        "max_depth"        : trial.suggest_int(  "max_depth",          3,    10),
        "min_child_samples": trial.suggest_int(  "min_child_samples", 10,   200),
        "subsample"        : trial.suggest_float("subsample",         0.5,   1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree",  0.4,  1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "subsample_freq"   : 1,
    }
    n_est  = trial.suggest_int("n_estimators", 100, 1000)
    dtrain = lgb.Dataset(X_tr.values, label=y_tr.values)
    dval   = lgb.Dataset(X_va.values, label=y_va.values, reference=dtrain)
    cbs    = [lgb.log_evaluation(-1)]
    try:
        cbs.append(lgb.early_stopping(50, verbose=False))
    except Exception as _e_lgb:
        logger.debug(f"lgb.early_stopping no disponible en esta versión → "
                     f"{type(_e_lgb).__name__}: {_e_lgb}")
    model = lgb.train(params, dtrain, num_boost_round=n_est,
                      valid_sets=[dval], valid_names=["val"], callbacks=cbs)
    return pinball_loss(y_va.values, model.predict(X_va.values), tau)


def optimizar_hiperparametros_lgbm(X_tr, y_tr, X_va, y_va, n_trials, fold_num):
    logger.info(f"    Optuna[LGBM] fold {fold_num} ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42 + fold_num))
    study.optimize(lambda t: _objetivo_optuna_lgbm(t, X_tr, y_tr, X_va, y_va),
                   n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def entrenar_quantiles_lgbm(X_tr, y_tr, best_params, quantiles):
    if not _LGBM_OK:
        raise ImportError("lightgbm no está instalado")
    n_est = int(best_params.get("n_estimators", 300))
    hp    = {k: v for k, v in best_params.items() if k != "n_estimators"}
    modelos = {}
    for tau in quantiles:
        params = {"objective": "quantile", "alpha": tau,
                  "verbosity": -1, "seed": 42, "subsample_freq": 1, **hp}
        dtrain = lgb.Dataset(X_tr.values, label=y_tr.values)
        modelos[tau] = lgb.train(params, dtrain, num_boost_round=n_est,
                                 callbacks=[lgb.log_evaluation(-1)])
    return modelos


def predecir_lgbm(modelos, X):
    preds_raw = {tau: m.predict(X.values) for tau, m in modelos.items()}
    taus   = sorted(preds_raw)
    matrix = np.sort(np.column_stack([preds_raw[t] for t in taus]), axis=1)
    return {t: matrix[:, i] for i, t in enumerate(taus)}


# ── XGBoost QT ────────────────────────────────────────────────────────────────

def _objetivo_optuna_xgb_qt_tau(trial, tau, X_tr, y_tr, X_va, y_va, std_y):
    s = (std_y * S_FACTOR_FIJO if S_FIJO
         else trial.suggest_float("s", std_y * S_MIN_FACTOR, std_y * S_MAX_FACTOR, log=True))
    params = {
        "learning_rate"   : trial.suggest_float("learning_rate",   0.01,  0.3,  log=True),
        "max_depth"       : trial.suggest_int(  "max_depth",         3,    10),
        "min_child_weight": trial.suggest_int(  "min_child_weight", 10,   200),
        "colsample_bytree": trial.suggest_float("colsample_bytree",  0.4,  1.0),
        "subsample"       : trial.suggest_float("subsample",         0.5,  1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method"     : "hist",
        "nthread"         : _XGB_NTHREAD,   # limita threads por trial en paralelo
        "seed"            : 42,
    }
    if MAX_DELTA_STEP_FACTOR is not None:
        params["max_delta_step"] = MAX_DELTA_STEP_FACTOR * std_y
    n_est  = trial.suggest_int("n_estimators", 100, 1000)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    model  = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(tau, s, std_y),
        custom_metric=make_pinball_metric(tau),
        evals=[(dval, "val")],
        callbacks=[_PinballEarlyStopping(rounds=50)],
        verbose_eval=False,
    )
    return pinball_loss(y_va.values, model.predict(dval), tau)


def _worker_optuna_tau(args):
    """
    Función de módulo (picklable) para ProcessPoolExecutor.
    Cada proceso hijo corre Optuna + entrenamiento final para un cuantil.
    """
    tau, X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num = args
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42 + fold_num + int(tau * 100)),
    )
    study.optimize(
        lambda t: _objetivo_optuna_xgb_qt_tau(t, tau, X_tr, y_tr, X_va, y_va, std_y),
        n_trials=n_trials, show_progress_bar=False,
    )
    bp    = study.best_params
    if S_FIJO:
        bp["s"] = std_y * S_FACTOR_FIJO
    s     = bp["s"]
    n_est = bp["n_estimators"]
    params = {k: v for k, v in bp.items() if k not in ("s", "n_estimators")}
    params.update({"tree_method": "hist", "seed": 42, "nthread": _XGB_NTHREAD})
    if MAX_DELTA_STEP_FACTOR is not None:
        params["max_delta_step"] = MAX_DELTA_STEP_FACTOR * std_y
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    model  = xgb.train(
        params, dtrain, num_boost_round=n_est,
        obj=make_quantile_objective(tau, s, std_y),
        verbose_eval=False,
    )
    # trial_values: insumo de la curva de convergencia en guardar_hp_report.
    # Se filtran los trials podados/fallidos (value=None), que romperían el
    # accumulate(min) del gráfico.
    trial_values = [float(t.value) for t in study.trials if t.value is not None]
    return tau, model, bp, study.best_value, trial_values


def _entrenar_fold_xgb_qt(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
    """
    Entrena un modelo por cuantil con Optuna independiente para cada uno.
    Los estudios corren en paralelo con ProcessPoolExecutor (un proceso por
    cuantil, GIL independiente) → paralelismo real en múltiples núcleos.
    Cada XGBoost usa _XGB_NTHREAD threads → sin over-subscription de CPU.
    """
    worker_args = [
        (tau, X_tr, y_tr, X_va, y_va, std_y, get_n_trials(tau), fold_num)
        for tau in QUANTILES
    ]

    modelos      = {}
    best_by_tau  = {}
    optuna_meta  = {}   # {tau: {"best_pinball_val": …, "trial_values": [...]}}

    # Se remide en cada fold: los folds tardíos tienen más filas y menos RAM
    # libre, así que el valor del fold 1 no sirve para el fold 3.
    n_workers = get_max_workers_optuna()
    n_rondas  = -(-len(QUANTILES) // n_workers)   # ceil division
    try:
        import psutil
        _libre = f" | RAM libre={psutil.virtual_memory().available / 1e9:.1f} GB"
    except ImportError:
        _libre = ""
    logger.info(f"    [xgb_qt] fold {fold_num}: {n_workers} workers paralelos "
                f"× {n_rondas} ronda(s) para {len(QUANTILES)} cuantiles{_libre}")

    # Monitor de RAM: mide el consumo real de los workers para calibrar
    # _MEM_POR_WORKER_GB con datos en vez de la estimación analítica.
    _stop_mon  = threading.Event()
    _mem_stats = {}
    _mon = threading.Thread(target=_monitor_rss_hijos,
                            args=(_stop_mon, _mem_stats), daemon=True)
    _mon.start()

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_worker_optuna_tau, args): args[0] for args in worker_args}
            for fut in as_completed(futures):
                tau, model, bp, best_val, trial_values = fut.result()
                modelos[tau]     = model
                best_by_tau[tau] = bp
                optuna_meta[tau] = {"best_pinball_val": float(best_val),
                                    "trial_values": trial_values}
                logger.info(f"    [xgb_qt] τ={tau:.2f} fold {fold_num}: "
                            f"pinball/VAL={best_val:.4f}  s={bp['s']:.3f}  "
                            f"n_est={bp['n_estimators']}")
    finally:
        _stop_mon.set()
        _mon.join(timeout=3)
        _pico = _mem_stats.get("pico")
        if _pico:
            _n   = _mem_stats["n_en_pico"]
            _por = _pico / _n / 1e9
            logger.info(f"    [mem] fold {fold_num}: pico hijos={_pico/1e9:.2f} GB "
                        f"con {_n} proceso(s) → {_por:.2f} GB/worker "
                        f"(_MEM_POR_WORKER_GB={_MEM_POR_WORKER_GB})")
            if _por > _MEM_POR_WORKER_GB:
                logger.warning(f"    [mem] consumo real > estimación: sube "
                               f"_MEM_POR_WORKER_GB a ~{_por * 1.2:.1f} "
                               f"para no arriesgar OOM")

    # Mean model — reg:squarederror with best Q50 hyperparameters as base
    bp_mean = best_by_tau.get(0.50, list(best_by_tau.values())[0])
    params_mean = {k: v for k, v in bp_mean.items() if k not in ("s", "n_estimators")}
    params_mean.update({"objective": "reg:squarederror",
                        "tree_method": "hist",
                        "nthread": _XGB_NTHREAD,
                        "seed": 42})
    dtrain_mean = xgb.DMatrix(X_tr, label=y_tr)
    modelos["mean"] = xgb.train(params_mean, dtrain_mean,
                                num_boost_round=bp_mean["n_estimators"],
                                verbose_eval=False)
    logger.info(f"    [xgb_qt] mean fold {fold_num}: reg:squarederror "
                f"(n_est={bp_mean['n_estimators']})")

    bp_representativo = best_by_tau.get(0.50, list(best_by_tau.values())[0])
    # Se adjunta el dict completo {tau: bp} bajo una key interna -- todo el
    # código que ya lee best_params como dict plano (manifest, métricas por
    # fold vía _hp(...), línea ~3613) sigue viendo exactamente las mismas
    # keys que antes (learning_rate, s, n_estimators, ...), sin cambios.
    # _retrain_train_val_qt es el único consumidor de esta key nueva.
    bp_representativo["_best_by_tau_full"] = best_by_tau
    # _optuna_meta: {tau: {best_pinball_val, trial_values}}. Se consume y se
    # elimina en el bucle de grupos de evaluar_banco (igual que
    # _best_by_tau_full), así que el manifest y las métricas por fold siguen
    # viendo exactamente las mismas keys que antes.
    bp_representativo["_optuna_meta"] = optuna_meta
    return modelos, bp_representativo


def _entrenar_train_only_qt(X_tr, y_tr, best_by_tau, std_y, h_val_actual, fold_num):
    """
    Entrena modelos por tau SOLO sobre train, usando hiperparámetros YA
    encontrados (por Optuna, sobre el h representativo del grupo) — no
    corre ninguna búsqueda nueva. Hermana de _retrain_train_val_qt, pero
    sin el paso final de train+val (esta se usa para el modelo "solo-train"
    que evalúa contra VAL como out-of-sample honesto, un h a la vez).

    h_val_actual: el h exacto que se está entrenando (para logging) —
    puede ser distinto del h representativo del grupo (best_by_tau viene
    del representante, se REUTILIZA aquí para cada h del mismo grupo).
    """
    col_names = list(X_tr.columns)
    dtr       = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=col_names)

    modelos_h = {}
    for tau, bp in best_by_tau.items():
        s      = bp["s"]
        n_est  = bp["n_estimators"]
        params = {k: v for k, v in bp.items() if k not in ("s", "n_estimators")}
        params.update({"tree_method": "hist", "seed": 42, "nthread": _XGB_NTHREAD})
        modelos_h[tau] = xgb.train(
            params, dtr, num_boost_round=n_est,
            obj=make_quantile_objective(tau, s, std_y),
            verbose_eval=False,
        )

    bp_mean = best_by_tau.get(0.50, list(best_by_tau.values())[0])
    params_mean = {k: v for k, v in bp_mean.items() if k not in ("s", "n_estimators")}
    params_mean.update({"objective": "reg:squarederror",
                        "tree_method": "hist", "nthread": _XGB_NTHREAD, "seed": 42})
    modelos_h["mean"] = xgb.train(
        params_mean, dtr, num_boost_round=bp_mean["n_estimators"], verbose_eval=False,
    )
    logger.debug(f"    [train_only] fold {fold_num} h={h_val_actual}: "
                f"{len(y_tr)} obs, HP heredados del grupo, sin Optuna propio")
    return modelos_h


def _retrain_train_val_qt(X_tr, y_tr, X_va, y_va, best_by_tau, std_y, fold_num):
    """
    Reentrenamiento final sobre train+val con HPs fijos (ya optimizados por Optuna).

    Diseño deliberado sin early stopping: añadir early stopping requeriría un set
    de evaluación independiente; la única opción sería test, lo que introduciría
    leakage directo. Se usa el n_estimators encontrado por Optuna, que fue
    seleccionado con early stopping activo sobre val durante la búsqueda.

    s viene de best_by_tau[tau] -- calibrado por Optuna, no derivado de std_y.
    std_y se mantiene de y_train para consistencia con el objetivo original.
    preds_val NO usa este modelo; se evalúa con el modelo solo-train para
    mantener val como estimación out-of-sample honesta.
    """
    X_tv      = pd.concat([X_tr, X_va], ignore_index=True)
    y_tv      = pd.concat([y_tr, y_va], ignore_index=True)
    col_names = list(X_tv.columns)
    dtv       = xgb.DMatrix(X_tv.values, label=y_tv.values, feature_names=col_names)

    modelos_tv = {}
    for tau, bp in best_by_tau.items():
        s      = bp["s"]
        n_est  = bp["n_estimators"]
        params = {k: v for k, v in bp.items() if k not in ("s", "n_estimators")}
        params.update({"tree_method": "hist", "seed": 42, "nthread": _XGB_NTHREAD})
        modelos_tv[tau] = xgb.train(
            params, dtv, num_boost_round=n_est,
            obj=make_quantile_objective(tau, s, std_y),
            verbose_eval=False,
        )

    bp_mean = best_by_tau.get(0.50, list(best_by_tau.values())[0])
    params_mean = {k: v for k, v in bp_mean.items() if k not in ("s", "n_estimators")}
    params_mean.update({"objective": "reg:squarederror",
                        "tree_method": "hist", "nthread": _XGB_NTHREAD, "seed": 42})
    modelos_tv["mean"] = xgb.train(
        params_mean, dtv,
        num_boost_round=bp_mean["n_estimators"],
        verbose_eval=False,
    )
    logger.info(f"    [retrain_tv] fold {fold_num}: train+val "
                f"({len(y_tv)} obs, s de Optuna, std_y de train, sin early stopping)")
    return modelos_tv



# ── Dispatchers ───────────────────────────────────────────────────────────────

def entrenar_fold(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num):
    if MODELO_CV == "xgb":
        bp = optimizar_hiperparametros(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num)
        return entrenar_quantiles(X_tr, y_tr, bp, QUANTILES, std_y), bp
    elif MODELO_CV == "lgbm":
        if not _LGBM_OK:
            raise ImportError("MODELO_CV='lgbm' requiere lightgbm instalado")
        bp = optimizar_hiperparametros_lgbm(X_tr, y_tr, X_va, y_va, n_trials, fold_num)
        return entrenar_quantiles_lgbm(X_tr, y_tr, bp.copy(), QUANTILES), bp
    elif MODELO_CV == "xgb_qt":
        return _entrenar_fold_xgb_qt(X_tr, y_tr, X_va, y_va, std_y, n_trials, fold_num)
    raise ValueError(f"MODELO_CV desconocido: {MODELO_CV!r}")


def predecir_fold(modelos, X):
    if MODELO_CV == "lgbm":
        return predecir_lgbm(modelos, X)
    return predecir_y_corregir(modelos, X)


###############################################################################
# PARTE 5 — Preparación de datos  (devuelve también fechas_t para TEST)
###############################################################################

def get_feature_cols(df):
    excluir = COLS_EXCLUIR | {"fecha_th"}
    return [c for c in df.columns
            if c not in excluir and df[c].dtype.kind in ("f", "i", "u", "b")]


def preparar_fold_data(df, fold, cols_feat, banco):
    """
    Retorna:
      X_train, y_train, X_val, y_val, X_test, y_test,
      h_train, h_val, h_test, fechas_t_test, año_corte_regimen

    fechas_t_test: array de pd.Timestamp con la fecha de origen de cada fila
    de TEST — necesario para construir los fan charts por snapshot.

    año_corte_regimen: bloque HMM usado para el feature de régimen en ESTE
    fold (None si USAR_FEATURE_REGIMEN=False o no había bloque válido) —
    se propaga hasta _guardar_preds_test para que step006 sepa qué transmat
    usar (ver cargar_transmat_regimen).
    """
    train_start = fold["train_start"]
    train_end   = fold["train_end"]
    val_start   = fold["val_start"]
    val_end   = fold["val_end"]
    test_start  = fold["test_start"]
    test_end    = fold["test_end"]

    mask_train = (df["fecha_t"] >= train_start) & (df["fecha_t"] <= train_end)
    mask_val   = (df["fecha_t"] >= val_start)   & (df["fecha_t"] <  val_end)
    mask_test  = (df["fecha_t"] >= test_start)  & (df["fecha_t"] <= test_end)

    df_fold_all = df[mask_train | mask_val | mask_test].copy()

    garch_cols = [c for c in ["garch_vol", "garch_vol_tc", "garch_vol_embi"]
                  if c in df_fold_all.columns]
    if garch_cols:
        df_fold_all = reemplazar_garch_fold(df_fold_all, train_end)

    frac_cols = [c for c in df_fold_all.columns if c.endswith("_frac")]
    if frac_cols:
        df_fold_all = reemplazar_ffd_fold(df_fold_all, train_end)

    año_corte_regimen = None
    if USAR_FEATURE_REGIMEN:
        df_fold_all, año_corte_regimen = reemplazar_regimen_fold(
            df_fold_all, train_end, banco)

    df_train = df_fold_all[df_fold_all["fecha_t"] <= train_end]

    # Burn-in: excluir primeros BURN_IN_DIAS_HAB donde MA22 aún no maduró
    if BURN_IN_DIAS_HAB > 0:
        burn_cutoff = fold.get("burn_cutoff",
                               train_start + pd.offsets.BusinessDay(BURN_IN_DIAS_HAB))
        df_train = df_train[df_train["fecha_t"] >= burn_cutoff]
    df_val   = df_fold_all[(df_fold_all["fecha_t"] >= val_start) &
                           (df_fold_all["fecha_t"] <  val_end)]
    df_test  = df_fold_all[df_fold_all["fecha_t"] >= test_start]

    medianas_fold = df_train[cols_feat].median()
    df_train = df_train.copy(); df_val = df_val.copy(); df_test = df_test.copy()
    df_train[cols_feat] = df_train[cols_feat].fillna(medianas_fold)
    df_val[cols_feat]   = df_val[cols_feat].fillna(medianas_fold)
    df_test[cols_feat]  = df_test[cols_feat].fillna(medianas_fold)

    mt  = df_train["target"].notna()
    mv  = df_val["target"].notna()
    mte = df_test["target"].notna()

    X_train        = df_train.loc[mt,  cols_feat].copy()
    y_train        = df_train.loc[mt,  "target"].copy()
    h_train        = df_train.loc[mt,  "h"].values

    X_val          = df_val.loc[mv,    cols_feat].copy()
    y_val          = df_val.loc[mv,    "target"].copy()
    h_val          = df_val.loc[mv,    "h"].values

    X_test         = df_test.loc[mte,  cols_feat].copy()
    y_test         = df_test.loc[mte,  "target"].copy()
    h_test         = df_test.loc[mte,  "h"].values
    fechas_t_test  = pd.to_datetime(df_test.loc[mte, "fecha_t"].values)

    # Convertir columnas pandas ExtensionArray (Int8, boolean, etc.) a float64
    # para compatibilidad con XGBoost DMatrix (no acepta dtype object/nullable)
    for _df in [X_train, X_val, X_test]:
        ext_cols = [c for c in _df.columns
                    if pd.api.types.is_extension_array_dtype(_df[c])]
        if ext_cols:
            _df[ext_cols] = _df[ext_cols].astype("float64")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            h_train, h_val, h_test, fechas_t_test, año_corte_regimen,
            df_fold_all)


###############################################################################
# PARTE 6 — Métricas
###############################################################################

def calcular_metricas_fold(preds, y_true, fold, periodo="test"):
    row = {
        "fold"            : fold["fold"],
        "train_start"     : fold["train_start"].date(),
        "train_end"       : fold["train_end"].date(),
        "n_train_fechas"  : fold["n_train_fechas"],
        "val_start"       : fold["val_start"].date(),
        "val_end"         : fold["val_end"].date(),
        "n_val_fechas"    : fold["n_val_fechas"],
        "test_start"      : fold["test_start"].date(),
        "test_end"        : fold["test_end"].date(),
        "n_test_fechas"   : fold["n_test_fechas"],
        f"n_obs_{periodo}": int(len(y_true)),
        "periodo_metricas": periodo,
        "expanding"       : EXPANDING,
    }
    q_preds = {t: v for t, v in preds.items() if t != "mean"}
    for tau in sorted(q_preds.keys()):
        row[f"pinball_q{int(tau*100):02d}"] = round(
            pinball_loss(y_true, q_preds[tau], tau), 2)
    if {0.05, 0.95}.issubset(q_preds):
        row["coverage_90"] = round(coverage(y_true, q_preds[0.05], q_preds[0.95]), 4)
        row["winkler_90"]  = round(winkler_score(y_true, q_preds[0.05], q_preds[0.95]), 2)
    row["crps_approx"] = round(crps_approx(y_true, q_preds), 2)
    return row


def calcular_metricas_por_h(preds, y_true, h_arr, fold_num):
    rows = []
    for h in np.unique(h_arr):
        mask = h_arr == h
        if mask.sum() < 5:
            continue
        row = {"fold": fold_num, "h": int(h)}
        q_preds = {t: v for t, v in preds.items() if t != "mean"}
        for tau in sorted(q_preds.keys()):
            row[f"pinball_q{int(tau*100):02d}"] = round(
                pinball_loss(y_true[mask], q_preds[tau][mask], tau), 2)
        if {0.05, 0.95}.issubset(q_preds):
            row["coverage_90"] = round(
                coverage(y_true[mask], q_preds[0.05][mask], q_preds[0.95][mask]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


###############################################################################
# PARTE 7 — Visualización
###############################################################################

def graficar_metricas_wfcv(df_test_m, banco):
    if df_test_m.empty:
        return
    folds = df_test_m["fold"].values
    modo  = "EXPANDING" if EXPANDING else "ROLLING"

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Walk-forward CV v3 [{modo}] — {banco}  [métricas TEST out-of-sample]\n"
        f"TRAIN {VENTANA_TRAIN_AÑOS}yr{'(min)' if EXPANDING else ''} / "
        f"VAL {VENTANA_VAL_AÑOS}yr (Optuna) / TEST {VENTANA_TEST_AÑOS}yr / "
        f"paso {PASO_AÑOS}yr / purge {PURGE_DIAS_HAB}dh / burn-in {BURN_IN_DIAS_HAB}dh",
        fontweight="bold", fontsize=10,
    )
    metricas_config = [
        ("pinball_q50", "Pinball Q50 — TEST OOS",   "steelblue",  None),
        ("coverage_90", "Coverage 90% — TEST OOS",  "seagreen",   0.90),
        ("winkler_90",  "Winkler score — TEST",     "darkorange", None),
        ("crps_approx", "CRPS aprox. — TEST",       "crimson",    None),
    ]
    for ax, (col, titulo, color, hline) in zip(axes, metricas_config):
        if col not in df_test_m.columns:
            ax.set_visible(False); continue
        vals = df_test_m[col].values.astype(float)
        ax.plot(folds, vals, "o-", color=color, lw=2, ms=7, zorder=4)
        if len(folds) >= 3:
            z    = np.polyfit(folds, vals, 1)
            xfit = np.linspace(folds[0], folds[-1], 50)
            ax.plot(xfit, np.polyval(z, xfit), "--", color=color, alpha=0.5, lw=1.2,
                    label=f"tendencia: {z[0]:+.2e}/fold")
            ax.legend(fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="grey", lw=1.2, ls="--", alpha=0.6,
                       label=f"nominal={hline:.0%}")
            ax.legend(fontsize=8)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Fold", fontsize=9)
        ax.set_xticks(folds)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _, _c=col: f"{x:.1%}" if _c == "coverage_90" else f"{x:,.0f}"))
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v3_metricas_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico métricas TEST: {nombre}")


def graficar_comparacion_val_test(df_val_m, df_test_m, banco):
    if df_val_m.empty or df_test_m.empty:
        return
    if "coverage_90" not in df_val_m.columns:
        return
    folds = df_test_m["fold"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"VAL (Optuna, sesgo) vs TEST (OOS real) — {banco} "
        f"[{'EXPANDING' if EXPANDING else 'ROLLING'}]",
        fontweight="bold", fontsize=11,
    )
    for ax, col, titulo in [
        (axes[0], "coverage_90", "Coverage 90%"),
        (axes[1], "pinball_q50", "Pinball Q50"),
    ]:
        if col not in df_val_m.columns or col not in df_test_m.columns:
            continue
        v_val  = df_val_m[col].values.astype(float)
        v_test = df_test_m[col].values.astype(float)
        ax.plot(folds, v_val,  "s--", color="steelblue", lw=1.5, ms=7, label="VAL (Optuna)")
        ax.plot(folds, v_test, "o-",  color="crimson",   lw=2,   ms=7, label="TEST (OOS)")
        if col == "coverage_90":
            ax.axhline(0.90, color="grey", lw=1.2, ls=":", alpha=0.7, label="nominal 90%")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
        diff = float(np.mean(v_val - v_test))
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Fold  (sesgo promedio VAL−TEST: {diff:+.4f})", fontsize=9)
        ax.set_xticks(folds)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v3_val_vs_test_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Comparación VAL vs TEST: {nombre}")


def graficar_fanchart_acum_test_fold(
    preds_test: dict,
    y_test: np.ndarray,
    h_test: np.ndarray,
    fechas_t_test,
    fold: dict,
    banco: str,
    dir_out: Path | None = None,
    variante: str = "banda",
):
    """
    Fan chart de flujo neto ACUMULADO (cumsum sobre h) para un fold TEST.
    Consolida las 3 variantes que antes eran funciones separadas — mismo
    cálculo base (cumsum sobre h por snapshot de origen), solo cambia qué
    capas se dibujan:

      variante="banda"      (antes graficar_fanchart_acum_test_fold):
        bandas Q01-Q99/Q05-Q95, línea Q50+media, dispersión de puntos
        realizados coloreada por dentro/fuera de banda. Nombre de archivo:
        fanchart_acum_test_fold{N}_{banco}.png

      variante="punto"      (antes graficar_fanchart_acum_punto_test_fold):
        solo líneas limpias (realizado, mediana, media), sin banda ni
        dispersión — más legible cuando hay muchos orígenes superpuestos.
        Nombre: fanchart_acum_punto_test_fold{N}_{banco}.png

      variante="punto_q05"  (antes graficar_fanchart_acum_punto_q05_test_fold):
        igual que "punto" + línea de escenario estresado en el día terminal:
        cumsum(Q50) hasta h-1, cae a Q05 solo en el último día h.
        Nombre: fanchart_acum_puntq05_test_fold{N}_{banco}.png
    """
    if variante not in ("banda", "punto", "punto_q05"):
        raise ValueError(f"variante debe ser 'banda'/'punto'/'punto_q05', recibido: {variante!r}")

    fechas_unicas = pd.DatetimeIndex(sorted(set(fechas_t_test)))
    if len(fechas_unicas) == 0:
        return

    test_start = fold["test_start"]
    n_snap     = FANCHART_N_SNAPSHOTS
    meses_paso = 12.0 / n_snap

    origenes = []
    for i in range(n_snap):
        target  = test_start + pd.DateOffset(months=int(round(i * meses_paso)))
        diffs   = np.abs((fechas_unicas - target).total_seconds())
        nearest = fechas_unicas[np.argmin(diffs)]
        if nearest not in origenes:
            origenes.append(nearest)

    if not origenes:
        return

    ncols = 2
    nrows = int(np.ceil(len(origenes) / ncols))
    modo  = "EXPANDING" if EXPANDING else "ROLLING"

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5), sharey=False)
    axes_flat = np.array(axes).flatten()

    _titulos_suptitle = {
        "banda":     f"Fan chart ACUMULADO TEST OOS — Fold {fold['fold']} — {banco} [{modo}]",
        "punto":     f"Fan chart ACUMULADO TEST OOS — Fold {fold['fold']} — {banco} [{modo}]",
        "punto_q05": f"Acumulado TEST OOS — Realizado / Media / Mediana / Q05-acum — "
                     f"Fold {fold['fold']} — {banco} [{modo}]",
    }
    fig.suptitle(
        f"{_titulos_suptitle[variante]}\n"
        f"TEST: {fold['test_start'].date()} → {fold['test_end'].date()}  |  "
        f"TRAIN hasta: {fold['train_end'].date()}",
        fontweight="bold", fontsize=11,
    )

    for ax, t0 in zip(axes_flat, origenes):
        mask = np.array([pd.Timestamp(f) == t0 for f in fechas_t_test])
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        h_s = h_test[mask]
        y_s = y_test[mask]
        p_s = {tau: arr[mask] for tau, arr in preds_test.items()}

        order = np.argsort(h_s)
        h_s   = h_s[order]
        y_s   = y_s[order]
        p_s   = {tau: arr[order] for tau, arr in p_s.items()}

        y_cum = np.cumsum(y_s)
        p_cum = {tau: np.cumsum(arr) for tau, arr in p_s.items()}

        if variante == "banda":
            if {0.01, 0.99}.issubset(p_cum):
                ax.fill_between(h_s, p_cum[0.01] / 1e6, p_cum[0.99] / 1e6,
                                alpha=0.12, color="steelblue", label="Q01-Q99 (CV)")
            if {0.05, 0.95}.issubset(p_cum):
                ax.fill_between(h_s, p_cum[0.05] / 1e6, p_cum[0.95] / 1e6,
                                alpha=0.28, color="steelblue", label="Q05-Q95 (CV)")
            if 0.50 in p_cum:
                lw_q50 = 1.4 if "mean" in p_cum else 1.8
                ls_q50 = "--" if "mean" in p_cum else "-"
                ax.plot(h_s, p_cum[0.50] / 1e6, color="steelblue", lw=lw_q50,
                        ls=ls_q50, zorder=3, label="Q50 (CV)")
            if "mean" in p_cum:
                ax.plot(h_s, p_cum["mean"] / 1e6, color="navy", lw=2.2,
                        zorder=4, label="Media (CV)")

            q_lo_cum   = p_cum.get(0.05, np.full_like(y_cum, -np.inf))
            q_hi_cum   = p_cum.get(0.95, np.full_like(y_cum,  np.inf))
            dentro_cum = (y_cum >= q_lo_cum) & (y_cum <= q_hi_cum)

            ax.plot(h_s, y_cum / 1e6, color="dimgray", lw=1.0, ls="--",
                    zorder=4, alpha=0.75, label="Realizado acum.")
            ax.scatter(h_s[dentro_cum],  y_cum[dentro_cum]  / 1e6, color="seagreen", s=20,
                       zorder=5, label="Dentro Q05-Q95")
            ax.scatter(h_s[~dentro_cum], y_cum[~dentro_cum] / 1e6, color="crimson",  s=20,
                       zorder=5, label="Fuera Q05-Q95")

            cov_snap = float(dentro_cum.mean())
            ax.set_title(
                f"Origen: {t0.strftime('%Y-%m-%d')}\nCoverage 90% acum.: {cov_snap:.1%}",
                fontsize=9, fontweight="bold",
            )
        else:
            ax.plot(h_s, y_cum / 1e6, color="dimgray", lw=2.0, ls="--",
                    zorder=4 if variante == "punto" else 5, label="Realizado acum.")
            if 0.50 in p_cum:
                ax.plot(h_s, p_cum[0.50] / 1e6, color="steelblue", lw=2.0,
                        ls="--", zorder=3,
                        label="Mediana Q50 (CV)" if variante == "punto" else "Mediana Q50 acum.")
            if "mean" in p_cum:
                ax.plot(h_s, p_cum["mean"] / 1e6, color="crimson", lw=2.0,
                        zorder=4, label="Media (CV)" if variante == "punto" else "Media acum.")

            if variante == "punto_q05":
                if 0.50 in p_s and 0.05 in p_s:
                    med_stress_cum = np.cumsum(p_s[0.50]) - (p_s[0.50] - p_s[0.05])
                    ax.plot(h_s, med_stress_cum / 1e6, color="darkorange", lw=1.8,
                            ls=":", zorder=2, label="Mediana acum. − riesgo P5 día h")

            ax.set_title(f"Origen: {t0.strftime('%Y-%m-%d')}", fontsize=9, fontweight="bold")

        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=8)
        ax.set_ylabel("Flujo acumulado D-R (MM USD)", fontsize=8)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3, ls="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, alpha=0.25)
        if variante != "banda":
            ax.legend(fontsize=8, loc="best")

    if variante == "banda":
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            axes_flat[0].legend(handles, labels, fontsize=7, loc="best")
    for ax in axes_flat[len(origenes):]:
        ax.set_visible(False)

    plt.tight_layout()
    _dir = dir_out if dir_out is not None else DIR_FANCHARTS
    _sufijo_archivo = {"banda": "fanchart_acum_test_fold",
                      "punto": "fanchart_acum_punto_test_fold",
                      "punto_q05": "fanchart_acum_puntq05_test_fold"}[variante]
    nombre = _dir / f"{_sufijo_archivo}{fold['fold']:02d}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _etiqueta_log = {"banda": "ACUMULADO", "punto": "ACUMULADO PUNTO",
                     "punto_q05": "ACUMULADO PUNTO+Q05"}[variante]
    logger.info(f"  Fan chart {_etiqueta_log} TEST fold {fold['fold']}: {nombre.name}")


def graficar_cobertura_por_h(df_por_h, banco, sufijo="test"):
    if df_por_h.empty or "coverage_90" not in df_por_h.columns:
        return
    pivot = df_por_h.pivot(index="fold", columns="h", values="coverage_90")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.25), 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0.70, vmax=1.0, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Coverage 90%")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Fold {i}" for i in pivot.index], fontsize=8)
    ax.set_xlabel("Horizonte h (días hábiles)", fontsize=9)
    ax.set_title(
        f"Coverage 90% por fold y horizonte — {banco} [{sufijo.upper()} OOS] "
        f"[{'EXPANDING' if EXPANDING else 'ROLLING'}]\n"
        "(verde=bien calibrado ~90%, rojo=sub/sobre-cobertura)",
        fontsize=10,
    )
    xticks_pos = range(0, len(pivot.columns), 10)
    ax.set_xticks([p for p in xticks_pos if p < len(pivot.columns)])
    ax.set_xticklabels([str(pivot.columns[p]) for p in xticks_pos
                        if p < len(pivot.columns)], fontsize=8)
    plt.tight_layout()
    nombre = DIR_PLOTS / f"wfcv_v3_cobertura_{sufijo}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Heatmap cobertura ({sufijo.upper()}): {nombre}")


def graficar_hiperparametros_wfcv(df_test_m, banco):
    if df_test_m.empty:
        return
    HP_CONFIG = [
        ("s_optimo",         "s (smoothing arctan)",  "steelblue",   "log"),
        ("learning_rate",    "learning rate",          "darkorange",  "log"),
        ("max_depth",        "max depth",              "seagreen",    "linear"),
        ("n_estimators",     "n_estimators",           "crimson",     "linear"),
        ("min_child_weight", "min child weight",       "purple",      "linear"),
        ("subsample",        "subsample",              "saddlebrown", "linear"),
        ("colsample_bytree", "colsample bytree",       "teal",        "linear"),
        ("reg_alpha",        "reg_alpha (L1)",         "goldenrod",   "log"),
        ("reg_lambda",       "reg_lambda (L2)",        "slategrey",   "log"),
    ]
    hp_presentes = [(col, lbl, clr, sc) for col, lbl, clr, sc in HP_CONFIG
                    if col in df_test_m.columns]
    if not hp_presentes:
        return
    n     = len(hp_presentes)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    folds = df_test_m["fold"].values
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    axes_flat = np.array(axes).flatten()
    fig.suptitle(
        f"Estabilidad HP — Walk-forward CV v3 [{' EXPANDING' if EXPANDING else 'ROLLING'}] — {banco}\n"
        f"TRAIN {VENTANA_TRAIN_AÑOS}yr / VAL {VENTANA_VAL_AÑOS}yr / TEST {VENTANA_TEST_AÑOS}yr  "
        f"({len(folds)} folds)",
        fontweight="bold", fontsize=11,
    )
    for ax, (col, label, color, scale) in zip(axes_flat, hp_presentes):
        vals    = df_test_m[col].values.astype(float)
        mu, sig = vals.mean(), vals.std()
        ax.plot(folds, vals, "o-", color=color, lw=2, ms=7, zorder=4, label=label)
        ax.axhline(mu, color=color, lw=1.0, ls="--", alpha=0.5, label=f"media={mu:.4g}")
        if sig > 0:
            ax.axhspan(mu - sig, mu + sig, alpha=0.08, color=color,
                       label=f"±1σ ({sig:.4g})")
        if len(folds) >= 3:
            z    = np.polyfit(folds, vals if scale == "linear"
                              else np.log(np.maximum(vals, 1e-12)), 1)
            xfit = np.linspace(folds[0], folds[-1], 50)
            yfit = np.polyval(z, xfit)
            if scale == "log":
                yfit = np.exp(yfit)
            ax.plot(xfit, yfit, "-", color="black", lw=1.0, alpha=0.4, label="tendencia")
        if scale == "log":
            ax.set_yscale("log")
        cv = sig / abs(mu) if abs(mu) > 1e-9 else 0.0
        ax.set_title(f"{label}  (CV={cv:.1%})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Fold", fontsize=8)
        ax.set_xticks(folds)
        ax.legend(fontsize=7, framealpha=0.85, loc="best")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    nombre = DIR_PLOTS / f"wfcv_v3_hiperparametros_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Gráfico hiperparámetros: {nombre}")


def _extraer_importancias(modelos, cols_feat):
    """
    Devuelve {feature: gain_promedio_entre_cuantiles} para XGBoost o LightGBM.
    Los features sin importancia en algún cuantil reciben gain=0.
    """
    acum = {f: 0.0 for f in cols_feat}
    n    = 0
    for tau, model in modelos.items():
        if MODELO_CV == "lgbm" and _LGBM_OK:
            names  = model.feature_name()
            gains  = model.feature_importance(importance_type="gain")
            imp    = dict(zip(names, gains.astype(float)))
        else:
            # xgb.Booster: solo devuelve features con importancia > 0
            imp = model.get_score(importance_type="gain")

        for feat in cols_feat:
            acum[feat] = acum[feat] + float(imp.get(feat, 0.0))
        n += 1

    if n > 0:
        acum = {f: v / n for f, v in acum.items()}
    return acum


def graficar_importancia_por_fold(
    importancias_folds: list,
    cols_feat: list,
    banco: str,
):
    """
    Genera dos gráficos a partir de la lista de importancias por fold:

    1. Heatmap (Features × Folds) — ganancia normalizada por fold.
       Verde intenso = feature dominante en ese fold; blanco/amarillo = marginal.
       Útil para detectar features consistentes vs. régimen-dependientes.

    2. Rank-stability (Top-10) — muestra cómo cambia el ranking de las features
       más importantes entre folds.  Línea plana = feature robusto;
       línea con saltos grandes = feature régimen-dependiente.

    También guarda un CSV con las ganancias brutas por fold.
    """
    if not importancias_folds:
        return

    # ── Pivot: filas = feature, columnas = fold ───────────────────────────────
    # importancias_folds ahora guarda "importancias_por_h" (un dict {h: {feat:
    # gain}} por fold, ya que cada h entrena su propio modelo) en vez del
    # "importancias" plano de antes. Se promedia entre horizontes para
    # mantener el mismo significado del heatmap: una columna por fold.
    registros = []
    for item in importancias_folds:
        fold_id = item["fold"]
        if "importancias_por_h" in item:
            imp_por_h = item["importancias_por_h"]
            imp = {}
            for feat in cols_feat:
                vals = [d.get(feat, 0.0) for d in imp_por_h.values()]
                imp[feat] = float(np.mean(vals)) if vals else 0.0
        else:
            imp = item["importancias"]   # compat: manifests viejos sin por-h
        total   = sum(imp.values()) or 1.0
        for feat, gain in imp.items():
            registros.append({
                "fold"     : fold_id,
                "feature"  : feat,
                "gain_norm": gain / total,
                "gain_raw" : gain,
            })

    df_imp = pd.DataFrame(registros)
    if df_imp.empty:
        return

    pivot_norm = df_imp.pivot_table(
        index="feature", columns="fold", values="gain_norm", aggfunc="mean"
    ).fillna(0.0)
    pivot_raw  = df_imp.pivot_table(
        index="feature", columns="fold", values="gain_raw",  aggfunc="mean"
    ).fillna(0.0)

    # Ordenar por importancia media descendente
    pivot_norm["_mean"] = pivot_norm.mean(axis=1)
    pivot_norm = pivot_norm.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    pivot_raw  = pivot_raw.loc[pivot_norm.index]  # mismo orden

    TOP_N  = min(25, len(pivot_norm))
    modo   = "EXPANDING" if EXPANDING else "ROLLING"
    folds  = sorted(pivot_norm.columns.tolist())

    # ── Gráfico 1: Heatmap ────────────────────────────────────────────────────
    top_feats_h = pivot_norm.iloc[:TOP_N]
    fig_h, ax_h = plt.subplots(
        figsize=(max(8, len(folds) * 1.2), max(6, TOP_N * 0.45))
    )

    im = ax_h.imshow(
        top_feats_h.values, aspect="auto", cmap="YlOrRd",
        vmin=0.0, vmax=max(top_feats_h.values.max() * 1.05, 1e-9),
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax_h, label="Ganancia normalizada por fold")

    ax_h.set_yticks(range(TOP_N))
    ax_h.set_yticklabels(top_feats_h.index.tolist(), fontsize=8)
    ax_h.set_xticks(range(len(folds)))
    ax_h.set_xticklabels([f"F{c}" for c in folds], fontsize=9)
    ax_h.set_xlabel("Fold", fontsize=10)
    ax_h.set_ylabel("Feature", fontsize=10)
    ax_h.set_title(
        f"Importancia de features por fold — {banco} [{modo}]\n"
        f"Top {TOP_N} features · ganancia XGBoost normalizada por fold "
        f"(TRAIN {VENTANA_TRAIN_AÑOS}yr / VAL {VENTANA_VAL_AÑOS}yr / TEST {VENTANA_TEST_AÑOS}yr)",
        fontweight="bold", fontsize=11,
    )

    # Anotar valores en celdas cuando la tabla es pequeña
    if TOP_N <= 20 and len(folds) <= 12:
        for i in range(TOP_N):
            for j in range(len(folds)):
                val = top_feats_h.values[i, j]
                ax_h.text(
                    j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if val > top_feats_h.values.max() * 0.6 else "black",
                )

    plt.tight_layout()
    nombre_h = DIR_PLOTS / f"wfcv_v3_importancia_heatmap_{banco}.png"
    plt.savefig(nombre_h, dpi=150, bbox_inches="tight")
    plt.close(fig_h)
    logger.info(f"  Heatmap importancia: {nombre_h.name}")

    # ── Gráfico 2: Rank-stability ─────────────────────────────────────────────
    TOP_RANK   = min(10, len(pivot_norm))
    top_feats_r = pivot_norm.iloc[:TOP_RANK].index.tolist()

    # Rango en cada fold (1 = mayor ganancia) sobre TODOS los features
    rank_data = {feat: [] for feat in top_feats_r}
    for fold_id in folds:
        col_vals = pivot_norm[fold_id]
        ranked   = col_vals.rank(ascending=False, method="min")
        for feat in top_feats_r:
            rank_data[feat].append(int(ranked.get(feat, len(pivot_norm) + 1)))

    fig_r, ax_r = plt.subplots(figsize=(max(7, len(folds) * 1.2), 5))
    cmap_r = plt.cm.tab10
    for i, feat in enumerate(top_feats_r):
        ax_r.plot(
            folds, rank_data[feat], "o-",
            lw=2, ms=7, color=cmap_r(i / max(TOP_RANK, 1)),
            label=feat, alpha=0.85,
        )

    ax_r.invert_yaxis()   # rango 1 arriba
    ax_r.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_r.set_xticks(folds)
    ax_r.set_xticklabels([f"Fold {f}" for f in folds], fontsize=9)
    ax_r.set_xlabel("Fold", fontsize=10)
    ax_r.set_ylabel("Ranking (1 = mayor ganancia)", fontsize=10)
    ax_r.set_title(
        f"Estabilidad de ranking — Top {TOP_RANK} features — {banco} [{modo}]\n"
        f"Línea plana = feature robusto · saltos grandes = régimen-dependiente",
        fontweight="bold", fontsize=10,
    )
    ax_r.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.85)
    ax_r.grid(True, alpha=0.25)
    plt.tight_layout()
    nombre_r = DIR_PLOTS / f"wfcv_v3_importancia_ranking_{banco}.png"
    plt.savefig(nombre_r, dpi=150, bbox_inches="tight")
    plt.close(fig_r)
    logger.info(f"  Ranking importancia: {nombre_r.name}")

    # ── CSV importancias brutas ────────────────────────────────────────────────
    df_csv = pivot_raw.copy().reset_index()
    df_csv.columns.name = None
    ruta_csv = DIR_MODO / f"wfcv_v3_importancias_{banco}.csv"
    df_csv.to_csv(ruta_csv, index=False)
    logger.info(f"  CSV importancias: {ruta_csv.name}")


def graficar_fanchart_test_fold(
    preds_test: dict,
    y_test: np.ndarray,
    h_test: np.ndarray,
    fechas_t_test,
    fold: dict,
    banco: str,
    preds_overlay: dict | None = None,   # step004 predictions para comparación
    dir_out: Path | None = None,         # carpeta de salida; None → DIR_FANCHARTS
):
    """
    Fan chart TEST out-of-sample para un fold.

    Selecciona hasta FANCHART_N_SNAPSHOTS fechas de origen separadas ~3 meses
    dentro del período TEST. Para cada fecha muestra:
      · Bandas Q01-Q99 (muy transparente) y Q05-Q95 (media)
      · Línea Q50 (mediana predicha)
      · Puntos realizados: verde = dentro de Q05-Q95 / rojo = fuera

    Ayuda a detectar si el modelo sobre/sub-estima en horizontes cortos o largos,
    y en qué trimestres del período TEST falla la cobertura.
    """
    fechas_unicas = pd.DatetimeIndex(sorted(set(fechas_t_test)))
    if len(fechas_unicas) == 0:
        return

    # Seleccionar fechas de origen: una cada ~(12/N_SNAPSHOTS) meses del TEST
    test_start = fold["test_start"]
    n_snap     = FANCHART_N_SNAPSHOTS
    meses_paso = 12.0 / n_snap

    origenes = []
    for i in range(n_snap):
        target  = test_start + pd.DateOffset(months=int(round(i * meses_paso)))
        diffs   = np.abs((fechas_unicas - target).total_seconds())
        nearest = fechas_unicas[np.argmin(diffs)]
        if nearest not in origenes:
            origenes.append(nearest)

    if not origenes:
        return

    # Layout 2×2
    ncols = 2
    nrows = int(np.ceil(len(origenes) / ncols))
    modo  = "EXPANDING" if EXPANDING else "ROLLING"

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5), sharey=False)
    axes_flat = np.array(axes).flatten()

    s4_tag = "  |  🟠 naranja = Step004 (GARCH global, lookahead)" if preds_overlay is not None else ""
    fig.suptitle(
        f"Fan chart TEST OOS — Fold {fold['fold']} — {banco} [{modo}]\n"
        f"TEST: {fold['test_start'].date()} → {fold['test_end'].date()}  |  "
        f"TRAIN hasta: {fold['train_end'].date()}{s4_tag}",
        fontweight="bold", fontsize=11,
    )

    for ax, t0 in zip(axes_flat, origenes):
        mask = np.array([pd.Timestamp(f) == t0 for f in fechas_t_test])
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        h_s = h_test[mask]
        y_s = y_test[mask]
        p_s = {tau: arr[mask] for tau, arr in preds_test.items()}

        order = np.argsort(h_s)
        h_s   = h_s[order]
        y_s   = y_s[order]
        p_s   = {tau: arr[order] for tau, arr in p_s.items()}

        # Predicciones step004 overlay (misma máscara y orden)
        p_s4 = None
        if preds_overlay is not None:
            p_s4 = {tau: arr[mask][order] for tau, arr in preds_overlay.items()
                    if tau != "mean"}

        # Bandas de incertidumbre — step005 (azul)
        if {0.01, 0.99}.issubset(p_s):
            ax.fill_between(h_s, p_s[0.01] / 1e6, p_s[0.99] / 1e6,
                            alpha=0.12, color="steelblue", label="Q01-Q99 (CV)")
        if {0.05, 0.95}.issubset(p_s):
            ax.fill_between(h_s, p_s[0.05] / 1e6, p_s[0.95] / 1e6,
                            alpha=0.28, color="steelblue", label="Q05-Q95 (CV)")
        if 0.50 in p_s:
            lw_q50 = 1.4 if "mean" in p_s else 1.8
            ls_q50 = "--" if "mean" in p_s else "-"
            ax.plot(h_s, p_s[0.50] / 1e6, color="steelblue", lw=lw_q50,
                    ls=ls_q50, zorder=3, label="Q50 (CV)")
        if "mean" in p_s:
            ax.plot(h_s, p_s["mean"] / 1e6, color="navy", lw=2.2,
                    zorder=4, label="Media (CV)")

        # Overlay step004 — naranja discontinuo
        if p_s4 is not None:
            if {0.05, 0.95}.issubset(p_s4):
                ax.fill_between(h_s, p_s4[0.05] / 1e6, p_s4[0.95] / 1e6,
                                alpha=0.10, color="darkorange")
            if 0.50 in p_s4:
                ax.plot(h_s, p_s4[0.50] / 1e6, color="darkorange", lw=1.6,
                        ls="--", zorder=4, alpha=0.85, label="Q50 Step004")

        # Realizados: línea punteada + scatter coloreado por cobertura
        q_lo   = p_s.get(0.05, np.full_like(y_s, -np.inf))
        q_hi   = p_s.get(0.95, np.full_like(y_s,  np.inf))
        dentro = (y_s >= q_lo) & (y_s <= q_hi)

        ax.plot(h_s, y_s / 1e6, color="dimgray", lw=1.0, ls="--",
                zorder=4, alpha=0.75, label="Realizado")
        ax.scatter(h_s[dentro],  y_s[dentro]  / 1e6, color="seagreen", s=20,
                   zorder=5, label="Dentro Q05-Q95")
        ax.scatter(h_s[~dentro], y_s[~dentro] / 1e6, color="crimson",  s=20,
                   zorder=5, label="Fuera Q05-Q95")

        cov_snap = float(dentro.mean())
        ax.set_title(
            f"Origen: {t0.strftime('%Y-%m-%d')}\nCoverage 90%: {cov_snap:.1%}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=8)
        ax.set_ylabel("Flujo D-R (MM USD)", fontsize=8)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3, ls="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, alpha=0.25)

    # Leyenda en el primer panel; ocultar paneles sobrantes
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(handles, labels, fontsize=7, loc="best")
    for ax in axes_flat[len(origenes):]:
        ax.set_visible(False)

    plt.tight_layout()
    _dir = dir_out if dir_out is not None else DIR_FANCHARTS
    nombre = _dir / f"fanchart_test_fold{fold['fold']:02d}_{banco}.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Fan chart TEST fold {fold['fold']}: {nombre.name}")


###############################################################################
# PARTE 7-bis — Diagnóstico de features (gain / block-perm / SHAP)
#
# v3.7.0: granularidad fold × h × τ. Antes se promediaba sobre τ dentro de
# _diag_*_promedio y luego sobre h en evaluar_banco, dejando una matriz
# folds × features. Ahora se conservan ambas dimensiones — mismo cómputo,
# más información. Las tres señales quedan normalizadas de forma que sus
# celdas son comparables entre horizontes, folds y cuantiles (ver el docstring
# de cada una); por eso los heatmaps NO se re-normalizan por fold.
###############################################################################

def _diag_predict_un_modelo(model, X):
    if MODELO_CV == "lgbm" and _LGBM_OK:
        return model.predict(X.values)
    return model.predict(xgb.DMatrix(X))


def _tau_label(tau) -> str:
    """Etiqueta canónica de un modelo: 'q01'…'q99' o 'mean'."""
    return "mean" if tau == "mean" else f"q{int(round(float(tau) * 100)):02d}"


def _orden_taus(modelos: dict) -> list:
    """Cuantiles en orden ascendente y 'mean' al final."""
    qs = sorted(t for t in modelos if t != "mean")
    return qs + (["mean"] if "mean" in modelos else [])


def _sincos_pairs(cols) -> dict:
    """
    Detecta features cíclicos codificados como par _sin / _cos.

    Devuelve {base: (sin_col, cos_col)} solo cuando AMBOS componentes están
    presentes. Sustituye a la lista fija _CICL_BASES_PERM: un feature cíclico
    nuevo en la matriz entra al diagnóstico sin tocar código.
    """
    cols = list(cols)
    sin_map = {c[:-4]: c for c in cols if c.endswith("_sin")}
    cos_map = {c[:-4]: c for c in cols if c.endswith("_cos")}
    return {base: (sin_map[base], cos_map[base])
            for base in sin_map if base in cos_map}


def _consolidar_sincos_pivot(pivot: pd.DataFrame, pairs: dict) -> pd.DataFrame:
    """
    En un pivot (features × h), SUMA las filas sin+cos en una única fila 'base'
    y elimina las individuales.

      • perm : las filas ya llevan el Δ conjunto (sin=joint, cos=0) → suma = joint.
      • gain/shap : la suma aproxima la importancia total del feature circular.

    Colisión de nombres: 'dias_al_cierre_mes' existe también como feature crudo,
    así que la base derivada de 'dias_al_cierre_mes_sin/_cos' coincide con una
    fila que YA está en el pivot. Concatenar sin más produce índice duplicado y
    el graficado revienta con ValueError. Con colisión se etiqueta la fila
    consolidada como '<base>_sincos', lo que además deja ver por separado cuánto
    aporta cada codificación del mismo calendario.
    """
    rows_new: dict = {}
    rows_drop: list = []
    for base, (sin_c, cos_c) in pairs.items():
        if sin_c in pivot.index and cos_c in pivot.index:
            nombre = f"{base}_sincos" if base in pivot.index else base
            rows_new[nombre] = pivot.loc[sin_c] + pivot.loc[cos_c]
            rows_drop += [sin_c, cos_c]
    if rows_new:
        pivot = pd.concat([pivot.drop(index=rows_drop), pd.DataFrame(rows_new).T])
    return pivot


def _shap_compat_booster(model):
    """
    Parchea xgb.Booster.predict a nivel de CLASE (una vez por sesión) para
    traducir el kwarg deprecado ntree_limit a iteration_range (removido en
    XGBoost 2.0).

    Por qué a nivel de clase y no de instancia: SHAP < 0.43 llama a predict
    desde dos sitios distintos —
      1. TreeExplainer.__init__ (expected_value)  → vía atributo de instancia
      2. _compute_tree_shap / assert_additivity   → xgb.Booster.predict(self, …)
         como método de clase, saltándose el lookup de instancia.
    Un parche por instancia arregla (1) pero no (2), y el fallo se traga en el
    except de más abajo dejando la columna shap_* en NaN SIN error visible.

    Devuelve el Booster listo para TreeExplainer (los modelos del core ya son
    xgb.Booster crudos; para LGBM se devuelve el modelo tal cual).
    """
    if MODELO_CV == "lgbm" and _LGBM_OK:
        return model
    if not getattr(xgb.Booster, "_shap_ntree_patched", False):
        _orig_cls = xgb.Booster.predict

        def _compat(self, data, ntree_limit=None, **kw):
            if ntree_limit is not None and "iteration_range" not in kw:
                kw["iteration_range"] = (0, int(ntree_limit))
            return _orig_cls(self, data, **kw)

        xgb.Booster.predict = _compat
        xgb.Booster._shap_ntree_patched = True
        logger.debug("xgb.Booster.predict parcheado para SHAP < 0.43 (una sola vez)")
    return model.get_booster() if hasattr(model, "get_booster") else model


def _diag_gain_h(modelos: dict, cols_feat: list) -> dict:
    """
    Participación de cada feature en la ganancia total del modelo, por cuantil.

    Dos decisiones que cambian respecto de la versión anterior:

    1. total_gain en vez de gain. XGBoost define 'gain' como la ganancia
       PROMEDIO de los splits que usan el feature: uno usado una sola vez en un
       split afortunado obtiene un valor enorme, mientras que otro usado en
       treinta splits de calidad media obtiene uno moderado. total_gain mide la
       contribución acumulada, que es lo que interesa.

    2. Se normaliza a participación (share del total). El gain crudo escala como
       n·_scale², y _scale = π(s²+std_y²)²/(2s³) depende de std_y y del s del
       grupo — entre grupos de un mismo fold hay hasta 5x de diferencia sin que
       los datos cambien. Al dividir por la suma ese factor, común a todos los
       features del mismo modelo, se cancela y los valores pasan a ser
       comparables entre horizontes, folds y cuantiles. SIN esta normalización
       el heatmap de gain no se puede leer a lo ancho de h.

    Returns {tau: Series de participaciones (suman 1)}
    """
    out: dict = {}
    for tau, model in modelos.items():
        if MODELO_CV == "lgbm" and _LGBM_OK:
            imp = dict(zip(model.feature_name(),
                           model.feature_importance(importance_type="gain").astype(float)))
        else:
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            imp = booster.get_score(importance_type="total_gain")
        s = pd.Series({f: float(imp.get(f, 0.0)) for f in cols_feat}, dtype=float)
        total = s.sum()
        out[tau] = s / total if total > 0 else s
    return out


def _perdida_de_tau(tau):
    """
    Pérdida natural del modelo: pinball para los cuantiles, MSE para 'mean'
    (se entrena con reg:squarederror). Los valores de la fila 'mean' son
    interpretables pero NO comparables con los de las filas de cuantiles.
    """
    if tau == "mean":
        return lambda yy, pp: float(np.mean((yy - pp) ** 2))
    return lambda yy, pp, _t=tau: float(pinball_loss(yy, pp, _t))


def _diag_perm_h(modelos: dict, X_val: pd.DataFrame, y_val, cols_feat: list,
                 fold_num: int) -> dict:
    """
    Block-permutation importance en VAL (OOS), por cuantil.

    Devuelve Δloss RELATIVO a la pérdida base (Δloss / |loss_base|), que es
    adimensional y por tanto comparable entre horizontes, folds y cuantiles.
    La versión anterior devolvía el Δpinball crudo, en unidades del target, que
    escala con std_y y no se puede comparar a lo ancho de h.

    NO se normaliza a participación como gain y shap: estos Δ pueden ser
    NEGATIVOS —permutar un feature a veces mejora el loss por azar— y con
    valores de signo mixto la suma puede acercarse a cero y las participaciones
    explotan.

    Pares sin/cos (convención v3.7.0): se permutan con el MISMO vector de
    shuffle (Strobl et al. 2008 — permutar solo un componente deja al modelo
    recuperar señal del otro y subestima la importancia real del cíclico), y el
    Δ conjunto completo se almacena en la columna _sin, con 0.0 en la _cos.
    _consolidar_sincos_pivot las SUMA → Δ conjunto correcto. La columna _cos
    valiendo exactamente 0.0 en el CSV es parte del diseño, no un bug.

    Returns {tau: Series de Δloss relativo (con signo)}
    """
    X = X_val.reset_index(drop=True)[cols_feat].copy()
    y = np.asarray(y_val)
    if DIAG_PERM_MAX_SAMPLES is not None and len(X) > DIAG_PERM_MAX_SAMPLES:
        X = X.iloc[:DIAG_PERM_MAX_SAMPLES].copy()
        y = y[:DIAG_PERM_MAX_SAMPLES]
    n  = len(X)
    bs = max(2, min(DIAG_BLOCK_SIZE, n // 3))
    block_starts = np.arange(0, n, bs)
    rng = np.random.default_rng(42 + fold_num)

    pairs     = _sincos_pairs(cols_feat)
    paired_cs = {c for (sc, cc) in pairs.values() for c in (sc, cc)}

    out: dict = {}
    for tau, model in modelos.items():
        _loss      = _perdida_de_tau(tau)
        base_loss  = _loss(y, _diag_predict_un_modelo(model, X))
        feat_deltas: dict = {}

        # ── Features individuales ────────────────────────────────────────────
        for c in cols_feat:
            if c in paired_cs:
                continue
            orig   = X[c].values.copy()
            deltas = []
            for _ in range(DIAG_N_REPEATS):
                perm    = rng.permutation(block_starts)
                new_col = np.concatenate([orig[s:s + bs] for s in perm])[:n]
                Xp = X.copy(); Xp[c] = new_col
                deltas.append(_loss(y, _diag_predict_un_modelo(model, Xp)) - base_loss)
            feat_deltas[c] = float(np.mean(deltas))

        # ── Pares sin/cos: un solo shuffle, Δ conjunto en _sin, 0 en _cos ────
        for base, (sin_c, cos_c) in pairs.items():
            orig_sin = X[sin_c].values.copy()
            orig_cos = X[cos_c].values.copy()
            deltas   = []
            for _ in range(DIAG_N_REPEATS):
                perm = rng.permutation(block_starts)
                Xp   = X.copy()
                Xp[sin_c] = np.concatenate([orig_sin[s:s + bs] for s in perm])[:n]
                Xp[cos_c] = np.concatenate([orig_cos[s:s + bs] for s in perm])[:n]
                deltas.append(_loss(y, _diag_predict_un_modelo(model, Xp)) - base_loss)
            feat_deltas[sin_c] = float(np.mean(deltas))   # Δ joint completo
            feat_deltas[cos_c] = 0.0                      # placeholder; suma → joint

        den = abs(base_loss) if base_loss != 0 else 1.0
        out[tau] = pd.Series(feat_deltas).reindex(cols_feat).fillna(0.0) / den

    return out


def _diag_perm_familias_h(modelos: dict, X_val: pd.DataFrame, y_val,
                          familias: dict, cols_feat: list, fold_num: int) -> dict:
    """
    Block-permutation CONJUNTA por familia — ver FAMILIAS_PERM para el porqué.

    Mismo esquema de bloques que _diag_perm_h, pero con semilla distinta a
    propósito: si compartieran semilla, el shuffle de una familia de 1 feature
    coincidiría exactamente con el de _diag_perm_h para esa columna — no es un
    problema de correctitud, pero desperdicia la variación entre repeticiones.

    Returns {tau: {familia: Δloss_relativo}}. Familias sin ningún feature
    presente se omiten.
    """
    X = X_val.reset_index(drop=True)[cols_feat].copy()
    y = np.asarray(y_val)
    if DIAG_PERM_MAX_SAMPLES is not None and len(X) > DIAG_PERM_MAX_SAMPLES:
        X = X.iloc[:DIAG_PERM_MAX_SAMPLES].copy()
        y = y[:DIAG_PERM_MAX_SAMPLES]
    n  = len(X)
    bs = max(2, min(DIAG_BLOCK_SIZE, n // 3))
    block_starts = np.arange(0, n, bs)
    rng = np.random.default_rng(4300 + fold_num)   # distinta de la de _diag_perm_h

    out: dict = {}
    for tau, model in modelos.items():
        _loss     = _perdida_de_tau(tau)
        base_loss = _loss(y, _diag_predict_un_modelo(model, X))
        den       = abs(base_loss) if base_loss != 0 else 1.0

        fam_deltas: dict = {}
        for fam, cols in familias.items():
            cols_presentes = [c for c in cols if c in X.columns]
            if not cols_presentes:
                continue
            deltas = []
            for _ in range(DIAG_N_REPEATS):
                perm = rng.permutation(block_starts)
                Xp   = X.copy()
                for c in cols_presentes:
                    orig = X[c].values
                    Xp[c] = np.concatenate([orig[s:s + bs] for s in perm])[:n]
                deltas.append(_loss(y, _diag_predict_un_modelo(model, Xp)) - base_loss)
            fam_deltas[fam] = float(np.mean(deltas)) / den
        out[tau] = fam_deltas
    return out


def _diag_shap_h(modelos: dict, X_val: pd.DataFrame, cols_feat: list,
                 fold_num: int) -> dict:
    """
    SHAP |mean| en VAL (OOS), por cuantil, normalizado a participación.

    |shap| está en unidades del target; dividir por la suma lo vuelve
    adimensional y comparable entre horizontes, folds y cuantiles.

    Returns {tau: Series de participaciones (suman 1)}
    """
    if not _SHAP_OK:
        return {}
    X = X_val.reset_index(drop=True)[cols_feat]
    if DIAG_SHAP_MAX_SAMPLES and len(X) > DIAG_SHAP_MAX_SAMPLES:
        X = X.sample(DIAG_SHAP_MAX_SAMPLES, random_state=42 + fold_num)

    out: dict = {}
    for tau, model in modelos.items():
        try:
            explainer = shap.TreeExplainer(_shap_compat_booster(model))
        except Exception as e:
            logger.debug(f"      [diag] SHAP {_tau_label(tau)} falló [init]: {e}")
            continue
        try:
            # check_additivity=False evita la llamada interna a
            # predict(ntree_limit=N) que SHAP < 0.43 usa para validar que
            # sum(SHAP) == predicción del modelo.
            sv = explainer.shap_values(X, check_additivity=False)
            s  = pd.Series(np.abs(sv).mean(axis=0), index=cols_feat).fillna(0.0)
            total = s.sum()
            out[tau] = s / total if total > 0 else s
        except Exception as e:
            logger.debug(f"      [diag] SHAP {_tau_label(tau)} falló [values]: {e}")
    return out


def diagnosticar_h(modelos, X_val, y_val, cols_feat, fold_num, h_val_actual):
    """
    Una fila por modelo (cuantiles + 'mean'):
        {fold, h, tau, gain_<feat>, perm_<feat>, shap_<feat>}

    Las tres señales ya se calculaban por cuantil y se promediaban al final;
    conservar el desglose no cuesta cómputo adicional.
    """
    gain = _diag_gain_h(modelos, cols_feat)
    perm = _diag_perm_h(modelos, X_val, y_val, cols_feat, fold_num)
    shp  = _diag_shap_h(modelos, X_val, cols_feat, fold_num)

    _vacio = pd.Series(dtype=float)
    filas: list = []
    for tau in _orden_taus(modelos):
        g = gain.get(tau, _vacio)
        p = perm.get(tau, _vacio)
        s = shp.get(tau,  _vacio)
        row: dict = {"fold": fold_num, "h": int(h_val_actual), "tau": _tau_label(tau)}
        for f in cols_feat:
            row[f"gain_{f}"] = float(g.get(f, 0.0))
            row[f"perm_{f}"] = float(p.get(f, 0.0))
            row[f"shap_{f}"] = float(s.get(f, np.nan))
        filas.append(row)
    return filas


def guardar_diag_familias(diag_rows, diag_familia_rows, familias,
                          dir_modo, banco, fecha_hoy):
    """
    Compara la permutación INDIVIDUAL (Σ perm_<feat> de cada familia, ya
    calculada feature por feature en diag_rows) contra la CONJUNTA de la
    familia completa.

    Lectura de ratio_redundancia = |Δ_conjunto| / |Σ Δ_individual|:
      ~1     → los features de la familia son ~independientes; el perm
               individual bajo YA reflejaba bajo aporte real.
      >> 1   → redundantes entre sí: el modelo compensaba con las columnas
               correlacionadas cuando se barajaba solo una, así que la suma de
               los Δ individuales subestimaba el aporte real del grupo.
      Δ_conjunto ≈ 0 (con cualquier ratio) → la familia completa no aporta.

    DEBE llamarse ANTES de guardar_diag_y_plots(): esa función vacía diag_rows
    para liberar memoria antes de graficar, y acá se necesitan las columnas
    perm_<feat> todavía intactas.
    """
    if not diag_familia_rows or not diag_rows:
        return None

    df_ind = pd.DataFrame(diag_rows)
    df_fam = pd.DataFrame(diag_familia_rows)

    filas = []
    for fam, cols in familias.items():
        if fam not in df_fam.columns:
            continue
        cols_perm = [f"perm_{c}" for c in cols if f"perm_{c}" in df_ind.columns]
        if not cols_perm:
            continue
        ind = df_ind[["fold", "h", "tau"] + cols_perm].copy()
        ind["delta_individual_sum"] = ind[cols_perm].sum(axis=1)
        merged = ind[["fold", "h", "tau", "delta_individual_sum"]].merge(
            df_fam[["fold", "h", "tau", fam]].rename(columns={fam: "delta_conjunto"}),
            on=["fold", "h", "tau"], how="inner",
        )
        merged["familia"] = fam
        filas.append(merged)

    if not filas:
        logger.warning("  [diag] guardar_diag_familias: ninguna familia de "
                       "FAMILIAS_PERM tiene features en la matriz — nada que comparar.")
        return None

    df_cmp = pd.concat(filas, ignore_index=True)
    eps = 1e-9
    # ratio por FILA: se guarda en el CSV para inspeccionar casos puntuales,
    # pero NO se promedia tal cual — si el denominador de una fila pasa cerca
    # de cero, ese ratio explota y arrastra cualquier promedio simple.
    # "media de ratios" ≠ "ratio de medias" cuando el denominador ronda cero.
    df_cmp["ratio_redundancia"] = (
        df_cmp["delta_conjunto"].abs() / (df_cmp["delta_individual_sum"].abs() + eps)
    )

    ruta_csv = dir_modo / f"diag_perm_familias_{banco}_{fecha_hoy}.csv"
    df_cmp.to_csv(ruta_csv, index=False)
    logger.info(f"  [diag] CSV perm por familias: {ruta_csv.name} ({len(df_cmp)} filas)")

    resumen = (df_cmp.groupby(["familia", "tau"], observed=True)
               [["delta_individual_sum", "delta_conjunto"]].mean())
    resumen["ratio_redundancia"] = (
        resumen["delta_conjunto"].abs() / (resumen["delta_individual_sum"].abs() + eps)
    )
    resumen = resumen.round(4)
    print("\nPerm por familia — individual (Σ) vs conjunta, por τ:")
    print(resumen.to_string())
    print("\n  ratio_redundancia = |Δ_conjunto promedio| / |Σ Δ_individual promedio|")
    print("  (ratio de medias, no media de ratios)")
    print("  ~1   → independientes (el perm individual bajo ya reflejaba bajo aporte)")
    print("  >> 1 → redundantes entre sí (juntas importan más que la suma de las partes)")
    print("  Δ_conjunto ≈ 0 → la familia completa no aporta, con cualquier ratio")
    print(f"[OK] Perm por familias: {ruta_csv.name}")
    return resumen


def guardar_diag_y_plots(diag_rows, cols_feat, dir_modo, banco, fecha_hoy):
    """
    Guarda el CSV de diagnóstico (fold × h × τ) y los heatmaps de importancia.

    Las tres señales están normalizadas de forma que sus celdas son comparables
    entre horizontes, folds y cuantiles, así que NO se re-normaliza por fila ni
    por fold: el color es el valor directo y el heatmap se lee tanto por fila
    ("¿en qué h importa este feature?") como por columna ("¿qué features pesan
    en este h?"). La versión anterior dividía por el máximo de cada fold, lo que
    forzaba a 1 el máximo de todos los folds y hacía imposible comparar
    magnitudes entre ellos.

    Salidas:
      - diag_features_por_h_<banco>_<fecha>.csv       siempre, es lo valioso
      - diag_por_fold_tau/<señal>_fold<NN>_<tau>.png  individuales
      - diag_por_fold_tau/panel_<señal>_<tau>.png     folds lado a lado
      - diag_por_fold_tau/convergencia_<tau>.png      barras gain/perm/SHAP

    CONSUME diag_rows: lo vacía tras volcarlo al CSV — son miles de filas ×
    cientos de columnas de dicts que no hacen falta durante el graficado, que
    es justo la fase donde la memoria se agota.
    """
    if not diag_rows:
        return

    # El CSV va PRIMERO y fuera de cualquier guarda de matplotlib: es el
    # producto que importa y no debe perderse porque el graficado falle.
    df_d = pd.DataFrame(diag_rows)
    ruta_csv = dir_modo / f"diag_features_por_h_{banco}_{fecha_hoy}.csv"
    df_d.to_csv(ruta_csv, index=False)
    n_filas_diag = len(df_d)
    logger.info(f"  [diag] CSV fold×h×τ: {ruta_csv.name}  ({n_filas_diag:,} filas)")

    diag_rows.clear()
    gc.collect()

    if "tau" not in df_d.columns:
        logger.warning("  [diag] diag_rows sin columna 'tau' — CSV guardado, sin heatmaps")
        return

    dir_diag = dir_modo / "diag_por_fold_tau"
    dir_diag.mkdir(parents=True, exist_ok=True)

    folds = sorted(df_d["fold"].unique())
    taus  = [t for t in [f"q{int(x * 100):02d}" for x in QUANTILES] + ["mean"]
             if t in set(df_d["tau"])]

    SENALES = [
        # (prefijo, etiqueta, cmap, divergente)
        ("gain", "Gain (TRAIN) · participación del total",       "YlOrRd", False),
        ("perm", "Block-Perm (VAL, OOS) · Δloss relativo",       "RdBu_r", True),
        ("shap", "SHAP |mean| (VAL, OOS) · participación total", "YlOrRd", False),
    ]

    def _pivot(sub, senal, feat_cols, rename):
        """(features × h) para un subconjunto ya filtrado por fold y τ."""
        pv = (sub[["h"] + feat_cols].rename(columns=rename)
              .groupby("h").mean().T)
        return _consolidar_sincos_pivot(pv, _sincos_pairs(list(pv.index)))

    n_generados = 0
    # {señal: {τ: serie feature→valor}} — insumo del gráfico de barras de
    # convergencia. Colapsa fold Y h para volver a un número por feature, pero
    # solo DENTRO de un mismo τ, nunca mezclando cuantiles.
    agregados: dict = {}

    for senal, etiqueta, cmap, divergente in SENALES:
        feat_cols = [c for c in df_d.columns if c.startswith(f"{senal}_")]
        if not feat_cols:
            continue
        rename = {c: c[len(senal) + 1:] for c in feat_cols}

        agregados[senal] = {}
        for ta in taus:
            sub_ta = df_d.loc[df_d["tau"] == ta, feat_cols].rename(columns=rename)
            if sub_ta.empty:
                continue
            serie = sub_ta.mean(axis=0)
            serie = _consolidar_sincos_pivot(
                serie.to_frame("v"), _sincos_pairs(list(serie.index)))["v"]
            agregados[senal][ta] = serie

        # Escala de color FIJA por señal, sobre todas las τ y folds: deja ver a
        # simple vista si un cuantil concentra más peso que otro.
        pv_global = _pivot(df_d, senal, feat_cols, rename)
        vals = pv_global.values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if divergente:
            lim = float(np.percentile(np.abs(vals), 99)) or 1.0
            vmin, vmax = -lim, lim
        else:
            vmin = 0.0
            vmax = float(np.percentile(vals, 99)) or 1.0

        # Orden de filas POR CUANTIL, no compartido. Con un orden único (el
        # promedio de |valor| de TODAS las taus mezcladas) el panel de q01 y el
        # de q99 mostraban el mismo ranking aunque cada cuantil pese features
        # distintos — dias_al_cierre_mes dominando todos los paneles era en
        # parte artefacto de ese orden compartido. Lo que se mantiene fijo es
        # el orden DENTRO de un mismo τ al comparar folds.
        top_n = min(DIAG_TOP_N, len(pv_global.index))
        orden_por_tau: dict = {}
        for ta in taus:
            pv_ta = _pivot(df_d[df_d["tau"] == ta], senal, feat_cols, rename)
            orden_por_tau[ta] = (pv_ta.abs().mean(axis=1)
                                 .sort_values(ascending=False).index.tolist()[:top_n])

        pivots: dict = {}
        for fo in folds:
            for ta in taus:
                sub = df_d[(df_d["fold"] == fo) & (df_d["tau"] == ta)]
                if sub.empty:
                    continue
                pivots[(fo, ta)] = (_pivot(sub, senal, feat_cols, rename)
                                    .reindex(orden_por_tau[ta]))

        def _dibujar(ax, pv, titulo, con_yticks):
            hs = pv.columns.tolist()
            im = ax.imshow(pv.values, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_yticks(range(len(pv)))
            ax.set_yticklabels(pv.index.tolist() if con_yticks else [], fontsize=7)
            # Ticks en múltiplos de 10; si el fold no tiene ningún h divisible
            # por 10 (p.ej. corridas parciales o un H_GRUPOS reducido) el eje
            # quedaría sin etiquetas: se cae a ~8 ticks equiespaciados.
            xt = [i for i, h in enumerate(hs) if h % 10 == 0]
            if len(xt) < 2:
                xt = list(np.unique(np.linspace(0, len(hs) - 1,
                                                min(8, len(hs))).astype(int)))
            ax.set_xticks(xt)
            ax.set_xticklabels([str(hs[i]) for i in xt], fontsize=8)
            ax.set_xlabel("Horizonte h (días hábiles)", fontsize=9)
            ax.set_title(titulo, fontsize=10, fontweight="bold")
            return im

        # ── Figuras individuales: una por (fold, τ) ──────────────────────────
        # Sin bbox_inches="tight": renderiza la figura dos veces (una para
        # medir el bbox) y duplica el pico de memoria; son ~190 figuras.
        if DIAG_PLOTS_INDIVIDUALES:
            for (fo, ta), pv in pivots.items():
                fig, ax = plt.subplots(figsize=(max(10, len(pv.columns) * 0.18),
                                                max(6, top_n * 0.42)))
                im = _dibujar(ax, pv,
                              f"{etiqueta}\n{banco} · Fold {fo} · {ta} · top {top_n} features",
                              con_yticks=True)
                plt.colorbar(im, ax=ax, label=etiqueta.split(" · ")[-1])
                plt.tight_layout()
                fig.savefig(dir_diag / f"{senal}_fold{fo:02d}_{ta}.png", dpi=DIAG_PLOT_DPI)
                plt.close(fig)
                n_generados += 1
                if n_generados % 20 == 0:      # matplotlib fragmenta el heap
                    plt.close("all")
                    gc.collect()

        # ── Panel resumen: folds lado a lado, un archivo por τ ───────────────
        for ta in taus:
            pvs = [(fo, pivots[(fo, ta)]) for fo in folds if (fo, ta) in pivots]
            if not pvs:
                continue
            ancho = min(max(6.0, len(pvs) * 5.2), DIAG_PANEL_ANCHO_MAX)
            fig, axes = plt.subplots(1, len(pvs),
                                     figsize=(ancho, max(6, top_n * 0.42)),
                                     gridspec_kw={"wspace": 0.08})
            axes = np.atleast_1d(axes)
            for j, (fo, pv) in enumerate(pvs):
                im = _dibujar(axes[j], pv, f"Fold {fo}", con_yticks=(j == 0))
            fig.suptitle(
                f"{etiqueta} — {banco} · {ta}\n"
                f"Filas ordenadas por importancia propia de τ={ta} (igual entre folds, "
                f"distinto entre cuantiles) · escala de color común a toda la señal",
                fontsize=11, fontweight="bold", y=1.01)
            fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.01,
                         label=etiqueta.split(" · ")[-1])
            # bbox_inches="tight" SÍ acá (y no en las individuales): el suptitle
            # va con y=1.01, fuera del lienzo, y sin recorte sale cortado. Son
            # ~24 paneles, no ~190 figuras, así que el doble render se paga.
            fig.savefig(dir_diag / f"panel_{senal}_{ta}.png",
                        dpi=DIAG_PLOT_DPI, bbox_inches="tight")
            plt.close(fig)
            n_generados += 1

        del pivots, pv_global
        plt.close("all")
        gc.collect()

    # ── Barras gain/perm/SHAP, una figura por cuantil ────────────────────────
    _COLORES_CONV = {"gain": "#4C72B0", "perm": "#55A868", "shap": "#C44E52"}
    _ETIQ_CONV = {"gain": "gain (TRAIN, in-sample)", "perm": "perm (VAL, OOS)",
                  "shap": "SHAP |mean| (VAL, OOS)"}

    if all(s in agregados for s in ("gain", "perm", "shap")):
        for ta in taus:
            series_ta = {s: agregados[s][ta] for s in ("gain", "perm", "shap")
                         if ta in agregados[s]}
            if len(series_ta) < 3:
                continue
            # Orden por |SHAP| descendente — mismo árbitro (OOS + atribución
            # por instancia) que ordena los heatmaps, para que ambos tipos de
            # figura sean consistentes entre sí dentro de un mismo τ.
            orden_feat = (series_ta["shap"].abs()
                          .sort_values(ascending=False).index.tolist()[:DIAG_TOP_N])

            fig, ax = plt.subplots(figsize=(11, max(6, len(orden_feat) * 0.32)))
            y    = np.arange(len(orden_feat))
            alto = 0.26
            for i, senal in enumerate(("gain", "perm", "shap")):
                serie  = series_ta[senal].reindex(orden_feat)
                escala = float(serie.abs().max()) or 1.0
                ax.barh(y + (i - 1) * alto, (serie / escala).to_numpy(), height=alto,
                        color=_COLORES_CONV[senal], label=_ETIQ_CONV[senal])
            ax.set_yticks(y)
            ax.set_yticklabels(orden_feat, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=0.6)
            # symlog: en q01/q99 un solo feature satura su señal, así que en
            # escala lineal el resto del top-25 queda como líneas invisibles
            # pegadas a 0. symlog comprime esa cola y mantiene el signo de perm
            # (puede ser negativo) sin romperse en 0, a diferencia de log.
            ax.set_xscale("symlog", linthresh=0.02)
            ax.set_xlabel("Importancia normalizada (cada señal / su propio máximo "
                          "en este τ) — escala symlog")
            ax.set_title(
                f"gain / perm / SHAP — {banco} · τ={ta}  "
                f"(agregado sobre folds y horizontes h)\n"
                f"Convergencia gain≈perm≈SHAP → feature genuinamente útil en este cuantil",
                fontsize=11, fontweight="bold")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, axis="x", which="both", alpha=0.25)
            plt.tight_layout()
            fig.savefig(dir_diag / f"convergencia_{ta}.png", dpi=DIAG_PLOT_DPI)
            plt.close(fig)
            n_generados += 1

        del agregados
        plt.close("all")
        gc.collect()

    logger.info(f"  [diag] Heatmaps: {n_generados} figuras en {dir_diag.name}")
    print(f"[OK] Diagnóstico features: {n_filas_diag:,} filas (fold × h × τ) | "
          f"{n_generados} figuras en {dir_diag}")


def guardar_hp_report(hp_rows, dir_modo, banco, fecha_hoy):
    """
    CSV + gráficos del reporte de hiperparámetros Optuna.

    Diferencia con v4_3: acá Optuna corre un estudio INDEPENDIENTE por cuantil
    (ver _entrenar_fold_xgb_qt), no uno por grupo. Eso permite una pregunta que
    v4_3 no puede responder — ¿las colas necesitan HP distintos al centro? — a
    cambio de una figura por τ en vez de una sola.

    1. hp_convergencia_<τ>.png — curva 'mejor encontrado hasta el trial t' por
       fold, un panel por grupo. Diagnostica si TRIALS_FLAT alcanza.
    2. hp_estabilidad_<τ>.png  — un subpanel por HP, una línea por grupo, a lo
       largo de los folds. Detecta inestabilidad entre folds.
    """
    if not hp_rows:
        return
    df_hp = pd.DataFrame([{k: v for k, v in r.items() if k != "trial_values"}
                          for r in hp_rows])
    ruta_csv = dir_modo / f"hp_report_{banco}_{fecha_hoy}.csv"
    df_hp.to_csv(ruta_csv, index=False)
    logger.info(f"  [HP] CSV: {ruta_csv.name}")

    grupo_order  = [g for g in H_GRUPOS if g in set(df_hp["grupo"])]
    grupo_colors = {"muy_corto": "#6D28D9", "corto": "#0369A1",
                    "medio": "#047857", "largo": "#B45309"}
    folds_sorted = sorted(df_hp["fold"].unique())
    taus_hp      = [t for t in [f"q{int(x * 100):02d}" for x in QUANTILES]
                    if t in set(df_hp["tau"])]
    n_generados  = 0

    # ── Gráfico 1: convergencia Optuna, una figura por τ ─────────────────────
    for ta in taus_hp:
        rows_ta = [r for r in hp_rows if r.get("tau") == ta and r.get("trial_values")]
        if not rows_ta:
            continue
        n_g = max(len(grupo_order), 1)
        fig, axes = plt.subplots(1, n_g, figsize=(5 * n_g, 4.5),
                                 gridspec_kw={"wspace": 0.35})
        axes = np.atleast_1d(axes)
        n_trials_ref = max(len(r["trial_values"]) for r in rows_ta)
        fig.suptitle(
            f"Convergencia Optuna por grupo — {banco} · τ={ta}\n"
            f"Curva 'mejor hasta trial t' · ¿converge antes del último trial?",
            fontsize=11, fontweight="bold", y=1.01)

        cmap_folds = plt.cm.Set1
        for ax, grupo in zip(axes, grupo_order):
            rows_g = [r for r in rows_ta if r["grupo"] == grupo]
            for i, row in enumerate(rows_g):
                tv = row["trial_values"]
                if not tv:
                    continue
                best_so_far = list(itertools.accumulate(tv, min))
                color = cmap_folds(i / max(len(folds_sorted), 1))
                ax.plot(range(1, len(best_so_far) + 1), best_so_far,
                        lw=1.6, color=color, alpha=0.85, label=f"Fold {row['fold']}")
                last_improvement = max(
                    (j for j, (a, b) in enumerate(zip(best_so_far, best_so_far[1:]))
                     if b < a - 1e-8), default=len(best_so_far) - 1)
                ax.axvline(last_improvement + 1, color=color, lw=0.7, ls=":", alpha=0.5)
            h_rep = rows_g[0]["h_rep"] if rows_g else "?"
            ax.set_title(f"{grupo}\n(h_rep={h_rep})", fontsize=10, fontweight="bold")
            ax.set_xlabel("Trial #", fontsize=9)
            ax.set_ylabel("Pinball val (mejor acumulado)", fontsize=8)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
            ax.grid(True, alpha=0.25)
            ax.axvline(n_trials_ref, color="#DC2626", lw=1.2, ls="--", alpha=0.6,
                       label=f"N_TRIALS={n_trials_ref}")
            ax.legend(fontsize=7, loc="upper right")
        # rect reserva el 14% superior: sin él tight_layout pega los ejes al
        # borde y el suptitle se superpone a los títulos de cada grupo.
        plt.tight_layout(rect=[0, 0, 1, 0.86])
        fig.savefig(dir_modo / f"hp_convergencia_{banco}_{ta}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_generados += 1

    # ── Gráfico 2: estabilidad de HP por fold, una figura por τ ──────────────
    hp_names = ["max_depth", "min_child_weight", "learning_rate", "reg_alpha",
                "reg_lambda", "subsample", "colsample_bytree", "s", "n_estimators"]
    hp_labels = {"max_depth": "max_depth", "min_child_weight": "min_child_weight",
                 "learning_rate": "learning_rate", "reg_alpha": "reg_alpha (L1)",
                 "reg_lambda": "reg_lambda (L2)", "subsample": "subsample",
                 "colsample_bytree": "colsample_bytree",
                 "s": "s (smoothing arctan)", "n_estimators": "n_estimators"}
    hp_names = [h for h in hp_names if h in df_hp.columns]

    for ta in taus_hp:
        sub_ta = df_hp[df_hp["tau"] == ta]
        if sub_ta.empty or not hp_names:
            continue
        ncols = 4
        nrows = (len(hp_names) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2),
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.38},
                                 squeeze=False)
        # list(...) y no .flat: flat es un iterador y el zip de abajo lo
        # consumiría, dejando vacío el slice que oculta los paneles sobrantes.
        axes_flat = list(axes.flat)
        fig.suptitle(
            f"Estabilidad de Hiperparámetros por fold — {banco} · τ={ta}\n"
            f"Línea por grupo · variación entre folds indica inestabilidad",
            fontsize=11, fontweight="bold", y=1.01)

        for ax, hp in zip(axes_flat, hp_names):
            for grupo in grupo_order:
                sg = sub_ta[(sub_ta["grupo"] == grupo) & sub_ta[hp].notna()]
                if sg.empty:
                    continue
                color = grupo_colors.get(grupo, "#444444")
                ax.plot(sg["fold"], sg[hp], "o-", lw=1.6, ms=5,
                        color=color, alpha=0.85, label=grupo)
                ax.fill_between(sg["fold"], sg[hp].min(), sg[hp].max(),
                                alpha=0.06, color=color)
            ax.set_title(hp_labels.get(hp, hp), fontsize=9, fontweight="bold")
            ax.set_xlabel("Fold", fontsize=8)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=7, loc="best")
        for ax in axes_flat[len(hp_names):]:
            ax.set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(dir_modo / f"hp_estabilidad_{banco}_{ta}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_generados += 1

    # ── Tabla resumen + diagnóstico de N_TRIALS ──────────────────────────────
    print("\nHiperparámetros óptimos por fold, grupo y cuantil:")
    col_print = ["fold", "grupo", "h_rep", "tau", "max_depth", "min_child_weight",
                 "learning_rate", "reg_alpha", "reg_lambda", "subsample",
                 "colsample_bytree", "s", "n_estimators", "best_pinball_val"]
    df_show = df_hp[[c for c in col_print if c in df_hp.columns]].copy()
    for c, fmt in [("learning_rate", "{:.4f}"), ("reg_alpha", "{:.3f}"),
                   ("reg_lambda", "{:.3f}"), ("subsample", "{:.3f}"),
                   ("colsample_bytree", "{:.3f}"), ("s", "{:.4g}"),
                   ("best_pinball_val", "{:.4f}")]:
        if c in df_show.columns:
            df_show[c] = df_show[c].map(lambda x, _f=fmt: _f.format(x)
                                        if pd.notna(x) else "—")
    print(df_show.to_string(index=False))

    last_improvement_stats, n_trials_max = [], 0
    for r in hp_rows:
        tv = r.get("trial_values")
        if not tv:
            continue
        n_trials_max = max(n_trials_max, len(tv))
        best_so_far  = list(itertools.accumulate(tv, min))
        last_improvement_stats.append(max(
            (j for j, (a, b) in enumerate(zip(best_so_far, best_so_far[1:]))
             if b < a - 1e-8), default=0) + 1)
    if last_improvement_stats:
        p50 = int(np.percentile(last_improvement_stats, 50))
        p90 = int(np.percentile(last_improvement_stats, 90))
        print(f"\nDiagnóstico de trials (máx observado={n_trials_max}): "
              f"última mejora mediana en trial #{p50}, p90 en trial #{p90}.")
        if p90 >= n_trials_max - 2:
            print("  ⚠  La convergencia llega hasta el final — considera subir TRIALS_FLAT.")
        else:
            print(f"  ✓  Converge antes del límite ({p90} < {n_trials_max}).")

    logger.info(f"  [HP] {n_generados} figuras en {dir_modo.name}")
    print(f"[OK] HP report guardado en: {dir_modo}")


def _reg(tablas: dict, nombre: str, obj):
    """
    Registra una tabla para el Excel de resumen y la devuelve, de modo que el
    sitio de impresión pueda seguir siendo print(_reg(...)).

    Se guarda la versión NUMÉRICA, no la formateada: en consola conviene
    "91.5%" pero en Excel conviene 0.915, que se puede ordenar y graficar.
    """
    if obj is None:
        return obj
    tablas[nombre] = (obj.to_frame(name=nombre)
                      if isinstance(obj, pd.Series) else obj.copy())
    return obj


def guardar_tablas_excel(tablas: dict, dir_modo, banco, fecha_hoy):
    """
    Un Excel con una hoja por tabla de consola.

    Complementa a consola_*.txt: el .txt conserva el formato exacto de pantalla,
    este archivo conserva los valores numéricos para analizarlos.

    Si openpyxl no está o el archivo está abierto en Excel, cae a CSVs sueltos
    en vez de romper la corrida.
    """
    if not tablas:
        return
    ruta = dir_modo / f"tablas_resumen_{banco}_{fecha_hoy}.xlsx"
    try:
        with pd.ExcelWriter(ruta, engine="openpyxl") as xl:
            for nombre, df in tablas.items():
                df.to_excel(xl, sheet_name=nombre[:31])   # Excel: 31 chars máx
    except Exception as e:
        logger.warning(f"  No se pudo escribir {ruta.name} ({e}) — CSVs sueltos")
        for nombre, df in tablas.items():
            df.to_csv(dir_modo / f"tabla_{nombre}_{banco}_{fecha_hoy}.csv")
        print(f"[OK] Tablas de consola: {len(tablas)} CSVs en {dir_modo}")
        return
    logger.info(f"  Tablas de consola: {ruta.name} ({len(tablas)} hojas)")
    print(f"[OK] Tablas de consola: {ruta.name} ({len(tablas)} hojas)")


class _Tee:
    """
    Duplica sys.stdout a un archivo para conservar las tablas de consola.

    Intercepta stdout (los print de las funciones de reporte nuevas) Y engancha
    un FileHandler al logger raíz. Esto último NO estaba en v4_3, donde el
    logging se dejaba fuera a propósito; acá es necesario porque este script
    emite las tablas de resumen (métricas por fold, promedios TEST, sesgo
    VAL−TEST) con logger.info, no con print — sin el handler el consola_*.txt
    saldría con las tablas del diagnóstico pero sin las del resumen.

    __getattr__ delega el resto de atributos al stdout original (isatty,
    encoding, fileno…) para que cualquier librería que los consulte siga
    funcionando.
    """

    def __init__(self, ruta, capturar_logging: bool = True):
        self._orig = sys.stdout           # primero: __getattr__ depende de él
        self._f    = open(ruta, "w", encoding="utf-8")
        sys.stdout = self
        self._handler = None
        if capturar_logging:
            self._handler = logging.StreamHandler(self._f)
            self._handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
            logging.getLogger().addHandler(self._handler)

    def write(self, s):
        self._orig.write(s)
        try:
            self._f.write(s)
        except OSError:
            # H: es una unidad de red; una corrida larga (>2h en la practica)
            # puede sobrevivir a un corte momentaneo de la conexion, pero eso
            # invalida el handle del archivo. Sin este guard, el siguiente
            # write() propaga el OSError hasta matar el proceso justo al
            # final — despues de horas de computo ya persistido en los
            # parquets de cada fold, perdiendo solo el resumen de consola.
            pass

    def flush(self):
        self._orig.flush()
        try:
            self._f.flush()
        except OSError:
            pass

    def __getattr__(self, nombre):
        orig = self.__dict__.get("_orig")
        if orig is None:
            raise AttributeError(nombre)
        return getattr(orig, nombre)

    def cerrar(self):
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            try:
                self._handler.close()
            except OSError:
                pass
            self._handler = None
        sys.stdout = self._orig
        try:
            self._f.close()
        except OSError:
            pass



###############################################################################
###############################################################################
# Overlay sobreencaje -- lee ajuste de step007 y aplica a las predicciones
###############################################################################

def _aplicar_overlay_sobreencaje(
    preds: dict,
    h_arr: np.ndarray,
    fechas_t: pd.DatetimeIndex,
    df_hist: "pd.DataFrame | None" = None,
) -> dict:
    """
    Aplica factor multiplicativo de sobreencaje sobre los cuantiles proyectados.

    Para cada fecha_t, itera sobre TODOS los cierres trimestrales dentro del horizonte
    de proyección (hasta 75 DH). Cada cierre tiene su propio factor independiente
    aplicado a su propia ventana de 7 DH. Los factores no se solapan.

    Por cierre fecha_cierre_k:

    FUERA de la ventana (fecha_t < inicio de los últimos 7 DH del trimestre k):
      - mask_ventana = h_inicio_k..h_cierre_k  (7 DH completos de la ventana k)
      - peor_usado   = peor_total               (ningún retiro conocido aún)
      - f = peor_total / |Q[TAU]_acum_7días_k|

    DENTRO de la ventana (fecha_t >= inicio de los últimos 7 DH del trimestre k):
      - mask_ventana = h=1..h_cierre_k          (días futuros restantes)
      - flujo_neto   = suma neta de flujos realizados en la ventana hasta fecha_t
                       (positivo=depósito aumenta peor_restante; negativo=retiro lo reduce)
      - peor_usado   = max(0, peor_total + flujo_neto)
      - f = peor_usado / |Q[TAU]_acum_restante_k|
      → numerador y denominador se reducen al mismo ritmo → f estable

    peor_total se lee del Excel como el valor más reciente disponible a fecha_t.
    Si Q4 aún no tiene entrada propia, hereda el último disponible (Q3).
    """
    if not OVERLAY_SOBREENCAJE:
        return preds

    try:
        df_aj = pd.read_excel(
            RUTA_AJUSTE_OVERLAY, sheet_name="Ajuste_diario",
            index_col=0, parse_dates=True,
        )
        df_aj.index = pd.DatetimeIndex(df_aj.index)
    except Exception as e:
        logger.warning(f"[OVERLAY] No se pudo leer Ajuste_diario: {e} -- sin ajuste")
        return preds

    if "peor_total" not in df_aj.columns:
        logger.warning("[OVERLAY] Columna peor_total no encontrada en Ajuste_diario -- sin ajuste")
        return preds

    preds_adj = {tau: arr.copy() for tau, arr in preds.items()}
    meta_rows = []   # metadatos del factor por (fecha_t, cierre_fecha)

    # Conjunto de feriados PER+USA para lookup O(1)
    _feriados_set = set(pd.DatetimeIndex(_FERIADOS_PEUSA).normalize())

    for fecha_t in sorted(set(fechas_t)):
        # Saltar feriados que aparezcan en los datos por forward-fill del origen
        if pd.Timestamp(fecha_t).normalize() in _feriados_set:
            continue

        # Guard rápido: si no hay ninguna fila del Excel antes de fecha_t, saltar.
        if not (df_aj.index <= fecha_t).any():
            continue

        # Log informativo si fecha_t es el propio cierre trimestral.
        # La ventana de ESE trimestre ya venció (bh empieza en fecha_t+1BDay),
        # pero los trimestres futuros (Q4, Q1…) sí se procesan en el inner loop.
        if fecha_t.month in [3, 6, 9, 12]:
            _prox_bday = fecha_t + BDAY_PE
            if _prox_bday.month != fecha_t.month:
                logger.info(
                    f"[OVERLAY] {fecha_t.date()} — DIA CIERRE TRIMESTRAL "
                    f"(prox BDay={_prox_bday.date()}): ventana Q{fecha_t.quarter} cerrada, "
                    f"procesando trimestres futuros"
                )

        # Horizonte de días hábiles proyectados (h=1..75 desde fecha_t+1BDay)
        bh = pd.bdate_range(
            start=fecha_t + BDAY_PE,
            periods=OVERLAY_VENTANA_DH + 75,
            freq=BDAY_PE,
        )
        bh_list = list(bh)
        df_bh = pd.DataFrame({"fecha": bh, "mes": bh.month, "anio": bh.year, "trim": bh.quarter})

        # Todos los cierres trimestrales dentro del rango de proyección
        cierres_proy = (
            df_bh[df_bh["mes"].isin([3, 6, 9, 12])]
            .groupby(["anio", "trim"])["fecha"].max()
            .sort_values().tolist()
        )
        if not cierres_proy:
            continue

        mask_origen = (fechas_t == fecha_t)

        # ── Procesar CADA cierre trimestral de forma independiente ────────────
        # Tracking para extensión cuando C2 queda fuera del horizonte h=75:
        # si la ventana de C2 está en h>75, se extiende el factor de C1 al
        # resto del horizonte [h_cierre_C1+1 .. 75] como mejor estimado.
        _last_applied_f  = None   # factor activo del último cierre procesado
        _last_h_cierre   = 0      # h_cierre del último cierre procesado
        _last_peor_total = 0.0

        for fecha_cierre in cierres_proy:

            # h correspondiente a este cierre en el vector de proyección
            try:
                h_cierre = bh_list.index(fecha_cierre) + 1
            except ValueError:
                _diffs = [abs((d - fecha_cierre).days) for d in bh_list]
                h_cierre = _diffs.index(min(_diffs)) + 1

            h_inicio = max(1, h_cierre - OVERLAY_VENTANA_DH + 1)

            # Si el cierre cae fuera del horizonte de predicción (h_cierre > 75):
            # aplicar el factor a h=75 solo infla el ratio por usar un único paso.
            # Extender el factor de C1 a h=[_last_h_cierre+1..75] es mejor estimado.
            if h_cierre > 75:
                if _last_applied_f is not None and _last_h_cierre < 75:
                    h_ext_ini = _last_h_cierre + 1
                    mask_ext  = mask_origen & (h_arr >= h_ext_ini) & (h_arr <= 75)
                    if mask_ext.any():
                        for tau in preds_adj:
                            preds_adj[tau][mask_ext] *= _last_applied_f
                        n_ext = int(mask_ext.sum())
                        meta_rows.append({
                            "fecha_t": fecha_t, "cierre_fecha": fecha_cierre,
                            "h_cierre": h_cierre, "n_inciertos": n_ext,
                            "peor_total": _last_peor_total,
                            "retiro_conocido": 0.0, "retiro_pasado": 0.0,
                            "retiro_t2": 0.0, "peor_restante": _last_peor_total,
                            "q_tau_acum": np.nan, "factor_f": _last_applied_f,
                            "overlay_activo": True, "razon_no_activo": "ext_f_c1",
                        })
                        logger.info(
                            f"[OVERLAY] {fecha_t.date()} [EXT] | cierre: {fecha_cierre.date()} "
                            f"(ventana h=[{h_inicio},{h_cierre}] fuera de h=75) | "
                            f"extendiendo f_C1={_last_applied_f:.3f} a h=[{h_ext_ini},75]"
                        )
                continue

            # peor_total específico para el tipo de trimestre de este cierre.
            # Filtra el Excel por el mismo mes de cierre (3=Mar, 6=Jun, 9=Sep, 12=Dic)
            # y toma el valor más reciente ≤ fecha_t.
            # Si no hay histórico del mismo trimestre, hereda el último disponible.
            _mes_cierre = pd.Timestamp(fecha_cierre).month
            _disp_q = df_aj.index[
                (df_aj.index.month == _mes_cierre) & (df_aj.index <= fecha_t)
            ]
            if len(_disp_q) > 0:
                peor_total = float(df_aj.loc[_disp_q[-1], "peor_total"])
            else:
                _disp_g = df_aj.index[df_aj.index <= fecha_t]
                peor_total = float(df_aj.loc[_disp_g[-1], "peor_total"])
            if peor_total <= 0:
                _last_h_cierre = h_cierre
                continue

            fecha_inicio_ventana_real = fecha_cierre - (OVERLAY_VENTANA_DH - 1) * BDAY_PE
            dentro_ventana = (pd.Timestamp(fecha_t) >= fecha_inicio_ventana_real)

            # mask_ventana y n_inciertos para ESTE cierre.
            # OUT con cierre previo procesado: cubrir desde h_cierre_C1+1 hasta h_cierre
            # (los flujos intermedios entre cierres también contribuyen al Q2).
            if dentro_ventana:
                h_inicio_mask = 1
            elif _last_h_cierre > 0:
                h_inicio_mask = _last_h_cierre + 1
            else:
                h_inicio_mask = h_inicio  # primer cierre sin antecesor: ventana de 7 DH
            mask_ventana  = mask_origen & (h_arr >= h_inicio_mask) & (h_arr <= h_cierre)
            n_inciertos   = max(1, h_cierre - h_inicio_mask + 1)

            if not mask_ventana.any():
                h_disponibles = h_arr[mask_origen & (h_arr <= h_cierre)]
                if len(h_disponibles) == 0:
                    # h_cierre < h_min (ej. h_cierre=1 cuando h_min=2): no hay
                    # predicciones para este cierre; registrar para diagnóstico.
                    meta_rows.append({
                        "fecha_t": fecha_t, "cierre_fecha": fecha_cierre,
                        "h_cierre": h_cierre, "n_inciertos": 0,
                        "peor_total": peor_total, "retiro_conocido": 0.0,
                        "retiro_pasado": 0.0, "retiro_t2": 0.0,
                        "peor_restante": peor_total,
                        "q_tau_acum": np.nan, "factor_f": np.nan,
                        "overlay_activo": False, "razon_no_activo": "h_cierre_lt_hmin",
                    })
                    logger.debug(
                        f"[OVERLAY] {fecha_t.date()} | cierre: {fecha_cierre.date()} "
                        f"h_cierre={h_cierre} < h_min — sin predicciones, se omite"
                    )
                    _last_h_cierre = h_cierre
                    continue
                h_fb = h_disponibles[np.abs(h_disponibles - h_cierre).argmin()]
                mask_ventana = mask_origen & (h_arr == h_fb)
                n_inciertos  = 1

            # ── Netting: flujo neto realizado en la ventana de ESTE cierre ────
            # _H_MIN viene de OVERLAY_CONOCIMIENTO_ANTICIPADO (config global).
            # df_hist contiene targets de todos los folds; el lookback de _H_MIN
            # días hábiles garantiza que nunca se accede a fechas > fecha_t.
            retiro_pasado = 0.0
            if dentro_ventana and df_hist is not None:
                _H_MIN = OVERLAY_CONOCIMIENTO_ANTICIPADO
                for _d in pd.bdate_range(start=fecha_inicio_ventana_real,
                                          end=min(fecha_t, fecha_cierre),
                                          freq=BDAY_PE):
                    _d_lookback = _d - _H_MIN * BDAY_PE
                    _rows = df_hist[
                        (df_hist["fecha_t"] == _d_lookback) & (df_hist["h"] == _H_MIN)
                    ]
                    if not _rows.empty:
                        retiro_pasado += float(_rows["target"].values[0])

            peor_usado = (
                max(0.0, peor_total + retiro_pasado) if dentro_ventana else peor_total
            )
            if peor_usado < 1e-6:
                meta_rows.append({
                    "fecha_t": fecha_t, "cierre_fecha": fecha_cierre,
                    "h_cierre": h_cierre, "n_inciertos": n_inciertos,
                    "peor_total": peor_total, "retiro_conocido": retiro_pasado,
                    "retiro_pasado": retiro_pasado, "retiro_t2": 0.0,
                    "peor_restante": 0.0,
                    "q_tau_acum": 0.0, "factor_f": 1.0, "overlay_activo": False,
                    "razon_no_activo": "peor_consumido",
                })
                _last_h_cierre = h_cierre
                continue

            _vals_ventana = preds[OVERLAY_TAU_REFERENCIA][mask_ventana]
            _n_nan_ventana = int(np.isnan(_vals_ventana).sum())
            if _n_nan_ventana > 0:
                _frac_nan = _n_nan_ventana / len(_vals_ventana)
                if _frac_nan > 0.5:
                    # Más de la mitad de la ventana sin predicción (grupo de
                    # horizonte saltado) -- un factor calculado sobre tan poca
                    # data no es confiable. Se omite el overlay para ESTA
                    # ventana en vez de multiplicar por un factor basado en
                    # 1-2 días sueltos (o, sin este guard, por NaN si toda la
                    # ventana faltara -- np.sum() sin nan-safe propagaría eso
                    # a filas que sí tenían predicción válida antes de overlay).
                    logger.warning(
                        f"[OVERLAY] {fecha_t.date()} cierre {fecha_cierre.date()}: "
                        f"{_n_nan_ventana}/{len(_vals_ventana)} horizontes sin predicción "
                        f"en la ventana (grupo de h saltado) — overlay omitido aquí, "
                        f"sin tocar los cuantiles ya calculados (validos o NaN) de esas filas.")
                    _last_h_cierre = h_cierre
                    continue
                logger.debug(
                    f"[OVERLAY] {fecha_t.date()} cierre {fecha_cierre.date()}: "
                    f"{_n_nan_ventana}/{len(_vals_ventana)} horizontes sin predicción en "
                    f"la ventana — factor calculado ignorando esos (np.nansum).")
            q01_acum = float(np.nansum(_vals_ventana))
            if q01_acum >= 0 or abs(q01_acum) < 1e-6:
                meta_rows.append({
                    "fecha_t": fecha_t, "cierre_fecha": fecha_cierre,
                    "h_cierre": h_cierre, "n_inciertos": n_inciertos,
                    "peor_total": peor_total, "retiro_conocido": retiro_pasado,
                    "retiro_pasado": retiro_pasado, "retiro_t2": 0.0,
                    "peor_restante": peor_usado,
                    "q_tau_acum": q01_acum, "factor_f": 1.0, "overlay_activo": False,
                    "razon_no_activo": "q_tau>=0_o_nulo",
                })
                if dentro_ventana:
                    logger.info(
                        f"[OVERLAY] {fecha_t.date()} [IN] | cierre: {fecha_cierre.date()} "
                        f"h=[{h_inicio_mask},{h_cierre}] n_inc={n_inciertos} | "
                        f"Q{int(OVERLAY_TAU_REFERENCIA*100):02d}_acum={q01_acum:+.0f} >= 0 "
                        f"— modelo predice flujo positivo, f=1.0 (sin ajuste multiplicativo)"
                    )
                _last_h_cierre = h_cierre
                continue

            # f = peor_usado / |Q05_acum| — misma lógica IN/OUT para cada cierre
            f = peor_usado / abs(q01_acum)
            if f <= 1.0:
                meta_rows.append({
                    "fecha_t": fecha_t, "cierre_fecha": fecha_cierre,
                    "h_cierre": h_cierre, "n_inciertos": n_inciertos,
                    "peor_total": peor_total, "retiro_conocido": retiro_pasado,
                    "retiro_pasado": retiro_pasado, "retiro_t2": 0.0,
                    "peor_restante": peor_usado,
                    "q_tau_acum": q01_acum, "factor_f": f, "overlay_activo": False,
                    "razon_no_activo": "f<=1",
                })
                _last_h_cierre = h_cierre
                continue

            for tau in preds_adj:
                preds_adj[tau][mask_ventana] *= f

            meta_rows.append({
                "fecha_t": fecha_t, "cierre_fecha": fecha_cierre,
                "h_cierre": h_cierre, "n_inciertos": n_inciertos,
                "peor_total": peor_total, "retiro_conocido": retiro_pasado,
                "retiro_pasado": retiro_pasado, "retiro_t2": 0.0,
                "peor_restante": peor_usado,
                "q_tau_acum": q01_acum, "factor_f": f, "overlay_activo": True,
                "razon_no_activo": "",
            })

            _zona = "IN" if dentro_ventana else "OUT"
            logger.info(
                f"[OVERLAY] {fecha_t.date()} [{_zona}] | cierre: {fecha_cierre.date()} "
                f"h=[{h_inicio_mask},{h_cierre}] n_inc={n_inciertos} | "
                f"Q{int(OVERLAY_TAU_REFERENCIA*100):02d}_acum={q01_acum:+.0f} | "
                f"peor={peor_total:,.0f} | flujo_neto={retiro_pasado:+,.0f} | "
                f"peor_restante={peor_usado:,.0f} | f={f:.3f}"
            )

            # Actualizar tracking para extensión a C2 si queda fuera de h=75
            _last_applied_f  = f
            _last_h_cierre   = h_cierre
            _last_peor_total = peor_total

    return preds_adj, meta_rows



###############################################################################
# PARTE 8 — Pipeline principal
###############################################################################

def _cargar_metadata_disco(banco: str) -> dict:
    """
    Encuentra y carga el metadata JSON más reciente para el banco y MODELO_CV
    en DIR_MODELOS. Lanza FileNotFoundError si no existe.
    """
    sfx = "lgbm_wfcv_v3" if MODELO_CV == "lgbm" else f"{MODELO_CV}_wfcv_v3"
    candidatos = sorted(DIR_MODELOS.glob(f"metadata_{sfx}_{banco}_*.json"), reverse=True)
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró metadata para banco={banco} modelo={MODELO_CV} en {DIR_MODELOS}"
        )
    meta = json.loads(candidatos[0].read_text(encoding="utf-8"))
    logger.info(f"  [REPLOT] Metadata cargado: {candidatos[0].name}")
    return meta


def _cargar_modelos_fold_disco(fold_info: dict, banco: str) -> dict:
    """
    Carga desde disco los modelos de un fold específico — uno por cada h
    entrenado (estrategia directa por grupos), no un solo dict plano.
    Usa fecha_hoy guardada en fold_info (del manifest original).

    Devuelve {h: {tau: Booster, "mean": Booster}} — reconstruye qué h's
    existen a partir de los archivos realmente presentes en disco (no
    asume que TODOS los h de H_GRUPOS se hayan entrenado ese fold, por si
    algún grupo se omitió por datos insuficientes).
    """
    sfx      = "lgbm_wfcv_v3" if MODELO_CV == "lgbm" else f"{MODELO_CV}_wfcv_v3"
    ext      = ".txt" if MODELO_CV == "lgbm" else ".json"
    fold_num = fold_info["fold"]
    fecha    = fold_info["fecha_hoy"]

    modelos_por_h: dict = {}
    for h_ind in range(H_MIN, H_MAX_DIAS_HAB + 1):
        ruta_q50 = DIR_MODELOS / f"{sfx}_{banco}_fold{fold_num:02d}_h{h_ind:02d}_q50_{fecha}{ext}"
        if not ruta_q50.exists():
            continue   # este h no se entrenó en este fold (grupo omitido, etc.)

        modelos_h = {}
        for tau in QUANTILES:
            ruta = DIR_MODELOS / f"{sfx}_{banco}_fold{fold_num:02d}_h{h_ind:02d}_q{int(tau*100):02d}_{fecha}{ext}"
            if not ruta.exists():
                raise FileNotFoundError(
                    f"Modelo fold {fold_num} h={h_ind} incompleto — falta {ruta}")
            if MODELO_CV == "lgbm":
                import lightgbm as lgb_load
                modelos_h[tau] = lgb_load.Booster(model_file=str(ruta))
            else:
                b = xgb.Booster(); b.load_model(str(ruta)); modelos_h[tau] = b

        ruta_mean = DIR_MODELOS / f"{sfx}_{banco}_fold{fold_num:02d}_h{h_ind:02d}_mean_{fecha}{ext}"
        if ruta_mean.exists():
            b = xgb.Booster(); b.load_model(str(ruta_mean)); modelos_h["mean"] = b

        modelos_por_h[h_ind] = modelos_h

    if not modelos_por_h:
        raise FileNotFoundError(
            f"Ningún modelo encontrado para fold {fold_num} (banco={banco}, "
            f"fecha={fecha}) con el patrón h-individual esperado.")

    logger.info(f"  [REPLOT] Fold {fold_num} cargado desde disco "
               f"({len(modelos_por_h)} horizontes)")
    return modelos_por_h


def _cargar_modelos_step004(banco: str) -> dict | None:
    """
    Carga los modelos del step004 (eval) para el banco indicado.
    Retorna dict {tau: Booster} o None si no se encuentran archivos.
    Soporta naming xgb_qt y xgb.
    """
    if not DIR_MODELOS_STEP004.exists():
        logger.warning(f"  [S4] Directorio step004 no encontrado: {DIR_MODELOS_STEP004}")
        return None

    sfx = _s4_prefijo
    # Buscar metadata para obtener fecha del modelo
    metas = sorted(DIR_MODELOS_STEP004.glob(f"metadata_{sfx}_{banco}_*.json"), reverse=True)
    if not metas:
        # fallback: sin metadata, buscar modelos directamente
        metas = sorted(DIR_MODELOS_STEP004.glob(f"metadata_{sfx}_wfcv_v3_{banco}_*.json"), reverse=True)
    if not metas:
        logger.warning(f"  [S4] No se encontró metadata step004 en {DIR_MODELOS_STEP004}")
        return None

    meta  = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha = metas[0].stem.split("_")[-1]
    quantiles_s4 = meta.get("quantiles", QUANTILES)

    modelos_s4 = {}
    for tau in quantiles_s4:
        ruta = DIR_MODELOS_STEP004 / f"{sfx}_{banco}_q{int(tau*100):02d}_{fecha}.json"
        if not ruta.exists():
            logger.warning(f"  [S4] Modelo no encontrado: {ruta.name}")
            return None
        b = xgb.Booster(); b.load_model(str(ruta))
        modelos_s4[tau] = b

    logger.info(f"  [S4] Modelos step004 cargados: {metas[0].name}  "
                f"({len(modelos_s4)} cuantiles)")
    return modelos_s4


def evaluar_banco(banco: str):
    modo = "EXPANDING" if EXPANDING else "ROLLING"
    logger.info(f"\n{'='*65}")
    logger.info(f"BANCO: {banco}  — Walk-Forward CV v3  [{modo}]  [TEST OOS]")
    logger.info(f"{'='*65}")
    logger.info(
        f"  TRAIN {VENTANA_TRAIN_AÑOS}yr{'(min)' if EXPANDING else ''} | "
        f"VAL {VENTANA_VAL_AÑOS}yr (Optuna) | TEST {VENTANA_TEST_AÑOS}yr (métricas) | "
        f"paso {PASO_AÑOS}yr | purge {PURGE_DIAS_HAB}dh | burn-in {BURN_IN_DIAS_HAB}dh | "
        f"trials={'adaptivo' if ADAPTIVE_TRIALS else f'flat={TRIALS_FLAT}'} | "
        f"workers_optuna={MAX_WORKERS_OPTUNA if MAX_WORKERS_OPTUNA else 'auto'}"
    )
    # Advertencia ruidosa: una corrida debug produce metricas submuestreadas que
    # se ven normales en los CSV. Sin este aviso es facil archivarlas como si
    # fueran una corrida real.
    if MODO_DEBUG:
        _n_h = sum(len(hs) for hs, _ in H_GRUPOS.values())
        logger.warning("  " + "!" * 63)
        logger.warning(f"  !! MODO_DEBUG ACTIVO — corrida de diagnostico, "
                       f"NO usar sus metricas")
        logger.warning(f"  !! {_n_h} horizontes (normal: 74) | "
                       f"TRIALS_FLAT={TRIALS_FLAT} | "
                       f"SHAP<={DIAG_SHAP_MAX_SAMPLES} | "
                       f"DIAG_N_REPEATS={DIAG_N_REPEATS}")
        logger.warning("  " + "!" * 63)
    if ADAPTIVE_TRIALS:
        _tau_trials = {tau: get_n_trials(tau) for tau in QUANTILES}
        logger.info(f"  Trials por cuantil: " +
                    " | ".join(f"τ={t:.2f}→{n}" for t, n in _tau_trials.items()))

    t_inicio = time.time()

    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    if df.empty or df["target"].notna().sum() < 500:
        logger.warning(f"  [{banco}] Datos insuficientes — omitiendo")
        return None

    # Columnas baseline del feature de régimen (NaN por defecto): necesarias
    # para que get_feature_cols las detecte automáticamente. Cada fold las
    # sobrescribe sin leakage en reemplazar_regimen_fold; si por algún motivo
    # un fold no tiene bloque HMM válido, quedan NaN y se imputan con la
    # mediana de TRAIN igual que cualquier otro feature (mismo mecanismo ya
    # existente más abajo, sin necesidad de casos especiales).
    if USAR_FEATURE_REGIMEN:
        df["regimen_hmm"]   = np.nan
        df["regimen_sigma"] = np.nan

    cols_feat = get_feature_cols(df)
    fechas    = pd.DatetimeIndex(df["fecha_t"].unique())
    logger.info(f"  [{banco}] {len(df):,} filas | {len(cols_feat)} features | "
                f"rango: {fechas.min().date()} → {fechas.max().date()}")

    # Cargar modelos step004 para comparación (opcional)
    modelos_s4 = _cargar_modelos_step004(banco) if COMPARAR_CON_STEP004 else None
    if COMPARAR_CON_STEP004 and modelos_s4 is None:
        logger.warning("  [S4] No se pudo cargar step004 — comparación desactivada")

    folds = generar_folds(
        fechas_disponibles=fechas,
        ventana_train_años=VENTANA_TRAIN_AÑOS,
        ventana_val_años=VENTANA_VAL_AÑOS,
        ventana_test_años=VENTANA_TEST_AÑOS,
        paso_años=PASO_AÑOS,
        purge_dias_hab=PURGE_DIAS_HAB,
        purge_val_test=PURGE_VAL_TEST,
        expanding=EXPANDING,
        fold_fecha_inicio=FOLD_FECHA_INICIO,
    )
    if not folds:
        logger.error(f"  [{banco}] No se generaron folds")
        return None

    if N_MAX_FOLDS is not None and len(folds) > N_MAX_FOLDS:
        logger.info(f"  [{banco}] Limitando a {N_MAX_FOLDS} folds "
                    f"(de {len(folds)} disponibles) — "
                    f"datos desde {folds[N_MAX_FOLDS]['test_start'].date()} quedan OOS")
        folds = folds[:N_MAX_FOLDS]

    # Folds manuales (p.ej. réplica exacta del split de step004)
    if FOLDS_MANUALES:
        n_previos = 0 if SOLO_FOLDS_MANUALES else len(folds)
        folds_man = resolver_folds_manuales(FOLDS_MANUALES, fechas, n_previos)
        if SOLO_FOLDS_MANUALES:
            folds = folds_man
            logger.info(f"  [{banco}] Modo SOLO_FOLDS_MANUALES — "
                        f"{len(folds_man)} fold(s) manual(es) en lugar de los automáticos")
        else:
            folds = folds + folds_man
            logger.info(f"  [{banco}] +{len(folds_man)} fold(s) manual(es) añadido(s)")

    logger.info(f"  [{banco}] {len(folds)} folds generados:")
    for f in folds:
        n_train_yr = round(f["n_train_fechas"] / 252, 1)
        tag = " [MANUAL]" if f.get("_manual") else ""
        logger.info(
            f"    Fold {f['fold']:2d}{tag} | TRAIN {f['train_start'].date()} → "
            f"{f['train_end'].date()} ({n_train_yr}yr, {f['n_train_fechas']}dh) | "
            f"VAL  {f['val_start'].date()} → {f['val_end'].date()} | "
            f"TEST {f['test_start'].date()} → {f['test_end'].date()}"
        )

    # Regimenes HMM alineados a ESTOS folds. Va aca porque necesita los folds ya
    # cerrados (train_end definitivos, incluidos N_MAX_FOLDS y los manuales) y
    # antes del bucle, que es donde reemplazar_regimen_fold lee los parquets.
    # Se piden TODAS las entidades, no solo `banco`: rho_ij necesita el bloque de
    # la contraparte, y con BANCOS_A_EVALUAR=[FOCO, RESTO, ...] el primer
    # evaluar_banco corre antes de que exista el parquet del segundo.
    asegurar_regimenes_hmm(BANCOS_A_EVALUAR, folds)

    resultados_test   = []
    resultados_val    = []
    por_h_test        = []
    por_h_val         = []
    importancias_folds = []
    # v3.7.0 — acumuladores del diagnóstico fold × h × τ (reemplazan a
    # diag_por_fold, que guardaba una sola entrada promediada por fold).
    # Snapshot previo para el diff de tracemalloc entre folds (ver
    # DIAGNOSTICO_MEMORIA). None en el fold 1: no hay con que comparar todavia.
    _snap_prev = None
    if DIAGNOSTICO_MEMORIA and not tracemalloc.is_tracing():
        tracemalloc.start(1)

    diag_rows          = []   # una fila por (fold, h, τ) con gain_/perm_/shap_
    diag_familia_rows  = []   # una fila por (fold, h, τ) con Δ conjunto por familia
    hp_rows            = []   # una fila por (fold, grupo, τ) para el reporte HP
    tablas_consola     = {}   # {nombre_hoja: DataFrame} para el Excel de resumen
    modelos_ultimo    = None
    params_ultimo     = None
    folds_manifest    = []   # registro de todos los folds para fan chart histórico
    fecha_hoy         = pd.Timestamp.today().strftime("%Y%m%d")

    # ── Modo regenerar plots: carga metadata del disco para obtener fecha_hoy ───
    _meta_disco = None
    if SOLO_REGENERAR_PLOTS:
        try:
            _meta_disco   = _cargar_metadata_disco(banco)
            folds_manifest = _meta_disco.get("folds_manifest", [])
            # Indexar por fold_num para acceso rápido
            _fm_idx = {fi["fold"]: fi for fi in folds_manifest}
            logger.info(f"  [REPLOT] {len(folds_manifest)} folds en manifest — "
                        f"solo se regenerarán los fan charts")
        except FileNotFoundError as _e_meta:
            logger.error(f"  [REPLOT] {_e_meta}")
            logger.error("  [REPLOT] Ejecuta primero con SOLO_REGENERAR_PLOTS=False")
            return None
    
    all_preds_base    = []   # acumula predicciones base (sin overlay) para exportar
    all_preds_overlay = []   # acumula predicciones finales (con overlay) para exportar
    all_overlay_meta  = []   # acumula metadata del factor por fecha_t
    
    for fold in folds:
        t_fold = time.time()
        logger.info(f"\n  ── Fold {fold['fold']}/{len(folds)} ──────────────────────")

        try:
            (X_train, y_train,
             X_val,   y_val,
             X_test,  y_test,
             h_train, h_val, h_test,
             fechas_t_test, año_corte_regimen,
             df_fold_all) = preparar_fold_data(df, fold, cols_feat, banco)
        except Exception as e:
            logger.warning(f"  Fold {fold['fold']}: error preparando datos — {e}")
            continue

        if USAR_FEATURE_REGIMEN:
            if año_corte_regimen is not None:
                logger.info(f"    [REGIMEN] feature tomado del bloque HMM "
                           f"año_corte={año_corte_regimen}")
            else:
                logger.info(f"    [REGIMEN] sin bloque HMM válido — feature queda NaN/mediana")

        if len(X_train) < 200 or len(X_val) < 20 or len(X_test) < 20:
            logger.warning(f"  Fold {fold['fold']}: datos insuficientes — omitiendo")
            continue

        std_y = float(y_train.std())
        if std_y < 1.0:
            logger.warning(f"    Fold {fold['fold']}: std_y={std_y:.4f} anormalmente bajo — "
                           f"forzado a 1.0 para evitar división por cero en objetivo GARCH")
            std_y = 1.0
        logger.info(f"    X_train={len(X_train):,} | X_val={len(X_val):,} | "
                    f"X_test={len(X_test):,} | std_y={std_y:,.0f}")

        # Descomposición h=2: se evalúa UNA VEZ por fold, ANTES del if/else,
        # porque _predecir_por_h (que la consume) corre en AMBOS modos —
        # entrenamiento normal y SOLO_REGENERAR_PLOTS. Definirla solo dentro
        # de la rama de entrenamiento provocaría NameError al regenerar plots.
        #
        # En modo replot los modelos vienen del disco: si fueron entrenados
        # con la descomposición activa, predicen D(t+2) y hay que
        # destransformar igual. El guard sobre R_conf_t2 sigue aplicando: si
        # la columna no está disponible ahora, no se puede destransformar y
        # se omite (con aviso), que es la degradación correcta.
        _h2_desc_ok = _h2_descomponible(X_train, X_val, X_test)

        if SOLO_REGENERAR_PLOTS:
            # ── Modo replot: carga modelos del disco, salta Optuna ───────────
            fold_num  = fold["fold"]
            fold_info = _fm_idx.get(fold_num)
            if fold_info is None:
                logger.warning(f"  [REPLOT] Fold {fold_num} no está en el manifest — omitiendo")
                continue
            try:
                modelos = _cargar_modelos_fold_disco(fold_info, banco)
            except FileNotFoundError as _e_load:
                logger.warning(f"  [REPLOT] {_e_load} — omitiendo fold {fold_num}")
                continue
            best_params = {}
            modelos_final = modelos
        else:
            # ── Modo normal: Optuna por GRUPO + un modelo por h individual ──
            # Estrategia DIRECTA (adoptada de step005_walk_forward_cv_4_final.py):
            # h ya NO es feature (ver COLS_EXCLUIR). Por cada grupo de H_GRUPOS,
            # Optuna busca hiperparámetros UNA VEZ sobre el h representativo;
            # esos mismos HP se reutilizan (sin nueva búsqueda) para entrenar
            # un modelo dedicado a cada h individual del grupo.
            modelos_por_h        = {}   # {h: {tau:model,"mean":model}} — solo TRAIN
            modelos_final_por_h  = {}   # {h: {tau:model,"mean":model}} — TRAIN+VAL
            best_params_por_grupo = {}  # {nombre_grupo: bp_representativo} — manifest

            for nombre_grupo, (hs_grupo, h_rep) in H_GRUPOS.items():
                m_tr_rep = h_train == h_rep
                m_va_rep = h_val   == h_rep

                if m_tr_rep.sum() < 30 or m_va_rep.sum() < 10:
                    # Fallback: NO se salta el grupo entero por que el h
                    # representativo puntual sea escaso -- eso dejaba TODOS
                    # los h del grupo sin modelo, y preds_test/preds_val
                    # quedaban en NaN para esas filas. pinball_loss usa
                    # np.mean() sin filtrar NaN, así que UN grupo saltado
                    # contaminaba las métricas del FOLD COMPLETO, no solo
                    # esos horizontes. En vez de eso, se busca Optuna sobre
                    # el POOL de todo el grupo (todos sus h juntos).
                    m_tr_pool = np.isin(h_train, hs_grupo)
                    m_va_pool = np.isin(h_val,   hs_grupo)
                    if m_tr_pool.sum() < 30 or m_va_pool.sum() < 10:
                        logger.warning(
                            f"    [{nombre_grupo}] muy pocos datos incluso en "
                            f"el pool completo del grupo (train={int(m_tr_pool.sum())}, "
                            f"val={int(m_va_pool.sum())}) — grupo omitido este fold")
                        continue
                    logger.warning(
                        f"    [{nombre_grupo}] h_rep={h_rep} con datos insuficientes "
                        f"(train={int(m_tr_rep.sum())}, val={int(m_va_rep.sum())}) — "
                        f"usando el POOL del grupo completo para Optuna en su lugar "
                        f"(train={int(m_tr_pool.sum())}, val={int(m_va_pool.sum())})")
                    m_tr_rep, m_va_rep = m_tr_pool, m_va_pool

                _, best_params_grupo = entrenar_fold(
                    X_train[m_tr_rep], y_train[m_tr_rep],
                    X_val[m_va_rep],   y_val[m_va_rep],
                    std_y, get_n_trials(0.5), fold["fold"]
                )
                best_params_por_grupo[nombre_grupo] = best_params_grupo

                if MODELO_CV == "xgb_qt":
                    _best_by_tau_grupo = best_params_grupo.pop("_best_by_tau_full", None)
                    if _best_by_tau_grupo is None:
                        raise RuntimeError(
                            f"entrenar_fold no devolvió '_best_by_tau_full' para "
                            f"el grupo {nombre_grupo!r}.")
                else:
                    _best_by_tau_grupo = None   # xgb/lgbm: un solo dict plano

                # ── Reporte HP: una fila por (fold, grupo, τ) ────────────────
                # _optuna_meta se saca ANTES de que best_params_grupo llegue al
                # manifest, para que ese dict siga teniendo solo las keys planas
                # de siempre (learning_rate, s, n_estimators, ...).
                _opt_meta = best_params_grupo.pop("_optuna_meta", {}) or {}
                if _best_by_tau_grupo:
                    for _tau_hp, _bp_hp in _best_by_tau_grupo.items():
                        _meta_hp = _opt_meta.get(_tau_hp, {})
                        hp_rows.append({
                            "fold" : fold["fold"], "grupo": nombre_grupo,
                            "h_rep": h_rep,        "tau"  : _tau_label(_tau_hp),
                            **{k: v for k, v in _bp_hp.items()
                               if not k.startswith("_")},
                            "best_pinball_val": _meta_hp.get("best_pinball_val", np.nan),
                            "trial_values"    : _meta_hp.get("trial_values", []),
                        })

                logger.info(f"    [{nombre_grupo}] HP buscados en h_rep={h_rep} "
                           f"(train={int(m_tr_rep.sum())}, val={int(m_va_rep.sum())}) "
                           f"— aplicando a h={hs_grupo}")

                # Un modelo dedicado por cada h individual del grupo, con los
                # MISMOS hiperparámetros del representante (sin Optuna de nuevo).
                for h_ind in hs_grupo:
                    m_tr_h = h_train == h_ind
                    m_va_h = h_val   == h_ind
                    if m_tr_h.sum() < 30:
                        logger.warning(f"      h={h_ind}: muy pocos datos train "
                                       f"({int(m_tr_h.sum())}) — omitiendo")
                        continue

                    # ── Descomposición h=2: entrenar sobre D = target + R_conf ──
                    # Solo para h=H_MIN y solo si el guard pasó. Los demás
                    # horizontes usan el target sin tocar.
                    _y_tr_h = y_train[m_tr_h]
                    _y_va_h = y_val[m_va_h]
                    if h_ind == H_MIN and _h2_desc_ok:
                        _y_tr_h = _y_tr_h + X_train.loc[m_tr_h, COL_R_CONF_H2].to_numpy(dtype=float)
                        if m_va_h.sum() > 0:
                            _y_va_h = _y_va_h + X_val.loc[m_va_h, COL_R_CONF_H2].to_numpy(dtype=float)
                        logger.info(f"      h={h_ind}: entrenando sobre D = target + "
                                   f"{COL_R_CONF_H2} (descomposición activa)")

                    if MODELO_CV == "xgb_qt":
                        modelos_por_h[h_ind] = _entrenar_train_only_qt(
                            X_train[m_tr_h], _y_tr_h,
                            _best_by_tau_grupo, std_y, h_ind, fold["fold"]
                        )
                        if m_va_h.sum() >= 5:
                            modelos_final_por_h[h_ind] = _retrain_train_val_qt(
                                X_train[m_tr_h], _y_tr_h,
                                X_val[m_va_h],   _y_va_h,
                                _best_by_tau_grupo, std_y, fold["fold"]
                            )
                        else:
                            logger.warning(f"      h={h_ind}: muy pocos datos val "
                                           f"({int(m_va_h.sum())}) para retrain — "
                                           f"usando modelo solo-train también para TEST")
                            modelos_final_por_h[h_ind] = modelos_por_h[h_ind]
                    else:
                        X_tv_h = pd.concat([X_train[m_tr_h], X_val[m_va_h]], ignore_index=True)
                        y_tv_h = pd.concat([pd.Series(np.asarray(_y_tr_h)),
                                            pd.Series(np.asarray(_y_va_h))], ignore_index=True)
                        modelos_final_por_h[h_ind] = entrenar_quantiles(
                            X_tv_h, y_tv_h, best_params_grupo, QUANTILES, std_y)
                        modelos_por_h[h_ind] = entrenar_quantiles(
                            X_train[m_tr_h], _y_tr_h, best_params_grupo,
                            QUANTILES, std_y)

            # best_params/modelos/modelos_final: alias para el resto del fold
            # (importancias, diagnóstico, manifest) que aún esperan estas
            # variables — ahora representan el conjunto POR-H, no un dict plano.
            best_params   = best_params_por_grupo
            modelos       = modelos_por_h
            modelos_final = modelos_final_por_h

            # Importancia de features — promedio entre todos los h entrenados
            # (cada h aporta su propio conjunto de modelos por tau).
            try:
                imp_por_h = {h: _extraer_importancias(m, cols_feat)
                            for h, m in modelos_final_por_h.items()}
                importancias_folds.append({"fold": fold["fold"], "importancias_por_h": imp_por_h})
            except Exception as _e_imp:
                logger.warning(f"    Importancia fold {fold['fold']}: {_e_imp}")

            if DIAGNOSTICO_FEATURES:
                # v3.7.0: se conserva el desglose fold × h × τ en vez de
                # promediarlo dos veces. diag_rows crece con una fila por
                # (fold, h, τ); guardar_diag_y_plots lo vuelca a CSV y lo vacía.
                logger.info(f"    [diag] Fold {fold['fold']}: gain(train) + "
                            f"block-perm(val) + shap(val) por h y por τ")
                for h_ind, m_h in modelos_por_h.items():
                    m_va_h = h_val == h_ind
                    if m_va_h.sum() < 10:
                        continue
                    _Xv_h, _yv_h = X_val[m_va_h], y_val[m_va_h]
                    try:
                        diag_rows.extend(
                            diagnosticar_h(m_h, _Xv_h, _yv_h, cols_feat,
                                           fold["fold"], h_ind))
                    except Exception as _e_diag:
                        logger.warning(f"    [diag] Fold {fold['fold']} h={h_ind} "
                                       f"falló: {_e_diag}")
                        continue
                    # Permutación conjunta por familia — único diagnóstico que
                    # distingue "no aporta" de "aporta pero es redundante
                    # consigo mismo". ~4% de cómputo extra sobre la individual.
                    try:
                        _perm_fam = _diag_perm_familias_h(
                            m_h, _Xv_h, _yv_h, FAMILIAS_PERM, cols_feat, fold["fold"])
                        for _tau_f, _deltas in _perm_fam.items():
                            diag_familia_rows.append({
                                "fold": fold["fold"], "h": int(h_ind),
                                "tau": _tau_label(_tau_f), **_deltas})
                    except Exception as _e_fam:
                        logger.warning(f"    [diag] Fold {fold['fold']} h={h_ind} "
                                       f"familias falló: {_e_fam}")

        # ── predecir_fold ahora opera por h y se apila en el orden original ──
        # de X_test/X_val (h_test/h_val ya indican a qué modelo va cada fila).
        def _predecir_por_h(modelos_dict_por_h, X_df, h_arr):
            claves_tau = sorted({t for m in modelos_dict_por_h.values() for t in m},
                                key=lambda t: (t != "mean", t))
            out = {t: np.full(len(X_df), np.nan) for t in claves_tau}
            for h_ind, m_h in modelos_dict_por_h.items():
                mask = h_arr == h_ind
                if not mask.any():
                    continue
                preds_h = predecir_fold(m_h, X_df[mask])
                # Descomposición h=2: el modelo predijo D(t+2); hay que volver
                # a la escala del target restando el retiro confirmado.
                #   q_τ(target) = max(q_τ(D), 0) − R_conf_t2
                # Sin este paso los cuantiles quedarían desplazados por
                # R_conf_t2 — un error silencioso y del tamaño del retiro.
                if h_ind == H_MIN and _h2_desc_ok:
                    _r = X_df.loc[mask, COL_R_CONF_H2].to_numpy(dtype=float)
                    preds_h = _destransformar_h2(preds_h, _r)
                for t, arr in preds_h.items():
                    out[t][mask] = arr
            return out

        preds_test = _predecir_por_h(modelos_final, X_test, h_test)
        preds_val  = _predecir_por_h(modelos,       X_val,  h_val)

        # ── Calibración post-hoc (por horizonte h) ───────────────────────────
        # Shift calculado por separado para cada h en VAL y aplicado solo a
        # filas del mismo h en TEST/VAL. Evita que residuos de horizontes
        # largos (mayor varianza) contaminen el percentil de horizontes cortos.
        # CALIBRACION_MAX_SHIFT_FACTOR limita el shift a ±factor×std_y para
        # evitar que un sesgo grande del modelo distorsione el eje Y.
        if CALIBRACION_POSTHOC and 0.50 in preds_val:
            _cap = (CALIBRACION_MAX_SHIFT_FACTOR * std_y
                    if CALIBRACION_MAX_SHIFT_FACTOR is not None else np.inf)
            _shifts_h = {}
            for _h in np.unique(h_val):
                _mask_h = h_val == _h
                _res_h  = y_val.values[_mask_h] - preds_val[0.50][_mask_h]
                if len(_res_h) >= 2:
                    _s = float(np.percentile(_res_h, CALIBRACION_PERCENTIL))
                    _shifts_h[int(_h)] = float(np.clip(_s, -_cap, _cap))
            _adj_test = np.array([_shifts_h.get(int(h), 0.0) for h in h_test])
            _adj_val  = np.array([_shifts_h.get(int(h), 0.0) for h in h_val])
            preds_test = {tau: arr + _adj_test for tau, arr in preds_test.items()}
            preds_val  = {tau: arr + _adj_val  for tau, arr in preds_val.items()}
            logger.info(
                f"    [CALIBRACION] P{CALIBRACION_PERCENTIL} por-h "
                f"(cap=±{_cap:,.0f} = ±{CALIBRACION_MAX_SHIFT_FACTOR}×std_y): "
                f"h=1→{_shifts_h.get(1,0):,.0f}  "
                f"h=38→{_shifts_h.get(38,0):,.0f}  "
                f"h=75→{_shifts_h.get(75,0):,.0f}"
            )
            
        # -- Overlay sobreencaje -----------------------------------------------
        # Lee peor_total desde Excel (step007) y aplica factor multiplicativo
        # uniforme sobre todo el horizonte, por fecha_t unica.
        # -- Snapshot base (antes de overlay) -- siempre se guarda ----------------
        _base_dict = {tau: arr.copy() for tau, arr in preds_test.items()}

        if OVERLAY_SOBREENCAJE:
            preds_test, _meta_fold = _aplicar_overlay_sobreencaje(preds_test, h_test, fechas_t_test, df)
            all_overlay_meta.extend(_meta_fold)
            
        # -- Scaffolding compartido ------------------------------------------------
        _scaffold = pd.DataFrame({
            "banco"  : banco,
            "fold"   : fold["fold"],
            "fecha_t": pd.DatetimeIndex(fechas_t_test),
            "h"      : h_test,
            "target" : y_test.values,
        })

        # Predicciones base (sin overlay)
        _preds_base_df = _scaffold.copy()
        for _tau, _arr in _base_dict.items():
            _col = "mean" if _tau == "mean" else f"q{int(_tau * 100):02d}"
            _preds_base_df[_col] = _arr
        all_preds_base.append(_preds_base_df)

        # Predicciones finales (post-overlay si activo, igual que base si no)
        _preds_df = _scaffold.copy()
        for _tau, _arr in preds_test.items():
            _col = "mean" if _tau == "mean" else f"q{int(_tau * 100):02d}"
            _preds_df[_col] = _arr
        all_preds_overlay.append(_preds_df)    
        
        # ── Estimacion de rho_s en VAL (anti-leakage) ──────────────────────
        _rho_s_val = None
        _rho_ij    = None   # rho transversal; queda None si la entidad no tiene
                            # contraparte (SISTEMA) o falta su parquet
        if ESTIMAR_RHO_EN_VAL and USAR_FEATURE_REGIMEN and año_corte_regimen is not None:
            # Nivel 3 (exclusión): si el fold HMM es degenerado (state collapse),
            # la clasificación de régimen en VAL no es confiable — los pares
            # (z_{t-1}, z_t) estarían mal etiquetados por régimen y el rho_s
            # resultante sería basura. Se omite la estimación y se usa el
            # fallback del orquestador (columnas rho_s_* ausentes → fallback).
            _pkl_path_diag = DIR_REGIMEN_HMM / f"modelo_hmm_{banco}_{año_corte_regimen}.pkl"
            _fold_degenerado = False
            if _pkl_path_diag.exists() and HMM_EXCLUIR_FOLDS_DEGENERADOS_WF:
                try:
                    import pickle as _pkl_diag
                    with open(_pkl_path_diag, "rb") as _pfd:
                        _diag_ok = _pkl_diag.load(_pfd).get("diag_ok", True)
                    _fold_degenerado = not _diag_ok
                except Exception as _e_pkl:
                    logger.debug(
                        f"  [RHO_VAL] No se pudo leer diag_ok del pickle "
                        f"{_pkl_path_diag.name} → {type(_e_pkl).__name__}: {_e_pkl}. "
                        f"Se asume fold NO degenerado (diag_ok=True).")
            if not _fold_degenerado:
                logger.debug(
                    f"  [RHO_VAL] Fold año_corte={año_corte_regimen}: "
                    f"_fold_degenerado=False (HMM estable o pickle no leído) — "
                    f"se procederá a estimar rho en VAL.")
            else:
                # Sin esto, un fold degenerado salta TODO el bloque de rho_s/
                # rho_ij en absoluto silencio: ni warning ni error, la corrida
                # termina "exitosa" con coverage/pinball normales y sin que
                # nada en el log indique por que rho_ij nunca aparecio. Costo
                # real: una tarde de diagnostico a ciegas buscando el bug en
                # la estimacion cuando el problema era el HMM del fold, no la
                # formula de correlacion.
                logger.warning(
                    f"  [RHO_VAL] Fold año_corte={año_corte_regimen} banco={banco}: "
                    f"HMM DEGENERADO (diag_ok=False en {_pkl_path_diag.name}, "
                    f"umbral HMM_MIN_DIAG_TRANSMAT en step005_validar_hmm_v5.py) "
                    f"— se omiten rho_s y rho_ij para este fold. Fallback del "
                    f"orquestador (columnas rho_s_*/rho_ij ausentes). Si esto "
                    f"ocurre en TODOS los folds de una entidad, esa entidad "
                    f"probablemente tiene muy pocas observaciones para 3 "
                    f"regimenes — considerar N_ESTADOS=2 en validar_hmm_v5.")

        if ESTIMAR_RHO_EN_VAL and USAR_FEATURE_REGIMEN and año_corte_regimen is not None and not _fold_degenerado:
            try:
                _h_min_rho = H_MIN_RHO_VAL if H_MIN_RHO_VAL is not None else int(h_val.min())

                # Se levanta lo que YA calculó step005_validar_hmm_v3.py —
                # mismas dos funciones que ya usa reemplazar_regimen_fold()
                # para el feature de XGBoost. Nada se reclasifica ni se
                # vuelve a estimar aqui: estado, sigma y flujo por fecha ya
                # estan en estados_regimen_hmm_<banco>.parquet.
                _df_estados_rho = _cargar_estados_regimen_disco(banco)

                if _df_estados_rho is not None:
                    _bloque_rho = (
                        _df_estados_rho[_df_estados_rho["año_corte"] == año_corte_regimen]
                        .drop_duplicates("fecha")
                        .set_index("fecha")[["estado", "sigma", "flujo"]]
                        .sort_index()
                    )
                    if not _bloque_rho.empty:
                        _rho_s_val = _estimar_rho_val_fold(
                            _bloque_rho,
                            h_min=_h_min_rho,
                            max_gap_dias=MAX_GAP_DIAS_RHO,
                            rho_default=RHO_DEFAULT_VAL,
                            tau_ewma_pares=TAU_EWMA_RHO_PARES,
                        )
                        logger.info(f"    [RHO_VAL] rho_s estimado (año_corte={año_corte_regimen}): {_rho_s_val}")

                        # ── rho_ij transversal (D2 del paper) ──────────────
                        # Mismo bloque año_corte que phi, asi que hereda su
                        # anti-leakage: no ve nada posterior a train_end.
                        _contraparte = _banco_contraparte(banco)
                        if _contraparte is not None:
                            _df_otro = _cargar_estados_regimen_disco(_contraparte)
                            if _df_otro is None:
                                logger.info(
                                    f"    [RHO_IJ] Sin parquet de {_contraparte} "
                                    f"— se omite rho_ij. Con HMM_INTERNO=True y "
                                    f"esa entidad en BANCOS_A_EVALUAR se genera sola.")
                            else:
                                _bloque_otro = (
                                    _df_otro[_df_otro["año_corte"] == año_corte_regimen]
                                    .drop_duplicates("fecha")
                                    .set_index("fecha")[["estado", "sigma", "flujo"]]
                                    .sort_index()
                                )
                                # Regimen para condicionar: el de SISTEMA si esta,
                                # NO el propio. Cada entidad tiene su HMM y sus
                                # etiquetas difieren el mismo dia, asi que usar el
                                # propio haria que rho_ij calculado desde FOCO no
                                # coincida con el calculado desde RESTO — y esa
                                # simetria es justamente el chequeo gratis que da
                                # correr las dos caras en una pasada.
                                _df_sist = _cargar_estados_regimen_disco("SISTEMA")
                                _est_cond = None
                                if _df_sist is not None:
                                    _bl_sist = _df_sist[
                                        _df_sist["año_corte"] == año_corte_regimen]
                                    if not _bl_sist.empty:
                                        _est_cond = (_bl_sist.drop_duplicates("fecha")
                                                     .set_index("fecha")["estado"])
                                _rho_ij = _estimar_rho_transversal(
                                    _bloque_rho, _bloque_otro,
                                    estados_cond=_est_cond,
                                    tau_ewma_pares=TAU_EWMA_RHO_PARES,
                                )
                                if _rho_ij:
                                    _rho_ij["contraparte"] = _contraparte
                                    logger.info(
                                        f"    [RHO_IJ] {banco} vs {_contraparte} "
                                        f"(año_corte={año_corte_regimen}): {_rho_ij}")
                    else:
                        # Bloque vacio para ESTE año_corte, pero el resto del
                        # parquet (otros folds, misma corrida de
                        # step005_validar_hmm*.py) sirve para inferir cuantos
                        # estados hay — todos comparten el mismo N_ESTADOS.
                        _n_est_fallback = _n_estados_de(_df_estados_rho)
                        logger.warning(f"  [RHO_VAL] Bloque vacio para año_corte={año_corte_regimen} "
                                       f"— usando rho_default")
                        _rho_s_val = {s: RHO_DEFAULT_VAL for s in range(_n_est_fallback)}
                else:
                    logger.warning("  [RHO_VAL] estados_regimen_hmm no disponible — usando rho_default")
                    _rho_s_val = {s: RHO_DEFAULT_VAL for s in range(_n_estados_de(None))}
            except Exception as _e_rho:
                logger.warning(f"  [RHO_VAL] Error estimando rho en VAL: {_e_rho}")

        if GUARDAR_PREDS_TEST and not SOLO_REGENERAR_PLOTS:
            _regimen_hmm_test   = (X_test["regimen_hmm"].values
                                   if "regimen_hmm" in X_test.columns else None)
            _regimen_sigma_test = (X_test["regimen_sigma"].values
                                   if "regimen_sigma" in X_test.columns else None)
            _guardar_preds_test(
                preds_test, y_test.values, h_test, fechas_t_test,
                fold["fold"], banco, fecha_hoy, DIR_MODO,
                regimen_hmm=_regimen_hmm_test,
                regimen_sigma=_regimen_sigma_test,
                año_corte_regimen=año_corte_regimen,
                rho_s_val=_rho_s_val,
                rho_ij=_rho_ij)
            
        if not SOLO_REGENERAR_PLOTS:
            row_test = calcular_metricas_fold(preds_test, y_test.values, fold, "test")
            row_val  = calcular_metricas_fold(preds_val,  y_val.values,  fold, "val")
            row_test["tiempo_min"] = round((time.time() - t_fold) / 60, 2)
            row_val["modelo_cv"]   = MODELO_CV
            row_test["modelo_cv"]  = MODELO_CV

            def _hp(d, key, default=0):
                return d.get(key, default)

            for row in (row_test, row_val):
                if MODELO_CV == "lgbm":
                    row.update({
                        "s_optimo"        : 0.0,
                        "learning_rate"   : round(_hp(best_params, "learning_rate"), 4),
                        "max_depth"       : int(_hp(best_params, "num_leaves")),
                        "n_estimators"    : int(_hp(best_params, "n_estimators")),
                        "min_child_weight": int(_hp(best_params, "min_child_samples")),
                        "subsample"       : round(_hp(best_params, "subsample"), 3),
                        "colsample_bytree": round(_hp(best_params, "colsample_bytree"), 3),
                        "reg_alpha"       : round(_hp(best_params, "reg_alpha"), 5),
                        "reg_lambda"      : round(_hp(best_params, "reg_lambda"), 5),
                    })
                else:
                    row.update({
                        "s_optimo"        : round(_hp(best_params, "s"), 4),
                        "learning_rate"   : round(_hp(best_params, "learning_rate"), 4),
                        "max_depth"       : int(_hp(best_params, "max_depth")),
                        "n_estimators"    : int(_hp(best_params, "n_estimators")),
                        "min_child_weight": int(_hp(best_params, "min_child_weight")),
                        "subsample"       : round(_hp(best_params, "subsample"), 3),
                        "colsample_bytree": round(_hp(best_params, "colsample_bytree"), 3),
                        "reg_alpha"       : round(_hp(best_params, "reg_alpha"), 5),
                        "reg_lambda"      : round(_hp(best_params, "reg_lambda"), 5),
                    })

            resultados_test.append(row_test)
            resultados_val.append(row_val)
            por_h_test.append(calcular_metricas_por_h(preds_test, y_test.values, h_test, fold["fold"]))
            por_h_val.append(calcular_metricas_por_h(preds_val,  y_val.values,  h_val,  fold["fold"]))

            cov_t = row_test.get("coverage_90", float("nan"))
            cov_v = row_val.get("coverage_90",  float("nan"))
            logger.info(
                f"    TEST: coverage={cov_t:.1%}  pinball_Q50={row_test.get('pinball_q50','?'):,.0f}  "
                f"winkler={row_test.get('winkler_90','?'):,.0f}  "
                f"| VAL coverage={cov_v:.1%} (sesgo={cov_v-cov_t:+.1%})  "
                f"({row_test['tiempo_min']} min)"
            )

        # Fan charts TEST — folds manuales van a carpeta separada
        preds_s4  = predecir_y_corregir(modelos_s4, X_test) if modelos_s4 is not None else None
        _fanchart_dir = DIR_FANCHARTS_MANUALES if fold.get("_manual") else None
        graficar_fanchart_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test, fold, banco,
            preds_overlay=preds_s4,
            dir_out=_fanchart_dir,
        )
        graficar_fanchart_acum_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test, fold, banco,
            dir_out=_fanchart_dir, variante="banda",
        )
        graficar_fanchart_acum_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test, fold, banco,
            dir_out=_fanchart_dir, variante="punto",
        )
        graficar_fanchart_acum_test_fold(
            preds_test, y_test.values, h_test, fechas_t_test, fold, banco,
            dir_out=_fanchart_dir, variante="punto_q05",
        )

        modelos_ultimo = modelos_final
        params_ultimo  = best_params

        if not SOLO_REGENERAR_PLOTS:
            # ── Guardar modelo del fold + manifest ───────────────────────────
            sfx = "lgbm_wfcv_v3" if MODELO_CV == "lgbm" else f"{MODELO_CV}_wfcv_v3"
            ext = ".txt" if MODELO_CV == "lgbm" else ".json"
            fold_num = fold["fold"]

            garch_fold = {}
            try:
                garch_fold = _extraer_garch_params_fold(df, fold["train_end"])
            except Exception as _eg:
                logger.warning(f"  Fold {fold_num}: no se pudo extraer GARCH — {_eg}")

            if GUARDAR_MODELOS_TODOS_FOLDS:
                for h_ind, modelos_h in modelos_final.items():
                    for tau, model in modelos_h.items():
                        if tau == "mean":
                            ruta_f = (DIR_MODELOS /
                                      f"{sfx}_{banco}_fold{fold_num:02d}_h{h_ind:02d}_mean_{fecha_hoy}{ext}")
                        else:
                            ruta_f = (DIR_MODELOS /
                                      f"{sfx}_{banco}_fold{fold_num:02d}_h{h_ind:02d}_q{int(tau*100):02d}_{fecha_hoy}{ext}")
                        model.save_model(str(ruta_f))
                logger.info(f"    Modelos fold {fold_num} ({len(modelos_final)} horizontes) "
                           f"guardados en {DIR_MODELOS.name}/")

            folds_manifest.append({
                "fold"       : fold_num,
                "train_start": str(fold["train_start"].date()),
                "train_end"  : str(fold["train_end"].date()),
                "test_start" : str(fold["test_start"].date()),
                "test_end"   : str(fold["test_end"].date()),
                "fecha_hoy"  : fecha_hoy,
                "garch"      : garch_fold,
            })

        del X_train, y_train, X_val, y_val, X_test, y_test

        # Los modelos del fold (74 horizontes x 8 modelos x 2 dicts ~ 1,200
        # boosters) solo se soltaban al reasignar los dicts en el fold siguiente,
        # DESPUES de que preparar_fold_data ya habia cargado la matriz nueva: en
        # ese punto convivian los modelos viejos con los datos nuevos. Soltarlos
        # aca adelanta la liberacion al final del fold que los produjo.
        # Se asigna None en vez de 'del' porque segun MODELO_CV algunos de estos
        # nombres pueden no haberse creado, y 'del' lanzaria NameError.
        modelos = modelos_final = modelos_por_h = modelos_final_por_h = None
        gc.collect()

        # RSS del padre tras liberar: si crece fold a fold hay retencion real;
        # si se mantiene y aun asi baja la RAM libre del sistema, es
        # fragmentacion del heap (memoria devuelta por Python pero no al SO).
        try:
            import psutil
            _rss = psutil.Process().memory_info().rss / 1e9
            _av  = psutil.virtual_memory().available / 1e9
            logger.info(f"    [mem] fin fold {fold['fold']}: RSS padre={_rss:.2f} GB "
                        f"| RAM libre={_av:.1f} GB")
        except ImportError:
            pass

        # Atribucion del crecimiento a la linea que lo asigna. Se compara contra
        # el fold anterior, no contra el arranque: lo que interesa es lo que
        # queda retenido DESPUES de la limpieza, no el pico de trabajo.
        if DIAGNOSTICO_MEMORIA:
            _snap = tracemalloc.take_snapshot()
            if _snap_prev is not None:
                _top = [st for st in _snap.compare_to(_snap_prev, "lineno")
                        if st.size_diff > 0][:10]
                _tot = sum(st.size_diff for st in _top) / 1e6
                logger.info(f"    [memtop] fold {fold['fold']} vs fold anterior "
                            f"(+{_tot:.1f} MB en los 10 mayores, solo Python):")
                for _st in _top:
                    _fr = _st.traceback[0]
                    _arch = Path(_fr.filename).name
                    logger.info(f"      +{_st.size_diff / 1e6:8.2f} MB  "
                                f"{_arch}:{_fr.lineno}")
            _snap_prev = _snap
        
    # -- Exportar predicciones a parquet (input para orquestador/video) ----------
    if all_preds_base:
        df_base_all = pd.concat(all_preds_base, ignore_index=True)
        ruta_base = DIR_MODO / f"preds_base_{banco}_{fecha_hoy}.parquet"
        df_base_all.to_parquet(ruta_base, index=False)
        logger.info(f"  [{banco}] Predicciones base guardadas: {ruta_base.name}")

    if all_preds_overlay:
        df_preds_all = pd.concat(all_preds_overlay, ignore_index=True)
        ruta_preds = DIR_MODO / f"preds_overlay_{banco}_{fecha_hoy}.parquet"
        df_preds_all.to_parquet(ruta_preds, index=False)
        logger.info(f"  [{banco}] Predicciones overlay guardadas: {ruta_preds.name}")

    # -- Exportar metadata del overlay a CSV (diagnóstico) -----------------------
    if all_overlay_meta:
        df_meta = pd.DataFrame(all_overlay_meta)
        ruta_meta = DIR_MODO / f"overlay_meta_{banco}_{fecha_hoy}.csv"
        df_meta.to_csv(ruta_meta, index=False)
        logger.info(f"  [{banco}] Metadata overlay guardada: {ruta_meta.name}")

    if SOLO_REGENERAR_PLOTS:
        logger.info(f"\n  [REPLOT] Fan charts regenerados para {banco}. "
                    f"Métricas y metadata no actualizados.")
        return None

    if not resultados_test:
        logger.error(f"  [{banco}] Ningún fold completado")
        return None

    df_test_m  = pd.DataFrame(resultados_test)
    df_val_m   = pd.DataFrame(resultados_val)
    df_por_h_t = pd.concat(por_h_test, ignore_index=True) if por_h_test else pd.DataFrame()
    df_por_h_v = pd.concat(por_h_val,  ignore_index=True) if por_h_val  else pd.DataFrame()
    # fecha_hoy ya fue definido antes del loop para consistencia en nombres de archivo

    def _save(df, nombre):
        ruta = DIR_MODO / nombre
        df.to_csv(ruta, index=False)
        logger.info(f"    {nombre}")

    logger.info(f"  [{banco}] Archivos en: {DIR_MODO}")
    _save(df_test_m,  f"wfcv_v3_metricas_TEST_{banco}_{fecha_hoy}.csv")
    _save(df_val_m,   f"wfcv_v3_metricas_VAL_{banco}_{fecha_hoy}.csv")
    if not df_por_h_t.empty:
        _save(df_por_h_t, f"wfcv_v3_por_h_TEST_{banco}_{fecha_hoy}.csv")
    if not df_por_h_v.empty:
        _save(df_por_h_v, f"wfcv_v3_por_h_VAL_{banco}_{fecha_hoy}.csv")

    cols_hp    = ["fold", "train_start", "train_end", "modelo_cv", "expanding",
                  "s_optimo", "learning_rate", "max_depth", "n_estimators",
                  "min_child_weight", "subsample", "colsample_bytree",
                  "reg_alpha", "reg_lambda"]
    cols_hp_ok = [c for c in cols_hp if c in df_test_m.columns]
    _save(df_test_m[cols_hp_ok], f"wfcv_v3_hiperparametros_{banco}_{fecha_hoy}.csv")

    tag = f"{banco}_{MODELO_CV}_{'exp' if EXPANDING else 'rol'}"
    graficar_metricas_wfcv(df_test_m, tag)
    graficar_comparacion_val_test(df_val_m, df_test_m, tag)
    graficar_cobertura_por_h(df_por_h_t, tag, "test")
    graficar_cobertura_por_h(df_por_h_v, tag, "val")
    graficar_hiperparametros_wfcv(df_test_m, tag)
    graficar_importancia_por_fold(importancias_folds, cols_feat, tag)

    # ── Reporte de hiperparámetros Optuna (fold × grupo × τ) ─────────────────
    if hp_rows:
        try:
            guardar_hp_report(hp_rows, DIR_MODO, tag, fecha_hoy)
            _reg(tablas_consola, "hp_report",
                 pd.DataFrame([{k: v for k, v in r.items() if k != "trial_values"}
                               for r in hp_rows]))
        except Exception as _e_hp:
            logger.warning(f"  [HP] guardar_hp_report falló: {_e_hp}")

    # ── Diagnóstico de features (fold × h × τ) ───────────────────────────────
    # ORDEN OBLIGATORIO: guardar_diag_familias necesita las columnas perm_<feat>
    # de diag_rows, y guardar_diag_y_plots vacía diag_rows para liberar memoria
    # antes de graficar. Invertir el orden deja la comparación por familias en
    # silencio, sin error.
    if DIAGNOSTICO_FEATURES and diag_rows:
        try:
            _resumen_fam = guardar_diag_familias(
                diag_rows, diag_familia_rows, FAMILIAS_PERM,
                DIR_MODO, tag, fecha_hoy)
            _reg(tablas_consola, "perm_familias", _resumen_fam)
        except Exception as _e_fam:
            logger.warning(f"  [diag] guardar_diag_familias falló: {_e_fam}")
        try:
            guardar_diag_y_plots(diag_rows, cols_feat, DIR_MODO, tag, fecha_hoy)
        except Exception as _e_cons:
            logger.warning(f"  [diag] guardar_diag_y_plots falló: {_e_cons}")

    if GUARDAR_MODELO_FINAL and modelos_ultimo is not None:
        ultimo = folds[-1]

        # Reusar GARCH del último fold ya calculado en el manifest (evita 3ª estimación)
        garch_params_prod = folds_manifest[-1]["garch"] if folds_manifest else {}
        if garch_params_prod:
            logger.info(f"  GARCH producción (del manifest): {list(garch_params_prod.keys())}")
        else:
            try:
                garch_params_prod = _extraer_garch_params_fold(df, ultimo["train_end"])
            except Exception as _eg:
                logger.warning(f"  No se pudieron extraer GARCH params: {_eg}")

        for h_ind, modelos_h in modelos_ultimo.items():
            for tau, model in modelos_h.items():
                sfx = "lgbm_wfcv_v3" if MODELO_CV == "lgbm" else f"{MODELO_CV}_wfcv_v3"
                ext = ".txt" if MODELO_CV == "lgbm" else ".json"
                if tau == "mean":
                    ruta_m = DIR_MODELOS / f"{sfx}_{banco}_h{h_ind:02d}_mean_{fecha_hoy}{ext}"
                else:
                    ruta_m = DIR_MODELOS / f"{sfx}_{banco}_h{h_ind:02d}_q{int(tau*100):02d}_{fecha_hoy}{ext}"
                model.save_model(str(ruta_m))
        metadata = {
            "banco": banco, "modelo": f"{MODELO_CV}_wfcv_v3",
            "fecha_entrenamiento": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "version": "v3 — expanding/rolling + TEST OOS",
            "config": {
                "expanding"          : EXPANDING,
                "ventana_train_años" : VENTANA_TRAIN_AÑOS,
                "ventana_val_años"   : VENTANA_VAL_AÑOS,
                "ventana_test_años"  : VENTANA_TEST_AÑOS,
                "paso_años"          : PASO_AÑOS,
                "h_max_dias_hab"     : H_MAX_DIAS_HAB,
                "lookback_max_dias"  : LOOKBACK_MAX_DIAS,
                "embargo_dias_hab"   : EMBARGO_DIAS_HAB,
                "purge_dias_hab"     : PURGE_DIAS_HAB,
                "purge_val_test"     : PURGE_VAL_TEST,
                "burn_in_dias_hab"   : BURN_IN_DIAS_HAB,
                "adaptive_trials"    : ADAPTIVE_TRIALS,
                "trials_flat"        : TRIALS_FLAT,
                "trials_por_tau"     : TRIALS_POR_TAU if ADAPTIVE_TRIALS else {},
            },
            "anti_leakage": {
                "purga_train_val"  : (f"{PURGE_DIAS_HAB} dh post-TRAIN = "
                                      f"h_max({H_MAX_DIAS_HAB}) + lookback({LOOKBACK_MAX_DIAS}) "
                                      f"+ embargo({EMBARGO_DIAS_HAB})"),
                "purga_val_test"   : f"{PURGE_VAL_TEST} dh post-VAL (simétrico)",
                "embargo"          : (f"{EMBARGO_DIAS_HAB} dh por autocorrelación serial "
                                      f"en bordes (adicional a la purga mecánica)"),
                "burn_in"          : f"{BURN_IN_DIAS_HAB} dh inicio TRAIN excluidos (MA22 warm-up)",
                "garch_por_fold"   : "ω/α/β estimados en TRAIN, propagados a VAL+TEST",
                "medianas_por_fold": "calculadas en TRAIN, aplicadas a VAL+TEST",
                "val_test_sep"     : "VAL=Optuna only / TEST=métricas OOS only",
            },
            "ultimo_fold": {
                "train_start": str(ultimo["train_start"].date()),
                "train_end"  : str(ultimo["train_end"].date()),
                "test_start" : str(ultimo["test_start"].date()),
                "test_end"   : str(ultimo["test_end"].date()),
            },
            "garch_produccion": {
                "train_end" : str(ultimo["train_end"].date()),
                "series"    : garch_params_prod,
                "uso"       : (
                    "Usar omega/alpha/beta para propagar GARCH desde train_end "
                    "en producción — garantiza consistencia entrenamiento-predicción"
                ),
            },
            "n_folds": len(folds), "quantiles": QUANTILES,
            "features": cols_feat, "best_params_ultimo_fold": params_ultimo,
            "folds_manifest": folds_manifest,
            "guardar_todos_folds": GUARDAR_MODELOS_TODOS_FOLDS,
        }
        ruta_meta = DIR_MODELOS / f"metadata_{MODELO_CV}_wfcv_v3_{banco}_{fecha_hoy}.json"
        with open(ruta_meta, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)

    t_total = time.time() - t_inicio
    logger.info(f"\n  {'─'*60}")
    logger.info(f"  RESUMEN — {banco}  [{modo}]  [TEST OOS]")
    logger.info(f"  {'─'*60}")
    cols_d  = ["fold", "train_start", "train_end", "test_start", "test_end",
               "pinball_q50", "coverage_90", "winkler_90", "crps_approx"]
    cols_ok = [c for c in cols_d if c in df_test_m.columns]
    logger.info("\n" + df_test_m[cols_ok].to_string(index=False))

    prom_t = {c: df_test_m[c].mean() for c in ["pinball_q50", "coverage_90",
              "winkler_90", "crps_approx"] if c in df_test_m.columns}
    logger.info(f"\n  Promedios TEST:")
    for k, v in prom_t.items():
        logger.info(f"    {k:20s}: {f'{v:.1%}' if k=='coverage_90' else f'{v:,.2f}'}")

    if "coverage_90" in df_val_m.columns and "coverage_90" in df_test_m.columns:
        sesgo = df_val_m["coverage_90"].mean() - df_test_m["coverage_90"].mean()
        logger.info(f"\n  Sesgo VAL−TEST coverage: {sesgo:+.2%}")

    # ── Excel de resumen: una hoja por tabla, valores NUMÉRICOS ──────────────
    # Complementa a consola_*.txt, que conserva el formato de pantalla pero no
    # se puede ordenar ni graficar.
    _reg(tablas_consola, "metricas_TEST", df_test_m)
    _reg(tablas_consola, "metricas_VAL",  df_val_m)
    if not df_por_h_t.empty:
        _reg(tablas_consola, "por_h_TEST", df_por_h_t)
    if not df_por_h_v.empty:
        _reg(tablas_consola, "por_h_VAL",  df_por_h_v)
    if prom_t:
        _reg(tablas_consola, "promedios_TEST", pd.Series(prom_t, name="valor"))
    try:
        guardar_tablas_excel(tablas_consola, DIR_MODO, tag, fecha_hoy)
    except Exception as _e_xls:
        logger.warning(f"  guardar_tablas_excel falló: {_e_xls}")

    logger.info(f"\n  ✓ Completado en {t_total/60:.1f} min  ({len(folds)} folds)")
    return df_test_m


###############################################################################
# Main
###############################################################################

def main():
    """
    Envoltura fina: abre el log de consola y delega en _main_interno, para no
    reindentar el cuerpo entero dentro de un try/finally.
    """
    DIR_MODO.mkdir(parents=True, exist_ok=True)
    _tee = None
    try:
        _tee = _Tee(DIR_MODO /
                    f"consola_{pd.Timestamp.today().strftime('%Y%m%d')}.txt")
    except Exception as _e_tee:
        logger.warning(f"No se pudo abrir el log de consola: {_e_tee}")
    try:
        _main_interno()
    finally:
        if _tee is not None:
            _tee.cerrar()


def _main_interno():
    modo = "EXPANDING" if EXPANDING else "ROLLING"
    logger.info("=" * 65)
    logger.info(f"STEP005 v3 — Walk-Forward CV [{modo}] + TEST OOS  [{MODELO_CV.upper()}]")
    logger.info("=" * 65)
    logger.info(f"  EXPANDING={EXPANDING}  TRAIN_min={VENTANA_TRAIN_AÑOS}yr  "
                f"VAL={VENTANA_VAL_AÑOS}yr  TEST={VENTANA_TEST_AÑOS}yr  "
                f"paso={PASO_AÑOS}yr")
    # Qué matriz y qué entidades, explícito: con el botón en False el log es el
    # mismo de antes salvo esta línea, y con True evita la duda de si la corrida
    # leyó la matriz particionada o la v1.
    if PARTICIONES:
        logger.info(f"  PARTICIONES=True  particion={PARTICION!r}  "
                    f"entidad={ENTIDAD!r}  ->  {BANCOS_A_EVALUAR}")
    else:
        logger.info(f"  PARTICIONES=False (sin partición)  ->  {BANCOS_A_EVALUAR}")
    logger.info(f"  Matriz: {RUTA_MATRIZ.name}")

    if MODELO_CV == "lgbm" and not _LGBM_OK:
        logger.error("MODELO_CV='lgbm' pero lightgbm no está instalado.")
        return
    if not RUTA_MATRIZ.exists():
        logger.error(f"Matriz no encontrada: {RUTA_MATRIZ}")
        return

    t0 = time.time()
    todos = []
    for banco in BANCOS_A_EVALUAR:
        # Reapuntar las carpetas a ESTA entidad. Solo con PARTICIONES=True: con
        # False la rama no se ejecuta y los globales conservan el valor que les
        # dio el bloque de rutas, que es el de siempre.
        #
        # Se reasignan los globales en vez de pasar las rutas por parámetro
        # porque hay 56 referencias a estas cinco constantes repartidas por el
        # archivo. Las tres funciones que más las usan (guardar_diag_y_plots,
        # guardar_hp_report, guardar_tablas_excel) ya reciben dir_modo y banco
        # como argumentos, así que la parte delicada ya estaba preparada.
        if PARTICIONES:
            global DIR_MODO, DIR_MODELOS, DIR_PLOTS
            global DIR_FANCHARTS, DIR_FANCHARTS_MANUALES
            _d = dirs_de_banco(banco)
            DIR_MODO, DIR_MODELOS  = _d["modo"], _d["modelos"]
            DIR_PLOTS              = _d["plots"]
            DIR_FANCHARTS          = _d["fancharts"]
            DIR_FANCHARTS_MANUALES = _d["fancharts_man"]
            logger.info(f"  [{banco}] salida → {DIR_MODO}")

        df_m = evaluar_banco(banco)
        if df_m is not None:
            todos.append((banco, df_m))

    if todos:
        logger.info("\n" + "=" * 65)
        logger.info(f"RESUMEN GLOBAL — TEST OOS  [{modo}]")
        logger.info("=" * 65)
        for banco, df_m in todos:
            avg_cov = df_m["coverage_90"].mean() if "coverage_90" in df_m.columns else float("nan")
            avg_pb  = df_m["pinball_q50"].mean()  if "pinball_q50"  in df_m.columns else float("nan")
            logger.info(f"  {banco:15s}: {len(df_m)} folds | "
                        f"coverage_90_avg={avg_cov:.1%} | pinball_Q50_avg={avg_pb:,.0f}")

    logger.info(f"\n✓ Total: {(time.time()-t0)/60:.1f} min  →  {DIR_MODO}")


if __name__ == "__main__":
    main()
