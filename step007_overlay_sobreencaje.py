# -*- coding: utf-8 -*-
"""
step007_overlay_sobreencaje.py
==============================
Overlay comportamental para retiros de sobreencaje en cierres trimestrales.

Problema que resuelve
----------------------
El modelo cuantílico predice el flujo neto del sistema bancario para los
siguientes 75 días hábiles.  En cierres trimestrales (mar/jun/sep/dic) ciertos
bancos -- en particular BBVA -- ejecutan retiros masivos de encaje + overnight que
exceden el percentil 1% pronosticado.  Este overlay ajusta esa asimetría de
forma transparente y auditable, sin re-entrenar el modelo.

Mapeo retiros <-> saldos
-----------------------
Los nombres de bancos en Transacciones_BancaLocal (columna Broker) difieren de
los nombres en el archivo de saldo CC+OVN.  TablaSaldosRetiros.xlsx contiene el
mapeo explícito entre ambas fuentes, incluyendo cambios históricos de nombre
(CONTINEN -> BBVA, FINANCIERO -> BANCO PICHINCHA).
Bancos con BANCOS_SALDOS = "NA" (BONY, FEDERAL) son excluidos del overlay.

Lógica en 3 pasos
------------------
Paso 1 -- Detección
    Para cada banco B y cada uno de los últimos N cierres trimestrales:
        ratio[B,q] = |retiro_7dh[B,q]| / saldo_max_mes[B,q]
    El banco contribuye al ajuste si ratio[B,q] > UMBRAL_ACTIVACION en AL MENOS UNO
    de los últimos N cierres ("alguna_vez_activa").
    "activa" (último trimestre > umbral) se reporta también para diagnóstico.

Paso 2 -- Peor retiro potencial por banco
    Para bancos con alguna_vez_activa = True:
        worst_ratio_B = max(ratio[B,q]) sobre TODOS los cierres históricos con ratio > umbral
        p95_saldo_B   = P{PERCENTIL_SALDO} del saldo desde inicio del trimestre ANTERIOR hasta fecha_ref-1
                        (trimestre anterior completo + días corridos del trimestre actual sin incluir hoy)
        peor_B        = worst_ratio_B × p95_saldo_B × (1 + FACTOR_SEGURIDAD)

Paso 3 -- Factor multiplicativo y ajuste
    peor_total = sum peor_B   (bancos con alguna_vez_activa)
    f          = peor_total / |Q01_h_cierre|
    Si f > 1 -> Q_adj[tau, h_cierre] = Q[tau, h_cierre] × f
    Aplica de forma simétrica a todos los cuantiles y la mediana.
    Solo afecta horizontes h que correspondan a un cierre trimestral.

Arquitectura
------------
step007 calcula el ajuste (peor_total); step005 aplica el factor.

Uso en step005
--------------
    from step007_overlay_sobreencaje import get_peor_total, OVERLAY_SOBREENCAJE_ACTIVO

    peor_total, fecha_cierre = get_peor_total(fecha_origen)  # 0.0, None si sin bancos activos
    if peor_total > 0:
        # Localizar el h del primer cierre en el array de horizontes
        bh = pd.bdate_range(start=fecha_origen + BDay(1), periods=75)
        h_cierre = int(np.where(bh == fecha_cierre)[0][0]) + 1
        idx = np.where(h_arr == h_cierre)[0]
        q01_cierre = float(preds[0.01][idx[0]]) if len(idx) else float(preds[0.01][-1])
        f = peor_total / abs(q01_cierre) if q01_cierre < 0 else 1.0
        if f > 1.0:
            # Factor uniforme para toda la ventana de proyeccion
            preds = {tau: arr * f for tau, arr in preds.items()}

Uso desde step006 / postproceso (DataFrame con fecha_t, h, q01..q99)
----------------------------------------------------------------------
    from step007_overlay_sobreencaje import aplicar_overlay_df

    df_preds_adj = aplicar_overlay_df(df_preds)   # lee fecha_t desde el DataFrame

Diagnóstico / outliers standalone
----------------------------------
    python step007_overlay_sobreencaje.py
    -> carga Saldos_CCOVN, detecta outliers por banco y muestra bancos activos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

###############################################################################
# Configuración
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")

# -- Master switch -------------------------------------------------------------
OVERLAY_SOBREENCAJE_ACTIVO = False

# -- Parámetros del overlay ----------------------------------------------------
UMBRAL_ACTIVACION     = 0.50   # ratio |retiro_7dh| / saldo_P95_mes para activar
N_TRIMESTRES_LOOKBACK = 4      # cierres para la tabla Señal (display); el worst_ratio usa todo el historial
MIN_TRIMESTRES_ACTIVO = 1      # mín. trimestres (dentro del lookback) con ratio > umbral para activar
VENTANA_RETIRO_DH     = 7      # días hábiles de la ventana antes del cierre
FACTOR_SEGURIDAD      = 0.00   # margen extra sobre peor_B  (0.00->0.10 = 0%->10%)
PERCENTIL_SALDO       = 0.95   # percentil del saldo usado en el denominador del ratio y en peor_B

# -- Detección de outliers (diagnóstico) --------------------------------------
OUTLIER_ZSCORE = 4.0   # |z| > 4 sigma -> sospechoso (muy conservador para datos de saldo)
OUTLIER_IQR    = 6.0   # valor > mediana ± 6×IQR -> outlier severo

# -- Rutas de datos ------------------------------------------------------------
# Tabla de mapeo BANCOS_RETIROS <-> BANCOS_SALDOS
RUTA_TABLA = Path(__file__).parent / "TablaSaldosRetiros.xlsx"

# Saldo fin del día (CC + Overnight) por banco -- formato wide:
#   col 0 = fecha | col 1..N = bancos (nombre = BANCOS_SALDOS)
RUTA_SALDO = BASE_SISTEMA / "1. Data" / "Raw" / "Saldos_CCOVN.xlsx"

# Alternativa si el archivo tiene nombre diferente:
# RUTA_SALDO = BASE_SISTEMA / "1. Data" / "Raw" / "Saldo fin del dia CC+OVN.xlsx"

# Transacciones bancarias -- formato transaccional:
#   columnas: Broker (= BANCOS_RETIROS), Fecha Valor, Delivery Principal Usd
RUTA_TRANS = BASE_SISTEMA / "1. Data" / "Raw" / "Transacciones_BancaLocal.xlsx"

# -- Cuantiles -----------------------------------------------------------------
QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]
Q_COLS    = [f"q{int(t * 100):02d}" for t in QUANTILES]   # q01, q05, q50, q95, q99


###############################################################################
# Carga del mapeo BANCOS_RETIROS <-> BANCOS_SALDOS
###############################################################################

def cargar_tabla_mapeo(ruta: Path = RUTA_TABLA) -> pd.DataFrame:
    """
    Lee TablaSaldosRetiros.xlsx.
    Retorna DataFrame con columnas BANCOS_RETIROS, BANCOS_SALDOS.
    Excluye automáticamente filas donde BANCOS_SALDOS = 'NA' (sin datos de saldo).
    """
    df = pd.read_excel(ruta)
    df.columns = df.columns.str.strip()
    df["BANCOS_RETIROS"] = df["BANCOS_RETIROS"].astype(str).str.strip()
    df["BANCOS_SALDOS"]  = df["BANCOS_SALDOS"].astype(str).str.strip()
    df = df[df["BANCOS_SALDOS"].str.upper() != "NA"].reset_index(drop=True)
    return df


def mapeo_retiros_a_saldos(ruta: Path = RUTA_TABLA) -> dict[str, str]:
    """Retorna {nombre_retiros: nombre_saldos} excluyendo bancos sin saldo."""
    df = cargar_tabla_mapeo(ruta)
    return dict(zip(df["BANCOS_RETIROS"], df["BANCOS_SALDOS"]))


def mapeo_saldos_a_retiros(ruta: Path = RUTA_TABLA) -> dict[str, str]:
    """Retorna {nombre_saldos: nombre_retiros} excluyendo bancos sin saldo."""
    df = cargar_tabla_mapeo(ruta)
    return dict(zip(df["BANCOS_SALDOS"], df["BANCOS_RETIROS"]))


###############################################################################
# Carga de datos de saldo y transacciones
###############################################################################

def _cargar_saldos(ruta: Path) -> pd.DataFrame:
    """
    Lee el archivo de saldo fin del día CC + OVN.
    Formato: primera columna = fecha, columnas restantes = bancos (BANCOS_SALDOS).

    El archivo contiene días calendario (incluyendo sábados, domingos y feriados).
    Se filtran solo días hábiles: lunes-viernes (dayofweek < 5) y con al menos
    un banco con dato no-NaN (elimina feriados donde no hubo operaciones).

    Retorna DataFrame con index=fechas hábiles, columnas=bancos.
    """
    raw = pd.read_excel(ruta, header=0)
    col_fecha   = raw.columns[0]
    cols_bancos = raw.columns[1:].tolist()

    raw[col_fecha] = pd.to_datetime(raw[col_fecha], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=[col_fecha]).sort_values(col_fecha).set_index(col_fecha)
    raw.index.name = "fecha"

    for c in cols_bancos:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw.columns = [str(c).strip() for c in raw.columns]

    # Filtro 1: excluir sábados (5) y domingos (6)
    raw = raw[raw.index.dayofweek < 5]

    # Filtro 2: excluir feriados -- filas donde todos los bancos son NaN
    raw = raw[raw.notna().any(axis=1)]

    return raw


def _bancos_en_transacciones(ruta: Path) -> set[str]:
    """Retorna nombres únicos en la columna Broker de Transacciones_BancaLocal."""
    df = pd.read_excel(ruta, usecols=["Broker"])
    return {str(b).strip() for b in df["Broker"].dropna().unique()}


###############################################################################
# Análisis de outliers por banco (diagnóstico)
###############################################################################

def detectar_outliers_banco(
    serie: pd.Series,
    zscore_thresh: float = OUTLIER_ZSCORE,
    iqr_thresh: float = OUTLIER_IQR,
) -> pd.DataFrame:
    """
    Detecta valores atípicos en una serie de saldo diario.

    Usa dos criterios independientes:
      - Z-score: |z| > zscore_thresh
      - IQR: valor fuera de [Q25 - iqr_thresh×IQR, Q75 + iqr_thresh×IQR]

    Retorna DataFrame con columnas: fecha, valor, z_score, outlier_zscore,
    outlier_iqr, outlier (True si cualquiera de los dos criterios se activa).
    """
    s = serie.dropna()
    if len(s) < 30:
        return pd.DataFrame()

    mu, sigma = s.mean(), s.std()
    q25, q75  = s.quantile(0.25), s.quantile(0.75)
    iqr       = q75 - q25

    df_out = pd.DataFrame({
        "fecha":  s.index,
        "valor":  s.values,
    })
    df_out["z_score"]       = (s.values - mu) / (sigma if sigma > 0 else 1)
    df_out["outlier_zscore"] = df_out["z_score"].abs() > zscore_thresh
    df_out["outlier_iqr"]   = (
        (s.values < q25 - iqr_thresh * iqr) |
        (s.values > q75 + iqr_thresh * iqr)
    )
    df_out["outlier"] = df_out["outlier_zscore"] | df_out["outlier_iqr"]
    return df_out[df_out["outlier"]].reset_index(drop=True)


def analizar_outliers_saldos(
    df_saldo: pd.DataFrame,
    bancos_objetivo: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Corre detectar_outliers_banco para cada banco en bancos_objetivo.
    Retorna {banco: DataFrame_outliers}.
    """
    resultado = {}
    for banco in bancos_objetivo:
        if banco not in df_saldo.columns:
            continue
        outliers = detectar_outliers_banco(df_saldo[banco])
        if not outliers.empty:
            resultado[banco] = outliers
    return resultado


###############################################################################
# Calendario y fechas de cierre trimestral
###############################################################################

def _cierres_trimestrales(calendario: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Último día hábil de cada trimestre (meses 3, 6, 9, 12) en el calendario."""
    df = pd.DataFrame({"fecha": calendario})
    df["mes"]  = df["fecha"].dt.month
    df["anio"] = df["fecha"].dt.year
    df["trim"] = df["fecha"].dt.quarter
    return (
        df[df["mes"].isin([3, 6, 9, 12])]
        .groupby(["anio", "trim"])["fecha"]
        .max()
        .sort_values()
        .tolist()
    )


def _quarter_start(fecha: pd.Timestamp) -> pd.Timestamp:
    """Primer día del trimestre al que pertenece `fecha`."""
    return pd.Timestamp(fecha.year, ((fecha.month - 1) // 3) * 3 + 1, 1)


def _prev_quarter_start(fecha: pd.Timestamp) -> pd.Timestamp:
    """Primer día del trimestre anterior al que pertenece `fecha`."""
    q = fecha.quarter
    if q == 1:
        return pd.Timestamp(fecha.year - 1, 10, 1)
    return pd.Timestamp(fecha.year, (q - 2) * 3 + 1, 1)


def _ventana_antes_cierre(
    fecha_cierre: pd.Timestamp,
    calendario: pd.DatetimeIndex,
    n: int,
) -> pd.DatetimeIndex:
    """Últimos n días hábiles del calendario terminando en fecha_cierre."""
    pos = int(calendario.searchsorted(fecha_cierre, side="right"))
    return calendario[max(0, pos - n): pos]


###############################################################################
# Paso 1 -- Detección por banco
###############################################################################

def _ratio_banco_cierre(
    df_saldo: pd.DataFrame,
    df_flujos: pd.DataFrame,
    col_saldo: str,
    fecha_cierre: pd.Timestamp,
    calendario: pd.DatetimeIndex,
    n_dh: int,
) -> float | None:
    """
    ratio = |retiro_neto_acumulado en ventana n_dh dh| / saldo_P95 del mes del cierre.
    Retorna None si el flujo neto es positivo (ingreso) -- la estrategia solo aplica a retiros.
    """
    mask_mes = (
        (df_saldo.index.year  == fecha_cierre.year) &
        (df_saldo.index.month == fecha_cierre.month)
    )
    col_data  = df_saldo.loc[mask_mes, col_saldo]
    # Si hay columnas duplicadas, loc devuelve DataFrame -- tomar la primera columna
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    serie_mes = col_data.dropna()
    if serie_mes.empty or float(serie_mes.max()) <= 0:
        return None
    saldo_max = float(serie_mes.quantile(PERCENTIL_SALDO))

    ventana     = _ventana_antes_cierre(fecha_cierre, calendario, n_dh)
    disponibles = df_flujos.index.intersection(ventana)
    if disponibles.empty:
        return None
    retiro = float(df_flujos.loc[disponibles, col_saldo].sum(skipna=True))
    # Solo cuenta retiro neto (saldo baja). Si hubo ingreso neto, no aplica la estrategia.
    if retiro >= 0:
        return None
    return abs(retiro) / saldo_max


def detectar_bancos_activos(
    df_saldo: pd.DataFrame,
    df_flujos: pd.DataFrame,
    bancos_saldo: list[str],
    cierres: list[pd.Timestamp],
    calendario: pd.DatetimeIndex,
    cierres_lookback: list[pd.Timestamp] | None = None,
    n_dh: int = VENTANA_RETIRO_DH,
    umbral: float = UMBRAL_ACTIVACION,
    min_trimestres: int = MIN_TRIMESTRES_ACTIVO,
) -> dict[str, dict]:
    """
    Evalúa la estrategia de sobreencaje para cada banco.

    cierres          : historial completo -> se computan todos los ratios (para worst_ratio).
    cierres_lookback : ventana de activación (últimos N_TRIMESTRES_LOOKBACK).
                       Si None se usa cierres[-N_TRIMESTRES_LOOKBACK:].

    "activa"            = True si ratio en el ÚLTIMO cierre del lookback > umbral.
    "alguna_vez_activa" = True si ratio > umbral en >= min_trimestres del lookback.
    """
    if cierres_lookback is None:
        cierres_lookback = cierres[-N_TRIMESTRES_LOOKBACK:]
    lb_set = set(cierres_lookback)

    resultado: dict[str, dict] = {}
    for col in bancos_saldo:
        if col not in df_saldo.columns:
            continue
        ratios: dict[pd.Timestamp, float] = {}
        for fc in cierres:
            r = _ratio_banco_cierre(df_saldo, df_flujos, col, fc, calendario, n_dh)
            if r is not None:
                ratios[fc] = r

        # Activación restringida al lookback (últimos N trimestres)
        ratios_lb  = {fc: r for fc, r in ratios.items() if fc in lb_set}
        n_activos  = sum(1 for r in ratios_lb.values() if r > umbral)
        activa     = bool(ratios_lb.get(max(ratios_lb), 0) > umbral) if ratios_lb else False
        alguna_vez_activa = n_activos >= min_trimestres

        resultado[col] = {
            "activa":               activa,
            "alguna_vez_activa":    alguna_vez_activa,
            "n_trimestres_activos": n_activos,
            "ratios":               ratios,
        }
    return resultado


###############################################################################
# Paso 2 -- Peor retiro potencial por banco
###############################################################################

def _peor_B(
    df_saldo: pd.DataFrame,
    det: dict[str, dict],
    col_saldo: str,
    cierres: list[pd.Timestamp],
    fecha_ref: pd.Timestamp,
    factor_seg: float = FACTOR_SEGURIDAD,
    umbral: float = UMBRAL_ACTIVACION,
    percentil: float = PERCENTIL_SALDO,
) -> float:
    """
    peor_B = worst_ratio_B × P{percentil}_saldo_trim_actual × (1 + factor_seg)

    worst_ratio      : máximo ratio histórico sobre TODOS los cierres donde el banco
                       usó la estrategia (cierres debe incluir todo el historial).
    p{percentil}_saldo: percentil del saldo desde el inicio del trimestre actual hasta
                        fecha_ref, inclusive -- captura lo disponible hoy para retirar.
    """
    ratios = det[col_saldo]["ratios"]
    ratios_activos = {fc: r for fc, r in ratios.items() if r > umbral}
    if not ratios_activos:
        return 0.0

    worst_ratio = max(ratios_activos.values())

    # Saldo: trimestre anterior completo + días corridos del trimestre actual (sin incluir fecha_ref)
    pq_start = _prev_quarter_start(fecha_ref)
    mask = (df_saldo.index >= pq_start) & (df_saldo.index < fecha_ref)
    s = df_saldo.loc[mask, col_saldo]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    if s.empty or float(s.max()) <= 0:
        return 0.0

    p_saldo = float(s.quantile(percentil))
    return worst_ratio * p_saldo * (1.0 + factor_seg)


###############################################################################
# Paso 3 -- Factor multiplicativo
###############################################################################

def _factor_overlay(peor_total: float, q01_val: float) -> float:
    """
    f = peor_total / |Q01_h_cierre|
    Retorna 1.0 si el modelo ya cubre el peor caso o si Q01 >= 0.
    """
    if q01_val >= 0.0 or abs(q01_val) < 1e-6:
        return 1.0
    f = peor_total / abs(q01_val)
    return f if f > 1.0 else 1.0


###############################################################################
# Carga unificada de datos (saldo + flujos + mapeo)
###############################################################################

def _preparar_datos(
    ruta_saldo: Path,
    ruta_tabla: Path,
    fecha_origen: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[pd.Timestamp], pd.DatetimeIndex] | None:
    """
    Carga saldos, aplica el mapeo, calcula flujos y selecciona los últimos
    N cierres históricos.  Retorna None si hay un error crítico.
    """
    # Mapeo BANCOS_RETIROS -> BANCOS_SALDOS
    try:
        tabla = cargar_tabla_mapeo(ruta_tabla)
    except Exception as e:
        logger.warning(f"[OVERLAY] No se pudo leer TablaSaldosRetiros: {e}")
        return None

    # Saldos por banco
    try:
        df_saldo_raw = _cargar_saldos(ruta_saldo)
    except Exception as e:
        logger.warning(f"[OVERLAY] No se pudo leer saldo CC+OVN: {e}")
        return None

    # Columnas de saldo disponibles en el archivo (normalizar)
    cols_saldo_disponibles = {c.strip().upper(): c for c in df_saldo_raw.columns}

    # Filtrar bancos del mapeo que existen en el archivo de saldo
    bancos_saldo: list[str] = []
    for _, row in tabla.iterrows():
        col_buscada = row["BANCOS_SALDOS"].strip().upper()
        col_real    = cols_saldo_disponibles.get(col_buscada)
        if col_real and col_real not in bancos_saldo:
            bancos_saldo.append(col_real)

    if not bancos_saldo:
        logger.warning("[OVERLAY] Ningún banco del mapeo encontrado en Saldos_CCOVN -- sin ajuste")
        return None

    df_saldo  = df_saldo_raw[bancos_saldo].copy()
    df_flujos = df_saldo.diff()
    calendario = pd.DatetimeIndex(df_saldo.index)

    todos_cierres = _cierres_trimestrales(calendario)
    cierres = [fc for fc in todos_cierres if fc < fecha_origen]   # todo el historial

    if len(cierres) < 2:
        logger.warning("[OVERLAY] Insuficientes cierres históricos")
        return None

    return df_saldo, df_flujos, bancos_saldo, cierres, calendario


###############################################################################
# Interfaz pública principal -- step005 llama a esta función
###############################################################################

def get_peor_total(
    fecha_origen: pd.Timestamp,
    h_max: int = 75,
    ruta_saldo: Path = RUTA_SALDO,
    ruta_tabla: Path  = RUTA_TABLA,
) -> tuple[float, pd.Timestamp | None]:
    """
    Calcula el peor retiro potencial del sistema para una fecha de origen dada.

    Retorna (peor_total, fecha_primer_cierre):
      - peor_total       : magnitud del retiro esperado (M USD).  0.0 si no aplica ajuste.
      - fecha_primer_cierre: primer cierre trimestral dentro del horizonte h_max.
                            None si no hay cierre en el horizonte o no aplica ajuste.

    El ajuste se referencia al PRIMER cierre trimestral proyectado, que es cuando
    se produce el retiro de sobreencaje.  step005 usa fecha_primer_cierre para
    localizar el h correspondiente y obtener el Q01 de referencia:

        peor_total, fecha_cierre = get_peor_total(fecha_origen)
        if peor_total > 0:
            # Encontrar el h donde cae el primer cierre
            bh = pd.bdate_range(start=fecha_origen + BDay(1), periods=h_max)
            h_cierre = int(np.where(bh == fecha_cierre)[0][0]) + 1
            idx_cierre = np.where(h_arr == h_cierre)[0]
            q01_cierre = float(preds[0.01][idx_cierre[0]]) if len(idx_cierre) else float(preds[0.01][-1])
            f = peor_total / abs(q01_cierre) if q01_cierre < 0 else 1.0
            if f > 1.0:
                preds = {tau: arr * f for tau, arr in preds.items()}  # uniforme en todos los h
    """
    if not OVERLAY_SOBREENCAJE_ACTIVO:
        return 0.0, None

    resultado = _preparar_datos(ruta_saldo, ruta_tabla, fecha_origen)
    if resultado is None:
        return 0.0, None

    df_saldo, df_flujos, bancos_saldo, cierres, calendario = resultado

    # Primer cierre trimestral dentro del horizonte de proyeccion
    # (75dh ~ 3.4 meses > 1 trimestre, asi que siempre existe al menos uno)
    bh_futuro = pd.bdate_range(
        start=fecha_origen + pd.offsets.BDay(1),
        periods=h_max,
    )
    fecha_primer_cierre = _cierres_trimestrales(pd.DatetimeIndex(bh_futuro))[0]

    # Deteccion de bancos activos
    det = detectar_bancos_activos(
        df_saldo, df_flujos, bancos_saldo, cierres, calendario,
        cierres_lookback=cierres[-N_TRIMESTRES_LOOKBACK:],
    )
    activos = [b for b, info in det.items() if info["alguna_vez_activa"]]
    if not activos:
        logger.info(f"[OVERLAY] {fecha_origen.date()}: sin bancos activos -- sin ajuste")
        return 0.0, None

    peor_total = sum(
        _peor_B(df_saldo, det, b, cierres, fecha_ref=fecha_origen)
        for b in activos
    )
    logger.info(
        f"[OVERLAY] {fecha_origen.date()} | primer cierre: {fecha_primer_cierre.date()} | "
        f"bancos={activos} | peor_total={peor_total:,.0f}"
    )
    return peor_total, fecha_primer_cierre


###############################################################################
# Función auxiliar -- formato dict {tau: array}  (uso desde step005/step006)
###############################################################################

def aplicar_overlay_preds(
    preds: dict[float, np.ndarray],
    h_arr: np.ndarray,
    fechas_t,
    ruta_saldo: Path = RUTA_SALDO,
    ruta_tabla: Path  = RUTA_TABLA,
) -> dict[float, np.ndarray]:
    """
    Aplica el overlay sobreencaje a un dict de predicciones.

    Parametros
    ----------
    preds    : {0.01: arr, 0.05: arr, 0.50: arr, 0.95: arr, 0.99: arr}
               Cada array tiene longitud N (una entrada por par fecha_t x h).
    h_arr    : horizonte en dias habiles de cada fila, shape (N,).
    fechas_t : array-like de pd.Timestamp con la fecha de origen de cada fila,
               shape (N,).  Permite multiples origenes (walk-forward CV).

    Logica por cada fecha_t unica
    ------------------------------
    1. Identificar el primer cierre trimestral dentro del horizonte de 75dh.
    2. Definir la ventana de retiro: los ultimos VENTANA_RETIRO_DH (7) dias
       habiles hasta el cierre, inclusive:
           h_inicio = h_cierre - VENTANA_RETIRO_DH + 1
           h_cierre = posicion del cierre en bdate_range(fecha_t, 75dh)
    3. Calcular peor_total (retiro acumulado potencial de 7dh).
    4. Calcular Q01_acum_7dh = suma de Q01 en esos mismos 7 horizontes.
       (Misma escala que peor_total: ambos son retiros acumulados en 7dh.)
    5. f = peor_total / |Q01_acum_7dh|   (si Q01_acum < 0 y f > 1)
    6. Aplicar f UNIFORMEMENTE a todos los horizontes de esa fecha_t.

    Cache
    -----
    La deteccion de bancos (det) se cachea por trimestre.
    El peor_total se recomputa por fecha_t (ventana P95 crece diariamente).
    """
    if not OVERLAY_SOBREENCAJE_ACTIVO:
        return preds

    fechas_t = pd.DatetimeIndex(fechas_t)
    if len(fechas_t) == 0 or 0.01 not in preds:
        return preds

    # Cargar datos una sola vez usando la fecha maxima disponible
    fecha_max = fechas_t.max()
    resultado = _preparar_datos(ruta_saldo, ruta_tabla, fecha_max)
    if resultado is None:
        return preds

    df_saldo, df_flujos, bancos_saldo, _unused, calendario = resultado
    todos_cierres = _cierres_trimestrales(calendario)

    preds_adj = {tau: arr.copy() for tau, arr in preds.items()}
    _det_cache: dict[tuple, tuple[dict, list]] = {}  # cache trimestral

    for fecha_t in sorted(set(fechas_t)):
        # Cierres historicos visibles desde fecha_t (sin leakage)
        cierres_hist = [fc for fc in todos_cierres if fc < fecha_t]
        if len(cierres_hist) < 2:
            continue

        # Primer cierre trimestral dentro del horizonte de proyeccion (75dh)
        # 75dh ~ 3.4 meses > 1 trimestre -> siempre existe al menos uno
        bh_futuro = pd.bdate_range(
            start=fecha_t + pd.offsets.BDay(1), periods=VENTANA_RETIRO_DH + 75
        )
        cierres_proy = _cierres_trimestrales(pd.DatetimeIndex(bh_futuro))
        fecha_cierre = cierres_proy[0]

        # h del cierre (1-indexed) en el bdate_range estandar
        bh_lista = list(bh_futuro)
        try:
            h_cierre = bh_lista.index(fecha_cierre) + 1
        except ValueError:
            # Mismatch por feriado: usar el dia mas cercano
            diffs = [abs((d - fecha_cierre).days) for d in bh_lista]
            h_cierre = diffs.index(min(diffs)) + 1

        # Ventana de retiro: VENTANA_RETIRO_DH dias habiles antes del cierre (inclusive)
        h_inicio = max(1, h_cierre - VENTANA_RETIRO_DH + 1)

        # Deteccion de bancos activos (cache trimestral)
        cierres_key = tuple(cierres_hist)
        if cierres_key not in _det_cache:
            lb = list(cierres_hist[-N_TRIMESTRES_LOOKBACK:])
            det = detectar_bancos_activos(
                df_saldo, df_flujos, bancos_saldo,
                list(cierres_hist), calendario,
                cierres_lookback=lb,
            )
            activos = [b for b, info in det.items() if info["alguna_vez_activa"]]
            _det_cache[cierres_key] = (det, activos)
            if activos:
                logger.info(f"[OVERLAY] {fecha_t.date()} | bancos activos: {activos}")

        det_t, activos_t = _det_cache[cierres_key]
        if not activos_t:
            continue

        # peor_total: retiro acumulado en 7dh (ventana P95 crece diariamente)
        peor_total = sum(
            _peor_B(df_saldo, det_t, b, list(cierres_hist), fecha_ref=fecha_t)
            for b in activos_t
        )
        if peor_total <= 0:
            continue

        # Filas de esta fecha_t en el array
        mask_origen = (fechas_t == fecha_t)

        # Q01 acumulado en los mismos 7dh de la ventana de retiro proyectada
        # (misma escala que peor_total: ambos son retiros acumulados en 7dh)
        mask_ventana = mask_origen & (h_arr >= h_inicio) & (h_arr <= h_cierre)
        if not mask_ventana.any():
            # Fallback: h mas cercano al cierre disponible
            h_origen = h_arr[mask_origen]
            h_fb = h_origen[np.abs(h_origen - h_cierre).argmin()]
            mask_ventana = mask_origen & (h_arr == h_fb)

        q01_acum_7dh = float(preds[0.01][mask_ventana].sum())
        f = _factor_overlay(peor_total, q01_acum_7dh)
        if f <= 1.0:
            continue

        # Aplicar f UNIFORMEMENTE a TODOS los horizontes de esta fecha_t
        for tau in preds_adj:
            preds_adj[tau][mask_origen] *= f

        logger.info(
            f"[OVERLAY] {fecha_t.date()} | cierre: {fecha_cierre.date()} "
            f"h=[{h_inicio},{h_cierre}] | "
            f"Q01_acum_7dh={q01_acum_7dh:+.0f} | peor_total={peor_total:,.0f} | f={f:.3f}"
        )

    return preds_adj


###############################################################################
# Variante DataFrame -- columnas q01..q99  (uso desde step006 / postproceso)
###############################################################################

def aplicar_overlay_df(
    df: pd.DataFrame,
    fecha_origen: pd.Timestamp | None = None,
    ruta_saldo: Path = RUTA_SALDO,
    ruta_tabla: Path  = RUTA_TABLA,
) -> pd.DataFrame:
    """
    Variante para DataFrames con columnas q01..q99, columna h y (preferiblemente)
    columna fecha_t.

    Si el DataFrame tiene columna 'fecha_t', se usa directamente para el walk-forward
    (multiples origenes).  Si no, se usa fecha_origen como origen unico de respaldo.
    """
    q_cols_presentes = [c for c in Q_COLS if c in df.columns]
    if not q_cols_presentes or "h" not in df.columns:
        logger.warning("[OVERLAY] Columnas q0x/h no encontradas -- sin ajuste")
        return df

    if "fecha_t" in df.columns:
        fechas_t = pd.DatetimeIndex(df["fecha_t"])
    elif fecha_origen is not None:
        fechas_t = pd.DatetimeIndex([fecha_origen] * len(df))
    else:
        logger.warning("[OVERLAY] Sin columna fecha_t ni fecha_origen -- sin ajuste")
        return df

    preds = {float(c[1:]) / 100.0: df[c].values.copy() for c in q_cols_presentes}
    h_arr = df["h"].values

    preds_adj = aplicar_overlay_preds(preds, h_arr, fechas_t, ruta_saldo, ruta_tabla)

    df_adj = df.copy()
    for c in q_cols_presentes:
        tau = float(c[1:]) / 100.0
        if tau in preds_adj:
            df_adj[c] = preds_adj[tau]
    return df_adj


###############################################################################
# Diagnóstico standalone -- outliers + activación + exporta a Excel
###############################################################################

# Ruta de salida del diagnóstico Excel
DIR_OUTPUT_DIAG = BASE_SISTEMA / "2. Output" / "analisis_cc"


def _diagnostico(
    ruta_saldo: Path = RUTA_SALDO,
    ruta_tabla: Path  = RUTA_TABLA,
    fecha_ref: pd.Timestamp | None = None,
    dir_output: Path = DIR_OUTPUT_DIAG,
) -> None:
    """
    Genera el diagnóstico del overlay y lo exporta a Excel con 4 hojas:

      1. Cruce        -- mapeo BANCOS_RETIROS <-> BANCOS_SALDOS con estado OK/AUSENTE
      2. Outliers     -- fechas y valores atípicos por banco (zscore > 4sigma o IQR > 6×IQR)
      3. Ratios       -- ratio |retiro_7dh|/saldo_max por banco y cierre trimestral
      4. Resumen      -- bancos activos, peor_B, peor_total y parámetros del overlay

    El archivo se guarda en dir_output/diag_overlay_sobreencaje.xlsx.
    """
    if fecha_ref is None:
        fecha_ref = pd.Timestamp.today().normalize()

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Overlay Sobreencaje -- Diagnóstico  ({fecha_ref.date()})")
    print(sep)

    # -- 1. Mapeo --------------------------------------------------------------
    try:
        tabla_mapeo = cargar_tabla_mapeo(ruta_tabla)
    except Exception as e:
        print(f"\n  ERROR: No se pudo leer TablaSaldosRetiros.xlsx: {e}")
        return

    # -- 2. Cargar saldos ------------------------------------------------------
    try:
        df_saldo_raw = _cargar_saldos(ruta_saldo)
    except Exception as e:
        print(f"\n  ERROR: No se pudo leer Saldos_CCOVN.xlsx: {e}")
        return

    print(f"\n  Saldos CC+OVN: {len(df_saldo_raw):,} filas | "
          f"{df_saldo_raw.index.min().date()} -> {df_saldo_raw.index.max().date()}")

    # -- 3. Cruce mapeo <-> saldo ------------------------------------------------
    cols_saldo_upper = {c.strip().upper(): c for c in df_saldo_raw.columns}
    bancos_saldo: list[str] = []
    filas_cruce: list[dict] = []

    for _, row in tabla_mapeo.iterrows():
        nombre_ret = row["BANCOS_RETIROS"]
        nombre_sal = row["BANCOS_SALDOS"]
        col_real   = cols_saldo_upper.get(nombre_sal.strip().upper())
        if col_real:
            if col_real not in bancos_saldo:
                bancos_saldo.append(col_real)
            estado = "OK"
        else:
            estado = "AUSENTE EN SALDO"
        filas_cruce.append({
            "BANCOS_RETIROS": nombre_ret,
            "BANCOS_SALDOS":  nombre_sal,
            "Col_en_archivo": col_real or "",
            "Estado":         estado,
        })

    df_cruce = pd.DataFrame(filas_cruce)
    print(f"\n  Cruce mapeo <-> saldo:")
    for _, r in df_cruce.iterrows():
        print(f"    {r['BANCOS_RETIROS']:<20} -> {r['BANCOS_SALDOS']:<30}  [{r['Estado']}]")

    if not bancos_saldo:
        print("\n  Sin bancos en intersección -- revisar nombres en TablaSaldosRetiros.xlsx")
        return

    df_saldo  = df_saldo_raw[bancos_saldo].copy()
    df_flujos = df_saldo.diff()
    calendario = pd.DatetimeIndex(df_saldo.index)

    # -- 4. Outliers por banco -------------------------------------------------
    print(f"\n{'-'*72}")
    print(f"  OUTLIERS DE SALDO (zscore > {OUTLIER_ZSCORE}sigma  |  IQR > {OUTLIER_IQR}×IQR)")
    print(f"{'-'*72}")

    filas_outliers: list[dict] = []
    for banco in bancos_saldo:
        outs = detectar_outliers_banco(df_saldo[banco])
        if outs.empty:
            print(f"  {banco:<35}  sin outliers")
        else:
            print(f"  {banco:<35}  {len(outs)} outlier(s):")
            for _, r in outs.iterrows():
                metodo = ("zscore+IQR" if r["outlier_zscore"] and r["outlier_iqr"]
                          else "zscore" if r["outlier_zscore"] else "IQR")
                print(f"    {str(r['fecha'].date()):<12}  valor={r['valor']:>12,.1f}  "
                      f"z={r['z_score']:+.1f}sigma  [{metodo}]")
                filas_outliers.append({
                    "banco":          banco,
                    "fecha":          r["fecha"],
                    "valor":          r["valor"],
                    "z_score":        round(r["z_score"], 2),
                    "outlier_zscore": r["outlier_zscore"],
                    "outlier_iqr":    r["outlier_iqr"],
                    "metodo":         metodo,
                })

    df_outliers = pd.DataFrame(filas_outliers) if filas_outliers else pd.DataFrame(
        columns=["banco", "fecha", "valor", "z_score", "outlier_zscore", "outlier_iqr", "metodo"]
    )
    print(f"\n  Total outliers: {len(df_outliers)}")

    # -- 5. Ratios históricos --------------------------------------------------
    todos_cierres  = _cierres_trimestrales(calendario)
    # Para el diagnóstico usar TODOS los cierres históricos (no solo los últimos N)
    fecha_max_data = df_saldo.index.max()
    cierres_hist   = [fc for fc in todos_cierres if fc <= min(fecha_ref, fecha_max_data)]
    cierres_n     = cierres_hist[-N_TRIMESTRES_LOOKBACK:]   # últimos N para el overlay

    det = detectar_bancos_activos(df_saldo, df_flujos, bancos_saldo, cierres_hist, calendario,
                                  cierres_lookback=cierres_n)

    # Tabla pivot: banco × cierre -> ratio
    filas_ratios: list[dict] = []
    for banco, info in sorted(det.items()):
        fila: dict = {"banco": banco}
        for fc in cierres_hist:
            col_name = f"{fc.year}-Q{fc.quarter}"
            fila[col_name] = info["ratios"].get(fc)
        fila["activa_ultimo_trim"]  = info["activa"]
        fila["alguna_vez_activa"]   = info["alguna_vez_activa"]
        filas_ratios.append(fila)
    df_ratios = pd.DataFrame(filas_ratios)

    # Consola: solo últimos N cierres
    print(f"\n{'-'*72}")
    print(f"  RATIOS |retiro_7dh|/saldo_max -- últimos {N_TRIMESTRES_LOOKBACK} cierres")
    print(f"{'-'*72}")
    header = f"  {'Banco':<35} {'Estado':>12}   " + "   ".join(
        f"{fc.year}-Q{fc.quarter}" for fc in cierres_n
    )
    print(header)
    print(f"  {'-'*35} {'-'*12}   {'-'*40}")
    for banco, info in sorted(det.items()):
        estado = ("ACTIVO  [OK]" if info["activa"]
                  else "hist [OK]   " if info["alguna_vez_activa"]
                  else "inactivo")
        ratios_str = "   ".join(
            f"{info['ratios'].get(fc, float('nan')):.0%}" if fc in info["ratios"] else "  N/D "
            for fc in cierres_n
        )
        print(f"  {banco:<35} {estado:>12}   {ratios_str}")

    # -- 6. Resumen peor_B -----------------------------------------------------
    bancos_activos = [b for b, info in det.items() if info["alguna_vez_activa"]]
    peor_total = sum(
        _peor_B(df_saldo, det, b, cierres_hist, fecha_ref=fecha_ref)
        for b in bancos_activos
    )

    # P95 del saldo: trimestre anterior + días corridos del actual hasta fecha_ref (exclusive)
    pq_start_diag = _prev_quarter_start(fecha_ref)

    filas_resumen: list[dict] = []
    for b in sorted(det.keys()):
        info = det[b]
        ratios_act = {fc: r for fc, r in info["ratios"].items() if r > UMBRAL_ACTIVACION}
        worst_r = max(ratios_act.values()) if ratios_act else None
        mask_q = (df_saldo.index >= pq_start_diag) & (df_saldo.index < fecha_ref)
        s_q    = df_saldo.loc[mask_q, b].dropna()
        p95_sal = float(s_q.quantile(PERCENTIL_SALDO)) if not s_q.empty and float(s_q.max()) > 0 else None
        pb = (
            _peor_B(df_saldo, det, b, cierres_hist, fecha_ref=fecha_ref)
            if info["alguna_vez_activa"] else 0.0
        )
        filas_resumen.append({
            "banco":                 b,
            "activa_ultimo_trim":    info["activa"],
            "alguna_vez_activa":     info["alguna_vez_activa"],
            "n_trimestres_activos":  info["n_trimestres_activos"],
            "worst_ratio":           round(worst_r, 4) if worst_r is not None else None,
            "p95_saldo_trim_actual": round(p95_sal, 1) if p95_sal is not None else None,
            "factor_seg":            FACTOR_SEGURIDAD,
            "peor_B":                round(pb, 1),
        })

    # Fila total
    filas_resumen.append({
        "banco":                   "TOTAL (bancos alguna_vez_activa)",
        "activa_ultimo_trim":      None,
        "alguna_vez_activa":       True,
        "worst_ratio":             None,
        "p95_saldo_trim_actual":   None,
        "factor_seg":              None,
        "peor_B":              round(peor_total, 1),
    })
    df_resumen = pd.DataFrame(filas_resumen)

    # Hoja de parámetros
    df_params = pd.DataFrame([
        {"parametro": "fecha_ref",                  "valor": str(fecha_ref.date())},
        {"parametro": "OVERLAY_SOBREENCAJE_ACTIVO", "valor": str(OVERLAY_SOBREENCAJE_ACTIVO)},
        {"parametro": "UMBRAL_ACTIVACION",          "valor": f"{UMBRAL_ACTIVACION:.0%}"},
        {"parametro": "N_TRIMESTRES_LOOKBACK",      "valor": str(N_TRIMESTRES_LOOKBACK)},
        {"parametro": "MIN_TRIMESTRES_ACTIVO",      "valor": str(MIN_TRIMESTRES_ACTIVO)},
        {"parametro": "VENTANA_RETIRO_DH",          "valor": str(VENTANA_RETIRO_DH)},
        {"parametro": "FACTOR_SEGURIDAD",           "valor": f"{FACTOR_SEGURIDAD:.0%}"},
        {"parametro": "OUTLIER_ZSCORE",             "valor": str(OUTLIER_ZSCORE)},
        {"parametro": "OUTLIER_IQR",                "valor": str(OUTLIER_IQR)},
        {"parametro": "peor_total",                 "valor": f"{peor_total:,.1f}"},
        {"parametro": "bancos_activos",             "valor": ", ".join(bancos_activos) or "ninguno"},
    ])

    # Consola: resumen final
    print(f"\n{'-'*72}")
    print(f"  RESUMEN -- peor retiro potencial por banco activo")
    print(f"{'-'*72}")
    for b in bancos_activos:
        pb = _peor_B(df_saldo, det, b, cierres_hist, fecha_ref=fecha_ref)
        print(f"  {b:<35}  {pb:>14,.1f}")
    if bancos_activos:
        print(f"  {'-'*35}  {'-'*14}")
        print(f"  {'TOTAL':35}  {peor_total:>14,.1f}")

    # -- 7. Exportar a Excel ---------------------------------------------------
    dir_output.mkdir(parents=True, exist_ok=True)
    ruta_out = dir_output / "diag_overlay_sobreencaje.xlsx"

    with pd.ExcelWriter(ruta_out, engine="openpyxl") as writer:
        df_cruce.to_excel(writer, sheet_name="1_Cruce", index=False)
        df_outliers.to_excel(writer, sheet_name="2_Outliers", index=False)
        df_ratios.to_excel(writer, sheet_name="3_Ratios", index=False)
        df_resumen.to_excel(writer, sheet_name="4_Resumen", index=False)
        df_params.to_excel(writer, sheet_name="5_Parametros", index=False)

        # Hoja de saldos completos (una columna por banco, index=fecha)
        df_saldo.reset_index().rename(columns={"fecha": "fecha"}).to_excel(
            writer, sheet_name="Saldos_raw", index=False
        )

    print(f"\n  Excel exportado: {ruta_out}")
    print(f"\n{sep}\n")


###############################################################################
# Exportación de saldos y retiros raw por banco + señal trimestral
###############################################################################

def exportar_saldos_retiros(
    ruta_saldo: Path = RUTA_SALDO,
    ruta_tabla: Path  = RUTA_TABLA,
    fecha_ref: pd.Timestamp | None = None,
    dir_output: Path = DIR_OUTPUT_DIAG,
) -> None:
    """
    Genera saldos_retiros_bancos.xlsx con:
      - Una pestaña por banco: fecha | saldo | retiro_diario | es_cierre_trim
      - Última pestaña "Señal": banco × cierre trimestral -> ratio (%) y activación

    La columna es_cierre_trim marca con 1 el último día hábil de cada trimestre,
    facilitando identificar visualmente los eventos de sobreencaje.
    """
    if fecha_ref is None:
        fecha_ref = pd.Timestamp.today().normalize()

    # -- Cargar datos ----------------------------------------------------------
    try:
        tabla_mapeo = cargar_tabla_mapeo(ruta_tabla)
    except Exception as e:
        print(f"ERROR: No se pudo leer TablaSaldosRetiros.xlsx: {e}")
        return

    try:
        df_saldo_raw = _cargar_saldos(ruta_saldo)
    except Exception as e:
        print(f"ERROR: No se pudo leer Saldos_CCOVN.xlsx: {e}")
        return

    cols_saldo_upper = {c.strip().upper(): c for c in df_saldo_raw.columns}

    # Consolidar por banco_saldos: múltiples nombres en retiros (CONTINEN+BBVA,
    # FINANCIERO+PICHINCHA) se agrupan en una sola entrada con nombres combinados.
    # bancos_consolidados: {col_saldo: [nombre_ret1, nombre_ret2, ...]}
    from collections import defaultdict
    bancos_consolidados: dict[str, list[str]] = defaultdict(list)
    for _, row in tabla_mapeo.iterrows():
        nombre_ret = row["BANCOS_RETIROS"]
        nombre_sal = row["BANCOS_SALDOS"]
        col_real   = cols_saldo_upper.get(nombre_sal.strip().upper())
        if col_real:
            bancos_consolidados[col_real].append(nombre_ret)

    if not bancos_consolidados:
        print("ERROR: Sin bancos en intersección -- revisar TablaSaldosRetiros.xlsx")
        return

    # bancos_info: una entrada por banco_saldos único, nombre_ret combinado con "/"
    bancos_info: list[dict] = [
        {
            "retiros": " / ".join(nombres),   # ej. "CONTINEN / BBVA"
            "retiros_actual": nombres[-1],     # último nombre = nombre más reciente
            "saldos": col_sal,
        }
        for col_sal, nombres in bancos_consolidados.items()
    ]

    # Columnas únicas (ya garantizado por la consolidación)
    bancos_saldo_unique = [b["saldos"] for b in bancos_info]
    df_saldo  = df_saldo_raw[bancos_saldo_unique].copy()
    df_flujos = df_saldo.diff()
    calendario = pd.DatetimeIndex(df_saldo.index)

    # Cierres trimestrales históricos (todos, no solo los últimos N)
    todos_cierres  = _cierres_trimestrales(calendario)
    fecha_max_data = df_saldo.index.max()
    cierres_hist   = [fc for fc in todos_cierres if fc <= min(fecha_ref, fecha_max_data)]
    cierres_n      = cierres_hist[-N_TRIMESTRES_LOOKBACK:]

    # Marca de cierre trimestral en el calendario
    set_cierres = set(cierres_hist)

    # Ratios históricos para todos los cierres (usar lista única)
    det = detectar_bancos_activos(
        df_saldo, df_flujos, bancos_saldo_unique, cierres_hist, calendario,
        cierres_lookback=cierres_n,
    )

    # -- Exportar --------------------------------------------------------------
    dir_output.mkdir(parents=True, exist_ok=True)
    ruta_out = dir_output / "saldos_retiros_bancos.xlsx"

    print(f"\nExportando {ruta_out} ...")

    with pd.ExcelWriter(ruta_out, engine="openpyxl") as writer:

        # -- Pestaña por banco -------------------------------------------------
        for info in bancos_info:
            col_sal  = info["saldos"]
            nom_ret  = info["retiros"]         # "CONTINEN / BBVA" (nombres históricos)
            nom_act  = info["retiros_actual"]  # "BBVA" (nombre más reciente)

            # Nombre de hoja: nombre actual (más reciente), máx 31 chars
            sheet = nom_act[:31]

            df_banco = pd.DataFrame({
                "fecha":          df_saldo.index,
                "saldo":          df_saldo[col_sal].values,
                "retiro_diario":  df_flujos[col_sal].values,
                "es_cierre_trim": [1 if d in set_cierres else 0
                                   for d in df_saldo.index],
            })

            # Calcular retiro acumulado en ventana de VENTANA_RETIRO_DH dh
            # para cada cierre trimestral (columna auxiliar de referencia)
            retiro_cierre: list[float | None] = []
            for fecha in df_saldo.index:
                if fecha in set_cierres:
                    ventana = _ventana_antes_cierre(fecha, calendario, VENTANA_RETIRO_DH)
                    disp    = df_flujos.index.intersection(ventana)
                    val     = float(df_flujos.loc[disp, col_sal].sum()) if not disp.empty else None
                    retiro_cierre.append(val)
                else:
                    retiro_cierre.append(None)

            df_banco["retiro_acum_cierre"] = retiro_cierre

            df_banco.to_excel(writer, sheet_name=sheet, index=False)
            n_saldo_ok  = df_banco["saldo"].notna().sum()
            n_total     = len(df_banco)
            pct_ok      = n_saldo_ok / n_total * 100 if n_total else 0
            aviso = "  *** SIN DATOS ***" if n_saldo_ok == 0 else (
                f"  *** DATOS ESCASOS ({pct_ok:.0f}%) ***" if pct_ok < 20 else ""
            )
            print(f"  + {sheet:<25}  saldo válido: {n_saldo_ok:,}/{n_total:,} ({pct_ok:.0f}%){aviso}")

        # -- Última pestaña: Señal ---------------------------------------------
        # Filas: bancos | Columnas: cierre trimestral -> ratio y activación
        filas_senal: list[dict] = []
        for info in bancos_info:
            col_sal  = info["saldos"]
            nom_ret  = info["retiros"]
            ratios   = det.get(col_sal, {}).get("ratios", {})

            fila: dict = {
                "banco_retiros": nom_ret,    # nombres históricos combinados
                "banco_actual":  nom_act,    # nombre más reciente
                "banco_saldos":  col_sal,
            }
            for fc in cierres_hist:
                etiqueta = f"{fc.year}-Q{fc.quarter}"
                ratio    = ratios.get(fc)
                fila[f"ratio_{etiqueta}"]  = round(ratio, 4) if ratio is not None else None
                fila[f"señal_{etiqueta}"]  = (1 if ratio is not None and ratio > UMBRAL_ACTIVACION
                                               else 0 if ratio is not None else None)
            filas_senal.append(fila)

        df_senal = pd.DataFrame(filas_senal)

        # Reordenar: primero columnas de identificación, luego pares ratio/señal por cierre
        cols_id   = ["banco_retiros", "banco_actual", "banco_saldos"]
        cols_trim = [c for c in df_senal.columns if c not in cols_id]
        df_senal  = df_senal[cols_id + sorted(cols_trim)]

        df_senal.to_excel(writer, sheet_name="Señal_trimestral", index=False)
        print(f"  + Señal_trimestral  ({len(cierres_hist)} cierres históricos)")

        # -- Pestaña: Ajuste_diario --------------------------------------------
        # worst_ratio se cachea por trimestre (cambia cuando hay nuevo cierre).
        # p95_saldo usa el trimestre actual hasta t -> se recalcula cada día.
        filas_ajuste: list[dict] = []
        prev_cierres_key: tuple = ()
        cache_det_t: dict | None = None
        cache_activos: list[str] = []

        for t in calendario:
            cierres_t = tuple(fc for fc in todos_cierres if fc < t)

            if cierres_t != prev_cierres_key:
                prev_cierres_key = cierres_t
                if len(cierres_t) < 2:
                    cache_det_t  = None
                    cache_activos = []
                else:
                    lb_t = list(cierres_t[-N_TRIMESTRES_LOOKBACK:])
                    cache_det_t = detectar_bancos_activos(
                        df_saldo, df_flujos, bancos_saldo_unique,
                        list(cierres_t), calendario,
                        cierres_lookback=lb_t,
                    )
                    cache_activos = [
                        b for b, info in cache_det_t.items() if info["alguna_vez_activa"]
                    ]

            if cache_det_t is None:
                fila: dict = {
                    "fecha": t, "n_bancos_activos": 0,
                    "bancos_activos": "", "peor_total": float("nan"),
                }
                for b in bancos_saldo_unique:
                    fila[f"peor_B_{b}"] = float("nan")
            else:
                # peor_B usa p95 del trim. anterior + corrido actual hasta t (recálculo diario)
                peor_by_banco = {
                    b: _peor_B(df_saldo, cache_det_t, b, list(cierres_t), fecha_ref=t)
                    for b in bancos_saldo_unique
                }
                peor_total_t = sum(peor_by_banco.get(b, 0.0) for b in cache_activos)
                fila = {
                    "fecha":            t,
                    "n_bancos_activos": len(cache_activos),
                    "bancos_activos":   " / ".join(cache_activos),
                    "peor_total":       peor_total_t,
                }
                for b in bancos_saldo_unique:
                    fila[f"peor_B_{b}"] = (
                        peor_by_banco[b] if b in cache_activos else float("nan")
                    )

            # Mostrar solo los últimos N_TRIMESTRES_LOOKBACK para legibilidad
            ultimos = prev_cierres_key[-N_TRIMESTRES_LOOKBACK:]
            fila["cierres_referencia"] = (
                " | ".join(f"{fc.year}-Q{fc.quarter}" for fc in ultimos)
                if ultimos else ""
            )
            filas_ajuste.append(fila)

        df_ajuste = pd.DataFrame(filas_ajuste)
        cols_meta  = ["fecha", "n_bancos_activos", "bancos_activos", "peor_total"]
        cols_banco = [c for c in df_ajuste.columns if c.startswith("peor_B_")]
        cols_ref   = ["cierres_referencia"]
        df_ajuste  = df_ajuste[cols_meta + sorted(cols_banco) + cols_ref]

        df_ajuste.to_excel(writer, sheet_name="Ajuste_diario", index=False)
        print(f"  + Ajuste_diario     ({len(df_ajuste):,} filas -- peor_total diario)")

        # -- Pestaña: P95_saldo_diario -----------------------------------------
        # Para cada día hábil: P95 del saldo por banco y por trimestre de
        # referencia, más el p95_max usado en peor_B.  Solo cambia al cruzar
        # un cierre trimestral -> mismo caché que Ajuste_diario.

        # P95_saldo_diario: p95 del trimestre actual hasta t -- se recalcula cada día
        # porque el trimestre actual crece a diario (igual que en _peor_B).
        filas_p95: list[dict] = []

        for t in calendario:
            pq_start_t = _prev_quarter_start(t)
            ventana    = (f"{pq_start_t.year}-Q{pq_start_t.quarter} -> "
                          f"{t.year}-Q{t.quarter} (excl. {t.date()})")
            fila_p95   = {"fecha": t, "ventana_saldo": ventana}
            for b in bancos_saldo_unique:
                mask = (df_saldo.index >= pq_start_t) & (df_saldo.index < t)
                s    = df_saldo.loc[mask, b].dropna()
                fila_p95[f"p95_{b}"] = (
                    float(s.quantile(PERCENTIL_SALDO))
                    if not s.empty and float(s.max()) > 0
                    else float("nan")
                )
            filas_p95.append(fila_p95)

        df_p95   = pd.DataFrame(filas_p95)
        cols_hdr = ["fecha", "ventana_saldo"]
        cols_det = sorted(c for c in df_p95.columns if c not in cols_hdr)
        df_p95   = df_p95[cols_hdr + cols_det]

        df_p95.to_excel(writer, sheet_name="P95_saldo_diario", index=False)
        print(f"  + P95_saldo_diario  ({len(df_p95):,} filas -- P95 trim. actual por banco)")

    print(f"\nArchivo generado: {ruta_out}\n")


###############################################################################
# Visualización del ajuste overlay
###############################################################################

# Paleta categórica fija (CVD-safe, slot order from dataviz skill)
_COLORES_BANCOS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]


def graficar_ajuste_overlay(
    dir_output: Path = DIR_OUTPUT_DIAG,
    ruta_excel: Path | None = None,
) -> None:
    """
    Lee Ajuste_diario de saldos_retiros_bancos.xlsx y genera dos paneles:
      1. peor_total diario -- magnitud del ajuste a lo largo del tiempo
      2. Stacked area por banco -- qué bancos impulsan el ajuste

    Exporta el gráfico a DIR_OUTPUT_DIAG / "overlay_ajuste.png".
    """
    if ruta_excel is None:
        ruta_excel = dir_output / "saldos_retiros_bancos.xlsx"

    if not ruta_excel.exists():
        print(f"  No se encontró {ruta_excel} -- ejecuta exportar_saldos_retiros() primero")
        return

    print("Leyendo Ajuste_diario...")
    df = pd.read_excel(ruta_excel, sheet_name="Ajuste_diario", parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Columnas por banco
    cols_banco = sorted(c for c in df.columns if c.startswith("peor_B_"))
    nombres    = [c.replace("peor_B_", "") for c in cols_banco]

    # Solo bancos con algún valor no-NaN (bancos que fueron activos en algún momento)
    cols_activos = [c for c in cols_banco if df[c].notna().any()]
    nombres_act  = [c.replace("peor_B_", "") for c in cols_activos]

    # Rellenar NaN con 0 para el área apilada
    df_stack = df[cols_activos].fillna(0)

    # Detectar cierres trimestrales (cambio en peor_total -- proxy de cierre)
    todos_cierres_vis = _cierres_trimestrales(pd.DatetimeIndex(df["fecha"]))

    # -- Layout ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.8]},
    )
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", labelsize=9)
        ax.yaxis.set_tick_params(length=0)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6, zorder=0)

    # -- Panel 1: peor_total ---------------------------------------------------
    mask_valid = df["peor_total"].notna() & (df["peor_total"] > 0)
    ax1.plot(
        df.loc[mask_valid, "fecha"],
        df.loc[mask_valid, "peor_total"] / 1e6,
        color="#2a78d6", linewidth=1.6, zorder=3,
    )
    ax1.fill_between(
        df.loc[mask_valid, "fecha"],
        df.loc[mask_valid, "peor_total"] / 1e6,
        alpha=0.12, color="#2a78d6", zorder=2,
    )
    ax1.set_ylabel("peor_total (M USD)", fontsize=9, color="#52514e")
    ax1.set_title("Magnitud del ajuste overlay sobreencaje", fontsize=11,
                  fontweight="bold", color="#0b0b0b", pad=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}M"))

    # Valor actual anotado
    ultimo_val = df.loc[mask_valid, "peor_total"].iloc[-1] if mask_valid.any() else None
    if ultimo_val is not None:
        ultima_fecha = df.loc[mask_valid, "fecha"].iloc[-1]
        ax1.annotate(
            f"{ultimo_val/1e6:,.1f}M",
            xy=(ultima_fecha, ultimo_val / 1e6),
            xytext=(8, 4), textcoords="offset points",
            fontsize=8.5, color="#2a78d6", fontweight="bold",
        )

    # -- Panel 2: stacked area por banco --------------------------------------
    fechas   = df["fecha"].values
    baseline = np.zeros(len(df))

    for i, (col, nombre) in enumerate(zip(cols_activos, nombres_act)):
        vals   = df_stack[col].values / 1e6
        color  = _COLORES_BANCOS[i % len(_COLORES_BANCOS)]
        top    = baseline + vals
        ax2.fill_between(fechas, baseline, top,
                         color=color, alpha=0.85, linewidth=0, label=nombre, zorder=2 + i)
        ax2.plot(fechas, top, color=color, linewidth=0.4, alpha=0.5, zorder=10 + i)
        baseline = top

    ax2.set_ylabel("peor_B por banco (M USD)", fontsize=9, color="#52514e")
    ax2.set_title("Contribución por banco al ajuste", fontsize=10,
                  color="#52514e", pad=6)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}M"))

    legend = ax2.legend(
        fontsize=8, ncol=min(len(nombres_act), 5),
        loc="upper left", framealpha=0.9,
        edgecolor="#e1e0d9", labelcolor="#0b0b0b",
    )
    legend.get_frame().set_linewidth(0.6)

    # -- Marcadores de cierre trimestral en ambos paneles ---------------------
    for fc in todos_cierres_vis:
        for ax in (ax1, ax2):
            ax.axvline(fc, color="#c3c2b7", linewidth=0.7,
                       linestyle="--", zorder=1, alpha=0.7)
        # Etiquetar cierres en el panel superior (solo Q4 para no saturar)
        if fc.month == 12:
            ax1.text(fc, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 1.0 else 0,
                     str(fc.year), fontsize=7, color="#898781",
                     ha="center", va="bottom")

    # -- Eje X -----------------------------------------------------------------
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax2.xaxis.get_majorticklabels(), color="#52514e", fontsize=9)

    fig.text(0.5, 0.01,
             "Líneas punteadas = cierres trimestrales  |  "
             f"UMBRAL_ACTIVACION={UMBRAL_ACTIVACION:.0%}  |  "
             f"N_TRIMESTRES_LOOKBACK={N_TRIMESTRES_LOOKBACK}  |  "
             f"VENTANA={VENTANA_RETIRO_DH}dh",
             ha="center", fontsize=7.5, color="#898781")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    ruta_png = dir_output / "overlay_ajuste.png"
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close()
    print(f"  Gráfico guardado: {ruta_png}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _diagnostico()
    exportar_saldos_retiros()
    graficar_ajuste_overlay()
