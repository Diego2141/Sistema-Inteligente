# -*- coding: utf-8 -*-
"""
step007_overlay_sobreencaje.py
==============================
Overlay comportamental para retiros de sobreencaje en cierres trimestrales.

Problema que resuelve
----------------------
El modelo cuantílico predice el flujo neto del sistema bancario para los
siguientes 75 días hábiles.  En cierres trimestrales (mar/jun/sep/dic) ciertos
bancos — en particular BBVA — ejecutan retiros masivos de encaje + overnight que
exceden el percentil 1% pronosticado.  Este overlay ajusta esa asimetría de
forma transparente y auditable, sin re-entrenar el modelo.

Mapeo retiros ↔ saldos
-----------------------
Los nombres de bancos en Transacciones_BancaLocal (columna Broker) difieren de
los nombres en el archivo de saldo CC+OVN.  TablaSaldosRetiros.xlsx contiene el
mapeo explícito entre ambas fuentes, incluyendo cambios históricos de nombre
(CONTINEN → BBVA, FINANCIERO → BANCO PICHINCHA).
Bancos con BANCOS_SALDOS = "NA" (BONY, FEDERAL) son excluidos del overlay.

Lógica en 3 pasos
------------------
Paso 1 — Detección
    Para cada banco B y cada uno de los últimos N cierres trimestrales:
        ratio[B,q] = |retiro_7dh[B,q]| / saldo_max_mes[B,q]
    El banco activa la estrategia si ratio[B, último trimestre] > UMBRAL_ACTIVACION.

Paso 2 — Peor retiro potencial por banco
    Solo para bancos activos:
        worst_ratio_B = max(ratio[B,q]) sobre cierres donde estrategia estuvo activa
        max_saldo_B   = max(saldo_max_mes[B,q]) sobre los últimos N cierres (sin filtro)
        peor_B        = worst_ratio_B × max_saldo_B × (1 + FACTOR_SEGURIDAD)

Paso 3 — Factor multiplicativo y ajuste
    peor_total = Σ peor_B   (bancos activos)
    f          = peor_total / |Q01_h_cierre|
    Si f > 1 → Q_adj[τ, h_cierre] = Q[τ, h_cierre] × f
    Aplica de forma simétrica a todos los cuantiles y la mediana.
    Solo afecta horizontes h que correspondan a un cierre trimestral.

Uso
---
    from step007_overlay_sobreencaje import aplicar_overlay_preds, aplicar_overlay_df

Diagnóstico / outliers standalone
----------------------------------
    python step007_overlay_sobreencaje.py
    → carga Saldos_CCOVN, detecta outliers por banco y muestra bancos activos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

###############################################################################
# Configuración
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")

# ── Master switch ─────────────────────────────────────────────────────────────
OVERLAY_SOBREENCAJE_ACTIVO = False

# ── Parámetros del overlay ────────────────────────────────────────────────────
UMBRAL_ACTIVACION     = 0.50   # ratio |retiro_7dh| / saldo_max_mes para activar
N_TRIMESTRES_LOOKBACK = 4      # cierres trimestrales históricos a revisar
VENTANA_RETIRO_DH     = 7      # días hábiles de la ventana antes del cierre
FACTOR_SEGURIDAD      = 0.00   # margen extra sobre peor_B  (0.00→0.10 = 0%→10%)

# ── Detección de outliers (diagnóstico) ──────────────────────────────────────
OUTLIER_ZSCORE = 4.0   # |z| > 4 σ → sospechoso (muy conservador para datos de saldo)
OUTLIER_IQR    = 6.0   # valor > mediana ± 6×IQR → outlier severo

# ── Rutas de datos ────────────────────────────────────────────────────────────
# Tabla de mapeo BANCOS_RETIROS ↔ BANCOS_SALDOS
RUTA_TABLA = Path(__file__).parent / "TablaSaldosRetiros.xlsx"

# Saldo fin del día (CC + Overnight) por banco — formato wide:
#   col 0 = fecha | col 1..N = bancos (nombre = BANCOS_SALDOS)
RUTA_SALDO = BASE_SISTEMA / "1. Data" / "Raw" / "Saldos_CCOVN.xlsx"

# Alternativa si el archivo tiene nombre diferente:
# RUTA_SALDO = BASE_SISTEMA / "1. Data" / "Raw" / "Saldo fin del dia CC+OVN.xlsx"

# Transacciones bancarias — formato transaccional:
#   columnas: Broker (= BANCOS_RETIROS), Fecha Valor, Delivery Principal Usd
RUTA_TRANS = BASE_SISTEMA / "1. Data" / "Raw" / "Transacciones_BancaLocal.xlsx"

# ── Cuantiles ─────────────────────────────────────────────────────────────────
QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]
Q_COLS    = [f"q{int(t * 100):02d}" for t in QUANTILES]   # q01, q05, q50, q95, q99


###############################################################################
# Carga del mapeo BANCOS_RETIROS ↔ BANCOS_SALDOS
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
    Retorna DataFrame con index=fecha, columnas=bancos.
    """
    raw = pd.read_excel(ruta, header=0)
    col_fecha   = raw.columns[0]
    cols_bancos = raw.columns[1:].tolist()

    raw[col_fecha] = pd.to_datetime(raw[col_fecha], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=[col_fecha]).sort_values(col_fecha).set_index(col_fecha)
    raw.index.name = "fecha"

    for c in cols_bancos:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Normalizar nombres: strip espacios
    raw.columns = [str(c).strip() for c in raw.columns]
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
      - IQR: valor fuera de [Q25 − iqr_thresh×IQR, Q75 + iqr_thresh×IQR]

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


def _ventana_antes_cierre(
    fecha_cierre: pd.Timestamp,
    calendario: pd.DatetimeIndex,
    n: int,
) -> pd.DatetimeIndex:
    """Últimos n días hábiles del calendario terminando en fecha_cierre."""
    pos = int(calendario.searchsorted(fecha_cierre, side="right"))
    return calendario[max(0, pos - n): pos]


###############################################################################
# Paso 1 — Detección por banco
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
    ratio = |retiro acumulado en ventana n_dh dh| / saldo_max del mes del cierre.
    """
    mask_mes = (
        (df_saldo.index.year  == fecha_cierre.year) &
        (df_saldo.index.month == fecha_cierre.month)
    )
    serie_mes = df_saldo.loc[mask_mes, col_saldo].dropna()
    if serie_mes.empty or serie_mes.max() <= 0:
        return None
    saldo_max = float(serie_mes.max())

    ventana     = _ventana_antes_cierre(fecha_cierre, calendario, n_dh)
    disponibles = df_flujos.index.intersection(ventana)
    if disponibles.empty:
        return None
    retiro = float(df_flujos.loc[disponibles, col_saldo].sum(skipna=True))
    return abs(retiro) / saldo_max


def detectar_bancos_activos(
    df_saldo: pd.DataFrame,
    df_flujos: pd.DataFrame,
    bancos_saldo: list[str],
    cierres: list[pd.Timestamp],
    calendario: pd.DatetimeIndex,
    n_dh: int = VENTANA_RETIRO_DH,
    umbral: float = UMBRAL_ACTIVACION,
) -> dict[str, dict]:
    """
    Evalúa la estrategia de sobreencaje para cada banco.

    Retorna:
        {col_saldo: {"activa": bool, "ratios": {fecha_cierre: ratio}}}

    "activa" = True si ratio en el ÚLTIMO cierre > umbral.
    """
    resultado: dict[str, dict] = {}
    for col in bancos_saldo:
        if col not in df_saldo.columns:
            continue
        ratios: dict[pd.Timestamp, float] = {}
        for fc in cierres:
            r = _ratio_banco_cierre(df_saldo, df_flujos, col, fc, calendario, n_dh)
            if r is not None:
                ratios[fc] = r
        activa = bool(ratios.get(max(ratios), 0) > umbral) if ratios else False
        resultado[col] = {"activa": activa, "ratios": ratios}
    return resultado


###############################################################################
# Paso 2 — Peor retiro potencial por banco
###############################################################################

def _peor_B(
    df_saldo: pd.DataFrame,
    det: dict[str, dict],
    col_saldo: str,
    cierres: list[pd.Timestamp],
    factor_seg: float = FACTOR_SEGURIDAD,
    umbral: float = UMBRAL_ACTIVACION,
) -> float:
    """
    peor_B = worst_ratio_B × max_saldo_B × (1 + factor_seg)
    """
    ratios = det[col_saldo]["ratios"]
    ratios_activos = {fc: r for fc, r in ratios.items() if r > umbral}
    if not ratios_activos:
        return 0.0

    worst_ratio = max(ratios_activos.values())

    saldos_max = []
    for fc in cierres:
        mask = (
            (df_saldo.index.year  == fc.year) &
            (df_saldo.index.month == fc.month)
        )
        s = df_saldo.loc[mask, col_saldo].dropna()
        if not s.empty:
            saldos_max.append(float(s.max()))

    if not saldos_max:
        return 0.0

    return worst_ratio * max(saldos_max) * (1.0 + factor_seg)


###############################################################################
# Paso 3 — Factor multiplicativo
###############################################################################

def _factor_overlay(peor_total: float, q01_val: float) -> float:
    """
    f = peor_total / |Q01_h_cierre|
    Retorna 1.0 si el modelo ya cubre el peor caso o si Q01 ≥ 0.
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
    # Mapeo BANCOS_RETIROS → BANCOS_SALDOS
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
        logger.warning("[OVERLAY] Ningún banco del mapeo encontrado en Saldos_CCOVN — sin ajuste")
        return None

    df_saldo  = df_saldo_raw[bancos_saldo].copy()
    df_flujos = df_saldo.diff()
    calendario = pd.DatetimeIndex(df_saldo.index)

    todos_cierres = _cierres_trimestrales(calendario)
    cierres = [fc for fc in todos_cierres if fc < fecha_origen][-N_TRIMESTRES_LOOKBACK:]

    if len(cierres) < 2:
        logger.warning("[OVERLAY] Insuficientes cierres históricos")
        return None

    return df_saldo, df_flujos, bancos_saldo, cierres, calendario


###############################################################################
# Función principal — formato dict {tau: array}  (uso desde step005)
###############################################################################

def aplicar_overlay_preds(
    preds: dict[float, np.ndarray],
    h_arr: np.ndarray,
    fecha_origen: pd.Timestamp,
    ruta_saldo: Path = RUTA_SALDO,
    ruta_tabla: Path  = RUTA_TABLA,
) -> dict[float, np.ndarray]:
    """
    Aplica el overlay sobreencaje a un dict de predicciones.

    Parámetros
    ----------
    preds        : {0.01: arr, 0.05: arr, 0.50: arr, 0.95: arr, 0.99: arr}
    h_arr        : horizonte de cada fila, shape (n,).
    fecha_origen : fecha de inicio del pronóstico (fecha_t en step005).
    """
    if not OVERLAY_SOBREENCAJE_ACTIVO:
        return preds

    resultado = _preparar_datos(ruta_saldo, ruta_tabla, fecha_origen)
    if resultado is None:
        return preds

    df_saldo, df_flujos, bancos_saldo, cierres, calendario = resultado

    det = detectar_bancos_activos(df_saldo, df_flujos, bancos_saldo, cierres, calendario)
    bancos_activos = [b for b, info in det.items() if info["activa"]]

    if not bancos_activos:
        logger.info("[OVERLAY] Sin bancos con estrategia activa — sin ajuste")
        return preds

    logger.info(f"[OVERLAY] Bancos activos ({len(bancos_activos)}): {bancos_activos}")

    peor_total = sum(_peor_B(df_saldo, det, b, cierres) for b in bancos_activos)
    logger.info(f"[OVERLAY] peor_total = {peor_total:,.1f}")

    # Mapeo h → fecha_pred usando pandas bdate_range (calendario estándar)
    h_max = int(h_arr.max())
    bh_future = pd.bdate_range(
        start=fecha_origen + pd.offsets.BDay(1),
        periods=h_max + 10,
    )
    h_to_fecha = {h + 1: bh_future[h] for h in range(len(bh_future))}
    cierres_futuros = set(_cierres_trimestrales(pd.DatetimeIndex(bh_future)))

    preds_adj = {tau: arr.copy() for tau, arr in preds.items()}

    for i, h in enumerate(h_arr):
        fecha_pred = h_to_fecha.get(int(h))
        if fecha_pred is None or fecha_pred not in cierres_futuros:
            continue
        q01_val = float(preds[0.01][i])
        f = _factor_overlay(peor_total, q01_val)
        if f <= 1.0:
            continue
        for tau in preds_adj:
            if tau in (0.01, 0.05, 0.50, 0.95, 0.99):
                preds_adj[tau][i] *= f
        logger.info(
            f"[OVERLAY] h={h:3d} ({fecha_pred.date()}) | "
            f"Q01={q01_val:+.1f} → {q01_val*f:+.1f} | f={f:.3f}"
        )

    return preds_adj


###############################################################################
# Variante DataFrame — columnas q01..q99  (uso desde step006 / postproceso)
###############################################################################

def aplicar_overlay_df(
    df: pd.DataFrame,
    fecha_origen: pd.Timestamp,
    ruta_saldo: Path = RUTA_SALDO,
    ruta_tabla: Path  = RUTA_TABLA,
) -> pd.DataFrame:
    """
    Variante para DataFrames con columnas q01, q05, q50, q95, q99 y columna h.
    """
    q_cols_presentes = [c for c in Q_COLS if c in df.columns]
    if not q_cols_presentes or "h" not in df.columns:
        logger.warning("[OVERLAY] Columnas q0x/h no encontradas — sin ajuste")
        return df

    preds = {float(c[1:]) / 100.0: df[c].values.copy() for c in q_cols_presentes}
    h_arr = df["h"].values

    preds_adj = aplicar_overlay_preds(preds, h_arr, fecha_origen, ruta_saldo, ruta_tabla)

    df_adj = df.copy()
    for c in q_cols_presentes:
        tau = float(c[1:]) / 100.0
        if tau in preds_adj:
            df_adj[c] = preds_adj[tau]
    return df_adj


###############################################################################
# Diagnóstico standalone — outliers + activación
###############################################################################

def _diagnostico(
    ruta_saldo: Path = RUTA_SALDO,
    ruta_tabla: Path  = RUTA_TABLA,
    fecha_ref: pd.Timestamp | None = None,
) -> None:
    """
    Imprime:
      1. Bancos del mapeo presentes/ausentes en Saldos_CCOVN
      2. Outliers de saldo por banco (zscore > 4σ o IQR > 6×IQR)
      3. Ratios históricos y bancos activos
      4. peor_B por banco y peor_total
    """
    if fecha_ref is None:
        fecha_ref = pd.Timestamp.today().normalize()

    sep = "═" * 72

    print(f"\n{sep}")
    print(f"  Overlay Sobreencaje — Diagnóstico  ({fecha_ref.date()})")
    print(sep)

    # ── Mapeo ────────────────────────────────────────────────────────────────
    try:
        tabla = cargar_tabla_mapeo(ruta_tabla)
    except Exception as e:
        print(f"\n  ERROR: No se pudo leer TablaSaldosRetiros.xlsx: {e}")
        return

    print(f"\n  Mapeo BANCOS_RETIROS ↔ BANCOS_SALDOS ({len(tabla)} bancos con saldo):")
    for _, row in tabla.iterrows():
        print(f"    {row['BANCOS_RETIROS']:<20} → {row['BANCOS_SALDOS']}")

    # ── Cargar saldos ────────────────────────────────────────────────────────
    try:
        df_saldo_raw = _cargar_saldos(ruta_saldo)
    except Exception as e:
        print(f"\n  ERROR: No se pudo leer Saldos_CCOVN.xlsx: {e}")
        return

    print(f"\n  Saldos CC+OVN: {len(df_saldo_raw):,} filas | "
          f"{df_saldo_raw.index.min().date()} → {df_saldo_raw.index.max().date()}")
    print(f"  Columnas en archivo ({len(df_saldo_raw.columns)}): {list(df_saldo_raw.columns)}")

    # Cruce mapeo ↔ saldo
    cols_saldo_upper = {c.strip().upper(): c for c in df_saldo_raw.columns}
    bancos_saldo: list[str] = []
    print(f"\n  Cruce mapeo ↔ saldo:")
    for _, row in tabla.iterrows():
        nombre_ret  = row["BANCOS_RETIROS"]
        nombre_sal  = row["BANCOS_SALDOS"]
        col_real    = cols_saldo_upper.get(nombre_sal.strip().upper())
        if col_real:
            if col_real not in bancos_saldo:
                bancos_saldo.append(col_real)
            estado = "OK"
        else:
            estado = "AUSENTE EN SALDO"
        print(f"    {nombre_ret:<20} → {nombre_sal:<30}  [{estado}]")

    if not bancos_saldo:
        print("\n  Sin bancos en intersección — revisar nombres en TablaSaldosRetiros.xlsx")
        return

    df_saldo = df_saldo_raw[bancos_saldo].copy()

    # ── Outliers por banco ───────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  OUTLIERS DE SALDO POR BANCO (zscore > {OUTLIER_ZSCORE}σ  |  IQR > {OUTLIER_IQR}×IQR)")
    print(f"{'─'*72}")

    total_outliers = 0
    for banco in bancos_saldo:
        outs = detectar_outliers_banco(df_saldo[banco])
        if outs.empty:
            print(f"  {banco:<35}  sin outliers")
        else:
            total_outliers += len(outs)
            print(f"\n  {banco:<35}  {len(outs)} outliers:")
            for _, r in outs.iterrows():
                metodo = ("zscore+IQR" if r["outlier_zscore"] and r["outlier_iqr"]
                          else "zscore" if r["outlier_zscore"] else "IQR")
                print(f"    {str(r['fecha'].date()):<12}  valor={r['valor']:>12,.1f}  "
                      f"z={r['z_score']:+.1f}σ  [{metodo}]")

    print(f"\n  Total outliers detectados: {total_outliers}")

    # ── Activación de estrategia ─────────────────────────────────────────────
    df_flujos  = df_saldo.diff()
    calendario = pd.DatetimeIndex(df_saldo.index)

    todos_cierres = _cierres_trimestrales(calendario)
    cierres = [fc for fc in todos_cierres if fc < fecha_ref][-N_TRIMESTRES_LOOKBACK:]

    print(f"\n{'─'*72}")
    print(f"  DETECCIÓN DE ESTRATEGIA — últimos {N_TRIMESTRES_LOOKBACK} cierres trimestrales")
    print(f"{'─'*72}")
    for fc in cierres:
        print(f"  {fc.date()}")

    det = detectar_bancos_activos(df_saldo, df_flujos, bancos_saldo, cierres, calendario)

    print(f"\n  {'Banco (saldo)':<35} {'Estado':>12}   Ratios por cierre trimestral")
    print(f"  {'─'*35} {'─'*12}   {'─'*35}")
    for banco, info in sorted(det.items()):
        estado = "ACTIVO  ✓" if info["activa"] else "inactivo"
        ratios_str = "  ".join(
            f"{fc.year}-Q{fc.quarter}: {r:.0%}"
            for fc, r in sorted(info["ratios"].items())
        )
        print(f"  {banco:<35} {estado:>12}   {ratios_str}")

    # ── peor_B y peor_total ─────────────────────────────────────────────────
    bancos_activos = [b for b, info in det.items() if info["activa"]]
    peor_total = sum(_peor_B(df_saldo, det, b, cierres) for b in bancos_activos)

    print(f"\n{'─'*72}")
    print(f"  PEOR RETIRO POTENCIAL (bancos activos)")
    print(f"{'─'*72}")
    for b in bancos_activos:
        pb = _peor_B(df_saldo, det, b, cierres)
        print(f"  {b:<35}  {pb:>14,.1f}")
    if bancos_activos:
        print(f"  {'─'*35}  {'─'*14}")
        print(f"  {'peor_total':35}  {peor_total:>14,.1f}")

    print(f"\n  Parámetros activos:")
    print(f"    OVERLAY_SOBREENCAJE_ACTIVO = {OVERLAY_SOBREENCAJE_ACTIVO}")
    print(f"    UMBRAL_ACTIVACION          = {UMBRAL_ACTIVACION:.0%}")
    print(f"    N_TRIMESTRES_LOOKBACK      = {N_TRIMESTRES_LOOKBACK}")
    print(f"    VENTANA_RETIRO_DH          = {VENTANA_RETIRO_DH} dh")
    print(f"    FACTOR_SEGURIDAD           = {FACTOR_SEGURIDAD:.0%}")
    print(f"    OUTLIER_ZSCORE / IQR       = {OUTLIER_ZSCORE} / {OUTLIER_IQR}")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _diagnostico()
