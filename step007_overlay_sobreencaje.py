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

    Intución: proyecta el peor comportamiento porcentual observado sobre el saldo
    máximo reciente → cubre el escenario en que el banco sigue depositando encaje.

Paso 3 — Factor multiplicativo y ajuste
    peor_total = Σ peor_B   (bancos activos)
    f          = peor_total / |Q01_h_cierre|

    Si f > 1 → el modelo es optimista → ajuste:
        Q_adj[τ, h_cierre] = Q[τ, h_cierre] × f
    Aplica de forma simétrica a todos los cuantiles y la mediana.
    Solo afecta horizontes h que correspondan a un cierre trimestral.

Uso desde step005 (modo dict)
------------------------------
    from step007_overlay_sobreencaje import aplicar_overlay_preds
    preds_test = aplicar_overlay_preds(preds_test, h_arr, fecha_origen)

Uso desde step006 / postproceso (modo DataFrame)
-------------------------------------------------
    from step007_overlay_sobreencaje import aplicar_overlay_df
    df_preds = aplicar_overlay_df(df_preds, fecha_origen)

Diagnóstico rápido (standalone)
---------------------------------
    python step007_overlay_sobreencaje.py
    → muestra bancos activos, ratios históricos y peor_total estimado.
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
VENTANA_RETIRO_DH     = 7      # días hábiles de la ventana de retiro (antes del cierre)
FACTOR_SEGURIDAD      = 0.00   # margen adicional sobre peor_B  (0.00→0.10 = 0%→10%)

# ── Rutas de datos ────────────────────────────────────────────────────────────
# Saldo fin del día (CC + Overnight) por banco — formato wide:
#   col 0 = fecha | col 1..N = bancos (una columna por banco)
RUTA_SALDO = BASE_SISTEMA / "1. Data" / "Raw" / "Saldo fin del dia CC+OVN.xlsx"

# Alternativa si el archivo se llama Saldos_CCOVN.xlsx:
# RUTA_SALDO = BASE_SISTEMA / "1. Data" / "Raw" / "Saldos_CCOVN.xlsx"

# Transacciones bancarias — formato transaccional:
#   columnas: Broker (banco), Fecha Valor, Delivery Principal Usd
RUTA_TRANS = BASE_SISTEMA / "1. Data" / "Raw" / "Transacciones_BancaLocal.xlsx"

# Alias históricos de nombres de banco (ej. CONTINEN → BBVA antes de fusión)
ALIAS_BANCOS = {
    "CONTINEN": "BBVA",
}

# ── Cuantiles ─────────────────────────────────────────────────────────────────
QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]
Q_COLS    = [f"q{int(t * 100):02d}" for t in QUANTILES]   # q01, q05, q50, q95, q99


###############################################################################
# Carga de datos
###############################################################################

def _cargar_saldos(ruta: Path) -> pd.DataFrame:
    """
    Lee el archivo de saldo fin del día CC + OVN.

    Formato esperado: primera columna = fecha, columnas restantes = un banco cada una.
    Retorna DataFrame con index=fecha, columnas=bancos, valores en MM USD (o la unidad
    del archivo original — el overlay usa ratios, por lo que la unidad no importa).
    """
    raw = pd.read_excel(ruta, header=0)
    col_fecha   = raw.columns[0]
    cols_bancos = raw.columns[1:].tolist()

    raw[col_fecha] = pd.to_datetime(raw[col_fecha], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=[col_fecha]).sort_values(col_fecha)
    raw = raw.set_index(col_fecha)
    raw.index.name = "fecha"

    for c in cols_bancos:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Aplicar alias de nombres históricos
    alias_lower = {k.lower(): v for k, v in ALIAS_BANCOS.items()}
    nuevas_cols = {c: alias_lower.get(str(c).lower(), str(c)) for c in cols_bancos}
    raw = raw.rename(columns=nuevas_cols)

    # Consolidar columnas con el mismo nombre tras alias (ej. CONTINEN + BBVA → BBVA)
    raw = raw.T.groupby(raw.columns).sum().T

    return raw


def _bancos_transacciones(ruta: Path) -> set[str]:
    """
    Retorna el conjunto de bancos únicos presentes en Transacciones_BancaLocal.xlsx
    (columna 'Broker', aplicando alias).
    """
    df = pd.read_excel(ruta, usecols=["Broker"])
    alias_lower = {k.lower(): v for k, v in ALIAS_BANCOS.items()}
    bancos = {alias_lower.get(str(b).lower(), str(b)) for b in df["Broker"].dropna().unique()}
    return bancos


def _bancos_comunes(cols_saldo: list[str], bancos_trans: set[str]) -> list[str]:
    """
    Intersección case-insensitive entre columnas del saldo y bancos de transacciones.
    Retorna nombres tal como aparecen en el archivo de saldo.
    """
    trans_lower = {b.lower() for b in bancos_trans}
    return [c for c in cols_saldo if str(c).lower() in trans_lower]


###############################################################################
# Calendario y fechas de cierre trimestral
###############################################################################

def _cierres_trimestrales(calendario: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """
    Devuelve el último día hábil de cada trimestre (meses 3, 6, 9, 12)
    presente en el calendario dado.
    """
    df = pd.DataFrame({"fecha": calendario})
    df["mes"]  = df["fecha"].dt.month
    df["anio"] = df["fecha"].dt.year
    df["trim"] = df["fecha"].dt.quarter
    cierres = (
        df[df["mes"].isin([3, 6, 9, 12])]
        .groupby(["anio", "trim"])["fecha"]
        .max()
        .sort_values()
        .tolist()
    )
    return cierres


def _ventana_antes_cierre(
    fecha_cierre: pd.Timestamp,
    calendario: pd.DatetimeIndex,
    n: int,
) -> pd.DatetimeIndex:
    """
    Devuelve los últimos n días hábiles del calendario que terminan en
    fecha_cierre (inclusive), es decir la ventana [cierre − (n−1) dh, cierre].
    """
    pos = int(calendario.searchsorted(fecha_cierre, side="right"))
    return calendario[max(0, pos - n): pos]


###############################################################################
# Paso 1 — Detección por banco
###############################################################################

def _ratio_banco_cierre(
    df_saldo: pd.DataFrame,
    df_flujos: pd.DataFrame,
    banco: str,
    fecha_cierre: pd.Timestamp,
    calendario: pd.DatetimeIndex,
    n_dh: int,
) -> float | None:
    """
    ratio = |retiro acumulado en ventana de n_dh dh antes del cierre|
             / saldo máximo en el mes del cierre.

    Retorna None si faltan datos suficientes.
    """
    mask_mes = (
        (df_saldo.index.year  == fecha_cierre.year) &
        (df_saldo.index.month == fecha_cierre.month)
    )
    serie_mes = df_saldo.loc[mask_mes, banco].dropna()
    if serie_mes.empty or serie_mes.max() <= 0:
        return None
    saldo_max = float(serie_mes.max())

    ventana     = _ventana_antes_cierre(fecha_cierre, calendario, n_dh)
    disponibles = df_flujos.index.intersection(ventana)
    if disponibles.empty:
        return None
    retiro = float(df_flujos.loc[disponibles, banco].sum(skipna=True))

    return abs(retiro) / saldo_max


def detectar_bancos_activos(
    df_saldo: pd.DataFrame,
    df_flujos: pd.DataFrame,
    bancos: list[str],
    cierres: list[pd.Timestamp],
    calendario: pd.DatetimeIndex,
    n_dh: int = VENTANA_RETIRO_DH,
    umbral: float = UMBRAL_ACTIVACION,
) -> dict[str, dict]:
    """
    Evalúa si cada banco activó la estrategia de sobreencaje.

    Retorna:
        {banco: {"activa": bool, "ratios": {fecha_cierre: ratio}}}

    "activa" = True si ratio en el ÚLTIMO cierre > umbral.
    """
    resultado: dict[str, dict] = {}

    for banco in bancos:
        if banco not in df_saldo.columns:
            continue

        ratios: dict[pd.Timestamp, float] = {}
        for fc in cierres:
            r = _ratio_banco_cierre(df_saldo, df_flujos, banco, fc, calendario, n_dh)
            if r is not None:
                ratios[fc] = r

        if ratios:
            ultimo = max(ratios)
            activa = ratios[ultimo] > umbral
        else:
            activa = False

        resultado[banco] = {"activa": activa, "ratios": ratios}

    return resultado


###############################################################################
# Paso 2 — Peor retiro potencial por banco
###############################################################################

def _peor_B(
    df_saldo: pd.DataFrame,
    det: dict[str, dict],
    banco: str,
    cierres: list[pd.Timestamp],
    factor_seg: float = FACTOR_SEGURIDAD,
    umbral: float = UMBRAL_ACTIVACION,
) -> float:
    """
    peor_B = worst_ratio_B × max_saldo_B × (1 + factor_seg)

    worst_ratio_B : peor ratio en cierres donde la estrategia estuvo activa
    max_saldo_B   : saldo máximo en cualquiera de los N cierres (sin filtro de activación)
    """
    ratios = det[banco]["ratios"]
    ratios_activos = {fc: r for fc, r in ratios.items() if r > umbral}
    if not ratios_activos:
        return 0.0

    worst_ratio = max(ratios_activos.values())

    saldos_max = []
    for fc in cierres:
        if banco not in df_saldo.columns:
            continue
        mask = (
            (df_saldo.index.year  == fc.year) &
            (df_saldo.index.month == fc.month)
        )
        s = df_saldo.loc[mask, banco].dropna()
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

    Retorna 1.0 (sin ajuste) cuando:
      - Q01 ≥ 0 (modelo pronostica ingreso neto en ese horizonte)
      - |Q01| es despreciable
      - f ≤ 1 (el modelo ya cubre el peor histórico)
    """
    if q01_val >= 0.0 or abs(q01_val) < 1e-6:
        return 1.0
    f = peor_total / abs(q01_val)
    return f if f > 1.0 else 1.0


###############################################################################
# Función principal — formato dict {tau: array}  (uso desde step005)
###############################################################################

def aplicar_overlay_preds(
    preds: dict[float, np.ndarray],
    h_arr: np.ndarray,
    fecha_origen: pd.Timestamp,
    ruta_saldo: Path = RUTA_SALDO,
    ruta_trans: Path  = RUTA_TRANS,
) -> dict[float, np.ndarray]:
    """
    Aplica el overlay sobreencaje a un dict de predicciones.

    Parámetros
    ----------
    preds        : {0.01: arr, 0.05: arr, 0.50: arr, 0.95: arr, 0.99: arr}
                   Cada array tiene shape (n,) donde n = número de filas (fecha × h).
    h_arr        : horizonte de cada fila, shape (n,).
    fecha_origen : fecha de inicio del pronóstico (fecha_t en step005).
    ruta_saldo   : ruta al archivo de saldo CC+OVN por banco.
    ruta_trans   : ruta a Transacciones_BancaLocal.xlsx.

    Retorna
    -------
    dict con misma estructura que preds, con valores ajustados solo en
    horizontes que corresponden a un cierre trimestral y solo si f > 1.
    """
    if not OVERLAY_SOBREENCAJE_ACTIVO:
        return preds

    # ── Cargar datos ──────────────────────────────────────────────────────────
    try:
        df_saldo = _cargar_saldos(ruta_saldo)
    except Exception as e:
        logger.warning(f"[OVERLAY] No se pudo cargar saldo CC+OVN: {e} — sin ajuste")
        return preds

    try:
        bancos_trans = _bancos_transacciones(ruta_trans)
    except Exception as e:
        logger.warning(f"[OVERLAY] No se pudo cargar Transacciones_BancaLocal: {e} — sin ajuste")
        return preds

    bancos = _bancos_comunes(list(df_saldo.columns), bancos_trans)
    if not bancos:
        logger.warning("[OVERLAY] Sin bancos en intersección saldo ∩ transacciones — sin ajuste")
        return preds

    df_saldo  = df_saldo[bancos]
    df_flujos = df_saldo.diff()                         # flujo = Δ(saldo)
    calendario = pd.DatetimeIndex(df_saldo.index)

    # ── Paso 1: cierres históricos ────────────────────────────────────────────
    todos_cierres = _cierres_trimestrales(calendario)
    cierres = [fc for fc in todos_cierres if fc < fecha_origen][-N_TRIMESTRES_LOOKBACK:]

    if len(cierres) < 2:
        logger.warning("[OVERLAY] Insuficientes cierres históricos — sin ajuste")
        return preds

    det = detectar_bancos_activos(df_saldo, df_flujos, bancos, cierres, calendario)
    bancos_activos = [b for b, info in det.items() if info["activa"]]

    if not bancos_activos:
        logger.info("[OVERLAY] Sin bancos con estrategia activa — sin ajuste")
        return preds

    logger.info(f"[OVERLAY] Bancos activos ({len(bancos_activos)}): {bancos_activos}")

    # ── Paso 2: peor caso agregado ────────────────────────────────────────────
    peor_total = sum(
        _peor_B(df_saldo, det, b, cierres)
        for b in bancos_activos
    )
    logger.info(f"[OVERLAY] peor_total = {peor_total:,.1f} (unidad del archivo saldo)")

    # ── Construir mapa h → fecha_pred usando bdate_range ──────────────────────
    h_max = int(h_arr.max())
    bh_future = pd.bdate_range(
        start=fecha_origen + pd.offsets.BDay(1),
        periods=h_max + 10,
    )
    h_to_fecha: dict[int, pd.Timestamp] = {
        h + 1: bh_future[h] for h in range(len(bh_future))
    }

    # Cierres trimestrales futuros en el horizonte de pronóstico
    cierres_futuros = set(_cierres_trimestrales(pd.DatetimeIndex(bh_future)))

    # ── Paso 3: aplicar factor por horizonte ──────────────────────────────────
    preds_adj = {tau: arr.copy() for tau, arr in preds.items()}
    ajustes_aplicados: list[tuple[int, pd.Timestamp, float, float]] = []

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

        ajustes_aplicados.append((int(h), fecha_pred, q01_val, f))

    if ajustes_aplicados:
        for h, fd, q01, f in ajustes_aplicados:
            logger.info(
                f"[OVERLAY] h={h:3d} ({fd.date()}) | "
                f"Q01={q01:+.1f} → Q01_adj={q01*f:+.1f} | f={f:.3f}"
            )
    else:
        logger.info("[OVERLAY] f ≤ 1 en todos los cierres — sin ajuste aplicado")

    return preds_adj


###############################################################################
# Variante DataFrame — columnas q01/q05/q50/q95/q99  (uso desde step006)
###############################################################################

def aplicar_overlay_df(
    df: pd.DataFrame,
    fecha_origen: pd.Timestamp,
    ruta_saldo: Path = RUTA_SALDO,
    ruta_trans: Path  = RUTA_TRANS,
) -> pd.DataFrame:
    """
    Variante para DataFrames con columnas q01, q05, q50, q95, q99 y columna h.

    Útil para postprocesar parquets guardados por step005 o DataFrames de step006.
    Retorna una copia del DataFrame con los cuantiles ajustados.
    """
    q_cols_presentes = [c for c in Q_COLS if c in df.columns]
    if not q_cols_presentes or "h" not in df.columns:
        logger.warning("[OVERLAY] Columnas q0x/h no encontradas en DataFrame — sin ajuste")
        return df

    # Convertir a formato dict {tau: array}
    preds = {float(c[1:]) / 100.0: df[c].values.copy() for c in q_cols_presentes}
    h_arr = df["h"].values

    preds_adj = aplicar_overlay_preds(preds, h_arr, fecha_origen, ruta_saldo, ruta_trans)

    df_adj = df.copy()
    for c in q_cols_presentes:
        tau = float(c[1:]) / 100.0
        if tau in preds_adj:
            df_adj[c] = preds_adj[tau]

    return df_adj


###############################################################################
# Diagnóstico standalone
###############################################################################

def _diagnostico(
    ruta_saldo: Path = RUTA_SALDO,
    ruta_trans: Path  = RUTA_TRANS,
    fecha_ref: pd.Timestamp | None = None,
) -> None:
    """
    Imprime el estado actual del overlay:
      - Bancos en intersección saldo ∩ transacciones
      - Ratios históricos por cierre trimestral
      - Bancos activos
      - peor_B por banco activo y peor_total
    """
    if fecha_ref is None:
        fecha_ref = pd.Timestamp.today().normalize()

    print(f"\n{'═'*70}")
    print(f"  Overlay Sobreencaje — Diagnóstico  ({fecha_ref.date()})")
    print(f"{'═'*70}")

    df_saldo = _cargar_saldos(ruta_saldo)
    bancos_trans = _bancos_transacciones(ruta_trans)
    bancos = _bancos_comunes(list(df_saldo.columns), bancos_trans)

    print(f"\n  Bancos en saldo CC+OVN      : {len(df_saldo.columns)}")
    print(f"  Bancos en transacciones     : {len(bancos_trans)}")
    print(f"  Bancos en intersección      : {len(bancos)}  → {bancos}")

    df_saldo  = df_saldo[bancos]
    df_flujos = df_saldo.diff()
    calendario = pd.DatetimeIndex(df_saldo.index)

    todos_cierres = _cierres_trimestrales(calendario)
    cierres = [fc for fc in todos_cierres if fc < fecha_ref][-N_TRIMESTRES_LOOKBACK:]

    print(f"\n  Últimos {N_TRIMESTRES_LOOKBACK} cierres trimestrales analizados:")
    for fc in cierres:
        print(f"    {fc.date()}")

    det = detectar_bancos_activos(df_saldo, df_flujos, bancos, cierres, calendario)

    print(f"\n  {'Banco':<30} {'Estado':>10}   Ratios por cierre")
    print(f"  {'─'*30} {'─'*10}   {'─'*40}")
    for banco, info in sorted(det.items()):
        estado = "ACTIVO  ✓" if info["activa"] else "inactivo"
        ratios_str = "  ".join(
            f"{fc.strftime('%Y-Q%q')} {r:.0%}"  # type: ignore[str-format]
            for fc, r in sorted(info["ratios"].items())
        )
        # Fallback si strftime %q no está disponible en el OS
        try:
            ratios_str = "  ".join(
                f"{fc.year}-Q{fc.quarter}: {r:.0%}"
                for fc, r in sorted(info["ratios"].items())
            )
        except Exception:
            pass
        print(f"  {banco:<30} {estado:>10}   {ratios_str}")

    bancos_activos = [b for b, info in det.items() if info["activa"]]
    peor_total = sum(_peor_B(df_saldo, det, b, cierres) for b in bancos_activos)

    print(f"\n  peor_B por banco activo:")
    for b in bancos_activos:
        pb = _peor_B(df_saldo, det, b, cierres)
        print(f"    {b:<30}  {pb:>12,.1f}")

    print(f"\n  peor_total (Σ bancos activos): {peor_total:>12,.1f}")
    print(f"\n  Parámetros activos:")
    print(f"    OVERLAY_SOBREENCAJE_ACTIVO = {OVERLAY_SOBREENCAJE_ACTIVO}")
    print(f"    UMBRAL_ACTIVACION          = {UMBRAL_ACTIVACION:.0%}")
    print(f"    N_TRIMESTRES_LOOKBACK      = {N_TRIMESTRES_LOOKBACK}")
    print(f"    VENTANA_RETIRO_DH          = {VENTANA_RETIRO_DH} dh")
    print(f"    FACTOR_SEGURIDAD           = {FACTOR_SEGURIDAD:.0%}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _diagnostico()
