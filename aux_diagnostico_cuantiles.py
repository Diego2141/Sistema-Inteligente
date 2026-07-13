# -*- coding: utf-8 -*-
"""
aux_diagnostico_cuantiles.py
============================
Genera un Excel de diagnóstico que cruza:
  - overlay_meta_{banco}_{fecha}.csv   -> factor_f, peor_total, peor_restante, etc.
  - preds_overlay_{banco}_{fecha}.parquet -> cuantiles Q01/Q05/Q50/Q95/Q99 y mean
  - preds_base_{banco}_{fecha}.parquet    -> cuantiles PRE-overlay (sin ajuste)
  - Ajuste_diario (saldos_retiros_bancos.xlsx) -> peor_total de referencia

El Excel permite validar visualmente que:
  1. El factor f tiene magnitud razonable (estructuralmente acotado).
  2. Los cuantiles proyectados se inflan al acercarse al cierre trimestral.
  3. La inflación es proporcional al peor_restante (netting correcto).

SHEETS
------
  "Metodologia"      : descripción de la lógica IN/OUT de la ventana
  "Factor_overlay"   : una fila por fecha_t con factor, razon_no_activo, etc.
  "Cuantiles_origen" : una fila por (fecha_t, h) con Q01/Q05/Q50/Q95/Q99 y target
  "Pre_vs_Post"      : pre y post ajuste por (fecha_t, h) con delta y meta
  "Factor_evolucion" : evolución del factor por cierre_fecha a través del tiempo
  "Pivot_Q05"        : tabla cruzada fecha_t × h del Q05 (para heatmap visual)
  "Pivot_Factor"     : tabla cruzada fecha_t × cierre con factor_f asignado
  "Calendario_BDay"  : días hábiles 2025-2026 según calendario PER+USA

USO
---
  1. Ejecutar step005 con OVERLAY_SOBREENCAJE = True
  2. Ajustar BANCO y FECHA_HOY (o dejar AUTO para usar el archivo más reciente)
  3. python aux_diagnostico_cuantiles.py
  4. Abrir diagnostico_cuantiles_{banco}.xlsx en 2. Output/analisis_cc/
"""

from __future__ import annotations
from pathlib import Path
import glob

import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_STEP005  = BASE_SISTEMA / "2. Output" / "step005_wfcv_v3"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "analisis_cc"

BANCO     = "SISTEMA"
FECHA_HOY = "AUTO"   # "AUTO" usa el archivo más reciente; o pon "2025-07-10"

# Cuantiles a mostrar en la hoja Cuantiles_origen / Pre_vs_Post
TAUS_MOSTRAR = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

# Calendario PER+USA (mismo que step005) — usado para generar Calendario_BDay
_ANOS_FERIADOS = range(2018, 2032)
try:
    import holidays as _hlib
    _dias_pe = _hlib.Peru(years=_ANOS_FERIADOS)
    _dias_us = _hlib.UnitedStates(years=_ANOS_FERIADOS)
    _FERIADOS_PEUSA = pd.to_datetime(
        sorted(set(list(_dias_pe.keys()) + list(_dias_us.keys())))
    )
except ImportError:
    _f_pe = pd.to_datetime(
        [f"{y}-01-01" for y in _ANOS_FERIADOS]
        + [f"{y}-05-01" for y in _ANOS_FERIADOS]
        + [f"{y}-06-29" for y in _ANOS_FERIADOS]
        + [f"{y}-07-28" for y in _ANOS_FERIADOS]
        + [f"{y}-07-29" for y in _ANOS_FERIADOS]
        + [f"{y}-08-30" for y in _ANOS_FERIADOS]
        + [f"{y}-10-08" for y in _ANOS_FERIADOS]
        + [f"{y}-11-01" for y in _ANOS_FERIADOS]
        + [f"{y}-12-08" for y in _ANOS_FERIADOS]
        + [f"{y}-12-24" for y in _ANOS_FERIADOS]
        + [f"{y}-12-25" for y in _ANOS_FERIADOS]
    )
    _f_us = pd.to_datetime(
        [f"{y}-07-04" for y in _ANOS_FERIADOS]
        + [f"{y}-06-19" for y in _ANOS_FERIADOS]
        + [f"{y}-11-11" for y in _ANOS_FERIADOS]
    )
    _FERIADOS_PEUSA = pd.to_datetime(
        sorted(set(_f_pe.tolist() + _f_us.tolist()))
    )

from pandas.tseries.offsets import CustomBusinessDay
BDAY_PE = CustomBusinessDay(holidays=_FERIADOS_PEUSA)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _encontrar_ultimo(patron: str) -> Path:
    archivos = sorted(glob.glob(patron, recursive=True))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontró archivo con patrón: {patron}\n"
            f"  -> Asegúrate de haber corrido step005 con OVERLAY_SOBREENCAJE = True"
        )
    return Path(archivos[-1])


def _cargar_datos(banco: str, fecha_hoy: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if fecha_hoy == "AUTO":
        ruta_overlay = _encontrar_ultimo(str(DIR_STEP005 / "**" / f"preds_overlay_{banco}_*.parquet"))
        ruta_meta    = _encontrar_ultimo(str(DIR_STEP005 / "**" / f"overlay_meta_{banco}_*.csv"))
    else:
        ruta_overlay = _encontrar_ultimo(str(DIR_STEP005 / "**" / f"preds_overlay_{banco}_{fecha_hoy}.parquet"))
        ruta_meta    = _encontrar_ultimo(str(DIR_STEP005 / "**" / f"overlay_meta_{banco}_{fecha_hoy}.csv"))

    df_overlay = pd.read_parquet(ruta_overlay)
    df_meta    = pd.read_csv(ruta_meta, parse_dates=["fecha_t", "cierre_fecha"])

    print(f"[OK] Predicciones overlay cargadas : {ruta_overlay.name}  ({len(df_overlay):,} filas)")
    print(f"[OK] Metadata cargada              : {ruta_meta.name}   ({len(df_meta):,} filas)")

    # preds_base (pre-overlay) — opcional; puede no existir si se corrió sin guardar base
    df_base = None
    try:
        if fecha_hoy == "AUTO":
            ruta_base = _encontrar_ultimo(str(DIR_STEP005 / "**" / f"preds_base_{banco}_*.parquet"))
        else:
            ruta_base = _encontrar_ultimo(str(DIR_STEP005 / "**" / f"preds_base_{banco}_{fecha_hoy}.parquet"))
        df_base = pd.read_parquet(ruta_base)
        print(f"[OK] Predicciones base cargadas    : {ruta_base.name}  ({len(df_base):,} filas)")
    except FileNotFoundError:
        print("[WARN] preds_base_*.parquet no encontrado — hoja Pre_vs_Post usará solo post-overlay")

    return df_overlay, df_meta, df_base


# ── Sheet builders ────────────────────────────────────────────────────────────

def _sheet_factor(df_meta: pd.DataFrame) -> pd.DataFrame:
    cols_orden = [
        "fecha_t", "cierre_fecha", "h_cierre", "n_inciertos",
        "peor_total", "retiro_pasado", "retiro_t2", "retiro_conocido", "peor_restante",
        "q_tau_acum", "factor_f", "overlay_activo", "razon_no_activo",
    ]
    cols = [c for c in cols_orden if c in df_meta.columns]
    df = df_meta[cols].copy().sort_values(["fecha_t"])
    df["fecha_t"]      = pd.to_datetime(df["fecha_t"]).dt.date
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"]).dt.date
    return df


def _sheet_cuantiles(df_preds: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    df = df_preds.copy()
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])

    tau_cols = []
    for tau in TAUS_MOSTRAR:
        col = f"q{int(tau * 100):02d}"
        if col in df.columns:
            tau_cols.append(col)
    if "mean" in df.columns:
        tau_cols.append("mean")

    meta_slim = (
        df_meta[["fecha_t", "cierre_fecha", "h_cierre", "factor_f", "overlay_activo", "razon_no_activo"]]
        .rename(columns={"h_cierre": "h_cierre_overlay"})
    )
    meta_slim["fecha_t"] = pd.to_datetime(meta_slim["fecha_t"])

    df = df.merge(meta_slim, on="fecha_t", how="left")

    cols_base = ["banco", "fold", "fecha_t", "h", "target"]
    cols = [c for c in cols_base if c in df.columns] + tau_cols + [
        "cierre_fecha", "h_cierre_overlay", "factor_f", "overlay_activo", "razon_no_activo"
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["fecha_t", "h"])
    df["fecha_t"] = df["fecha_t"].dt.date
    if "cierre_fecha" in df.columns:
        df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"]).dt.date
    return df


def _sheet_pre_post(
    df_overlay: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_base: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Compara predicciones pre-overlay (base) y post-overlay por (fecha_t, h).
    Incluye delta por cada cuantil, retiro_pasado, peor_restante y factor_f.
    Si df_base es None, solo se muestran los valores post-overlay con meta.
    """
    tau_cols = []
    for tau in TAUS_MOSTRAR:
        col = f"q{int(tau * 100):02d}"
        if col in df_overlay.columns:
            tau_cols.append(col)
    if "mean" in df_overlay.columns:
        tau_cols.append("mean")

    meta_cols = [
        "fecha_t", "cierre_fecha", "h_cierre", "n_inciertos",
        "peor_total", "retiro_pasado", "peor_restante", "q_tau_acum",
        "factor_f", "overlay_activo", "razon_no_activo",
    ]
    meta_slim = df_meta[[c for c in meta_cols if c in df_meta.columns]].copy()
    meta_slim["fecha_t"] = pd.to_datetime(meta_slim["fecha_t"])

    key_cols = ["banco", "fold", "fecha_t", "h"]

    df_post = df_overlay.copy()
    df_post["fecha_t"] = pd.to_datetime(df_post["fecha_t"])

    if df_base is not None:
        df_pre = df_base.copy()
        df_pre["fecha_t"] = pd.to_datetime(df_pre["fecha_t"])

        # Renombrar columnas de base con sufijo _pre
        rename_pre = {c: f"{c}_pre" for c in tau_cols if c in df_pre.columns}
        df_pre = df_pre.rename(columns=rename_pre)

        # Renombrar columnas de overlay con sufijo _post
        rename_post = {c: f"{c}_post" for c in tau_cols if c in df_post.columns}
        df_post = df_post.rename(columns=rename_post)

        base_keys = [c for c in key_cols if c in df_pre.columns]
        post_keys = [c for c in key_cols if c in df_post.columns]
        df_merged = df_pre[base_keys + list(rename_pre.values())].merge(
            df_post[post_keys + list(rename_post.values())],
            on=[c for c in key_cols if c in base_keys and c in post_keys],
            how="outer",
        )

        # Columnas target: preferir post (tienen target real)
        if "target" in df_post.columns:
            df_merged = df_merged.merge(
                df_post[[c for c in key_cols if c in post_keys] + ["target"]],
                on=[c for c in key_cols if c in post_keys],
                how="left",
            )

        # Deltas
        for tc in tau_cols:
            pre_col  = f"{tc}_pre"
            post_col = f"{tc}_post"
            if pre_col in df_merged.columns and post_col in df_merged.columns:
                df_merged[f"{tc}_delta"] = df_merged[post_col] - df_merged[pre_col]
    else:
        # Sin base: solo overlay
        df_merged = df_post.copy()

    # Merge meta
    merge_key = [c for c in ["fecha_t"] if c in df_merged.columns]
    df_merged = df_merged.merge(meta_slim, on=merge_key, how="left", suffixes=("", "_meta"))

    df_merged = df_merged.sort_values(["fecha_t", "h"] if "h" in df_merged.columns else ["fecha_t"])
    df_merged["fecha_t"] = df_merged["fecha_t"].dt.date
    for col in ["cierre_fecha"]:
        if col in df_merged.columns:
            df_merged[col] = pd.to_datetime(df_merged[col]).dt.date

    return df_merged


def _sheet_factor_evolucion(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Muestra cómo evoluciona el factor_f a lo largo del tiempo para cada cierre trimestral.
    Una fila por fecha_t, columnas: cierre_fecha, factor_f, overlay_activo, razon_no_activo,
    retiro_pasado, peor_restante, q_tau_acum — útil para ver la progresión dentro de
    la ventana de 7 DH.
    """
    cols = [
        "fecha_t", "cierre_fecha", "h_cierre", "n_inciertos",
        "peor_total", "retiro_pasado", "peor_restante",
        "q_tau_acum", "factor_f", "overlay_activo", "razon_no_activo",
    ]
    df = df_meta[[c for c in cols if c in df_meta.columns]].copy()
    df["fecha_t"]      = pd.to_datetime(df["fecha_t"]).dt.date
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"]).dt.date
    df = df.sort_values(["cierre_fecha", "fecha_t"])
    return df


def _sheet_pivot_q05(df_preds: pd.DataFrame) -> pd.DataFrame:
    if "q05" not in df_preds.columns:
        return pd.DataFrame({"nota": ["Columna q05 no disponible en el parquet"]})
    df = df_preds[["fecha_t", "h", "q05"]].copy()
    df["fecha_t"] = pd.to_datetime(df["fecha_t"]).dt.date
    pivot = df.pivot_table(index="fecha_t", columns="h", values="q05", aggfunc="mean")
    pivot.columns = [f"h{int(c)}" for c in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


def _sheet_pivot_factor(df_meta: pd.DataFrame) -> pd.DataFrame:
    df = df_meta[["fecha_t", "cierre_fecha", "factor_f", "overlay_activo"]].copy()
    df["fecha_t"]      = pd.to_datetime(df["fecha_t"]).dt.date
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"]).dt.date
    pivot = df.pivot_table(
        index="fecha_t", columns="cierre_fecha", values="factor_f", aggfunc="first"
    ).reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


def _sheet_calendario_bday() -> pd.DataFrame:
    """
    Lista todos los días hábiles de 2025 y 2026 según el calendario combinado
    PER+USA (mismo BDAY_PE que usa step005). Incluye columna de día de semana
    y una bandera que indica si ese día es feriado en alguno de los dos países.
    """
    # Todos los días calendario 2025-2026
    todos = pd.date_range(start="2025-01-01", end="2026-12-31", freq="D")

    # Días hábiles según BDAY_PE
    bdays = pd.bdate_range(start="2025-01-01", end="2026-12-31", freq=BDAY_PE)
    bdays_set = set(bdays)

    # Feriados en el rango
    feriados_rango = set(
        f for f in _FERIADOS_PEUSA
        if pd.Timestamp("2025-01-01") <= f <= pd.Timestamp("2026-12-31")
    )

    rows = []
    for d in bdays:
        rows.append({
            "fecha":        d.date(),
            "dia_semana":   d.day_name(),
            "trimestre":    d.quarter,
            "es_feriado":   d in feriados_rango,
            "es_fin_trim":  (d.month in [3, 6, 9, 12] and (d + BDAY_PE).month != d.month),
        })

    df = pd.DataFrame(rows)
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Agregar nombre del feriado si está disponible
    try:
        import holidays as _hlib
        _pe = _hlib.Peru(years=[2025, 2026])
        _us = _hlib.UnitedStates(years=[2025, 2026])
        def _nombre(d):
            ts = d.date() if hasattr(d, "date") else d
            return _pe.get(ts) or _us.get(ts) or ""
        df["nombre_feriado"] = df["fecha"].apply(_nombre)
    except ImportError:
        df["nombre_feriado"] = ""

    df["fecha"] = df["fecha"].dt.date
    return df


def _sheet_metodologia() -> pd.DataFrame:
    rows = [
        ("Concepto", "Overlay Sobreencaje — Lógica vigente"),
        ("", ""),
        ("Zona FUERA de ventana", "fecha_t < inicio de los últimos 7 DH del trimestre"),
        ("  Fórmula", "f = peor_total / |Q[TAU]_acum_7días|"),
        ("  mask_ventana", "h_inicio..h_cierre (los 7 DH completos de la ventana)"),
        ("  peor_usado", "peor_total (sin netting; ningún retiro conocido aún)"),
        ("", ""),
        ("Zona DENTRO de ventana", "fecha_t >= inicio de los últimos 7 DH del trimestre"),
        ("  Fórmula", "f = peor_usado / |Q[TAU]_acum_restante|"),
        ("  mask_ventana", "h=1..h_cierre (días futuros restantes en la ventana)"),
        ("  retiro_pasado", "Flujos negativos YA realizados en la ventana hasta fecha_t"),
        ("  peor_usado", "max(0, peor_total − retiro_pasado)"),
        ("  Comparabilidad", "Numerador y denominador se reducen al mismo ritmo → f estable"),
        ("", ""),
        ("peor_total", "Máximo retiro neto acumulado en 7 DH, histórico por banco"),
        ("|Q[TAU]_acum|", "Suma del cuantil τ=5% sobre mask_ventana"),
        ("n_inciertos", "Número de días en mask_ventana"),
        ("", ""),
        ("T+2", "Desactivado por el momento (N=0)"),
        ("", ""),
        ("Factor = 1 (sin ajuste)", "Si f ≤ 1 o Q[TAU]_acum ≥ 0"),
        ("razon_no_activo", "q_tau>=0_o_nulo | f<=1 | peor_consumido"),
        ("", ""),
        ("Últimos 2 DH del trimestre", ""),
        ("  Cierre trimestral (Sep 30, Dic 31…)", "Skipped: ventana ya venció (prox BDay en otro mes)"),
        ("  Penúltimo DH (Sep 27, Dic 30…)", "Q05[h=1] >= 0 → modelo predice flujo positivo, f=1.0"),
    ]
    return pd.DataFrame(rows, columns=["Campo", "Descripción"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n=== Diagnóstico Cuantiles + Overlay: {BANCO} ===\n")

    df_overlay, df_meta, df_base = _cargar_datos(BANCO, FECHA_HOY)

    ruta_out = DIR_OUTPUT / f"diagnostico_cuantiles_{BANCO}.xlsx"

    with pd.ExcelWriter(ruta_out, engine="openpyxl") as writer:
        _sheet_metodologia().to_excel(writer, sheet_name="Metodologia", index=False)
        _sheet_factor(df_meta).to_excel(writer, sheet_name="Factor_overlay", index=False)
        _sheet_cuantiles(df_overlay, df_meta).to_excel(writer, sheet_name="Cuantiles_origen", index=False)
        _sheet_pre_post(df_overlay, df_meta, df_base).to_excel(writer, sheet_name="Pre_vs_Post", index=False)
        _sheet_factor_evolucion(df_meta).to_excel(writer, sheet_name="Factor_evolucion", index=False)
        _sheet_pivot_q05(df_overlay).to_excel(writer, sheet_name="Pivot_Q05", index=False)
        _sheet_pivot_factor(df_meta).to_excel(writer, sheet_name="Pivot_Factor", index=False)
        _sheet_calendario_bday().to_excel(writer, sheet_name="Calendario_BDay", index=False)

    print(f"\n[OK] Excel de diagnóstico generado correctamente:")
    print(f"     {ruta_out}")
    print(
        f"     Sheets: Metodologia | Factor_overlay | Cuantiles_origen | "
        f"Pre_vs_Post | Factor_evolucion | Pivot_Q05 | Pivot_Factor | Calendario_BDay"
    )


if __name__ == "__main__":
    main()
