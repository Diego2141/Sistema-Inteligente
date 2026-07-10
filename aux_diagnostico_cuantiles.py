# -*- coding: utf-8 -*-
"""
aux_diagnostico_cuantiles.py
============================
Genera un Excel de diagnóstico que cruza:
  - overlay_meta_{banco}_{fecha}.csv   -> factor_f, peor_total, peor_restante, etc.
  - preds_overlay_{banco}_{fecha}.parquet -> cuantiles Q01/Q05/Q50/Q95/Q99 y mean
  - Ajuste_diario (saldos_retiros_bancos.xlsx) -> peor_total de referencia

El Excel permite validar visualmente que:
  1. El factor f tiene magnitud razonable (estructuralmente acotado).
  2. Los cuantiles proyectados se inflan al acercarse al cierre trimestral.
  3. La inflación es proporcional al peor_restante (netting correcto).

SHEETS
------
  "Factor_overlay"   : una fila por fecha_t con factor, razon_no_activo, etc.
  "Cuantiles_origen" : una fila por (fecha_t, h) con Q01/Q05/Q50/Q95/Q99 y target
  "Pivot_Q05"        : tabla cruzada fecha_t × h del Q05 (para heatmap visual)
  "Pivot_Factor"     : tabla cruzada fecha_t × cierre con factor_f asignado

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

# Cuantiles a mostrar en la hoja Cuantiles_origen
TAUS_MOSTRAR = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _encontrar_ultimo(patron: str) -> Path:
    archivos = sorted(glob.glob(patron))
    if not archivos:
        raise FileNotFoundError(f"No se encontró archivo con patrón: {patron}")
    return Path(archivos[-1])


def _cargar_datos(banco: str, fecha_hoy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fecha_hoy == "AUTO":
        ruta_preds = _encontrar_ultimo(str(DIR_STEP005 / f"preds_overlay_{banco}_*.parquet"))
        ruta_meta  = _encontrar_ultimo(str(DIR_STEP005 / f"overlay_meta_{banco}_*.csv"))
    else:
        ruta_preds = DIR_STEP005 / f"preds_overlay_{banco}_{fecha_hoy}.parquet"
        ruta_meta  = DIR_STEP005 / f"overlay_meta_{banco}_{fecha_hoy}.csv"

    df_preds = pd.read_parquet(ruta_preds)
    df_meta  = pd.read_csv(ruta_meta, parse_dates=["fecha_t", "cierre_fecha"])

    print(f"Predicciones : {ruta_preds.name}  ({len(df_preds):,} filas)")
    print(f"Metadata     : {ruta_meta.name}   ({len(df_meta):,} filas)")
    return df_preds, df_meta


# ── Sheet builders ────────────────────────────────────────────────────────────

def _sheet_factor(df_meta: pd.DataFrame) -> pd.DataFrame:
    cols_orden = [
        "fecha_t", "cierre_fecha", "h_cierre", "n_inciertos",
        "peor_total", "retiro_conocido", "peor_restante",
        "q_tau_acum", "factor_f", "overlay_activo", "razon_no_activo",
    ]
    cols = [c for c in cols_orden if c in df_meta.columns]
    df = df_meta[cols].copy().sort_values(["fecha_t"])
    df["fecha_t"]     = pd.to_datetime(df["fecha_t"]).dt.date
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"]).dt.date
    return df


def _sheet_cuantiles(df_preds: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    df = df_preds.copy()
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])

    # Columnas de cuantiles disponibles en el parquet
    tau_cols = []
    for tau in TAUS_MOSTRAR:
        col = f"q{int(tau * 100):02d}"
        if col in df.columns:
            tau_cols.append(col)
    if "mean" in df.columns:
        tau_cols.append("mean")

    # Merge con factor del overlay (uno por fecha_t)
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
    df["fecha_t"]     = pd.to_datetime(df["fecha_t"]).dt.date
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"]).dt.date
    # Una fila por fecha_t, una columna por cierre (puede haber uno o dos cierres activos)
    pivot = df.pivot_table(
        index="fecha_t", columns="cierre_fecha", values="factor_f", aggfunc="first"
    ).reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


def _sheet_metodologia() -> pd.DataFrame:
    rows = [
        ("Concepto", "Overlay BBVA Sobreencaje"),
        ("", ""),
        ("Fórmula factor", "f = peor_restante × n_inciertos / (VDH × |Q[TAU]_acum|)"),
        ("peor_total", "Máximo retiro neto acumulado en 7 días hábiles, histórico por banco"),
        ("retiro_conocido", "Suma de retiros reales ya ocurridos en la ventana (solo negativos)"),
        ("peor_restante", "max(0, peor_total - retiro_conocido)"),
        ("n_inciertos", "Días hábiles dentro de la ventana que aún son inciertos"),
        ("VDH", "Ventana en días hábiles = 7"),
        ("|Q[TAU]_acum|", "Suma del cuantil τ=5% para los días inciertos de la ventana"),
        ("", ""),
        ("", ""),
        ("T+2 conocimiento anticipado", "Los bancos notifican 2 días hábiles antes."),
        ("", "h=1,2 son conocidos; si h_cierre ≤ 2, saltar al siguiente trimestre."),
        ("", ""),
        ("Factor = 1 (sin ajuste)", "Si f ≤ 1 o Q[TAU]_acum ≥ 0"),
        ("razon_no_activo", "q_tau>=0_o_nulo: cuantil proyectado positivo o nulo"),
        ("", "f<=1: overlay no necesario (modelo ya conservador)"),
        ("", "h_cierre>MAX: cierre demasiado lejano para activar"),
    ]
    return pd.DataFrame(rows, columns=["Campo", "Descripción"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n=== Diagnóstico Cuantiles + Overlay: {BANCO} ===\n")

    df_preds, df_meta = _cargar_datos(BANCO, FECHA_HOY)

    ruta_out = DIR_OUTPUT / f"diagnostico_cuantiles_{BANCO}.xlsx"

    with pd.ExcelWriter(ruta_out, engine="openpyxl") as writer:
        _sheet_metodologia().to_excel(writer, sheet_name="Metodologia", index=False)
        _sheet_factor(df_meta).to_excel(writer, sheet_name="Factor_overlay", index=False)
        _sheet_cuantiles(df_preds, df_meta).to_excel(writer, sheet_name="Cuantiles_origen", index=False)
        _sheet_pivot_q05(df_preds).to_excel(writer, sheet_name="Pivot_Q05", index=False)
        _sheet_pivot_factor(df_meta).to_excel(writer, sheet_name="Pivot_Factor", index=False)

    print(f"\nExcel guardado en:\n  {ruta_out}")


if __name__ == "__main__":
    main()
