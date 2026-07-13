# -*- coding: utf-8 -*-
"""
aux_diagnostico_cuantiles.py
============================
Genera un Excel de diagnóstico que cruza:
  - overlay_meta_{banco}_{fecha}.csv   -> factor_f por (fecha_t, cierre_fecha)
  - preds_overlay_{banco}_{fecha}.parquet -> cuantiles Q01/Q05/Q50/Q95/Q99 y mean
  - preds_base_{banco}_{fecha}.parquet    -> cuantiles PRE-overlay (sin ajuste)

LÓGICA DE CIERRES
-----------------
Para cada fecha_t (fecha de observación), el overlay busca hasta 2 cierres
trimestrales dentro del horizonte de proyección h=1..75:
  - Cierre C1: el más cercano (ej. Sep 30 si fecha_t es Sep 18)
  - Cierre C2: el siguiente (ej. Dic 31 si fecha_t es Sep 18) — puede no existir
                si el cierre más cercano está suficientemente lejos.

Cada cierre tiene su propia ventana de 7 DH, su propio factor f (IN/OUT), y
una fila independiente en overlay_meta. Las ventanas no se solapan.

Edge case: si h_cierre_C2 > 75, la ventana cae fuera del horizonte de predicción;
el overlay aplica el factor solo a h=75 (fallback) — se reporta en Diagnostico.

SHEETS
------
  "Metodologia"      : descripción de la lógica multi-cierre IN/OUT
  "Diagnostico"      : validación: n_cierres por fecha_t, edge cases, resumen
  "Resumen_Factores" : una fila por fecha_t con cierre más próximo y factores efectivos (C1/C2/EXT)
  "Factor_C1_C2"     : una fila por fecha_t con factor C1 y C2 lado a lado (detalle completo)
  "Factor_overlay"   : una fila por (fecha_t, cierre_fecha) — formato largo
  "Factor_evolucion" : evolución del factor por cierre_fecha a través del tiempo
  "Cuantiles_origen" : una fila por (fecha_t, h) con cuantiles post-overlay
  "Pre_vs_Post"      : pre y post ajuste por (fecha_t, h) con delta
  "Pivot_Q05"        : tabla cruzada fecha_t × h del Q05
  "Pivot_Factor"     : tabla cruzada fecha_t × cierre con factor_f
  "Calendario_BDay"  : días hábiles 2025-2026 según calendario PER+USA

USO
---
  1. Ejecutar step005 con OVERLAY_SOBREENCAJE = True
  2. Ajustar BANCO y FECHA_HOY (o dejar AUTO)
  3. python aux_diagnostico_cuantiles.py
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
FECHA_HOY = "AUTO"

TAUS_MOSTRAR   = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
H_MAX_OVERLAY  = 75   # horizonte máximo de predicción (h_min=2 en la matriz)

# Calendario PER+USA (mismo que step005)
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

    print(f"[OK] Predicciones overlay : {ruta_overlay.name}  ({len(df_overlay):,} filas)")
    print(f"[OK] Metadata             : {ruta_meta.name}   ({len(df_meta):,} filas)")
    print(f"     Filas meta por fecha_t: min={df_meta.groupby('fecha_t').size().min()} "
          f"max={df_meta.groupby('fecha_t').size().max()}")

    df_base = None
    try:
        if fecha_hoy == "AUTO":
            ruta_base = _encontrar_ultimo(str(DIR_STEP005 / "**" / f"preds_base_{banco}_*.parquet"))
        else:
            ruta_base = _encontrar_ultimo(str(DIR_STEP005 / "**" / f"preds_base_{banco}_{fecha_hoy}.parquet"))
        df_base = pd.read_parquet(ruta_base)
        print(f"[OK] Predicciones base    : {ruta_base.name}  ({len(df_base):,} filas)")
    except FileNotFoundError:
        print("[WARN] preds_base_*.parquet no encontrado — hoja Pre_vs_Post usará solo post-overlay")

    return df_overlay, df_meta, df_base


def _meta_por_cierre(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Asigna a cada fila de meta el rango de h que le corresponde (h_inicio_mask..h_cierre)."""
    df = df_meta.copy()
    df["fecha_t"]      = pd.to_datetime(df["fecha_t"])
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"])
    # h_inicio_mask: dentro=1, fuera=h_cierre-6 (approx OVERLAY_VENTANA_DH=7)
    # Lo inferimos de n_inciertos y h_cierre si están disponibles
    if "n_inciertos" in df.columns and "h_cierre" in df.columns:
        df["h_inicio_mask"] = df["h_cierre"] - df["n_inciertos"] + 1
    return df


# ── Sheet builders ────────────────────────────────────────────────────────────

def _sheet_diagnostico(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Valida la lógica multi-cierre:
    - Confirma que cada fecha_t tiene 1 o 2 cierres (nunca 3+)
    - Identifica edge cases: h_cierre > H_MAX (cierre fuera del horizonte de pred.)
    - Muestra resumen de cuántas fechas tienen 1 vs 2 cierres
    """
    df = df_meta.copy()
    df["fecha_t"]      = pd.to_datetime(df["fecha_t"])
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"])

    g = df.groupby("fecha_t")
    n_cierres   = g["cierre_fecha"].nunique().rename("n_cierres")
    h_max_obs   = g["h_cierre"].max().rename("h_cierre_max")
    activos     = g["overlay_activo"].any().rename("alguno_activo") if "overlay_activo" in df.columns else None

    resumen_ft = pd.DataFrame({"n_cierres": n_cierres, "h_cierre_max": h_max_obs})
    if activos is not None:
        resumen_ft["alguno_activo"] = activos
    resumen_ft = resumen_ft.reset_index()

    # Edge case: h_cierre > H_MAX → ventana fuera del horizonte
    edge = resumen_ft[resumen_ft["h_cierre_max"] > H_MAX_OVERLAY].copy()

    rows = [
        ("VALIDACIÓN LÓGICA MULTI-CIERRE", ""),
        ("", ""),
        ("Total fecha_t distintas", str(resumen_ft["fecha_t"].nunique())),
        ("fecha_t con 1 cierre",   str((resumen_ft["n_cierres"] == 1).sum())),
        ("fecha_t con 2 cierres",  str((resumen_ft["n_cierres"] == 2).sum())),
        ("fecha_t con 3+ cierres", str((resumen_ft["n_cierres"] >= 3).sum())),
        ("Max cierres por fecha_t",str(resumen_ft["n_cierres"].max())),
        ("", ""),
        ("EDGE CASE: h_cierre > 75", f"{len(edge)} fechas con cierre fuera del horizonte pred."),
        ("  Interpretación",
         "El factor se aplica a h=75 (fallback). Ocurre cuando el 2do cierre "
         "está a >75 BDays de fecha_t. Normal para dates muy cercanas al cierre anterior."),
        ("", ""),
        ("RESULTADO", "OK — máximo 2 cierres por fecha_t" if resumen_ft["n_cierres"].max() <= 2
                      else "ALERTA — hay fecha_t con 3+ cierres, revisar"),
    ]
    df_resumen = pd.DataFrame(rows, columns=["Campo", "Valor"])

    # Agregar detalle de edge cases si los hay
    if not edge.empty:
        edge_rows = [("", ""), ("DETALLE EDGE CASES (h_cierre > 75)", "")]
        for _, r in edge.head(20).iterrows():
            edge_rows.append((str(r["fecha_t"].date()), f"h_cierre_max={r['h_cierre_max']}"))
        df_resumen = pd.concat(
            [df_resumen, pd.DataFrame(edge_rows, columns=["Campo", "Valor"])],
            ignore_index=True
        )

    return df_resumen


def _sheet_factor_c1_c2(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot: una fila por fecha_t con los factores C1 (cierre más cercano) y
    C2 (cierre siguiente) lado a lado. Muestra h_cierre, factor_f, overlay_activo
    y razon_no_activo para cada cierre.
    """
    df = df_meta.copy()
    df["fecha_t"]      = pd.to_datetime(df["fecha_t"])
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"])

    # Ordenar para asignar rank 1 (C1=más cercano) y 2 (C2=siguiente)
    df = df.sort_values(["fecha_t", "cierre_fecha"])
    df["rank_cierre"] = df.groupby("fecha_t").cumcount() + 1  # 1=C1, 2=C2

    cols_mostrar = [c for c in
        ["h_cierre", "factor_f", "overlay_activo", "razon_no_activo",
         "peor_total", "retiro_pasado", "peor_restante", "q_tau_acum", "n_inciertos"]
        if c in df.columns]

    pivots = []
    for rank, label in [(1, "C1"), (2, "C2")]:
        sub = df[df["rank_cierre"] == rank][["fecha_t", "cierre_fecha"] + cols_mostrar].copy()
        sub = sub.rename(columns={c: f"{label}_{c}" for c in cols_mostrar})
        sub = sub.rename(columns={"cierre_fecha": f"{label}_cierre_fecha"})
        pivots.append(sub)

    result = pivots[0]
    if len(pivots) > 1:
        result = result.merge(pivots[1], on="fecha_t", how="left")

    result = result.sort_values("fecha_t")
    result["fecha_t"] = result["fecha_t"].dt.date
    for col in [c for c in result.columns if "cierre_fecha" in c]:
        result[col] = pd.to_datetime(result[col]).dt.date

    return result


def _sheet_factor(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Formato largo: una fila por (fecha_t, cierre_fecha)."""
    cols_orden = [
        "fecha_t", "cierre_fecha", "h_cierre", "n_inciertos",
        "peor_total", "retiro_pasado", "peor_restante",
        "q_tau_acum", "factor_f", "overlay_activo", "razon_no_activo",
    ]
    cols = [c for c in cols_orden if c in df_meta.columns]
    df = df_meta[cols].copy().sort_values(["fecha_t", "cierre_fecha"])
    df["fecha_t"]      = pd.to_datetime(df["fecha_t"]).dt.date
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"]).dt.date
    return df


def _sheet_factor_evolucion(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Evolución del factor por cierre_fecha a través del tiempo (útil para ver IN/OUT)."""
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


def _sheet_cuantiles(df_preds: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Cuantiles post-overlay por (fecha_t, h). Para el merge con meta, usa el
    cierre cuya ventana contiene h (h_inicio_mask <= h <= h_cierre).
    Si h no cae en ninguna ventana, deja las columnas de meta en blanco.
    """
    df = df_preds.copy()
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])

    tau_cols = [f"q{int(t*100):02d}" for t in TAUS_MOSTRAR if f"q{int(t*100):02d}" in df.columns]
    if "mean" in df.columns:
        tau_cols.append("mean")

    # Meta con h_inicio_mask inferido
    meta = _meta_por_cierre(df_meta).copy()
    meta_cols = [c for c in
        ["fecha_t", "cierre_fecha", "h_inicio_mask", "h_cierre",
         "factor_f", "overlay_activo", "razon_no_activo", "n_inciertos"]
        if c in meta.columns or c == "h_inicio_mask"]

    # Join (fecha_t, h) con la fila de meta cuya ventana contiene h
    def _buscar_meta(row):
        ft = row["fecha_t"]
        h  = row["h"]
        candidatos = meta[
            (meta["fecha_t"] == ft) &
            (meta.get("h_inicio_mask", meta["h_cierre"] - meta.get("n_inciertos", 7) + 1) <= h) &
            (meta["h_cierre"] >= h)
        ] if "h_inicio_mask" in meta.columns else meta[
            (meta["fecha_t"] == ft) &
            (meta["h_cierre"] - meta.get("n_inciertos", 7) + 1 <= h) &
            (meta["h_cierre"] >= h)
        ]
        if candidatos.empty:
            return pd.Series({"cierre_fecha": pd.NaT, "factor_f": np.nan,
                               "overlay_activo": False, "razon_no_activo": ""})
        r = candidatos.iloc[0]
        return pd.Series({
            "cierre_fecha":   r["cierre_fecha"],
            "factor_f":       r.get("factor_f", np.nan),
            "overlay_activo": r.get("overlay_activo", False),
            "razon_no_activo": r.get("razon_no_activo", ""),
        })

    # Para performance, hacer merge directo en lugar de apply fila a fila
    # Expandir meta a rango de h por fila
    meta_h_rows = []
    for _, m in meta.iterrows():
        if "h_inicio_mask" in meta.columns:
            h_ini = int(m["h_inicio_mask"]) if not pd.isna(m.get("h_inicio_mask")) else 1
        else:
            h_ini = max(1, int(m["h_cierre"]) - int(m.get("n_inciertos", 7)) + 1)
        h_fin = int(m["h_cierre"])
        for h in range(h_ini, h_fin + 1):
            meta_h_rows.append({
                "fecha_t":        m["fecha_t"],
                "h":              h,
                "cierre_fecha":   m["cierre_fecha"],
                "factor_f":       m.get("factor_f", np.nan),
                "overlay_activo": m.get("overlay_activo", False),
                "razon_no_activo": m.get("razon_no_activo", ""),
            })
    df_meta_h = pd.DataFrame(meta_h_rows) if meta_h_rows else pd.DataFrame(
        columns=["fecha_t", "h", "cierre_fecha", "factor_f", "overlay_activo", "razon_no_activo"])

    df = df.merge(df_meta_h, on=["fecha_t", "h"], how="left")

    cols_base = ["banco", "fold", "fecha_t", "h", "target"]
    cols = [c for c in cols_base if c in df.columns] + tau_cols + [
        "cierre_fecha", "factor_f", "overlay_activo", "razon_no_activo"
    ]
    df = df[[c for c in cols if c in df.columns]].sort_values(["fecha_t", "h"])
    df["fecha_t"] = df["fecha_t"].dt.date
    if "cierre_fecha" in df.columns:
        df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"]).dt.date
    return df


def _sheet_pre_post(df_overlay: pd.DataFrame, df_meta: pd.DataFrame,
                    df_base: pd.DataFrame | None) -> pd.DataFrame:
    """Pre vs post overlay por (fecha_t, h) con delta por cuantil."""
    tau_cols = [f"q{int(t*100):02d}" for t in TAUS_MOSTRAR if f"q{int(t*100):02d}" in df_overlay.columns]
    if "mean" in df_overlay.columns:
        tau_cols.append("mean")

    # Meta resumida por (fecha_t, h): mismo enfoque que _sheet_cuantiles
    meta = _meta_por_cierre(df_meta).copy()
    meta_h_rows = []
    for _, m in meta.iterrows():
        h_ini = int(m["h_inicio_mask"]) if "h_inicio_mask" in meta.columns and not pd.isna(m.get("h_inicio_mask")) \
                else max(1, int(m["h_cierre"]) - int(m.get("n_inciertos", 7)) + 1)
        h_fin = int(m["h_cierre"])
        for h in range(h_ini, h_fin + 1):
            meta_h_rows.append({
                "fecha_t":        m["fecha_t"],
                "h":              h,
                "cierre_fecha":   m["cierre_fecha"],
                "factor_f":       m.get("factor_f", np.nan),
                "retiro_pasado":  m.get("retiro_pasado", np.nan),
                "peor_restante":  m.get("peor_restante", np.nan),
                "q_tau_acum":     m.get("q_tau_acum", np.nan),
                "overlay_activo": m.get("overlay_activo", False),
            })
    df_meta_h = pd.DataFrame(meta_h_rows) if meta_h_rows else pd.DataFrame()

    key_cols = ["banco", "fold", "fecha_t", "h"]
    df_post = df_overlay.copy()
    df_post["fecha_t"] = pd.to_datetime(df_post["fecha_t"])

    if df_base is not None:
        df_pre = df_base.copy()
        df_pre["fecha_t"] = pd.to_datetime(df_pre["fecha_t"])
        rename_pre  = {c: f"{c}_pre"  for c in tau_cols if c in df_pre.columns}
        rename_post = {c: f"{c}_post" for c in tau_cols if c in df_post.columns}
        df_pre  = df_pre.rename(columns=rename_pre)
        df_post = df_post.rename(columns=rename_post)
        jk = [c for c in key_cols if c in df_pre.columns and c in df_post.columns]
        df_merged = df_pre[jk + list(rename_pre.values())].merge(
            df_post[jk + list(rename_post.values())], on=jk, how="outer")
        if "target" in df_post.columns:
            df_merged = df_merged.merge(df_post[jk + ["target"]], on=jk, how="left")
        for tc in tau_cols:
            if f"{tc}_pre" in df_merged.columns and f"{tc}_post" in df_merged.columns:
                df_merged[f"{tc}_delta"] = df_merged[f"{tc}_post"] - df_merged[f"{tc}_pre"]
    else:
        df_merged = df_post.copy()

    if not df_meta_h.empty:
        df_merged = df_merged.merge(df_meta_h, on=["fecha_t", "h"], how="left")

    df_merged = df_merged.sort_values(["fecha_t", "h"] if "h" in df_merged.columns else ["fecha_t"])
    df_merged["fecha_t"] = df_merged["fecha_t"].dt.date
    if "cierre_fecha" in df_merged.columns:
        df_merged["cierre_fecha"] = pd.to_datetime(df_merged["cierre_fecha"]).dt.date
    return df_merged


def _sheet_resumen_factores(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila por fecha_t mostrando el cierre más próximo y los factores
    efectivamente aplicados (C1, C2 o EXT).

    Columnas:
      fecha_t, cierre_proximo, h_cierre_prox, zona_C1,
      factor_C1, c1_activo, c1_razon,
      cierre_siguiente, h_cierre_sig, zona_C2,
      factor_C2, c2_activo, c2_razon
    """
    df = df_meta.copy()
    df["fecha_t"]      = pd.to_datetime(df["fecha_t"])
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"])
    df = df.sort_values(["fecha_t", "cierre_fecha"])
    df["rank"] = df.groupby("fecha_t").cumcount() + 1   # 1=C1, 2=C2

    def _zona(row):
        """Inferir IN / OUT / EXT desde n_inciertos y razon_no_activo."""
        razon = str(row.get("razon_no_activo", ""))
        if razon == "ext_f_c1":
            return "EXT"
        n   = row.get("n_inciertos", 7)
        h_c = row.get("h_cierre", 0)
        # si n_inciertos == h_cierre la ventana empieza en h=1 → estamos DENTRO
        if pd.notna(n) and pd.notna(h_c) and int(n) == int(h_c):
            return "IN"
        return "OUT"

    c1 = df[df["rank"] == 1].set_index("fecha_t")
    c2 = df[df["rank"] == 2].set_index("fecha_t")

    rows = []
    for ft in sorted(df["fecha_t"].unique()):
        r1 = c1.loc[ft] if ft in c1.index else None
        r2 = c2.loc[ft] if ft in c2.index else None

        # C1
        if r1 is not None:
            activo_c1 = bool(r1.get("overlay_activo", False))
            factor_c1 = float(r1["factor_f"]) if activo_c1 and pd.notna(r1.get("factor_f")) else np.nan
            zona_c1   = _zona(r1)
            razon_c1  = "" if activo_c1 else str(r1.get("razon_no_activo", ""))
            cierre1   = r1["cierre_fecha"]
            h_c1      = int(r1["h_cierre"]) if pd.notna(r1.get("h_cierre")) else np.nan
        else:
            activo_c1, factor_c1, zona_c1, razon_c1 = False, np.nan, "", ""
            cierre1, h_c1 = pd.NaT, np.nan

        # C2 (puede ser activo, EXT, o ninguno)
        if r2 is not None:
            activo_c2 = bool(r2.get("overlay_activo", False))
            razon_c2  = str(r2.get("razon_no_activo", ""))
            zona_c2   = _zona(r2)
            if activo_c2 and pd.notna(r2.get("factor_f")):
                factor_c2 = float(r2["factor_f"])
            elif razon_c2 == "ext_f_c1":
                # extensión: el factor aplicado fue el de C1
                factor_c2 = float(r2["factor_f"]) if pd.notna(r2.get("factor_f")) else factor_c1
                activo_c2 = True   # el factor SÍ se aplicó (como extensión)
            else:
                factor_c2 = np.nan
            cierre2 = r2["cierre_fecha"]
            h_c2    = int(r2["h_cierre"]) if pd.notna(r2.get("h_cierre")) else np.nan
        else:
            activo_c2, factor_c2, zona_c2, razon_c2 = False, np.nan, "", ""
            cierre2, h_c2 = pd.NaT, np.nan

        rows.append({
            "fecha_t":          pd.Timestamp(ft).date(),
            "cierre_proximo":   cierre1.date() if pd.notna(cierre1) else None,
            "h_cierre_prox":    h_c1,
            "zona_C1":          zona_c1,
            "factor_C1":        round(factor_c1, 4) if pd.notna(factor_c1) else np.nan,
            "c1_activo":        activo_c1,
            "c1_razon":         razon_c1,
            "cierre_siguiente": cierre2.date() if pd.notna(cierre2) else None,
            "h_cierre_sig":     h_c2,
            "zona_C2":          zona_c2,
            "factor_C2":        round(factor_c2, 4) if pd.notna(factor_c2) else np.nan,
            "c2_activo":        activo_c2,
            "c2_razon":         "" if activo_c2 else razon_c2,
        })

    return pd.DataFrame(rows)


def _sheet_pivot_q05(df_preds: pd.DataFrame) -> pd.DataFrame:
    if "q05" not in df_preds.columns:
        return pd.DataFrame({"nota": ["Columna q05 no disponible"]})
    df = df_preds[["fecha_t", "h", "q05"]].copy()
    df["fecha_t"] = pd.to_datetime(df["fecha_t"]).dt.date
    pivot = df.pivot_table(index="fecha_t", columns="h", values="q05", aggfunc="mean")
    pivot.columns = [f"h{int(c)}" for c in pivot.columns]
    return pivot.reset_index()


def _sheet_pivot_factor(df_meta: pd.DataFrame) -> pd.DataFrame:
    df = df_meta[["fecha_t", "cierre_fecha", "factor_f"]].copy()
    df["fecha_t"]      = pd.to_datetime(df["fecha_t"]).dt.date
    df["cierre_fecha"] = pd.to_datetime(df["cierre_fecha"]).dt.date
    pivot = df.pivot_table(
        index="fecha_t", columns="cierre_fecha", values="factor_f", aggfunc="first"
    ).reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


def _sheet_calendario_bday() -> pd.DataFrame:
    bdays = pd.bdate_range(start="2025-01-01", end="2026-12-31", freq=BDAY_PE)
    feriados_rango = set(
        f for f in _FERIADOS_PEUSA
        if pd.Timestamp("2025-01-01") <= f <= pd.Timestamp("2026-12-31")
    )
    rows = []
    for d in bdays:
        rows.append({
            "fecha":       d.date(),
            "dia_semana":  d.day_name(),
            "trimestre":   d.quarter,
            "es_feriado":  d in feriados_rango,
            "es_fin_trim": (d.month in [3, 6, 9, 12] and (d + BDAY_PE).month != d.month),
        })
    df = pd.DataFrame(rows)
    df["fecha"] = pd.to_datetime(df["fecha"])
    try:
        import holidays as _hlib
        _pe = _hlib.Peru(years=[2025, 2026])
        _us = _hlib.UnitedStates(years=[2025, 2026])
        df["nombre_feriado"] = df["fecha"].apply(
            lambda d: _pe.get(d.date()) or _us.get(d.date()) or "")
    except ImportError:
        df["nombre_feriado"] = ""
    df["fecha"] = df["fecha"].dt.date
    return df


def _sheet_metodologia() -> pd.DataFrame:
    rows = [
        ("LÓGICA MULTI-CIERRE", "Overlay Sobreencaje — vigente"),
        ("", ""),
        ("Cierres por fecha_t", "1 o 2 (nunca 3): el horizonte de 75 DH cubre ≤2 trimestres"),
        ("  C1", "Cierre más cercano (ej. Sep 30 si fecha_t=Sep 18)"),
        ("  C2", "Siguiente cierre (ej. Dic 31 si fecha_t=Sep 18) — puede no existir"),
        ("  Ventanas", "No se solapan: h∈[h_ini_C1, h_cierre_C1] y h∈[h_ini_C2, h_cierre_C2]"),
        ("", ""),
        ("FUERA de ventana", "fecha_t < inicio de los últimos 7 DH del trimestre k"),
        ("  Factor",   "f_k = peor_total / |Q05_acum_7DH_k|"),
        ("  Ventana",  "h_inicio_k..h_cierre_k (7 DH completos)"),
        ("", ""),
        ("DENTRO de ventana", "fecha_t >= inicio de los últimos 7 DH del trimestre k"),
        ("  Factor",      "f_k = peor_restante_k / |Q05_acum_restante_k|"),
        ("  Ventana",     "h=1..h_cierre_k (días futuros restantes)"),
        ("  flujo_neto",  "Suma neta realizada en la ventana hasta fecha_t"),
        ("  peor_restante","max(0, peor_total + flujo_neto)"),
        ("  Nota",        "flujo_neto>0 (depósito) sube peor_restante; <0 (retiro) lo baja"),
        ("", ""),
        ("peor_total",  "Valor más reciente en Excel Ajuste_diario ≤ fecha_t"),
        ("  Si C2 sin datos", "Hereda peor_total de C1 (último disponible)"),
        ("", ""),
        ("EDGE CASE: h_cierre > 75", "Cierre fuera del horizonte de predicción"),
        ("  Qué ocurre", "mask_ventana vacía → fallback: factor aplicado solo a h=75"),
        ("  Cuándo",     "C2 muy lejano (ej. Q1 siguiente año desde fecha_t de diciembre)"),
        ("", ""),
        ("Factor = 1 (sin ajuste)", "Si f≤1, Q05_acum≥0, o peor_consumido"),
    ]
    return pd.DataFrame(rows, columns=["Campo", "Descripción"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n=== Diagnóstico Cuantiles + Overlay: {BANCO} ===\n")

    df_overlay, df_meta, df_base = _cargar_datos(BANCO, FECHA_HOY)

    # Diagnóstico en consola
    n_por_ft = df_meta.groupby("fecha_t")["cierre_fecha"].nunique()
    print(f"\n[DIAGNÓSTICO CIERRES]")
    print(f"  fecha_t con 1 cierre : {(n_por_ft==1).sum()}")
    print(f"  fecha_t con 2 cierres: {(n_por_ft==2).sum()}")
    print(f"  fecha_t con 3+       : {(n_por_ft>=3).sum()}  ← debe ser 0")
    n_edge = (df_meta["h_cierre"] > H_MAX_OVERLAY).sum()
    print(f"  filas con h_cierre > {H_MAX_OVERLAY} (edge case): {n_edge}")
    activos = df_meta[df_meta.get("overlay_activo", pd.Series(False, index=df_meta.index))]
    if "overlay_activo" in df_meta.columns:
        print(f"  filas con overlay activo : {df_meta['overlay_activo'].sum()}")

    ruta_out = DIR_OUTPUT / f"diagnostico_cuantiles_{BANCO}.xlsx"

    with pd.ExcelWriter(ruta_out, engine="openpyxl") as writer:
        _sheet_metodologia().to_excel(writer, sheet_name="Metodologia", index=False)
        _sheet_diagnostico(df_meta).to_excel(writer, sheet_name="Diagnostico", index=False)
        _sheet_resumen_factores(df_meta).to_excel(writer, sheet_name="Resumen_Factores", index=False)
        _sheet_factor_c1_c2(df_meta).to_excel(writer, sheet_name="Factor_C1_C2", index=False)
        _sheet_factor(df_meta).to_excel(writer, sheet_name="Factor_overlay", index=False)
        _sheet_factor_evolucion(df_meta).to_excel(writer, sheet_name="Factor_evolucion", index=False)
        _sheet_cuantiles(df_overlay, df_meta).to_excel(writer, sheet_name="Cuantiles_origen", index=False)
        _sheet_pre_post(df_overlay, df_meta, df_base).to_excel(writer, sheet_name="Pre_vs_Post", index=False)
        _sheet_pivot_q05(df_overlay).to_excel(writer, sheet_name="Pivot_Q05", index=False)
        _sheet_pivot_factor(df_meta).to_excel(writer, sheet_name="Pivot_Factor", index=False)
        _sheet_calendario_bday().to_excel(writer, sheet_name="Calendario_BDay", index=False)

    print(f"\n[OK] Excel generado: {ruta_out}")
    print("     Sheets: Metodologia | Diagnostico | Resumen_Factores | Factor_C1_C2 |")
    print("             Factor_overlay | Factor_evolucion | Cuantiles_origen | Pre_vs_Post |")
    print("             Pivot_Q05 | Pivot_Factor | Calendario_BDay")


if __name__ == "__main__":
    main()
