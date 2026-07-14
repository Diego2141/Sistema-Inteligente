# -*- coding: utf-8 -*-
"""
aux_exportar_fanchart_excel.py
Exporta a Excel los datos que usa aux_fanchart_video_xgb_qt.py para cada frame:
  - Cuantiles (Q01, Q05, Q50, Q95, Q99) de step005
  - Valores realizados (target = D_th - R_th)
  - Fecha pronosticada (fecha_th) para cada (fecha_t, h)

Hoja "base"    : predicciones puras (sin overlay sobreencaje)
Hoja "overlay" : predicciones con overlay sobreencaje aplicado
Hoja "resumen" : una fila por fecha_t con rangos de h y cobertura de realizados
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Rutas (idénticas a aux_fanchart_video_xgb_qt.py) ─────────────────────────
BASE_SISTEMA      = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ       = BASE_SISTEMA / "1. Data"   / "Clean"  / "matriz_features.parquet"
DIR_PREDS_STEP005 = BASE_SISTEMA / "2. Output" / "step005_wfcv_v3"
RUTA_OUTPUT       = BASE_SISTEMA / "2. Output" / "aux_fanchart_datos_excel" / "fanchart_datos.xlsx"

BANCO = "SISTEMA"

# ── Columnas de cuantiles disponibles en los parquets de step005 ──────────────
Q_COLS   = ["q01", "q05", "q50", "q95", "q99"]
META_COLS = ["fecha_t", "h", "banco", "fold"]
EXTRA_COLS = ["mean"]   # presente si el modelo exportó media


# ─────────────────────────────────────────────────────────────────────────────
def cargar_preds_step005(banco: str) -> dict:
    """
    Replica _cargar_preds_step005 del video:
    carga y concatena todos los parquets base/overlay de step005 para el banco.
    Filas duplicadas por (fecha_t, h) se resuelven conservando la más reciente.
    """
    result = {}
    for tipo in ("base", "overlay"):
        candidatos = sorted(DIR_PREDS_STEP005.glob(f"*/preds_{tipo}_{banco}_*.parquet"))
        if not candidatos:
            print(f"  [{tipo.upper()}] No se encontró preds_{tipo}_{banco}_*.parquet")
            result[tipo] = None
            continue

        dfs = []
        for ruta in candidatos:
            try:
                _df = pd.read_parquet(ruta)
                _df["fecha_t"] = pd.to_datetime(_df["fecha_t"])
                dfs.append(_df)
            except Exception as e:
                print(f"  [{tipo.upper()}] Error leyendo {ruta.name}: {e}")

        if not dfs:
            result[tipo] = None
            continue

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        key_cols = [c for c in ["fecha_t", "h", "banco", "fold"] if c in df.columns]
        df = (df.drop_duplicates(subset=key_cols, keep="last")
                .sort_values(["fecha_t", "h"])
                .reset_index(drop=True))

        n_fechas = df["fecha_t"].nunique()
        rng = f"{df['fecha_t'].min().date()} → {df['fecha_t'].max().date()}"
        print(f"  [{tipo.upper()}] {len(candidatos)} archivo(s) | "
              f"{len(df):,} filas | {n_fechas} fechas_t | {rng}")
        result[tipo] = df

    return result


def cargar_fecha_th(banco: str) -> pd.DataFrame:
    """
    Lee (fecha_t, h, fecha_th) de la matriz de features.
    fecha_th es la fecha calendario del horizonte h desde fecha_t,
    calculada con el calendario hábil PE+USA.
    """
    print("  Leyendo fecha_th desde la matriz de features...")
    cols = ["fecha_t", "h", "fecha_th"]
    df = pd.read_parquet(RUTA_MATRIZ,
                         filters=[("banco", "==", banco)],
                         columns=cols)
    df["fecha_t"]  = pd.to_datetime(df["fecha_t"])
    df["fecha_th"] = pd.to_datetime(df["fecha_th"])
    df = df.drop_duplicates(subset=["fecha_t", "h"]).sort_values(["fecha_t", "h"])
    print(f"  fecha_th: {len(df):,} pares (fecha_t, h)")
    return df


def enriquecer(df_preds: pd.DataFrame, df_fth: pd.DataFrame) -> pd.DataFrame:
    """
    Une cuantiles + target + fecha_th, elige las columnas finales y convierte
    los montos de USD a millones para consistencia con el video.
    """
    df = df_preds.merge(df_fth[["fecha_t", "h", "fecha_th"]],
                        on=["fecha_t", "h"], how="left")

    # Columnas base en el orden de presentación
    base = ["fecha_t", "fecha_th", "h"]
    if "fold" in df.columns:
        base.append("fold")

    q_presentes    = [c for c in Q_COLS   if c in df.columns]
    extra_presentes = [c for c in EXTRA_COLS if c in df.columns]
    target_col     = ["target"] if "target" in df.columns else []

    cols_ord = base + q_presentes + extra_presentes + target_col

    df = df[cols_ord].copy()

    # Convertir a millones (el video divide entre 1e6)
    for col in q_presentes + extra_presentes + target_col:
        df[col] = df[col] / 1_000_000

    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)
    return df


def construir_resumen(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Una fila por fecha_t:
      - rango de h disponible
      - n_realizados: cuántos horizontes tienen target no nulo
      - pct_realizados: cobertura de realizados sobre el rango de h
      - q50_h1: mediana en h=1 (primer día pronosticado)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    g = df.groupby("fecha_t")

    resumen = pd.DataFrame({
        "fecha_t"      : g["fecha_t"].first(),
        "fecha_th_min" : g["fecha_th"].min() if "fecha_th" in df.columns else np.nan,
        "fecha_th_max" : g["fecha_th"].max() if "fecha_th" in df.columns else np.nan,
        "h_min"        : g["h"].min(),
        "h_max"        : g["h"].max(),
        "n_horizontes" : g["h"].count(),
        "n_realizados" : g["target"].apply(lambda x: x.notna().sum()) if "target" in df.columns else np.nan,
    }).reset_index(drop=True)

    if "target" in df.columns:
        resumen["pct_realizados"] = (resumen["n_realizados"] / resumen["n_horizontes"] * 100).round(1)

    if "q50" in df.columns:
        q50_h1 = df[df["h"] == df["h"].min()].set_index("fecha_t")["q50"]
        resumen["q50_h_min_MM"] = resumen["fecha_t"].map(q50_h1)

    resumen["fuente"] = label
    return resumen


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  aux_exportar_fanchart_excel.py  |  banco: {BANCO}")
    print(f"{'='*60}\n")

    # 1. Cargar predicciones step005
    print("1. Cargando parquets de step005...")
    preds = cargar_preds_step005(BANCO)

    # 2. Cargar fecha_th desde la matriz
    print("\n2. Cargando fecha_th desde matriz de features...")
    df_fth = cargar_fecha_th(BANCO)

    # 3. Enriquecer con fecha_th
    print("\n3. Enriqueciendo con fecha_th...")
    hojas = {}
    resumenes = []
    for tipo in ("base", "overlay"):
        if preds[tipo] is None:
            print(f"  [{tipo.upper()}] Sin datos — hoja omitida")
            continue
        df_rich = enriquecer(preds[tipo], df_fth)
        hojas[tipo] = df_rich
        resumenes.append(construir_resumen(df_rich, tipo))
        print(f"  [{tipo.upper()}] {len(df_rich):,} filas × {df_rich.shape[1]} columnas listas")

    if not hojas:
        print("\n⚠  No hay datos de predicción disponibles. Verificar DIR_PREDS_STEP005.")
        return

    # 4. Hoja resumen
    if resumenes:
        df_resumen = pd.concat(resumenes, ignore_index=True).sort_values(["fecha_t", "fuente"])
        hojas["resumen"] = df_resumen

    # 5. Exportar
    RUTA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n4. Exportando a Excel...")
    print(f"   {RUTA_OUTPUT}")

    with pd.ExcelWriter(RUTA_OUTPUT, engine="openpyxl", datetime_format="YYYY-MM-DD") as writer:
        for nombre, df_hoja in hojas.items():
            df_hoja.to_excel(writer, sheet_name=nombre, index=False)
            ws = writer.sheets[nombre]
            # Ancho automático de columnas
            for col_cells in ws.columns:
                max_len = max(
                    (len(str(cell.value)) if cell.value is not None else 0)
                    for cell in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 30)
            print(f"   Hoja '{nombre}': {df_hoja.shape[0]:,} filas × {df_hoja.shape[1]} cols")

    print(f"\n✓ Archivo exportado correctamente.")
    print(f"\nContenido:")
    print(f"  · Hoja 'base'    : cuantiles Q01/Q05/Q50/Q95/Q99 + realizados + fecha_th (sin overlay)")
    print(f"  · Hoja 'overlay' : ídem con overlay sobreencaje aplicado")
    print(f"  · Hoja 'resumen' : una fila por fecha_t con cobertura de realizados")
    print(f"\nUnidades de cuantiles y target: millones USD")


if __name__ == "__main__":
    main()
