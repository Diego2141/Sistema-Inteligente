# -*- coding: utf-8 -*-
"""
aux_download_bloomberg.py
Descarga series Bloomberg y las guarda en bloomberg_series.xlsx (Raw/).

Tickers:
  - PERU CDS USD SR 5Y D14 Corp  →  CDS_PERU_5Y  (spread CDS Perú 5 años)
  - SPPETIPP Index                →  PETROLEUM    (S&P GSCI Petroleum)

Requiere Bloomberg Terminal activo y pdblp instalado:
  pip install pdblp

Si pdblp no está disponible, el script muestra instrucciones para
descargar manualmente con el Add-In Bloomberg de Excel.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_SISTEMA  = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_SALIDA   = BASE_SISTEMA / "1. Data" / "Raw" / "bloomberg_series.xlsx"

FECHA_INICIO  = "20000101"
FECHA_FIN     = pd.Timestamp.today().strftime("%Y%m%d")

TICKERS = {
    "PERU CDS USD SR 5Y D14 Corp": "CDS_PERU_5Y",
    "SPPETIPP Index":              "PETROLEUM",
}

FIELD = "PX_LAST"

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA VÍA pdblp
# ─────────────────────────────────────────────────────────────────────────────
def _descargar_via_pdblp() -> pd.DataFrame:
    import pdblp
    print("Conectando a Bloomberg Terminal...")
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()

    dfs = []
    for ticker, nombre in TICKERS.items():
        print(f"  Descargando {ticker} → {nombre}...")
        df = con.bdh(ticker, FIELD, FECHA_INICIO, FECHA_FIN, longdata=False)
        df.columns = [nombre]
        dfs.append(df)

    con.stop()
    resultado = pd.concat(dfs, axis=1)
    resultado.index = pd.to_datetime(resultado.index)
    resultado.index.name = "fecha"
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DESDE EXCEL MANUAL (fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _leer_desde_excel() -> pd.DataFrame:
    """
    Lee bloomberg_series.xlsx si ya fue creado manualmente.

    Formato esperado (exportado con BDH desde Bloomberg Excel Add-In):
      Columna A : fecha  (dd/mm/yyyy o similar)
      Columna B : CDS_PERU_5Y
      Columna C : PETROLEUM
    """
    if not RUTA_SALIDA.exists():
        raise FileNotFoundError(
            f"No se encontró {RUTA_SALIDA}.\n"
            "  Para crearlo manualmente con el Add-In Bloomberg de Excel:\n"
            "  1. Abre Excel con Bloomberg activo.\n"
            "  2. En celda A1 escribe la fecha inicial: 01/01/2000\n"
            "  3. En B1 usa: =BDH(\"PERU CDS USD SR 5Y D14 Corp\",\"PX_LAST\",A1,TODAY())\n"
            "  4. En C1 usa: =BDH(\"SPPETIPP Index\",\"PX_LAST\",A1,TODAY())\n"
            "  5. Guarda como 'bloomberg_series.xlsx' en:\n"
            f"     {RUTA_SALIDA.parent}"
        )

    print(f"Leyendo desde {RUTA_SALIDA}...")
    df = pd.read_excel(RUTA_SALIDA, index_col=0, parse_dates=True)
    df.index.name = "fecha"
    # Normalizar nombres de columnas si difieren
    rename_map = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        if "cds" in col_lower:
            rename_map[col] = "CDS_PERU_5Y"
        elif "petroleum" in col_lower or "sppetipp" in col_lower or "petrol" in col_lower:
            rename_map[col] = "PETROLEUM"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LIMPIEZA Y GUARDADO
# ─────────────────────────────────────────────────────────────────────────────
def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Filtrar desde 2000 en adelante
    df = df[df.index >= "2000-01-01"]

    # Forward-fill máximo 5 días (fines de semana y feriados)
    df = df.ffill(limit=5)

    # Reporte de NaNs
    for col in df.columns:
        n_nan = df[col].isna().sum()
        pct   = 100 * n_nan / len(df)
        print(f"  {col:20s}: {len(df):,} obs | NaN={n_nan:,} ({pct:.1f}%) | "
              f"rango [{df[col].min():.2f}, {df[col].max():.2f}]")
    return df


def guardar(df: pd.DataFrame):
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(RUTA_SALIDA)
    print(f"\nGuardado en: {RUTA_SALIDA}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import pdblp  # noqa: F401
        df_raw = _descargar_via_pdblp()
    except ImportError:
        print("pdblp no instalado (pip install pdblp).")
        print("Intentando leer desde Excel existente...\n")
        df_raw = _leer_desde_excel()

    print("\nResumen de series descargadas:")
    df_clean = _limpiar(df_raw)
    guardar(df_clean)

    print("\nPrimeras filas:")
    print(df_clean.head())
    print("\nÚltimas filas:")
    print(df_clean.tail())
