# -*- coding: utf-8 -*-
"""
aux_download_bloomberg.py
Lee series Bloomberg desde Excel y genera bloomberg_series.xlsx (Raw/).

══════════════════════════════════════════════════════════════════════════════
INSTRUCCIONES — Bloomberg Excel Add-In (en el terminal Bloomberg)
══════════════════════════════════════════════════════════════════════════════

1. Abre Excel con Bloomberg activo.
2. Crea UN archivo con DOS hojas: "CDS" y "PETROL".
3. En la hoja "CDS", celda A1, pega esta fórmula:

   =BDH("PERU CDS USD SR 5Y D14 Corp","PX_LAST","01/01/2000",TODAY(),
        "Days=W","Fill=P","Per=CD","DateFormat=YYYY-MM-DD")

4. En la hoja "PETROL", celda A1, pega esta fórmula:

   =BDH("SPPETIPP Index","PX_LAST","01/01/2000",TODAY(),
        "Days=W","Fill=P","Per=CD","DateFormat=YYYY-MM-DD")

   Parámetros:
     Days=W    → solo días hábiles (lunes-viernes)
     Fill=P    → rellena días sin cotización con valor previo
     Per=CD    → periodicidad diaria
     DateFormat=YYYY-MM-DD → formato ISO para evitar ambigüedades

5. Espera que Bloomberg cargue los datos (puede tardar unos segundos).
6. Guarda el archivo como:
   bloomberg_series.xlsx
   y cópialo a la carpeta Raw del proyecto.

══════════════════════════════════════════════════════════════════════════════
El script detecta automáticamente el formato de salida de Bloomberg BDH
(con o sin filas de encabezado, distintos formatos de fecha).
══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_ENTRADA = BASE_SISTEMA / "1. Data" / "Raw" / "bloomberg_series.xlsx"
RUTA_SALIDA  = RUTA_ENTRADA  # sobreescribe el mismo archivo después de limpiar

HOJAS = {
    "CDS":    "CDS_PERU_5Y",   # hoja → nombre de columna en output
    "PETROL": "PETROLEUM",
}


# ─────────────────────────────────────────────────────────────────────────────
# LECTOR ROBUSTO — maneja distintos formatos de BDH
# ─────────────────────────────────────────────────────────────────────────────
def _detectar_fila_datos(raw: pd.DataFrame) -> int:
    """
    Bloomberg BDH genera entre 0 y ~5 filas de metadatos antes de los datos.
    Escanea filas hasta encontrar la primera con una fecha válida en col 0.
    """
    for i, row in raw.iterrows():
        val = row.iloc[0]
        try:
            ts = pd.to_datetime(val, dayfirst=False, errors="raise")
            if pd.Timestamp("1990-01-01") <= ts <= pd.Timestamp("2100-01-01"):
                return i
        except Exception:
            continue
    raise ValueError("No se encontró ninguna fila con fecha válida en la hoja.")


def _leer_hoja(ruta: Path, hoja: str, nombre: str) -> pd.Series:
    """
    Lee una hoja de Excel exportada con BDH y retorna una Series limpia.
    Maneja:
      - Filas de encabezado Bloomberg (Ticker / Field / Dates / PX_LAST)
      - Fechas en formato YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, YYYYMMDD
      - Columna de fecha en col A, valor en col B
    """
    raw = pd.read_excel(ruta, sheet_name=hoja, header=None, dtype=str)

    fila_inicio = _detectar_fila_datos(raw)
    datos = raw.iloc[fila_inicio:, :2].copy()
    datos.columns = ["fecha", nombre]
    datos = datos.dropna(subset=["fecha"])

    # Parsear fechas con múltiples formatos
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d", "%d-%b-%y"):
        try:
            datos["fecha"] = pd.to_datetime(datos["fecha"], format=fmt, errors="raise")
            break
        except Exception:
            continue
    else:
        datos["fecha"] = pd.to_datetime(datos["fecha"], infer_datetime_format=True,
                                         errors="coerce")

    datos[nombre] = pd.to_numeric(datos[nombre], errors="coerce")
    datos = datos.dropna(subset=["fecha", nombre])
    datos = datos.set_index("fecha").sort_index()
    datos = datos[~datos.index.duplicated(keep="last")]
    datos.index.name = "fecha"

    print(f"  [{hoja}] → {nombre}: {len(datos):,} obs | "
          f"{datos.index.min().date()} → {datos.index.max().date()} | "
          f"rango [{datos[nombre].min():.2f}, {datos[nombre].max():.2f}]")
    return datos[nombre]


# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA VÍA pdblp (cuando Bloomberg está en la misma máquina)
# ─────────────────────────────────────────────────────────────────────────────
TICKERS_PDBLP = {
    "PERU CDS USD SR 5Y D14 Corp": "CDS_PERU_5Y",
    "SPPETIPP Index":              "PETROLEUM",
}

def _descargar_via_pdblp() -> pd.DataFrame:
    import pdblp
    print("Conectando a Bloomberg Terminal (pdblp)...")
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    dfs = []
    for ticker, nombre in TICKERS_PDBLP.items():
        print(f"  {ticker} → {nombre}...")
        df = con.bdh(ticker, "PX_LAST", "20000101",
                     pd.Timestamp.today().strftime("%Y%m%d"), longdata=False)
        df.columns = [nombre]
        dfs.append(df)
    con.stop()
    resultado = pd.concat(dfs, axis=1)
    resultado.index = pd.to_datetime(resultado.index)
    resultado.index.name = "fecha"
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DESDE EXCEL MANUAL
# ─────────────────────────────────────────────────────────────────────────────
def _leer_desde_excel() -> pd.DataFrame:
    if not RUTA_ENTRADA.exists():
        raise FileNotFoundError(
            f"\nNo se encontró: {RUTA_ENTRADA}\n"
            "  → Sigue las instrucciones al inicio de este script para\n"
            "    exportar los datos desde Bloomberg Excel Add-In."
        )

    print(f"Leyendo desde: {RUTA_ENTRADA}\n")

    # Verificar hojas disponibles
    xl = pd.ExcelFile(RUTA_ENTRADA)
    hojas_disponibles = xl.sheet_names
    print(f"  Hojas encontradas: {hojas_disponibles}")

    series = []
    for hoja, nombre in HOJAS.items():
        if hoja not in hojas_disponibles:
            # Búsqueda flexible por nombre parcial
            match = next((h for h in hojas_disponibles
                          if nombre[:3].lower() in h.lower()
                          or hoja.lower() in h.lower()), None)
            if match:
                print(f"  Hoja '{hoja}' no encontrada → usando '{match}'")
                hoja = match
            else:
                print(f"  ADVERTENCIA: hoja '{hoja}' no encontrada en el Excel. "
                      f"  {nombre} quedará como NaN.")
                continue
        s = _leer_hoja(RUTA_ENTRADA, hoja, nombre)
        series.append(s)

    if not series:
        raise ValueError("No se pudo leer ninguna hoja del archivo Excel.")

    return pd.concat(series, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────
def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.index >= "2000-01-01"].copy()
    df = df.ffill(limit=5)   # cubre feriados y fines de semana

    n_nan_total = df.isna().sum().sum()
    if n_nan_total > 0:
        print(f"\n  NaN residuales tras ffill: {n_nan_total} "
              f"(filas con algún NaN: {df.isna().any(axis=1).sum()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("aux_download_bloomberg.py")
    print("=" * 65)

    try:
        import pdblp  # noqa: F401
        df_raw = _descargar_via_pdblp()
    except ImportError:
        df_raw = _leer_desde_excel()

    print("\nLimpiando series...")
    df_clean = _limpiar(df_raw)

    # Guardar bloomberg_series.xlsx con formato limpio (una sola hoja "datos")
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(RUTA_SALIDA, engine="openpyxl") as writer:
        df_clean.to_excel(writer, sheet_name="datos", index=True)
        # Preservar hojas originales si existen
        if RUTA_ENTRADA.exists():
            xl_orig = pd.ExcelFile(RUTA_ENTRADA)
            for hoja in xl_orig.sheet_names:
                if hoja != "datos":
                    xl_orig.parse(hoja).to_excel(writer, sheet_name=hoja, index=False)

    print(f"\nGuardado en: {RUTA_SALIDA}")
    print(f"  Filas: {len(df_clean):,} | Columnas: {list(df_clean.columns)}")
    print("\nPrimeras filas:")
    print(df_clean.head(3).to_string())
    print("\nÚltimas filas:")
    print(df_clean.tail(3).to_string())
