# -*- coding: utf-8 -*-
"""
aux_download_bloomberg.py
Lee series Bloomberg desde Excel y genera bloomberg_series.xlsx (Raw/).

══════════════════════════════════════════════════════════════════════════════
INSTRUCCIONES — Bloomberg Excel Add-In (en el terminal Bloomberg)
══════════════════════════════════════════════════════════════════════════════

Usa el mismo template que ya tienes funcionando. En cada hoja configura:

  Archivo: DataBBG.xlsx  (en 1. Data\Raw\)

  Hoja "BVL":   MXNUPEGE Index       — Bolsa de Valores de Lima
  Hoja "CDS":   CPERU1U5 CBIN Curncy — CDS Perú 5Y
  Hoja "Cobre": HG1 COMB Comdty      — Cobre COMEX (USD/lb)

El script reconoce automáticamente el formato estándar Bloomberg:
  Fila 1: Security   | <ticker>
  Fila 2: Start Date | <fecha>
  Fila 3: End Date   | <fecha>
  Fila 4: Period     | D
  Fila 5: Currency   | <moneda>
  Fila 6: (vacía)
  Fila 7: Date       | PX_LAST
  Fila 8+: datos

Guarda el archivo como bloomberg_series.xlsx y cópialo a:
  1. Data\Raw\bloomberg_series.xlsx
══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_ENTRADA = BASE_SISTEMA / "1. Data" / "Raw" / "DataBBG.xlsx"
RUTA_SALIDA  = BASE_SISTEMA / "1. Data" / "Raw" / "bloomberg_series.xlsx"

HOJAS = {
    "BVL":   "BVL",         # MXNUPEGE Index — Bolsa de Valores de Lima
    "CDS":   "CDS_PERU_5Y", # CPERU1U5 CBIN Curncy — CDS Perú 5Y
    "Cobre": "COPPER",      # HG1 COMB Comdty — Cobre COMEX (USD/lb)
}

# Formato Bloomberg con template estándar:
#   Fila 1: Security  | <ticker>
#   Fila 2: Start Date| <fecha>
#   Fila 3: End Date  | <fecha>
#   Fila 4: Period    | D
#   Fila 5: Currency  | PEN
#   Fila 6: (vacía)
#   Fila 7: Date      | PX_LAST   ← encabezado de columnas
#   Fila 8+: datos    | valores


# ─────────────────────────────────────────────────────────────────────────────
# LECTOR ROBUSTO — maneja formato Bloomberg BDH
# ─────────────────────────────────────────────────────────────────────────────
# Formatos de fecha en orden de prioridad (DD/MM/YYYY primero = estándar Perú)
_DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d", "%d/%m/%Y %H:%M", "%m/%d/%Y",
                 "%Y%m%d", "%d-%b-%y", "%d-%b-%Y"]


def _detectar_fila_datos(raw: pd.DataFrame) -> int:
    """
    Bloomberg BDH genera metadatos en las primeras filas (Security, Dates, etc.).
    Los metadatos tienen texto en col A; los datos tienen fecha en col A.
    Retorna el índice de la primera fila con fecha válida en col A.
    """
    for i, row in raw.iterrows():
        val = str(row.iloc[0]).strip()
        # Ignorar etiquetas conocidas de Bloomberg
        if val.lower() in ("security", "start date", "end date", "period",
                           "currency", "date", "dates", "nan", ""):
            continue
        for fmt in _DATE_FORMATS:
            try:
                ts = pd.to_datetime(val.split(" ")[0], format=fmt.split(" ")[0])
                if pd.Timestamp("1990-01-01") <= ts <= pd.Timestamp("2100-01-01"):
                    return i
            except Exception:
                continue
    raise ValueError("No se encontró ninguna fila con fecha válida en la hoja.")


def _parsear_fechas(serie: pd.Series) -> pd.Series:
    """Intenta múltiples formatos de fecha en orden de prioridad DD/MM/YYYY."""
    for fmt in _DATE_FORMATS:
        try:
            resultado = pd.to_datetime(serie, format=fmt, errors="raise")
            return resultado
        except Exception:
            continue
    # Fallback: inferencia automática con dayfirst=True (formato Perú)
    return pd.to_datetime(serie, dayfirst=True, errors="coerce")


def _leer_hoja(ruta: Path, hoja: str, nombre: str) -> pd.Series:
    """
    Lee una hoja de Excel exportada con BDH y retorna una Series limpia.
    Compatible con el template Bloomberg estándar (6 filas de metadatos).
    """
    raw = pd.read_excel(ruta, sheet_name=hoja, header=None, dtype=str)

    fila_inicio = _detectar_fila_datos(raw)
    datos = raw.iloc[fila_inicio:, :2].copy()
    datos.columns = ["fecha", nombre]
    datos = datos.dropna(subset=["fecha"])
    datos = datos[datos["fecha"].str.strip() != ""]

    datos["fecha"] = _parsear_fechas(datos["fecha"])
    datos[nombre]  = pd.to_numeric(datos[nombre], errors="coerce")
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
    "MXNUPEGE Index":      "BVL",
    "CPERU1U5 CBIN Curncy": "CDS_PERU_5Y",
    "HG1 COMB Comdty":     "COPPER",
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
