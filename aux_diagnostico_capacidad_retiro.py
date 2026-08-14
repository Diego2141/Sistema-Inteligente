"""
¿Por qué capacidad_retiro_th tiene tantos vacíos y valores repetidos?

Diagnóstico de solo lectura sobre matriz_features.parquet. No re-entrena nada.

Dos patrones observados, dos causas distintas
-----------------------------------------------
VACÍOS      → por diseño. capacidad_retiro_th queda en NaN cuando fecha_th cae
              en un mes calendario distinto al de fecha_t (el período de encaje
              se reinicia al cierre y no hay forma de proyectar el requerimiento
              del mes siguiente sin sus propios datos). Con h hasta 75 días
              hábiles, la MAYORÍA de las filas de cada fecha_t caen fuera del
              mes — eso es esperado, no un error.

REPETIDOS   → sospecha de staleness. merge_asof(direction="backward") no tiene
              ningún resguardo de qué tan viejo es el dato encontrado. Si
              bbva_encaje.xlsx termina en una fecha anterior al máximo fecha_t
              de la matriz, todo origen posterior a ese corte recibe LA MISMA
              fila (la última disponible) — insumos congelados. Combinado con
              los dos MAX(0,·) de la fórmula (que saturan en los extremos),
              produce el mismo número exacto en cientos de filas.

Qué mide este script
---------------------
1. Los valores más repetidos de capacidad_retiro_th y cuántas filas cada uno.
2. Para el valor más repetido: el rango de fecha_t que lo produce. Si se
   concentra en la COLA más reciente de la matriz, confirma staleness.
3. La fecha máxima real de bbva_encaje_features_modelo.xlsx, para comparar
   directamente contra el fecha_t donde empiezan los repetidos.
4. La cobertura de vacíos por grupo de horizonte, para confirmar que decae
   con h como predice el diseño (no que sea un problema aparte).

Uso: ejecutar directamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RUTA_MATRIZ = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\matriz_features.parquet")
RUTA_ENCAJE = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\2. Output\bbva_encaje_features_modelo.xlsx")

BINS_H   = [1, 5, 15, 30, 50, 75]
LABELS_H = ["h02-05", "h06-15", "h16-30", "h31-50", "h51-75"]


def main() -> None:
    if not RUTA_MATRIZ.exists():
        raise FileNotFoundError(f"No se encuentra {RUTA_MATRIZ}")

    cols = ["fecha_t", "fecha_th", "h", "capacidad_retiro_th"]
    df = pd.read_parquet(RUTA_MATRIZ, columns=cols)
    df["fecha_t"]  = pd.to_datetime(df["fecha_t"])
    df["fecha_th"] = pd.to_datetime(df["fecha_th"])

    print(f"Filas totales     : {len(df):,}")
    print(f"fecha_t           : {df['fecha_t'].min().date()} → {df['fecha_t'].max().date()}")
    print(f"cobertura no-nulo : {df['capacidad_retiro_th'].notna().mean():.1%}")

    # ── 1) Valores más repetidos ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1 — Valores más repetidos (no nulos)")
    print("=" * 70)
    vc = df["capacidad_retiro_th"].dropna().value_counts().head(10)
    n_no_nulo = df["capacidad_retiro_th"].notna().sum()
    for val, n in vc.items():
        print(f"  {val:>18,.0f}  :  {n:>7,} filas  ({n/n_no_nulo:.1%} de los no-nulos)")

    # ── 2) ¿Dónde vive el valor más repetido? ───────────────────────────────
    if len(vc) > 0:
        top_val = vc.index[0]
        sub = df[df["capacidad_retiro_th"] == top_val]
        print("\n" + "=" * 70)
        print(f"2 — Rango de fecha_t que produce el valor más repetido ({top_val:,.0f})")
        print("=" * 70)
        print(f"  fecha_t: {sub['fecha_t'].min().date()} → {sub['fecha_t'].max().date()}")
        print(f"  fecha_t distintas involucradas: {sub['fecha_t'].nunique():,}")
        print(f"  Comparar este rango contra la fecha máxima del Excel de encaje (sección 3).")
        print(f"  Si el mínimo de este rango coincide con esa fecha máxima (o cae poco después),")
        print(f"  confirma que son fechas SIN dato fresco de encaje — insumos congelados.")

    # ── 3) Fecha máxima real del archivo de encaje ──────────────────────────
    print("\n" + "=" * 70)
    print("3 — Última fecha disponible en bbva_encaje_features_modelo.xlsx")
    print("=" * 70)
    if RUTA_ENCAJE.exists():
        enc = pd.read_excel(RUTA_ENCAJE, sheet_name="Datos", usecols=["fecha"])
        enc["fecha"] = pd.to_datetime(enc["fecha"])
        print(f"  Última fecha: {enc['fecha'].max().date()}")
        print(f"  Máximo fecha_t de la matriz: {df['fecha_t'].max().date()}")
        gap = (df["fecha_t"].max() - enc["fecha"].max()).days
        print(f"  Brecha: {gap} días calendario sin dato fresco de encaje")
    else:
        print(f"  No se encuentra {RUTA_ENCAJE} — ajuste RUTA_ENCAJE si está en otra carpeta.")

    # ── 4) Vacíos por grupo de horizonte — debe decaer con h por diseño ─────
    print("\n" + "=" * 70)
    print("4 — Cobertura no-nula por grupo de horizonte")
    print("=" * 70)
    df["h_grupo"] = pd.cut(df["h"], bins=BINS_H, labels=LABELS_H)
    cob_h = df.groupby("h_grupo", observed=True)["capacidad_retiro_th"].apply(lambda s: s.notna().mean())
    print(cob_h.map("{:.1%}".format).to_string())
    print("\n  Si decae monótonamente con h, los vacíos son el diseño esperado")
    print("  (fecha_th sale del mes calendario de fecha_t cada vez más seguido).")


if __name__ == "__main__":
    main()
