# -*- coding: utf-8 -*-
"""
aux_exportar_matriz_excel.py
Exporta la matriz de features del banco SISTEMA a Excel para análisis del equipo.
Lee directamente del parquet generado por step001_build_feature_matrix.py.
"""

import pandas as pd
from pathlib import Path

RUTA_PARQUET = (r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente"
                r"\1. Data\Clean\matriz_features.parquet")

RUTA_OUTPUT  = (r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente"
                r"\2. Output\matriz_SISTEMA.xlsx")

BANCO        = "SISTEMA"
H_FILTRO     = 4      # None = todos los horizontes; entero = un solo h (h_min=2)

# ─────────────────────────────────────────────────────────────────────────────
print("Leyendo parquet...")
df = pd.read_parquet(RUTA_PARQUET, filters=[("banco", "==", BANCO)])
print(f"  Filas cargadas: {len(df):,}  |  Columnas: {df.shape[1]}")

if H_FILTRO is not None:
    df = df[df["h"] == H_FILTRO]
    print(f"  Filtrado a h={H_FILTRO}: {len(df):,} filas")

# Ordenar por fecha y horizonte
df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

print(f"Exportando a Excel... (puede tomar unos segundos)")
Path(RUTA_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
df.to_excel(RUTA_OUTPUT, index=False, sheet_name="SISTEMA")
print(f"Archivo guardado en:\n  {RUTA_OUTPUT}")
print(f"\nDimensiones exportadas: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"Rango de fechas: {df['fecha_t'].min().date()} → {df['fecha_t'].max().date()}")
print(f"Horizontes: h={df['h'].min()} a h={df['h'].max()}")
