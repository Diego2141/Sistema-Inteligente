# -*- coding: utf-8 -*-
"""
aux_exportar_matriz_excel_v2.py
Exporta a Excel la matriz de features de los grupos de la PARTICIÓN generada por
step001_build_feature_matrix_v2.py, más una hoja de conciliación.

Diferencia con v1: v1 exportaba una sola hoja con el banco SISTEMA. v2 detecta
los grupos de la partición en la propia matriz (FOCO_* y RESTO_*), exporta una
hoja por grupo, y agrega la hoja que motiva todo esto:

  CONCILIACION — para cada (fecha_t, h), el target de SISTEMA contra la suma de
  FOCO + RESTO. Tiene que cerrar exacto. Si no cierra, los dos modelos no están
  partiendo el mismo objeto que el modelo agregado, y compararlos no significa
  nada. Es la verificación que no se puede hacer sobre datos sintéticos.

  RESUMEN — participación del foco en el total por año, que es el número de
  concentración leído directamente de la matriz en vez de recalculado aparte.

Se corre sin argumentos (runfile desde Spyder). Editar las constantes de abajo.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
RUTA_PARQUET = (r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente"
                r"\1. Data\Clean\matriz_features.parquet")

RUTA_OUTPUT  = (r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente"
                r"\2. Output\matriz_particion.xlsx")

# None = detectar los grupos en la matriz (SISTEMA + FOCO_* + RESTO_*).
# Para forzar un subconjunto, poner la lista a mano.
GRUPOS = None

H_FILTRO = 5        # None = todos los horizontes; entero = un solo h
TOL      = 1e-3     # tolerancia de la conciliación, en las unidades del target

MAX_FILAS_HOJA = 1_000_000   # Excel corta en 1.048.576

# ─────────────────────────────────────────────────────────────────────────────
def _detectar_columna(df, candidatas, etiqueta):
    for c in candidatas:
        if c in df.columns:
            return c
    raise KeyError(f"No se encontró la columna de {etiqueta}. "
                   f"Probadas: {candidatas}. Disponibles: {list(df.columns)[:25]}...")


print("Leyendo la columna banco para detectar los grupos...")
_bancos = pd.read_parquet(RUTA_PARQUET, columns=["banco"])["banco"]
disponibles = sorted(_bancos.unique())
print(f"  Grupos en la matriz: {disponibles}")

if GRUPOS is None:
    foco  = [g for g in disponibles if str(g).startswith("FOCO_")]
    resto = [g for g in disponibles if str(g).startswith("RESTO_")]
    grupos = (["SISTEMA"] if "SISTEMA" in disponibles else []) + foco + resto
    if not foco or not resto:
        print("\n  AVISO: no se detectaron grupos FOCO_*/RESTO_* en la matriz.")
        print("  La matriz parece haberse generado con particion_activa=None,")
        print("  o con step001 v1. No hay conciliación que hacer.")
else:
    grupos = list(GRUPOS)
    foco  = [g for g in grupos if str(g).startswith("FOCO_")]
    resto = [g for g in grupos if str(g).startswith("RESTO_")]

if not grupos:
    raise SystemExit("No hay grupos para exportar.")
print(f"  A exportar: {grupos}")

print("\nLeyendo la matriz...")
df = pd.read_parquet(RUTA_PARQUET, filters=[("banco", "in", grupos)])
print(f"  Filas: {len(df):,}  |  Columnas: {df.shape[1]}")

if H_FILTRO is not None:
    df = df[df["h"] == H_FILTRO]
    print(f"  Filtrado a h={H_FILTRO}: {len(df):,} filas")

col_target = _detectar_columna(df, ["target", "y", "target_neto"], "target")
df = df.sort_values(["banco", "fecha_t", "h"]).reset_index(drop=True)

# ── Conciliación ─────────────────────────────────────────────────────────────
# Se hace sobre el target y no sobre las features: el target es lo que los
# modelos predicen, y es donde la partición tiene que cerrar para que la
# recombinación tenga sentido. Que las features de FOCO y RESTO no sumen a las
# de SISTEMA es normal y esperable (una volatilidad no es aditiva).
conc = pd.DataFrame()
if foco and resto and "SISTEMA" in grupos:
    print("\nConciliando SISTEMA contra FOCO + RESTO...")
    llaves = [c for c in ("fecha_t", "h") if c in df.columns]
    piv = df.pivot_table(index=llaves, columns="banco",
                         values=col_target, aggfunc="first")
    necesarias = ["SISTEMA", foco[0], resto[0]]
    faltan = [c for c in necesarias if c not in piv.columns]
    if faltan:
        print(f"  No se puede conciliar, faltan columnas: {faltan}")
    else:
        conc = piv[necesarias].dropna().copy()
        conc.columns = ["sistema", "foco", "resto"]
        conc["foco_mas_resto"] = conc["foco"] + conc["resto"]
        conc["diferencia"] = conc["sistema"] - conc["foco_mas_resto"]
        # Participación del foco. Se deja vacía cuando el sistema está cerca de
        # cero: ahí el cociente explota sin que el foco haya cambiado, y un
        # 1.800% en la planilla se lee como un dato y no como un denominador
        # chico. Es el mismo criterio de la tabla trimestral del one-pager.
        piso = conc["sistema"].abs().median() * 0.15
        conc["foco_pct_sistema"] = np.where(
            conc["sistema"].abs() > piso,
            conc["foco"] / conc["sistema"].replace(0, np.nan) * 100,
            np.nan)
        conc = conc.reset_index()

        peor = float(conc["diferencia"].abs().max())
        n_mal = int((conc["diferencia"].abs() > TOL).sum())
        print(f"  Filas conciliadas : {len(conc):,}")
        print(f"  Peor diferencia   : {peor:.6g}")
        if n_mal:
            print(f"  *** {n_mal:,} filas superan la tolerancia de {TOL} ***")
            print("      La partición NO cierra. Revisar aplicar_particion() y")
            print("      la suma de SISTEMA antes de comparar modelos.")
        else:
            print(f"  OK: FOCO + RESTO == SISTEMA en todas las filas (tol {TOL})")

# ── Resumen por grupo y año ──────────────────────────────────────────────────
resumen = (df.groupby("banco")[col_target]
             .agg(filas="size", media="mean", desv="std",
                  minimo="min", maximo="max")
             .reset_index())

por_anio = pd.DataFrame()
if not conc.empty and "fecha_t" in conc.columns:
    conc["anio"] = pd.to_datetime(conc["fecha_t"]).dt.year
    # Suma de sumas, no promedio de cocientes: el promedio de participaciones
    # diarias no es la participación del año, y la diferencia entre las dos
    # lecturas ya causó confusión antes en este proyecto.
    por_anio = (conc.groupby("anio")[["sistema", "foco", "resto"]].sum()
                    .assign(foco_pct=lambda d: d["foco"] / d["sistema"] * 100)
                    .reset_index())

# ── Escritura ────────────────────────────────────────────────────────────────
print(f"\nExportando a Excel...")
Path(RUTA_OUTPUT).parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(RUTA_OUTPUT, engine="openpyxl") as xw:
    if not conc.empty:
        conc.to_excel(xw, sheet_name="CONCILIACION", index=False)
    if not por_anio.empty:
        por_anio.to_excel(xw, sheet_name="RESUMEN_ANUAL", index=False)
    resumen.to_excel(xw, sheet_name="RESUMEN_GRUPOS", index=False)

    for g in grupos:
        sub = df[df["banco"] == g]
        if sub.empty:
            continue
        if len(sub) > MAX_FILAS_HOJA:
            print(f"  {g}: {len(sub):,} filas excede el límite de Excel, "
                  f"se truncan las más antiguas")
            sub = sub.tail(MAX_FILAS_HOJA)
        # Los nombres de hoja de Excel toleran 31 caracteres
        sub.to_excel(xw, sheet_name=str(g)[:31], index=False)
        print(f"  {g}: {len(sub):,} filas")

print(f"\nArchivo guardado en:\n  {RUTA_OUTPUT}")
if not conc.empty:
    print(f"\nRango de fechas: {pd.to_datetime(conc['fecha_t']).min().date()}"
          f" → {pd.to_datetime(conc['fecha_t']).max().date()}")
print("\nHojas: CONCILIACION (el check), RESUMEN_ANUAL, RESUMEN_GRUPOS, "
      "y una por grupo.")
