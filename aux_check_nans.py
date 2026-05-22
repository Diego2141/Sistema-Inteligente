# -*- coding: utf-8 -*-
"""
aux_check_nans.py
Diagnóstico de NaN en los splits TRAIN / VAL / TEST para un banco dado.
Corre localmente donde esté el Parquet.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"

BANCO        = "SISTEMA"
SEMANAS_VAL  = 26
SEMANAS_TEST = 13

# ── Leer ────────────────────────────────────────────────────────────────────
df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", BANCO)])
df["fecha_t"] = pd.to_datetime(df["fecha_t"])
df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

# ── Split ────────────────────────────────────────────────────────────────────
fechas = np.sort(df["fecha_t"].unique())
n = len(fechas)
n_test = min(SEMANAS_TEST * 5, n // 6)
n_val  = min(SEMANAS_VAL  * 5, n // 4)
corte_val  = fechas[n - n_test - n_val]
corte_test = fechas[n - n_test]

df_train = df[df["fecha_t"] <  corte_val].copy()
df_val   = df[(df["fecha_t"] >= corte_val) & (df["fecha_t"] < corte_test)].copy()
df_test  = df[df["fecha_t"] >= corte_test].copy()

print(f"TRAIN: {df_train['fecha_t'].min().date()} → {df_train['fecha_t'].max().date()} "
      f"({df_train['fecha_t'].nunique()} fechas, {len(df_train):,} filas)")
print(f"VAL  : {df_val['fecha_t'].min().date()} → {df_val['fecha_t'].max().date()} "
      f"({df_val['fecha_t'].nunique()} fechas, {len(df_val):,} filas)")
print(f"TEST : {df_test['fecha_t'].min().date()} → {df_test['fecha_t'].max().date()} "
      f"({df_test['fecha_t'].nunique()} fechas, {len(df_test):,} filas)")
print()

# ── NaN por feature ──────────────────────────────────────────────────────────
excluir = {"fecha_t", "banco", "target"}
cols = [c for c in df.columns if c not in excluir]

resultados = []
for col in cols:
    nan_tr = df_train[col].isna().sum()
    nan_v  = df_val[col].isna().sum()
    nan_te = df_test[col].isna().sum()
    resultados.append({
        "feature"  : col,
        "NaN_train": nan_tr, "%_train": round(100 * nan_tr / len(df_train), 1),
        "NaN_val"  : nan_v,  "%_val"  : round(100 * nan_v  / len(df_val),   1),
        "NaN_test" : nan_te, "%_test" : round(100 * nan_te / len(df_test),  1),
    })

df_res = pd.DataFrame(resultados).sort_values("NaN_val", ascending=False)

print("=" * 80)
print("FEATURES CON NaN EN VAL O TEST")
print("=" * 80)
con_nan = df_res[(df_res["NaN_val"] > 0) | (df_res["NaN_test"] > 0)]
pd.set_option("display.max_rows", 80)
pd.set_option("display.width", 120)
print(con_nan.to_string(index=False))
print(f"\nFeatures con NaN en val o test : {len(con_nan)} de {len(cols)}")
print(f"Features sin NaN en ningún split: {len(df_res) - len(con_nan)} de {len(cols)}")

print()
print("=" * 80)
print("DETALLE — primeras fechas del VAL con NaN (para identificar causa)")
print("=" * 80)
cols_con_nan_val = con_nan[con_nan["NaN_val"] > 0]["feature"].tolist()
for col in cols_con_nan_val[:10]:
    fechas_nan = df_val[df_val[col].isna()]["fecha_t"].unique()
    print(f"  {col:30s}: {len(fechas_nan)} fechas con NaN "
          f"| primera: {pd.Timestamp(fechas_nan.min()).date()} "
          f"| última: {pd.Timestamp(fechas_nan.max()).date()}")
