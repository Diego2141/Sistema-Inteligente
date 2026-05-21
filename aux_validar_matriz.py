# -*- coding: utf-8 -*-
"""
aux_validar_matriz.py
Verifica la integridad de la matriz de features generada por step001 y
produce gráficos diagnósticos para validar su calidad antes del modelado.

Lee el Parquet en trozos para evitar errores de memoria en matrices grandes.
Pico de RAM ≈ max(columnas_clave × n_filas, BATCH_SIZE × n_features).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_validar_matriz"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100_000   # filas por bloque en el pase de NaN/rangos

# ─────────────────────────────────────────────────────────────────
# Resumen de avance
# ─────────────────────────────────────────────────────────────────
sep = "=" * 65
print(f"\n{sep}")
print("  SISTEMA DE PREDICCIÓN DE LIQUIDEZ — ESTADO DEL PROYECTO")
print(sep)
print("""
  LO QUE ESTÁ LISTO
  ─────────────────
  ✓ Calendario hábil conjunto Perú + EE.UU. (feriados PE y US)
  ✓ Carga y limpieza de transacciones bancarias (2010–2026)
      · Alias histórico CONTINEN → BBVA unificado
      · 12 bancos pequeños agrupados en Otros_bancos (umbral 1%)
  ✓ Features bancarias por entidad
      · Valores de hoy: R_t0, D_t0 (aviso anticipado)
      · Rezagos: t-1, t-2, t-3, t-5, t-22
      · Volatilidades y medias móviles: 5d y 22d
      · Confirmados próximos: R_conf_t1, R_conf_t2, D_conf_t1
  ✓ Features macroeconómicas (2010–2026)
      · VIX, T10Y (Yahoo Finance)
      · FED_FUNDS diario (FRED)
      · EMBI Perú, Tasa Ref, TC compra/venta (BCRP Add-In)
  ✓ Features estacionales para fecha futura t+h
      · Posición en mes/trimestre, feriados PE+US, elecciones
  ✓ Matriz de features exportada a Parquet (~5M filas, float32)
      · 12 entidades × 4,271 fechas × 89 horizontes
      · Lectura eficiente banco por banco (peak RAM < 200 MB)

  PRÓXIMO PASO
  ────────────
  → step002_train_model.py
     Entrenar LightGBM con quantile regression para los quantiles
     [0.01, 0.05, 0.50, 0.95, 0.99]. Un solo modelo por banco con h
     como feature numérico. Validación walk-forward (expanding window).
""")
print(sep + "\n")

# ─────────────────────────────────────────────────────────────────
# 1. Metadata sin cargar datos
# ─────────────────────────────────────────────────────────────────
print(f"Leyendo metadata: {RUTA_MATRIZ}")
pf      = pq.ParquetFile(RUTA_MATRIZ)
n_total = pf.metadata.num_rows
all_cols = pf.schema_arrow.names

cols_id     = ["fecha_t", "banco", "h", "log_h"]
cols_target = ["target"]
cols_feat   = [c for c in all_cols if c not in cols_id + cols_target]

print(f"  Filas totales : {n_total:,}")
print(f"  Columnas      : {len(all_cols)}  ({len(cols_feat)} features + "
      f"{len(cols_id)} id + 1 target)")

# ─────────────────────────────────────────────────────────────────
# 2. Columnas clave — lectura ligera (4 cols × n_filas ≈ 160 MB)
# ─────────────────────────────────────────────────────────────────
print("Cargando columnas clave (fecha_t, banco, h, target)...")
df_key = pd.read_parquet(RUTA_MATRIZ, columns=["fecha_t", "banco", "h", "target"])
bancos = sorted(df_key["banco"].unique())
hs     = sorted(df_key["h"].unique())

# ─────────────────────────────────────────────────────────────────
# 3. Reporte de integridad en consola
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  VERIFICACIÓN DE INTEGRIDAD — MATRIZ DE FEATURES")
print(f"{'='*65}")

print(f"\n[1] Dimensiones")
print(f"    Filas       : {n_total:,}")
print(f"    Features    : {len(cols_feat)}")
print(f"    Bancos      : {len(bancos)}  →  {bancos}")
print(f"    Horizontes h: {min(hs)} … {max(hs)}  ({len(hs)} valores)")
print(f"    Fechas t    : {df_key['fecha_t'].min().date()} → {df_key['fecha_t'].max().date()}")

print(f"\n[2] Target (neto = D - R) por banco")
tgt = df_key.dropna(subset=["target"])
print(f"    Observaciones con target: {len(tgt):,} / {n_total:,} "
      f"({len(tgt)/n_total:.1%})")
stats = (
    tgt.groupby("banco")["target"]
    .agg(n="count", media="mean", std="std",
         p5=lambda x: x.quantile(0.05), p50="median",
         p95=lambda x: x.quantile(0.95))
    .round(0)
)
print(stats.to_string())

# ─────────────────────────────────────────────────────────────────
# 4. Pase único en bloques: NaN + min/max por feature
#    Peak RAM = BATCH_SIZE × n_features × 8 bytes ≈ 36 MB por bloque
# ─────────────────────────────────────────────────────────────────
print(f"\n[3/5] Calculando NaN y rangos en bloques de {BATCH_SIZE:,} filas...")
null_sum = {c: 0   for c in cols_feat}
col_min  = {c: None for c in cols_feat}
col_max  = {c: None for c in cols_feat}
filas_leidas = 0

for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=cols_feat):
    df_batch = batch.to_pandas()
    for c in cols_feat:
        serie = df_batch[c]
        null_sum[c] += int(serie.isna().sum())
        validos = serie.dropna()
        if len(validos):
            bmin, bmax = validos.min(), validos.max()
            col_min[c] = bmin if col_min[c] is None else min(col_min[c], bmin)
            col_max[c] = bmax if col_max[c] is None else max(col_max[c], bmax)
    filas_leidas += len(df_batch)
    print(f"  ... {filas_leidas:,} / {n_total:,} filas", end="\r")
print()

nan_pct = pd.Series({c: null_sum[c] / n_total for c in cols_feat})

print(f"\n[3] Valores faltantes por feature (solo con NaN)")
nan_pct_nonzero = nan_pct[nan_pct > 0].sort_values(ascending=False)
if nan_pct_nonzero.empty:
    print("    Sin NaN en features ✓")
else:
    for feat, pct in nan_pct_nonzero.items():
        print(f"    {feat:<35s}: {pct:.1%}")

print(f"\n[4] Filas por banco y año")
df_key["año"] = df_key["fecha_t"].dt.year
pivot_cov = df_key.groupby(["banco", "año"]).size().unstack(fill_value=0)
print(pivot_cov.to_string())
df_key.drop(columns=["año"], inplace=True)

print(f"\n[5] Rango de features clave")
checks = {"h": (1, None), "dia_semana": (0, 4), "mes": (1, 12), "pos_en_mes": (1, None)}
for feat, (vmin, vmax) in checks.items():
    if feat not in cols_feat:
        continue
    fmin, fmax = col_min[feat], col_max[feat]
    if fmin is None:
        print(f"    {feat:<20s}: sin datos")
        continue
    ok = (fmin >= vmin) and (vmax is None or fmax <= vmax)
    marca = "✓" if ok else "✗ REVISAR"
    print(f"    {feat:<20s}: [{fmin:.0f}, {fmax:.0f}]  {marca}")

cols_bin = [c for c in cols_feat if c.startswith("is_")]
for feat in cols_bin:
    fmin, fmax = col_min.get(feat), col_max.get(feat)
    if fmin is None:
        continue
    ok = (fmin >= 0) and (fmax <= 1)
    marca = "✓" if ok else f"✗ rango [{fmin}, {fmax}]"
    print(f"    {feat:<35s}: {marca}")

print(f"\n{'='*65}\n")

# ─────────────────────────────────────────────────────────────────
# 5. Gráficos diagnósticos
# ─────────────────────────────────────────────────────────────────

# ── FIGURA 1: NaN por feature ─────────────────────────────────────
nan_all = nan_pct.sort_values(ascending=True)
colores_nan = ["#d73027" if p > 0.3 else "#fc8d59" if p > 0.05
               else "#91bfdb" if p > 0 else "#4575b4" for p in nan_all]

fig1, ax = plt.subplots(figsize=(10, max(6, len(nan_all) * 0.28)))
bars = ax.barh(nan_all.index, nan_all.values * 100, color=colores_nan,
               edgecolor="white", height=0.7)
for bar, pct in zip(bars, nan_all.values):
    if pct > 0:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1%}", va="center", fontsize=7.5)
ax.axvline(0,  color="black",  lw=0.8)
ax.axvline(5,  color="#fc8d59", lw=1, ls="--", alpha=0.7, label="5%")
ax.axvline(30, color="#d73027", lw=1, ls="--", alpha=0.7, label="30%")
ax.set_xlabel("% valores faltantes")
ax.set_title("Valores Faltantes por Feature", fontweight="bold")
ax.legend(fontsize=8)
ax.set_xlim(0, max(nan_all.max() * 100 + 10, 15))
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(DIR_OUTPUT / "01_nan_por_feature.png", dpi=150, bbox_inches="tight")
plt.show()

# ── FIGURA 3: Target por banco (boxplot) ──────────────────────────
tgt_banco = [
    df_key.loc[(df_key["banco"] == b) & df_key["target"].notna(), "target"].values / 1e6
    for b in bancos
]

fig3, ax = plt.subplots(figsize=(max(10, len(bancos) * 1.4), 6))
bp = ax.boxplot(tgt_banco, labels=bancos, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", lw=1.5))
for patch in bp["boxes"]:
    patch.set_facecolor("steelblue")
    patch.set_alpha(0.6)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel("Neto (MM USD)")
ax.set_title("Distribución del Target por Banco (sin outliers extremos)", fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(DIR_OUTPUT / "03_target_por_banco.png", dpi=150, bbox_inches="tight")
plt.show()

# ── FIGURA 4: Cobertura temporal por banco (heatmap) ──────────────
df_key["año"] = df_key["fecha_t"].dt.year
pivot      = df_key.groupby(["banco", "año"]).size().unstack(fill_value=0)
pivot_norm = pivot.div(pivot.max(axis=1), axis=0)
df_key.drop(columns=["año"], inplace=True)

fig4, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.6),
                                  max(4,  len(pivot.index)  * 0.5)))
im = ax.imshow(pivot_norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=8)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        n = pivot.values[i, j]
        if n > 0:
            ax.text(j, i, f"{n:,}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if pivot_norm.values[i, j] > 0.5 else "black")
plt.colorbar(im, ax=ax, label="Densidad relativa")
ax.set_title("Cobertura Temporal por Banco (filas en la matriz)", fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_OUTPUT / "04_cobertura_temporal.png", dpi=150, bbox_inches="tight")
plt.show()

# ── FIGURA 5: Correlación con target (h mínimo disponible) ───────────────────
# Usa el menor horizonte de la matriz — los features de t son más predictivos
# cuando la fecha objetivo está más cercana
h_min_disp = int(df_key["h"].min())
print(f"Cargando filas h={h_min_disp} para correlaciones...")
df_h1 = pd.read_parquet(
    RUTA_MATRIZ,
    filters=[("h", "==", h_min_disp)],
    columns=cols_feat + ["target"],
)
df_h1 = df_h1.dropna(subset=["target"])
feat_num = [c for c in cols_feat if pd.api.types.is_numeric_dtype(df_h1[c])]
corrs = (
    df_h1[feat_num + ["target"]]
    .corr()["target"]
    .drop("target")
    .dropna()
    .sort_values(key=abs, ascending=True)
)
colores_corr = ["#d73027" if v < 0 else "#4575b4" for v in corrs]

fig5, ax = plt.subplots(figsize=(10, max(6, len(corrs) * 0.28)))
ax.barh(corrs.index, corrs.values, color=colores_corr, edgecolor="white", height=0.7)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Correlación de Pearson con target")
ax.set_title(f"Correlación de Features con Target (h = {h_min_disp})", fontweight="bold")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(DIR_OUTPUT / "05_correlacion_features_target.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nGráficos guardados en: {DIR_OUTPUT}")

# ─────────────────────────────────────────────────────────────────
# 6. Exportar muestra a Excel para inspección visual
#    Lee las primeras N filas del Parquet sin cargar el archivo completo
# ─────────────────────────────────────────────────────────────────
N_MUESTRA = 2_000
print(f"\nExportando muestra de {N_MUESTRA:,} filas a Excel...")

batch_muestra = next(pf.iter_batches(batch_size=N_MUESTRA))
df_muestra = batch_muestra.to_pandas()

ruta_muestra = DIR_OUTPUT / "muestra_matriz_features.xlsx"
with pd.ExcelWriter(ruta_muestra, engine="openpyxl") as writer:
    df_muestra.to_excel(writer, sheet_name="Muestra", index=False)

    # Hoja de resumen: NaN por feature
    resumen_nan = (
        nan_pct.rename("pct_nan")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    resumen_nan["pct_nan"] = (resumen_nan["pct_nan"] * 100).round(1)
    resumen_nan.to_excel(writer, sheet_name="NaN_por_feature", index=False)

print(f"  Muestra guardada en: {ruta_muestra}")

# ─────────────────────────────────────────────────────────────────
# 7. Resumen de archivos del proyecto
# ─────────────────────────────────────────────────────────────────
sep = "=" * 65
print(f"\n{sep}")
print("  ARCHIVOS PRINCIPALES DEL PROYECTO")
print(sep)
print("""
  SCRIPTS  (raíz del repositorio)
  ├── step001_build_feature_matrix.py   → construye la matriz de features
  │     Ejecutar primero. Genera matriz_features.parquet.
  │
  ├── aux_visualizar_series.py          → gráficos exploratorios de las
  │     series de transacciones bancarias (retiros, depósitos, bancos).
  │
  ├── aux_validar_matriz.py             → validación de la matriz de features
  │     (este script). Gráficos de NaN, cobertura y correlaciones.
  │
  ├── aux_fanchart_ejemplo.py           → fan chart para una fecha calendario
  │     fija a través de todos los años (ej. 6 de julio).
  │
  └── aux_validacion_flujos.py          → validación cruzada entre
        Transacciones_BancaLocal.xlsx y DepositosSF.xlsx del BCRP.

  DATOS  (no versionados en git — solo en disco local)
  ├── 1. Data/Raw/
  │   ├── Transacciones_BancaLocal.xlsx → transacciones diarias por banco
  │   └── series_bcrp.xlsx              → EMBI, Tasa Ref, TC compra/venta
  │
  └── 1. Data/Clean/
      ├── matriz_features.parquet       → matriz de entrenamiento (~5M filas)
      └── diccionario_variables.xlsx    → descripción de cada feature

  OUTPUTS  (generados automáticamente)
  ├── 2. Output/aux_validar_matriz/
  │   ├── 01_nan_por_feature.png
  │   ├── 03_target_por_banco.png
  │   ├── 04_cobertura_temporal.png
  │   ├── 05_correlacion_features_target.png
  │   └── muestra_matriz_features.xlsx  → muestra de 2,000 filas + NaN
  │
  ├── 2. Output/aux_fanchart_ejemplo/
  │   └── fanchart_<BANCO>_julio06_todos_anios.png
  │
  ├── 2. Output/aux_visualizar_series/  → gráficos de series bancarias
  └── 2. Output/aux_validacion_flujos/  → validación DepositosSF vs Tx
""")
print(sep)
