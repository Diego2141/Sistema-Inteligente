# -*- coding: utf-8 -*-
"""
aux_estrategia_encaje.py
Identifica en qué meses BBVA utilizó la estrategia de sobreencaje temprano
y retiro tardío: acumular encaje muy por encima del exigible en los primeros
días del mes y luego retirar fuerte en los últimos días.

Señal principal: faltante_lag1
  faltante = exigible_total_mes − encaje_acumulado(t-1)
  · Negativo temprano (días 1-10) → ya cubrió más de lo necesario (sobreencaje)
  · Retorno a 0 al cierre → retiró el exceso a fin de mes

Fuente: matriz_features.parquet (banco BBVA, h=2 para obtener 1 fila por fecha_t)

Outputs → DIR_OUTPUT /
    00_heatmap_estrategia.png   — heatmap año × mes, intensidad del sobreencaje
    01_trayectoria_mensual.png  — perfil promedio de faltante por tipo de mes
    02_serie_tiempo.png         — timeline del score mensual con clasificación
    resultados_estrategia.xlsx  — tabla detallada por mes
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

###############################################################################
# CONFIGURACIÓN
###############################################################################

BASE         = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_PARQUET = BASE / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE / "2. Output" / "aux_estrategia_encaje"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO_REF = "BBVA"   # banco con datos de encaje

# Umbral de sobreencaje: faltante normalizado < -UMBRAL_SOBREENCAJE en los
# primeros DIA_CORTE días hábiles del mes → estrategia activa
UMBRAL_SOBREENCAJE = 0.25   # 25% de sobrecobertura respecto al exigible total
DIA_CORTE          = 12     # primeros N días hábiles del mes

###############################################################################
# CARGA
###############################################################################

print("=" * 65)
print("  aux_estrategia_encaje — Identificación de sobreencaje BBVA")
print("=" * 65)

if not RUTA_PARQUET.exists():
    raise FileNotFoundError(
        f"Matriz no encontrada: {RUTA_PARQUET}\n"
        "Ejecutar step001_build_feature_matrix.py primero."
    )

print(f"\n  Cargando matriz  banco='{BANCO_REF}'  h=2 ...")
try:
    df = pd.read_parquet(
        RUTA_PARQUET,
        filters=[("banco", "==", BANCO_REF), ("h", "==", 2)],
        columns=["fecha_t", "faltante_lag1", "encaje_lag1",
                 "techo_restante_lag1", "proporcion_usada"],
    )
except Exception:
    df_all = pd.read_parquet(RUTA_PARQUET)
    df = df_all[(df_all["banco"] == BANCO_REF) & (df_all["h"] == 2)][
        ["fecha_t", "faltante_lag1", "encaje_lag1",
         "techo_restante_lag1", "proporcion_usada"]
    ].copy()

df["fecha_t"] = pd.to_datetime(df["fecha_t"])
df = df.sort_values("fecha_t").reset_index(drop=True)

# Filtrar solo filas con datos de encaje
df = df.dropna(subset=["faltante_lag1", "encaje_lag1"])

if df.empty:
    raise SystemExit(
        "  No hay datos de encaje para BBVA. "
        "Verificar que banco_encaje='BBVA' en PARAMS de step001."
    )

print(f"  Filas disponibles: {len(df):,}  |  "
      f"{df['fecha_t'].min().date()} → {df['fecha_t'].max().date()}")

###############################################################################
# CONSTRUCCIÓN DE INDICADORES POR MES
###############################################################################

# Día hábil dentro del mes (rank 1, 2, 3, ...)
df["mes_periodo"] = df["fecha_t"].dt.to_period("M")
df["dia_habil"]   = (df.groupby("mes_periodo")["fecha_t"]
                     .rank(method="first").astype(int))
df["n_dias_mes"]  = df.groupby("mes_periodo")["dia_habil"].transform("max")

# Normalizar faltante por encaje_lag1 medio del mes (proxy del exigible)
# faltante = exigible_total - encaje_acum  → si encaje_lag1 ≈ encaje_diario,
# encaje_lag1 × n_dias_mes ≈ exigible_total
df["encaje_medio_mes"] = df.groupby("mes_periodo")["encaje_lag1"].transform("mean")
df["faltante_norm"]    = df["faltante_lag1"] / (
    df["encaje_medio_mes"] * df["n_dias_mes"]
).replace(0, np.nan)

# ── Métricas por mes ─────────────────────────────────────────────────────────

def _metricas_mes(g):
    """Para un grupo mensual, devuelve los indicadores de la estrategia."""
    g = g.sort_values("dia_habil")

    # Ventana temprana (días 1..DIA_CORTE) y tardía (últimos 5 días)
    early = g[g["dia_habil"] <= DIA_CORTE]
    late  = g[g["dia_habil"] > g["n_dias_mes"].iloc[0] - 5]

    # Faltante mínimo en ventana temprana (más negativo = más sobreencaje)
    min_faltante_early = early["faltante_norm"].min() if len(early) else np.nan
    dia_min            = (early.loc[early["faltante_norm"].idxmin(), "dia_habil"]
                          if len(early) and not early["faltante_norm"].isna().all()
                          else np.nan)

    # Retiro tardío: cuánto aumenta el faltante en los últimos 5 días
    # (positivo = faltante sube = están retirando encaje)
    retiro_tardio = (late["faltante_norm"].max() - late["faltante_norm"].min()
                     if len(late) >= 2 else np.nan)

    # Proporcion_usada en últimos días (qué fracción del techo ya usaron)
    prop_fin = late["proporcion_usada"].max() if len(late) else np.nan

    # Score combinado: intensidad de sobreencaje temprano + retiro tardío
    score_sobreencaje = -min_faltante_early if np.isfinite(min_faltante_early) else np.nan
    score = (score_sobreencaje * 0.7 + (retiro_tardio if np.isfinite(retiro_tardio) else 0) * 0.3
             if np.isfinite(score_sobreencaje) else np.nan)

    estrategia = (
        np.isfinite(min_faltante_early)
        and min_faltante_early < -UMBRAL_SOBREENCAJE
        and np.isfinite(dia_min)
        and dia_min <= DIA_CORTE
    )

    return pd.Series({
        "min_faltante_norm_early": round(min_faltante_early, 3) if np.isfinite(min_faltante_early) else np.nan,
        "dia_min_faltante":        dia_min,
        "retiro_tardio_norm":      round(retiro_tardio,       3) if np.isfinite(retiro_tardio) else np.nan,
        "prop_usada_fin":          round(prop_fin,            3) if np.isfinite(prop_fin) else np.nan,
        "score":                   round(score,               3) if np.isfinite(score) else np.nan,
        "estrategia":              estrategia,
        "n_dias":                  int(g["n_dias_mes"].iloc[0]),
    })

print("\n  Calculando métricas por mes ...")
df_meses = (df.groupby("mes_periodo")
            .apply(_metricas_mes)
            .reset_index())
df_meses["anio"] = df_meses["mes_periodo"].dt.year
df_meses["mes"]  = df_meses["mes_periodo"].dt.month
df_meses["mes_nombre"] = df_meses["mes_periodo"].dt.strftime("%Y-%m")

n_est = df_meses["estrategia"].sum()
n_tot = df_meses["estrategia"].notna().sum()
print(f"  Meses con estrategia detectada: {n_est} / {n_tot} "
      f"({100*n_est/max(n_tot,1):.0f}%)")

###############################################################################
# GRÁFICO 1 — HEATMAP AÑO × MES
###############################################################################

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

pivot_score = df_meses.pivot(index="anio", columns="mes", values="score").sort_index()
pivot_estrat = df_meses.pivot(index="anio", columns="mes", values="estrategia").sort_index()

MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

fig, ax = plt.subplots(figsize=(14, max(4, len(pivot_score) * 0.55 + 1.5)))
fig.suptitle("Estrategia sobreencaje BBVA — intensidad por año y mes\n"
             "(más oscuro = mayor sobreencaje temprano + retiro tardío)",
             fontweight="bold", fontsize=10)

vmax = pivot_score.values[np.isfinite(pivot_score.values)].max() if np.any(np.isfinite(pivot_score.values)) else 1
cmap = plt.cm.YlOrRd

im = ax.imshow(pivot_score.values.astype(float), aspect="auto",
               cmap=cmap, vmin=0, vmax=vmax)

ax.set_xticks(range(12))
ax.set_xticklabels(MESES)
ax.set_yticks(range(len(pivot_score)))
ax.set_yticklabels(pivot_score.index.astype(str))

# Marcar celdas con estrategia detectada
for i, anio in enumerate(pivot_score.index):
    for j, mes in enumerate(range(1, 13)):
        val = pivot_score.loc[anio, mes] if mes in pivot_score.columns else np.nan
        est = pivot_estrat.loc[anio, mes] if mes in pivot_estrat.columns else False
        if np.isfinite(val):
            color_txt = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color_txt)
            if est:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="blue", linewidth=2))

plt.colorbar(im, ax=ax, label="Score estrategia", shrink=0.7)
ax.set_xlabel("Mes")
ax.set_ylabel("Año")

# Leyenda
from matplotlib.patches import Patch, Rectangle
legend_elems = [
    Patch(facecolor=cmap(0.3), label="Sobreencaje moderado"),
    Patch(facecolor=cmap(0.8), label="Sobreencaje intenso"),
    Rectangle((0,0),1,1, fill=False, edgecolor="blue", linewidth=2,
              label=f"Estrategia confirmada (sobreencaje >{UMBRAL_SOBREENCAJE*100:.0f}% en día ≤{DIA_CORTE})"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=7,
          bbox_to_anchor=(1.35, 1.0))

plt.tight_layout()
p1 = DIR_OUTPUT / "00_heatmap_estrategia.png"
fig.savefig(p1, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\n  Guardado: {p1.name}")

###############################################################################
# GRÁFICO 2 — PERFIL PROMEDIO DE FALTANTE (estrategia vs normal)
###############################################################################

meses_est  = set(df_meses.loc[df_meses["estrategia"], "mes_periodo"].astype(str))
meses_norm = set(df_meses.loc[~df_meses["estrategia"] & df_meses["estrategia"].notna(),
                               "mes_periodo"].astype(str))

df["tipo"] = df["mes_periodo"].astype(str).map(
    lambda m: "Estrategia" if m in meses_est else
              ("Normal" if m in meses_norm else None)
)

fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle("Perfil promedio de faltante normalizado por día hábil del mes\n"
             "(meses con estrategia vs meses normales)", fontweight="bold")

for tipo, color, ls in [("Estrategia", "#E53935", "-"), ("Normal", "#1976D2", "--")]:
    sub = df[df["tipo"] == tipo]
    if sub.empty:
        continue
    perfil = sub.groupby("dia_habil")["faltante_norm"].mean()
    dias   = perfil.index.values
    vals   = perfil.values
    ci     = sub.groupby("dia_habil")["faltante_norm"].std()
    ax.plot(dias, vals, color=color, linestyle=ls, linewidth=2, label=tipo)
    ax.fill_between(dias, vals - ci.reindex(dias).fillna(0).values,
                    vals + ci.reindex(dias).fillna(0).values,
                    alpha=0.12, color=color)

ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax.axhline(-UMBRAL_SOBREENCAJE, color="gray", linewidth=0.7, linestyle="-.",
           label=f"Umbral sobreencaje (−{UMBRAL_SOBREENCAJE*100:.0f}%)")
ax.axvline(DIA_CORTE, color="orange", linewidth=0.9, linestyle="--",
           label=f"Día corte = {DIA_CORTE}")
ax.set_xlabel("Día hábil del mes")
ax.set_ylabel("Faltante normalizado")
ax.legend(fontsize=8)
ax.set_xlim(1, 22)

plt.tight_layout()
p2 = DIR_OUTPUT / "01_trayectoria_mensual.png"
fig.savefig(p2, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Guardado: {p2.name}")

###############################################################################
# GRÁFICO 3 — SERIE DE TIEMPO DEL SCORE
###############################################################################

df_meses_clean = df_meses.dropna(subset=["score"]).copy()
df_meses_clean["fecha_plot"] = pd.to_datetime(
    df_meses_clean["mes_periodo"].dt.start_time
)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle("Estrategia sobreencaje BBVA — evolución temporal", fontweight="bold")

# Panel superior: score mensual
ax = axes[0]
colores = df_meses_clean["estrategia"].map({True: "#E53935", False: "#90A4AE"})
ax.bar(df_meses_clean["fecha_plot"], df_meses_clean["score"],
       width=20, color=colores, alpha=0.85)
ax.axhline(df_meses_clean["score"].quantile(0.75), color="orange",
           linewidth=0.9, linestyle="--", label="P75 score")
# Rolling 12 meses
sc_roll = df_meses_clean.set_index("fecha_plot")["score"].rolling("365D").mean()
ax.plot(sc_roll.index, sc_roll.values, color="black", linewidth=1.5,
        label="Media móvil 12m")
ax.set_ylabel("Score estrategia")
ax.legend(fontsize=8)

# Patches para los años
from matplotlib.patches import Patch
legend_patch = [
    Patch(facecolor="#E53935", alpha=0.85, label="Estrategia activa"),
    Patch(facecolor="#90A4AE", alpha=0.85, label="Sin estrategia"),
]
ax.legend(handles=legend_patch + [
    plt.Line2D([0],[0], color="black", lw=1.5, label="Media móvil 12m"),
    plt.Line2D([0],[0], color="orange", lw=0.9, ls="--", label="P75 score"),
], fontsize=8)

# Panel inferior: faltante mínimo normalizado temprano
ax2 = axes[1]
ax2.bar(df_meses_clean["fecha_plot"],
        df_meses_clean["min_faltante_norm_early"],
        width=20,
        color=df_meses_clean["estrategia"].map({True: "#E53935", False: "#90A4AE"}),
        alpha=0.85)
ax2.axhline(-UMBRAL_SOBREENCAJE, color="navy", linewidth=0.9, linestyle="--",
            label=f"Umbral −{UMBRAL_SOBREENCAJE*100:.0f}%")
ax2.axhline(0, color="black", linewidth=0.6, linestyle=":")
ax2.set_ylabel("Faltante normalizado (días 1–12)")
ax2.set_xlabel("Fecha")
ax2.legend(fontsize=8)
ax2.invert_yaxis()   # negativo hacia arriba = visualmente más intuitivo

plt.tight_layout()
p3 = DIR_OUTPUT / "02_serie_tiempo.png"
fig.savefig(p3, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Guardado: {p3.name}")

###############################################################################
# EXPORTAR EXCEL
###############################################################################

print("\n  Exportando Excel ...")
RUTA_EXCEL = DIR_OUTPUT / "resultados_estrategia.xlsx"

cols_export = ["mes_nombre", "anio", "mes", "estrategia", "score",
               "min_faltante_norm_early", "dia_min_faltante",
               "retiro_tardio_norm", "prop_usada_fin", "n_dias"]
df_export = df_meses[cols_export].copy()
df_export.columns = [
    "Mes", "Año", "Nro_mes", "Estrategia",
    "Score", "Sobreencaje_temprano", "Dia_minimo_faltante",
    "Retiro_tardio", "Proporcion_usada_fin", "Dias_habiles_mes"
]

try:
    with pd.ExcelWriter(RUTA_EXCEL, engine="openpyxl") as writer:
        df_export.to_excel(writer, sheet_name="Por_mes", index=False)

        # Resumen anual: cuántos meses con estrategia por año
        resumen = (df_meses.groupby("anio")
                   .agg(n_meses_total=("estrategia", "count"),
                        n_meses_estrategia=("estrategia", "sum"),
                        score_medio=("score", "mean"),
                        score_max=("score", "max"))
                   .reset_index())
        resumen["pct_estrategia"] = (resumen["n_meses_estrategia"]
                                     / resumen["n_meses_total"] * 100).round(1)
        resumen.to_excel(writer, sheet_name="Resumen_anual", index=False)

        # Meses clasificados como estrategia
        df_solo_est = df_export[df_export["Estrategia"]].copy()
        df_solo_est.to_excel(writer, sheet_name="Meses_estrategia", index=False)

    print(f"  Guardado: {RUTA_EXCEL.name}")
except Exception as e:
    print(f"  AVISO: no se pudo exportar Excel — {e}")

###############################################################################
# RESUMEN CONSOLA
###############################################################################

print("\n" + "=" * 65)
print("  RESUMEN ANUAL")
print("=" * 65)
print(f"  {'Año':<6} {'Meses estrategia':>18} {'Score medio':>12} {'Frecuencia':>12}")
print(f"  {'-'*6} {'-'*18} {'-'*12} {'-'*12}")
for _, r in resumen.sort_values("anio").iterrows():
    flag = " ← TRIMESTRAL" if r["n_meses_estrategia"] in (3,4) else (
           " ← MENSUAL" if r["n_meses_estrategia"] >= 8 else "")
    print(f"  {int(r['anio']):<6} "
          f"{int(r['n_meses_estrategia']):>6} / {int(r['n_meses_total']):<10} "
          f"{r['score_medio']:>11.3f}  "
          f"{r['pct_estrategia']:>10.0f}%{flag}")

print(f"\n  Archivos en: {DIR_OUTPUT}")
print("=" * 65)
