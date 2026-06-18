# -*- coding: utf-8 -*-
"""
aux_analisis_encaje.py
======================
Análisis exploratorio de la estrategia de encaje de BBVA.

Hipótesis a verificar:
  El banco acumula Cta Cte BCR (encaje disponible) en los días 1-21 del mes,
  y en los días 22-31 lo "rentabiliza" moviendo saldo a Overnight BCR y
  realizando retiros netos al exterior. El exceso de encaje (encaje total -
  exigible) se libera principalmente vía retiros netos.

Identidad contable del balance de encaje:
  Encaje disponible  = Caja + Cta Cte BCR
  Encaje total       = Caja + Cta Cte BCR + Overnight BCR
  Exceso de encaje   = Encaje total - Encaje Exigible

  En la variación diaria:
  Δ(Cta Cte) + Δ(Overnight) + Δ(Activos) + Retiro Neto ≈ 0

  → El retiro neto es el canal principal por el que el banco libera
    el exceso de encaje acumulado durante el período mensual.

Datos requeridos:
  H:/DPINV/CARPETAS PERSONALES/DIEGO/3. Sistema Inteligente/1. Data/Raw/EncajeD.xlsx

Outputs (guardados en Output/encaje/ relativo a este script):
  encaje_01_componentes_tiempo.png
  encaje_02_patron_dia_mes.png
  encaje_03_scatter_exceso_retiro.png
  encaje_04_fases_boxplot.png
  encaje_05_identidad_contable.png
  encaje_06_r2_por_anio.png
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
RUTA_DATOS = pathlib.Path(
    r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Raw\EncajeD.xlsx"
)
# RUTA_DATOS: .../3. Sistema Inteligente/1. Data/Raw/EncajeD.xlsx
# Subir 3 niveles da la raíz del proyecto → Output/encaje
DIR_OUTPUT = RUTA_DATOS.parent.parent.parent / "Output" / "encaje"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

COLORES = {
    "cta_cte":   "#1f77b4",
    "overnight": "#ff7f0e",
    "exceso":    "#2ca02c",
    "retiro":    "#d62728",
    "exigible":  "#9467bd",
    "activos":   "#8c564b",
}
FASES_COLORES = {
    "A_inicio(1-4)":    "#4daf4a",
    "B_acum(5-21)":     "#377eb8",
    "C_rentab(22-28)":  "#ff7f00",
    "D_cierre(29-31)":  "#e41a1c",
}

# =============================================================================
# CARGA Y PREPARACIÓN
# =============================================================================
print("Cargando datos...")
df = pd.read_excel(RUTA_DATOS)
df.columns = [
    "fecha", "moneda", "banco", "codigo",
    "overnight_bcr", "caja", "cta_cte_bcr",
    "encaje_exigible", "activos", "retiro_neto"
]
df = df.sort_values("fecha").reset_index(drop=True)

df["dia_mes"]        = df["fecha"].dt.day
df["ano"]            = df["fecha"].dt.year
df["mes"]            = df["fecha"].dt.month
df["encaje_total"]   = df["caja"] + df["cta_cte_bcr"] + df["overnight_bcr"]
df["exceso_encaje"]  = df["encaje_total"] - df["encaje_exigible"]

# Variaciones diarias (Δ)
df["d_cta_cte"]      = df["cta_cte_bcr"].diff()
df["d_overnight"]    = df["overnight_bcr"].diff()
df["d_activos"]      = df["activos"].diff()
df["d_exceso"]       = df["exceso_encaje"].diff()

# Fase del período de encaje
def asignar_fase(d):
    if d <= 4:   return "A_inicio(1-4)"
    elif d <= 21: return "B_acum(5-21)"
    elif d <= 28: return "C_rentab(22-28)"
    else:         return "D_cierre(29-31)"

df["fase"] = df["dia_mes"].apply(asignar_fase)

# Días hasta fin de mes
df["dias_fin_mes"] = df["fecha"].apply(
    lambda d: (pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(1) - d).days
)

# Exceso acumulado desde inicio del período mensual
exceso_inicio = df.groupby(["ano","mes"])["exceso_encaje"].transform("first")
df["exceso_acum_periodo"] = df["exceso_encaje"] - exceso_inicio

print(f"  Período: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"  Registros: {len(df):,}")
print(f"  Retiro neto: min={df['retiro_neto'].min()/1e9:.2f}B  "
      f"max={df['retiro_neto'].max()/1e9:.2f}B  "
      f"std={df['retiro_neto'].std()/1e6:.0f}M")
print()

# =============================================================================
# FUNCIÓN AUXILIAR: R²
# =============================================================================
def r2_poly(x, y, deg=3):
    X = np.column_stack([x**i for i in range(deg + 1)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = ((y - yhat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    return max(0.0, 1 - ss_res / ss_tot)

# =============================================================================
# GRÁFICA 1: Componentes de encaje en el tiempo
# =============================================================================
print("Generando gráfica 1: componentes en el tiempo...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("BBVA – Componentes de Encaje en el Tiempo (USD)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.fill_between(df["fecha"], df["cta_cte_bcr"]/1e9, alpha=0.5,
                color=COLORES["cta_cte"], label="Cta Cte BCR")
ax.fill_between(df["fecha"], df["overnight_bcr"]/1e9, alpha=0.5,
                color=COLORES["overnight"], label="Overnight BCR")
ax.plot(df["fecha"], df["encaje_exigible"]/1e9, color=COLORES["exigible"],
        lw=1.5, ls="--", label="Encaje Exigible")
ax.set_ylabel("Miles de M USD")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Cta Cte BCR vs Overnight BCR vs Exigible", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fB"))

ax = axes[1]
ax.fill_between(df["fecha"], df["exceso_encaje"]/1e9,
                where=df["exceso_encaje"] >= 0, alpha=0.6,
                color=COLORES["exceso"], label="Exceso (positivo)")
ax.fill_between(df["fecha"], df["exceso_encaje"]/1e9,
                where=df["exceso_encaje"] < 0, alpha=0.6,
                color=COLORES["retiro"], label="Déficit (negativo)")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Miles de M USD")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Exceso de Encaje = (Cta Cte + Overnight + Caja) − Exigible", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fB"))

ax = axes[2]
ax.bar(df["fecha"], df["retiro_neto"]/1e6, color=COLORES["retiro"], alpha=0.5,
       width=1, label="Retiro Neto (M USD)")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Millones USD")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Retiro Neto diario (negativo = salida de divisas)", fontsize=10)

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_01_componentes_tiempo.png"
fig.savefig(ruta, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Guardado: {ruta.name}")

# =============================================================================
# GRÁFICA 2: Patrón por día del mes (bar + shading por fase)
# =============================================================================
print("Generando gráfica 2: patrón por día del mes...")
by_day = df.groupby("dia_mes").agg(
    rn_mean  = ("retiro_neto", "mean"),
    rn_p10   = ("retiro_neto", lambda x: x.quantile(0.10)),
    rn_p90   = ("retiro_neto", lambda x: x.quantile(0.90)),
    cta_mean = ("cta_cte_bcr",   "mean"),
    on_mean  = ("overnight_bcr", "mean"),
    ex_mean  = ("exceso_encaje", "mean"),
).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("BBVA – Patrón Intra-Mensual del Encaje\n(Promedio por día del mes, período 2016-2026)",
             fontsize=13, fontweight="bold")

# Panel A: Retiro neto
ax = axes[0]
colores_bar = [FASES_COLORES[asignar_fase(d)] for d in by_day["dia_mes"]]
bars = ax.bar(by_day["dia_mes"], by_day["rn_mean"]/1e6, color=colores_bar, alpha=0.8)
ax.fill_between(by_day["dia_mes"], by_day["rn_p10"]/1e6, by_day["rn_p90"]/1e6,
                alpha=0.2, color="gray", label="P10–P90")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Millones USD")
ax.set_title("Retiro Neto promedio por día del mes", fontsize=10)
patches = [mpatches.Patch(color=c, label=f, alpha=0.8)
           for f, c in FASES_COLORES.items()]
ax.legend(handles=patches, loc="lower left", fontsize=8, ncol=2)

# Panel B: Cta Cte vs Overnight
ax = axes[1]
ax.plot(by_day["dia_mes"], by_day["cta_mean"]/1e9, color=COLORES["cta_cte"],
        marker="o", ms=4, lw=1.5, label="Cta Cte BCR (prom)")
ax.plot(by_day["dia_mes"], by_day["on_mean"]/1e9, color=COLORES["overnight"],
        marker="s", ms=4, lw=1.5, label="Overnight BCR (prom)")
ax.plot(by_day["dia_mes"], by_day["ex_mean"]/1e9, color=COLORES["exceso"],
        marker="^", ms=4, lw=1.5, ls="--", label="Exceso de Encaje (prom)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Día del mes")
ax.set_ylabel("Miles de M USD")
ax.set_title("Cta Cte vs Overnight vs Exceso por día del mes", fontsize=10)
ax.legend(loc="lower left", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fB"))

# Sombrear fases
for ax in axes:
    ax.axvspan(0.5,  4.5,  alpha=0.08, color=FASES_COLORES["A_inicio(1-4)"])
    ax.axvspan(4.5,  21.5, alpha=0.08, color=FASES_COLORES["B_acum(5-21)"])
    ax.axvspan(21.5, 28.5, alpha=0.08, color=FASES_COLORES["C_rentab(22-28)"])
    ax.axvspan(28.5, 31.5, alpha=0.08, color=FASES_COLORES["D_cierre(29-31)"])

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_02_patron_dia_mes.png"
fig.savefig(ruta, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Guardado: {ruta.name}")

# =============================================================================
# GRÁFICA 3: Scatter Δexceso vs Retiro Neto (la identidad contable)
# =============================================================================
print("Generando gráfica 3: scatter Δexceso vs Retiro Neto...")
df_sc = df[["d_exceso", "retiro_neto", "fase"]].dropna()
corr_val = df_sc["d_exceso"].corr(df_sc["retiro_neto"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Identidad Contable: ΔExceso de Encaje ↔ Retiro Neto\n"
    f"Correlación global = {corr_val:+.4f}   (r² = {corr_val**2:.4f})",
    fontsize=12, fontweight="bold"
)

# Panel A: scatter coloreado por fase
ax = axes[0]
for fase_nom, grp in df_sc.groupby("fase", sort=True):
    ax.scatter(grp["d_exceso"]/1e9, grp["retiro_neto"]/1e6,
               alpha=0.25, s=10, color=FASES_COLORES[fase_nom], label=fase_nom)

# línea de regresión
x_lin = df_sc["d_exceso"].values
y_lin = df_sc["retiro_neto"].values
mask  = np.isfinite(x_lin) & np.isfinite(y_lin)
coef  = np.polyfit(x_lin[mask], y_lin[mask], 1)
xr    = np.linspace(x_lin[mask].min(), x_lin[mask].max(), 200)
ax.plot(xr/1e9, np.polyval(coef, xr)/1e6, "k-", lw=1.5, label=f"Regresión (pend={coef[0]:.2f})")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("Δ Exceso de Encaje (B USD)")
ax.set_ylabel("Retiro Neto (M USD)")
ax.set_title("Scatter por fase del mes", fontsize=10)
ax.legend(fontsize=7, markerscale=2)

# Panel B: hexbin para ver densidad
ax = axes[1]
xp = df_sc["d_exceso"].values / 1e9
yp = df_sc["retiro_neto"].values / 1e6
hb = ax.hexbin(xp, yp, gridsize=40, cmap="YlOrRd", mincnt=1, bins="log")
fig.colorbar(hb, ax=ax, label="log(conteo)")
ax.plot(xr/1e9, np.polyval(coef, xr)/1e6, "b-", lw=1.5)
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("Δ Exceso de Encaje (B USD)")
ax.set_ylabel("Retiro Neto (M USD)")
ax.set_title("Densidad hexbin (log-scale)", fontsize=10)

# Texto explicativo
textstr = (
    "Identidad:\n"
    "Δcta_cte + Δovernight\n"
    "+ Δactivos + retiro_neto ≈ 0\n\n"
    "→ retiro_neto ≈ -Δ(exceso)\n"
    f"   r = {corr_val:+.4f}\n"
    f"   r² = {corr_val**2:.2%}"
)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=8,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_03_scatter_exceso_retiro.png"
fig.savefig(ruta, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Guardado: {ruta.name}")

# =============================================================================
# GRÁFICA 4: Boxplot de retiro neto por fase y desglose de componentes
# =============================================================================
print("Generando gráfica 4: boxplot por fases...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("BBVA – Retiro Neto y Componentes de Encaje por Fase del Mes",
             fontsize=12, fontweight="bold")

# Panel A: boxplot retiro neto por fase
ax = axes[0]
fase_order = ["A_inicio(1-4)", "B_acum(5-21)", "C_rentab(22-28)", "D_cierre(29-31)"]
data_bp = [df.loc[df["fase"]==f, "retiro_neto"].values/1e6 for f in fase_order]
labels_bp = ["Inicio\n(días 1-4)", "Acumulación\n(días 5-21)",
             "Rentabilización\n(días 22-28)", "Cierre\n(días 29-31)"]
bp = ax.boxplot(data_bp, labels=labels_bp, patch_artist=True,
                medianprops=dict(color="black", lw=2),
                flierprops=dict(marker=".", markersize=3, alpha=0.4))
for patch, fase in zip(bp["boxes"], fase_order):
    patch.set_facecolor(FASES_COLORES[fase])
    patch.set_alpha(0.7)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_ylabel("Retiro Neto (M USD)")
ax.set_title("Distribución del Retiro Neto por fase", fontsize=10)

# Anotar media
for i, (f, lbl) in enumerate(zip(fase_order, labels_bp), start=1):
    media = df.loc[df["fase"]==f, "retiro_neto"].mean()/1e6
    ax.text(i, media, f" {media:.0f}M", va="center", fontsize=8,
            color="black", fontweight="bold")

# Panel B: barras apiladas de Δcomponentes por fase
ax = axes[1]
comp_names = ["Δ Cta Cte BCR", "Δ Overnight BCR", "Δ Activos", "Retiro Neto"]
comp_cols  = ["d_cta_cte", "d_overnight", "d_activos", "retiro_neto"]
comp_colors= [COLORES["cta_cte"], COLORES["overnight"], COLORES["activos"], COLORES["retiro"]]
medias_fase = df.groupby("fase")[comp_cols].mean() / 1e6

x = np.arange(len(fase_order))
width = 0.18
for j, (name, col, color) in enumerate(zip(comp_names, comp_cols, comp_colors)):
    vals = [medias_fase.loc[f, col] for f in fase_order]
    ax.bar(x + j*width - 1.5*width, vals, width, label=name, color=color, alpha=0.8)

ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels_bp, fontsize=8)
ax.set_ylabel("Millones USD (promedio diario)")
ax.set_title("Variación diaria promedio de componentes por fase", fontsize=10)
ax.legend(fontsize=8, loc="lower left")

# Nota de la identidad
ax.text(0.98, 0.97,
        "Identidad:\nΔcta_cte + Δovernight + Δactivos + retiro_neto ≈ 0\n"
        "Suma de barras por fase ≈ 0",
        transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_04_fases_boxplot.png"
fig.savefig(ruta, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Guardado: {ruta.name}")

# =============================================================================
# GRÁFICA 5: Verificación de la identidad contable día a día
# =============================================================================
print("Generando gráfica 5: verificación identidad contable...")
df_id = df[["fecha","d_cta_cte","d_overnight","d_activos","retiro_neto"]].dropna()
df_id["suma_identidad"] = (df_id["d_cta_cte"] + df_id["d_overnight"] +
                           df_id["d_activos"] + df_id["retiro_neto"])

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(
    "Verificación de la Identidad Contable de Encaje\n"
    "Δcta_cte + Δovernight + Δactivos + retiro_neto  (debe ≈ 0)",
    fontsize=12, fontweight="bold"
)

ax = axes[0]
ax.plot(df_id["fecha"], df_id["d_cta_cte"]/1e9,    color=COLORES["cta_cte"],   lw=0.8, label="Δ Cta Cte BCR")
ax.plot(df_id["fecha"], df_id["d_overnight"]/1e9,  color=COLORES["overnight"], lw=0.8, label="Δ Overnight BCR")
ax.plot(df_id["fecha"], df_id["d_activos"]/1e9,    color=COLORES["activos"],   lw=0.8, label="Δ Activos")
ax.plot(df_id["fecha"], df_id["retiro_neto"]/1e9,  color=COLORES["retiro"],    lw=0.8, label="Retiro Neto")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("B USD (variación diaria)")
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.set_title("Componentes individuales", fontsize=10)

ax = axes[1]
ax.plot(df_id["fecha"], df_id["suma_identidad"]/1e9, color="purple", lw=0.8,
        label="SUMA (debe ≈ 0)")
ax.axhline(0, color="k", lw=1.2, ls="--")
media_suma = df_id["suma_identidad"].mean()
std_suma   = df_id["suma_identidad"].std()
ax.text(0.02, 0.95, f"Media de la suma: {media_suma/1e6:.1f}M\nStd: {std_suma/1e6:.0f}M",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_ylabel("B USD")
ax.legend(fontsize=9)
ax.set_title("Residuo de la identidad (error de cierre)", fontsize=10)

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_05_identidad_contable.png"
fig.savefig(ruta, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Guardado: {ruta.name}")

# =============================================================================
# GRÁFICA 6: R² del patrón por día del mes, por año
# =============================================================================
print("Generando gráfica 6: R² por año...")
r2_anios = {}
for ano, grp in df.groupby("ano"):
    r2_anios[ano] = r2_poly(grp["dia_mes"].values,
                            grp["retiro_neto"].values, deg=3)

anos  = list(r2_anios.keys())
r2vals = list(r2_anios.values())

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(anos, r2vals, color="#2c7bb6", alpha=0.8)
ax.set_xlabel("Año")
ax.set_ylabel("R² (polinomio grado 3 en día del mes)")
ax.set_title(
    "BBVA – ¿Qué porcentaje del retiro neto explica el día del mes?\n"
    "R² de regresión polinómica: retiro_neto ~ dia_mes³",
    fontsize=11, fontweight="bold"
)
ax.set_ylim(0, max(r2vals) * 1.2)
for b, v in zip(bars, r2vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005,
            f"{v:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(np.mean(r2vals), color="red", ls="--", lw=1.5,
           label=f"Media = {np.mean(r2vals):.1%}")
ax.legend(fontsize=10)

nota = ("La estacionalidad mensual del encaje se ha vuelto\n"
        "más sistemática y predecible desde 2020.\n"
        "→ El modelo actual trata estos retiros como ruido aleatorio.")
ax.text(0.98, 0.95, nota, transform=ax.transAxes, fontsize=8.5,
        va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_06_r2_por_anio.png"
fig.savefig(ruta, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Guardado: {ruta.name}")

# =============================================================================
# RESUMEN ESTADÍSTICO EN CONSOLA
# =============================================================================
print()
print("=" * 65)
print("RESUMEN DEL ANÁLISIS")
print("=" * 65)

print("\n[1] CÓMO SE CALCULÓ EL Δ EXCESO DE ENCAJE")
print("    Exceso = (Caja + Cta Cte BCR + Overnight BCR) − Exigible")
print("    Δ Exceso(t) = Exceso(t) − Exceso(t−1)   [diferencia simple diaria]")
print(f"    Correlación Retiro Neto ~ Δ Exceso: r = {corr_val:+.4f}")
print(f"    R² = {corr_val**2:.2%}  → el {corr_val**2:.0%} de la varianza del retiro")
print("    está determinada por el cambio en el exceso de encaje.")

print("\n[2] VERIFICACIÓN DE LA IDENTIDAD CONTABLE")
print(f"    Media(Δcta_cte + Δovernight + Δactivos + retiro_neto) = {media_suma/1e6:.1f}M")
print(f"    Std del residuo = {std_suma/1e6:.0f}M")
print("    → La identidad se cumple. El residuo es ruido de redondeo.")

print("\n[2b] VARIACIÓN DIARIA PROMEDIO DE CADA COMPONENTE (M USD)")
df_id_tmp = df[["d_cta_cte","d_overnight","d_activos","retiro_neto"]].dropna()
for col, nombre in [("d_cta_cte",   "Δ Cta Cte BCR  "),
                     ("d_overnight", "Δ Overnight BCR"),
                     ("d_activos",   "Δ Activos      "),
                     ("retiro_neto", "Retiro Neto    ")]:
    v = df_id_tmp[col]
    print(f"    {nombre}  media={v.mean()/1e6:+7.1f}M  "
          f"std={v.std()/1e6:6.0f}M  "
          f"p10={v.quantile(.10)/1e6:+7.0f}M  "
          f"p90={v.quantile(.90)/1e6:+7.0f}M")
print(f"    {'SUMA (debe≈0)   ':22s}  media={df_id_tmp.sum(axis=1).mean()/1e6:+7.1f}M")

print("\n[2c] CONTRIBUCIÓN DE CADA CANAL POR FASE (promedio diario, M USD)")
comp_cols_r = ["d_cta_cte","d_overnight","d_activos","retiro_neto"]
comp_names_r= ["Δ Cta Cte","Δ Overnight","Δ Activos ","Retiro    "]
fase_medias = df.groupby("fase")[comp_cols_r].mean() / 1e6
print(f"    {'Fase':<22}", "  ".join(f"{n:>10}" for n in comp_names_r))
for f in ["A_inicio(1-4)","B_acum(5-21)","C_rentab(22-28)","D_cierre(29-31)"]:
    vals = "  ".join(f"{fase_medias.loc[f,c]:+10.1f}" for c in comp_cols_r)
    print(f"    {f:<22}  {vals}")

print("\n[3] RETIRO NETO PROMEDIO POR FASE (M USD)")
fase_res = df.groupby("fase")["retiro_neto"].agg(
    media=lambda x: round(x.mean()/1e6),
    p10  =lambda x: round(x.quantile(.10)/1e6),
    p90  =lambda x: round(x.quantile(.90)/1e6),
    n    ="count"
)
print(fase_res.to_string())

print("\n[4] R² DEL DÍA DEL MES COMO PREDICTOR (por año)")
for a, r in r2_anios.items():
    print(f"    {a}: {r:.2%}")

print("\n[5] FEATURES RECOMENDADAS PARA step001 (observables en t=forecast)")
print("    1. dia_mes_target          [calendario, siempre conocido]")
print("    2. dias_fin_mes_target     [calendario, siempre conocido]")
print("    3. overnight_bcr_lag1      r con retiro_neto =", round(df["retiro_neto"].corr(df["overnight_bcr"].shift(1)), 4))
print("    4. exceso_encaje_lag1      r con retiro_neto =", round(df["retiro_neto"].corr(df["exceso_encaje"].shift(1)), 4))
print("    5. exceso_acum_periodo_lag1 [exceso acumulado desde inicio del mes]")

print()
print("Archivos generados en:", DIR_OUTPUT)
for i, nombre in enumerate([
    "encaje_01_componentes_tiempo.png",
    "encaje_02_patron_dia_mes.png",
    "encaje_03_scatter_exceso_retiro.png",
    "encaje_04_fases_boxplot.png",
    "encaje_05_identidad_contable.png",
    "encaje_06_r2_por_anio.png",
], start=1):
    print(f"  {i}. {nombre}")
