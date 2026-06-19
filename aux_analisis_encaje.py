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
DIR_OUTPUT = RUTA_DATOS.parent.parent.parent / "2. Output" / "encaje"
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
# Encaje = Caja + Cta Cte BCR  (Overnight NO cuenta para el encaje)
# Cuando Cta Cte cae → se compensa con: Δovernight + Δactivos + retiro_neto
df["encaje"]         = df["caja"] + df["cta_cte_bcr"]
df["exceso_encaje"]  = df["encaje"] - df["encaje_exigible"]

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
# FEATURES DERIVADAS: faltante, techo, retiro acumulado
# =============================================================================
# Acumulado de encaje (Caja + Cta Cte) dentro del mes — días calendario
df["encaje_acum_mes"] = df.groupby(["ano","mes"])["encaje"].cumsum()

# Exigible total del mes = exigible diario × días calendario del mes
df["dias_en_mes"]        = df["fecha"].dt.daysinmonth
df["exigible_total_mes"] = df["encaje_exigible"] * df["dias_en_mes"]

# Faltante = lo que aún necesita acumular para cerrar el mes en compliance
# positivo → falta acumular; negativo → ya cumplió y excedió
df["faltante"]      = df["exigible_total_mes"] - df["encaje_acum_mes"]
df["faltante_lag1"] = df["faltante"].shift(1)   # observable en t=día de pronóstico

# Retiro acumulado dentro del mes (negativo = salida neta acumulada)
df["retiro_acum_mes"]      = df.groupby(["ano","mes"])["retiro_neto"].cumsum()
df["retiro_acum_mes_lag1"] = df["retiro_acum_mes"].shift(1)

# CC + ON: instrumento combinado observable antes del cierre del mes
df["cc_on"] = df["cta_cte_bcr"] + df["overnight_bcr"]

# Techo operativo: CC + ON al día que queda a 10 días hábiles del cierre del mes.
# Se calcula una vez por mes y se propaga hacia adelante desde esa fecha.
df["techo_10h"] = np.nan

for (ano_i, mes_i), _ in df.groupby(["ano","mes"]):
    inicio_mes = pd.Timestamp(ano_i, mes_i, 1)
    fin_mes    = inicio_mes + pd.offsets.MonthEnd(1)
    bdays      = pd.bdate_range(start=inicio_mes, end=fin_mes)
    n_bd       = len(bdays)
    if n_bd < 10:
        continue
    fecha_techo_mes = bdays[n_bd - 10]   # 10° día hábil antes del último día hábil

    # CC+ON en esa fecha (o el día calendario más reciente anterior si no hay dato)
    mask_mes   = (df["ano"] == ano_i) & (df["mes"] == mes_i)
    df_mes_ord = df.loc[mask_mes].sort_values("fecha")
    cands      = df_mes_ord[df_mes_ord["fecha"] <= fecha_techo_mes]
    if cands.empty:
        continue
    val_techo = cands.iloc[-1]["cc_on"]

    mask_post = mask_mes & (df["fecha"] >= fecha_techo_mes)
    df.loc[mask_post, "techo_10h"] = val_techo

# Presupuesto restante de retiro y proporción del techo ya usada
df["techo_restante"]   = df["techo_10h"] - df["retiro_acum_mes_lag1"].abs()
df["proporcion_usada"] = (
    df["retiro_acum_mes_lag1"].abs() / df["techo_10h"].replace(0, np.nan)
).clip(lower=0)

# =============================================================================
# ESTADO LIBRE: banco puede retirar (transición irreversible dentro del mes)
# Condición de entrada: faltante_lag1 ≤ encaje_lag1
#   → con un día más al nivel actual, ya cumpliría el exigible del mes
# Una vez activo, permanece el resto del mes (encaje_acum solo crece)
# =============================================================================
df["encaje_lag1"] = df["encaje"].shift(1)

# condicion_libre = False en el primer día del mes (faltante_lag1 viene del mes anterior)
_primer_dia_mes = df.groupby(["ano","mes"]).cumcount() == 0
df["condicion_libre"] = (
    (~_primer_dia_mes) &
    df["faltante_lag1"].notna() &
    df["encaje_lag1"].notna() &
    (df["faltante_lag1"] <= df["encaje_lag1"])
).astype(int)

df["libre"] = (
    df.groupby(["ano","mes"])["condicion_libre"]
    .cummax()
    .astype(bool)
)

# Días desde inicio del estado libre en el mes (0 = primer día libre)
df["libre_prev"]        = df.groupby(["ano","mes"])["libre"].shift(1).fillna(False)
df["es_inicio_libre"]   = df["libre"] & ~df["libre_prev"]
df["dias_desde_libre"]  = df.groupby(["ano","mes"])["libre"].cumsum().where(df["libre"]) - 1

# Resumen mensual con día de inicio libre y métricas de cumplimiento
def _resumen_mes(g):
    dia_libre = g.loc[g["es_inicio_libre"], "dia_mes"].values
    return pd.Series({
        "dia_inicio_libre":  dia_libre[0] if len(dia_libre) > 0 else np.nan,
        "retiro_acum_fin":   g["retiro_acum_mes"].iloc[-1],
        "techo_mes":         g["techo_10h"].dropna().iloc[0] if g["techo_10h"].notna().any() else np.nan,
        "n_dias_libre":      int(g["libre"].sum()),
        "n_dias_frenado":    int((~g["libre"]).sum()),
        "retiro_en_libre":   g.loc[g["libre"],   "retiro_neto"].sum(),
        "retiro_en_frenado": g.loc[~g["libre"],  "retiro_neto"].sum(),
    })

df_mensual = df.groupby(["ano","mes"]).apply(_resumen_mes).reset_index()

n_techo = df["techo_10h"].notna().sum()
n_libre = df["libre"].sum()
print(f"  Faltante calculado:    {df['faltante_lag1'].notna().sum():,} días")
print(f"  Techo 10h disponible:  {n_techo:,} días ({n_techo/len(df)*100:.1f}% del total)")
print(f"  Días en estado libre:  {n_libre:,} ({n_libre/len(df)*100:.1f}%)")
print(f"  Día inicio libre (mediana del mes): {df_mensual['dia_inicio_libre'].median():.0f}")
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


def r2_ols(X_mat, y_vec):
    """OLS R² con numpy lstsq; filtra NaN automáticamente."""
    mask = np.isfinite(X_mat).all(axis=1) & np.isfinite(y_vec)
    if mask.sum() < 10:
        return np.nan
    X_m, y_m = X_mat[mask], y_vec[mask]
    coef, *_ = np.linalg.lstsq(X_m, y_m, rcond=None)
    yhat   = X_m @ coef
    ss_res = ((y_m - yhat)**2).sum()
    ss_tot = ((y_m - y_m.mean())**2).sum()
    return float(max(0.0, 1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0


# Comparación de R² para retiro_neto diario — se usa en gráfica 8 y consola [7]
_y    = df["retiro_neto"].values
_ones = np.ones(len(df))
_dia  = df["dia_mes"].values.astype(float)
_falt = df["faltante_lag1"].values
_exc  = df["exceso_encaje"].shift(1).values
_on1  = df["overnight_bcr"].shift(1).values
_rest = df["techo_restante"].values
_prop = df["proporcion_usada"].values

R2 = {
    "dia_mes³":              r2_ols(np.column_stack([_ones, _dia, _dia**2, _dia**3]), _y),
    "faltante_lag1":         r2_ols(np.column_stack([_ones, _falt]), _y),
    "exceso_lag1":           r2_ols(np.column_stack([_ones, _exc]), _y),
    "overnight_lag1":        r2_ols(np.column_stack([_ones, _on1]), _y),
    "dia_mes³ + faltante":   r2_ols(np.column_stack([_ones, _dia, _dia**2, _dia**3, _falt]), _y),
    "dia_mes³ + exceso":     r2_ols(np.column_stack([_ones, _dia, _dia**2, _dia**3, _exc]), _y),
    "techo_restante_lag1":   r2_ols(np.column_stack([_ones, _rest]), _y),
    "proporcion_usada_lag1": r2_ols(np.column_stack([_ones, _prop]), _y),
}

# =============================================================================
# GRÁFICA 1: Componentes de encaje en el tiempo
# =============================================================================
print("Generando gráfica 1: componentes en el tiempo...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("BBVA – Componentes de Encaje en el Tiempo (USD)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.fill_between(df["fecha"], df["cta_cte_bcr"]/1e9, alpha=0.6,
                color=COLORES["cta_cte"], label="Cta Cte BCR (encaje)")
ax.plot(df["fecha"], df["overnight_bcr"]/1e9, color=COLORES["overnight"],
        lw=0.8, alpha=0.8, label="Overnight BCR (no es encaje)")
ax.plot(df["fecha"], df["encaje_exigible"]/1e9, color=COLORES["exigible"],
        lw=1.5, ls="--", label="Encaje Exigible")
ax.set_ylabel("B USD")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Encaje = Caja + Cta Cte BCR  |  Overnight: instrumento separado (no encaje)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fB"))

ax = axes[1]
ax.fill_between(df["fecha"], df["exceso_encaje"]/1e9,
                where=df["exceso_encaje"] >= 0, alpha=0.6,
                color=COLORES["exceso"], label="Exceso (positivo)")
ax.fill_between(df["fecha"], df["exceso_encaje"]/1e9,
                where=df["exceso_encaje"] < 0, alpha=0.6,
                color=COLORES["retiro"], label="Déficit (negativo)")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("B USD")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Exceso de Encaje = (Caja + Cta Cte BCR) − Exigible  [sin overnight]", fontsize=10)
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

# Panel B: Cta Cte vs Overnight (movimiento espejo)
ax = axes[1]
ax.plot(by_day["dia_mes"], by_day["cta_mean"]/1e9, color=COLORES["cta_cte"],
        marker="o", ms=4, lw=1.5, label="Cta Cte BCR — encaje (prom)")
ax.plot(by_day["dia_mes"], by_day["on_mean"]/1e9, color=COLORES["overnight"],
        marker="s", ms=4, lw=1.5, label="Overnight BCR — no encaje (prom)")
ax.plot(by_day["dia_mes"], by_day["ex_mean"]/1e9, color=COLORES["exceso"],
        marker="^", ms=4, lw=1.5, ls="--", label="Exceso de Encaje = (Caja+Cta Cte)−Exigible")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Día del mes")
ax.set_ylabel("B USD")
ax.set_title("Cta Cte baja en días 22-28 → Overnight sube (sustitución)  |  días 29-31: Overnight baja → retiro", fontsize=10)
ax.legend(loc="upper right", fontsize=8)
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
# GRÁFICA 3: Identidad contable — distribución de la caída de Cta Cte
# Cuando Δcta_cte < 0 → se distribuye en Δovernight, Δactivos, retiro_neto
# =============================================================================
print("Generando gráfica 3: identidad contable Δcta_cte → canales...")

df_sc = df[["d_cta_cte", "d_overnight", "d_activos", "retiro_neto", "fase"]].dropna()
corr_val = df_sc["d_cta_cte"].corr(df_sc["retiro_neto"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Identidad Contable: Δ Cta Cte BCR → Overnight + Activos + Retiro Neto\n"
    "Cuando Cta Cte cae, el dinero va al exterior (retiro), overnight y activos",
    fontsize=11, fontweight="bold"
)

# Panel A: scatter Δcta_cte vs retiro_neto por fase
ax = axes[0]
for fase_nom, grp in df_sc.groupby("fase", sort=True):
    ax.scatter(grp["d_cta_cte"]/1e9, grp["retiro_neto"]/1e6,
               alpha=0.25, s=10, color=FASES_COLORES[fase_nom], label=fase_nom)
x_lin = df_sc["d_cta_cte"].values
y_lin = df_sc["retiro_neto"].values
mask  = np.isfinite(x_lin) & np.isfinite(y_lin)
coef  = np.polyfit(x_lin[mask], y_lin[mask], 1)
xr    = np.linspace(x_lin[mask].min(), x_lin[mask].max(), 200)
ax.plot(xr/1e9, np.polyval(coef, xr)/1e6, "k-", lw=1.5,
        label=f"Regresión (r={corr_val:+.3f})")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("Δ Cta Cte BCR (B USD)")
ax.set_ylabel("Retiro Neto (M USD)")
ax.set_title("Δ Cta Cte vs Retiro Neto por fase", fontsize=10)
ax.legend(fontsize=7, markerscale=2)

# Panel B: barras de destino de la caída de Cta Cte por fase
ax = axes[1]
fase_order_g = ["A_inicio(1-4)", "B_acum(5-21)", "C_rentab(22-28)", "D_cierre(29-31)"]
labels_g     = ["Inicio\n(1-4)", "Acum.\n(5-21)", "Rentab.\n(22-28)", "Cierre\n(29-31)"]
canales      = ["d_overnight", "d_activos", "retiro_neto"]
canales_nom  = ["→ Overnight BCR", "→ Activos", "→ Retiro Neto"]
canales_col  = [COLORES["overnight"], COLORES["activos"], COLORES["retiro"]]
medias_f     = df.groupby("fase")[canales].mean() / 1e6
x = np.arange(len(fase_order_g)); width = 0.22
for j, (n, c, col) in enumerate(zip(canales_nom, canales, canales_col)):
    vals = [medias_f.loc[f, c] for f in fase_order_g]
    ax.bar(x + j*width - width, vals, width, label=n, color=col, alpha=0.85)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels_g, fontsize=9)
ax.set_ylabel("M USD (promedio diario)")
ax.set_title("Destino de la caída de Cta Cte por fase\n(promedio diario por canal)", fontsize=10)
ax.legend(fontsize=9)
ax.text(0.98, 0.97,
        "Días 22-28: Cta Cte → Overnight (82%)\n"
        "Días 29-31: Overnight → Retiro Neto\n"
        "El retiro grande ocurre cuando el banco\n"
        "convierte overnight en divisas al exterior.",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_03_canales_cta_cte.png"
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
# GRÁFICA 7: Faltante de encaje como predictor del retiro neto
# =============================================================================
print("Generando gráfica 7: faltante vs retiro neto...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "BBVA – Faltante de Encaje como Predictor del Retiro Neto\n"
    "faltante(t−1) = exigible_total_mes − encaje_acumulado(t−1)",
    fontsize=12, fontweight="bold"
)

# Panel A: Faltante promedio ± 1σ por día del mes
ax = axes[0]
by_day_fal = df.groupby("dia_mes")["faltante"].agg(["mean","std","median"]) / 1e9
ax.fill_between(by_day_fal.index,
                by_day_fal["mean"] - by_day_fal["std"],
                by_day_fal["mean"] + by_day_fal["std"],
                alpha=0.2, color="#1f77b4", label="±1σ")
ax.plot(by_day_fal.index, by_day_fal["mean"],   color="#1f77b4", lw=2,    label="Promedio")
ax.plot(by_day_fal.index, by_day_fal["median"], color="#ff7f0e", lw=1.5,
        ls="--", label="Mediana")
ax.axhline(0, color="k", lw=0.8, ls="--", label="Faltante = 0 (compliance exacto)")
for fase, col in FASES_COLORES.items():
    d0, d1 = {"A_inicio(1-4)":(0.5,4.5), "B_acum(5-21)":(4.5,21.5),
               "C_rentab(22-28)":(21.5,28.5), "D_cierre(29-31)":(28.5,31.5)}[fase]
    ax.axvspan(d0, d1, alpha=0.08, color=col)
ax.set_xlabel("Día del mes")
ax.set_ylabel("B USD")
ax.set_title("Faltante promedio por día del mes\n"
             "(positivo = aún necesita acumular; negativo = ya cumplió)", fontsize=10)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fB"))

# Panel B: Retiro neto mediano y P10-P90 por quintil de faltante_lag1
ax = axes[1]
df_q = df[["retiro_neto","faltante_lag1"]].dropna().copy()
try:
    df_q["quintil_idx"] = pd.qcut(df_q["faltante_lag1"], 5, labels=False,
                                   duplicates="drop")
except Exception:
    df_q["quintil_idx"] = pd.cut(df_q["faltante_lag1"], 5, labels=False)

quintil_stats = df_q.groupby("quintil_idx")["retiro_neto"].agg(
    mediana  = "median",
    p10      = lambda x: x.quantile(0.10),
    p90      = lambda x: x.quantile(0.90),
    n        = "count",
)
falt_limits = df_q.groupby("quintil_idx")["faltante_lag1"].agg(["min","max"]) / 1e9

x_pos = np.arange(len(quintil_stats))
ax.bar(x_pos, quintil_stats["mediana"]/1e6, color="#2c7bb6", alpha=0.8, label="Mediana")
ax.vlines(x_pos, quintil_stats["p10"]/1e6, quintil_stats["p90"]/1e6,
          color="black", lw=2.5, label="P10–P90")
ax.axhline(0, color="k", lw=0.8, ls="--")
xlabels = []
for i, (idx, row) in enumerate(falt_limits.iterrows()):
    xlabels.append(f"Q{i+1}\n[{row['min']:.1f}B, {row['max']:.1f}B]")
ax.set_xticks(x_pos)
ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("Retiro Neto (M USD)")
ax.set_title("Retiro Neto por quintil de faltante_lag1\n"
             "Q1=más faltante (necesita acumular) → Q5=ya cumplido/excedido", fontsize=10)
ax.legend(fontsize=9)
ax.text(0.98, 0.97,
        "Patrón esperado:\nQ1 (falta mucho): retiros pequeños\n"
        "Q5 (ya cumplió): retiros grandes\n"
        "→ banco libre de retirar cuando\n  tiene exceso acumulado",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_07_faltante_predictor.png"
fig.savefig(ruta, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Guardado: {ruta.name}")

# =============================================================================
# GRÁFICA 8: Techo operativo (compliance) y comparación de R²
# =============================================================================
print("Generando gráfica 8: techo operativo y R² comparativo...")

# Retiro acumulado y techo al cierre de cada mes
fin_mes_rows = []
for (ano_i, mes_i), grp in df.groupby(["ano","mes"]):
    grp_s = grp.sort_values("fecha")
    techo_val = grp_s["techo_10h"].dropna()
    if techo_val.empty:
        continue
    fin_mes_rows.append({
        "ano":             ano_i,
        "retiro_acum_fin": grp_s["retiro_acum_mes"].iloc[-1],
        "techo_mes":       techo_val.iloc[0],
    })
fin_mes_df = pd.DataFrame(fin_mes_rows).dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("BBVA – Techo Operativo de Encaje y Comparación de Predictores",
             fontsize=12, fontweight="bold")

# Panel A: Scatter techo vs retiro acumulado al cierre del mes (por año)
ax = axes[0]
anios_u = sorted(fin_mes_df["ano"].unique())
cmap_c  = plt.cm.plasma
for a in anios_u:
    sub = fin_mes_df[fin_mes_df["ano"] == a]
    col = cmap_c((a - min(anios_u)) / max(1, max(anios_u) - min(anios_u)))
    ax.scatter(sub["techo_mes"]/1e9, sub["retiro_acum_fin"]/1e9,
               color=col, alpha=0.75, s=50, label=str(a))

# Línea de compliance total: retiro_acum = −techo (límite teórico)
max_t = fin_mes_df["techo_mes"].max() / 1e9
ax.plot([0, max_t], [0, -max_t], "k--", lw=1.2,
        label="Límite: retiro = techo", alpha=0.6)
ax.axhline(0, color="gray", lw=0.5)

n_comply   = (fin_mes_df["retiro_acum_fin"].abs() <= fin_mes_df["techo_mes"]).sum()
n_total_m  = len(fin_mes_df)
pct_comply = n_comply / n_total_m * 100

ax.set_xlabel("Techo 10h (CC+ON, B USD)")
ax.set_ylabel("Retiro Acumulado en el Mes (B USD)")
ax.set_title(f"Techo vs Retiro Acumulado al Cierre del Mes\n"
             f"Compliance: {n_comply}/{n_total_m} meses ({pct_comply:.0f}%)", fontsize=10)
ax.legend(fontsize=7, ncol=2, loc="lower right")

# Panel B: R² comparativo — barras horizontales
ax = axes[1]
nombres_r2 = list(R2.keys())
valores_r2 = [v * 100 if v is not None and not np.isnan(v) else 0.0 for v in R2.values()]

colores_r2 = []
for n in nombres_r2:
    if "+" in n:          colores_r2.append("#d62728")   # combinación
    elif "dia_mes" in n:  colores_r2.append("#1f77b4")   # solo calendario
    elif "techo" in n or "proporcion" in n: colores_r2.append("#ff7f0e")   # features techo
    else:                 colores_r2.append("#2ca02c")   # features encaje individuales

y_pos = np.arange(len(nombres_r2))
bars_r2 = ax.barh(y_pos, valores_r2, color=colores_r2, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(nombres_r2, fontsize=9)
ax.set_xlabel("R² (%)")
ax.set_title("R² de distintos predictores\nvs Retiro Neto diario (regresión OLS)", fontsize=10)
for i, v in enumerate(valores_r2):
    ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlim(0, max(valores_r2) * 1.35)

from matplotlib.patches import Patch as _Patch
ax.legend(handles=[
    _Patch(color="#1f77b4", label="Solo día del mes"),
    _Patch(color="#2ca02c", label="Features de encaje (solo)"),
    _Patch(color="#ff7f0e", label="Techo / Proporción"),
    _Patch(color="#d62728", label="Combinación"),
], fontsize=8, loc="lower right")

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_08_techo_r2_comparativo.png"
fig.savefig(ruta, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Guardado: {ruta.name}")

# =============================================================================
# GRÁFICA 9: Estado libre — día de inicio y comportamiento del retiro
# =============================================================================
print("Generando gráfica 9: estado libre (día de inicio y retiro)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "BBVA – Estado Libre de Encaje\n"
    "Libre = 'con un día más al nivel actual ya cumplo el exigible del mes'",
    fontsize=12, fontweight="bold"
)

# Panel A: Boxplot del día de inicio libre por año
ax = axes[0]
anios_u = sorted(df_mensual["dia_inicio_libre"].dropna().pipe(
    lambda s: df_mensual.loc[s.index, "ano"]
).unique())
anios_u = sorted(df_mensual["ano"].unique())
data_box  = [df_mensual.loc[df_mensual["ano"] == a, "dia_inicio_libre"].dropna().values
             for a in anios_u]
data_box  = [d for d in data_box if len(d) > 0]
anios_box = [a for a, d in zip(anios_u,
             [df_mensual.loc[df_mensual["ano"] == a, "dia_inicio_libre"].dropna().values
              for a in anios_u]) if len(d) > 0]

bp = ax.boxplot(data_box, labels=anios_box, patch_artist=True,
                medianprops=dict(color="black", lw=2),
                flierprops=dict(marker=".", markersize=4, alpha=0.5))
for patch in bp["boxes"]:
    patch.set_facecolor("#2c7bb6")
    patch.set_alpha(0.7)

med_global = df_mensual["dia_inicio_libre"].median()
ax.axhline(med_global, color="red", ls="--", lw=1.5,
           label=f"Mediana global = día {med_global:.0f}")
ax.set_xlabel("Año")
ax.set_ylabel("Día del mes")
ax.set_title("¿Qué día del mes entra en estado libre?\n"
             "(por año — mediana y dispersión)", fontsize=10)
ax.legend(fontsize=9)
# Sombrear fases de referencia
for fase, col in FASES_COLORES.items():
    d0, d1 = {"A_inicio(1-4)":(0.5,4.5), "B_acum(5-21)":(4.5,21.5),
               "C_rentab(22-28)":(21.5,28.5), "D_cierre(29-31)":(28.5,31.5)}[fase]
    ax.axhspan(d0, d1, alpha=0.07, color=col)

# Panel B: Retiro neto por estado (frenado / recién libre / libre establecido)
ax = axes[1]
df_reg = df.copy()
df_reg["estado"] = "Frenado"
df_reg.loc[df_reg["libre"] & (df_reg["dias_desde_libre"] <= 2),  "estado"] = "Libre (días 1-3)"
df_reg.loc[df_reg["libre"] & (df_reg["dias_desde_libre"] > 2),   "estado"] = "Libre (día 4+)"

orden_est = ["Frenado", "Libre (días 1-3)", "Libre (día 4+)"]
colores_est = {"Frenado": "#d62728", "Libre (días 1-3)": "#ff7f0e", "Libre (día 4+)": "#2ca02c"}
data_est = [df_reg.loc[df_reg["estado"] == e, "retiro_neto"].values / 1e6 for e in orden_est]

bp2 = ax.boxplot(data_est, labels=orden_est, patch_artist=True,
                 medianprops=dict(color="black", lw=2),
                 flierprops=dict(marker=".", markersize=3, alpha=0.3),
                 showfliers=True)
for patch, est in zip(bp2["boxes"], orden_est):
    patch.set_facecolor(colores_est[est])
    patch.set_alpha(0.75)

ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_ylabel("Retiro Neto (M USD)")
ax.set_title("Retiro Neto por estado de encaje\n"
             "(¿el retiro grande ocurre después de entrar en estado libre?)", fontsize=10)

for i, (est, data) in enumerate(zip(orden_est, data_est), start=1):
    med = float(np.median(data))
    n   = len(data)
    ax.text(i, med + 5, f"{med:.0f}M\nn={n:,}", ha="center", fontsize=8,
            fontweight="bold", color="black")

plt.tight_layout()
ruta = DIR_OUTPUT / "encaje_09_estado_libre.png"
fig.savefig(ruta, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Guardado: {ruta.name}")

# =============================================================================
# EXPORTAR EXCEL CON DATOS DE ANÁLISIS
# =============================================================================
print("Exportando Excel con datos del análisis...")

ruta_excel = DIR_OUTPUT / "encaje_analisis_datos.xlsx"

# Hoja 1: datos diarios con todas las features computadas
cols_diario = [
    "fecha", "dia_mes", "ano", "mes", "fase",
    "caja", "cta_cte_bcr", "overnight_bcr", "activos", "encaje_exigible",
    "encaje", "exceso_encaje",
    "encaje_acum_mes", "exigible_total_mes", "faltante", "faltante_lag1",
    "encaje_lag1", "condicion_libre", "libre", "dias_desde_libre",
    "retiro_neto", "retiro_acum_mes",
    "cc_on", "techo_10h", "techo_restante", "proporcion_usada",
    "d_cta_cte", "d_overnight", "d_activos",
]
cols_diario_ok = [c for c in cols_diario if c in df.columns]
df_export = df[cols_diario_ok].copy()
# Escalar a millones para legibilidad
cols_m = ["caja","cta_cte_bcr","overnight_bcr","activos","encaje_exigible",
          "encaje","exceso_encaje","encaje_acum_mes","exigible_total_mes",
          "faltante","faltante_lag1","encaje_lag1",
          "retiro_neto","retiro_acum_mes","cc_on","techo_10h",
          "techo_restante","d_cta_cte","d_overnight","d_activos"]
for c in cols_m:
    if c in df_export.columns:
        df_export[c] = df_export[c] / 1e6

# Hoja 2: resumen mensual
df_mensual_exp = df_mensual.copy()
for c in ["retiro_acum_fin","techo_mes","retiro_en_libre","retiro_en_frenado"]:
    if c in df_mensual_exp.columns:
        df_mensual_exp[c] = df_mensual_exp[c] / 1e6

# Hoja 3: R² comparativo
df_r2 = pd.DataFrame([
    {"predictor": k, "R2_pct": round(v * 100, 2) if v is not None and not np.isnan(v) else np.nan}
    for k, v in R2.items()
])

# Hoja 4: estadísticas por estado (frenado / libre)
df_estado_stats = (
    df_reg.groupby("estado")["retiro_neto"]
    .agg(media="mean", mediana="median",
         p10=lambda x: x.quantile(0.10),
         p90=lambda x: x.quantile(0.90),
         n="count")
    .reset_index()
)
df_estado_stats[["media","mediana","p10","p90"]] /= 1e6

with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
    df_export.to_excel(writer, sheet_name="Diario_M_USD", index=False)
    df_mensual_exp.to_excel(writer, sheet_name="Mensual", index=False)
    df_r2.to_excel(writer, sheet_name="R2_comparativo", index=False)
    df_estado_stats.to_excel(writer, sheet_name="Estado_libre_frenado", index=False)

print(f"  Guardado: {ruta_excel.name}")
print(f"    → Hoja 'Diario_M_USD':        {len(df_export):,} filas — features diarias (M USD)")
print(f"    → Hoja 'Mensual':              {len(df_mensual_exp):,} filas — resumen por mes")
print(f"    → Hoja 'R2_comparativo':       {len(df_r2)} predictores")
print(f"    → Hoja 'Estado_libre_frenado': estadísticas por régimen")

# =============================================================================
# RESUMEN ESTADÍSTICO EN CONSOLA
# =============================================================================
print()
print("=" * 65)
print("RESUMEN DEL ANÁLISIS")
print("=" * 65)

print("\n[1] DEFINICIÓN CORRECTA DE ENCAJE Y EXCESO")
print("    Encaje     = Caja + Cta Cte BCR  (Overnight NO es encaje)")
print("    Exceso     = Encaje − Exigible")
print("    Δ Exceso(t)= Exceso(t) − Exceso(t−1)  [diferencia simple diaria]")
print()
print("    NOTA: Una versión anterior incluía Overnight en el exceso,")
print("    lo que generaba una correlación espuria r=+0.90 con el retiro.")
print("    Δexceso_malo = Δcta_cte + Δovernight + Δcaja  (overnight en ambos lados)")
print("    Identidad:    Δcta_cte + Δovernight + Δactivos + retiro_neto = 0")
print("    → El overnight se cancelaba consigo mismo, inflando el r artificialmente.")

print("\n[2] CORRELACIONES CORRECTAS CON RETIRO NETO")
corr_data = {
    "Δ Cta Cte BCR (driver principal)": df["retiro_neto"].corr(df["d_cta_cte"]),
    "Δ Exceso encaje (correcto, sin ON)": df["retiro_neto"].corr(df["d_exceso"]),
    "Δ Activos                        ": df["retiro_neto"].corr(df["d_activos"]),
    "Δ Overnight BCR                  ": df["retiro_neto"].corr(df["d_overnight"]),
    "Exceso encaje (nivel)            ": df["retiro_neto"].corr(df["exceso_encaje"]),
}
for nombre, r in corr_data.items():
    print(f"    {nombre}  r={r:+.4f}  R²={r**2:.2%}")
print()
print("    → El r=+0.90 reportado anteriormente era INCORRECTO (overnight incluido).")
print("    → El valor correcto de Δexceso es r=+0.38 / R²=14%.")
print("    → Lo más valioso para el pronóstico sigue siendo el patrón calendárico.")

print("\n[3] VERIFICACIÓN DE LA IDENTIDAD CONTABLE")
df_id_tmp = df[["d_cta_cte","d_overnight","d_activos","retiro_neto"]].dropna().copy()
df_id_tmp["residuo"] = df_id_tmp.sum(axis=1)
res = df_id_tmp["residuo"] / 1e6
print(f"    Media residuo:  {res.mean():+.1f}M")
print(f"    Std residuo:    {res.std():.0f}M")
print(f"    Min / Max:      {res.min():.0f}M  /  {res.max():.0f}M")
print()
print(f"    Días con |residuo| < 10M  (≈ cero):    {(res.abs()<10).sum():4d}  ({(res.abs()<10).mean()*100:.1f}%)")
print(f"    Días con |residuo| < 100M (pequeño):   {(res.abs()<100).sum():4d}  ({(res.abs()<100).mean()*100:.1f}%)")
print(f"    Días con |residuo| >= 500M (grande):   {(res.abs()>=500).sum():4d}  ({(res.abs()>=500).mean()*100:.1f}%)")
print(f"    Mediana del residuo absoluto:          {res.abs().quantile(0.50):.0f}M")
print(f"    P90 del residuo absoluto:              {res.abs().quantile(0.90):.0f}M")
print()
print("    → CONCLUSIÓN: la identidad NO se cumple perfectamente.")
print("    → La mitad de los días el residuo supera 155M — no es ruido de redondeo.")
print("    → EncajeD.xlsx captura solo 5 variables; el balance real del banco")
print("      incluye otros canales no observados (bonos, posición interbancaria,")
print("      depósitos de clientes, posición FX, etc.) que absorben la diferencia.")
print("    → No podemos afirmar que el retiro sea 'la válvula principal'")
print("      con los datos disponibles.")

print("\n[3b] VARIACIÓN DIARIA PROMEDIO DE CADA COMPONENTE (M USD)")
for col, nombre in [("d_cta_cte",   "Δ Cta Cte BCR  "),
                     ("d_overnight", "Δ Overnight BCR"),
                     ("d_activos",   "Δ Activos      "),
                     ("retiro_neto", "Retiro Neto    ")]:
    v = df_id_tmp[col]
    print(f"    {nombre}  media={v.mean()/1e6:+7.1f}M  "
          f"std={v.std()/1e6:6.0f}M  "
          f"p10={v.quantile(.10)/1e6:+7.0f}M  "
          f"p90={v.quantile(.90)/1e6:+7.0f}M")
print(f"    {'Residuo (no≈0)  ':22s}  media={res.mean():+7.1f}M  "
      f"std={res.std():6.0f}M")

print("\n[3c] DISTRIBUCIÓN DE LA CAÍDA DE CTA CTE (días con Δcta_cte < 0)")
mask_cae = df["d_cta_cte"] < 0
total = df.loc[mask_cae, "d_cta_cte"].sum()
d_on  = df.loc[mask_cae, "d_overnight"].sum()
d_act = df.loc[mask_cae, "d_activos"].sum()
d_ret = df.loc[mask_cae, "retiro_neto"].sum()
print(f"    Caída total Cta Cte: {total/1e9:+.1f}B")
print(f"      → Overnight:       {d_on/1e9:+.1f}B  ({d_on/abs(total)*100:.0f}%)")
print(f"      → Activos:         {d_act/1e9:+.1f}B  ({d_act/abs(total)*100:.0f}%)")
print(f"      → Retiro Neto:     {d_ret/1e9:+.1f}B  ({d_ret/abs(total)*100:.0f}%)")

print("\n[3d] CONTRIBUCIÓN DE CADA CANAL POR FASE (promedio diario, M USD)")
comp_cols_r = ["d_cta_cte","d_overnight","d_activos","retiro_neto"]
comp_names_r= ["Δ Cta Cte","Δ Overnight","Δ Activos ","Retiro    "]
fase_medias = df.groupby("fase")[comp_cols_r].mean() / 1e6
print(f"    {'Fase':<22}", "  ".join(f"{n:>10}" for n in comp_names_r))
for f in ["A_inicio(1-4)","B_acum(5-21)","C_rentab(22-28)","D_cierre(29-31)"]:
    vals = "  ".join(f"{fase_medias.loc[f,c]:+10.1f}" for c in comp_cols_r)
    print(f"    {f:<22}  {vals}")

print("\n[4] RETIRO NETO PROMEDIO POR FASE (M USD)")
fase_res = df.groupby("fase")["retiro_neto"].agg(
    media=lambda x: round(x.mean()/1e6),
    p10  =lambda x: round(x.quantile(.10)/1e6),
    p90  =lambda x: round(x.quantile(.90)/1e6),
    n    ="count"
)
print(fase_res.to_string())

print("\n[5] R² DEL DÍA DEL MES COMO PREDICTOR (por año)")
print("    (Este resultado NO cambia con la corrección de encaje)")
for a, r in r2_anios.items():
    print(f"    {a}: {r:.2%}")

print("\n[7] FALTANTE DE ENCAJE COMO PREDICTOR DEL RETIRO NETO")
r_fal = df["retiro_neto"].corr(df["faltante_lag1"])
print(f"    r(retiro_neto, faltante_lag1) = {r_fal:+.4f}  R²={r_fal**2:.2%}")
print()
print("    Retiro neto (M USD) por quintil de faltante_lag1:")
_df_q2 = df[["retiro_neto","faltante_lag1"]].dropna().copy()
try:
    _df_q2["qidx"] = pd.qcut(_df_q2["faltante_lag1"], 5, labels=False, duplicates="drop")
except Exception:
    _df_q2["qidx"] = pd.cut(_df_q2["faltante_lag1"], 5, labels=False)
for qi, grp in _df_q2.groupby("qidx"):
    lims = f"[{grp['faltante_lag1'].min()/1e9:.1f}B, {grp['faltante_lag1'].max()/1e9:.1f}B]"
    print(f"      Q{int(qi)+1} {lims:25s}  "
          f"media={grp['retiro_neto'].mean()/1e6:+7.0f}M  "
          f"mediana={grp['retiro_neto'].median()/1e6:+7.0f}M  "
          f"n={len(grp):4d}")
print()
print("    R² comparativo (retiro_neto diario):")
for nombre, val in R2.items():
    v = val * 100 if val is not None and not np.isnan(val) else float("nan")
    print(f"      {nombre:<30s}: {v:.2f}%")
print()
print("    → Conclusión: faltante_lag1 agrega ~1-2% de R² sobre día del mes.")
print("    → Su valor principal es el comportamiento condicional (Q1 vs Q5).")
print("    → techo_restante y proporcion_usada tienen R² similar al faltante solo.")

print("\n[8] TECHO OPERATIVO: COMPLIANCE Y LÍMITE DE RETIROS")
print(f"    Techo = CC + ON al día hábil que queda a 10 días hábiles del cierre del mes")
print(f"    Meses con techo calculado: {len(fin_mes_df)}")
print(f"    Compliance (|retiro_acum| ≤ techo): {n_comply}/{n_total_m} meses ({pct_comply:.0f}%)")
print()
print("    Distribución de retiro_acum_fin por mes:")
_raf = fin_mes_df["retiro_acum_fin"] / 1e9
_tet = fin_mes_df["techo_mes"] / 1e9
print(f"      retiro_acum_fin:  media={_raf.mean():+.2f}B  std={_raf.std():.2f}B  "
      f"min={_raf.min():.2f}B  max={_raf.max():.2f}B")
print(f"      techo_mes:        media={_tet.mean():+.2f}B  std={_tet.std():.2f}B  "
      f"min={_tet.min():.2f}B  max={_tet.max():.2f}B")
_ratio = (_raf.abs() / _tet.replace(0, np.nan))
print(f"      uso del techo:    media={_ratio.mean():.1%}  P90={_ratio.quantile(0.90):.1%}  "
      f"max={_ratio.max():.1%}")
print()
print("    → Si compliance < 100%, revisar si hay meses de cierre de encaje especiales.")
print("    → El techo es útil como restricción operativa, no como predictor diario.")
print("    → Para pronóstico: mejor usarlo en la última semana del mes (días 22-31).")

print("\n[10] ESTADO LIBRE: DÍA DE INICIO Y COMPORTAMIENTO DEL RETIRO")
print(f"    Condición: faltante_lag1 ≤ encaje_lag1")
print(f"    (con un día más al nivel actual, el banco ya cumpliría el exigible)")
print()
med_libre = df_mensual["dia_inicio_libre"].median()
p10_libre = df_mensual["dia_inicio_libre"].quantile(0.10)
p90_libre = df_mensual["dia_inicio_libre"].quantile(0.90)
print(f"    Día de inicio libre — mediana: {med_libre:.0f}  P10: {p10_libre:.0f}  P90: {p90_libre:.0f}")
print(f"    Meses sin estado libre: {df_mensual['dia_inicio_libre'].isna().sum()} "
      f"({df_mensual['dia_inicio_libre'].isna().mean()*100:.0f}%)")
print()
print("    Retiro neto (M USD) por estado:")
for est in ["Frenado", "Libre (días 1-3)", "Libre (día 4+)"]:
    sub = df_reg.loc[df_reg["estado"] == est, "retiro_neto"] / 1e6
    print(f"      {est:<22s}  media={sub.mean():+7.0f}M  "
          f"mediana={sub.median():+7.0f}M  "
          f"p10={sub.quantile(.10):+7.0f}M  "
          f"p90={sub.quantile(.90):+7.0f}M  "
          f"n={len(sub):,}")
print()
print("    Día de inicio libre por año:")
for _, row in df_mensual.groupby("ano")["dia_inicio_libre"].median().reset_index().iterrows():
    print(f"      {int(row['ano'])}: mediana día {row['dia_inicio_libre']:.0f}")
print()
print("    → Si el retiro grande ocurre principalmente en estado 'Libre (día 4+)',")
print("      el estado libre es un predictor confiable del timing del retiro.")

print("\n[9] RESUMEN: ¿VALE LA PENA INCLUIR ESTAS FEATURES EN EL MODELO?")
print("    Feature               | R² diario | Utilidad principal")
print("    ─────────────────────────────────────────────────────────────────")
r2_dia_val = R2.get("dia_mes³", 0) * 100
r2_fal_val = R2.get("faltante_lag1", 0) * 100
r2_com_val = R2.get("dia_mes³ + faltante", 0) * 100
r2_exc_val = R2.get("exceso_lag1", 0) * 100
r2_tec_val = R2.get("techo_restante_lag1", 0) * 100
print(f"    dia_mes (calendario)  | {r2_dia_val:5.1f}%    | Predictor principal — patrón estacional")
print(f"    faltante_lag1         | {r2_fal_val:5.1f}%    | Estado acumulación → quintiles condicionales")
print(f"    exceso_lag1           | {r2_exc_val:5.1f}%    | Nivel de encaje de ayer")
print(f"    dia_mes + faltante    | {r2_com_val:5.1f}%    | Ganancia marginal vs solo calendario")
print(f"    techo_restante_lag1   | {r2_tec_val:5.1f}%    | Presupuesto de retiro restante")
print("    ─────────────────────────────────────────────────────────────────")
print("    → Recomendación: incluir faltante_lag1 y exceso_lag1 (bajo costo,")
print("      información sobre estado de encaje) aunque R² incremental sea bajo.")
print("    → techo_restante es más útil para días 22-31 (última semana del mes).")

print("\n[6] FEATURES RECOMENDADAS PARA step001 (observables en t=forecast)")
r_on_lag1  = round(df["retiro_neto"].corr(df["overnight_bcr"].shift(1)), 4)
r_exc_lag1 = round(df["retiro_neto"].corr(df["exceso_encaje"].shift(1)), 4)
r_cta_lag1 = round(df["retiro_neto"].corr(df["cta_cte_bcr"].shift(1)), 4)
print(f"    1. dia_mes_target           [calendario]  — principal predictor (R²=13-35% por año)")
print(f"    2. dias_fin_mes_target      [calendario]  — complementario al día del mes")
print(f"    3. cta_cte_bcr_lag1         r={r_cta_lag1:+.4f}  [nivel de encaje de ayer]")
print(f"    4. exceso_encaje_lag1       r={r_exc_lag1:+.4f}  [exceso encaje de ayer, sin overnight]")
print(f"    5. overnight_bcr_lag1       r={r_on_lag1:+.4f}  [overnight de ayer → proxy rentabilización]")
print(f"    6. exceso_acum_periodo_lag1          [exceso acumulado desde inicio del mes]")

print()
print("Archivos generados en:", DIR_OUTPUT)
for i, nombre in enumerate([
    "encaje_01_componentes_tiempo.png",
    "encaje_02_patron_dia_mes.png",
    "encaje_03_canales_cta_cte.png",
    "encaje_04_fases_boxplot.png",
    "encaje_05_identidad_contable.png",
    "encaje_06_r2_por_anio.png",
    "encaje_07_faltante_predictor.png",
    "encaje_08_techo_r2_comparativo.png",
    "encaje_09_estado_libre.png",
    "encaje_analisis_datos.xlsx",
], start=1):
    print(f"  {i}. {nombre}")
