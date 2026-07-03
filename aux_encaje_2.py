# -*- coding: utf-8 -*-
"""
aux_encaje_2.py
===============
Análisis de encaje BBVA desde cero.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, GoodFriday,
    USFederalHolidayCalendar,
)
from pandas.tseries.offsets import Easter, Day as _Day

RUTA              = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Raw\bbva_encaje.xlsx")
RUTA_TRANSACCIONES = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Raw\Transacciones_BancaLocal.xlsx")

# ── Carga ──────────────────────────────────────────────────────────────────────
df = pd.read_excel(RUTA, header=0)
df = df.iloc[:, :9].copy()
df.columns = [
    "fecha", "entidad", "codigo",
    "overnight", "cta_cte", "caja",
    "tose", "exigible", "retiro_neto"
]
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

print(f"Período  : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"Filas    : {len(df):,}")
print()

# ── Columnas derivadas ─────────────────────────────────────────────────────────
df["encaje"]         = df["cta_cte"] + df["caja"]
df["encaje_ovn"]     = df["encaje"] + df["overnight"]
df["var_encaje_ovn"] = df["encaje_ovn"].diff()

df["anio_mes"]          = df["fecha"].dt.to_period("M")
df["NecAcumMes"]        = df.groupby("anio_mes")["exigible"].cumsum()
df["EncajeAcumMes"]     = df.groupby("anio_mes")["encaje"].cumsum()

df["dia_mes"]           = df["fecha"].dt.day
df["dias_en_mes"]       = df["fecha"].dt.days_in_month
df["dias_restantes"]    = df["dias_en_mes"] - df["dia_mes"]

# Estimaciones del exigible total del mes
df["ExigibleTotalMes_A"] = df["exigible"] * df["dias_en_mes"]
df["ExigibleTotalMes_C"] = (df["NecAcumMes"] / df["dia_mes"]) * df["dias_en_mes"]

# Feature principal (sin leakage): usar Opción C
df["ExigibleTotalMes_est"] = df["ExigibleTotalMes_C"]

# Avance actual (solo lo acumulado hasta hoy)
df["Avance"] = df["EncajeAcumMes"] / df["ExigibleTotalMes_est"]

UMBRAL_90 = 0.90

# brecha_90: distancia al umbral del 90% en puntos porcentuales del requerimiento mensual
#   < 0 → banco ya superó el umbral (libre para hacer retiros)
#   > 0 → banco aún no llega al 90% (debe seguir acumulando encaje)
df["brecha_90"]    = UMBRAL_90 - df["Avance"]

# ritmo_encaje: encaje de hoy relativo al exigible diario (sin supuesto de distribución)
#   > 1 → encaje cubre el exigible del día
#   < 1 → encaje por debajo del ritmo requerido
df["ritmo_encaje"] = df["encaje"] / df["exigible"]

# dia_liberacion_90: primer día del mes donde brecha_90 ≤ 0 (Avance ≥ 90%)
_anio_mes_tmp = df["fecha"].dt.to_period("M")
df["_ya_lib"] = df["Avance"] >= UMBRAL_90
_dias_lib_map = (
    df.assign(_am=_anio_mes_tmp)
    .groupby("_am")
    .apply(lambda g: g.loc[g["_ya_lib"], "dia_mes"].iloc[0]
                     if g["_ya_lib"].any() else np.nan)
)
df["dia_liberacion_90"] = _anio_mes_tmp.map(_dias_lib_map)
df.drop(columns=["_ya_lib"], inplace=True)

# ── Validación: A y C vs real al cierre de cada mes ──────────────────────────

# Error diario para cada observación (vs real del mes completo)
ExigibleReal = df.groupby("anio_mes")["exigible"].transform("sum")
df["error_A_pct"] = (df["ExigibleTotalMes_A"] - ExigibleReal) / ExigibleReal * 100
df["error_C_pct"] = (df["ExigibleTotalMes_C"] - ExigibleReal) / ExigibleReal * 100

# Resumen global
mape_A = df["error_A_pct"].abs().mean()
mape_C = df["error_C_pct"].abs().mean()

# Error promedio absoluto por día del mes
err_dia = df.groupby("dia_mes")[["error_A_pct","error_C_pct"]].apply(
    lambda g: pd.Series({
        "MAPE_A": g["error_A_pct"].abs().mean(),
        "MAPE_C": g["error_C_pct"].abs().mean(),
    })
).reset_index()

print(f"\n── Validación Opción A vs C (error promedio diario) ────────────────")
print(f"  MAPE global A : {mape_A:.3f}%")
print(f"  MAPE global C : {mape_C:.3f}%")
print()
print(f"  {'Día':>4}  {'MAPE A':>8}  {'MAPE C':>8}  {'Mejor':>6}")
for _, row in err_dia.iterrows():
    mejor = "C" if row["MAPE_C"] < row["MAPE_A"] else "A"
    print(f"  {int(row['dia_mes']):>4}  {row['MAPE_A']:>7.2f}%  {row['MAPE_C']:>7.2f}%  {mejor:>6}")

# Resumen al cierre
validacion = (
    df.groupby("anio_mes")
    .apply(lambda g: pd.Series({
        "fecha_cierre": g["fecha"].iloc[-1],
        "real"        : g["exigible"].sum(),
        "estimado_A"  : g["ExigibleTotalMes_A"].iloc[-1],
        "estimado_C"  : g["ExigibleTotalMes_C"].iloc[-1],
        "error_A_pct" : g["error_A_pct"].iloc[-1],
        "error_C_pct" : g["error_C_pct"].iloc[-1],
    }))
    .reset_index(drop=True)
)

df.drop(columns=["error_A_pct", "error_C_pct"], inplace=True)

df.drop(columns=["anio_mes", "ExigibleTotalMes_A", "ExigibleTotalMes_C"], inplace=True)

print(df[["fecha", "encaje", "encaje_ovn", "var_encaje_ovn"]].tail(10).to_string(index=False))

# ── Balance mensual: Σ(var_encaje_ovn) vs Σ(retiro_neto) ─────────────────────
balance = (
    df.groupby(df["fecha"].dt.to_period("M"))
    .agg(
        suma_var_ovn  = ("var_encaje_ovn", "sum"),
        suma_retiro   = ("retiro_neto",    "sum"),
    )
    .reset_index()
)
balance["residuo"]     = balance["suma_var_ovn"] - balance["suma_retiro"]
balance["cumple"]      = balance["residuo"].abs() < 1e6   # tolerancia 1M
balance["fecha"]       = balance["fecha"].dt.to_timestamp()

print(f"\n── Balance mensual (Σ var_encaje_ovn vs Σ retiro_neto) ─────────────")
print(f"  Meses analizados : {len(balance)}")
print(f"  Meses que cumplen: {balance['cumple'].sum()}  ({balance['cumple'].mean()*100:.0f}%)")
print(f"  Residuo mediano  : {balance['residuo'].median()/1e6:.0f}M")
print(f"  Residuo P90 abs  : {balance['residuo'].abs().quantile(0.90)/1e6:.0f}M")
print()
print(balance[["fecha","suma_var_ovn","suma_retiro","residuo","cumple"]]
      .rename(columns={"suma_var_ovn":"Σ VarOVN","suma_retiro":"Σ Retiro","residuo":"Residuo"})
      .to_string(index=False))

# ── Export ─────────────────────────────────────────────────────────────────────
DIR_OUT = RUTA.parent.parent.parent / "2. Output" / "encaje_bbva"
DIR_OUT.mkdir(parents=True, exist_ok=True)
ruta_out = DIR_OUT / "bbva_encaje_features.xlsx"

# ── Gráfico validación A vs C ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Validación Opción A vs C — Error promedio por día del mes\n"
             f"MAPE global  A={mape_A:.2f}%  |  C={mape_C:.2f}%",
             fontsize=12, fontweight="bold")

# Panel A — MAPE por día del mes
ax = axes[0]
ax.plot(err_dia["dia_mes"], err_dia["MAPE_A"], color="#1f77b4",
        marker="o", ms=4, lw=1.5, label=f"Opción A  (global {mape_A:.2f}%)")
ax.plot(err_dia["dia_mes"], err_dia["MAPE_C"], color="#ff7f0e",
        marker="^", ms=4, lw=1.5, label=f"Opción C  (global {mape_C:.2f}%)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Día del mes")
ax.set_ylabel("MAPE promedio (%)")
ax.set_title("Error promedio absoluto por día del mes")
ax.legend(fontsize=9)

# Panel B — scatter real vs estimado (muestra de días)
ax = axes[1]
idx = df.sample(min(2000, len(df)), random_state=42).index
ExigibleTotalMes_A_plot = df.loc[idx, "exigible"] * df.loc[idx, "dias_en_mes"]
ax.scatter(ExigibleReal.loc[idx]/1e9, ExigibleTotalMes_A_plot/1e9,
           alpha=0.2, s=8, color="#1f77b4", label="Opción A")
ax.scatter(ExigibleReal.loc[idx]/1e9, df.loc[idx, "ExigibleTotalMes_est"]/1e9,
           alpha=0.2, s=8, color="#ff7f0e", label="Opción C")
lim = [ExigibleReal.min()/1e9, ExigibleReal.max()/1e9]
ax.plot(lim, lim, "k--", lw=1, label="Perfecta")
ax.set_xlabel("Exigible real del mes (B USD)")
ax.set_ylabel("Estimado (B USD)")
ax.set_title("Real vs Estimado — todos los días")
ax.legend(fontsize=9)

plt.tight_layout()
ruta_val = DIR_OUT / "validacion_opciones.png"
fig.savefig(ruta_val, dpi=130, bbox_inches="tight")
plt.close()
print(f"\nGráfico validación guardado: {ruta_val.name}")

# ── Gráfico de dispersión ─────────────────────────────────────────────────────
x = balance["suma_var_ovn"] / 1e9
y = balance["suma_retiro"]  / 1e9

coef = np.polyfit(x, y, 1)
xr   = np.linspace(x.min(), x.max(), 200)
r    = np.corrcoef(x, y)[0, 1]

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(x, y, c=balance["fecha"].dt.year,
                cmap="plasma", alpha=0.8, s=60, edgecolors="white", lw=0.4)
ax.plot(xr, np.polyval(coef, xr), color="black", lw=1.5, ls="--",
        label=f"Tendencia  r = {r:+.3f}")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.axvline(0, color="gray", lw=0.5, ls=":")
plt.colorbar(sc, ax=ax, label="Año")
ax.set_xlabel("Σ Var EncajeOVN  (B USD)", fontsize=11)
ax.set_ylabel("Σ Retiro Neto  (B USD)", fontsize=11)
ax.set_title("Balance mensual: variación de EncajeOVN vs Retiro Neto\n"
             "Cada punto = un mes  |  si la ecuación se cumple → nube en diagonal",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
ruta_plot = DIR_OUT / "balance_scatter.png"
fig.savefig(ruta_plot, dpi=130, bbox_inches="tight")
plt.close()
print(f"\nGráfico guardado: {ruta_plot.name}")

# ── Regresión diaria: var_encaje_ovn → retiro_neto ───────────────────────────
df_reg = df[["fecha", "var_encaje_ovn", "retiro_neto"]].dropna()
x_d = df_reg["var_encaje_ovn"].values
y_d = df_reg["retiro_neto"].values

coef_d = np.polyfit(x_d, y_d, 1)
yhat   = np.polyval(coef_d, x_d)
r_d    = np.corrcoef(x_d, y_d)[0, 1]
r2_d   = r_d ** 2
ss_res = ((y_d - yhat) ** 2).sum()
ss_tot = ((y_d - y_d.mean()) ** 2).sum()

print(f"\n── Regresión diaria: var_encaje_ovn → retiro_neto ──────────────────")
print(f"  N          : {len(df_reg):,}")
print(f"  r          : {r_d:+.4f}")
print(f"  R²         : {r2_d:.2%}")
print(f"  Pendiente  : {coef_d[0]:.4f}  (β = 1 implicaría identidad perfecta)")
print(f"  Intercepto : {coef_d[1]/1e6:.1f}M")

xr_d = np.linspace(np.percentile(x_d, 1), np.percentile(x_d, 99), 300)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Regresión diaria: Var EncajeOVN → Retiro Neto",
             fontsize=13, fontweight="bold")

# Panel A — scatter con regresión
ax = axes[0]
sc = ax.scatter(x_d / 1e9, y_d / 1e9,
                c=df_reg["fecha"].dt.year, cmap="plasma",
                alpha=0.25, s=12, linewidths=0)
ax.plot(xr_d / 1e9, np.polyval(coef_d, xr_d) / 1e9,
        color="black", lw=2, label=f"r = {r_d:+.3f}  |  R² = {r2_d:.1%}")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.axvline(0, color="gray", lw=0.5, ls=":")
plt.colorbar(sc, ax=ax, label="Año")
ax.set_xlabel("Var EncajeOVN  (B USD)", fontsize=10)
ax.set_ylabel("Retiro Neto  (B USD)", fontsize=10)
ax.set_title(f"Scatter diario  (N={len(df_reg):,})", fontsize=10)
ax.legend(fontsize=9)

# Panel B — residuos en el tiempo
residuos = (y_d - yhat) / 1e9
ax = axes[1]
ax.bar(df_reg["fecha"], residuos, color=np.where(residuos >= 0, "#2196F3", "#F44336"),
       alpha=0.5, width=1.2)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Residuo  (B USD)", fontsize=10)
ax.set_title("Residuos diarios en el tiempo", fontsize=10)

plt.tight_layout()
ruta_daily = DIR_OUT / "regresion_diaria_scatter.png"
fig.savefig(ruta_daily, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Gráfico guardado: {ruta_daily.name}")

# ── Análisis día de liberación ────────────────────────────────────────────────
_am = df["fecha"].dt.to_period("M")

# Resumen mensual: día de liberación + retiro pre/post umbral
_post = df["brecha_90"] <= 0
lib_resumen = (
    df.groupby(_am)
    .agg(
        dia_liberacion_90     = ("dia_liberacion_90", "first"),
        retiro_post_umbral    = ("retiro_neto", lambda g: g[_post.loc[g.index]].mean()),
        retiro_pre_umbral     = ("retiro_neto", lambda g: g[~_post.loc[g.index]].mean()),
        pct_dias_post_umbral  = ("brecha_90",   lambda g: (g <= 0).mean()),
    )
    .reset_index()
)
lib_resumen["fecha"] = lib_resumen["fecha"].dt.to_timestamp()

print(f"\n── Día de liberación (brecha_90 ≤ 0  ↔  Avance ≥ 90%) ─────────────")
print(f"  Umbral             : {UMBRAL_90:.0%}")
print(f"  Días post-umbral   : {_post.sum():,}  ({_post.mean()*100:.0f}% del total)")
print(f"  Día mediano de liberación : día {lib_resumen['dia_liberacion_90'].median():.0f} del mes")
print(f"  Meses sin liberación      : {lib_resumen['dia_liberacion_90'].isna().sum()}")
print(f"\n  Retiro promedio post-umbral : {df.loc[_post, 'retiro_neto'].mean()/1e6:+.0f}M")
print(f"  Retiro promedio pre-umbral  : {df.loc[~_post, 'retiro_neto'].mean()/1e6:+.0f}M")

# ── Gráfico día de liberación ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=False)
fig.suptitle(
    f"Análisis de Liberación — Umbral {UMBRAL_90:.0%}\n"
    "brecha_90 = 90% − Avance  |  negativo = banco ya cruzó el umbral",
    fontsize=12, fontweight="bold",
)

# Panel 1: día de liberación a lo largo del tiempo
ax = axes[0]
valid = lib_resumen.dropna(subset=["dia_liberacion_90"])
ax.bar(valid["fecha"], valid["dia_liberacion_90"],
       width=20, color="#1565C0", alpha=0.75)
ax.axhline(valid["dia_liberacion_90"].median(), color="red", lw=1.2, ls="--",
           label=f"Mediana = día {valid['dia_liberacion_90'].median():.0f}")
ax.set_ylabel("Día del mes")
ax.set_title("Día del mes en que brecha_90 cruza 0 (Avance ≥ 90%)")
ax.legend(fontsize=9)
ax.set_ylim(0, 31)

# Panel 2: brecha_90 con retiro neto superpuesto (año representativo)
ax = axes[1]
anio_ej = 2023
mask_ej = df["fecha"].dt.year == anio_ej
df_ej2  = df[mask_ej]
cols_b  = np.where(df_ej2["brecha_90"] <= 0, "#4CAF50", "#1565C0")
ax.bar(df_ej2["fecha"], df_ej2["brecha_90"] * 100,
       color=cols_b, alpha=0.75, width=1)
ax.axhline(0, color="red", lw=1.2, ls="--", label="Umbral 90%")
ax2 = ax.twinx()
ax2.plot(df_ej2["fecha"], df_ej2["retiro_neto"] / 1e9,
         color="navy", lw=1.0, alpha=0.7, label="Retiro neto")
ax2.set_ylabel("Retiro neto (B USD)", fontsize=9, color="navy")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}B"))
ax.set_ylabel("brecha_90 (pp)", fontsize=9)
ax.set_title(f"brecha_90 y retiro neto — {anio_ej}  (verde = banco post-umbral)")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

# Panel 3: retiro promedio pre vs post umbral por mes
ax = axes[2]
ax.bar(lib_resumen["fecha"],
       lib_resumen["retiro_post_umbral"] / 1e6,
       width=20, color="#4CAF50", alpha=0.8, label="Post-umbral (brecha ≤ 0)")
ax.bar(lib_resumen["fecha"],
       lib_resumen["retiro_pre_umbral"] / 1e6,
       width=20, color="#1565C0", alpha=0.6, label="Pre-umbral (brecha > 0)",
       bottom=lib_resumen["retiro_post_umbral"].fillna(0) / 1e6)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Retiro neto promedio (M USD)")
ax.set_title("Retiro neto promedio pre vs post umbral (por mes)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}M"))

plt.tight_layout()
ruta_lib = DIR_OUT / "dia_liberacion_90.png"
fig.savefig(ruta_lib, dpi=130, bbox_inches="tight")
plt.close()
print(f"\nGráfico liberación guardado: {ruta_lib.name}")

# ── Análisis brecha_90 y ritmo_encaje ────────────────────────────────────────
df_b = df[["fecha", "brecha_90", "ritmo_encaje", "retiro_neto"]].dropna()

r_b_same = np.corrcoef(df_b["brecha_90"], df_b["retiro_neto"])[0, 1]
r_b_lag1 = np.corrcoef(df_b["brecha_90"].iloc[:-1], df_b["retiro_neto"].iloc[1:])[0, 1]
r_r_same = np.corrcoef(df_b["ritmo_encaje"], df_b["retiro_neto"])[0, 1]
r_r_lag1 = np.corrcoef(df_b["ritmo_encaje"].iloc[:-1], df_b["retiro_neto"].iloc[1:])[0, 1]

print(f"\n── Análisis brecha_90 y ritmo_encaje ───────────────────────────────")
print(f"  Días post-umbral (brecha ≤ 0) : {(df_b['brecha_90'] <= 0).sum():,}  ({(df_b['brecha_90'] <= 0).mean()*100:.0f}%)")
print(f"  Mediana brecha_90             : {df_b['brecha_90'].median()*100:+.1f} pp")
print(f"  P25 / P75 brecha_90           : {df_b['brecha_90'].quantile(0.25)*100:+.1f} pp  /  {df_b['brecha_90'].quantile(0.75)*100:+.1f} pp")
print(f"\n  brecha_90    vs retiro (mismo día) : r = {r_b_same:+.4f}")
print(f"  brecha_90    vs retiro (lag 1 día) : r = {r_b_lag1:+.4f}")
print(f"  ritmo_encaje vs retiro (mismo día) : r = {r_r_same:+.4f}")
print(f"  ritmo_encaje vs retiro (lag 1 día) : r = {r_r_lag1:+.4f}")

df_b["quintil_brecha"] = pd.qcut(df_b["brecha_90"], q=5,
                                  labels=["Q1 (más neg.)", "Q2", "Q3", "Q4", "Q5 (más pos.)"])
qt_b = df_b.groupby("quintil_brecha", observed=True)["retiro_neto"].mean() / 1e6
print(f"\n  Retiro neto promedio por quintil de brecha_90:")
print(f"  (Q1 = banco más por encima del 90% | Q5 = banco más lejos del umbral)")
for q, v in qt_b.items():
    print(f"    {q}: {v:+.0f}M")

# ── Gráfico brecha_90 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 13))
fig.suptitle(
    "Análisis de brecha_90 y ritmo_encaje\n"
    "brecha_90 = 90% − Avance  |  negativo = banco ya superó el umbral",
    fontsize=12, fontweight="bold",
)

# Panel 1: scatter brecha_90 vs retiro_neto
ax = axes[0]
sc = ax.scatter(df_b["brecha_90"] * 100, df_b["retiro_neto"] / 1e9,
                c=df_b["fecha"].dt.year, cmap="plasma",
                alpha=0.2, s=10, linewidths=0)
coef_b = np.polyfit(df_b["brecha_90"], df_b["retiro_neto"], 1)
xr_b   = np.linspace(df_b["brecha_90"].quantile(0.01), df_b["brecha_90"].quantile(0.99), 200)
ax.plot(xr_b * 100, np.polyval(coef_b, xr_b) / 1e9, color="black", lw=2,
        label=f"r (mismo día) = {r_b_same:+.3f}  |  r (lag 1) = {r_b_lag1:+.3f}")
ax.axvline(0, color="red", lw=1.2, ls="--", label="Umbral 90%")
ax.axhline(0, color="gray", lw=0.5, ls=":")
plt.colorbar(sc, ax=ax, label="Año")
ax.set_xlabel("brecha_90 (pp)  — negativo = ya superó el 90%", fontsize=10)
ax.set_ylabel("Retiro neto (B USD)", fontsize=10)
ax.set_title("Dispersión: brecha_90 vs retiro neto diario")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}B"))

# Panel 2: scatter ritmo_encaje vs retiro_neto
ax = axes[1]
sc2 = ax.scatter(df_b["ritmo_encaje"].clip(0, 3), df_b["retiro_neto"] / 1e9,
                 c=df_b["fecha"].dt.year, cmap="plasma",
                 alpha=0.2, s=10, linewidths=0)
coef_r = np.polyfit(df_b["ritmo_encaje"].clip(0, 3), df_b["retiro_neto"], 1)
xr_r   = np.linspace(0, 3, 200)
ax.plot(xr_r, np.polyval(coef_r, xr_r) / 1e9, color="black", lw=2,
        label=f"r (mismo día) = {r_r_same:+.3f}  |  r (lag 1) = {r_r_lag1:+.3f}")
ax.axvline(1, color="red", lw=1.2, ls="--", label="ritmo = 1× exigible")
ax.axhline(0, color="gray", lw=0.5, ls=":")
plt.colorbar(sc2, ax=ax, label="Año")
ax.set_xlabel("ritmo_encaje  (encaje / exigible diario)", fontsize=10)
ax.set_ylabel("Retiro neto (B USD)", fontsize=10)
ax.set_title("Dispersión: ritmo_encaje vs retiro neto diario")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}B"))

# Panel 3: retiro promedio por quintil de brecha
ax = axes[2]
colores_q = ["#388e3c", "#a5d6a7", "#fff9c4", "#ef9a9a", "#d32f2f"]
ax.bar(qt_b.index.astype(str), qt_b.values,
       color=colores_q, alpha=0.85, edgecolor="white")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Quintil de brecha_90  (Q1 = más negativa = banco más holgado)", fontsize=9)
ax.set_ylabel("Retiro neto promedio (M USD)", fontsize=10)
ax.set_title("Retiro neto promedio por quintil de brecha_90")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}M"))

plt.tight_layout()
ruta_brecha = DIR_OUT / "brecha_90.png"
fig.savefig(ruta_brecha, dpi=130, bbox_inches="tight")
plt.close()
print(f"\nGráfico brecha_90 guardado: {ruta_brecha.name}")

# ── Export ─────────────────────────────────────────────────────────────────────
with pd.ExcelWriter(ruta_out, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Datos", index=False)
    balance.to_excel(writer, sheet_name="Balance_Mensual", index=False)
    df_reg[["fecha", "var_encaje_ovn", "retiro_neto"]].assign(
        residuo=y_d - yhat
    ).to_excel(writer, sheet_name="Regresion_Diaria", index=False)
    validacion.to_excel(writer, sheet_name="Validacion_OpcionA", index=False)
    lib_resumen.to_excel(writer, sheet_name="Liberacion_Mensual", index=False)

print(f"\nExportado: {ruta_out}")
print(f"  Hoja 'Datos'             : {len(df):,} filas")
print(f"  Hoja 'Balance_Mensual'   : {len(balance)} filas")
print(f"  Hoja 'Liberacion_Mensual': {len(lib_resumen)} filas")

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESTRATEGIA SOBREENCAJE BBVA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Estrategia sobreencaje BBVA")
print("=" * 65)

# Fracción de la mediana del saldo mensual que define el umbral (adaptable)
UMBRAL_SALDO         = 0.50
# % mínimo del retiro 15d que debe concentrarse en los últimos N días hábiles
UMBRAL_CONCENTRACION = 0.90
# Ventana de días hábiles al cierre del mes (aplica a BBVA y Sistema)
N_DIAS_HABILES       = 6

# ── Retiro neto diario del SISTEMA (toda la banca local) ──────────────────────
# Porcentaje = retiro sistema últimos 6d / retiro sistema mes completo
_sis_ok = False
try:
    _tx = pd.read_excel(RUTA_TRANSACCIONES)
    _tx["fecha"] = pd.to_datetime(_tx["Fecha Valor"])
    _tx["monto"] = pd.to_numeric(_tx["Delivery Principal Usd"], errors="coerce")
    _flujo_sis = (_tx.groupby("fecha")["monto"]
                  .sum()
                  .rename("retiro_neto_sis")
                  .reset_index())
    df = df.merge(_flujo_sis, on="fecha", how="left")
    _sis_ok = True
    print(f"  Sistema cargado: {len(_flujo_sis):,} días hábiles "
          f"({_flujo_sis['fecha'].min().date()} → {_flujo_sis['fecha'].max().date()})")
except Exception as _e:
    df["retiro_neto_sis"] = np.nan
    print(f"  AVISO: no se cargó Transacciones_BancaLocal — {_e}")

# ── Calendario de días hábiles: excluye feriados Perú + USA ──────────────────
class _PeruCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("AnioNuevo",      month=1,  day=1),
        Holiday("JuevesSanto",    month=1,  day=1, offset=[Easter(), _Day(-3)]),
        GoodFriday,                               # Viernes Santo
        Holiday("Trabajo",        month=5,  day=1),
        Holiday("SanPedro",       month=6,  day=29),
        Holiday("FiestasP1",      month=7,  day=28),
        Holiday("FiestasP2",      month=7,  day=29),
        Holiday("SantaRosa",      month=8,  day=30),
        Holiday("Angamos",        month=10, day=8),
        Holiday("TodosSantos",    month=11, day=1),
        Holiday("Inmaculada",     month=12, day=8),
        Holiday("Navidad",        month=12, day=25),
    ]

_f_ini = df["fecha"].min()
_f_fin = df["fecha"].max()
_hols_peru = set(_PeruCalendar().holidays(_f_ini, _f_fin).normalize())
_hols_usa  = set(USFederalHolidayCalendar().holidays(_f_ini, _f_fin).normalize())
_hols_all  = _hols_peru | _hols_usa

df["_is_bday"] = (
    (df["fecha"].dt.weekday < 5) &          # lunes a viernes
    (~df["fecha"].isin(_hols_all))          # no feriado Perú ni USA
)
n_bday = df["_is_bday"].sum()
print(f"  Días hábiles (sin feriados Perú/USA): {n_bday} / {len(df)} "
      f"({100*n_bday/len(df):.1f}%)")

df["_am"] = df["fecha"].dt.to_period("M")


def _met_estrat(g):
    g      = g.sort_values("fecha")
    # Solo días hábiles: lunes-viernes, sin feriados Perú ni USA
    g_bday = g[g["_is_bday"]] if "_is_bday" in g.columns else g
    late   = g_bday.tail(N_DIAS_HABILES)   # últimos N días hábiles del mes
    late15 = g_bday.tail(15)               # últimos 15 días hábiles del mes

    retiro_acum_6d  = late["retiro_neto"].sum()  if len(late)   else np.nan
    retiro_acum_15d = late15["retiro_neto"].sum() if len(late15) else np.nan
    # mediana del saldo usa todos los días (incluye no-hábiles si existen)
    mediana_saldo   = g["encaje_ovn"].median()
    umbral          = UMBRAL_SALDO * mediana_saldo

    estrategia = (np.isfinite(retiro_acum_6d) and np.isfinite(umbral)
                  and retiro_acum_6d < -umbral)

    # Concentración BBVA: fracción del retiro 15d que ocurre en los últimos 6d
    if (np.isfinite(retiro_acum_6d) and np.isfinite(retiro_acum_15d)
            and retiro_acum_15d < 0 and retiro_acum_6d < 0):
        concentracion = retiro_acum_6d / retiro_acum_15d
    else:
        concentracion = np.nan
    concentrada = np.isfinite(concentracion) and concentracion >= UMBRAL_CONCENTRACION

    # Sistema: retiro neto acumulado últimos N días hábiles
    sis_6d  = late["retiro_neto_sis"].sum()    if "retiro_neto_sis" in late.columns   else np.nan
    sis_mes = g_bday["retiro_neto_sis"].sum()  if "retiro_neto_sis" in g_bday.columns else np.nan

    return pd.Series({
        "retiro_acum_6d_M":  round(retiro_acum_6d  / 1e6, 1) if np.isfinite(retiro_acum_6d)  else np.nan,
        "retiro_acum_15d_M": round(retiro_acum_15d / 1e6, 1) if np.isfinite(retiro_acum_15d) else np.nan,
        "mediana_saldo_M":   round(mediana_saldo   / 1e6, 1) if np.isfinite(mediana_saldo)   else np.nan,
        "umbral_M":          round(umbral           / 1e6, 1) if np.isfinite(umbral)          else np.nan,
        "estrategia":        estrategia,
        "concentracion":     round(concentracion, 4) if np.isfinite(concentracion) else np.nan,
        "concentrada":       concentrada,
        "sis_retiro_6d_M":   round(sis_6d  / 1e6, 1) if np.isfinite(sis_6d)  else np.nan,
        "sis_retiro_mes_M":  round(sis_mes / 1e6, 1) if np.isfinite(sis_mes) else np.nan,
    })


print(f"\n  Umbral estrategia:    retiro 6d > {UMBRAL_SALDO*100:.0f}% mediana saldo mensual")
print(f"  Umbral concentración: ≥ {UMBRAL_CONCENTRACION*100:.0f}% del retiro 15d ocurre en últimos 6d")
print("\n  Calculando indicadores por mes ...")
dm = df.groupby("_am").apply(_met_estrat).reset_index()
dm["anio"]       = dm["_am"].dt.year
dm["mes"]        = dm["_am"].dt.month
dm["fecha_plot"] = dm["_am"].dt.to_timestamp()

n_est  = dm["estrategia"].sum()
n_conc = dm["concentrada"].sum()
n_both = (dm["estrategia"] & dm["concentrada"]).sum()
print(f"  Meses con estrategia detectada:  {n_est}  / {len(dm)} ({100*n_est/max(len(dm),1):.0f}%)")
print(f"  Meses con retiro concentrado:    {n_conc} / {len(dm)} ({100*n_conc/max(len(dm),1):.0f}%)")
print(f"  Meses con ambas condiciones:     {n_both} / {len(dm)} ({100*n_both/max(len(dm),1):.0f}%)")

# Diagnóstico: muestra los 10 meses con mayor retiro para verificar el umbral
print("\n── Top 10 meses por retiro acumulado 5d ────────────────────────────────────────")
print(f"  {'Mes':<10} {'Ret6d (M)':>10} {'Ret15d (M)':>11} {'Med.saldo (M)':>14} "
      f"{'Umbral':>8} {'Estrat':>7} {'Conc%':>6} {'Concent':>8}")
print(f"  {'-'*10} {'-'*10} {'-'*11} {'-'*14} {'-'*8} {'-'*7} {'-'*6} {'-'*8}")
for _, _r in dm.nsmallest(10, "retiro_acum_6d_M").iterrows():
    _conc_pct = f"{_r['concentracion']*100:.0f}%" if np.isfinite(_r["concentracion"]) else "n/a"
    _est_lbl  = "SÍ" if _r["estrategia"] else "NO"
    _con_lbl  = "SÍ" if _r["concentrada"] else "NO"
    print(f"  {str(_r['_am']):<10} {_r['retiro_acum_6d_M']:>10,.1f} "
          f"{_r['retiro_acum_15d_M']:>11,.1f} {_r['mediana_saldo_M']:>14,.1f} "
          f"{_r['umbral_M']:>8,.1f} {_est_lbl:>7} {_conc_pct:>6} {_con_lbl:>8}")

# ── Heatmap año × mes (intensidad = retiro / mediana saldo) ───────────────────
_MESES_ABR = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
              "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Intensidad normalizada: retiro / umbral  (> 1 = estrategia activada)
dm["_intensidad"] = (-dm["retiro_acum_6d_M"] / dm["umbral_M"]).clip(lower=0)
# % BBVA: retiro BBVA / mediana saldo mensual
dm["_pct"] = (-dm["retiro_acum_6d_M"] / dm["mediana_saldo_M"].replace(0, np.nan) * 100).clip(lower=0)
# % Sistema en celda: BBVA / Sistema (cuánto representa BBVA del retiro del sistema)
_mask_ratio = (dm["retiro_acum_6d_M"] < 0) & (dm["sis_retiro_6d_M"] < 0)
dm["_sis_ratio"] = np.where(
    _mask_ratio,
    dm["retiro_acum_6d_M"] / dm["sis_retiro_6d_M"].replace(0, np.nan) * 100,
    np.nan,
)

piv_int      = dm.pivot(index="anio", columns="mes", values="_intensidad").sort_index()
piv_est      = dm.pivot(index="anio", columns="mes", values="estrategia").sort_index()
piv_ret      = dm.pivot(index="anio", columns="mes", values="retiro_acum_6d_M").sort_index()
piv_pct      = dm.pivot(index="anio", columns="mes", values="_pct").sort_index()
piv_conc     = dm.pivot(index="anio", columns="mes", values="concentracion").sort_index()
piv_cond     = dm.pivot(index="anio", columns="mes", values="concentrada").sort_index()
piv_sis      = dm.pivot(index="anio", columns="mes", values="sis_retiro_6d_M").sort_index()
piv_sis_rat  = dm.pivot(index="anio", columns="mes", values="_sis_ratio").sort_index()

_vals = piv_int.values.astype(float)
_vmax = float(np.nanmax(_vals)) if np.any(np.isfinite(_vals)) else 2.0
_cmap = plt.cm.YlOrRd

fig, ax = plt.subplots(figsize=(14, max(4, len(piv_int) * 0.6 + 2)))
fig.suptitle(
    f"Estrategia sobreencaje BBVA — Intensidad del retiro por año y mes\n"
    f"Intensidad = retiro {N_DIAS_HABILES}d / ({UMBRAL_SALDO*100:.0f}% × mediana saldo)  ·  "
    f"> 1 = estrategia activada (borde azul)  ·  Celda: BBVA retiro / (% saldo)  ·  Sis: monto (% BBVA/Sis)",
    fontweight="bold", fontsize=10,
)
im = ax.imshow(_vals, aspect="auto", cmap=_cmap, vmin=0, vmax=_vmax)

ax.set_xticks(range(12))
ax.set_xticklabels(_MESES_ABR)
ax.set_yticks(range(len(piv_int)))
ax.set_yticklabels(piv_int.index.astype(str))

for _i, _anio in enumerate(piv_int.index):
    for _j, _mes in enumerate(range(1, 13)):
        _vi = piv_int.loc[_anio, _mes]  if _mes in piv_int.columns  else np.nan
        _ve = piv_est.loc[_anio, _mes]  if _mes in piv_est.columns  else False
        _vr = piv_ret.loc[_anio, _mes]  if _mes in piv_ret.columns  else np.nan
        _vp = piv_pct.loc[_anio, _mes]  if _mes in piv_pct.columns  else np.nan
        _vs = piv_sis.loc[_anio, _mes]     if _mes in piv_sis.columns     else np.nan
        _vsr = piv_sis_rat.loc[_anio, _mes] if _mes in piv_sis_rat.columns else np.nan
        _fi  = float(_vi)  if _vi  is not None else np.nan
        _fr  = float(_vr)  if _vr  is not None else np.nan
        _fp  = float(_vp)  if (_vp  is not None and _vp  is not pd.NA) else np.nan
        _fs  = float(_vs)  if (_vs  is not None and _vs  is not pd.NA) else np.nan
        _fsr = float(_vsr) if (_vsr is not None and _vsr is not pd.NA) else np.nan
        if not np.isfinite(_fi):
            continue
        _ctxt = "white" if _fi > _vmax * 0.55 else "black"
        # BBVA — parte superior: monto + (% mediana saldo)
        if np.isfinite(_fr) and np.isfinite(_fp):
            _bbva_lbl = f"{_fr:,.0f}\n({_fp:.0f}%)"
        elif np.isfinite(_fr):
            _bbva_lbl = f"{_fr:,.0f}"
        else:
            _bbva_lbl = ""
        if _bbva_lbl:
            ax.text(_j, _i - 0.14, _bbva_lbl, ha="center", va="center",
                    fontsize=5.3, color=_ctxt, linespacing=1.2,
                    fontweight="bold" if _ve else "normal")
        # Sistema — parte inferior: monto + (% BBVA/Sistema)
        if np.isfinite(_fs):
            _sis_lbl = (f"Sis:{_fs:,.0f} ({_fsr:.0f}%)"
                        if np.isfinite(_fsr) else f"Sis:{_fs:,.0f}")
            ax.text(_j, _i + 0.30, _sis_lbl, ha="center", va="center",
                    fontsize=4.5, color=_ctxt, style="italic")
        if _ve:
            ax.add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), 1, 1,
                         fill=False, edgecolor="#1565C0", linewidth=2.2))

plt.colorbar(im, ax=ax, label=f"Retiro / ({UMBRAL_SALDO*100:.0f}% × mediana saldo)",
             shrink=0.75)
ax.set_xlabel("Mes")
ax.set_ylabel("Año")
from matplotlib.patches import Patch as _Patch
ax.legend(handles=[
    _Patch(facecolor=_cmap(0.4), label="Retiro moderado"),
    _Patch(facecolor=_cmap(0.85), label="Retiro intenso"),
    plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#1565C0", lw=2,
                  label=f"Estrategia activada (retiro > {UMBRAL_SALDO*100:.0f}% saldo)"),

], loc="lower right", fontsize=7, framealpha=0.9)
plt.tight_layout()
_p1 = DIR_OUT / "00_heatmap_estrategia.png"
fig.savefig(_p1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Guardado: {_p1.name}")

# ── Heatmap adicional: % retiro sobre mediana saldo del mes PREVIO ─────────────
# mediana_saldo del mes anterior (shift=1); primer mes usa el propio mes
dm_s = dm.sort_values("fecha_plot").reset_index(drop=True)
dm_s["_saldo_prev_M"] = dm_s["mediana_saldo_M"].shift(1).fillna(dm_s["mediana_saldo_M"])

dm_s["_int_prev"]  = (-dm_s["retiro_acum_6d_M"] / (UMBRAL_SALDO * dm_s["_saldo_prev_M"].replace(0, np.nan))).clip(lower=0)
dm_s["_pct_prev"]  = (-dm_s["retiro_acum_6d_M"] / dm_s["_saldo_prev_M"].replace(0, np.nan) * 100).clip(lower=0)
_mask_ratio2 = (dm_s["retiro_acum_6d_M"] < 0) & (dm_s["sis_retiro_6d_M"] < 0)
dm_s["_sis_ratio2"] = np.where(
    _mask_ratio2,
    dm_s["retiro_acum_6d_M"] / dm_s["sis_retiro_6d_M"].replace(0, np.nan) * 100,
    np.nan,
)

piv_int2     = dm_s.pivot(index="anio", columns="mes", values="_int_prev").sort_index()
piv_pct2     = dm_s.pivot(index="anio", columns="mes", values="_pct_prev").sort_index()
piv_sis_rat2 = dm_s.pivot(index="anio", columns="mes", values="_sis_ratio2").sort_index()

_vals2 = piv_int2.values.astype(float)
_vmax2 = float(np.nanmax(_vals2)) if np.any(np.isfinite(_vals2)) else 2.0

fig2, ax2 = plt.subplots(figsize=(14, max(4, len(piv_int2) * 0.6 + 2)))
fig2.suptitle(
    f"Estrategia sobreencaje BBVA — Intensidad usando saldo del mes PREVIO\n"
    f"Intensidad = retiro {N_DIAS_HABILES}d / ({UMBRAL_SALDO*100:.0f}% × mediana saldo mes anterior)  ·  "
    f"> 1 = estrategia activada (borde azul)  ·  Celda: BBVA retiro / (% saldo previo)  ·  Sis: monto (% BBVA/Sis)",
    fontweight="bold", fontsize=10,
)
im2 = ax2.imshow(_vals2, aspect="auto", cmap=_cmap, vmin=0, vmax=_vmax2)

ax2.set_xticks(range(12))
ax2.set_xticklabels(_MESES_ABR)
ax2.set_yticks(range(len(piv_int2)))
ax2.set_yticklabels(piv_int2.index.astype(str))

for _i, _anio in enumerate(piv_int2.index):
    for _j, _mes in enumerate(range(1, 13)):
        _vi = piv_int2.loc[_anio, _mes] if _mes in piv_int2.columns else np.nan
        _ve = piv_est.loc[_anio, _mes]  if _mes in piv_est.columns  else False
        _vr = piv_ret.loc[_anio, _mes]  if _mes in piv_ret.columns  else np.nan
        _vp = piv_pct2.loc[_anio, _mes] if _mes in piv_pct2.columns else np.nan
        _vs  = piv_sis.loc[_anio, _mes]      if _mes in piv_sis.columns      else np.nan
        _vsr = piv_sis_rat2.loc[_anio, _mes] if _mes in piv_sis_rat2.columns else np.nan
        _fi  = float(_vi)  if _vi  is not None else np.nan
        _fr  = float(_vr)  if _vr  is not None else np.nan
        _fp  = float(_vp)  if (_vp  is not None and _vp  is not pd.NA) else np.nan
        _fs  = float(_vs)  if (_vs  is not None and _vs  is not pd.NA) else np.nan
        _fsr = float(_vsr) if (_vsr is not None and _vsr is not pd.NA) else np.nan
        if not np.isfinite(_fi):
            continue
        _ctxt = "white" if _fi > _vmax2 * 0.55 else "black"
        # BBVA — parte superior: monto + (% saldo previo)
        if np.isfinite(_fr) and np.isfinite(_fp):
            _bbva_lbl = f"{_fr:,.0f}\n({_fp:.0f}%)"
        elif np.isfinite(_fr):
            _bbva_lbl = f"{_fr:,.0f}"
        else:
            _bbva_lbl = ""
        if _bbva_lbl:
            ax2.text(_j, _i - 0.14, _bbva_lbl, ha="center", va="center",
                     fontsize=5.3, color=_ctxt, linespacing=1.2,
                     fontweight="bold" if _ve else "normal")
        # Sistema — parte inferior: monto + (% BBVA/Sistema)
        if np.isfinite(_fs):
            _sis_lbl = (f"Sis:{_fs:,.0f} ({_fsr:.0f}%)"
                        if np.isfinite(_fsr) else f"Sis:{_fs:,.0f}")
            ax2.text(_j, _i + 0.30, _sis_lbl, ha="center", va="center",
                     fontsize=4.5, color=_ctxt, style="italic")
        if _ve:
            ax2.add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), 1, 1,
                          fill=False, edgecolor="#1565C0", linewidth=2.2))

plt.colorbar(im2, ax=ax2, label=f"Retiro / ({UMBRAL_SALDO*100:.0f}% × mediana saldo previo)",
             shrink=0.75)
ax2.set_xlabel("Mes")
ax2.set_ylabel("Año")
ax2.legend(handles=[
    _Patch(facecolor=_cmap(0.4), label="Retiro moderado"),
    _Patch(facecolor=_cmap(0.85), label="Retiro intenso"),
    plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#1565C0", lw=2,
                  label=f"Estrategia activada (retiro > {UMBRAL_SALDO*100:.0f}% saldo previo)"),

], loc="lower right", fontsize=7, framealpha=0.9)
plt.tight_layout()
_p1b = DIR_OUT / "00b_heatmap_estrategia_saldo_prev.png"
fig2.savefig(_p1b, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Guardado: {_p1b.name}")

# ── Timeline: retiro acumulado 5d vs umbral ────────────────────────────────────
_dm_c = dm.dropna(subset=["retiro_acum_6d_M"]).copy()
_cb   = _dm_c["estrategia"].map({True: "#E53935", False: "#90A4AE"})

fig, ax = plt.subplots(figsize=(15, 5))
fig.suptitle("Retiro acumulado últimos 6 días hábiles vs umbral mensual",
             fontweight="bold")
ax.bar(_dm_c["fecha_plot"], _dm_c["retiro_acum_6d_M"], width=22,
       color=_cb, alpha=0.85)
ax.step(_dm_c["fecha_plot"], -_dm_c["umbral_M"], where="mid",
        color="#B71C1C", lw=1.5, ls="--",
        label=f"−Umbral ({UMBRAL_SALDO*100:.0f}% × mediana saldo)")
ax.axhline(0, color="black", lw=0.6, ls=":")
ax.set_ylabel("Retiro acumulado 5d (M USD)")
ax.set_xlabel("Fecha")
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color="#E53935", alpha=0.85, label="Estrategia activa"),
    plt.Rectangle((0,0),1,1, color="#90A4AE", alpha=0.85, label="Sin estrategia"),
    plt.Line2D([0],[0], color="#B71C1C", lw=1.5, ls="--",
               label=f"−Umbral ({UMBRAL_SALDO*100:.0f}% × mediana saldo)"),
], fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}M"))
plt.tight_layout()
_p2 = DIR_OUT / "01_retiro_vs_umbral.png"
fig.savefig(_p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Guardado: {_p2.name}")

# ── Resumen por año ────────────────────────────────────────────────────────────
_res = (
    dm.groupby("anio")
    .agg(meses_datos=("estrategia", "count"),
         meses_estrat=("estrategia", "sum"))
    .reset_index()
)
_res["clasificacion"] = _res["meses_estrat"].apply(
    lambda n: "MENSUAL"    if n >= 8 else
              "TRIMESTRAL" if n >= 3 else
              "NO APLICA"  if n == 0 else
              "ESPORÁDICA"
)

# ── Export Excel ───────────────────────────────────────────────────────────────
try:
    _ruta_est = DIR_OUT / "resultados_estrategia.xlsx"
    with pd.ExcelWriter(_ruta_est, engine="openpyxl") as _wr:
        (dm[["_am", "anio", "mes", "estrategia",
             "retiro_acum_6d_M", "retiro_acum_15d_M",
             "mediana_saldo_M", "umbral_M",
             "concentracion", "concentrada",
             "sis_retiro_6d_M", "sis_retiro_mes_M"]]
         .assign(bbva_pct_saldo=dm["_pct"].round(1),
                 bbva_pct_del_sistema=dm["_sis_ratio"].round(1))
         .rename(columns={"_am": "mes_periodo",
                          "concentracion": "conc_6d_vs_15d"})
         .to_excel(_wr, sheet_name="Por_mes", index=False))
        _res.rename(columns={"meses_datos":  "meses_con_datos",
                             "meses_estrat": "meses_estrategia"}).to_excel(
            _wr, sheet_name="Resumen_anual", index=False)
    print(f"  Guardado: {_ruta_est.name}")
except Exception as _e:
    print(f"  AVISO: no se pudo exportar Excel — {_e}")

# ── Tabla resumen consola ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ESTRATEGIA SOBREENCAJE BBVA — RESUMEN POR AÑO")
print(f"  Umbral: retiro 6d > {UMBRAL_SALDO*100:.0f}% de la mediana del saldo mensual")
print("=" * 55)
print(f"  {'Año':<6}  {'Meses c/estrat.':>16}  {'Clasificación'}")
print(f"  {'-'*6}  {'-'*16}  {'-'*14}")
for _, _r in _res.iterrows():
    _flag = " ◄" if _r["clasificacion"] in ("MENSUAL", "TRIMESTRAL") else ""
    print(f"  {int(_r['anio']):<6}  "
          f"{int(_r['meses_estrat']):>5} / {int(_r['meses_datos']):<9}  "
          f"{_r['clasificacion']}{_flag}")
print("=" * 55)
print(f"\n  Archivos en: {DIR_OUT}")

# Limpiar columnas temporales
df.drop(columns=["_am", "_is_bday", "retiro_neto_sis"], inplace=True, errors="ignore")
dm.drop(columns=["_intensidad", "_pct", "_sis_ratio"], inplace=True, errors="ignore")
dm_s.drop(columns=["_saldo_prev_M", "_int_prev", "_pct_prev"], inplace=True, errors="ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESTRATEGIA SOBREENCAJE — OTROS BANCOS (BCP, IBK, SCO, CITI)
# ══════════════════════════════════════════════════════════════════════════════
RUTA_BANCOS = Path(
    r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Raw\bancos_encaje.xlsx"
)

_BANCOS = ["BCP", "IBK", "SCO", "CITI"]


def _procesar_banco_df(df_raw):
    """Estandariza columnas, calcula derivadas y marca días hábiles."""
    db = df_raw.iloc[:, :9].copy()
    db.columns = [
        "fecha", "entidad", "codigo",
        "overnight", "cta_cte", "caja",
        "tose", "exigible", "retiro_neto",
    ]
    db["fecha"]      = pd.to_datetime(db["fecha"])
    db               = db.sort_values("fecha").reset_index(drop=True)
    db["encaje"]     = db["cta_cte"] + db["caja"]
    db["encaje_ovn"] = db["encaje"] + db["overnight"]

    # Días hábiles: reutiliza _hols_all del scope principal
    db["_is_bday"] = (
        (db["fecha"].dt.weekday < 5) &
        (~db["fecha"].isin(_hols_all))
    )

    # Retiro neto del sistema (ya calculado en scope principal)
    if _sis_ok:
        db = db.merge(_flujo_sis, on="fecha", how="left")
    else:
        db["retiro_neto_sis"] = np.nan

    db["_am"] = db["fecha"].dt.to_period("M")
    return db


def _heatmap_saldo_prev_banco(dm_b, nombre, dir_out_b):
    """Genera el heatmap de intensidad usando saldo del mes previo para un banco."""
    dm_b = dm_b.sort_values("fecha_plot").reset_index(drop=True)
    dm_b["_saldo_prev_M"] = dm_b["mediana_saldo_M"].shift(1).fillna(dm_b["mediana_saldo_M"])

    dm_b["_int_prev"] = (
        -dm_b["retiro_acum_6d_M"] /
        (UMBRAL_SALDO * dm_b["_saldo_prev_M"].replace(0, np.nan))
    ).clip(lower=0)
    dm_b["_pct_prev"] = (
        -dm_b["retiro_acum_6d_M"] /
        dm_b["_saldo_prev_M"].replace(0, np.nan) * 100
    ).clip(lower=0)

    _mask = (dm_b["retiro_acum_6d_M"] < 0) & (dm_b["sis_retiro_6d_M"] < 0)
    dm_b["_sis_ratio"] = np.where(
        _mask,
        dm_b["retiro_acum_6d_M"] / dm_b["sis_retiro_6d_M"].replace(0, np.nan) * 100,
        np.nan,
    )

    piv_int  = dm_b.pivot(index="anio", columns="mes", values="_int_prev").sort_index()
    piv_est  = dm_b.pivot(index="anio", columns="mes", values="estrategia").sort_index()
    piv_ret  = dm_b.pivot(index="anio", columns="mes", values="retiro_acum_6d_M").sort_index()
    piv_pct  = dm_b.pivot(index="anio", columns="mes", values="_pct_prev").sort_index()
    piv_sis  = dm_b.pivot(index="anio", columns="mes", values="sis_retiro_6d_M").sort_index()
    piv_sisr = dm_b.pivot(index="anio", columns="mes", values="_sis_ratio").sort_index()

    _vals = piv_int.values.astype(float)
    _vmax = float(np.nanmax(_vals)) if np.any(np.isfinite(_vals)) else 2.0
    _cmap = plt.cm.YlOrRd

    fig, ax = plt.subplots(figsize=(14, max(4, len(piv_int) * 0.6 + 2)))
    fig.suptitle(
        f"Estrategia sobreencaje {nombre} — Intensidad usando saldo del mes PREVIO\n"
        f"Intensidad = retiro {N_DIAS_HABILES}d / ({UMBRAL_SALDO*100:.0f}% × mediana saldo mes anterior)  ·  "
        f"> 1 = estrategia activada (borde azul)  ·  Celda: {nombre} retiro / (% saldo previo)  ·  Sis: monto (% {nombre}/Sis)",
        fontweight="bold", fontsize=10,
    )
    im = ax.imshow(_vals, aspect="auto", cmap=_cmap, vmin=0, vmax=_vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(_MESES_ABR)
    ax.set_yticks(range(len(piv_int)))
    ax.set_yticklabels(piv_int.index.astype(str))

    for _i, _anio in enumerate(piv_int.index):
        for _j, _mes in enumerate(range(1, 13)):
            _vi  = piv_int.loc[_anio, _mes]  if _mes in piv_int.columns  else np.nan
            _ve  = piv_est.loc[_anio, _mes]  if _mes in piv_est.columns  else False
            _vr  = piv_ret.loc[_anio, _mes]  if _mes in piv_ret.columns  else np.nan
            _vp  = piv_pct.loc[_anio, _mes]  if _mes in piv_pct.columns  else np.nan
            _vs  = piv_sis.loc[_anio, _mes]  if _mes in piv_sis.columns  else np.nan
            _vsr = piv_sisr.loc[_anio, _mes] if _mes in piv_sisr.columns else np.nan
            _fi  = float(_vi)  if _vi  is not None else np.nan
            _fr  = float(_vr)  if _vr  is not None else np.nan
            _fp  = float(_vp)  if (_vp  is not None and _vp  is not pd.NA) else np.nan
            _fs  = float(_vs)  if (_vs  is not None and _vs  is not pd.NA) else np.nan
            _fsr = float(_vsr) if (_vsr is not None and _vsr is not pd.NA) else np.nan
            if not np.isfinite(_fi):
                continue
            _ctxt = "white" if _fi > _vmax * 0.55 else "black"
            if np.isfinite(_fr) and np.isfinite(_fp):
                _bbva_lbl = f"{_fr:,.0f}\n({_fp:.0f}%)"
            elif np.isfinite(_fr):
                _bbva_lbl = f"{_fr:,.0f}"
            else:
                _bbva_lbl = ""
            if _bbva_lbl:
                ax.text(_j, _i - 0.14, _bbva_lbl, ha="center", va="center",
                        fontsize=5.3, color=_ctxt, linespacing=1.2,
                        fontweight="bold" if _ve else "normal")
            if np.isfinite(_fs):
                _sis_lbl = (f"Sis:{_fs:,.0f} ({_fsr:.0f}%)"
                            if np.isfinite(_fsr) else f"Sis:{_fs:,.0f}")
                ax.text(_j, _i + 0.30, _sis_lbl, ha="center", va="center",
                        fontsize=4.5, color=_ctxt, style="italic")
            if _ve:
                ax.add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), 1, 1,
                             fill=False, edgecolor="#1565C0", linewidth=2.2))

    plt.colorbar(im, ax=ax, label=f"Retiro / ({UMBRAL_SALDO*100:.0f}% × mediana saldo previo)",
                 shrink=0.75)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Año")
    ax.legend(handles=[
        _Patch(facecolor=_cmap(0.4), label="Retiro moderado"),
        _Patch(facecolor=_cmap(0.85), label="Retiro intenso"),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#1565C0", lw=2,
                      label=f"Estrategia activada (retiro > {UMBRAL_SALDO*100:.0f}% saldo previo)"),
    ], loc="lower right", fontsize=7, framealpha=0.9)
    plt.tight_layout()

    nombre_lower = nombre.lower()
    _p = dir_out_b / f"00b_heatmap_{nombre_lower}_saldo_prev.png"
    fig.savefig(_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{nombre}] Guardado: {_p.name}")

    dm_b.drop(columns=["_saldo_prev_M", "_int_prev", "_pct_prev", "_sis_ratio"],
              inplace=True, errors="ignore")


print("\n" + "=" * 65)
print("  Estrategia sobreencaje — Otros bancos")
print("=" * 65)

# Medianas de componentes BBVA (reutiliza df ya cargado)
_med_list = []
_med_bbva = (
    df.groupby(df["fecha"].dt.to_period("M"))
    .agg(
        med_overnight=("overnight", "median"),
        med_cta_cte=("cta_cte",    "median"),
        med_caja=("caja",          "median"),
    )
    .reset_index()
)
_med_bbva["fecha_plot"] = _med_bbva["fecha"].dt.to_timestamp()
_med_bbva["banco"]      = "BBVA"
_med_list.append(_med_bbva)

for _banco in _BANCOS:
    print(f"\n  Procesando {_banco} ...")
    try:
        _df_raw = pd.read_excel(RUTA_BANCOS, sheet_name=_banco, header=0)
        _db     = _procesar_banco_df(_df_raw)

        print(f"    Período : {_db['fecha'].min().date()} → {_db['fecha'].max().date()}")
        print(f"    Filas   : {len(_db):,}")

        _dm_b = _db.groupby("_am").apply(_met_estrat).reset_index()
        _dm_b["anio"]       = _dm_b["_am"].dt.year
        _dm_b["mes"]        = _dm_b["_am"].dt.month
        _dm_b["fecha_plot"] = _dm_b["_am"].dt.to_timestamp()

        _n_est = _dm_b["estrategia"].sum()
        print(f"    Estrategia detectada: {_n_est} / {len(_dm_b)} meses")

        # Todos los heatmaps van a la misma carpeta que BBVA
        _heatmap_saldo_prev_banco(_dm_b, _banco, DIR_OUT)

        # Acumular medianas de componentes para gráfico de evolución
        _med_b = (
            _db.groupby("_am")
            .agg(
                med_overnight=("overnight", "median"),
                med_cta_cte=("cta_cte",    "median"),
                med_caja=("caja",          "median"),
            )
            .reset_index()
        )
        _med_b["fecha_plot"] = _med_b["_am"].dt.to_timestamp()
        _med_b["banco"]      = _banco
        _med_list.append(_med_b)

    except Exception as _exc:
        print(f"  AVISO [{_banco}]: {_exc}")

# ── Gráfico: evolución mediana saldo mensual — 5 bancos + sistema ────────────
if _med_list:
    _med_all = pd.concat(_med_list, ignore_index=True)
    for _col in ["med_overnight", "med_cta_cte", "med_caja"]:
        _med_all[_col] = _med_all[_col] / 1e6
    _med_all["med_total"] = (
        _med_all["med_overnight"] + _med_all["med_cta_cte"] + _med_all["med_caja"]
    )

    _bancos_5 = ["BBVA"] + [b for b in _BANCOS if b in _med_all["banco"].unique()]

    _colores_banco = {
        "BBVA":    "#1565C0",
        "BCP":     "#E53935",
        "IBK":     "#F57C00",
        "SCO":     "#43A047",
        "CITI":    "#8E24AA",
        "SISTEMA": "#424242",
        "Otros":   "#BDBDBD",
    }
    _colores_comp = {
        "med_cta_cte":   "#1565C0",
        "med_caja":      "#43A047",
        "med_overnight": "#F57C00",
    }
    _labels_comp = {
        "med_cta_cte":   "Cta Cte BCR",
        "med_caja":      "Caja",
        "med_overnight": "Overnight BCR",
    }

    # Cargar hoja SISTEMA (6 columnas: sin entidad/codigo/retiro_neto)
    _med_sis_df = None
    try:
        _db_sis = pd.read_excel(RUTA_BANCOS, sheet_name="SISTEMA", header=0)
        _db_sis = _db_sis.iloc[:, :6].copy()
        _db_sis.columns = ["fecha", "overnight", "cta_cte", "caja", "tose", "exigible"]
        _db_sis["fecha"] = pd.to_datetime(_db_sis["fecha"])
        _db_sis["_am"]   = _db_sis["fecha"].dt.to_period("M")
        _med_sis = (
            _db_sis.groupby("_am")
            .agg(
                med_overnight=("overnight", "median"),
                med_cta_cte=("cta_cte",    "median"),
                med_caja=("caja",          "median"),
            )
            .reset_index()
        )
        _med_sis["fecha_plot"] = _med_sis["_am"].dt.to_timestamp()
        _med_sis["banco"]      = "SISTEMA"
        for _col in ["med_overnight", "med_cta_cte", "med_caja"]:
            _med_sis[_col] = _med_sis[_col] / 1e6
        _med_sis["med_total"] = (
            _med_sis["med_overnight"] + _med_sis["med_cta_cte"] + _med_sis["med_caja"]
        )
        _med_all    = pd.concat([_med_all, _med_sis], ignore_index=True)
        _med_sis_df = _med_sis
        print(f"  [SISTEMA] Cargado: {len(_med_sis)} meses")
    except Exception as _exc_sis:
        print(f"  AVISO [SISTEMA]: {_exc_sis}")

    _bancos_todos = _bancos_5 + (["SISTEMA"] if _med_sis_df is not None else [])

    # ── 02: Evolución por entidad — subplots individuales (incluye SISTEMA) ──────
    _n_sub = len(_bancos_todos)
    fig_ev, axes_ev = plt.subplots(_n_sub, 1, figsize=(15, 3.2 * _n_sub), sharex=False)
    if _n_sub == 1:
        axes_ev = [axes_ev]

    fig_ev.suptitle(
        "Evolución mediana saldo mensual por entidad y componente\n"
        "(Cta Cte BCR + Caja + Overnight BCR)",
        fontweight="bold", fontsize=12,
    )
    for _ax_b, _bnk in zip(axes_ev, _bancos_todos):
        _d = _med_all[_med_all["banco"] == _bnk].sort_values("fecha_plot")
        if _d.empty:
            _ax_b.set_visible(False)
            continue
        _bottom = np.zeros(len(_d))
        for _comp in ["med_cta_cte", "med_caja", "med_overnight"]:
            _vals_c = _d[_comp].fillna(0).values
            _ax_b.fill_between(_d["fecha_plot"], _bottom, _bottom + _vals_c,
                               alpha=0.75, color=_colores_comp[_comp],
                               label=_labels_comp[_comp])
            _bottom += _vals_c
        _ax_b.plot(_d["fecha_plot"], _d["med_total"],
                   color="black", lw=1.2, ls="--", label="Total encaje OVN")
        _ax_b.set_ylabel("M USD", fontsize=9)
        _ax_b.set_title(_bnk, fontsize=10, fontweight="bold", loc="left")
        _ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}M"))
        _ax_b.legend(fontsize=7, loc="upper left", ncol=4, framealpha=0.8)

    plt.tight_layout()
    _p_ev = DIR_OUT / "02_evolucion_saldo_mensual.png"
    fig_ev.savefig(_p_ev, dpi=150, bbox_inches="tight")
    plt.close(fig_ev)
    print(f"\n  Guardado: {_p_ev.name}")

    # ── 03: Comparativo líneas + apilada (5 bancos + Otros = Sistema - 5) ────────
    fig_cmp, (ax_cmp, ax_stk) = plt.subplots(
        2, 1, figsize=(15, 11), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig_cmp.suptitle(
        "Comparación saldo total encaje OVN — 5 entidades + sistema\n"
        "(mediana mensual  ·  Cta Cte BCR + Caja + Overnight BCR)",
        fontweight="bold", fontsize=12,
    )

    # Panel superior — líneas individuales; SISTEMA en punteado negro
    for _bnk in _bancos_todos:
        _d = _med_all[_med_all["banco"] == _bnk].sort_values("fecha_plot")
        if _d.empty:
            continue
        _is_sis = _bnk == "SISTEMA"
        ax_cmp.plot(
            _d["fecha_plot"], _d["med_total"],
            color=_colores_banco.get(_bnk, "gray"),
            lw=2.2 if _is_sis else 1.8,
            ls="--" if _is_sis else "-",
            label=_bnk,
        )
        _last = _d.iloc[-1]
        ax_cmp.annotate(
            f"{_bnk}  {_last['med_total']:,.0f}M",
            xy=(_last["fecha_plot"], _last["med_total"]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=_colores_banco.get(_bnk, "gray"), va="center",
        )
    ax_cmp.axhline(0, color="black", lw=0.5, ls=":")
    ax_cmp.set_ylabel("M USD", fontsize=10)
    ax_cmp.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}M"))
    ax_cmp.legend(fontsize=9, loc="upper left")

    # Panel inferior — apilada: 5 bancos + "Otros bancos" = Sistema − Σ 5
    _pivot_stk = (
        _med_all[_med_all["banco"].isin(_bancos_5)][["fecha_plot", "banco", "med_total"]]
        .pivot(index="fecha_plot", columns="banco", values="med_total")
        .sort_index()
        .reindex(columns=_bancos_5)
        .fillna(0)
    )

    # "Otros" alineado al mismo índice que el pivot
    if _med_sis_df is not None:
        _sis_aln = (
            _med_sis_df.set_index("fecha_plot")["med_total"]
            .reindex(_pivot_stk.index)
            .ffill()
        )
        _otros_vals = (_sis_aln - _pivot_stk.sum(axis=1)).clip(lower=0).values
    else:
        _otros_vals = None

    # Calcular denominador total para % (Σ 5 bancos + Otros)
    _total_last = _pivot_stk.iloc[-1].sum()
    if _otros_vals is not None:
        _total_last += _otros_vals[-1]

    _bottom_stk = np.zeros(len(_pivot_stk))
    for _bnk in _bancos_5:
        if _bnk not in _pivot_stk.columns:
            continue
        _vals_stk = _pivot_stk[_bnk].values
        ax_stk.fill_between(
            _pivot_stk.index, _bottom_stk, _bottom_stk + _vals_stk,
            alpha=0.80, color=_colores_banco.get(_bnk, "gray"), label=_bnk,
        )
        _bnk_last = _pivot_stk[_bnk].iloc[-1]
        _pct_last = _bnk_last / _total_last * 100 if _total_last > 0 else 0
        ax_stk.annotate(
            f"{_bnk} {_pct_last:.0f}%",
            xy=(_pivot_stk.index[-1], _bottom_stk[-1] + _bnk_last / 2),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=_colores_banco.get(_bnk, "gray"), va="center",
        )
        _bottom_stk += _vals_stk

    if _otros_vals is not None and _otros_vals.sum() > 0:
        ax_stk.fill_between(
            _pivot_stk.index, _bottom_stk, _bottom_stk + _otros_vals,
            alpha=0.70, color=_colores_banco["Otros"], label="Otros bancos",
        )
        _pct_otros = _otros_vals[-1] / _total_last * 100 if _total_last > 0 else 0
        ax_stk.annotate(
            f"Otros {_pct_otros:.0f}%",
            xy=(_pivot_stk.index[-1], _bottom_stk[-1] + _otros_vals[-1] / 2),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color="#616161", va="center",
        )

    ax_stk.set_ylabel("M USD (total sistema)", fontsize=10)
    ax_stk.set_xlabel("Fecha", fontsize=10)
    ax_stk.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}M"))
    ax_stk.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    _p_cmp = DIR_OUT / "03_comparacion_saldo_bancos.png"
    fig_cmp.savefig(_p_cmp, dpi=150, bbox_inches="tight")
    plt.close(fig_cmp)
    print(f"  Guardado: {_p_cmp.name}")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURES PARA MODELO — EXPORT CON FÓRMULAS EXCEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Features para modelo — export con fórmulas Excel")
print("=" * 65)

# ── Nuevos features (valores del día t, sin lag) ──────────────────────────────
# avance_mes ya existe como "Avance" en df
df["avance_mes"]      = df["Avance"]
df["ratio_ovn_total"] = df["overnight"] / df["encaje_ovn"].replace(0, np.nan)

# Mínimo de encaje diario que el banco debe mantener los días restantes para
# seguir cumpliendo el umbral del 90% al cierre del mes.
# dias_restantes = dias que faltan (sin contar hoy), mínimo 1 para evitar /0.
_encaje_min_por_dia = (
    (df["ExigibleTotalMes_est"] - df["EncajeAcumMes"])
    .clip(lower=0)
    / df["dias_restantes"].clip(lower=1)
)
# exceso_abs: capacidad de retiro total del banco hoy sin incumplir el 100%
# del exigible al cierre del mes.
#   = exceso de encaje computable sobre el mínimo necesario (req. 100%)
#   + overnight completo (no computa encaje → retirable libremente)
_exceso_encaje = (df["encaje"] - _encaje_min_por_dia).clip(lower=0)
df["exceso_abs"] = _exceso_encaje + df["overnight"]

# ── Versiones lag1 (información disponible en t-1 para el modelo) ─────────────
df["avance_mes_lag1"]      = df["avance_mes"].shift(1)
df["exceso_abs_lag1"]      = df["exceso_abs"].shift(1)
df["encaje_ovn_lag1"]      = df["encaje_ovn"].shift(1)
df["ratio_ovn_total_lag1"] = df["ratio_ovn_total"].shift(1)

# ── Construcción del DataFrame de export ──────────────────────────────────────
_COLS_EXPORT = [
    # Series base (origen)
    "fecha", "overnight", "cta_cte", "caja", "tose", "exigible", "retiro_neto",
    # Derivadas intermedias
    "encaje", "encaje_ovn", "var_encaje_ovn",
    "dia_mes", "dias_en_mes",
    "NecAcumMes", "EncajeAcumMes", "ExigibleTotalMes_est",
    # Features sin lag (valor en t — para verificación)
    "avance_mes", "exceso_abs", "ratio_ovn_total",
    # Features con lag1 (para modelo — valor conocido en t-1)
    "avance_mes_lag1", "exceso_abs_lag1", "encaje_ovn_lag1", "ratio_ovn_total_lag1",
]
df_fe = df[[c for c in _COLS_EXPORT if c in df.columns]].copy()

# ── Letras de columna Excel ───────────────────────────────────────────────────
def _col_letter(n):
    s, n = "", n + 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

_cols_fe = [c for c in _COLS_EXPORT if c in df.columns]
_cl = {col: _col_letter(i) for i, col in enumerate(_cols_fe)}

# ── Tabla de fórmulas Excel (fórmula válida en fila 3 = segunda fila de datos) ─
#    Fila 1 = encabezado, Fila 2 = primera fila de datos, Fila 3 = fila general
_C = _cl   # alias corto
_formulas_doc = [
    # (columna,  fórmula Excel fila 3,  descripción)
    ("fecha",
     "— origen",
     "Fecha del dato"),
    ("overnight",
     "— origen",
     "Depósito overnight en BCRP. No computa encaje pero es parte de la posición BCRP."),
    ("cta_cte",
     "— origen",
     "Cuenta corriente en BCRP. Sí computa encaje. Instrumento principal de cumplimiento."),
    ("caja",
     "— origen",
     "Efectivo en bóveda. Sí computa encaje."),
    ("tose",
     "— origen",
     "Tasa de obligaciones sujetas a encaje. Define el nivel de exigencia del mes."),
    ("exigible",
     "— origen",
     "Requerimiento diario de encaje = depósitos × TOSE / días_calendario_mes."),
    ("retiro_neto",
     "— origen",
     "Retiro neto del banco desde BCRP ese día. Negativo = salida de fondos."),
    ("encaje",
     f"={_C['cta_cte']}3+{_C['caja']}3",
     "Encaje computable = cta_cte + caja."),
    ("encaje_ovn",
     f"={_C['overnight']}3+{_C['cta_cte']}3+{_C['caja']}3",
     "Posición total en BCRP = overnight + cta_cte + caja. Techo máximo de retiro posible."),
    ("var_encaje_ovn",
     f"={_C['encaje_ovn']}3-{_C['encaje_ovn']}2",
     "Variación diaria de la posición total BCRP. Negativo = el banco retiró fondos."),
    ("dia_mes",
     f"=DIA({_C['fecha']}3)",
     "Día calendario del mes (1–31). Denominador del promedio diario del exigible."),
    ("dias_en_mes",
     f"=DIA(FIN.MES({_C['fecha']}3,0))",
     "Total días calendario del mes. Multiplicador de la proyección mensual."),
    ("NecAcumMes",
     f"=SI(MES({_C['fecha']}3)=MES({_C['fecha']}2),{_C['NecAcumMes']}2+{_C['exigible']}3,{_C['exigible']}3)",
     "Σ exigible acumulado desde el 1ro del mes (reset en cambio de mes)."),
    ("EncajeAcumMes",
     f"=SI(MES({_C['fecha']}3)=MES({_C['fecha']}2),{_C['EncajeAcumMes']}2+{_C['encaje']}3,{_C['encaje']}3)",
     "Σ encaje acumulado desde el 1ro del mes (reset en cambio de mes)."),
    ("ExigibleTotalMes_est",
     f"=({_C['NecAcumMes']}3/{_C['dia_mes']}3)*{_C['dias_en_mes']}3",
     "Proyección del total mensual: (NecAcumMes / día_calendario) × días_en_mes. "
     "NOTA: válido si el dato tiene filas para días calendario. "
     "Si solo hay días hábiles, reemplazar dia_mes por CONTAR.SI del mes hasta esa fila."),
    ("avance_mes",
     f"={_C['EncajeAcumMes']}3/{_C['ExigibleTotalMes_est']}3",
     "Avance = EncajeAcumMes / ExigibleTotalMes_est. "
     "0→1: fracción del requerimiento mensual estimado ya cubierta. >1 = cumplió."),
    ("exceso_abs",
     f"=MAX(0,{_C['encaje']}3-MAX(0,({_C['ExigibleTotalMes_est']}3-{_C['EncajeAcumMes']}3)/MAX(1,{_C['dias_en_mes']}3-{_C['dia_mes']}3)))+{_C['overnight']}3",
     "Capacidad de retiro total del banco hoy sin incumplir el 100% del exigible al cierre. "
     "= MAX(0, encaje - encaje_min_por_dia) + overnight. "
     "encaje_min_por_dia = MAX(0, (ExigibleTotalMes_est - EncajeAcumMes) / dias_restantes). "
     "El overnight se suma íntegro: no computa encaje y es libremente retirable."),
    ("ratio_ovn_total",
     f"={_C['overnight']}3/{_C['encaje_ovn']}3",
     "overnight / encaje_ovn. Alto = banco aún no inició retiro (fondos en overnight). "
     "Si cae respecto al día anterior, el retiro ya comenzó."),
    ("avance_mes_lag1",
     f"={_C['avance_mes']}2",
     "avance_mes del día anterior. Feature principal del modelo."),
    ("exceso_abs_lag1",
     f"={_C['exceso_abs']}2",
     "exceso_abs del día anterior. Monto retiable conocido en t-1."),
    ("encaje_ovn_lag1",
     f"={_C['encaje_ovn']}2",
     "Posición total BCRP del día anterior. Techo del retiro posible."),
    ("ratio_ovn_total_lag1",
     f"={_C['ratio_ovn_total']}2",
     "ratio_ovn_total del día anterior. Indica si el retiro ya inició."),
]

df_formulas = pd.DataFrame(
    _formulas_doc,
    columns=["columna", "formula_excel_fila3", "descripcion"]
)

# ── Export ─────────────────────────────────────────────────────────────────────
ruta_fe = DIR_OUT / "bbva_encaje_features_modelo.xlsx"
with pd.ExcelWriter(ruta_fe, engine="openpyxl") as _wr:
    df_fe.to_excel(_wr, sheet_name="Datos", index=False)
    df_formulas.to_excel(_wr, sheet_name="Formulas", index=False)

print(f"\n  Exportado: {ruta_fe.name}")
print(f"  Hoja 'Datos'    : {len(df_fe):,} filas × {len(df_fe.columns)} columnas")
print(f"  Hoja 'Formulas' : {len(df_formulas)} columnas documentadas")
print(f"\n  Columnas exportadas:")
for _g, _cols in [
    ("Base (origen)",          ["fecha","overnight","cta_cte","caja","tose","exigible","retiro_neto"]),
    ("Intermedias",            ["encaje","encaje_ovn","var_encaje_ovn","dia_mes","dias_en_mes","NecAcumMes","EncajeAcumMes","ExigibleTotalMes_est"]),
    ("Features sin lag (t)",   ["avance_mes","exceso_abs","ratio_ovn_total"]),
    ("Features lag1 (modelo)", ["avance_mes_lag1","exceso_abs_lag1","encaje_ovn_lag1","ratio_ovn_total_lag1"]),
]:
    print(f"    {_g}: {', '.join(_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST DE PODER PREDICTIVO — correlación con retiro_neto en h=1..75 días hábiles
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Test poder predictivo — features encaje vs retiro_neto(t+h)")
print("=" * 65)

# Filtrar a días hábiles (retiro_neto solo es significativo en estos días)
_is_bday_fe = (
    (df["fecha"].dt.weekday < 5) &
    (~df["fecha"].isin(_hols_all))
)
_db = df[_is_bday_fe & (df["fecha"] >= "2019-01-01")][
    ["fecha", "retiro_neto",
     "avance_mes_lag1", "exceso_abs_lag1",
     "encaje_ovn_lag1", "ratio_ovn_total_lag1",
     "dias_restantes"]
].copy().reset_index(drop=True)

# exceso_dia_lag1: exceso disponible dividido entre días que quedan.
# Aunque el banco puede retirar todo en un día, este ratio captura
# la "presión temporal": a fin de mes con exceso alto, el banco ya
# no puede postergar — el ratio es grande precisamente cuando la
# probabilidad de retirar es mayor. Se prueba aquí antes de incorporar.
_db["exceso_dia_lag1"] = (
    _db["exceso_abs_lag1"] / _db["dias_restantes"].clip(lower=1)
)

_FEAT_TEST  = ["avance_mes_lag1", "exceso_abs_lag1", "exceso_dia_lag1",
               "encaje_ovn_lag1", "ratio_ovn_total_lag1"]
_LABELS     = ["avance_mes", "exceso_abs", "exceso_dia", "encaje_ovn", "ratio_ovn"]
# h=1 excluido: el retiro del día siguiente ya se conoce en t
_HORIZONTES = [2, 3, 5, 10, 15, 22, 30, 45, 60, 75]

# ── Correlaciones Pearson y Spearman por horizonte ───────────────────────────
_corr_p = pd.DataFrame(np.nan, index=_FEAT_TEST, columns=_HORIZONTES)
_corr_s = pd.DataFrame(np.nan, index=_FEAT_TEST, columns=_HORIZONTES)

for _h in _HORIZONTES:
    _ret_h = _db["retiro_neto"].shift(-_h)   # retiro_neto h días hábiles adelante
    for _f in _FEAT_TEST:
        _mask = _db[_f].notna() & _ret_h.notna()
        if _mask.sum() > 50:
            _corr_p.loc[_f, _h] = _db.loc[_mask, _f].corr(_ret_h[_mask])
            _corr_s.loc[_f, _h] = _db.loc[_mask, _f].corr(
                _ret_h[_mask], method="spearman"
            )

print(f"\n  Pearson r  (feature_lag1 vs retiro_neto a h días hábiles):")
print(f"  {'Feature':<22}", end="")
for _h in _HORIZONTES:
    print(f"  h={_h:>2}", end="")
print()
for _f, _lbl in zip(_FEAT_TEST, _LABELS):
    print(f"  {_lbl:<22}", end="")
    for _h in _HORIZONTES:
        _v = _corr_p.loc[_f, _h]
        print(f"  {_v:+.3f}" if np.isfinite(_v) else "    n/a", end="")
    print()

print(f"\n  Spearman ρ (feature_lag1 vs retiro_neto a h días hábiles):")
print(f"  {'Feature':<22}", end="")
for _h in _HORIZONTES:
    print(f"  h={_h:>2}", end="")
print()
for _f, _lbl in zip(_FEAT_TEST, _LABELS):
    print(f"  {_lbl:<22}", end="")
    for _h in _HORIZONTES:
        _v = _corr_s.loc[_f, _h]
        print(f"  {_v:+.3f}" if np.isfinite(_v) else "    n/a", end="")
    print()

# ── Análisis por quintil: avance_mes y exceso_dia (comparación) ───────────────
def _print_quintiles(feat_col, label):
    print(f"\n  Retiro promedio por quintil de {label}:")
    _mask_q = _db[feat_col].notna()
    _q = pd.qcut(_db.loc[_mask_q, feat_col],
                 q=5, labels=["Q1 (<20%)", "Q2", "Q3", "Q4", "Q5 (>80%)"])
    _qt = (
        _db.loc[_mask_q].assign(_q=_q)
        .groupby("_q", observed=True)["retiro_neto"]
        .agg(n="count", mean=lambda x: x.mean() / 1e6, median=lambda x: x.median() / 1e6)
    )
    print(f"  {'Quintil':<12}  {'N':>6}  {'Media (M)':>10}  {'Mediana (M)':>12}")
    for _qi, _row in _qt.iterrows():
        print(f"  {str(_qi):<12}  {int(_row['n']):>6}  "
              f"{_row['mean']:>+10.1f}  {_row['median']:>+12.1f}")

_print_quintiles("avance_mes_lag1",  "avance_mes_lag1")
_print_quintiles("exceso_dia_lag1",  "exceso_dia_lag1  [PRUEBA — exceso/dias_rest]")
_print_quintiles("exceso_abs_lag1",  "exceso_abs_lag1  [referencia]")

# ── Heatmap correlación Pearson ───────────────────────────────────────────────
_corr_plot = _corr_p.copy().astype(float)
_vabs = _corr_plot.abs().max().max()
_vabs = max(_vabs, 0.05)

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
fig.suptitle(
    "Poder predictivo features encaje BBVA vs retiro_neto(t+h)  [2019–actualidad]\n"
    "feature_lag1: información disponible en t-1 | h en días hábiles | h=1 excluido (ya conocido)",
    fontsize=11, fontweight="bold",
)

# Panel izquierdo: heatmap
ax = axes[0]
_im = ax.imshow(_corr_plot.values, aspect="auto",
                cmap="RdBu_r", vmin=-_vabs, vmax=_vabs)
ax.set_xticks(range(len(_HORIZONTES)))
ax.set_xticklabels([f"h={_h}" for _h in _HORIZONTES], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(_FEAT_TEST)))
ax.set_yticklabels(_LABELS, fontsize=9)
for _i, _f in enumerate(_FEAT_TEST):
    for _j, _h in enumerate(_HORIZONTES):
        _v = _corr_plot.loc[_f, _h]
        if np.isfinite(_v):
            ax.text(_j, _i, f"{_v:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(_v) > _vabs * 0.6 else "black")
plt.colorbar(_im, ax=ax, label="Pearson r")
ax.set_title("Pearson r — correlación lineal", fontsize=10)

# Panel derecho: líneas por feature
ax = axes[1]
_colores_f = ["#1565C0", "#E53935", "#43A047", "#F57C00"]
for _f, _lbl, _col in zip(_FEAT_TEST, _LABELS, _colores_f):
    _vals = [_corr_p.loc[_f, _h] for _h in _HORIZONTES]
    ax.plot(_HORIZONTES, _vals, marker="o", ms=4, lw=1.5,
            color=_col, label=_lbl)
ax.axhline(0, color="black", lw=0.6, ls=":")
ax.axhline( 0.1, color="gray", lw=0.5, ls="--", alpha=0.5)
ax.axhline(-0.1, color="gray", lw=0.5, ls="--", alpha=0.5)
ax.set_xlabel("Horizonte h (días hábiles)")
ax.set_ylabel("Pearson r")
ax.set_title("Decaimiento de la correlación por horizonte", fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(-_vabs * 1.3, _vabs * 1.3)

plt.tight_layout()
_ruta_test = DIR_OUT / "test_poder_predictivo_features.png"
fig.savefig(_ruta_test, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\n  Gráfico guardado: {_ruta_test.name}")

