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

RUTA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Raw\bbva_encaje.xlsx")

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
