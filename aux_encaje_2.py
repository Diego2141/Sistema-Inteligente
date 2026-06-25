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

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESTRATEGIA SOBREENCAJE BBVA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Estrategia sobreencaje BBVA")
print("=" * 65)

# Fracción de la mediana del saldo mensual que define el umbral (adaptable)
UMBRAL_SALDO        = 0.50
# % mínimo del retiro 15d que debe concentrarse en los últimos 5d hábiles
UMBRAL_CONCENTRACION = 0.90

df["_am"] = df["fecha"].dt.to_period("M")


def _met_estrat(g):
    g    = g.sort_values("fecha")
    late = g.tail(5)    # últimos 5 días hábiles del mes
    late15 = g.tail(15) # últimos 15 días hábiles del mes

    retiro_acum_5d  = late["retiro_neto"].sum()  if len(late)   else np.nan
    retiro_acum_15d = late15["retiro_neto"].sum() if len(late15) else np.nan
    mediana_saldo   = g["encaje_ovn"].median()
    umbral          = UMBRAL_SALDO * mediana_saldo

    estrategia = (np.isfinite(retiro_acum_5d) and np.isfinite(umbral)
                  and retiro_acum_5d < -umbral)

    # Concentración: fracción del retiro 15d que ocurre en los últimos 5d
    # Solo aplica cuando ambos acumulados son salidas netas (< 0)
    if (np.isfinite(retiro_acum_5d) and np.isfinite(retiro_acum_15d)
            and retiro_acum_15d < 0 and retiro_acum_5d < 0):
        concentracion = retiro_acum_5d / retiro_acum_15d
    else:
        concentracion = np.nan

    concentrada = np.isfinite(concentracion) and concentracion >= UMBRAL_CONCENTRACION

    return pd.Series({
        "retiro_acum_5d_M":  round(retiro_acum_5d  / 1e6, 1) if np.isfinite(retiro_acum_5d)  else np.nan,
        "retiro_acum_15d_M": round(retiro_acum_15d / 1e6, 1) if np.isfinite(retiro_acum_15d) else np.nan,
        "mediana_saldo_M":   round(mediana_saldo   / 1e6, 1) if np.isfinite(mediana_saldo)   else np.nan,
        "umbral_M":          round(umbral           / 1e6, 1) if np.isfinite(umbral)          else np.nan,
        "estrategia":        estrategia,
        "concentracion":     round(concentracion, 4) if np.isfinite(concentracion) else np.nan,
        "concentrada":       concentrada,
    })


print(f"\n  Umbral estrategia:    retiro 5d > {UMBRAL_SALDO*100:.0f}% mediana saldo mensual")
print(f"  Umbral concentración: ≥ {UMBRAL_CONCENTRACION*100:.0f}% del retiro 15d ocurre en últimos 5d")
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
print(f"  {'Mes':<10} {'Ret5d (M)':>10} {'Ret15d (M)':>11} {'Med.saldo (M)':>14} "
      f"{'Umbral':>8} {'Estrat':>7} {'Conc%':>6} {'Concent':>8}")
print(f"  {'-'*10} {'-'*10} {'-'*11} {'-'*14} {'-'*8} {'-'*7} {'-'*6} {'-'*8}")
for _, _r in dm.nsmallest(10, "retiro_acum_5d_M").iterrows():
    _conc_pct = f"{_r['concentracion']*100:.0f}%" if np.isfinite(_r["concentracion"]) else "n/a"
    _est_lbl  = "SÍ" if _r["estrategia"] else "NO"
    _con_lbl  = "SÍ" if _r["concentrada"] else "NO"
    print(f"  {str(_r['_am']):<10} {_r['retiro_acum_5d_M']:>10,.1f} "
          f"{_r['retiro_acum_15d_M']:>11,.1f} {_r['mediana_saldo_M']:>14,.1f} "
          f"{_r['umbral_M']:>8,.1f} {_est_lbl:>7} {_conc_pct:>6} {_con_lbl:>8}")

# ── Heatmap año × mes (intensidad = retiro / mediana saldo) ───────────────────
_MESES_ABR = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
              "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Intensidad normalizada: retiro / umbral  (> 1 = estrategia activada)
dm["_intensidad"] = (-dm["retiro_acum_5d_M"] / dm["umbral_M"]).clip(lower=0)
# Porcentaje: retiro / mediana saldo del mes
dm["_pct"] = (-dm["retiro_acum_5d_M"] / dm["mediana_saldo_M"].replace(0, np.nan) * 100).clip(lower=0)

piv_int  = dm.pivot(index="anio", columns="mes", values="_intensidad").sort_index()
piv_est  = dm.pivot(index="anio", columns="mes", values="estrategia").sort_index()
piv_ret  = dm.pivot(index="anio", columns="mes", values="retiro_acum_5d_M").sort_index()
piv_pct  = dm.pivot(index="anio", columns="mes", values="_pct").sort_index()
piv_conc = dm.pivot(index="anio", columns="mes", values="concentracion").sort_index()
piv_cond = dm.pivot(index="anio", columns="mes", values="concentrada").sort_index()

_vals = piv_int.values.astype(float)
_vmax = float(np.nanmax(_vals)) if np.any(np.isfinite(_vals)) else 2.0
_cmap = plt.cm.YlOrRd

fig, ax = plt.subplots(figsize=(14, max(4, len(piv_int) * 0.6 + 2)))
fig.suptitle(
    f"Estrategia sobreencaje BBVA — Intensidad del retiro por año y mes\n"
    f"Intensidad = retiro 5d / ({UMBRAL_SALDO*100:.0f}% × mediana saldo)  ·  "
    "> 1 = estrategia activada (borde azul)  ·  Celda: retiro M USD / % mediana saldo",
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
        _vc = piv_conc.loc[_anio, _mes] if _mes in piv_conc.columns else np.nan
        _vk = piv_cond.loc[_anio, _mes] if _mes in piv_cond.columns else False
        _fi = float(_vi) if _vi is not None else np.nan
        _fr = float(_vr) if _vr is not None else np.nan
        _fp = float(_vp) if _vp is not None else np.nan
        _fc = float(_vc) if (_vc is not None and _vc is not pd.NA) else np.nan
        if not np.isfinite(_fi):
            continue
        _ctxt = "white" if _fi > _vmax * 0.55 else "black"
        _conc_str = f"\nc:{_fc*100:.0f}%" if np.isfinite(_fc) else ""
        if np.isfinite(_fr) and np.isfinite(_fp):
            _lbl = f"{_fr:,.0f}\n({_fp:.0f}%){_conc_str}"
        elif np.isfinite(_fr):
            _lbl = f"{_fr:,.0f}{_conc_str}"
        else:
            _lbl = ""
        ax.text(_j, _i, _lbl, ha="center", va="center",
                fontsize=5.3, color=_ctxt, linespacing=1.25,
                fontweight="bold" if _ve else "normal")
        if _ve:
            ax.add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), 1, 1,
                         fill=False, edgecolor="#1565C0", linewidth=2.2))
        if _vk:
            ax.add_patch(plt.Rectangle((_j - 0.38, _i - 0.38), 0.76, 0.76,
                         fill=False, edgecolor="#2E7D32", linewidth=1.5,
                         linestyle="--"))

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
    plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#2E7D32", lw=1.5,
                  linestyle="--",
                  label=f"Retiro concentrado (≥{UMBRAL_CONCENTRACION*100:.0f}% del acum. 15d en últimos 5d)"),
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

dm_s["_int_prev"]  = (-dm_s["retiro_acum_5d_M"] / (UMBRAL_SALDO * dm_s["_saldo_prev_M"].replace(0, np.nan))).clip(lower=0)
dm_s["_pct_prev"]  = (-dm_s["retiro_acum_5d_M"] / dm_s["_saldo_prev_M"].replace(0, np.nan) * 100).clip(lower=0)

piv_int2 = dm_s.pivot(index="anio", columns="mes", values="_int_prev").sort_index()
piv_pct2 = dm_s.pivot(index="anio", columns="mes", values="_pct_prev").sort_index()

_vals2 = piv_int2.values.astype(float)
_vmax2 = float(np.nanmax(_vals2)) if np.any(np.isfinite(_vals2)) else 2.0

fig2, ax2 = plt.subplots(figsize=(14, max(4, len(piv_int2) * 0.6 + 2)))
fig2.suptitle(
    f"Estrategia sobreencaje BBVA — Intensidad usando saldo del mes PREVIO\n"
    f"Intensidad = retiro 5d / ({UMBRAL_SALDO*100:.0f}% × mediana saldo mes anterior)  ·  "
    "> 1 = estrategia activada (borde azul)  ·  Celda: retiro M USD / % saldo previo",
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
        _vc = piv_conc.loc[_anio, _mes] if _mes in piv_conc.columns else np.nan
        _vk = piv_cond.loc[_anio, _mes] if _mes in piv_cond.columns else False
        _fi = float(_vi) if _vi is not None else np.nan
        _fr = float(_vr) if _vr is not None else np.nan
        _fp = float(_vp) if _vp is not None else np.nan
        _fc = float(_vc) if (_vc is not None and _vc is not pd.NA) else np.nan
        if not np.isfinite(_fi):
            continue
        _ctxt = "white" if _fi > _vmax2 * 0.55 else "black"
        _conc_str = f"\nc:{_fc*100:.0f}%" if np.isfinite(_fc) else ""
        if np.isfinite(_fr) and np.isfinite(_fp):
            _lbl = f"{_fr:,.0f}\n({_fp:.0f}%){_conc_str}"
        elif np.isfinite(_fr):
            _lbl = f"{_fr:,.0f}{_conc_str}"
        else:
            _lbl = ""
        ax2.text(_j, _i, _lbl, ha="center", va="center",
                 fontsize=5.3, color=_ctxt, linespacing=1.25,
                 fontweight="bold" if _ve else "normal")
        if _ve:
            ax2.add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), 1, 1,
                          fill=False, edgecolor="#1565C0", linewidth=2.2))
        if _vk:
            ax2.add_patch(plt.Rectangle((_j - 0.38, _i - 0.38), 0.76, 0.76,
                          fill=False, edgecolor="#2E7D32", linewidth=1.5,
                          linestyle="--"))

plt.colorbar(im2, ax=ax2, label=f"Retiro / ({UMBRAL_SALDO*100:.0f}% × mediana saldo previo)",
             shrink=0.75)
ax2.set_xlabel("Mes")
ax2.set_ylabel("Año")
ax2.legend(handles=[
    _Patch(facecolor=_cmap(0.4), label="Retiro moderado"),
    _Patch(facecolor=_cmap(0.85), label="Retiro intenso"),
    plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#1565C0", lw=2,
                  label=f"Estrategia activada (retiro > {UMBRAL_SALDO*100:.0f}% saldo previo)"),
    plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#2E7D32", lw=1.5,
                  linestyle="--",
                  label=f"Retiro concentrado (≥{UMBRAL_CONCENTRACION*100:.0f}% del acum. 15d en últimos 5d)"),
], loc="lower right", fontsize=7, framealpha=0.9)
plt.tight_layout()
_p1b = DIR_OUT / "00b_heatmap_estrategia_saldo_prev.png"
fig2.savefig(_p1b, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Guardado: {_p1b.name}")

# ── Timeline: retiro acumulado 5d vs umbral ────────────────────────────────────
_dm_c = dm.dropna(subset=["retiro_acum_5d_M"]).copy()
_cb   = _dm_c["estrategia"].map({True: "#E53935", False: "#90A4AE"})

fig, ax = plt.subplots(figsize=(15, 5))
fig.suptitle("Retiro acumulado últimos 5 días hábiles vs umbral mensual",
             fontweight="bold")
ax.bar(_dm_c["fecha_plot"], _dm_c["retiro_acum_5d_M"], width=22,
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
             "retiro_acum_5d_M", "retiro_acum_15d_M",
             "mediana_saldo_M", "umbral_M",
             "concentracion", "concentrada"]]
         .rename(columns={"_am": "mes_periodo",
                          "concentracion": "conc_5d_vs_15d"})
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
print(f"  Umbral: retiro 5d > {UMBRAL_SALDO*100:.0f}% de la mediana del saldo mensual")
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
df.drop(columns=["_am"], inplace=True)
dm.drop(columns=["_intensidad", "_pct"], inplace=True, errors="ignore")
dm_s.drop(columns=["_saldo_prev_M", "_int_prev", "_pct_prev"], inplace=True, errors="ignore")
