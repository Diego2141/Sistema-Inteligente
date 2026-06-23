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

# ── Día de liberación (umbral 90%) ────────────────────────────────────────────
# NecMinDiario_90: encaje mínimo por día para los días restantes y cerrar en ≥90%
#   = max(0, 90%·exigible_total - acumulado) / días_restantes
#   Si ya se acumuló ≥90%, la necesidad es 0: banco completamente libre
UMBRAL_90 = 0.90
df["NecMinDiario_90"] = (
    (UMBRAL_90 * df["ExigibleTotalMes_est"] - df["EncajeAcumMes"]).clip(lower=0)
    / df["dias_restantes"].clip(lower=1)
)

# libre_90: el encaje de hoy supera la necesidad mínima → banco puede hacer retiros
df["libre_90"] = df["encaje"] >= df["NecMinDiario_90"]

# dia_liberacion_90: primer día del mes donde Avance ≥ 90%
#   = banco ya aseguró el umbral aunque encaje = 0 los días restantes
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

# Resumen mensual: día de liberación + retiro_neto en días libres vs no libres
lib_resumen = (
    df.groupby(_am)
    .agg(
        dia_liberacion_90 = ("dia_liberacion_90", "first"),
        retiro_dias_libres    = ("retiro_neto",
                                 lambda g: g[df.loc[g.index, "libre_90"]].mean()),
        retiro_dias_no_libres = ("retiro_neto",
                                 lambda g: g[~df.loc[g.index, "libre_90"]].mean()),
        pct_dias_libres       = ("libre_90", "mean"),
    )
    .reset_index()
)
lib_resumen["fecha"] = lib_resumen["fecha"].dt.to_timestamp()

print(f"\n── Día de liberación (Avance ≥ 90%) ────────────────────────────────")
print(f"  Umbral        : {UMBRAL_90:.0%}")
print(f"  Días libres   : {df['libre_90'].sum():,}  ({df['libre_90'].mean()*100:.0f}% del total)")
print(f"  Día mediano de liberación  : día {lib_resumen['dia_liberacion_90'].median():.0f} del mes")
print(f"  Meses sin liberación       : {lib_resumen['dia_liberacion_90'].isna().sum()}")
print(f"\n  Retiro promedio en días libres   : {df.loc[df['libre_90'], 'retiro_neto'].mean()/1e6:+.0f}M")
print(f"  Retiro promedio en días no libres: {df.loc[~df['libre_90'], 'retiro_neto'].mean()/1e6:+.0f}M")

# ── Gráfico día de liberación ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=False)
fig.suptitle(
    f"Análisis de Liberación — Umbral {UMBRAL_90:.0%}\n"
    "'Libre' = encaje de hoy ≥ encaje mínimo para cerrar el mes en ≥90%",
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
ax.set_title("Día del mes en que Avance acumulado cruza 90% (liberación total)")
ax.legend(fontsize=9)
ax.set_ylim(0, 31)

# Panel 2: NecMinDiario_90 vs encaje real (año representativo)
ax = axes[1]
anio_ej = 2023
mask_ej = df["fecha"].dt.year == anio_ej
df_ej2  = df[mask_ej]
ax.fill_between(df_ej2["fecha"], df_ej2["NecMinDiario_90"] / 1e9,
                alpha=0.25, color="#F44336", label="Mínimo necesario (90%)")
ax.plot(df_ej2["fecha"], df_ej2["NecMinDiario_90"] / 1e9,
        color="#F44336", lw=1.2)
ax.plot(df_ej2["fecha"], df_ej2["encaje"] / 1e9,
        color="#1565C0", lw=1.4, label="Encaje actual")

# Sombrear días libres
for _, row in df_ej2[df_ej2["libre_90"]].iterrows():
    ax.axvspan(row["fecha"], row["fecha"] + pd.Timedelta(days=1),
               alpha=0.10, color="#4CAF50", zorder=0)

from matplotlib.patches import Patch
ax.legend(handles=[
    ax.lines[1], ax.lines[0],
    Patch(color="#4CAF50", alpha=0.4, label="Días libres (retiros posibles)"),
], fontsize=9)
ax.set_ylabel("B USD")
ax.set_title(f"Encaje actual vs Mínimo necesario para ≥90% — Año {anio_ej}")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}B"))

# Panel 3: retiro neto promedio días libres vs no libres por mes
ax = axes[2]
ax.bar(lib_resumen["fecha"],
       lib_resumen["retiro_dias_libres"] / 1e6,
       width=20, color="#4CAF50", alpha=0.8, label="Días libres")
ax.bar(lib_resumen["fecha"],
       lib_resumen["retiro_dias_no_libres"] / 1e6,
       width=20, color="#F44336", alpha=0.6, label="Días no libres",
       bottom=lib_resumen["retiro_dias_libres"].fillna(0) / 1e6)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Retiro neto promedio (M USD)")
ax.set_title("Retiro neto promedio según estado de liberación (por mes)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}M"))

plt.tight_layout()
ruta_lib = DIR_OUT / "dia_liberacion_90.png"
fig.savefig(ruta_lib, dpi=130, bbox_inches="tight")
plt.close()
print(f"\nGráfico liberación guardado: {ruta_lib.name}")

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
