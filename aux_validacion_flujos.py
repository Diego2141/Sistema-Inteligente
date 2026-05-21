# -*- coding: utf-8 -*-
"""
aux_validacion_flujos.py
Validación cruzada entre Transacciones_BancaLocal.xlsx y DepositosSF.xlsx.

Hipótesis: la variación diaria de Depósitos del SF en el BCRP (PD04651MD)
debe correlacionar con el flujo neto D-R de las transacciones de banca local.

Resultado esperado: correlación > 0.85
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_validacion_flujos"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

RUTA_TRANSACCIONES = BASE_SISTEMA / "1. Data" / "Raw" / "Transacciones_BancaLocal.xlsx"
RUTA_DEPOSITOS_SF  = BASE_SISTEMA / "1. Data" / "Raw" / "DepositosSF.xlsx"
RUTA_OUTPUT_PNG    = DIR_OUTPUT / "validacion_flujos.png"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar DepositosSF (hoja "Diarias", formato fecha: 03Ene00)
# ─────────────────────────────────────────────────────────────────────────────
MESES_ES = {
    "Ene": "Jan", "Feb": "Feb", "Mar": "Mar", "Abr": "Apr",
    "May": "May", "Jun": "Jun", "Jul": "Jul", "Ago": "Aug",
    "Set": "Sep", "Oct": "Oct", "Nov": "Nov", "Dic": "Dec",
}

def _parse_fecha_bcrp(s):
    s = str(s).strip()
    for es, en in MESES_ES.items():
        s = s.replace(es, en)
    return pd.to_datetime(s, format="%d%b%y", errors="coerce")


df_sf = pd.read_excel(RUTA_DEPOSITOS_SF, sheet_name="Diarias", header=None)
df_sf.columns = ["fecha", "DepSF_Total", "DepBancaEmp", "DepBanNacion", "DepOtros"]
df_sf = df_sf.iloc[1:].copy()  # quitar fila de descripciones

df_sf["fecha"] = df_sf["fecha"].apply(_parse_fecha_bcrp)
for c in ["DepSF_Total", "DepBancaEmp", "DepBanNacion", "DepOtros"]:
    df_sf[c] = pd.to_numeric(df_sf[c], errors="coerce")

df_sf = df_sf.dropna(subset=["fecha"]).set_index("fecha").sort_index()

# Forward-fill valores "n.d." antes de calcular variación diaria
df_sf["DepSF_Total"] = df_sf["DepSF_Total"].ffill()
df_sf["delta_DepSF"] = df_sf["DepSF_Total"].diff()

print(f"DepositosSF:    {len(df_sf):,} obs | "
      f"{df_sf.index.min().date()} → {df_sf.index.max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Cargar Transacciones y calcular flujo neto diario
# ─────────────────────────────────────────────────────────────────────────────
df_tx = pd.read_excel(RUTA_TRANSACCIONES)
df_tx["Fecha Valor"] = pd.to_datetime(df_tx["Fecha Valor"])
df_tx["monto"]       = pd.to_numeric(df_tx["Delivery Principal Usd"], errors="coerce")
df_tx["R"]           = df_tx["monto"].clip(upper=0).abs()   # negativos → retiros
df_tx["D"]           = df_tx["monto"].clip(lower=0)          # positivos → depósitos

flujo = df_tx.groupby("Fecha Valor")[["R", "D"]].sum()
flujo["neto"] = flujo["D"] - flujo["R"]
flujo.index.name = "fecha"

print(f"Transacciones:  {len(flujo):,} días | "
      f"{flujo.index.min().date()} → {flujo.index.max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Unir series y calcular correlación
# ─────────────────────────────────────────────────────────────────────────────
df_comp = flujo[["neto", "R", "D"]].join(
    df_sf[["DepSF_Total", "delta_DepSF"]], how="inner"
).dropna()

corr  = df_comp["neto"].corr(df_comp["delta_DepSF"])
slope, intercept = np.polyfit(df_comp["delta_DepSF"], df_comp["neto"] / 1e6, 1)

print(f"\nPeriodo común:  {len(df_comp):,} días | "
      f"{df_comp.index.min().date()} → {df_comp.index.max().date()}")
print(f"Correlación     Δ(DepSF) vs Flujo Neto : {corr:.3f}")
print(f"Regresión       slope={slope:.3f}  intercept={intercept:.2f}")

if corr >= 0.85:
    print("  ✓ Validación OK — correlación por encima del umbral 0.85")
else:
    print("  ✗ Validación ALERTA — correlación por debajo de 0.85, revisar datos")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Gráficos
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 13))

# Panel 1: Nivel DepSF_Total
ax1 = axes[0]
ax1.plot(df_sf.index, df_sf["DepSF_Total"], color="steelblue", lw=0.8)
ax1.set_title(
    "Depósitos del Sistema Financiero en el BCRP — PD04651MD (millones USD)",
    fontweight="bold"
)
ax1.set_ylabel("Millones USD")
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.tick_params(axis="x", rotation=45)
ax1.grid(True, alpha=0.3)

# Panel 2: Δ DepSF vs Flujo neto
ax2 = axes[1]
ax2.plot(df_comp.index, df_comp["delta_DepSF"], color="steelblue",
         lw=0.7, alpha=0.8, label="Δ Dep. SF en BCRP (PD04651MD)")
ax2.plot(df_comp.index, df_comp["neto"] / 1e6, color="darkorange",
         lw=0.7, alpha=0.8, label="Flujo neto D−R (Transacciones)")
ax2.axhline(0, color="black", lw=0.5)
ax2.set_title(
    f"Variación diaria: Δ Depósitos SF vs Flujo Neto  |  Correlación = {corr:.3f}",
    fontweight="bold"
)
ax2.set_ylabel("Millones USD")
ax2.legend(fontsize=9)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.tick_params(axis="x", rotation=45)
ax2.grid(True, alpha=0.3)

# Panel 3: Scatter
ax3 = axes[2]
ax3.scatter(df_comp["delta_DepSF"], df_comp["neto"] / 1e6,
            alpha=0.2, s=5, color="steelblue")
x_line = np.linspace(df_comp["delta_DepSF"].min(), df_comp["delta_DepSF"].max(), 100)
ax3.plot(x_line, slope * x_line + intercept, color="red", lw=1.5,
         label=f"y = {slope:.2f}x + {intercept:.2f}")
ax3.set_xlabel("Δ Depósitos SF BCRP (millones USD)")
ax3.set_ylabel("Flujo Neto D−R Transacciones (millones USD)")
ax3.set_title("Scatter: validación Δ Depósitos SF vs Flujo Neto", fontweight="bold")
ax3.axhline(0, color="black", lw=0.4)
ax3.axvline(0, color="black", lw=0.4)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.suptitle(
    f"Validación cruzada — Transacciones Banca Local vs DepositosSF BCRP\n"
    f"Correlación = {corr:.3f}  |  Slope = {slope:.3f}",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(RUTA_OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nGráfico guardado en: {RUTA_OUTPUT_PNG}")
