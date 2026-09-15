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

# ── Validación de magnitud ────────────────────────────────────────────────────
neto_m    = df_comp["neto"] / 1e6          # flujo neto en millones
delta_sf  = df_comp["delta_DepSF"]         # delta DepSF ya en millones

mae       = (neto_m - delta_sf).abs().mean()
rmse      = np.sqrt(((neto_m - delta_sf) ** 2).mean())
std_ratio = neto_m.std() / delta_sf.std()
# Ratio diario: evitar divisiones por cero
mask_nz   = delta_sf.abs() > 1
ratio_ser = (neto_m[mask_nz] / delta_sf[mask_nz])
ratio_med  = ratio_ser.median()
ratio_mean = ratio_ser.mean()
ratio_p25  = ratio_ser.quantile(0.25)
ratio_p75  = ratio_ser.quantile(0.75)

print(f"\n── Validación de magnitud ──────────────────────────────")
print(f"  MAE  (millones USD/día)     : {mae:,.1f}")
print(f"  RMSE (millones USD/día)     : {rmse:,.1f}")
print(f"  Ratio std  (neto/delta_SF)  : {std_ratio:.3f}  (ideal=1.0)")
print(f"  Ratio diario media          : {ratio_mean:.3f}  (ideal=1.0)")
print(f"  Ratio diario mediana        : {ratio_med:.3f}  (ideal=1.0)")
print(f"  Ratio diario P25-P75        : [{ratio_p25:.3f}, {ratio_p75:.3f}]")
if abs(std_ratio - 1) < 0.10:
    print("  ✓ Escala OK — ratio de desviaciones estándar dentro del ±10%")
else:
    print(f"  ✗ Diferencia de escala — ratio std = {std_ratio:.3f}, revisar unidades")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Gráficos
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 21))

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

# Panel 4: Ratio rolling (flujo_neto / delta_DepSF) en el tiempo
ax4 = axes[3]
roll_ratio      = ratio_ser.rolling(22, min_periods=10).median()
roll_ratio_mean = ratio_ser.rolling(22, min_periods=10).mean()
ax4.plot(roll_ratio.index, roll_ratio, color="steelblue", lw=1.0,
         label="Ratio rolling 22d (mediana)")
ax4.plot(roll_ratio_mean.index, roll_ratio_mean, color="darkorange", lw=1.0,
         alpha=0.75, label="Ratio rolling 22d (media)")
ax4.axhline(1.0, color="black", lw=0.8, ls="--", label="Ideal = 1.0")
ax4.axhline(ratio_med, color="steelblue", lw=0.8, ls=":",
            label=f"Mediana global = {ratio_med:.3f}")
ax4.axhline(ratio_mean, color="darkorange", lw=0.8, ls=":",
            label=f"Media global = {ratio_mean:.3f}")
ax4.fill_between(roll_ratio.index, 0.85, 1.15, alpha=0.08,
                 color="seagreen", label="Banda ±15%")
ax4.set_ylim(-1, 3)
ax4.set_title("Ratio de magnitud rolling: Flujo Neto / Δ Depósitos SF\n"
              "(ratio=1 → escala perfecta; ratio>1 → flujo neto mayor que delta DepSF)",
              fontweight="bold")
ax4.set_ylabel("Ratio")
ax4.legend(fontsize=8)
ax4.xaxis.set_major_locator(mdates.YearLocator(2))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax4.tick_params(axis="x", rotation=45)
ax4.grid(True, alpha=0.3)

# Panel 5: Distribución comparada (histograma superpuesto)
ax5 = axes[4]
lim = float(np.percentile(np.abs(neto_m.dropna()), 99))
bins = np.linspace(-lim, lim, 80)
ax5.hist(neto_m.clip(-lim, lim), bins=bins, alpha=0.55,
         color="darkorange", label="Flujo Neto D−R (Transacciones)", density=True)
ax5.hist(delta_sf.clip(-lim, lim), bins=bins, alpha=0.55,
         color="steelblue", label="Δ Depósitos SF BCRP", density=True)
ax5.set_title(f"Distribución comparada de magnitudes\n"
              f"std Neto={neto_m.std():,.0f}  |  std ΔDepSF={delta_sf.std():,.0f}  "
              f"|  Ratio stds={std_ratio:.3f}",
              fontweight="bold")
ax5.set_xlabel("Millones USD")
ax5.set_ylabel("Densidad")
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

plt.suptitle(
    f"Validación cruzada — Transacciones Banca Local vs DepositosSF BCRP\n"
    f"Correlación = {corr:.3f}  |  Slope = {slope:.3f}  |  MAE = {mae:,.0f} MM/día  |  "
    f"Ratio std = {std_ratio:.3f}  |  Ratio media = {ratio_mean:.3f}  |  Ratio mediana = {ratio_med:.3f}",
    fontsize=11, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(RUTA_OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nGráfico guardado en: {RUTA_OUTPUT_PNG}")
