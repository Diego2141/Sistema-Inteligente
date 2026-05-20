# -*- coding: utf-8 -*-
"""
aux_visualizar_series.py
Visualiza las series base (no transformaciones) y features de calendario
del pipeline de predicción de liquidez en ME.
"""

import sys
sys.path.insert(0, r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")

import importlib
import step001_build_feature_matrix as bfm
importlib.reload(bfm)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import CustomBusinessDay

PARAMS = bfm.PARAMS.copy()

# ─────────────────────────────────────────────────────────────────
# Cargar datos reales
# ─────────────────────────────────────────────────────────────────
peru_bday, peru_holidays, fechas_igv, fechas_elecciones = bfm.build_peru_calendar(
    años=PARAMS["años_calendario"], ruta_igv=PARAMS["ruta_igv"], ruta_elecciones=None
)
datos    = bfm.load_manual_data(PARAMS)
df_macro = bfm.download_external_series(PARAMS)
df_banc  = datos["bancarios"]

# ─────────────────────────────────────────────────────────────────
# BLOQUE 1: Series externas base
# ─────────────────────────────────────────────────────────────────
series_macro = {
    "VIX":           "Índice VIX",
    "TC_PEN_USD":    "Tipo de Cambio USD/PEN",
    "EMBI_PERU":     "EMBI Perú (pbs)",
    "TASA_REF_BCRP": "Tasa Ref. BCRP (%)",
    "FED_FUNDS":     "Fed Funds (%)",
    "T10Y":          "Bono Tesoro 10Y (%)",
}

fig1, axes = plt.subplots(3, 2, figsize=(14, 11))
axes = axes.flatten()
for i, (col, titulo) in enumerate(series_macro.items()):
    ax = axes[i]
    if col in df_macro.columns and df_macro[col].notna().any():
        s = df_macro[col].dropna()
        ax.plot(s.index, s.values, lw=0.8, color="steelblue")
        ax.set_title(titulo, fontweight="bold", fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3)
        ultima = s.index.max()
        ax.annotate(f"Última: {ultima.date()}", xy=(0.98, 0.05),
                    xycoords="axes fraction", ha="right", fontsize=7, color="red")
    else:
        ax.text(0.5, 0.5, f"{col}\n(sin datos)", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(titulo, fontsize=10)

plt.suptitle("Series Externas Base — Datos Reales", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(
    r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\series_base_macro.png",
    dpi=150, bbox_inches="tight"
)
plt.show()

# ─────────────────────────────────────────────────────────────────
# BLOQUE 2: Series bancarias por banco
# ─────────────────────────────────────────────────────────────────
if not df_banc.empty:
    bancos = df_banc["banco"].unique()
    n = len(bancos)
    fig2, axes2 = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes2 = axes2.reshape(1, -1)

    for i, banco in enumerate(bancos):
        sub = df_banc[df_banc["banco"] == banco].set_index("fecha")
        ax_r = axes2[i, 0]
        ax_d = axes2[i, 1]

        ax_r.bar(sub.index, sub["R"], color="tomato", width=1, alpha=0.7)
        ax_r.set_title(f"{banco} — Retiros (USD)", fontsize=9, fontweight="bold")
        ax_r.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_r.tick_params(axis="x", rotation=45, labelsize=7)
        ax_r.grid(True, alpha=0.3)

        ax_d.bar(sub.index, sub["D"], color="steelblue", width=1, alpha=0.7)
        ax_d.set_title(f"{banco} — Depósitos (USD)", fontsize=9, fontweight="bold")
        ax_d.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_d.tick_params(axis="x", rotation=45, labelsize=7)
        ax_d.grid(True, alpha=0.3)

    plt.suptitle("Retiros y Depósitos por Banco — Datos Reales", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\series_bancarias.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()

    print("\nResumen por banco (USD):")
    resumen = (
        df_banc.groupby("banco")[["R", "D"]]
        .sum()
        .assign(neto=lambda x: x["D"] - x["R"])
    )
    print(resumen.to_string())
else:
    print("Sin datos bancarios cargados.")

# ─────────────────────────────────────────────────────────────────
# BLOQUE 3: Features de calendario — año 2024
# ─────────────────────────────────────────────────────────────────
peru_bday2 = CustomBusinessDay(holidays=peru_holidays)
fechas_cal = pd.date_range("2024-01-01", "2024-12-31", freq=peru_bday2)
df_cal = bfm._build_seasonal_table(
    fechas_cal, peru_holidays, fechas_igv, fechas_elecciones, peru_bday2
)

cols_cal = [
    "dia_semana", "mes", "pos_en_mes", "dias_al_cierre_mes",
    "dias_desde_cierre_mes", "is_quincena", "is_cierre_encaje",
    "is_fiestas_patrias", "is_fin_anio", "is_pre_feriado",
    "is_post_feriado", "is_eleccion",
]

fig3, axes3 = plt.subplots(4, 3, figsize=(16, 12))
axes3 = axes3.flatten()

for i, col in enumerate(cols_cal):
    ax = axes3[i]
    if col in df_cal.columns:
        ax.plot(df_cal.index, df_cal[col], lw=0.8, color="steelblue")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f"{col}\n(no disponible)", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)
        ax.set_title(col, fontsize=9)

plt.suptitle("Features de Calendario — Año 2024", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(
    r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\features_calendario.png",
    dpi=150, bbox_inches="tight"
)
plt.show()
