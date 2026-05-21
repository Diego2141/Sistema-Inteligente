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
# BLOQUE 3: Flujos del sistema completo (D, R, neto)
# ─────────────────────────────────────────────────────────────────
if not df_banc.empty:
    fechas_con_data = set(df_banc.groupby("fecha").groups.keys())
    sistema = df_banc.groupby("fecha")[["R", "D"]].sum()

    cal_completo = pd.bdate_range(
        start=sistema.index.min(), end=sistema.index.max(), freq=peru_bday
    )
    sistema = sistema.reindex(cal_completo, fill_value=0)
    sistema.index.name = "fecha"
    sistema["neto"] = sistema["D"] - sistema["R"]

    d_mm    =  sistema["D"] / 1e6
    r_mm    = -sistema["R"] / 1e6
    neto_mm =  sistema["neto"] / 1e6

    fig3, ax = plt.subplots(figsize=(14, 6))

    ax.bar(sistema.index, d_mm,    color="steelblue", width=1, alpha=0.8, zorder=1, label="Depósitos")
    ax.bar(sistema.index, r_mm,    color="tomato",    width=1, alpha=0.8, zorder=1, label="Retiros")
    ax.plot(sistema.index, neto_mm, color="black",    lw=0.8, alpha=0.9,  zorder=2, label="Neto")
    ax.axhline(0, color="black", lw=0.6, zorder=3)

    ax.set_ylabel("MM USD (neto diario)")
    ax.set_title("Flujos Sistema Financiero — Depósitos / Retiros / Neto (millones USD)",
                 fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, loc="upper left",
              handles=[
                  plt.Line2D([0],[0], color="black", lw=1.5, label="Neto"),
                  plt.Rectangle((0,0),1,1, fc="steelblue", alpha=0.8, label="Depósitos"),
                  plt.Rectangle((0,0),1,1, fc="tomato",    alpha=0.8, label="Retiros"),
              ])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\flujos_sistema.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()

    # Exportar flujos diarios a Excel
    ruta_flujos_clean = (
        r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente"
        r"\1. Data\Clean\flujos_diarios.xlsx"
    )
    flujos_export = pd.DataFrame({
        "depositos_musd":  sistema["D"] / 1e6,
        "retiros_musd":    sistema["R"] / 1e6,
        "neto_musd":       sistema["neto"] / 1e6,
    })
    flujos_export.index.name = "fecha"
    flujos_export.to_excel(ruta_flujos_clean)
    print(f"\nFlujos diarios exportados: {ruta_flujos_clean}")
    print(f"  Filas  : {len(flujos_export):,}")
    print(f"  Periodo: {flujos_export.index.min().date()} → {flujos_export.index.max().date()}")

# ─────────────────────────────────────────────────────────────────
# BLOQUE 3b: Flujos sistema — zoom últimos 2 años
# ─────────────────────────────────────────────────────────────────
if not df_banc.empty:
    fecha_fin_3b   = sistema.index.max()
    fecha_ini_3b   = fecha_fin_3b - pd.DateOffset(years=2)
    mask_2y        = (sistema.index >= fecha_ini_3b) & (sistema.index <= fecha_fin_3b)
    sis_2y         = sistema[mask_2y]

    d_2y    =  sis_2y["D"] / 1e6
    r_2y    = -sis_2y["R"] / 1e6
    neto_2y =  sis_2y["neto"] / 1e6

    fig3b, ax3b = plt.subplots(figsize=(14, 6))

    ax3b.bar(sis_2y.index, d_2y,    color="steelblue", width=1, alpha=0.8, zorder=1)
    ax3b.bar(sis_2y.index, r_2y,    color="tomato",    width=1, alpha=0.8, zorder=1)
    ax3b.plot(sis_2y.index, neto_2y, color="black",    lw=0.9, alpha=0.9, zorder=2)
    ax3b.axhline(0, color="black", lw=0.6, zorder=3)

    ax3b.set_ylabel("MM USD (neto diario)")
    ax3b.set_title(
        f"Flujos Sistema Financiero — Zoom Últimos 2 Años "
        f"({fecha_ini_3b.strftime('%b %Y')} – {fecha_fin_3b.strftime('%b %Y')})",
        fontweight="bold"
    )
    ax3b.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3b.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax3b.tick_params(axis="x", rotation=0)
    ax3b.legend(
        fontsize=8, loc="upper left",
        handles=[
            plt.Line2D([0], [0], color="black", lw=1.5, label="Neto"),
            plt.Rectangle((0, 0), 1, 1, fc="steelblue", alpha=0.8, label="Depositos"),
            plt.Rectangle((0, 0), 1, 1, fc="tomato",    alpha=0.8, label="Retiros"),
        ]
    )
    ax3b.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\flujos_sistema_2y.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()

# ─────────────────────────────────────────────────────────────────
# BLOQUE 4: Features de calendario — año 2021
# (28-jul=miércoles hábil → is_fiestas_patrias; 11-abr y 6-jun → elecciones)
# ─────────────────────────────────────────────────────────────────
peru_bday2 = CustomBusinessDay(holidays=peru_holidays)
fechas_cal = pd.date_range("2021-01-01", "2021-12-31", freq=peru_bday2)
df_cal = bfm._build_seasonal_table(
    fechas_cal, peru_holidays, fechas_igv, fechas_elecciones, peru_bday2
)

# Diagnóstico: verificar si las fechas electorales de 2021 son feriados
fechas_elec_2021 = [pd.Timestamp("2021-04-11"), pd.Timestamp("2021-06-06")]
for fe in fechas_elec_2021:
    en_feriados = fe in peru_holidays
    en_cal      = fe in fechas_cal
    print(f"  {fe.date()} — en peru_holidays: {en_feriados} | en fechas_cal: {en_cal} "
          f"| is_eleccion en df_cal: "
          f"{int(df_cal.loc[fe, 'is_eleccion']) if fe in df_cal.index else 'NO está en índice'}")

cols_cal = [
    "dia_semana", "mes", "pos_en_mes", "dias_al_cierre_mes",
    "dias_desde_cierre_mes", "total_bdays_mes",
    "is_quincena", "is_cierre_encaje",
    "is_penult_bday_trim", "is_ultimo_bday_trim",
    "is_1er_bday_trim", "is_2do_bday_trim", "is_3er_bday_trim",
    "is_fiestas_patrias", "is_fin_anio",
    "is_pre_feriado", "is_post_feriado",
    "is_igv", "dias_al_igv",
    "is_pre_eleccion", "is_post_eleccion",
]

n_cols_cal = len(cols_cal)
n_rows_cal = (n_cols_cal + 2) // 3
fig4, axes4 = plt.subplots(n_rows_cal, 3, figsize=(16, 4 * n_rows_cal))
axes4 = axes4.flatten()

for i, col in enumerate(cols_cal):
    ax = axes4[i]
    if col in df_cal.columns:
        ax.plot(df_cal.index, df_cal[col], lw=0.8, color="steelblue")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)
        if df_cal[col].max() <= 1 and df_cal[col].min() >= 0:
            ax.set_ylim(-0.1, 1.3)
    else:
        ax.text(0.5, 0.5, f"{col}\n(no disponible)", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)
        ax.set_title(col, fontsize=9)

for j in range(n_cols_cal, len(axes4)):
    axes4[j].set_visible(False)

plt.suptitle("Features de Calendario — Año 2021", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(
    r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\features_calendario_2021.png",
    dpi=150, bbox_inches="tight"
)
plt.show()

# ─────────────────────────────────────────────────────────────────
# BLOQUE 5: Flujos sistema 2021 con marcadores electorales
# ─────────────────────────────────────────────────────────────────
if not df_banc.empty:
    elec_1ra = pd.Timestamp("2021-04-11")
    elec_2da = pd.Timestamp("2021-06-06")

    mask21 = (sistema.index >= "2021-01-01") & (sistema.index <= "2021-12-31")
    sis21  = sistema[mask21]

    d21    =  sis21["D"] / 1e6
    r21    = -sis21["R"] / 1e6
    neto21 =  sis21["neto"] / 1e6

    fig5, ax5 = plt.subplots(figsize=(14, 6))

    ax5.bar(sis21.index, d21,    color="steelblue", width=1, alpha=0.8, label="Depositos")
    ax5.bar(sis21.index, r21,    color="tomato",    width=1, alpha=0.8, label="Retiros")
    ax5.plot(sis21.index, neto21, color="black",    lw=1.0, alpha=0.9,  label="Neto")
    ax5.axhline(0, color="black", lw=0.6)

    ax5.axvspan(elec_1ra, elec_2da, color="crimson", alpha=0.07)
    ax5.axvline(elec_1ra, color="crimson", lw=1.8, ls="--",
                label="1ra Vuelta (11-abr)")
    ax5.axvline(elec_2da, color="purple",  lw=1.8, ls="--",
                label="2da Vuelta (6-jun)")

    ymin = min(r21.min(), neto21.min()) * 1.15
    ax5.text(elec_1ra, ymin, "1ra Vuelta\n(11-abr)",
             rotation=90, va="bottom", ha="right", fontsize=8, color="crimson")
    ax5.text(elec_2da, ymin, "2da Vuelta\n(6-jun)",
             rotation=90, va="bottom", ha="right", fontsize=8, color="purple")

    ax5.set_ylabel("MM USD (neto diario)")
    ax5.set_title("Transferencias Exterior - 2021", fontweight="bold", fontsize=12)
    ax5.xaxis.set_major_locator(mdates.MonthLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax5.tick_params(axis="x", rotation=0)
    ax5.legend(fontsize=8, loc="upper right")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\flujos_2021_electoral.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
