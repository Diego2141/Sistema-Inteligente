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
    sistema = df_banc.groupby("fecha")[["R", "D"]].sum()
    sistema["neto"] = sistema["D"] - sistema["R"]

    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes3[0].plot(sistema.index, sistema["D"] / 1e6, lw=0.7, color="steelblue", alpha=0.9)
    axes3[0].set_title("Depósitos — Sistema Total (millones USD)", fontweight="bold")
    axes3[0].set_ylabel("Millones USD")
    axes3[0].set_ylim(bottom=0)
    axes3[0].grid(True, alpha=0.3)

    axes3[1].plot(sistema.index, sistema["R"] / 1e6, lw=0.7, color="tomato", alpha=0.9)
    axes3[1].set_title("Retiros — Sistema Total (millones USD)", fontweight="bold")
    axes3[1].set_ylabel("Millones USD")
    axes3[1].set_ylim(bottom=0)
    axes3[1].grid(True, alpha=0.3)

    axes3[2].plot(sistema.index, sistema["neto"] / 1e6, lw=0.7, color="darkorange", alpha=0.9)
    axes3[2].axhline(0, color="black", lw=0.8)
    axes3[2].set_title("Flujo Neto D−R — Sistema Total (millones USD)", fontweight="bold")
    axes3[2].set_ylabel("Millones USD")
    axes3[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes3[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes3[2].tick_params(axis="x", rotation=45)
    axes3[2].grid(True, alpha=0.3)

    plt.suptitle("Flujos Sistema Financiero — Datos Reales", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\flujos_sistema.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()

    # ── Verificación: misma serie como línea en millones USD
    # (equivalente visual al panel naranja de aux_validacion_flujos.py)
    fig3b, ax3b = plt.subplots(figsize=(14, 4))
    neto_mm = sistema["neto"] / 1e6  # convertir a millones USD
    ax3b.plot(neto_mm.index, neto_mm.values, lw=0.7, color="darkorange", alpha=0.9,
              label="Flujo neto D−R (Transacciones)")
    ax3b.axhline(0, color="black", lw=0.6)
    ax3b.set_title(
        "Flujo Neto D−R como línea (millones USD)\n"
        "→ Debe coincidir visualmente con la línea naranja de aux_validacion_flujos.py",
        fontweight="bold"
    )
    ax3b.set_ylabel("Millones USD")
    ax3b.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3b.tick_params(axis="x", rotation=45)
    ax3b.grid(True, alpha=0.3)
    ax3b.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(
        r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\flujo_neto_linea.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"\nFlujo neto — estadísticas (millones USD):")
    print(f"  Media  : {neto_mm.mean():.1f}  |  Std: {neto_mm.std():.1f}")
    print(f"  Mín    : {neto_mm.min():.1f}  |  Máx: {neto_mm.max():.1f}")
    print(f"  Obs    : {len(neto_mm):,}  |  Periodo: {neto_mm.index.min().date()} → {neto_mm.index.max().date()}")

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
    "dias_desde_cierre_mes", "is_quincena", "is_cierre_encaje",
    "is_fiestas_patrias", "is_fin_anio", "is_pre_feriado",
    "is_post_feriado", "is_pre_eleccion", "is_post_eleccion",
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
# BLOQUE 5: Zoom en ventanas electorales 2021
# Elecciones siempre en domingo → is_eleccion nunca activa en días hábiles
# is_pre/post_eleccion captura los 7 días calendario alrededor
# ─────────────────────────────────────────────────────────────────
elecciones_2021 = [
    ("1ra vuelta", pd.Timestamp("2021-04-11")),
    ("2da vuelta", pd.Timestamp("2021-06-06")),
]

fig5, axes5 = plt.subplots(2, 1, figsize=(14, 8))

for ax, (vuelta, elec_day) in zip(axes5, elecciones_2021):
    ventana_ini = elec_day - pd.Timedelta(days=14)
    ventana_fin = elec_day + pd.Timedelta(days=14)
    mask = (df_cal.index >= ventana_ini) & (df_cal.index <= ventana_fin)
    sub  = df_cal[mask]

    dias_hab = sub.index
    x = range(len(dias_hab))
    etiquetas = [f"{d.strftime('%a %d-%b')}" for d in dias_hab]

    # Barras pre (naranja) y post (verde)
    pre_vals  = sub["is_pre_eleccion"].values  if "is_pre_eleccion"  in sub.columns else [0]*len(sub)
    post_vals = sub["is_post_eleccion"].values if "is_post_eleccion" in sub.columns else [0]*len(sub)

    ax.bar(x, pre_vals,  color="darkorange", alpha=0.8, label="is_pre_eleccion",  width=0.4, align="edge")
    ax.bar([xi + 0.4 for xi in x], post_vals, color="steelblue", alpha=0.8,
           label="is_post_eleccion", width=0.4, align="edge")

    # Línea vertical en domingo de elección (no es día hábil, cae entre barras)
    # Calcular posición interpolada del domingo dentro del eje x
    dias_antes = sum(1 for d in dias_hab if d < elec_day)
    ax.axvline(x=dias_antes - 0.2, color="red", lw=2, ls="--",
               label=f"Domingo elección\n({elec_day.strftime('%d %b')})")

    # Anotar el domingo
    ax.annotate(f"ELECCIÓN\n{elec_day.strftime('%d-%b')}\n(domingo)",
                xy=(dias_antes - 0.2, 0.5), xytext=(dias_antes + 0.5, 0.85),
                fontsize=8, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

    ax.set_xticks(list(x))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 (inactivo)", "1 (activo)"])
    ax.set_title(f"Ventana electoral — {vuelta} ({elec_day.strftime('%d %b %Y')})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

plt.suptitle(
    "Features Electorales 2021 — is_pre/post_eleccion (ventana ±14 días calendario)\n"
    "is_eleccion siempre 0: las elecciones peruanas son en domingo (día no hábil)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\features_electorales_2021.png",
    dpi=150, bbox_inches="tight"
)
plt.show()
