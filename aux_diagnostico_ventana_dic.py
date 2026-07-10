# -*- coding: utf-8 -*-
"""
aux_diagnostico_ventana_dic.py
==============================
Evalúa la consistencia de los cuantiles Q05 para la ventana de cierre
de diciembre (últimos 7 días hábiles) a través de distintas fechas de origen.

Hipótesis: la predicción Q05 para los MISMOS días calendario (ej. Dic 27,
Dic 30, Dic 31) debería ser similar independientemente de si el origen es
Sep 15 u Oct 10. Si hay alta varianza, el modelo es ruidoso a horizontes
lejanos y eso explica la inestabilidad del factor f.

Outputs en DIR_OUTPUT:
  - PNG por año: Q05 por día de diciembre según fecha de origen
  - PNG por año: Q05_acum de la ventana según fecha de origen
  - Excel: datos crudos + pivots para análisis detallado
"""

from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pandas.tseries.offsets import CustomBusinessDay

# ── Config ───────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_STEP005  = BASE_SISTEMA / "2. Output" / "step005_wfcv_v3"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "analisis_cc"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO          = "SISTEMA"
VENTANA_DH     = 7    # días hábiles de la ventana de cierre
DIA_INICIO_DIC = 19   # filtrar días de diciembre >= este valor (cubre ventana con feriados)

# Calendario peruano (igual que overlay en step005)
_ANOS = range(2018, 2032)
_FERIADOS_PE = pd.to_datetime(
    [f"{y}-01-01" for y in _ANOS]
    + [f"{y}-12-08" for y in _ANOS]
    + [f"{y}-12-24" for y in _ANOS]
    + [f"{y}-12-25" for y in _ANOS]
)
BDAY_PE = CustomBusinessDay(holidays=_FERIADOS_PE)


# ── 1. Cargar parquet base ────────────────────────────────────────────────────
def cargar_base() -> pd.DataFrame:
    patron = str(DIR_STEP005 / "**" / f"preds_base_{BANCO}_*.parquet")
    archivos = sorted(glob.glob(patron, recursive=True))
    if not archivos:
        raise FileNotFoundError(f"No se encontró: {patron}")
    df = pd.read_parquet(archivos[-1])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    print(f"[OK] Parquet cargado: {Path(archivos[-1]).name}  ({len(df):,} filas)")
    return df


# ── 2. Mapear (fecha_t, h) → fecha_target ─────────────────────────────────────
def mapear_fecha_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada fecha_t única, construye el calendario PE y mapea h → fecha_calendario.
    Solo procesa h >= 40 para limitar el costo (las ventanas de diciembre están ahí).
    """
    df_far = df[df["h"] >= 40].copy()
    partes = []
    fechas_unicas = sorted(df_far["fecha_t"].unique())
    print(f"Mapeando {len(fechas_unicas)} fechas de origen (h>=40)...")

    for fecha_t in fechas_unicas:
        grp = df_far[df_far["fecha_t"] == fecha_t]
        h_max = int(grp["h"].max())
        bh = pd.bdate_range(start=fecha_t + BDAY_PE, periods=h_max, freq=BDAY_PE)
        h_to_date = {i + 1: d for i, d in enumerate(bh)}
        sub = grp.copy()
        sub["fecha_target"] = sub["h"].map(h_to_date)
        partes.append(sub)

    return pd.concat(partes, ignore_index=True)


# ── 3. Filtrar ventana de diciembre ──────────────────────────────────────────
def filtrar_ventana_dic(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["fecha_target"].dt.month == 12)
        & (df["fecha_target"].dt.day >= DIA_INICIO_DIC)
    )
    return df[mask].copy()


# ── 4. Gráfico de consistencia por año ───────────────────────────────────────
def graficar_consistencia(df_vent: pd.DataFrame) -> None:
    df_vent = df_vent.copy()
    df_vent["anio_target"] = df_vent["fecha_target"].dt.year
    df_vent["label_target"] = df_vent["fecha_target"].dt.strftime("%-d %b")  # "26 Dec"

    for anio in sorted(df_vent["anio_target"].unique()):
        sub = df_vent[df_vent["anio_target"] == anio].copy()
        if sub.empty:
            continue

        dias_dic = sorted(sub["fecha_target"].unique())
        colores  = plt.cm.tab10(np.linspace(0, 0.9, len(dias_dic)))

        fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=False)

        # ── Panel 1: Q05 por día de diciembre, a través del tiempo ────────
        ax1 = axes[0]
        for dia, color in zip(dias_dic, colores):
            serie = sub[sub["fecha_target"] == dia].set_index("fecha_t")["q05"].sort_index() / 1e6
            label = pd.Timestamp(dia).strftime("%-d %b")
            ax1.plot(serie.index, serie.values, marker=".", lw=1.4,
                     label=label, color=color)
        ax1.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax1.set_title(
            f"Consistencia Q05 — ventana cierre Dic {anio}  |  SISTEMA\n"
            "Q05 predicho para cada día de diciembre, según fecha de origen",
            fontweight="bold", fontsize=11,
        )
        ax1.set_ylabel("Q05 (MM)", fontsize=10)
        ax1.legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.9)
        ax1.grid(True, alpha=0.25)
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(12))
        fig.autofmt_xdate()

        # ── Panel 2: Q05 acumulado de la ventana ────────────────────────
        ax2 = axes[1]
        suma = (
            sub.groupby("fecha_t")["q05"].sum() / 1e6
        ).sort_index()
        ax2.plot(suma.index, suma.values, marker="o", color="crimson",
                 lw=1.8, label="Q05_acum ventana (suma 7 días hábiles)")
        ax2.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax2.set_title(
            f"Q05 acumulado de la ventana (7 días hábiles Dic {anio}) — por fecha de origen",
            fontsize=10, style="italic",
        )
        ax2.set_ylabel("Q05_acum ventana (MM)", fontsize=10)
        ax2.set_xlabel("Fecha de origen (t)", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.25)
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(12))
        fig.autofmt_xdate()

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        nombre = f"consistencia_ventana_dic{anio}_{BANCO}.png"
        plt.savefig(DIR_OUTPUT / nombre, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Guardado: {nombre}")


# ── 5. Exportar Excel ─────────────────────────────────────────────────────────
def exportar_excel(df_vent: pd.DataFrame) -> None:
    df_vent = df_vent.copy()
    df_vent["anio_target"] = df_vent["fecha_target"].dt.year

    ruta = DIR_OUTPUT / f"consistencia_ventana_dic_{BANCO}.xlsx"
    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:

        # Datos crudos
        cols_out = [c for c in ["fecha_t", "h", "fecha_target", "q01", "q05",
                                  "q50", "q95", "q99", "target"] if c in df_vent.columns]
        df_vent[cols_out].to_excel(writer, sheet_name="Datos_crudos", index=False)

        # Pivot: filas = fecha_t (origen), columnas = fecha_target (diciembre)
        pivot_q05 = df_vent.pivot_table(
            index="fecha_t", columns="fecha_target",
            values="q05", aggfunc="mean",
        ) / 1e6
        pivot_q05.columns = [pd.Timestamp(c).strftime("%d-%b-%Y") for c in pivot_q05.columns]
        pivot_q05.to_excel(writer, sheet_name="Pivot_Q05_por_dia_dic")

        # Q05_acum de la ventana por fecha_t y año
        suma = (
            df_vent.groupby(["anio_target", "fecha_t"])["q05"]
            .sum()
            .reset_index()
            .rename(columns={"q05": "q05_acum"})
        )
        suma["q05_acum_MM"] = suma["q05_acum"] / 1e6
        suma.to_excel(writer, sheet_name="Q05_acum_ventana", index=False)

        # Coeficiente de variación del Q05_acum por año (indicador de ruido)
        cv = (
            suma.groupby("anio_target")["q05_acum_MM"]
            .agg(media="mean", std="std", cv_pct=lambda x: 100 * x.std() / abs(x.mean()))
            .reset_index()
        )
        cv.to_excel(writer, sheet_name="CV_por_anio", index=False)

    print(f"[OK] Excel guardado: {ruta}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n=== Diagnóstico ventana diciembre — {BANCO} ===\n")

    df      = cargar_base()
    df_map  = mapear_fecha_target(df)
    df_vent = filtrar_ventana_dic(df_map)

    n_fechas = df_vent["fecha_t"].nunique()
    print(f"\nFechas de origen con al menos un día en ventana Dic: {n_fechas}")
    print(f"Rango de fechas origen: "
          f"{df_vent['fecha_t'].min().date()} → {df_vent['fecha_t'].max().date()}")
    print(f"Años de diciembre cubiertos: {sorted(df_vent['fecha_target'].dt.year.unique())}")

    print("\nGenerando gráficos de consistencia...")
    graficar_consistencia(df_vent)

    print("\nExportando Excel...")
    exportar_excel(df_vent)

    print("\n[OK] Diagnóstico completo.")
