# -*- coding: utf-8 -*-
"""
aux_fanchart_horizontes.py
Fan chart del modelo para múltiples fechas de origen.

Lógica:
  - Identifica todas las fechas del TEST donde h=90 tiene target realizado.
  - Filtra cada N días hábiles (PASO_FECHAS) para no generar demasiados gráficos.
  - Por cada fecha de origen genera un PNG con 3 subplots:
      1. Flujo diario D-R: bandas + realizado
      2. Flujo diario D-R: solo modelo (mediana visible)
      3. Flujo neto acumulado D-R: bandas + realizado acumulado
  - Guarda todos los PNGs en DIR_OUTPUT.

Parámetros ajustables:
  PASO_FECHAS : cada cuántos días hábiles tomar una fecha de origen (5 = semanal)
  N_FECHAS_MAX: límite de gráficos a generar (None = todos)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data"   / "Clean"  / "matriz_features.parquet"
DIR_MODELOS  = BASE_SISTEMA / "2. Output" / "modelos"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_fanchart_horizontes"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO      = "SISTEMA"
CORTE_VAL  = pd.Timestamp("2022-07-01")   # alineado con step003
CORTE_TEST = pd.Timestamp("2023-01-03")   # inicio de datos de tasas (allocation)

# ── Parámetros del loop ───────────────────────────────────────────────────────
PASO_FECHAS  = 2    # cada 2 días hábiles; usar 1 para todas las fechas
N_FECHAS_MAX = None # None = generar todos; ej: 6 para las 6 primeras fechas válidas


# ── 1. Cargar modelo más reciente ────────────────────────────────────────────
def cargar_modelos(banco: str, dir_modelos: Path) -> tuple[dict, list[str]]:
    metas = sorted(dir_modelos.glob(f"metadata_{banco}_*.json"), reverse=True)
    if not metas:
        raise FileNotFoundError(f"No se encontró metadata para banco={banco}")
    meta  = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha = metas[0].stem.split("_")[-1]
    cols_feat = meta["features"]
    quantiles = meta["quantiles"]

    modelos = {}
    for tau in quantiles:
        ruta = dir_modelos / f"lgbm_{banco}_q{int(tau*100):02d}_{fecha}.txt"
        if not ruta.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
        modelos[tau] = lgb.Booster(model_file=str(ruta))

    print(f"Modelo cargado : {metas[0].name}")
    print(f"  Quantiles    : {quantiles}")
    print(f"  Features     : {len(cols_feat)}")
    return modelos, cols_feat


# ── 2. Leer datos y preparar medianas de imputación ─────────────────────────
def leer_datos(banco: str, cols_feat: list[str]):
    """Carga el parquet, hace el split y calcula medianas de TRAIN.
    Retorna: df completo, medianas de TRAIN, lista de fechas válidas de origen."""
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    df_train = df[df["fecha_t"] < CORTE_VAL].copy()
    df_test  = df[df["fecha_t"] >= CORTE_TEST].copy()

    cols_excluir = {"fecha_t", "banco", "target"}
    cols_num = [c for c in df.columns if c not in cols_excluir]
    medianas = df_train[cols_num].median()

    print(f"\nTRAIN : hasta {CORTE_VAL.date()}")
    print(f"TEST  : {CORTE_TEST.date()} → "
          f"{df_test['fecha_t'].max().date()} ({df_test['fecha_t'].nunique()} fechas)")

    # Fechas del TEST donde h=90 tiene target realizado
    fechas_validas = np.sort(
        df_test[(df_test["h"] == 90) & df_test["target"].notna()]["fecha_t"].unique()
    )

    if len(fechas_validas) == 0:
        raise ValueError("Ninguna fecha del TEST tiene h=90 realizado. "
                         "Aumenta SEMANAS_TEST o espera más datos.")

    print(f"Fechas válidas de origen (h=90 realizado): "
          f"{pd.Timestamp(fechas_validas[0]).date()} → "
          f"{pd.Timestamp(fechas_validas[-1]).date()} "
          f"({len(fechas_validas)} fechas)")

    return df, medianas, cols_num, fechas_validas, cols_feat


def preparar_fecha(df, fecha_origen: pd.Timestamp, medianas, cols_num, cols_feat) -> pd.DataFrame:
    """Extrae y prepara las filas de una fecha de origen específica."""
    df_fecha = df[df["fecha_t"] == fecha_origen].copy().sort_values("h")
    cols_num_ok = [c for c in cols_num if c in df_fecha.columns]
    df_fecha[cols_num_ok] = df_fecha[cols_num_ok].fillna(medianas)

    faltantes = set(cols_feat) - set(df_fecha.columns)
    if faltantes:
        for c in faltantes:
            df_fecha[c] = 0.0
    return df_fecha


# ── 3. Predecir ──────────────────────────────────────────────────────────────
def predecir(modelos: dict, df_fecha: pd.DataFrame, cols_feat: list[str]) -> pd.DataFrame:
    cols_ok = [c for c in cols_feat if c in df_fecha.columns]
    X = df_fecha[cols_ok].copy()

    resultado = df_fecha[["h", "target"]].copy().reset_index(drop=True)
    for tau, model in modelos.items():
        resultado[f"q{int(tau*100):02d}"] = model.predict(X)

    q_cols = sorted([c for c in resultado.columns if c.startswith("q")])
    resultado[q_cols] = np.sort(resultado[q_cols].values, axis=1)
    return resultado


# ── 4. Graficar ───────────────────────────────────────────────────────────────
def _dibujar_bandas(ax, hs, resultado):
    ax.fill_between(hs,
                    resultado["q01"] / 1e6, resultado["q99"] / 1e6,
                    alpha=0.12, color="steelblue", label="Q01–Q99 (98%)")
    ax.fill_between(hs,
                    resultado["q05"] / 1e6, resultado["q95"] / 1e6,
                    alpha=0.28, color="steelblue", label="Q05–Q95 (90%)")
    ax.plot(hs, resultado["q05"] / 1e6,
            color="steelblue", lw=1.0, ls=":", alpha=0.7, label="Q05 / Q95 (borde)")
    ax.plot(hs, resultado["q95"] / 1e6,
            color="steelblue", lw=1.0, ls=":", alpha=0.7)
    ax.plot(hs, resultado["q50"] / 1e6,
            color="crimson", lw=2.0, label="Mediana predicha (Q50)", zorder=5)
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_xlim(hs.min() - 1, hs.max() + 1)


def graficar(resultado: pd.DataFrame, fecha_origen: pd.Timestamp, banco: str,
             idx: int, total: int):
    hs        = resultado["h"].values
    realizado = resultado["target"].values / 1e6
    mask_real = ~np.isnan(realizado)
    h_max_real = int(resultado.loc[mask_real, "h"].max()) if mask_real.any() else 0

    titulo_base = (
        f"Fan Chart — {banco}  |  Fecha de origen: {fecha_origen.strftime('%d %b %Y')}  |  "
        f"h = 1 … 90 días hábiles\n"
        f"Realizado disponible: h = 1 … {h_max_real}  |  "
        f"Proyección pura: h = {h_max_real + 1} … 90"
    )

    cum_q01  = np.cumsum(resultado["q01"].values / 1e6)
    cum_q05  = np.cumsum(resultado["q05"].values / 1e6)
    cum_q50  = np.cumsum(resultado["q50"].values / 1e6)
    cum_q95  = np.cumsum(resultado["q95"].values / 1e6)
    cum_q99  = np.cumsum(resultado["q99"].values / 1e6)
    cum_real = np.where(mask_real, np.nancumsum(np.where(mask_real, realizado, 0)), np.nan)
    cum_real[~mask_real] = np.nan

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16), sharex=True,
                                        gridspec_kw={"hspace": 0.08})

    # ── Subplot 1: flujo diario — bandas + realizado ──────────────────────
    _dibujar_bandas(ax1, hs, resultado)
    if mask_real.any():
        ax1.plot(hs[mask_real], realizado[mask_real],
                 color="black", lw=2, label="Realizado (D−R)", zorder=5)
        ax1.scatter(hs[mask_real], realizado[mask_real],
                    color="black", s=18, zorder=6)
    if h_max_real > 0:
        ax1.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                    label=f"Último dato realizado (h={h_max_real})")
    ax1.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
    ax1.set_title(titulo_base, fontweight="bold", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ── Subplot 2: flujo diario — solo modelo ────────────────────────────
    _dibujar_bandas(ax2, hs, resultado)
    if h_max_real > 0:
        ax2.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                    label=f"Último dato realizado (h={h_max_real})")
    ax2.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
    ax2.set_title("Solo proyección del modelo (sin realizado) — mediana visible",
                  fontsize=10, style="italic")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ── Subplot 3: flujo neto ACUMULADO ──────────────────────────────────
    ax3.fill_between(hs, cum_q01, cum_q99,
                     alpha=0.12, color="steelblue", label="Q01–Q99 acum. (98%)")
    ax3.fill_between(hs, cum_q05, cum_q95,
                     alpha=0.28, color="steelblue", label="Q05–Q95 acum. (90%)")
    ax3.plot(hs, cum_q05, color="steelblue", lw=1.0, ls=":", alpha=0.7)
    ax3.plot(hs, cum_q95, color="steelblue", lw=1.0, ls=":", alpha=0.7)
    ax3.plot(hs, cum_q50,
             color="crimson", lw=2.0, label="Mediana acumulada (Q50)", zorder=5)
    if mask_real.any():
        hs_real = hs[mask_real]
        ax3.plot(hs_real, cum_real[mask_real],
                 color="black", lw=2, label="Realizado acumulado (D−R)", zorder=6)
        ax3.scatter(hs_real, cum_real[mask_real],
                    color="black", s=18, zorder=7)
    if h_max_real > 0:
        ax3.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                    label=f"Último dato realizado (h={h_max_real})")
    ax3.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
    ax3.set_xlabel("Horizonte h (días hábiles desde t)", fontsize=11)
    ax3.set_ylabel("Flujo neto acumulado (MM USD)", fontsize=11)
    ax3.set_title("Flujo neto acumulado D−R desde la fecha de origen",
                  fontsize=10, style="italic")
    ax3.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax3.set_xlim(hs.min() - 1, hs.max() + 1)

    nombre = f"fanchart_{banco}_{fecha_origen.strftime('%Y%m%d')}_h1_h90.png"
    plt.savefig(DIR_OUTPUT / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)  # cierra para no acumular figuras en memoria
    print(f"  [{idx}/{total}] Guardado: {nombre}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    modelos, cols_feat = cargar_modelos(BANCO, DIR_MODELOS)
    df, medianas, cols_num, fechas_validas, cols_feat = leer_datos(BANCO, cols_feat)

    # Seleccionar fechas cada PASO_FECHAS días hábiles
    fechas_selec = fechas_validas[::PASO_FECHAS]
    if N_FECHAS_MAX is not None:
        fechas_selec = fechas_selec[:N_FECHAS_MAX]

    print(f"\nGenerando {len(fechas_selec)} gráfico(s) "
          f"(paso={PASO_FECHAS} días hábiles, límite={N_FECHAS_MAX}):")
    for i, f_orig in enumerate(fechas_selec, 1):
        fecha_origen = pd.Timestamp(f_orig)
        df_fecha  = preparar_fecha(df, fecha_origen, medianas, cols_num, cols_feat)
        resultado = predecir(modelos, df_fecha, cols_feat)
        graficar(resultado, fecha_origen, BANCO, idx=i, total=len(fechas_selec))

    print(f"\n✓ Todos los gráficos guardados en: {DIR_OUTPUT}")
