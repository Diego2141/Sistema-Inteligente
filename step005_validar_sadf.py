# -*- coding: utf-8 -*-
"""
step005_validar_sadf.py
=======================
Genera gráfico de validación del SADF (sadf_60d y sadf_120d) sobre la
serie de flujos netos del SISTEMA, usando la matriz de features ya construida.

No modifica ningún archivo — solo genera el gráfico de validación.

Uso:
    python step005_validar_sadf.py
    runfile('...step005_validar_sadf.py', wdir='...')
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output"

BANCO        = "SISTEMA"
H_REF        = 2          # h mínimo — usamos target(t+2) como proxy de flujo diario

# SADF params
TAU_MIN      = 30         # mínimo de obs para regresión ADF
VENTANAS     = [60, 120]  # días hábiles
LAGS         = 1

# Episodios de stress conocidos para validación visual (ajustar según datos reales)
EPISODIOS = [
    ("2018-09-01", "2018-12-31", "Stress 2018"),
    ("2020-03-01", "2020-06-30", "COVID"),
    ("2022-01-01", "2022-06-30", "Estrés 2022"),
]

# ── SADF (numpy vectorizado) ───────────────────────────────────────────────────

def calcular_sadf(values: np.ndarray, tau_min: int = 30,
                  ventana_max: int = 120, lags: int = 1) -> np.ndarray:
    """
    Para cada punto t calcula el supremo del t-estadístico ADF sobre todas
    las ventanas [t0, t] con t0 en [t - ventana_max, t - tau_min].
    Implementado con numpy.linalg.lstsq (sin statsmodels) para velocidad.
    """
    n    = len(values)
    sadf = np.full(n, np.nan)
    dy   = np.diff(values)            # length n-1

    for t in range(ventana_max, n):
        t_stats = []
        t0_ini  = max(lags + 1, t - ventana_max)
        t0_fin  = t - tau_min + 1

        for t0 in range(t0_ini, t0_fin):
            y_win  = values[t0 : t + 1]   # length = t - t0 + 1
            dy_win = dy[t0 : t]            # length = t - t0
            T      = len(dy_win)
            if T < lags + 3:
                continue

            Y       = dy_win[lags:]
            X_cols  = [np.ones(T - lags), y_win[lags:-1]]
            for l in range(1, lags + 1):
                X_cols.append(dy_win[lags - l : T - l])
            X_mat = np.column_stack(X_cols)

            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
                resid  = Y - X_mat @ coeffs
                s2     = np.sum(resid ** 2) / max(len(Y) - X_mat.shape[1], 1)
                XtXi   = np.linalg.inv(X_mat.T @ X_mat)
                se     = np.sqrt(max(s2 * XtXi[1, 1], 1e-12))
                t_stats.append(coeffs[1] / se)
            except Exception:
                continue

        if t_stats:
            sadf[t] = max(t_stats)

    return sadf


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_flujo_neto(ruta: Path, banco: str, h_ref: int) -> pd.Series:
    """
    Extrae la serie de flujo neto diario:
      - filtra banco y h = h_ref (mínimo horizonte disponible)
      - usa target(fecha_t + h) como proxy del flujo diario
      - indexa por fecha_t + h_ref días hábiles
    """
    print(f"Cargando matriz: {ruta}")
    df = pd.read_parquet(ruta, filters=[("banco", "==", banco),
                                        ("h",     "==", h_ref)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values("fecha_t").reset_index(drop=True)

    # La fecha real del flujo es fecha_t + h días hábiles
    offsets = pd.tseries.offsets.BusinessDay(h_ref)
    df["fecha_flujo"] = df["fecha_t"] + offsets

    serie = df.set_index("fecha_flujo")["target"].dropna()
    print(f"  {banco} | {len(serie):,} obs | {serie.index.min().date()} → "
          f"{serie.index.max().date()}")
    return serie


# ── Gráfico de validación ─────────────────────────────────────────────────────

def graficar_sadf(serie: pd.Series, sadf_dict: dict[str, np.ndarray]) -> None:
    """
    Panel superior : flujo neto diario (área)
    Panel inferior : sadf_60d y sadf_120d con umbral percentil 95
    Bandas sombreadas : episodios de stress conocidos
    """
    fechas = serie.index
    vals   = serie.values

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1]}
    )
    fig.suptitle(
        f"Validación SADF — {BANCO}\n"
        f"(tau_min={TAU_MIN} dh | ventanas {VENTANAS[0]} y {VENTANAS[1]} dh)",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: flujo neto ───────────────────────────────────────────────────
    colores_flujo = np.where(vals >= 0, "#2196F3", "#F44336")
    ax1.bar(fechas, vals, color=colores_flujo, alpha=0.7, width=1.2)
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.set_ylabel("Flujo neto (unidades originales)", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"
    ))
    ax1.set_title("Flujo neto diario (azul=entradas, rojo=salidas)", fontsize=10)

    # ── Panel 2: SADF ─────────────────────────────────────────────────────────
    colores_sadf = ["#E91E63", "#FF9800"]
    for (nombre, sadf_vals), color in zip(sadf_dict.items(), colores_sadf):
        ax2.plot(fechas, sadf_vals, lw=1.2, color=color,
                 label=nombre, alpha=0.85)
        # Umbral percentil 95
        p95 = np.nanpercentile(sadf_vals, 95)
        ax2.axhline(p95, color=color, lw=0.8, ls=":", alpha=0.7,
                    label=f"{nombre} P95 = {p95:.2f}")

    ax2.axhline(0, color="k", lw=0.6, ls="--")
    ax2.set_ylabel("SADF (t-stat supremo)", fontsize=10)
    ax2.set_title("SADF: spikes deben coincidir con episodios de stress conocidos",
                  fontsize=10)
    ax2.legend(fontsize=9, loc="upper left")

    # ── Bandas de episodios conocidos ─────────────────────────────────────────
    for ini, fin, etiqueta in EPISODIOS:
        for ax in (ax1, ax2):
            ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin),
                       alpha=0.12, color="gold", zorder=0)
        # Etiqueta solo en panel superior
        ax1.text(
            pd.Timestamp(ini) + (pd.Timestamp(fin) - pd.Timestamp(ini)) / 2,
            ax1.get_ylim()[1] * 0.88,
            etiqueta, ha="center", fontsize=8, color="#7B6000",
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8)
        )

    plt.tight_layout()

    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    ruta = DIR_OUTPUT / f"validacion_sadf_{BANCO}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"\n  Gráfico guardado: {ruta}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Cargar serie
    serie = cargar_flujo_neto(RUTA_MATRIZ, BANCO, H_REF)

    # 2. Calcular SADF para cada ventana
    vals = serie.values
    sadf_dict = {}
    for ventana in VENTANAS:
        nombre = f"sadf_{ventana}d"
        print(f"\nCalculando {nombre}  ({len(vals):,} puntos × {ventana} ventana)…")
        print("  Esto puede tardar 2-5 minutos — implementado en numpy puro.")
        sadf_vals = calcular_sadf(vals, tau_min=TAU_MIN,
                                  ventana_max=ventana, lags=LAGS)
        sadf_dict[nombre] = sadf_vals
        p95 = np.nanpercentile(sadf_vals, 95)
        p99 = np.nanpercentile(sadf_vals, 99)
        n_sobre_p95 = int(np.sum(sadf_vals > p95))
        print(f"  P95={p95:.2f} | P99={p99:.2f} | días sobre P95: {n_sobre_p95}")

    # 3. Estadísticas por episodio de stress
    print("\n── Porcentaje de días de episodios con SADF > P95 ──────────────────")
    for nombre, sadf_vals in sadf_dict.items():
        p95 = np.nanpercentile(sadf_vals, 95)
        s_sadf = pd.Series(sadf_vals, index=serie.index)
        for ini, fin, etiqueta in EPISODIOS:
            mask = (s_sadf.index >= ini) & (s_sadf.index <= fin)
            sub  = s_sadf[mask]
            if len(sub) > 0:
                pct = (sub > p95).mean() * 100
                print(f"  {nombre} | {etiqueta}: {pct:.1f}% días sobre P95")

    # 4. Gráfico
    graficar_sadf(serie, sadf_dict)


if __name__ == "__main__":
    main()
