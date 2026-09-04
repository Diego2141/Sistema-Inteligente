# -*- coding: utf-8 -*-
"""
aux_plot_ffd_features.py
Compara series originales vs. FFD (Fixed-Width Fracdiff, López de Prado Cap. 5)
para EMBI_PERU, T10Y, CDS_PERU_5Y y VIX.

Panel por serie (4 filas × 3 columnas):
  Col 1 — Serie original (nivel, I(1))
  Col 2 — FFD con d óptimo (estacionaria, preserva memoria)
  Col 3 — Primera diferencia Δ1 (referencia: d=1.0, pierde toda la memoria)

Ejecutar después de step001 (requiere parquet de features).
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE        = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_PARQ   = BASE / "1. Data" / "Clean" / "features_SISTEMA.parquet"
RUTA_PNG    = BASE / "3. Files" / "ffd_features_comparison.png"
CORTE_TRAIN = "2010-12-31"   # mismo que _FD_TRAIN_CUTOFF en step001

SERIES = {
    "EMBI_PERU":   "EMBI Perú (pb)",
    "T10Y":        "UST 10Y (%)",
    "CDS_PERU_5Y": "CDS Perú 5Y (pb)",
    "VIX":         "VIX",
}

COLOR_ORIG = "#2166ac"
COLOR_FFD  = "#1a9850"
COLOR_D1   = "#d95f02"
COLOR_VLINE= "#e31a1c"

# ─────────────────────────────────────────────────────────────────────────────
# FFD helpers (auto-contenidos, sin dependencia de step001)
# ─────────────────────────────────────────────────────────────────────────────
_THRES = 1e-4
_RANGO = np.arange(0.0, 1.05, 0.05)


def _pesos(d, n):
    w = [1.0]
    for k in range(1, n):
        w.append(w[-1] * (k - 1 - d) / k)
    w = np.array(w[::-1])
    return w[np.abs(w) > _THRES]


def _ffd(serie, d):
    s = serie.dropna()
    p = _pesos(d, len(s))
    L = len(p)
    return pd.Series(
        {s.index[i]: float(np.dot(p, s.iloc[i - L + 1: i + 1]))
         for i in range(L - 1, len(s))},
        name=serie.name,
    )


def _calibrar_d(serie, cutoff):
    from statsmodels.tsa.stattools import adfuller
    s = serie.dropna()
    s_tr = s[s.index <= cutoff]
    if len(s_tr) < 50:
        return 1.0
    for d in _RANGO:
        try:
            sd = _ffd(s_tr, d) if d > 0 else s_tr
            if len(sd) < 20:
                continue
            _, p, *_ = adfuller(sd)
            if p < 0.05:
                return float(d)
        except Exception:
            continue
    return 1.0


def _adf_pvalue(serie):
    from statsmodels.tsa.stattools import adfuller
    try:
        return adfuller(serie.dropna())[1]
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────────────────────
def _cargar_series():
    if not RUTA_PARQ.exists():
        raise FileNotFoundError(
            f"Parquet no encontrado: {RUTA_PARQ}\n"
            "  Ejecuta step001_build_feature_matrix.py primero."
        )
    df = pd.read_parquet(RUTA_PARQ)
    ts = (df.drop_duplicates("fecha_t")
            .set_index("fecha_t")
            .sort_index())
    return ts


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_ax(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def graficar(ts):
    n_series = sum(1 for s in SERIES if s in ts.columns)
    if n_series == 0:
        raise ValueError("Ninguna de las series FFD encontrada en el parquet.")

    fig, axes = plt.subplots(n_series, 3, figsize=(15, 3.2 * n_series),
                             sharex=False)
    if n_series == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Diferenciación Fraccional — López de Prado Cap. 5\n"
        f"d calibrado en TRAIN (datos ≤ {CORTE_TRAIN})   |   "
        "Col 1: Original  |  Col 2: FFD  |  Col 3: Δ1 (referencia)",
        fontsize=10, fontweight="bold",
    )

    row = 0
    for nombre, etiqueta in SERIES.items():
        if nombre not in ts.columns:
            continue

        serie = ts[nombre].dropna()
        d_opt = _calibrar_d(serie, CORTE_TRAIN)
        ffd   = _ffd(serie, d_opt)
        d1    = serie.diff(1).dropna()

        p_orig = _adf_pvalue(serie)
        p_ffd  = _adf_pvalue(ffd)
        p_d1   = _adf_pvalue(d1)

        corte_ts = pd.Timestamp(CORTE_TRAIN)

        # ── Col 1: Original ──────────────────────────────────────────────────
        ax0 = axes[row, 0]
        ax0.plot(serie.index, serie.values, color=COLOR_ORIG, lw=0.8)
        ax0.axvline(corte_ts, color=COLOR_VLINE, ls="--", lw=0.8, alpha=0.7,
                    label=f"Corte TRAIN ({CORTE_TRAIN[:4]})")
        ax0.set_title(f"{etiqueta}  —  Original\nADF p={p_orig:.3f}  {'✗ raíz unitaria' if p_orig >= 0.05 else '✓ estacionaria'}",
                      fontsize=8)
        ax0.legend(fontsize=6, loc="upper left")
        _fmt_ax(ax0)

        # ── Col 2: FFD ───────────────────────────────────────────────────────
        ax1 = axes[row, 1]
        ax1.plot(ffd.index, ffd.values, color=COLOR_FFD, lw=0.8)
        ax1.axvline(corte_ts, color=COLOR_VLINE, ls="--", lw=0.8, alpha=0.7)
        ax1.axhline(0, color="gray", ls=":", lw=0.5)
        ax1.set_title(f"{etiqueta}  —  FFD  (d={d_opt:.2f})\nADF p={p_ffd:.3f}  {'✓ estacionaria' if p_ffd < 0.05 else '✗ no estacionaria'}",
                      fontsize=8)
        _fmt_ax(ax1)

        # ── Col 3: Primera diferencia ─────────────────────────────────────────
        ax2 = axes[row, 2]
        ax2.plot(d1.index, d1.values, color=COLOR_D1, lw=0.6, alpha=0.8)
        ax2.axvline(corte_ts, color=COLOR_VLINE, ls="--", lw=0.8, alpha=0.7)
        ax2.axhline(0, color="gray", ls=":", lw=0.5)
        ax2.set_title(f"{etiqueta}  —  Δ1  (d=1.0, referencia)\nADF p={p_d1:.3f}  (pierde toda la memoria)",
                      fontsize=8)
        _fmt_ax(ax2)

        row += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    RUTA_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(RUTA_PNG, dpi=150, bbox_inches="tight")
    print(f"\nPlot guardado: {RUTA_PNG}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("aux_plot_ffd_features.py")
    print("=" * 65)

    print(f"\nLeyendo parquet: {RUTA_PARQ}")
    ts = _cargar_series()

    print(f"Fechas disponibles: {ts.index.min().date()} → {ts.index.max().date()}")
    print(f"Series a graficar : {[s for s in SERIES if s in ts.columns]}\n")

    print("Calibrando d (puede tardar ~30s)...")
    graficar(ts)
