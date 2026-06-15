# -*- coding: utf-8 -*-
"""
step005_validar_sadf.py
=======================
Valida dos detectores de régimen antes de integrarlos en la matriz de features:

  1. SADF sobre garch_vol (volatilidad del flujo neto)
     — detecta explosividad en la volatilidad (stress auto-reforzado)

  2. HMM de 3 estados sobre [flujo_neto, garch_vol]
     — clasifica el régimen oculto: calma / stress moderado / stress severo

Gráfico de 3 paneles:
  Panel 1: Flujo neto diario (azul/rojo)
  Panel 2: SADF sobre volatilidad con umbrales P95/P99
  Panel 3: Régimen HMM coloreado (verde=calma, amarillo=moderado, rojo=severo)

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
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

try:
    from hmmlearn import hmm as hmmlearn_hmm
    _HMM_OK = True
except ImportError:
    _HMM_OK = False
    print("AVISO: hmmlearn no instalado — panel HMM omitido.")

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output"

BANCO        = "SISTEMA"
H_REF        = 2          # h mínimo — usamos target y garch_vol de este h

# SADF params (aplicado sobre garch_vol)
TAU_MIN      = 30
VENTANAS     = [60, 120]
LAGS         = 1

# HMM params
N_ESTADOS    = 3
HMM_INICIO   = "2016-01-01"   # entrenar solo post-cambio estructural

# Episodios de stress conocidos (ajustar fechas según datos reales)
EPISODIOS = [
    ("2018-09-01", "2018-12-31", "Stress 2018"),
    ("2020-03-01", "2020-06-30", "COVID"),
    ("2022-01-01", "2022-06-30", "Estrés 2022"),
]

# ── SADF (numpy vectorizado) ───────────────────────────────────────────────────

def calcular_sadf(values: np.ndarray, tau_min: int = 30,
                  ventana_max: int = 120, lags: int = 1) -> np.ndarray:
    """Supremum ADF sobre todas las subventanas [t0, t] con t0 en [t-ventana_max, t-tau_min]."""
    n    = len(values)
    sadf = np.full(n, np.nan)
    dy   = np.diff(values)

    for t in range(ventana_max, n):
        t_stats = []
        for t0 in range(max(lags + 1, t - ventana_max), t - tau_min + 1):
            y_win  = values[t0 : t + 1]
            dy_win = dy[t0 : t]
            T      = len(dy_win)
            if T < lags + 3:
                continue
            Y      = dy_win[lags:]
            X_cols = [np.ones(T - lags), y_win[lags:-1]]
            for l in range(1, lags + 1):
                X_cols.append(dy_win[lags - l : T - l])
            X_mat  = np.column_stack(X_cols)
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


# ── Carga de datos ─────────────────────────────────────────────────────────────

def cargar_datos(ruta: Path, banco: str, h_ref: int) -> pd.DataFrame:
    """
    Carga flujo neto (target) y garch_vol para un banco y h_ref dados.
    Indexa por fecha_flujo = fecha_t + h_ref días hábiles.
    """
    print(f"Cargando matriz: {ruta}")
    cols = ["fecha_t", "banco", "h", "target", "garch_vol"]
    df   = pd.read_parquet(
        ruta,
        columns=cols,
        filters=[("banco", "==", banco), ("h", "==", h_ref)]
    )
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values("fecha_t").reset_index(drop=True)

    offset = pd.tseries.offsets.BusinessDay(h_ref)
    df["fecha"] = df["fecha_t"] + offset
    df = df.set_index("fecha").sort_index()

    print(f"  {banco} | {len(df):,} obs | "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


# ── HMM ───────────────────────────────────────────────────────────────────────

def entrenar_hmm(df: pd.DataFrame, inicio: str) -> tuple:
    """
    Entrena HMM de 3 estados sobre [flujo_norm, vol_norm] desde `inicio`.
    Devuelve (modelo, estados_toda_serie, sorted_states).
    sorted_states[0] = calma, sorted_states[2] = stress severo.
    """
    from sklearn.preprocessing import StandardScaler

    df_train = df[df.index >= inicio].copy()
    df_train = df_train[["target", "garch_vol"]].dropna()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.values)

    modelo = hmmlearn_hmm.GaussianHMM(
        n_components=N_ESTADOS,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
    )
    modelo.fit(X_train)

    # Ordenar estados por varianza ascendente: 0=calma, 2=stress severo
    varianzas   = [np.linalg.det(modelo.covars_[s]) for s in range(N_ESTADOS)]
    sorted_states = np.argsort(varianzas)   # índices de menor a mayor varianza

    # Predecir sobre toda la serie disponible (no solo desde inicio)
    df_full  = df[["target", "garch_vol"]].dropna()
    X_full   = scaler.transform(df_full.values)
    estados_raw = modelo.predict(X_full)

    # Reasignar etiquetas: 0=calma, 1=moderado, 2=severo
    mapa = {sorted_states[i]: i for i in range(N_ESTADOS)}
    estados = np.array([mapa[e] for e in estados_raw])

    # Estadísticas de la matriz de transición
    trans = modelo.transmat_
    print(f"\n  Matriz de transición HMM (estados ordenados por varianza):")
    labels_orden = [f"E{sorted_states[i]}" for i in range(N_ESTADOS)]
    for i in range(N_ESTADOS):
        fila = "  ".join([f"{trans[sorted_states[i], sorted_states[j]]:.2f}"
                          for j in range(N_ESTADOS)])
        nombre = ["Calma     ", "Moderado  ", "Severo    "][i]
        print(f"    {nombre}: {fila}")

    return modelo, df_full.index, estados, sorted_states


# ── Gráfico 3 paneles ─────────────────────────────────────────────────────────

def graficar(df: pd.DataFrame, sadf_dict: dict,
             hmm_result=None) -> None:

    n_panels = 3 if hmm_result is not None else 2
    h_ratios = [1.8, 1, 1] if n_panels == 3 else [2, 1]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(19, 10 if n_panels == 3 else 8),
        sharex=True, gridspec_kw={"height_ratios": h_ratios}
    )
    if n_panels == 2:
        axes = list(axes) + [None]
    ax1, ax2, ax3 = axes

    fig.suptitle(
        f"Validación SADF (sobre volatilidad) y HMM — {BANCO}\n"
        f"SADF: tau_min={TAU_MIN} dh | ventanas {VENTANAS[0]} y {VENTANAS[1]} dh  |  "
        f"HMM: {N_ESTADOS} estados, entrenado desde {HMM_INICIO}",
        fontsize=12, fontweight="bold"
    )

    fechas = df.index
    flujo  = df["target"].values
    vol    = df["garch_vol"].values

    # ── Panel 1: flujo neto ────────────────────────────────────────────────────
    colores = np.where(flujo >= 0, "#2196F3", "#F44336")
    ax1.bar(fechas, flujo, color=colores, alpha=0.65, width=1.2)
    ax1.axhline(0, color="k", lw=0.7, ls="--")
    ax1.set_ylabel("Flujo neto", fontsize=9)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B"))
    ax1.set_title("Flujo neto diario (azul=entradas, rojo=salidas netas)", fontsize=9)

    # ── Panel 2: SADF sobre volatilidad ───────────────────────────────────────
    colores_s = ["#E91E63", "#FF6F00"]
    for (nombre, svals), color in zip(sadf_dict.items(), colores_s):
        ax2.plot(fechas, svals, lw=1.0, color=color, label=nombre, alpha=0.85)
        p95 = np.nanpercentile(svals, 95)
        p99 = np.nanpercentile(svals, 99)
        ax2.axhline(p95, color=color, lw=0.9, ls=":",
                    label=f"{nombre} P95={p95:.2f}")
        ax2.axhline(p99, color=color, lw=0.9, ls="--", alpha=0.6,
                    label=f"{nombre} P99={p99:.2f}")

    ax2.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax2.set_ylabel("SADF (t-stat sup.)", fontsize=9)
    ax2.set_title(
        "SADF sobre garch_vol — spikes positivos = volatilidad explosiva", fontsize=9)
    ax2.legend(fontsize=8, loc="upper left", ncol=2)

    # ── Panel 3: HMM regímenes ─────────────────────────────────────────────────
    if ax3 is not None and hmm_result is not None:
        idx_hmm, estados = hmm_result
        colores_hmm = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}
        nombres_hmm = {0: "Calma", 1: "Stress moderado", 2: "Stress severo"}

        # Colorear fondo por régimen
        for estado, color in colores_hmm.items():
            mask = estados == estado
            fechas_e = idx_hmm[mask]
            for f in fechas_e:
                ax3.axvspan(f, f + pd.offsets.BusinessDay(1),
                            alpha=0.6, color=color, linewidth=0)

        # Superponer garch_vol normalizada para referencia
        vol_full = df.loc[idx_hmm, "garch_vol"].values
        vol_norm = (vol_full - np.nanmean(vol_full)) / (np.nanstd(vol_full) + 1e-12)
        ax3.plot(idx_hmm, vol_norm, color="k", lw=0.7, alpha=0.6,
                 label="garch_vol (norm.)")

        ax3.set_ylabel("Régimen HMM", fontsize=9)
        ax3.set_title("Régimen HMM: verde=calma  amarillo=moderado  rojo=severo", fontsize=9)

        parches = [mpatches.Patch(color=c, label=nombres_hmm[e])
                   for e, c in colores_hmm.items()]
        ax3.legend(handles=parches, fontsize=8, loc="upper left")

    # ── Bandas de episodios en todos los paneles ───────────────────────────────
    for ini, fin, etiqueta in EPISODIOS:
        for ax in [ax1, ax2] + ([ax3] if ax3 else []):
            ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin),
                       alpha=0.10, color="navy", zorder=0)
        mid = pd.Timestamp(ini) + (pd.Timestamp(fin) - pd.Timestamp(ini)) / 2
        ymax = ax1.get_ylim()[1]
        ax1.text(mid, ymax * 0.88, etiqueta, ha="center", fontsize=8,
                 color="#1A237E",
                 bbox=dict(boxstyle="round,pad=0.2", fc="lavender", alpha=0.8))

    plt.tight_layout()
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    ruta = DIR_OUTPUT / f"validacion_sadf_hmm_{BANCO}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"\n  Gráfico guardado: {ruta}")
    plt.show()


# ── Estadísticas de validación ────────────────────────────────────────────────

def reportar_estadisticas(df, sadf_dict, hmm_result=None):
    print("\n── SADF: % días sobre P95 en episodios de stress ───────────────────")
    for nombre, svals in sadf_dict.items():
        p95 = np.nanpercentile(svals, 95)
        s   = pd.Series(svals, index=df.index)
        for ini, fin, etiqueta in EPISODIOS:
            sub = s[(s.index >= ini) & (s.index <= fin)]
            if len(sub) > 0:
                pct = (sub > p95).mean() * 100
                print(f"  {nombre:<12} | {etiqueta:<15}: {pct:5.1f}% sobre P95")

    if hmm_result is not None:
        idx_hmm, estados = hmm_result
        print("\n── HMM: distribución de días por estado ────────────────────────────")
        nombres = {0: "Calma    ", 1: "Moderado ", 2: "Severo   "}
        total   = len(estados)
        for e in range(N_ESTADOS):
            n = (estados == e).sum()
            print(f"  Estado {e} {nombres[e]}: {n:,} días ({n/total*100:.1f}%)")

        print("\n── HMM: % días en stress severo (estado 2) por episodio ────────────")
        for ini, fin, etiqueta in EPISODIOS:
            mask = (idx_hmm >= ini) & (idx_hmm <= fin)
            sub  = estados[mask]
            if len(sub) > 0:
                pct = (sub == 2).mean() * 100
                print(f"  {etiqueta:<20}: {pct:5.1f}% días en estado severo")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Cargar datos
    df = cargar_datos(RUTA_MATRIZ, BANCO, H_REF)

    # 2. SADF sobre garch_vol
    vol = df["garch_vol"].fillna(method="ffill").values
    sadf_dict = {}
    for ventana in VENTANAS:
        nombre = f"sadf_vol_{ventana}d"
        print(f"\nCalculando {nombre}  ({len(vol):,} puntos, ~2-5 min)…")
        sadf_dict[nombre] = calcular_sadf(vol, tau_min=TAU_MIN,
                                          ventana_max=ventana, lags=LAGS)
        p95 = np.nanpercentile(sadf_dict[nombre], 95)
        p99 = np.nanpercentile(sadf_dict[nombre], 99)
        print(f"  P95={p95:.2f} | P99={p99:.2f}")

    # 3. HMM (si hmmlearn disponible)
    hmm_result = None
    if _HMM_OK:
        print(f"\nEntrenando HMM ({N_ESTADOS} estados, desde {HMM_INICIO})…")
        try:
            _, idx_hmm, estados, sorted_states = entrenar_hmm(df, HMM_INICIO)
            hmm_result = (idx_hmm, estados)
            print("  HMM OK")
        except Exception as e:
            print(f"  HMM falló: {e}")
            hmm_result = None

    # 4. Estadísticas
    reportar_estadisticas(df, sadf_dict, hmm_result)

    # 5. Gráfico
    graficar(df, sadf_dict, hmm_result)


if __name__ == "__main__":
    main()
