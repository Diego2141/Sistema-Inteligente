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
VENTANAS     = [60, 120, 252]
LAGS         = 1

# HMM params
N_ESTADOS        = 3
HMM_INICIO       = "2010-01-01"   # usar toda la historia disponible
HMM_MIN_AÑOS     = 2              # mínimo de historia antes de etiquetar (expanding)
                                  # → primeros labels desde 2012 (entrenado en 2010-2011)
HMM_ROLLING_AÑOS = 6              # tamaño ventana rolling fija (en años)

# Episodios de stress conocidos
EPISODIOS = [
    ("2020-03-01", "2020-12-31", "COVID-19"),
    ("2021-03-01", "2021-08-31", "Elecciones 2021"),
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
#
# Modelo: Gaussian HMM de K=3 estados ocultos S_t ∈ {0,1,2}
#
# Componentes:
#   π_i   = P(S_1 = i)                         distribución inicial
#   a_ij  = P(S_t = j | S_{t-1} = i)           matriz de transición A (3×3)
#   b_i(x)= N(x | μ_i, Σ_i)                   emisión Gaussiana por estado
#
# Observación: X_t = [flujo_neto_norm, garch_vol_norm]  (vector 2D)
#
# Entrenamiento (Baum-Welch / EM):
#   Maximiza  P(X_1,...,X_T | π, A, {μ_i,Σ_i})  iterando:
#     E-step: calcular γ_t(i) = P(S_t=i | X_{1:T})  con forward-backward
#     M-step: actualizar π, A, μ_i, Σ_i usando γ_t(i)
#
# Predicción de estados (Viterbi):
#   S*_{1:T} = argmax P(S_{1:T} | X_{1:T}, π, A, {μ_i,Σ_i})
#   Algoritmo de programación dinámica O(K²·T)
#
# Ordenamiento post-entrenamiento:
#   Estados reordenados por det(Σ_i) ascendente → 0=calma, 2=severo


def _fit_hmm_single(X_train: np.ndarray) -> tuple:
    """Ajusta un HMM y devuelve (modelo, scaler, sorted_states)."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    modelo = hmmlearn_hmm.GaussianHMM(
        n_components=N_ESTADOS, covariance_type="full",
        n_iter=1000, random_state=42,
    )
    modelo.fit(Xs)
    varianzas     = [np.linalg.det(modelo.covars_[s]) for s in range(N_ESTADOS)]
    sorted_states = np.argsort(varianzas)   # 0=calma, 2=severo
    return modelo, scaler, sorted_states


def _predecir_estados(modelo, scaler, sorted_states, X: np.ndarray) -> np.ndarray:
    """Predice estados reordenados (0=calma, 2=severo) para X."""
    Xs          = scaler.transform(X)
    estados_raw = modelo.predict(Xs)
    mapa        = {sorted_states[i]: i for i in range(N_ESTADOS)}
    return np.array([mapa[e] for e in estados_raw])


def hmm_muestra_completa(df: pd.DataFrame, inicio: str) -> tuple:
    """
    VERSION CON LEAKAGE — entrena en 2016→hoy y etiqueta toda la serie.
    Solo para comparación visual.
    """
    df_base = df[df.index >= inicio][["target", "garch_vol"]].dropna()
    modelo, scaler, sorted_states = _fit_hmm_single(df_base.values)

    df_full = df[["target", "garch_vol"]].dropna()
    estados = _predecir_estados(modelo, scaler, sorted_states, df_full.values)

    trans = modelo.transmat_
    print(f"\n  [MUESTRA COMPLETA] Matriz de transición:")
    for i in range(N_ESTADOS):
        fila   = "  ".join([f"{trans[sorted_states[i], sorted_states[j]]:.2f}"
                             for j in range(N_ESTADOS)])
        nombre = ["Calma    ", "Moderado ", "Severo   "][i]
        print(f"    {nombre}: {fila}")

    return df_full.index, estados


def hmm_rolling(df: pd.DataFrame, inicio: str,
                ventana_años: int = 6) -> tuple:
    """
    VERSION SIN LEAKAGE — rolling window de longitud fija.

    Para etiquetar el año Y:
      - Entrena HMM en [Y - ventana_años, Y-1]  (ventana fija)
      - Requiere que haya al menos ventana_años de historia desde `inicio`
      - Predice estados del año Y con ese modelo

    Con ventana_años=6 e inicio=2010:
      - Primer entrenamiento: 2010-2015 (6 años) → etiqueta 2016
      - 2do entrenamiento:    2011-2016 (6 años) → etiqueta 2017
      - ...y así sucesivamente
    """
    df_base    = df[df.index >= inicio][["target", "garch_vol"]].dropna()
    año_inicio = df_base.index.year.min()
    años       = sorted(df_base.index.year.unique())
    todos_idx     = []
    todos_estados = []

    print(f"\n  [ROLLING {ventana_años}a] HMM año a año | inicio={inicio}")
    print(f"  Primeras etiquetas desde {año_inicio + ventana_años} "
          f"(ventana fija {ventana_años} años)")

    for año in años:
        años_historia = año - año_inicio
        if años_historia < ventana_años:
            print(f"    {año}: historia insuficiente ({años_historia} años < {ventana_años}) — omitiendo")
            continue

        año_ini_ventana = año - ventana_años
        X_train = df_base[
            (df_base.index.year >= año_ini_ventana) &
            (df_base.index.year < año)
        ].values
        try:
            modelo, scaler, sorted_states = _fit_hmm_single(X_train)
        except Exception as e:
            print(f"    {año}: HMM falló — {e}")
            continue

        df_año = df_base[df_base.index.year == año]
        if df_año.empty:
            continue

        estados_año = _predecir_estados(modelo, scaler, sorted_states, df_año.values)
        todos_idx.extend(df_año.index.tolist())
        todos_estados.extend(estados_año.tolist())
        pct_severo = (np.array(estados_año) == 2).mean() * 100
        print(f"    {año}: ventana [{año_ini_ventana}-{año-1}] ({len(X_train):,} obs) → "
              f"severo={pct_severo:.0f}%")

    return pd.DatetimeIndex(todos_idx), np.array(todos_estados)


def hmm_expanding(df: pd.DataFrame, inicio: str,
                  min_años: int = 2) -> tuple:
    """
    VERSION SIN LEAKAGE — expanding window por año con mínimo de historia.

    Para etiquetar el año Y:
      - Requiere al menos `min_años` de historia previa
      - Entrena HMM en datos desde `inicio` hasta fin del año Y-1
      - Predice estados del año Y con ese modelo

    Con min_años=2 e inicio=2010:
      - Primer entrenamiento: 2010-2011 (2 años) → etiqueta 2012
      - 2do entrenamiento:    2010-2012 (3 años) → etiqueta 2013
      - ...y así sucesivamente (historia siempre crece)
    """
    df_base       = df[df.index >= inicio][["target", "garch_vol"]].dropna()
    año_inicio    = df_base.index.year.min()
    años          = sorted(df_base.index.year.unique())
    todos_idx     = []
    todos_estados = []

    print(f"\n  [EXPANDING] HMM año a año | inicio={inicio} | min_años={min_años}")
    print(f"  Primeras etiquetas desde {año_inicio + min_años} "
          f"(entrenado en {año_inicio}-{año_inicio + min_años - 1})")

    for año in años:
        años_historia = año - año_inicio
        if años_historia < min_años:
            print(f"    {año}: historia insuficiente ({años_historia} años < {min_años}) — omitiendo")
            continue

        X_train = df_base[df_base.index.year < año].values
        try:
            modelo, scaler, sorted_states = _fit_hmm_single(X_train)
        except Exception as e:
            print(f"    {año}: HMM falló — {e}")
            continue

        df_año = df_base[df_base.index.year == año]
        if df_año.empty:
            continue

        estados_año = _predecir_estados(modelo, scaler, sorted_states,
                                        df_año.values)
        todos_idx.extend(df_año.index.tolist())
        todos_estados.extend(estados_año.tolist())
        pct_severo = (np.array(estados_año) == 2).mean() * 100
        print(f"    {año}: entrenado en {len(X_train):,} obs ({años_historia} años) → "
              f"severo={pct_severo:.0f}%")

    return pd.DatetimeIndex(todos_idx), np.array(todos_estados)


# ── Helper ────────────────────────────────────────────────────────────────────

def df_base_year(inicio: str) -> int:
    return pd.Timestamp(inicio).year


# ── Gráfico 5 paneles ─────────────────────────────────────────────────────────

def _pintar_regimen(ax, idx, estados, vol_ref_idx, vol_ref_vals):
    """Pinta fondo por régimen HMM y superpone garch_vol normalizada."""
    colores_hmm = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}
    for estado, color in colores_hmm.items():
        fechas_e = idx[estados == estado]
        for f in fechas_e:
            ax.axvspan(f, f + pd.offsets.BusinessDay(1),
                       alpha=0.55, color=color, linewidth=0)
    vol_n = (vol_ref_vals - np.nanmean(vol_ref_vals)) / (np.nanstd(vol_ref_vals) + 1e-12)
    ax.plot(vol_ref_idx, vol_n, color="k", lw=0.6, alpha=0.55)
    parches = [
        mpatches.Patch(color="#4CAF50", label="Calma"),
        mpatches.Patch(color="#FFC107", label="Moderado"),
        mpatches.Patch(color="#F44336", label="Severo"),
    ]
    ax.legend(handles=parches, fontsize=8, loc="upper left")
    ax.set_ylabel("Régimen HMM", fontsize=9)


def graficar(df: pd.DataFrame, sadf_dict: dict,
             hmm_completo=None, hmm_expanding_=None,
             hmm_rolling_=None) -> None:

    n_panels = 2 + (1 if hmm_completo   is not None else 0) \
                 + (1 if hmm_expanding_  is not None else 0) \
                 + (1 if hmm_rolling_    is not None else 0)
    h_ratios = [2.0] + [1.0] * (n_panels - 1)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(19, 4 * n_panels),
        sharex=True, gridspec_kw={"height_ratios": h_ratios}
    )
    axes = list(axes)

    ventanas_str = ", ".join(f"{v}d" for v in VENTANAS)
    fig.suptitle(
        f"Validación SADF y HMM — {BANCO}\n"
        f"SADF: tau_min={TAU_MIN} dh | ventanas {ventanas_str}  |  "
        f"HMM: {N_ESTADOS} estados | expanding (mín {HMM_MIN_AÑOS}a) vs rolling ({HMM_ROLLING_AÑOS}a)",
        fontsize=12, fontweight="bold"
    )

    ax_idx = 0
    fechas = df.index
    flujo  = df["target"].values

    # ── Panel 1: flujo neto ────────────────────────────────────────────────────
    ax1 = axes[ax_idx]; ax_idx += 1
    colores = np.where(flujo >= 0, "#2196F3", "#F44336")
    ax1.bar(fechas, flujo, color=colores, alpha=0.65, width=1.2)
    ax1.axhline(0, color="k", lw=0.7, ls="--")
    ax1.set_ylabel("Flujo neto", fontsize=9)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B"))
    ax1.set_title("Flujo neto diario (azul=entradas, rojo=salidas netas)", fontsize=9)

    # ── Panel 2: SADF ─────────────────────────────────────────────────────────
    ax2 = axes[ax_idx]; ax_idx += 1
    colores_s = ["#E91E63", "#FF6F00", "#7B1FA2"]
    for (nombre, svals), color in zip(sadf_dict.items(), colores_s):
        ax2.plot(fechas, svals, lw=1.0, color=color, label=nombre, alpha=0.85)
        p95 = np.nanpercentile(svals, 95)
        p99 = np.nanpercentile(svals, 99)
        ax2.axhline(p95, color=color, lw=0.9, ls=":", label=f"P95={p95:.2f}")
        ax2.axhline(p99, color=color, lw=0.9, ls="--", alpha=0.6,
                    label=f"P99={p99:.2f}")
    ax2.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax2.set_ylabel("SADF (t-stat sup.)", fontsize=9)
    ax2.set_title("SADF sobre garch_vol — spikes positivos = volatilidad explosiva",
                  fontsize=9)
    ax2.legend(fontsize=8, loc="upper left", ncol=2)

    # ── Panel 3: HMM muestra completa (con leakage) ───────────────────────────
    if hmm_completo is not None:
        ax3 = axes[ax_idx]; ax_idx += 1
        idx_c, est_c = hmm_completo
        vol_c = df.loc[df.index.isin(idx_c), "garch_vol"]
        _pintar_regimen(ax3, idx_c, est_c, vol_c.index, vol_c.values)
        ax3.set_title(
            "HMM muestra completa (CON leakage) — entrenado en 2016→hoy, etiqueta toda la historia",
            fontsize=9, color="#B71C1C"
        )

    # ── Panel 4: HMM expanding (sin leakage) ──────────────────────────────────
    if hmm_expanding_ is not None:
        ax4 = axes[ax_idx]; ax_idx += 1
        idx_e, est_e = hmm_expanding_
        vol_e = df.loc[df.index.isin(idx_e), "garch_vol"]
        _pintar_regimen(ax4, idx_e, est_e, vol_e.index, vol_e.values)
        ax4.set_title(
            f"HMM expanding (SIN leakage, mín {HMM_MIN_AÑOS}a) — historia crece: "
            f"etiqueta desde {df_base_year(HMM_INICIO) + HMM_MIN_AÑOS}",
            fontsize=9, color="#1B5E20"
        )

    # ── Panel 5: HMM rolling (sin leakage) ────────────────────────────────────
    if hmm_rolling_ is not None:
        ax5 = axes[ax_idx]; ax_idx += 1
        idx_r, est_r = hmm_rolling_
        vol_r = df.loc[df.index.isin(idx_r), "garch_vol"]
        _pintar_regimen(ax5, idx_r, est_r, vol_r.index, vol_r.values)
        ax5.set_title(
            f"HMM rolling {HMM_ROLLING_AÑOS}a (SIN leakage) — ventana fija: "
            f"etiqueta desde {df_base_year(HMM_INICIO) + HMM_ROLLING_AÑOS}",
            fontsize=9, color="#0D47A1"
        )

    # ── Bandas de episodios ────────────────────────────────────────────────────
    for ini, fin, etiqueta in EPISODIOS:
        for ax in axes:
            ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin),
                       alpha=0.10, color="navy", zorder=0)
        mid  = pd.Timestamp(ini) + (pd.Timestamp(fin) - pd.Timestamp(ini)) / 2
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

def reportar_estadisticas(df, sadf_dict, hmm_completo=None,
                          hmm_expanding_=None, hmm_rolling_=None):
    print("\n── SADF: % días sobre P95 en episodios de stress ───────────────────")
    for nombre, svals in sadf_dict.items():
        p95 = np.nanpercentile(svals, 95)
        s   = pd.Series(svals, index=df.index)
        for ini, fin, etiqueta in EPISODIOS:
            sub = s[(s.index >= ini) & (s.index <= fin)]
            if len(sub) > 0:
                pct = (sub > p95).mean() * 100
                print(f"  {nombre:<18} | {etiqueta:<15}: {pct:5.1f}% sobre P95")

    variantes = [
        ("MUESTRA COMPLETA (leakage)    ", hmm_completo),
        (f"EXPANDING  mín {HMM_MIN_AÑOS}a (sin leak)", hmm_expanding_),
        (f"ROLLING    {HMM_ROLLING_AÑOS}a  (sin leak) ", hmm_rolling_),
    ]
    for tag, hmm_res in variantes:
        if hmm_res is None:
            continue
        idx_h, estados = hmm_res
        total = len(estados)
        print(f"\n── HMM {tag}: distribución de estados ──────")
        for e, nom in enumerate(["Calma   ", "Moderado", "Severo  "]):
            n = (estados == e).sum()
            print(f"  Estado {e} {nom}: {n:,} días ({n/total*100:.1f}%)")
        print(f"  % días en stress severo (E2) por episodio:")
        for ini, fin, etiqueta in EPISODIOS:
            mask = (idx_h >= ini) & (idx_h <= fin)
            sub  = estados[mask]
            if len(sub) > 0:
                pct = (sub == 2).mean() * 100
                print(f"    {etiqueta:<20}: {pct:5.1f}% en estado severo")



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

    # 3. HMM — tres versiones
    hmm_c  = None   # muestra completa (con leakage, solo referencia)
    hmm_ex = None   # expanding: historia crece, min HMM_MIN_AÑOS
    hmm_ro = None   # rolling:   ventana fija HMM_ROLLING_AÑOS

    if _HMM_OK:
        print(f"\nEntrenando HMM muestra completa (con leakage)…")
        try:
            hmm_c = hmm_muestra_completa(df, HMM_INICIO)
            print("  OK")
        except Exception as e:
            print(f"  Falló: {e}")

        print(f"\nEntrenando HMM expanding (sin leakage, mín {HMM_MIN_AÑOS} años)…")
        try:
            hmm_ex = hmm_expanding(df, HMM_INICIO, min_años=HMM_MIN_AÑOS)
            print("  OK")
        except Exception as e:
            print(f"  Falló: {e}")

        print(f"\nEntrenando HMM rolling (sin leakage, ventana {HMM_ROLLING_AÑOS} años)…")
        try:
            hmm_ro = hmm_rolling(df, HMM_INICIO, ventana_años=HMM_ROLLING_AÑOS)
            print("  OK")
        except Exception as e:
            print(f"  Falló: {e}")

    # 4. Estadísticas
    reportar_estadisticas(df, sadf_dict, hmm_c, hmm_ex, hmm_ro)

    # 5. Gráfico (hasta 5 paneles)
    graficar(df, sadf_dict, hmm_c, hmm_ex, hmm_ro)


if __name__ == "__main__":
    main()
