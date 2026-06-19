# -*- coding: utf-8 -*-
"""
step005_validar_sadf.py
=======================
Valida el detector de régimen HMM (expanding window, GARCH por ventana, sin leakage)
antes de integrarlo en la matriz de features.

Único output: gráfico de evolución — cada panel re-etiqueta toda la historia
con el modelo GARCH(1,1) + HMM entrenado hasta ese año.

Uso:
    python step005_validar_sadf.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.optimize import minimize

try:
    from hmmlearn import hmm as hmmlearn_hmm
    from sklearn.preprocessing import StandardScaler
    _HMM_OK = True
except ImportError:
    _HMM_OK = False
    print("AVISO: hmmlearn o sklearn no instalado.")

# ── Configuración ──────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
RUTA_BCRP    = BASE_SISTEMA / "1. Data" / "Raw" / "series_bcrp.xlsx"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output"

BANCO              = "SISTEMA"
H_REF              = 2
N_ESTADOS          = 3
HMM_INICIO         = "2010-01-01"
HMM_PRIMER_VENTANA = 6   # años del primer bloque in-sample

EPISODIOS = [
    ("2020-03-01", "2020-12-31", "COVID-19"),
    ("2021-03-01", "2021-08-31", "Elecciones 2021"),
]

# ── Carga de datos ─────────────────────────────────────────────────────────────

def cargar_datos():
    df = pd.read_parquet(
        RUTA_MATRIZ,
        columns=["fecha_t", "banco", "h", "target"],
        filters=[("banco", "==", BANCO), ("h", "==", H_REF)],
    )
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values("fecha_t").reset_index(drop=True)
    df["fecha"] = df["fecha_t"] + pd.tseries.offsets.BusinessDay(H_REF)
    df = df.set_index("fecha").sort_index()
    print(f"  {BANCO} | {len(df):,} obs | {df.index.min().date()} → {df.index.max().date()}")
    return df["target"].dropna()


def cargar_embig():
    if not RUTA_BCRP.exists():
        return pd.Series(dtype=float, name="EMBI_PERU")
    try:
        raw = pd.read_excel(RUTA_BCRP, sheet_name="EMBIG", header=None)
        header_row = next(
            (i for i, row in raw.iterrows()
             if any(str(v).strip().lower() in ("date", "fecha") for v in row if pd.notna(v))),
            5,
        )
        df = pd.read_excel(RUTA_BCRP, sheet_name="EMBIG", header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        col_f = next((c for c in df.columns if c.lower() in ("date", "fecha")), df.columns[0])
        col_v = next((c for c in df.columns if c.lower() in ("valores", "value", "values")), df.columns[1])
        df = df[[col_f, col_v]].copy()
        df.columns = ["fecha", "EMBI_PERU"]
        _m = {"Ene":"Jan","Feb":"Feb","Mar":"Mar","Abr":"Apr","May":"May","Jun":"Jun",
              "Jul":"Jul","Ago":"Aug","Set":"Sep","Oct":"Oct","Nov":"Nov","Dic":"Dec"}
        def _parse(s):
            s = str(s).strip()
            for es, en in _m.items():
                s = s.replace(es, en)
            return pd.to_datetime(s, errors="coerce", dayfirst=True)
        df["fecha"] = df["fecha"].apply(_parse)
        df["EMBI_PERU"] = pd.to_numeric(
            df["EMBI_PERU"].astype(str).str.replace(",", "."), errors="coerce"
        )
        return df.dropna(subset=["fecha"]).set_index("fecha").sort_index()["EMBI_PERU"]
    except Exception as e:
        print(f"  AVISO EMBIG: {e}")
        return pd.Series(dtype=float, name="EMBI_PERU")


# ── GARCH(1,1) por ventana ─────────────────────────────────────────────────────

def _garch_fit(arr: np.ndarray):
    """MLE de GARCH(1,1). Retorna (ω, α, β, escala, var_unc) o None."""
    escala = float(np.std(arr))
    if escala < 1e-9 or len(arr) < 60:
        return None
    x = (arr / escala).astype(float)
    n, vu = len(x), float(np.var(x))

    def _s2(w, a, b):
        s2 = np.empty(n); s2[0] = vu
        for t in range(1, n):
            s2[t] = w + a * x[t-1]**2 + b * s2[t-1]
        return s2

    def _nll(p):
        w, a, b = p
        if w <= 0 or a <= 0 or b <= 0 or a + b >= 0.9999:
            return 1e10
        s2 = _s2(w, a, b)
        if np.any(s2 <= 0):
            return 1e10
        return 0.5 * float(np.sum(np.log(s2) + x**2 / s2))

    try:
        res = minimize(_nll, [0.01, 0.08, 0.88], method="L-BFGS-B",
                       bounds=[(1e-7, 0.5)] * 2 + [(1e-7, 0.9999)],
                       options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7})
        return None if res.fun > 1e9 else (*res.x, escala, vu)
    except Exception:
        return None


def _garch_vol(params: tuple, arr: np.ndarray, s2_init: float = None) -> np.ndarray:
    """Recursión causal σ_t. s2_init=None → usa var_unc (in-sample)."""
    w, a, b, escala, vu = params
    x = (arr / escala).astype(float)
    n = len(x)
    s2 = np.empty(n)
    s2[0] = vu if s2_init is None else s2_init
    for t in range(1, n):
        s2[t] = w + a * x[t-1]**2 + b * s2[t-1]
    return np.sqrt(np.maximum(s2, 0)) * escala


def _garch_last_s2(params: tuple, arr: np.ndarray) -> float:
    """Propaga la recursión GARCH por arr y retorna el último σ²."""
    w, a, b, escala, vu = params
    x = (arr / escala).astype(float)
    s2 = vu
    for t in range(1, len(x)):
        s2 = w + a * x[t-1]**2 + b * s2
    return w + a * x[-1]**2 + b * s2   # σ²_{n+1}


# ── HMM ───────────────────────────────────────────────────────────────────────

def _fit_hmm(X: np.ndarray):
    scaler = StandardScaler()
    modelo = hmmlearn_hmm.GaussianHMM(
        n_components=N_ESTADOS, covariance_type="full",
        n_iter=1000, random_state=42,
    )
    modelo.fit(scaler.fit_transform(X))
    ss = np.argsort([np.linalg.det(modelo.covars_[s]) for s in range(N_ESTADOS)])
    return modelo, scaler, ss


def _predecir(modelo, scaler, ss, X: np.ndarray) -> np.ndarray:
    raw = modelo.predict(scaler.transform(X))
    mapa = {ss[i]: i for i in range(N_ESTADOS)}
    return np.array([mapa[e] for e in raw])


# ── Evolución expanding ────────────────────────────────────────────────────────

def hmm_evolucion(flujo: pd.Series, primer_ventana: int = HMM_PRIMER_VENTANA) -> dict:
    """
    Para cada año Y: re-estima GARCH+HMM en [inicio, Y] y re-etiqueta toda la
    historia con ese modelo. Devuelve {año: (DatetimeIndex, estados, sigma_series)}.

    GARCH re-estimado en cada bloque → ω, α, β solo ven datos hasta Y.
    """
    año_inicio   = flujo.index.year.min()
    año_fin_prim = año_inicio + primer_ventana - 1
    años_oos     = sorted(y for y in flujo.index.year.unique() if y > año_fin_prim)

    resultados = {}
    print(f"\n  [EVOLUCIÓN] GARCH+HMM re-estimado por bloque | primer_ventana={primer_ventana}a")

    def _bloque(flujo_bloque, label):
        params = _garch_fit(flujo_bloque.values)
        if params is None:
            print(f"    {label}: GARCH falló")
            return None
        w, a, b, escala, _ = params
        sigma = _garch_vol(params, flujo_bloque.values)
        X = np.column_stack([flujo_bloque.values, sigma])
        modelo, scaler, ss = _fit_hmm(X)
        estados = _predecir(modelo, scaler, ss, X)
        pct = (estados == 2).mean() * 100
        print(f"    hasta {label}: {len(flujo_bloque):,} obs | "
              f"GARCH ω={w:.4f} α={a:.3f} β={b:.3f} (α+β={a+b:.3f}) | "
              f"severo={pct:.0f}%")
        return flujo_bloque.index, estados, pd.Series(sigma, index=flujo_bloque.index)

    # Bloque base
    flujo_prim = flujo[flujo.index.year <= año_fin_prim]
    try:
        res = _bloque(flujo_prim, str(año_fin_prim))
        if res:
            resultados[año_fin_prim] = res
    except Exception as e:
        print(f"    Bloque base falló: {e}")

    for año in años_oos:
        flujo_hasta = flujo[flujo.index.year <= año]
        try:
            res = _bloque(flujo_hasta, str(año))
            if res:
                resultados[año] = res
        except Exception as e:
            print(f"    {año}: falló — {e}")

    return resultados


# ── Gráfico de evolución ───────────────────────────────────────────────────────

def graficar_evolucion(flujo: pd.Series, evol: dict, embig: pd.Series = None) -> None:
    años = sorted(evol.keys())
    n    = len(años)
    if n == 0:
        return

    tiene_embig = embig is not None and not embig.empty
    n_rows   = (1 if tiene_embig else 0) + 1 + n
    h_ratios = ([1.2] if tiene_embig else []) + [2.0] + [1.0] * n

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(19, 2.0 + 2.5 * n),
        sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.05},
    )
    fig.suptitle(
        f"Evolución del HMM — {BANCO}  |  {N_ESTADOS} estados\n"
        f"Cada panel re-etiqueta TODA la historia con el modelo GARCH+HMM entrenado hasta ese año",
        fontsize=12, fontweight="bold", y=0.995,
    )

    colores_hmm = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}
    sig = 0   # índice de panel siguiente

    # Panel EMBIG
    if tiene_embig:
        ax = axes[sig]; sig += 1
        ev = embig.reindex(flujo.index.union(embig.index)).sort_index()
        ev = ev[ev.index >= flujo.index.min()]
        ax.plot(ev.index, ev.values, color="#6A1B9A", lw=0.8)
        ax.fill_between(ev.index, ev.values, alpha=0.12, color="#6A1B9A")
        ax.set_ylabel("EMBIG\n(pbs)", fontsize=8)
        ax.set_title("EMBIG Perú — riesgo país (pbs)", fontsize=9)
        for ini, fin, _ in EPISODIOS:
            ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), alpha=0.10, color="navy", zorder=0)

    # Panel flujo neto
    ax = axes[sig]; sig += 1
    cols = np.where(flujo.values >= 0, "#2196F3", "#F44336")
    ax.bar(flujo.index, flujo.values, color=cols, alpha=0.65, width=1.2)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_ylabel("Flujo neto", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B"))
    ax.set_title("Flujo neto diario (azul=entradas, rojo=salidas netas)", fontsize=9)
    for ini, fin, etiqueta in EPISODIOS:
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), alpha=0.10, color="navy", zorder=0)
        mid  = pd.Timestamp(ini) + (pd.Timestamp(fin) - pd.Timestamp(ini)) / 2
        ymax = ax.get_ylim()[1]
        ax.text(mid, ymax * 0.88, etiqueta, ha="center", fontsize=8, color="#1A237E",
                bbox=dict(boxstyle="round,pad=0.2", fc="lavender", alpha=0.8))

    # Paneles HMM por año
    for i, año in enumerate(años):
        ax_i = axes[sig + i]
        idx_a, est_a, sigma_a = evol[año]

        for estado, color in colores_hmm.items():
            for f in idx_a[est_a == estado]:
                ax_i.axvspan(f, f + pd.offsets.BusinessDay(1),
                             alpha=0.55, color=color, linewidth=0)

        sig_n = (sigma_a - sigma_a.mean()) / (sigma_a.std() + 1e-12)
        ax_i.plot(sig_n.index, sig_n.values, color="k", lw=0.5, alpha=0.5)

        for ini, fin, _ in EPISODIOS:
            if pd.Timestamp(ini) <= idx_a.max():
                ax_i.axvspan(pd.Timestamp(ini),
                             min(pd.Timestamp(fin), idx_a.max()),
                             alpha=0.12, color="navy", zorder=0)

        pct = (est_a == 2).mean() * 100
        ax_i.set_ylabel(str(año), fontsize=8, rotation=0, labelpad=35, va="center")
        ax_i.set_title(f"Modelo entrenado hasta {año}  ({len(idx_a):,} obs)  —  "
                       f"severo={pct:.0f}%", fontsize=8, loc="left", pad=2)
        ax_i.set_yticks([])

        if i == 0:
            parches = [mpatches.Patch(color=c, label=l)
                       for c, l in zip(["#4CAF50", "#FFC107", "#F44336"],
                                       ["Calma", "Moderado", "Severo"])]
            ax_i.legend(handles=parches, fontsize=7, loc="upper left", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(top=0.93, hspace=0.05)
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    ruta = DIR_OUTPUT / f"evolucion_hmm_{BANCO}.png"
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    print(f"\n  Gráfico guardado: {ruta}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not _HMM_OK:
        print("ERROR: instalar hmmlearn y sklearn antes de ejecutar.")
        return
    flujo = cargar_datos()
    evol  = hmm_evolucion(flujo[flujo.index >= HMM_INICIO])
    embig = cargar_embig()
    graficar_evolucion(flujo, evol, embig=embig)


if __name__ == "__main__":
    main()
