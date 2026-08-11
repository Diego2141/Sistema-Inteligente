"""
aux_backtest_liquidez.py
========================
Backtest ASIMÉTRICO de los intervalos, orientado al dimensionamiento de un
portafolio de liquidez.

Por qué no basta coverage_90 / coverage_98
------------------------------------------
La cobertura es simétrica: mezcla en un solo número dos riesgos con costos
muy distintos. Con target = D − R (negativo = salida neta de fondos):

    target < q05   →  la salida fue PEOR que la prevista: el buffer se
                      queda corto. Es el error caro.
    target > q95   →  entró más de lo previsto: liquidez ociosa. Solo
                      cuesta oportunidad… salvo en horizontes largos, donde
                      un intervalo demasiado ancho produce SOBREASIGNACIÓN.

De ahí que se reporten los dos lados por separado, con el énfasis en el
inferior a horizontes cortos y en el superior a horizontes largos.

Las cuatro métricas
-------------------
 1. Tasa de brecha unilateral       P(target < q_τ), nominal τ
 2. Expected Shortfall              E[q_τ − target | target < q_τ], en MM
                                    + peor episodio y déficit acumulado
 3. Kupiec POF (LR_uc)              ¿la tasa observada es compatible con τ?
                                    χ²(1), valor crítico 3.841 al 5%
 4. Christoffersen (LR_ind, LR_cc)  ¿las brechas están AGRUPADAS? Cinco
                                    brechas aisladas se gestionan; cinco
                                    consecutivas son una crisis de liquidez.
                                    χ²(1) y χ²(2) (crítico 5.991)

ADVERTENCIA DE POTENCIA
-----------------------
Con ~118 observaciones por (fold, h) y τ=0.05 se esperan 5.9 brechas. Los
tests de VaR tienen poca potencia con esas cifras y con τ=0.01 (1.2 brechas
esperadas) prácticamente ninguna. El script reporta el rango de brechas que
NO se rechaza, para que se lea con esa salvedad. Por eso incluye también un
agregado por horizonte pooleando folds (~708 obs), donde sí hay potencia.

Uso:  python aux_backtest_liquidez.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_WFCV     = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"

BANCO = "SISTEMA"
MODO  = "exp_arctan"          # carpeta fold_<MODO>

TAU_PRINCIPAL = 0.05           # nivel de referencia para el dimensionamiento
TAUS_INFERIOR = [0.01, 0.05]   # colas que cuestan iliquidez
TAUS_SUPERIOR = [0.95, 0.99]   # colas que cuestan sobreasignación

# Tramos por criticidad para la asignación. h=2 va aparte: es el más crítico
# porque no da margen de reacción.
TRAMOS = [(2, 2, "h02 (inmediato)"), (3, 5, "h03-05 (esta semana)"),
          (6, 10, "h06-10 (prox. semana)"), (11, 22, "h11-22 (prox. mes)"),
          (23, 45, "h23-45"), (46, 75, "h46-75 (largo)")]

CHI2_1GL_95 = 3.841            # χ²(1) al 5%
CHI2_2GL_95 = 5.991            # χ²(2) al 5%


def _titulo(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def _fmt_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Porcentajes sin usar applymap/DataFrame.map (incompatibles entre pandas 1.x y 2.x)."""
    return df.apply(lambda c: c.map(lambda v: f"{v:.1%}" if pd.notna(v) else ""))


def kupiec_lr(n: int, x: int, p: float) -> float:
    """
    Kupiec Proportion-of-Failures. H0: la tasa de brecha observada es p.

        LR_uc = −2·ln[(1−p)^(n−x) · p^x] + 2·ln[(1−x/n)^(n−x) · (x/n)^x]

    ~ χ²(1). Los casos x=0 y x=n se resuelven analíticamente porque el término
    con ln(0) tiene límite 0 (0·ln 0 → 0).
    """
    if n == 0:
        return np.nan
    if x == 0:
        return float(-2 * n * np.log(1 - p))
    if x == n:
        return float(-2 * n * np.log(p))
    ph = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - ph) + x * np.log(ph)
    return float(2 * (ll1 - ll0))


def christoffersen(brechas: np.ndarray, p: float) -> tuple[float, float, int]:
    """
    Test de independencia de Christoffersen sobre la secuencia de brechas.

    H0: la probabilidad de brecha no depende de si el día anterior hubo brecha.
    Rechazarlo significa que las brechas se AGRUPAN — lo peor para liquidez.

    Returns (LR_ind ~ χ²(1), LR_cc = LR_uc + LR_ind ~ χ²(2), racha_max)
    """
    b = np.asarray(brechas, dtype=bool)
    n = len(b)
    if n < 2:
        return np.nan, np.nan, 0

    # racha máxima de brechas consecutivas
    racha = mx = 0
    for v in b:
        racha = racha + 1 if v else 0
        mx = max(mx, racha)

    prev, cur = b[:-1], b[1:]
    n00 = int((~prev & ~cur).sum()); n01 = int((~prev & cur).sum())
    n10 = int((prev & ~cur).sum());  n11 = int((prev & cur).sum())

    d0, d1 = n00 + n01, n10 + n11
    if d0 == 0 or d1 == 0:          # no hay transiciones desde uno de los estados
        return np.nan, np.nan, mx

    pi01, pi11 = n01 / d0, n11 / d1
    pi = (n01 + n11) / (d0 + d1)
    if pi in (0.0, 1.0):            # sin brechas o todo brechas: no informativo
        return np.nan, np.nan, mx

    def _t(cnt, prob):              # 0·ln(0) → 0
        return cnt * np.log(prob) if cnt > 0 and prob > 0 else 0.0

    ll1 = _t(n00, 1 - pi01) + _t(n01, pi01) + _t(n10, 1 - pi11) + _t(n11, pi11)
    ll0 = _t(n00 + n10, 1 - pi) + _t(n01 + n11, pi)
    lr_ind = float(2 * (ll1 - ll0))
    lr_cc = kupiec_lr(n, int(b.sum()), p) + lr_ind
    return lr_ind, lr_cc, mx


def rango_no_rechazo(n: int, p: float) -> tuple[int, int]:
    """Nº de brechas que Kupiec NO rechaza al 5%. Mide la potencia disponible."""
    ok = [x for x in range(n + 1) if kupiec_lr(n, x, p) <= CHI2_1GL_95]
    return (min(ok), max(ok)) if ok else (0, 0)


def metricas_grupo(g: pd.DataFrame, tau: float, inferior: bool) -> dict:
    """Las 4 métricas para un subconjunto ya filtrado (un fold×h, o un pool)."""
    col = f"q{int(tau * 100):02d}"
    y = g["target"].to_numpy()
    q = g[col].to_numpy()
    n = len(y)

    if inferior:
        brecha = y < q
        deficit = np.where(brecha, q - y, 0.0)    # cuánta liquidez faltó
        p_nom = tau
    else:
        brecha = y > q
        deficit = np.where(brecha, y - q, 0.0)
        p_nom = 1.0 - tau

    x = int(brecha.sum())
    lr_uc = kupiec_lr(n, x, p_nom)
    lr_ind, lr_cc, racha = christoffersen(brecha, p_nom)
    d_br = deficit[brecha]

    return {
        "n": n,
        "brechas": x,
        "tasa": x / n if n else np.nan,
        "tasa_nominal": p_nom,
        "es_mm": float(d_br.mean() / 1e6) if x else 0.0,
        "es_max_mm": float(d_br.max() / 1e6) if x else 0.0,
        "es_total_mm": float(d_br.sum() / 1e6),
        "racha_max": racha,
        "kupiec_LR": lr_uc,
        "kupiec_rechaza": bool(lr_uc > CHI2_1GL_95) if np.isfinite(lr_uc) else False,
        "christof_LR": lr_ind,
        "christof_rechaza": bool(lr_ind > CHI2_1GL_95) if np.isfinite(lr_ind) else False,
        "cc_LR": lr_cc,
        "cc_rechaza": bool(lr_cc > CHI2_2GL_95) if np.isfinite(lr_cc) else False,
    }


def tabla(df: pd.DataFrame, claves: list[str], tau: float, inferior: bool) -> pd.DataFrame:
    filas = []
    for k, g in df.groupby(claves, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        filas.append({**dict(zip(claves, k)), **metricas_grupo(g, tau, inferior)})
    return pd.DataFrame(filas)


# ===========================================================================
def main() -> None:
    dir_fold = DIR_WFCV / f"fold_{MODO}"
    paths = sorted(dir_fold.glob(f"preds_base_{BANCO}_*.parquet"))
    if not paths:
        print(f"[ERROR] sin preds_base en {dir_fold}")
        return
    p = paths[-1]
    print(f"Leyendo {p.name}")
    df = pd.read_parquet(p)

    qcols = [f"q{int(t*100):02d}" for t in TAUS_INFERIOR + TAUS_SUPERIOR]
    faltan = [c for c in ["target"] + qcols if c not in df.columns]
    if faltan:
        print(f"[ERROR] faltan columnas: {faltan}")
        return

    df = df.dropna(subset=["target"] + qcols).sort_values(["fold", "h", "fecha_t"])
    print(f"  {len(df):,} filas | folds {sorted(df['fold'].unique())} | "
          f"h {df['h'].min()}–{df['h'].max()}")

    def _tramo(h):
        for lo, hi, nom in TRAMOS:
            if lo <= h <= hi:
                return nom
        return "otro"
    df["tramo"] = df["h"].map(_tramo)
    orden_tramo = [t[2] for t in TRAMOS]

    # ── Potencia disponible ───────────────────────────────────────────────
    _titulo("Potencia de los tests con el tamaño de muestra disponible")
    n_fh = int(df.groupby(["fold", "h"]).size().median())
    n_h = int(df.groupby("h").size().median())
    print(f"   {'nivel':>22} {'n':>7} {'brechas esperadas':>19} "
          f"{'rango que NO rechaza':>22}")
    print("   " + "-" * 74)
    for etiq, nn in (("por (fold, h)", n_fh), ("por h, pooleando folds", n_h)):
        for t in TAUS_INFERIOR:
            lo, hi = rango_no_rechazo(nn, t)
            print(f"   {etiq + f'  τ={t}':>22} {nn:>7} {nn*t:>19.1f} "
                  f"{f'{lo} – {hi} brechas':>22}")
    print("\n   Un rango ancho significa que el test no puede distinguir un modelo")
    print("   bien calibrado de uno que no lo está: léelo como cota, no como prueba.")

    # ── Tablas ────────────────────────────────────────────────────────────
    tau = TAU_PRINCIPAL
    t_fh = tabla(df, ["fold", "h"], tau, inferior=True)
    t_h  = tabla(df, ["h"], tau, inferior=True)
    t_tr = tabla(df, ["fold", "tramo"], tau, inferior=True)
    t_tr_pool = tabla(df, ["tramo"], tau, inferior=True)

    _titulo(f"Cola INFERIOR (q{int(tau*100):02d}) — el buffer se queda corto")
    piv = t_tr.pivot(index="fold", columns="tramo", values="tasa")
    piv = piv[[c for c in orden_tramo if c in piv.columns]]
    print(f"   Tasa de brecha por fold y tramo (nominal {tau:.0%}):")
    print(_fmt_pct(piv).to_string())

    print(f"\n   Pooleando folds — {len(df.groupby('h').size())} horizontes, "
          f"~{n_h} obs cada uno:")
    print(f"   {'tramo':>24} {'n':>7} {'brechas':>8} {'tasa':>7} {'ES med':>9} "
          f"{'ES max':>9} {'racha':>6} {'Kupiec':>8}")
    print("   " + "-" * 88)
    t_tr_pool["_o"] = t_tr_pool["tramo"].map({t: i for i, t in enumerate(orden_tramo)})
    for _, r in t_tr_pool.sort_values("_o").iterrows():
        marca = " RECHAZA" if r["kupiec_rechaza"] else ""
        print(f"   {r['tramo']:>24} {r['n']:>7,} {r['brechas']:>8} {r['tasa']:>7.1%} "
              f"{r['es_mm']:>9,.0f} {r['es_max_mm']:>9,.0f} {r['racha_max']:>6} "
              f"{r['kupiec_LR']:>8.2f}{marca}")
    print(f"\n   ES en MM. 'racha' = brechas consecutivas máximas.")
    print(f"   Kupiec > {CHI2_1GL_95}: la tasa NO es compatible con {tau:.0%}.")

    # ── Cola superior: sobreasignación en horizontes largos ───────────────
    _titulo("Cola SUPERIOR (q95) — intervalo ancho de más: sobreasignación")
    t_sup = tabla(df, ["tramo"], 0.95, inferior=False)
    t_sup["_o"] = t_sup["tramo"].map({t: i for i, t in enumerate(orden_tramo)})
    print(f"   {'tramo':>24} {'n':>7} {'brechas':>8} {'tasa':>7} "
          f"{'nominal':>8} {'Kupiec':>8}")
    print("   " + "-" * 70)
    for _, r in t_sup.sort_values("_o").iterrows():
        marca = " RECHAZA" if r["kupiec_rechaza"] else ""
        print(f"   {r['tramo']:>24} {r['n']:>7,} {r['brechas']:>8} {r['tasa']:>7.1%} "
              f"{r['tasa_nominal']:>8.1%} {r['kupiec_LR']:>8.2f}{marca}")
    print("\n   Tasa por DEBAJO del nominal ⇒ el intervalo se pasa de ancho por")
    print("   arriba ⇒ en tramos largos eso es capital inmovilizado de más.")

    # ── Excel ─────────────────────────────────────────────────────────────
    _titulo("Escritura del Excel")
    fecha = p.stem.split("_")[-1]
    ruta = dir_fold / f"backtest_liquidez_{BANCO}_{fecha}.xlsx"
    hojas = {
        "inferior_fold_h":   t_fh,
        "inferior_por_h":    t_h,
        "inferior_tramo":    t_tr,
        "inferior_tramo_pool": t_tr_pool.drop(columns="_o"),
        "superior_tramo":    t_sup.drop(columns="_o"),
        "inferior_q01_fold_h": tabla(df, ["fold", "h"], 0.01, inferior=True),
        "detalle_brechas":   df.loc[df["target"] < df[f"q{int(tau*100):02d}"],
                                    ["banco", "fold", "h", "tramo", "fecha_t",
                                     "fecha_th", "target", "q01", "q05", "q95"]]
                               .assign(deficit_mm=lambda d:
                                       (d["q05"] - d["target"]) / 1e6),
    }
    try:
        with pd.ExcelWriter(ruta, engine="openpyxl") as xl:
            for nom, d in hojas.items():
                d.to_excel(xl, sheet_name=nom[:31], index=False)
    except Exception as e:
        print(f"  [AVISO] no se pudo escribir el Excel ({e}) — CSVs sueltos")
        for nom, d in hojas.items():
            d.to_csv(dir_fold / f"backtest_{nom}_{BANCO}_{fecha}.csv", index=False)
        return
    print(f"  {ruta.name}")
    for nom, d in hojas.items():
        print(f"    {nom:<22}: {len(d):>7,} filas")


if __name__ == "__main__":
    main()
