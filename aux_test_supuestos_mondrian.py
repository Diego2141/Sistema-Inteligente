"""
¿Se sostienen los supuestos de una calibración conformal Mondrian por posición
en el mes? Tres pruebas sobre los parquets de VAL ya guardados. No re-entrena.

Qué se quiere calibrar
----------------------
Para la fila i = (fold, h, fecha_t), con fecha_th = fecha_t + h días hábiles:

    w_lo = q50 - q05                    s_lo = (q50 - y) / w_lo
    w_hi = q95 - q50                    s_hi = (y - q50) / w_hi

s_lo > 1 equivale exactamente a que y cruzó q05, así que el cuantil 0.95 de
s_lo dentro de una categoría es el factor por el que habría que multiplicar la
semi-amplitud para que esa cola quede en 5%:

    k(kappa) = Quantile_0.95( s | kappa )        (estadístico de orden
                                                  ceil(0.95*(n+1)) — CQR)
    r(kappa) = k(kappa) / k_marginal              factor de forma, adimensional

La propuesta es aplicar k_f(kappa) = k_f^marg * r(kappa), separando el NIVEL
(por fold, estimable con datos recientes) de la FORMA (estructural, estimada
agrupando folds). Eso solo tiene sentido si se cumplen tres supuestos.

Las tres pruebas
----------------
  1. r(kappa) invariante en h    → si falla, hay que meter h en la taxonomía
  2. r(kappa) estable en el tiempo → si falla, NO es estructural y el método
                                     entero no aplica. Es la prueba crítica.
  3. el score multiplicativo normaliza entre modelos → si falla, no se pueden
     agrupar folds y hay que calibrar dentro de cada uno

Sobre cómo leer la prueba 2
---------------------------
Las filas NO son independientes: una misma fecha_th aparece en hasta 74 filas.
La varianza de r la gobierna el número de FECHAS distintas, que en la categoría
'cierre -1' es ~10 por fold. Con tan poco, r va a variar entre folds aunque el
supuesto sea cierto. Por eso cada r lleva un intervalo por bootstrap de bloques
sobre fechas_th: la pregunta no es si los r difieren, sino si difieren MÁS de
lo que explica el ruido.

Uso: ejecutar directamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración — espejo de step005_walk_forward_cv_4.py
# ---------------------------------------------------------------------------
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"

EXPANDING     = False   # debe coincidir con la corrida que se quiere leer
AJUSTE_ARCTAN = True
BANCO         = "SISTEMA"
FECHA_TAG     = ""      # "" → la más reciente

ALPHA      = 0.05       # nivel de cada cola
J_BORDE    = 3          # cuántos días de cada extremo del mes son categoría propia
MIN_FECHAS = 19         # ceil(0.95*(n+1)) <= n exige n >= 19
N_BOOT     = 400
SEED       = 42

BINS_H   = [1, 5, 15, 30, 50, 75]
LABELS_H = ["h02-05", "h06-15", "h16-30", "h31-50", "h51-75"]

ORDEN = ["inicio +1", "inicio +2", "inicio +3", "resto del mes",
         "cierre -3", "cierre -2", "cierre -1"]


def _dir_modo() -> Path:
    sfx = "_arctan" if AJUSTE_ARCTAN else ""
    return DIR_OUTPUT / f"fold_{'exp' if EXPANDING else 'roll'}{sfx}"


def _cargar_val(dir_modo: Path) -> pd.DataFrame:
    pat = f"preds_val_fold*_{BANCO}_*.parquet"
    files = sorted(dir_modo.glob(pat))
    if FECHA_TAG:
        files = [f for f in files if FECHA_TAG in f.name]
    if not files:
        raise FileNotFoundError(f"No hay {pat} en {dir_modo}.")
    tags = sorted({f.stem.split("_")[-1] for f in files})
    if len(tags) > 1:
        files = [f for f in files if tags[-1] in f.name]
        print(f"[aviso] varias corridas {tags}; se usa {tags[-1]}")
    print(f"Archivos: {', '.join(f.name for f in files)}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------

def _preparar(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("fecha_t", "fecha_th"):
        d[c] = pd.to_datetime(d[c])
    d["w_lo"] = (d["q50"] - d["q05"]).clip(lower=1.0)
    d["w_hi"] = (d["q95"] - d["q50"]).clip(lower=1.0)
    d["s_lo"] = (d["q50"] - d["target"]) / d["w_lo"]
    d["s_hi"] = (d["target"] - d["q50"]) / d["w_hi"]
    d["h_grupo"] = pd.cut(d["h"], bins=BINS_H, labels=LABELS_H)

    # Posición dentro del mes, sobre los días hábiles presentes en los datos.
    # Meses con menos de 15 fechas son bordes de ventana: el conteo quedaría
    # truncado y etiquetaría mal, así que se descartan.
    fe = pd.DataFrame({"fecha_th": sorted(d["fecha_th"].unique())})
    fe["mes"] = fe["fecha_th"].dt.to_period("M")
    cnt = fe.groupby("mes")["fecha_th"].transform("size")
    fe["p_ini"] = fe.groupby("mes").cumcount() + 1
    fe["p_fin"] = cnt - fe["p_ini"] + 1
    fe.loc[cnt < 15, ["p_ini", "p_fin"]] = np.nan

    def _bucket(r):
        if np.isnan(r["p_ini"]):
            return None
        if r["p_ini"] <= J_BORDE:
            return f"inicio +{int(r['p_ini'])}"
        if r["p_fin"] <= J_BORDE:
            return f"cierre -{int(r['p_fin'])}"
        return "resto del mes"

    fe["pos"] = fe.apply(_bucket, axis=1)
    d = d.merge(fe[["fecha_th", "pos"]], on="fecha_th", how="left")
    return d[d["pos"].notna()]


def _k_conformal(s: np.ndarray, alpha: float = ALPHA) -> float:
    """Estadístico de orden ceil((n+1)(1-alpha)) — corrección de muestra finita."""
    s = np.sort(np.asarray(s, dtype=float))
    n = len(s)
    if n == 0:
        return np.nan
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[min(idx, n - 1)])       # min(): sin corrección si n < 19


def _k_boot(sub: pd.DataFrame, col: str, rng) -> tuple[float, float, float]:
    """
    k con intervalo por bootstrap de bloques sobre fecha_th.

    Remuestrear FECHAS y no filas es lo que respeta la correlación: las ~74
    filas de una misma fecha comparten el mismo y y fallan juntas.
    """
    k = _k_conformal(sub[col].values)
    fechas = sub["fecha_th"].unique()
    if len(fechas) < 3:
        return k, np.nan, np.nan
    por_fecha = {f: g[col].values for f, g in sub.groupby("fecha_th")}
    reps = []
    for _ in range(N_BOOT):
        pick = rng.choice(fechas, size=len(fechas), replace=True)
        reps.append(_k_conformal(np.concatenate([por_fecha[f] for f in pick])))
    lo, hi = np.percentile(reps, [5, 95])
    return k, float(lo), float(hi)


def _tabla_r(d: pd.DataFrame, col: str, by: str, rng) -> pd.DataFrame:
    """r(pos) = k(pos)/k_marginal, dentro de cada nivel de `by`."""
    filas = {}
    for niv, g in d.groupby(by, observed=True):
        k_marg = _k_conformal(g[col].values)
        fila = {}
        for pos in ORDEN:
            sub = g[g["pos"] == pos]
            nf = sub["fecha_th"].nunique()
            if nf == 0 or not np.isfinite(k_marg) or k_marg <= 0:
                fila[pos] = np.nan
                continue
            k, lo, hi = _k_boot(sub, col, rng)
            fila[pos] = k / k_marg
            fila[f"{pos}__lo"] = lo / k_marg
            fila[f"{pos}__hi"] = hi / k_marg
            fila[f"{pos}__nf"] = nf
        filas[niv] = fila
    return pd.DataFrame(filas).T


def _print_r(t: pd.DataFrame, titulo: str) -> None:
    print(f"\n{titulo}")
    cols = [c for c in ORDEN if c in t.columns]
    enc = f"{'':<12}" + "".join(f"{c:>17}" for c in cols)
    print(enc)
    for niv in t.index:
        linea = f"{str(niv):<12}"
        for c in cols:
            v  = t.loc[niv, c]
            lo = t.loc[niv, f"{c}__lo"] if f"{c}__lo" in t.columns else np.nan
            hi = t.loc[niv, f"{c}__hi"] if f"{c}__hi" in t.columns else np.nan
            if not np.isfinite(v):
                linea += f"{'—':>17}"
            elif np.isfinite(lo):
                linea += f"{v:>7.2f} [{lo:.2f},{hi:.2f}]"
            else:
                linea += f"{v:>17.2f}"
        print(linea)


def main() -> None:
    rng = np.random.default_rng(SEED)
    d = _preparar(_cargar_val(_dir_modo()))
    print(f"Filas VAL clasificadas: {len(d):,}  |  folds: {sorted(d['fold'].unique())}")
    print(f"fecha_th: {d['fecha_th'].min():%Y-%m-%d} → {d['fecha_th'].max():%Y-%m-%d}")

    # ── 0) Tamaño muestral efectivo ────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"0 — Tamaño muestral: fecha_th DISTINTAS (mínimo para calibrar: {MIN_FECHAS})")
    print("=" * 78)
    piv = (d.groupby(["fold", "pos"], observed=True)["fecha_th"].nunique()
           .unstack("pos").reindex(columns=ORDEN))
    piv.loc["POOL 1..4"] = [d.loc[d["pos"] == p, "fecha_th"].nunique() for p in ORDEN]
    print(piv.to_string())
    insuf = (piv < MIN_FECHAS).sum().sum()
    print(f"\n  celdas por debajo del mínimo: {insuf} de {piv.size}")

    # ── 1) ¿r es invariante en h? ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("1 — ¿r(pos) es invariante en el horizonte?  [pool de folds]")
    print("=" * 78)
    print("  Si el efecto de calendario es propiedad de la fecha de resultado y no")
    print("  del horizonte, las filas deben coincidir dentro de sus intervalos.")
    for col, lado in (("s_lo", "cola inferior"), ("s_hi", "cola superior")):
        _print_r(_tabla_r(d, col, "h_grupo", rng), f"  {col}  ({lado})")

    # ── 2) ¿r es estable en el tiempo?  — LA PRUEBA CRÍTICA ────────────────
    print("\n" + "=" * 78)
    print("2 — ¿r(pos) es estable entre folds?  [PRUEBA CRÍTICA]")
    print("=" * 78)
    print("  Si r es estructural debe repetirse fold a fold. Con ~10 fechas por")
    print("  categoría los puntos van a diferir por ruido: la pregunta es si los")
    print("  intervalos al 90% se solapan, no si los puntos coinciden.")
    for col, lado in (("s_lo", "cola inferior"), ("s_hi", "cola superior")):
        _print_r(_tabla_r(d, col, "fold", rng), f"  {col}  ({lado})")

    # ── 3) ¿El score multiplicativo normaliza entre modelos? ───────────────
    print("\n" + "=" * 78)
    print("3 — ¿El score normaliza entre modelos?  [solo 'resto del mes']")
    print("=" * 78)
    print("  Si el cociente por la semi-amplitud del propio modelo hace comparables")
    print("  los folds, la distribución de s en la categoría neutra debe ser la")
    print("  misma. k≈1 indica además que el modelo calibra bien fuera de bordes.\n")
    r0 = d[d["pos"] == "resto del mes"]
    filas = []
    for f, g in r0.groupby("fold"):
        fila = {"fold": f, "n_fechas": g["fecha_th"].nunique()}
        for col in ("s_lo", "s_hi"):
            k, lo, hi = _k_boot(g, col, rng)
            fila[f"med_{col}"] = float(np.median(g[col]))
            fila[f"k_{col}"]   = k
            fila[f"ic_{col}"]  = f"[{lo:.2f},{hi:.2f}]"
        filas.append(fila)
    print(pd.DataFrame(filas).set_index("fold").round(3).to_string())

    print("\n" + "=" * 78)
    print("Cómo decidir")
    print("=" * 78)
    print("  Prueba 2 con intervalos solapados  → r es estructural; implementar")
    print("  Prueba 2 con r discrepante entre folds más allá del ruido → PARAR:")
    print("     el efecto no es estable y una constante por categoría no sirve")
    print("  Prueba 1 con r dependiente de h    → agregar h_grupo a la taxonomía")
    print("  Prueba 3 con k dispares entre folds → calibrar dentro de cada fold")
    print("  Fila 0 con celdas bajo el mínimo   → fusionar categorías (J=2 o J=1)")


if __name__ == "__main__":
    main()
