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
# Fuente de fecha_th cuando los parquets de VAL no la traen. Hasta esta
# revisión el scaffold de VAL guardaba solo fecha_t/h, así que para las
# corridas ya hechas hay que recuperar el mapeo (fecha_t, h) -> fecha_th de la
# matriz. Reconstruirlo con un calendario propio seria arriesgado: peru_bday
# excluye feriados y un calendario distinto etiquetaria mal las posiciones.
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"

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

def _adjuntar_fecha_th(d: pd.DataFrame) -> pd.DataFrame:
    """Recupera fecha_th desde la matriz si el parquet de VAL no la trae."""
    if "fecha_th" in d.columns:
        return d
    if not RUTA_MATRIZ.exists():
        raise FileNotFoundError(
            f"Los parquets de VAL no traen fecha_th y no se encuentra "
            f"{RUTA_MATRIZ}. Vuelva a correr step005 (ya guarda fecha_th en VAL) "
            f"o corrija RUTA_MATRIZ."
        )
    print(f"[info] VAL sin fecha_th — se recupera de {RUTA_MATRIZ.name}")
    mp = (pd.read_parquet(RUTA_MATRIZ, columns=["fecha_t", "h", "fecha_th"])
          .drop_duplicates(subset=["fecha_t", "h"]))
    mp["fecha_t"] = pd.to_datetime(mp["fecha_t"])
    d = d.copy()
    d["fecha_t"] = pd.to_datetime(d["fecha_t"])
    n0 = len(d)
    d = d.merge(mp, on=["fecha_t", "h"], how="left")
    n_sin = int(d["fecha_th"].isna().sum())
    if n_sin:
        print(f"[aviso] {n_sin:,} de {n0:,} filas sin fecha_th en la matriz — se descartan")
        d = d[d["fecha_th"].notna()]
    return d


def _preparar(df: pd.DataFrame) -> pd.DataFrame:
    d = _adjuntar_fecha_th(df)
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
    vals = sub[col].to_numpy(dtype=float)
    k = _k_conformal(vals)
    # factorize en vez de agrupar por la fecha: groupby devuelve claves
    # Timestamp y .unique() devuelve datetime64 según la versión de pandas,
    # y emparejarlas por clave rompe. Con códigos enteros no hay ambigüedad.
    codes, uniq = pd.factorize(sub["fecha_th"])
    n_g = len(uniq)
    if n_g < 3:
        return k, np.nan, np.nan
    por_grupo = [vals[codes == g] for g in range(n_g)]
    reps = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        pick = rng.integers(0, n_g, size=n_g)
        reps[b] = _k_conformal(np.concatenate([por_grupo[g] for g in pick]))
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


TAUS_LO = [0.01, 0.05, 0.40]     # cuantiles por debajo de la mediana
TAUS_HI = [0.60, 0.95, 0.99]     # por encima
N_BOOT_TAU = 150                 # menos réplicas: son 6 τ × 7 posiciones


def _score_tau(d: pd.DataFrame, tau: float) -> tuple[pd.Series, float]:
    """
    Score y nivel del percentil a estimar para el cuantil tau.

    Para tau < 0.5:  q_tau^cal = q50 - k*(q50 - q_tau), y se quiere que la cola
    quede en tau, así que k es el percentil (1-tau) de (q50-y)/(q50-q_tau).
    Para tau > 0.5 es simétrico con el percentil tau.

    El nivel es lo que fija cuántos datos hacen falta: tau=0.05 pide el
    percentil 95 (n>=19) y tau=0.01 pide el 99 (n>=99).
    """
    col = f"q{int(tau * 100):02d}"
    if tau < 0.5:
        w = (d["q50"] - d[col]).clip(lower=1.0)
        return (d["q50"] - d["target"]) / w, 1.0 - tau
    w = (d[col] - d["q50"]).clip(lower=1.0)
    return (d["target"] - d["q50"]) / w, tau


def seccion_4(d: pd.DataFrame, rng) -> None:
    print("\n" + "=" * 78)
    print("4 — ¿La FORMA del factor es invariante en τ?")
    print("=" * 78)
    print("  q01 y q99 exigen el percentil 99 del score: 99 fechas por categoría,")
    print("  y hay 27. No se pueden calibrar por categoría con datos propios.")
    print("  La salida es prestarles la forma de q05/q95, lo cual solo vale si el")
    print("  perfil por día del mes NO cambia con el nivel de τ. Se contrasta")
    print("  comparando τ=0.40 con τ=0.05, que están mucho más separados que")
    print("  0.05 y 0.01: si ahí la forma coincide, extrapolar es defendible.\n")

    for grupo, nombre in ((TAUS_LO, "cola inferior"), (TAUS_HI, "cola superior")):
        print(f"\n  r(pos) por τ — {nombre}   [pool de folds]")
        print(f"{'':<8}" + "".join(f"{p:>17}" for p in ORDEN))
        formas = {}
        for tau in grupo:
            s, nivel = _score_tau(d, tau)
            dd = d.assign(_s=s)
            k_marg = _k_conformal(dd["_s"].values, alpha=1 - nivel)
            fila, vec = f"q{int(tau*100):02d}    ", []
            for pos in ORDEN:
                sub = dd[dd["pos"] == pos]
                if len(sub) == 0 or not np.isfinite(k_marg) or k_marg <= 0:
                    fila += f"{'—':>17}"; vec.append(np.nan); continue
                _bk = N_BOOT_TAU
                k = _k_conformal(sub["_s"].values, alpha=1 - nivel) / k_marg
                vec.append(k)
                nf = sub["fecha_th"].nunique()
                marca = "*" if nf < int(np.ceil(1 / (1 - nivel))) - 1 else " "
                fila += f"{k:>16.2f}{marca}"
            print(fila)
            formas[tau] = np.array(vec, dtype=float)

        ref = grupo[1]      # q05 / q95: el nivel donde sí hay datos
        print(f"\n    similitud de forma contra q{int(ref*100):02d}:")
        for tau in grupo:
            if tau == ref:
                continue
            a, b = formas[tau], formas[ref]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 3:
                print(f"      q{int(tau*100):02d}: insuficiente")
                continue
            corr = float(np.corrcoef(a[m], b[m])[0, 1])
            dif  = float(np.max(np.abs(a[m] / b[m] - 1)))
            print(f"      q{int(tau*100):02d}: correlación {corr:+.3f} | "
                  f"máxima discrepancia relativa {dif:.0%}")
    print("\n  * = menos fechas que el mínimo para ese nivel; el valor es el máximo")
    print("      observado, no un percentil. Informativo, no utilizable.")


def seccion_5(d: pd.DataFrame, rng) -> None:
    print("\n" + "=" * 78)
    print("5 — ¿Conviene además desplazar la mediana por categoría?")
    print("=" * 78)
    print("  El hit-rate de τ=0.50 está clavado en 0.40 en todas las corridas: la")
    print("  mediana queda por debajo de lo realizado. Si ese sesgo depende del día")
    print("  del mes, parte de la asimetría que encontramos en el ancho es sesgo de")
    print("  ubicación disfrazado de asimetría de escala, y conviene corregirlo")
    print("  antes de estimar los factores de ancho.")
    print("  delta se expresa como fracción de la semi-amplitud para que sea")
    print("  comparable entre horizontes; 0.10 significa un décimo del intervalo.\n")

    d = d.assign(_sh=(d["target"] - d["q50"]) / ((d["q95"] - d["q05"]) / 2).clip(lower=1.0))
    filas = {}
    for pos in ORDEN:
        sub = d[d["pos"] == pos]
        fila = {"n_fechas": sub["fecha_th"].nunique()}
        med = float(np.median(sub["_sh"])) if len(sub) else np.nan
        fila["POOL"] = med
        # bootstrap de bloques sobre la mediana: converge mucho mejor que un
        # percentil de cola, así que 27 fechas alcanzan de sobra
        codes, uniq = pd.factorize(sub["fecha_th"])
        vals = sub["_sh"].to_numpy(dtype=float)
        if len(uniq) >= 3:
            por = [vals[codes == g] for g in range(len(uniq))]
            reps = [np.median(np.concatenate([por[g] for g in
                    rng.integers(0, len(uniq), size=len(uniq))]))
                    for _ in range(N_BOOT_TAU)]
            lo, hi = np.percentile(reps, [5, 95])
            fila["IC90"] = f"[{lo:+.2f},{hi:+.2f}]"
            fila["≠0"] = "sí" if lo * hi > 0 else "no"
        for f, g in sub.groupby("fold"):
            fila[f"fold{f}"] = float(np.median(g["_sh"]))
        filas[pos] = fila
    t = pd.DataFrame(filas).T
    for c in t.columns:
        if c not in ("n_fechas", "IC90", "≠0"):
            t[c] = pd.to_numeric(t[c]).round(3)
    print(t.to_string())
    print("\n  delta > 0 → lo realizado supera a la mediana predicha (sub-pronóstico)")
    print("  '≠0' = sí → el desplazamiento es distinguible de cero al 90%")
    print("  Si el signo cambia entre folds en una categoría, ese desplazamiento")
    print("  es ruido y corregirlo empeoraría el pronóstico puntual.")


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

    seccion_4(d, rng)
    seccion_5(d, rng)

    print("\n" + "=" * 78)
    print("Cómo decidir")
    print("=" * 78)
    print("  Prueba 2 con intervalos solapados  → r es estructural; implementar")
    print("  Prueba 2 con r discrepante entre folds más allá del ruido → PARAR:")
    print("     el efecto no es estable y una constante por categoría no sirve")
    print("  Prueba 1 con r dependiente de h    → agregar h_grupo a la taxonomía")
    print("  Prueba 3 con k dispares entre folds → calibrar dentro de cada fold")
    print("  Fila 0 con celdas bajo el mínimo   → fusionar categorías (J=2 o J=1)")
    print("  Prueba 4 con forma estable entre τ → q01/q99 heredan la de q05/q95")
    print("  Prueba 4 con forma dependiente de τ → dejar la banda del 98% sin")
    print("     condicionar por calendario, y decirlo")
    print("  Prueba 5 con '≠0'=sí y mismo signo en los 4 folds → corregir también")
    print("     la ubicación; si el signo cambia entre folds, dejar q50 intacto")


if __name__ == "__main__":
    main()
