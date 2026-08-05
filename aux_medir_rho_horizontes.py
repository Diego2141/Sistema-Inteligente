"""
aux_medir_rho_horizontes.py
===========================
Mide la autocorrelación intra-cluster (rho) que decide si agrupar horizontes
vecinos en step005 vale la pena, y con qué ancho de banda.

El problema
-----------
Hoy cada modelo se entrena sobre un solo horizonte: ~500 filas de TRAIN y ~121
de VAL, de las cuales solo ~1.2 caen en la cola del 1%. Agrupar h vecinos
multiplica las filas, pero NO son independientes: las m filas de una misma
fecha comparten el mismo vector de features x_t, y sus targets son días
consecutivos de la misma serie.

El tamaño efectivo no es n·m sino n·m / DEFF, con

    DEFF  =  1 + 2 · Σ_{k=1}^{m-1} (1 − k/m) · rho_k

(se reduce a 1 + (m−1)·rho cuando rho_k es constante).

Qué mide este script
--------------------
  A · rho NO CONDICIONAL — autocorrelación de la serie de flujo neto diario.
      Es una cota superior: conditionar sobre las features suele reducirla.

  B · rho CONDICIONAL — correlación de los RESIDUOS del modelo (target − q50)
      entre horizontes de una misma fecha. Es el número que realmente entra
      en el DEFF. Requiere preds_base_*.parquet de una corrida previa.

  C · Tamaño efectivo y ganancia real para distintos anchos de banda.

Uso:  python aux_medir_rho_horizontes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_WFCV     = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"

BANCO      = "SISTEMA"
MODO       = "exp_arctan"          # carpeta fold_<MODO> de la que leer los residuos
LAG_MAX    = 25                    # lags de autocorrelación a reportar
ANCHOS     = [3, 5, 10, 15, 20]    # anchos de banda a evaluar
N_FECHAS   = 500                   # filas de TRAIN por (fold, h) hoy

# Bandas candidatas (h_lo, h_hi) — la partición que se propuso
BANDAS = [(2, 2), (3, 5), (6, 10), (11, 15),
          (16, 25), (26, 40), (41, 55), (56, 75)]


def _titulo(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def acf(x: np.ndarray, lag_max: int) -> np.ndarray:
    """Autocorrelación muestral para lags 1..lag_max, ignorando NaN por par."""
    x = np.asarray(x, dtype=float)
    out = np.full(lag_max, np.nan)
    for k in range(1, lag_max + 1):
        a, b = x[:-k], x[k:]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 30:
            out[k - 1] = float(np.corrcoef(a[m], b[m])[0, 1])
    return out


def deff(rho: np.ndarray, m: int) -> float:
    """
    Efecto de diseño de un bloque de m observaciones consecutivas.

        DEFF = 1 + 2 · Σ_{k=1}^{m-1} (1 − k/m) · rho_k

    rho_k faltante se trata como 0 (hipótesis optimista: se avisa si ocurre).
    """
    if m <= 1:
        return 1.0
    s = 0.0
    for k in range(1, m):
        r = rho[k - 1] if k - 1 < len(rho) and np.isfinite(rho[k - 1]) else 0.0
        s += (1 - k / m) * r
    return max(1.0 + 2 * s, 1e-9)


def tabla_bandas(rho: np.ndarray, etiqueta: str) -> None:
    """Tamaño efectivo y ganancia por ancho de banda."""
    print(f"\n   {etiqueta}")
    print(f"   {'ancho m':>8} {'filas nominales':>16} {'DEFF':>7} "
          f"{'n efectivo':>12} {'ganancia':>10}")
    print("   " + "-" * 58)
    for m in ANCHOS:
        d = deff(rho, m)
        nef = N_FECHAS * m / d
        print(f"   {m:>8} {N_FECHAS * m:>16,} {d:>7.2f} "
              f"{nef:>12,.0f} {nef / N_FECHAS:>9.1f}x")


# ===========================================================================
def main() -> None:
    print(f"pandas {pd.__version__} | numpy {np.__version__}")
    rho_cond = None

    # ── A · rho no condicional ────────────────────────────────────────────
    _titulo("A · Autocorrelación de la serie de flujo neto diario (no condicional)")
    try:
        df = pd.read_parquet(RUTA_MATRIZ, columns=["banco", "h", "fecha_th", "target"])
        df = df[(df["banco"] == BANCO) & (df["h"] == df["h"].min())]
        serie = (df.dropna(subset=["target"])
                   .drop_duplicates("fecha_th")
                   .sort_values("fecha_th")["target"].to_numpy())
        print(f"   serie diaria reconstruida: {len(serie):,} días hábiles")
        print(f"   media={serie.mean():.3e}  sd={serie.std():.3e}")
        rho_unc = acf(serie, LAG_MAX)
        print(f"\n   {'lag':>5} {'rho':>8}      {'lag':>5} {'rho':>8}")
        for i in range(0, min(LAG_MAX, 10)):
            j = i + 10
            der = (f"      {j+1:>5} {rho_unc[j]:>8.3f}"
                   if j < LAG_MAX and np.isfinite(rho_unc[j]) else "")
            print(f"   {i+1:>5} {rho_unc[i]:>8.3f}{der}")
        tabla_bandas(rho_unc, "Con rho NO condicional (cota superior):")
    except FileNotFoundError:
        print(f"   [AVISO] no se encontró {RUTA_MATRIZ}")
        rho_unc = None
    except Exception as e:
        print(f"   [AVISO] {type(e).__name__}: {e}")
        rho_unc = None

    # ── B · rho condicional, desde los residuos del modelo ────────────────
    _titulo("B · Correlación de los RESIDUOS del modelo entre horizontes")
    dir_fold = DIR_WFCV / f"fold_{MODO}"
    paths = sorted(dir_fold.glob(f"preds_base_{BANCO}_*.parquet"))
    if not paths:
        print(f"   [AVISO] sin preds_base en {dir_fold}")
        print("           corre step005 primero, o ajusta MODO")
    else:
        p = paths[-1]
        print(f"   leyendo {p.name}")
        pr = pd.read_parquet(p, columns=["fold", "fecha_t", "h", "target", "q50"])
        pr = pr.dropna(subset=["target", "q50"])
        pr["resid"] = pr["target"] - pr["q50"]
        print(f"   {len(pr):,} filas | folds {sorted(pr['fold'].unique())}")

        piv = pr.pivot_table(index=["fold", "fecha_t"], columns="h", values="resid")
        hs = sorted(c for c in piv.columns)
        print(f"   matriz residuos: {piv.shape[0]:,} fechas × {len(hs)} horizontes")

        # rho_k = correlación media entre columnas separadas k horizontes
        rho_cond = np.full(LAG_MAX, np.nan)
        for k in range(1, LAG_MAX + 1):
            vals = []
            for h in hs:
                if h + k in piv.columns:
                    a, b = piv[h].to_numpy(), piv[h + k].to_numpy()
                    m = np.isfinite(a) & np.isfinite(b)
                    if m.sum() > 30:
                        vals.append(np.corrcoef(a[m], b[m])[0, 1])
            if vals:
                rho_cond[k - 1] = float(np.mean(vals))

        print(f"\n   {'lag':>5} {'rho':>8}      {'lag':>5} {'rho':>8}")
        for i in range(0, min(LAG_MAX, 10)):
            j = i + 10
            der = (f"      {j+1:>5} {rho_cond[j]:>8.3f}"
                   if j < LAG_MAX and np.isfinite(rho_cond[j]) else "")
            print(f"   {i+1:>5} {rho_cond[i]:>8.3f}{der}")
        tabla_bandas(rho_cond, "Con rho CONDICIONAL (el que decide):")

    # ── C · las bandas propuestas ─────────────────────────────────────────
    rho = rho_cond if rho_cond is not None else rho_unc
    if rho is None:
        print("\n[AVISO] sin rho medido — no se puede evaluar la partición")
        return

    fuente = "condicional" if rho_cond is not None else "NO condicional (cota superior)"
    _titulo(f"C · Partición propuesta, con rho {fuente}")
    print(f"   {'banda':>10} {'m':>4} {'filas nom.':>12} {'DEFF':>7} "
          f"{'n efectivo':>12} {'obs cola 1%':>12}")
    print("   " + "-" * 64)
    for lo, hi in BANDAS:
        m = hi - lo + 1
        d = deff(rho, m)
        nef = N_FECHAS * m / d
        print(f"   {f'{lo}-{hi}':>10} {m:>4} {N_FECHAS * m:>12,} {d:>7.2f} "
              f"{nef:>12,.0f} {nef * 0.01:>12.1f}")

    # ── Veredicto ─────────────────────────────────────────────────────────
    _titulo("VEREDICTO")
    r1 = rho[0] if np.isfinite(rho[0]) else float("nan")
    g5 = 5 / deff(rho, 5)
    print(f"   rho al lag 1 ({fuente}) : {r1:+.3f}")
    print(f"   ganancia con bandas de 5 : {g5:.1f}x\n")
    if not np.isfinite(r1):
        print("   No se pudo estimar rho.")
    elif g5 >= 3.0:
        print("   AGRUPAR VALE LA PENA. La correlación es baja y las réplicas")
        print("   aportan información casi independiente. Bandas de 5-10.")
    elif g5 >= 2.0:
        print("   AGRUPAR AYUDA, con moderación. Bandas de 5-10; ensancharlas")
        print("   más rinde poco (ver tabla C).")
    elif g5 >= 1.5:
        print("   GANANCIA MODESTA. Habría que sopesarla contra el costo:")
        print("   se pierde especialización por horizonte, los diagnósticos")
        print("   bajan de resolución y h entra como feature dominante.")
    else:
        print("   NO COMPENSA. Las réplicas son casi redundantes: agrupar")
        print("   multiplica filas sin añadir información. Conviene atacar")
        print("   el problema por otra vía.")

    if rho_cond is None:
        print("\n   NOTA: esto usa rho NO condicional, que sobreestima la")
        print("   dependencia. El número real sería mejor. Corre step005 y")
        print("   repite para obtener el rho condicional de la sección B.")


if __name__ == "__main__":
    main()
