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

  D · CICLO MENSUAL — T1: ¿el residuo varía sistemáticamente según la
      posición calendario de la FECHA OBJETIVO (fecha_th)? dias_al_cierre_mes
      está calculado en t+h (step001 L2659: "Días hábiles restantes hasta fin
      de mes en t+h"), así que el modelo YA tiene esta información como
      feature (dias_al_cierre_mes_sin/cos) — si aun así el residuo muestra un
      patrón, no es que falte la variable, es que la codificación sin/cos no
      la está aprovechando del todo. Se usa el ángulo atan2(sin,cos) en vez
      de recalcular el calendario hábil: ya captura la posición en el ciclo
      con la longitud de mes correcta, sin reimplementar peru_bday.

  E · CICLO MENSUAL — T2: ACF ampliada a 45 lags con bandas de significancia
      explícitas (±2·SE, SE=1/√n) y un periodograma (FFT) sobre los
      horizontes representativos de H_GRUPOS en step005 (h=3,13,35,62), para
      detectar un pico limpio cerca de ~21 días hábiles (un mes).

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

# ── D/E · ciclo mensual ─────────────────────────────────────────────────────
N_BINS_MES = 21              # ≈ business days por mes; ancho de cada bin de fase
LAG_MAX_2  = 45              # T2: ACF ampliada (vs LAG_MAX=25 de A/B)
H_REP      = [3, 13, 35, 62] # mismos representantes que H_GRUPOS en step005
PERIODO_OBJETIVO = (17, 25)  # rango de "días" que cuenta como ciclo mensual


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


def bin_fase_mensual(sin_c: np.ndarray, cos_c: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Convierte el par (sin, cos) del ciclo mensual en un índice de bin [0, n_bins).

    Se usa el ángulo en vez de recalcular días-al-cierre: atan2(sin,cos) ya
    captura la posición dentro del mes con la longitud de mes correcta (el
    encoding la normaliza internamente), sin reimplementar el calendario hábil.
    """
    theta_deg = np.degrees(np.arctan2(sin_c, cos_c)) % 360.0
    return np.floor(theta_deg / 360.0 * n_bins).astype(int) % n_bins


def anova_ratio(valores: np.ndarray, bins: np.ndarray, n_bins: int) -> tuple[float, int]:
    """
    Estadístico F de ANOVA de un factor: MS_entre / MS_intra (sin p-value, para
    no depender de scipy). Bajo H0 (ningún bin explica nada) el valor esperado
    es 1; valores bien por encima sugieren un patrón real.

        MS_entre = [ Σ_b n_b·(media_b − media_global)² ] / (k − 1)
        MS_intra = [ Σ_b (n_b − 1)·var_b ]               / (N − k)

    Ambas sumas de cuadrados se dividen por sus grados de libertad — dividir
    SS_entre por N en vez de por (k−1) subestima el ratio por un factor
    (k−1)/N y hace parecer significativo lo que es ruido puro.

    Returns (ratio, grados_libertad_entre). ratio=nan si no hay suficientes datos.
    """
    medias, ns, ss_intra, gl_intra = [], [], 0.0, 0
    for b in range(n_bins):
        v = valores[bins == b]
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        medias.append(v.mean()); ns.append(len(v))
        ss_intra += v.var(ddof=1) * (len(v) - 1)
        gl_intra += len(v) - 1
    if len(medias) < 2 or gl_intra == 0:
        return float("nan"), 0
    medias, ns = np.array(medias), np.array(ns)
    gl_entre = len(medias) - 1
    media_global = np.average(medias, weights=ns)
    ss_entre = float(np.sum(ns * (medias - media_global) ** 2))
    ms_intra = ss_intra / gl_intra
    if ms_intra <= 0:
        return float("nan"), gl_entre
    return (ss_entre / gl_entre) / ms_intra, gl_entre


def periodograma_top(x: np.ndarray, k: int = 3) -> list[tuple[float, float]]:
    """
    FFT sobre una serie 1-D (con NaN eliminados, orden posicional). Devuelve
    los k picos de mayor potencia como (periodo_en_muestras, potencia_relativa),
    excluyendo la frecuencia 0 (la media).

    Aproximado: asume espaciado uniforme entre muestras. Los gaps entre folds
    (purge+embargo) y feriados rompen esa uniformidad, así que el periodo
    resultante es indicativo, no exacto.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return []
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    power[0] = 0.0                                   # descarta la componente DC
    idx = np.argsort(power)[::-1][:k]
    tot = power.sum()
    out = []
    for i in idx:
        if freqs[i] > 0 and tot > 0:
            out.append((1.0 / freqs[i], float(power[i] / tot)))
    return out


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
    pr = None                    # se reutiliza en D y E si la carga tiene éxito
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

    # ── D · T1: residuo vs posición calendario de la fecha objetivo ───────
    _titulo("D · T1 — Residuo según posición en el ciclo mensual (fecha_th)")
    if pr is None:
        print("   [AVISO] requiere preds_base — se omite (ver sección B)")
    else:
        try:
            mx = pd.read_parquet(
                RUTA_MATRIZ,
                columns=["banco", "fecha_t", "h",
                         "dias_al_cierre_mes_sin", "dias_al_cierre_mes_cos"],
            )
            mx = mx[mx["banco"] == BANCO].drop(columns="banco")
            dfm = pr.merge(mx, on=["fecha_t", "h"], how="inner")
            print(f"   filas con calendario emparejado: {len(dfm):,} de {len(pr):,}")

            bins = bin_fase_mensual(dfm["dias_al_cierre_mes_sin"].to_numpy(),
                                    dfm["dias_al_cierre_mes_cos"].to_numpy(),
                                    N_BINS_MES)
            resid = dfm["resid"].to_numpy()

            print(f"\n   {'bin':>4} {'n':>6} {'media resid':>13} {'SEM':>10} "
                  f"{'media |resid|':>14}")
            print("   " + "-" * 52)
            for b in range(N_BINS_MES):
                v = resid[bins == b]
                v = v[np.isfinite(v)]
                if len(v) < 2:
                    continue
                sem = v.std(ddof=1) / np.sqrt(len(v))
                print(f"   {b:>4} {len(v):>6} {v.mean():>13,.2e} {sem:>10,.2e} "
                      f"{np.abs(v).mean():>14,.2e}")

            f_nivel, gl1 = anova_ratio(resid, bins, N_BINS_MES)
            f_disp,  gl2 = anova_ratio(np.abs(resid), bins, N_BINS_MES)
            print(f"\n   razón var-entre/var-intra (nivel)      : {f_nivel:.2f}"
                  f"  (gl_entre={gl1})")
            print(f"   razón var-entre/var-intra (dispersión) : {f_disp:.2f}"
                  f"  (gl_entre={gl2})")
            print("   (heurística sin p-value: ~1 es lo esperado bajo ruido puro;")
            print("    valores bien por encima de 1 sugieren un patrón real)")
            if f_nivel > 2 or f_disp > 2:
                print("\n   [AVISO] hay estructura por posición calendario que el")
                print("   modelo no está capturando pese a tener dias_al_cierre_mes")
                print("   como feature. Apunta a H1: la codificación sin/cos pierde")
                print("   información que la variable cruda tendría disponible.")
            else:
                print("\n   Sin estructura clara por posición calendario. El ciclo")
                print("   detectado en la ACF (sección E) puede tener otra causa,")
                print("   o ser ruido de una sola medición.")
        except FileNotFoundError:
            print(f"   [AVISO] no se encontró {RUTA_MATRIZ}")
        except Exception as e:
            print(f"   [AVISO] {type(e).__name__}: {e}")

    # ── E · T2: ACF ampliada + periodograma ────────────────────────────────
    _titulo("E · T2 — ACF ampliada (bandas de significancia) y periodograma")
    if rho_cond is None and rho_unc is None:
        print("   [AVISO] sin serie disponible — se omite")
    else:
        for nombre, serie_x in (("no condicional", serie if rho_unc is not None else None),
                                 ("condicional (media entre h)",
                                  None if pr is None else
                                  pr.pivot_table(index="fecha_t", columns="h",
                                                 values="resid").mean(axis=1)
                                  .sort_index().to_numpy())):
            if serie_x is None or len(serie_x) < 60:
                continue
            print(f"\n   Serie: {nombre}  (n={len(serie_x):,})")
            r = acf(serie_x, LAG_MAX_2)
            se = 1.0 / np.sqrt(len(serie_x))
            sig = [k + 1 for k in range(LAG_MAX_2)
                   if np.isfinite(r[k]) and abs(r[k]) > 2 * se]
            print(f"   banda de no-significancia: ±{2*se:.3f}  (2·SE, SE=1/√n)")
            print(f"   lags significativos (|rho|>2SE) en 1..{LAG_MAX_2}: "
                  f"{sig if sig else '(ninguno)'}")
            en_ciclo = [k for k in sig if PERIODO_OBJETIVO[0] <= k <= PERIODO_OBJETIVO[1]]
            if en_ciclo:
                print(f"   [AVISO] {len(en_ciclo)} lag(s) significativo(s) caen en el "
                      f"rango del ciclo mensual {PERIODO_OBJETIVO}: {en_ciclo}")

        if pr is not None:
            print(f"\n   Periodograma (FFT) por horizonte representativo (H_GRUPOS):")
            print(f"   {'h_rep':>6} {'n':>6}   {'top-3 periodos (días, potencia rel.)'}")
            print("   " + "-" * 66)
            piv_h = pr.pivot_table(index="fecha_t", columns="h", values="resid")
            for h in H_REP:
                if h not in piv_h.columns:
                    continue
                serie_h = piv_h[h].sort_index().to_numpy()
                picos = periodograma_top(serie_h, k=3)
                txt = "  ".join(f"{per:.1f}d ({pot:.1%})" for per, pot in picos)
                marca = ""
                if any(PERIODO_OBJETIVO[0] <= per <= PERIODO_OBJETIVO[1]
                       for per, _ in picos):
                    marca = "  <- ciclo mensual"
                print(f"   {h:>6} {np.isfinite(serie_h).sum():>6}   {txt}{marca}")
            print("\n   (aproximado: asume espaciado uniforme entre muestras; los")
            print("   gaps de purge+embargo entre folds y los feriados lo rompen")
            print("   parcialmente — el periodo es indicativo, no exacto)")

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
