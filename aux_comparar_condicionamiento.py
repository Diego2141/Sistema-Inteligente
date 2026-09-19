# -*- coding: utf-8 -*-
"""
aux_comparar_condicionamiento.py — ¿regimen (HMM) o calendario (4 baldes)?

    runfile("aux_comparar_condicionamiento.py")

Compara dos corridas de step006 que difieren SOLO en CONDICIONAR_POR, sobre el
mismo BANCO y la misma geometria de fold.

DOS FASES, y la primera no es tramite
  Fase 0 — INSPECCION. Antes de calcular una sola metrica, verifica que los dos
           parquet sean comparables. Tres de los chequeos son BLOQUEANTES porque
           invalidan todo lo que viene despues, y ninguno se detecta mirando el
           esquema (aux_verificar_estructura_paths ya cubre esa parte):

             - el mismo y_realizado_acum donde se solapan. Si difiere, las dos
               corridas vieron datos distintos y no hay comparacion posible.
             - percentil_acum NO identico. Si lo es, corriste el mismo modo dos
               veces: el guard de CONDICIONAR_POR no disparo, o "auto" leyo el
               mismo preds_test. Es un falso negativo perfecto — el script
               reportaria "empate" y el empate seria un artefacto.
             - interseccion no vacia de (fecha_t, ventana, tau).

  Fase 1 — METRICAS, sobre la interseccion.

POR QUE NO ALCANZA CON EL COVERAGE
Los origenes son diarios y las ventanas llegan a 75 dias, asi que las
observaciones se solapan: las independientes son ~n_origenes/ventana. Con 476
origenes, a v=63 son 7.6. Para separar una tasa de excedencia del 5% de una del
10% con 80% de potencia hacen falta ~185. No falta un poco: falta un orden de
magnitud.

La raiz es que el coverage tira la magnitud — convierte cada observacion en un
bit. La metrica principal de este script es el PINBALL LOSS, que es la regla de
puntuacion estrictamente propia para un cuantil y usa la magnitud entera.

Y el ranking por ANCHO DE BANDA esta deliberadamente excluido: en modo
calendario la secuencia de baldes es determinista (secuencia_regimen con
baldes_fijos), o sea una fuente de varianza Monte Carlo MENOS por construccion.
Sus bandas salen mas angostas aunque el modelo no sea mejor. Se reporta como
diagnostico, nunca como criterio.
"""

# PEP 604 (`list | None`) necesita Python 3.10+; esto difiere la evaluacion
# de las anotaciones y lo vuelve valido desde 3.7. Todos los archivos del
# repo la traen — sin ella el modulo ni siquiera importa en el entorno de
# Spyder, que es mas viejo.
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    from aux_verificar_estructura_paths import (
        COLUMNAS_CONTRATO, resolver_ruta, describir, comparar_estructura,
    )
except ImportError as e:
    raise ImportError(
        "Falta aux_verificar_estructura_paths.py en la misma carpeta — de ahi "
        "salen resolver_ruta y la comparacion de esquema, que no se duplican aca."
    ) from e

###############################################################################
# Configuración
###############################################################################

BASE = (r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\2. Output"
        r"\step006_simulacion\xgb_qt_expanding_310.5\CONJUNTO_BBVA_1_0.5")

# {etiqueta: ruta}. La etiqueta se usa en todos los reportes.
# El de regimen conserva el nombre viejo (es anterior a sufijo_config); el de
# calendario ya lo lleva. resolver_ruta tolera las dos formas.
CORRIDAS = {
    "regimen":    rf"{BASE}\simulacion_paths_CONJUNTO_BBVA.parquet",
    "calendario": rf"{BASE}\cond_calendario"
                  rf"\simulacion_paths_CONJUNTO_BBVA_1_0.5_condcalendario.parquet",
}

# Cual es el modelo de REFERENCIA del skill score. El otro es el retador.
REFERENCIA = "regimen"

# ── Celdas PRE-ESPECIFICADAS ────────────────────────────────────────────────
# Son 74 ventanas x 9 taus = 666 comparaciones. Mirarlas todas y quedarse con
# las que ganan es pesca. Se fijan de antemano:
#   taus  -> la cola que define el requerimiento de liquidez (salida neta)
#   ventanas -> corto / un mes / un trimestre
# El resto se calcula igual y queda en el DataFrame de salida, pero el veredicto
# se lee en estas nueve celdas.
TAUS_FOCO    = [0.01, 0.05]
VENTANAS_FOCO = [5, 22, 63]

# Lag de la correccion Newey-West del Diebold-Mariano. None = la ventana, que es
# exactamente el solapamiento de las observaciones.
LAG_NW = None

ALPHA = 0.05          # nivel de los tests
TOL_REALIZADO = 1e-6  # tolerancia relativa al comparar y_realizado_acum

###############################################################################
# Contador de checklist
###############################################################################

_OK = _FALLA = 0
_BLOQUEANTES = []


def chk(nombre: str, cond: bool, detalle: str = "", bloqueante: bool = False) -> bool:
    global _OK, _FALLA
    if cond:
        _OK += 1
        print(f"[OK] {nombre}" + (f"  — {detalle}" if detalle else ""))
    else:
        _FALLA += 1
        print(f"[{'BLOQUEA' if bloqueante else 'FALLA'}] {nombre}"
              + (f"  — {detalle}" if detalle else ""))
        if bloqueante:
            _BLOQUEANTES.append(nombre)
    return cond


###############################################################################
# Métricas
###############################################################################

def pinball(y, q, tau):
    """
    Perdida cuantilica (check loss):

        L_tau(y,q) = (y - q) * (tau - 1{y < q})

    Estrictamente PROPIA para el cuantil tau: su minimizador en esperanza es el
    cuantil verdadero, no otra cosa. De ahi que no se pueda ganar ni ensanchando
    ni angostando la banda — a diferencia del ancho, que premia angostar, y del
    coverage, que es insensible a por CUANTO se rompio el piso.
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    return (y - q) * (tau - (y < q).astype(float))


def var_newey_west(d, lag: int) -> float:
    """
    Varianza de la media de d con kernel de Bartlett hasta `lag`.

        var = gamma_0 + 2 * sum_{k=1}^{L} (1 - k/(L+1)) * gamma_k

    Sin esto, un t-test sobre observaciones que comparten 74 de 75 dias de
    ventana declara significancia sobre ruido: a v=75 los 476 origenes son ~6
    observaciones independientes, y el error estandar ingenuo sale ~8x mas chico
    de lo que debe.

    El kernel de Bartlett garantiza var >= 0 en teoria; en muestras chicas puede
    dar <= 0 igual. En ese caso se cae a gamma_0 y se avisa con el flag.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 3:
        return np.nan
    dc = d - d.mean()
    g0 = float(dc @ dc) / n
    tot = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(dc[k:] @ dc[:-k]) / n
        tot += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return tot if tot > 0 else g0


def diebold_mariano(d, lag: int) -> dict:
    """
    H0: E[L_retador - L_referencia] = 0, con errores estandar HAC.

    Negativo y significativo -> el RETADOR pierde menos, o sea gana.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"n": n, "dbar": np.nan, "dm": np.nan, "p": np.nan,
                "lag": lag, "n_efectivo": np.nan}
    v = var_newey_west(d, lag)
    dbar = float(d.mean())
    se = np.sqrt(v / n) if (np.isfinite(v) and v > 0) else np.nan
    dm = dbar / se if (se and np.isfinite(se) and se > 0) else np.nan
    # p bilateral por normal asintotica
    from scipy.stats import norm
    p = float(2 * (1 - norm.cdf(abs(dm)))) if np.isfinite(dm) else np.nan
    # Inflacion del error estandar respecto del ingenuo: es la medida directa de
    # cuanta independencia se perdio por el solapamiento.
    se_naif = float(d.std(ddof=1) / np.sqrt(n))
    return {"n": n, "dbar": dbar, "dm": dm, "p": p, "lag": lag,
            "inflacion_se": se / se_naif if (se and se_naif > 0) else np.nan}


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """IC de Wilson para una proporcion. Mejor que el normal con k chico."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(c - h, 0.0), min(c + h, 1.0))


def rank_histogram(df_ancho: pd.DataFrame, taus: list) -> tuple:
    """
    PIT discreto: ¿en cual de los 10 intervalos que definen los 9 cuantiles cae
    el realizado?

    Bajo calibracion perfecta, la frecuencia de cada bin debe igualar su ancho
    nominal: (0.01, 0.04, 0.05, 0.15, 0.25, 0.25, 0.15, 0.05, 0.04, 0.01) para
    la grilla de taus de step006.

    POR QUE ESTE Y NO ANDERSON-DARLING: el A² de step006 necesita el PIT
    CONTINUO, que se calcula sobre las trayectorias simuladas. El parquet solo
    guarda 9 cuantiles, asi que desde aca el PIT es necesariamente discreto. Es
    menos potente, pero es lo que los datos permiten — y no se lo disfraza de A².

    El chi2 se calcula sobre el n EFECTIVO (n/ventana), no sobre n: los bins de
    dias consecutivos estan correlacionados por el solapamiento y el chi2 crudo
    rechazaria siempre.
    """
    cols = [f"q{int(round(t * 100)):02d}" for t in taus]
    Q = df_ancho[cols].values
    y = df_ancho["y_realizado"].values
    ok = np.isfinite(y) & np.isfinite(Q).all(axis=1)
    Q, y = Q[ok], y[ok]
    if len(y) == 0:
        return None, np.nan, np.nan
    # bin = cuantos cuantiles quedaron por debajo del realizado -> 0..len(taus)
    bins = (Q < y[:, None]).sum(axis=1)
    obs = np.bincount(bins, minlength=len(taus) + 1).astype(float)
    bordes = np.concatenate([[0.0], np.asarray(taus, dtype=float), [1.0]])
    nominal = np.diff(bordes)
    return obs / obs.sum(), nominal, len(y)


def chi2_efectivo(frec, nominal, n_efectivo) -> tuple:
    """chi2 de bondad de ajuste con el n efectivo. Conservador a proposito."""
    from scipy.stats import chi2 as _chi2
    if frec is None or not np.isfinite(n_efectivo) or n_efectivo < 5:
        return np.nan, np.nan
    esp = nominal * n_efectivo
    obs = frec * n_efectivo
    usable = esp >= 1.0        # bins con esperado < 1 rompen la aproximacion
    if usable.sum() < 2:
        return np.nan, np.nan
    stat = float(((obs[usable] - esp[usable]) ** 2 / esp[usable]).sum())
    gl = int(usable.sum() - 1)
    return stat, float(1 - _chi2.cdf(stat, gl))


def kupiec_lr(k: int, n: float, tau: float) -> tuple:
    """
    Test de cobertura incondicional de Kupiec (razon de verosimilitud).

        LR_uc = 2 * [ k*ln(pi/tau) + (n-k)*ln((1-pi)/(1-tau)) ],   pi = k/n

    Asintoticamente chi2(1). Es el test estandar de "la tasa empirica es tau".

    Se evalua sobre el n EFECTIVO (n/ventana), no sobre n: con observaciones que
    comparten 74 de 75 dias de ventana, el n nominal miente por un factor de ~8
    y el LR rechazaria practicamente siempre. Escalar k y n por igual mantiene la
    proporcion y corrige los grados de informacion, que es lo unico que el test
    necesita.
    """
    from scipy.stats import chi2 as _chi2
    if not np.isfinite(n) or n < 5:
        return np.nan, np.nan
    pi = k / n
    if pi <= 0:
        lr = 2.0 * n * np.log(1.0 / (1.0 - tau))
    elif pi >= 1:
        lr = 2.0 * n * np.log(1.0 / tau)
    else:
        lr = 2.0 * (k * np.log(pi / tau) + (n - k) * np.log((1 - pi) / (1 - tau)))
    return float(lr), float(1 - _chi2.cdf(lr, 1))


def submuestras_no_solapadas(exc: np.ndarray, ventana: int) -> dict:
    """
    Tasa de excedencia sobre submuestras SIN solapamiento: se toma un origen cada
    `ventana` dias, con los `ventana` desfases posibles.

    Por que vale la pena ademas del n efectivo: escalar n por 1/ventana es una
    correccion de grados de informacion, pero sigue siendo una aproximacion sobre
    la muestra solapada. Cada submuestra de aca es genuinamente independiente —
    dos origenes separados por `ventana` dias habiles no comparten ni un dia de
    acumulacion. El rango entre desfases muestra cuanta de la tasa observada es
    sensible a por donde se corta, que es la variabilidad real que el n nominal
    esconde.

    Es descriptivo: los desfases comparten la misma serie, asi que su dispersion
    no es un error estandar. Se reporta como rango, no como IC.
    """
    exc = np.asarray(exc, dtype=float)
    v = max(int(ventana), 1)
    tasas = [exc[off::v].mean() for off in range(v) if len(exc[off::v]) >= 5]
    if not tasas:
        return {"media": np.nan, "min": np.nan, "max": np.nan, "n_sub": 0,
                "n_por_sub": 0}
    return {"media": float(np.mean(tasas)), "min": float(np.min(tasas)),
            "max": float(np.max(tasas)), "n_sub": len(tasas),
            "n_por_sub": int(np.ceil(len(exc) / v))}


def cobertura_intervalo(m: pd.DataFrame, etq: str, ventana: int,
                        tau_lo: float, tau_hi: float) -> dict:
    """
    Cobertura del INTERVALO central [q_lo, q_hi]: fraccion de origenes en que el
    realizado cayo dentro.

    Es el numero que el proyecto viene reportando (88.8% en FOCO_GLOBALES, 83.0%
    en RESTO_GLOBALES contra 90% nominal), y no se deriva de las dos tasas de una
    cola: un modelo puede tener las dos colas mal en la MISMA direccion —corrido,
    no mal escalado— y aun asi cubrir el 90%. Por eso van las dos vistas.
    """
    sub = m[m["ventana"] == ventana]
    lo = sub[sub["tau"] == tau_lo].set_index("fecha_t")[f"percentil_acum_{etq}"]
    hi = sub[sub["tau"] == tau_hi].set_index("fecha_t")[f"percentil_acum_{etq}"]
    y = sub.drop_duplicates("fecha_t").set_index("fecha_t")["y_realizado_acum"]
    idx = lo.index.intersection(hi.index).intersection(y.dropna().index)
    if len(idx) < 10:
        return {"n": len(idx), "cobertura": np.nan}
    dentro = ((y[idx] >= lo[idx]) & (y[idx] <= hi[idx])).values
    n, k = len(idx), int(dentro.sum())
    n_ef = n / max(ventana, 1)
    lo_w, hi_w = wilson(int(round(k * n_ef / n)), int(round(n_ef)))
    nominal = tau_hi - tau_lo
    return {"n": n, "n_ef": n_ef, "cobertura": k / n, "nominal": nominal,
            "ic_lo": lo_w, "ic_hi": hi_w, "cubre": bool(lo_w <= nominal <= hi_w),
            "brecha": k / n - nominal}


def reporte_cobertura(m: pd.DataFrame, etqs: list, taus_disp: list) -> pd.DataFrame:
    """Seccion de cobertura: por cola, por intervalo, y estructura de plazos."""
    ref, ret = etqs
    ventanas = sorted(m["ventana"].unique().tolist())

    print("\n" + "=" * 78)
    print("COBERTURA")
    print("=" * 78)

    # ── A. Tasa de excedencia por cola, en las celdas pre-especificadas ──────
    print("\n--- A. EXCEDENCIA de la cola inferior (romper el piso de estres) ---")
    print("    tasa = P(realizado < percentil). Bien calibrado => tasa ~ tau.")
    print("    IC de Wilson y Kupiec sobre el n EFECTIVO (n/ventana).")
    print("    submuestras: un origen cada `ventana` dias, los v desfases.\n")
    filas_a = []
    for v in VENTANAS_FOCO:
        for t in TAUS_FOCO:
            sub = (m[(m["ventana"] == v) & (m["tau"] == t)]
                   .dropna(subset=["y_realizado_acum"]).sort_values("fecha_t"))
            if len(sub) < 10:
                continue
            y = sub["y_realizado_acum"].values
            for e in etqs:
                exc = (y < sub[f"percentil_acum_{e}"].values)
                n, k = len(exc), int(exc.sum())
                n_ef = n / max(v, 1)
                k_ef = k * n_ef / n
                lo_w, hi_w = wilson(int(round(k_ef)), int(round(n_ef)))
                lr, p = kupiec_lr(k_ef, n_ef, t)
                ss = submuestras_no_solapadas(exc, v)
                filas_a.append({
                    "modelo": e, "tau": t, "ventana": v, "n": n,
                    "n_ef": round(n_ef, 1), "tasa": k / n,
                    "ic_lo": lo_w, "ic_hi": hi_w,
                    "cubre": bool(lo_w <= t <= hi_w),
                    "kupiec_p": p,
                    "sub_min": ss["min"], "sub_max": ss["max"],
                })
    df_a = pd.DataFrame(filas_a)
    if not df_a.empty:
        print(df_a.to_string(index=False, float_format=lambda x: f"{x:,.4g}"))

    # ── B. Cobertura del INTERVALO, estructura de plazos ────────────────────
    print("\n--- B. COBERTURA DEL INTERVALO central (el numero comparable al "
          "83.0% ya medido) ---")
    filas_b = []
    pares = [(0.05, 0.95, "90%"), (0.01, 0.99, "98%")]
    pares = [(a, b, etiq) for a, b, etiq in pares
             if a in taus_disp and b in taus_disp]
    # estructura de plazos legible: las de foco mas una grilla espaciada
    v_muestra = sorted(set(VENTANAS_FOCO) |
                       set(ventanas[::max(len(ventanas) // 8, 1)]) |
                       {ventanas[0], ventanas[-1]})
    for tau_lo, tau_hi, etiq in pares:
        print(f"\n  intervalo [{tau_lo:g}, {tau_hi:g}]  nominal {etiq}")
        print(f"  {'ventana':>7} {'n_ef':>6} "
              + " ".join(f"{e[:10]:>11}" for e in etqs)
              + f"  {'brecha nom':>12}")   # del RETADOR contra el nominal
        for v in v_muestra:
            r = {e: cobertura_intervalo(m, e, v, tau_lo, tau_hi) for e in etqs}
            if any(not np.isfinite(r[e].get("cobertura", np.nan)) for e in etqs):
                continue
            marca = {e: ("" if r[e]["cubre"] else " *") for e in etqs}
            print(f"  {v:>7} {r[etqs[0]]['n_ef']:>6.1f} "
                  + " ".join(f"{r[e]['cobertura']:>9.1%}{marca[e]:<2}" for e in etqs)
                  + f"  {r[ret]['brecha']:>+11.1%}")
            for e in etqs:
                filas_b.append({"modelo": e, "intervalo": etiq, "ventana": v,
                                **{k: val for k, val in r[e].items()}})
        print(f"  (* = el nominal {etiq} cae FUERA del IC de Wilson sobre n_ef)")

    df_b = pd.DataFrame(filas_b)

    # ── C. Resumen sobre TODAS las ventanas ─────────────────────────────────
    if pares:
        tau_lo, tau_hi, etiq = pares[0]
        print(f"\n--- C. RESUMEN sobre las {len(ventanas)} ventanas "
              f"(intervalo {etiq}) ---")
        for e in etqs:
            cobs = [cobertura_intervalo(m, e, v, tau_lo, tau_hi) for v in ventanas]
            cobs = [c for c in cobs if np.isfinite(c.get("cobertura", np.nan))]
            if not cobs:
                continue
            vals = np.array([c["cobertura"] for c in cobs])
            dentro = sum(c["cubre"] for c in cobs)
            # tendencia con el horizonte: si la cobertura se deteriora al alargar
            # el plazo, el problema esta en la acumulacion, no en las marginales.
            vs = np.array([v for v, c in zip(ventanas, cobs)])
            pend = float(np.polyfit(vs, vals, 1)[0]) * 10 if len(vals) > 2 else np.nan
            print(f"  {e:<11} media {vals.mean():6.1%}  "
                  f"rango [{vals.min():.1%}, {vals.max():.1%}]  "
                  f"nominal dentro del IC en {dentro}/{len(cobs)} ventanas  "
                  f"tendencia {pend:+.2%} cada 10 dias")
        print("\n  Los dos modelos heredan las MISMAS marginales de step005, asi "
              "que una brecha\n  de cobertura comun NO la explica el "
              "condicionamiento: phi y rho describen\n  dependencia, no ancho. "
              "Lo comparable entre modelos es la DIFERENCIA.")

    return df_a, df_b


###############################################################################
# Fase 0 — inspección
###############################################################################

def inspeccionar(dfs: dict) -> pd.DataFrame:
    """
    Verifica que los dos parquet sean comparables y devuelve la interseccion.

    Devuelve un DataFrame largo con columnas percentil_<etiqueta> y
    y_realizado_acum unica, o None si algo bloqueante fallo.
    """
    print("\n" + "=" * 78)
    print("FASE 0 — INSPECCION")
    print("=" * 78)

    etqs = list(dfs)
    a, b = dfs[etqs[0]], dfs[etqs[1]]

    # ── Esquema: se delega en la funcion que ya existe ───────────────────────
    difs = comparar_estructura(describir(a), describir(b))
    chk("esquema identico (columnas, orden, tipos, grilla de tau)", not difs,
        "; ".join(difs) if difs else "identicas", bloqueante=True)

    for e, df in dfs.items():
        d = describir(df)
        f0, f1, nf = d["fechas"]
        vs = sorted(df["ventana"].unique().tolist())
        n_nan = int(df["y_realizado_acum"].isna().sum())
        print(f"  {e:<11}: {len(df):>9,} filas | {nf} origenes "
              f"{f0:%Y-%m-%d}..{f1:%Y-%m-%d} | {len(vs)} ventanas | "
              f"y_realizado NaN: {n_nan:,}")

    # ── Interseccion ─────────────────────────────────────────────────────────
    clave = ["fecha_t", "ventana", "tau"]
    m = a.merge(b, on=clave, suffixes=(f"_{etqs[0]}", f"_{etqs[1]}"), how="inner")
    frac = len(m) / max(len(a), len(b))
    chk("la interseccion de (fecha_t, ventana, tau) no esta vacia", len(m) > 0,
        f"{len(m):,} filas ({frac:.1%} del mayor)", bloqueante=True)
    if len(m) == 0:
        return None
    if frac < 0.99:
        print(f"  [AVISO] se pierde {1-frac:.1%} de las filas al intersecar. En "
              f"modo calendario step005 excluye los meses incompletos del "
              f"etiquetado y _baldes_th devuelve -1 fuera del eje habil, asi que "
              f"algo de perdida es esperable.")

    # ── BLOQUEANTE 1: el mismo realizado ─────────────────────────────────────
    ya, yb = m[f"y_realizado_acum_{etqs[0]}"], m[f"y_realizado_acum_{etqs[1]}"]
    amb = ya.notna() & yb.notna()
    difr = np.abs(ya[amb] - yb[amb]) / np.maximum(np.abs(ya[amb]), 1e-9)
    n_dif = int((difr > TOL_REALIZADO).sum())
    chk("mismo y_realizado_acum donde se solapan", n_dif == 0,
        f"{n_dif:,} filas difieren (max rel {difr.max():.2e})" if len(difr)
        else "sin pares comparables", bloqueante=True)

    # ── BLOQUEANTE 2: las bandas SON distintas ───────────────────────────────
    pa, pb = m[f"percentil_acum_{etqs[0]}"], m[f"percentil_acum_{etqs[1]}"]
    iguales = int(np.isclose(pa, pb, rtol=1e-9, atol=1e-9).sum())
    chk("percentil_acum NO es identico entre las dos corridas",
        iguales < 0.99 * len(m),
        f"{iguales:,}/{len(m):,} celdas identicas ({iguales/len(m):.1%}) — si es "
        f"~100%, las dos corridas son el MISMO modo y el 'empate' seria un "
        f"artefacto", bloqueante=True)
    dif_rel = np.abs(pa - pb) / np.maximum(np.abs(pa), 1e-9)
    print(f"  diferencia relativa de las bandas: mediana {np.median(dif_rel):.2%}, "
          f"p90 {np.quantile(dif_rel, 0.90):.2%}, max {dif_rel.max():.2%}")

    # ── No bloqueantes ───────────────────────────────────────────────────────
    sin_real = int(m[f"y_realizado_acum_{etqs[0]}"].isna().sum())
    chk("hay realizado para puntuar", sin_real < len(m),
        f"{sin_real:,} filas sin y_realizado ({sin_real/len(m):.1%}) — se excluyen")

    # Cruces de cuantiles, en las dos
    for e in etqs:
        o = m.sort_values(clave)
        d = o.groupby(["fecha_t", "ventana"], sort=False)[f"percentil_acum_{e}"].diff()
        n_cruces = int((d < -1e-9).sum())
        chk(f"{e}: percentil_acum no decrece con tau", n_cruces == 0,
            f"{n_cruces} cruces")

    # Celdas pre-especificadas presentes
    faltan = [(v, t) for v in VENTANAS_FOCO for t in TAUS_FOCO
              if not ((m["ventana"] == v) & (m["tau"] == t)).any()]
    chk("las celdas pre-especificadas existen en la interseccion", not faltan,
        f"faltan {faltan}" if faltan else
        f"{len(VENTANAS_FOCO)}x{len(TAUS_FOCO)} celdas")

    # Se normaliza a un y_realizado unico (ya se verifico que coinciden)
    m["y_realizado_acum"] = m[f"y_realizado_acum_{etqs[0]}"]
    return m


###############################################################################
# Fase 1 — métricas
###############################################################################

def evaluar_celda(sub: pd.DataFrame, etqs: list, tau: float, ventana: int) -> dict:
    """Todas las metricas de UNA celda (tau, ventana)."""
    sub = sub.dropna(subset=["y_realizado_acum"]).sort_values("fecha_t")
    n = len(sub)
    if n < 10:
        return {"tau": tau, "ventana": ventana, "n": n}

    y = sub["y_realizado_acum"].values
    ref, ret = etqs[0], etqs[1]
    L = {e: pinball(y, sub[f"percentil_acum_{e}"].values, tau) for e in etqs}

    lag = ventana if LAG_NW is None else LAG_NW
    dm = diebold_mariano(L[ret] - L[ref], lag)

    fila = {
        "tau": tau, "ventana": ventana, "n": n,
        "n_independiente": n / max(ventana, 1),
        f"pinball_{ref}": float(np.mean(L[ref])),
        f"pinball_{ret}": float(np.mean(L[ret])),
        # skill > 0 -> el retador pierde menos
        "skill": 1 - float(np.mean(L[ret])) / float(np.mean(L[ref]))
                 if np.mean(L[ref]) != 0 else np.nan,
        "dm_stat": dm["dm"], "dm_p": dm["p"], "dm_lag": dm["lag"],
        "inflacion_se": dm.get("inflacion_se", np.nan),
    }

    # Coverage por cola, con Wilson sobre el n EFECTIVO
    for e in etqs:
        q = sub[f"percentil_acum_{e}"].values
        exc = (y < q) if tau < 0.5 else (y > q)
        k, n_ef = int(exc.sum()), n / max(ventana, 1)
        fila[f"cobertura_{e}"] = k / n
        lo, hi = wilson(int(round(k * n_ef / n)), int(round(n_ef)))
        fila[f"cubre_tau_{e}"] = bool(lo <= tau <= hi)
        # Shortfall condicional: cuando se rompe el piso, ¿por cuanto?
        fila[f"shortfall_{e}"] = float(np.mean(np.abs(y[exc] - q[exc]))) if k else 0.0
    return fila


def main():
    print("=" * 78)
    print("COMPARACION DE CONDICIONAMIENTO — regimen vs calendario")
    print("=" * 78)

    rutas = {e: resolver_ruta(r) for e, r in CORRIDAS.items()}
    faltan = []
    for e, r in rutas.items():
        try:
            if not r.exists():
                faltan.append((e, r))
        except OSError:
            faltan.append((e, r))
    if faltan:
        print("\nNo se pudo leer:")
        for e, r in faltan:
            print(f"  - {e}: {r}")
        print("\nSin acceso a H: se corre el AUTOTEST, que valida la logica.")
        return autotest()

    dfs = {}
    for e, r in rutas.items():
        df = pd.read_parquet(r)
        falt = [c for c in COLUMNAS_CONTRATO if c not in df.columns]
        if falt:
            raise ValueError(f"{r.name}: faltan columnas {falt}")
        df["fecha_t"] = pd.to_datetime(df["fecha_t"])
        dfs[e] = df
        print(f"  {e:<11} <- {r.name}")

    m = inspeccionar(dfs)
    if _BLOQUEANTES:
        print(f"\n{'=' * 78}\nABORTA: {len(_BLOQUEANTES)} chequeo(s) bloqueante(s) "
              f"— {_BLOQUEANTES}\nLas metricas no se calculan porque no serian "
              f"interpretables.\n{'=' * 78}")
        return 1

    # El orden importa: la referencia primero, el retador segundo.
    etqs = [REFERENCIA] + [e for e in dfs if e != REFERENCIA]
    ref, ret = etqs

    print("\n" + "=" * 78)
    print(f"FASE 1 — METRICAS   (referencia={ref}, retador={ret})")
    print("=" * 78)

    filas = []
    for v in sorted(m["ventana"].unique()):
        for t in sorted(m["tau"].unique()):
            sub = m[(m["ventana"] == v) & (m["tau"] == t)]
            f = evaluar_celda(sub, etqs, float(t), int(v))
            if "skill" in f:
                filas.append(f)
    res = pd.DataFrame(filas)

    foco = res[res["tau"].isin(TAUS_FOCO) & res["ventana"].isin(VENTANAS_FOCO)]
    print("\n--- CELDAS PRE-ESPECIFICADAS (el veredicto se lee aca) ---")
    print(f"    skill > 0  =>  {ret} pierde menos  |  dm_p < {ALPHA} => significativo\n")
    cols = ["tau", "ventana", "n", "n_independiente", f"pinball_{ref}",
            f"pinball_{ret}", "skill", "dm_stat", "dm_p", "inflacion_se"]
    print(foco[cols].to_string(index=False,
          float_format=lambda x: f"{x:,.4g}"))

    print("\n--- COBERTURA en las mismas celdas (diagnostico, NO ranking) ---")
    cols_c = ["tau", "ventana", f"cobertura_{ref}", f"cobertura_{ret}",
              f"cubre_tau_{ref}", f"cubre_tau_{ret}",
              f"shortfall_{ref}", f"shortfall_{ret}"]
    print(foco[cols_c].to_string(index=False,
          float_format=lambda x: f"{x:,.4g}"))

    # ── Rank histogram por ventana de foco ───────────────────────────────────
    taus_all = sorted(m["tau"].unique().tolist())
    print("\n--- PIT DISCRETO (rank histogram) por ventana ---")
    for v in VENTANAS_FOCO:
        print(f"  ventana={v}")
        for e in etqs:
            sub = m[m["ventana"] == v]
            ancho = sub.pivot(index="fecha_t", columns="tau",
                              values=f"percentil_acum_{e}")
            ancho.columns = [f"q{int(round(t*100)):02d}" for t in ancho.columns]
            ancho["y_realizado"] = (sub.drop_duplicates("fecha_t")
                                    .set_index("fecha_t")["y_realizado_acum"])
            frec, nom, n_us = rank_histogram(ancho, taus_all)
            stat, p = chi2_efectivo(frec, nom, n_us / max(v, 1)) if frec is not None else (np.nan, np.nan)
            if frec is not None:
                # n_ef va SIEMPRE junto al p-valor. Sin el, el p se lee al
                # reves: crece con la ventana (0.38 -> 0.58 -> 0.69) mientras el
                # ajuste EMPEORA, porque n_ef = n/ventana se desploma y el test
                # pierde toda la potencia. Un p alto con n_ef=7 no dice "ajusta
                # bien", dice "no se puede saber".
                print(f"    {e:<11} n_ef={n_us/max(v,1):5.1f}  chi2_ef={stat:7.2f} "
                      f"p={p:6.3f}  frec={np.round(frec, 3).tolist()}")
                print(f"    {'':<11} {'':>26}  nom ={np.round(nom, 3).tolist()}")
        if n_us / max(v, 1) < 20:
            print(f"    [AVISO] v={v}: {n_us/max(v,1):.1f} observaciones "
                  f"independientes — el chi2 no tiene potencia, leelo como "
                  f"descriptivo y mira la FRECUENCIA, no el p.")

    # ── Cobertura: seccion propia ────────────────────────────────────────────
    df_cola, df_int = reporte_cobertura(m, etqs, taus_all)

    # ── Sharpness: se REPORTA, no se rankea ──────────────────────────────────
    print("\n--- ANCHO DE BANDA q05-q95 (diagnostico, NUNCA criterio) ---")
    print(f"    {ret} tiene una fuente de ruido Monte Carlo MENOS por "
          f"construccion\n    (secuencia de baldes determinista), asi que sale "
          f"mas angosto aunque no sea mejor.")
    for v in VENTANAS_FOCO:
        anchos = {}
        for e in etqs:
            lo = m[(m["ventana"] == v) & (m["tau"] == 0.05)][f"percentil_acum_{e}"]
            hi = m[(m["ventana"] == v) & (m["tau"] == 0.95)][f"percentil_acum_{e}"]
            anchos[e] = float((hi.values - lo.values).mean()) if len(lo) == len(hi) else np.nan
        rel = anchos[ret] / anchos[ref] - 1 if anchos[ref] else np.nan
        print(f"    v={v:<3} {ref}={anchos[ref]:,.1f}  {ret}={anchos[ret]:,.1f}  "
              f"({rel:+.1%})")

    # ── Veredicto ────────────────────────────────────────────────────────────
    gana = int((foco["skill"] > 0).sum())
    signif = foco[foco["dm_p"] < ALPHA]
    print("\n" + "=" * 78)
    print(f"VEREDICTO sobre {len(foco)} celdas pre-especificadas")
    print(f"  {ret} pierde menos en {gana}/{len(foco)} celdas "
          f"(skill medio {foco['skill'].mean():+.2%})")
    print(f"  significativas al {ALPHA:.0%}: {len(signif)}/{len(foco)}"
          + (f" — {ret} gana {int((signif['skill']>0).sum())}, "
             f"pierde {int((signif['skill']<=0).sum())}" if len(signif) else ""))
    if len(signif) == 0:
        print(f"  Ninguna diferencia significativa. Con {foco['n_independiente'].min():.0f}-"
              f"{foco['n_independiente'].max():.0f} observaciones independientes "
              f"por celda eso es\n  el resultado ESPERABLE, no un empate "
              f"demostrado: el test no tiene potencia\n  para descartar "
              f"diferencias chicas.")
    print("=" * 78)

    res.attrs["referencia"] = ref
    autotest()
    return {"celdas": res, "cobertura_cola": df_cola, "cobertura_intervalo": df_int}


###############################################################################
# Autotest sintético
###############################################################################

def _sintetico(n=400, ventana=22, sesgo=0.0, escala=1.0, seed=0):
    """
    Dos modelos sobre el MISMO realizado, con solapamiento real: y_acum es la
    suma movil de `ventana` dias, que es lo que hace que las observaciones
    consecutivas compartan informacion.
    """
    rng = np.random.default_rng(seed)
    diario = rng.normal(0, 1, n + ventana)
    y = np.array([diario[i:i + ventana].sum() for i in range(n)])
    sd = np.sqrt(ventana)
    from scipy.stats import norm
    taus = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    filas = []
    fechas = pd.bdate_range("2024-01-01", periods=n)
    for i, ft in enumerate(fechas):
        for t in taus:
            filas.append({"fecha_t": ft, "ventana": ventana, "tau": t,
                          "percentil_acum": norm.ppf(t) * sd * escala + sesgo,
                          "y_realizado_acum": y[i]})
    return pd.DataFrame(filas, columns=COLUMNAS_CONTRATO)


def autotest() -> int:
    print("\n" + "=" * 78)
    print("AUTOTEST SINTETICO (no necesita acceso a H:)")
    print("=" * 78)
    from scipy.stats import norm

    # 1. El pinball se minimiza en el cuantil VERDADERO (es propio).
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 200_000)
    tau = 0.05
    cand = np.linspace(norm.ppf(tau) - 0.5, norm.ppf(tau) + 0.5, 51)
    perd = [pinball(y, c, tau).mean() for c in cand]
    chk("autotest 1. el pinball se minimiza en el cuantil verdadero",
        abs(cand[int(np.argmin(perd))] - norm.ppf(tau)) < 0.05,
        f"argmin={cand[int(np.argmin(perd))]:.3f} vs {norm.ppf(tau):.3f}")

    # 2. Un modelo IDENTICO a si mismo: skill 0 y DM no significativo.
    a = _sintetico(seed=3)
    d = pinball(a["y_realizado_acum"], a["percentil_acum"], 0.05) * 0
    r = diebold_mariano(d, 22)
    chk("autotest 2. dos modelos identicos: diferencial nulo",
        not np.isfinite(r["dm"]) or abs(r["dbar"]) < 1e-12, f"dbar={r['dbar']:.2e}")

    # 3. [neg] un modelo claramente SESGADO tiene que perder, y detectarse.
    base = _sintetico(seed=3)
    malo = _sintetico(seed=3, sesgo=-8.0)
    sub_b = base[base["tau"] == 0.05].sort_values("fecha_t")
    sub_m = malo[malo["tau"] == 0.05].sort_values("fecha_t")
    y = sub_b["y_realizado_acum"].values
    Lb = pinball(y, sub_b["percentil_acum"].values, 0.05)
    Lm = pinball(y, sub_m["percentil_acum"].values, 0.05)
    r = diebold_mariano(Lm - Lb, 22)
    chk("autotest 3. [neg] un modelo sesgado pierde y el DM lo detecta",
        r["dbar"] > 0 and r["p"] < 0.05, f"dbar={r['dbar']:+.3f} p={r['p']:.4f}")

    # 4. [neg] LA CORRECCION HAC IMPORTA: sin ella, un diferencial nulo en media
    #    pero muy autocorrelacionado se declara significativo.
    rng = np.random.default_rng(7)
    e = rng.normal(0, 1, 500)
    d_ac = np.array([e[max(i-40, 0):i+1].mean() for i in range(500)])  # muy persistente
    naif = d_ac.mean() / (d_ac.std(ddof=1) / np.sqrt(len(d_ac)))
    hac = diebold_mariano(d_ac, 40)
    chk("autotest 4. la correccion HAC infla el error estandar frente al ingenuo",
        hac["inflacion_se"] > 2.0, f"inflacion={hac['inflacion_se']:.1f}x "
        f"(t ingenuo={naif:+.2f} -> DM={hac['dm']:+.2f})")

    # 5. var_newey_west con lag 0 es gamma_0
    x = rng.normal(0, 1, 100)
    chk("autotest 5. Newey-West con lag 0 == varianza simple",
        abs(var_newey_west(x, 0) - x.var()) < 1e-10)

    # 6. [neg] percentiles identicos se detectan como no comparables
    a = _sintetico(seed=5)
    m = a.merge(a, on=["fecha_t", "ventana", "tau"], suffixes=("_x", "_y"))
    ig = int(np.isclose(m["percentil_acum_x"], m["percentil_acum_y"]).sum())
    chk("autotest 6. [neg] dos corridas identicas dan ~100% de celdas iguales",
        ig >= 0.99 * len(m), f"{ig}/{len(m)} — es lo que el bloqueante 2 ataja")

    # 7. [neg] un realizado distinto se detecta
    b = _sintetico(seed=9)
    m = a.merge(b, on=["fecha_t", "ventana", "tau"], suffixes=("_x", "_y"))
    dif = int((np.abs(m["y_realizado_acum_x"] - m["y_realizado_acum_y"]) > 1e-6).sum())
    chk("autotest 7. [neg] un y_realizado distinto se detecta",
        dif > 0, f"{dif}/{len(m)} filas difieren")

    # 8. rank histogram de un modelo BIEN calibrado se parece al nominal
    a = _sintetico(n=2000, seed=11)
    taus = sorted(a["tau"].unique().tolist())
    anc = a.pivot(index="fecha_t", columns="tau", values="percentil_acum")
    anc.columns = [f"q{int(round(t*100)):02d}" for t in anc.columns]
    anc["y_realizado"] = a.drop_duplicates("fecha_t").set_index("fecha_t")["y_realizado_acum"]
    frec, nom, n_us = rank_histogram(anc, taus)
    chk("autotest 8. rank histogram de un modelo calibrado ~ nominal",
        float(np.abs(frec - nom).max()) < 0.05,
        f"max|frec-nom|={np.abs(frec-nom).max():.3f}")

    # 9. [neg] un modelo con bandas ANGOSTAS manda masa a los bins extremos
    est = _sintetico(n=2000, seed=11, escala=0.5)
    anc2 = est.pivot(index="fecha_t", columns="tau", values="percentil_acum")
    anc2.columns = [f"q{int(round(t*100)):02d}" for t in anc2.columns]
    anc2["y_realizado"] = est.drop_duplicates("fecha_t").set_index("fecha_t")["y_realizado_acum"]
    frec2, nom2, _ = rank_histogram(anc2, taus)
    chk("autotest 9. [neg] bandas angostas -> exceso en los bins extremos",
        (frec2[0] + frec2[-1]) > 3 * (nom2[0] + nom2[-1]),
        f"extremos {frec2[0]+frec2[-1]:.3f} vs nominal {nom2[0]+nom2[-1]:.3f}")

    # 10. Wilson: cobertura razonable con k chico
    lo, hi = wilson(1, 20)
    chk("autotest 10. Wilson con k=1,n=20 da un IC no degenerado",
        0 < lo < 0.05 < hi < 1, f"[{lo:.3f}, {hi:.3f}]")

    # ── Cobertura ────────────────────────────────────────────────────────────
    # 11. Kupiec NO rechaza cuando la tasa es la nominal.
    lr, p = kupiec_lr(5, 100, 0.05)
    chk("autotest 11. Kupiec con tasa == tau no rechaza",
        lr < 1e-9 and p > 0.99, f"LR={lr:.2e} p={p:.3f}")

    # 12. [neg] Kupiec SI rechaza con una tasa muy distinta y n suficiente.
    lr, p = kupiec_lr(25, 100, 0.05)
    chk("autotest 12. [neg] Kupiec rechaza 25% contra tau=5%",
        p < 0.001, f"LR={lr:.1f} p={p:.2e}")

    # 13. La correccion por n efectivo AFLOJA el test: la misma proporcion con
    #     n/22 no rechaza lo que con n nominal si. Es el punto de todo el
    #     tratamiento del solapamiento.
    _, p_nom = kupiec_lr(38, 476, 0.05)          # 8% con n nominal
    _, p_ef  = kupiec_lr(38 / 22, 476 / 22, 0.05)  # la MISMA proporcion, n efectivo
    chk("autotest 13. el n efectivo afloja Kupiec (no rechaza lo que el n "
        "nominal si)", p_nom < 0.05 < p_ef, f"p_nominal={p_nom:.4f} -> p_ef={p_ef:.3f}")

    # 14. Submuestras no solapadas: cantidad y tamaño correctos, y la media
    #     coincide con la tasa global cuando no hay estructura.
    exc = (rng.random(440) < 0.10)
    ss = submuestras_no_solapadas(exc, 22)
    chk("autotest 14. submuestras no solapadas: v desfases del tamaño correcto",
        ss["n_sub"] == 22 and ss["n_por_sub"] == 20
        and abs(ss["media"] - exc.mean()) < 1e-9,
        f"{ss['n_sub']} desfases de {ss['n_por_sub']}, media {ss['media']:.3f}")

    # 15. [neg] con violaciones EN RACIMO el rango entre desfases se abre — es
    #     justamente lo que el n nominal esconde.
    exc_racimo = np.zeros(440, dtype=bool)
    exc_racimo[::22] = True              # una por bloque, siempre en el mismo sitio
    ss_r = submuestras_no_solapadas(exc_racimo, 22)
    chk("autotest 15. [neg] violaciones en racimo abren el rango entre desfases",
        (ss_r["max"] - ss_r["min"]) > 0.9,
        f"rango [{ss_r['min']:.2f}, {ss_r['max']:.2f}] vs media {ss_r['media']:.2f}")

    print(f"\n{'=' * 78}")
    print(f"RESULTADO: {_OK} OK / {_FALLA} FALLA")
    print("=" * 78)
    return 1 if _FALLA else 0


if __name__ == "__main__":
    _res = main()
    if isinstance(_res, int):
        sys.exit(_res)
    # Al namespace GLOBAL: con runfile() de Spyder es lo que los deja en el
    # Variable Explorer para seguir explorandolos a mano.
    resultados     = _res["celdas"]               # pinball/DM por celda
    tabla_cola     = _res["cobertura_cola"]       # excedencia por tau
    tabla_intervalo = _res["cobertura_intervalo"]  # cobertura del intervalo central
