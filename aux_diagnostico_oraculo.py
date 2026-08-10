# -*- coding: utf-8 -*-
"""
aux_diagnostico_oraculo.py
==========================
¿La brecha de cobertura es un problema de CALIBRACIÓN o de HETEROCEDASTICIDAD
CONDICIONAL? Es la pregunta que decide si conviene seguir invirtiendo en
post-proceso (CQR, ACI) o volver a tocar el modelo.

La lógica
---------
Se construye un ORÁCULO: la mejor calibración global posible, estimada sobre el
propio TEST. Es deliberadamente tramposo — usa el futuro — y por eso mismo es
una COTA SUPERIOR de lo que cualquier método honesto puede lograr.

    · Si el oráculo global alcanza 90% Y la cobertura queda plana dentro de cada
      bin condicionante  ->  el problema es solo de nivel. ACI lo resuelve.

    · Si el oráculo global alcanza 90% en el agregado PERO la cobertura sigue
      variando fuerte entre bins  ->  hay heterocedasticidad condicional. Ningún
      desplazamiento ni escalamiento global la arregla, porque para cubrir los
      bins difíciles tiene que sobre-ensanchar los fáciles.

El oráculo llega a 90% exacto por construcción (dos parámetros libres, uno por
cola), así que el resultado informativo NO es el 90% — es la DISPERSIÓN que
queda entre bins después de aplicarlo.

Los dos oráculos
----------------
    Aditivo         Q05 + c_lo,  Q95 + c_hi
    Multiplicativo  Q50 - s_lo*(Q50-Q05),  Q50 + s_hi*(Q95-Q50)

El multiplicativo preserva la heterocedasticidad que el modelo SÍ capturó
(reescala la banda que el modelo ya predijo); el aditivo la destruye (mueve
todo lo mismo). Comparar ambos ya dice algo: si el multiplicativo cuesta menos
anchura para la misma cobertura, la señal de escala del modelo sirve.

El número que importa
---------------------
    ORÁCULO CONDICIONAL: reescala cada bin por separado. Alcanza 90% en todos
    los bins por construcción. La diferencia de anchura contra el oráculo global
    es EL COSTO EN MM USD de la heterocedasticidad no modelada.

Por qué se evalúan VARIAS dimensiones
-------------------------------------
La estructura condicional puede vivir en dimensiones distintas, y cada una
implica un remedio distinto. Mirar una sola produce falsos negativos: en una
prueba con datos sintéticos de régimen de volatilidad temporal, la dispersión
por decil de anchura fue 1.5% (invisible) mientras que por mes fue 22%.

    DIMENSIÓN            ¿condicionable ex-ante?   REMEDIO
    anchura predicha     SÍ (se conoce en t)       calibrar por bin de anchura
    horizonte h          SÍ (se conoce en t)       calibrar por h
    tiempo (mes)         NO (es el futuro)         ACI / feature de volatilidad
    fold                 NO                        idem — es tiempo agregado

Si la estructura está en anchura u h, se corrige con post-proceso condicional
y punto. Si está en el tiempo, hace falta adaptación online (ACI) o meterle al
modelo un feature que anticipe el régimen.

Inferencia
----------
Las filas NO son independientes: 74 horizontes comparten origen y los orígenes
vecinos comparten datos. Un intervalo binomial sobre n=44,000 sería absurdo. Se
usa bootstrap de bloques móviles sobre FECHAS DE ORIGEN (bloque de 22 dh = 1
mes), que respeta las dos fuentes de dependencia.

Salidas
-------
    2. Output/step006_cqr/cqr_<MODO>/
        diagnostico_oraculo_<BANCO>_<fecha>.png      6 paneles
        diagnostico_oraculo_<BANCO>_<fecha>.xlsx     tablas por bin

Uso: ajustar _MODO_SUFIJO y ejecutar.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_WFCV     = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"
DIR_CQR      = BASE_SISTEMA / "2. Output" / "step006_cqr"

_MODO_SUFIJO = "exp_arctan"
BANCO        = "SISTEMA"

DIR_FOLD = DIR_WFCV / f"fold_{_MODO_SUFIJO}"
DIR_OUT  = DIR_CQR  / f"cqr_{_MODO_SUFIJO}"

COBERTURA_OBJETIVO = 0.90      # intervalo Q05–Q95
N_DECILES          = 10        # bins de anchura predicha
BOOT_B             = 400       # réplicas del bootstrap
BOOT_BLOQUE        = 22        # días hábiles por bloque (≈ 1 mes)
SEMILLA            = 20260810

# Paleta validada con scripts/validate_palette.js (light, surface #fcfcfb):
# lightness PASS · chroma PASS · CVD ΔE 9.2 PASS · normal-vision ΔE 27.6 PASS
C_MODELO  = "#2a78d6"   # modelo crudo
C_ORACULO = "#eb6834"   # oráculo global
C_COND    = "#1baf7a"   # oráculo condicional (siempre con etiqueta directa)
C_REF     = "#898781"   # líneas de referencia — nunca lleva significado de serie
C_TINTA   = "#0b0b0b"
C_TINTA2  = "#52514e"
C_GRID    = "#e1e0d9"
SURFACE   = "#fcfcfb"

EPS = 1e-9


def _titulo(t: str) -> None:
    print(f"\n{'=' * 78}\n  {t}\n{'=' * 78}")


def _h_grupo(h):
    if h <= 5:  return "muy_corto"
    if h <= 20: return "corto"
    if h <= 50: return "medio"
    return "largo"


# ── 1. Datos ──────────────────────────────────────────────────────────────────
def cargar() -> pd.DataFrame:
    paths = sorted(DIR_FOLD.glob(f"preds_base_{BANCO}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Sin preds_base_{BANCO}_*.parquet en {DIR_FOLD}")
    df = pd.read_parquet(paths[-1])
    print(f"[OK] {paths[-1].name}: {len(df):,} filas")

    faltan = [c for c in ("target", "q05", "q50", "q95", "h", "fold", "fecha_t")
              if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas: {faltan}")

    df = df.dropna(subset=["target", "q05", "q50", "q95"]).copy()
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    # Semi-anchuras. El reordenamiento de step005 garantiza q05<=q50<=q95, pero
    # pueden quedar en cero tras un empate; el clip evita divisiones explosivas.
    df["w_lo"] = np.maximum(df["q50"] - df["q05"], EPS)
    df["w_hi"] = np.maximum(df["q95"] - df["q50"], EPS)
    df["ancho"] = df["q95"] - df["q05"]

    df["h_grupo"] = df["h"].apply(_h_grupo)
    print(f"     folds {sorted(df['fold'].unique())} | h {df['h'].min()}–{df['h'].max()} | "
          f"{df['fecha_t'].nunique():,} fechas de origen")
    return df


# ── 2. Oráculos ───────────────────────────────────────────────────────────────
def oraculo_multiplicativo(y, q50, w_lo, w_hi, alpha_cola):
    """
    Factores s_lo, s_hi tales que la cobertura de cada cola sea exactamente
    alpha_cola sobre ESTAS mismas filas.

        Y < Q50 - s*w_lo   <=>   (Q50-Y)/w_lo > s
        => P(cola baja) = alpha  <=>  s = cuantil_{1-alpha} de (Q50-Y)/w_lo

    Forma cerrada, sin optimización numérica.
    """
    u_lo = (q50 - y) / w_lo
    u_hi = (y - q50) / w_hi
    s_lo = float(np.quantile(u_lo, 1.0 - alpha_cola))
    s_hi = float(np.quantile(u_hi, 1.0 - alpha_cola))
    return s_lo, s_hi


def oraculo_aditivo(y, q05, q95, alpha_cola):
    """Desplazamientos c_lo, c_hi para cobertura exacta por cola."""
    c_lo = float(np.quantile(y - q05, alpha_cola))
    c_hi = float(np.quantile(y - q95, 1.0 - alpha_cola))
    return c_lo, c_hi


def aplicar_mult(q50, w_lo, w_hi, s_lo, s_hi):
    return q50 - s_lo * w_lo, q50 + s_hi * w_hi


# ── 3. Bootstrap de bloques sobre fechas de origen ────────────────────────────
def bootstrap_cobertura(df, dentro, bin_id, n_bins, B=BOOT_B, L=BOOT_BLOQUE,
                        semilla=SEMILLA):
    """
    IC percentil 90% de la cobertura por bin, vía bootstrap de bloques móviles
    sobre fechas de origen.

    Remuestrear fechas (no filas) preserva la dependencia entre los 74 horizontes
    de un mismo origen; los bloques de L fechas consecutivas preservan además la
    autocorrelación entre orígenes vecinos. Un IC binomial ignoraría ambas y
    saldría ~5x demasiado angosto.
    """
    rng = np.random.default_rng(semilla)

    fechas = np.sort(df["fecha_t"].unique())
    n_f    = len(fechas)
    codigo = pd.Series(np.arange(n_f), index=fechas)
    f_cod  = df["fecha_t"].map(codigo).values

    # Filas agrupadas por fecha (índices contiguos: df viene ordenado por fecha)
    orden  = np.argsort(f_cod, kind="stable")
    f_ord  = f_cod[orden]
    inicio = np.searchsorted(f_ord, np.arange(n_f), side="left")
    fin    = np.searchsorted(f_ord, np.arange(n_f), side="right")

    n_bloques = int(np.ceil(n_f / L))
    out = np.full((B, n_bins), np.nan)

    for b in range(B):
        arranques = rng.integers(0, max(1, n_f - L + 1), size=n_bloques)
        fechas_b  = np.concatenate([np.arange(a, min(a + L, n_f)) for a in arranques])
        filas     = np.concatenate([orden[inicio[f]:fin[f]] for f in fechas_b])

        n_tot = np.bincount(bin_id[filas], minlength=n_bins).astype(float)
        n_in  = np.bincount(bin_id[filas], weights=dentro[filas].astype(float),
                            minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[b] = np.where(n_tot > 0, n_in / n_tot, np.nan)

    lo = np.nanpercentile(out, 5, axis=0)
    hi = np.nanpercentile(out, 95, axis=0)
    return lo, hi


# ── 4. Cobertura por bin ──────────────────────────────────────────────────────
def cobertura_por_bin(dentro, bin_id, n_bins):
    n_tot = np.bincount(bin_id, minlength=n_bins).astype(float)
    n_in  = np.bincount(bin_id, weights=dentro.astype(float), minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_tot > 0, n_in / n_tot, np.nan), n_tot


# ── 5. Evaluación de una dimensión condicionante ──────────────────────────────
def evaluar_dimension(df, nombre, etiquetas, y, q50, w_lo, w_hi,
                      dentro_orc, dentro_raw, alpha_cola, ancho_global_mm,
                      condicionable):
    """
    Para una partición dada calcula:
      · cobertura por bin tras el oráculo global (+ IC bootstrap de bloques)
      · cuántos bins tienen su IC 90% fuera del objetivo
      · el oráculo CONDICIONAL a esa partición y su ahorro de anchura

    El ahorro es el costo, en MM USD, de no modelar esa dimensión: es la anchura
    extra que el oráculo global gasta en todas las filas para poder cubrir los
    bins difíciles.
    """
    codigos, _ = pd.factorize(etiquetas)
    n_bins = int(codigos.max()) + 1

    cov_orc, n_tot = cobertura_por_bin(dentro_orc, codigos, n_bins)
    cov_raw, _     = cobertura_por_bin(dentro_raw, codigos, n_bins)
    lo_ic, hi_ic   = bootstrap_cobertura(df, dentro_orc, codigos, n_bins)

    fuera = (hi_ic < COBERTURA_OBJETIVO) | (lo_ic > COBERTURA_OBJETIVO)

    # Oráculo condicional a esta partición
    q05_c = np.empty_like(q50); q95_c = np.empty_like(q50)
    s_lo_b = np.full(n_bins, np.nan); s_hi_b = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = codigos == i
        if m.sum() < 30:                    # bin demasiado chico para una cola al 5%
            q05_c[m] = q50[m] - w_lo[m]; q95_c[m] = q50[m] + w_hi[m]
            continue
        sl, sh = oraculo_multiplicativo(y[m], q50[m], w_lo[m], w_hi[m], alpha_cola)
        q05_c[m], q95_c[m] = aplicar_mult(q50[m], w_lo[m], w_hi[m], sl, sh)
        s_lo_b[i], s_hi_b[i] = sl, sh

    ancho_cond_mm = float((q95_c - q05_c).mean() / 1e6)
    cov_cond      = float(((y >= q05_c) & (y <= q95_c)).mean())

    return {
        "dimension": nombre,
        "condicionable": condicionable,
        "n_bins": n_bins,
        "codigos": codigos,
        "niveles": pd.Series(etiquetas).astype(str).drop_duplicates().tolist(),
        "n": n_tot, "cov_raw": cov_raw, "cov_orc": cov_orc,
        "ic_lo": lo_ic, "ic_hi": hi_ic, "fuera": fuera,
        "n_fuera": int(fuera.sum()),
        "dispersion": float(np.nanmax(cov_orc) - np.nanmin(cov_orc)),
        "s_lo": s_lo_b, "s_hi": s_hi_b,
        "ancho_cond_mm": ancho_cond_mm,
        "cov_cond": cov_cond,
        "ahorro_mm": ancho_global_mm - ancho_cond_mm,
        "ahorro_pct": (ancho_global_mm - ancho_cond_mm) / max(ancho_cond_mm, EPS),
    }


def main() -> None:
    alpha_cola = (1.0 - COBERTURA_OBJETIVO) / 2.0      # 0.05 por lado
    df = cargar()

    y   = df["target"].values
    q50 = df["q50"].values
    q05 = df["q05"].values
    q95 = df["q95"].values
    w_lo, w_hi = df["w_lo"].values, df["w_hi"].values

    # ── A. Oráculos globales ──────────────────────────────────────────────────
    _titulo("A · Oráculo global — la mejor calibración de nivel posible")

    cov_crudo = float(((y >= q05) & (y <= q95)).mean())
    sh_crudo  = float((q95 - q05).mean() / 1e6)
    print(f"  Modelo crudo          cobertura {cov_crudo:6.1%}   anchura {sh_crudo:8.1f} MM")

    s_lo, s_hi = oraculo_multiplicativo(y, q50, w_lo, w_hi, alpha_cola)
    q05_m, q95_m = aplicar_mult(q50, w_lo, w_hi, s_lo, s_hi)
    cov_m = float(((y >= q05_m) & (y <= q95_m)).mean())
    sh_m  = float((q95_m - q05_m).mean() / 1e6)
    print(f"  Oráculo multiplicativo cobertura {cov_m:6.1%}   anchura {sh_m:8.1f} MM"
          f"   (s_lo={s_lo:.3f}  s_hi={s_hi:.3f})")

    c_lo, c_hi = oraculo_aditivo(y, q05, q95, alpha_cola)
    q05_a, q95_a = q05 + c_lo, q95 + c_hi
    cov_a = float(((y >= q05_a) & (y <= q95_a)).mean())
    sh_a  = float((q95_a - q05_a).mean() / 1e6)
    print(f"  Oráculo aditivo        cobertura {cov_a:6.1%}   anchura {sh_a:8.1f} MM"
          f"   (c_lo={c_lo/1e6:+.1f}MM  c_hi={c_hi/1e6:+.1f}MM)")

    print(f"\n  s_lo={s_lo:.3f} significa que la cola BAJA necesita ser "
          f"{s_lo:.2f}x más ancha de lo que el modelo predice.")
    print(f"  s_hi={s_hi:.3f} idem para la cola ALTA. "
          f"Asimetría = {s_lo/max(s_hi, EPS):.2f}x")
    mejor = "multiplicativo" if sh_m <= sh_a else "aditivo"
    print(f"  Para la MISMA cobertura, el oráculo {mejor} es más angosto "
          f"({abs(sh_m - sh_a):.1f} MM de diferencia)")

    # ── B. ¿Queda estructura después del oráculo global? ──────────────────────
    _titulo("B · Cobertura condicional TRAS el oráculo global (la prueba)")

    dentro_orc = ((y >= q05_m) & (y <= q95_m))
    dentro_raw = ((y >= q05) & (y <= q95))
    df["_dentro_orc"] = dentro_orc
    df["_dentro_raw"] = dentro_raw

    df["dec_ancho"] = pd.qcut(df["ancho"], N_DECILES, labels=False, duplicates="drop")
    df["mes"]   = df["fecha_t"].dt.to_period("M").astype(str)
    df["h_bin"] = pd.cut(df["h"], bins=[1, 5, 10, 20, 30, 40, 50, 60, 75],
                         labels=["2-5", "6-10", "11-20", "21-30",
                                 "31-40", "41-50", "51-60", "61-75"])
    ancho_medio = df.groupby("dec_ancho")["ancho"].mean().values / 1e6

    # Se evalúan CUATRO particiones. Mirar solo una produce falsos negativos:
    # la heterocedasticidad temporal es invisible en los deciles de anchura,
    # porque dentro de cada decil los regímenes se mezclan por igual.
    DIMS = [
        ("decil de anchura", df["dec_ancho"].values,          True),
        ("horizonte h",      df["h_bin"].astype(str).values,  True),
        ("mes calendario",   df["mes"].values,                False),
        ("fold",             df["fold"].values,               False),
    ]

    print(f"  Bootstrap de bloques: B={BOOT_B}, bloque={BOOT_BLOQUE} dh sobre "
          f"{df['fecha_t'].nunique():,} fechas de origen\n")

    resultados = []
    for nombre, etiq, cond in DIMS:
        r = evaluar_dimension(df, nombre, etiq, y, q50, w_lo, w_hi,
                              dentro_orc, dentro_raw, alpha_cola, sh_m, cond)
        resultados.append(r)
        print(f"  {nombre:>18} | bins {r['n_bins']:>3} | "
              f"dispersión {r['dispersion']:>6.1%} | "
              f"miscalibrados {r['n_fuera']:>2}/{r['n_bins']:<3} | "
              f"ahorro condicional {r['ahorro_mm']:>8,.0f} MM ({r['ahorro_pct']:>+6.1%})")

    peor = max(resultados, key=lambda r: r["dispersion"])
    print(f"\n  Dimensión con más estructura residual: {peor['dimension'].upper()} "
          f"(dispersión {peor['dispersion']:.1%})")

    # Detalle de la dimensión de anchura (test directo de la señal de escala)
    r_ancho = resultados[0]
    _titulo("C · Detalle — decil de anchura predicha (¿ordena bien la escala?)")
    print(f"  {'Decil':>5} {'Ancho MM':>10} {'n':>8} {'Cob crudo':>10} "
          f"{'Cob oráculo':>12} {'IC 90%':>18} {'s_lo':>7} {'s_hi':>7}")
    print("  " + "-" * 88)
    for i in range(r_ancho["n_bins"]):
        print(f"  {i+1:>5} {ancho_medio[i]:>10,.0f} {int(r_ancho['n'][i]):>8,} "
              f"{r_ancho['cov_raw'][i]:>10.1%} {r_ancho['cov_orc'][i]:>12.1%} "
              f"[{r_ancho['ic_lo'][i]:>6.1%},{r_ancho['ic_hi'][i]:>6.1%}] "
              f"{r_ancho['s_lo'][i]:>7.3f} {r_ancho['s_hi'][i]:>7.3f}")
    tend = r_ancho["s_lo"][-1] - r_ancho["s_lo"][0]
    print(f"\n  Tendencia de s_lo del decil 1 al {r_ancho['n_bins']}: {tend:+.3f}")
    print("  s_lo CRECIENTE -> el modelo subestima justo donde ya predice más")
    print("  incertidumbre: su ranking de escala está sesgado.")
    print("  s_lo PLANO -> la forma de la señal de escala es correcta y solo")
    print("  falta un factor global; el post-proceso por anchura no aporta.")

    # Detalle de la dimensión temporal
    _titulo("D · Detalle — dimensión temporal (deriva)")
    for r in resultados:
        if r["dimension"] not in ("mes calendario", "fold"):
            continue
        print(f"\n  Por {r['dimension']}  (dispersión {r['dispersion']:.1%}, "
              f"{r['n_fuera']}/{r['n_bins']} miscalibrados)")
        orden = np.argsort(r["niveles"])
        for i in orden:
            marca = "  <-- fuera" if r["fuera"][i] else ""
            print(f"    {r['niveles'][i]:>12}  n={int(r['n'][i]):>7,}  "
                  f"crudo {r['cov_raw'][i]:>6.1%}  ->  oráculo {r['cov_orc'][i]:>6.1%}"
                  f"{marca}")

    rango  = r_ancho["dispersion"]
    n_fuera = r_ancho["n_fuera"]
    n_bins  = r_ancho["n_bins"]
    sh_c    = r_ancho["ancho_cond_mm"]
    cov_c   = r_ancho["cov_cond"]
    costo   = r_ancho["ahorro_mm"]
    df_s    = pd.DataFrame({"decil": np.arange(1, n_bins + 1),
                            "s_lo": r_ancho["s_lo"], "s_hi": r_ancho["s_hi"],
                            "ancho_pred_mm": ancho_medio})
    tablas = {r["dimension"]: pd.DataFrame({
        "nivel": r["niveles"], "n": r["n"].astype(int),
        "cob_cruda": r["cov_raw"], "cob_oraculo": r["cov_orc"],
        "ic_lo": r["ic_lo"], "ic_hi": r["ic_hi"],
        "s_lo": r["s_lo"], "s_hi": r["s_hi"]}) for r in resultados}

    # ── E. Escala predicha vs realizada ───────────────────────────────────────
    _titulo("E · ¿La anchura predicha ordena bien el error realizado?")

    df["err_abs"] = np.abs(y - q50)
    esc = (df.groupby("dec_ancho")
             .agg(ancho_pred=("ancho", "mean"),
                  err_p90=("err_abs", lambda s: float(np.quantile(s, 0.90))),
                  err_med=("err_abs", "median"))
             .reset_index())
    esc["ratio"] = esc["err_p90"] / np.maximum(esc["ancho_pred"], EPS)
    print(f"  {'Decil':>5} {'Ancho pred MM':>14} {'P90|error| MM':>14} {'ratio':>8}")
    print("  " + "-" * 46)
    for _, r in esc.iterrows():
        print(f"  {int(r['dec_ancho'])+1:>5} {r['ancho_pred']/1e6:>14,.0f} "
              f"{r['err_p90']/1e6:>14,.0f} {r['ratio']:>8.3f}")
    rho = float(np.corrcoef(df["ancho"].values, df["err_abs"].values)[0, 1])
    print(f"\n  Correlación anchura predicha vs |error| realizado: {rho:+.3f}")
    print("  (>0 = el modelo SÍ ordena la incertidumbre; ~0 = su anchura no informa)")

    # ── F. Veredicto ──────────────────────────────────────────────────────────
    _titulo("F · Veredicto")
    print(f"  Modelo crudo {cov_crudo:.1%}  ->  oráculo global {cov_m:.1%} "
          f"(por construcción)  con {sh_m - sh_crudo:+,.0f} MM de anchura\n")
    print(f"  {'Dimensión':>18} {'ex-ante':>9} {'dispersión':>11} "
          f"{'miscalib.':>10} {'ahorro condicional':>20}")
    print("  " + "-" * 74)
    for r in resultados:
        print(f"  {r['dimension']:>18} {'sí' if r['condicionable'] else 'no':>9} "
              f"{r['dispersion']:>11.1%} {r['n_fuera']:>4}/{r['n_bins']:<5} "
              f"{r['ahorro_mm']:>12,.0f} MM ({r['ahorro_pct']:+.1%})")

    # El criterio NO usa el conteo de bins fuera del IC: con 28 meses y un IC al
    # 90% se esperan ~2.8 falsos por azar, y sin corrección por multiplicidad ese
    # conteo dispara solo. En una prueba sintética SIN estructura temporal marcó
    # 10/28 meses. Se decide por magnitud —dispersión y ahorro— que es lo que
    # tiene consecuencia económica. El conteo se reporta, no decide.
    UMBRAL_DISP, UMBRAL_AHORRO = 0.05, 0.05
    def _relevante(r):
        return (r["dispersion"] >= UMBRAL_DISP
                or r["ahorro_pct"] >= UMBRAL_AHORRO)

    malas     = [r for r in resultados if _relevante(r)]
    ex_ante   = [r for r in malas if r["condicionable"]]
    temporal  = [r for r in malas if not r["condicionable"]]

    print()
    if not malas:
        print("  -> CALIBRACIÓN DE NIVEL. Tras un único factor global la cobertura")
        print("     queda plana en TODAS las dimensiones evaluadas. La estructura")
        print("     condicional está bien capturada y solo falla el nivel.")
        print("     REMEDIO: ACI / conformal online. No hace falta tocar el modelo.")
    else:
        print("  -> HETEROCEDASTICIDAD CONDICIONAL. Ni siquiera un oráculo global")
        print("     —que ve el futuro— reparte bien la cobertura: para cubrir los")
        print("     bins difíciles sobre-ensancha los fáciles. Ningún método de")
        print("     calibración GLOBAL (CQR, ACI escalar) puede cerrar esto.")
        print(f"     Dimensiones afectadas: {', '.join(r['dimension'] for r in malas)}")
        print()
        if ex_ante:
            print("     · Parte de la estructura es CONDICIONABLE EX-ANTE "
                  f"({', '.join(r['dimension'] for r in ex_ante)}):")
            print("       se corrige con post-proceso condicional, sin reentrenar.")
            print(f"       Ahorro disponible: "
                  f"{max(r['ahorro_mm'] for r in ex_ante):,.0f} MM de anchura.")
        if temporal:
            print("     · Parte vive en el TIEMPO "
                  f"({', '.join(r['dimension'] for r in temporal)}):")
            print("       no se puede condicionar ex-ante porque es el futuro.")
            print("       REMEDIO: ACI (se adapta online) o un feature que anticipe")
            print("       el régimen de volatilidad. Un delta fijo estimado sobre")
            print("       VAL nunca lo va a capturar — es lo que ya se observó.")

    # ── Salidas ───────────────────────────────────────────────────────────────
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    fecha = date.today().strftime("%Y%m%d")

    resumen = pd.DataFrame([
        {"metrica": "cobertura_cruda",          "valor": cov_crudo},
        {"metrica": "cobertura_oraculo_global", "valor": cov_m},
        {"metrica": "cobertura_oraculo_cond",   "valor": cov_c},
        {"metrica": "anchura_cruda_mm",         "valor": sh_crudo},
        {"metrica": "anchura_oraculo_global_mm","valor": sh_m},
        {"metrica": "anchura_oraculo_cond_mm",  "valor": sh_c},
        {"metrica": "costo_heterocedast_mm",    "valor": costo},
        {"metrica": "s_lo_global",              "valor": s_lo},
        {"metrica": "s_hi_global",              "valor": s_hi},
        {"metrica": "dispersion_deciles",       "valor": rango},
        {"metrica": "deciles_miscalibrados",    "valor": n_fuera},
        {"metrica": "corr_ancho_error",         "valor": rho},
    ] + [
        {"metrica": f"dispersion_{r['dimension'].replace(' ', '_')}",
         "valor": r["dispersion"]} for r in resultados
    ] + [
        {"metrica": f"ahorro_mm_{r['dimension'].replace(' ', '_')}",
         "valor": r["ahorro_mm"]} for r in resultados
    ])
    det = pd.DataFrame({
        "decil": np.arange(1, n_bins + 1), "n": r_ancho["n"].astype(int),
        "ancho_pred_mm": ancho_medio, "cob_cruda": r_ancho["cov_raw"],
        "cob_oraculo": r_ancho["cov_orc"], "ic_lo": r_ancho["ic_lo"],
        "ic_hi": r_ancho["ic_hi"],
        "s_lo_cond": r_ancho["s_lo"], "s_hi_cond": r_ancho["s_hi"],
    })

    ruta_xl = DIR_OUT / f"diagnostico_oraculo_{BANCO}_{fecha}.xlsx"
    try:
        with pd.ExcelWriter(ruta_xl, engine="openpyxl") as xl:
            resumen.to_excel(xl, sheet_name="resumen", index=False)
            det.to_excel(xl, sheet_name="por_decil_ancho", index=False)
            esc.to_excel(xl, sheet_name="escala_pred_vs_real", index=False)
            for dim, t in tablas.items():
                t.to_excel(xl, sheet_name=f"por_{dim}".replace(" ", "_")[:31],
                           index=False)
        print(f"\n[OK] {ruta_xl.name}")
    except Exception as e:
        print(f"\n[AVISO] Excel no escrito ({e}) — se guardan CSVs")
        resumen.to_csv(DIR_OUT / f"diag_oraculo_resumen_{fecha}.csv", index=False)
        det.to_csv(DIR_OUT / f"diag_oraculo_deciles_{fecha}.csv", index=False)

    graficar(det, esc, df_s, resultados, df, y, q50,
             (cov_crudo, cov_m, cov_c), (sh_crudo, sh_m, sh_c),
             costo, rho, DIR_OUT / f"diagnostico_oraculo_{BANCO}_{fecha}.png")


# ── Gráfico ───────────────────────────────────────────────────────────────────
def graficar(det, esc, df_s, resultados, df, y, q50, covs, shs, costo, rho, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("[AVISO] matplotlib no disponible — sin gráfico")
        return

    cov_crudo, cov_m, cov_c = covs
    sh_crudo, sh_m, sh_c = shs

    fig, axes = plt.subplots(3, 2, figsize=(15, 15.5),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.24})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c3c2b7")
        ax.tick_params(colors=C_TINTA2, labelsize=8.5)
        ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)

    fig.suptitle(
        "¿Calibración o heterocedasticidad condicional?  —  "
        f"{BANCO} · fold_{_MODO_SUFIJO}",
        fontsize=13, fontweight="bold", color=C_TINTA, y=0.985)
    fig.text(0.5, 0.963,
             "El oráculo ve el futuro: es la COTA SUPERIOR de cualquier calibración honesta. "
             "Lo informativo no es que llegue a 90%, sino la dispersión que queda.",
             ha="center", fontsize=9, color=C_TINTA2, style="italic")

    x = det["decil"].values
    pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")

    # ── Panel 1 · EL panel: cobertura por decil de anchura ───────────────────
    ax = axes[0, 0]
    ax.axhline(COBERTURA_OBJETIVO, color=C_REF, lw=1.4, ls="--", zorder=2)
    ax.text(0.99, COBERTURA_OBJETIVO, "objetivo 90% ", va="bottom", ha="right",
            fontsize=8, color=C_REF, transform=ax.get_yaxis_transform())
    ax.fill_between(x, det["ic_lo"], det["ic_hi"], color=C_ORACULO,
                    alpha=0.16, lw=0, zorder=2)
    ax.plot(x, det["cob_cruda"], "o-", color=C_MODELO, lw=2, ms=8,
            zorder=4, label="Modelo crudo")
    ax.plot(x, det["cob_oraculo"], "s-", color=C_ORACULO, lw=2, ms=8,
            zorder=5, label="Tras oráculo global")
    ax.set_xticks(x)
    ax.set_xlabel("Decil de anchura predicha  (1 = banda más angosta)",
                  fontsize=9, color=C_TINTA2)
    ax.set_ylabel("Cobertura empírica", fontsize=9, color=C_TINTA2)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("1 · Cobertura por decil de anchura predicha\n"
                 "banda naranja = IC 90% (bootstrap de bloques)",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=C_GRID, labelcolor=C_TINTA)

    # ── Panel 2 · factor de reescalamiento necesario por decil ───────────────
    ax = axes[0, 1]
    ax.axhline(1.0, color=C_REF, lw=1.4, ls="--", zorder=2)
    ax.text(0.99, 1.0, "el modelo ya acierta ", va="bottom", ha="right",
            fontsize=8, color=C_REF, transform=ax.get_yaxis_transform())
    ax.plot(x, df_s["s_lo"], "o-", color=C_MODELO, lw=2, ms=8, zorder=4,
            label="cola baja (retiros)")
    ax.plot(x, df_s["s_hi"], "s-", color=C_ORACULO, lw=2, ms=8, zorder=4,
            label="cola alta")
    ax.set_xticks(x)
    ax.set_xlabel("Decil de anchura predicha", fontsize=9, color=C_TINTA2)
    ax.set_ylabel("Factor de ensanchamiento necesario", fontsize=9, color=C_TINTA2)
    ax.set_title("2 · Cuánto hay que ensanchar cada decil\n"
                 "pendiente ≠ 0 ⇒ la señal de escala del modelo está sesgada",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=C_GRID, labelcolor=C_TINTA)

    # ── Panel 3 · anchura predicha vs error realizado ────────────────────────
    ax = axes[1, 0]
    ap = esc["ancho_pred"].values / 1e6
    ep = esc["err_p90"].values / 1e6
    lim = max(ap.max(), ep.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=C_REF, lw=1.4, ls="--", zorder=2)
    # Etiqueta por debajo de la diagonal, donde nunca cae la serie si el modelo
    # subestima (que es el caso que este panel busca mostrar).
    ax.text(lim * 0.62, lim * 0.54, "identidad", fontsize=8, color=C_REF,
            ha="left", va="top", rotation=45, rotation_mode="anchor")
    ax.plot(ap, ep, "o-", color=C_MODELO, lw=2, ms=8, zorder=4)
    for i, (a, e) in enumerate(zip(ap, ep)):
        if i in (0, len(ap) - 1):
            ax.annotate(f"D{i+1}", (a, e), textcoords="offset points",
                        xytext=(9, -10), fontsize=8, color=C_TINTA2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Anchura predicha Q95−Q05 (MM USD)", fontsize=9, color=C_TINTA2)
    ax.set_ylabel("P90 del |error| realizado (MM USD)", fontsize=9, color=C_TINTA2)
    ax.set_title(f"3 · ¿La anchura predice el error?   corr = {rho:+.3f}\n"
                 "por encima de la identidad ⇒ el modelo subestima ahí",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")
    ax.grid(color=C_GRID, lw=0.6, zorder=0)

    # ── Panel 4 · EL VEREDICTO: dónde queda estructura ───────────────────────
    # Horizontal porque las etiquetas son texto; ordenado por magnitud para que
    # la dimensión culpable quede arriba y se lea sin buscar.
    ax = axes[1, 1]
    rs   = sorted(resultados, key=lambda r: r["dispersion"])
    ypos = np.arange(len(rs))
    disp = [r["dispersion"] for r in rs]
    # El color codifica el REMEDIO, no el ranking: naranja = corregible ex-ante,
    # azul = solo vía adaptación online. Es la distinción accionable.
    cols = [C_ORACULO if r["condicionable"] else C_MODELO for r in rs]
    ax.barh(ypos, disp, color=cols, height=0.58, zorder=3)
    ax.axvline(0.05, color=C_REF, lw=1.4, ls="--", zorder=4)
    ax.text(0.05, -0.48, " umbral 5pp", fontsize=8, color=C_REF, va="center")
    for yv, d, r in zip(ypos, disp, rs):
        ax.text(d + max(disp) * 0.02, yv, f"{d:.1%}  ·  {r['n_fuera']}/{r['n_bins']} bins",
                va="center", fontsize=8.5, color=C_TINTA)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["dimension"] for r in rs], fontsize=9)
    ax.set_xlim(0, max(max(disp) * 1.42, 0.07))
    ax.xaxis.set_major_formatter(pct)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Dispersión de cobertura tras el oráculo global",
                  fontsize=9, color=C_TINTA2)
    ax.set_title("4 · ¿Dónde queda estructura?   (naranja = corregible ex-ante,\n"
                 "azul = solo con adaptación online)",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")

    # ── Panel 5 · cobertura por horizonte ────────────────────────────────────
    ax = axes[2, 0]
    ph = (df.groupby("h")
            .agg(cr=("_dentro_raw", "mean"), co=("_dentro_orc", "mean"))
            .reset_index())
    ax.axhline(COBERTURA_OBJETIVO, color=C_REF, lw=1.4, ls="--", zorder=2)
    ax.plot(ph["h"], ph["cr"], color=C_MODELO, lw=1.8, zorder=4, label="Modelo crudo")
    ax.plot(ph["h"], ph["co"], color=C_ORACULO, lw=1.8, zorder=5,
            label="Tras oráculo global")
    ax.set_xlabel("Horizonte h (días hábiles)", fontsize=9, color=C_TINTA2)
    ax.set_ylabel("Cobertura empírica", fontsize=9, color=C_TINTA2)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("5 · Cobertura por horizonte\n"
                 "estructura residual ⇒ el oráculo global no la ve",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=C_GRID, labelcolor=C_TINTA)

    # ── Panel 6 · cobertura en el tiempo ─────────────────────────────────────
    ax = axes[2, 1]
    rt = next((r for r in resultados if r["dimension"] == "mes calendario"), None)
    orden = np.argsort(rt["niveles"])
    niveles = [rt["niveles"][i] for i in orden]
    xm = np.arange(len(niveles))
    ax.axhline(COBERTURA_OBJETIVO, color=C_REF, lw=1.4, ls="--", zorder=2)
    ax.fill_between(xm, rt["ic_lo"][orden], rt["ic_hi"][orden],
                    color=C_ORACULO, alpha=0.16, lw=0, zorder=2)
    ax.plot(xm, rt["cov_raw"][orden], "o-", color=C_MODELO, lw=1.8, ms=5,
            zorder=4, label="Modelo crudo")
    ax.plot(xm, rt["cov_orc"][orden], "s-", color=C_ORACULO, lw=1.8, ms=5,
            zorder=5, label="Tras oráculo global")
    paso = max(1, len(niveles) // 10)
    ax.set_xticks(xm[::paso])
    ax.set_xticklabels(niveles[::paso], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Cobertura empírica", fontsize=9, color=C_TINTA2)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("6 · Cobertura mes a mes\n"
                 "deriva persistente ⇒ el problema es temporal, no de escala",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=C_GRID, labelcolor=C_TINTA)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[OK] {ruta.name}")


if __name__ == "__main__":
    main()
