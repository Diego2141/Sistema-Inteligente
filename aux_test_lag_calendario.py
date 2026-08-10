# -*- coding: utf-8 -*-
"""
aux_test_lag_calendario.py
==========================
¿Un rezago alineado por CALENDARIO aporta información de ESCALA que el modelo
todavía no está usando?

Motivación
----------
aux_diagnostico_oraculo dejó establecido que la señal de escala del modelo está
SUB-AMPLIFICADA: la anchura predicha varía 2.8x entre el decil 1 y el 10 mientras
el error realizado varía 5.7x (elasticidad ~0.59 en vez de 1). El modelo reacciona
en la dirección correcta pero con la mitad de la amplitud necesaria. Le falta una
variable que expanda el rango dinámico.

Candidata: el flujo en la posición equivalente del mes anterior, de hace dos meses
y de hace tres. La hipótesis es que `sigma_22d` —que el modelo ya tiene— PROMEDIA
sobre todas las posiciones del mes y por eso destruye justamente la variación que
hace falta, mientras que un rezago alineado por calendario compara peras con peras.

Este script NO reentrena nada. Responde si vale la pena tocar step001.

Restricción de disponibilidad
-----------------------------
El rezago k del objetivo t+h cae en t+h-21k días hábiles, observable en t solo si
h <= 21k:

    h <= 21     lags 1, 2, 3
    22-42       lags 2, 3
    43-63       lag 3
    64-75       ninguno

Como hay UN MODELO POR h, cada uno usaría exactamente los lags que le tocan. El
script respeta esa máscara: nunca usa un valor que el modelo no podría ver.

El número que decide
--------------------
NO la correlación simple —todo lo que mide magnitud correlaciona con todo lo que
mide magnitud— sino la CORRELACIÓN PARCIAL controlando por la anchura predicha:
¿aporta algo POR ENCIMA de lo que el modelo ya usa? Y el cara a cara contra
`sigma_22d`, que es el competidor directo.

Se usan rangos (Spearman) porque el error realizado tiene colas muy pesadas: en el
decil 10 el P90 del |error| supera el ancho completo del intervalo.

Salidas
-------
    2. Output/step006_cqr/cqr_<MODO>/
        test_lag_calendario_<BANCO>_<fecha>.png
        test_lag_calendario_<BANCO>_<fecha>.xlsx
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
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"

_MODO_SUFIJO = "exp_arctan"
BANCO        = "SISTEMA"

DIR_FOLD = DIR_WFCV / f"fold_{_MODO_SUFIJO}"
DIR_OUT  = DIR_CQR  / f"cqr_{_MODO_SUFIJO}"

LAGS_MESES  = [1, 2, 3]

# Competidores: variables de escala que el modelo YA tiene. Se toman las que
# existan en la matriz, en este orden de preferencia para el titular. La matriz
# cambia entre experimentos (la reducida no trae sigma_22d), así que se descubren
# del esquema en vez de fijarlas.
COMPETIDORES_POSIBLES = ["sigma_22d", "sigma_flujo_ratio", "garch_vol",
                         "ma_flujo_20d", "ma_flujo_5d", "sigma_flujo"]
BOOT_B      = 300
BOOT_BLOQUE = 22
SEMILLA     = 20260810
N_BINS      = 10

# Paleta validada (light, surface #fcfcfb): lightness/chroma/CVD/normal-vision PASS
C_CAND  = "#eb6834"   # candidato: rezago de calendario
C_COMP  = "#2a78d6"   # competidor: sigma_22d (lo que el modelo ya tiene)
C_REF   = "#898781"
C_TINTA = "#0b0b0b"
C_TINTA2 = "#52514e"
C_GRID  = "#e1e0d9"
SURFACE = "#fcfcfb"

EPS = 1e-9
NOMBRE_COMPETIDOR = None   # se resuelve en cargar_preds() según lo hallado


def _titulo(t: str) -> None:
    print(f"\n{'=' * 78}\n  {t}\n{'=' * 78}")


# ── Estadística ───────────────────────────────────────────────────────────────
def _rangos(x):
    """Rango normalizado; NaN se mantiene NaN."""
    s = pd.Series(x)
    return (s.rank(method="average") / s.notna().sum()).values


def _matriz_rangos(df, cols):
    """
    Rangos de todas las columnas de una vez, más la máscara de filas completas.

    Se pre-calcula FUERA del bootstrap: rankear 44k filas seis veces por réplica
    domina el costo y vuelve el script inusable con B=300.
    """
    M = np.column_stack([_rangos(df[c]) for c in cols])
    ok = np.isfinite(M).all(axis=1)
    return M, ok


def _parcial_M(M, filas=None):
    """
    Correlación parcial sobre una matriz ya rankeada: col 0 vs col 1,
    controlando por las columnas 2..k. `filas` selecciona un submuestreo.
    """
    A = M if filas is None else M[filas]
    if len(A) < 30:
        return np.nan
    x, y = A[:, 0].copy(), A[:, 1].copy()
    if A.shape[1] > 2:
        Z = np.column_stack([np.ones(len(A)), A[:, 2:]])
        x = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        y = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if x.std() < EPS or y.std() < EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def corr_parcial(x, y, Z=None):
    """
    Correlación de Spearman entre x e y controlando por Z (lista de series).

    Sobre rangos y no sobre niveles porque el error realizado tiene colas muy
    pesadas: en el decil 10 el P90 del |error| supera el ancho completo del
    intervalo, y una correlación de Pearson la dominarían un puñado de filas.
    """
    cols = [pd.Series(np.asarray(v)).reset_index(drop=True)
            for v in ([x, y] + list(Z or []))]
    M = np.column_stack([_rangos(c) for c in cols])
    ok = np.isfinite(M).all(axis=1)
    return _parcial_M(M[ok])


def bootstrap_M(df, M, ok, B=BOOT_B, L=BOOT_BLOQUE, semilla=SEMILLA):
    """
    IC 90% por bootstrap de bloques móviles sobre FECHAS DE ORIGEN.

    Las filas no son independientes: 74 horizontes comparten origen y los orígenes
    vecinos comparten datos. Un IC analítico sobre n=44,000 sería varias veces
    demasiado angosto y cualquier correlación parecería significativa.

    Se remuestrean FECHAS en bloques de L días consecutivos; los rangos ya vienen
    calculados sobre la muestra completa.
    """
    rng = np.random.default_rng(semilla)
    d = df.loc[ok] if len(ok) == len(df) else df
    pos = np.flatnonzero(ok)

    fechas = np.sort(d["fecha_t"].unique())
    n_f = len(fechas)
    if n_f < 2 * L:
        return np.nan, np.nan
    cod = pd.Series(np.arange(n_f), index=fechas)
    f_cod = d["fecha_t"].map(cod).values

    orden  = np.argsort(f_cod, kind="stable")
    f_ord  = f_cod[orden]
    inicio = np.searchsorted(f_ord, np.arange(n_f), side="left")
    fin    = np.searchsorted(f_ord, np.arange(n_f), side="right")

    n_bloques = int(np.ceil(n_f / L))
    out = []
    for _ in range(B):
        arr = rng.integers(0, max(1, n_f - L + 1), size=n_bloques)
        fb  = np.concatenate([np.arange(a, min(a + L, n_f)) for a in arr])
        loc = np.concatenate([orden[inicio[f]:fin[f]] for f in fb])
        v = _parcial_M(M[pos], loc)
        if np.isfinite(v):
            out.append(v)
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


# ── 1. Serie diaria de flujo + calendario ─────────────────────────────────────
def _columnas_matriz():
    """Nombres de columna de la matriz sin cargarla (solo el esquema)."""
    try:
        import pyarrow.dataset as pads
        return list(pads.dataset(str(RUTA_MATRIZ)).schema.names)
    except Exception:
        pass
    try:
        import pyarrow.parquet as pq
        return list(pq.read_schema(str(RUTA_MATRIZ)).names)
    except Exception:
        return list(pd.read_parquet(RUTA_MATRIZ).columns)


def cargar_serie_diaria():
    """
    Serie diaria de flujo desde la matriz de features.

    Una fila por fecha_t (la matriz tiene 74 filas por fecha, una por horizonte;
    R_t0/D_t0 son constantes dentro de la fecha porque están anclados en t).

    Los competidores se descubren del esquema: la matriz cambia entre
    experimentos y pedir una columna inexistente aborta la lectura entera.
    """
    disponibles = _columnas_matriz()
    comp = [c for c in COMPETIDORES_POSIBLES if c in disponibles]
    faltan = [c for c in ("fecha_t", "R_t0", "D_t0") if c not in disponibles]
    if faltan:
        raise KeyError(f"La matriz no tiene {faltan} — sin ellas no hay serie de flujo")
    if not comp:
        print("[AVISO] la matriz no trae ninguna variable de escala conocida; "
              "el test solo controlará por la anchura predicha y h")
    else:
        print(f"[OK] competidores hallados en la matriz: {comp}")

    cols = ["fecha_t", "R_t0", "D_t0"] + comp
    try:
        mat = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", BANCO)],
                              columns=cols)
    except Exception:
        mat = pd.read_parquet(RUTA_MATRIZ, columns=cols + ["banco"])
        mat = mat[mat["banco"] == BANCO]

    mat["fecha_t"] = pd.to_datetime(mat["fecha_t"])
    d = (mat.drop_duplicates("fecha_t")
            .sort_values("fecha_t")
            .set_index("fecha_t"))

    serie = pd.DataFrame(index=d.index)
    serie["flujo"]  = d["D_t0"] - d["R_t0"]      # neto, consistente con el target
    serie["retiro"] = d["R_t0"]                   # solo retiro (lo operativo)
    for c in comp:
        serie[c] = d[c]
    serie.attrs["competidores"] = comp

    # Distancia al cierre de mes, en las DOS métricas. Cuál corresponde es una
    # pregunta abierta: el período de encaje se computa sobre días calendario,
    # pero el modelo lleva dias_al_cierre_mes en hábiles. Se prueban ambas.
    idx = serie.index
    fin_mes = idx + pd.offsets.MonthEnd(0)
    serie["dcm_cal"] = (fin_mes - idx).days

    df_h = pd.DataFrame({"f": idx, "ym": idx.to_period("M")})
    # rank descendente dentro del mes: el último día hábil = 0
    serie["dcm_hab"] = (df_h.groupby("ym")["f"]
                            .rank(ascending=False, method="first").values - 1).astype(int)
    serie["ym"] = idx.to_period("M")

    print(f"[OK] serie diaria: {len(serie):,} días | "
          f"{serie.index.min().date()} → {serie.index.max().date()}")
    return serie, comp


def construir_lags(serie, col_flujo, modo_dcm):
    """
    {(año-mes, dcm) -> fecha} para localizar el día equivalente de meses previos.

    Alineando por distancia AL CIERRE, no por posición desde el inicio: los meses
    tienen entre 19 y 23 días hábiles, y anclar al final hace que el último día
    siempre sea 0 — que además es donde se ata la restricción de encaje.
    """
    key = f"dcm_{modo_dcm}"
    idx_map = {}
    for f, ym, dcm in zip(serie.index, serie["ym"], serie[key]):
        idx_map.setdefault((ym, int(dcm)), f)     # primera ocurrencia
    valores = serie[col_flujo]
    return idx_map, valores


# ── 2. Muestra: predicciones de TEST ──────────────────────────────────────────
def cargar_preds(serie, comp):
    paths = sorted(DIR_FOLD.glob(f"preds_base_{BANCO}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Sin preds_base en {DIR_FOLD}")
    df = pd.read_parquet(paths[-1])
    print(f"[OK] {paths[-1].name}: {len(df):,} filas")

    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.dropna(subset=["target", "q05", "q50", "q95"]).copy()

    # fecha_th: si no viene, se deriva del calendario hábil de la propia serie
    if "fecha_th" in df.columns and df["fecha_th"].notna().any():
        df["fecha_th"] = pd.to_datetime(df["fecha_th"])
    else:
        dias = np.array(serie.index)
        pos  = pd.Series(np.arange(len(dias)), index=serie.index)
        p_t  = df["fecha_t"].map(pos)
        p_th = p_t + df["h"]
        ok   = p_t.notna() & (p_th < len(dias))
        df["fecha_th"] = pd.NaT
        df.loc[ok, "fecha_th"] = dias[p_th[ok].astype(int).values]
        print(f"     fecha_th derivada del calendario ({int(ok.sum()):,} filas)")

    df = df.dropna(subset=["fecha_th"]).copy()

    # Variables a explicar y control
    df["r"]     = (df["target"] - df["q50"]).abs()          # escala NO capturada
    df["r_lo"]  = (df["q50"] - df["target"]).clip(lower=0)  # solo sorpresas abajo
    df["ancho"] = df["q95"] - df["q05"]                     # escala SÍ predicha

    # Competidor: la mejor variable de escala que el modelo YA tiene, entre las
    # halladas en el esquema de la matriz. Se guarda con el alias interno
    # "sigma_22d" para no reescribir el resto del script — el nombre real se
    # imprime una vez para que quede claro qué se está comparando.
    global NOMBRE_COMPETIDOR
    if comp:
        NOMBRE_COMPETIDOR = comp[0]
        df["sigma_22d"] = df["fecha_t"].map(serie[NOMBRE_COMPETIDOR])
        print(f"[OK] competidor usado: {NOMBRE_COMPETIDOR} "
              f"(alias interno 'sigma_22d' en el resto del reporte)")
    else:
        NOMBRE_COMPETIDOR = None
        df["sigma_22d"] = np.nan
        print("[AVISO] sin competidor disponible en la matriz — las columnas "
              "'sigma_22d' del reporte quedan vacías (comparación omitida)")
    return df


def agregar_candidato(df, serie, col_flujo, modo_dcm, etiqueta):
    """
    Añade pos_lag_1/2/3 y esc_<etiqueta> respetando la máscara de disponibilidad.

    Un rezago solo se usa si su fecha es <= fecha_t. Nunca se le da al modelo un
    valor que no podría observar; sin esa máscara el test daría un resultado
    espectacular y falso.
    """
    key = f"dcm_{modo_dcm}"
    idx_map, valores = construir_lags(serie, col_flujo, modo_dcm)

    dcm_th = df["fecha_th"].map(serie[key])
    ym_th  = df["fecha_th"].dt.to_period("M")

    cols_abs, n_disp = [], []
    for k in LAGS_MESES:
        claves = list(zip(ym_th - k, dcm_th.fillna(-1).astype(int)))
        fechas = pd.Series([idx_map.get(c) for c in claves], index=df.index)
        fechas = pd.to_datetime(fechas)

        vals = fechas.map(valores)
        # Máscara de disponibilidad point-in-time
        vals = vals.where(fechas.notna() & (fechas <= df["fecha_t"]))

        df[f"lag{k}_{etiqueta}"] = vals
        cols_abs.append(vals.abs())
        n_disp.append(int(vals.notna().sum()))

    # Mediana de los |valores| disponibles: con 3 observaciones un solo rezago es
    # demasiado inestable, y la inestabilidad del estimador es exactamente lo que
    # viene castigando a este proyecto.
    df[f"esc_{etiqueta}"] = pd.concat(cols_abs, axis=1).median(axis=1, skipna=True)
    df[f"n_lags_{etiqueta}"] = pd.concat(cols_abs, axis=1).notna().sum(axis=1)

    cob = df[f"esc_{etiqueta}"].notna().mean()
    print(f"     {etiqueta:>18}: lags disponibles {n_disp} | "
          f"cobertura del candidato {cob:.1%}")
    return df


# ── 3. Test ───────────────────────────────────────────────────────────────────
def main() -> None:
    serie, comp = cargar_serie_diaria()
    df = cargar_preds(serie, comp)

    _titulo("Construcción de candidatos (con máscara point-in-time)")
    variantes = []
    for col_flujo in ("flujo", "retiro"):
        for modo in ("hab", "cal"):
            et = f"{col_flujo}_{modo}"
            df = agregar_candidato(df, serie, col_flujo, modo, et)
            variantes.append(et)

    _titulo("A · ¿Aporta escala POR ENCIMA de lo que el modelo ya usa?")
    print("  Spearman contra r = |target − q50|, controlando por anchura Y por h\n")
    _et_comp = NOMBRE_COMPETIDOR or "sin competidor"
    print(f"  {'candidato':>18} {'simple':>9} {'| ancho':>10} "
          f"{'IC 90%':>18} {'| ancho+'+_et_comp[:10]:>15}")
    print("  " + "-" * 76)

    filas = []
    for et in variantes + ["sigma_22d"]:
        col = et if et == "sigma_22d" else f"esc_{et}"
        sub = df.dropna(subset=[col, "r", "ancho"])
        if len(sub) < 100:
            continue
        c_simple = corr_parcial(sub[col], sub["r"])
        M, ok    = _matriz_rangos(sub, [col, "r", "ancho", "h"])
        c_anch   = _parcial_M(M[ok])
        lo, hi   = bootstrap_M(sub, M, ok)
        sub2 = sub.dropna(subset=["sigma_22d"])
        c_both = (np.nan if et == "sigma_22d" or len(sub2) < 100 else
                  corr_parcial(sub2[col], sub2["r"],
                               [sub2["ancho"], sub2["h"], sub2["sigma_22d"]]))
        marca = f"  <- competidor ({NOMBRE_COMPETIDOR})" if et == "sigma_22d" else ""
        print(f"  {et:>18} {c_simple:>+9.3f} {c_anch:>+10.3f} "
              f"[{lo:>+6.3f},{hi:>+6.3f}] "
              f"{'—' if not np.isfinite(c_both) else f'{c_both:+.3f}':>15}{marca}")
        filas.append({"candidato": et, "n": len(sub), "corr_simple": c_simple,
                      "parcial_ancho": c_anch, "ic_lo": lo, "ic_hi": hi,
                      "parcial_ancho_sigma": c_both})
    tab = pd.DataFrame(filas)

    print("\n  'simple' sube por construcción: todo lo que mide magnitud")
    print("  correlaciona con todo lo que mide magnitud. El número que decide es")
    print("  '| ancho': lo que aporta POR ENCIMA de la escala que el modelo ya predice.")
    print(f"  '| ancho+{_et_comp}' responde si alinear por calendario le gana a promediar.")

    # ── B. Solo la cola baja (lo operativo) ───────────────────────────────────
    _titulo("B · Ídem contra r_lo = max(q50 − target, 0)  — solo retiros")
    print(f"  {'candidato':>18} {'| ancho':>10} {'IC 90%':>18}")
    print("  " + "-" * 50)
    for et in variantes + ["sigma_22d"]:
        col = et if et == "sigma_22d" else f"esc_{et}"
        sub = df.dropna(subset=[col, "r_lo", "ancho"])
        if len(sub) < 100:
            continue
        M, ok  = _matriz_rangos(sub, [col, "r_lo", "ancho", "h"])
        c      = _parcial_M(M[ok])
        lo, hi = bootstrap_M(sub, M, ok)
        tab.loc[tab["candidato"] == et, "parcial_lo"] = c
        print(f"  {et:>18} {c:>+10.3f} [{lo:>+6.3f},{hi:>+6.3f}]")

    # ── C. Por tramo de horizonte ─────────────────────────────────────────────
    _titulo("C · Por tramo de horizonte (la disponibilidad cambia con h)")
    mejor = tab.loc[tab["parcial_ancho"].abs().idxmax(), "candidato"]
    col_m = mejor if mejor == "sigma_22d" else f"esc_{mejor}"
    print(f"  Mejor candidato de la sección A: {mejor}\n")
    df["h_bin"] = pd.cut(df["h"], [1, 21, 42, 63, 75],
                         labels=["2-21 (3 lags)", "22-42 (2 lags)",
                                 "43-63 (1 lag)", "64-75 (ninguno)"])
    print(f"  {'tramo':>18} {'n':>8} {'cob.':>7} {'| ancho':>10} "
          f"{(NOMBRE_COMPETIDOR or 'competidor')[:11]:>11}")
    print("  " + "-" * 60)
    filas_h = []
    for tramo, g in df.groupby("h_bin", observed=True):
        s = g.dropna(subset=[col_m, "r", "ancho"])
        cob = g[col_m].notna().mean()
        c  = corr_parcial(s[col_m], s["r"], [s["ancho"]]) if len(s) >= 100 else np.nan
        s2 = g.dropna(subset=["sigma_22d", "r", "ancho"])
        cs = corr_parcial(s2["sigma_22d"], s2["r"], [s2["ancho"]]) if len(s2) >= 100 else np.nan
        print(f"  {str(tramo):>18} {len(s):>8,} {cob:>7.0%} "
              f"{c:>+10.3f} {cs:>+11.3f}")
        filas_h.append({"tramo": str(tramo), "n": len(s), "cobertura": cob,
                        "parcial_candidato": c, "parcial_sigma": cs})
    tab_h = pd.DataFrame(filas_h)

    # ── D. ¿Expande el rango dinámico? ────────────────────────────────────────
    _titulo("D · ¿Expandiría el rango dinámico de la anchura?")
    print("  El diagnóstico del oráculo midió elasticidad ~0.59 (el modelo mueve")
    print("  la anchura 2.8x cuando el error se mueve 5.7x). Acá se mide cuánto")
    print("  ordena el candidato el error DENTRO de cada decil de anchura —")
    print("  o sea, la variación que el modelo se está perdiendo.\n")

    sub = df.dropna(subset=[col_m, "r", "ancho"]).copy()
    sub["dec_ancho"] = pd.qcut(sub["ancho"], N_BINS, labels=False, duplicates="drop")
    sub["dec_cand"] = (sub.groupby("dec_ancho")[col_m]
                          .transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                                       labels=False,
                                                       duplicates="drop")))
    sub = sub.dropna(subset=["dec_cand"])
    piv = (sub.groupby("dec_cand")["r"].median() / 1e6) if len(sub) else pd.Series(dtype=float)
    if piv.empty:
        n_anchuras = df["ancho"].nunique()
        print(f"  [AVISO] sin quintiles utilizables — se omite la sección "
              f"(anchura predicha con {n_anchuras} valor(es) distinto(s); "
              f"qcut necesita variación real en 'ancho')")
    print(f"  {'quintil del candidato (intra decil de anchura)':>48} {'mediana |error| MM':>20}")
    print("  " + "-" * 70)
    for q, v in piv.items():
        print(f"  {'Q'+str(int(q)+1):>48} {v:>20,.0f}")
    if len(piv) >= 2 and piv.iloc[0] > 0:
        ratio = piv.iloc[-1] / piv.iloc[0]
        print(f"\n  Rango dinámico intra-decil: {ratio:.2f}x")
        print("  >1.3x significa que el candidato separa filas de escala distinta")
        print("  que el modelo hoy trata igual — exactamente lo que falta.")

    # ── Veredicto ─────────────────────────────────────────────────────────────
    _titulo("Veredicto")
    fila_m = tab.loc[tab["candidato"] == mejor].iloc[0]
    fila_s = tab.loc[tab["candidato"] == "sigma_22d"]
    p_sig  = float(fila_s["parcial_ancho"].iloc[0]) if len(fila_s) else np.nan
    # Umbral sobre el PISO del IC, no sobre el punto. 0.05 y no 0.02 porque en el
    # mundo sintetico SIN estructura mensual apareció un espurio de +0.062 antes
    # de controlar por h; el margen protege de lo que quede sin controlar.
    UMBRAL_SIGNIF = 0.05
    signif = np.isfinite(fila_m["ic_lo"]) and (fila_m["ic_lo"] > UMBRAL_SIGNIF)

    print(f"  Mejor candidato          : {mejor}")
    print(f"  Parcial | ancho          : {fila_m['parcial_ancho']:+.3f} "
          f"[{fila_m['ic_lo']:+.3f}, {fila_m['ic_hi']:+.3f}]")
    print(f"  Competidor ({NOMBRE_COMPETIDOR or 'ninguno'})"
          f"{' ' * max(0, 12 - len(NOMBRE_COMPETIDOR or 'ninguno'))}: {p_sig:+.3f}")
    print(f"  Parcial | ancho + sigma  : {fila_m['parcial_ancho_sigma']:+.3f}")
    print()
    if not signif:
        print(f"  (criterio: piso del IC 90% > {UMBRAL_SIGNIF:+.2f})\n")
        print("  -> NO CONSTRUIR. El rezago de calendario no aporta escala por encima")
        print("     de lo que el modelo ya usa. Nos ahorramos tocar step001 y una")
        print("     corrida completa de revalidación.")
    elif np.isfinite(fila_m["parcial_ancho_sigma"]) and abs(fila_m["parcial_ancho_sigma"]) < UMBRAL_SIGNIF:
        print(f"  -> REDUNDANTE. Aporta sobre la anchura, pero NO sobre {NOMBRE_COMPETIDOR}:")
        print("     la información ya está en el modelo y el problema no es que")
        print("     falte la variable sino cómo la usa. Alinear por calendario no")
        print("     le gana a promediar.")
    else:
        print("  -> CONSTRUIR. Aporta información de escala incremental sobre la")
        print(f"     anchura predicha Y sobre {NOMBRE_COMPETIDOR}. Alinear por calendario sí")
        print("     preserva variación que promediar destruye.")
        print(f"     Alineación ganadora: {'hábiles' if mejor.endswith('hab') else 'calendario'}")
        print(f"     Serie ganadora     : {'flujo neto' if 'flujo' in mejor else 'retiro'}")

    # ── Salidas ───────────────────────────────────────────────────────────────
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    fecha = date.today().strftime("%Y%m%d")
    ruta_xl = DIR_OUT / f"test_lag_calendario_{BANCO}_{fecha}.xlsx"
    try:
        with pd.ExcelWriter(ruta_xl, engine="openpyxl") as xl:
            tab.to_excel(xl, sheet_name="candidatos", index=False)
            tab_h.to_excel(xl, sheet_name="por_horizonte", index=False)
            piv.reset_index().to_excel(xl, sheet_name="rango_dinamico", index=False)
        print(f"\n[OK] {ruta_xl.name}")
    except Exception as e:
        print(f"\n[AVISO] Excel no escrito ({e})")
        tab.to_csv(DIR_OUT / f"test_lag_candidatos_{fecha}.csv", index=False)

    graficar(tab, tab_h, sub, col_m, mejor, piv,
             DIR_OUT / f"test_lag_calendario_{BANCO}_{fecha}.png")


# ── Gráfico ───────────────────────────────────────────────────────────────────
def graficar(tab, tab_h, sub, col_m, mejor, piv, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[AVISO] matplotlib no disponible")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.26})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c3c2b7")
        ax.tick_params(colors=C_TINTA2, labelsize=8.5)
        ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)

    fig.suptitle("¿Aporta el rezago de calendario información de escala nueva?",
                 fontsize=13, fontweight="bold", color=C_TINTA, y=0.98)
    fig.text(0.5, 0.945,
             "Lo que decide es la correlación PARCIAL controlando por la anchura "
             "que el modelo ya predice, no la simple.",
             ha="center", fontsize=9, color=C_TINTA2, style="italic")

    # Panel 1 · simple vs parcial
    ax = axes[0, 0]
    t = tab.sort_values("parcial_ancho")
    yp = np.arange(len(t))
    cols = [C_COMP if c == "sigma_22d" else C_CAND for c in t["candidato"]]
    ax.barh(yp - 0.19, t["corr_simple"], height=0.34, color=cols, alpha=0.35, zorder=3)
    ax.barh(yp + 0.19, t["parcial_ancho"], height=0.34, color=cols, zorder=3)
    for i, (s, p) in enumerate(zip(t["corr_simple"], t["parcial_ancho"])):
        ax.text(s, i - 0.19, f" {s:+.2f}", va="center", fontsize=7.5, color=C_TINTA2)
        ax.text(p, i + 0.19, f" {p:+.2f}", va="center", fontsize=8, color=C_TINTA)
    ax.axvline(0, color=C_REF, lw=1.2)
    ax.set_yticks(yp); ax.set_yticklabels(t["candidato"], fontsize=8.5)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Spearman vs |error| realizado", fontsize=9, color=C_TINTA2)
    ax.set_title("1 · Simple (pálido) vs parcial | anchura (sólido)\n"
                 f"azul = {NOMBRE_COMPETIDOR or 'sin competidor'}, lo que el modelo ya tiene",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")

    # Panel 2 · parcial con IC
    ax = axes[0, 1]
    t2 = tab.dropna(subset=["ic_lo"]).sort_values("parcial_ancho")
    yp2 = np.arange(len(t2))
    cols2 = [C_COMP if c == "sigma_22d" else C_CAND for c in t2["candidato"]]
    ax.errorbar(t2["parcial_ancho"], yp2,
                xerr=[t2["parcial_ancho"] - t2["ic_lo"], t2["ic_hi"] - t2["parcial_ancho"]],
                fmt="none", ecolor=C_REF, elinewidth=1.4, capsize=4, zorder=3)
    ax.scatter(t2["parcial_ancho"], yp2, c=cols2, s=70, zorder=4)
    ax.axvline(0, color=C_REF, lw=1.2, ls="--")
    ax.set_yticks(yp2); ax.set_yticklabels(t2["candidato"], fontsize=8.5)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Parcial | anchura, con IC 90%", fontsize=9, color=C_TINTA2)
    ax.set_title("2 · ¿Sobrevive a la dependencia de los datos?\n"
                 "IC por bootstrap de bloques sobre fechas de origen",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")

    # Panel 3 · por horizonte
    ax = axes[1, 0]
    xh = np.arange(len(tab_h))
    ax.bar(xh - 0.19, tab_h["parcial_candidato"], width=0.36, color=C_CAND,
           zorder=3, label=mejor)
    ax.bar(xh + 0.19, tab_h["parcial_sigma"], width=0.36, color=C_COMP,
           zorder=3, label=NOMBRE_COMPETIDOR or "sin competidor")
    ax.axhline(0, color=C_REF, lw=1.2)
    ax.set_xticks(xh)
    ax.set_xticklabels(tab_h["tramo"], fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("Parcial | anchura", fontsize=9, color=C_TINTA2)
    ax.set_title("3 · Por tramo de horizonte\n"
                 "la disponibilidad de rezagos cae al alejarse",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=C_GRID, labelcolor=C_TINTA)

    # Panel 4 · rango dinámico intra-decil
    ax = axes[1, 1]
    xq = np.arange(len(piv))
    ax.bar(xq, piv.values, width=0.56, color=C_CAND, zorder=3)
    for i, v in enumerate(piv.values):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5, color=C_TINTA)
    ax.set_xticks(xq); ax.set_xticklabels([f"Q{i+1}" for i in xq], fontsize=9)
    ax.set_xlabel("Quintil del candidato, DENTRO del decil de anchura",
                  fontsize=9, color=C_TINTA2)
    ax.set_ylabel("Mediana del |error| (MM USD)", fontsize=9, color=C_TINTA2)
    ratio = piv.iloc[-1] / piv.iloc[0] if len(piv) >= 2 and piv.iloc[0] > 0 else np.nan
    ax.set_title(f"4 · Variación que el modelo se pierde: {ratio:.2f}x\n"
                 "filas que hoy reciben la misma banda y no deberían",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[OK] {ruta.name}")


if __name__ == "__main__":
    main()
