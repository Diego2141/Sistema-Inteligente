# -*- coding: utf-8 -*-
"""
aux_verificar_pos_ventana.py
============================
Checklist de correctitud de la VENTANA DESLIZANTE de la familia *_pos: de los
candidatos de LAGS_POSICION_MES entran los N_REZAGOS_OBJETIVO rezagos más
recientes que hayan sobrevivido a la máscara point-in-time.

Lo que se verifica:

  1. Máscara point-in-time      ningún rezago posterior a fecha_t
  2. Conteo constante            n == objetivo en todo el rango de h
  3. Identidad en h corto        hasta h=21 el resultado no cambia respecto al
                                 comportamiento previo (lista [1,2,3,4] sin tope)
  4. Recencia máxima             los rezagos elegidos son los MÁS RECIENTES
                                 disponibles, no unos cualesquiera
  5. Contención contra el previo el extremo sobre >= observaciones es más extremo
  6. Cobertura                   plana en h
  7. Coherencia interna          min <= max, max_abs == max(|min|,|max|)
  8. El ancla se aplica          referencia="inicio" != "cierre"

Dos modos:

  python aux_verificar_pos_ventana.py --sintetico
      Ejercita _build_lag_posicion_mes() sobre series construidas a propósito.
      No necesita los datos de producción.

  python aux_verificar_pos_ventana.py --matriz RUTA.parquet
      Valida la matriz ya construida.

Sale con código 1 si alguna verificación falla, para poder encadenarlo.
"""

import argparse
import os
import sys
import importlib.util
import pathlib

import numpy as np
import pandas as pd

RUTA_STEP001 = pathlib.Path(__file__).with_name("step001_build_feature_matrix.py")

_FALLAS = []


def check(nombre, condicion, detalle=""):
    ok = bool(condicion)
    print(f"  [{'OK ' if ok else 'FALLA'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        _FALLAS.append(nombre)
    return ok


def _cargar_step001():
    """
    Importa step001 sin ejecutar su main().

    step001 pide credenciales de proxy en el import si BCRP_PROXY no está en el
    entorno. Este script no hace ninguna llamada de red, solo ejercita funciones
    puras sobre series en memoria, así que se declara un centinela para saltar el
    prompt. Si la variable ya existe no se toca.
    """
    os.environ.setdefault("BCRP_PROXY", "http://verificacion-local-sin-red")
    spec = importlib.util.spec_from_file_location("_step001", RUTA_STEP001)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Modo sintético
# ─────────────────────────────────────────────────────────────────────────────
HORIZONTES = [1, 5, 10, 21, 22, 40, 42, 43, 63, 64, 70, 75]


def modo_sintetico():
    mod = _cargar_step001()
    fn = mod._build_lag_posicion_mes
    CAND = mod.LAGS_POSICION_MES
    OBJ = mod.N_REZAGOS_OBJETIVO

    print(f"\nCandidatos de rezago : {CAND}")
    print(f"Rezagos objetivo     : {OBJ}")

    peru_bday = pd.tseries.offsets.BDay()
    idx = pd.bdate_range("2013-01-01", "2026-06-30", freq=peru_bday)

    # SERIE TRAMPA: el valor de cada día es su propio ordinal de calendario. Con
    # esto cualquier valor devuelto identifica sin ambigüedad la fecha de la que
    # salió. Es lo que permite convertir "¿usó el futuro?" y "¿eligió los más
    # recientes?" en comparaciones aritméticas exactas en vez de inspecciones.
    serie_ordinal = pd.Series(np.arange(len(idx), dtype="float64"), index=idx)
    ordinal_de = pd.Series(np.arange(len(idx)), index=idx)

    pos = {d: i for i, d in enumerate(idx)}
    filas_t, filas_th, filas_h = [], [], []
    for t in idx[idx >= "2014-06-02"][::5]:
        i = pos[t]
        for h in HORIZONTES:
            if i + h < len(idx):
                filas_t.append(t); filas_th.append(idx[i + h]); filas_h.append(h)
    fechas_t, fechas_th = pd.Series(filas_t), pd.Series(filas_th)
    hs = np.array(filas_h)
    print(f"Filas sintéticas     : {len(fechas_t):,}\n")

    ord_t = ordinal_de.reindex(pd.DatetimeIndex(fechas_t)).to_numpy()

    def corre(lags, n_obj, ref="cierre", serie=serie_ordinal):
        mx, mn, mxs, n = fn(serie, fechas_th, fechas_t, lags, peru_bday,
                            referencia=ref, n_objetivo=n_obj)
        return {"max_abs": mx.to_numpy(), "min": mn.to_numpy(),
                "max": mxs.to_numpy(), "n": n.to_numpy()}

    vent = corre(CAND, OBJ)                 # ventana deslizante (nueva)
    prev = corre([1, 2, 3, 4], None)        # comportamiento anterior
    todos = corre(CAND, None)               # candidatos sin tope, para recencia

    # ── 1. Máscara point-in-time ────────────────────────────────────────────
    print("1. Máscara point-in-time (ningún rezago posterior a fecha_t)")
    hay = np.isfinite(vent["max"])
    peor = float(np.max(np.where(hay, vent["max"] - ord_t, -np.inf)))
    check("max(ordinal_rezago) <= ordinal(fecha_t)", peor <= 0,
          f"peor excedente = {peor:+.0f} ruedas")

    # ── 2. Conteo constante ─────────────────────────────────────────────────
    print("\n2. Conteo de rezagos por horizonte")
    print("      h   n_previo   n_ventana   rezagos elegidos (mediana)")
    n_prev_h, n_vent_h = [], []
    for h in HORIZONTES:
        m = hs == h
        npv, nvt = int(np.median(prev["n"][m])), int(np.median(vent["n"][m]))
        n_prev_h.append(npv); n_vent_h.append(nvt)
        # El rezago más reciente elegido se deduce del ordinal: cuánto más
        # antiguo que fecha_t es el valor más nuevo que entró.
        atraso = int(np.median(ord_t[m] - vent["max_abs"][m])) if m.any() else 0
        print(f"   {h:>4}   {npv:>8}   {nvt:>9}   ~{atraso} ruedas de atraso el más nuevo")
    check("n_previo DECRECE con h (el problema que se corrige)",
          n_prev_h[0] > n_prev_h[-1],
          f"h={HORIZONTES[0]}: {n_prev_h[0]} → h={HORIZONTES[-1]}: {n_prev_h[-1]}")
    check("n_ventana es CONSTANTE en todo el rango de h",
          len(set(n_vent_h)) == 1, f"valores = {sorted(set(n_vent_h))}")
    check(f"n_ventana == {OBJ}", all(v == OBJ for v in n_vent_h))

    # ── 3. Identidad donde el previo ya llegaba al objetivo ─────────────────
    # Donde el comportamiento previo ya tenía sus 4 rezagos, la ventana elige
    # exactamente los mismos y el resultado tiene que ser IDÉNTICO. Es lo que
    # deja intactas las mediciones de importancia ya hechas en ese tramo.
    #
    # La condición se toma del contador y NO de un umbral de h. La primera
    # versión de este test usaba h <= 21 y falló en `min`: en el borde de la
    # muestra y en los recortes por largo de mes el previo se quedaba con 3
    # rezagos, y ahí la ventana baja a buscar un quinto candidato para completar
    # el cupo. Con la serie ordinal el `max` lo fija el rezago más nuevo (el
    # mismo en ambos) pero el `min` lo fija el más viejo, que pasa a ser más
    # viejo. O sea que difiere justo donde la ventana MEJORA. Es el mismo error
    # que fijar el umbral por la regla h <= 21k: hay que leer el conteo real.
    print("\n3. Identidad con el previo donde este ya llegaba al objetivo")
    lleno = prev["n"] == OBJ
    corto = hs <= 21
    print(f"   filas con n_previo == {OBJ}: {lleno.mean():.1%} del total, "
          f"{lleno[corto].mean():.1%} dentro de h <= 21")
    for campo in ("min", "max", "max_abs"):
        a, b = vent[campo][lleno], prev[campo][lleno]
        check(f"{campo} idéntico al previo cuando n_previo == {OBJ}",
              np.allclose(np.nan_to_num(a, nan=-9e18),
                          np.nan_to_num(b, nan=-9e18), atol=1e-9),
              f"{int(lleno.sum()):,} filas")
    # Donde el previo se quedaba corto, la ventana tiene que haber tomado MÁS,
    # que es exactamente la mejora que se buscaba.
    corto_prev = prev["n"] < OBJ
    check(f"donde n_previo < {OBJ}, la ventana toma más rezagos",
          bool((vent["n"][corto_prev] >= prev["n"][corto_prev]).all())
          and bool((vent["n"][corto_prev] > prev["n"][corto_prev]).any()),
          f"{int(corto_prev.sum()):,} filas afectadas")
    largo = hs > 21
    dif = float((~np.isclose(np.nan_to_num(vent["min"][largo], nan=-9e18),
                             np.nan_to_num(prev["min"][largo], nan=-9e18))).mean())
    check("y DIFIERE en h > 21 (si no, la ventana no está haciendo nada)",
          dif > 0.5, f"difiere en {dif:.0%} de las filas")

    # ── 4. Recencia máxima ──────────────────────────────────────────────────
    # El valor más nuevo que entró con la ventana tiene que ser el mismo que
    # entraría sin tope: si la ventana eligiera rezagos viejos en vez de los
    # recientes, el máximo ordinal sería menor.
    print("\n4. Los rezagos elegidos son los MÁS RECIENTES disponibles")
    m = np.isfinite(vent["max"]) & np.isfinite(todos["max"])
    check("max(ordinal) con ventana == max(ordinal) sin tope",
          bool(np.allclose(vent["max"][m], todos["max"][m], atol=1e-9)),
          f"{int(m.sum()):,} filas — el rezago más nuevo es el mismo")

    # ── 5. Contención contra el previo ──────────────────────────────────────
    # Donde la ventana toma más observaciones que el previo, el conjunto del
    # previo está contenido en el de la ventana (ambos parten del mismo rezago
    # más reciente), así que el mínimo solo puede bajar y el máximo subir.
    print("\n5. Contención donde la ventana toma más rezagos que el previo")
    mas = vent["n"] > prev["n"]
    m1 = mas & np.isfinite(vent["min"]) & np.isfinite(prev["min"])
    check("min_ventana <= min_previo",
          bool((vent["min"][m1] <= prev["min"][m1] + 1e-9).all()),
          f"{int(m1.sum()):,} filas")
    m2 = mas & np.isfinite(vent["max"]) & np.isfinite(prev["max"])
    check("max_ventana >= max_previo",
          bool((vent["max"][m2] >= prev["max"][m2] - 1e-9).all()),
          f"{int(m2.sum()):,} filas")

    # ── 6. Cobertura ────────────────────────────────────────────────────────
    print("\n6. Cobertura (fracción no-NaN) por horizonte")
    cob = np.array([np.isfinite(vent["min"][hs == h]).mean() for h in HORIZONTES])
    print("      h   cobertura")
    for h, c in zip(HORIZONTES, cob):
        print(f"   {h:>4}   {c:>9.1%}")
    check("cobertura >= 99% en todo h", bool((cob >= 0.99).all()),
          f"mínimo = {cob.min():.1%}")

    # ── 7. Coherencia interna ───────────────────────────────────────────────
    print("\n7. Coherencia de los tres extremos devueltos")
    m3 = np.isfinite(vent["min"]) & np.isfinite(vent["max"])
    check("min <= max", bool((vent["min"][m3] <= vent["max"][m3] + 1e-9).all()))
    # Con la serie ordinal todo es positivo, así que para este test hace falta
    # una serie con signo: si no, max_abs coincide con max por construcción y la
    # verificación no probaría nada.
    rng = np.random.default_rng(0)
    serie_sgn = pd.Series(rng.normal(size=len(idx)), index=idx)
    vs = corre(CAND, OBJ, serie=serie_sgn)
    m4 = np.isfinite(vs["max_abs"]) & np.isfinite(vs["min"]) & np.isfinite(vs["max"])
    check("max_abs == max(|min|, |max|) (serie con signo)",
          bool(np.allclose(vs["max_abs"][m4],
                           np.maximum(np.abs(vs["min"][m4]), np.abs(vs["max"][m4])),
                           atol=1e-9)),
          f"{int(m4.sum()):,} filas")
    check("la serie con signo produce mínimos negativos",
          bool((vs["min"][m4] < 0).any()), "si no, el test anterior es vacío")

    # ── 8. El ancla se aplica ───────────────────────────────────────────────
    print("\n8. El parámetro referencia se está aplicando")
    ini = corre(CAND, OBJ, ref="inicio")
    m5 = np.isfinite(ini["min"]) & np.isfinite(vent["min"])
    d = float((ini["min"][m5] != vent["min"][m5]).mean())
    check("referencia='inicio' difiere de 'cierre'", d > 0.30,
          f"difieren en {d:.0%} de las filas (nota de ANCLA_POSICION_MES: ~74%)")


# ─────────────────────────────────────────────────────────────────────────────
# Modo matriz
# ─────────────────────────────────────────────────────────────────────────────
COLS = ["esc_neto_min_pos", "esc_neto_max_pos", "esc_retiro_pos",
        "acum_neto_min_pos", "acum_neto_max_pos",
        "esc_neto_max_pos_ap", "esc_deposito_pos_ap"]


def modo_matriz(ruta):
    p = pathlib.Path(ruta)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    print(f"\nMatriz: {p.name} — {len(df):,} filas, {df.shape[1]} columnas\n")

    print("1. Presencia de columnas de la familia")
    for c in COLS:
        check(f"existe {c}", c in df.columns)

    print("\n2. Ninguna columna _fx sobreviviente")
    # El intento anterior (familia paralela de conteo fijo) se revirtió. Si
    # quedan columnas _fx la matriz se construyó con una versión vieja del
    # código y las comparaciones no serían contra lo que dice el diccionario.
    fx = [c for c in df.columns if c.endswith("_pos_fx")]
    check("sin columnas *_pos_fx", not fx, str(fx) if fx else "ninguna")

    col_h = next((c for c in ("h", "horizonte", "horizonte_h") if c in df.columns), None)
    check("existe la columna de horizonte", col_h is not None, str(col_h))
    if col_h is None or _FALLAS:
        return

    print("\n3. Cobertura por horizonte (debe ser plana, no decreciente)")
    g = df.groupby(col_h)["esc_neto_min_pos"].apply(lambda s: s.notna().mean())
    print(g.to_string(float_format=lambda v: f"{v:.1%}"))
    check("cobertura >= 99% en todo h", bool((g >= 0.99).all()),
          f"mínimo = {g.min():.1%} en h={int(g.idxmin())}")
    # La firma del bug que se corrigió era cobertura o conteo cayendo con h. Que
    # la cobertura sea plana no prueba que el conteo lo sea, para eso hace falta
    # n_lags_pos, pero una cobertura decreciente sí lo desmentiría.
    check("cobertura no decrece entre el h mínimo y el máximo",
          float(g.iloc[-1]) >= float(g.iloc[0]) - 0.01,
          f"h={int(g.index[0])}: {g.iloc[0]:.1%} → h={int(g.index[-1])}: {g.iloc[-1]:.1%}")

    if "n_lags_pos" in df.columns:
        print("\n4. Conteo de rezagos (columna presente)")
        gn = df.groupby(col_h)["n_lags_pos"].median()
        print(gn.to_string())
        check("n_lags_pos constante en todo h", gn.nunique() == 1,
              f"valores = {sorted(gn.unique().tolist())}")
    else:
        print("\n4. Conteo de rezagos")
        print("   n_lags_pos no está en la matriz (GUARDAR_N_LAGS_POS = False).")
        print("   Para verificar el conteo directamente, correr step001 con esa")
        print("   bandera en True. El log de step001 ya reporta el reparto.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # required=False a propósito: desde Spyder se corre con runfile() sin
    # argumentos, y un parser que aborta ahí obliga a editar el archivo o a
    # pasar por la consola. Sin argumentos cae al autotest sintético, que es el
    # modo que no necesita datos y por lo tanto el sensato por defecto.
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--sintetico", action="store_true",
                   help="autotest sobre series construidas (no necesita datos)")
    g.add_argument("--matriz", metavar="RUTA",
                   help="valida una matriz de features ya construida")
    a = ap.parse_args()
    if not a.sintetico and not a.matriz:
        a.sintetico = True
        print("(sin argumentos: corriendo el autotest sintético; "
              "para validar una matriz usar --matriz RUTA)\n")

    print("=" * 74)
    print("CHECKLIST — ventana deslizante de rezagos, familia *_pos")
    print("=" * 74)

    modo_sintetico() if a.sintetico else modo_matriz(a.matriz)

    print("\n" + "=" * 74)
    if _FALLAS:
        print(f"RESULTADO: {len(_FALLAS)} verificación(es) FALLARON")
        for f in _FALLAS:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULTADO: todas las verificaciones pasaron")
    print("=" * 74)


if __name__ == "__main__":
    main()
