# -*- coding: utf-8 -*-
"""
aux_verificar_pos_fijo.py
=========================
Checklist de correctitud de la familia *_pos_fx (rezagos de posición del mes con
CONTEO FIJO, LAGS_POSICION_MES_FIJO = [4,5,6,7]).

Dos modos:

  python aux_verificar_pos_fijo.py --sintetico
      Ejercita _build_lag_posicion_mes() sobre una serie construida a propósito.
      No necesita los datos de producción: corre en cualquier máquina y es el
      modo que prueba las propiedades que no dependen de los datos reales
      (look-ahead, conteo, contención entre juegos de rezagos).

  python aux_verificar_pos_fijo.py --matriz RUTA.parquet
      Valida la matriz ya construida: presencia de columnas, cobertura por
      horizonte y las mismas relaciones de contención sobre datos reales.

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

# ── Estado del checklist ─────────────────────────────────────────────────────
_FALLAS = []


def check(nombre, condicion, detalle=""):
    """Registra un ítem del checklist y lo imprime."""
    ok = bool(condicion)
    print(f"  [{'OK ' if ok else 'FALLA'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        _FALLAS.append(nombre)
    return ok


def _cargar_step001():
    """
    Importa step001 sin ejecutar su main(). Se usa importlib y no un import
    normal porque el nombre del módulo empieza con dígitos en algunos forks y
    porque así queda explícito que solo se quieren dos símbolos.

    step001 pide credenciales de proxy en el import si BCRP_PROXY no está en el
    entorno. Este script no hace ninguna llamada de red —solo ejercita funciones
    puras sobre series construidas en memoria— así que se declara un valor
    centinela para saltar el prompt. Si la variable ya existe no se toca, para
    no pisar una configuración real.
    """
    os.environ.setdefault("BCRP_PROXY", "http://verificacion-local-sin-red")
    spec = importlib.util.spec_from_file_location("_step001", RUTA_STEP001)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Modo sintético
# ─────────────────────────────────────────────────────────────────────────────
def modo_sintetico():
    mod = _cargar_step001()
    fn = mod._build_lag_posicion_mes
    LAGS_ADAPT = mod.LAGS_POSICION_MES
    LAGS_FIJO = mod.LAGS_POSICION_MES_FIJO

    print(f"\nRezagos adaptativos : {LAGS_ADAPT}")
    print(f"Rezagos conteo fijo : {LAGS_FIJO}")

    # Calendario de días hábiles Lun-Vie, 12 años. Sin feriados: la máscara
    # point-in-time y la aritmética de posición no dependen de ellos, y así el
    # test no queda atado al calendario peruano.
    peru_bday = pd.tseries.offsets.BDay()
    idx = pd.bdate_range("2014-01-01", "2026-06-30", freq=peru_bday)

    # SERIE TRAMPA: el valor de cada día es su propio ordinal de calendario.
    # Con esto, cualquier valor devuelto identifica sin ambigüedad la fecha de la
    # que salió, y el máximo devuelto es exactamente el ordinal de la fecha más
    # reciente que el feature llegó a mirar. Es lo que convierte el test de
    # look-ahead en una comparación aritmética exacta en vez de una inspección.
    serie_ordinal = pd.Series(np.arange(len(idx), dtype="float64"), index=idx)
    ordinal_de = pd.Series(np.arange(len(idx)), index=idx)

    # Orígenes: uno cada 5 ruedas, dejando 7 meses de calentamiento para que los
    # rezagos 4-7 existan.
    origenes = idx[(idx >= "2015-01-02")][::5]
    horizontes = [1, 5, 10, 21, 22, 40, 42, 43, 63, 64, 70, 75]

    filas_t, filas_th, filas_h = [], [], []
    pos_por_origen = {d: i for i, d in enumerate(idx)}
    for t in origenes:
        i = pos_por_origen[t]
        for h in horizontes:
            if i + h >= len(idx):
                continue
            filas_t.append(t)
            filas_th.append(idx[i + h])
            filas_h.append(h)

    fechas_t = pd.Series(filas_t)
    fechas_th = pd.Series(filas_th)
    hs = np.array(filas_h)
    print(f"Filas sintéticas    : {len(fechas_t):,} "
          f"({len(origenes):,} orígenes × {len(horizontes)} horizontes)\n")

    res = {}
    for etiqueta, lags in (("adapt", LAGS_ADAPT), ("fx", LAGS_FIJO)):
        mx, mn_s, mx_s, n = fn(serie_ordinal, fechas_th, fechas_t, lags, peru_bday)
        res[etiqueta] = {"max_abs": mx.to_numpy(), "min_sgn": mn_s.to_numpy(),
                         "max_sgn": mx_s.to_numpy(), "n": n.to_numpy()}

    ord_t = ordinal_de.reindex(pd.DatetimeIndex(fechas_t)).to_numpy()

    # ── 1. Sin look-ahead ───────────────────────────────────────────────────
    # El valor es el ordinal de la fecha de la que salió, así que el máximo
    # devuelto no puede superar el ordinal de fecha_t. Cualquier exceso es una
    # fecha futura entrando al feature.
    print("1. Máscara point-in-time (ningún rezago posterior a fecha_t)")
    for etiqueta in ("adapt", "fx"):
        v = res[etiqueta]["max_sgn"]
        hay = np.isfinite(v)
        exceso = np.where(hay, v - ord_t, -np.inf)
        peor = float(np.nanmax(exceso)) if hay.any() else -np.inf
        check(f"{etiqueta}: max(ordinal_rezago) <= ordinal(fecha_t)",
              peor <= 0,
              f"peor excedente = {peor:+.0f} ruedas")

    # ── 2. Conteo de rezagos ────────────────────────────────────────────────
    print("\n2. Conteo de rezagos por horizonte")
    n_ad, n_fx = res["adapt"]["n"], res["fx"]["n"]
    tabla = []
    for h in horizontes:
        m = hs == h
        tabla.append((h, int(np.median(n_ad[m])), int(np.median(n_fx[m])),
                      int(n_fx[m].min()), int(n_fx[m].max())))
    print("     h   n_adapt   n_fx(mediana)   n_fx(min-max)")
    for h, na, nf, lo, hi in tabla:
        print(f"   {h:>3}   {na:>7}   {nf:>13}   {lo:>6}-{hi}")

    # La premisa del bloque: el conteo adaptativo cae con h, el fijo no.
    n_ad_h = np.array([np.median(n_ad[hs == h]) for h in horizontes])
    n_fx_h = np.array([np.median(n_fx[hs == h]) for h in horizontes])
    check("n_adapt DECRECE con h (es el problema que motiva _fx)",
          n_ad_h[0] > n_ad_h[-1],
          f"h={horizontes[0]}: {n_ad_h[0]:.0f} rezagos → h={horizontes[-1]}: {n_ad_h[-1]:.0f}")
    check("n_fx es CONSTANTE en todo el rango de h",
          len(set(n_fx_h.tolist())) == 1,
          f"valores observados = {sorted(set(n_fx_h.tolist()))}")
    check(f"n_fx == {len(LAGS_FIJO)} (los {len(LAGS_FIJO)} rezagos entran siempre)",
          bool((n_fx_h == len(LAGS_FIJO)).all()))

    # ── 3. Contención de extremos donde los juegos se solapan ────────────────
    # Los rezagos adaptativos sobreviven de mayor a menor: el rezago k exige
    # h <= 21k, así que a medida que h crece van cayendo los CHICOS primero y el
    # conjunto vivo es {k_min..4}. La contención con {4,5,6,7} se da entonces
    # exactamente cuando queda un solo rezago, o sea n_adapt == 1 -> {4}.
    #
    # La condición se toma del n devuelto y NO de un umbral de h: la regla
    # h <= 21k es una aproximación (los meses tienen 19-23 ruedas y el lookup usa
    # aritmética de mes calendario, no de 21 días), así que el h en que cae el
    # tercer rezago se corre unas ruedas según el calendario. Fijar h a mano
    # metería filas con el rezago 3 vivo, que no está en el juego fijo, y el test
    # fallaría por construcción y no por un defecto del feature.
    print("\n3. Contención de extremos donde adapt ⊆ fx (n_adapt == 1)")
    solo4 = res["adapt"]["n"] == 1
    check("existe un tramo con n_adapt == 1 para comparar", bool(solo4.any()),
          f"{int(solo4.sum()):,} filas, h ∈ "
          f"[{hs[solo4].min() if solo4.any() else '-'}, "
          f"{hs[solo4].max() if solo4.any() else '-'}]")
    m = solo4 & np.isfinite(res["adapt"]["min_sgn"]) & np.isfinite(res["fx"]["min_sgn"])
    check("min_fx <= min_adapt",
          bool((res["fx"]["min_sgn"][m] <= res["adapt"]["min_sgn"][m] + 1e-9).all()),
          f"{m.sum():,} filas comparadas")
    m2 = solo4 & np.isfinite(res["adapt"]["max_sgn"]) & np.isfinite(res["fx"]["max_sgn"])
    check("max_fx >= max_adapt",
          bool((res["fx"]["max_sgn"][m2] >= res["adapt"]["max_sgn"][m2] - 1e-9).all()),
          f"{m2.sum():,} filas comparadas")

    # Control negativo: donde el adaptativo TODAVÍA tiene rezagos que el fijo no
    # cubre (n_adapt >= 2, o sea el rezago 3 o menores siguen vivos), la
    # contención no tiene por qué valer. Se comprueba que efectivamente NO vale,
    # porque si valiera igual sería señal de que los dos juegos están devolviendo
    # lo mismo y el bloque _fx no está usando los rezagos que declara.
    m3 = (res["adapt"]["n"] >= 2) & np.isfinite(res["adapt"]["max_sgn"]) \
        & np.isfinite(res["fx"]["max_sgn"])
    viola = float((res["fx"]["max_sgn"][m3] < res["adapt"]["max_sgn"][m3] - 1e-9).mean())
    check("control negativo: con n_adapt >= 2 la contención se rompe",
          viola > 0.0,
          f"{viola:.0%} de las filas la rompen, como corresponde")

    # ── 4. Cobertura ────────────────────────────────────────────────────────
    print("\n4. Cobertura (fracción no-NaN) por horizonte")
    cob_ad = np.array([np.isfinite(res["adapt"]["min_sgn"][hs == h]).mean() for h in horizontes])
    cob_fx = np.array([np.isfinite(res["fx"]["min_sgn"][hs == h]).mean() for h in horizontes])
    print("     h   cob_adapt   cob_fx")
    for h, a, f in zip(horizontes, cob_ad, cob_fx):
        print(f"   {h:>3}   {a:>9.1%}   {f:>6.1%}")
    check("cobertura _fx >= 99% en todo h", bool((cob_fx >= 0.99).all()),
          f"mínimo = {cob_fx.min():.1%}")

    # ── 5. Coherencia interna de los extremos ───────────────────────────────
    print("\n5. Coherencia de los tres extremos devueltos")
    for etiqueta in ("adapt", "fx"):
        r = res[etiqueta]
        m3 = np.isfinite(r["min_sgn"]) & np.isfinite(r["max_sgn"])
        check(f"{etiqueta}: min_sgn <= max_sgn",
              bool((r["min_sgn"][m3] <= r["max_sgn"][m3] + 1e-9).all()))
        m4 = np.isfinite(r["max_abs"]) & m3
        esperado = np.maximum(np.abs(r["min_sgn"][m4]), np.abs(r["max_sgn"][m4]))
        check(f"{etiqueta}: max_abs == max(|min_sgn|, |max_sgn|)",
              bool(np.allclose(r["max_abs"][m4], esperado, atol=1e-9)))

    # ── 6. Independencia del ancla ──────────────────────────────────────────
    # El bloque _fx usa la misma referencia que el adaptativo. Se comprueba que
    # cambiar el ancla efectivamente cambia el resultado, porque si no, el
    # parámetro estaría siendo ignorado y la comparación _ap/_fx no significaría
    # nada.
    print("\n6. El parámetro referencia se está aplicando")
    _, mn_ini, _, _ = fn(serie_ordinal, fechas_th, fechas_t, LAGS_FIJO,
                         peru_bday, referencia="inicio")
    m5 = np.isfinite(mn_ini.to_numpy()) & np.isfinite(res["fx"]["min_sgn"])
    difieren = float((mn_ini.to_numpy()[m5] != res["fx"]["min_sgn"][m5]).mean())
    check("referencia='inicio' difiere de 'cierre' en una fracción sustantiva",
          difieren > 0.30,
          f"difieren en {difieren:.0%} de las filas (esperado ~74% por la nota "
          f"de ANCLA_POSICION_MES)")


# ─────────────────────────────────────────────────────────────────────────────
# Modo matriz
# ─────────────────────────────────────────────────────────────────────────────
COLS_FX = ["esc_neto_min_pos_fx", "esc_neto_max_pos_fx", "esc_retiro_pos_fx"]
PARES = [("esc_neto_min_pos", "esc_neto_min_pos_fx", "min"),
         ("esc_neto_max_pos", "esc_neto_max_pos_fx", "max"),
         ("esc_retiro_pos",   "esc_retiro_pos_fx",   "max")]


def modo_matriz(ruta):
    p = pathlib.Path(ruta)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    print(f"\nMatriz: {p.name} — {len(df):,} filas, {df.shape[1]} columnas\n")

    print("1. Presencia de columnas")
    for c in COLS_FX:
        check(f"existe {c}", c in df.columns)
    if _FALLAS:
        return

    col_h = next((c for c in ("h", "horizonte", "horizonte_h") if c in df.columns), None)
    check("existe la columna de horizonte", col_h is not None, str(col_h))
    if col_h is None:
        return

    print("\n2. Cobertura por horizonte (adaptativa contra conteo fijo)")
    g = df.groupby(col_h)
    cob = pd.DataFrame({
        "adapt": g["esc_neto_min_pos"].apply(lambda s: s.notna().mean()),
        "fx":    g["esc_neto_min_pos_fx"].apply(lambda s: s.notna().mean()),
    })
    print(cob.to_string(float_format=lambda v: f"{v:.1%}"))
    check("cobertura _fx >= 99% en todo h", bool((cob["fx"] >= 0.99).all()),
          f"mínimo = {cob['fx'].min():.1%}")

    # La contención solo vale donde el juego adaptativo se redujo al rezago 4.
    # Con n_lags_pos en la matriz (GUARDAR_N_LAGS_POS = True) la condición es
    # exacta; sin él se usa un umbral de h conservador, porque el h en que cae el
    # tercer rezago depende del calendario y no de la regla aproximada h <= 21k.
    # En el autotest sintético ese corte quedó entre h=64 y h=70.
    if "n_lags_pos" in df.columns:
        print("\n3. Contención de extremos donde adapt ⊆ fx (n_lags_pos == 1)")
        alto = df[df["n_lags_pos"] == 1]
    else:
        print("\n3. Contención de extremos donde adapt ⊆ fx (h >= 70, aproximado)")
        print("   nota: para la condición exacta correr con GUARDAR_N_LAGS_POS = True")
        alto = df[df[col_h] >= 70]
    if len(alto) == 0:
        print("   (sin filas en el tramo comparable, se omite)")
    else:
        for c_ad, c_fx, modo in PARES:
            m = alto[c_ad].notna() & alto[c_fx].notna()
            if not m.any():
                continue
            if modo == "min":
                ok = (alto.loc[m, c_fx] <= alto.loc[m, c_ad] + 1e-6).all()
            else:
                ok = (alto.loc[m, c_fx] >= alto.loc[m, c_ad] - 1e-6).all()
            check(f"{c_fx} {'<=' if modo == 'min' else '>='} {c_ad}",
                  bool(ok), f"{int(m.sum()):,} filas")

    print("\n4. Correlación con su par adaptativo")
    # No es un test de aprobar/fallar: si la correlación fuera ~1 las columnas
    # serían redundantes y no valdría el costo de permutación en step005.
    for c_ad, c_fx, _ in PARES:
        m = df[c_ad].notna() & df[c_fx].notna()
        if m.sum() > 10:
            r = df.loc[m, c_ad].corr(df.loc[m, c_fx])
            print(f"   corr({c_ad}, {c_fx}) = {r:.3f}"
                  + ("   <- muy alta, revisar si aporta" if r > 0.97 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sintetico", action="store_true",
                   help="autotest sobre serie construida (no necesita datos)")
    g.add_argument("--matriz", metavar="RUTA",
                   help="valida una matriz de features ya construida")
    a = ap.parse_args()

    print("=" * 72)
    print("CHECKLIST — familia *_pos_fx (rezagos de posición, conteo fijo)")
    print("=" * 72)

    if a.sintetico:
        modo_sintetico()
    else:
        modo_matriz(a.matriz)

    print("\n" + "=" * 72)
    if _FALLAS:
        print(f"RESULTADO: {len(_FALLAS)} verificación(es) FALLARON")
        for f in _FALLAS:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULTADO: todas las verificaciones pasaron")
    print("=" * 72)


if __name__ == "__main__":
    main()
