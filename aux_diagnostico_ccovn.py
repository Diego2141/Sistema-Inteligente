# -*- coding: utf-8 -*-
"""
aux_diagnostico_ccovn.py
========================
¿Dónde se pierden los datos de CCOVN?

Nace de un caso concreto: tras actualizar Saldos_CCOVN.xlsx, la matriz exportada
salió con datos faltantes. La cadena tiene cuatro eslabones y cada uno puede
perder filas por un motivo distinto, así que adivinar cuál falló es caro. Este
script los recorre en orden y reporta el estado al final de cada uno:

    1. El Excel crudo          load_ccovn_data()      rango, columnas, NaN
    2. El emparejamiento       _mapear_bancos_ccovn() qué banco quedó sin match
    3. El df ancho             armar_ccovn_ancho()    sistema/foco/resto
    4. Las features            build_ccovn_features() cobertura por columna

Usa las funciones de PRODUCCIÓN, no copias de su lógica: si alguna cambia, este
diagnóstico cambia con ella en vez de quedar mintiendo.

Los tres modos de falla que se ven en la práctica:

  · ENCABEZADO QUE CAMBIÓ DE GRAFÍA. El emparejamiento es por nombre
    (ALIAS_CCOVN primero, subcadena normalizada después). Un banco que deja de
    emparejar queda con ccovn_propio_lag1 en NaN, pero ccovn_sistema_lag1 sigue
    bien: el sistema suma TODAS las columnas del Excel, empareje o no.

  · FECHAS QUE NO SON FECHAS. Si una columna de fecha viene como texto en
    algunas filas, pd.to_datetime las deja en NaT y esas filas desaparecen del
    merge sin ningún error.

  · RANGO MÁS CORTO QUE EL DE LA MATRIZ. Si el archivo nuevo arranca después o
    termina antes que el resto de las fuentes, las features quedan en NaN en los
    bordes. Es el caso más difícil de ver a ojo porque el archivo "está bien".

Uso
    python aux_diagnostico_ccovn.py
    python aux_diagnostico_ccovn.py --matriz "1. Data/Clean/matriz_features_particiones_bbva.parquet"

Desde Spyder, sin argumentos:
    runfile('aux_diagnostico_ccovn.py')
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys

import numpy as np
import pandas as pd

# step001 pide credenciales de proxy en el import si BCRP_PROXY no está
# definida. Se fija un centinela antes de importarlo para que no bloquee.
os.environ.setdefault("BCRP_PROXY", "http://centinela-sin-red")

AQUI = pathlib.Path(__file__).parent
RUTA_V2 = AQUI / "step001_build_feature_matrix_v2.py"

# Matriz a contrastar cuando se corre sin argumentos. Vacía o inexistente, el
# script hace igual los pasos 1-4 sobre el Excel, que es la parte que importa.
RUTA_MATRIZ_DEFECTO = "1. Data/Clean/matriz_features_particiones_bbva.parquet"


def cargar_v2():
    spec = importlib.util.spec_from_file_location("_s1v2", RUTA_V2)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s1v2"] = mod
    spec.loader.exec_module(mod)
    return mod


def seccion(n, titulo):
    print(f"\n{'='*74}\n{n}. {titulo}\n{'='*74}")


def diagnosticar(mod, ruta_matriz=None):
    params = dict(mod.PARAMS)

    # ── 1. El Excel crudo ────────────────────────────────────────────────────
    seccion(1, "El Excel crudo — load_ccovn_data()")
    ruta = params.get("ruta_ccovn", "")
    print(f"   ruta: {ruta}")
    if not ruta or not pathlib.Path(ruta).exists():
        print("   [FALLA] el archivo no existe en esa ruta — nada más que revisar")
        return
    raw = mod.load_ccovn_data(params)
    if raw.empty:
        print("   [FALLA] load_ccovn_data() devolvió vacío")
        return

    idx = pd.DatetimeIndex(raw.index)
    print(f"   filas: {len(raw):,}  |  columnas: {raw.shape[1]}")
    print(f"   rango: {idx.min().date()} → {idx.max().date()}")
    # Fechas no parseadas: el síntoma silencioso más común.
    n_nat = int(idx.isna().sum())
    print(f"   fechas NaT (no parsearon): {n_nat}" + ("  <-- REVISAR" if n_nat else ""))

    # Huecos grandes: un mes entero faltante no se ve en el rango total.
    dif = pd.Series(idx.sort_values()).diff().dt.days.dropna()
    huecos = dif[dif > 7]
    if len(huecos):
        print(f"   huecos > 7 días: {len(huecos)} (máx {int(huecos.max())} días)")
        for i in huecos.sort_values(ascending=False).head(3).index:
            print(f"     ~{pd.Timestamp(idx.sort_values()[i]).date()} "
                  f"({int(dif[i])} días sin dato)")
    else:
        print("   sin huecos mayores a 7 días")

    print("\n   Última fecha con dato, por columna del Excel:")
    fin_global = idx.max()
    for c in raw.columns:
        s = raw[c].dropna()
        if s.empty:
            print(f"     {str(c)[:34]:34s} SIN NINGÚN DATO  <-- REVISAR")
            continue
        ult = pd.Timestamp(s.index.max())
        atraso = (fin_global - ult).days
        marca = "  <-- se corta antes" if atraso > 7 else ""
        print(f"     {str(c)[:34]:34s} {ult.date()}  ({100*len(s)/len(raw):5.1f}% con dato){marca}")

    # ── 2. El emparejamiento ─────────────────────────────────────────────────
    seccion(2, "Emparejamiento banco → columna — _mapear_bancos_ccovn()")
    # Los canónicos salen de ALIAS_CCOVN más lo que haya en la matriz, para no
    # depender de tener los datos bancarios cargados.
    canonicos = sorted(set(mod.ALIAS_CCOVN) | {"BONY", "FEDERAL"})
    cols_raw = [c for c in raw.columns if c != "sistema"]
    mapeo = mod._mapear_bancos_ccovn(cols_raw, canonicos)
    sin_match = sorted(b for b, c in mapeo.items() if c is None)
    print(f"   {len(canonicos) - len(sin_match)}/{len(canonicos)} bancos emparejados")
    for b in canonicos:
        col = mapeo.get(b)
        print(f"     {b:12s} -> {col if col else 'SIN MATCH  <-- su ccovn_propio queda en NaN'}")
    if sin_match:
        print(f"\n   Sin match: {sin_match}")
        print("   Si alguno de estos NO estaba sin match antes de actualizar el")
        print("   archivo, cambió la grafía de su encabezado. Comparar contra")
        print("   ALIAS_CCOVN en step001 (línea ~1089).")

    # ── 3. El df ancho ───────────────────────────────────────────────────────
    seccion(3, "El df ancho — armar_ccovn_ancho()")
    for nombre, rep in [("sin partición", None),
                        ("partición bbva", {"activa": "bbva",
                                            "nombre_foco": "FOCO_BBVA",
                                            "nombre_resto": "RESTO_BBVA",
                                            "bancos_foco": ["BBVA"],
                                            "bancos_resto": [b for b in canonicos
                                                             if b != "BBVA"]})]:
        ancho, rep_match = mod.armar_ccovn_ancho(raw, canonicos, rep)
        if ancho.empty:
            print(f"   [{nombre}] devolvió vacío  <-- REVISAR")
            continue
        cols_clave = [c for c in ("sistema", "foco", "resto") if c in ancho.columns]
        cob = {c: f"{100*ancho[c].notna().mean():.1f}%" for c in cols_clave}
        print(f"   [{nombre}] columnas={ancho.shape[1]}  cobertura {cob}")
        if rep and rep_match.get("foco_sin_match"):
            print(f"      foco sin match: {rep_match['foco_sin_match']}  "
                  f"<-- subestima 'foco' Y 'resto' (resto = sistema - foco)")

    # ── 4. Las features ──────────────────────────────────────────────────────
    seccion(4, "Las features — build_ccovn_features()")
    ancho, _ = mod.armar_ccovn_ancho(raw, canonicos, None)
    # build_peru_calendar devuelve (peru_bday, peru_holidays, fechas_elecciones):
    # el calendario hábil es el elemento 0, no el 1.
    bday = mod.build_peru_calendar(params["años_calendario"])[0]
    try:
        feats = mod.build_ccovn_features(ancho, bday)
    except Exception as e:
        print(f"   [FALLA] build_ccovn_features abortó: {type(e).__name__}: {e}")
        return
    print(f"   {feats.shape[1]} columnas generadas, {len(feats):,} filas\n")
    print(f"   {'columna':38s} {'% con dato':>11s}  {'último dato':>12s}")
    for c in sorted(feats.columns):
        s = feats[c].dropna()
        ult = pd.Timestamp(s.index.max()).date() if len(s) else "—"
        pct = 100 * feats[c].notna().mean()
        marca = "  <--" if pct < 50 else ""
        print(f"   {c[:38]:38s} {pct:10.1f}% {str(ult):>13}{marca}")

    # ── 5. Contraste con la matriz exportada ─────────────────────────────────
    if not ruta_matriz or not pathlib.Path(ruta_matriz).exists():
        print(f"\n   (sin matriz para contrastar: {ruta_matriz})")
        return
    seccion(5, "Contraste con la matriz exportada")
    df = pd.read_parquet(ruta_matriz)
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    cc = sorted(c for c in df.columns if "ccovn" in c or c.startswith("share_"))
    if not cc:
        print("   la matriz no trae ninguna columna ccovn/share "
              "(¿todas en FEATURES_EXCLUIR?)")
        return
    print(f"   matriz: {len(df):,} filas, {df['fecha_t'].min().date()} → "
          f"{df['fecha_t'].max().date()}")

    # EL CONTRASTE DECISIVO. build_ccovn_features reindexa entre el min y el max
    # del PROPIO archivo CCOVN (línea ~2421) y hace ffill SIN LÍMITE. De ahí
    # salen los dos síntomas opuestos:
    #
    #   · Si el CCOVN EMPIEZA DESPUÉS que la matriz, el ffill no rellena hacia
    #     atrás: todas las fechas previas quedan en NaN. Es la causa más común
    #     de "faltan datos" justo después de actualizar el archivo, cuando la
    #     exportación trajo solo los años recientes.
    #   · Si TERMINA ANTES, esas fechas no existen en el reindex y también
    #     quedan en NaN.
    #
    # Y el reverso, más traicionero: una COLUMNA suelta que se corta antes del
    # fin del archivo NO deja NaN — el ffill la propaga como línea plana hasta
    # el final. No falta el dato: está congelado. Por eso la sección 1 reporta
    # la última fecha con dato POR COLUMNA y no solo el rango global.
    ini_cc, fin_cc = pd.Timestamp(idx.min()), pd.Timestamp(idx.max())
    ini_mx, fin_mx = df["fecha_t"].min(), df["fecha_t"].max()
    d_ini = (ini_cc - ini_mx).days
    d_fin = (fin_mx - fin_cc).days
    print(f"\n   CCOVN: {ini_cc.date()} → {fin_cc.date()}")
    if d_ini > 5:
        print(f"   [CAUSA PROBABLE] el CCOVN empieza {d_ini} días DESPUÉS que la "
              f"matriz.\n       El ffill no rellena hacia atrás: todo lo anterior "
              f"a {ini_cc.date()} queda en NaN.")
    if d_fin > 5:
        print(f"   [CAUSA PROBABLE] el CCOVN termina {d_fin} días ANTES que la "
              f"matriz.\n       Las fechas posteriores a {fin_cc.date()} quedan "
              f"en NaN.")
    if d_ini <= 5 and d_fin <= 5:
        print("   los rangos coinciden: el faltante no viene de la cobertura "
              "temporal del archivo")
    print(f"   columnas ccovn: {cc}\n")
    print(f"   {'entidad':14s} " + " ".join(f"{c[:16]:>17s}" for c in cc))
    for banco in sorted(df["banco"].unique()):
        sub = df[df["banco"] == banco]
        fila = " ".join(f"{100*sub[c].notna().mean():16.1f}%" for c in cc)
        print(f"   {banco:14s} {fila}")
    print("\n   Una entidad con 0% en ccovn_propio_lag1 y >0% en ccovn_sistema_lag1")
    print("   es el síntoma de un banco sin match (ver sección 2). Ambos en 0%")
    print("   apunta al Excel, no al emparejamiento.")

    # Dónde se corta, en el tiempo
    print("\n   Cobertura por año de la primera columna ccovn:")
    c0 = cc[0]
    por_anio = df.groupby(df["fecha_t"].dt.year)[c0].apply(lambda s: 100*s.notna().mean())
    for anio, pct in por_anio.items():
        marca = "  <--" if pct < 50 else ""
        print(f"     {anio}: {pct:5.1f}%{marca}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matriz", metavar="RUTA", default=None)
    a = ap.parse_args()
    ruta = a.matriz or RUTA_MATRIZ_DEFECTO

    print("=" * 74)
    print("DIAGNÓSTICO DE LA CADENA CCOVN")
    print("=" * 74)
    mod = cargar_v2()
    diagnosticar(mod, ruta)


if __name__ == "__main__":
    main()
