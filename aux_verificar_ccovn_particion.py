# -*- coding: utf-8 -*-
"""
aux_verificar_ccovn_particion.py
=================================
Checklist de correctitud de las features CC+OVN (Saldos_CCOVN.xlsx) resueltas
por entidad (propio/contraparte) en step001 v2.

Lo que se verifica:

  1. Sin partición activa    propio == sistema, contraparte == None, para
                             cualquier banco individual (equivalente a v1 salvo
                             el renombre)
  2. SISTEMA                 propio == sistema SIEMPRE, contraparte == None,
                             con y sin partición
  3. FOCO / RESTO            contraparte mutua, y ccovn_foco + ccovn_resto
                             reconstruye ccovn_sistema (misma identidad que ya
                             se exige sobre el target en aux_verificar_particion)
  4. Banco individual == foco  cuando la composición del foco es exactamente un
                             banco (ej. "bbva"), ese banco hereda propio=foco y
                             contraparte=resto — mismo valor que FOCO_x
  5. Banco fuera de la partición  propio = su propio saldo, contraparte = None
                             (no hereda nada de un lado que no le corresponde)
  6. Emparejamiento CCOVN↔transacciones  cobertura de la clasificación de la
                             partición contra los headers de Saldos_CCOVN.xlsx,
                             con un caso de headers deliberadamente distintos
                             (mayúsculas, acentos, razón social completa)
  7. share y exceso recompuestos  ccovn_propio/ccovn_sistema y el componente
                             idiosincrático dan lo mismo calculados antes o
                             después de rezagar (la propiedad que permite no
                             recalcular share dentro de build_ccovn_features)
  8. No hay leakage           todo shift(1), ningún valor de t entra a t-1
  9. Nombres viejos no sobreviven  ccovn_bbva_lag1 / var_ccovn_bbva_lag1 no
                             deberían aparecer en ninguna matriz nueva

Uso:
    python aux_verificar_ccovn_particion.py --sintetico
    python aux_verificar_ccovn_particion.py --matriz RUTA.parquet

Sale con código 1 si alguna verificación falla.
"""

import argparse
import os
import sys
import importlib.util
import pathlib

import numpy as np
import pandas as pd

RUTA_V2 = pathlib.Path(__file__).with_name("step001_build_feature_matrix_v2.py")

_FALLAS = []


def check(nombre, condicion, detalle=""):
    ok = bool(condicion)
    print(f"  [{'OK ' if ok else 'FALLA'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        _FALLAS.append(nombre)
    return ok


def _cargar_v2():
    os.environ.setdefault("BCRP_PROXY", "http://verificacion-local-sin-red")
    spec = importlib.util.spec_from_file_location("_step001v2", RUTA_V2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Universo sintético
# ─────────────────────────────────────────────────────────────────────────────
# Headers de Saldos_CCOVN.xlsx DELIBERADAMENTE distintos a los nombres canónicos
# de Transacciones_BancaLocal — es el caso real que el emparejamiento por
# subcadena tiene que resolver, y CITIBANK se deja SIN columna en absoluto para
# ejercitar la cobertura incompleta.
BANCOS_TX = ["BBVA", "CREDITO", "INTERBANK", "SCOTIABANK", "CITIBANK", "PICHINCHA"]
HEADERS_CCOVN = {
    "BBVA":       "BBVA Continental S.A.",
    "CREDITO":    "banco de crédito del perú",
    "INTERBANK":  "INTERBANK",
    "SCOTIABANK": "Scotiabank Perú",
    # CITIBANK: sin columna en CCOVN a propósito.
    "PICHINCHA":  "Banco Pichincha",
}


def _serie_sintetica(n=500, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2023-01-02", periods=n, freq="B")
    tx = pd.DataFrame(index=idx)
    for b in BANCOS_TX:
        esc = {"BBVA": 900, "CREDITO": 700, "INTERBANK": 300,
              "SCOTIABANK": 250, "CITIBANK": 60, "PICHINCHA": 40}[b]
        tx[f"{b}_R"] = np.abs(rng.normal(esc, esc * 0.3, n))
        tx[f"{b}_D"] = np.abs(rng.normal(esc, esc * 0.3, n))

    ccovn = pd.DataFrame(index=idx)
    for b, header in HEADERS_CCOVN.items():
        esc = {"BBVA": 5000, "CREDITO": 4000, "INTERBANK": 1500,
              "SCOTIABANK": 1200, "PICHINCHA": 200}[b]
        ccovn[header] = np.abs(rng.normal(esc, esc * 0.2, n)).cumsum() / 40 + esc
    ccovn["sistema"] = ccovn.sum(axis=1)
    return tx, ccovn


def modo_sintetico():
    mod = _cargar_v2()
    peru_bday = pd.tseries.offsets.BDay()
    df_tx, df_ccovn_raw = _serie_sintetica()

    # ── 6. Emparejamiento y cobertura, primero en aislado ───────────────────
    print("6. Emparejamiento de headers de CCOVN contra bancos canónicos")
    dfp_bbva, rep_bbva = mod.aplicar_particion(df_tx, "bbva")
    ancho, rep_match = mod.armar_ccovn_ancho(df_ccovn_raw, BANCOS_TX, rep_bbva)
    check("BBVA (razón social completa) se empareja", rep_match["mapeo"]["BBVA"] is not None,
          f"-> {rep_match['mapeo']['BBVA']!r}")
    check("CREDITO (minúsculas + acento) se empareja", rep_match["mapeo"]["CREDITO"] is not None,
          f"-> {rep_match['mapeo']['CREDITO']!r}")
    check("CITIBANK (sin columna) queda sin match", rep_match["mapeo"]["CITIBANK"] is None)
    check("el faltante se reporta explícito", "CITIBANK" in rep_match["sin_match"])
    # bbva_share activo en la particion "bbva": foco = {BBVA}, cobertura 100%
    check("cobertura del foco (partición bbva) == 100%",
          abs(rep_match.get("cobertura_foco", 0) - 1.0) < 1e-9,
          f"cobertura_foco = {rep_match.get('cobertura_foco')}")
    print()

    # Con la partición "globales" (foco incluye a CITIBANK, que no matchea):
    # la cobertura tiene que reflejar el hueco.
    dfp_glob, rep_glob = mod.aplicar_particion(df_tx, "globales")
    _, rep_match_g = mod.armar_ccovn_ancho(df_ccovn_raw, BANCOS_TX, rep_glob)
    n_foco_g = len(rep_glob["bancos_foco"])
    check("partición 'globales': cobertura del foco < 100% (CITIBANK sin match)",
          rep_match_g.get("cobertura_foco", 1.0) < 1.0,
          f"cobertura_foco = {rep_match_g.get('cobertura_foco'):.0%} "
          f"de {n_foco_g} bancos")

    # ── Pipeline completo para "bbva" ────────────────────────────────────────
    print("\nPipeline completo, partición 'bbva'")
    ccovn_feat = mod.build_ccovn_features(ancho, peru_bday)
    print(f"  Columnas generadas: {sorted(ccovn_feat.columns)}\n")

    NOMBRE_SISTEMA = "SISTEMA"

    def _resolver_y_extraer(banco):
        cp, cc = mod.resolver_ccovn_lados(banco, NOMBRE_SISTEMA, rep_bbva)
        propio = ccovn_feat.get(f"ccovn_{cp}_lag1")
        contra = ccovn_feat.get(f"ccovn_{cc}_lag1") if cc else None
        return cp, cc, propio, contra

    # ── 2. SISTEMA ───────────────────────────────────────────────────────────
    print("2. SISTEMA")
    cp, cc, propio, contra = _resolver_y_extraer(NOMBRE_SISTEMA)
    check("SISTEMA: clave propio == 'sistema'", cp == "sistema")
    # Con partición activa la contraparte de SISTEMA es el FOCO, y NO None.
    # Esta aserción decía lo contrario: se escribió antes de descubrir que sin
    # contraparte, SISTEMA se quedaba con share_propio == 1 por construcción y
    # perdía la señal de concentración que en v1 viajaba como bbva_share_lag1
    # (la variable del hallazgo 3). Quedó desactualizada tras ese cambio.
    check("SISTEMA: contraparte == 'foco' (restituye la concentración)",
          cc == "foco", f"obtenido {cc!r}")
    check("SISTEMA: propio == ccovn_sistema_lag1",
          propio.equals(ccovn_feat["ccovn_sistema_lag1"]))
    check("SISTEMA: contraparte == ccovn_foco_lag1",
          contra is not None and contra.equals(ccovn_feat["ccovn_foco_lag1"]))

    # ── 3. FOCO / RESTO ──────────────────────────────────────────────────────
    print("\n3. FOCO_BBVA / RESTO_BBVA")
    cp_f, cc_f, prop_f, cont_f = _resolver_y_extraer("FOCO_BBVA")
    cp_r, cc_r, prop_r, cont_r = _resolver_y_extraer("RESTO_BBVA")
    check("FOCO: propio='foco', contraparte='resto'", (cp_f, cc_f) == ("foco", "resto"))
    check("RESTO: propio='resto', contraparte='foco'", (cp_r, cc_r) == ("resto", "foco"))
    check("FOCO.propio == RESTO.contraparte (mismo objeto, dos ángulos)",
          prop_f.equals(cont_r))
    check("RESTO.propio == FOCO.contraparte", prop_r.equals(cont_f))
    m = prop_f.notna() & prop_r.notna() & ccovn_feat["ccovn_sistema_lag1"].notna()
    peor = float((prop_f[m] + prop_r[m] - ccovn_feat["ccovn_sistema_lag1"][m]).abs().max())
    check("ccovn_foco + ccovn_resto == ccovn_sistema (rezagados)", peor < 1e-6,
          f"peor desvío {peor:.3g}")

    # ── 4. Banco individual cuya composición == foco exacto ─────────────────
    print("\n4. BBVA individual (composición del foco == {BBVA})")
    cp_b, cc_b, prop_b, cont_b = _resolver_y_extraer("BBVA")
    check("BBVA: propio='banco_BBVA'", cp_b == "banco_BBVA")
    check("BBVA: contraparte='resto' (hereda el lado del foco)", cc_b == "resto")
    check("BBVA.propio == FOCO_BBVA.propio (BBVA es el único miembro del foco)",
          prop_b.equals(prop_f))
    check("BBVA.contraparte == RESTO_BBVA.propio", cont_b.equals(prop_r))

    # ── 5. Banco individual fuera de la partición ────────────────────────────
    print("\n5. CREDITO (parte de RESTO_BBVA, pero no es RESTO_BBVA exacto)")
    cp_c, cc_c, prop_c, cont_c = _resolver_y_extraer("CREDITO")
    check("CREDITO: propio='banco_CREDITO' (su propio saldo, no el del grupo)",
          cp_c == "banco_CREDITO")
    check("CREDITO: SIN contraparte (no es un lado exacto de la partición)",
          cc_c is None)
    check("CREDITO.propio != RESTO_BBVA.propio (CREDITO es solo una parte del resto)",
          not prop_c.dropna().equals(prop_r.reindex(prop_c.index).dropna()))

    # ── 1. Sin partición activa ───────────────────────────────────────────────
    print("\n1. Sin partición activa")
    ancho_sin, _ = mod.armar_ccovn_ancho(df_ccovn_raw, BANCOS_TX, {"activa": None})
    feat_sin = mod.build_ccovn_features(ancho_sin, peru_bday)
    cp_s, cc_s, prop_s, cont_s = mod.resolver_ccovn_lados("BBVA", NOMBRE_SISTEMA, None), None, None, None
    cp_s2, cc_s2 = mod.resolver_ccovn_lados("BBVA", NOMBRE_SISTEMA, None)
    check("sin partición, BBVA: propio='banco_BBVA', sin contraparte",
          (cp_s2, cc_s2) == ("banco_BBVA", None))
    check("sin partición, SISTEMA: propio='sistema', sin contraparte",
          mod.resolver_ccovn_lados(NOMBRE_SISTEMA, NOMBRE_SISTEMA, None) == ("sistema", None))

    # ── 7. share y exceso: propiedad de conmutación con el rezago ───────────
    print("\n7. share_propio y exceso: recomponer después de rezagar == antes")
    sis_lag = ccovn_feat["ccovn_sistema_lag1"]
    var_sis_lag = ccovn_feat["var_ccovn_sistema_lag1"]
    var_foco_lag = ccovn_feat["var_ccovn_foco_lag1"]
    share_despues = prop_f / sis_lag.replace(0, np.nan)
    exceso_despues = var_foco_lag - share_despues * var_sis_lag
    # Cálculo "antes": share y exceso sobre las series SIN rezagar, shift(1) al final —
    # exactamente como lo hacía v1 (bbva_share_lag1 / var_ccovn_bbva_exceso_lag1).
    sis_raw = ancho["sistema"]; foco_raw = ancho["foco"]
    idx_bd = pd.bdate_range(sis_raw.index.min(), sis_raw.index.max(), freq="B")
    sis_bd, foco_bd = sis_raw.reindex(idx_bd).ffill(), foco_raw.reindex(idx_bd).ffill()
    share_antes = (foco_bd / sis_bd.replace(0, np.nan)).shift(1)
    exceso_antes = (foco_bd.diff() - (foco_bd / sis_bd.replace(0, np.nan)) * sis_bd.diff()).shift(1)
    m2 = share_despues.notna() & share_antes.notna()
    check("share_propio: recomponer después de rezagar == calcular antes",
          bool(np.allclose(share_despues[m2], share_antes.reindex(share_despues.index)[m2], atol=1e-9)),
          f"{int(m2.sum()):,} filas comparadas")
    m3 = exceso_despues.notna() & exceso_antes.notna()
    check("var_ccovn_propio_exceso: recomponer después == calcular antes",
          bool(np.allclose(exceso_despues[m3], exceso_antes.reindex(exceso_despues.index)[m3], atol=1e-6)),
          f"{int(m3.sum()):,} filas comparadas")

    # ── 8. Sin leakage ────────────────────────────────────────────────────────
    print("\n8. Máscara temporal — todo es shift(1)")
    # Serie trampa: valor = ordinal de la fecha. ccovn_<clave>_lag1(t) tiene que
    # ser exactamente el ordinal de la rueda ANTERIOR a t, nunca el de t.
    idx = pd.bdate_range("2023-01-02", periods=200, freq="B")
    trampa = pd.DataFrame({"sistema": np.arange(len(idx), dtype="float64")}, index=idx)
    trampa["banco_X"] = trampa["sistema"]
    feat_trampa = mod.build_ccovn_features(trampa, peru_bday)
    ordinal = pd.Series(np.arange(len(idx)), index=idx)
    esperado = ordinal.shift(1)
    m4 = feat_trampa["ccovn_sistema_lag1"].notna() & esperado.notna()
    check("ccovn_sistema_lag1(t) == ordinal(t-1), nunca ordinal(t)",
          bool((feat_trampa["ccovn_sistema_lag1"][m4] == esperado[m4]).all()))

    # ── 9. Nombres viejos no sobreviven ──────────────────────────────────────
    print("\n9. Nombres de columna heredados de v1")
    check("ccovn_bbva_lag1 NO existe en la salida", "ccovn_bbva_lag1" not in ccovn_feat.columns)
    check("var_ccovn_bbva_lag1 NO existe en la salida", "var_ccovn_bbva_lag1" not in ccovn_feat.columns)


# ─────────────────────────────────────────────────────────────────────────────
def modo_matriz(ruta):
    p = pathlib.Path(ruta)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    print(f"\nMatriz: {p.name} — {len(df):,} filas, {df.shape[1]} columnas\n")

    print("1. Columnas nuevas presentes, viejas ausentes")
    nuevas = ["ccovn_propio_lag1", "var_ccovn_propio_lag1",
              "ccovn_contraparte_lag1", "var_ccovn_contraparte_lag1"]
    for c in nuevas:
        check(f"existe {c}", c in df.columns)
    for c in ("ccovn_bbva_lag1", "var_ccovn_bbva_lag1"):
        check(f"{c} ya no existe", c not in df.columns)

    if "banco" not in df.columns:
        print("  (sin columna 'banco', no se puede verificar por entidad)")
        return

    print("\n2. ccovn_propio_lag1 == ccovn_sistema_lag1 para SISTEMA")
    if {"ccovn_propio_lag1", "ccovn_sistema_lag1"} <= set(df.columns):
        sub = df[df["banco"] == "SISTEMA"]
        m = sub["ccovn_propio_lag1"].notna() & sub["ccovn_sistema_lag1"].notna()
        if m.any():
            peor = float((sub.loc[m, "ccovn_propio_lag1"]
                         - sub.loc[m, "ccovn_sistema_lag1"]).abs().max())
            check("SISTEMA: propio == sistema", peor < 1e-6, f"peor desvío {peor:.3g}")

    print("\n3. Cobertura de ccovn_contraparte_lag1 por grupo")
    foco = [g for g in df["banco"].unique() if str(g).startswith("FOCO_")]
    if foco and "ccovn_contraparte_lag1" in df.columns:
        cob = df.groupby("banco")["ccovn_contraparte_lag1"].apply(lambda s: s.notna().mean())
        print(cob.to_string(float_format=lambda v: f"{v:.1%}"))
        check(f"{foco[0]} tiene contraparte con cobertura razonable",
              float(cob.get(foco[0], 0)) > 0.5,
              "si es 0%, el grupo del foco no está recibiendo el saldo del resto")
        check("SISTEMA tiene contraparte en NaN",
              float(cob.get("SISTEMA", 0)) < 0.01 if "SISTEMA" in cob.index else True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--sintetico", action="store_true")
    g.add_argument("--matriz", metavar="RUTA")
    a = ap.parse_args()
    if not a.sintetico and not a.matriz:
        a.sintetico = True
        print("(sin argumentos: corriendo el autotest sintético; "
              "para validar una matriz usar --matriz RUTA)\n")

    print("=" * 74)
    print("CHECKLIST — CC+OVN resuelto por partición (propio/contraparte)")
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
