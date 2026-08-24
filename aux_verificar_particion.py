# -*- coding: utf-8 -*-
"""
aux_verificar_particion.py
==========================
Checklist de correctitud de las particiones del sistema de step001 v2.

Lo que se verifica:

  1. Exhaustividad y exclusividad   todo banco cae en foco o resto, nunca en
                                    ambos ni en ninguno
  2. Reconstrucción del total       FOCO + RESTO == suma de los bancos crudos,
                                    día por día, en R y en D
  3. Orden de llamada               partir DESPUÉS de agrupar_bancos() falla
                                    ruidosamente en vez de dar números malos
  4. Bancos chicos del foco         un global por debajo del umbral sigue
                                    quedando del lado del foco
  5. Aislamiento en el agrupamiento los agregados no compiten con bancos reales
                                    por el umbral ni caen en Otros_bancos
  6. SISTEMA no se duplica          la suma del total excluye los agregados
  7. Ambas particiones              "bbva" y "globales" dan grupos distintos y
                                    ambos reconstruyen el mismo total
  8. Sin partición                  PARTICION_ACTIVA=None reproduce v1

Uso:
    python aux_verificar_particion.py --sintetico
    python aux_verificar_particion.py --matriz RUTA.parquet

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
    """Importa step001 v2 sin ejecutar su main() ni pedir credenciales de proxy."""
    os.environ.setdefault("BCRP_PROXY", "http://verificacion-local-sin-red")
    spec = importlib.util.spec_from_file_location("_step001v2", RUTA_V2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Universo sintético: mezcla deliberada de bancos locales y globales, con dos
# globales DIMINUTOS (Deutsche e ICBC) muy por debajo del umbral del 1%. Son el
# caso que motiva todo el diseño: si la partición corriera después de agrupar,
# esos dos ya estarían dentro de Otros_bancos y terminarían contados como banca
# local, que es exactamente el error que no se puede ver a simple vista.
BANCOS = {
    "BBVA":        1000.0,
    "CREDITO":      900.0,
    "INTERBANK":    400.0,
    "SCOTIABANK":   350.0,
    "CITIBANK":      60.0,
    "PICHINCHA":     40.0,
    "DEUTSCHE":       2.0,   # global diminuto
    "ICBC":           1.5,   # global diminuto
    "MIBANCO":        3.0,   # local diminuto
}
GLOBALES_ESPERADOS = {"BBVA", "SCOTIABANK", "CITIBANK", "DEUTSCHE", "ICBC"}


def _pivot_sintetico(n=600, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2022-01-03", periods=n, freq="B")
    df = pd.DataFrame(index=idx)
    for b, escala in BANCOS.items():
        df[f"{b}_R"] = np.abs(rng.normal(escala, escala * 0.3, n))
        df[f"{b}_D"] = np.abs(rng.normal(escala, escala * 0.3, n))
    return df


def modo_sintetico():
    mod = _cargar_v2()
    df = _pivot_sintetico()
    bancos = sorted(BANCOS)
    print(f"\nUniverso sintético: {len(bancos)} bancos, {len(df)} días")
    print(f"Globales esperados : {sorted(GLOBALES_ESPERADOS)}\n")

    # ── 1-2. Por cada partición: exhaustividad, exclusividad, reconstrucción ─
    reportes = {}
    for nombre in ("bbva", "globales"):
        print(f"1-2. Partición '{nombre}'")
        dfp, rep = mod.aplicar_particion(df, nombre, nombre_otros="Otros_bancos")
        reportes[nombre] = (dfp, rep)
        foco, resto = rep["bancos_foco"], rep["bancos_resto"]

        check(f"{nombre}: exhaustiva (foco + resto cubre todo)",
              sorted(foco + resto) == bancos,
              f"{len(foco)} foco + {len(resto)} resto = {len(foco) + len(resto)}")
        check(f"{nombre}: exclusiva (sin solapamiento)",
              not (set(foco) & set(resto)))
        for s in ("_R", "_D"):
            total = dfp[[f"{b}{s}" for b in bancos]].sum(axis=1)
            suma = dfp[f"{rep['nombre_foco']}{s}"] + dfp[f"{rep['nombre_resto']}{s}"]
            peor = float((total - suma).abs().max())
            check(f"{nombre}: FOCO + RESTO == total en {s}", peor < 1e-6,
                  f"peor desvío {peor:.3g}")
        print()

    # ── 3. La clasificación es la correcta, no solo consistente ─────────────
    # Una partición puede cerrar perfecto y aun así poner los bancos del lado
    # equivocado. Se compara contra la lista esperada, no solo contra sí misma.
    print("3. Clasificación contra el resultado esperado")
    _, rep_g = reportes["globales"]
    check("globales: el foco es exactamente el conjunto esperado",
          set(rep_g["bancos_foco"]) == GLOBALES_ESPERADOS,
          f"obtenido = {sorted(rep_g['bancos_foco'])}")
    _, rep_b = reportes["bbva"]
    check("bbva: el foco es solo BBVA", set(rep_b["bancos_foco"]) == {"BBVA"},
          f"obtenido = {sorted(rep_b['bancos_foco'])}")
    check("las dos particiones dan grupos DISTINTOS",
          set(rep_b["bancos_foco"]) != set(rep_g["bancos_foco"]),
          "si coincidieran, 'globales' no estaría agregando nada")

    # ── 4. Los globales diminutos quedan del lado del foco ──────────────────
    print("\n4. Bancos del foco por debajo del umbral de agrupamiento")
    diminutos = {"DEUTSCHE", "ICBC"}
    check("Deutsche e ICBC entran al foco de 'globales'",
          diminutos <= set(rep_g["bancos_foco"]),
          "son los que se perderían si la partición corriera después de agrupar")

    # ── 5. Orden de llamada: partir después tiene que fallar ruidosamente ───
    print("\n5. Guardia de orden de llamada")
    df_ya_agrupado = df.copy()
    df_ya_agrupado["Otros_bancos_R"] = 0.0
    df_ya_agrupado["Otros_bancos_D"] = 0.0
    try:
        mod.aplicar_particion(df_ya_agrupado, "globales", nombre_otros="Otros_bancos")
        check("partir después de agrupar_bancos() aborta", False,
              "no lanzó excepción: el error pasaría inadvertido")
    except RuntimeError as e:
        check("partir después de agrupar_bancos() aborta", True,
              f"RuntimeError: {str(e)[:58]}...")

    # ── 6. Aislamiento de los agregados en agrupar_bancos ───────────────────
    print("\n6. Los agregados no participan del agrupamiento")
    dfp, rep = reportes["globales"]
    derivadas = mod.columnas_derivadas(rep)
    df_agr, lista, _ = mod.agrupar_bancos(dfp, 0.01, [], "Otros_bancos",
                                          excluir=derivadas)
    check("FOCO y RESTO no aparecen en lista_bancos",
          not ({rep["nombre_foco"], rep["nombre_resto"]} & set(lista)),
          f"lista = {lista}")
    # Si los agregados hubieran entrado al cálculo de volumen, los bancos reales
    # habrían perdido participación y alguno grande podría caer bajo el umbral.
    check("los bancos grandes siguen siendo individuales",
          {"BBVA", "CREDITO", "INTERBANK"} <= set(lista),
          f"lista = {lista}")
    check("los agregados sobreviven como columnas",
          all(f"{n}{s}" in df_agr.columns
              for n in (rep["nombre_foco"], rep["nombre_resto"]) for s in ("_R", "_D")))

    # ── 7. SISTEMA no se duplica ────────────────────────────────────────────
    # Réplica de la suma de build_full_matrix: sin la exclusión, FOCO y RESTO
    # entrarían al barrido por sufijo y el total saldría al doble.
    print("\n7. La suma de SISTEMA excluye los agregados")
    total_real = df[[f"{b}_R" for b in bancos]].sum(axis=1)
    con_excl = df_agr[[c for c in df_agr.columns
                       if c.endswith("_R") and c[:-2] not in derivadas]].sum(axis=1)
    sin_excl = df_agr[[c for c in df_agr.columns if c.endswith("_R")]].sum(axis=1)
    check("SISTEMA con exclusión == total real",
          float((con_excl - total_real).abs().max()) < 1e-6)
    ratio = float((sin_excl / total_real).mean())
    check("sin exclusión el total se DUPLICA (control negativo)",
          abs(ratio - 2.0) < 0.01,
          f"ratio = {ratio:.3f}× — confirma que la exclusión es necesaria")

    # ── 8. Sin partición reproduce v1 ───────────────────────────────────────
    print("\n8. PARTICION_ACTIVA = None")
    df_none, rep_none = mod.aplicar_particion(df, None)
    check("devuelve el DataFrame intacto", df_none.equals(df))
    check("reporte sin partición activa", rep_none.get("activa") is None)
    check("columnas_derivadas solo trae SISTEMA",
          mod.columnas_derivadas(rep_none) == {"SISTEMA"})

    # ── 9. Roles relativos de CCOVN ─────────────────────────────────────────
    # Regresión del bug de la colisión propio/sistema: para la entidad SISTEMA,
    # clave_propio vale "sistema", o sea que la fuente del rol "propio" es LA
    # MISMA columna que una de las comunes. La version con select + rename
    # duplicaba ccovn_propio_lag1 (df["..."] devolvia un DataFrame y la division
    # posterior reventaba con TypeError) y ademas hacia desaparecer
    # ccovn_sistema_lag1, que quedaba en NaN sin ningun aviso.
    #
    # Se ejercita armar_sub_ccovn(), la funcion de produccion, no una copia.
    print("\n9. Roles relativos de CCOVN (propio / contraparte)")
    _, rep_b = reportes["bbva"]
    idx = pd.bdate_range("2024-01-01", periods=60)
    ccovn = pd.DataFrame({
        "ccovn_sistema_lag1":     np.arange(60.) + 100,
        "var_ccovn_sistema_lag1": np.ones(60),
        "ccovn_foco_lag1":        np.arange(60.) * 0 + 40,
        "var_ccovn_foco_lag1":    np.ones(60) * 0.4,
        "ccovn_resto_lag1":       np.arange(60.) + 60,
        "var_ccovn_resto_lag1":   np.ones(60) * 0.6,
        "ccovn_vs_dia_mes_lag1":  np.zeros(60),
        "residuo_ccovn_lag1":     np.zeros(60),
    }, index=idx)

    esperado = {"SISTEMA": ("sistema", "foco"),
                "FOCO_BBVA": ("foco", "resto"),
                "RESTO_BBVA": ("resto", "foco")}
    for entidad, (cp_esp, cc_esp) in esperado.items():
        cp, cc = mod.resolver_ccovn_lados(entidad, "SISTEMA", rep_b)
        check(f"{entidad}: roles = ({cp_esp}, {cc_esp})", (cp, cc) == (cp_esp, cc_esp),
              f"obtenido = ({cp}, {cc})")

        sub = mod.armar_sub_ccovn(ccovn, cp, cc)   # lanza si hay duplicados
        check(f"{entidad}: sin columnas duplicadas en _sub",
              not sub.columns.duplicated().any())
        check(f"{entidad}: ccovn_sistema_lag1 sobrevive",
              "ccovn_sistema_lag1" in sub.columns,
              "si falta, el bucle de relleno la dejaria en NaN sin avisar")
        check(f"{entidad}: propio y contraparte presentes",
              {"ccovn_propio_lag1", "ccovn_contraparte_lag1"} <= set(sub.columns))

    # La contraparte de SISTEMA es el foco, que es lo que restituye la señal de
    # concentración que en v1 viajaba como bbva_share_lag1.
    sub_sis = mod.armar_sub_ccovn(ccovn, *mod.resolver_ccovn_lados("SISTEMA", "SISTEMA", rep_b))
    share_c = (sub_sis["ccovn_contraparte_lag1"] / sub_sis["ccovn_sistema_lag1"]).iloc[0]
    check("SISTEMA: share_contraparte es el foco sobre el total",
          abs(share_c - 40 / 100) < 1e-9, f"{share_c:.3f} (esperado 0.400)")
    check("SISTEMA: propio == sistema (share_propio seria constante 1)",
          bool((sub_sis["ccovn_propio_lag1"] == sub_sis["ccovn_sistema_lag1"]).all()),
          "por eso produccion lo anula en vez de emitir una constante")

    # Sin partición no hay contraparte, y el armado tiene que seguir cerrando.
    cp0, cc0 = mod.resolver_ccovn_lados("SISTEMA", "SISTEMA", {"activa": None})
    sub0 = mod.armar_sub_ccovn(ccovn, cp0, cc0)
    check("sin partición: SISTEMA no tiene contraparte", cc0 is None)
    check("sin partición: _sub igual queda sin duplicados",
          not sub0.columns.duplicated().any() and "ccovn_sistema_lag1" in sub0.columns)

    # ── 10. Partición inexistente ───────────────────────────────────────────
    print("\n10. Nombre de partición inválido")
    try:
        mod.aplicar_particion(df, "no_existe")
        check("una partición desconocida aborta", False, "no lanzó excepción")
    except ValueError as e:
        check("una partición desconocida aborta", True, f"ValueError: {str(e)[:50]}...")


def modo_matriz(ruta):
    p = pathlib.Path(ruta)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    print(f"\nMatriz: {p.name} — {len(df):,} filas, {df.shape[1]} columnas\n")

    print("1. Grupos presentes en la columna banco")
    check("existe la columna banco", "banco" in df.columns)
    if "banco" not in df.columns:
        return
    grupos = sorted(df["banco"].unique())
    print(f"   {grupos}")
    foco = [g for g in grupos if g.startswith("FOCO_")]
    resto = [g for g in grupos if g.startswith("RESTO_")]
    check("hay exactamente un FOCO y un RESTO", len(foco) == 1 and len(resto) == 1,
          f"foco={foco} resto={resto}")
    check("SISTEMA sigue presente", "SISTEMA" in grupos)
    if not (foco and resto):
        return

    print("\n2. El target de FOCO + RESTO reconstruye el de SISTEMA")
    # Es la verificación que importa sobre datos reales: si no cierra acá, los
    # dos modelos no están particionando el mismo objeto que el modelo agregado.
    col_t = next((c for c in ("target", "y", "target_neto") if c in df.columns), None)
    check("existe la columna target", col_t is not None, str(col_t))
    if col_t is None:
        return
    llaves = [c for c in ("fecha_t", "h") if c in df.columns]
    piv = df.pivot_table(index=llaves, columns="banco", values=col_t, aggfunc="first")
    comunes = piv[["SISTEMA", foco[0], resto[0]]].dropna()
    dif = (comunes["SISTEMA"] - comunes[foco[0]] - comunes[resto[0]]).abs()
    check("SISTEMA == FOCO + RESTO en el target",
          float(dif.max()) < 1e-3,
          f"{len(comunes):,} filas comparadas, peor desvío {dif.max():.4g}")

    print("\n3. Cobertura de features de encaje por grupo")
    if "encaje_lag1" in df.columns:
        cob = df.groupby("banco")["encaje_lag1"].apply(lambda s: s.notna().mean())
        print(cob.to_string(float_format=lambda v: f"{v:.1%}"))
        check(f"{foco[0]} recibe features de encaje", float(cob.get(foco[0], 0)) > 0.5,
              "si es 0%, el grupo del foco corre sin encaje y la comparación "
              "contra BBVA mide la falta de features, no la partición")
    else:
        print("   encaje_lag1 no está en la matriz (excluida en FEATURES_EXCLUIR)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # required=False a propósito: desde Spyder se corre con runfile() sin
    # argumentos, y un parser que aborta ahí obliga a editar el archivo o a
    # pasar por la consola. Sin argumentos cae al autotest sintético, que es el
    # modo que no necesita datos y por lo tanto el sensato por defecto.
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--sintetico", action="store_true")
    g.add_argument("--matriz", metavar="RUTA")
    a = ap.parse_args()
    if not a.sintetico and not a.matriz:
        a.sintetico = True
        print("(sin argumentos: corriendo el autotest sintético; "
              "para validar una matriz usar --matriz RUTA)\n")

    print("=" * 74)
    print("CHECKLIST — particiones del sistema (step001 v2)")
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
