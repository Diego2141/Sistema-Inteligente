# -*- coding: utf-8 -*-
"""
aux_verificar_hmm_interno.py
=============================
Checklist de correctitud de la generacion interna de regimenes HMM y la
correlacion transversal rho_ij en step005_walk_forward_cv_3.7.py.

Que hace el codigo que este checklist verifica
-----------------------------------------------
step005_walk_forward_cv_3.7.py ya no depende de que step005_validar_hmm_v5.py
se haya corrido aparte con su constante BANCO cambiada a mano. Al arrancar
cada corrida, `asegurar_regimenes_hmm(BANCOS_A_EVALUAR, folds)`:

  1. Determina que entidades necesitan regimen: las de BANCOS_A_EVALUAR, MAS
     la contraparte de cada una (FOCO_<P> <-> RESTO_<P>) y SISTEMA — aunque
     no esten en la lista a evaluar. La contraparte hace falta porque rho_ij
     es una correlacion ENTRE los dos lados de la particion; SISTEMA porque
     el rho_ij condicional se condiciona por su regimen, no por el propio
     (dos HMMs distintos etiquetarian el mismo dia distinto, y usar el propio
     volveria rho_ij asimetrico entre FOCO y RESTO).
  2. Por cada una, comprueba si el parquet YA tiene un bloque (año_corte) por
     cada train_end de los folds actuales. Si no, importa
     step005_validar_hmm_v5, ajusta el HMM con esos train_end como cortes
     exactos, y guarda estados_regimen_hmm_<banco>.parquet + transmat +
     pickles por fold.
  3. Es deliberadamente barato: NO corre el walk-forward de XGBoost de la
     contraparte (Optuna, retrain, diagnosticos — minutos), solo el ajuste
     del HMM (segundos), porque rho_ij no consume ninguna columna que ese
     entrenamiento produzca.

Con los parquets ya alineados, dentro de cada fold:

  - _estimar_rho_val_fold(...)         -> phi_s (persistencia temporal, D1)
  - _estimar_rho_transversal(...)      -> rho_ij (correlacion entre grupos, D2)

usando exactamente la misma base (z = flujo/sigma, misma ponderacion EWMA,
misma formula de Pearson ponderada) para que sean compatibles con la ecuacion
Sigma_e = R o. (11' - phi phi') de la metodologia de agregacion por grupos.

Que se verifica aqui
---------------------
  1. Importabilidad          step005_validar_hmm_v5 no revienta al importarse
                              (regresion real: PEP 604 sin __future__.annotations
                              en Python < 3.10 — ver commit 1a6ab8c)
  2. Contraparte              FOCO_<P> <-> RESTO_<P>; SISTEMA y sin-particion
                              devuelven None (no tienen contraparte)
  3. Deteccion de cortes      un parquet con grilla anual, mientras los folds
                              avanzan cada 0.5 años, se detecta como NO
                              cubierto (control negativo — el bug que motivo
                              _cortes_cubren_folds en primer lugar)
  4. Expansion de entidades   pedir una sola entidad genera tambien su
                              contraparte y SISTEMA; sin particion NO inventa
                              contrapartes que no existen
  5. Idempotencia             una segunda pasada con los parquets ya alineados
                              no vuelve a ajustar nada
  6. rho_ij: exactitud        sobre series sinteticas de correlacion conocida
                              por construccion, con error < 0.06
  7. rho_ij: simetria         rho_ij calculado FOCO->RESTO == RESTO->FOCO,
                              exacto — es la propiedad que rompe si el
                              condicionamiento usa el regimen propio en vez
                              del de SISTEMA
  8. rho_ij: invariante       ante la escala de sigma (no deberia cambiar el
                              resultado — z ya esta estandarizado)
  9. rho_ij: guardas          muestra corta, entrada None, sigma=0 devuelven
                              {} en vez de reventar o devolver basura
  10. rho_ij: clip            dos series identicas dan <= 0.98, nunca 1.0
                              (rho=1 vuelve R singular, la factorizacion de
                              Cholesky del algoritmo de simulacion falla)

Uso
---
    python aux_verificar_hmm_interno.py                  (autotest sintetico)
    python aux_verificar_hmm_interno.py --matriz RUTA.parquet --banco FOCO_BBVA

Con --matriz se agrega una seccion adicional de verificaciones sobre datos
reales (existencia de parquets, cobertura de rho_ij en el ultimo fold,
simetria si el parquet de la contraparte tambien existe). Sin --matriz, todas
las funciones de PRODUCCION se ejercitan igual, mediante importlib —no una
reimplementacion— sobre datos sinteticos.

Sale con codigo 1 si alguna verificacion falla.
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.getLogger().setLevel(logging.WARNING)   # silenciar el ruido de cv3.7

RUTA_CV37 = Path(__file__).parent / "step005_walk_forward_cv_3.7.py"
RUTA_HMM5 = Path(__file__).parent / "step005_validar_hmm_v5.py"

# Ruta por defecto para el modo --matriz sin argumento explicito, igual
# convencion que aux_verificar_particion.py: si no esta montada la unidad de
# red, se cae al sintetico en vez de abortar.
RUTA_MATRIZ_DEFECTO = None
try:
    _p = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente"
              r"\1. Data\Clean\matriz_features_particiones_bbva.parquet")
    if _p.exists():
        RUTA_MATRIZ_DEFECTO = str(_p)
except OSError:
    pass

_FALLAS: list[str] = []


def _check(nombre: str, condicion: bool, detalle: str = "") -> bool:
    ok = bool(condicion)
    print(f"  [{'OK ' if ok else 'FALLA'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        _FALLAS.append(nombre)
    return ok


def _cargar_cv37():
    """
    Importa step005_walk_forward_cv_3.7.py por ruta de archivo, ejerciendo el
    codigo de PRODUCCION real (no una copia). main() esta bajo
    `if __name__ == "__main__"`, asi que importarlo no dispara ninguna
    corrida — solo define funciones y evalua los bloques de configuracion a
    nivel de modulo (PARTICIONES=False por defecto no dispara ningun assert).
    """
    spec = importlib.util.spec_from_file_location("_cv37_verif", RUTA_CV37)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Series sinteticas
# ─────────────────────────────────────────────────────────────────────────────

def _bloque(z, fechas, sigma=1.0, estado=0):
    """DataFrame con la forma exacta de estados_regimen_hmm_<banco>.parquet
    (columnas flujo/sigma/estado, indexado por fecha) que consumen
    _estimar_rho_val_fold y _estimar_rho_transversal."""
    n = len(z)
    est = estado if hasattr(estado, "__len__") else np.full(n, estado)
    return pd.DataFrame({"flujo": np.asarray(z) * sigma,
                         "sigma": np.full(n, sigma),
                         "estado": est},
                        index=pd.DatetimeIndex(fechas))


def _folds_sinteticos():
    return [{"fold": i + 1, "train_end": pd.Timestamp(d)} for i, d in
            enumerate(("2022-07-01", "2023-01-01", "2023-07-01"))]


# ─────────────────────────────────────────────────────────────────────────────
# Modo sintetico
# ─────────────────────────────────────────────────────────────────────────────

def modo_sintetico():
    print("\n--- 1. Importabilidad de step005_validar_hmm_v5 ---")
    try:
        spec = importlib.util.spec_from_file_location("_hmm5_verif", RUTA_HMM5)
        hmm5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hmm5)
        _check("step005_validar_hmm_v5 importa sin TypeError", True)
        _check("hmm_evolucion expuesta", hasattr(hmm5, "hmm_evolucion"))
        _check("cargar_datos acepta banco/ruta_matriz",
              {"banco", "ruta_matriz"} <= set(
                  hmm5.cargar_datos.__code__.co_varnames[
                      :hmm5.cargar_datos.__code__.co_argcount]))
    except Exception as e:
        _check("step005_validar_hmm_v5 importa sin TypeError", False,
              f"{type(e).__name__}: {e}")

    print("\n--- 2. Carga de funciones de produccion (step005_walk_forward_cv_3.7) ---")
    try:
        cv37 = _cargar_cv37()
        _check("step005_walk_forward_cv_3.7 importa", True)
    except Exception as e:
        _check("step005_walk_forward_cv_3.7 importa", False, f"{type(e).__name__}: {e}")
        return   # nada mas se puede probar sin el modulo

    print("\n--- 3. _banco_contraparte ---")
    casos = [("FOCO_BBVA", "RESTO_BBVA"), ("RESTO_BBVA", "FOCO_BBVA"),
             ("FOCO_GLOBALES", "RESTO_GLOBALES"), ("SISTEMA", None),
             ("Otros_bancos", None)]
    for banco, esperado in casos:
        got = cv37._banco_contraparte(banco)
        _check(f"contraparte({banco}) == {esperado}", got == esperado, f"obtuvo {got!r}")

    print("\n--- 4. _cortes_cubren_folds: control negativo (grilla desalineada) ---")
    tmp = Path("_verif_hmm_tmp"); tmp.mkdir(exist_ok=True)
    cv37.DIR_REGIMEN_HMM = tmp
    folds = _folds_sinteticos()
    train_ends = [f["train_end"] for f in folds]
    try:
        # Ausente
        _check("parquet ausente -> NO cubre",
              cv37._cortes_cubren_folds("X", train_ends) is False)
        # Grilla anual (desalineada respecto a folds cada 0.5 años)
        pd.DataFrame({"año_corte": pd.to_datetime(["2022-01-01", "2023-01-01"]),
                      "fecha": pd.to_datetime(["2022-01-01", "2023-01-01"])}
                     ).to_parquet(tmp / "estados_regimen_hmm_X.parquet")
        _check("cortes en grilla anual -> NO cubre (control negativo)",
              cv37._cortes_cubren_folds("X", train_ends) is False,
              "si esto da True, un HMM ajustado con una grilla distinta a la "
              "de los folds pasaria como 'ya cubierto' sin avisar")
        # Cortes exactos
        pd.DataFrame({"año_corte": train_ends, "fecha": train_ends}
                     ).to_parquet(tmp / "estados_regimen_hmm_X.parquet")
        _check("cortes == train_end -> SI cubre",
              cv37._cortes_cubren_folds("X", train_ends) is True)
    finally:
        for f in tmp.glob("*.parquet"):
            f.unlink()

    print("\n--- 5. asegurar_regimenes_hmm: expansion a contraparte + SISTEMA ---")
    import types
    generados = []

    def _fake_cargar_datos(banco, ruta_matriz):
        return pd.Series([1.0])

    def _fake_evolucion(flujo, fechas_corte):
        return {}

    def _fake_guardar(evol, flujo, banco, dir_output):
        generados.append(banco)
        pd.DataFrame({"año_corte": train_ends, "fecha": train_ends}
                     ).to_parquet(dir_output / f"estados_regimen_hmm_{banco}.parquet")

    fake = types.ModuleType("step005_validar_hmm_v5")
    fake._HMM_OK = True
    fake.cargar_datos = _fake_cargar_datos
    fake.hmm_evolucion = _fake_evolucion
    fake.guardar_objetos_simulacion = _fake_guardar
    sys.modules["step005_validar_hmm_v5"] = fake

    cv37.HMM_INTERNO = True
    cv37.USAR_FEATURE_REGIMEN = True
    cv37._regimen_cache.clear()
    cv37._hmm_generado.clear()
    try:
        cv37.asegurar_regimenes_hmm(["FOCO_BBVA"], folds)
        _check("pedir solo FOCO_BBVA genera tambien RESTO_BBVA",
              "RESTO_BBVA" in generados, f"generados={sorted(generados)}")
        _check("pedir solo FOCO_BBVA genera tambien SISTEMA",
              "SISTEMA" in generados, f"generados={sorted(generados)}")

        generados.clear()
        cv37._regimen_cache.clear()
        cv37._hmm_generado.clear()
        for f in tmp.glob("*.parquet"):
            f.unlink()
        cv37.asegurar_regimenes_hmm(["SISTEMA"], folds)
        _check("sin particion (SISTEMA solo) NO inventa contrapartes",
              generados == ["SISTEMA"], f"generados={sorted(generados)}")

        print("\n--- 6. asegurar_regimenes_hmm: idempotencia ---")
        generados.clear()
        cv37.asegurar_regimenes_hmm(["SISTEMA"], folds)
        _check("segunda pasada con parquets alineados no reajusta nada",
              generados == [], f"generados={sorted(generados)}")
    finally:
        del sys.modules["step005_validar_hmm_v5"]
        for f in tmp.glob("*.parquet"):
            f.unlink()
        tmp.rmdir()

    print("\n--- 7. _estimar_rho_transversal: exactitud sobre series conocidas ---")
    f = pd.bdate_range("2021-01-01", periods=600)
    rng = np.random.default_rng(7)
    for rho_true in (0.0, 0.5, -0.7, 0.9):
        a = rng.standard_normal(len(f))
        b = rho_true * a + np.sqrt(1 - rho_true**2) * rng.standard_normal(len(f))
        out = cv37._estimar_rho_transversal(_bloque(a, f), _bloque(b, f),
                                            tau_ewma_pares=1e6)
        err = abs(out["global"] - rho_true)
        _check(f"rho_true={rho_true:+.2f} recuperado (err={err:.4f})", err < 0.06)

    print("\n--- 8. _estimar_rho_transversal: simetria ---")
    a = rng.standard_normal(len(f))
    b = 0.6 * a + 0.8 * rng.standard_normal(len(f))
    r1 = cv37._estimar_rho_transversal(_bloque(a, f), _bloque(b, f))["global"]
    r2 = cv37._estimar_rho_transversal(_bloque(b, f), _bloque(a, f))["global"]
    _check("rho_ij(FOCO,RESTO) == rho_ij(RESTO,FOCO)", abs(r1 - r2) < 1e-9,
          f"{r1:.6f} vs {r2:.6f}")

    print("\n--- 9. _estimar_rho_transversal: invariante de escala ---")
    r_s1 = cv37._estimar_rho_transversal(_bloque(a, f, sigma=1.0), _bloque(b, f, sigma=1.0))["global"]
    r_s2 = cv37._estimar_rho_transversal(_bloque(a, f, sigma=1e6), _bloque(b, f, sigma=3e5))["global"]
    _check("resultado no depende de la escala de sigma", abs(r_s1 - r_s2) < 1e-9)

    print("\n--- 10. _estimar_rho_transversal: guardas ---")
    MIN = cv37.MIN_PARES_RHO_REGIMEN
    _check("muestra corta -> {} (no rompe)",
          cv37._estimar_rho_transversal(_bloque(a[:10], f[:10]), _bloque(b[:10], f[:10])) == {})
    _check("clasif=None -> {} (no rompe)",
          cv37._estimar_rho_transversal(None, _bloque(b, f)) == {})
    z0 = _bloque(a, f); z0["sigma"] = 0.0
    _check("sigma=0 en una entidad -> {} (no rompe)",
          cv37._estimar_rho_transversal(z0, _bloque(b, f)) == {})

    print("\n--- 11. _estimar_rho_transversal: clip a 0.98 ---")
    out = cv37._estimar_rho_transversal(_bloque(a, f), _bloque(a, f), tau_ewma_pares=1e6)
    _check("series identicas -> rho <= 0.98, no 1.0 (Cholesky necesita R no singular)",
          out["global"] <= 0.98, f"obtuvo {out['global']}")


# ─────────────────────────────────────────────────────────────────────────────
# Modo con datos reales
# ─────────────────────────────────────────────────────────────────────────────

def modo_matriz(ruta_matriz: str, banco: str):
    print(f"\n--- Verificaciones sobre datos reales ---")
    print(f"  Matriz: {ruta_matriz}")
    print(f"  Banco : {banco}")

    dir_regimen = None
    try:
        cv37 = _cargar_cv37()
        cv37.RUTA_MATRIZ = Path(ruta_matriz)
        dir_regimen = cv37.DIR_REGIMEN_HMM
    except Exception as e:
        _check("step005_walk_forward_cv_3.7 importa", False, f"{type(e).__name__}: {e}")
        return

    ruta_parquet = dir_regimen / f"estados_regimen_hmm_{banco}.parquet"
    if not _check(f"existe {ruta_parquet.name}", ruta_parquet.exists(),
                 "correr la CV de esta entidad primero (o esperar a que "
                 "HMM_INTERNO lo genere)"):
        return

    df = pd.read_parquet(ruta_parquet)
    _check("columnas flujo/sigma/estado/año_corte presentes",
          {"flujo", "sigma", "estado", "año_corte"} <= set(df.columns))

    contra = cv37._banco_contraparte(banco)
    if contra is None:
        print(f"  (informativo) {banco} no tiene contraparte — sin particion "
              f"activa, o es SISTEMA: rho_ij no aplica.")
        return

    ruta_contra = dir_regimen / f"estados_regimen_hmm_{contra}.parquet"
    if not _check(f"existe {ruta_contra.name} (contraparte)", ruta_contra.exists(),
                 "asegurar_regimenes_hmm deberia haberlo generado solo — "
                 "revisar HMM_INTERNO=True y que la corrida haya llegado a "
                 "generar folds"):
        return

    # Buscar un preds_test_fold*.parquet de esta entidad para leer rho_ij real
    dir_salida = None
    for p in cv37.DIR_OUTPUT.glob(f"**/preds_test_fold*_{banco}_*.parquet"):
        dir_salida = p
        break
    if dir_salida is None:
        print(f"  (informativo) sin preds_test_fold*_{banco}_*.parquet — "
              f"correr la CV completa de {banco} para verificar rho_ij "
              f"persistido; las verificaciones de parquets de regimen ya "
              f"pasaron arriba.")
        return

    dfp = pd.read_parquet(dir_salida)
    tiene_rho_ij = "rho_ij" in dfp.columns and dfp["rho_ij"].notna().any()
    _check(f"rho_ij presente en {dir_salida.name}", tiene_rho_ij)
    if tiene_rho_ij:
        val = dfp["rho_ij"].dropna().iloc[0]
        n_pares = dfp["n_pares_rho_ij"].dropna().iloc[0] if "n_pares_rho_ij" in dfp.columns else None
        _check(f"rho_ij en rango valido (-1, 1): {val:+.4f}", -1.0 < val < 1.0)
        print(f"    n_pares_rho_ij={n_pares}  contraparte_rho_ij="
              f"{dfp.get('contraparte_rho_ij', pd.Series(['?'])).iloc[0]}")

    # Simetria: si tambien existe preds_test de la contraparte, comparar
    dir_salida_contra = None
    for p in cv37.DIR_OUTPUT.glob(f"**/preds_test_fold*_{contra}_*.parquet"):
        dir_salida_contra = p
        break
    if dir_salida_contra is not None and tiene_rho_ij:
        dfc = pd.read_parquet(dir_salida_contra)
        if "rho_ij" in dfc.columns and dfc["rho_ij"].notna().any():
            val_c = dfc["rho_ij"].dropna().iloc[0]
            _check(f"simetria: rho_ij({banco})={val:+.4f} == "
                  f"rho_ij({contra})={val_c:+.4f}",
                  abs(val - val_c) < 1e-6,
                  "una diferencia aca indica desalineamiento de fechas entre "
                  "los dos bloques HMM, no ruido de estimacion")
    elif tiene_rho_ij:
        print(f"  (informativo) sin preds_test de {contra} para comparar "
              f"simetria — correr tambien esa entidad completa el chequeo.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--sintetico", action="store_true")
    g.add_argument("--matriz", metavar="RUTA")
    ap.add_argument("--banco", default="FOCO_BBVA",
                    help="entidad a verificar en modo --matriz (default: FOCO_BBVA)")
    a = ap.parse_args()

    if not a.sintetico and not a.matriz:
        if RUTA_MATRIZ_DEFECTO:
            a.matriz = RUTA_MATRIZ_DEFECTO
            print(f"(sin argumentos: usando RUTA_MATRIZ_DEFECTO = {RUTA_MATRIZ_DEFECTO!r})\n")
        else:
            a.sintetico = True
            print("(sin argumentos: RUTA_MATRIZ_DEFECTO no existe en disco, "
                 "corriendo el autotest sintetico; para validar una corrida "
                 "real usar --matriz RUTA --banco NOMBRE)\n")

    print("=" * 74)
    print("CHECKLIST — regimenes HMM internos y rho_ij (step005_walk_forward_cv_3.7)")
    print("=" * 74)

    modo_sintetico()
    if a.matriz:
        modo_matriz(a.matriz, a.banco)

    print("\n" + "=" * 74)
    if _FALLAS:
        print(f"RESULTADO: {len(_FALLAS)} verificación(es) FALLARON")
        for fnombre in _FALLAS:
            print(f"  - {fnombre}")
        sys.exit(1)
    print("RESULTADO: todas las verificaciones pasaron")
    print("=" * 74)


if __name__ == "__main__":
    main()
