# -*- coding: utf-8 -*-
"""
aux_validar_grilla_step006.py — ¿la grilla clasifica bien las 8 configuraciones?

    python aux_validar_grilla_step006.py

Construye un arbol de directorios que imita el H: real —step005 corrido con
PARTICIONES=True para bbva y globales, en expanding y rolling, en regimen y
calendario, todos con val=1 / test=0.5— y corre la FASE 0 de verdad contra el.

NO MIRA H:. Construye su propio arbol en un directorio temporal y redirige
BASE_SISTEMA ahi. Valida la LOGICA de la grilla, no que tus preds existan — eso
lo dice la fase 0 de aux_correr_grilla_step006.py con EJECUTAR=False, que si lee
el disco real. Este harness se corre despues de tocar el codigo, no antes de
cada corrida.

POR QUE EXISTE APARTE DEL AUTOTEST DE LA GRILLA
El autotest de aux_correr_grilla_step006 valida la expansion, la derivacion y la
plomeria del subproceso, pero no puede ejercitar la fase 0 completa: sin una
jerarquia de carpetas con preds_test de verdad, validar() sale temprano y el
camino interesante nunca se recorre.

Eso oculto un bug real: validar() usaba DIR_MODO —que es dir_modo_de(GRUPOS[0]),
la carpeta de FOCO— para los DOS grupos, y reportaba "sin preds_test de RESTO_*"
en las 8 configuraciones, todas perfectas. step006 resuelve una carpeta por
grupo dentro del bucle de cargar_preds_de_grupos. El bug sobrevivio a 22 checks
y lo encontro este harness en la primera corrida.

Ejercita el CODIGO DE PRODUCCION: copia step006 y la grilla con BASE_SISTEMA y
ORQ redirigidos al arbol de prueba, sin reimplementar nada.
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# La carpeta donde vive este archivo, no una ruta absoluta: el script se
# copia a H: junto a los demas y ahi una ruta del contenedor no existe.
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

OK = FALLA = 0


def chk(n, c, d=""):
    global OK, FALLA
    print(f"[{'OK' if c else 'FALLA'}] {n}" + (f"  — {d}" if d else ""))
    if c:
        OK += 1
    else:
        FALLA += 1


# ── 1. El arbol de H: ────────────────────────────────────────────────────────
H = Path(tempfile.mkdtemp())
OUT = H / "2. Output"
WF = OUT / "step005_wfcv_v3"

# La config REAL de la grilla, para no duplicarla: el arbol de prueba se
# construye con las etiquetas y la geometria de fold que la grilla va a buscar.
# Hardcodearlas hacia que el harness validara contra un disco imaginario
# distinto del que la grilla lee — el peor tipo de test verde.
_g = (REPO / "aux_correr_grilla_step006.py").read_text(encoding="utf-8")
_ns = {"__file__": str(REPO / "aux_correr_grilla_step006.py")}
exec(compile(_g.split("\n# Botones que NO varian")[0], "<cfg>", "exec"), _ns)
CORRIDAS = _ns["CORRIDAS_STEP005"]

PARTICIONES = ["GLOBALES", "BBVA"]
MODOS = ["regimen", "calendario"]
N_ESTADOS = 3


def _fmt(x):
    """0.5 -> '0.5', 1 -> '1'. Misma regla que step005/006."""
    return f"{x:g}"


def subnivel(banco, corrida):
    """<banco>_<val>_<test>, como lo arma etiqueta_corrida() de step006."""
    return (f"{banco}_{_fmt(corrida['VENTANA_VAL_AÑOS'])}"
            f"_{_fmt(corrida['VENTANA_TEST_AÑOS'])}")


def preds(banco, modo):
    """Un preds_test con las columnas que la fase 0 verifica."""
    f = pd.bdate_range("2024-04-30", periods=40)
    filas = []
    for ft in f:
        for h in (2, 5, 22):
            filas.append({
                "fecha_t": ft, "h": h, "banco": banco,
                "y_realizado": 0.0, "regimen_hmm": 0,
                "año_corte_regimen": "2024-01-01",
                "condicionar_por": modo,
                **{f"rho_s_{s}": -0.3 for s in range(N_ESTADOS)},
                "rho_ij": -0.04,
                **({"balde_th": 1} if modo == "calendario" else {}),
            })
    return pd.DataFrame(filas)


for corrida in CORRIDAS:
    etq = corrida["ETIQUETA_CORRIDA"]
    for p in PARTICIONES:
        for modo in MODOS:
            for cara in ("FOCO", "RESTO"):
                banco = f"{cara}_{p}"
                d = WF / etq / subnivel(banco, corrida)
                if modo == "calendario":
                    d = d / "cond_calendario"
                d.mkdir(parents=True, exist_ok=True)
                for fold in (1, 2, 3, 4):
                    preds(banco, modo).to_parquet(
                        d / f"preds_test_fold{fold}_{banco}_20240430.parquet",
                        index=False)

# transmat del HMM (solo hace falta para el modo regimen)
pd.DataFrame({"año_corte": ["2024-01-01"] * N_ESTADOS**2,
              "i": np.repeat(range(N_ESTADOS), N_ESTADOS),
              "j": list(range(N_ESTADOS)) * N_ESTADOS,
              "p": [1 / N_ESTADOS] * N_ESTADOS**2}).to_parquet(
    OUT / "transmat_hmm_SISTEMA.parquet", index=False)

# Salidas YA HECHAS de bbva expanding: la de regimen con el nombre LEGADO
# (anterior a sufijo_config) y la de calendario con el nombre nuevo.
_exp = next(c for c in CORRIDAS if "expanding" in c["ETIQUETA_CORRIDA"])
sim = (OUT / "step006_simulacion" / _exp["ETIQUETA_CORRIDA"]
       / subnivel("CONJUNTO_BBVA", _exp))
sim.mkdir(parents=True, exist_ok=True)
(sim / "simulacion_paths_CONJUNTO_BBVA.parquet").touch()
(sim / "cond_calendario").mkdir(exist_ok=True)
(sim / "cond_calendario" /
 f"simulacion_paths_{subnivel('CONJUNTO_BBVA', _exp)}_condcalendario.parquet").touch()

print(f"Arbol de prueba en {H}\n")

# ── 2. Copias del codigo con las rutas redirigidas ──────────────────────────
orq_tmp = REPO / "_val_step006.py"
gri_tmp = REPO / "_val_grilla.py"
try:
    src = (REPO / "step006_orquestador_vf_7.py").read_text(encoding="utf-8")
    src2, n = re.subn(r'(?m)^BASE_SISTEMA = Path\(r"[^"]*"\)',
                      f'BASE_SISTEMA = Path(r"{H}")', src, count=1)
    assert n == 1, "no se pudo redirigir BASE_SISTEMA"
    orq_tmp.write_text(src2, encoding="utf-8")

    g = (REPO / "aux_correr_grilla_step006.py").read_text(encoding="utf-8")
    g, n = re.subn(r'(?m)^ORQ = REPO / "step006_orquestador_vf_7\.py"',
                   f'ORQ = REPO / "{orq_tmp.name}"', g, count=1)
    assert n == 1, "no se pudo redirigir ORQ"
    g = re.sub(r'(?m)^EJECUTAR = True', 'EJECUTAR = False', g, count=1)
    gri_tmp.write_text(g, encoding="utf-8")

    # ── 3. Correr la fase 0 real ────────────────────────────────────────────
    import importlib.util
    spec = importlib.util.spec_from_file_location("valg", gri_tmp)
    m = importlib.util.module_from_spec(spec)
    sys.modules["valg"] = m
    spec.loader.exec_module(m)

    configs = m.expandir(m.GRILLA, m.CORRIDAS_STEP005)
    # Derivado, no hardcodeado: el harness sigue a la grilla vigente en vez de
    # fallar por contabilidad cada vez que alguien la recorta.
    esperadas = len(m.expandir(m.GRILLA)) * len(m.CORRIDAS_STEP005)
    chk("1. la grilla expande al producto grilla x corridas de step005",
        len(configs) == esperadas,
        f"{len(configs)} = {len(m.expandir(m.GRILLA))} x {len(m.CORRIDAS_STEP005)}")

    listas, hechas, sininputs, avisos = [], [], [], []
    for cfg in configs:
        d = m.derivar(cfg)
        probs, info = m.validar(cfg, d)
        etq = f"{cfg['ETIQUETA_CORRIDA']} | {d.get('SUFIJO_CONFIG', '?')}"
        avisos += info.get("avisos", [])
        if probs:
            sininputs.append((etq, probs))
        elif info.get("hecha"):
            hechas.append((etq, info["salida"].name))
        else:
            listas.append((etq, d["GRUPOS"], info.get("folds"), info.get("origenes")))

    print()
    for e, g_, f_, o_ in listas:
        print(f"  [LISTA]     {e}  grupos={g_} folds={list(f_.values())} origenes={o_}")
    for e, nom in hechas:
        print(f"  [YA HECHA]  {e}  -> {nom}")
    for e, p_ in sininputs:
        print(f"  [SIN INPUTS] {e}")
        for x in p_:
            print(f"        - {x}")
    print()

    # Las expectativas se DERIVAN de la grilla y del arbol, no se hardcodean:
    # el arbol tiene salidas preexistentes SOLO para bbva expanding, asi que
    # esas son las que deben salir YA HECHAS sea cual sea la grilla vigente.
    esp_hechas = [c for c in configs
                  if c["PARTICION"] == "bbva" and "expanding" in c["ETIQUETA_CORRIDA"]]
    chk("2. ninguna configuracion queda sin inputs", not sininputs,
        f"{len(sininputs)} sin inputs")
    chk("3. las de bbva expanding se detectan YA HECHAS",
        len(hechas) == len(esp_hechas),
        f"{len(hechas)}/{len(esp_hechas)}: {[h[0].split('| ')[-1] for h in hechas]}")
    chk("4. el resto queda LISTA, y nada se pierde",
        len(listas) == len(configs) - len(esp_hechas)
        and len(listas) + len(hechas) == len(configs),
        f"{len(listas)} a correr de {len(configs)}")
    # El aviso del nombre legado solo aplica si la grilla incluye la
    # configuracion que en el arbol tiene ese nombre (bbva expanding regimen).
    hay_legado = any(c["PARTICION"] == "bbva" and "expanding" in c["ETIQUETA_CORRIDA"]
                     and c["CONDICIONAR_POR"] == "regimen" for c in configs)
    chk("5. la salida con nombre LEGADO se reconoce igual",
        (not hay_legado) or any("hecha con el nombre ANTERIOR" in a for a in avisos),
        next((a[:60] for a in avisos if "ANTERIOR" in a), "n/a"))
    chk("6. las que van a correr leen los 4 folds de cada grupo",
        all(f_ and set(f_.values()) == {4} for _, _, f_, _ in listas))
    chk("7. el modo conjunto carga las DOS caras de la particion",
        all(len(g_) == 2 for _, g_, _, _ in listas))

    # Las de calendario tienen que resolver al subnivel cond_calendario
    # Calendario se prueba SIEMPRE, este o no en la grilla vigente: un all()
    # sobre una lista vacia pasa trivialmente, y un chequeo que no puede fallar
    # es decorativo. Se construyen las configuraciones a proposito.
    cal = [m.derivar(m.canonicalizar(
        {"PARTICIONES": True, "PARTICION": p_, "ENTIDAD": "SISTEMA",
         "CONDICIONAR_POR": "calendario", **e})) 
        for p_ in ("globales", "bbva") for e in m.CORRIDAS_STEP005]
    chk("8. las de calendario leen del subnivel cond_calendario",
        cal and all("cond_calendario" in str(d["DIR_MODO"]) for d in cal),
        f"{len(cal)} configuraciones (construidas, esten o no en la grilla)")
    chk("9. las de calendario ESCRIBEN en cond_calendario",
        cal and all("cond_calendario" in str(d["DIR_SALIDA"]) for d in cal))

    reg = [m.derivar(m.canonicalizar(
        {"PARTICIONES": True, "PARTICION": p_, "ENTIDAD": "SISTEMA",
         "CONDICIONAR_POR": "regimen", **e}))
        for p_ in ("globales", "bbva") for e in m.CORRIDAS_STEP005]
    chk("10. [neg] las de regimen NO leen del subnivel de calendario",
        reg and all("cond_calendario" not in str(d["DIR_MODO"]) for d in reg),
        f"{len(reg)} configuraciones")

    # ── Control negativo fuerte: si el preds dice otro modo, se detecta ──────
    _roll = CORRIDAS[-1]
    d_cal = (WF / _roll["ETIQUETA_CORRIDA"]
             / subnivel("FOCO_GLOBALES", _roll) / "cond_calendario")
    for f in d_cal.glob("*.parquet"):
        df = pd.read_parquet(f)
        df["condicionar_por"] = "regimen"          # discordante a proposito
        df.to_parquet(f, index=False)
    cfg_mal = m.canonicalizar(
        {"PARTICIONES": True, "PARTICION": "globales", "ENTIDAD": "SISTEMA",
         "CONDICIONAR_POR": "calendario", **m.CORRIDAS_STEP005[-1]})
    probs, _ = m.validar(cfg_mal, m.derivar(cfg_mal))
    chk("11. [neg] un preds_test del OTRO modo se detecta en fase 0",
        any("CONDICIONAR_POR" in p for p in probs), str(probs)[:80])

    # ── Y si faltan las columnas rho_s_* ────────────────────────────────────
    d_reg = WF / _roll["ETIQUETA_CORRIDA"] / subnivel("FOCO_BBVA", _roll)
    for f in d_reg.glob("*.parquet"):
        df = pd.read_parquet(f)
        df.drop(columns=[c for c in df.columns if c.startswith("rho_s_")]).to_parquet(
            f, index=False)
    cfg_mal2 = m.canonicalizar(
        {"PARTICIONES": True, "PARTICION": "bbva", "ENTIDAD": "SISTEMA",
         "CONDICIONAR_POR": "regimen", **m.CORRIDAS_STEP005[-1]})
    probs, _ = m.validar(cfg_mal2, m.derivar(cfg_mal2))
    chk("12. [neg] sin columnas rho_s_* se detecta en fase 0",
        any("rho_s_" in p for p in probs), str(probs)[:80])

finally:
    orq_tmp.unlink(missing_ok=True)
    gri_tmp.unlink(missing_ok=True)
    shutil.rmtree(H, ignore_errors=True)

print(f"\nval_grilla: {OK} OK / {FALLA} FALLA")
sys.exit(1 if FALLA else 0)
