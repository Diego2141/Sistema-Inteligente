# -*- coding: utf-8 -*-
"""
aux_correr_grilla_step006.py — corre step006 sobre varias configuraciones.

    runfile("aux_correr_grilla_step006.py")

DOS FASES, y la primera es la que justifica el script
  Fase 0 — VALIDACION. Para CADA configuracion, sin correr nada: que la
           combinacion sea valida, que existan los preds_test de sus grupos, que
           la transmat este donde tiene que estar, y que las columnas de
           preds_test digan lo que la config declara. Recien despues arranca el
           bucle.

           Es lo unico que evita el peor desenlace posible: dejar una grilla
           corriendo de noche y encontrar a la mañana que la tercera de cinco
           aborto a los cuarenta minutos por una columna que faltaba, y que las
           dos siguientes ni arrancaron.

  Fase 1 — BUCLE. Una configuracion por SUBPROCESO.

POR QUE SUBPROCESO Y NO UN IMPORT
Los cuatro botones de step006 son constantes de nivel de modulo, leidas AL
IMPORTAR: BANCO, GRUPOS, DIR_MODO, DIR_SALIDA y SUFIJO_CONFIG se derivan ahi.
Reimportar no las recalcula (sys.modules cachea) y recargar con importlib deja
el estado a medio camino. Un proceso nuevo por configuracion es la unica forma
limpia — y ademas devuelve la memoria entre corridas, que con N_JOBS>1 en
Windows (spawn) no es un detalle.

La copia temporal se escribe en la carpeta del repo, no en el temp del sistema:
los workers de multiprocessing en Windows REIMPORTAN el archivo, y necesitan
resolver `from step006_simulacion_paths_vf7 import ...` desde su ubicacion.

LA GRILLA SE CANONICALIZA ANTES DE EXPANDIR
Con PARTICIONES=True, ENTIDAD no se lee: las tres entidades producen la MISMA
corrida, el mismo BANCO y el mismo archivo. Expandir el producto cartesiano sin
canonicalizar correria el modo conjunto tres veces para pisar su propia salida
dos veces. Son horas. Ver canonicalizar().
"""

import itertools
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
ORQ = REPO / "step006_orquestador_vf_7.py"

###############################################################################
# Configuración
###############################################################################

# La grilla: un valor o una lista por boton. Se expande como producto
# cartesiano, se canonicaliza y se descartan las combinaciones invalidas.
GRILLA = {
    "PARTICIONES":     [True],
    "PARTICION":       ["globales", "bbva"],
    "ENTIDAD":         ["SISTEMA"],   # inerte con PARTICIONES=True
    "CONDICIONAR_POR": ["regimen"],   # agregar "calendario" cuando toque
}

# ── Corridas de step005 a leer — PARAMETROS ACOPLADOS ───────────────────────
# Estos tres NO son ejes independientes y por eso no van en GRILLA: la etiqueta
# nombra la carpeta de step005 y las dos ventanas nombran el subnivel de adentro,
# asi que solo existen en disco las combinaciones que step005 produjo.
#
# step005 arma la etiqueta como f"{MODELO_CV}_{modo}_{TRAIN}{VAL}{TEST}", una
# concatenacion pelada de los tres numeros:
#
#     xgb_qt_expanding_310.5  ->  "3" + "1" + "0.5"   (train 3, val 1, test 0.5)
#     xgb_qt_rolling_310.5    ->  idem, con EXPANDING=False
#
# Ojo con confundirla con una carpeta de otra geometria: un "xgb_qt_rolling_30.51"
# seria "3"+"0.5"+"1", o sea val y test INVERTIDOS — otra corrida, otro subnivel.
# La geometria se declara aca explicitamente y no se deduce del nombre; si las
# dos no coinciden, el subnivel que step006 arma no existe en disco.
#
# Cada entrada es un TRIPLE que se aplica ENTERO y la grilla se cruza contra
# ella. Se mantiene asi aunque hoy las dos compartan geometria: el dia que entre
# una corrida con otras ventanas, no hay nada que rehacer, y el acoplamiento
# queda documentado donde importa.
#
# Si una ruta no existe, la fase 0 la nombra exacta. Correr primero con
# EJECUTAR=False para verlo.
CORRIDAS_STEP005 = [
    {"ETIQUETA_CORRIDA": "xgb_qt_expanding_310.5",
     "VENTANA_VAL_AÑOS": 1, "VENTANA_TEST_AÑOS": 0.5},
    {"ETIQUETA_CORRIDA": "xgb_qt_rolling_310.5",
     "VENTANA_VAL_AÑOS": 1, "VENTANA_TEST_AÑOS": 0.5},
]

# Botones que NO varian en la grilla pero se quieren fijar para toda la corrida.
# Se sustituyen igual que los otros. Util sobre todo para N_JOBS y N_PATHS:
# una grilla de prueba con N_PATHS=1000 tarda una fraccion y ya muestra si las
# formas son las esperadas.
FIJOS = {
    # "N_PATHS": 10000,
    # "N_JOBS": 4,          # ver CLAUDE.md: -1 en Windows puede dar WinError 1455
}

# Si la salida principal de una configuracion ya existe, se saltea. Hace que la
# grilla sea REANUDABLE: si se corta a la mitad, volver a correrla retoma donde
# quedo en vez de repetir horas.
SALTAR_SI_EXISTE = True

# False = solo valida y muestra el plan, no corre nada. Conviene la primera vez.
EJECUTAR = True

# Carpeta de los logs, uno por configuracion.
DIR_LOGS = REPO / "logs_grilla"

###############################################################################
# Expansión de la grilla
###############################################################################

def canonicalizar(cfg: dict) -> dict:
    """
    Lleva una configuracion a su forma unica.

    Regla (b) de la tabla de combinaciones: con PARTICIONES=True, ENTIDAD no se
    lee. Las tres entidades derivan el mismo BANCO=CONJUNTO_<P>, el mismo
    SUFIJO_CONFIG y el mismo archivo de salida — son la MISMA corrida. Se fuerza
    a "SISTEMA" para que el dedupe las colapse en una.

    Sin esto, la grilla de arriba correria el modo conjunto tres veces por cada
    modo de condicionamiento: seis corridas identicas pisandose entre si.
    """
    cfg = dict(cfg)
    if cfg.get("PARTICIONES"):
        cfg["ENTIDAD"] = "SISTEMA"
    return cfg


def expandir(grilla: dict, acoplados: list | None = None) -> list:
    """
    Producto cartesiano de `grilla` x `acoplados` -> canonicalizar -> dedupe.

    `acoplados` es una lista de dicts que se aplican ENTEROS, no por ejes. Es la
    forma de expresar que ETIQUETA_CORRIDA y las dos ventanas del fold describen
    UNA corrida de step005 y solo valen juntos.
    """
    acoplados = [{}] if not acoplados else acoplados
    claves = list(grilla)
    valores = [v if isinstance(v, (list, tuple)) else [v] for v in grilla.values()]
    vistas, salida = set(), []
    for combo in itertools.product(*valores):
        for extra in acoplados:
            cfg = canonicalizar({**dict(zip(claves, combo)), **extra})
            firma = tuple(sorted(cfg.items()))
            if firma not in vistas:
                vistas.add(firma)
                salida.append(cfg)
    return salida


def _sustituir(src: str, botones: dict) -> str:
    """
    Reescribe los literales de nivel superior. Falla ruidosamente si un boton no
    aparece exactamente una vez: una sustitucion silenciosa que no ocurre
    dejaria la corrida con la config del archivo, no con la de la grilla — y el
    log diria que corrio lo que se pidio.
    """
    for k, v in botones.items():
        patron = rf"(?m)^{re.escape(k)}\s*=\s*(?:True|False|\"[^\"]*\"|'[^']*'|-?\d+(?:\.\d+)?)"
        nuevo = f"{k} = {v!r}"
        src, n = re.subn(patron, nuevo.replace("\\", "\\\\"), src, count=1)
        if n != 1:
            raise ValueError(
                f"No se pudo sustituir {k} en step006_orquestador_vf_7.py "
                f"({n} coincidencias). ¿Cambio el nombre o el formato del boton?")
    return src


def derivar(cfg: dict) -> dict:
    """
    Deriva BANCO, GRUPOS, DIR_MODO, DIR_SALIDA y SUFIJO_CONFIG EJECUTANDO la
    cabecera del propio step006, no reimplementando sus reglas.

    Es la diferencia entre validar y adivinar: si manana cambia como se arma
    _SUF_SALIDA, este script lo sigue solo. Y _validar_combinacion() corre de
    paso, asi que una combinacion invalida se rechaza con SU mensaje.

    Devuelve {} con la clave "error" si la combinacion no es valida.
    """
    src = ORQ.read_text(encoding="utf-8")
    cab = src.split("\n# ── Backtest extendido")[0]
    ns = {"__name__": "grilla_cfg", "__file__": str(ORQ)}
    try:
        exec(compile(_sustituir(cab, cfg), "<cabecera>", "exec"), ns)
    except ValueError as e:
        return {"error": str(e)}
    d = {k: ns[k] for k in ("BANCO", "GRUPOS", "DIR_MODO", "DIR_SALIDA",
                            "SUFIJO_CONFIG", "DIR_REGIMEN_HMM",
                            "BANCO_REGIMEN", "ETIQUETA_CORRIDA")}
    # La FUNCION, no solo el resultado: cada grupo tiene SU carpeta y hay que
    # resolverla una por una (ver validar). DIR_MODO es solo la del primero.
    d["dir_modo_de"] = ns["dir_modo_de"]
    return d


###############################################################################
# Fase 0 — validación de inputs
###############################################################################

def detectar_hecha(dir_salida: Path, sufijo: str, banco: str) -> tuple:
    """
    ¿Esta configuracion ya tiene su salida? Devuelve (hecha, ruta, aviso|None).

    Acepta TAMBIEN el nombre anterior a sufijo_config. Las corridas previas al
    renombrado dejaron simulacion_paths_<BANCO>.parquet; buscar solo el nombre
    nuevo las daria por no hechas y SALTAR_SI_EXISTE volveria a correr horas de
    computo ya hecho — el problema exacto que ese flag existe para evitar.

    El nombre viejo no dice el modo, pero la CARPETA si: el modo calendario
    escribe en el subnivel cond_calendario, asi que un archivo con nombre legado
    en dir_salida solo puede venir del modo de esa carpeta. Por eso se busca
    unicamente ahi, sin glob recursivo.
    """
    salida = dir_salida / f"simulacion_paths_{sufijo}.parquet"
    legado = dir_salida / f"simulacion_paths_{banco}.parquet"
    try:
        if salida.exists():
            return True, salida, None
        if legado.exists():
            return True, legado, (
                f"hecha con el nombre ANTERIOR ({legado.name}). Se la saltea "
                f"igual; renombrala a {salida.name} si queres que el nombre "
                f"lleve la configuracion.")
    except OSError:
        pass
    return False, salida, None


def validar(cfg: dict, d: dict) -> tuple:
    """
    ¿Esta configuracion tiene con que correr? Devuelve (lista_de_problemas, info).

    Los chequeos estan ordenados por cuando fallarian SIN este script:
      1. combinacion invalida      -> al importar (ya es inmediato)
      2. preds_test ausentes       -> al minuto, FileNotFoundError
      3. transmat ausente          -> al minuto
      4. condicionar_por discordante -> DESPUES de cargar preds y fitear las
         marginales, o sea a la hora. Este es el que justifica la fase 0.
      5. columnas rho_s_* / balde_th ausentes -> igual de tarde
    """
    problemas, info = [], {}
    if "error" in d:
        return [f"combinacion invalida: {d['error'][:90]}..."], info

    import pandas as pd

    # CADA grupo tiene SU carpeta, y se resuelve con la funcion del propio
    # step006 (cargar_preds_de_grupos hace exactamente esto: dir_modo_de(g)
    # dentro del bucle). Usar una sola carpeta para todos los grupos era un bug:
    # DIR_MODO es dir_modo_de(GRUPOS[0]), o sea la de FOCO, y los preds de RESTO
    # viven en RESTO_<P>_<val>_<test>, otra carpeta. Daba "sin preds_test de
    # RESTO_*" en configuraciones que estaban perfectas.
    modo = cfg["CONDICIONAR_POR"]
    for g in d["GRUPOS"]:
        try:
            dir_g = Path(d["dir_modo_de"](g))
            if not dir_g.is_dir():
                problemas.append(f"no existe la carpeta de preds de {g}: {dir_g}")
                continue
        except OSError as e:
            problemas.append(f"no se pudo inspeccionar la carpeta de {g} "
                             f"({type(e).__name__})")
            continue
        archivos = sorted(dir_g.glob(f"preds_test_fold*_{g}_*.parquet"))
        if not archivos:
            problemas.append(f"sin preds_test de {g} en {dir_g.name}")
            continue
        info.setdefault("folds", {})[g] = len(archivos)
        # Se lee UN fold: las columnas y condicionar_por son constantes por
        # corrida, asi que alcanza para saber si el contenido cuadra, y evita
        # cargar cientos de MB solo para validar.
        try:
            df = pd.read_parquet(archivos[0])
        except Exception as e:                        # noqa: BLE001
            problemas.append(f"{g}: no se pudo leer {archivos[0].name} ({e})")
            continue

        if "condicionar_por" in df.columns:
            cp = str(df["condicionar_por"].iloc[0])
            if cp != modo:
                problemas.append(
                    f"{g}: preds_test es de CONDICIONAR_POR='{cp}' y la config "
                    f"declara '{modo}' — step006 abortaria DESPUES de fitear")
        else:
            info.setdefault("avisos", []).append(
                f"{g}: preds_test sin columna condicionar_por (parquet viejo)")

        if not [c for c in df.columns if re.fullmatch(r"rho_s_\d+", str(c))]:
            problemas.append(f"{g}: sin columnas rho_s_* "
                             f"(correr step005 con ESTIMAR_RHO_EN_VAL=True)")
        if modo == "calendario" and "balde_th" not in df.columns:
            problemas.append(f"{g}: modo calendario sin columna balde_th")
        if cfg["PARTICIONES"] and "rho_ij" not in df.columns:
            problemas.append(f"{g}: modo conjunto sin columna rho_ij")
        info["origenes"] = int(df["fecha_t"].nunique())

    # La transmat solo hace falta si se muestrea la cadena de Markov.
    if modo != "calendario":
        banco_reg = d["BANCO_REGIMEN"] or d["BANCO"]
        ruta = Path(d["DIR_REGIMEN_HMM"]) / f"transmat_hmm_{banco_reg}.parquet"
        try:
            if not ruta.exists():
                problemas.append(f"falta la transmat: {ruta.name}")
        except OSError:
            problemas.append(f"no se pudo verificar {ruta.name}")

    # ¿Ya esta hecha?
    hecha, salida, aviso = detectar_hecha(Path(d["DIR_SALIDA"]),
                                          d["SUFIJO_CONFIG"], d["BANCO"])
    info["hecha"], info["salida"] = hecha, salida
    if aviso:
        info.setdefault("avisos", []).append(aviso)
    return problemas, info


###############################################################################
# Fase 1 — ejecución
###############################################################################

def correr(cfg: dict, d: dict, i: int, total: int) -> dict:
    """Una configuracion, un subproceso, un log."""
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    suf = d["SUFIJO_CONFIG"]
    tmp = REPO / f"_grilla_step006_{suf}.py"
    # El nombre del log lleva la corrida de step005 ademas del sufijo: los logs
    # de todas las configuraciones caen en la MISMA carpeta, y con dos corridas
    # de igual geometria de fold el sufijo coincide — se distinguirian solo por
    # la hora, que es justo lo que uno no recuerda al volver a buscarlos.
    _etq = cfg.get("ETIQUETA_CORRIDA", "")
    log = DIR_LOGS / f"{datetime.now():%Y%m%d_%H%M%S}_{_etq}_{suf}.log"

    src = _sustituir(ORQ.read_text(encoding="utf-8"), {**cfg, **FIJOS})
    tmp.write_text(src, encoding="utf-8")

    print(f"\n{'─' * 78}\n[{i}/{total}] {suf}\n  log -> {log}")
    t0 = time.time()
    try:
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run([sys.executable, tmp.name], cwd=str(REPO),
                                  stdout=fh, stderr=subprocess.STDOUT)
        rc = proc.returncode
    finally:
        # El temporal se borra SIEMPRE, tambien si el subproceso reventa o si se
        # interrumpe con Ctrl-C: si no, la carpeta del repo se llena de copias de
        # 1800 lineas que el proximo grep confunde con codigo vigente.
        tmp.unlink(missing_ok=True)

    dt = time.time() - t0
    estado = "OK" if rc == 0 else f"FALLO (rc={rc})"
    print(f"  {estado} en {dt/60:.1f} min")
    if rc != 0:
        cola = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        print("  ultimas lineas del log:")
        for ln in cola:
            print(f"    {ln}")
    return {"sufijo": suf, "rc": rc, "minutos": dt / 60, "log": log}


def main() -> int:
    print("=" * 78)
    print("GRILLA DE CONFIGURACIONES — step006_orquestador_vf_7")
    print("=" * 78)

    configs = expandir(GRILLA, CORRIDAS_STEP005)
    n_bruto = max(len(CORRIDAS_STEP005), 1)
    for v in GRILLA.values():
        n_bruto *= len(v) if isinstance(v, (list, tuple)) else 1
    print(f"\nGrilla x {len(CORRIDAS_STEP005)} corrida(s) de step005: {n_bruto}"
          f"  ->  tras canonicalizar y deduplicar: {len(configs)}")
    if n_bruto != len(configs):
        print(f"  ({n_bruto - len(configs)} descartadas: con PARTICIONES=True "
              f"ENTIDAD no se lee, las entidades colapsan en una sola corrida)")

    print("\n" + "=" * 78)
    print("FASE 0 — VALIDACION DE INPUTS")
    print("=" * 78)

    # Dos categorias distintas, y mezclarlas confunde:
    #   podadas  -> la tabla de combinaciones no las contempla. Es PODA de la
    #               grilla, esperable y no accionable: la grilla se escribe como
    #               producto cartesiano justamente para no tener que enumerar a
    #               mano cuales existen.
    #   sin_inputs -> la combinacion es valida pero le falta un archivo. ESO si
    #               hay que resolverlo, y dice exactamente que.
    plan, podadas, sin_inputs, hechas = [], [], [], []
    for cfg in configs:
        d = derivar(cfg)
        problemas, info = validar(cfg, d)
        # La corrida de step005 va SIEMPRE en la etiqueta: cuando dos corridas
        # comparten geometria de fold, SUFIJO_CONFIG es identico entre ellas y
        # las lineas del plan quedarian indistinguibles.
        etiqueta = d.get("SUFIJO_CONFIG") or (
            f"PARTICIONES={cfg['PARTICIONES']} ENTIDAD={cfg['ENTIDAD']} "
            f"cond={cfg['CONDICIONAR_POR']}")
        etiqueta = f"{cfg['ETIQUETA_CORRIDA']} | {etiqueta}" if cfg.get(
            "ETIQUETA_CORRIDA") else etiqueta
        if "error" in d:
            podadas.append(etiqueta)
            continue
        if problemas:
            print(f"\n[SIN INPUTS] {etiqueta}")
            for p in problemas:
                print(f"    - {p}")
            sin_inputs.append((etiqueta, problemas))
            continue
        for av in info.get("avisos", []):
            print(f"[AVISO] {etiqueta}: {av}")
        if SALTAR_SI_EXISTE and info.get("hecha"):
            print(f"[YA HECHA] {etiqueta}  ->  {info['salida'].name}")
            hechas.append(etiqueta)
            continue
        folds = info.get("folds", {})
        print(f"[LISTA] {etiqueta}  grupos={d['GRUPOS']}  "
              f"folds={list(folds.values())}  origenes={info.get('origenes', '?')}")
        plan.append((cfg, d))

    if podadas:
        print(f"\n[PODADAS] {len(podadas)} combinacion(es) que la tabla no "
              f"contempla — es la poda normal del producto cartesiano:")
        for e in podadas:
            print(f"    - {e}")

    print("\n" + "=" * 78)
    print(f"PLAN: {len(plan)} a correr | {len(hechas)} ya hechas | "
          f"{len(sin_inputs)} sin inputs | {len(podadas)} podadas")
    print("=" * 78)

    if not EJECUTAR:
        print("\nEJECUTAR=False — solo se valido. Poner True para correr.")
        return 0
    if not plan:
        print("\nNada que correr.")
        return 0

    resultados = []
    t0 = time.time()
    for i, (cfg, d) in enumerate(plan, 1):
        resultados.append(correr(cfg, d, i, len(plan)))
        # Proyeccion, misma idea que el Cronometro de step006: saber en la
        # primera si faltan 20 minutos o seis horas cambia lo que uno hace.
        if i < len(plan):
            hechas_min = (time.time() - t0) / 60
            print(f"  acumulado {hechas_min:.0f} min | faltan {len(plan)-i} | "
                  f"proyeccion {hechas_min / i * len(plan):.0f} min en total")

    print("\n" + "=" * 78)
    print("RESUMEN")
    print("=" * 78)
    for r in resultados:
        print(f"  {'OK   ' if r['rc'] == 0 else 'FALLO'}  {r['minutos']:6.1f} min  "
              f"{r['sufijo']}")
    fallos = [r for r in resultados if r["rc"] != 0]
    print(f"\n  {len(resultados) - len(fallos)}/{len(resultados)} OK  |  "
          f"total {(time.time() - t0)/60:.0f} min")
    if fallos:
        print(f"  Logs de los fallos: "
              f"{[str(r['log'].name) for r in fallos]}")
    return 1 if fallos else 0


###############################################################################
# Autotest
###############################################################################

def autotest() -> int:
    ok = falla = 0

    def chk(n, c, d=""):
        nonlocal ok, falla
        print(f"[{'OK' if c else 'FALLA'}] {n}" + (f"  — {d}" if d else ""))
        if c:
            ok += 1
        else:
            falla += 1

    print("\n" + "=" * 78)
    print("AUTOTEST (no necesita acceso a H:)")
    print("=" * 78)

    # 1. Canonicalizacion: con PARTICIONES=True las tres entidades colapsan.
    g = {"PARTICIONES": [True], "PARTICION": ["bbva"],
         "ENTIDAD": ["SISTEMA", "FOCO", "RESTO"], "CONDICIONAR_POR": ["regimen"]}
    chk("1. con PARTICIONES=True las 3 entidades colapsan en 1 corrida",
        len(expandir(g)) == 1, f"{len(expandir(g))} configuracion(es)")

    # 2. [neg] con PARTICIONES=False NO colapsan: ahi ENTIDAD si se lee.
    g2 = dict(g, PARTICIONES=[False])
    chk("2. [neg] con PARTICIONES=False las 3 entidades son 3 corridas",
        len(expandir(g2)) == 3, f"{len(expandir(g2))}")

    # 3. La grilla del archivo se reduce como corresponde.
    c = expandir(GRILLA, CORRIDAS_STEP005)
    chk("3. la grilla se cruza con las corridas acopladas de step005",
        len(c) == len(expandir(GRILLA)) * len(CORRIDAS_STEP005),
        f"{len(c)} configs = {len(expandir(GRILLA))} x {len(CORRIDAS_STEP005)}")

    # 4. Las invalidas las rechaza _validar_combinacion del PROPIO step006.
    d = derivar({"PARTICIONES": False, "PARTICION": "bbva", "ENTIDAD": "FOCO",
                 "CONDICIONAR_POR": "calendario"})
    chk("4. [neg] calendario con PARTICIONES=False se rechaza con el mensaje "
        "de step006", "error" in d and "PARTICIONES=False" in d["error"],
        d.get("error", "")[:55])

    # 5. Una valida deriva lo esperado, ejecutando la cabecera real.
    d = derivar({"PARTICIONES": True, "PARTICION": "bbva", "ENTIDAD": "SISTEMA",
                 "CONDICIONAR_POR": "calendario"})
    chk("5. una config valida deriva BANCO, GRUPOS y SUFIJO_CONFIG",
        d.get("BANCO") == "CONJUNTO_BBVA"
        and d.get("SUFIJO_CONFIG", "").endswith("_condcalendario")
        and len(d.get("GRUPOS", [])) == 2,
        f"{d.get('BANCO')} | {d.get('SUFIJO_CONFIG')} | {d.get('GRUPOS')}")

    # 6. La salida del modo calendario va al subnivel cond_calendario.
    chk("6. calendario escribe en el subnivel cond_calendario",
        "cond_calendario" in str(d.get("DIR_SALIDA", "")),
        str(d.get("DIR_SALIDA", ""))[-40:])

    # 7. [neg] un boton inexistente falla ruidosamente, no en silencio.
    try:
        _sustituir("X = 1\n", {"NO_EXISTE": True})
        chk("7. [neg] sustituir un boton inexistente aborta", False, "no abortó")
    except ValueError as e:
        chk("7. [neg] sustituir un boton inexistente aborta",
            "NO_EXISTE" in str(e), str(e)[:50])

    # 8. La sustitucion cambia el literal y no rompe la sintaxis.
    src = _sustituir(ORQ.read_text(encoding="utf-8"),
                     {"PARTICIONES": False, "ENTIDAD": "RESTO"})
    compile(src, "<sust>", "exec")
    chk("8. el archivo sustituido sigue compilando",
        "PARTICIONES = False" in src and "ENTIDAD = 'RESTO'" in src)

    # 9. Cada configuracion escribe en una RUTA distinta. Se compara la ruta
    #    completa y no solo el sufijo: dos corridas de step005 con la misma
    #    geometria de fold darian el mismo SUFIJO_CONFIG y distinto DIR_SALIDA,
    #    y eso no es colision. Lo que si lo seria es la ruta repetida.
    rutas = []
    for cfg in c:
        dd = derivar(cfg)
        if "error" not in dd:
            rutas.append(f"{dd['DIR_SALIDA']}/simulacion_paths_{dd['SUFIJO_CONFIG']}")
    chk("9. cada configuracion escribe en una ruta distinta (nada se pisa)",
        len(set(rutas)) == len(rutas), f"{len(set(rutas))}/{len(rutas)} distintas")

    # 9b. El triple acoplado se aplica ENTERO: la etiqueta manda la carpeta y las
    #     ventanas declaradas mandan el sufijo. Se prueba con un triple
    #     SINTETICO de geometria distinta, no con CORRIDAS_STEP005: hoy las dos
    #     entradas comparten val/test, asi que usarlas no distinguiria si las
    #     ventanas se aplican o se ignoran — el chequeo pasaria por casualidad.
    d_roll = derivar(expandir(
        {"PARTICIONES": [True], "PARTICION": ["globales"],
         "ENTIDAD": ["SISTEMA"], "CONDICIONAR_POR": ["regimen"]},
        [{"ETIQUETA_CORRIDA": "xgb_qt_rolling_310.5",
          "VENTANA_VAL_AÑOS": 0.5, "VENTANA_TEST_AÑOS": 1}])[0])
    chk("9b. el triple acoplado se aplica entero (etiqueta + las dos ventanas)",
        "rolling" in str(d_roll["DIR_SALIDA"])
        and d_roll["SUFIJO_CONFIG"].startswith("CONJUNTO_GLOBALES_0.5_1"),
        f"{d_roll['SUFIJO_CONFIG']}")

    # 9g. Con la geometria REAL, expanding y rolling comparten SUFIJO_CONFIG y
    #     se distinguen solo por la carpeta. No es colision —las rutas difieren—
    #     pero si vuelve homonimos los archivos al sacarlos de su carpeta, que es
    #     justo lo que sufijo_config existia para evitar. Queda medido aca para
    #     que el dia que se decida meter la etiqueta en el nombre haya un
    #     chequeo que lo refleje.
    porsuf = {}
    for cfg_ in expandir(GRILLA, CORRIDAS_STEP005):
        dd = derivar(cfg_)
        if "error" not in dd:
            porsuf.setdefault(dd["SUFIJO_CONFIG"], set()).add(cfg_["ETIQUETA_CORRIDA"])
    homon = {k: v for k, v in porsuf.items() if len(v) > 1}
    chk("9g. homonimos entre corridas de step005 (esperado con igual geometria)",
        len(homon) == len(porsuf) if len(CORRIDAS_STEP005) > 1 else True,
        f"{len(homon)}/{len(porsuf)} sufijos se repiten entre carpetas")

    # 9h. CADA grupo tiene SU carpeta de preds. Fue un bug real: validar() usaba
    #     DIR_MODO —que es dir_modo_de(GRUPOS[0]), la de FOCO— para los dos
    #     grupos, y reportaba "sin preds_test de RESTO_*" en configuraciones que
    #     estaban perfectas. step006 resuelve una por grupo dentro del bucle de
    #     cargar_preds_de_grupos; aca se verifica que las dos rutas difieran.
    d_dos = derivar({"PARTICIONES": True, "PARTICION": "globales",
                     "ENTIDAD": "SISTEMA", "CONDICIONAR_POR": "regimen"})
    g1, g2 = d_dos["GRUPOS"]
    p1, p2 = d_dos["dir_modo_de"](g1), d_dos["dir_modo_de"](g2)
    chk("9h. cada grupo resuelve a SU carpeta de preds (no comparten)",
        str(p1) != str(p2) and g1 in str(p1) and g2 in str(p2),
        f"{Path(p1).name} vs {Path(p2).name}")

    # ── Deteccion de "ya hecha", con el nombre nuevo y con el legado ────────
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        suf, banco = "CONJUNTO_BBVA_1_0.5_condregimen", "CONJUNTO_BBVA"
        h, r, av = detectar_hecha(td, suf, banco)
        chk("9c. sin archivos, no esta hecha", (not h) and av is None
            and r.name == f"simulacion_paths_{suf}.parquet")

        (td / f"simulacion_paths_{banco}.parquet").touch()
        h, r, av = detectar_hecha(td, suf, banco)
        chk("9d. el nombre LEGADO cuenta como hecha (no re-correr horas)",
            h and av is not None and r.name == f"simulacion_paths_{banco}.parquet",
            (av or "")[:48])

        (td / f"simulacion_paths_{suf}.parquet").touch()
        h, r, av = detectar_hecha(td, suf, banco)
        chk("9e. con los dos presentes gana el nombre NUEVO, sin aviso",
            h and av is None and r.name == f"simulacion_paths_{suf}.parquet")

        # [neg] el legado NO se busca fuera de su carpeta: el modo calendario
        # escribe en cond_calendario, asi que un legado del padre no es suyo.
        sub = td / "cond_calendario"
        sub.mkdir()
        h, _, _ = detectar_hecha(sub, "CONJUNTO_BBVA_1_0.5_condcalendario", banco)
        chk("9f. [neg] el legado del padre NO cuenta para el subnivel "
            "cond_calendario", not h)

    # ── Plomeria del subproceso ─────────────────────────────────────────────
    # Es la parte que solo se ejercita corriendo de verdad, y la que mas caro
    # sale que falle: si el temporal no se borra, la carpeta del repo se llena
    # de copias de 1800 lineas; si la sustitucion no llega al subproceso, la
    # grilla corre ocho veces la MISMA config y el log dice que corrio otra.
    # Se prueba contra un orquestador de mentira, sin tocar H:.
    import shutil
    orq_real, logs_real = ORQ, DIR_LOGS
    fake = REPO / "_fake_orq_autotest.py"
    try:
        globals()["ORQ"] = fake
        globals()["DIR_LOGS"] = REPO / "_logs_autotest"

        fake.write_text('PARTICIONES = True\nENTIDAD = "SISTEMA"\n'
                        'print("corriendo", PARTICIONES, ENTIDAD)\n',
                        encoding="utf-8")
        r = correr({"PARTICIONES": False, "ENTIDAD": "RESTO"},
                   {"SUFIJO_CONFIG": "AUTOTEST_OK"}, 1, 1)
        texto = r["log"].read_text(encoding="utf-8")
        chk("10. el subproceso corre y devuelve rc=0", r["rc"] == 0)
        chk("11. la sustitucion LLEGA al subproceso (no corre la config del "
            "archivo)", "corriendo False RESTO" in texto, texto.strip())
        chk("12. el temporal se borra tras el exito",
            not (REPO / "_grilla_step006_AUTOTEST_OK.py").exists())

        fake.write_text('PARTICIONES = True\nENTIDAD = "SISTEMA"\n'
                        'raise RuntimeError("explota a proposito")\n',
                        encoding="utf-8")
        r2 = correr({"PARTICIONES": True, "ENTIDAD": "SISTEMA"},
                    {"SUFIJO_CONFIG": "AUTOTEST_FALLA"}, 1, 1)
        chk("13. [neg] un subproceso que revienta devuelve rc != 0", r2["rc"] != 0)
        chk("14. [neg] el temporal se borra IGUAL cuando el subproceso falla",
            not (REPO / "_grilla_step006_AUTOTEST_FALLA.py").exists())
        chk("15. [neg] el traceback queda en el log",
            "explota a proposito" in r2["log"].read_text(encoding="utf-8"))
        chk("16. un fallo NO corta la grilla (correr devuelve, no lanza)",
            isinstance(r2, dict) and "minutos" in r2)
    finally:
        globals()["ORQ"] = orq_real
        globals()["DIR_LOGS"] = logs_real
        fake.unlink(missing_ok=True)
        shutil.rmtree(REPO / "_logs_autotest", ignore_errors=True)

    print(f"\n{'=' * 78}\nAUTOTEST: {ok} OK / {falla} FALLA\n{'=' * 78}")
    return 1 if falla else 0


if __name__ == "__main__":
    _rc = main()
    autotest()
    sys.exit(_rc)
