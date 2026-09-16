# -*- coding: utf-8 -*-
"""
step006_orquestador.py
=======================
Glue code que conecta los outputs REALES de:
  - step005_walk_forward_cv_4.py  → preds_test_fold*.parquet (con regimen_hmm,
    regimen_sigma, año_corte_regimen ya mergeados, sin leakage).
  - step005_validar_hmm_ewma.py (o la variante con winsor/EWMA que dejaste
    fija) → transmat_hmm_<banco>.parquet.

con el módulo step006_simulacion_paths_vf7.py (PCHIP+GPD + cópula AR(1) +
backtest
de 3 piezas). No reimplementa nada de la simulación/backtest — solo arma los
inputs correctos y llama a las funciones de ese módulo.

Por qué se agrupa por año_corte_regimen
----------------------------------------
Cada fold de XGBoost (step005_walk_forward_cv_4.py) puede haber usado un
bloque HMM distinto para su feature de régimen (el más reciente sin leakage
respecto a su propio train_end). Eso significa que distintos orígenes del
período de test pueden corresponder a matrices de transición DIFERENTES.
Se simula cada grupo (mismo año_corte_regimen) con SU propia matriz, y se
concatena el resultado al final — la métrica de backtest se calcula sobre
el conjunto completo, igual que se evaluaría en producción.

Por qué el estado inicial es POR ORIGEN, no uno global
-------------------------------------------------------
Para backtest histórico, cada fecha de origen YA tiene un estado de régimen
clasificado (columna regimen_hmm, calculado sin leakage por el HMM). Usar ESE
estado como punto de partida de la simulación de cada origen es lo correcto
— no hay que asumir un único "estado actual" para todo el período de test.

Uso:
    python step006_orquestador.py
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm as _norm_dist

# vf7, no vf6. Verificado por AST: vf7 = vf6 + backtest y NADA mas. Son
# identicas las tres clases de marginal (PchipGPD, AzzaliniT, SplitT),
# simular_regimen_path, simular_un_path, simular_paths_origen,
# pipeline_simulacion, calcular_percentiles_acumulado,
# fitear_distribuciones_por_horizonte, cargar_preds_test_reales y los tres
# generadores de fanchart; tampoco cambio ninguna constante existente.
# vf7 AGREGA solo herramienta de backtest: _lag_newey_west_auto (ancho de
# banda Newey-West 1994), _wilson (intervalo de Wilson para una proporcion,
# reemplaza Wald que se degrada cerca de 0 y 1) y anderson_darling_uniforme
# (A^2 para H0: u~U(0,1) — scipy.stats.anderson NO trae dist="uniform"), y
# reescribe backtest_completo / backtest_flujo_neto_completo /
# backtest_pieza1_violaciones / backtest_pieza3_pit / _print_backtest.
#
# Por que importa: main() llama a backtest_completo, asi que con vf6 se
# corria el backtest viejo teniendo el mejorado en disco. Y las tres
# funciones nuevas son exactamente las que hacen falta para diagnosticar el
# coverage por debajo del nominal (83.0% en RESTO_GLOBALES contra 90%):
# Wilson dice si la brecha es significativa, Newey-West corrige por el
# racimo de excedencias, y Anderson-Darling sobre el PIT es la version
# afilada de la validacion V1 del paper.
from step006_simulacion_paths_vf7 import (
    pipeline_simulacion,
    graficar_fanchart_origen,
    simular_regimen_path,
    calcular_percentiles_acumulado,
    backtest_completo,
    backtest_flujo_neto_completo,
    cargar_preds_test_reales,
    estimar_rho_por_regimen,
    fitear_distribuciones_por_horizonte,
    generar_fancharts_todos_origenes,
    generar_fancharts_neto_todos_origenes,
    generar_fancharts_integrados_todos_origenes,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


###############################################################################
# Configuración — ajustar a tus rutas reales
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
MODELO       = "expanding"

# ═════════════════════════════════════════════════════════════════════════════
# BOTON: N=1 (SISTEMA) o N=2 (simulacion CONJUNTA por grupos)
# ═════════════════════════════════════════════════════════════════════════════
# False → una sola entidad, el motor vigente. Es el caso N=1 del paper
#         ("Simulacion conjunta de flujos netos por grupos del sistema
#         financiero", seccion 6.2): con N=1 la ecuacion (5) entrega
#         sigma_e^2 = 1 - phi^2 y la recursion (4) se reduce a
#         z_h = phi*z_{h-1} + sqrt(1-phi^2)*w_h, que es literalmente
#         simular_un_path() de step006_simulacion_paths_vf7.py. El propio paper
#         lo dice: "el esquema es una extension estricta del motor vigente, no
#         un reemplazo". Con False NADA cambia respecto de antes de este boton.
#
# True  → carga FOCO_<P> y RESTO_<P> y los simula JUNTOS, siguiendo el
#         algoritmo P1-P5 de la seccion 6. El objeto de decision es el sistema
#         agregado, asi que no hay ENTIDAD que elegir: entran las dos.
#
# POR QUE NO SE PUEDE "CORRER DOS VECES Y SUMAR" — es la razon por la que este
# boton cambia el flujo de control y no solo la configuracion. En el paso P2 el
# ruido eta_h es UN VECTOR de N componentes sorteado una sola vez y multiplicado
# por L(s_h) = chol(Sigma_e(s_h)); ahi, y solo ahi, entra rho_ij al sistema. Dos
# corridas independientes con semillas distintas producen rho_ij = 0 efectivo,
# sin importar que rho_ij se haya estimado. Y sumar los percentiles de cada
# corrida es el error de no-aditividad que el paper descarta en su seccion 2.
PARTICIONES = False

# Solo se leen con PARTICIONES=True. Se declaran igual aca para que la
# configuracion viva en un unico lugar, mismo criterio que step005.
PARTICION = "globales"    # "bbva" | "globales" — debe coincidir con step005

# Solo se lee con PARTICIONES=False: QUE entidad corre el motor N=1.
#
# Existe porque antes de este boton BANCO era un string libre y se le podia
# poner "FOCO_BBVA" para correr N=1 sobre esa entidad y obtener SUS fan charts.
# Al derivar BANCO de PARTICIONES esa flexibilidad se perdio: False forzaba
# SISTEMA. ENTIDAD la devuelve, con los mismos valores que usa step005.
#
# Con PARTICIONES=True este valor se ignora: el modo conjunto corre las DOS
# caras de la particion por definicion, no hay entidad que elegir.
#
# Para que sirve en la practica: main_conjunto() no genera fan charts (los tres
# generadores del modulo esperan rho_por_regimen escalar), asi que si se quiere
# el fan chart de FOCO o de RESTO por separado hay que correrlos en N=1.
ENTIDAD = "SISTEMA"       # "SISTEMA" | "FOCO" | "RESTO"

# ── Sobre QUE estan estratificados los phi_i(s) y rho_ij(s) que se van a usar ─
# "auto"       → lo toma de la columna condicionar_por de preds_test.
# "regimen"    → declara que son estados del HMM (se muestrea la cadena de Markov).
# "calendario" → declara que son baldes de posicion en el mes (s_h deterministo,
#                sin cadena de Markov; requiere la columna balde_th).
#
# El valor declarado se CONTRASTA contra lo que dicen los datos y, si no
# coinciden, la corrida aborta. La verificacion no es burocracia: el subindice s
# de rho_s_* significa cosas distintas en los dos modos, y aplicar el indice
# equivocado no produce ningun error visible — se aplicaria phi(cierre) en los
# dias que el HMM llama "moderado" y nunca phi(transicion), devolviendo paths
# plausibles con el significado cambiado. Justo el caso que este guard atajo la
# primera vez que aparecio.
#
# Declararlo explicitamente (en vez de "auto") deja el modo a la vista en la
# cabecera del archivo y convierte un preds_test inesperado en un aborto
# inmediato en lugar de una corrida silenciosa en el otro modo.
CONDICIONAR_POR = "auto"

if CONDICIONAR_POR not in ("auto", "regimen", "calendario"):
    raise ValueError(
        f"CONDICIONAR_POR={CONDICIONAR_POR!r} invalido — debe ser 'auto', "
        f"'regimen' o 'calendario'.")

# Geometria del fold de la corrida de step005 que se quiere leer. NO reconfigura
# nada: solo reconstruye el nombre del subnivel de carpeta que step005 crea con
# PARTICIONES=True (dirs_de_banco -> etiqueta_corrida). Si en step005 cambia
# VENTANA_VAL_AÑOS o VENTANA_TEST_AÑOS, hay que reflejarlo aca o el loader
# apunta a una carpeta que no existe.
VENTANA_VAL_AÑOS  = 1
VENTANA_TEST_AÑOS = 0.5


def _fmt_anios(x: float) -> str:
    """0.5 -> '0.5', 1.0 -> '1'. Copiado literal de step005_walk_forward_cv_3.7."""
    return f"{x:g}"


def etiqueta_corrida(banco: str) -> str:
    """
    Identidad de la corrida: entidad + geometria del fold. Ej: FOCO_BBVA_1_0.5

    Replica etiqueta_corrida() de step005_walk_forward_cv_3.7.py, que es la que
    define el subnivel de carpeta donde viven los preds_test con
    PARTICIONES=True. Las dos definiciones tienen que moverse juntas.
    """
    return f"{banco}_{_fmt_anios(VENTANA_VAL_AÑOS)}_{_fmt_anios(VENTANA_TEST_AÑOS)}"


# Grupos que entran a la simulacion. Con PARTICIONES=False es ["SISTEMA"] y
# BANCO queda en "SISTEMA" igual que antes; con True son las DOS caras de la
# particion, en orden fijo (FOCO primero) para que el indice i de phi_i, de las
# filas/columnas de R y del vector Z sea siempre el mismo.
if not PARTICIONES:
    if ENTIDAD == "SISTEMA":
        BANCO = "SISTEMA"
    elif ENTIDAD in ("FOCO", "RESTO"):
        BANCO = f"{ENTIDAD}_{PARTICION.upper()}"
    else:
        raise ValueError(f"ENTIDAD={ENTIDAD!r} no es valida. Opciones: "
                         f"'SISTEMA', 'FOCO', 'RESTO'.")
    GRUPOS = [BANCO]
else:
    GRUPOS = [f"FOCO_{PARTICION.upper()}", f"RESTO_{PARTICION.upper()}"]
    # Etiqueta del agregado para nombres de archivo de salida. No es una entidad
    # de step005: es la suma S_h = sum_i X_{i,h} de la ecuacion (1) del paper.
    BANCO  = f"CONJUNTO_{PARTICION.upper()}"

# ¿Los preds_test viven en un subnivel de carpeta, o en la base?
#
# Lo decide la corrida de STEP005 que los produjo, no esta config: step005
# agrega el subnivel etiqueta_corrida(banco) cuando SU PARTICIONES=True
# (dirs_de_banco), y esta config no tiene forma de saber con que flags se corrio
# aquello. Por eso no se adivina: se MIRA EL DISCO (ver dir_modo_de).

# Entidad de la que salen las ETIQUETAS de regimen. DEBE coincidir con
# BANCO_REGIMEN de step005_walk_forward_cv_3.7.py: la columna regimen_hmm de
# preds_test se usa aca como estado_inicial de la cadena de Markov, y la matriz
# de transicion que la propaga tiene que ser la del MISMO HMM. Un estado "2" de
# SISTEMA no significa lo mismo que un "2" de FOCO_BBVA — son ajustes distintos,
# con medias y covarianzas propias, aunque ambos esten ordenados por volatilidad.
# Mezclarlos daria paths de regimen sin sentido, sin ningun error visible.
# None → cada entidad usa su propia transmat (comportamiento anterior).
BANCO_REGIMEN = "SISTEMA"

# Debe coincidir EXACTAMENTE con el DIR_MODO de step005_walk_forward_cv_4.py
# (se construye ahí como DIR_OUTPUT / f"{MODELO_CV}_{modo}_{ventanas}";
# pega aquí el valor resultante de esa corrida — ej. con MODELO_CV="xgb",
# EXPANDING=True, VENTANA_TRAIN_AÑOS=5, VENTANA_VAL_AÑOS=0.5, VENTANA_TEST_AÑOS=1):
_DIR_MODO_BASE = (BASE_SISTEMA / "2. Output" / "step005_wfcv_v3" / "xgb_qt_expanding_310.5")


class Cronometro:
    """
    Reloj por ETAPA, no solo total.

    El total dice "tardo 5 horas" y no sirve para decidir nada. El desglose dice
    cual de las tres etapas se lleva el tiempo —fitear las marginales, simular, o
    dibujar los fan charts— que es lo que determina que optimizar.

    Y lo mas util en una corrida larga: al cerrar el primer año_corte PROYECTA el
    total. Saber en el minuto 5 si faltan 20 minutos o 6 horas cambia lo que uno
    hace con la tarde.
    """

    def __init__(self, n_bloques: int, etiqueta: str = ""):
        self.t0 = time.time()
        self.n_bloques = max(int(n_bloques), 1)
        self.etiqueta = etiqueta
        self.acum: dict[str, float] = {}
        self.bloques_hechos = 0
        self._t_etapa = None
        self._nombre_etapa = None

    def etapa(self, nombre: str):
        """Context manager: acumula el tiempo de esa etapa y lo loguea."""
        crono = self

        class _Ctx:
            def __enter__(self):
                crono._t_etapa = time.time()
                crono._nombre_etapa = nombre
                return crono

            def __exit__(self, *exc):
                dt = time.time() - crono._t_etapa
                crono.acum[nombre] = crono.acum.get(nombre, 0.0) + dt
                logger.info(f"    [t] {nombre}: {crono._fmt(dt)}")
                return False
        return _Ctx()

    def cerrar_bloque(self) -> None:
        """Marca un año_corte terminado y proyecta lo que falta."""
        self.bloques_hechos += 1
        transcurrido = time.time() - self.t0
        if self.bloques_hechos >= self.n_bloques:
            return
        por_bloque = transcurrido / self.bloques_hechos
        restante = por_bloque * (self.n_bloques - self.bloques_hechos)
        logger.info(
            f"  [t] bloque {self.bloques_hechos}/{self.n_bloques} en "
            f"{self._fmt(transcurrido)} — proyeccion: faltan "
            f"~{self._fmt(restante)}, total ~{self._fmt(transcurrido + restante)}")

    def resumen(self) -> None:
        total = time.time() - self.t0
        logger.info(f"\n  ── Tiempos{' ' + self.etiqueta if self.etiqueta else ''} "
                    f"─────────────────────────────")
        for nombre, dt in sorted(self.acum.items(), key=lambda kv: -kv[1]):
            logger.info(f"    {nombre:28s} {self._fmt(dt):>10s}   "
                        f"{dt / total * 100:5.1f}%")
        _otros = total - sum(self.acum.values())
        if _otros > 0.05 * total:
            logger.info(f"    {'(resto: I/O, concat, etc.)':28s} "
                        f"{self._fmt(_otros):>10s}   {_otros / total * 100:5.1f}%")
        logger.info(f"    {'TOTAL':28s} {self._fmt(total):>10s}")

    @staticmethod
    def _fmt(seg: float) -> str:
        if seg < 60:
            return f"{seg:.1f}s"
        if seg < 3600:
            return f"{seg / 60:.1f}min"
        return f"{seg / 3600:.2f}h"


def _avisar_salida_sin_modo() -> None:
    """
    Con CONDICIONAR_POR="auto" las salidas NO se separan por modo, porque el modo
    recien se conoce al leer preds_test y estas rutas se arman al importar. O sea
    que una corrida de calendario pisa el .parquet y los PNG de una de regimen.

    Se avisa al arrancar la corrida y no al importar: a nivel de modulo el aviso
    saldria tambien al cargar el archivo desde un checklist, donde no significa
    nada.
    """
    if CONDICIONAR_POR == "auto":
        logger.warning(
            "CONDICIONAR_POR='auto': las salidas NO se separan por modo, asi que "
            "una corrida de calendario va a pisar los .parquet y los PNG de una "
            "de regimen. Declaralo ('regimen' o 'calendario') para que convivan.")


def dir_modo_de(banco: str) -> Path:
    """
    Carpeta donde step005 dejo los preds_test de esa entidad. Se RESUELVE
    mirando el disco, no derivando de la config.

    POR QUE NO SE DEDUCE
    step005 agrega un subnivel etiqueta_corrida(banco) cuando SU PARTICIONES=True
    (dirs_de_banco). Esta config no sabe con que flags se corrio aquello, y la
    deduccion "PARTICIONES or ENTIDAD != 'SISTEMA'" fallaba en un caso real: los
    preds de SISTEMA generados por una corrida de step005 CON particiones quedan
    en .../SISTEMA_1_0.5/, pero step006 con PARTICIONES=False los buscaba en la
    base y no los encontraba.

    Mirar el disco elimina la ambiguedad de raiz. Y no es solo comodidad: sin
    esto, el glob de cargar_preds_test_reales que busca en la carpeta PADRE
    puede levantar preds_test de una corrida vieja de OTRA entidad y simularla
    en silencio, creyendo simular la que se pidio.

    Prioridad y por que ese orden:
      1. el subnivel, si existe y tiene preds de ESE banco — es el resultado de
         la corrida mas especifica, la que se hizo con particiones;
      2. la base, si tiene preds de ese banco;
      3. si ninguna los tiene, el subnivel igual, para que el error de
         cargar_preds_de_grupos liste la carpeta correcta.
    """
    sub  = _DIR_MODO_BASE / etiqueta_corrida(banco)
    pat  = f"preds_test_fold*_{banco}_*.parquet"

    # Candidatas EN ORDEN DE PRIORIDAD.
    #
    # step005 separa los modos en un subnivel cond_<modo> (dirs_de_banco), y
    # regimen —el caso historico— no lleva subnivel. Asi que el botón de este
    # archivo no solo declara el modo: tambien elige DONDE buscar.
    #   CONDICIONAR_POR="calendario" -> primero cond_calendario/
    #   "regimen"                    -> las carpetas sin subnivel de modo
    #   "auto"                       -> las sin subnivel primero (el caso
    #                                   historico) y cond_* despues, asi una
    #                                   corrida de regimen nunca queda tapada
    #                                   por una de calendario hecha despues.
    _cands = []
    if CONDICIONAR_POR == "calendario":
        _cands += [sub / "cond_calendario", _DIR_MODO_BASE / "cond_calendario"]
    _cands += [sub, _DIR_MODO_BASE]
    if CONDICIONAR_POR == "auto":
        _cands += [sub / "cond_calendario", _DIR_MODO_BASE / "cond_calendario"]

    try:
        for c in _cands:
            if c.is_dir() and any(c.glob(pat)):
                if c.name.startswith("cond_"):
                    logger.info(f"  preds de {banco} en el subnivel de modo "
                                f"{c.name!r}")
                return c
    except OSError as e:
        # La unidad de red puede no estar montada al importar el modulo (p.ej.
        # al correr un checklist). No es motivo para abortar el import: se cae
        # al comportamiento deducido y el error real aparece al cargar.
        logger.debug(f"  No se pudo inspeccionar {sub} ({type(e).__name__}: {e})")
    # Nada en disco: se devuelve la candidata MAS especifica de esta config, para
    # que el error de cargar_preds_de_grupos nombre donde deberian estar.
    return _cands[0] if (PARTICIONES or banco != "SISTEMA"
                         or CONDICIONAR_POR == "calendario") else _DIR_MODO_BASE


DIR_MODO = dir_modo_de(GRUPOS[0])

# Carpeta donde step005_validar_hmm_*.py guardó transmat_hmm_<banco>.parquet
# (su DIR_OUTPUT — normalmente la misma "2. Output" del proyecto).
DIR_REGIMEN_HMM = BASE_SISTEMA / "2. Output"

# Salida de este orquestador
# _SUF_SALIDA: subnivel por corrida. A diferencia de dir_modo_de —que RESUELVE
# donde dejo los preds una corrida ajena y por eso mira el disco— aca se DECIDE
# donde escribir, asi que es una regla y no una deteccion.
#
# Con PARTICIONES=False y ENTIDAD="SISTEMA" es "" y las rutas quedan EXACTAMENTE
# como estaban antes de los botones. En cualquier otro caso separa las salidas
# —del conjunto o de una entidad de particion— de las de SISTEMA, que comparten
# nombre de archivo y se pisarian.
_SUF_SALIDA = "" if (not PARTICIONES and ENTIDAD == "SISTEMA") else etiqueta_corrida(BANCO)

# ── Separacion por MODO de condicionamiento, igual que step005 ──────────────
# Sin esto, una corrida de calendario pisa el simulacion_paths_*.parquet y los
# PNG de fan chart de una corrida de regimen: mismo nombre, misma carpeta. Es el
# mismo problema que cond_<modo> resuelve aguas arriba, un nivel mas abajo.
#
# Solo se puede separar cuando el modo esta DECLARADO: con "auto" no se sabe
# cual es hasta leer preds_test, y estas rutas se arman al importar el modulo.
# Por eso declarar CONDICIONAR_POR no es solo documentacion — es lo que hace que
# las salidas de los dos modos convivan.
if CONDICIONAR_POR == "calendario":
    _SUF_SALIDA = f"{_SUF_SALIDA}/cond_calendario" if _SUF_SALIDA else "cond_calendario"
# El aviso para "auto" va dentro de la corrida (_avisar_salida_sin_modo), no
# aca: a nivel de modulo dispararia con solo importar el archivo — p.ej. al
# correr un checklist — y ahi no significa nada.
DIR_SALIDA = (BASE_SISTEMA / "2. Output" / "step006_simulacion" /
              "xgb_qt_expanding_310.5" / _SUF_SALIDA)

# Columna de Prophet en df_preds, si tu 'target' es un RESIDUO de Prophet que
# hay que sumar de vuelta. None si 'target'/'y_realizado' ya es el flujo
# directo (caso típico si XGBoost predice el flujo, no un residuo).♣
PROPHET_COL = None

VENTANAS = list(np.linspace(2,75,74))
VENTANAS = [int(i) for i in VENTANAS]
#VENTANAS = [2, 5,10, 22,44,66,75]     # plazos en días hábiles (igual que step006)

N_PATHS  = 10000               # paths simulados por fecha de origen
N_JOBS   = -1                 # procesos paralelos para fitear skew-t/PIT
                              # (-1 = todos los cores; 1 = serial). El fit de
                              # cada skew-t es ~0.2s — sobre miles de filas el
                              # cómputo serial es impracticable, ver
                              # step006_simulacion_paths_vf7.py.
TAU_BACKTEST = 0.05           # cuantil usado para la pieza 1/2 del backtest
H_REFERENCIA_RHO = None      # horizonte para estimar ρ_s; None = autodetecta el
                              # mínimo h presente en los datos (revisa el log al
                              # correr — fija un entero aquí si quieres forzar otro)
SEED = 42

# ── Fan charts de flujo acumulado (uno por día de origen) ───────────────────
GENERAR_FANCHARTS    = True
DIR_FLUJOS_ACUMULADOS = (BASE_SISTEMA / "2. Output" / "flujos_acumulados" /
                       "xgb_qt_expanding_310.5" / _SUF_SALIDA)
N_PATHS_FANCHART      = 100  # paths por origen (default aumentado para aprovechar
                               # la paralelización — con 8 cores el tiempo de cómputo
                               # es comparable al anterior con 1000 paths serial)
BANDAS_FANCHART       = None   # None = usa BANDAS_FANCHART_DEFAULT de
                               # step006_simulacion_paths_vf7.py (P40-60 ... P01-99)

# ── Fan charts de flujo neto diario (sin acumulación) ────────────────────────
# Los percentiles se extraen analíticamente de las AzzaliniT ya fiteadas —
# no requieren simulación adicional. Las bandas pueden no ser monótonas en h.
GENERAR_FANCHARTS_NETO = True
DIR_FLUJOS_NETOS = (BASE_SISTEMA / "2. Output" / "flujos_netos" /
                  "xgb_qt_expanding_310.5" / _SUF_SALIDA)

# ── Fanchart INTEGRADO (3 filas: neto XGBoost crudo / neto distribución / ────
#     acumulado simulado) — imagen adicional, NO reemplaza las anteriores.
GENERAR_FANCHARTS_INTEGRADO = True
DIR_FLUJOS_INTEGRADOS = (BASE_SISTEMA / "2. Output" / "flujos_integrados" /
                       "xgb_qt_expanding_310.5" / _SUF_SALIDA)

# ── Backtest extendido: taus evaluados ───────────────────────────────────────
# TAU_BACKTEST_ACUM : tau existente del backtest acumulado (no cambia).
# TAUS_BACKTEST_EXTRA: taus adicionales para el backtest acumulado (e.g. 0.01).
# TAUS_BACKTEST_NETO : taus para el backtest de flujos netos (pointwise).
TAU_BACKTEST_ACUM   = 0.05
TAUS_BACKTEST_EXTRA = [0.01]
TAUS_BACKTEST_NETO  = [0.01, 0.05]


###############################################################################
# Loader liviano de transmat_hmm_<banco>.parquet (sin importar el script HMM
# completo — evita depender de hmmlearn solo para leer un parquet pequeño)
###############################################################################

def cargar_transmat(banco: str, año_corte,
                    dir_regimen: Path = DIR_REGIMEN_HMM) -> np.ndarray:
    """
    Lee transmat_hmm_<banco>.parquet y devuelve la matriz NxN del año_corte
    pedido. año_corte puede ser string ISO "YYYY-MM-DD" (nuevo) o int (legacy).
    La comparación se hace por string para ser insensible al tipo del parquet.
    """
    ruta = dir_regimen / f"transmat_hmm_{banco}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Corre primero step005_validar_hmm_*.py "
            f"con GUARDAR_OBJETOS_SIMULACION=True.")
    df_t = pd.read_parquet(ruta)
    # Normalizar a string para comparación tipo-insensible
    df_t["_año_corte_str"] = df_t["año_corte"].astype(str)
    año_corte_str = str(año_corte)
    fila = df_t[df_t["_año_corte_str"] == año_corte_str]
    if fila.empty:
        disponibles = sorted(df_t["año_corte"].astype(str).unique().tolist())
        raise ValueError(f"No hay transmat para año_corte={año_corte} en {ruta}. "
                        f"Disponibles: {disponibles}")
    n_estados = int(np.sqrt(len([c for c in df_t.columns if re.fullmatch(r"p\d+", c)])))
    cols = [f"p{i}{j}" for i in range(n_estados) for j in range(n_estados)]
    return fila.iloc[0][cols].values.astype(float).reshape(n_estados, n_estados)


###############################################################################
# Orquestación principal
###############################################################################

def _detectar_n_estados_rho(columnas) -> int | None:
    """
    Detecta cuántos estados hay a partir de las columnas rho_s_0, rho_s_1, ...
    realmente presentes en el parquet — nunca asume 2 o 3 fijo, porque
    step005_validar_hmm*.py puede haberse corrido con N_ESTADOS=2 o 3.

    Devuelve None si no hay al menos rho_s_0 y rho_s_1, o si la secuencia
    tiene huecos (p.ej. rho_s_0 y rho_s_2 sin rho_s_1 — eso es un dato
    corrupto, no un caso válido de N_ESTADOS, y se trata como ausente).
    """
    indices = sorted(
        int(m.group(1)) for c in columnas
        if (m := re.fullmatch(r"rho_s_(\d+)", str(c)))
    )
    if len(indices) < 2 or indices != list(range(len(indices))):
        return None
    return len(indices)


###############################################################################
# Algebra de la simulacion conjunta (paper, secciones 4 y 5)
#
# Vive aca y no en step006_simulacion_paths_vf7.py por dos razones: (a) son
# funciones PURAS —numpy y nada mas— que se validan sin tocar la unidad H:, y
# (b) el modulo de simulacion ya es el motor N=1 y conviene no moverlo mientras
# el conjunto este en prueba. Si el modo conjunto se consolida, el lugar natural
# de estas cuatro funciones es el modulo.
###############################################################################

def construir_sigma_e(phi, R) -> np.ndarray:
    """
    Ecuacion (5) del paper:  Sigma_e = R (x) (11' - phi phi')

    elemento a elemento:  (Sigma_e)_ij = rho_ij * (1 - phi_i*phi_j)
                          (Sigma_e)_ii = 1 - phi_i^2

    (x) es el producto de Hadamard. Sigma_e NO es un parametro libre: queda
    determinada al exigir que el VAR(1) sea estacionario con matriz de
    correlacion contemporanea igual a R. Esa calibracion es lo que garantiza
    Var(Z_{i,h}) = 1 para todo h, y de ahi que U = Phi_N(Z) sea exactamente
    uniforme y que las marginales simuladas reproduzcan F_{i,h} sin sesgo. Si
    Sigma_e se fija de otro modo, la varianza latente deriva con el horizonte
    (seccion 4).
    """
    phi = np.asarray(phi, dtype=float).ravel()
    R   = np.asarray(R, dtype=float)
    if R.shape != (len(phi), len(phi)):
        raise ValueError(f"R debe ser {len(phi)}x{len(phi)}, es {R.shape}")
    return R * (np.ones_like(R) - np.outer(phi, phi))


def cota_rho_n2(phi_1: float, phi_2: float) -> float:
    """
    Ecuacion (7): cota cerrada sobre |rho_12| para que Sigma_e sea PSD con N=2.

        |rho_12| <= sqrt((1 - phi_1^2)(1 - phi_2^2)) / (1 - phi_1*phi_2)

    Vale exactamente 1 cuando phi_1 == phi_2 —o sea NO restringe— y se estrecha
    a medida que las persistencias divergen. Por eso el problema puede pasar
    inadvertido con dos grupos de comportamiento parecido y aparecer al
    generalizar a N>2 (Cuadro 1 del paper).

    Solo informativa/diagnostica: la verificacion operativa es lambda_min.
    """
    num = np.sqrt(max((1.0 - phi_1**2) * (1.0 - phi_2**2), 0.0))
    den = 1.0 - phi_1 * phi_2
    if den <= 1e-12:
        return 0.0
    return float(min(num / den, 1.0))


def lambda_estrella(phi, R, tol: float = 1e-6) -> float:
    """
    Ecuacion (8): encogimiento sobre R hasta que Sigma_e sea PSD.

        R(lam) = lam*R + (1-lam)*I
        lam*   = max{ lam en (0,1] : lambda_min(Sigma_e(lam)) >= 0 }

    resuelto por biseccion. Devuelve 1.0 si Sigma_e ya es PSD con R sin tocar.

    Se prefiere al truncamiento espectral (Sigma_e = P max(D,0) P') porque lam*
    es UN numero interpretable y reportable —el factor de atenuacion uniforme
    aplicado a la correlacion transversal— en vez de una deformacion arbitraria
    de la estructura.

    Que lam* resulte bajo es informacion sustantiva, no un detalle numerico:
    indica que la restriccion de Phi diagonal (supuesto A2) es demasiado rigida
    para los datos, y sugiere pasar a un VAR(1) completo.

    Validado contra el ejemplo del paper: N=3, phi=(0.90,0.50,0.10),
    rho=(0.7,0.6,0.5) -> lambda_min(Sigma_e) = -0.114 y lam* = 0.72.
    """
    R = np.asarray(R, dtype=float)
    I = np.eye(len(R))

    def _lmin(lam: float) -> float:
        return float(np.linalg.eigvalsh(construir_sigma_e(phi, lam * R + (1 - lam) * I)).min())

    if _lmin(1.0) >= 0.0:
        return 1.0
    # lam=0 da R=I -> Sigma_e = diag(1-phi_i^2), PSD siempre que |phi_i|<1, asi
    # que la biseccion tiene solucion garantizada dentro de [0, 1].
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _lmin(mid) >= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


def preparar_bloques_conjuntos(phi_por_s: dict, R_por_s: dict,
                              etiqueta: str = "") -> dict:
    """
    Precomputo del algoritmo (seccion 6): una sola vez por fold.

    Para cada regimen s: Sigma_e(s) por (5), verificacion PSD, eventual
    encogimiento (8), y L(s) = chol(Sigma_e(s)). Ademas L0 = chol(R) del regimen
    inicial, para la inicializacion estacionaria del paso P1.

    Parametros
    ----------
    phi_por_s : {s: array (N,)}  — phi_i(s) por grupo, en el orden de GRUPOS.
    R_por_s   : {s: array (N,N)} — matriz de correlacion transversal por regimen.

    Devuelve
    --------
    {"Phi": {s: array (N,)}, "L": {s: array (N,N)}, "L0": array (N,N),
     "lam": {s: float}, "lmin": {s: float}, "cota": {s: float|None}}

    La verificacion PSD es OBLIGATORIA antes de simular, no opcional: la matriz
    11' - phi phi' de (5) no es PSD en general, de modo que el teorema de Schur
    sobre productos de Hadamard no aplica y Sigma_e puede no admitir Cholesky.
    """
    Phi, L, lam, lmin, cota = {}, {}, {}, {}, {}
    for s in sorted(phi_por_s):
        phi = np.asarray(phi_por_s[s], dtype=float).ravel()
        R   = np.asarray(R_por_s[s], dtype=float)
        N   = len(phi)
        if np.any(np.abs(phi) >= 1.0):
            raise ValueError(f"|phi_i| >= 1 en el regimen {s}: {phi} — "
                             f"la condicion (2) exige |phi_i| < 1.")
        lmin[s] = float(np.linalg.eigvalsh(construir_sigma_e(phi, R)).min())
        cota[s] = cota_rho_n2(phi[0], phi[1]) if N == 2 else None
        lam[s]  = lambda_estrella(phi, R)
        if lam[s] < 1.0:
            R = lam[s] * R + (1 - lam[s]) * np.eye(N)
            logger.warning(
                f"  [CONJ]{etiqueta} regimen {s}: Sigma_e NO era PSD "
                f"(lambda_min={lmin[s]:+.4f}) — encogimiento (8) con "
                f"lambda*={lam[s]:.4f}. Un lambda* bajo indica que Phi diagonal "
                f"(supuesto A2) es demasiado rigida para estos datos y sugiere "
                f"un VAR(1) completo.")
        Phi[s] = phi
        # jitter minimo: con lambda_min == 0 exacto (el borde que deja la
        # biseccion) Cholesky puede fallar por redondeo.
        Se = construir_sigma_e(phi, R)
        try:
            L[s] = np.linalg.cholesky(Se)
        except np.linalg.LinAlgError:
            L[s] = np.linalg.cholesky(Se + 1e-10 * np.eye(N))
        _txt_cota = f" cota(7)={cota[s]:.3f}" if cota[s] is not None else ""
        logger.info(f"  [CONJ]{etiqueta} regimen {s}: phi={np.round(phi, 4).tolist()} "
                    f"lambda_min={lmin[s]:+.4f} lambda*={lam[s]:.4f}{_txt_cota}")

    s0 = sorted(R_por_s)[0]
    R0 = np.asarray(R_por_s[s0], dtype=float)
    if lam[s0] < 1.0:
        R0 = lam[s0] * R0 + (1 - lam[s0]) * np.eye(len(R0))
    try:
        L0 = np.linalg.cholesky(R0)
    except np.linalg.LinAlgError:
        L0 = np.linalg.cholesky(R0 + 1e-10 * np.eye(len(R0)))
    return {"Phi": Phi, "L": L, "L0": L0, "lam": lam, "lmin": lmin, "cota": cota}


def simular_un_path_conjunto(fecha_t, horizontes, regimenes_path, bloques,
                             distribuciones, grupos, rng) -> np.ndarray:
    """
    Pasos P1-P4 del algoritmo, para una replica j.

    P1  Z_{h0-1} = L0 @ xi,            xi ~ N(0, I_N)   (inicio estacionario)
    P2  Z_h      = Phi(s_h) * Z_{h-1} + L(s_h) @ eta_h, eta_h ~ N(0, I_N)
    P3  U_{i,h}  = Phi_N(Z_{i,h})
    P4  X_{i,h}  = F^-1_{i,h}(U_{i,h})   con la marginal PROPIA de ese par (i,h)

    Devuelve array (N, H) en escala de flujos. La AGREGACION (P5) la hace el
    llamador, porque necesita acumular por ventana.

    Por que Phi(s) se aplica como producto elemento a elemento y no como
    matriz: Phi = diag(phi_1..phi_N) por el supuesto A2 (sin efectos cruzados
    con rezago), asi que Phi @ Z == phi * Z y el producto matricial seria
    trabajo de mas.

    eta_h es UN sorteo de N componentes compartido por los grupos: es el unico
    punto donde rho_ij entra al sistema, via L(s_h) = chol(Sigma_e(s_h)).
    Sortear un eta por grupo destruiria la dependencia transversal.
    """
    N, H = len(grupos), len(horizontes)
    _s_fb = sorted(bloques["Phi"])[0]
    regimenes_path = np.asarray(regimenes_path)

    Z_todos = np.empty((N, H))
    z = bloques["L0"] @ rng.standard_normal(N)              # P1
    W = rng.standard_normal((H, N))                         # un eta_h por horizonte
    for k in range(H):
        s = int(regimenes_path[k])
        # Un regimen sin bloque solo puede pasar si la transmat tiene mas estados
        # que rho_s_* en preds_test; el guard de main() lo descarta antes, esto
        # es el cinturon.
        phi = bloques["Phi"].get(s, bloques["Phi"][_s_fb])
        L   = bloques["L"].get(s,   bloques["L"][_s_fb])
        z   = phi * z + L @ W[k]                            # P2
        Z_todos[:, k] = z

    U = _norm_dist.cdf(Z_todos)                             # P3, vectorizado
    X = np.zeros((N, H))
    _ft = pd.Timestamp(fecha_t)
    for i, g in enumerate(grupos):
        for k in range(H):
            dist = distribuciones.get((g, _ft, int(horizontes[k])))
            if dist is not None:                            # P4
                X[i, k] = float(dist.ppf(np.clip(U[i, k], 1e-7, 1.0 - 1e-7))[0])
    return X


def secuencia_regimen(estado_inicial, matriz_transicion, H, rng,
                      baldes_fijos=None) -> np.ndarray:
    """
    La secuencia s_{h0..H} de una replica. Dos modos, y la diferencia es
    metodologica, no de implementacion:

    baldes_fijos=None  -> modo REGIMEN. s_h es latente: se muestrea de la cadena
                          de Markov con simular_regimen_path. Cada replica tiene
                          su propia trayectoria, y eso agrega varianza Monte
                          Carlo sobre la del ruido.

    baldes_fijos dado   -> modo CALENDARIO. s_h es DETERMINISTA: la posicion en
                          el mes de t+h se conoce con certeza al decidir, asi que
                          la secuencia es la MISMA en todas las replicas y no se
                          sortea nada. Es la ventaja practica del calendario que
                          motivo el boton: una fuente de ruido menos, y la
                          matriz de transicion deja de hacer falta.

    Un balde -1 (t+h fuera del eje habil o en un mes incompleto; ver _baldes_th
    de step005) se resuelve con el bloque de menor indice, igual que hace
    simular_un_path_conjunto con un estado sin bloque.
    """
    if baldes_fijos is None:
        return simular_regimen_path(estado_inicial, matriz_transicion, H, rng)
    b = np.asarray(baldes_fijos, dtype=int)
    if len(b) != H:
        raise ValueError(f"baldes_fijos tiene {len(b)} elementos y se esperaban "
                         f"{H} (uno por horizonte).")
    return b


def simular_paths_origen_conjunto(
    fecha_t, df_por_grupo: dict, distribuciones: dict, estado_inicial: int,
    matriz_transicion: np.ndarray, bloques: dict, n_paths: int,
    ventanas: list, grupos: list, seed: int = 42, baldes_fijos=None,
) -> dict:
    """
    Paso P5 y ecuacion (9): agrega los grupos DENTRO de cada replica y devuelve
    la distribucion del acumulado del SISTEMA por ventana.

        S_h^(j) = sum_i X_{i,h}^(j)          (agregacion transversal)
        C_{h0:h}^(j) = sum_{k=h0}^{h} S_k^(j)  (acumulacion temporal)

    El orden de las operaciones es la esencia del metodo (seccion 6): se suma
    primero —dentro de la replica, donde la dependencia ya esta incorporada— y
    el cuantil se toma despues, sobre las J replicas. Invertir el orden
    reintroduce el error de no-aditividad de cuantiles.

    La trayectoria de regimen se sortea UNA vez por replica y se comparte entre
    los grupos: es el s_h de un solo subindice del supuesto A3.

    Devuelve {ventana: array (n_paths,)} — igual que simular_paths_origen del
    modulo, asi que calcular_percentiles_acumulado se reusa sin cambios.
    """
    # Grilla de horizontes comun: ya viene alineada por el inner join del loader,
    # asi que basta tomarla del primer grupo.
    horizontes = (df_por_grupo[grupos[0]]
                  .sort_values("h")["h"].values.astype(int))
    H = len(horizontes)
    acum = {v: np.empty(n_paths) for v in ventanas}
    idx_por_v = {v: np.where(horizontes <= v)[0] for v in ventanas}

    for p in range(n_paths):
        rng_p = np.random.default_rng(seed + p)
        regimenes = secuencia_regimen(
            estado_inicial, matriz_transicion, H, rng_p,
            baldes_fijos=baldes_fijos)                        # P0 (compartido)
        X = simular_un_path_conjunto(
            fecha_t, horizontes, regimenes, bloques,
            distribuciones, grupos, rng_p)                    # P1-P4
        S = X.sum(axis=0)                                     # P5, S_h
        for v in ventanas:
            i_v = idx_por_v[v]
            acum[v][p] = float(S[i_v].sum()) if len(i_v) else 0.0
    return acum


def simular_fanchart_origen_conjunto(
    fecha_t, horizontes, estado_inicial, matriz_transicion, bloques,
    distribuciones, grupos, n_paths, seed: int = 42, baldes_fijos=None,
) -> np.ndarray:
    """
    Version conjunta de simular_fanchart_origen del modulo: devuelve el
    acumulado del SISTEMA (suma de los grupos) en CADA horizonte.

    Devuelve array (n_paths, len(horizontes)); columna k = distribucion de
    C_{h0:h_k} = sum_{j<=k} sum_i X_{i,h_j}. Es el mismo shape y el mismo
    significado que espera graficar_fanchart_origen, asi que el graficador se
    reusa SIN tocarlo — lo unico que cambia es de donde salen los paths.

    La suma entre grupos ocurre DENTRO de la replica, antes del cumsum, que es
    el orden que exige la seccion 6 del paper. Un fan chart armado a partir de
    los percentiles de FOCO y de RESTO por separado seria el error de
    no-aditividad dibujado.
    """
    H = len(horizontes)
    acumulados = np.empty((n_paths, H))
    for p in range(n_paths):
        rng_p = np.random.default_rng(seed + p)
        regimenes = secuencia_regimen(
            estado_inicial, matriz_transicion, H, rng_p,
            baldes_fijos=baldes_fijos)
        X = simular_un_path_conjunto(
            fecha_t, horizontes, regimenes, bloques,
            distribuciones, grupos, rng_p)
        acumulados[p, :] = np.cumsum(X.sum(axis=0))
    return acumulados


def generar_fancharts_conjunto(
    dfg: dict, distribuciones: dict, estado_por_origen, matriz_transicion,
    bloques: dict, grupos: list, dir_salida: Path, banco: str,
    n_paths: int, seed: int = 42, bandas=None, col_balde: str | None = None,
) -> list:
    """
    Un fan chart PNG por origen, con la banda del acumulado CONJUNTO.

    Serial a proposito, no por descuido: los tres generadores del modulo
    paralelizan con multiprocessing, y en Windows eso es spawn —cada worker
    reimporta numpy/scipy/pandas y cuesta ~150-250 MB de commit charge— que es
    justo lo que hizo fallar esta corrida con WinError 1455. Con
    N_PATHS_FANCHART (100 por defecto, no los 10.000 de la simulacion) el costo
    serial es chico y no vale arriesgar el pagefile.
    """
    dir_salida.mkdir(parents=True, exist_ok=True)
    g0 = grupos[0]
    origenes = sorted(dfg[g0]["fecha_t"].unique())
    logger.info(f"  [fanchart_conj] {len(origenes)} origenes x {n_paths} paths "
                f"-> {dir_salida}")
    rutas = []
    for i_o, ft in enumerate(origenes):
        ft = pd.Timestamp(ft)
        sub0 = dfg[g0][dfg[g0]["fecha_t"] == ft].sort_values("h")
        horizontes = sub0["h"].values.astype(int)
        _bf = (sub0[col_balde].values.astype(int)
               if col_balde and col_balde in sub0.columns else None)
        acum = simular_fanchart_origen_conjunto(
            ft, horizontes, int(estado_por_origen.get(ft, 0)),
            matriz_transicion, bloques, distribuciones, grupos,
            n_paths=n_paths, seed=seed + i_o, baldes_fijos=_bf)

        # Realizado del SISTEMA: se suman los grupos por horizonte y recien
        # despues se acumula — mismo orden que en la simulacion.
        y_real = np.zeros(len(horizontes), dtype=float)
        for g in grupos:
            sg = dfg[g][dfg[g]["fecha_t"] == ft].sort_values("h")
            if "y_realizado" in sg.columns and len(sg) == len(horizontes):
                y_real += sg["y_realizado"].values.astype(float)
        y_acum = dict(zip(horizontes.tolist(), np.cumsum(y_real).tolist()))

        rutas.append(graficar_fanchart_origen(
            ft, horizontes, acum, y_acum, dir_salida,
            banco=banco, bandas=bandas))
        if (i_o + 1) % 25 == 0:
            logger.info(f"    [fanchart_conj] {i_o+1}/{len(origenes)}")
    logger.info(f"  [fanchart_conj] {len(rutas)} imagenes generadas")
    return rutas


def cargar_preds_de_grupos(grupos: list) -> dict:
    """
    Carga el preds_test de cada grupo desde SU carpeta y los alinea sobre la
    grilla comun (fecha_t, h) por INNER JOIN.

    El inner join no es defensivo, es la condicion A5 del paper puesta en
    practica: la suma S_h = sum_i X_{i,h} solo identifica al sistema si los
    grupos particionan exhaustivamente. Un origen presente en FOCO y ausente en
    RESTO no es agregable — sumar ahi seria reportar medio sistema como si fuera
    el total.

    Guard: si en la carpeta de un grupo no hay preds_test suyos, aborta listando
    lo que si hay. Sin el guard, cargar_preds_test_reales glob-ea en esa carpeta
    y, si quedaron archivos de otra entidad con el patron, los levanta en
    silencio (ver dir_modo_de).
    """
    out = {}
    for g in grupos:
        d = dir_modo_de(g)
        if not d.exists():
            raise FileNotFoundError(
                f"No existe la carpeta {d} para el grupo {g}. Con "
                f"PARTICIONES=True step005 escribe en un subnivel "
                f"etiqueta_corrida(banco) = '{etiqueta_corrida(g)}'; verifica "
                f"que VENTANA_VAL_AÑOS={VENTANA_VAL_AÑOS} y "
                f"VENTANA_TEST_AÑOS={VENTANA_TEST_AÑOS} coincidan con la "
                f"corrida de step005. Subcarpetas presentes en "
                f"{_DIR_MODO_BASE}: "
                f"{sorted(p.name for p in _DIR_MODO_BASE.glob('*') if p.is_dir())}")
        if not list(d.glob(f"preds_test_fold*_{g}_*.parquet")):
            raise FileNotFoundError(
                f"No hay preds_test_fold*_{g}_*.parquet en {d}. Archivos de "
                f"preds presentes: "
                f"{sorted(p.name for p in d.glob('preds_test_fold*.parquet'))[:8]}")
        df = cargar_preds_test_reales(d, g)
        logger.info(f"  {g}: {len(df):,} filas | {df['fecha_t'].nunique()} origenes "
                    f"| folds {sorted(df['fold'].unique())}")
        out[g] = df

    # Grilla comun (fecha_t, h): interseccion sobre TODOS los grupos.
    claves = None
    for g in grupos:
        k = set(zip(out[g]["fecha_t"], out[g]["h"].astype(int)))
        claves = k if claves is None else (claves & k)
    if not claves:
        raise ValueError("Los grupos no comparten ningun par (fecha_t, h) — "
                         "no hay nada que agregar. Revisa que las corridas de "
                         "step005 usen la MISMA geometria de folds.")
    for g in grupos:
        n0 = len(out[g])
        m = [(ft, int(h)) in claves for ft, h in zip(out[g]["fecha_t"], out[g]["h"])]
        out[g] = out[g].loc[m].sort_values(["fecha_t", "h"]).reset_index(drop=True)
        if len(out[g]) != n0:
            logger.warning(f"  {g}: {n0 - len(out[g]):,} de {n0:,} filas quedan "
                           f"fuera de la grilla comun (A5: la suma solo "
                           f"identifica al sistema si los grupos particionan)")
    logger.info(f"  grilla comun: {len(claves):,} pares (fecha_t, h) | "
                f"{len(set(ft for ft, _ in claves)):,} origenes")
    return out


def _leer_phi_rho_de_grupo(df_grupo: pd.DataFrame, n_estados: int) -> tuple:
    """
    Extrae phi_i(s) y rho_ij(s) de las columnas constantes por fold que escribe
    step005 (ver _guardar_preds_test): rho_s_0..rho_s_{n-1} y rho_ij / rho_ij_s*.

    Devuelve ({s: phi}, {s: rho_ij}) con rho_ij cayendo al global cuando el
    condicional de ese regimen no existe — misma politica que
    _estimar_rho_transversal, que omite un estado con menos pares que el minimo
    y deja que se use el global.
    """
    fila = df_grupo.drop_duplicates("año_corte_regimen").iloc[0]
    phi = {s: float(fila[f"rho_s_{s}"]) for s in range(n_estados)}
    rho_global = float(fila["rho_ij"]) if "rho_ij" in df_grupo.columns else 0.0
    rho = {}
    for s in range(n_estados):
        col = f"rho_ij_s{s}"
        v = fila[col] if col in df_grupo.columns else np.nan
        rho[s] = float(v) if pd.notna(v) else rho_global
    return phi, rho


def main_conjunto():
    """
    Modo PARTICIONES=True — algoritmo P1-P5 del paper sobre N grupos.

    Estructura, igual que main(): se agrupa por año_corte_regimen porque cada
    fold del walk-forward puede haber usado un bloque HMM distinto, y se simula
    cada grupo con SU transmat. La diferencia es que dentro de cada
    año_corte_regimen los N grupos se simulan JUNTOS.

    Genera el fan chart del acumulado CONJUNTO (GENERAR_FANCHARTS) con
    generar_fancharts_conjunto, que reusa graficar_fanchart_origen del modulo
    sin tocarlo. NO genera los de flujo neto ni el integrado: esos dos leen los
    percentiles de las marginales de UNA entidad, y en el conjunto la marginal
    del agregado no existe en forma cerrada — hay que simularla, que es
    justamente lo que hace el acumulado.
    """
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    _avisar_salida_sin_modo()
    logger.info(f"MODO CONJUNTO — N={len(GRUPOS)} grupos: {GRUPOS}")

    preds = cargar_preds_de_grupos(GRUPOS)

    # n_estados y coherencia de modo: los guards de main() aplicados a cada grupo
    n_est = {g: _detectar_n_estados_rho(preds[g].columns) for g in GRUPOS}
    if any(v is None for v in n_est.values()):
        logger.error(f"Faltan columnas rho_s_* en algun grupo: {n_est}. Corre "
                     f"step005 con ESTIMAR_RHO_EN_VAL=True.")
        return
    if len(set(n_est.values())) != 1:
        logger.error(f"Los grupos traen distinto numero de rho_s_*: {n_est} — "
                     f"vienen de corridas con N_ESTADOS distinto. Regenera.")
        return
    n_estados = next(iter(n_est.values()))
    _modos = set()
    for g in GRUPOS:
        cp = (str(preds[g]["condicionar_por"].iloc[0])
              if "condicionar_por" in preds[g].columns else None)
        _modos.add(cp)
        if cp == "calendario" and "balde_th" not in preds[g].columns:
            logger.error(
                f"{g}: preds_test viene con CONDICIONAR_POR='calendario' pero sin "
                f"la columna balde_th. Sin ella no se puede saber que balde le "
                f"toca a cada horizonte: s_h es deterministo pero derivarlo aca "
                f"exigiria reconstruir el calendario de feriados. Regenera con un "
                f"step005 que incluya _baldes_th.")
            return
        if cp is not None and cp not in ("regimen", "calendario"):
            logger.error(f"{g}: CONDICIONAR_POR='{cp}' desconocido — no soportado.")
            return
    if len(_modos) > 1:
        logger.error(f"Los grupos vienen de modos DISTINTOS: {sorted(_modos)}. "
                     f"phi_i y rho_ij quedarian estratificados por particiones de "
                     f"dias diferentes y Sigma_e dejaria de significar lo que la "
                     f"derivacion dice. Regenera los dos con el mismo modo.")
        return
    _detectado = next(iter(_modos)) or "regimen"
    if CONDICIONAR_POR == "auto":
        _MODO_COND = _detectado
    elif CONDICIONAR_POR != _detectado:
        logger.error(
            f"CONDICIONAR_POR='{CONDICIONAR_POR}' declarado en este archivo, pero "
            f"preds_test dice '{_detectado}'. No se continua: el subindice s de "
            f"rho_s_* significa cosas distintas en los dos modos, y usar el "
            f"indice equivocado no da ningun error visible — aplicaria "
            f"phi(cierre) en los dias que el HMM llama 'moderado'. Corregi "
            f"CONDICIONAR_POR aca, o regenera preds_test con el step005 que "
            f"corresponde.")
        return
    else:
        _MODO_COND = CONDICIONAR_POR
    _CALENDARIO = _MODO_COND == "calendario"
    logger.info(f"  condicionado por: {_MODO_COND} "
                f"({'declarado' if CONDICIONAR_POR != 'auto' else 'detectado'})"
                + ("  (s_h DETERMINISTA: no se muestrea cadena de Markov, la "
                   "secuencia de baldes es la misma en todas las replicas)"
                   if _CALENDARIO else ""))

    resultados, filas_diag = [], []
    _grupos_ac = list(preds[GRUPOS[0]].groupby("año_corte_regimen"))
    crono = Cronometro(len(_grupos_ac), etiqueta="(modo conjunto)")
    for año_corte, df_ref in _grupos_ac:
        try:
            _seed_offset = int(str(año_corte).replace("-", "")) % 100_000
        except Exception:
            _seed_offset = 0
        try:
            # En modo calendario la transmat NO se usa: la secuencia de baldes
            # es deterministica. Se carga igual cuando existe, porque su diag(A)
            # va al diagnostico, pero su ausencia deja de ser motivo para omitir
            # el grupo — y el guard de n_estados vs len(A) tampoco aplica, porque
            # los 4 baldes no tienen por que coincidir con los estados del HMM.
            A = cargar_transmat(BANCO_REGIMEN if BANCO_REGIMEN else GRUPOS[0], año_corte)
        except (FileNotFoundError, ValueError) as e:
            if not _CALENDARIO:
                logger.warning(f"  año_corte_regimen={año_corte}: {e} — grupo omitido.")
                continue
            logger.info(f"  año_corte_regimen={año_corte}: sin transmat, pero en "
                        f"modo calendario no hace falta — se sigue.")
            A = np.eye(n_estados)
        if not _CALENDARIO and n_estados != len(A):
            logger.error(f"  año_corte_regimen={año_corte}: {n_estados} columnas "
                         f"rho_s_* contra transmat de {len(A)} estados — omitido.")
            continue

        # Sub-df de cada grupo para ESTE año_corte
        dfg = {g: preds[g][preds[g]["año_corte_regimen"] == año_corte] for g in GRUPOS}
        if any(d.empty for d in dfg.values()):
            logger.warning(f"  año_corte_regimen={año_corte}: algun grupo sin filas "
                           f"— omitido.")
            continue

        # ── phi_i(s) por grupo y R(s) ────────────────────────────────────────
        phi_rho = {g: _leer_phi_rho_de_grupo(dfg[g], n_estados) for g in GRUPOS}
        if any(np.isnan(list(phi_rho[g][0].values())).any() for g in GRUPOS):
            logger.error(f"  año_corte_regimen={año_corte}: rho_s_* con NaN en "
                         f"algun grupo — omitido (probable mezcla de preds_test "
                         f"de una corrida sin ESTIMAR_RHO_EN_VAL).")
            continue
        phi_por_s, R_por_s = {}, {}
        for s in range(n_estados):
            phi_por_s[s] = np.array([phi_rho[g][0][s] for g in GRUPOS], dtype=float)
            # rho_ij es SIMETRICO y se verifico identico desde las dos caras, asi
            # que se toma del primer grupo. Con N>2 habria que leer la matriz
            # completa; con N=2 un escalar la determina.
            r = phi_rho[GRUPOS[0]][1][s]
            R = np.eye(len(GRUPOS))
            R[0, 1] = R[1, 0] = r
            R_por_s[s] = R

        bloques = preparar_bloques_conjuntos(
            phi_por_s, R_por_s, etiqueta=f" año_corte={año_corte}")
        for s in range(n_estados):
            filas_diag.append({
                "año_corte_regimen": str(año_corte), "regimen": s,
                **{f"phi_{g}": float(phi_por_s[s][i]) for i, g in enumerate(GRUPOS)},
                "rho_ij": float(R_por_s[s][0, 1]),
                "lambda_min_sigma_e": bloques["lmin"][s],
                "lambda_estrella": bloques["lam"][s],
                "cota_ec7": bloques["cota"][s],
                "diag_A": float(np.diag(A)[s]) if s < len(A) else np.nan,
            })

        # ── Marginales por GRUPO: clave (grupo, fecha_t, h) ──────────────────
        distribuciones = {}
        with crono.etapa("fitear marginales"):
            for g in GRUPOS:
                logger.info(f"  año_corte_regimen={año_corte}: fiteando marginales de {g} "
                            f"({len(dfg[g]):,} pares)...")
                for (ft, h), d in fitear_distribuciones_por_horizonte(
                        dfg[g], taus=None, n_jobs=N_JOBS).items():
                    distribuciones[(g, ft, h)] = d

        # En calendario el estado inicial es irrelevante (P1 no arranca una cadena
        # de Markov), pero se deja por uniformidad de firma: secuencia_regimen lo
        # ignora cuando recibe baldes_fijos.
        if "regimen_hmm" in df_ref.columns and df_ref["regimen_hmm"].notna().any():
            estado_por_origen = (df_ref.drop_duplicates("fecha_t")
                                 .set_index("fecha_t")["regimen_hmm"]
                                 .fillna(0).astype(int))
        else:
            estado_por_origen = pd.Series(0, index=df_ref["fecha_t"].unique())
        ventanas = [v for v in VENTANAS if v >= int(dfg[GRUPOS[0]]["h"].min())]

        origenes = sorted(dfg[GRUPOS[0]]["fecha_t"].unique())
        logger.info(f"  año_corte_regimen={año_corte}: simulando {len(origenes)} "
                    f"origenes x {N_PATHS} replicas, "
                    f"diag(A)={np.diag(A).round(3).tolist()}")
        _t_sim = time.time()
        for i_o, ft in enumerate(origenes):
            df_por_grupo = {g: dfg[g][dfg[g]["fecha_t"] == ft] for g in GRUPOS}
            # En calendario, la secuencia de baldes de ESTE origen sale de la
            # columna balde_th, ordenada por h igual que los horizontes.
            _bf = None
            if _CALENDARIO:
                _s0 = df_por_grupo[GRUPOS[0]].sort_values("h")
                _bf = _s0["balde_th"].values.astype(int)
            acum = simular_paths_origen_conjunto(
                ft, df_por_grupo, distribuciones,
                estado_inicial=int(estado_por_origen.get(ft, 0)),
                matriz_transicion=A, bloques=bloques, n_paths=N_PATHS,
                ventanas=ventanas, grupos=GRUPOS, seed=SEED + _seed_offset + i_o,
                baldes_fijos=_bf)
            # y realizado del SISTEMA = suma de los grupos, por ventana
            real = {}
            for v in ventanas:
                tot = 0.0
                for g in GRUPOS:
                    d = df_por_grupo[g]
                    tot += float(d.loc[d["h"] <= v, "y_realizado"].sum()) \
                        if "y_realizado" in d.columns else np.nan
                real[v] = tot
            for v, tmap in calcular_percentiles_acumulado(acum).items():
                for tau, val in tmap.items():
                    resultados.append({"fecha_t": ft, "ventana": v, "tau": tau,
                                       "percentil_acum": val,
                                       "y_realizado_acum": real.get(v, np.nan)})
            if (i_o + 1) % 25 == 0:
                # ETA dentro del bloque: con 476 origenes el primer aviso llega a
                # los 25 y ya dice si esto son minutos u horas.
                _el = time.time() - _t_sim
                _falta = _el / (i_o + 1) * (len(origenes) - i_o - 1)
                logger.info(f"    origen {i_o+1}/{len(origenes)} — "
                            f"{Cronometro._fmt(_el)} transcurridos, faltan "
                            f"~{Cronometro._fmt(_falta)}")
        crono.acum["simular paths"] = (crono.acum.get("simular paths", 0.0)
                                       + time.time() - _t_sim)
        logger.info(f"    [t] simular paths: "
                    f"{Cronometro._fmt(time.time() - _t_sim)}")

        if GENERAR_FANCHARTS:
            # El fan chart del CONJUNTO: la banda del acumulado de FOCO+RESTO
            # con la dependencia ya incorporada. Es el unico grafico que muestra
            # lo que aporta el metodo — sin el, el modo conjunto produce un
            # parquet de numeros y nada que mirar.
            with crono.etapa("fan charts"):
                generar_fancharts_conjunto(
                    dfg, distribuciones, estado_por_origen, A, bloques, GRUPOS,
                    dir_salida=DIR_FLUJOS_ACUMULADOS, banco=BANCO,
                    n_paths=N_PATHS_FANCHART, seed=SEED + _seed_offset,
                    bandas=BANDAS_FANCHART,
                    col_balde="balde_th" if _CALENDARIO else None)

        crono.cerrar_bloque()

    if not resultados:
        logger.error("Ningun año_corte_regimen pudo simularse.")
        return

    df_sim = pd.DataFrame(resultados)
    ruta = DIR_SALIDA / f"simulacion_paths_{BANCO}.parquet"
    df_sim.to_parquet(ruta, index=False)
    logger.info(f"Simulacion conjunta: {len(df_sim):,} filas -> {ruta.name}")

    df_diag = pd.DataFrame(filas_diag)
    ruta_d = DIR_SALIDA / f"diagnostico_sigma_e_{BANCO}.csv"
    df_diag.to_csv(ruta_d, index=False)
    logger.info("\n" + df_diag.to_string(index=False))
    logger.info(f"Diagnostico Sigma_e / lambda*: {ruta_d.name}")
    crono.resumen()
    if (df_diag["lambda_estrella"] < 1.0).any():
        logger.warning(
            "  Algun regimen necesito encogimiento (8). Un lambda* bajo indica "
            "que Phi diagonal (A2) es demasiado rigida — considerar VAR(1) completo.")


def main():
    if PARTICIONES:
        return main_conjunto()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    _avisar_salida_sin_modo()

    # ── 1. Cargar predicciones TEST reales (todos los folds, sin duplicados) ──
    logger.info(f"Cargando preds_test de {BANCO} desde {DIR_MODO} ...")
    df_preds = cargar_preds_test_reales(DIR_MODO, BANCO)
    h_min, h_max = int(df_preds["h"].min()), int(df_preds["h"].max())
    logger.info(f"  {len(df_preds):,} filas | "
               f"{df_preds['fecha_t'].nunique()} orígenes | "
               f"{sorted(df_preds['fold'].unique())} folds | "
               f"h ∈ [{h_min}, {h_max}]")

    ventanas_validas = [v for v in VENTANAS if v >= h_min]
    if len(ventanas_validas) < len(VENTANAS):
        omitidas = [v for v in VENTANAS if v < h_min]
        logger.warning(f"  VENTANAS={omitidas} son menores al horizonte mínimo "
                       f"real (h_min={h_min}) — no hay datos que acumular ahí. "
                       f"Se omiten de esta corrida; ajusta VENTANAS en la config "
                       f"si quieres otro plazo corto.")
        globals()["VENTANAS"] = ventanas_validas

    if "regimen_hmm" not in df_preds.columns or df_preds["regimen_hmm"].isna().all():
        logger.error("No hay feature de régimen en los preds_test (regimen_hmm "
                     "ausente o todo NaN). Corre step005_walk_forward_cv_4.py "
                     "con USAR_FEATURE_REGIMEN=True antes de simular.")
        return

    # ── 2. ρ_s por régimen — leído de VAL (estimado sin leakage en step005) ─────
    #
    # ANTI-LEAKAGE: step005_walk_forward_cv_4.py (con ESTIMAR_RHO_EN_VAL=True)
    # clasifica el período de VALIDACION con los parámetros HMM ya fijos del
    # fold de TRAIN, calcula z_t=Phi^-1(PIT) con los cuantiles de XGBoost en
    # VAL y los realizados, y guarda rho_s_0/rho_s_1/... en los parquets de
    # preds_test — una columna por estado (2 o 3, según N_ESTADOS con el que
    # se corrió step005_validar_hmm*.py; nunca se asume un número fijo). Aquí
    # se leen esos valores por grupo (año_corte_regimen) en vez de re-estimar
    # sobre TEST donde los regímenes eran NaN/mediana imputados. Si no
    # existen las columnas (run previo sin el flag) se cae al método antiguo
    # con un aviso, para no romper compatibilidad hacia atrás.

    _n_estados_rho = _detectar_n_estados_rho(df_preds.columns)
    _tiene_rho_val = _n_estados_rho is not None

    # ── 3. Agrupar por año_corte_regimen y simular cada grupo con su propia
    #       matriz de transición y el estado inicial real de cada origen ──────
    if "año_corte_regimen" not in df_preds.columns:
        logger.error("Falta columna 'año_corte_regimen' en preds_test — "
                     "no se puede saber qué matriz de transición usar.")
        return

    grupos = df_preds.dropna(subset=["año_corte_regimen"]).groupby("año_corte_regimen")
    sin_grupo = df_preds["año_corte_regimen"].isna().sum()
    if sin_grupo > 0:
        n_fechas_sin = df_preds.loc[df_preds["año_corte_regimen"].isna(),
                                    "fecha_t"].nunique()
        logger.warning(f"  {sin_grupo:,} filas ({n_fechas_sin} orígenes) sin "
                       f"año_corte_regimen — se EXCLUYEN de la simulación "
                       f"(no hay matriz de transición asociada).")

    distribuciones_todos = {}   # acumula distribuciones de todos los grupos
    resultados_sim = []
    rho_regimen_rows = []
    
    for año_corte, df_grupo in grupos:
        # año_corte puede ser string ISO "2022-07-01" (nuevo) o int (legacy).
        # NO forzar int() — rompe con fechas ISO. Se mantiene el tipo original.
        # Para la semilla (que debe ser int), derivar un hash reproducible.
        try:
            _seed_offset = int(str(año_corte).replace("-", "")) % 100_000
        except Exception:
            _seed_offset = 0
        try:
            A = cargar_transmat(BANCO if BANCO_REGIMEN is None else BANCO_REGIMEN,
                                año_corte)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"  año_corte_regimen={año_corte}: {e} — grupo omitido "
                           f"({df_grupo['fecha_t'].nunique()} orígenes afectados).")
            continue

        # ── Guard: los rho_s_* tienen que estar indexados por los MISMOS
        # estados que simula la cadena de Markov ─────────────────────────────
        # La trayectoria de regimen se muestrea de A (NxN del HMM) y con cada
        # estado s se elige rho_por_regimen[s]. Si preds_test trae mas (o menos)
        # rho_s_* que estados tiene A, los dos indices dejan de significar lo
        # mismo y NADA truena: la recursion AR(1) recibe un float valido y
        # devuelve paths plausibles con el rho equivocado. Casos que esto ataja:
        #   - preds_test de una corrida de step005 con CONDICIONAR_POR=
        #     "calendario": trae 4 columnas rho_s_0..rho_s_3 (los baldes
        #     resto/apertura/cierre/transicion) mientras A sigue siendo la
        #     transmat 3x3 del HMM. Aplicaria phi(apertura) en los dias que el
        #     HMM llama "moderado" y nunca usaria phi(transicion).
        #
        #     ASIMETRIA A TENER PRESENTE: el modo calendario SI esta soportado en
        #     main_conjunto (secuencia_regimen con baldes_fijos), pero NO en esta
        #     ruta N=1. La razon es que aca la simulacion la hace
        #     pipeline_simulacion del modulo, que llama a simular_regimen_path
        #     internamente y no expone por donde inyectar una secuencia fija.
        #     Soportarlo exigiria tocar vf7, que es el motor vigente — trabajo
        #     aparte. Con CONDICIONAR_POR="calendario" y PARTICIONES=False la
        #     corrida aborta aca, a proposito.
        #   - mezclar preds_test de una corrida con N_ESTADOS=2 y transmat de
        #     una con N_ESTADOS=3 (o al reves).
        # Chequeo EXPLICITO primero: la columna condicionar_por dice sobre que
        # esta estratificado el subindice s. El chequeo por conteo de abajo es
        # el respaldo para parquets viejos que no la traen.
        _cond_por = (str(df_grupo["condicionar_por"].iloc[0])
                     if "condicionar_por" in df_grupo.columns else None)
        _esperado = "regimen" if CONDICIONAR_POR == "auto" else CONDICIONAR_POR
        if _cond_por is not None and _cond_por != _esperado:
            logger.error(
                f"  año_corte_regimen={año_corte}: preds_test viene de una "
                f"corrida de step005 con CONDICIONAR_POR='{_cond_por}' — los "
                f"rho_s_* estan estratificados por eso y no por el estado del "
                f"HMM que muestrea A. Grupo omitido "
                f"({df_grupo['fecha_t'].nunique()} orígenes afectados). Ese "
                f"modo aun no esta soportado aqui: falta reemplazar el muestreo "
                f"de A por el balde deterministico de t+h.")
            continue

        if _tiene_rho_val and _n_estados_rho != len(A):
            logger.error(
                f"  año_corte_regimen={año_corte}: preds_test trae "
                f"{_n_estados_rho} columnas rho_s_* pero la transmat de "
                f"{BANCO if BANCO_REGIMEN is None else BANCO_REGIMEN} tiene "
                f"{len(A)} estados — grupo omitido "
                f"({df_grupo['fecha_t'].nunique()} orígenes afectados). "
                f"Si viene de step005 con CONDICIONAR_POR='calendario', ese "
                f"modo aun no esta soportado aqui. Si no, regenera preds_test "
                f"y transmat con el mismo N_ESTADOS.")
            continue

        estado_por_origen = (
            df_grupo.drop_duplicates("fecha_t")
            .set_index("fecha_t")["regimen_hmm"]
            .astype(int)
        )

        # ── rho_s para ESTE grupo: del parquet (VAL) o fallback antiguo ────
        if _tiene_rho_val:
            _primera = df_grupo.drop_duplicates("año_corte_regimen").iloc[0]
            rho_por_regimen = {s: float(_primera[f"rho_s_{s}"]) for s in range(_n_estados_rho)}
            # Guard: la columna puede existir GLOBALMENTE (pasa _tiene_rho_val)
            # pero venir NaN para ESTE grupo específico si se mezcló un
            # preds_test_fold*.parquet viejo (guardado antes de activar
            # ESTIMAR_RHO_EN_VAL) con otros nuevos — pd.concat rellena con
            # NaN las columnas ausentes del archivo viejo. Un rho=NaN no
            # truena en la recursión AR(1) (NaN es un float válido), se
            # propaga en silencio si no se detecta aquí.
            if any(np.isnan(v) for v in rho_por_regimen.values()):
                logger.error(f"  año_corte_regimen={año_corte}: rho_s_0.."
                            f"rho_s_{_n_estados_rho - 1} contiene NaN para "
                            f"este grupo — grupo omitido "
                            f"({df_grupo['fecha_t'].nunique()} orígenes afectados). "
                            f"Probable mezcla de preds_test_fold*.parquet de una "
                            f"corrida sin ESTIMAR_RHO_EN_VAL con otros que sí lo "
                            f"tienen. Regenera todos los folds con el mismo flag.")
                continue
            logger.info(f"  año_corte_regimen={año_corte}: rho_s leido de VAL: {rho_por_regimen}")
        else:
            # Fallback: estimar sobre test (comportamiento anterior)
            _todos_str = df_preds["año_corte_regimen"].astype(str)
            if str(año_corte) == _todos_str.min():
                logger.warning("  [RHO] Columnas rho_s_* no encontradas en preds_test. "
                               "Corre step005_walk_forward_cv_4.py con ESTIMAR_RHO_EN_VAL=True.")
            h_ref = H_REFERENCIA_RHO if H_REFERENCIA_RHO is not None else int(df_preds["h"].min())
            rho_por_regimen = estimar_rho_por_regimen(
                df_grupo, h_referencia=h_ref, n_jobs=N_JOBS)

        logger.info(f"  año_corte_regimen={año_corte}: "
                   f"{df_grupo['fecha_t'].nunique()} orígenes | "
                   f"diag(A)={np.diag(A).round(3).tolist()} | "
                   f"rho_s={rho_por_regimen}")
 
        # Acumular para reporte
        diag_A = np.diag(A)
        # Las columnas se derivan de cuantos estados hay (no fijas en 0/1/2):
        # con la version fija, una corrida con mas estados perdia el ultimo rho
        # en el reporte en silencio. El guard de arriba ya garantiza que
        # rho_por_regimen y diag_A tienen el mismo largo.
        _fila_reg = {"año_corte_regimen": str(año_corte),
                     "n_origenes":        df_grupo["fecha_t"].nunique()}
        for _s in range(max(len(diag_A), len(rho_por_regimen))):
            _fila_reg[f"rho_s_{_s}"] = rho_por_regimen.get(_s, np.nan)
            _fila_reg[f"diag_A_{_s}"] = (float(diag_A[_s])
                                         if _s < len(diag_A) else np.nan)
        rho_regimen_rows.append(_fila_reg)
        
        distribuciones_grupo = fitear_distribuciones_por_horizonte(
            df_grupo, taus=None, n_jobs=N_JOBS)
        distribuciones_todos.update(distribuciones_grupo)   # acumular sin re-fitear

        df_sim_grupo = pipeline_simulacion(
            df_grupo,
            estado_inicial    = estado_por_origen,
            matriz_transicion = A,
            rho_por_regimen   = rho_por_regimen,
            ventanas          = VENTANAS,
            n_paths           = N_PATHS,
            prophet_col       = PROPHET_COL,
            seed              = SEED + _seed_offset,
            n_jobs            = N_JOBS,
            distribuciones    = distribuciones_grupo,
        )
        resultados_sim.append(df_sim_grupo)

        if GENERAR_FANCHARTS:
            generar_fancharts_todos_origenes(
                df_grupo,
                distribuciones    = distribuciones_grupo,
                estado_inicial    = estado_por_origen,
                matriz_transicion = A,
                rho_por_regimen   = rho_por_regimen,
                dir_salida        = DIR_FLUJOS_ACUMULADOS,
                banco             = BANCO,
                n_paths           = N_PATHS_FANCHART,
                prophet_col       = PROPHET_COL,
                seed              = SEED + _seed_offset,
                bandas            = BANDAS_FANCHART,
                n_jobs            = N_JOBS,
            )

        if GENERAR_FANCHARTS_NETO:
            generar_fancharts_neto_todos_origenes(
                df_grupo,
                distribuciones = distribuciones_grupo,
                dir_salida     = DIR_FLUJOS_NETOS,
                banco          = BANCO,
                bandas         = BANDAS_FANCHART,
                n_jobs         = N_JOBS,
            )

        if GENERAR_FANCHARTS_INTEGRADO:
            generar_fancharts_integrados_todos_origenes(
                df_grupo,
                distribuciones    = distribuciones_grupo,
                estado_inicial    = estado_por_origen,
                matriz_transicion = A,
                rho_por_regimen   = rho_por_regimen,
                dir_salida        = DIR_FLUJOS_INTEGRADOS,
                banco             = BANCO,
                n_paths           = N_PATHS_FANCHART,
                prophet_col       = PROPHET_COL,
                seed              = SEED + _seed_offset,
                bandas            = BANDAS_FANCHART,
                n_jobs            = N_JOBS,
            )

    if not resultados_sim:
        logger.error("Ningún grupo pudo simularse — revisa que existan los "
                     "transmat_hmm_<banco>.parquet correspondientes.")
        return

    df_sim = pd.concat(resultados_sim, ignore_index=True)
    logger.info(f"Simulación completa: {len(df_sim):,} filas "
               f"({df_sim['fecha_t'].nunique()} orígenes simulados)")

    # ── 3b. Guardar parámetros de distribuciones ──────────────────────────────
    # Se guarda un parquet ligero con los parámetros de cada distribución ajustada.
    # El diagnóstico (diagnostico_fit_skewt.py) lo carga para reconstruir los
    # objetos sin re-fitear — el costo más caro del pipeline.
    # Formato: fecha_t | h | dist_type | p1 | p2 | p3 | p4
    #   AzzaliniT → p1=xi,  p2=omega,   p3=alpha,   p4=nu
    #   SplitT    → p1=loc, p2=scale_l, p3=scale_r, p4=df
    #   PchipGPD  → p1=xi_tail, p2=beta_l, p3=beta_r, p4=nu_origen
    #               (⚠ NO permite reconstruir el objeto: el cuerpo PCHIP
    #                necesita los cuantiles completos, no caben en 4 columnas.
    #                Se guardan como metadatos de cola para diagnóstico; para
    #                re-fitear hay que releer los cuantiles del parquet de
    #                preds_test.)
    logger.info(f"Guardando parámetros de {len(distribuciones_todos):,} distribuciones...")
    dist_rows = []
    for (fecha_t, h), d in distribuciones_todos.items():
        if hasattr(d, "xi_tail"):        # PchipGPD (híbrido)
            dist_rows.append({"fecha_t": fecha_t, "h": int(h),
                              "dist_type": "PchipGPD",
                              "p1": d.xi_tail, "p2": d._beta_l,
                              "p3": d._beta_r, "p4": d.nu_origen})
        elif hasattr(d, "xi"):           # AzzaliniT
            dist_rows.append({"fecha_t": fecha_t, "h": int(h),
                              "dist_type": "AzzaliniT",
                              "p1": d.xi, "p2": d.omega, "p3": d.alpha, "p4": d.nu})
        elif hasattr(d, "loc"):          # SplitT
            dist_rows.append({"fecha_t": fecha_t, "h": int(h),
                              "dist_type": "SplitT",
                              "p1": d.loc, "p2": d.scale_l, "p3": d.scale_r, "p4": d.df})
        else:
            # Tipo desconocido: se registra sin parámetros en vez de tronar
            # al final de una corrida larga por un AttributeError.
            logger.warning(f"  Distribución de tipo inesperado en "
                          f"({fecha_t}, h={h}): {type(d).__name__} — "
                          f"guardada sin parámetros.")
            dist_rows.append({"fecha_t": fecha_t, "h": int(h),
                              "dist_type": type(d).__name__,
                              "p1": np.nan, "p2": np.nan,
                              "p3": np.nan, "p4": np.nan})
    df_dists = pd.DataFrame(dist_rows)
    df_dists["fecha_t"] = pd.to_datetime(df_dists["fecha_t"])
    ruta_dists = DIR_SALIDA / f"distribuciones_{BANCO}.parquet"
    df_dists.to_parquet(ruta_dists, index=False)
    logger.info(f"  Guardado: {ruta_dists.name}  "
                f"({df_dists['dist_type'].value_counts().to_dict()})")

    
    # ── 4. Backtest acumulado — tau principal + taus adicionales ──────────────
    logger.info(f"Backtest acumulado (3 piezas) tau={TAU_BACKTEST_ACUM}...")
    df_bt, raw_acum_principal = backtest_completo(
        df_sim, tau_col=TAU_BACKTEST_ACUM, ventanas=VENTANAS,
        return_raw=True)
 
    dfs_bt_extra        = []
    raw_acum_por_tau    = {TAU_BACKTEST_ACUM: raw_acum_principal}
 
    for tau_extra in TAUS_BACKTEST_EXTRA:
        if tau_extra != TAU_BACKTEST_ACUM:
            logger.info(f"  Backtest acumulado tau={tau_extra}...")
            df_bt_e, raw_extra = backtest_completo(
                df_sim, tau_col=tau_extra, ventanas=VENTANAS,
                return_raw=True)
            df_bt_e.insert(0, "tau", tau_extra)
            dfs_bt_extra.append(df_bt_e)
            raw_acum_por_tau[tau_extra] = raw_extra
 
    df_bt_acum_all = pd.concat(
        [df_bt.assign(tau=TAU_BACKTEST_ACUM)] + dfs_bt_extra,
        ignore_index=True)
 
    # ── 4b. Backtest flujo neto ───────────────────────────────────────────────
    logger.info(f"Backtest flujo neto (3 piezas) taus={TAUS_BACKTEST_NETO}...")
    df_bt_neto, raw_neto_por_tau = backtest_flujo_neto_completo(
        df_preds,
        distribuciones  = distribuciones_todos,
        taus_eval       = TAUS_BACKTEST_NETO,
        horizontes_eval = VENTANAS,
        return_raw      = True,
    )
 
    # ── 5. Guardar resultados principales ─────────────────────────────────────
    ruta_sim     = DIR_SALIDA / f"simulacion_paths_{BANCO}.parquet"
    ruta_bt      = DIR_SALIDA / f"backtest_acum_{BANCO}.parquet"
    ruta_bt_neto = DIR_SALIDA / f"backtest_neto_{BANCO}.parquet"
    df_sim.to_parquet(ruta_sim, index=False)

    # Los flags de las 3 piezas ahora pueden valer None ("no concluyente"),
    # además de True/False. Se normalizan a object con None — NO a dtype
    # nullable "boolean": ese dtype introduce pd.NA, y _pasa_cell() en
    # reporte_backtest.py solo contempla None y float('nan'), así que un
    # pd.NA lo hace tronar con "boolean value of NA is ambiguous" al construir
    # la tabla. object+None es además lo que p2_independencia ya venía
    # produciendo, así que el esquema del parquet no cambia de forma.
    for _df in (df_bt_acum_all, df_bt_neto):
        for _col in ("p1_pasa", "p2_independencia", "p3_uniformidad",
                     "p3_test_valido"):
            if _col in _df.columns:
                _df[_col] = (_df[_col].astype(object)
                                      .where(_df[_col].notna(), None))

    # Resumen de las 3 piezas en el log: si un test vuelve a romperse en
    # silencio, se ve en la corrida en vez de descubrirse en el PDF tres
    # semanas después. Las tres piezas pueden devolver None ("no concluyente")
    # y ese conteo es lo primero que hay que mirar antes de leer PASA/FALLA:
    #   · Pieza 1 → None cuando no hubo NINGUNA violación observada. Con cero
    #     violaciones el intervalo de Wilson colapsa a [0, z²/(n_eff+z²)], un
    #     número que no depende del plazo. No es evidencia de calibración.
    #   · Pieza 2 → None cuando hay menos de 3 violaciones: sin violaciones no
    #     hay duraciones entre ellas y la Weibull no se puede ajustar. En el
    #     acumulado esto vacía la mayoría de las celdas, y la causa está en la
    #     Pieza 3: con el PIT medio subiendo a ~0.75 el realizado casi nunca
    #     rompe el piso a plazos largos.
    #   · Pieza 3 → None cuando el submuestreo no solapado deja menos de 20
    #     PITs, o si el Anderson-Darling falla.
    _PIEZAS = (("Pieza 1", "p1_pasa"), ("Pieza 2", "p2_independencia"),
               ("Pieza 3", "p3_uniformidad"))
    for _nombre, _df in (("acumulado", df_bt_acum_all), ("neto", df_bt_neto)):
        for _etq, _col in _PIEZAS:
            if _col not in _df.columns:
                continue
            _tot   = len(_df)
            _nd    = int(_df[_col].isna().sum())
            _pasa  = int((_df[_col] == True).sum())
            _falla = _tot - _nd - _pasa
            logger.info(f"  {_etq} ({_nombre}): {_tot - _nd}/{_tot} celdas "
                        f"concluyentes | PASA={_pasa} | FALLA={_falla} | "
                        f"N/D={_nd}")
            if _nd == _tot:
                logger.warning(f"  {_etq} ({_nombre}): NINGUNA celda "
                               f"concluyente — revisar antes de reportar.")
            elif _nd > _tot // 2:
                logger.warning(f"  {_etq} ({_nombre}): {_nd} de {_tot} celdas "
                               f"sin veredicto — el resultado agregado no es "
                               f"representativo.")

    df_bt_acum_all.to_parquet(ruta_bt, index=False)
    df_bt_neto.to_parquet(ruta_bt_neto, index=False)
 
    # ── 6. Guardar datos crudos para reporte completo ─────────────────────────
    logger.info("Guardando datos crudos para reporte...")
 
    # 6a. PITs acumulados: (tau, ventana, pit)
    pits_acum_rows = []
    for tau, raw in raw_acum_por_tau.items():
        for v, arr in raw["pits"].items():
            for val in arr:
                pits_acum_rows.append({"tau": tau, "ventana": v, "pit": float(val)})
    if pits_acum_rows:
        pd.DataFrame(pits_acum_rows).to_parquet(
            DIR_SALIDA / f"pits_acum_{BANCO}.parquet", index=False)
 
    # 6b. PITs netos: (tau, h, pit)
    pits_neto_rows = []
    for tau, raw in raw_neto_por_tau.items():
        for h, arr in raw["pits"].items():
            for val in arr:
                pits_neto_rows.append({"tau": tau, "h": h, "pit": float(val)})
    if pits_neto_rows:
        pd.DataFrame(pits_neto_rows).to_parquet(
            DIR_SALIDA / f"pits_neto_{BANCO}.parquet", index=False)
 
    # 6c. Indicadores de violación: (tau, ventana, fecha_t, I_t)
    indic_rows = []
    for tau, raw in raw_acum_por_tau.items():
        for v, d in raw["indicadores"].items():
            for ft, it in zip(d["fechas_t"], d["I"]):
                indic_rows.append({
                    "tau": tau, "ventana": v,
                    "fecha_t": pd.Timestamp(ft), "I_t": float(it)})
    if indic_rows:
        pd.DataFrame(indic_rows).to_parquet(
            DIR_SALIDA / f"indicadores_acum_{BANCO}.parquet", index=False)
 
    # 6d. Duraciones entre violaciones: (tau, ventana, duracion)
    durs_rows = []
    for tau, raw in raw_acum_por_tau.items():
        for v, arr in raw["duraciones"].items():
            for val in arr:
                durs_rows.append({"tau": tau, "ventana": v, "duracion": float(val)})
    if durs_rows:
        pd.DataFrame(durs_rows).to_parquet(
            DIR_SALIDA / f"duraciones_acum_{BANCO}.parquet", index=False)
 
    # 6e. rho_s + diag(A) por grupo HMM
    if rho_regimen_rows:
        pd.DataFrame(rho_regimen_rows).to_parquet(
            DIR_SALIDA / f"rho_regimen_{BANCO}.parquet", index=False)
 
    # 6f. Config del reporte (reproducibilidad)
    import json as _json
    config_reporte = {
        "banco":               BANCO,
        "ventanas":            VENTANAS,
        "n_paths":             N_PATHS,
        "tau_backtest_acum":   TAU_BACKTEST_ACUM,
        "taus_backtest_extra": TAUS_BACKTEST_EXTRA,
        "taus_backtest_neto":  TAUS_BACKTEST_NETO,
        "seed":                SEED,
        "dir_modo":            str(DIR_MODO),
        "fecha_corrida":       pd.Timestamp.today().strftime("%Y-%m-%d %H:%M"),
    }
    with open(DIR_SALIDA / f"config_reporte_{BANCO}.json", "w",
              encoding="utf-8") as _f:
        _json.dump(config_reporte, _f, indent=2, ensure_ascii=False)
 
    logger.info(f"  Datos reporte guardados en {DIR_SALIDA}:")
    for nombre in [
        f"backtest_acum_{BANCO}.parquet",
        f"backtest_neto_{BANCO}.parquet",
        f"pits_acum_{BANCO}.parquet",
        f"pits_neto_{BANCO}.parquet",
        f"indicadores_acum_{BANCO}.parquet",
        f"duraciones_acum_{BANCO}.parquet",
        f"rho_regimen_{BANCO}.parquet",
        f"config_reporte_{BANCO}.json",
    ]:
        ruta = DIR_SALIDA / nombre
        logger.info(f"    {'✓' if ruta.exists() else '✗'} {nombre}")
        
    logger.info(f"Guardado: {ruta_sim.name}")
    logger.info(f"Guardado: {ruta_bt.name}  ({len(df_bt_acum_all)} filas — taus={df_bt_acum_all['tau'].unique().tolist()})")
    logger.info(f"Guardado: {ruta_bt_neto.name}  ({len(df_bt_neto)} filas — backtest neto)")


    print("\n" + "=" * 78)
    print("=" * 78)
    print(df_bt_acum_all.to_string(index=False))
    print("\n" + "=" * 78)
    print(f"  BACKTEST FLUJO NETO — {BANCO}  (τ={TAUS_BACKTEST_NETO})")
    print("=" * 78)
    print(df_bt_neto.to_string(index=False))
    print("=" * 78)

    return df_sim, df_bt_acum_all, df_bt_neto


if __name__ == "__main__":
    main()
