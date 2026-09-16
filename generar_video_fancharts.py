# -*- coding: utf-8 -*-
"""
generar_video_fancharts.py
============================
Ensambla los fan charts PNG que guarda step006_orquestador_vf_7.py en un video
que avanza día por día, en orden cronológico de fecha de origen.

step006 produce TRES familias de fan chart, cada una en su carpeta y con su
propio prefijo de archivo:

    GENERAR_FANCHARTS            flujos_acumulados/   fanchart_<banco>_*.png
    GENERAR_FANCHARTS_NETO       flujos_netos/        fanchart_neto_<banco>_*.png
    GENERAR_FANCHARTS_INTEGRADO  flujos_integrados/   fanchart_integrado_<banco>_*.png

Este script arma un video por cada tipo que se le pida en TIPOS_FANCHART.

POR QUÉ LOS TRES SALEN DE UNA SOLA ENTRADA DE CONFIGURACIÓN
La versión anterior tenía las tres cosas escritas por separado: una constante
DIR_FLUJOS_ACUMULADOS que en realidad apuntaba a flujos_integrados, un patrón
de regex con "fanchart_integrado_" hardcodeado, y un RUTA_VIDEO que no decía de
qué tipo era. Cambiar de tipo exigía editar tres lugares y era fácil editar uno
solo — el resultado era un FileNotFoundError cuyo mensaje, además, nombraba un
patrón 'fanchart_integrado_<banco>_f1_*.png' que no existía en ningún glob y
mandaba a activar GENERAR_FANCHARTS cuando los PNG que buscaba los produce
GENERAR_FANCHARTS_INTEGRADO. Ahora la carpeta, el prefijo del archivo y el
nombre del video se derivan de la misma entrada del dict _CFG, así que no
pueden desincronizarse.

Requiere: pip install imageio imageio-ffmpeg

Uso:
    python generar_video_fancharts.py
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

# Import compatible con imageio viejo y nuevo.
#
# `imageio.v2` recién existe en versiones relativamente nuevas. Anaconda suele
# traer un imageio antiguo arrastrado por scikit-image, y ahí
# `import imageio.v2 as imageio` falla con
#     ModuleNotFoundError: No module named 'imageio.v2'
# que es fácil de confundir con "imageio no está instalado" cuando en realidad
# está: lo que falta es el submódulo.
#
# La API de nivel superior (`imread` / `imwrite` / `get_writer`) es exactamente
# la que `v2` expone, así que el fallback no cambia el comportamiento — solo
# emite un DeprecationWarning en las versiones nuevas. Verificado: las dos rutas
# producen el mismo mp4 con format="FFMPEG" + codec="libx264".
#
# Se prefiere `v2` cuando está disponible para no depender de un default que
# ImageIO v3 va a cambiar.
try:
    import imageio.v2 as imageio            # imageio >= ~2.10
except ModuleNotFoundError:
    import imageio                          # imageio viejo: misma API, sin v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


###############################################################################
# Configuración
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")

# ── Los MISMOS botones que step006_orquestador_vf_7.py ──────────────────────
# De acá sale BANCO y el subnivel de carpeta, en vez de escribirlos a mano.
# Tienen que coincidir con los de la corrida de step006 que dejó los PNG.
#
# Antes BANCO era un literal y la ruta no sabía del subnivel _SUF_SALIDA que
# step006 agrega con PARTICIONES=True o con una ENTIDAD de partición, así que
# para el conjunto había que escribir a mano
#     ETIQUETA_CORRIDA = "xgb_qt_expanding_310.5/CONJUNTO_BBVA_1_0.5"
# metiendo una barra adentro de lo que debería ser un solo nivel. Funcionaba
# por cómo pathlib parsea la cadena, pero era un remiendo.
PARTICIONES = False
PARTICION   = "globales"   # "bbva" | "globales"
ENTIDAD     = "SISTEMA"    # "SISTEMA" | "FOCO" | "RESTO" — solo con PARTICIONES=False

# Geometría del fold, para reconstruir etiqueta_corrida() igual que step005/006.
VENTANA_VAL_AÑOS  = 1
VENTANA_TEST_AÑOS = 0.5

# Debe coincidir con el subdirectorio que usa step006_orquestador_vf_7.py al
# guardar los PNG (ahí se construye como f"{MODELO_CV}_{modo}_{ventanas}").
ETIQUETA_CORRIDA = "xgb_qt_expanding_310.5"


def _fmt_anios(x: float) -> str:
    """0.5 -> '0.5', 1.0 -> '1'. Misma regla que step005 y step006."""
    return f"{x:g}"


def etiqueta_corrida(banco: str) -> str:
    """Identidad de la corrida: entidad + geometría. Ej: CONJUNTO_BBVA_1_0.5"""
    return f"{banco}_{_fmt_anios(VENTANA_VAL_AÑOS)}_{_fmt_anios(VENTANA_TEST_AÑOS)}"


if not PARTICIONES:
    if ENTIDAD == "SISTEMA":
        BANCO = "SISTEMA"
    elif ENTIDAD in ("FOCO", "RESTO"):
        BANCO = f"{ENTIDAD}_{PARTICION.upper()}"
    else:
        raise ValueError(f"ENTIDAD={ENTIDAD!r} no es valida. Opciones: "
                         f"'SISTEMA', 'FOCO', 'RESTO'.")
else:
    # El conjunto: step006 nombra el agregado así (main_conjunto).
    BANCO = f"CONJUNTO_{PARTICION.upper()}"

# Sobre que estaba condicionada la corrida de step006 que dejo estos PNG.
# "auto" = no se separaron por modo (step006 avisa cuando eso pasa).
CONDICIONAR_POR = "auto"      # "auto" | "regimen" | "calendario"

if CONDICIONAR_POR not in ("auto", "regimen", "calendario"):
    raise ValueError(f"CONDICIONAR_POR={CONDICIONAR_POR!r} invalido — debe ser "
                     f"'auto', 'regimen' o 'calendario'.")

# Mismo predicado que _SUF_SALIDA de step006, porque es la MISMA decisión:
# dónde escribió esos PNG. Si los dos divergen, este script busca donde no es.
_SUF = "" if (not PARTICIONES and ENTIDAD == "SISTEMA") else etiqueta_corrida(BANCO)
if CONDICIONAR_POR == "calendario":
    _SUF = f"{_SUF}/cond_calendario" if _SUF else "cond_calendario"

# Con PARTICIONES=True solo existe la familia "acumulado": main_conjunto genera
# únicamente ese fan chart (el neto y el integrado leen percentiles de la
# marginal de UNA entidad, y la del agregado no existe en forma cerrada). Pedir
# los otros dos no rompe nada —se omiten con un aviso— pero conviene saberlo.

# Qué videos armar. Lista, no un valor único: step006 genera las TRES familias de
# PNG en una sola corrida, así que lo natural es armar los tres videos también.
# Se puede reducir a ["integrado"] para reproducir el comportamiento anterior.
TIPOS_FANCHART = ["acumulado", "neto", "integrado"]

# tipo -> (carpeta de step006, prefijo del nombre de archivo).
# Única fuente de esos dos datos: de acá salen la ruta, el patrón del glob y el
# nombre del video, así que no pueden quedar apuntando a cosas distintas.
# Los prefijos son los que step006_simulacion_paths_vf7.py usa al hacer savefig:
#   fanchart_{banco}_{YYYYMMDD}.png            (acumulado simulado)
#   fanchart_neto_{banco}_{YYYYMMDD}.png       (flujo neto diario, sin acumular)
#   fanchart_integrado_{banco}_{YYYYMMDD}.png  (3 filas: neto crudo/dist/acum)
_CFG = {
    "acumulado": ("flujos_acumulados", "fanchart",           "GENERAR_FANCHARTS"),
    "neto":      ("flujos_netos",      "fanchart_neto",      "GENERAR_FANCHARTS_NETO"),
    "integrado": ("flujos_integrados", "fanchart_integrado", "GENERAR_FANCHARTS_INTEGRADO"),
}

_tipos_malos = [t for t in TIPOS_FANCHART if t not in _CFG]
if _tipos_malos:
    raise ValueError(f"TIPOS_FANCHART contiene {_tipos_malos}, que no existen. "
                     f"Opciones: {sorted(_CFG)}")

FPS = 4   # cuadros por segundo — 4 = ~0.25s por día de origen.
          # Subir (p.ej. 8-10) parta un avance más rápido; bajar (p.ej. 1-2)
          # para poder leer cada cuadro con calma.


def dir_frames_de(tipo: str) -> Path:
    """
    Carpeta donde step006 dejó los PNG de ese tipo de fan chart.

    Se RESUELVE mirando el disco, igual que dir_modo_de() del orquestador y por
    el mismo motivo: el subnivel lo decidió la corrida que escribió los PNG, y
    esta config puede no reflejarla (frames viejos de antes del botón, o una
    corrida hecha con otros flags). Prioridad:
      1. el subnivel que corresponde a esta config, si tiene frames de ESE banco;
      2. la carpeta base, si los tiene — cubre los PNG anteriores al botón;
      3. si ninguna, el subnivel igual, para que el FileNotFoundError nombre la
         carpeta donde deberían estar.
    """
    base = BASE_SISTEMA / "2. Output" / _CFG[tipo][0] / ETIQUETA_CORRIDA
    sub  = base / _SUF if _SUF else base
    pat  = f"{_CFG[tipo][1]}_{BANCO}_*.png"
    try:
        if sub.is_dir() and any(sub.glob(pat)):
            return sub
        if sub != base and base.is_dir() and any(base.glob(pat)):
            logger.info(f"  [{tipo}] frames encontrados en la carpeta base, no en "
                        f"{_SUF!r} — se usa {base}")
            return base
    except OSError as e:
        logger.debug(f"  No se pudo inspeccionar {sub} ({type(e).__name__}: {e})")
    return sub


def ruta_video_de(tipo: str, banco: str = BANCO) -> Path:
    """
    Video de salida, junto a sus frames. El tipo va EN EL NOMBRE: antes los tres
    habrían quedado como video_SISTEMA.mp4 y el último habría pisado a los otros
    dos si alguna vez compartían carpeta.
    """
    return dir_frames_de(tipo) / f"video_{tipo}_{banco}.mp4"


###############################################################################
# Ensamblado
###############################################################################

def listar_pngs_ordenados(dir_flujos: Path, banco: str,
                          prefijo: str = "fanchart_integrado") -> list[Path]:
    """
    Lista los PNG <prefijo>_<banco>_<YYYYMMDD>.png ordenados por fecha, extraída
    del nombre de archivo y no del orden alfabético del filesystem — aunque en
    este caso coinciden, porque YYYYMMDD ya ordena correctamente como texto.

    El regex hace match ANCLADO (re.match sobre el nombre completo + \\.png), así
    que el prefijo "fanchart" del acumulado no captura por accidente los
    "fanchart_neto_..." ni los "fanchart_integrado_...": después de "fanchart_"
    el regex exige el nombre del banco, no otra palabra. De todos modos las tres
    familias viven en carpetas distintas, así que esto es cinturón y tirantes.
    """
    patron = re.compile(rf"{re.escape(prefijo)}_{re.escape(banco)}_(\d{{8}})\.png")
    archivos = []
    for ruta in dir_flujos.glob(f"{prefijo}_{banco}_*.png"):
        m = patron.fullmatch(ruta.name)
        if m:
            archivos.append((m.group(1), ruta))
    archivos.sort(key=lambda t: t[0])   # YYYYMMDD ordena correctamente como string
    return [ruta for _, ruta in archivos]


def _verificar_imageio_ffmpeg() -> None:
    """
    Verifica que imageio-ffmpeg esté instalado ANTES de intentar escribir el
    video. Sin esto, imageio puede caer silenciosamente en otro plugin
    (p.ej. TiffWriter) que no entiende kwargs de video como 'fps', y el error
    resultante (TypeError dentro de un plugin interno) no dice nada sobre la
    causa real. Con esta verificación, el mensaje de error es directo.
    """
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        raise ImportError(
            "Falta el paquete 'imageio-ffmpeg' (o no se está detectando "
            "correctamente). Sin él, imageio no puede escribir video y cae "
            "en otro plugin incompatible (ese es el origen del error "
            "'TiffWriter.write() got an unexpected keyword argument fps' "
            "si ya lo viste). Instalar con:\n\n"
            "    pip install imageio-ffmpeg\n\n"
            "y volver a correr este script."
        )
    # La version de imageio no se exige, porque el import del tope ya cae a la
    # API legacy si falta el submodulo v2. Pero se loguea: si algo raro pasa con
    # el writer, el numero de version es el primer dato que uno quiere ver, y en
    # Anaconda es comun tener un imageio viejo arrastrado por scikit-image.
    _v = getattr(imageio, "__version__", None)
    if _v is None:
        import imageio as _base
        _v = getattr(_base, "__version__", "desconocida")
    logger.info(f"imageio {_v} | imageio-ffmpeg {imageio_ffmpeg.__version__} | "
                f"API {'v2' if hasattr(imageio, 'imread') and imageio.__name__.endswith('v2') else 'legacy'}")


def generar_video(tipo: str = "integrado",
                  banco: str = BANCO,
                  dir_flujos: Path | None = None,
                  ruta_salida: Path | None = None,
                  fps: int = FPS) -> Path:
    """
    Arma el video de UN tipo de fan chart. dir_flujos y ruta_salida se derivan
    de `tipo` si no se pasan: así el llamador normal no puede combinar una
    carpeta con el patrón de otro tipo.
    """
    if tipo not in _CFG:
        raise ValueError(f"tipo={tipo!r} invalido. Opciones: {sorted(_CFG)}")
    _carpeta, _prefijo, _flag = _CFG[tipo]
    dir_flujos  = dir_frames_de(tipo) if dir_flujos is None else Path(dir_flujos)
    ruta_salida = ruta_video_de(tipo, banco) if ruta_salida is None else Path(ruta_salida)

    _verificar_imageio_ffmpeg()

    rutas = listar_pngs_ordenados(dir_flujos, banco, prefijo=_prefijo)
    if not rutas:
        # El mensaje nombra el patrón REAL que se buscó y el flag REAL que
        # produce esos PNG — antes mencionaba un 'fanchart_integrado_..._f1_*'
        # inexistente y mandaba a activar GENERAR_FANCHARTS incluso cuando los
        # frames los genera GENERAR_FANCHARTS_INTEGRADO.
        raise FileNotFoundError(
            f"No se encontraron PNGs '{_prefijo}_{banco}_<YYYYMMDD>.png' en "
            f"{dir_flujos}. Corre primero step006_orquestador_vf_7.py con "
            f"{_flag}=True. Si la carpeta existe pero está vacía, revisá que "
            f"ETIQUETA_CORRIDA={ETIQUETA_CORRIDA!r} y BANCO={banco!r} coincidan "
            f"con esa corrida.")

    logger.info(f"[{tipo}] {len(rutas)} fan charts encontrados — "
               f"desde {rutas[0].name} hasta {rutas[-1].name}")
    logger.info(f"Ensamblando video a {fps} fps "
               f"(~{len(rutas)/fps:.1f}s de duración)...")

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    # format="FFMPEG" EXPLÍCITO: sin esto, imageio intenta adivinar el plugin
    # por la extensión del archivo, y si el plugin de ffmpeg no se detecta
    # bien en el entorno, puede caer en otro plugin incompatible sin avisar
    # claramente — ese fue el origen real del TypeError de TiffWriter.
    try:
        with imageio.get_writer(str(ruta_salida), format="FFMPEG", fps=fps,
                                codec="libx264", quality=8) as writer:
            for i, ruta in enumerate(rutas):
                frame = imageio.imread(ruta)
                writer.append_data(frame)
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i+1}/{len(rutas)} cuadros añadidos")
    except Exception as e:
        logger.error(
            f"Fallo escribiendo el video con el plugin FFMPEG ({type(e).__name__}: {e}). "
            f"Si el error menciona 'ffmpeg' no encontrado, confirma que "
            f"imageio-ffmpeg esté instalado en ESTE mismo entorno de Python "
            f"(no en otro, si tienes varios — Anaconda vs. el Python suelto)."
        )
        raise

    logger.info(f"[{tipo}] Video guardado: {ruta_salida}")
    return ruta_salida


def generar_videos(tipos: list | None = None, banco: str = BANCO,
                   fps: int = FPS) -> dict:
    """
    Arma un video por cada tipo pedido y devuelve {tipo: ruta | None}.

    Un tipo sin frames NO aborta la corrida: se avisa y se sigue con los otros.
    Antes, un FileNotFoundError en el primer tipo mataba todo, de modo que con
    los tres flags activos en step006 pero uno de los tres sin generar (o con
    otro BANCO), se perdían los dos videos que sí se podían armar.
    """
    tipos = list(TIPOS_FANCHART if tipos is None else tipos)
    _verificar_imageio_ffmpeg()      # una vez, no una por tipo
    out = {}
    for tipo in tipos:
        try:
            out[tipo] = generar_video(tipo=tipo, banco=banco, fps=fps)
        except FileNotFoundError as e:
            logger.warning(f"[{tipo}] sin frames — se omite. {e}")
            out[tipo] = None

    hechos = [t for t, r in out.items() if r is not None]
    faltan = [t for t, r in out.items() if r is None]
    logger.info(f"Videos generados ({len(hechos)}/{len(tipos)}): {hechos or '—'}")
    if faltan:
        logger.warning(f"Sin generar: {faltan}")
    return out


if __name__ == "__main__":
    generar_videos()
