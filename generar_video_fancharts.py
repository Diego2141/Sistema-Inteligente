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

import imageio.v2 as imageio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


###############################################################################
# Configuración
###############################################################################

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
BANCO        = "SISTEMA"

# Debe coincidir con el subdirectorio que usa step006_orquestador_vf_7.py al
# guardar los PNG (ahí se construye como f"{MODELO_CV}_{modo}_{ventanas}").
ETIQUETA_CORRIDA = "xgb_qt_expanding_310.5"

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
    """Carpeta donde step006 dejó los PNG de ese tipo de fan chart."""
    return BASE_SISTEMA / "2. Output" / _CFG[tipo][0] / ETIQUETA_CORRIDA


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
