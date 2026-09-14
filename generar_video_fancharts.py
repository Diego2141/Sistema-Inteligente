# -*- coding: utf-8 -*-
"""
generar_video_fancharts.py
============================
Ensambla todos los fan charts PNG guardados por step006_orquestador.py
(carpeta flujos_acumulados) en un video que avanza día por día, en orden
cronológico de fecha de origen.

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

# Misma carpeta donde step006_orquestador.py guardó los PNGs
DIR_FLUJOS_ACUMULADOS = BASE_SISTEMA / "2. Output" / "flujos_integrados" / "xgb_qt_expanding_310.5" 

# Video de salida (en la misma carpeta, por defecto)
RUTA_VIDEO = DIR_FLUJOS_ACUMULADOS / f"video_{BANCO}.mp4"

FPS = 4   # cuadros por segundo — 4 = ~0.25s por día de origen.
          # Subir (p.ej. 8-10) parta un avance más rápido; bajar (p.ej. 1-2)
          # para poder leer cada cuadro con calma.


###############################################################################
# Ensamblado
###############################################################################

def listar_pngs_ordenados(dir_flujos: Path, banco: str) -> list[Path]:
    """
    Lista los PNG fanchart_<banco>_<YYYYMMDD>.png ordenados por fecha
    (extraída del nombre de archivo, no por orden alfabético del filesystem
    — aunque en este caso coinciden porque el formato YYYYMMDD ya ordena
    correctamente como texto).
    """
    patron = re.compile(rf"fanchart_integrado_{re.escape(banco)}_(\d{{8}})\.png")
    archivos = []
    for ruta in dir_flujos.glob(f"fanchart_integrado_{banco}_*.png"):
        m = patron.match(ruta.name)
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


def generar_video(dir_flujos: Path = DIR_FLUJOS_ACUMULADOS,
                  banco: str = BANCO,
                  ruta_salida: Path = RUTA_VIDEO,
                  fps: int = FPS) -> Path:
    _verificar_imageio_ffmpeg()

    rutas = listar_pngs_ordenados(dir_flujos, banco)
    if not rutas:
        raise FileNotFoundError(
            f"No se encontraron PNGs 'fanchart_integrado_{banco}_f1_*.png' en {dir_flujos}. "
            f"Corre primero step006_orquestador.py con GENERAR_FANCHARTS=True.")

    logger.info(f"{len(rutas)} fan charts encontrados — "
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

    logger.info(f"Video guardado: {ruta_salida}")
    return ruta_salida


if __name__ == "__main__":
    generar_video()
