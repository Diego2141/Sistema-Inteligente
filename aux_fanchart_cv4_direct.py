# -*- coding: utf-8 -*-
"""
aux_fanchart_cv4_direct.py
==========================
Fan charts para el modelo cv4 DIRECT (step005_walk_forward_cv_4.py).

Genera por cada fecha de origen seleccionada un PNG con 3 subplots:
  1. Flujo diario D-R: bandas + realizado
  2. Flujo diario D-R: solo modelo (sin realizado)
  3. Flujo neto acumulado D-R: bandas + realizado acumulado

Diferencia clave vs cv3: los parquets están en step005_wfcv_v4_direct/fold_exp/ o fold_roll/
según el flag EXPANDING (True → expanding/fold_exp/fan_exp; False → rolling/fold_roll/fan_roll).
y el nombre del banco en el filename es SISTEMA (no BancaLocal).

Dos botones independientes controlan la salida:

    FOTOS_DIARIAS = True/False   un PNG por fecha de origen
    VIDEO_DIARIO  = True/False   la secuencia completa como video

El video va en orden cronológico y lleva tipo + instante de creación, de modo
que corridas sucesivas nunca se pisan entre sí:

    video_fanchart_cv4_<TIPO>_<BANCO>_<YYYYmmdd_HHMMSS>.<mp4|gif>

El TIPO es _MODO_SUFIJO por defecto (exp / roll / roll_arctan); se puede fijar
a mano con VIDEO_TIPO. Backend: mp4 si hay imageio-ffmpeg o binario ffmpeg
(ver RUTA_FFMPEG, el mismo build portable de aux_fanchart_video_xgb_qt.py);
si no, GIF con Pillow como último recurso. Con SOLO_VIDEO = True se reencodea
a partir de los PNG que ya están en disco, sin volver a graficar.
"""

from __future__ import annotations

import os

# ffmpeg portable (mismo build que usa aux_fanchart_video_xgb_qt.py). Se antepone
# al PATH para que shutil.which lo encuentre sin instalarlo en el sistema.
RUTA_FFMPEG = (r"H:\DPINV\CARPETAS PERSONALES\DIEGO\2. Python\Paquetes python"
               r"\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin")
if RUTA_FFMPEG and os.path.isdir(RUTA_FFMPEG):
    os.environ["PATH"] = RUTA_FFMPEG + os.pathsep + os.environ.get("PATH", "")

from datetime import datetime
from pathlib import Path
import gc
import logging
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib
# Agg ANTES de importar pyplot: corre desde Spyder/runfile, cuyo backend por
# defecto (Qt/Tk) registra cada figura en el panel de Plots aunque el script
# la cierre con plt.close(fig) — esa copia extra es lo que va acumulando
# memoria a lo largo de cientos de PNG hasta que RendererAgg no consigue el
# buffer contiguo que necesita ("Out of memory"). Agg es puramente headless:
# no hay panel que capture nada, savefig() es la única salida.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, GoodFriday,
    USFederalHolidayCalendar, Easter,
)
from pandas.tseries.offsets import CustomBusinessDay, Day as _Day

# Silencia loggers internos de matplotlib que heredan el nivel DEBUG del root logger
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def _build_bday() -> CustomBusinessDay:
    """Calendario hábil PE+USA idéntico al de step001 (AbstractHolidayCalendar + Easter)."""
    _f0, _f1 = "2009-01-01", "2042-12-31"

    class _PeruCal(AbstractHolidayCalendar):
        rules = [
            Holiday("AnioNuevo",   month=1,  day=1),
            Holiday("JuevesSanto", month=1,  day=1, offset=[Easter(), _Day(-3)]),
            GoodFriday,
            Holiday("Trabajo",     month=5,  day=1),
            Holiday("SanPedro",    month=6,  day=29),
            Holiday("FiestasP1",   month=7,  day=28),
            Holiday("FiestasP2",   month=7,  day=29),
            Holiday("SantaRosa",   month=8,  day=30),
            Holiday("Angamos",     month=10, day=8),
            Holiday("TodosSantos", month=11, day=1),
            Holiday("Inmaculada",  month=12, day=8),
            Holiday("Nochebuena",  month=12, day=24),
            Holiday("Navidad",     month=12, day=25),
        ]

    hols = set(_PeruCal().holidays(_f0, _f1).normalize())
    hols |= set(USFederalHolidayCalendar().holidays(_f0, _f1).normalize())
    return CustomBusinessDay(holidays=sorted(hols))

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")

# "exp" → fold_exp / fan_exp  |  "roll" → fold_roll / fan_roll  (debe coincidir con EXPANDING de step005)
_MODO_SUFIJO = "exp_arctan"

DIR_PREDS    = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct" / f"fold_{_MODO_SUFIJO}"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_fanchart_cv4_direct" / f"fan_{_MODO_SUFIJO}"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"

BANCO = "SISTEMA"

# ── Parámetros ─────────────────────────────────────────────────────────────────
PASO_FECHAS  = 1     # cada N días hábiles tomar una fecha de origen
N_FECHAS_MAX = None  # None = todos; ej: 10 para limitar cantidad de gráficos
COLOR_BANDA  = "steelblue"   # distinto al rojo de cv3 para distinguir visualmente

EXPORTAR_EXCEL = True
RUTA_EXCEL = DIR_OUTPUT / "fanchart_cv4_datos.xlsx"

# ══ BOTONES ═══════════════════════════════════════════════════════════════════
# Los dos son independientes:
#   FOTOS_DIARIAS=True,  VIDEO_DIARIO=False -> solo los PNG (comportamiento previo)
#   FOTOS_DIARIAS=False, VIDEO_DIARIO=True  -> solo el video; los PNG se generan en
#                                              una carpeta temporal y se borran
#   ambos True                              -> PNG en disco + video
#   ambos False                             -> no se grafica nada (solo Excel)
FOTOS_DIARIAS = True
VIDEO_DIARIO  = True

# True -> NO regrafica; arma el video con los PNG que ya están en DIR_OUTPUT.
# Útil para reencodear (otro fps, otro ancho) sin repetir 600 figuras.
SOLO_VIDEO = False
# ══════════════════════════════════════════════════════════════════════════════

# ── Video ─────────────────────────────────────────────────────────────────────
# Los PNG conservan su nombre de siempre (se sobrescriben entre corridas). El
# VIDEO en cambio lleva tipo + timestamp, así que cada corrida deja su propio
# archivo y nunca se pisa con el anterior:
#
#     video_fanchart_cv4_<TIPO>_<BANCO>_<YYYYmmdd_HHMMSS>.<mp4|gif>
#
VIDEO_TIPO      = None    # None -> usa _MODO_SUFIJO ("exp", "roll", "roll_arctan"…)
VIDEO_FPS       = 6       # frames por segundo (6 -> ~0.17 s por fecha de origen)
VIDEO_ANCHO_MAX = 1280    # los PNG son ~2400px; reducir es lo que hace el video manejable
VIDEO_FORMATO   = "auto"  # "auto" (mp4 si hay encoder, si no gif) | "mp4" | "gif"
VIDEO_COLORES   = 128     # tamaño de paleta del GIF (256 max; menos = archivo más chico)
VIDEO_POR_FOLD  = False   # True -> además del video global, uno por fold

# ── Ejes Y fijos ──────────────────────────────────────────────────────────────
# Con escala automática cada frame reescala y el video se vuelve ilegible: las
# bandas parecen del mismo ancho siempre porque el eje se ajusta a ellas. Con
# límites fijos las fechas son comparables entre sí. Se calcula un rango por
# panel (el panel 2 es mucho más estrecho que el 1, que incluye el realizado).
YLIM_FIJO    = True
YLIM_DIARIO  = None   # None -> automático global; o fijar a mano, ej. (-3000, 3000)


# ── 1. Cargar parquet ─────────────────────────────────────────────────────────
def cargar_parquet(banco: str) -> pd.DataFrame:
    candidatos = sorted(DIR_PREDS.glob(f"preds_base_{banco}_*.parquet"))
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró preds_base_{banco}_*.parquet en {DIR_PREDS}\n"
            f"  -> Corre step005_walk_forward_cv_4.py primero"
        )
    ruta = candidatos[-1]
    df = pd.read_parquet(ruta)
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    if "fecha_th" in df.columns:
        df["fecha_th"] = pd.to_datetime(df["fecha_th"])
    print(f"[OK] Parquet cargado: {ruta.name}  ({len(df):,} filas)")
    print(f"     Folds: {sorted(df['fold'].unique())}  |  h: {df['h'].min()}–{df['h'].max()}")
    print(f"     fecha_t: {df['fecha_t'].min().date()} → {df['fecha_t'].max().date()}")
    return df


# ── 2. Enriquecer fecha_th y target desde la matriz ───────────────────────────
def _enriquecer_desde_matriz(df: pd.DataFrame, banco: str) -> pd.DataFrame:
    """
    Reemplaza fecha_th y target desde la matriz de step001 para garantizar
    coherencia (mismo calendario que el que generó los features).
    """
    try:
        cols = ["fecha_t", "h", "target"]
        mat = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)],
                              columns=cols + (["fecha_th"] if True else []))
    except Exception:
        try:
            mat = pd.read_parquet(RUTA_MATRIZ, columns=cols)
            mat = mat[mat["banco"] == banco] if "banco" in mat.columns else mat
        except Exception as e:
            print(f"[AVISO] No se pudo cargar matriz: {e}")
            return df

    mat["fecha_t"] = pd.to_datetime(mat["fecha_t"])

    # Derivar fecha_th desde secuencia de días hábiles si no está en la matriz
    if "fecha_th" not in mat.columns:
        dias = np.sort(mat["fecha_t"].unique())
        n = len(dias)
        pos_dict = {pd.Timestamp(d): i for i, d in enumerate(dias)}
        ft_idx = mat["fecha_t"].map(pos_dict)
        th_idx = ft_idx + mat["h"]
        valido = ft_idx.notna() & (th_idx < n)
        mat["fecha_th"] = pd.NaT
        mat.loc[valido, "fecha_th"] = pd.DatetimeIndex(
            dias[th_idx[valido].astype(int).values]
        )
    else:
        mat["fecha_th"] = pd.to_datetime(mat["fecha_th"])

    # Fallback con calendario PE+USA para filas que quedaron NaT
    # (ocurre cuando fecha_t está cerca del final del histórico y h es grande,
    #  por lo que th_idx supera el rango de la matriz)
    nat_mask = mat["fecha_th"].isna()
    if nat_mask.any():
        bday = _build_bday()
        fechas_nat = pd.DatetimeIndex(mat.loc[nat_mask, "fecha_t"].unique())
        records = []
        for ft in fechas_nat:
            h_max_local = int(mat.loc[nat_mask & (mat["fecha_t"] == ft), "h"].max())
            proj = pd.date_range(start=ft + bday, periods=h_max_local, freq=bday)
            for h_idx, fth in enumerate(proj, start=1):
                records.append((ft, h_idx, fth))
        th_fill = (
            pd.DataFrame(records, columns=["fecha_t", "h", "fecha_th"])
            .astype({"h": int})
        )
        mat = mat.drop(columns=["fecha_th"]).merge(th_fill, on=["fecha_t", "h"], how="left")
        mat["fecha_th"] = pd.to_datetime(mat["fecha_th"])
        print(f"[OK] {nat_mask.sum():,} fecha_th completadas con calendario PE+USA")

    mapping = mat[["fecha_t", "h", "fecha_th", "target"]].drop_duplicates(
        subset=["fecha_t", "h"]
    )

    # Drop columnas que vienen del parquet y reemplazar desde matriz
    for col in ["fecha_th", "target"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(mapping, on=["fecha_t", "h"], how="left")
    print(f"[OK] fecha_th y target reemplazados desde matriz "
          f"({df['fecha_th'].notna().sum():,}/{len(df):,} filas con fecha_th)")
    return df


# ── 3. Fechas válidas ─────────────────────────────────────────────────────────
def fechas_validas(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df["fecha_t"].unique())


# ── 4. Preparar resultado para una fecha ─────────────────────────────────────
def preparar_resultado(df: pd.DataFrame, fecha_origen: pd.Timestamp) -> pd.DataFrame:
    res = df[df["fecha_t"] == fecha_origen].sort_values("h").reset_index(drop=True)
    if res.empty:
        raise ValueError(f"Sin datos para fecha_t={fecha_origen.date()}")
    return res


# ── 5. Graficar ───────────────────────────────────────────────────────────────
def _dibujar_bandas(ax, hs, res):
    ax.fill_between(hs, res["q01"] / 1e6, res["q99"] / 1e6,
                    alpha=0.10, color=COLOR_BANDA, label="Q01–Q99 (98%)")
    ax.fill_between(hs, res["q05"] / 1e6, res["q95"] / 1e6,
                    alpha=0.22, color=COLOR_BANDA, label="Q05–Q95 (90%)")
    if "q40" in res.columns and "q60" in res.columns:
        ax.fill_between(hs, res["q40"] / 1e6, res["q60"] / 1e6,
                        alpha=0.42, color=COLOR_BANDA, label="Q40–Q60 (20%)")
    ax.plot(hs, res["q05"] / 1e6, color=COLOR_BANDA, lw=1.0, ls=":", alpha=0.7)
    ax.plot(hs, res["q95"] / 1e6, color=COLOR_BANDA, lw=1.0, ls=":", alpha=0.7)
    ax.plot(hs, res["q50"] / 1e6, color="steelblue", lw=2.0,
            label="Mediana predicha (Q50)", zorder=5)
    if "mean" in res.columns:
        ax.plot(hs, res["mean"] / 1e6, color="navy", lw=1.5, ls="--",
                alpha=0.7, label="Media predicha", zorder=4)
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_xlim(hs.min() - 1, hs.max() + 1)


def rangos_globales(df: pd.DataFrame):
    """
    Límites Y fijos para los 3 paneles, calculados sobre TODAS las fechas.

    Devuelve (ylim1, ylim2, ylim3) o None si faltan columnas. Se calcula un
    rango por panel a propósito: el panel 1 incluye el realizado (picos de
    ±4000 MM) y el panel 2 solo las bandas del modelo (±1500 MM); usar el mismo
    rango en ambos aplastaría el panel 2 hasta volverlo inútil.
    """
    def _pad(lo, hi, f=0.05):
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        p = (hi - lo) * f
        return (lo - p, hi + p)

    if not {"q01", "q99"}.issubset(df.columns):
        return None

    d = df.sort_values(["fecha_t", "h"])

    # Panel 1: bandas + realizado   |   Panel 2: solo el modelo
    lo_m, hi_m = float(d["q01"].min()), float(d["q99"].max())
    if "target" in d.columns and d["target"].notna().any():
        lo1 = min(lo_m, float(d["target"].min()))
        hi1 = max(hi_m, float(d["target"].max()))
    else:
        lo1, hi1 = lo_m, hi_m
    ylim1 = YLIM_DIARIO if YLIM_DIARIO else _pad(lo1 / 1e6, hi1 / 1e6)
    ylim2 = _pad(lo_m / 1e6, hi_m / 1e6)

    # Panel 3: acumulado por fecha de origen (banda Q40–Q60, mediana, realizado)
    series = []
    for c in ("q40", "q60", "q50"):
        if c in d.columns:
            series.append(d.groupby("fecha_t")[c].cumsum())
    if "target" in d.columns:
        series.append(d.assign(_t=d["target"].fillna(0.0))
                       .groupby("fecha_t")["_t"].cumsum())
    if series:
        acum = pd.concat(series)
        ylim3 = _pad(float(acum.min()) / 1e6, float(acum.max()) / 1e6)
    else:
        ylim3 = None

    return ylim1, ylim2, ylim3


def graficar(res: pd.DataFrame, fecha_origen: pd.Timestamp, banco: str,
             idx: int, total: int, fold_num: int,
             dir_out: Path = None, ylims=None) -> Path:
    hs        = res["h"].values
    realizado = res["target"].values / 1e6
    mask_real = ~np.isnan(realizado)
    h_max_real = int(res.loc[mask_real, "h"].max()) if mask_real.any() else 0

    cum_q01  = np.cumsum(res["q01"].values / 1e6)
    cum_q05  = np.cumsum(res["q05"].values / 1e6)
    cum_q40  = np.cumsum(res["q40"].values / 1e6) if "q40" in res.columns else None
    cum_q50  = np.cumsum(res["q50"].values / 1e6)
    cum_q60  = np.cumsum(res["q60"].values / 1e6) if "q60" in res.columns else None
    cum_q95  = np.cumsum(res["q95"].values / 1e6)
    cum_q99  = np.cumsum(res["q99"].values / 1e6)
    cum_real = np.where(mask_real, np.nancumsum(np.where(mask_real, realizado, 0)), np.nan)
    cum_real[~mask_real] = np.nan

    titulo = (
        f"Fan Chart cv4 DIRECT — {banco}  |  Fold {fold_num}  |  "
        f"Origen: {fecha_origen.strftime('%d %b %Y')}  |  h = 2…{int(hs.max())} dh\n"
        f"Realizado: h ≤ {h_max_real}  |  "
        f"Proyección pura: h > {h_max_real}  |  [1 modelo por h, h ∉ features]"
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16), sharex=True,
                                         gridspec_kw={"hspace": 0.08})
    # try/finally: si algo revienta a mitad de dibujo (o savefig() se queda
    # sin memoria para el buffer de RendererAgg), fig NO debe quedar sin
    # cerrar. Antes, la excepción se escapaba directo al try/except del
    # bucle principal y saltaba el plt.close(fig) de más abajo — esa fig
    # quedaba viva en el registro interno de pyplot para siempre, y cada
    # fallo dejaba más memoria retenida que hacía más probable el próximo
    # fallo (el patrón de rachas de "omitida" que se ve en el log).
    try:
        # Panel 1: bandas + realizado
        _dibujar_bandas(ax1, hs, res)
        if mask_real.any():
            ax1.plot(hs[mask_real], realizado[mask_real],
                     color="black", lw=2, label="Realizado (D−R)", zorder=6)
            ax1.scatter(hs[mask_real], realizado[mask_real], color="black", s=18, zorder=7)
        if h_max_real > 0:
            ax1.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                        label=f"Último dato realizado (h={h_max_real})")
        ax1.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
        ax1.set_title(titulo, fontweight="bold", fontsize=11)
        ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

        # Panel 2: solo modelo
        _dibujar_bandas(ax2, hs, res)
        if h_max_real > 0:
            ax2.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                        label=f"Último dato realizado (h={h_max_real})")
        ax2.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
        ax2.set_title("Solo proyección del modelo — mediana visible", fontsize=10, style="italic")
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

        # Panel 3: acumulado — solo Q40-Q60 + mediana
        if cum_q40 is not None and cum_q60 is not None:
            ax3.fill_between(hs, cum_q40, cum_q60, alpha=0.45, color=COLOR_BANDA,
                             label="Q40–Q60 acum. (20%)")
        ax3.plot(hs, cum_q50, color="steelblue", lw=2.0,
                 label="Mediana acumulada (Q50)", zorder=5)
        if mask_real.any():
            ax3.plot(hs[mask_real], cum_real[mask_real],
                     color="black", lw=2, label="Realizado acumulado", zorder=6)
            ax3.scatter(hs[mask_real], cum_real[mask_real], color="black", s=18, zorder=7)
        if h_max_real > 0:
            ax3.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                        label=f"Último dato realizado (h={h_max_real})")
        ax3.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
        ax3.set_xlabel("Horizonte h (días hábiles desde t)", fontsize=11)
        ax3.set_ylabel("Flujo neto acumulado (MM USD)", fontsize=11)
        ax3.set_title("Flujo neto acumulado D−R desde la fecha de origen",
                      fontsize=10, style="italic")
        ax3.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax3.grid(True, alpha=0.25)
        ax3.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax3.set_xlim(hs.min() - 1, hs.max() + 1)

        # Ejes Y fijos: imprescindible para el video, útil para comparar PNGs entre sí
        if ylims:
            for ax, yl in zip((ax1, ax2, ax3), ylims):
                if yl:
                    ax.set_ylim(*yl)

        nombre = f"fanchart_cv4_{banco}_f{fold_num}_{fecha_origen.strftime('%Y%m%d')}.png"
        ruta_png = Path(dir_out or DIR_OUTPUT) / nombre
        plt.savefig(ruta_png, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    print(f"  [{idx}/{total}] {nombre}")
    return ruta_png


# ── 6. Exportar Excel ─────────────────────────────────────────────────────────
def exportar_excel(df: pd.DataFrame) -> None:
    """Exporta una hoja por fold con columnas fecha_t, fecha_th, h, target, q01..q99, mean."""
    try:
        import openpyxl
    except ImportError:
        print("[AVISO] openpyxl no instalado — omitiendo exportación Excel")
        return

    q_cols = sorted(
        [c for c in df.columns if c.startswith("q") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    col_order = [c for c in ["fecha_t", "fecha_th", "h", "target"] + q_cols + ["mean"]
                 if c in df.columns]

    # Convertir a MM USD
    mm_cols = [c for c in ["target"] + q_cols + ["mean"] if c in df.columns]

    with pd.ExcelWriter(RUTA_EXCEL, engine="openpyxl") as writer:
        for fold_num, gdf in df.groupby("fold"):
            gdf_out = gdf[col_order].copy().sort_values(["fecha_t", "h"])
            for c in mm_cols:
                gdf_out[c] = gdf_out[c] / 1e6
            sheet = f"Fold_{fold_num:02d}"
            gdf_out.to_excel(writer, sheet_name=sheet, index=False)
            print(f"  [EXCEL] Hoja '{sheet}': {len(gdf_out):,} filas")

    print(f"[OK] Excel guardado: {RUTA_EXCEL}")


# ── 7. Video ──────────────────────────────────────────────────────────────────
# Tres backends, en orden de preferencia. Ninguno se asume presente:
#   1. imageio + imageio-ffmpeg  -> .mp4  (streaming, sin binario externo)
#   2. binario ffmpeg en el PATH -> .mp4  (streaming vía pipe rawvideo)
#   3. Pillow                    -> .gif  (siempre disponible: matplotlib lo requiere)
#
# Los tres escriben frame a frame. Nada acumula la secuencia completa en RAM:
# con ~600 PNG de 2400x2400 eso serían decenas de GB.

def _detectar_backend(formato: str):
    """Retorna (extension, backend) según lo pedido y lo instalado."""
    if formato not in ("auto", "mp4", "gif"):
        raise ValueError(f"VIDEO_FORMATO inválido: {formato!r}")

    if formato == "gif":
        return "gif", "pillow"

    try:
        import imageio                      # noqa: F401
        import imageio_ffmpeg               # noqa: F401
        return "mp4", "imageio"
    except ImportError:
        pass

    if shutil.which("ffmpeg"):
        return "mp4", "ffmpeg"

    if formato == "mp4":
        print("[AVISO] VIDEO_FORMATO='mp4' pero no hay imageio-ffmpeg ni binario "
              "ffmpeg — se genera GIF en su lugar")
    return "gif", "pillow"


def _dim_canvas(paths, ancho_max: int):
    """
    Lienzo común para todos los frames.

    savefig(bbox_inches='tight') recorta al contenido, así que los PNG NO salen
    todos del mismo tamaño (varía con el largo de títulos y ticks). Un encoder
    de video exige dimensión constante, de modo que se toma el máximo y cada
    frame se centra dentro. Ancho y alto se fuerzan a par: libx264 lo requiere.

    Image.open solo lee la cabecera para consultar .size — no decodifica píxeles.
    """
    from PIL import Image

    w_max = h_max = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                w_max, h_max = max(w_max, im.width), max(h_max, im.height)
        except Exception:
            continue
    if w_max == 0:
        return None

    esc = min(1.0, float(ancho_max) / w_max)
    w, h = int(round(w_max * esc)), int(round(h_max * esc))
    return (w - w % 2, h - h % 2)


def _frames(paths, size, para_gif: bool, colores: int):
    """Generador de frames ya normalizados al lienzo `size`. Uno vivo a la vez."""
    from PIL import Image

    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"  [AVISO] frame omitido {Path(p).name}: {e}")
            continue
        im.thumbnail(size, Image.LANCZOS)          # conserva aspecto
        lienzo = Image.new("RGB", size, (255, 255, 255))
        lienzo.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
        yield (lienzo.convert("P", palette=Image.ADAPTIVE, colors=colores)
               if para_gif else lienzo)


def crear_video(paths, tipo: str, banco: str = BANCO, dir_out: Path = None,
                fps: int = None, ancho_max: int = None,
                formato: str = None, colores: int = None):
    """
    Arma un video con los PNG de `paths`, en ese orden.

    Nombre: video_fanchart_cv4_<tipo>_<banco>_<YYYYmmdd_HHMMSS>.<ext>
    El timestamp es el de creación del video, así que dos corridas del mismo
    tipo nunca se sobrescriben.

    Retorna la ruta escrita, o None si no se pudo generar.
    """
    dir_out   = DIR_OUTPUT      if dir_out   is None else dir_out
    fps       = VIDEO_FPS       if fps       is None else fps
    ancho_max = VIDEO_ANCHO_MAX if ancho_max is None else ancho_max
    formato   = VIDEO_FORMATO   if formato   is None else formato
    colores   = VIDEO_COLORES   if colores   is None else colores

    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        print("[AVISO] sin PNG para el video")
        return None

    ext, backend = _detectar_backend(formato)
    size = _dim_canvas(paths, ancho_max)
    if size is None:
        print("[AVISO] no se pudo leer ningún PNG — sin video")
        return None

    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta  = Path(dir_out) / f"video_fanchart_cv4_{tipo}_{banco}_{sello}.{ext}"
    ruta.parent.mkdir(parents=True, exist_ok=True)

    print(f"  backend={backend}  frames={len(paths)}  "
          f"lienzo={size[0]}x{size[1]}  fps={fps}")

    try:
        if backend == "imageio":
            import imageio
            with imageio.get_writer(str(ruta), fps=fps, codec="libx264",
                                    quality=8, macro_block_size=None) as w:
                for fr in _frames(paths, size, False, colores):
                    w.append_data(np.asarray(fr))

        elif backend == "ffmpeg":
            cmd = [
                shutil.which("ffmpeg"), "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "pipe:0",
                "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", str(ruta),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                for fr in _frames(paths, size, False, colores):
                    proc.stdin.write(fr.tobytes())
            finally:
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError(f"ffmpeg salió con código {proc.returncode}")

        else:  # pillow -> gif
            gen = _frames(paths, size, True, colores)
            try:
                primero = next(gen)
            except StopIteration:
                print("[AVISO] ningún frame legible — sin video")
                return None
            # append_images acepta un generador: Pillow lo consume perezosamente,
            # así que el GIF se escribe sin materializar la secuencia en memoria.
            primero.save(ruta, save_all=True, append_images=gen,
                         duration=max(20, int(round(1000.0 / max(fps, 1)))),
                         loop=0, optimize=False)

    except Exception as e:
        print(f"[AVISO] falló la generación del video ({type(e).__name__}: {e})")
        if ruta.exists():
            try:
                ruta.unlink()          # no dejar un archivo truncado
            except OSError:
                pass
        return None

    mb = ruta.stat().st_size / 1e6
    print(f"[OK] video: {ruta.name}  ({mb:,.1f} MB)")
    return ruta


def _orden_desde_nombre(p: Path):
    """Clave (fold, fecha) parseada de fanchart_cv4_<banco>_f<fold>_<YYYYMMDD>.png."""
    partes = p.stem.split("_")
    fold, fecha = 0, ""
    for s in partes:
        if s.startswith("f") and s[1:].isdigit():
            fold = int(s[1:])
        elif len(s) == 8 and s.isdigit():
            fecha = s
    return (fold, fecha, p.name)


def crear_video_desde_carpeta(tipo: str, banco: str = BANCO,
                              dir_png: Path = None, **kw):
    """Arma el video con los PNG ya presentes en disco (no regenera figuras)."""
    dir_png = DIR_OUTPUT if dir_png is None else Path(dir_png)
    paths = sorted(dir_png.glob(f"fanchart_cv4_{banco}_f*_*.png"),
                   key=_orden_desde_nombre)
    if not paths:
        print(f"[AVISO] sin PNG fanchart_cv4_{banco}_* en {dir_png}")
        return None
    print(f"\nArmando video desde {len(paths)} PNG en disco...")
    return crear_video(paths, tipo, banco, dir_out=dir_png, **kw)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _tipo = VIDEO_TIPO if VIDEO_TIPO else _MODO_SUFIJO

    if SOLO_VIDEO:
        crear_video_desde_carpeta(_tipo, BANCO)
        raise SystemExit

    df = cargar_parquet(BANCO)
    df = _enriquecer_desde_matriz(df, BANCO)

    if FOTOS_DIARIAS or VIDEO_DIARIO:
        fechas = fechas_validas(df)
        fechas_sel = fechas[::PASO_FECHAS]
        if N_FECHAS_MAX is not None:
            fechas_sel = fechas_sel[:N_FECHAS_MAX]
        total = len(fechas_sel)

        # Si solo se quiere el video, los PNG son un intermedio desechable
        _tmp = None if FOTOS_DIARIAS else tempfile.mkdtemp(prefix="fanchart_cv4_")
        dir_png = DIR_OUTPUT if FOTOS_DIARIAS else Path(_tmp)

        _ylims = rangos_globales(df) if YLIM_FIJO else None
        if _ylims:
            _f = lambda t: f"[{t[0]:,.0f}, {t[1]:,.0f}]" if t else "auto"
            print(f"\nEjes Y fijos  diario={_f(_ylims[0])}  "
                  f"modelo={_f(_ylims[1])}  acum={_f(_ylims[2])}  (MM USD)")

        destino = "PNG en disco" if FOTOS_DIARIAS else "frames temporales (solo video)"
        print(f"\nGenerando {total} fan chart(s) — {destino}")

        # Rutas en el orden en que se generan (cronológico) = orden de los frames
        pngs, pngs_por_fold = [], {}
        for i, fecha_np in enumerate(fechas_sel, 1):
            fecha = pd.Timestamp(fecha_np)
            try:
                res = preparar_resultado(df, fecha)
                fold_num = int(res["fold"].iloc[0]) if "fold" in res.columns else 0
                ruta_png = graficar(res, fecha, BANCO, i, total, fold_num,
                                    dir_out=dir_png, ylims=_ylims)
                pngs.append(ruta_png)
                pngs_por_fold.setdefault(fold_num, []).append(ruta_png)
            except Exception as e:
                print(f"  [{i}/{total}] {fecha.date()} omitida: {e}")

            # gc.collect() cada 25 figuras: el objeto Figure ya se cierra en
            # graficar() (try/finally), pero el recolector de Python no
            # necesariamente libera el buffer C de Agg de inmediato — en una
            # corrida de cientos de figuras esa demora es lo que fragmenta
            # el heap del proceso hasta que un buffer contiguo grande ya no
            # entra. Forzarlo cada cierto número de iteraciones es más barato
            # que dejar que se acumule.
            if i % 25 == 0:
                gc.collect()

        if FOTOS_DIARIAS:
            print(f"\n[OK] {len(pngs)} gráficos guardados en: {DIR_OUTPUT}")

        try:
            if VIDEO_DIARIO and pngs:
                print(f"\nGenerando video ({_tipo})...")
                crear_video(pngs, _tipo, BANCO)          # siempre a DIR_OUTPUT

                if VIDEO_POR_FOLD:
                    for fold_num in sorted(pngs_por_fold):
                        print(f"\nGenerando video fold {fold_num}...")
                        crear_video(pngs_por_fold[fold_num],
                                    f"{_tipo}_f{fold_num}", BANCO)
        finally:
            if _tmp:
                shutil.rmtree(_tmp, ignore_errors=True)
                print(f"[OK] frames temporales eliminados")
    else:
        print("\n[INFO] FOTOS_DIARIAS y VIDEO_DIARIO en False — no se grafica nada")

    if EXPORTAR_EXCEL:
        print("\nExportando Excel...")
        exportar_excel(df)
