# -*- coding: utf-8 -*-
import os
os.environ["PATH"] += r";H:\DPINV\CARPETAS PERSONALES\DIEGO\2. Python\Paquetes python\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin"
"""
aux_fanchart_video_xgb_qt.py
Genera un video MP4 (o GIF como fallback) animando el fan chart XGBoost QT
(per-quantile Optuna) a través de todas las fechas de origen válidas del período TEST.

Idéntico a aux_fanchart_video_xgb.py, salvo:
  1. Apunta a modelos_xgb/eval/ con los modelos QT (step004_train_xgboost_qt.py)
  2. Color de bandas: tomato (distingue de XGBoost estándar darkorange y LightGBM steelblue)
  3. Títulos y nombres de archivo llevan sufijo _xgb_qt

Cada frame muestra:
  - Subplot superior : flujo diario D-R  (bandas Q01-Q99, Q05-Q95, mediana, realizado)
  - Subplot inferior : flujo neto acumulado D-R (mismas bandas + realizado acumulado)

Los ejes Y están fijados al rango global para comparabilidad entre fechas.

Requisitos:
  - ffmpeg en el PATH para MP4  (conda install -c conda-forge ffmpeg)
  - pillow como fallback para GIF  (pip install pillow)
"""

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data"   / "Clean"  / "matriz_features.parquet"
DIR_MODELOS  = BASE_SISTEMA / "2. Output" / "modelos_xgb" / "eval"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_fanchart_horizontes_xgb_qt"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO      = "SISTEMA"
CORTE_VAL  = pd.Timestamp("2022-07-01")
CORTE_TEST = pd.Timestamp("2023-01-03")

# ── Parámetros del video ──────────────────────────────────────────────────────
PASO_FECHAS = 1    # 1 = todos los días hábiles válidos; 2 = cada 2, etc.
FPS         = 2    # frames por segundo
DPI         = 72   # resolución reducida para evitar out-of-memory

COLOR = "tomato"  # distingue XGBoost QT de XGBoost estándar (darkorange) y LightGBM (steelblue)


# ── 1. Cargar modelos XGBoost QT ──────────────────────────────────────────────
def cargar_modelos(banco: str, dir_modelos: Path):
    metas = sorted(dir_modelos.glob(f"metadata_xgb_{banco}_*.json"), reverse=True)
    if not metas:
        raise FileNotFoundError(
            f"No se encontró metadata XGBoost para banco={banco} en {dir_modelos}"
        )
    meta      = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha     = metas[0].stem.split("_")[-1]
    cols_feat = meta["features"]
    quantiles = meta["quantiles"]

    modelos = {}
    for tau in quantiles:
        ruta = dir_modelos / f"xgb_{banco}_q{int(tau*100):02d}_{fecha}.json"
        if not ruta.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
        booster = xgb.Booster()
        booster.load_model(str(ruta))
        modelos[tau] = booster

    print(f"Modelo XGBoost QT cargado : {metas[0].name}  ({len(cols_feat)} features)")
    return modelos, cols_feat, quantiles


# ── 2. Datos ──────────────────────────────────────────────────────────────────
def cargar_datos(banco: str, cols_feat: list[str]):
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    df_train = df[df["fecha_t"] < CORTE_VAL].copy()
    df_test  = df[df["fecha_t"] >= CORTE_TEST].copy()

    cols_excluir = {"fecha_t", "banco", "target"}
    cols_num     = [c for c in df.columns if c not in cols_excluir]
    medianas     = df_train[cols_num].median()

    fechas_validas = np.sort(
        df_test[(df_test["h"] == 90) & df_test["target"].notna()]["fecha_t"].unique()
    )
    print(f"TEST  : {CORTE_TEST.date()} → {df_test['fecha_t'].max().date()}")
    print(f"Fechas válidas : {pd.Timestamp(fechas_validas[0]).date()} → "
          f"{pd.Timestamp(fechas_validas[-1]).date()} ({len(fechas_validas)} fechas)")

    return df, medianas, cols_num, fechas_validas


# ── 3. Predecir una fecha ─────────────────────────────────────────────────────
def predecir_fecha(df, fecha_origen, medianas, cols_num, cols_feat, modelos):
    df_f    = df[df["fecha_t"] == fecha_origen].copy().sort_values("h")
    cols_ok = [c for c in cols_num if c in df_f.columns]
    df_f[cols_ok] = df_f[cols_ok].fillna(medianas)
    for c in set(cols_feat) - set(df_f.columns):
        df_f[c] = 0.0

    cols_x = [c for c in cols_feat if c in df_f.columns]
    dmat   = xgb.DMatrix(df_f[cols_x].copy())

    res = df_f[["h", "target"]].copy().reset_index(drop=True)
    for tau, booster in modelos.items():
        res[f"q{int(tau*100):02d}"] = booster.predict(dmat)

    q_cols = sorted([c for c in res.columns if c.startswith("q")])
    res[q_cols] = np.sort(res[q_cols].values, axis=1)
    return res


# ── 4. Pre-computar todos los resultados ──────────────────────────────────────
def precomputar(df, medianas, cols_num, cols_feat, modelos, fechas_validas):
    fechas_sel = fechas_validas[::PASO_FECHAS]
    total      = len(fechas_sel)
    print(f"\nPre-computando {total} fechas...")
    frames = []

    for i, f in enumerate(fechas_sel, 1):
        fecha_origen = pd.Timestamp(f)
        res  = predecir_fecha(df, fecha_origen, medianas, cols_num, cols_feat, modelos)
        hs   = res["h"].values
        real = res["target"].values / 1e6
        mask = ~np.isnan(real)
        h_max_r = int(res.loc[mask, "h"].max()) if mask.any() else 0

        cum_q01 = np.cumsum(res["q01"].values / 1e6)
        cum_q05 = np.cumsum(res["q05"].values / 1e6)
        cum_q50 = np.cumsum(res["q50"].values / 1e6)
        cum_q95 = np.cumsum(res["q95"].values / 1e6)
        cum_q99 = np.cumsum(res["q99"].values / 1e6)
        cum_r   = np.where(mask, np.nancumsum(np.where(mask, real, 0)), np.nan)
        cum_r[~mask] = np.nan

        frames.append({
            "fecha": fecha_origen,
            "hs": hs, "real": real, "mask": mask, "h_max_r": h_max_r,
            "q01": res["q01"].values / 1e6,
            "q05": res["q05"].values / 1e6,
            "q50": res["q50"].values / 1e6,
            "q95": res["q95"].values / 1e6,
            "q99": res["q99"].values / 1e6,
            "cum_q01": cum_q01, "cum_q05": cum_q05, "cum_q50": cum_q50,
            "cum_q95": cum_q95, "cum_q99": cum_q99, "cum_r": cum_r,
        })
        if i % 5 == 0 or i == total:
            print(f"  {i}/{total}  {fecha_origen.date()}")

    return frames


# ── 5. Calcular rangos Y globales ─────────────────────────────────────────────
def rangos_globales(frames):
    y1_min = min(min(f["q01"].min(),
                     f["real"][f["mask"]].min() if f["mask"].any() else 0)
                 for f in frames)
    y1_max = max(max(f["q99"].max(),
                     f["real"][f["mask"]].max() if f["mask"].any() else 0)
                 for f in frames)
    y3_min = min(min(f["cum_q01"].min(),
                     np.nanmin(f["cum_r"]) if f["mask"].any() else 0)
                 for f in frames)
    y3_max = max(max(f["cum_q99"].max(),
                     np.nanmax(f["cum_r"]) if f["mask"].any() else 0)
                 for f in frames)
    pad1 = (y1_max - y1_min) * 0.05
    pad3 = (y3_max - y3_min) * 0.05
    return (y1_min - pad1, y1_max + pad1), (y3_min - pad3, y3_max + pad3)


# ── 6. Animar ─────────────────────────────────────────────────────────────────
def animar(frames, ylim1, ylim3, banco):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"hspace": 0.08})
    hs = frames[0]["hs"]

    for ax, ylim in [(ax1, ylim1), (ax3, ylim3)]:
        ax.set_xlim(hs.min() - 1, hs.max() + 1)
        ax.set_ylim(*ylim)
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    ax1.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
    ax3.set_ylabel("Flujo neto acumulado (MM USD)", fontsize=11)
    ax3.set_xlabel("Horizonte h (días hábiles desde t)", fontsize=11)

    legend_diario = [
        Patch(facecolor=COLOR, alpha=0.12, label="Q01–Q99 (98%)"),
        Patch(facecolor=COLOR, alpha=0.40, label="Q05–Q95 (90%)"),
        Line2D([0], [0], color="crimson", lw=2,   label="Mediana Q50"),
        Line2D([0], [0], color="black",   lw=2,   label="Realizado (D−R)"),
        Line2D([0], [0], color="red",     lw=1.2, ls="--", alpha=0.7,
               label="Último dato realizado"),
    ]
    legend_acum = [
        Patch(facecolor=COLOR, alpha=0.12, label="Q01–Q99 acum."),
        Patch(facecolor=COLOR, alpha=0.40, label="Q05–Q95 acum."),
        Line2D([0], [0], color="crimson", lw=2, label="Mediana acumulada Q50"),
        Line2D([0], [0], color="black",   lw=2, label="Realizado acumulado"),
    ]

    title = fig.suptitle("", fontsize=12, fontweight="bold", y=0.995)

    def update(i):
        f = frames[i]
        ax1.cla(); ax3.cla()

        for ax, ylim in [(ax1, ylim1), (ax3, ylim3)]:
            ax.set_xlim(hs.min() - 1, hs.max() + 1)
            ax.set_ylim(*ylim)
            ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
            ax.grid(True, alpha=0.25)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax1.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
        ax3.set_ylabel("Flujo neto acumulado (MM USD)", fontsize=11)
        ax3.set_xlabel("Horizonte h (días hábiles desde t)", fontsize=11)

        # ── Subplot diario ────────────────────────────────────────────────
        ax1.fill_between(hs, f["q01"], f["q99"], alpha=0.12, color=COLOR)
        ax1.fill_between(hs, f["q05"], f["q95"], alpha=0.28, color=COLOR)
        ax1.plot(hs, f["q05"], color=COLOR, lw=1.0, ls=":", alpha=0.7)
        ax1.plot(hs, f["q95"], color=COLOR, lw=1.0, ls=":", alpha=0.7)
        ax1.plot(hs, f["q50"], color="crimson", lw=2.0, zorder=5)
        if f["mask"].any():
            ax1.plot(hs[f["mask"]], f["real"][f["mask"]], color="black", lw=2, zorder=6)
            ax1.scatter(hs[f["mask"]], f["real"][f["mask"]], color="black", s=18, zorder=7)
        if f["h_max_r"] > 0:
            ax1.axvline(f["h_max_r"], color="red", lw=1.2, ls="--", alpha=0.7)
        ax1.legend(handles=legend_diario, loc="upper right", fontsize=9, framealpha=0.9)

        # ── Subplot acumulado ─────────────────────────────────────────────
        ax3.fill_between(hs, f["cum_q01"], f["cum_q99"], alpha=0.12, color=COLOR)
        ax3.fill_between(hs, f["cum_q05"], f["cum_q95"], alpha=0.28, color=COLOR)
        ax3.plot(hs, f["cum_q05"], color=COLOR, lw=1.0, ls=":", alpha=0.7)
        ax3.plot(hs, f["cum_q95"], color=COLOR, lw=1.0, ls=":", alpha=0.7)
        ax3.plot(hs, f["cum_q50"], color="crimson", lw=2.0, zorder=5)
        if f["mask"].any():
            ax3.plot(hs[f["mask"]], f["cum_r"][f["mask"]], color="black", lw=2, zorder=6)
            ax3.scatter(hs[f["mask"]], f["cum_r"][f["mask"]], color="black", s=18, zorder=7)
        if f["h_max_r"] > 0:
            ax3.axvline(f["h_max_r"], color="red", lw=1.2, ls="--", alpha=0.7)
        ax3.legend(handles=legend_acum, loc="upper left", fontsize=9, framealpha=0.9)

        title.set_text(
            f"Fan Chart XGBoost QT — {banco}  |  Origen: {f['fecha'].strftime('%d %b %Y')}  |  "
            f"Realizado hasta h={f['h_max_r']}  —  frame {i+1}/{len(frames)}"
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 // FPS, blit=False)

    nombre_mp4 = DIR_OUTPUT / f"fanchart_xgb_qt_{banco}_animacion.mp4"
    nombre_gif = DIR_OUTPUT / f"fanchart_xgb_qt_{banco}_animacion.gif"

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    if not ffmpeg_ok and len(frames) > 200:
        print(f"\n⚠️  ffmpeg no encontrado y hay {len(frames)} frames.")
        print("   Un GIF de este tamaño agotará la memoria.")
        print("   Instala ffmpeg:  conda install -c conda-forge ffmpeg")
        print("   O reduce PASO_FECHAS para bajar el número de frames.")
        plt.close(fig)
        return None

    try:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=1800)
        anim.save(str(nombre_mp4), writer=writer, dpi=DPI)
        print(f"\n✓ Video guardado: {nombre_mp4}")
        plt.close(fig)
        return nombre_mp4
    except Exception as e:
        print(f"  ffmpeg no disponible ({e}). Guardando como GIF...")
    try:
        writer_gif = animation.PillowWriter(fps=FPS)
        anim.save(str(nombre_gif), writer=writer_gif, dpi=DPI)
        print(f"\n✓ GIF guardado: {nombre_gif}")
        plt.close(fig)
        return nombre_gif
    except Exception as e2:
        print(f"  Error guardando GIF: {e2}")
        print("  Instala ffmpeg:  conda install -c conda-forge ffmpeg")
        plt.close(fig)
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    modelos, cols_feat, _ = cargar_modelos(BANCO, DIR_MODELOS)
    df, medianas, cols_num, fechas_validas = cargar_datos(BANCO, cols_feat)

    frames = precomputar(df, medianas, cols_num, cols_feat, modelos, fechas_validas)
    _, ylim3 = rangos_globales(frames)
    ylim1 = (-3000, 3000)  # límite fijo flujo diario (MM USD)

    print(f"\nRango Y flujo diario   : {ylim1[0]:.0f} → {ylim1[1]:.0f} MM USD (fijo)")
    print(f"Rango Y acumulado      : {ylim3[0]:.0f} → {ylim3[1]:.0f} MM USD")
    print(f"\nGenerando video ({len(frames)} frames, {FPS} fps)...")

    animar(frames, ylim1, ylim3, BANCO)
