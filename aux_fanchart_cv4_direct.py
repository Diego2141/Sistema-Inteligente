# -*- coding: utf-8 -*-
"""
aux_fanchart_cv4_direct.py
==========================
Fan charts para el modelo cv4 DIRECT (step005_walk_forward_cv_4.py).

Genera por cada fecha de origen seleccionada un PNG con 3 subplots:
  1. Flujo diario D-R: bandas + realizado
  2. Flujo diario D-R: solo modelo (sin realizado)
  3. Flujo neto acumulado D-R: bandas + realizado acumulado

Diferencia clave vs cv3: los parquets están en step005_wfcv_v4_direct/fold_exp/
y el nombre del banco en el filename es SISTEMA (no BancaLocal).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, GoodFriday,
    USFederalHolidayCalendar, Easter,
)
from pandas.tseries.offsets import CustomBusinessDay, Day as _Day


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
DIR_PREDS    = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct" / "fold_exp"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_fanchart_cv4_direct"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"

BANCO = "SISTEMA"

# ── Parámetros ─────────────────────────────────────────────────────────────────
PASO_FECHAS  = 1     # cada N días hábiles tomar una fecha de origen
N_FECHAS_MAX = None  # None = todos; ej: 10 para limitar cantidad de gráficos
COLOR_BANDA  = "steelblue"   # distinto al rojo de cv3 para distinguir visualmente

EXPORTAR_EXCEL = True
RUTA_EXCEL = DIR_OUTPUT / "fanchart_cv4_datos.xlsx"


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


def graficar(res: pd.DataFrame, fecha_origen: pd.Timestamp, banco: str,
             idx: int, total: int, fold_num: int):
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

    # Panel 3: acumulado
    ax3.fill_between(hs, cum_q01, cum_q99, alpha=0.10, color=COLOR_BANDA,
                     label="Q01–Q99 acum. (98%)")
    ax3.fill_between(hs, cum_q05, cum_q95, alpha=0.22, color=COLOR_BANDA,
                     label="Q05–Q95 acum. (90%)")
    if cum_q40 is not None and cum_q60 is not None:
        ax3.fill_between(hs, cum_q40, cum_q60, alpha=0.42, color=COLOR_BANDA,
                         label="Q40–Q60 acum. (20%)")
    ax3.plot(hs, cum_q05, color=COLOR_BANDA, lw=1.0, ls=":", alpha=0.7)
    ax3.plot(hs, cum_q95, color=COLOR_BANDA, lw=1.0, ls=":", alpha=0.7)
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

    nombre = f"fanchart_cv4_{banco}_f{fold_num}_{fecha_origen.strftime('%Y%m%d')}.png"
    plt.savefig(DIR_OUTPUT / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{idx}/{total}] {nombre}")


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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = cargar_parquet(BANCO)
    df = _enriquecer_desde_matriz(df, BANCO)

    fechas = fechas_validas(df)
    fechas_sel = fechas[::PASO_FECHAS]
    if N_FECHAS_MAX is not None:
        fechas_sel = fechas_sel[:N_FECHAS_MAX]

    total = len(fechas_sel)
    print(f"\nGenerando {total} fan chart(s) en {DIR_OUTPUT}")

    for i, fecha_np in enumerate(fechas_sel, 1):
        fecha = pd.Timestamp(fecha_np)
        try:
            res = preparar_resultado(df, fecha)
            fold_num = int(res["fold"].iloc[0]) if "fold" in res.columns else 0
            graficar(res, fecha, BANCO, i, total, fold_num)
        except Exception as e:
            print(f"  [{i}/{total}] {fecha.date()} omitida: {e}")

    print(f"\n[OK] {total} gráficos guardados en: {DIR_OUTPUT}")

    if EXPORTAR_EXCEL:
        print("\nExportando Excel...")
        exportar_excel(df)
